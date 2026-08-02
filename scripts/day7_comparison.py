"""对比模型 Probing: DINOv2 和 SigLIP"""
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from probing.probes import RidgeProbe, MLPProbe
from probing.baselines import ConstantVelocityBaseline


class DINOv2Prober:
    """DINOv2 表征提取与 Probing"""

    def __init__(self, model_name='facebook/dinov2-base'):
        from transformers import AutoModel, AutoImageProcessor

        print(f"加载 DINOv2: {model_name}")
        self.model = AutoModel.from_pretrained(model_name).cuda().eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.num_layers = len(self.model.encoder.layer)
        print(f"层数: {self.num_layers}")

    def extract_features(self, images, layer_idx):
        """提取指定层的表征"""
        inputs = self.processor(images=images, return_tensors="pt").to('cuda')

        with torch.no_grad():
            # 注册 hook
            features = []
            def hook(module, input, output):
                features.append(output.detach().cpu())

            handle = self.model.encoder.layer[layer_idx].register_forward_hook(hook)
            outputs = self.model(**inputs)
            handle.remove()

        # DINOv2 输出: (batch, num_patches+1, hidden_dim)
        # 去掉 [CLS] token
        hidden = features[0][:, 1:, :]  # (batch, num_patches, hidden_dim)
        return hidden

    def get_patch_grid(self, image_size):
        """获取 patch 网格尺寸"""
        # DINOv2: 14x14 patch, 224 输入 -> 16x16 grid
        patch_size = 14
        grid_h = image_size[0] // patch_size
        grid_w = image_size[1] // patch_size
        return grid_h, grid_w


class SigLIPProber:
    """SigLIP 表征提取与 Probing"""

    def __init__(self, model_name='google/siglip-base-patch16-224'):
        from transformers import AutoModel, AutoProcessor

        print(f"加载 SigLIP: {model_name}")
        self.model = AutoModel.from_pretrained(model_name).cuda().eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.num_layers = len(self.model.vision_model.encoder.layers)
        print(f"层数: {self.num_layers}")

    def extract_features(self, images, layer_idx):
        """提取指定层的表征"""
        inputs = self.processor(images=images, return_tensors="pt").to('cuda')

        with torch.no_grad():
            features = []
            def hook(module, input, output):
                features.append(output.detach().cpu())

            handle = self.model.vision_model.encoder.layers[layer_idx].register_forward_hook(hook)
            outputs = self.model.vision_model(**inputs)
            handle.remove()

        # SigLIP 输出: (batch, num_patches+1, hidden_dim)
        hidden = features[0][:, 1:, :]  # 去掉 [CLS]
        return hidden

    def get_patch_grid(self, image_size):
        """获取 patch 网格尺寸"""
        # SigLIP: 16x16 patch, 224 输入 -> 14x14 grid
        patch_size = 16
        grid_h = image_size[0] // patch_size
        grid_w = image_size[1] // patch_size
        return grid_h, grid_w


def extract_ball_tokens(hidden, ball_positions, image_size, grid_size):
    """
    从 hidden states 提取球所在 patch 的 token

    Args:
        hidden: (num_patches, hidden_dim)
        ball_positions: [(x, y, radius), ...]
        image_size: (width, height)
        grid_size: (grid_h, grid_w)
    """
    grid_h, grid_w = grid_size
    patch_h = image_size[1] / grid_h
    patch_w = image_size[0] / grid_w

    all_tokens = []
    for x, y, r in ball_positions:
        # 计算 patch 坐标
        col = int(x / patch_w)
        row = int(y / patch_h)

        # 边界
        col = min(max(col, 0), grid_w - 1)
        row = min(max(row, 0), grid_h - 1)

        # 展平索引
        idx = row * grid_w + col
        if idx < hidden.shape[0]:
            all_tokens.append(hidden[idx].numpy())

    return np.array(all_tokens) if all_tokens else None


def run_probing_for_model(prober, model_name, seq_paths, target_layers, target_type='velocity',
                          max_sequences=None, probe_type='ridge', num_frames=2):
    """对指定模型运行 probing"""

    print(f"\n{'='*60}")
    print(f"{model_name} - {target_type} probing")
    print(f"{'='*60}")

    if max_sequences:
        seq_paths = seq_paths[:max_sequences]

    results = {}

    for layer_idx in target_layers:
        print(f"\nLayer {layer_idx}/{prober.num_layers-1}")

        all_features = []
        all_targets = []
        all_prev_velocities = []

        for seq_path in tqdm(seq_paths, desc=f"Layer {layer_idx}"):
            try:
                with open(seq_path / 'trajectory.json', 'r') as f:
                    traj_data = json.load(f)

                frame_files = sorted(seq_path.glob('frame_*.png'))
                if len(frame_files) < num_frames + 1:
                    continue

                # 处理每一帧
                for frame_idx in range(len(frame_files) - num_frames):
                    from PIL import Image

                    # 加载多帧
                    images = [Image.open(frame_files[frame_idx + i]).convert('RGB')
                             for i in range(num_frames)]

                    target_frame_idx = frame_idx + num_frames - 1

                    # 获取球位置
                    ball_positions = []
                    frame_targets = []
                    frame_prev_vel = []

                    for ball_id, ball in enumerate(traj_data['balls']):
                        traj = ball['trajectory']
                        if target_frame_idx < len(traj):
                            state = traj[target_frame_idx]
                            ball_positions.append((state['x'], state['y'], state['radius']))

                            # 目标
                            if target_type == 'velocity' and target_frame_idx + 1 < len(traj):
                                next_state = traj[target_frame_idx + 1]
                                frame_targets.append([next_state['vx'], next_state['vy']])
                            elif target_type == 'position' and target_frame_idx + 1 < len(traj):
                                next_state = traj[target_frame_idx + 1]
                                frame_targets.append([next_state['x'], next_state['y']])
                            else:
                                frame_targets.append([0, 0])

                            frame_prev_vel.append([state['vx'], state['vy']])

                    if not ball_positions:
                        continue

                    # 提取特征（只用最后一帧）
                    hidden = prober.extract_features([images[-1]], layer_idx)
                    hidden = hidden[0]  # (num_patches, hidden_dim)

                    # 获取 grid
                    grid_size = prober.get_patch_grid(images[-1].size)

                    # 提取球的 token
                    ball_tokens = extract_ball_tokens(
                        hidden, ball_positions,
                        images[-1].size, grid_size
                    )

                    if ball_tokens is None or len(ball_tokens) == 0:
                        continue

                    # 对多球取平均，得到固定维度特征
                    # ball_tokens: (num_balls, hidden_dim) -> (hidden_dim,)
                    # frame_targets: (num_balls, 2) -> (2,)
                    # frame_prev_vel: (num_balls, 2) -> (2,)
                    avg_feature = ball_tokens.mean(axis=0)
                    avg_target = np.array(frame_targets).mean(axis=0)
                    avg_prev_vel = np.array(frame_prev_vel).mean(axis=0)

                    all_features.append(avg_feature)
                    all_targets.append(avg_target)
                    all_prev_velocities.append(avg_prev_vel)

                    if len(all_features) % 20 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

            except Exception as e:
                continue

        if not all_features:
            print(f"Layer {layer_idx}: 无数据")
            continue

        # 合并（avg_feature 是 1D，用 np.array 而非 concatenate）
        X = np.array(all_features)  # (num_samples, hidden_dim)
        y = np.array(all_targets)   # (num_samples, 2)
        prev_vel = np.array(all_prev_velocities)  # (num_samples, 2)

        print(f"Layer {layer_idx}: {len(X)} 样本, dim {X.shape[1]}")

        # 训练探针
        if probe_type == 'ridge':
            probe = RidgeProbe()
        else:
            probe = MLPProbe()

        result = probe.fit_evaluate(X, y)
        result['layer_idx'] = layer_idx

        # 基线
        baseline = ConstantVelocityBaseline()
        baseline_result = baseline.evaluate(prev_vel, y)
        result['baseline_r2'] = baseline_result['r2']

        print(f"Layer {layer_idx}: R² = {result['r2']:.4f}, Baseline = {baseline_result['r2']:.4f}")

        results[layer_idx] = result

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results_dinov2')
    parser.add_argument('--model', type=str, default='dinov2',
                       choices=['dinov2', 'siglip'])
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['small', 'base', 'large'])
    parser.add_argument('--target_type', type=str, default='velocity')
    parser.add_argument('--layers', type=str, default='0,4,8,11',
                       help='层号，逗号分隔')
    parser.add_argument('--max_seq', type=int, default=500)
    parser.add_argument('--probe_type', type=str, default='ridge')
    parser.add_argument('--num_frames', type=int, default=2)
    args = parser.parse_args()

    print("="*60)
    print(f"对比模型 Probing: {args.model.upper()}")
    print("="*60)

    # 加载数据
    with open(Path(args.data_dir) / 'split.json', 'r') as f:
        split = json.load(f)
    train_meta = split['train']

    data_dir = Path(args.data_dir)
    train_paths = [data_dir / item['path'] for item in train_meta]
    print(f"Train: {len(train_paths)} 段")

    # 解析层
    target_layers = [int(x) for x in args.layers.split(',')]

    # 初始化模型
    if args.model == 'dinov2':
        model_name = f'facebook/dinov2-{args.model_size}'
        prober = DINOv2Prober(model_name)
    else:
        model_name = f'google/siglip-{args.model_size}-patch16-224'
        prober = SigLIPProber(model_name)

    # 运行 probing
    results = run_probing_for_model(
        prober, args.model.upper(), train_paths, target_layers,
        target_type=args.target_type,
        max_sequences=args.max_seq,
        probe_type=args.probe_type,
        num_frames=args.num_frames
    )

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    all_results = list(results.values())
    with open(output_dir / f'summary_{args.target_type}.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # 打印汇总
    print("\n" + "="*60)
    print(f"{args.model.upper()} Probing 完成!")
    print("="*60)
    print(f"{'Layer':<8} {'R²':<10} {'Baseline R²':<12}")
    print("-"*30)
    for r in all_results:
        print(f"{r['layer_idx']:<8} {r['r2']:<10.4f} {r['baseline_r2']:<12.4f}")

    print(f"\n结果保存至: {output_dir}")


if __name__ == '__main__':
    main()
