"""Day 3: 流式表征提取与 Probing 主脚本"""
import sys
import os
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.qwen_vl import QwenVLHookManager
from models.patch_utils import PatchLocator
from probing.probes import RidgeProbe, ControlTaskProbe
from probing.baselines import ConstantVelocityBaseline


def load_split_data(data_dir):
    """加载 train/test 划分"""
    with open(Path(data_dir) / 'split.json', 'r') as f:
        split = json.load(f)
    return split['train'], split['test']


def get_sequence_paths(metadata_list, data_dir):
    """获取序列完整路径"""
    data_dir = Path(data_dir)
    paths = []
    for item in metadata_list:
        seq_path = data_dir / item['path']
        if seq_path.exists():
            paths.append(seq_path)
    return paths


def extract_targets_from_trajectory(traj_data, target_type='velocity'):
    """
    从轨迹数据提取目标值

    Args:
        traj_data: 轨迹数据
        target_type: 'velocity', 'position', 'acceleration'
    """
    targets = []
    for ball in traj_data['balls']:
        traj = ball['trajectory']
        ball_targets = []

        for i in range(len(traj)):
            if target_type == 'velocity' and i + 1 < len(traj):
                # 下一帧速度
                ball_targets.append([traj[i+1]['vx'], traj[i+1]['vy']])
            elif target_type == 'position' and i + 1 < len(traj):
                # 下一帧位置
                ball_targets.append([traj[i+1]['x'], traj[i+1]['y']])
            elif target_type == 'acceleration':
                # 当前帧加速度
                ball_targets.append([traj[i]['ax'], traj[i]['ay']])
            else:
                ball_targets.append([0, 0])

        targets.append(ball_targets)
    return targets


def run_probing_for_layer(model, locator, seq_paths, layer_idx, target_type='velocity',
                          max_sequences=None, probe_type='ridge', num_frames=1):
    """
    对单层运行 probing

    Args:
        num_frames: 输入帧数（1=单帧，2/3=多帧）

    Returns:
        dict: probing 结果
    """
    print(f"\n{'='*60}")
    print(f"Layer {layer_idx} - {target_type} probing ({num_frames}帧输入)")
    print(f"{'='*60}")

    if max_sequences:
        seq_paths = seq_paths[:max_sequences]

    all_features = []
    all_targets = []
    all_prev_velocities = []  # 用于基线

    # 收集数据
    for seq_path in tqdm(seq_paths, desc=f"Layer {layer_idx} 数据收集"):
        try:
            # 加载轨迹
            with open(seq_path / 'trajectory.json', 'r') as f:
                traj_data = json.load(f)

            # 加载帧
            frame_files = sorted(seq_path.glob('frame_*.png'))
            if len(frame_files) < num_frames + 1:  # 需要足够的帧
                continue

            # 提取目标
            targets = extract_targets_from_trajectory(traj_data, target_type)

            # 处理每一帧（跳过最后几帧，确保有足够的帧用于输入和目标）
            for frame_idx in range(len(frame_files) - num_frames):
                from PIL import Image

                # 加载多帧
                images = []
                for offset in range(num_frames):
                    img = Image.open(frame_files[frame_idx + offset]).convert('RGB')
                    images.append(img)

                # 使用最后一帧的位置（因为这是要预测的目标帧的前一帧）
                target_frame_idx = frame_idx + num_frames - 1

                # 获取该帧球位置
                ball_positions = []
                frame_targets = []
                frame_prev_vel = []

                for ball_id, ball in enumerate(traj_data['balls']):
                    traj = ball['trajectory']
                    if target_frame_idx < len(traj) and ball_id < len(targets):
                        state = traj[target_frame_idx]
                        ball_positions.append((state['x'], state['y'], state['radius']))
                        frame_targets.append(targets[ball_id][target_frame_idx])
                        frame_prev_vel.append([state['vx'], state['vy']])

                if not ball_positions:
                    continue

                # 提取特征（多帧输入）
                results = model.forward_with_hooks(images, layer_indices=[layer_idx])

                if layer_idx not in results['layer_outputs'] or not results['layer_outputs'][layer_idx]:
                    continue

                hidden = results['layer_outputs'][layer_idx][0]  # (num_tokens, hidden_dim)

                # 获取 grid 信息（多帧时 grid 可能不同）
                grid_thw = model.get_grid_info(results['inputs'])
                if grid_thw is None:
                    continue

                # 多帧时，grid_thw 的 temporal 维度 > 1
                # 我们使用最后一帧的 grid 信息
                if len(grid_thw[0]) == 3:
                    t, h, w = grid_thw[0].tolist()
                    # 计算每张图的 token 数
                    tokens_per_frame = h * w
                    # 使用最后一帧的 token
                    if hidden.shape[0] >= t * tokens_per_frame:
                        # 只保留最后一帧的 token
                        hidden = hidden[-tokens_per_frame:]
                        grid_thw_single = torch.tensor([[1, h, w]])
                    else:
                        grid_thw_single = grid_thw[0:1]
                else:
                    grid_thw_single = grid_thw[0:1]

                # 提取球的 token 并 mean-pool（使用最后一帧的位置）
                ball_tokens, ball_mask = locator.extract_ball_tokens(
                    hidden, ball_positions,
                    images[-1].width, images[-1].height,
                    grid_thw_single[0]
                )

                # Mean-pool over tokens for each ball
                mask_expanded = ball_mask[..., np.newaxis]
                ball_repr = (ball_tokens * mask_expanded).sum(axis=1) / (ball_mask.sum(axis=1, keepdims=True) + 1e-8)

                all_features.append(ball_repr.numpy())
                all_targets.append(np.array(frame_targets))
                all_prev_velocities.append(np.array(frame_prev_vel))

                # 清理
                model.clear_layer_outputs()

                # 定期清理显存
                if len(all_features) % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        except Exception as e:
            print(f"处理 {seq_path} 失败: {e}")
            continue

    if not all_features:
        print(f"Layer {layer_idx}: 无有效数据")
        return None

    # 合并数据
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_targets, axis=0)
    prev_vel = np.concatenate(all_prev_velocities, axis=0)

    print(f"Layer {layer_idx}: 收集 {len(X)} 个样本, 特征维度 {X.shape[1]}")

    # 训练探针
    if probe_type == 'ridge':
        probe = RidgeProbe()
    else:
        from probing.probes import MLPProbe
        probe = MLPProbe()

    result = probe.fit_evaluate(X, y)
    result['layer_idx'] = layer_idx
    result['target_type'] = target_type
    result['n_samples'] = len(X)

    # 计算基线
    baseline = ConstantVelocityBaseline()
    baseline_result = baseline.evaluate(prev_vel, y)
    result['baseline_r2'] = baseline_result['r2']
    result['r2_over_baseline'] = result['r2'] - baseline_result['r2']

    print(f"Layer {layer_idx}: R² = {result['r2']:.4f}, "
          f"Baseline R² = {baseline_result['r2']:.4f}, "
          f"Diff = {result['r2_over_baseline']:.4f}")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--target_type', type=str, default='velocity',
                       choices=['velocity', 'position', 'acceleration'])
    parser.add_argument('--layers', type=str, default='all',
                       help='all 或逗号分隔的层号，如 0,5,10,15,20,25,30')
    parser.add_argument('--max_seq', type=int, default=None,
                       help='最大序列数（用于测试）')
    parser.add_argument('--probe_type', type=str, default='ridge',
                       choices=['ridge', 'mlp'])
    parser.add_argument('--num_frames', type=int, default=1,
                       help='输入帧数：1=单帧, 2/3=多帧')
    args = parser.parse_args()

    print("="*60)
    print("Day 3: 流式表征提取与 Probing")
    print("="*60)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 加载数据
    print("\n加载数据集...")
    train_meta, test_meta = load_split_data(args.data_dir)
    train_paths = get_sequence_paths(train_meta, args.data_dir)
    test_paths = get_sequence_paths(test_meta, args.data_dir)

    print(f"Train: {len(train_paths)} 段")
    print(f"Test: {len(test_paths)} 段")

    # 确定要 probing 的层
    if args.layers == 'all':
        layer_indices = list(range(32))  # 全部 32 层
    else:
        layer_indices = [int(x) for x in args.layers.split(',')]

    print(f"Probing 层: {layer_indices}")

    # 初始化模型
    print("\n加载模型...")
    model = QwenVLHookManager()
    locator = PatchLocator()

    # 对所有层运行 probing
    all_results = []

    for layer_idx in layer_indices:
        result = run_probing_for_layer(
            model, locator, train_paths, layer_idx,
            target_type=args.target_type,
            max_sequences=args.max_seq,
            probe_type=args.probe_type,
            num_frames=args.num_frames
        )

        if result:
            all_results.append(result)

            # 保存中间结果
            result_file = output_dir / f'layer_{layer_idx:02d}_{args.target_type}.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)

        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()

    # 保存汇总结果
    summary_file = output_dir / f'summary_{args.target_type}.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # 打印汇总
    print("\n" + "="*60)
    print("Probing 完成！汇总:")
    print("="*60)
    print(f"{'Layer':<8} {'R²':<10} {'Baseline R²':<12} {'Diff':<10}")
    print("-"*40)
    for r in all_results:
        print(f"{r['layer_idx']:<8} {r['r2']:<10.4f} {r['baseline_r2']:<12.4f} {r['r2_over_baseline']:<10.4f}")

    print(f"\n结果保存至: {output_dir}")


if __name__ == '__main__':
    main()
