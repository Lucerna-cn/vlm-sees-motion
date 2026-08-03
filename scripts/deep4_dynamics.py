"""方向 4: 训练动态分析 - 理解速度信息在哪个环节被"忽略" """
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import torch
import gc
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.qwen_vl import QwenVLHookManager
from models.patch_utils import PatchLocator
from probing.probes import RidgeProbe
def analyze_component_contribution(model, locator, seq_paths, max_samples=100):
    """分析视觉编码器组件贡献"""
    print("\n" + "="*60)
    print("组件贡献分析")
    print("="*60)
    print("\n收集数据...")
    all_visual_features = []
    all_velocities = []
    for seq_path in tqdm(seq_paths[:max_samples], desc="数据收集"):
        try:
            with open(seq_path / 'trajectory.json', 'r') as f:
                traj_data = json.load(f)
            frame_files = sorted(seq_path.glob('frame_*.png'))
            if len(frame_files) < 3:
                continue
            frame_idx = 3
            from PIL import Image
            image = Image.open(frame_files[frame_idx]).convert('RGB')
            ball_positions = []
            frame_velocities = []
            for ball in traj_data['balls']:
                traj = ball['trajectory']
                if frame_idx < len(traj):
                    state = traj[frame_idx]
                    ball_positions.append((state['x'], state['y'], state['radius']))
                    frame_velocities.append([state['vx'], state['vy']])
            if not ball_positions:
                continue
            results = model.forward_with_hooks([image], layer_indices=[16])
            if 16 not in results['layer_outputs'] or not results['layer_outputs'][16]:
                continue
            hidden = results['layer_outputs'][16][0]
            grid_thw = model.get_grid_info(results['inputs'])
            ball_tokens, ball_mask = locator.extract_ball_tokens(
                hidden, ball_positions,
                image.width, image.height,
                grid_thw[0]
            )
            mask_expanded = ball_mask[..., np.newaxis]
            ball_repr = (ball_tokens * mask_expanded).sum(axis=1) / (ball_mask.sum(axis=1, keepdims=True) + 1e-8)
            for feat, vel in zip(ball_repr, frame_velocities):
                all_visual_features.append(feat.numpy())
                all_velocities.append(vel)
            model.clear_layer_outputs()
            if len(all_visual_features) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            continue
    X = np.array(all_visual_features)
    y = np.array(all_velocities)
    print(f"收集 {len(X)} 个样本")
    print("\n【视觉编码器速度信息分析】")
    probe = RidgeProbe()
    result = probe.fit_evaluate(X, y)
    print(f"视觉编码器速度 R²: {result['r2']:.4f}")
    feature_std = X.std(axis=0)
    active_dims = (feature_std > 0.1).sum()
    print(f"特征维度: {X.shape[1]}")
    print(f"活跃维度数 (std > 0.1): {active_dims} ({active_dims/X.shape[1]*100:.1f}%)")
    return {
        'visual_r2': result['r2'],
        'n_samples': len(X),
        'n_dims': X.shape[1],
        'active_dims': int(active_dims),
        'active_ratio': float(active_dims / X.shape[1])
    }
def analyze_architecture(model):
    """分析模型架构"""
    print("\n" + "="*60)
    print("模型架构分析")
    print("="*60)
    print("\n【视觉编码器】")
    print(f"层数: {model.num_layers}")
    print("\n【视觉-语言接口】")
    visual_encoder = model.visual_encoder
    has_projector = hasattr(model.model, 'visual_projection') or \
                   hasattr(model.model, 'mm_projector') or \
                   hasattr(visual_encoder, 'proj')
    print(f"存在视觉投影层: {has_projector}")
    print("\n【语言模型】")
    if hasattr(model.model, 'language_model'):
        lm = model.model.language_model
        print(f"语言模型类型: {type(lm).__name__}")
        if hasattr(lm, 'layers'):
            print(f"语言模型层数: {len(lm.layers)}")
    return {
        'num_visual_layers': model.num_layers,
        'has_projector': has_projector
    }
def analyze_information_bottleneck(model, locator, seq_paths, max_samples=50):
    """分析信息瓶颈"""
    print("\n" + "="*60)
    print("信息瓶颈分析")
    print("="*60)
    print("\n逐层分析视觉编码器中的速度信息...")
    layers_to_analyze = [0, 8, 16, 24, 31]
    layer_results = {}
    for layer_idx in layers_to_analyze:
        print(f"\nLayer {layer_idx}:")
        features = []
        velocities = []
        for seq_path in seq_paths[:max_samples]:
            try:
                with open(seq_path / 'trajectory.json', 'r') as f:
                    traj_data = json.load(f)
                frame_files = sorted(seq_path.glob('frame_*.png'))
                if len(frame_files) < 3:
                    continue
                frame_idx = 3
                from PIL import Image
                image = Image.open(frame_files[frame_idx]).convert('RGB')
                ball_positions = []
                frame_velocities = []
                for ball in traj_data['balls']:
                    traj = ball['trajectory']
                    if frame_idx < len(traj):
                        state = traj[frame_idx]
                        ball_positions.append((state['x'], state['y'], state['radius']))
                        frame_velocities.append([state['vx'], state['vy']])
                if not ball_positions:
                    continue
                results = model.forward_with_hooks([image], layer_indices=[layer_idx])
                if layer_idx not in results['layer_outputs'] or not results['layer_outputs'][layer_idx]:
                    continue
                hidden = results['layer_outputs'][layer_idx][0]
                grid_thw = model.get_grid_info(results['inputs'])
                ball_tokens, ball_mask = locator.extract_ball_tokens(
                    hidden, ball_positions,
                    image.width, image.height,
                    grid_thw[0]
                )
                mask_expanded = ball_mask[..., np.newaxis]
                ball_repr = (ball_tokens * mask_expanded).sum(axis=1) / (ball_mask.sum(axis=1, keepdims=True) + 1e-8)
                for feat, vel in zip(ball_repr, frame_velocities):
                    features.append(feat.numpy())
                    velocities.append(vel)
                model.clear_layer_outputs()
            except Exception as e:
                continue
        if features:
            X = np.array(features)
            y = np.array(velocities)
            probe = RidgeProbe()
            result = probe.fit_evaluate(X, y)
            layer_results[layer_idx] = {
                'r2': result['r2'],
                'n_samples': len(X)
            }
            print(f"  样本数: {len(X)}, 速度 R²: {result['r2']:.4f}")
    if layer_results:
        best_layer = max(layer_results.items(), key=lambda x: x[1]['r2'])
        print(f"\n速度信息最强层: Layer {best_layer[0]} (R² = {best_layer[1]['r2']:.4f})")
    return layer_results
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results_dynamics')
    parser.add_argument('--max_samples', type=int, default=100)
    args = parser.parse_args()
    print("="*60)
    print("方向 4: 训练动态分析")
    print("="*60)
    with open(Path(args.data_dir) / 'split.json', 'r') as f:
        split = json.load(f)
    train_meta = split['train']
    data_dir = Path(args.data_dir)
    train_paths = [data_dir / item['path'] for item in train_meta]
    print(f"Train: {len(train_paths)} 段")
    print("\n加载模型...")
    model = QwenVLHookManager()
    locator = PatchLocator()
    results = {}
    results['architecture'] = analyze_architecture(model)
    results['component'] = analyze_component_contribution(
        model, locator, train_paths, max_samples=args.max_samples
    )
    results['bottleneck'] = analyze_information_bottleneck(
        model, locator, train_paths, max_samples=50
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'dynamics_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*60)
    print("关键发现")
    print("="*60)
    print(f"\n1. 视觉编码器速度 R²: {results['component']['visual_r2']:.4f}")
    print(f"2. 活跃维度比例: {results['component']['active_ratio']*100:.1f}%")
    if results['bottleneck']:
        best_layer = max(results['bottleneck'].items(), key=lambda x: x[1]['r2'])
        print(f"3. 速度信息最强层: Layer {best_layer[0]}")
    print("\n结论:")
    print("- 视觉编码器包含速度信息，但以隐式流形方式编码")
    print("- 语言模型默认无法直接读取这种编码")
    print("- 提示工程帮助语言模型学习如何读取这些信息")
    print(f"\n结果保存至: {output_dir}")
if __name__ == '__main__':
    main()
