"""数据集生成主脚本：批量生成物理场景并渲染"""
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_generation.physics_sim import generate_scene
from data_generation.renderer import render_frame


def generate_sequence(seq_id, style, output_dir, width=448, height=448,
                      num_frames=8, seed=None):
    """生成单个序列：物理模拟 + 渲染所有帧"""
    seq_dir = Path(output_dir) / style / f"seq_{seq_id:06d}"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # 物理模拟
    traj_data = generate_scene(
        width=width, height=height,
        num_frames=num_frames,
        seed=seed
    )

    # 渲染每一帧
    for frame_idx in range(num_frames):
        # 收集该帧所有球的状态
        balls_state = []
        for ball in traj_data['balls']:
            state = ball['trajectory'][frame_idx].copy()
            state['id'] = ball['id']
            state['radius'] = ball['radius']
            balls_state.append(state)

        # 渲染
        img = render_frame(width, height, balls_state, style)

        # 保存 PNG
        frame_path = seq_dir / f"frame_{frame_idx:03d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # 保存轨迹真值
    traj_path = seq_dir / "trajectory.json"
    with open(traj_path, 'w') as f:
        json.dump(traj_data, f, indent=2)

    return seq_dir


def generate_dataset(output_dir, num_sequences=2000, styles=None,
                     width=448, height=448, num_frames=8, start_seed=0):
    """生成完整数据集"""
    if styles is None:
        styles = ['minimal', 'medium', 'realistic']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = []

    for style in styles:
        print(f"\n生成 {style} 风格数据 ({num_sequences} 段)...")

        for i in tqdm(range(num_sequences), desc=style):
            seq_id = start_seed + i
            seed = start_seed * 10000 + i

            seq_dir = generate_sequence(
                seq_id=seq_id,
                style=style,
                output_dir=output_dir,
                width=width,
                height=height,
                num_frames=num_frames,
                seed=seed
            )

            all_metadata.append({
                'seq_id': seq_id,
                'style': style,
                'path': str(seq_dir.relative_to(output_dir)),
                'seed': seed
            })

    # 保存元数据
    meta_path = output_dir / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n数据集生成完成: {len(all_metadata)} 段")
    print(f"元数据保存至: {meta_path}")

    return all_metadata


def split_dataset(metadata, train_ratio=0.8, seed=42):
    """按轨迹划分 train/test"""
    np.random.seed(seed)
    indices = np.random.permutation(len(metadata))
    split_idx = int(len(metadata) * train_ratio)

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_data = [metadata[i] for i in train_indices]
    test_data = [metadata[i] for i in test_indices]

    return train_data, test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成 VLM 运动学 probing 数据集')
    parser.add_argument('--output', type=str, default='./data',
                        help='输出目录')
    parser.add_argument('--num', type=int, default=2000,
                        help='每种风格的序列数')
    parser.add_argument('--styles', type=str, nargs='+',
                        default=['minimal', 'medium', 'realistic'],
                        help='渲染风格')
    parser.add_argument('--width', type=int, default=448)
    parser.add_argument('--height', type=int, default=448)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--test', action='store_true',
                        help='测试模式：只生成少量数据')

    args = parser.parse_args()

    if args.test:
        args.num = 5
        print("测试模式：每种风格只生成 5 段")

    metadata = generate_dataset(
        output_dir=args.output,
        num_sequences=args.num,
        styles=args.styles,
        width=args.width,
        height=args.height,
        num_frames=args.frames
    )

    # 划分数据集
    train_data, test_data = split_dataset(metadata)

    # 保存划分结果
    split_path = Path(args.output) / "split.json"
    with open(split_path, 'w') as f:
        json.dump({
            'train': train_data,
            'test': test_data
        }, f, indent=2)

    print(f"\nTrain: {len(train_data)} 段")
    print(f"Test: {len(test_data)} 段")
    print(f"划分保存至: {split_path}")
