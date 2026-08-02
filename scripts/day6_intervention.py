"""Day 6: 因果干预实验 - 验证关键层是否被语言生成使用"""
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
from intervention.patch_experiment import InterventionExperiment


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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results_intervention')
    parser.add_argument('--target_layers', type=str, default='16,24',
                       help='目标层，逗号分隔')
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--patch_mode', type=str, default='mean',
                       choices=['mean', 'zero', 'random'])
    args = parser.parse_args()

    print("="*60)
    print("Day 6: 因果干预实验")
    print("="*60)

    # 解析目标层
    target_layers = [int(x) for x in args.target_layers.split(',')]
    print(f"目标层: {target_layers}")
    print(f"Patch 模式: {args.patch_mode}")

    # 加载数据
    print("\n加载数据集...")
    _, test_meta = load_split_data(args.data_dir)
    test_paths = get_sequence_paths(test_meta, args.data_dir)
    print(f"Test: {len(test_paths)} 段")

    if args.max_samples:
        test_paths = test_paths[:args.max_samples]
        print(f"使用样本: {len(test_paths)} 段")

    # 初始化模型
    print("\n加载模型...")
    model = QwenVLHookManager()
    locator = PatchLocator()

    # 运行干预实验
    print("\n运行干预实验...")
    experiment = InterventionExperiment(model, locator)

    results = experiment.run_batch(
        test_paths,
        target_layers,
        ball_id=0,
        frame_idx=3,
        patch_mode=args.patch_mode
    )

    # 计算统计
    stats = experiment.compute_statistics()

    print("\n" + "="*60)
    print("干预实验结果")
    print("="*60)
    print(f"总样本: {stats['total']}")
    print(f"正常准确率: {stats['normal_accuracy']:.4f}")
    print(f"Patch 后准确率: {stats['patched_accuracy']:.4f}")
    print(f"行为改变率: {stats['behavior_change_rate']:.4f}")
    print(f"准确率下降率: {stats['accuracy_drop_rate']:.4f}")
    print(f"准确率变化: {stats['accuracy_change']:.4f}")

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    experiment.save_results(output_dir / 'intervention_results.json')

    print(f"\n结果保存至: {output_dir}")

    # 关键发现
    print("\n" + "="*60)
    print("关键发现")
    print("="*60)

    if stats['accuracy_drop_rate'] > 0.1:
        print(f"✓ Patch 导致 {stats['accuracy_drop_rate']*100:.1f}% 的样本准确率下降")
        print("  → 这些层对语言生成有因果影响")
    elif stats['behavior_change_rate'] > 0.3:
        print(f"✓ Patch 导致 {stats['behavior_change_rate']*100:.1f}% 的样本回答改变")
        print("  → 这些层影响语言生成，但方向不确定")
    else:
        print("✗ Patch 对语言生成影响很小")
        print("  → 这些层可能不被语言生成使用")
        print("  → 进一步证实语言/表征解耦")


if __name__ == '__main__':
    main()
