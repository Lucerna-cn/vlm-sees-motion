"""Day 5: 行为对照实验 - 验证 H3（语言准确率 vs 表征可解码性）"""
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import torch
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.qwen_vl import QwenVLHookManager
from behavior.question_gen import BehaviorEvaluator, generate_question, parse_answer


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


def run_behavior_evaluation(model, seq_paths, output_dir, max_samples=None):
    """
    运行行为评测

    Returns:
        list: 每个样本的评测结果
    """
    print("\n" + "="*60)
    print("行为对照实验")
    print("="*60)

    if max_samples:
        seq_paths = seq_paths[:max_samples]

    evaluator = BehaviorEvaluator(model)
    results = []

    for seq_path in tqdm(seq_paths, desc="行为评测"):
        try:
            # 对每个序列，随机选择一个球和帧进行提问
            with open(seq_path / 'trajectory.json', 'r') as f:
                traj_data = json.load(f)

            num_balls = traj_data['num_balls']

            # 随机选择一个球
            ball_id = np.random.randint(0, num_balls)

            # 使用中间帧（有前后文）
            frame_files = sorted(seq_path.glob('frame_*.png'))
            frame_idx = len(frame_files) // 2

            # 评测
            result = evaluator.evaluate_sequence(
                seq_path, ball_id=ball_id, frame_idx=frame_idx
            )
            results.append(result)

            # 定期清理
            if len(results) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"评测失败 {seq_path}: {e}")
            continue

    # 计算准确率
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / len(results) if results else 0

    print(f"\n行为评测完成:")
    print(f"  样本数: {len(results)}")
    print(f"  正确数: {correct}")
    print(f"  准确率: {accuracy:.4f}")

    # 保存结果
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'behavior_results.json', 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'num_samples': len(results),
            'num_correct': correct,
            'results': results
        }, f, indent=2, ensure_ascii=False)

    return results, accuracy


def analyze_decoupling(behavior_results, probing_summary_path=None):
    """
    分析语言准确率与表征可解码性的相关性

    如果 probing 结果可用，进行联合分析
    """
    print("\n" + "="*60)
    print("H3 验证: 语言 vs 表征 解耦分析")
    print("="*60)

    # 按风格分组统计
    style_stats = {}
    for r in behavior_results:
        seq_dir = Path(r['seq_dir'])
        style = seq_dir.parent.name  # minimal/medium/realistic

        if style not in style_stats:
            style_stats[style] = {'total': 0, 'correct': 0}

        style_stats[style]['total'] += 1
        if r['correct']:
            style_stats[style]['correct'] += 1

    print("\n按渲染风格的行为准确率:")
    for style, stats in style_stats.items():
        acc = stats['correct'] / stats['total']
        print(f"  {style}: {acc:.4f} ({stats['correct']}/{stats['total']})")

    # 分析错误案例
    errors = [r for r in behavior_results if not r['correct']]
    print(f"\n错误案例数: {len(errors)}")

    if errors:
        print("\n典型错误示例:")
        for i, err in enumerate(errors[:3]):
            print(f"\n  案例 {i+1}:")
            print(f"    问题: {err['question'][:100]}...")
            print(f"    正确答案: {err['correct_answer']}")
            print(f"    模型回答: {err['predicted_answer']}")
            print(f"    原始输出: {err['raw_response'][:100]}...")

    # 如果提供了 probing 结果，进行相关性分析
    if probing_summary_path and Path(probing_summary_path).exists():
        with open(probing_summary_path, 'r') as f:
            probing_results = json.load(f)

        # 使用最佳层的 R² 作为表征可解码性指标
        best_r2 = max(r['r2'] for r in probing_results)
        print(f"\n表征可解码性 (最佳层 R²): {best_r2:.4f}")
        print(f"语言行为准确率: {accuracy:.4f}")

        # 简单解读
        print("\n" + "-"*60)
        print("H3 验证初步结论:")
        if best_r2 < 0.5 and accuracy > 0.7:
            print("  ✓ 支持 H3: 语言表现好但表征信息弱 → 解耦现象存在")
        elif best_r2 > 0.7 and accuracy > 0.7:
            print("  ✗ 不支持 H3: 语言与表征均好 → 可能耦合")
        elif best_r2 < 0.5 and accuracy < 0.5:
            print("  ? 不确定: 语言与表征均差 → 需进一步分析")
        else:
            print(f"  ? 中间情况: 表征 R²={best_r2:.2f}, 行为准确率={accuracy:.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results_behavior')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大评测样本数')
    parser.add_argument('--probing_summary', type=str, default=None,
                       help='probing 结果路径（用于联合分析）')
    args = parser.parse_args()

    print("="*60)
    print("Day 5: 行为对照实验")
    print("="*60)

    # 加载数据
    print("\n加载数据集...")
    _, test_meta = load_split_data(args.data_dir)
    test_paths = get_sequence_paths(test_meta, args.data_dir)
    print(f"Test: {len(test_paths)} 段")

    # 初始化模型
    print("\n加载模型...")
    model = QwenVLHookManager()

    # 运行行为评测
    results, accuracy = run_behavior_evaluation(
        model, test_paths, args.output_dir,
        max_samples=args.max_samples
    )

    # 分析解耦现象
    analyze_decoupling(results, args.probing_summary)

    print(f"\n结果保存至: {args.output_dir}")


if __name__ == '__main__':
    main()
