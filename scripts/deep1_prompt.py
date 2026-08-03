"""方向 1: 提示工程干预 - 测试不同提示对物理推理的影响"""
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import torch
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.qwen_vl import QwenVLHookManager


# 定义不同的提示策略
PROMPTS = {
    "baseline": "图中{num_balls}个球正在运动。请预测编号为{ball_id}的球接下来最可能向哪个方向运动？\nA. 上\nB. 下\nC. 左\nD. 右\n\n请只回答选项字母（A/B/C/D）：",

    "physical": "请仔细分析这张物理运动场景。图中有{num_balls}个球，它们遵循物理定律运动。考虑速度、加速度和碰撞等因素。编号为{ball_id}的球接下来最可能向哪个方向运动？\nA. 上\nB. 下\nC. 左\nD. 右\n\n请基于物理分析回答选项字母：",

    "cot": "让我们一步步分析这个物理问题：\n1. 观察编号为{ball_id}的球当前的运动状态\n2. 分析其速度方向和大小\n3. 预测其接下来的运动轨迹\n\n根据以上分析，编号为{ball_id}的球接下来最可能向哪个方向运动？\nA. 上\nB. 下\nC. 左\nD. 右\n\n答案：",

    "explicit": "根据牛顿第一定律，物体在不受外力时保持匀速直线运动。观察编号为{ball_id}的球，其速度向量决定了接下来的运动方向。忽略其他因素，仅根据当前速度，该球将：\nA. 向上运动\nB. 向下运动\nC. 向左运动\nD. 向右运动\n\n选择：",

    "visual_grounding": "请先定位编号为{ball_id}的球在图像中的位置，然后观察其运动趋势。根据视觉线索（如位置变化、运动模糊等），预测该球接下来的运动方向：\nA. 上\nB. 下\nC. 左\nD. 右\n\n回答：",

    "anti_bias": "注意：请客观分析，不要有任何偏向。每个方向（上/下/左/右）的可能性应该基于实际观察，而非先验偏好。编号为{ball_id}的球接下来最可能向哪个方向运动？\nA. 上\nB. 下\nC. 左\nD. 右\n\n客观回答：",
}


def velocity_to_direction(vx, vy):
    """将速度向量转换为主要方向"""
    if abs(vx) > abs(vy):
        return 'D' if vx > 0 else 'C'  # 右/左
    else:
        return 'B' if vy > 0 else 'A'  # 下/上


def parse_answer(text):
    """从模型输出中解析选项"""
    text = text.strip().upper()
    for label in ['A', 'B', 'C', 'D']:
        if label in text:
            return label
    return 'A'


def evaluate_with_prompt(model, image, traj_data, ball_id, frame_idx, prompt_template):
    """使用指定提示评测"""
    ball = traj_data['balls'][ball_id]
    traj = ball['trajectory']

    # 获取正确答案
    if frame_idx + 1 < len(traj):
        next_state = traj[frame_idx + 1]
        correct_answer = velocity_to_direction(next_state['vx'], next_state['vy'])
    else:
        correct_answer = 'A'

    # 构建提示
    question = prompt_template.format(
        num_balls=traj_data['num_balls'],
        ball_id=ball_id
    )

    # 模型回答
    response = model.generate([image], [question], max_new_tokens=20)
    predicted_answer = parse_answer(response[0])

    return {
        'question': question,
        'correct_answer': correct_answer,
        'predicted_answer': predicted_answer,
        'raw_response': response[0],
        'correct': predicted_answer == correct_answer
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results_prompt')
    parser.add_argument('--max_samples', type=int, default=100)
    args = parser.parse_args()

    print("="*60)
    print("方向 1: 提示工程干预")
    print("="*60)
    print(f"测试 {len(PROMPTS)} 种提示策略")

    # 加载数据
    with open(Path(args.data_dir) / 'split.json', 'r') as f:
        split = json.load(f)
    test_meta = split['test']

    data_dir = Path(args.data_dir)
    test_paths = [data_dir / item['path'] for item in test_meta[:args.max_samples]]
    print(f"Test: {len(test_paths)} 段")

    # 初始化模型
    print("\n加载模型...")
    model = QwenVLHookManager()

    # 对每种提示策略进行评测
    all_results = {}

    for prompt_name, prompt_template in PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"提示策略: {prompt_name}")
        print(f"{'='*60}")

        results = []
        for seq_path in tqdm(test_paths, desc=prompt_name):
            try:
                with open(seq_path / 'trajectory.json', 'r') as f:
                    traj_data = json.load(f)

                frame_files = sorted(seq_path.glob('frame_*.png'))
                from PIL import Image
                image = Image.open(frame_files[3]).convert('RGB')

                ball_id = np.random.randint(0, traj_data['num_balls'])
                result = evaluate_with_prompt(
                    model, image, traj_data, ball_id, 3, prompt_template
                )
                result['seq_dir'] = str(seq_path)
                result['prompt_type'] = prompt_name
                results.append(result)

                if len(results) % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                continue

        # 统计
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / len(results) if results else 0

        # 选项分布
        from collections import Counter
        pred_counts = Counter(r['predicted_answer'] for r in results)

        all_results[prompt_name] = {
            'accuracy': accuracy,
            'num_samples': len(results),
            'num_correct': correct,
            'option_distribution': dict(pred_counts),
            'results': results
        }

        print(f"\n结果:")
        print(f"  准确率: {accuracy:.4f} ({correct}/{len(results)})")
        print(f"  选项分布: A={pred_counts.get('A',0)}, B={pred_counts.get('B',0)}, "
              f"C={pred_counts.get('C',0)}, D={pred_counts.get('D',0)}")

    # 汇总对比
    print("\n" + "="*60)
    print("提示策略对比汇总")
    print("="*60)
    print(f"{'策略':<20} {'准确率':<10} {'C选项占比':<12}")
    print("-"*45)

    for name, data in all_results.items():
        c_count = data['option_distribution'].get('C', 0)
        c_pct = c_count / data['num_samples'] * 100
        print(f"{name:<20} {data['accuracy']:<10.4f} {c_pct:<12.1f}%")

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 保存汇总
    summary = {name: {
        'accuracy': data['accuracy'],
        'num_samples': data['num_samples'],
        'option_distribution': data['option_distribution']
    } for name, data in all_results.items()}

    with open(output_dir / 'prompt_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # 保存详细结果
    for name, data in all_results.items():
        with open(output_dir / f'prompt_{name}.json', 'w') as f:
            json.dump(data['results'], f, indent=2, ensure_ascii=False)

    print(f"\n结果保存至: {output_dir}")

    # 关键发现
    print("\n" + "="*60)
    print("关键发现")
    print("="*60)

    best_prompt = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    baseline_acc = all_results['baseline']['accuracy']

    print(f"最佳策略: {best_prompt[0]} (准确率 {best_prompt[1]['accuracy']:.4f})")
    print(f"基线准确率: {baseline_acc:.4f}")
    print(f"提升: {(best_prompt[1]['accuracy'] - baseline_acc)*100:.1f}%")

    if best_prompt[1]['accuracy'] > baseline_acc + 0.1:
        print("\n✓ 提示工程显著改善物理推理！")
        print("  → 模型'有能力但未被激活'")
    elif best_prompt[1]['accuracy'] < baseline_acc + 0.05:
        print("\n✗ 提示工程几乎无效")
        print("  → 确认'功能性断层'，模型确实无法使用速度信息")


if __name__ == '__main__':
    main()
