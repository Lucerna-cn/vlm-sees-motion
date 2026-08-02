"""分析模型回答偏向：为何总选 A/C"""
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np


def analyze_answer_bias(behavior_results_path):
    """分析模型回答的偏向模式"""

    with open(behavior_results_path, 'r') as f:
        data = json.load(f)

    results = data['results']

    print("="*60)
    print("模型回答偏向分析")
    print("="*60)

    # 1. 总体分布
    print("\n【1. 回答选项分布】")
    predicted = [r['predicted_answer'] for r in results]
    correct = [r['correct_answer'] for r in results]

    pred_counts = Counter(predicted)
    correct_counts = Counter(correct)

    print("\n模型预测分布:")
    for label in ['A', 'B', 'C', 'D']:
        count = pred_counts.get(label, 0)
        pct = count / len(results) * 100
        print(f"  {label}: {count:3d} ({pct:5.1f}%)")

    print("\n正确答案分布:")
    for label in ['A', 'B', 'C', 'D']:
        count = correct_counts.get(label, 0)
        pct = count / len(results) * 100
        print(f"  {label}: {count:3d} ({pct:5.1f}%)")

    # 2. 混淆矩阵
    print("\n【2. 混淆矩阵】")
    print("        预测")
    print("      A   B   C   D")
    labels = ['A', 'B', 'C', 'D']
    confusion = {c: {p: 0 for p in labels} for c in labels}

    for r in results:
        confusion[r['correct_answer']][r['predicted_answer']] += 1

    for true_label in labels:
        row = [f"{confusion[true_label][p]:3d}" for p in labels]
        print(f"真实 {true_label}: {' '.join(row)}")

    # 3. 按风格分析
    print("\n【3. 按渲染风格的回答偏向】")
    style_bias = {}
    for r in results:
        style = Path(r['seq_dir']).parent.name
        if style not in style_bias:
            style_bias[style] = {'predicted': [], 'correct': []}
        style_bias[style]['predicted'].append(r['predicted_answer'])
        style_bias[style]['correct'].append(r['correct_answer'])

    for style, data in style_bias.items():
        pred_counts = Counter(data['predicted'])
        total = len(data['predicted'])
        print(f"\n{style}:")
        for label in ['A', 'B', 'C', 'D']:
            count = pred_counts.get(label, 0)
            pct = count / total * 100
            print(f"  {label}: {pct:5.1f}%", end="")
        print()

    # 4. 原始输出分析
    print("\n【4. 原始输出模式】")
    raw_responses = [r['raw_response'].strip() for r in results]

    # 检查是否只输出字母
    letter_only = sum(1 for r in raw_responses if len(r) == 1 and r in 'ABCD')
    print(f"只输出字母: {letter_only}/{len(raw_responses)} ({letter_only/len(raw_responses)*100:.1f}%)")

    # 常见完整输出
    response_counts = Counter(raw_responses)
    print("\n最常见输出:")
    for resp, count in response_counts.most_common(5):
        print(f"  '{resp}': {count} 次")

    # 5. 偏向原因假设
    print("\n【5. 偏向原因分析】")

    # 检查是否与位置相关
    position_bias = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    for r in results:
        # 从 seq_dir 推断不了位置，需要额外信息
        pass

    # 检查是否与问题中的数字相关
    num_balls_bias = {}
    for r in results:
        # 从问题中提取球的数量
        q = r['question']
        if '1个球' in q:
            n = 1
        elif '2个球' in q:
            n = 2
        elif '3个球' in q:
            n = 3
        else:
            continue

        if n not in num_balls_bias:
            num_balls_bias[n] = []
        num_balls_bias[n].append(r['predicted_answer'])

    print("\n按球数量的回答分布:")
    for n, preds in sorted(num_balls_bias.items()):
        counts = Counter(preds)
        print(f"  {n}个球: A={counts.get('A',0)}, B={counts.get('B',0)}, "
              f"C={counts.get('C',0)}, D={counts.get('D',0)}")

    # 6. 关键发现
    print("\n" + "="*60)
    print("【关键发现】")
    print("="*60)

    a_pct = pred_counts.get('A', 0) / len(results) * 100
    c_pct = pred_counts.get('C', 0) / len(results) * 100

    if a_pct + c_pct > 70:
        print(f"✓ 模型极度偏向 A ({a_pct:.1f}%) 和 C ({c_pct:.1f}%)，合计 {a_pct+c_pct:.1f}%")
        print("  可能原因:")
        print("  - 训练数据中的选项顺序偏向")
        print("  - 模型对'上'和'左'有先验偏好")
        print("  - 视觉特征与语言生成的映射错误")
    else:
        print("模型回答分布相对均匀")

    if pred_counts.get('A', 0) > len(results) * 0.4:
        print(f"✓ 极端偏向 A: {pred_counts.get('A', 0)}/{len(results)} 次")
        print("  建议: 检查是否所有问题都返回相同答案")

    return {
        'predicted_distribution': dict(pred_counts),
        'correct_distribution': dict(correct_counts),
        'confusion_matrix': confusion,
        'style_bias': {k: dict(Counter(v['predicted'])) for k, v in style_bias.items()}
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('results_path', type=str, help='行为评测结果 JSON 文件路径')
    args = parser.parse_args()

    analyze_answer_bias(args.results_path)
