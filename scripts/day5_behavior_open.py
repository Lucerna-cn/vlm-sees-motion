"""Day 5 补充: 开放式问题行为评测"""
import sys
from pathlib import Path
import json
import re
import numpy as np
from tqdm import tqdm
import torch
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.qwen_vl import QwenVLHookManager


def velocity_to_direction(vx, vy):
    """将速度向量转换为方向描述"""
    if abs(vx) > abs(vy):
        return '右' if vx > 0 else '左'
    else:
        return '下' if vy > 0 else '上'


def parse_open_answer(text):
    """从开放式回答中解析方向"""
    text = text.strip()

    # 直接匹配方向词
    directions = {
        '上': ['上', '向上', '上方', 'up', 'upward'],
        '下': ['下', '向下', '下方', 'down', 'downward'],
        '左': ['左', '向左', '左侧', 'left'],
        '右': ['右', '向右', '右侧', 'right']
    }

    text_lower = text.lower()
    for direction, keywords in directions.items():
        for kw in keywords:
            if kw in text_lower:
                return direction

    return '未知'


def evaluate_open_ended(model, image, traj_data, ball_id, frame_idx):
    """开放式问题评测"""
    ball = traj_data['balls'][ball_id]
    traj = ball['trajectory']

    # 获取正确答案
    if frame_idx + 1 < len(traj):
        next_state = traj[frame_idx + 1]
        correct_direction = velocity_to_direction(next_state['vx'], next_state['vy'])
    else:
        correct_direction = '未知'

    # 开放式问题
    question = f"图中{traj_data['num_balls']}个球正在运动。请描述编号为{ball_id}的球接下来的运动方向。"

    # 模型回答
    response = model.generate([image], [question], max_new_tokens=50)
    predicted_direction = parse_open_answer(response[0])

    return {
        'question': question,
        'correct_answer': correct_direction,
        'predicted_answer': predicted_direction,
        'raw_response': response[0],
        'correct': predicted_direction == correct_direction
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results_behavior_open')
    parser.add_argument('--max_samples', type=int, default=200)
    args = parser.parse_args()

    print("="*60)
    print("开放式问题行为评测")
    print("="*60)

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

    # 评测
    results = []
    for seq_path in tqdm(test_paths, desc="开放式评测"):
        try:
            with open(seq_path / 'trajectory.json', 'r') as f:
                traj_data = json.load(f)

            frame_files = sorted(seq_path.glob('frame_*.png'))
            from PIL import Image
            image = Image.open(frame_files[3]).convert('RGB')

            ball_id = np.random.randint(0, traj_data['num_balls'])
            result = evaluate_open_ended(model, image, traj_data, ball_id, 3)
            result['seq_dir'] = str(seq_path)
            results.append(result)

            if len(results) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"评测失败 {seq_path}: {e}")

    # 统计
    correct = sum(1 for r in results if r['correct'])
    unknown = sum(1 for r in results if r['predicted_answer'] == '未知')

    print(f"\n开放式评测完成:")
    print(f"  样本数: {len(results)}")
    print(f"  正确: {correct} ({correct/len(results)*100:.1f}%)")
    print(f"  无法解析: {unknown} ({unknown/len(results)*100:.1f}%)")

    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'open_results.json', 'w') as f:
        json.dump({
            'accuracy': correct / len(results),
            'unknown_rate': unknown / len(results),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n结果保存至: {output_dir}")


if __name__ == '__main__':
    main()
