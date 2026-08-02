"""行为对照：选项式问题生成与评测"""
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


DIRECTION_OPTIONS = ['上', '下', '左', '右']
DIRECTION_LABELS = ['A', 'B', 'C', 'D']


def velocity_to_direction(vx: float, vy: float) -> str:
    """
    将速度向量转换为主要方向

    注意：图像坐标系 y 轴向下为正
    """
    if abs(vx) > abs(vy):
        return '右' if vx > 0 else '左'
    else:
        return '下' if vy > 0 else '上'


def generate_question(traj_data: Dict, ball_id: int, frame_idx: int) -> Tuple[str, str]:
    """
    为指定球和帧生成选项式问题

    Returns:
        (question_text, correct_answer)
    """
    num_balls = traj_data['num_balls']
    ball = traj_data['balls'][ball_id]
    traj = ball['trajectory']

    # 获取当前帧和下一帧速度
    if frame_idx + 1 >= len(traj):
        frame_idx = len(traj) - 2

    next_state = traj[frame_idx + 1]
    vx, vy = next_state['vx'], next_state['vy']

    correct_direction = velocity_to_direction(vx, vy)
    correct_label = DIRECTION_LABELS[DIRECTION_OPTIONS.index(correct_direction)]

    question = (
        f"图中{num_balls}个球正在运动。请预测编号为{ball_id}的球接下来最可能向哪个方向运动？\n"
        f"A. 上\nB. 下\nC. 左\nD. 右\n\n"
        f"请只回答选项字母（A/B/C/D）："
    )

    return question, correct_label


def parse_answer(text: str) -> str:
    """从模型输出中解析选项"""
    text = text.strip().upper()

    # 直接匹配
    for label in DIRECTION_LABELS:
        if label in text:
            return label

    # 匹配中文
    for i, opt in enumerate(DIRECTION_OPTIONS):
        if opt in text:
            return DIRECTION_LABELS[i]

    # 默认返回 A
    return 'A'


class BehaviorEvaluator:
    """行为评测器"""

    def __init__(self, model_manager):
        self.model = model_manager
        self.results = []

    def evaluate_sequence(self, seq_dir: Path, ball_id: int = 0, frame_idx: int = 3):
        """
        评测单个序列

        Args:
            seq_dir: 序列目录
            ball_id: 要询问的球编号
            frame_idx: 使用的帧编号（默认中间帧）
        """
        seq_dir = Path(seq_dir)

        # 加载轨迹
        with open(seq_dir / 'trajectory.json', 'r') as f:
            traj_data = json.load(f)

        # 加载图像
        frame_files = sorted(seq_dir.glob('frame_*.png'))
        if frame_idx >= len(frame_files):
            frame_idx = len(frame_files) - 1

        from PIL import Image
        image = Image.open(frame_files[frame_idx]).convert('RGB')

        # 生成问题
        question, correct_answer = generate_question(traj_data, ball_id, frame_idx)

        # 模型回答
        response = self.model.generate([image], [question], max_new_tokens=10)
        predicted_answer = parse_answer(response[0])

        # 记录结果
        result = {
            'seq_dir': str(seq_dir),
            'ball_id': ball_id,
            'frame_idx': frame_idx,
            'question': question,
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'raw_response': response[0],
            'correct': predicted_answer == correct_answer
        }

        self.results.append(result)
        return result

    def evaluate_batch(self, seq_dirs: List[Path], **kwargs):
        """批量评测"""
        for seq_dir in seq_dirs:
            try:
                self.evaluate_sequence(seq_dir, **kwargs)
            except Exception as e:
                print(f"评测失败 {seq_dir}: {e}")

        return self.results

    def compute_accuracy(self):
        """计算准确率"""
        if not self.results:
            return 0.0
        correct = sum(1 for r in self.results if r['correct'])
        return correct / len(self.results)

    def save_results(self, output_path: Path):
        """保存结果"""
        with open(output_path, 'w') as f:
            json.dump({
                'accuracy': self.compute_accuracy(),
                'num_samples': len(self.results),
                'results': self.results
            }, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    # 测试问题生成
    traj_data = {
        'num_balls': 2,
        'balls': [{
            'id': 0,
            'trajectory': [
                {'x': 100, 'y': 100, 'vx': 50, 'vy': 10},
                {'x': 105, 'y': 101, 'vx': 50, 'vy': 10},
            ]
        }]
    }

    q, a = generate_question(traj_data, 0, 0)
    print(f"问题: {q}")
    print(f"正确答案: {a}")

    # 测试答案解析
    print(f"解析 'A': {parse_answer('A')}")
    print(f"解析 '答案是B': {parse_answer('答案是B')}")
    print(f"解析 '向右运动': {parse_answer('向右运动')}")
