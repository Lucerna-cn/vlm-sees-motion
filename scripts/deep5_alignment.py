"""方向 5: 跨模态对齐分析 - 理解视觉-语言对齐机制"""
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_similarity
import gc
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.qwen_vl import QwenVLHookManager
from models.patch_utils import PatchLocator
def analyze_grounding_vs_physics(model, locator, seq_paths, max_samples=100):
    """
    分析 grounding（位置理解）与物理推理的差异
    检查模型是否能正确定位球，但无法预测运动
    """
    print("\n" + "="*60)
    print("Grounding vs 物理推理分析")
    print("="*60)
    results = []
    for seq_path in tqdm(seq_paths[:max_samples], desc="分析中"):
        try:
            with open(seq_path / 'trajectory.json', 'r') as f:
                traj_data = json.load(f)
            frame_files = sorted(seq_path.glob('frame_*.png'))
            if len(frame_files) < 3:
                continue
            frame_idx = 3
            from PIL import Image
            image = Image.open(frame_files[frame_idx]).convert('RGB')
            # 测试 1: Grounding（位置理解）
            ball_id = 0
            ball = traj_data['balls'][ball_id]
            traj = ball['trajectory']
            state = traj[frame_idx]
            grounding_prompt = f"请描述编号为{ball_id}的球在图像中的位置。"
            grounding_response = model.generate([image], [grounding_prompt], max_new_tokens=50)
            # 测试 2: 物理推理（运动预测）
            physics_prompt = f"图中{traj_data['num_balls']}个球正在运动。请预测编号为{ball_id}的球接下来最可能向哪个方向运动？\nA. 上\nB. 下\nC. 左\nD. 右\n\n请只回答选项字母："
            physics_response = model.generate([image], [physics_prompt], max_new_tokens=10)
            # 分析回答
            # Grounding: 检查是否提到位置相关信息
            grounding_mentions_position = any(word in grounding_response[0] for word in ['位置', '上', '下', '左', '右', '中心', '边缘'])
            # Physics: 解析选项
            physics_answer = 'A'
            for label in ['A', 'B', 'C', 'D']:
                if label in physics_response[0]:
                    physics_answer = label
                    break
            # 正确答案
            vx, vy = state['vx'], state['vy']
            if abs(vx) > abs(vy):
                correct_answer = 'D' if vx > 0 else 'C'
            else:
                correct_answer = 'B' if vy > 0 else 'A'
            result = {
                'seq_dir': str(seq_path),
                'grounding_response': grounding_response[0],
                'grounding_mentions_position': grounding_mentions_position,
                'physics_response': physics_response[0],
                'physics_answer': physics_answer,
                'correct_answer': correct_answer,
                'physics_correct': physics_answer == correct_answer,
                'ball_position': {'x': state['x'], 'y': state['y']},
                'ball_velocity': {'vx': vx, 'vy': vy}
            }
            results.append(result)
        except Exception as e:
            continue
    # 统计分析
    print("\n" + "="*60)
    print("分析结果")
    print("="*60)
    grounding_success = sum(1 for r in results if r['grounding_mentions_position'])
    physics_success = sum(1 for r in results if r['physics_correct'])
    print(f"\nGrounding 成功率: {grounding_success}/{len(results)} ({grounding_success/len(results)*100:.1f}%)")
    print(f"物理推理成功率: {physics_success}/{len(results)} ({physics_success/len(results)*100:.1f}%)")
    # 分析 grounding 正确但物理错误的案例
    grounding_ok_physics_fail = [r for r in results if r['grounding_mentions_position'] and not r['physics_correct']]
    print(f"\nGrounding 正确但物理推理错误: {len(grounding_ok_physics_fail)}/{len(results)} ({len(grounding_ok_physics_fail)/len(results)*100:.1f}%)")
    if grounding_ok_physics_fail:
        print("\n典型案例:")
        for i, r in enumerate(grounding_ok_physics_fail[:3]):
            print(f"\n  案例 {i+1}:")
            print(f"    Grounding: {r['grounding_response'][:60]}...")
            print(f"    物理回答: {r['physics_answer']} (正确: {r['correct_answer']})")
    return {
        'grounding_success_rate': grounding_success / len(results) if results else 0,
        'physics_success_rate': physics_success / len(results) if results else 0,
        'grounding_ok_physics_fail_rate': len(grounding_ok_physics_fail) / len(results) if results else 0,
        'results': results
    }
def analyze_token_alignment(model, locator, seq_paths, max_samples=50):
    """
    分析视觉 token 与语言 token 的对齐方式
    检查视觉表征如何映射到语言输出
    """
    print("\n" + "="*60)
    print("视觉-语言 Token 对齐分析")
    print("="*60)
    print("\n分析视觉 token 与方向词汇的关联...")
    # 收集视觉 token 和对应的语言输出
    alignment_data = []
    for seq_path in tqdm(seq_paths[:max_samples], desc="对齐分析"):
        try:
            with open(seq_path / 'trajectory.json', 'r') as f:
                traj_data = json.load(f)
            frame_files = sorted(seq_path.glob('frame_*.png'))
            if len(frame_files) < 3:
                continue
            frame_idx = 3
            from PIL import Image
            image = Image.open(frame_files[frame_idx]).convert('RGB')
            # 获取视觉表征
            results = model.forward_with_hooks([image], layer_indices=[16])
            if 16 not in results['layer_outputs']:
                continue
            hidden = results['layer_outputs'][16][0]
            # 获取球的位置
            ball = traj_data['balls'][0]
            traj = ball['trajectory']
            state = traj[frame_idx]
            ball_positions = [(state['x'], state['y'], state['radius'])]
            # 提取球的 token
            grid_thw = model.get_grid_info(results['inputs'])
            ball_tokens, _ = locator.extract_ball_tokens(
                hidden, ball_positions,
                image.width, image.height,
                grid_thw[0]
            )
            # 获取球的视觉表征
            ball_repr = ball_tokens[0].mean(axis=0).numpy()
            # 生成方向描述
            direction_prompt = "球的运动方向是"
            direction_response = model.generate([image], [direction_prompt], max_new_tokens=5)
            # 记录
            alignment_data.append({
                'visual_repr_norm': float(np.linalg.norm(ball_repr)),
                'direction_response': direction_response[0],
                'true_vx': state['vx'],
                'true_vy': state['vy']
            })
            model.clear_layer_outputs()
        except Exception as e:
            continue
    print(f"\n收集 {len(alignment_data)} 个对齐样本")
    # 分析视觉表征与语言输出的关系
    if alignment_data:
        # 检查视觉表征范数与回答的关系
        norms = [d['visual_repr_norm'] for d in alignment_data]
        print(f"视觉表征范数: 均值 {np.mean(norms):.4f}, 标准差 {np.std(norms):.4f}")
        # 检查回答中的方向词
        direction_words = {'上': 0, '下': 0, '左': 0, '右': 0}
        for d in alignment_data:
            for word in direction_words:
                if word in d['direction_response']:
                    direction_words[word] += 1
        print(f"方向词出现次数: {direction_words}")
    return {
        'n_samples': len(alignment_data),
        'direction_words': direction_words if alignment_data else {}
    }
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results_alignment')
    parser.add_argument('--max_samples', type=int, default=100)
    args = parser.parse_args()
    print("="*60)
    print("方向 5: 跨模态对齐分析")
    print("="*60)
    # 加载数据
    with open(Path(args.data_dir) / 'split.json', 'r') as f:
        split = json.load(f)
    test_meta = split['test']
    data_dir = Path(args.data_dir)
    test_paths = [data_dir / item['path'] for item in test_meta]
    print(f"Test: {len(test_paths)} 段")
    # 初始化模型
    print("\n加载模型...")
    model = QwenVLHookManager()
    locator = PatchLocator()
    # 分析结果
    results = {}
    # 1. Grounding vs 物理推理
    results['grounding_vs_physics'] = analyze_grounding_vs_physics(
        model, locator, test_paths, max_samples=args.max_samples
    )
    # 2. Token 对齐分析
    results['token_alignment'] = analyze_token_alignment(
        model, locator, test_paths, max_samples=50
    )
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    # 保存汇总（不包含详细结果）
    summary = {
        'grounding_success_rate': results['grounding_vs_physics']['grounding_success_rate'],
        'physics_success_rate': results['grounding_vs_physics']['physics_success_rate'],
        'grounding_ok_physics_fail_rate': results['grounding_vs_physics']['grounding_ok_physics_fail_rate'],
        'token_alignment': results['token_alignment']
    }
    with open(output_dir / 'alignment_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)
    # 关键发现
    print("\n" + "="*60)
    print("关键发现")
    print("="*60)
    grounding_rate = results['grounding_vs_physics']['grounding_success_rate']
    physics_rate = results['grounding_vs_physics']['physics_success_rate']
    gap_rate = results['grounding_vs_physics']['grounding_ok_physics_fail_rate']
    print(f"\n1. Grounding（位置理解）: {grounding_rate*100:.1f}%")
    print(f"2. 物理推理（运动预测）: {physics_rate*100:.1f}%")
    print(f"3. Grounding 正确但物理错误: {gap_rate*100:.1f}%")
    if gap_rate > 0.5:
        print("\n✓ 核心发现: 模型能理解位置，但无法推理运动")
        print("  → 视觉-语言对齐**选择性**存在问题")
        print("  → 位置信息对齐良好，速度信息对齐失败")
    elif physics_rate > grounding_rate:
        print("\n✓ 物理推理优于 Grounding")
        print("  → 模型更擅长推理而非描述")
    else:
        print("\n? 两种能力相当")
    print(f"\n结果保存至: {output_dir}")
if __name__ == '__main__':
    main()
