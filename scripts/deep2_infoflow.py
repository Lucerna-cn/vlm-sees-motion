"""方向 2: 信息流分析 - 理解提示工程如何激活潜在能力"""
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.qwen_vl import QwenVLHookManager


PROMPTS = {
    "baseline": "图中{num_balls}个球正在运动。请预测编号为{ball_id}的球接下来最可能向哪个方向运动？\nA. 上\nB. 下\nC. 左\nD. 右\n\n请只回答选项字母（A/B/C/D）：",
    "physical": "请仔细分析这张物理运动场景。图中有{num_balls}个球，它们遵循物理定律运动。考虑速度、加速度和碰撞等因素。编号为{ball_id}的球接下来最可能向哪个方向运动？\nA. 上\nB. 下\nC. 左\nD. 右\n\n请基于物理分析回答选项字母：",
}


class AttentionAnalyzer:
    """注意力流分析器"""

    def __init__(self, model):
        self.model = model
        self.attention_maps = []

    def register_attention_hooks(self):
        """注册注意力 hook"""
        self.attention_maps = []

        def attention_hook(module, input, output):
            # output 包含 attention weights
            if hasattr(output, 'attentions') and output.attentions is not None:
                self.attention_maps.append(output.attentions.detach().cpu())

        # 对语言模型的每一层注册 hook
        # 注意：这里简化处理，实际需要根据模型结构调整
        pass

    def analyze_attention_pattern(self, image, question, target_tokens=['A', 'B', 'C', 'D']):
        """
        分析生成答案时对视觉 token 的注意力

        Returns:
            attention_to_visual: 对视觉 token 的平均注意力
            attention_to_text: 对文本 token 的平均注意力
        """
        # 使用 forward_with_hooks 获取中间表征
        results = self.model.forward_with_hooks([image], [question], layer_indices=[0, 16, 31])

        # 获取输入的 token 类型（视觉 vs 文本）
        inputs = results['inputs']

        # 简化分析：比较不同提示下的表征差异
        return results


def compute_representation_similarity(model, image, prompt1, prompt2):
    """
    计算两种提示下表征的相似性

    如果提示工程激活了潜在能力，两种提示的表征应该有显著差异
    """
    results1 = model.forward_with_hooks([image], [prompt1], layer_indices=[16])
    results2 = model.forward_with_hooks([image], [prompt2], layer_indices=[16])

    # 获取 Layer 16 的表征（转换为 float32 以支持 numpy）
    repr1 = results1['layer_outputs'][16][0].float().numpy() if 16 in results1['layer_outputs'] else None
    repr2 = results2['layer_outputs'][16][0].float().numpy() if 16 in results2['layer_outputs'] else None

    if repr1 is None or repr2 is None:
        return None

    # 计算余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity

    # 展平
    flat1 = repr1.flatten().reshape(1, -1)
    flat2 = repr2.flatten().reshape(1, -1)

    similarity = cosine_similarity(flat1, flat2)[0, 0]

    # 计算差异
    diff = np.abs(repr1 - repr2).mean()

    return {
        'cosine_similarity': float(similarity),
        'mean_absolute_difference': float(diff),
        'repr1_norm': float(np.linalg.norm(repr1)),
        'repr2_norm': float(np.linalg.norm(repr2))
    }


def analyze_logit_lens(model, image, question, target_words=['上', '下', '左', '右']):
    """
    Logit Lens 分析：追踪中间层到输出词汇的映射

    检查在生成答案时，模型对方向词汇的预测如何变化
    """
    # 这里简化实现，实际需要更复杂的 logit lens 分析
    # 返回生成结果
    response = model.generate([image], [question], max_new_tokens=10)
    return response[0]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results_infoflow')
    parser.add_argument('--max_samples', type=int, default=20)
    args = parser.parse_args()

    print("="*60)
    print("方向 2: 信息流分析")
    print("="*60)

    # 加载数据
    with open(Path(args.data_dir) / 'split.json', 'r') as f:
        split = json.load(f)
    test_meta = split['test']

    data_dir = Path(args.data_dir)
    test_paths = [data_dir / item['path'] for item in test_meta[:args.max_samples]]
    print(f"分析样本: {len(test_paths)} 段")

    # 初始化模型
    print("\n加载模型...")
    model = QwenVLHookManager()

    # 分析结果
    analysis_results = []

    print("\n分析提示对表征的影响...")
    for seq_path in tqdm(test_paths, desc="分析中"):
        try:
            with open(seq_path / 'trajectory.json', 'r') as f:
                traj_data = json.load(f)

            frame_files = sorted(seq_path.glob('frame_*.png'))
            from PIL import Image
            image = Image.open(frame_files[3]).convert('RGB')

            ball_id = 0  # 简化：只分析第一个球

            # 构建两种提示
            baseline_prompt = PROMPTS['baseline'].format(
                num_balls=traj_data['num_balls'], ball_id=ball_id
            )
            physical_prompt = PROMPTS['physical'].format(
                num_balls=traj_data['num_balls'], ball_id=ball_id
            )

            # 分析表征相似性
            similarity = compute_representation_similarity(
                model, image, baseline_prompt, physical_prompt
            )

            # 生成回答
            baseline_response = analyze_logit_lens(model, image, baseline_prompt)
            physical_response = analyze_logit_lens(model, image, physical_prompt)

            result = {
                'seq_dir': str(seq_path),
                'similarity': similarity,
                'baseline_response': baseline_response,
                'physical_response': physical_response,
                'responses_differ': baseline_response != physical_response
            }

            analysis_results.append(result)

        except Exception as e:
            print(f"分析失败 {seq_path}: {e}")
            continue

    # 统计分析
    print("\n" + "="*60)
    print("信息流分析结果")
    print("="*60)

    # 表征相似性统计
    similarities = [r['similarity']['cosine_similarity'] for r in analysis_results if r['similarity']]
    diffs = [r['similarity']['mean_absolute_difference'] for r in analysis_results if r['similarity']]

    if similarities:
        print(f"\n表征相似性 (baseline vs physical):")
        print(f"  余弦相似度: {np.mean(similarities):.4f} ± {np.std(similarities):.4f}")
        print(f"  平均差异: {np.mean(diffs):.4f} ± {np.std(diffs):.4f}")

        if np.mean(similarities) > 0.95:
            print("  → 两种提示下表征几乎相同")
        elif np.mean(similarities) > 0.8:
            print("  → 两种提示下表征相似但有差异")
        else:
            print("  → 两种提示下表征显著不同")

    # 回答差异统计
    differ_count = sum(1 for r in analysis_results if r['responses_differ'])
    print(f"\n回答差异:")
    print(f"  不同回答: {differ_count}/{len(analysis_results)} ({differ_count/len(analysis_results)*100:.1f}%)")

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'infoflow_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)

    print(f"\n结果保存至: {output_dir}")

    # 关键发现
    print("\n" + "="*60)
    print("关键发现")
    print("="*60)

    if np.mean(similarities) > 0.9 and differ_count > len(analysis_results) * 0.5:
        print("✓ 表征相似但回答不同")
        print("  → 提示工程主要影响**解码阶段**而非编码阶段")
        print("  → 视觉表征中的信息已足够，关键在于如何'读取'")
    elif np.mean(similarities) < 0.8:
        print("✓ 表征显著不同")
        print("  → 提示工程影响**编码阶段**")
        print("  → 不同提示导致不同的视觉信息提取")


if __name__ == '__main__':
    main()
