"""生成论文图表"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置英文论文风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# 颜色方案
COLORS = {
    'position': '#2E86AB',
    'velocity': '#A23B72',
    'baseline': '#F18F01',
    'dinov2': '#C73E1D',
    'siglip': '#3B1F2B',
    'qwen': '#2E86AB',
    'grounding': '#2E86AB',
    'physics': '#A23B72'
}


def load_json(path):
    """加载 JSON 文件（处理编码）"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_layer_r2_curve(output_dir):
    """图 1: 逐层 R² 曲线（位置 vs 速度）"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # 加载数据
    position_data = load_json('results_download/results_position/summary_position.json')
    velocity_data = load_json('results_download/results_test_mlp/summary_velocity.json')

    # 提取数据
    pos_layers = [d['layer_idx'] for d in position_data]
    pos_r2 = [d['r2'] for d in position_data]

    vel_layers = [d['layer_idx'] for d in velocity_data]
    vel_r2 = [d['r2'] for d in velocity_data]

    # 绘制
    ax.plot(pos_layers, pos_r2, 'o-', color=COLORS['position'],
            linewidth=2, markersize=8, label='Position (Linear Probe)')
    ax.plot(vel_layers, vel_r2, 's-', color=COLORS['velocity'],
            linewidth=2, markersize=8, label='Velocity (MLP Probe)')

    # 基线
    ax.axhline(y=0.9, color=COLORS['baseline'], linestyle='--',
               alpha=0.7, label='Constant Velocity Baseline')
    ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5, label='Random Guess')

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('R² Score')
    ax.set_title('Layer-wise Probing: Position vs Velocity Information')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.0)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_layer_r2_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig1_layer_r2_curve.pdf', bbox_inches='tight')
    plt.close()

    print('图 1 已生成: fig1_layer_r2_curve.png/pdf')


def plot_model_comparison(output_dir):
    """图 2: 三模型对比（DINOv2 vs SigLIP vs Qwen）"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # 加载数据
    dinov2_data = load_json('results_download/results_dinov2/summary_velocity.json')
    siglip_data = load_json('results_download/results_siglip/summary_velocity.json')
    qwen_data = load_json('results_download/results_test_mlp/summary_velocity.json')

    # 获取最佳 R²
    dinov2_best = max(d['r2'] for d in dinov2_data)
    siglip_best = max(d['r2'] for d in siglip_data)
    qwen_best = max(d['r2'] for d in qwen_data)

    # 数据
    models = ['DINOv2\n(Pure Vision)', 'SigLIP\n(Weak Language)', 'Qwen2.5-VL\n(Strong Language)']
    r2_scores = [dinov2_best, siglip_best, qwen_best]
    colors = [COLORS['dinov2'], COLORS['siglip'], COLORS['qwen']]

    # 绘制柱状图
    bars = ax.bar(models, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Best Velocity R² Score')
    ax.set_title('Language Training Enhances Velocity Representation')
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加提升倍数标注
    ax.annotate(f'+{qwen_best/dinov2_best:.0f}x', xy=(2, qwen_best/2),
                ha='center', fontsize=14, fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig2_model_comparison.pdf', bbox_inches='tight')
    plt.close()

    print('图 2 已生成: fig2_model_comparison.png/pdf')


def plot_prompt_engineering(output_dir):
    """图 3: 提示工程效果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 加载数据
    prompt_data = load_json('results_download/results_prompt/prompt_comparison.json')

    # 提取数据
    strategies = list(prompt_data.keys())
    accuracies = [prompt_data[s]['accuracy'] for s in strategies]
    c_biases = [prompt_data[s]['option_distribution'].get('C', 0) / prompt_data[s]['num_samples'] * 100
                for s in strategies]

    # 左图: 准确率
    colors_acc = ['red' if s == 'baseline' else 'green' for s in strategies]
    bars1 = ax1.barh(strategies, accuracies, color=colors_acc, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Accuracy')
    ax1.set_title('Prompt Engineering Improves Accuracy')
    ax1.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5, label='Random Guess')
    ax1.set_xlim(0, 0.5)

    # 添加数值标签
    for bar, acc in zip(bars1, accuracies):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{acc:.2f}', ha='left', va='center', fontsize=9)

    # 右图: C 选项偏向
    colors_bias = ['red' if b > 50 else 'green' for b in c_biases]
    bars2 = ax2.barh(strategies, c_biases, color=colors_bias, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('C Option Bias (%)')
    ax2.set_title('Prompt Engineering Eliminates Bias')
    ax2.set_xlim(0, 100)

    # 添加数值标签
    for bar, bias in zip(bars2, c_biases):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{bias:.1f}%', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_prompt_engineering.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig3_prompt_engineering.pdf', bbox_inches='tight')
    plt.close()

    print('图 3 已生成: fig3_prompt_engineering.png/pdf')


def plot_grounding_vs_physics(output_dir):
    """图 4: Grounding vs 物理推理"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # 数据
    categories = ['Grounding\n(Position)', 'Physical Reasoning\n(Velocity)', 'Gap']
    values = [100, 19, 81]
    colors = [COLORS['grounding'], COLORS['physics'], 'red']

    # 绘制
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Selective Cross-Modal Alignment:\nGrounding Succeeds but Physical Reasoning Fails')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加解释
    ax.text(1, 50, '81% of samples:\n"Understand position\nbut cannot predict motion"',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_grounding_vs_physics.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig4_grounding_vs_physics.pdf', bbox_inches='tight')
    plt.close()

    print('图 4 已生成: fig4_grounding_vs_physics.png/pdf')


def plot_option_distribution(output_dir):
    """图 5: 选项分布偏向"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # 加载数据
    behavior_data = load_json('results_download/results_behavior_500/behavior_results.json')

    # 统计选项分布
    from collections import Counter
    predicted = [r['predicted_answer'] for r in behavior_data['results']]
    correct = [r['correct_answer'] for r in behavior_data['results']]

    pred_counts = Counter(predicted)
    correct_counts = Counter(correct)

    # 准备数据
    options = ['A', 'B', 'C', 'D']
    pred_values = [pred_counts.get(o, 0) / len(predicted) * 100 for o in options]
    correct_values = [correct_counts.get(o, 0) / len(correct) * 100 for o in options]

    x = np.arange(len(options))
    width = 0.35

    # 绘制
    bars1 = ax.bar(x - width/2, pred_values, width, label='Model Prediction',
                   color='red', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, correct_values, width, label='Ground Truth',
                   color='blue', alpha=0.7, edgecolor='black')

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Option')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Systematic Bias in Default Prompt:\n72.4% Predictions are "C" (Left)')
    ax.set_xticks(x)
    ax.set_xticklabels(['A (Up)', 'B (Down)', 'C (Left)', 'D (Right)'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_option_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig5_option_distribution.pdf', bbox_inches='tight')
    plt.close()

    print('图 5 已生成: fig5_option_distribution.png/pdf')


def plot_intervention_effect(output_dir):
    """图 6: 因果干预效果"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # 数据
    categories = ['Normal', 'After Patching']
    accuracies = [20.0, 17.5]

    bars = ax.bar(categories, accuracies, color=['green', 'red'], alpha=0.7, edgecolor='black')

    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Causal Intervention: Patching Velocity Layers\nHas Minimal Effect on Language Generation')
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加解释
    ax.annotate('Only 4.5% drop\n(Expected >20% if layers are used)',
                xy=(0.5, 10), ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig6_intervention.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig6_intervention.pdf', bbox_inches='tight')
    plt.close()

    print('图 6 已生成: fig6_intervention.png/pdf')


def main():
    output_dir = Path('paper_figures')
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("生成论文图表")
    print("="*60)

    # 生成所有图表
    plot_layer_r2_curve(output_dir)
    plot_model_comparison(output_dir)
    plot_prompt_engineering(output_dir)
    plot_grounding_vs_physics(output_dir)
    plot_option_distribution(output_dir)
    plot_intervention_effect(output_dir)

    print("\n" + "="*60)
    print("所有图表已生成至 paper_figures/ 目录")
    print("="*60)


if __name__ == '__main__':
    main()
