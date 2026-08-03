"""方向 3: 表征几何分析 - 速度信息在高维空间中的几何结构"""
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.qwen_vl import QwenVLHookManager
from models.patch_utils import PatchLocator


def extract_features_for_geometry(model, locator, seq_paths, layer_idx=16, max_samples=200):
    """
    提取用于几何分析的特征

    Returns:
        features: (n_samples, hidden_dim) 特征矩阵
        velocities: (n_samples, 2) 速度标签
        directions: (n_samples,) 方向标签（上/下/左/右）
    """
    print(f"\n提取 Layer {layer_idx} 特征用于几何分析...")

    all_features = []
    all_velocities = []
    all_directions = []

    for seq_path in tqdm(seq_paths[:max_samples], desc="特征提取"):
        try:
            with open(seq_path / 'trajectory.json', 'r') as f:
                traj_data = json.load(f)

            frame_files = sorted(seq_path.glob('frame_*.png'))
            if len(frame_files) < 3:
                continue

            # 使用第 3 帧
            frame_idx = 3
            from PIL import Image
            image = Image.open(frame_files[frame_idx]).convert('RGB')

            # 获取球位置
            ball_positions = []
            frame_velocities = []

            for ball in traj_data['balls']:
                traj = ball['trajectory']
                if frame_idx < len(traj):
                    state = traj[frame_idx]
                    ball_positions.append((state['x'], state['y'], state['radius']))
                    frame_velocities.append([state['vx'], state['vy']])

            if not ball_positions:
                continue

            # 提取特征
            results = model.forward_with_hooks([image], layer_indices=[layer_idx])

            if layer_idx not in results['layer_outputs'] or not results['layer_outputs'][layer_idx]:
                continue

            hidden = results['layer_outputs'][layer_idx][0]  # (num_tokens, hidden_dim)

            # 获取 grid
            grid_thw = model.get_grid_info(results['inputs'])
            if grid_thw is None:
                continue

            # 提取球的 token 并 mean-pool
            ball_tokens, ball_mask = locator.extract_ball_tokens(
                hidden, ball_positions,
                image.width, image.height,
                grid_thw[0]
            )

            # Mean-pool
            mask_expanded = ball_mask[..., np.newaxis]
            ball_repr = (ball_tokens * mask_expanded).sum(axis=1) / (ball_mask.sum(axis=1, keepdims=True) + 1e-8)

            # 对每个球记录特征和速度
            for i, (feat, vel) in enumerate(zip(ball_repr, frame_velocities)):
                all_features.append(feat.numpy())
                all_velocities.append(vel)

                # 方向标签
                vx, vy = vel
                if abs(vx) > abs(vy):
                    direction = '右' if vx > 0 else '左'
                else:
                    direction = '下' if vy > 0 else '上'
                all_directions.append(direction)

            model.clear_layer_outputs()

            if len(all_features) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            continue

    features = np.array(all_features)
    velocities = np.array(all_velocities)
    directions = np.array(all_directions)

    print(f"提取完成: {len(features)} 个样本, 特征维度 {features.shape[1]}")

    return features, velocities, directions


def analyze_geometry(features, velocities, directions, output_dir):
    """分析表征的几何结构"""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("表征几何分析")
    print("="*60)

    results = {}

    # 1. 速度大小与表征范数的关系
    print("\n【1. 速度大小与表征范数】")
    speed = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    feature_norms = np.linalg.norm(features, axis=1)

    correlation = np.corrcoef(speed, feature_norms)[0, 1]
    results['speed_norm_correlation'] = float(correlation)
    print(f"速度大小与表征范数的相关系数: {correlation:.4f}")

    if abs(correlation) > 0.3:
        print("  → 速度大小与表征强度显著相关")
    else:
        print("  → 速度大小与表征强度无关（速度以方向而非强度编码）")

    # 2. 不同方向的表征分离度
    print("\n【2. 不同方向的表征分离度】")

    # 计算每个方向的平均表征
    direction_centroids = {}
    for d in ['上', '下', '左', '右']:
        mask = directions == d
        if mask.sum() > 0:
            direction_centroids[d] = features[mask].mean(axis=0)

    # 计算方向间的余弦相似度
    print("方向间余弦相似度:")
    directions_list = list(direction_centroids.keys())
    similarity_matrix = np.zeros((len(directions_list), len(directions_list)))

    for i, d1 in enumerate(directions_list):
        for j, d2 in enumerate(directions_list):
            sim = cosine_similarity(
                direction_centroids[d1].reshape(1, -1),
                direction_centroids[d2].reshape(1, -1)
            )[0, 0]
            similarity_matrix[i, j] = sim
            print(f"  {d1} vs {d2}: {sim:.4f}")

    results['direction_similarity_matrix'] = similarity_matrix.tolist()

    # 平均对角线（同类相似度）
    avg_intra = np.mean([similarity_matrix[i, i] for i in range(len(directions_list))])
    # 平均非对角线（异类相似度）
    mask = ~np.eye(len(directions_list), dtype=bool)
    avg_inter = np.mean(similarity_matrix[mask])

    results['avg_intra_direction_similarity'] = float(avg_intra)
    results['avg_inter_direction_similarity'] = float(avg_inter)
    results['direction_separation'] = float(avg_intra - avg_inter)

    print(f"\n方向分离度: {avg_intra - avg_inter:.4f}")
    if avg_intra - avg_inter > 0.1:
        print("  → 不同方向的表征显著分离")
    else:
        print("  → 不同方向的表征混合在一起")

    # 3. PCA 分析
    print("\n【3. PCA 主成分分析】")
    pca = PCA(n_components=min(10, features.shape[1]))
    pca_features = pca.fit_transform(features)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print(f"前 5 个主成分解释方差: {explained_variance[:5]}")
    print(f"前 5 个主成分累计解释方差: {cumulative_variance[:5]}")

    results['pca_explained_variance'] = explained_variance.tolist()

    # 检查速度信息是否集中在特定主成分
    # 计算每个主成分与速度的相关性
    pc_velocity_corr = []
    for i in range(min(5, pca_features.shape[1])):
        corr_vx = np.corrcoef(pca_features[:, i], velocities[:, 0])[0, 1]
        corr_vy = np.corrcoef(pca_features[:, i], velocities[:, 1])[0, 1]
        pc_velocity_corr.append((abs(corr_vx) + abs(corr_vy)) / 2)

    results['pc_velocity_correlation'] = pc_velocity_corr
    print(f"主成分与速度的相关性: {[f'{c:.3f}' for c in pc_velocity_corr]}")

    # 4. t-SNE 可视化
    print("\n【4. t-SNE 可视化】")
    if len(features) > 50:
        # 降采样以加速
        sample_idx = np.random.choice(len(features), min(200, len(features)), replace=False)
        tsne_features = features[sample_idx]
        tsne_directions = directions[sample_idx]

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_features)-1))
        tsne_results = tsne.fit_transform(tsne_features)

        # 绘图
        plt.figure(figsize=(10, 8))
        colors = {'上': 'red', '下': 'blue', '左': 'green', '右': 'orange'}

        for d in ['上', '下', '左', '右']:
            mask = tsne_directions == d
            if mask.sum() > 0:
                plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                           c=colors[d], label=d, alpha=0.6)

        plt.legend()
        plt.title('t-SNE: 不同运动方向的表征分布')
        plt.savefig(output_dir / 'tsne_directions.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"t-SNE 图保存至: {output_dir / 'tsne_directions.png'}")
        results['tsne_saved'] = True
    else:
        print("样本太少，跳过 t-SNE")
        results['tsne_saved'] = False

    # 5. 子空间分析
    print("\n【5. 速度信息子空间分析】")

    # 使用速度标签训练一个简单的线性分类器，看速度信息在哪个子空间
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # 将速度分为 4 类（上/下/左/右）
    direction_labels = np.array([['上', '下', '左', '右'].index(d) for d in directions])

    # 在完整特征空间上分类
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores_full = cross_val_score(clf, features, direction_labels, cv=3)

    print(f"完整特征空间方向分类准确率: {scores_full.mean():.4f} (+/- {scores_full.std():.4f})")
    results['direction_classification_accuracy'] = float(scores_full.mean())

    if scores_full.mean() > 0.4:
        print("  → 方向信息在表征中显著存在")
    else:
        print("  → 方向信息在表征中较弱")

    # 保存结果
    with open(output_dir / 'geometry_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果保存至: {output_dir}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results_geometry')
    parser.add_argument('--layer', type=int, default=16)
    parser.add_argument('--max_samples', type=int, default=200)
    args = parser.parse_args()

    print("="*60)
    print("方向 3: 表征几何分析")
    print("="*60)

    # 加载数据
    with open(Path(args.data_dir) / 'split.json', 'r') as f:
        split = json.load(f)
    train_meta = split['train']

    data_dir = Path(args.data_dir)
    train_paths = [data_dir / item['path'] for item in train_meta]
    print(f"Train: {len(train_paths)} 段")

    # 初始化模型
    print("\n加载模型...")
    model = QwenVLHookManager()
    locator = PatchLocator()

    # 提取特征
    features, velocities, directions = extract_features_for_geometry(
        model, locator, train_paths,
        layer_idx=args.layer,
        max_samples=args.max_samples
    )

    # 分析几何结构
    results = analyze_geometry(features, velocities, directions, args.output_dir)

    # 关键发现
    print("\n" + "="*60)
    print("关键发现")
    print("="*60)

    if results['direction_separation'] > 0.1:
        print("✓ 不同运动方向的表征显著分离")
        print("  → 速度信息以**方向选择性**方式编码")
    else:
        print("✗ 不同运动方向的表征混合")
        print("  → 速度信息以**连续流形**方式编码")

    if results['direction_classification_accuracy'] > 0.4:
        print(f"✓ 方向分类准确率 {results['direction_classification_accuracy']:.2f}")
        print("  → 速度信息在表征中**显式存在**")
    else:
        print(f"✗ 方向分类准确率仅 {results['direction_classification_accuracy']:.2f}")
        print("  → 速度信息在表征中**隐式存在**")


if __name__ == '__main__':
    main()
