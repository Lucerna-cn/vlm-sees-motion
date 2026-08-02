"""流式表征提取：逐层提取并立即训练探针，避免存储全部表征"""
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import gc

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.qwen_vl import QwenVLHookManager
from models.patch_utils import PatchLocator
from probing.probes import RidgeProbe, MLPProbe
from probing.baselines import ConstantVelocityBaseline


class StreamProber:
    """流式 probing 执行器"""

    def __init__(self, model_manager: QwenVLHookManager, patch_locator: PatchLocator):
        self.model = model_manager
        self.locator = patch_locator
        self.results = {}

    def load_sequence(self, seq_dir: Path):
        """加载单个序列的数据"""
        # 加载轨迹
        with open(seq_dir / 'trajectory.json', 'r') as f:
            traj_data = json.load(f)

        # 加载所有帧
        frames = []
        frame_files = sorted(seq_dir.glob('frame_*.png'))
        for f in frame_files:
            img = Image.open(f).convert('RGB')
            frames.append(img)

        return traj_data, frames

    def extract_frame_features(self, image, ball_positions, layer_indices=None):
        """
        提取单帧的多层表征

        Args:
            image: PIL Image
            ball_positions: [(x, y, radius), ...]
            layer_indices: 要提取的层，None 表示全部

        Returns:
            dict: {layer_idx: {'ball_tokens': tensor, 'mean_pooled': tensor}}
        """
        # 前向传播 with hooks
        results = self.model.forward_with_hooks(
            [image],
            layer_indices=layer_indices
        )

        # 获取 grid 信息
        grid_thw = self.model.get_grid_info(results['inputs'])
        if grid_thw is None:
            raise ValueError("无法获取 grid 信息")

        # 处理每一层
        features = {}
        for layer_idx, layer_outputs in results['layer_outputs'].items():
            if not layer_outputs:
                continue

            hidden = layer_outputs[0]  # (num_tokens, hidden_dim)

            # 提取每个球的 token
            ball_tokens, ball_mask = self.locator.extract_ball_tokens(
                hidden, ball_positions,
                image.width, image.height,
                grid_thw[0]  # batch 中第一张图
            )

            # Mean-pooled 全图表征
            mean_pooled = self.locator.extract_mean_pooled(hidden)

            features[layer_idx] = {
                'ball_tokens': ball_tokens.numpy(),  # (num_balls, max_tokens, hidden_dim)
                'ball_mask': ball_mask.numpy(),
                'mean_pooled': mean_pooled.numpy()   # (hidden_dim,)
            }

        # 清理
        self.model.clear_layer_outputs()

        return features

    def probe_layer(self, layer_idx, features_list, targets_list, probe_type='ridge'):
        """
        对单层训练探针

        Args:
            layer_idx: 层索引
            features_list: 该层所有样本的特征列表
            targets_list: 对应的目标值列表
            probe_type: 'ridge' 或 'mlp'

        Returns:
            dict: 探针结果
        """
        # 合并所有样本
        X = np.concatenate(features_list, axis=0)
        y = np.concatenate(targets_list, axis=0)

        # 划分 train/test (按轨迹已在外部划分)
        # 这里假设 features_list 已按 train/test 分开

        if probe_type == 'ridge':
            probe = RidgeProbe()
        else:
            probe = MLPProbe()

        results = probe.fit_evaluate(X, y)
        results['layer_idx'] = layer_idx
        results['probe_type'] = probe_type

        return results

    def run_stream_probing(self, data_dirs, targets_fn, layer_indices=None,
                           probe_type='ridge', batch_size=2):
        """
        流式 probing 主循环

        Args:
            data_dirs: 序列目录列表
            targets_fn: 从轨迹数据提取目标值的函数
            layer_indices: 要 probing 的层
            probe_type: 探针类型
            batch_size: 批大小
        """
        if layer_indices is None:
            layer_indices = list(range(self.model.num_layers))

        # 按层组织数据
        layer_features = {i: [] for i in layer_indices}
        layer_targets = {i: [] for i in layer_indices}

        print(f"开始流式 probing，共 {len(data_dirs)} 个序列")

        for seq_dir in tqdm(data_dirs, desc="处理序列"):
            seq_dir = Path(seq_dir)
            traj_data, frames = self.load_sequence(seq_dir)

            # 提取目标值（如速度）
            targets = targets_fn(traj_data)

            # 处理每一帧
            for frame_idx, frame in enumerate(frames):
                # 获取该帧的球位置
                ball_positions = []
                frame_targets = []
                for ball in traj_data['balls']:
                    traj = ball['trajectory']
                    if frame_idx < len(traj):
                        state = traj[frame_idx]
                        ball_positions.append((state['x'], state['y'], state['radius']))
                        # 目标值（如下一帧速度）
                        if frame_idx + 1 < len(traj):
                            next_state = traj[frame_idx + 1]
                            frame_targets.append([next_state['vx'], next_state['vy']])
                        else:
                            frame_targets.append([0, 0])  # 最后一帧无目标

                if not ball_positions:
                    continue

                # 提取特征
                features = self.extract_frame_features(
                    frame, ball_positions, layer_indices
                )

                # 存储（仅保留必要信息）
                for layer_idx in layer_indices:
                    if layer_idx in features:
                        # 对每个球取 mean-pooled token 作为其表征
                        ball_tokens = features[layer_idx]['ball_tokens']
                        ball_mask = features[layer_idx]['ball_mask']

                        # Mean-pool over tokens for each ball
                        mask_expanded = ball_mask[..., np.newaxis]
                        ball_repr = (ball_tokens * mask_expanded).sum(axis=1) / ball_mask.sum(axis=1, keepdims=True)

                        layer_features[layer_idx].append(ball_repr)
                        layer_targets[layer_idx].append(np.array(frame_targets))

                # 定期清理显存
                if frame_idx % batch_size == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        # 对每层训练探针
        results = {}
        for layer_idx in layer_indices:
            if layer_features[layer_idx]:
                print(f"\n训练 Layer {layer_idx} 探针...")
                result = self.probe_layer(
                    layer_idx,
                    layer_features[layer_idx],
                    layer_targets[layer_idx],
                    probe_type
                )
                results[layer_idx] = result

        return results


def extract_velocity_target(traj_data):
    """提取速度目标（示例）"""
    targets = []
    for ball in traj_data['balls']:
        traj = ball['trajectory']
        ball_targets = []
        for i in range(len(traj) - 1):
            ball_targets.append([traj[i+1]['vx'], traj[i+1]['vy']])
        targets.append(ball_targets)
    return targets


if __name__ == '__main__':
    print("流式 Probing 模块")
    print("使用方式: 在服务器上运行完整 probing 流程")
