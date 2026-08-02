"""Patch 定位工具：像素坐标与视觉 token 的映射"""
import torch
import numpy as np
from typing import Tuple, List, Optional


class PatchLocator:
    """将像素坐标转换为 Qwen2.5-VL 视觉 token 索引"""

    def __init__(self, patch_size=14, spatial_merge_size=2):
        """
        Args:
            patch_size: ViT patch 大小（Qwen2.5-VL 为 14）
            spatial_merge_size: 空间合并因子（Qwen2.5-VL 为 2，即 2x2 patch 合并为 1 token）
        """
        self.patch_size = patch_size
        self.merge_size = spatial_merge_size
        self.effective_patch_size = patch_size * spatial_merge_size

    def pixel_to_patch(self, x: float, y: float, image_width: int, image_height: int,
                       grid_thw: torch.Tensor) -> Tuple[int, int]:
        """
        将像素坐标转换为 patch 网格坐标

        Args:
            x, y: 像素坐标
            image_width, image_height: 原始图像尺寸
            grid_thw: (T, H, W) patch 网格尺寸

        Returns:
            (patch_row, patch_col) 在合并后的 token 网格中的位置
        """
        # grid_thw: [temporal, height_patches, width_patches]
        _, grid_h, grid_w = grid_thw.tolist()

        # 计算缩放比例（动态分辨率）
        scale_h = grid_h * self.effective_patch_size / image_height
        scale_w = grid_w * self.effective_patch_size / image_width

        # 转换到 patch 坐标
        patch_x = x * scale_w / self.effective_patch_size
        patch_y = y * scale_h / self.effective_patch_size

        # 合并后的 token 坐标
        token_col = int(patch_x)
        token_row = int(patch_y)

        # 边界检查
        token_col = min(max(token_col, 0), grid_w - 1)
        token_row = min(max(token_row, 0), grid_h - 1)

        return token_row, token_col

    def patch_to_token_index(self, row: int, col: int, grid_w: int) -> int:
        """将 (row, col) 转换为展平的 token 索引"""
        return row * grid_w + col

    def get_ball_token_indices(self, ball_x: float, ball_y: float, ball_radius: float,
                                image_width: int, image_height: int,
                                grid_thw: torch.Tensor) -> List[int]:
        """
        获取覆盖小球区域的所有 token 索引

        Returns:
            token 索引列表
        """
        _, grid_h, grid_w = grid_thw.tolist()

        # 获取球心 patch
        center_row, center_col = self.pixel_to_patch(
            ball_x, ball_y, image_width, image_height, grid_thw
        )

        # 计算半径覆盖的 patch 数
        scale_h = grid_h * self.effective_patch_size / image_height
        scale_w = grid_w * self.effective_patch_size / image_width
        patch_radius = int(ball_radius * max(scale_h, scale_w) / self.effective_patch_size) + 1

        # 收集覆盖区域的所有 token
        indices = []
        for dr in range(-patch_radius, patch_radius + 1):
            for dc in range(-patch_radius, patch_radius + 1):
                r, c = center_row + dr, center_col + dc
                if 0 <= r < grid_h and 0 <= c < grid_w:
                    idx = self.patch_to_token_index(r, c, grid_w)
                    indices.append(idx)

        return indices

    def extract_ball_tokens(self, hidden_states: torch.Tensor,
                            ball_positions: List[Tuple[float, float, float]],
                            image_width: int, image_height: int,
                            grid_thw: torch.Tensor) -> torch.Tensor:
        """
        从 hidden states 中提取小球对应的 token

        Args:
            hidden_states: (num_tokens, hidden_dim) 或 (batch, num_tokens, hidden_dim)
            ball_positions: [(x, y, radius), ...] 每个球的位置
            image_width, image_height: 原始图像尺寸
            grid_thw: patch 网格尺寸

        Returns:
            (num_balls, max_tokens_per_ball, hidden_dim) 的 tensor
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[0]  # 取 batch 中第一个

        all_ball_tokens = []
        max_tokens = 0

        for x, y, radius in ball_positions:
            indices = self.get_ball_token_indices(
                x, y, radius, image_width, image_height, grid_thw
            )
            tokens = hidden_states[indices]  # (num_tokens, hidden_dim)
            all_ball_tokens.append(tokens)
            max_tokens = max(max_tokens, len(indices))

        # Padding 到相同长度
        hidden_dim = hidden_states.shape[-1]
        padded = torch.zeros(len(ball_positions), max_tokens, hidden_dim)
        mask = torch.zeros(len(ball_positions), max_tokens)

        for i, tokens in enumerate(all_ball_tokens):
            n = tokens.shape[0]
            padded[i, :n] = tokens
            mask[i, :n] = 1

        return padded, mask

    def extract_mean_pooled(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """提取全图 mean-pooled 表征"""
        if hidden_states.dim() == 3:
            return hidden_states.mean(dim=1)  # (batch, hidden_dim)
        return hidden_states.mean(dim=0)  # (hidden_dim,)


def test_patch_locator():
    """测试 patch 定位"""
    locator = PatchLocator()

    # 模拟 448x448 图像，patch_size=14，merge_size=2
    # 有效 patch 大小 = 28，网格 = 448/28 = 16x16
    grid_thw = torch.tensor([1, 16, 16])

    # 球心在 (224, 224)，应该在 patch (8, 8)
    row, col = locator.pixel_to_patch(224, 224, 448, 448, grid_thw)
    print(f"球心 (224, 224) -> patch ({row}, {col})")

    # 获取覆盖区域的 token
    indices = locator.get_ball_token_indices(224, 224, 25, 448, 448, grid_thw)
    print(f"覆盖 token 数: {len(indices)}")
    print(f"Token 索引: {indices[:5]}...")

    # 测试提取
    hidden = torch.randn(256, 1024)  # 16x16=256 tokens, 1024 dim
    tokens, mask = locator.extract_ball_tokens(
        hidden, [(224, 224, 25)], 448, 448, grid_thw
    )
    print(f"提取 token shape: {tokens.shape}, mask shape: {mask.shape}")


if __name__ == '__main__':
    test_patch_locator()
