"""Day 1 综合测试：数据生成 + 模型加载 + Hook 验证"""
import sys
import os
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image


def test_data_generation():
    """测试 1: 数据生成"""
    print("\n" + "=" * 60)
    print("测试 1: 数据生成")
    print("=" * 60)

    from data_generation.physics_sim import generate_scene
    from data_generation.renderer import render_frame

    # 生成一个场景
    traj_data = generate_scene(width=448, height=448, num_frames=8, seed=42)
    print(f"✓ 物理模拟成功: {traj_data['num_balls']} 个球, 8 帧")

    # 渲染三种风格
    for style in ['minimal', 'medium', 'realistic']:
        balls_state = []
        for ball in traj_data['balls']:
            state = ball['trajectory'][0].copy()
            state['id'] = ball['id']
            state['radius'] = ball['radius']
            balls_state.append(state)

        img = render_frame(448, 448, balls_state, style)
        print(f"✓ {style} 渲染成功: shape {img.shape}")

    return True


def test_model_loading():
    """测试 2: 模型加载"""
    print("\n" + "=" * 60)
    print("测试 2: 模型加载")
    print("=" * 60)

    from models.qwen_vl import QwenVLHookManager

    manager = QwenVLHookManager()
    print(f"✓ 模型加载成功")
    print(f"  视觉编码器层数: {manager.num_layers}")
    print(f"  设备: {manager.device}")

    return manager


def test_hook_extraction(manager):
    """测试 3: Hook 提取"""
    print("\n" + "=" * 60)
    print("测试 3: Hook 提取")
    print("=" * 60)

    # 创建测试图像
    test_img = Image.fromarray(
        np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
    )

    # 测试单层 hook
    results = manager.forward_with_hooks([test_img], layer_indices=[0, 1])

    for layer_idx, outputs in results['layer_outputs'].items():
        if outputs:
            print(f"✓ Layer {layer_idx}: shape {outputs[0].shape}")

    # 检查 grid 信息
    grid_info = manager.get_grid_info(results['inputs'])
    if grid_info is not None:
        print(f"✓ Grid info: {grid_info}")

    return True


def test_patch_location():
    """测试 4: Patch 定位"""
    print("\n" + "=" * 60)
    print("测试 4: Patch 定位")
    print("=" * 60)

    from models.patch_utils import PatchLocator

    locator = PatchLocator()

    # 模拟 grid
    grid_thw = torch.tensor([1, 16, 16])

    # 测试坐标转换
    row, col = locator.pixel_to_patch(224, 224, 448, 448, grid_thw)
    print(f"✓ 像素 (224, 224) -> patch ({row}, {col})")

    # 测试 token 提取
    hidden = torch.randn(256, 1024)
    tokens, mask = locator.extract_ball_tokens(
        hidden, [(224, 224, 25)], 448, 448, grid_thw
    )
    print(f"✓ Token 提取: shape {tokens.shape}")

    return True


def test_generation(manager):
    """测试 5: 文本生成"""
    print("\n" + "=" * 60)
    print("测试 5: 文本生成")
    print("=" * 60)

    # 创建一个有明确内容的测试图像
    img_array = np.zeros((448, 448, 3), dtype=np.uint8)
    img_array[100:200, 100:200] = [255, 0, 0]  # 红色方块
    test_img = Image.fromarray(img_array)

    texts = manager.generate([test_img], ["这张图片里有什么颜色的方块？"], max_new_tokens=50)
    print(f"✓ 生成成功: {texts[0][:100]}...")

    return True


def main():
    print("=" * 60)
    print("Day 1 综合测试")
    print("=" * 60)

    results = {}

    # 测试 1: 数据生成
    try:
        results['data_generation'] = test_data_generation()
    except Exception as e:
        print(f"✗ 数据生成失败: {e}")
        results['data_generation'] = False

    # 测试 2: 模型加载
    try:
        manager = test_model_loading()
        results['model_loading'] = True
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        results['model_loading'] = False
        manager = None

    # 测试 3-5: 依赖模型的测试
    if manager:
        try:
            results['hook_extraction'] = test_hook_extraction(manager)
        except Exception as e:
            print(f"✗ Hook 提取失败: {e}")
            results['hook_extraction'] = False

        try:
            results['patch_location'] = test_patch_location()
        except Exception as e:
            print(f"✗ Patch 定位失败: {e}")
            results['patch_location'] = False

        try:
            results['generation'] = test_generation(manager)
        except Exception as e:
            print(f"✗ 文本生成失败: {e}")
            results['generation'] = False

        # 清理
        del manager
        torch.cuda.empty_cache()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")

    all_passed = all(results.values())
    print(f"\n总体: {'全部通过' if all_passed else '存在失败'}")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
