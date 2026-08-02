"""因果干预实验：对关键层 token 做 patch/消融，观察下游行为变化"""
import torch
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Optional, Callable


class TokenPatcher:
    """对视觉 token 进行干预"""

    def __init__(self, model_manager, patch_locator):
        self.model = model_manager
        self.locator = patch_locator
        self.patched_values = {}
        self.hooks = []

    def create_patch_hook(self, layer_idx: int, token_indices: List[int],
                          patch_mode: str = 'mean', patch_value: Optional[torch.Tensor] = None):
        """
        创建 patch hook

        Args:
            layer_idx: 目标层
            token_indices: 要 patch 的 token 索引
            patch_mode: 'mean' (均值替换), 'zero' (置零), 'random' (随机), 'custom' (自定义值)
            patch_value: 自定义 patch 值（当 patch_mode='custom' 时使用）
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Clone 避免原地修改
            hidden = hidden.clone()

            if patch_mode == 'mean':
                # 用所有 token 的均值替换
                mean_val = hidden.mean(dim=0, keepdim=True)
                for idx in token_indices:
                    hidden[idx] = mean_val
            elif patch_mode == 'zero':
                for idx in token_indices:
                    hidden[idx] = 0
            elif patch_mode == 'random':
                for idx in token_indices:
                    hidden[idx] = torch.randn_like(hidden[idx])
            elif patch_mode == 'custom' and patch_value is not None:
                for idx in token_indices:
                    hidden[idx] = patch_value

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        return hook

    def patch_layer(self, layer_idx: int, token_indices: List[int], **kwargs):
        """对指定层注册 patch hook"""
        hook_fn = self.create_patch_hook(layer_idx, token_indices, **kwargs)
        hook = self.model.visual_encoder.blocks[layer_idx].register_forward_hook(hook_fn)
        self.hooks.append(hook)
        return hook

    def clear_patches(self):
        """清除所有 patch"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class InterventionExperiment:
    """因果干预实验执行器"""

    def __init__(self, model_manager, patch_locator):
        self.model = model_manager
        self.locator = patch_locator
        self.patcher = TokenPatcher(model_manager, patch_locator)
        self.results = []

    def run_intervention(self, seq_dir: Path, target_layer: int,
                         ball_id: int = 0, frame_idx: int = 3,
                         patch_mode: str = 'mean'):
        """
        运行单个干预实验

        对比：正常推理 vs patch 后推理
        """
        from behavior.question_gen import generate_question, parse_answer
        from PIL import Image

        seq_dir = Path(seq_dir)

        # 加载数据
        with open(seq_dir / 'trajectory.json', 'r') as f:
            traj_data = json.load(f)

        frame_files = sorted(seq_dir.glob('frame_*.png'))
        image = Image.open(frame_files[frame_idx]).convert('RGB')

        # 获取球位置
        ball = traj_data['balls'][ball_id]
        traj = ball['trajectory']
        state = traj[frame_idx]
        ball_x, ball_y, ball_r = state['x'], state['y'], state['radius']

        # 生成问题
        question, correct_answer = generate_question(traj_data, ball_id, frame_idx)

        # 1. 正常推理
        normal_response = self.model.generate([image], [question], max_new_tokens=10)
        normal_answer = parse_answer(normal_response[0])
        normal_correct = (normal_answer == correct_answer)

        # 2. 获取要 patch 的 token
        # 需要先跑一次 forward 获取 grid 信息
        results = self.model.forward_with_hooks([image], layer_indices=[target_layer])
        grid_thw = self.model.get_grid_info(results['inputs'])

        token_indices = self.locator.get_ball_token_indices(
            ball_x, ball_y, ball_r,
            image.width, image.height,
            grid_thw[0]
        )

        # 3. Patch 后推理
        self.patcher.patch_layer(target_layer, token_indices, patch_mode=patch_mode)
        patched_response = self.model.generate([image], [question], max_new_tokens=10)
        patched_answer = parse_answer(patched_response[0])
        patched_correct = (patched_answer == correct_answer)
        self.patcher.clear_patches()

        # 记录结果
        result = {
            'seq_dir': str(seq_dir),
            'target_layer': target_layer,
            'ball_id': ball_id,
            'frame_idx': frame_idx,
            'patch_mode': patch_mode,
            'num_patched_tokens': len(token_indices),
            'correct_answer': correct_answer,
            'normal': {
                'response': normal_response[0],
                'answer': normal_answer,
                'correct': normal_correct
            },
            'patched': {
                'response': patched_response[0],
                'answer': patched_answer,
                'correct': patched_correct
            },
            'behavior_changed': normal_answer != patched_answer,
            'accuracy_drop': normal_correct and not patched_correct
        }

        self.results.append(result)
        return result

    def run_batch(self, seq_dirs: List[Path], target_layers: List[int], **kwargs):
        """批量运行干预实验"""
        for seq_dir in seq_dirs:
            for layer in target_layers:
                try:
                    self.run_intervention(seq_dir, layer, **kwargs)
                except Exception as e:
                    print(f"干预实验失败 {seq_dir} layer {layer}: {e}")

        return self.results

    def compute_statistics(self):
        """计算干预效果统计"""
        if not self.results:
            return {}

        total = len(self.results)
        normal_correct = sum(1 for r in self.results if r['normal']['correct'])
        patched_correct = sum(1 for r in self.results if r['patched']['correct'])
        behavior_changed = sum(1 for r in self.results if r['behavior_changed'])
        accuracy_drop = sum(1 for r in self.results if r['accuracy_drop'])

        return {
            'total': total,
            'normal_accuracy': normal_correct / total,
            'patched_accuracy': patched_correct / total,
            'behavior_change_rate': behavior_changed / total,
            'accuracy_drop_rate': accuracy_drop / total,
            'accuracy_change': (patched_correct - normal_correct) / total
        }

    def save_results(self, output_path: Path):
        """保存结果"""
        stats = self.compute_statistics()
        with open(output_path, 'w') as f:
            json.dump({
                'statistics': stats,
                'results': self.results
            }, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    print("因果干预实验模块")
    print("使用方式: 在服务器上运行干预实验")
