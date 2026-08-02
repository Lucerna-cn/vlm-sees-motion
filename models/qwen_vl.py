"""Qwen2.5-VL 模型加载、推理与多层 hook 封装"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Dict, List, Optional, Callable
import numpy as np


class QwenVLHookManager:
    """管理 Qwen2.5-VL 视觉编码器的多层 forward hook"""

    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                 cache_dir=None, device_map="auto", torch_dtype=torch.bfloat16):
        self.model_name = model_name
        self.cache_dir = cache_dir

        print(f"加载模型: {model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )

        self.device = self.model.device
        self.hooks = []
        self.layer_outputs = {}

        # 获取视觉编码器（兼容不同 transformers 版本）
        self.visual_encoder = self._find_visual_encoder()
        self.num_layers = len(self.visual_encoder.blocks)

        print(f"视觉编码器层数: {self.num_layers}")
        print(f"设备: {self.device}")

    def _find_visual_encoder(self):
        """查找视觉编码器（兼容不同模型结构）"""
        # transformers 5.x: model.model.visual
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
            return self.model.model.visual
        # transformers 4.x: model.visual
        elif hasattr(self.model, 'visual'):
            return self.model.visual
        # 其他可能的位置
        elif hasattr(self.model, 'vision_model'):
            return self.model.vision_model
        elif hasattr(self.model, 'vision_encoder'):
            return self.model.vision_encoder
        else:
            # 打印模型结构帮助调试
            print("模型属性:", [attr for attr in dir(self.model) if not attr.startswith('_')])
            raise AttributeError("无法找到视觉编码器，请检查模型结构")

    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        """注册 forward hook 到指定层"""
        if layer_indices is None:
            layer_indices = list(range(self.num_layers))

        self.clear_hooks()
        self.layer_outputs = {i: [] for i in layer_indices}

        def make_hook(layer_idx):
            def hook(module, input, output):
                # output 是 tuple (hidden_states, ...) 或 tensor
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # 立即移到 CPU 并 detach，避免显存累积
                self.layer_outputs[layer_idx].append(
                    hidden.detach().cpu()
                )
            return hook

        for idx in layer_indices:
            hook = self.visual_encoder.blocks[idx].register_forward_hook(
                make_hook(idx)
            )
            self.hooks.append(hook)

        return layer_indices

    def clear_hooks(self):
        """清除所有 hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_outputs = {}

    def get_layer_output(self, layer_idx: int, batch_idx: int = 0):
        """获取指定层的输出"""
        if layer_idx not in self.layer_outputs:
            return None
        if batch_idx >= len(self.layer_outputs[layer_idx]):
            return None
        return self.layer_outputs[layer_idx][batch_idx]

    def clear_layer_outputs(self):
        """清空层输出缓存（保留 hook）"""
        for key in self.layer_outputs:
            self.layer_outputs[key] = []

    @torch.no_grad()
    def forward_with_hooks(self, images, texts=None, layer_indices=None):
        """
        前向传播并捕获指定层输出

        Args:
            images: 单张图片 (PIL.Image) 或图片列表（多帧）
            texts: 文本提示
            layer_indices: 要 hook 的层
        """
        # 兼容单图和多图输入
        if not isinstance(images, list):
            images = [images]
            single_image = True
        else:
            single_image = False

        if texts is None:
            if single_image:
                texts = ["描述这张图片。"]
            else:
                texts = [f"这{len(images)}张图片是连续帧，描述物体的运动。"]

        # 构建输入（支持多图）
        if single_image:
            # 单图：一个 message 一张图
            messages = [
                [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[0]},
                        {"type": "text", "text": texts[0]}
                    ]
                }]
            ]
        else:
            # 多图：一个 message 多张图
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": texts[0]})
            messages = [[{"role": "user", "content": content}]]

        text_inputs = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        inputs = self.processor(
            text=text_inputs,
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # 注册 hook
        self.register_hooks(layer_indices)

        # 前向传播
        outputs = self.model(**inputs)

        # 收集结果
        results = {
            'logits': outputs.logits.cpu() if hasattr(outputs, 'logits') else None,
            'layer_outputs': {k: [t.cpu() for t in v] for k, v in self.layer_outputs.items()},
            'inputs': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
        }

        return results

    @torch.no_grad()
    def generate(self, images, texts, max_new_tokens=128, **kwargs):
        """生成文本"""
        messages = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": text}
                ]
            }]
            for img, text in zip(images, texts)
        ]

        text_inputs = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        inputs = self.processor(
            text=text_inputs,
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_texts

    def get_grid_info(self, inputs):
        """获取视觉 token 的 grid 信息（用于 patch 定位）"""
        # Qwen2.5-VL 的 image_grid_thw 包含每张图的 (T, H, W) patch 数
        if 'image_grid_thw' in inputs:
            return inputs['image_grid_thw']
        return None

    def __del__(self):
        self.clear_hooks()


def test_model_loading():
    """测试模型加载"""
    manager = QwenVLHookManager()

    # 创建测试图像
    from PIL import Image
    import numpy as np
    test_img = Image.fromarray(
        np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
    )

    # 测试推理
    print("\n测试推理...")
    texts = manager.generate([test_img], ["这张图片里有什么？"])
    print(f"输出: {texts[0]}")

    # 测试 hook
    print("\n测试 hook...")
    results = manager.forward_with_hooks([test_img], layer_indices=[0, 1, 2])
    for layer_idx, outputs in results['layer_outputs'].items():
        if outputs:
            print(f"  Layer {layer_idx}: shape {outputs[0].shape}")

    print("\n模型测试完成")


if __name__ == '__main__':
    test_model_loading()
