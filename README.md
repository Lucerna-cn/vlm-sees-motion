# Do VLMs "See" Motion? A Small Hypothesis, a Small Experiment

> 中文版请见文末（English first, Chinese below）

We spend a lot of time watching vision-language models describe images. They are great at naming objects, reading text, even counting apples. But there is a quieter question we kept coming back to: when a ball bounces across the screen, does the model actually *encode* that motion somewhere inside its visual features — or is it just guessing words?

This repo is the story of a small experiment we ran to find out. It did not answer the question the way we expected, and honestly, the twist at the end is the best part.

---

## The setup: a tiny world we control

To ask this question cleanly, we built a tiny world we control completely. We simulated **6000 short physics clips** with [pymunk](https://www.pymunk.org/) — 1 to 3 balls bouncing in a box with random gravity, elasticity and friction — and rendered them in three visual styles (`minimal`, `medium`, `realistic`). Every frame ships with ground-truth trajectory data: position, velocity, acceleration. Nothing here is ambiguous. If the model encodes velocity *anywhere* in its visual stream, we should be able to find it.

Then we asked two questions:

1. **Representation**: can a probe read velocity `(vx, vy)` out of the visual tokens at each layer of Qwen2.5-VL-3B-Instruct?
2. **Behavior**: can the model's language head actually *use* that information — say, predict which direction a ball is moving?

---

## Chapter 1 — Static images are silent

We started with the simplest possible test: a Ridge (linear) probe on a **single frame**. If velocity lives in the visual tokens in a readable, linear way, this should work.

| Layer | R² | Baseline R² |
|-------|-----|-------------|
| 0  | -0.0046 | 0.9131 |
| 8  | 0.1644  | 0.9131 |
| 16 | 0.3043  | 0.9131 |
| 24 | 0.3409  | 0.9131 |
| 31 | 0.2794  | 0.9131 |

R² ≈ 0. A static image carries essentially no linearly decodable velocity information — which makes sense: a photograph of a ball does not say how fast it is moving. The constant-velocity baseline (just "reuse the previous velocity") is at 0.91. So this was not a failure; it was a sanity check that passed.

## Chapter 2 — Two frames barely help

Maybe the model needs time. We fed **two consecutive frames** and probed again.

| Layer | R² | Baseline R² |
|-------|-----|-------------|
| 0  | 0.0187 | 0.9026 |
| 8  | 0.1793 | 0.9026 |
| 16 | 0.3202 | 0.9026 |
| 24 | 0.3598 | 0.9026 |
| 31 | 0.3519 | 0.9026 |

Better, but only slightly (+0.02 ~ 0.07 R²). Even with temporal context, linear probes peak around **0.36** at the upper layers — still far below the 0.90 baseline. If velocity were sitting in the features in plain sight, this should have been easy.

## Chapter 3 — The information is hiding non-linearly

Linear probes failing does not mean the information is absent. It might just be *organized* in a way a straight line cannot see. So we swapped the probe for a small 2-layer MLP.

| Layer | Ridge R² | MLP R² | Change |
|-------|----------|--------|--------|
| 16 | 0.3202 | **0.4164** | +0.096 |
| 24 | 0.3598 | 0.4075  | +0.048 |
| 31 | 0.3519 | 0.1543  | -0.198 |

At the mid-layers (16–24), the MLP jumps to **R² ≈ 0.42**. Velocity is there — it is just encoded non-linearly, in a form a linear readout cannot reach. (Layer 31 drops, which suggests overfitting or that the final visual layers have already transformed the signal into something sparse.)

## Chapter 4 — The twist

So the representations know the velocity. Then we asked the model itself, in plain language: *"Which direction is ball 0 moving next?"* (A/B/C/D, four choices, 25% chance).

| Metric | Value | Reading |
|--------|-------|---------|
| Overall accuracy | **15.0%** | Below random chance (25%) |
| realistic | 14.3% (6/42) | Poor on complex backgrounds |
| medium | 22.6% (7/31) | Close to chance |
| minimal | 7.4% (2/27) | Worst on the simplest scenes |

**15% — worse than a coin flip among four options.** The model even shows a systematic answer bias (it loves picking A or C). Yet the very same visual tokens decode velocity at R² = 0.42.

That is the dissociation: the information exists in the representation, but the language head cannot get at it. There is an information bottleneck between visual encoding and language generation — or, in plainer words, a model can *hold* a fact in its features and still fail to say it.

## Hypotheses summary

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1: position decodable | Pending | - |
| H2: velocity decodable | Partially supported | Non-linear R² = 0.42, linear R² = 0.36 |
| H3: representation–behavior dissociation | **Strongly supported** | Representation 42% vs. behavior 15%, gap ~27 pts |

We think this makes a neat workshop-paper core finding: *Qwen2.5-VL's visual representations implicitly encode velocity, but its language output cannot use that information for physical reasoning.*

---

## The world we built (dataset)

`data_generation/` generates the fully controlled dataset:

- 2D physics via **pymunk**: 1–3 balls, random gravity, wall/ball collisions, elasticity & friction
- 8 frames per sequence at 448×448
- Ground-truth trajectory JSON (position, velocity, acceleration per frame) next to the rendered PNGs
- 6000 sequences total: `minimal` / `medium` / `realistic` × 2000
- Sequence-level train/test split (`split.json`)

## Repository structure

```
vlm_kinematics_probing/
├── EXPERIMENT_LOG.md        # 完整实验记录（中文）Full experiment log (Chinese)
├── data_generation/
│   ├── physics_sim.py       # pymunk 2D 物理模拟 pymunk 2D physics simulation
│   ├── renderer.py          # 三档渲染器 minimal / medium / realistic renderers
│   └── generate_dataset.py  # 数据集生成 + 划分 dataset generation + train/test split
├── models/
│   ├── qwen_vl.py           # Qwen2.5-VL 加载 + 多层 hook 封装 loading + multi-layer forward hooks
│   └── patch_utils.py       # 像素坐标 → 视觉 token 映射 pixel coordinates -> visual token mapping
├── probing/
│   ├── probes.py            # Ridge / MLP / 对照任务探针 Ridge / MLP / control-task probes
│   ├── baselines.py         # 恒定速度等基线 constant-velocity & random baselines
│   └── extract_stream.py    # 流式表征提取 streaming representation extraction
├── behavior/
│   └── question_gen.py      # 行为问题生成与评测 multiple-choice question generation & evaluation
├── intervention/
│   └── patch_experiment.py  # 因果 token 干预 causal token-patching experiments
└── scripts/
    ├── day1_test.py         # 环境自检 environment sanity check
    ├── day3_probing.py      # Probing 主流程 main probing pipeline
    ├── day5_behavior.py     # 行为评测主流程 main behavior-evaluation pipeline
    ├── server.py            # AutoDL 服务器工具（凭据走环境变量）server helpers (credentials via env vars)
    ├── check_server.py      # 远程环境检查 remote environment check
    └── deploy.py            # 上传代码并远程测试 upload code & run tests remotely
```

## Installation

```bash
pip install -r requirements.txt
```

You also need Hugging Face access to download `Qwen/Qwen2.5-VL-3B-Instruct`.

## Reproduce the experiment

```bash
# 0. Sanity check (data generation + model loading + hooks)
python scripts/day1_test.py

# 1. Generate a small dataset (test mode: 5 sequences per style)
python data_generation/generate_dataset.py --test

# 2. Generate the full dataset (6000 sequences)
python data_generation/generate_dataset.py --output ./data --num 2000

# 3. Linear probing, single-frame input
python scripts/day3_probing.py --data_dir ./data --output_dir ./results \
    --layers 0,8,16,24,31 --num_frames 1 --probe_type ridge

# 4. Non-linear probing, two-frame input
python scripts/day3_probing.py --data_dir ./data --output_dir ./results \
    --layers 16,24,31 --num_frames 2 --probe_type mlp

# 5. Behavior evaluation (100 test sequences)
python scripts/day5_behavior.py --data_dir ./data \
    --output_dir ./results_behavior --max_samples 100
```

Results are saved as JSON under `results/` and `results_behavior/`.

## Notes

- Experiments ran on an AutoDL RTX 5090 (32 GB) with `torch.bfloat16`; the 3B model fits comfortably.
- `scripts/server.py` / `scripts/check_server.py` are AutoDL helpers. **Never commit real server credentials**: provide them via environment variables (`AUTODL_HOST`, `AUTODL_PORT`, `AUTODL_USERNAME`, `AUTODL_PASSWORD`), see [.env.example](.env.example).

## License

MIT — see [LICENSE](LICENSE).

---

# 大模型真的"看见"运动吗？一个小猜想，一个小实验

我们花了很多时间看视觉语言模型描述图片。它们很擅长说出物体的名字、读出文字、甚至数出苹果的个数。但有一个更安静的问题一直在我们心里打转：当一个球在屏幕上一闪而过，这个模型是真的把**运动**编码进了它的视觉特征里，还是只是在猜词？

这个仓库就是我们为找到答案而做的一个小实验的记录。结果和我们预想的不一样——而且结尾的反转，是这个实验最精彩的部分。

---

## 实验设计：一个完全可控的微型世界

为了把问题问得干净，我们搭了一个完全可控的微型世界：用 [pymunk](https://www.pymunk.org/) 模拟了 **6000 段短物理视频**——1 到 3 个小球在盒子里碰撞弹跳，重力、弹性、摩擦随机变化——并渲染成三种视觉风格（`minimal` 极简 / `medium` 中等 / `realistic` 类真实）。每一帧都带有真值轨迹数据：位置、速度、加速度。这个世界没有任何歧义：如果模型在视觉流**任何地方**编码了速度，我们应该能找到它。

然后我们问两个问题：

1. **表征**：Qwen2.5-VL-3B-Instruct 每一层的视觉 token 里，能否用探针读出速度 `(vx, vy)`？
2. **行为**：模型的语言输出能否真正**使用**这些信息——比如预测小球往哪个方向运动？

---

## 第一章——静态图像是沉默的

我们从最简单的测试开始：在**单帧**图像上用 Ridge（线性）探针。如果速度以可读的线性方式存在于视觉 token 中，这一步就应该成功。

| Layer | R² | Baseline R² |
|-------|-----|-------------|
| 0  | -0.0046 | 0.9131 |
| 8  | 0.1644  | 0.9131 |
| 16 | 0.3043  | 0.9131 |
| 24 | 0.3409  | 0.9131 |
| 31 | 0.2794  | 0.9131 |

R² ≈ 0。静态图像几乎不含可线性解码的速度信息——这很合理：一张小球的照片并不能告诉你它运动得多快。恒定速度基线（"直接复用上一帧速度"）有 0.91。所以这不是失败，而是一次通过的 sanity check。

## 第二章——两帧也只是略微帮忙

也许模型需要时间维度。我们输入**连续两帧**，再次探测。

| Layer | R² | Baseline R² |
|-------|-----|-------------|
| 0  | 0.0187 | 0.9026 |
| 8  | 0.1793 | 0.9026 |
| 16 | 0.3202 | 0.9026 |
| 24 | 0.3598 | 0.9026 |
| 31 | 0.3519 | 0.9026 |

有进步，但只有一点点（+0.02 ~ 0.07 R²）。即使有了时序信息，线性探针在高层的峰值也只有 **0.36** 左右——仍远低于 0.90 的基线。如果速度就摆在特征里，这一步本该轻而易举。

## 第三章——信息以非线性形式隐藏着

线性探针失败，不代表信息不存在。它可能只是以一种"直线看不出来"的方式组织着。于是我们把探针换成一个小型 2 层 MLP。

| Layer | Ridge R² | MLP R² | 变化 Change |
|-------|----------|--------|------------|
| 16 | 0.3202 | **0.4164** | +0.096 |
| 24 | 0.3598 | 0.4075  | +0.048 |
| 31 | 0.3519 | 0.1543  | -0.198 |

在中层（16–24），MLP 跃升到 **R² ≈ 0.42**。速度确实在那里——只是以非线性形式编码，线性读取够不着。（Layer 31 反而下降，可能过拟合，也可能最后的视觉层已经把信号变换得稀疏了。）

## 第四章——反转来了

所以表征"知道"速度。然后我们直接用自然语言问模型：*"编号为 0 的球接下来最可能向哪个方向运动？"*（A/B/C/D 四选一，随机水平 25%。）

| 指标 Metric | 数值 Value | 解读 Reading |
|------------|-----------|--------------|
| 整体准确率 Overall accuracy | **15.0%** | 低于随机水平（25%）Below random chance (25%) |
| realistic | 14.3% (6/42) | 复杂背景下表现差 Poor on complex backgrounds |
| medium | 22.6% (7/31) | 接近随机 Close to chance |
| minimal | 7.4% (2/27) | 极简场景反而最差 Worst on the simplest scenes |

**15%——四个选项里比瞎猜还差。** 模型甚至表现出系统性答案偏好（特别喜欢选 A 或 C）。然而，同一批视觉 token 却能解码出 R² = 0.42 的速度。

这就是解耦：信息存在于表征中，但语言输出够不到它。视觉编码与语言生成之间存在一个"信息断层"——通俗地说，模型可以在特征里**持有**一个事实，却无法把它**说出来**。

## 假设总结

| 假设 Hypothesis | 结果 Result | 证据 Evidence |
|----------------|------------|---------------|
| H1: 位置可解码 Position decodable | 待验证 Pending | - |
| H2: 速度可解码 Velocity decodable | 部分成立 Partially supported | 非线性 R² = 0.42，线性 R² = 0.36 |
| H3: 表征-行为解耦 Dissociation | **强成立 Strongly supported** | 表征 42% vs 行为 15%，差距 ~27 个百分点 |

我们觉得这可以成为 workshop paper 的核心发现：*Qwen2.5-VL 的视觉表征中隐式编码了速度信息，但其语言输出无法利用这些信息进行物理推理。*

---

## 数据集

`data_generation/` 生成完全受控的数据集：

- 基于 **pymunk** 的 2D 物理模拟：1–3 个小球、随机重力、墙壁/小球碰撞、弹性与摩擦
- 每段 8 帧、448×448 分辨率
- 轨迹真值 JSON（逐帧位置、速度、加速度）与渲染 PNG 一同保存
- 共 6000 段：`minimal` / `medium` / `realistic` 各 2000 段
- 按序列划分 train/test（`split.json`）

## 仓库结构

```
vlm_kinematics_probing/
├── EXPERIMENT_LOG.md        # 完整实验记录（中文）
├── data_generation/
│   ├── physics_sim.py       # pymunk 2D 物理模拟
│   ├── renderer.py          # 三档渲染器 minimal / medium / realistic
│   └── generate_dataset.py  # 数据集生成 + train/test 划分
├── models/
│   ├── qwen_vl.py           # Qwen2.5-VL 加载 + 多层 forward hook
│   └── patch_utils.py       # 像素坐标 → 视觉 token 映射
├── probing/
│   ├── probes.py            # Ridge / MLP / 对照任务探针
│   ├── baselines.py         # 恒定速度与随机基线
│   └── extract_stream.py    # 流式表征提取
├── behavior/
│   └── question_gen.py      # 选项式问题生成与评测
├── intervention/
│   └── patch_experiment.py  # 因果 token 干预实验
└── scripts/
    ├── day1_test.py         # 环境自检
    ├── day3_probing.py      # Probing 主流程
    ├── day5_behavior.py     # 行为评测主流程
    ├── server.py            # AutoDL 服务器工具（凭据走环境变量）
    ├── check_server.py      # 远程环境检查
    └── deploy.py            # 上传代码并远程测试
```

## 安装

```bash
pip install -r requirements.txt
```

你还需要通过 Hugging Face 下载 `Qwen/Qwen2.5-VL-3B-Instruct` 的模型权重。

## 复现实验

```bash
# 0. 环境自检（数据生成 + 模型加载 + hook 验证）
python scripts/day1_test.py

# 1. 生成小数据集（测试模式：每风格 5 段）
python data_generation/generate_dataset.py --test

# 2. 生成完整数据集（6000 段）
python data_generation/generate_dataset.py --output ./data --num 2000

# 3. 线性 Probing，单帧输入
python scripts/day3_probing.py --data_dir ./data --output_dir ./results \
    --layers 0,8,16,24,31 --num_frames 1 --probe_type ridge

# 4. 非线性 Probing，双帧输入
python scripts/day3_probing.py --data_dir ./data --output_dir ./results \
    --layers 16,24,31 --num_frames 2 --probe_type mlp

# 5. 行为评测（100 段测试序列）
python scripts/day5_behavior.py --data_dir ./data \
    --output_dir ./results_behavior --max_samples 100
```

结果以 JSON 形式保存在 `results/` 和 `results_behavior/`。

## 注意事项

- 实验在 AutoDL RTX 5090（32 GB）上以 `torch.bfloat16` 运行，3B 模型可以轻松加载。
- `scripts/server.py` / `scripts/check_server.py` 是 AutoDL 辅助工具。**切勿提交真实服务器凭据**：请通过环境变量提供（`AUTODL_HOST`、`AUTODL_PORT`、`AUTODL_USERNAME`、`AUTODL_PASSWORD`），参见 [.env.example](.env.example)。

## 许可证

MIT 协议——详见 [LICENSE](LICENSE)。
