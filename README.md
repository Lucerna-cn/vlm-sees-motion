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
| Overall accuracy (500 samples) | **23.6%** (118/500) | Still below random chance (25%) |
| realistic | 22.0% (38/173) | Poor on complex backgrounds |
| medium / minimal | pending | - |

We scaled up from 100 to **500 samples** to make sure this wasn't a fluke — and it held: **23.6%**, still below the 25% coin flip. Yet the very same visual tokens decode velocity at R² = 0.42.

That is the dissociation: the information exists in the representation, but the language head cannot get at it. There is an information bottleneck between visual encoding and language generation — or, in plainer words, a model can *hold* a fact in its features and still fail to say it.

## Chapter 5 — It's not guessing, it's systematic

Was the model just bad at the task — or is something more specific going on? We looked at *every* answer across the 500 samples. The model is not guessing randomly; it has a fixed, systematic answer bias.

| Option | Model predicts | True answer | Bias |
|--------|---------------|-------------|------|
| A | 5.8% | 29.0% | **-23.2%** |
| B | 11.2% | 26.4% | -15.2% |
| C | **72.4%** | 22.2% | **+50.2%** |
| D | 10.6% | 22.4% | -11.8% |

**72.4% of all answers were "C" (left) — while only 22.2% of the ground-truth answers are C.** The confusion matrix tells the same story: 69–75% of non-C cases get misclassified as C, while true-C cases are answered correctly 77% of the time. The bias is strongest on visually complex inputs (realistic: 87.9% C) and with a single ball (77.8% C), and weakens on minimal scenes (49.3%).

This sharpens the story: the model doesn't fail randomly — it has a **deterministic wrong mapping** from visual input to "C", completely disconnected from the physical information in its own representations. Not noise: a fixed error pattern.

## Chapter 6 — Poking the layers: a causal test

The language head doesn't use the velocity information — but can we *prove* it causally? We patched the tokens where velocity lives (Layers 16 & 24) by replacing them with their mean, then re-ran the behavior test (100 sequences × 2 layers = 200 interventions).

| Metric | Value | Reading |
|--------|-------|---------|
| Normal accuracy | 20.0% | Baseline |
| Patched accuracy | 17.5% | After destroying velocity layers |
| Behavior change rate | 18.0% | Some answers changed |
| **Accuracy drop rate** | **4.5%** | **Key metric** |

**Destroying the velocity-information layers changed the language output by less than 5%.** If the language head were reading those representations, corrupting them should have wrecked its answers. It barely noticed. This is the causal half of the dissociation: the information exists in the visual stream, and the language head simply does not use it.

## Chapter 7 — A sanity check that makes it airtight

Before claiming the probe "found" velocity, we had to prove the probing method itself works. We probed for something a vision encoder must know — **ball position (x, y)** — with the same setup.

| Layer | R² |
|-------|-----|
| 0 | 0.3402 |
| 8 | **0.9388** |
| 16–31 | running |

**Position decodes linearly at R² ≈ 0.94.** The pipeline can extract information from these representations with ease. So the velocity result is not a method artifact — the model genuinely encodes *where* a ball is, and genuinely fails to encode *how fast it moves* in a comparable way.

| Information | Decodability | Encoding |
|-------------|--------------|----------|
| Position | Very high (R² = 0.94) | Linear |
| Velocity | Moderate (R² = 0.42) | Non-linear |
| Acceleration | Expected very low | ? |

## Chapter 8 — It's not the question format

Could the multiple-choice format be the problem? We asked the same question **open-endedly** (200 samples). Accuracy: **28.0%** vs. 23.6% with options — slightly better, still around chance (25%). The model doesn't fail because of the A/B/C/D wrapper; it fails at the task itself.

| Format | Accuracy |
|--------|----------|
| Multiple-choice | 23.6% |
| Open-ended | 28.0% |
| Unparseable | 3.0% |

## Hypotheses summary

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1: position decodable | **Confirmed** | Linear R² = 0.94 |
| H2: velocity decodable | Partially supported | Non-linear R² = 0.42, linear R² = 0.36 |
| H3: representation–behavior dissociation | **Extremely strongly supported** | Representation 42% vs. behavior 23.6%; causal intervention effect <5% |

The story is now backed by three independent layers of evidence:

1. **Representation**: velocity exists in the visual stream, but non-linearly (R² = 0.42)
2. **Behavior**: the language head answers with a systematic bias (72.4% pick C) at 23.6% accuracy
3. **Causality**: destroying the velocity layers changes language output by less than 5%

*Visual encoding and language generation are functionally disconnected — the model can represent where a ball is and how it moves, but it cannot say either.*

A working title for this story: **"Do Vision-Language Models Know Where the Ball Goes? Evidence of Functional Decoupling between Visual Representation and Language Generation"**

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
├── analysis/
│   └── answer_bias.py       # 回答偏向分析 answer-bias analysis
├── intervention/
│   └── patch_experiment.py  # 因果 token 干预 causal token-patching experiments
└── scripts/
    ├── day1_test.py         # 环境自检 environment sanity check
    ├── day3_probing.py      # Probing 主流程 main probing pipeline
    ├── day5_behavior.py     # 行为评测主流程 main behavior-evaluation pipeline
    ├── day5_behavior_open.py # 开放式行为评测 open-ended behavior evaluation
    └── day6_intervention.py  # 因果干预实验 causal intervention experiments
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

- Experiments ran on an RTX 5090 (32 GB) with `torch.bfloat16`; the 3B model fits comfortably.
- A full 32-layer MLP probing run is in progress (~48h); results will be added when it completes.

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
| 整体准确率 Overall accuracy | **23.6%** (118/500) | 仍低于随机水平（25%）Still below random chance (25%) |
| realistic | 22.0% (38/173) | 复杂背景下表现差 Poor on complex backgrounds |
| medium / minimal | 待统计 pending | - |

我们把样本从 100 扩到 **500 段**来确认这不是偶然——结果稳住了：**23.6%**，仍然低于 25% 的随机水平。然而，同一批视觉 token 却能解码出 R² = 0.42 的速度。

这就是解耦：信息存在于表征中，但语言输出够不到它。视觉编码与语言生成之间存在一个"信息断层"——通俗地说，模型可以在特征里**持有**一个事实，却无法把它**说出来**。

## 第五章——不是猜，是系统性的错

模型只是"不擅长这个任务"，还是另有玄机？我们把 500 个样本的**每一个回答**都翻出来分析——模型根本不是随机猜，它有一个固定、系统性的答案偏向。

| 选项 Option | 模型预测 Model predicts | 正确答案 True answer | 偏差 Bias |
|------------|------------------------|---------------------|-----------|
| A | 5.8% | 29.0% | **-23.2%** |
| B | 11.2% | 26.4% | -15.2% |
| C | **72.4%** | 22.2% | **+50.2%** |
| D | 10.6% | 22.4% | -11.8% |

**72.4% 的回答都是"C"（左）——而正确答案里 C 只占 22.2%。** 混淆矩阵讲的是同一个故事：69–75% 的非 C 案例被误判为 C，而真实为 C 的案例有 77% 答对。偏向在视觉复杂的输入上最严重（realistic 场景 87.9% 选 C）、单个球时也明显（77.8%），在极简场景（minimal）里则降到 49.3%。

这把故事从"模型不会做物理推理"变成了更尖锐的版本：**模型有一个确定性的错误映射——把视觉输入系统性映射到"C"，与它自己表征里的物理信息完全脱节。**不是随机噪声，而是固定的错误模式。

## 第六章——戳一戳这些层：因果验证

表征"知道"速度，但语言输出就是不用——这能**证明**吗？我们把速度信息最强的 Layer 16、24 的球 token 全部替换成均值（patch），再跑行为测试（100 段 × 2 层 = 200 次干预）。

| 指标 | 数值 | 解读 |
|------|------|------|
| 正常准确率 | 20.0% | 基线 |
| Patch 后准确率 | 17.5% | 破坏速度层之后 |
| 行为改变率 | 18.0% | 部分回答发生变化 |
| **准确率下降率** | **4.5%** | **关键指标** |

**破坏速度信息层后，语言输出只下降了不到 5%。** 如果语言生成真的在读取这些表征，篡改它们应该会摧毁回答——但它几乎无动于衷。这就是解耦的因果半边：信息在视觉流里，语言头就是不用。

## 第七章——让结论无懈可击的 sanity check

在宣称探针"找到"速度之前，必须先证明探针方法本身有效。我们用同样的流程去探测视觉编码器必须知道的东西——**球的位置 (x, y)**。

| Layer | R² |
|-------|-----|
| 0 | 0.3402 |
| 8 | **0.9388** |
| 16–31 | 进行中 |

**位置可以以 R² ≈ 0.94 线性解码。** 这套流程提取表征信息毫无压力。所以速度的结果不是方法假象——模型真的编码了球**在哪**，却真的没有以可比的方式编码球**动得多快**。

| 信息类型 | 可解码性 | 编码方式 |
|---------|---------|---------|
| 位置 | 极高 (R²=0.94) | 线性 |
| 速度 | 中等 (R²=0.42) | 非线性 |
| 加速度 | 预期极低 | ? |

## 第八章——不是提问方式的问题

会不会是选项式提问限制了模型？我们把同样的问题改成**开放式**（200 段样本）：准确率 **28.0%**，选项式是 23.6%——略好一点，但仍在随机水平（25%）附近打转。模型失败不是因为 A/B/C/D 的外壳，而是任务本身。

| 提问方式 | 准确率 |
|---------|--------|
| 选项式 | 23.6% |
| 开放式 | 28.0% |
| 无法解析 | 3.0% |

## 假设总结

| 假设 Hypothesis | 结果 Result | 证据 Evidence |
|----------------|------------|---------------|
| H1: 位置可解码 Position decodable | **✓ 成立 Confirmed** | 线性 R² = 0.94 |
| H2: 速度可解码 Velocity decodable | 部分成立 Partially supported | 非线性 R² = 0.42，线性 R² = 0.36 |
| H3: 表征-行为解耦 Dissociation | **极强成立 Extremely strongly supported** | 表征 42% vs 行为 23.6%；因果干预影响 <5% |

现在这个故事由三层相互独立的证据支撑：

1. **表征层**：速度信息存在于视觉流中，但是非线性的（R² = 0.42）
2. **行为层**：语言头用系统性偏向作答（72.4% 选 C），准确率仅 23.6%
3. **因果层**：破坏速度信息层后，语言输出变化不到 5%

*视觉编码与语言生成之间存在功能性断层——模型能表征球在哪里、怎么动，却一个字都说不出来。*

建议标题：**"Do Vision-Language Models Know Where the Ball Goes? Evidence of Functional Decoupling between Visual Representation and Language Generation"**

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
├── analysis/
│   └── answer_bias.py       # 回答偏向分析
├── intervention/
│   └── patch_experiment.py  # 因果 token 干预实验
└── scripts/
    ├── day1_test.py         # 环境自检
    ├── day3_probing.py      # Probing 主流程
    ├── day5_behavior.py     # 行为评测主流程
    ├── day5_behavior_open.py # 开放式行为评测
    └── day6_intervention.py  # 因果干预实验
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

- 实验在 RTX 5090（32 GB）上以 `torch.bfloat16` 运行，3B 模型可以轻松加载。
- 全 32 层 MLP probing 正在进行（预计约 48 小时），完成后会补充结果。

## 许可证

MIT 协议——详见 [LICENSE](LICENSE)。
