# Action_Spotting 三阶段训练方案

## 目标

用 `/home/chang/Dataset/Action_Spotting` 训练 VLM_spotting 的开放动作时刻检测模型，路线是：

```text
Stage 1: connector-only SFT
Stage 2: LLM LoRA SFT
Stage 3: GRPO / RLVR
```

当前假设窗口设置：

```text
window_frames = 600
stride = 600
```

按 annotation 粗算，全集 train 约 **25,078 windows**。如果第一阶段排除 `soccernetv2` 和 `finediving`，并使用 `fs_perf` 而不是 `fs_comp`，clean train pool 约 **15,804 windows**。

## 数据口径

推荐第一阶段训练集：

```text
TouchMoment
finegym
tennis
fs_perf
soccernet_ball, 少量窗口
```

暂缓或 held-out：

```text
finediving     -> OOD held-out
soccernetv2    -> 暂缓，2.083fps + 秒级标注 + 不可见事件噪声
fs_comp        -> 有标注但帧不完整时先不用，用 fs_perf 起步
```

## Stage 1: Connector-only SFT

目标：先把 V-JEPA / ViT 双路视觉 token 接到 frozen LLM，让 frozen Qwen 能读 motion token 并稳定 copy frame index。

训练状态：

```text
trainable = connector
frozen    = V-JEPA, Qwen-ViT, Qwen LLM
loss      = answer token CE
```

推荐规模：

```text
8K-12K windows
第一版用 10K windows
```

推荐采样：

```text
TouchMoment:     2.5K
finegym:         3.5K
tennis:          1.5K
fs_perf:         1.0K
soccernet_ball:  1.5K
```

负样本控制：

```text
positive windows: 70%-85%
none negatives:   15%-30%
```

注意：Stage 1 不建议直接全集训练。connector 容易学到 dataset prior、position prior、class prior、none prior，而不是通用视觉 token 到 LLM token 的映射。

## Stage 1 Filtering

用 Stage 1 checkpoint 跑完整 train pool，记录每个 window：

```text
pred frames
scores
parse failure
nearest GT distance
hit@0/1/2/4/8
over-predict count
under-predict count
none false positive
none false negative
dataset / type / fps / window length
```

用途：

```text
1. 诊断 connector 是否接通
2. 发现难样本和噪声样本
3. 为 Stage 2 采样提供 difficulty signal
```

## Stage 2: LLM LoRA SFT

目标：让 LLM readout 适应 connector 产生的视觉 token 和多数据集 query，同时保持 connector 已学到的视觉对齐。

训练状态：

```text
trainable = LoRA + connector
connector lr 低
LoRA lr 正常
```

推荐规模：

```text
12K-18K windows
第一版用 15K windows
```

推荐采样：

```text
TouchMoment:     3K
finegym:         5K
tennis:          2K
fs_perf:         1K
soccernet_ball:  2K
soccernetv2:     0-2K clean subset, 第一版可先不加
finediving:      0, 保留 OOD
```

推荐超参范围：

```text
connector lr: 1e-5 ~ 5e-5
LoRA lr:      5e-5 ~ 1e-4
LoRA rank:    16
early stop:   以 in-domain val + finediving OOD 共同判断
```

Stage 2 之后再跑一次 filtering，覆盖完整 25K train pool，作为 GRPO 数据来源。

## Stage 3: GRPO / RLVR

目标：短程 metric calibration，修正边界偏移、漏检、过检和 none 错误。不要把 GRPO 当成重新训练模型。

训练状态建议：

```text
方案 A: trainable = LoRA, connector frozen
方案 B: trainable = LoRA + connector low-lr
```

推荐规模：

```text
2K-4K filtered hard windows
第一版用 3K windows
```

GRPO 数据从 Stage 2 filtering 结果中选，不随机采样。建议构成：

```text
30% boundary error 小但未命中，例如 3-10 帧偏差
25% under-predict，漏检
20% over-predict，多检
15% none false positive / none false negative
10% 长窗口 / 少见类 / 跨数据集 query
```

不要大量选择明显标注噪声、不可见事件或完全不可学样本。

Reward 建议：

```text
point spotting:
  reward = matched_F1@tol
  tol = 2/4/8 帧混合

可选轻微过检惩罚:
  reward = matched_F1@tol - 0.05 * extra_predictions
  reward clamp 到 [0, 1]
```

如果未来改成区间 temporal grounding：

```text
reward = temporal IoU(pred_span, gt_span)
```

推荐 GRPO 超参：

```text
num_generations: 4 起步，稳定后再试 8
steps:           100-300
lr:              1e-6 ~ 5e-6
early stop:      val mAP@2 或 OOD mAP 不涨就停
```

## 第一版完整配方

```text
1. Connector-only SFT
   10K clean windows
   trainable connector only
   eval in-domain + finediving OOD

2. Filtering
   run full train pool
   save per-window prediction/error/failure metadata

3. LoRA SFT
   15K sampled windows
   trainable LoRA + low-lr connector
   keep finediving held-out

4. Filtering
   run full 25K train pool
   select 3K hard windows

5. GRPO / RLVR
   3K hard windows
   num_generations=4
   100-200 steps first
   reward = matched F1 / frame-distance shaped reward
```

## 核心原则

```text
connector 阶段: 少量、干净、平衡，先接通视觉 token
LoRA SFT 阶段: 扩大数据，但 early stop，避免读出层过拟合
GRPO 阶段: 只吃 filtered hard set，短训，直接对齐评测指标
```

全集 25K windows 更适合作为 filtering pool 和后续扩训池，不适合作为 Stage 1 connector 的自然分布全集训练输入。
