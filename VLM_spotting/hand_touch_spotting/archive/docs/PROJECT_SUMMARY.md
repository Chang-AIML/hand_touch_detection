# VLM Spotting — 项目总结

> HOI4D 手-物 **touch/untouch 时刻检测**（moment spotting）：用**冻结 VLM + 视觉压缩连接器 + 语言 query → 生成帧号**的方式，精确定位"手接触物体""手松开物体"发生在第几帧。
> 数据集 `hand_touch_detection/data/HOI4D-v3`（train/val/test），评测 = action-spotting **mAP@0/1/2**（touch/untouch 两类平均）。

---

## 1. 任务与评测

- **输入**：一段视频 + 一个自然语言问题（"When does the hand touch/release the object?"）。
- **输出**：该类型**所有事件的帧号列表**（可能 0/1/多个），例如 `45, 120, 200`。
- **指标（E2E eval）**：`eval_nms.maps_quiet → common.score.compute_average_precision`。
  - 按分数排序 + 一对一最近未匹配 GT 匹配 + 插值精度，`AP = Σ(interp precision) / #GT`。
  - 报的数字 = **mAP@2**（容差 2 帧，15fps ≈ 133ms），touch/untouch 两类 AP 取平均。
  - **只有 AP 是 eval 认的数**；precision / recall 是诊断分解，不是 eval 指标。

---

## 2. 方法架构

```
每帧图像
  └─ V-JEPA 2.1 ViT-B (384px, patch16) ──► 24×24 = 576 spatial tokens × 768d  (grid_even, 单遍抽取)
        └─ FrameCompress: 8 个 learnable query 对 576 patch 做 attention-pool
              └─ concat(8×768=6144) → adaptor → 4096d，且 RMS 对齐 LLM 词嵌入
                    └─ 每帧 1 个"motion token"（4096d，视觉、无语言）
视频区序列: [帧号digit][motion_f] [帧号digit][motion_f] ... + 每隔5s插一张 ViT anchor
  + 问题文本 + " Answer:" 
    └─ 冻结 Qwen3-VL-8B ──► 生成帧号（idx-copy：从事件那一帧把"帧号把手"抄出来）
```

**关键设计**：
- **query-conditioned 压缩**：8 个 query 把 576 patch 压成 1 个富信息 token（填满 4096 维），远好于 mean-pool（rank-768 稀释 token）。
- **RMS 对齐**：注入的视觉 token 必须匹配 LLM 词嵌入的 RMS norm，否则冻结 LLM 崩掉。
- **idx-copy 机制**：输入端每个 motion token 前挂一个真实数字 token 作"帧号把手"，模型输出即"抄"事件帧的把手 → domain/taxonomy 无关。
- **语言由 LLM 在 prompt 里 grounding**，连接器保持纯视觉（见 §4 结论）。

---

## 3. 数据的 Q 和 A 怎么构造

- **样本单位** = 一个 `(视频, 事件类型)` 对；一个视频有 touch+untouch 就拆成 2 个样本。
- **Q（问题）**：`data/questions.py` 同义句库，切 `train`（8 种说法）/ `test`（4 种换动词的 held-out 说法）。
  - 训练随机抽 train 措辞（措辞增强）；主结果 eval 用固定的 `GENERIC_Q`（措辞不作混淆变量）。
  - held-out 措辞是"是否真 grounding 语言"的探针（Q3 设计）。
- **A（答案）**：该类型所有事件帧号、排序、逗号分隔的**文本**（`" 45, 120, 200" + <eos>`），无事件则 `"none"`。
- **训练目标**：teacher-forced 交叉熵，**只在答案数字+EOS 上算 loss**。
- **推理**：贪心生成 → 正则 `\d+` 抠帧号；每个检测分数 = 其数字 token 的平均 softmax 概率（mAP 排序用）。

---

## 4. 核心实验结果

### 4.1 连接器：压缩赢，语言注入伤（全部冻结 Qwen，val mAP@2）

| 连接器 | val mAP@2 |
|---|---|
| mean-pool baseline（feat_interleave, 768/帧） | 50.1 |
| **8-query 压缩，无语言（M1）** | **67.6** (+17.5) ✅ |
| 压缩 + 文本条件 query（问题进 query） | 11.5 ❌ |
| 压缩 + staining（patch 交叉注意问题）+ 文本 query | 2.2 ❌ |
| 压缩 + gated（init-0）语言 | 0.67 ❌ |

**结论**：**语言注入 V-JEPA→VLM 连接处 HURTS，query 压缩（纯视觉）帮助巨大。** 机制 = LLM 已在 prompt 里有问题，预先把语言混进视觉 token 会污染它们（**LLaVA 干净视觉 token > InstructBLIP 预注入语言**的教训）。连接器保持纯视觉、压硬（1 富 token/帧），语言交给 LLM。

### 4.2 SFT 最强模型（LLaVA 两阶段：先冻结连接器 → 再 +LoRA 解冻）

| 模型 | val @0/@1/@2 | test @0/@1/@2 |
|---|---|---|
| M1 frozen-compress | ~/~/67.6 | ~/~/~64.3 |
| **SFT-LoRA（sft_lora_v2，最强）** | 9.42 / 46.97 / **72.52** | 7.77 / 45.53 / **72.94** |

- SFT-LoRA **零 val→test 掉点**（比 M1 泛化更好）。**这是全项目最强模型。**
- 消融确认：LLM 是原始 Qwen3-VL-8B（未载入别的权重），只训连接器（0.34% 参数）就到 67.6。

### 4.3 RLVR reward 探索 —— 结论：换 reward 打不过 SFT@2（算法天花板）

目标（/goal）：找一个 reward 让 RLVR 超越 SFT。从 SFT init（72.52）出发，GRPO：

| reward | 机制 | @1（起46.97） | @2（起72.52） |
|---|---|---|---|
| sharp | 精确度分层 + 重罚多检 | ↑48.4 | 71.0–71.5 |
| f1 | 集合 F1@tol1 | ↑47.1 | 71.5 |
| graded | 2帧内给分 + 精确度 | ↑48.0 | 70.5–71.6 |
| **map** | **2帧内全平（无精确度梯度）** | ↑47.4 | **70.8** |
| KL-anchored kl=1/6/20 | 锚定到 SFT 防漂移 | 冻结 | 71.0–71.5（压不住 72.52）|

- **5 种 reward + 3 种 KL 强度，全部 @2 掉 ~1.5、@1 涨 ~1.5，没有任何 eval 越过 72.52**（best.pt 从未保存）。
- **机制**：SFT 已在**贪心解码的 @2 最优点**上；策略梯度优化的是**采样 rollout**，任何更新都扰动贪心 digit 预测 + 改变分数校准 → mAP@2 排序变差。**reward 形状只决定往哪扰动，改不了扰动本身；KL 只能"漂移↔冻结"两头不讨好。**
- **结论**：本任务（idx 生成 + 贪心解码 + GT 监督的 SFT 已最优）**RLVR 打不过 SFT@2，天花板就是 SFT**。RLVR 唯一稳定效果是小幅 @1/精确度增益。要真超越需**超出 GT 的新监督**或**非贪心评测**，都不是"选 reward"能解决的。

---

## 5. 可解释性（最强模型的 token 语义）

三个角度，弱→决定性：

| 方法 | 结果 | 判定 |
|---|---|---|
| 词表投影（token→最近 LLM 词） | 乱码（罕见 unicode/代码 token，无 hand/touch 词） | ❌ 非语言对齐 |
| Attention 热力图（8 query 看哪） | 弥散，接触帧仅略更集中（SFT 92 vs 83） | 〰️ 空间上不明确 |
| **线性探针**（单 token 分类接触/非接触，held-out val 平衡） | **AUC 0.89, acc 0.82** | ✅ **决定性** |

**结论**：接触信息能从**单个压缩 token 线性解码**（语义真实但**分布式**，非单区域、非词形）。SFT 几乎不改探针（0.889→0.892）→ **连接器早已编码好接触；SFT 的 +5 mAP 来自 LLM(LoRA) 更会"读"这个 token，不是 token 变好了。**

---

## 6. 指标深入分析（precision / recall / AP 对齐）

在 test（ndet=1980 ≈ ngt=1945，枚举数量几乎精确）：

| 容差 | precision（一对一匹配） | recall（覆盖率） | **AP = E2E eval** |
|---|---|---|---|
| @0 | 24% | 25% | **7.8** |
| @1 | 60% | 63% | **45.5** |
| @2 | 79% | 83% | **72.9** |

- **AP ≠ precision × recall**（0.79×0.83=0.66≠0.73）。AP = 排序后 PR 曲线下面积，比 prec/rec 都低（要两者同时高 + 分数排序），但比乘积高（用插值精度）。
- **@0 崩塌（AP 7.8 << prec 24）**：分数排不出"精确到帧"的命中（模型对"大概哪帧"有信心、对"精确帧"不区分）→ **分数校准是 @0 瓶颈**（与 RLVR 结论呼应）。
- **mAP@δ 曲线（test）**：

  | δ帧 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
  |---|---|---|---|---|---|---|---|---|---|
  | **mAP** | 7.8 | 45.5 | **72.9** | 82.8 | 88.2 | 90.1 | 91.4 | 92.2 | 92.9 |
  | touch | 5.5 | 37.0 | 67.0 | 79.7 | 87.3 | 90.2 | 92.0 | 93.1 | 93.9 |
  | untouch | 10.1 | 54.1 | 78.9 | 86.0 | 89.2 | 90.1 | 90.8 | 91.4 | 91.8 |

  - 前段极陡（0→2 涨 65 点），δ≥5 饱和 ~90，δ=10 也只 92.9（剩 ~7% 硬漏检天花板）。
  - **untouch（松手）紧容差更强、touch（抓取）松容差略高**：抓取"定位飘"（接触是渐变），松手"要么准要么彻底漏"（分离清晰）。

---

## 7. 代码结构（`hand_touch_spotting/`）

| 文件 | 作用 |
|---|---|
| `models/frame_compress.py` | **FrameCompress**：3 步 cross-attn，8 learnable query 压缩，`--stain/--use_text/--gate_lang`，RMS 对齐 |
| `models/idx_localizer.py` | idx-copy 定位器：prompt 组装、`_build`、`predict_multievent_batch`、`_answer_str` |
| `models/vjepa_adaptor.py` | mean-pool 路径的 adaptor（baseline） |
| `data/idx_grid_dataset.py` | 加载 `grid_even`（N/2,576,768）的多事件数据集 |
| `data/questions.py` | 问题同义句库（train/test 措辞分离） |
| `train/idx_compress_train.py` | Stage-1 训练：`--mode compress|mean`、`--lora`、`--init_fc`（两阶段） |
| `train/idx_rlvr_train.py` | GRPO/RLVR：reward `{tiered,smooth,map,sharp,f1,graded}` + `--kl` 锚定 |
| `scripts/67_extract_even_grids.py` | 单遍抽 V-JEPA 576-grid（分片 2 卡，线程/内存受控） |
| `scripts/68_idx_test_eval.py` | test 集评测某 checkpoint |
| `scripts/69–71` | 可解释性：词表投影 / attention 热图 / **线性探针** |
| `scripts/72–75` | 误差分布 / **precision 分布** / **PR 曲线** / **mAP@δ 曲线** |

**数据/环境**：
- 特征 `VLM_spotting/vjepa/grid_even`（单遍、未池化 576-token grid，fp16，2850 视频，~134MB/视频）。
- Python：`/home/chang_noroot/data2/myconda/envs_dirs/vlm_spot/bin/python`；跑 `CUDA_VISIBLE_DEVICES=0/1 OMP_NUM_THREADS=8`。
- 最强 checkpoint：`outputs/idx_compress/sft_lora_v2/best.pt`（fc+lora）；`outputs/idx_compress/comp_notext/best.pt`（M1）。
- **安全线**：batch=2（SFT）/ B=8 chunked-logprob（RLVR），98GB GPU，线程 ≤8/进程（历史上 OOM 把服务器搞崩过）。

---

## 8. 关键结论（一句话版）

1. **压缩 > mean-pool（67.6 vs 50），语言注入连接器 HURTS** —— 连接器纯视觉，语言交给 LLM。
2. **SFT-LoRA（冻结→解冻两阶段）= 72.5 val / 72.9 test，是最强模型。**
3. **RLVR 打不过 SFT@2**：换 reward / 加 KL 都不行，是算法天花板（RL 扰动 SFT 贪心最优点），不是 reward 选错。
4. **token 语义分布式**：接触可线性解码（AUC 0.89），但非词形、非单区域；SFT 增益来自 LLM 更会读 token。
5. **@0 瓶颈 = 分数校准**（分数排不出精确帧），这也是 RLVR 想补补不动的地方。

---

## 9. 未来方向：开放动作时刻检测（open action moment detection）

把本任务泛化：**混多个数据集**训一个开放词表的 moment 检测器。

- **架构现成**：冻结 VLM + 视觉压缩 + 语言 query + idx 生成 = 天然开放词表、domain 无关。**touch/untouch 本质就是"抓取"区间的 start/end**，泛化到任意动作起止是同一件事放大。
- **统一原语**：`(自然语言 query, [帧号列表])`；区间事件拆成 start+end；query-mode token 区分"找全部"vs"找匹配描述的那一个"。
- **可混数据集**：SoccerNet-v2 / Ego4D-PNR·state-change / Kinetics-GEBD（最"open"）/ HOI4D（精确点事件，先做）；Charades-STA / QVHighlights / Ego4D-NLQ（语言查询区间，后并入）。
- **真正的坑在数据口径**（非模型）：精度不一致（帧级 vs 秒级）、枚举语义冲突（穷举 vs 找一个）、moment 定义统一、fps/时长归一、缺席负样本。
- **必做验证**：能否定位**训练里没见过的动作类型/说法**（held-out 措辞探针）—— 这才是"真开放"vs"多域闭集"的判据。

---

## 10. 产物图表清单（`hand_touch_spotting/plot/`）

| 图 | 内容 |
|---|---|
| `compress_error_dist_test.png` | 每事件误差分布：M1 → SFT-LoRA（误差往 @0/@1 拉） |
| `precision_dist_test.png` | 预测精度分布 + precision/recall/**AP** vs 容差（红线=E2E） |
| `pr_curve_test.png` | PR 曲线（面积=AP=E2E eval），touch/untouch 分开 |
| `ap_vs_tol_test.png` | **mAP@δ 曲线**（E2E eval 随容差 0–10） |
| `attn_interp.png` / `attn_interp_sft.png` | 8-query attention 热图（接触 vs 非接触） |

---

*记忆索引*：`stage1-language-injection-verdict.md`（语言注入结论 + query 压缩 + SFT/RLVR 全过程）、`method-search-verdict.md`、`vlm-spotting-project.md`、`astrm-*`。
