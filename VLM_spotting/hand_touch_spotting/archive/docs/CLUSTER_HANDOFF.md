# Cluster Handoff — 从 HOI4D touch/untouch 走向「开放动作时刻检测」

> **这个包是什么**：一个已跑通的、基于**冻结 VLM + 视觉压缩 + 语言 query + idx 生成**的时刻检测项目（HOI4D touch/untouch，最强 mAP@2 = 72.9 test），加上我们讨论定下来的**下一步 = 多数据集混训**的完整计划。
> **你要在 cluster 上干什么**：把训练路径从「预抽特征落盘」改成 **on-the-fly（实时算 V-JEPA）**，然后**混多个 spotting 数据集联合训 LoRA**，目标是让读出层能跨数据集迁移（现在不能）。
> 配套文档：`TECHNICAL_REPORT.md`（细节完整报告）、`PROJECT_SUMMARY.md`（速览）。本文件是**给 cluster 落地用的行动手册**。

---

## 0. TL;DR — 先做什么

1. **拿代码 + 2 个 checkpoint（M1、SFT-LoRA）过去**，修硬编码路径（§4）。
2. **第 0 步实验（最便宜、先做）**：HOI4D + TACO 联合 SFT，看 TACO 零样本能不能从 **15 → 45+**（验证「混训修读出」假设）。用现有缓存特征，~1-2h。
3. **有效 → 改 on-the-fly 训练路径（§5.2）→ 子采样多数据集验证（半天）→ 全量混训**。
4. **硬件**：维持「冻结 8B + LoRA」就选 **8×A100-40G（无 NVSwitch 也行）**；真正的瓶颈是数据管线不是通信（§5.3）。

---

## 1. 项目一句话 + 已确定的结论（**别重走这些死路**）

**方法**：V-JEPA 2.1 ViT-B 每帧出 24×24=576 个 patch token → **8 个 learnable query 注意力压缩**成 1 个填满 4096 维的富 token（RMS 对齐）→ 注入**冻结 Qwen3-VL-8B** → LLM 用 **idx-copy** 生成事件帧号。语言 query 在 prompt 里、由 LLM grounding。

**已确认、不要再试的结论**：
- ✅ **压缩 >> mean-pool**（val mAP@2 67.6 vs 50）。瓶颈是 token 质量，不是 LLM 能力。
- ❌ **在连接处注入语言 HURTS**（11.5 / 2.2 / 0.67）。连接器保持**纯视觉**，语言交给 LLM（LlaVA > InstructBLIP 的教训）。**别再往连接器注语言。**
- ✅ **SFT-LoRA（LlaVA 式冻结→解冻）= 72.5 val / 72.9 test**，是最强模型，val→test 零掉点。
- ❌ **RLVR 打不过 SFT@2**：5 种 reward + 3 种 KL 强度全败，是**算法天花板**（策略梯度扰动 SFT 贪心最优点），不是 reward 选错。**别再花时间调 RLVR reward。**
- ✅ **可解释性**：线性探针 AUC 0.89 —— 接触信息从单个压缩 token **线性可解码**（语义真实但分布式）。SFT 的增益来自「LLM 更会读 token」，不是 token 变好。

---

## 2. 核心对话决策历史（时间线 + 为什么）

1. **早期两阶段**（LLM→idx → MS-TCN refine）→ 结论：refine 有用但瓶颈在 token 表示 → **转向 query 压缩**。
2. **核心实验**（§1）：压缩赢、语言注入伤、SFT 最强、RLVR 打不过。
3. **可解释性三连**（词表=乱码、attention=弥散、**线性探针 AUC 0.89**）。
4. **指标对齐**：只有 **AP（mAP@δ）= E2E eval**；precision/recall 是诊断分解。@0 只有 7.8 是因为**分数排不出精确帧**（分数校准是 @0 瓶颈）。
5. **迁移研究（这次的重点，直接决定 cluster 计划）**：
   - 新数据 **TACO**（touchmoment 里的 C* 视频，双手工具-物体；HOI4D 是单手家居）。
   - **探针（表示层）**：HOI4D 训的连接器在没见过的 TACO 上，接触仍**线性可解码（within-TACO 0.82）** → 表示**通用**。
   - **零样本端到端（读出层）**：SFT-LoRA 在 TACO 上 **mAP@2 只有 15**（同域 73），M1 **20**（同域 63）→ **模型不迁移**。
   - **关键反转**：**M1（全冻结）迁移比 SFT-LoRA 好（32% vs 21% 保留率）** → **微调越贴合 HOI4D，越搬不走**。
   - **结论**：**连接器迁移、读出不迁移** → 想要跨数据集可用的模型，**必须混训**（修读出层，连接器可冻着不动）。探针会让你过度乐观，端到端才是实锤。
6. **question 多样性讨论**：**内容丰富的 query（点名物体/工具/动作/模式）能把定位路由到 LLM 可迁移的语言知识上**，可能比混数据更根本 —— 但**先验证 question 是否 load-bearing**（现在用固定 GENERIC_Q，模型可能忽略问题）。
7. **多数据集资源讨论**：**磁盘是墙不是算力**（见 §5）。
8. **硬件讨论**：冻结 8B + LoRA + 数据并行 → **NVSwitch 用不上**，8 卡纯赢。

---

## 3. 环境与依赖

- **Python 3.11**，`torch 2.10.0+cu128`（CUDA 12.8），`transformers 5.12.1`，`numpy 2.4.4`，另需 `Pillow, matplotlib`。
- **Qwen3-VL-8B**：`Qwen/Qwen3-VL-8B-Instruct`（HF 自动下载，需 `transformers.Qwen3VLForConditionalGeneration`）。
- **V-JEPA 2.1 ViT-B**：**本地 repo + checkpoint**（不是 pip 包）：
  - repo：`repos/vjepa2`（native 仓库，`vjepa2_1_vit_base_384`）
  - ckpt：`repos/feature_extraction/ckpts/vjepa2_1_vitb_dist_vitG_384.pt`
  - 可用环境变量覆盖：`VJEPA_REPO` / `VJEPA_CKPT`。**这两样要一起拷到 cluster**（包里附了 requirements 说明，但 V-JEPA 权重较大，见 §10）。
- eval 依赖 `hand_touch_detection/` 里的 `common/score.py` + `methods/spot_head/eval_nms.py`（action-spotting AP）。**这个 sibling 仓库也要带上**（或至少这两个文件）。

---

## 4. ⚠️ 代码在 cluster 上会炸的地方（务必先改）

- **61 个源文件硬编码了 `/home/chang_noroot/data2/huyanh/Workspace/...` 绝对路径**（`_COMMON / LAB / GRID / FRAMES / OUT / VJEPA_*`）。cluster 上这些路径不存在 → 先全局替换成你的路径，或**抽成一个 `paths.py / 环境变量`**（推荐后者，顺手把 on-the-fly 改造一起做）。
- 关键路径常量（在多个脚本头部）：
  - `LAB` = HOI4D-v3 标注 json；`GRID` = `vjepa/grid_even`（预抽特征）；`FRAMES` = `dataset/hoi4d/frames`；`TM` = `dataset/hoi4d/touchmoment`。
- **特征（grid_even）不在包里**（357GB+）。cluster 上要么重抽、要么直接走 on-the-fly（推荐，见 §5.2）。
- **数据集本身不在包里**：HOI4D / TACO 的 frames、以及要混的 sports 数据集，需你在 cluster 上准备。

---

## 5. 下一步计划：多数据集混训

### 5.1 数据集 + 帧数（train split）
| 数据集 | train clip | train 帧 | grid_even 磁盘(预抽) | 状态 |
|---|---|---|---|---|
| finegym | 3,327 | 4.69M | 1.97 TB | 需抽 |
| soccernetv2 | 600 | 3.44M | 1.44 TB | 需抽 |
| soccernet_ball | 4 | 0.58M | 0.24 TB | 需抽 |
| tennis | 1,368 | 0.48M | 0.20 TB | 需抽 |
| fs_perf | 79 | 0.34M | 0.14 TB | 需抽 |
| finediving | 1,801 | 0.19M | 0.08 TB | 需抽 |
| TACO | 946 | 0.16M | 0.07 TB | 部分抽好 |
| **HOI4D**（已训） | 2,288 | 0.69M | 0.29 TB | ✅ 已有 |
| **train 合计** | **~10,413** | **~10.55M** | **~4.4 TB** | — |

- 训练**只吃 train split**；val/test 特征只给你要评测的集抽（HOI4D/TACO 已有）。
- 规模 = **HOI4D 的 ~15× 帧 / ~4.6× clip**。
- ⚠️ **两个数据集还没抽 jpg**（fs_comp、TouchMoment 原生）→ 要先从视频抽帧。

### 5.2 on-the-fly V-JEPA（**核心改造**）
**为什么**：预抽特征要 **~4.4 TB 磁盘**，撞墙（原机器只剩 1.2TB）。改成**训练时实时算 V-JEPA**，磁盘归零。
**为什么可行**：V-JEPA **冻结** → on-the-fly 只是 `no_grad + bf16` 前向，**不回传、不存激活** → ≈「贵一点的 dataloader」，不是端到端训 ViT。
**代价**：V-JEPA 从「跑 1 遍」变「每 epoch 跑 1 遍」→ 每步慢 ~3×（含 V-JEPA 前向），总算力 ~2× 于预抽，但**省掉 4.4TB 磁盘 + 独立抽取阶段**。
**怎么改**（代码量不大，抽取逻辑现成）：
- 现成函数：`data/vjepa_grid.py::_run_pass_grid`（单遍 even pass → 576-grid）、`data/vjepa_interleave.py::load_encoder / preprocess_frames`。
- 把它从「`scripts/67` 预抽落盘」搬进**训练 forward**：dataset 返回**帧（或帧路径）**而非读 `.npy`；训练 step 里对当前 batch 的帧调 `_run_pass_grid` → 得到 grid → 喂 `FrameCompress`。
- **混合方案（推荐）**：HOI4D+TACO 已抽好（~360GB，能塞）→ 读缓存；sports 巨无霸 → on-the-fly 流式。
- 优化：**降 fps（15→5，V-JEPA 计算和数据管线 ÷3）**、`torch.compile` + flash-attn。
- ⚠️ **`max_frames=320`**：finegym/soccernet 单 clip 上千帧 → 必须**切窗**（on-the-fly 更好切，按窗喂 V-JEPA）。

### 5.3 资源估算 + 硬件
- **磁盘**：on-the-fly → **0**；预抽 → 4.4TB（不可行）。
- **抽特征算力**（若预抽）：~0.01s/帧/GPU → train-only ~27 GPU-h/单卡。
- **训练**：on-the-fly、3 epoch、全量 ≈ **~3.5 天/单卡**（V-JEPA 主导）→ **8 卡数据并行 ~10-12h**。子采样验证版 ~1h。
- **硬件结论**：
  - **冻结 8B + LoRA → 用不到 NVSwitch**：纯数据并行，每步只 all-reduce ~400MB 的 LoRA 梯度（PCIe ~16ms，占比 <1%）。**8×A100-40G 无 NVSwitch 完全够用**，~8× 近线性提速。
  - 40G 显存够：16(8B)+0.4(V-JEPA)+1.2(LoRA+Adam)+~5-10(激活) ≈ 24-28G，batch/卡 1-2 + 梯度累积。
  - **4×NVSwitch 只有在你要 shard 模型时才值**（全量解冻 8B / 换更大 LLM）——不是现在的配方。
  - **真正的瓶颈 = 数据管线**（8 卡 on-the-fly 的 jpg 解码 / IO）：帧放 NVMe、多开 dataloader workers、降 fps。

### 5.4 数据口径统一（混训能不能 work 的前提）
- **统一原语**：`(自然语言 query, [帧号列表])`。区间事件拆成 **start+end** 两个 moment（touch/untouch 本就是「抓取」区间的 start/end）。
- **最大风险 = 标注语义一致性**：混训只在「各数据集的事件指同一物理时刻」时有效。语义不一致会注入标签噪声、反而伤 → 需 **per-dataset / query-mode 条件**（"找全部 X" vs "找匹配描述的一个"）。
- **精度异质**：帧级 vs 秒级容差不同 → 需**容差感知的 loss**（松标注不硬罚精确帧）。
- **fps/时长归一** + **缺席负样本**（`none`，已支持）。

---

## 6. 要建的工程（engineering tasks，按优先级）
1. **路径参数化**（§4）：抽 `paths.py` / 环境变量，去掉 61 处硬编码。
2. **on-the-fly 训练路径**（§5.2）：dataset 返回帧、训练 forward 内调 V-JEPA、支持缓存/流式混合。
3. **长 clip 切窗**（`max_frames=320`）：sports 数据集必需。
4. **多数据集 loader + query-mode 条件**（§5.4）：统一 `(query,[frames])` schema。
5. **降 fps + torch.compile**：喂数据 & 提速。
6.（可选，见 §7）**内容丰富 question** + **question load-bearing ablation**。

---

## 7. 启动顺序（实验优先级）
- **Step 0（最便宜，先做）**：**HOI4D+TACO 联合 SFT**，用现有缓存，测 TACO 零样本是否从 15→45+。有效 = 「混训修读出」证实。（脚本参考 `scripts/78_taco_zeroshot.py` 的 eval + `train/idx_compress_train.py` 的 SFT。）
- **Step 0.5（并行、几分钟）**：**question ablation** —— 换问题看输出变不变，判断模型是否在读 question（决定「内容丰富 question」这条路值不值得走）。
- **Step 1**：改 **on-the-fly**，**子采样多数据集**（每集封顶 ~150K 帧，总 ~1.5M）验证，半天。
- **Step 2**：**全量混训**（8 卡 ~10-12h），held-out 数据集测真·泛化。
- **Step 3（可选）**：内容丰富 query，把定位路由到语言 grounding。

## 8. 验证 gate / 风险
- **Gate A**：Step 0 若 TACO 明显恢复 → 混训方向对；只微涨 → 查标注口径 / TACO 本身难度。
- **Gate B（真·open 判据）**：能否定位**训练里没见过的动作类型/说法**（held-out 集 + held-out 措辞）。否则只是「多域闭集」。
- **风险**：标注语义不一致（§5.4）是最隐蔽的坑；数据管线 IO 是最可能的工程瓶颈。
- **别做**：连接器注语言、RLVR 调 reward（已证死路）。

---

## 9. 仓库地图 & 关键路径
| 路径 | 作用 |
|---|---|
| `models/frame_compress.py` | **FrameCompress**（8-query 压缩、RMS 对齐、语言开关，默认全关=纯视觉） |
| `models/idx_localizer.py` | idx-copy 定位器（prompt 组装 `_build`、`predict_multievent_batch`、`_answer_str`） |
| `models/wrapper.py` | 冻结 Qwen3-VL-8B 包装（`MODEL_ID`、`add_lora`） |
| `data/vjepa_grid.py` / `vjepa_interleave.py` | V-JEPA 抽取（**on-the-fly 复用 `_run_pass_grid` / `load_encoder`**） |
| `data/idx_grid_dataset.py` | 读 grid_even 的多事件数据集（**改 on-the-fly 从这动**） |
| `data/questions.py` | 问题同义句库（train/test 措辞分离） |
| `train/idx_compress_train.py` | Stage-1 训练（`--mode compress|mean`、`--lora`、`--init_fc`） |
| `train/idx_rlvr_train.py` | GRPO/RLVR（已证打不过 SFT，仅存档） |
| `scripts/67_extract_even_grids.py` | 预抽脚本（on-the-fly 的逻辑蓝本） |
| `scripts/68 / 78` | test 评测 / **TACO 零样本迁移评测** |
| `scripts/71 / 77` | 线性探针 / **跨数据集迁移探针** |
| `scripts/72-75` | 误差分布 / precision / PR 曲线 / mAP@δ 曲线 |

**关键 checkpoint（包内附）**：`outputs/idx_compress/sft_lora_v2/best.pt`（SFT-LoRA，最强，fc+lora）、`outputs/idx_compress/comp_notext/best.pt`（M1，纯连接器，迁移更好）。

---

## 10. 包内清单
- ✅ 全部源码（`models/ data/ train/ scripts/ eval/ localize/ configs/`）
- ✅ 2 个关键 checkpoint（SFT-LoRA、M1）
- ✅ 三份文档：`CLUSTER_HANDOFF.md`(本文)、`TECHNICAL_REPORT.md`、`PROJECT_SUMMARY.md`
- ✅ 关键结果图（`plot/*.png`）
- ❌ **不含**：grid_even 特征（357GB+，cluster 走 on-the-fly）、frames、完整 outputs、V-JEPA/Qwen 权重（V-JEPA 权重需单独拷 `VJEPA_CKPT`，Qwen 从 HF 下）
- ⚠️ **eval 依赖**：`hand_touch_detection/common/score.py` + `methods/spot_head/eval_nms.py`（需另带）

---
*打包日期：2026-07（HOI4D touch/untouch 项目；下一步 = 多数据集混训 + on-the-fly）*
