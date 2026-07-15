# 冻结 VLM 上的 HOI4D 手-物 Touch/Untouch 时刻检测 —— 技术报告

**项目**：`VLM_spotting/hand_touch_spotting`
**任务**：给定视频 + 自然语言问题，精确定位"手接触物体 / 手松开物体"发生在**第几帧**。
**一句话结论**：把 V-JEPA 的 576 空间 token 用 **8-query 注意力压缩**成 1 个富信息 token 注入**冻结** Qwen3-VL，让 LLM 用 **idx-copy** 生成帧号 —— 单阶段就到 **val mAP@2 67.6**（mean-pool 基线仅 50）；再走 **LLaVA 式冻结→解冻 SFT-LoRA** 到 **72.5 val / 72.9 test**，成为最强模型。**在连接处注入语言反而 HURTS**；**RLVR 无论换什么 reward 都打不过 SFT**（算法天花板，非 reward 选错）。

---

## 目录
1. 背景与问题定义
2. 评测协议（如何评估的）
3. 动机与总体思路
4. 实验历史（按时间顺序 · 每个实验含"为什么做")
5. 汇总结果表
6. 关键结论
7. 局限与未来方向
8. 附录：架构细节 / 数据构造 / 代码 / 环境 / 产物

---

## 1. 背景与问题定义

### 1.1 任务
HOI4D 是第一人称手-物交互数据集。我们要做的是 **temporal moment spotting**：对每段视频，预测两类**瞬时事件**发生的帧号：
- **touch**：手与物体**发生接触**的那一帧；
- **untouch**：手**松开物体**的那一帧。

一段视频里每类可能有 **0 个、1 个或多个**事件（多次抓取/释放）。**touch/untouch 本质是"一次抓取"这个时间区间的 start/end 边界** —— 这个观察后面在"开放动作时刻检测"里会被复用。

### 1.2 数据集
`hand_touch_detection/data/HOI4D-v3` 的 `train/val/test` 三份 json，每个视频含 `events: [{label: touch|untouch, frame: int}]`。有效视频（有对应 V-JEPA 特征的）约 2850 个，test 424 个视频、1945 个 GT 事件。

### 1.3 为什么用 VLM 做
此前的路线（见 §4.1–4.5）是纯视觉时序网络（MS-TCN + FiLM）。转向 VLM 的动机：
- **开放词表接口**：用自然语言问"什么时候接触/松开"，天然可泛化到任意动作查询，不绑固定标签头；
- **利用预训练知识**：冻结的大 VLM 自带丰富视觉-语言先验；
- **生成式定位**：让 LLM 直接"说出"帧号，避免逐帧分类头的类别耦合。

---

## 2. 评测协议（如何评估的）

**这是理解所有结果的前提。**

### 2.1 主指标：action-spotting mAP@δ
调用链 `eval_nms.maps_quiet → common.score.compute_average_precision`。对每个类别（touch/untouch）：
1. 把该类所有预测按**分数降序**排列；
2. 逐个预测，找**最近的、尚未被认领的** GT；若距离 ≤ 容差 δ，记为 TP，该 GT 标记为已认领（**一对一匹配**，杜绝重复计数）；
3. 每遇到一个 TP 记一次精度 `p = (累计TP)/(累计预测数)`；
4. 对精度序列做**从右到左的插值**（取右侧最大）；
5. `AP = Σ(插值精度) / (该类 GT 总数)`。**分母是 GT 总数**——所以漏检的 GT 贡献 0，AP 天然同时惩罚**漏检**和**误检排序**。

**mAP@δ = touch-AP@δ 与 untouch-AP@δ 的平均。** 报告的"72.9"= **mAP@2**（δ=2 帧，15fps ≈ 133ms 容差）。**只有 AP 是 E2E eval 认的数**；precision / recall 是我们后来做的诊断分解（§4.16），不是 eval 指标。

### 2.2 分数从哪来
idx 生成时每个事件的分数 = 其**数字 token 的平均 softmax 概率**（模型对"这个帧号"的生成置信度）。mAP 用它做全局排序。

### 2.3 Read-out（一个早期关键修复）
早期直接取 argmax / 单个 idx，@0 卡在 ~4。改成**稠密的逐帧打分场 + 高斯时序 soft-NMS**（保留所有峰、按分数衰减邻近同类）后，@0 从 4 提到 10–15。**"稠密打分 + soft-NMS" 远好于 "单点 argmax"**，这条贯穿始终。

### 2.4 推理流程
全视频多事件 idx 生成 → 正则 `\d+` 抠出所有帧号 + 其分数 → `maps_quiet`。val 用于选型，test 用于最终确认。

---

## 3. 动机与总体思路

**核心假设**：定位任务的瓶颈不是 LLM 的能力，而是**喂给 LLM 的视觉 token 质量**。V-JEPA 每帧给 24×24=576 个空间 patch token（768 维），此前被 **mean-pool 成 1 个 768 维 token** 再送进 4096 维的 LLM —— 这是个 rank-768 的稀释 token，信息被抹平。**假设：用 query 注意力把 576 patch 压成 1 个填满 4096 维的富 token，会显著提升枚举与定位。** 同时顺带回答一个设计问题：**在 V-JEPA→VLM 连接处注入语言，有没有帮助？**

---

## 4. 实验历史（按时间顺序）

> 每个实验：**① 为什么做 → ② 方法 → ③ 如何评估 → ④ 结果 → ⑤ 结论/引出下一步**。

### 4.1 起点：LLM→idx 生成（Stage-1，copy 机制）
- **① 为什么**：想让 LLM 直接"说出"事件帧号，而非逐帧分类，以获得开放接口。
- **② 方法**：prompt = 指令 + 逐帧 `[帧号数字token][motion token]` + 每 5s 一张 ViT anchor 图 + 问题 + `" Answer:"`；答案 = 逗号分隔帧号列表。**idx-copy**：输入端每个 motion token 前挂一个真实数字 token 作"帧号把手"，模型输出即"抄"事件帧的把手（`use_idx=False` 是消融，逼它 COUNT）。teacher-forced CE 只在答案 token 上。
- **③ 评估**：全视频多事件生成 → maps_quiet。
- **④ 结果**：能定位，但 **@0 长期卡在 ~4.1**，怀疑 off-by-one / soft-argmax 问题。
- **⑤ 引出**：@0 太低 → 需要更好的 read-out（§4.3）和更好的 token（§4.6）。

### 4.2 二阶段：idx → MS-TCN refine
- **① 为什么**：Stage-1 给出粗略帧号，用一个小网络在 idx 附近做精修，提精度。
- **② 方法**：`train/refine4.py` —— 多阶段 FiLM MS-TCN，在预测 idx 的 ±W 窗口内、用 query 做 FiLM 条件，输出每帧事件概率；`--out_E` 非对称输出窗、`--multi_gt`、`--centers`、稠密+soft-NMS 读出。
- **③ 评估**：`scripts/46_refine_error_dist.py` 画"每事件定位误差"直方图 + 累计召回 CDF（Stage-1 vs +refine）。
- **④ 结果**（旧 mean-pool `feat_interleave` 特征上）：**Stage-1 mAP@2 44.9 → 2-stage 56.7**；误差质量被拉向 0。窗口研究：**输入 ±24、输出锁 8–12 即可**，refine ±8 窗口就够（网络更小、效果不差）。
- **⑤ 引出**：2-stage work 了，但引出问题——瓶颈到底在 Stage-1 的 token 表示，还是在 refine？

### 4.3 Read-out 修复（dense + soft-NMS）
- **① 为什么**：@0 卡 4，怀疑读出方式（单 argmax）丢了精度。
- **② 方法**：改为逐帧稠密打分场 + 高斯时序 soft-NMS（见 §2.3）。
- **④ 结果**：@0 从 4 → 10–15。**读出方式本身是一大杠杆。**

### 4.4 方法搜索小结（overnight，`method-search-verdict`）
- **④ 结论**：在旧管线下，**最好的是 MS-TCN 直接在 RAW V-JEPA 上 + FiLM(query)**；**"语言空间特征"被证伪**；**idx 两阶段相比好的 refine 并没有额外增益**。→ 说明瓶颈在**表示**，不在后处理。

### 4.5 转折：为什么要做 query 压缩
- **① 动机**（用户提出）：mean-pool 把 576 patch 抹成 1 个 rank-768 稀释 token，是信息瓶颈。改用 **query-conditioned attention pooling**：K 个 learnable query 对 576 patch 做注意力池化，压成填满 4096 维的富 token。同时验证一个设计问题：**在连接处注入语言到底有没有用**（InstructBLIP 式）。
- **设计确认**（多轮讨论敲定）：learnable query 放在**注意力之后**（不是之前）；压缩后**仍过 Adaptor** 到 4096 维；必须 **RMS 对齐** LLM 词嵌入范数（否则冻结 LLM 崩）；**K=8**；时间维用 **插值**从 N/2 扩到 N（不做 even/odd 双遍）。

### 4.6 特征抽取（`grid_even`）
- **① 为什么**：压缩需要**未池化的 576-grid**，得先把 V-JEPA 特征存下来并行处理。
- **② 方法**：`scripts/67_extract_even_grids.py` —— V-JEPA 2.1 ViT-B（384px, patch16）单遍抽取，保留 24×24=576 patch × 768d 的 grid（`grid_even`），Conv3d tubelet_size=2 → T/2 时间步。线程/内存严格受限（`torch.set_num_threads(8)` + io-workers，因历史上 OOM 把服务器搞崩过）。分片 2 卡。
- **④ 产物**：`VLM_spotting/vjepa/grid_even`，fp16，~134MB/视频，2850 视频 ~377GB。

### 4.7 核心实验：mean-pool vs 压缩 vs 语言注入 ★
- **① 为什么**：直接回答两个问题——压缩有没有用？语言注入有没有用？（全部**冻结 Qwen3-VL-8B**，只训连接器 = 0.34% 参数，val mAP@2。）
- **② 方法**：`models/frame_compress.py`（FrameCompress：3 步 cross-attn，`--stain/--use_text/--gate_lang`，gates init-0）+ `train/idx_compress_train.py --mode compress|mean`。
- **④ 结果**：

  | 连接器 | val mAP@2 |
  |---|---|
  | mean-pool baseline（feat_interleave, 768/帧） | **50.1** |
  | **8-query 压缩，无语言（M1 = comp_notext）** | **67.6 (+17.5)** ✅ |
  | 压缩 + 文本条件 query（问题进 query） | 11.5 ❌ |
  | 压缩 + staining（patch 交叉注意问题）+ 文本 query | 2.2 ❌ |
  | 压缩 + gated（init-0）语言 | 0.67 ❌ |

- **⑤ 结论**：
  - **压缩帮助巨大**（+17.5）。机制：8 query 把 576 patch 压成填满 4096 维的富 token，LLM 枚举+定位大幅变好（**ndet≈ngt**，修复了此前"欠枚举"的召回天花板）。冻结 Qwen 本就有能力，瓶颈是 token 质量。
  - **语言注入连接处 HURTS**（越注入越差）。机制：LLM 已在 prompt 里有问题，**预先把语言混进视觉 token 会污染它们** —— 这是 **LLaVA（干净视觉 token，LLM 自己 grounding）> InstructBLIP（预注入语言到 Q-Former）** 的教训。→ **连接器保持纯视觉，压硬（1 富 token/帧），语言交给 LLM。**
  - 注意：压缩单阶段 67.6 已**超过**旧 mean-pool 2-stage 的 56.7 —— **更好的 token 让 refine 变得不必要**。

### 4.8 SFT：LLaVA 式冻结→解冻 ★
- **① 为什么**：M1 是"冻结 LLM 只训连接器"的 Phase-1。按 LLaVA 配方，Phase-2 解冻 LLM（加 LoRA）应进一步提升，尤其 @0（精确落点）。用户明确要求对比 **"冻结+解冻 from-scratch"** 而非联合训。
- **② 方法**：`idx_compress_train.py --lora 1 --init_fc <M1连接器>`；`W.add_lora(rank=16, alpha=32, target=all)`；训"连接器 + LoRA"。
- **③ 评估**：val 选型，随后在 test 上确认（`scripts/68`）。
- **④ 结果**：

  | 模型 | val @0/@1/@2 | test @0/@1/@2 |
  |---|---|---|
  | M1 frozen-compress | – / – / 67.6 | – / – / ~64.3 |
  | **SFT-LoRA（sft_lora_v2，最强）** | 9.42 / 46.97 / **72.52** | 7.77 / 45.53 / **72.94** |

- **⑤ 结论**：解冻有效（+5，尤其 @0：6.8→9–11）；**SFT-LoRA 是全项目最强模型**，且 **val→test 零掉点**（比 M1 的 -3.4 泛化更好）。消融确认 LLM 是原始 Qwen3-VL-8B（未载别的权重），只训连接器就到 67.6。（联合 from-scratch `comp_scratch_lora` 按用户要求砍掉。）

### 4.9 RLVR 初探（tiered reward，3 配置）
- **① 为什么**：SFT 是 token 级 CE，"metric-blind"。RLVR 用可验证 reward（到 GT 的帧距离）直接对齐指标，看能否再涨。
- **② 方法**：`train/idx_rlvr_train.py`（GRPO：每 prompt K 个 rollout，组内归一优势，策略梯度；reward = 到 GT 距离分层）。
- **④ 结果**：#1 从 M1、lr 1e-5：reward↑ 但 mAP↓（68→65）；#2 lr 3e-6：平（67.6）；#3 从 SFT-LoRA：平（~72）。
- **⑤ 引出**：**reward 涨、mAP 不涨 = reward-metric 错位**。用户判断"reward 有问题"，先停，转做可解释性；随后专门系统搜索 reward（§4.14）。

### 4.10 测试集验证
- **① 为什么**：val 结论要在 test 上确认。
- **④ 结果**：M1 test ~64.3；SFT-LoRA test **72.94**（@0 7.77 / @1 45.53）。val 结论成立，SFT-LoRA 泛化最好。

### 4.11 可解释性 I：词表投影
- **① 为什么**：想知道压缩 token 的语义——它和哪些词最像？
- **② 方法**：`scripts/69` —— RMS 对齐的 token 与 LLM 词嵌入做余弦，取最近词，对比接触/非接触帧。
- **④ 结果**：**乱码**（罕见 unicode/代码 token，无 hand/touch 词）。
- **⑤ 结论**：token 是为 idx 优化的，**不对齐语言**。→ 换 attention 视角（§4.12）。

### 4.12 可解释性 II：attention 热图
- **① 为什么**：token 不像词，那看它"在看哪"。
- **② 方法**：`scripts/70` —— 8 query 对 576 patch 的注意力 → 24×24 热图叠帧，接触 vs 非接触。
- **④ 结果**：**弥散**，没锁定手部；接触帧仅略更集中（max/mean：M1 105.8 vs 85.5；SFT 92.2 vs 83.0，SFT 更平——LoRA 让 LLM 承担更多 grounding，连接器不必尖峰）。
- **⑤ 结论**：空间上不明确，需要更定量的证据（§4.13）。

### 4.13 可解释性 III：线性探针 ★
- **① 为什么**：决定性地问——**接触信息能否从单个压缩 token 线性解码**？
- **② 方法**：`scripts/71` —— 对每帧压缩 token 训 logistic 回归（接触 ±2 vs 非接触 >15），held-out val、平衡、无 sklearn 用 torch 实现，AUC 用 Mann-Whitney。
- **④ 结果**：**AUC 0.889（M1）/ 0.892（SFT），acc 0.82**（chance=0.5）。
- **⑤ 结论**：接触**能从单 token 线性解码** —— 语义**真实但分布式**（非单区域、非词形）。**SFT 几乎不改探针（0.889→0.892）→ 连接器早已编码好接触；SFT 的 +5 mAP 来自 LLM(LoRA) 更会"读"这个 token，而非 token 变好。** 这把整条线串成了闭环。

### 4.14 RLVR reward 系统搜索（/goal）★
- **① 为什么**：目标明确——**找一个 reward 让 RLVR 超越 SFT（72.52）**。既然 tiered 失败，系统性地设计并对比多种 reward。
- **② 方法**：全部从 SFT init（72.52）出发，快迭代配置（`--train_limit 800`、`--evals_per_epoch 4`、加了逐步 heartbeat），双卡并行对比。新增 reward：
  - `sharp`：精确度分层（+1@0/+0.35@1/+0.1@2）+ **重罚多检(-0.5)** —— 打 @0/@1 分数校准。
  - `f1`：集合 F1@tol1 + 精确度 bonus —— 打枚举数量。
  - `graded`：2 帧内给分（1.5/1.2/1.0）+ 轻罚多检(-0.1) —— @2 对齐、保召回。
  - `map`：**2 帧内全平**（无精确度梯度）—— 纯 @2 召回、不拿 @2 换精确度。
- **③ 评估**：每次 eval 全 val 138 视频跑 maps_quiet；best.pt 仅在 @2 > 历史最好时保存。
- **④ 结果**：

  | reward | @1（起46.97） | @2（起72.52，轨迹） |
  |---|---|---|
  | sharp | ↑48.4 | 71.0 / 71.5 / 71.1 |
  | f1 | ↑47.1 | 71.5 |
  | graded | ↑48.0 | 70.5 → 71.6 → 71.4 → END 71.1 |
  | **map（全平内）** | ↑47.4 | **70.8** |

- **⑤ 结论（关键）**：**5 种 reward 全部 @2 掉 ~1.5、@1 涨 ~1.5，没有任何 eval 越过 72.52（best.pt 从未保存）。** 连 `map`（理论上绝不拿 @2 换精确度）都一样掉 → **排除了"reward 形状"这个变量**。→ 问题是**算法性的**（§4.15）。

### 4.15 KL-anchored RLVR（决定性实验）★
- **① 为什么**：既然 reward 形状不是杠杆，机制应是——**SFT 已在贪心解码的 @2 最优点上，策略梯度优化的是采样 rollout，每次更新都把贪心轨迹推离最优点**。修法：**KL 锚定到 SFT**，让 @2 不漂移、reward 只推 @1。
- **② 方法**：在 `idx_rlvr_train.py` 实现 KL —— 参考 = 冻结的 SFT 策略（LoRA+fc 快照），每步换权重前向算参考 logprob，k3 estimator 加到 loss（`--kl`）。对比 kl=1/6/20。
- **④ 结果**：kl=1 → 70.98（≈无 KL）；kl=6 → 71.49 **然后回落** 70.99；kl=20 → 71.23 且 **@1 增益被冻结**。跑满 2 epoch 仍震荡在 70.5–71.5，**从未到 72.52**。
- **⑤ 结论**：**KL 也压不住 @2，只能在"漂移↔冻结"之间权衡，最好也就是打平。** 综合 §4.14–4.15：**本任务（idx 生成 + 贪心解码 + GT 监督的 SFT 已最优）RLVR 打不过 SFT@2，天花板就是 SFT。** RLVR 唯一稳定效果是小幅 @1/精确度增益。要真超越需**超出 GT 的新监督**或**非贪心评测**——都不是"选 reward"能解决的。

### 4.16 指标深入对齐（precision / recall / AP / mAP@δ）
- **① 为什么**：需要向下拆解 AP、并确认与 E2E eval 一致；也解释"@0 为什么这么低"。
- **② 方法**：`scripts/73`（一对一匹配的 precision 分布 + precision/recall/AP vs 容差）、`scripts/74`（PR 曲线，面积=AP）、`scripts/75`（mAP@δ 曲线）。全部复用缓存预测、CPU、复现 `maps_quiet`。
- **④ 结果**（test，ndet=1980 ≈ ngt=1945，dup-FP=79）：

  | 容差 | precision（匹配后） | recall（覆盖） | **AP = E2E** |
  |---|---|---|---|
  | @0 | 24% | 25% | **7.8** |
  | @1 | 60% | 63% | **45.5** |
  | @2 | 79% | 83% | **72.9** |

  mAP@δ 曲线：

  | δ帧 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
  |---|---|---|---|---|---|---|---|---|---|
  | **mAP** | 7.8 | 45.5 | **72.9** | 82.8 | 88.2 | 90.1 | 91.4 | 92.2 | 92.9 |
  | touch | 5.5 | 37.0 | 67.0 | 79.7 | 87.3 | 90.2 | 92.0 | 93.1 | 93.9 |
  | untouch | 10.1 | 54.1 | 78.9 | 86.0 | 89.2 | 90.1 | 90.8 | 91.4 | 91.8 |

- **⑤ 结论**：
  - **AP ≠ precision×recall**（0.79×0.83=0.66≠0.73）；AP = 排序后 PR 曲线下面积，比 prec/rec 都低（要两者同时高 + 分数排序）。**只有 AP 是 E2E eval。**
  - **@0 崩塌（AP 7.8 << precision 24）= 分数校准问题**：模型对"大概哪帧"有信心、对"精确到帧"不区分，分数排不出精确命中 → PR 曲线极差。**这正是 RLVR 想补补不动的地方**（呼应 §4.15）。
  - 曲线前段极陡（0→2 涨 65 点），δ≥5 饱和 ~90，δ=10 也只 92.9（**剩 ~7% 硬漏检天花板**）。**untouch 紧容差更强、touch 松容差略高**：抓取"定位飘"（接触渐变）、松手"要么准要么彻底漏"（分离清晰）。

---

## 5. 汇总结果表

**连接器/训练（val mAP@2）**：mean-pool 50.1 → **压缩 M1 67.6** → **SFT-LoRA 72.5**；语言注入 11.5 / 2.2 / 0.67（全伤）。
**最强模型 test**：SFT-LoRA @0/@1/@2 = **7.77 / 45.53 / 72.94**。
**RLVR**：sharp/f1/graded/map + KL(1/6/20) 全部 @2 ≤ 71.5，**无一越过 72.52**。
**可解释性**：线性探针 AUC 0.89（分布式语义）；词表=乱码；attention=弥散。

---

## 6. 关键结论

1. **压缩 > mean-pool（67.6 vs 50，+17.5）**：瓶颈是 token 质量，8-query 把 576 patch 压成填满 4096 维的富 token，修复欠枚举（ndet≈ngt）。
2. **语言注入连接处 HURTS**：连接器保持纯视觉，语言交给 LLM 在 prompt 里 grounding（LLaVA > InstructBLIP）。
3. **SFT-LoRA（冻结→解冻两阶段）= 72.5 val / 72.9 test，最强模型**，且 val→test 零掉点。
4. **RLVR 打不过 SFT@2**：5 reward + 3 KL 全败，是**算法天花板**（RL 扰动 SFT 贪心最优点），不是 reward 选错。
5. **token 语义分布式**：接触可线性解码（AUC 0.89），但非词形非单区域；**SFT 的增益来自 LLM 更会读 token，不是 token 变好**。
6. **@0 瓶颈 = 分数校准**（分数排不出精确帧）—— 是 RLVR 也补不动的硬骨头。

---

## 7. 局限与未来方向

**局限**：@0 极低（分数不校准精确帧）；~7% 硬漏检天花板（δ=10 也补不回）；RLVR 在确定性 idx 任务上无法超越 SFT。

**未来：开放动作时刻检测（open action moment detection）**
- **架构现成**：冻结 VLM + 视觉压缩 + 语言 query + idx 生成 = 天然开放词表、domain 无关；**touch/untouch 就是"抓取"区间的 start/end**，泛化到任意动作起止是同一件事放大。
- **统一原语**：`(自然语言 query, [帧号列表])`；区间事件拆成 start+end；用 query-mode token 区分"找全部"vs"找匹配描述的一个"。
- **可混数据集**：SoccerNet-v2 / Ego4D-PNR·state-change / Kinetics-GEBD（最 open）/ HOI4D（精确点事件，先做）；Charades-STA / QVHighlights / Ego4D-NLQ（语言查询区间，后并入）。
- **真正的坑在数据口径**（非模型）：精度不一致（帧级 vs 秒级）、枚举语义冲突（穷举 vs 找一个）、moment 定义统一、fps/时长归一、缺席负样本。
- **必做验证**：能否定位**训练里没见过的动作类型/说法**（held-out 措辞探针，见 §附.数据）—— "真开放"vs"多域闭集"的判据。

---

## 8. 附录

### 8.1 架构细节
- **FrameCompress**（`models/frame_compress.py`）：先把 grid 从 N/2 线性插值到 N；3 步 cross-attn（①text←patch ②patch←text staining ③**8 个 learnable query←patch**，query 放**最后**）；concat(8×768=6144) → adaptor → 4096d；`set_target_rms_from(embed_tokens.weight)` **RMS 对齐**；gates（g_stain, g_qcond）init-0。`--stain/--use_text/--gate_lang` 控制语言注入（默认全关 = 纯视觉）。
- **IdxLocalizer**（`models/idx_localizer.py`）：`_build` 组装 prompt（指令 + 逐帧 `[idx][motion]` + 每 5s ViT anchor + 问题 + `" Answer:"` + 答案）；`compress` 非空走压缩路径，否则走 mean-pool adaptor；teacher-forced CE 只在答案；`predict_multievent_batch` 生成 + `_parse` 抠帧号与分数。

### 8.2 数据的 Q 和 A 怎么造
- **样本单位** = `(视频, 事件类型)`；一个视频有 touch+untouch → 2 样本。
- **Q**：`data/questions.py` 同义句库，`train`（8 种）/ `test`（4 种换动词的 held-out）。训练随机抽 train 措辞；主结果 eval 用固定 `GENERIC_Q`（措辞不作混淆变量）；held-out 措辞是"真 grounding 语言"的探针。
- **A**：该类所有事件帧号、排序、逗号分隔文本（`" 45, 120, 200" + <eos>`），无事件则 `"none"`。
- **copy 机制**：输入端"帧号把手" + 输出端"抄帧号"，domain/taxonomy 无关。

### 8.3 代码结构（`hand_touch_spotting/`）
| 文件 | 作用 |
|---|---|
| `models/frame_compress.py` | FrameCompress（8-query 压缩、RMS 对齐、语言开关） |
| `models/idx_localizer.py` | idx-copy 定位器（prompt 组装、生成、解析） |
| `models/vjepa_adaptor.py` | mean-pool 路径 adaptor（baseline） |
| `data/idx_grid_dataset.py` | 加载 `grid_even` 多事件数据集 |
| `data/questions.py` | 问题同义句库（train/test 措辞分离） |
| `train/idx_compress_train.py` | Stage-1 训练（`--mode compress|mean`、`--lora`、`--init_fc`） |
| `train/idx_rlvr_train.py` | GRPO/RLVR（reward {tiered,smooth,map,sharp,f1,graded} + `--kl` 锚定） |
| `train/refine4.py` | 旧 2-stage 的 FiLM MS-TCN refine |
| `scripts/67` | 单遍抽 V-JEPA 576-grid |
| `scripts/68` | test 集评测某 checkpoint |
| `scripts/69/70/71` | 可解释性：词表 / attention / **线性探针** |
| `scripts/72/73/74/75` | 误差分布 / **precision 分布** / **PR 曲线** / **mAP@δ 曲线** |

### 8.4 环境与安全线
- 特征：`VLM_spotting/vjepa/grid_even`（单遍未池化 576-grid，fp16，2850 视频 ~377GB）。
- Python：`/home/chang_noroot/data2/myconda/envs_dirs/vlm_spot/bin/python`；`CUDA_VISIBLE_DEVICES=0/1 OMP_NUM_THREADS=8`。
- 最强 checkpoint：`outputs/idx_compress/sft_lora_v2/best.pt`（fc+lora）；M1：`outputs/idx_compress/comp_notext/best.pt`。
- **OOM 安全线**（历史上 OOM 把服务器搞崩过）：batch=2（SFT）/ B=8 chunked-logprob（RLVR），98GB GPU，线程 ≤8/进程，不建巨型 RAM 缓存。

### 8.5 产物图表（`hand_touch_spotting/plot/`）
| 图 | 内容 |
|---|---|
| `compress_error_dist_test.png` | 每事件误差分布：M1 → SFT-LoRA |
| `precision_dist_test.png` | 精度分布 + precision/recall/**AP** vs 容差 |
| `pr_curve_test.png` | PR 曲线（面积=AP=E2E eval），touch/untouch 分开 |
| `ap_vs_tol_test.png` | **mAP@δ 曲线**（E2E，δ=0–10） |
| `attn_interp.png` / `attn_interp_sft.png` | 8-query attention 热图 |

---

*相关记忆*：`stage1-language-injection-verdict.md`、`method-search-verdict.md`、`vlm-spotting-project.md`。
*报告生成日期*：2026-07-04。
