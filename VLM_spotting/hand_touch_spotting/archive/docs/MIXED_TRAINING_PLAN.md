# 多数据集混训作战计划 v2（DPC 版）

> 目标：把 HOI4D 上验证过的「冻结 VLM + FrameCompress + idx-copy」扩到 Action_Spotting 全家桶，
> 训出跨数据集可用的开放动作时刻检测器。核心假设（迁移研究）：连接器通用、**读出层不迁移 → 混训修读出**。
> v2 变更（2026-07-06）：训练全部在 DPC；数据已在 PVC（tar shards + SQLite index）；V-JEPA 内化为模型内冻结子模块 on-the-fly；
> TouchMoment 以顶层 `{train,val,test}.json` 为准；4 GPU/Job 上限。
> 执行前提：**想清楚再动手**——本文件是「想清楚」的载体，不是执行指令。

---

## 0. 环境事实（来自 DPC_ACTION_SPOTTING_HANDOFF.md，2026-07-05/06 实测）

- PVC `action-spotting-pvc` 800Gi，已占 ~632G；**个人配额 1000Gi** → 余量 ~170G，**预抽特征彻底不可行**
  （grid_even 全量 ~4-5TB）→ **V-JEPA 2.1 ViT-B 冻结、作为模型子模块，训练 forward 里 no_grad+bf16 实时算**。
- 帧在 **tar shards + `index.sqlite`**（1860 万帧/17,120 clips），只能走 `as_dataset.py`（TarStore/ClipDataset，~5ms/帧随机读；帧按 clip 打包 → 窗口读近似顺序读）。**不能 glob jpg**。
- **每 Job 最多 4×A100-SXM4-40G**（NVLink，无 NVSwitch；>4 会 Suspended）。要更多 = 多 Job 跨节点 DDP（InfiniBand），先不搞。
- 两个"有标注无帧"：**fs_comp（PVC 上 0 帧）**、TouchMoment/native 行（= 顶层合并标注，帧其实就是 TouchMoment 的 H*/C* 包——上集群后第一件事验证 clip-id 解析，见 §6 sanity）。

---

## 1. 数据资产与批次（v2 定稿）

| 数据集 | train clips / 帧 | fps | 精度 | 判定 |
|---|---|---|---|---|
| TouchMoment（HOI4D H* + TACO C*） | 3,234 / 842K | 15 / 30 | 帧级 | ✅ 第一批（Step-0 主角） |
| tennis | 1,368 / 483K | 25 / 29.97 | 帧级 | ✅ 第一批 |
| finegym | 3,327 / 4.69M | 25–30 混 | 帧级 | ✅ 第一批（温度采样压体量） |
| soccernet_ball | 4 半场 / 577K | 25 | 帧级(40ms) | ✅ 第一批（窗口化后 ~1800 窗） |
| fs_comp / fs_perf | 见 §1.2 | 25 | 帧级 | ✅ 第一批（171 clip 起步） |
| finediving | 1,801 / 188K | -1(未知) | 帧级 | 🎯 **整集 held-out 零样本**，不训 |
| soccernetv2 | 600 半场 / 3.44M | **2.083** | **秒级** | ⏸ 二期（§1.3） |

### 1.1 TouchMoment 标注：以顶层 json 为准（2026-07-06 生成）
`/home/chang/Dataset/Action_Spotting/TouchMoment/{train,val,test}.json`（= annotation 修复版）：
- train 3,234（H 2,288 + C 946），touch 6,837 / untouch 6,488；**val 138（纯 HOI4D）**；test 649（H 424 + C 225）。
- val∩test = 0，无泄漏。**TACO 不设 val**（用户定）：选型看 HOI4D val，TACO 只在 test 上报。
- ⚠️ 集群 PVC 上的 TouchMoment 标注若还是旧版，把这三个文件同步上去（loose annotation 可编辑）。

### 1.2 fs_comp：分两步走
PVC 上 fs_comp 无帧，但 fs_perf 的 171 个 clip 帧在（且 fs_comp/fs_perf 的 371 条标注逐字段相同、clip 名一致）：
- **起步**：用 fs_comp 切分，clip id 映射到 `fs_perf/<video>` 读帧 → 可用 train 85 / val 28 / test 58。
- **补全（并行推进，非阻断）**：本地从 `fs_perf/video/`（11 个 1080p mp4 都在）按 clip 名帧区间补抽缺的 200 个 clip
  （模板 `extract_women2018_frames.py`，max_height=224），打包上传（~20G，配额余量内），`build_index_parallel.py` 重建 index（~8min）→ 全 371 clip。
- fs_perf 切分弃用（同数据、更易、会泄漏）。

### 1.3 soccernetv2 延后二期的三个理由（数据实测）
1. **17.8% 事件"not shown"**：SoccerNet-v2 标注自带 visibility 标志；not shown = 事件确实发生（有比赛记录），但**标注时刻的转播画面里根本看不到这个动作**（镜头在回放/观众/特写）。分类看更严重：**Kick-off 73%、Clearance 55%、Indirect free-kick 45%、Throw-in 29% 是不可见的**。对「从视觉特征定位时刻」的模型，这些标签教它在没有视觉证据的帧上开火——纯标签噪声。
2. **帧是 2.083fps 抽的**：V-JEPA tubelet(t,t+1) 在这里覆盖 ~1 秒真实运动（帧级域是 67-133ms），运动特征时间尺度差 10 倍，与"原生 fps fast feature"的设计前提冲突。
3. **标注本身秒级**（±1s ≈ ±2 帧@2.083fps），另有 30 条越界事件——需要容差感知 loss + not-shown 过滤 + clamp，是一个独立子项目，混进第一批会污染帧级监督信号。
二期方案：filter visible-only + 容差感知 loss（软标签窗口）+ 把它当"慢时钟域"单独条件化。

### 1.4 数据清洗清单（不变）
finegym 剔 `rrrgsW--AE8_E_031139_031215`（帧数不符）；finediving 剔 3 个事件非单调 clip（held-out 用，`03__67`/`03__71`/`14__3`）；
finediving fps=-1 → 布局按 25fps 假设（对 anchor 密度影响极小），发布前确认；tennis 结论只认 test（train/val 同源视频）。

---

## 2. 统一数据层设计（v2 要点）

统一原语不变：`(dataset, video, fps, window[s,e), query(class), answer=[窗内局部帧号] | "none")`

### 2.1 FrameSource 抽象（新增，关键重构）
帧访问全部走一个接口，两个实现：`DirFrameSource`（本地 jpg 目录，开发调试用）/ `TarFrameSource`（封装 TarStore，集群用）。
**受影响处**：窗口 dataset 读帧喂 V-JEPA；`IdxLocalizer._anchors_for_secs` 现在用 `glob(TOUCH_FRAMES_DIR/vid/*.jpg)` 读 anchor 图 —— 集群上必须改走 FrameSource。

### 2.2 窗口化（不变）+ 采样
窗 ≤320 帧、秒对齐；正:负窗 ≈ 4:1；稀有类全采+过采样；数据集温度采样 `p_i ∝ n_i^0.5`；每 epoch 定额窗口数。
Eval 滑窗 stride=W/2 → 全局坐标合并 → soft-NMS → per-dataset per-class mAP@{1,2,4}（`eval_nms.maps_quiet` 复用，与 E2E-Spot 文献可比）。

### 2.3 fps 泛化（用户已确认要改）
- `IdxLocalizer` 三处：① anchor 取帧 `self.fps` 写死 15 → **用 per-sample fps**（25/30fps 视频现在会拿错 anchor 帧）；
  ② 秒分组支持非整数 fps（29.97）→ `floor(f/fps)`；③ anchor 放置改按 motion-token 计数（≈5s×fps 个 token 一张），摆脱整数秒假设。
- V-JEPA 恒为原生帧率逐帧 1 token（方法卖点不动）；stride-2 降采样留作算力开关（默认关）。

### 2.4 on-the-fly V-JEPA 子模块（v2 核心工程）
- `_run_pass_grid`（even 单遍）从 scripts/67 搬进模型 forward：`frames → preprocess → V-JEPA(no_grad, bf16) → (N/2,576,768) → FrameCompress._interp → 压缩 token`。
- V-JEPA ViT-B ~0.4G 显存 + 8B LLM bf16 16G + LoRA/优化器 ~1.5G + 激活（grad ckpt）→ 40G A100 batch 1-2/卡可行。
- **eval 特征小缓存**：HOI4D val 138 clip 的 grid ≈ 17GB fp16 → 放 Job 本地 emptyDir/scratch，每 epoch eval 不重算（PVC 不占）。
- IO：TarStore 5ms/帧随机、窗口内近顺序；dataloader workers ≥8/卡 预取解码，V-JEPA 前向和 LLM 步交叠（prefetch 队列）。

### 2.5 QA 构造 → 详见 `QA_DESIGN.md`
四模式混合（类型枚举 / metadata-grounded 指代 / 负问题 / 序数），核心原则「多样性防坍缩、承重性逼 grounding」；
6 个第一批数据集里 5 个的指代问题可纯靠现成 metadata 造（HOI4D 物体类 from video id、fs 跳跃细类、finegym 动作描述、tennis 场地侧、soccernet_ball 队伍侧），仅 TACO 需本地 VLM captioning 补槽位。

---

## 3. 工程任务清单（v2 排序）

**P0 本地开发（不训练，只写代码+小验证）**
1. 补依赖：V-JEPA ViT-B ckpt（好办）；`hand_touch_detection/common/score.py` + `eval_nms.py`（eval 硬依赖，包里没有）。
2. 路径参数化（61 处硬编码 → paths.py/env）+ **FrameSource 抽象**（§2.1）。
3. on-the-fly V-JEPA 子模块（§2.4）+ `IdxLocalizer` fps 三处修复（§2.3）。
4. `WindowedSpottingDataset`（统一 json + 窗口采样 + 局部坐标 + query bank）+ 滑窗 eval harness。
5. fs_comp 200 clip 本地补抽（CPU 活，随时可做）。

**P1 集群 sanity（1×GPU 小 Job，分钟-小时级）**
6. holder pod 验证：顶层 TouchMoment json 同步；合并标注的 clip-id 全部能在 index 解析（§0 的 "native ❌" 疑点）；fs_comp→fs_perf clip 映射解析。
7. 镜像构建（cu12.x + torch + transformers + vjepa2 vendored + as_dataset.py + Pillow）推 docker.aiml.team。
8. **正确性 gate**：sft_lora_v2 + on-the-fly 特征在 HOI4D val 复现 mAP@2 = 72.5±0.5（一并验证 TarStore 读帧、V-JEPA 权重、eval 三件套）。

**P2 实验序列（都在 DPC，≤4×A100/Job）**
- **Step 0**：TouchMoment 顶层 train.json 联合 SFT（HOI4D+TACO）。init A/B：M1 fc + 新 LoRA vs sft_lora_v2 继续。
  **Gate A：TACO test mAP@2 15 → 40+，HOI4D test 掉点 ≤3**。规模 842K 帧/epoch → 1-2 卡数小时。
- **Step 0.5**：question ablation（换/乱问题看输出变不变），分钟级，决定 query 投入度。
- **Step 1**：六数据集子采样混训（每集 ≤5k 窗）。Gate：各集 val 非平凡 + HOI4D 不崩。
- **Step 2**：全量混训 3 epoch（窗口定额 ~25-30k/epoch ≈ 7-9M 帧前向）→ 4×A100 约 1-1.5 天。
  报告：各集 in-domain test mAP@{1,2,4} + **finediving 整集零样本** + **held-out 措辞** + E2E-Spot 文献对照。
- **Step 3（二期）**：soccernetv2（visible-only + 容差感知）、referring query、fs_comp 全 371、高分辨率重抽评估。

---

## 4. 训练配方（不变项从简）

fc + LoRA(r16, α32, all)，V-JEPA/Qwen 冻结；CE 只在答案 token；lr_fc 1e-4 / lr_lora 1e-4、AdamW、cosine、bf16+grad ckpt；
温度采样 + 窗口定额；死路不碰（连接器注语言、RLVR reward）。
DDP：单 Job 4×A100 纯数据并行（LoRA 梯度 all-reduce 很小，NVLink 绰绰有余）。

---

## 5. 风险表（v2 增补）

| 风险 | 对策 |
|---|---|
| TarStore IO 成瓶颈（8 worker × 4 卡随机读） | 帧按 clip 打包=窗口近顺序读；预取队列；实测后可调窗口采样局部性 |
| "native ❌" 若真不解析 | P1-6 第一时间验证；真缺就按 H*/C* 子标注读（帧同一批） |
| 4 GPU 上限拖慢全量 | 窗口定额控制 epoch 成本；必要时多 Job 跨节点 DDP（InfiniBand）或 admins 特批 |
| 224px 上采样 384 喂 V-JEPA | 先接受（E2E-Spot 在 224 成立）；弱则二期源视频重抽 |
| 标注语义漂移 / 类长尾 / 分数校准@0 | 同 v1：描述性 query；稀有类过采样+单列报告；@0 不在本期目标 |
| PVC 180 天回收 | checkpoint/日志及时回传本地 |

---

## 6. 一句话路线（v2）

**本地只写代码：FrameSource 抽象 + on-the-fly V-JEPA 子模块 + fps 三处修复 + 窗口数据层；集群上先过两道 sanity（clip-id 解析、72.5 复现），再按 Step 0（Gate A：TACO 15→40+）→ 子采样 → 全量（4×A100）推进；finediving 整集零样本 + held-out 措辞出「真开放」主结果；soccernetv2 因 not-shown 17.8%（Kick-off 73% 不可见）+ 2fps 慢时钟 + 秒级精度留二期专项。**
