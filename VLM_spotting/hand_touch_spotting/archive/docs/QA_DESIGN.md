# QA 构造设计 —— 防 LLM 退化 + 语言真 grounding

> 动机：question 太雷同 → LoRA 学会忽略 question（Q 退化成常量前缀/数据集 ID），闭集坍缩，开放词表迁移失效。
> 目标：让 question 既**多样**（防模式坍缩）又**承重**（load-bearing，答案必须依赖问题内容），并尽可能**结合画面内容**提问。
> 配套：MIXED_TRAINING_PLAN.md §2.5 的占位由本文件展开。

---

## 1. 退化机制诊断（为什么雷同 question 有害）

teacher-forced CE 只在答案上。若同一 (视频, 类别) 永远配同一句问题，梯度会发现：**忽略 question token、只看视觉 token + 位置先验，loss 一样降**——question 退化成"第 3 个数据集第 2 类"的 ID。表现为：换措辞输出不变（Step 0.5 ablation 可测）、held-out 措辞掉点大、新类零样本失效。

**关键洞察：多样性只防坍缩，承重性才逼 grounding。** 两者要分开设计：
- 多样性 = 同一语义多种说法（paraphrase bank）；
- 承重性 = **同一视觉输入、不同问题 → 不同答案**（模型不读题就答不对）。

HOI4D 原设置里承重性其实已经存在但很弱（同视频 touch/untouch 两问、答案不同）。混训的运动数据集是天然放大器：**一个 tennis 窗口有 6 个类同时出现**，"问 bounce 还是问 serve"决定答案集合——问题不读就必错。这是混训对 QA 质量的隐藏红利。

---

## 2. 四种 QA 模式（训练时按比例混合采样）

| 模式 | 占比(初值) | 例子 | 承重机制 |
|---|---|---|---|
| **M1 类型枚举**（现状扩强） | ~55% | "List every frame where the ball bounces." | 多类窗口内选类 |
| **M2 指代限定**（referring，画面结合的主形态） | ~20% | "When does the skater take off for the **axel**?" / "When does the hand grasp the **mug**?" | 同类事件的**子集选择**，metadata 可验证 |
| **M3 负问题** | ~15% | 在只有 pass/drive 的窗口问 "When is a goal scored?" → `none`；在 mug 视频问 "...touch the scissors?" → `none` | **最强的防忽略信号**：视觉相同、问题不同 → none vs 非 none |
| **M4 序数/限定量** | ~10% | "the **second** touch" / "the **last** bounce"（HOI4D 已有 first/second 传统） | 答案数量与位置依赖问题 |

约束：
- **"none" 总占率控制在自然事件缺失率附近（~15-20%）**，防模型偏向 none。
- M2 的答案语义 = "该类中匹配描述的子集"，与 M1 的"该类全部"不同 → 措辞模板必须让二者可区分（M1 用 every/all，M2 用限定名词），不用特殊 token，靠语言本身。
- eval 主结果仍用固定 GENERIC_Q（跨 run 可比）；M2/M3/held-out 措辞作为独立探针套件报告。

---

## 3. 每个数据集的"画面结合"素材（实测盘点）

| 数据集 | 现成 metadata（零成本、可验证） | M2 槽位 | 缺口 |
|---|---|---|---|
| HOI4D | video id `H*_C{n}_*` → **16 物体类**（C1-C9,C11-14,C17,18,C20；官方映射：toy car/mug/laptop/…需对 HOI4D 文档核对一次） | 物体名（"grasp the **kettle**"）；ordinal | ~~hand_anno 手侧/抓握~~（文件已移除；若找回可加 left/right hand、抓握型） |
| TACO | id 无语义（C{人}_{日期}_{序号}） | — | **唯一需要造的**：官方 TACO 有 (动词,工具,目标物) 三元组标注，优先去拿；拿不到就本地 VLM captioning（§4） |
| tennis | 类名自带 far/near court × serve/swing/bounce；video id 含选手名（osaka/federer…但换边风险，**用 court side 不用人名**） | 场地侧（"the player on the **far court**"）；回合序数（"first serve"） | train comment 无用（"extended dataset"） |
| fs_comp | comment = 跳跃/旋转细类：axel/lutz/flip/loop/salchow/toe_loop/flying_camel/flying_sit/flying_upright | **细类指代**（"the **lutz** takeoff"）——直接可用，覆盖率高 | 6 条噪声 comment 过滤 |
| finegym | comment = 元素 ID + **完整自然语言动作描述**（"round-off, flic-flac on, stretched salto backward with 1.5 turn off"）；类名含器械 BB/FX/UB/VT | 器械 + 描述片段（"the balance-beam dismount"、"the salto with 1.5 turn"）——**最富的语言素材** | 描述长，需截取/改写成问题（LLM 离线改写） |
| soccernet_ball | comment = 队伍侧 left/right | **队伍侧**（"When does the **left** team pass?"）——把答案集合砍半，强承重 | 类名需同义词表（DRIVE=dribble/carry 等） |
| finediving（held-out） | comment = 阶段转换描述（"Inward -> 3.5 Soms.Tuck"）、FINA code、难度分 | 零样本探针用：类名 + 描述性措辞两套 | 不训，只测 |

**结论：6 个第一批数据集里 5 个的 M2 指代问题可以纯靠现成 metadata 造出来（可验证、零噪声），只有 TACO 需要 VLM 看图造。** 这比预想的好很多。

---

## 4. TACO（及未来缺口）的 VLM captioning 管线

本地 5090 + 本地 Frames（`/home/chang/Dataset/Action_Spotting/TouchMoment/Frames`）离线跑，不占集群：

1. 对每个 train 事件，取事件帧 ±2 帧，喂 Qwen3-VL（现成在 HF cache）：
   "What tool is the hand holding, and what object is it acting on? Answer: tool=..., object=..."
2. **一致性过滤**：同 clip 多个事件的 tool 答案投票，不一致的 clip 只用 M1 问题（不造 M2）；名词做白名单归一（spoon/spatula/…）。
3. 产出写成 loose 标注字段（`events[].tool/object`），随标注文件同步上 PVC（loose 文件可编辑）。
4. 规模：TACO train 1,689 事件 × 5 帧 ≈ 8.5k 次 VLM 前向，5090 上个把小时。

**红线：caption 只用于丰富问题措辞，绝不改变答案**（答案永远来自人工事件标注）。caption 错了最多是指代词不准（mug 说成 cup），不会引入时间标签噪声。

---

## 5. 指代表达的两条纪律

1. **只用时间不变属性**：物体类别/颜色、场地侧、器械——在窗口内恒定。**禁止**瞬时状态描述（"touch the mug that is tipped over"）——那会把定位任务偷换成状态匹配，且状态只在答案帧成立时等于把答案泄进问题。
2. **可验证性**：M2/M3 的每个指代词必须能被 metadata（或过滤后的 caption）证实/证伪。造 M3 负物体时从**同数据集其他 clip 的槽位词表**里采样（问 mug 视频"scissors"），保证"none"是真的。

---

## 6. 措辞库生产管线（questions_v2）

1. 每 (dataset, class, 模式) 写 2-3 条种子模板 + 槽位 `{object|subtype|side|ordinal}`。
2. 用强 LLM 离线扩写到 20-30 条/类（改动词、句式：疑问/祈使/间接），自动过滤（保留类语义关键词、长度）+ 人工抽查 10%。
3. **train/held-out 双切分**：held-out 不只切句式，**槽位词也切**（train 用 "mug/cup"，held-out 用 "ceramic mug"；train 用 "axel"，held-out 用 "the 1.5-rotation forward-entry jump"）——后者才真正测语言知识路由。
4. 固化成 `questions_v2.py`（确定性种子），训练时每样本每 epoch 重采（沿用现有机制）。

---

## 7. 验证阶梯（每步便宜、可提前止损）

| 实验 | 成本 | 回答的问题 |
|---|---|---|
| **Q-ablation（Step 0.5）**：现有 sft_lora_v2 换/乱问题看输出 | 分钟 | 现在的模型到底读不读题（决定投入底线） |
| **Step 1 四臂对照**：(a) 单模板 (b) M1 bank (c) +M3 负问题 (d) +M2 指代 | 子采样规模各一跑 | 多样性 vs 承重性各贡献多少 |
| 观测指标 | — | in-domain mAP、**held-out 措辞 gap**、**finediving 零样本**（终极判据） |

预期（可证伪）：(a)→(b) 修 held-out gap；(b)→(c) 修"忽略问题"；(c)→(d) 提 finediving 零样本（语言知识路由的直接证据）。若 (d) 不动，说明连接器 token 里没有物体/细类信息可供语言选择——那时再回头看压缩器容量，而不是继续堆 QA。

---

## 8. 与现有代码的接缝

- `data/questions.py` → `questions_v2.py`：QUESTION_BANK 从 `{type: {train:[], test:[]}}` 扩为 `{dataset: {class: {mode: {train:[], heldout:[]}}}}` + 槽位填充函数。
- `WindowedSpottingDataset` 采样时决定 (模式, 措辞, 槽位)，M3 负采样需要窗口的"在场类别/物体"集合（从 events + metadata 得）。
- `_answer_str` 不变（M2 子集、M3 none 都是现有格式）。
- eval harness 增加两个探针套件跑法（held-out 措辞、M2 指代），主表仍 GENERIC_Q。
