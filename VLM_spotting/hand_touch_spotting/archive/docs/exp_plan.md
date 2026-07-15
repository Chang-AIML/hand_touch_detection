# Experiment Plan: VLM-Semantics + V-JEPA-Motion for Egocentric Hand-Touch Action Spotting

> **Audience: executing agent.** Sequentially-executable engineering spec. Each Phase has explicit inputs / outputs / task list / **acceptance gate**. **A Phase that fails its gate must NOT proceed to the next Phase.**
>
> **Scope: action spotting / temporal localization ONLY.** No robotics, no downstream policy, no VR. The deliverable is frame-level hand-touch (contact) and hand-untouch (separation) timestamps and their AP@tIoU.

---

## 0. Overview

### 0.1 Goal (one line)
Given an egocentric video and a natural-language question (e.g. "when does the left hand first contact the object"), output the **frame-level** timestamp(s) of hand-object contact / separation. Metrics: **AP@tIoU {0,1,2,4} + mAP + MAE (frames)**, reported separately for touch / untouch.

### 0.2 Core method (CONVERGED — do NOT change the architecture)
- **Two encoders:**
  - `ViT` (native to Qwen3-VL, **frozen**) @ 1 fps → semantic anchor tokens.
  - `V-JEPA 2.1` (**frozen**) → per-frame motion tokens via **even/odd interleave** (§3.4), then a trainable **adaptor**.
- **LLM:** Qwen3-VL-8B-Instruct (**frozen**, both ViT and LLM backbone).
- **Sequence layout:** `ViT anchor | VJEPA(1s) | ViT anchor | VJEPA(1s) | ... | Question | [LOC]×k [REJ]`, aligned **1 ViT group : F VJEPA tokens per second** (F = video fps; V-JEPA yields one motion token per frame after §3.4).
- **Localization readout:** take the **output hidden (latent)** at the `[LOC]` position and at each VJEPA position; compute `s(t) = sim(h_loc, h_vjepa(t))`; `soft-argmax` → `t*`.
- **Only 2 trainable components:** ① V-JEPA adaptor, ② `[LOC]`/`[REJ]` token embeddings. **Everything else frozen.**
- **Paradigm reference:** LISA's embedding-as-mask ported to the temporal axis = embedding-as-timestamp; multi-token + reject follows GSVA / VRS-HQ.

### 0.3 Three key assumptions (first two are gates)
- **A1 (Phase 0 gate):** V-JEPA output latents (after even/odd interleave, after frozen LLM) are temporally discriminative at the contact frame — `s(t)` peak is sharp (FWHM ≤ threshold). Backed by the user's prior result "V-JEPA + det head does spotting well", plus §3.4 guarantees every frame is a *real* feature (no interpolation). Still must be **quantitatively verified** because the question is whether sharpness survives the frozen LLM.
- **A2 (Phase 1 gate):** After training adaptor + `[LOC]` embedding, single-event AP@2 reaches a usable level.
- **A3 (non-gate, ablation):** LLM semantic conditioning (ViT anchor) contributes positively. If not, drop the LLM stream and fall back to pure V-JEPA contrastive spotting — still publishable.

### 0.4 Hardware allocation
| Use | GPU | Notes |
|---|---|---|
| Phase 0 diagnostic (inference only) | RTX 5090 (32G) | single card enough |
| Phase 1/2 training (frozen backbone + adaptor) | RTX Pro 6000 ×1 or L40S ×1 (48G) | 8B frozen backbone fits on 48G |
| Phase 3 ablation parallelism | Phoenix HPC / 4×RTX 4090 | sweep configs in parallel |

---

## 1. Environment & Dependencies

### 1.1 Software stack
```bash
# Python 3.10+
pip install "torch>=2.4" torchvision
pip install "transformers>=4.57"   # Qwen3-VL requires >=4.57
pip install accelerate peft
pip install decord opencv-python
pip install numpy scipy pandas einops matplotlib
# V-JEPA 2.1 weights: reuse the user's existing checkpoint + extraction code (extract_vjepa21.py)
```

### 1.2 Model weights — use Qwen3-VL, NOT Qwen3.5
- **Primary: `Qwen/Qwen3-VL-8B-Instruct`.** Reasons (architecture, not "which is stronger"):
  - Qwen3-VL is a **dense ViT + LLM** model. The design — inserting VJEPA tokens as independent visual tokens, interleaving with ViT anchors, reading a specific-layer output hidden for sim — depends on this cleanly decomposable token sequence.
  - **Do NOT use Qwen3.5-VL.** Qwen3.5 uses **early-fusion + Gated-Delta / sparse-MoE**. Early fusion makes inserting an external encoder's tokens messy; MoE sparse routing makes "select the mid-layer that is the fine-grained extraction layer" (Phase 0 gate) hard to control. The user's Qwen2.5-VL (dense) KV-probe experience transfers to Qwen3-VL (dense), not to 3.5's MoE.
  - The LLM is **fully frozen** and only a coarse semantic anchor — Qwen3.5's reasoning/agentic strengths give **zero benefit** here.
- **Fallback ONLY if Qwen3-VL is unavailable:** `Qwen2.5-VL-7B-Instruct`. Never 3.5. (Note: 2.5 loses text-timestamp alignment; handle time mapping manually.)
- **V-JEPA 2.1 encoder:** reuse the user's validated checkpoint. `TUBELET=2` (Conv3d, time kernel=stride=2). See §3.4.

### 1.3 Version sanity check (agent's first step)
```python
# scripts/00_env_check.py
# 1. Qwen3-VL loads, exposes hidden_states (output_hidden_states=True)
# 2. any intermediate layer hidden extractable (Phase 0/1 sweep)
# 3. V-JEPA 2.1 loads; confirm TUBELET=2, record feature_dim
# 4. print: LLM hidden_dim, VJEPA feature_dim, num_layers, video fps F
# Output: configs/model_dims.yaml
```
**Acceptance:** `model_dims.yaml` generated; dummy forward returns `hidden_states` tuple of length `num_layers + 1`.

---

## 2. Repo skeleton

```
hand_touch_spotting/
├── configs/
│   ├── model_dims.yaml
│   ├── data.yaml
│   └── train.yaml
├── data/
│   ├── dataset.py
│   ├── vjepa_interleave.py      # CORE §3.4: even/odd two-pass, per-frame tokens
│   ├── token_layout.py          # CORE §3.3: interleaved ViT|VJEPA sequence + time_index
│   └── annotations/
├── models/
│   ├── vjepa_adaptor.py         # trainable (1); supports 'interleave' mode
│   ├── loc_tokens.py            # trainable (2): [LOC]/[REJ] embeddings
│   └── wrapper.py               # frozen Qwen + token injection + hidden extraction
├── localize/
│   ├── sim_readout.py           # s(t)=sim(h_loc, h_vjepa), soft-argmax
│   └── matching.py              # Phase 2: multi-[LOC] <-> multi-GT (Hungarian)
├── train/
│   ├── loss.py                  # InfoNCE + neighbor hard-neg; REJ BCE
│   └── train_loop.py
├── eval/
│   ├── metrics.py               # AP@tIoU, mAP, MAE
│   └── evaluate.py
├── scripts/
│   ├── 00_env_check.py
│   ├── 10_phase0_diagnostic.py  # GATE 1
│   ├── 20_phase1_train.py
│   ├── 21_phase1_eval.py        # GATE 2
│   ├── 30_phase2_train.py
│   ├── 31_phase2_eval.py
│   └── 40_phase3_ablation.py
└── README.md
```

---

## 3. Data schema & token construction

### 3.1 Video input
- Egocentric RGB video at native fps `F`.
- Primary data: the hand-touch spotting set the user already ran V-JEPA / TSP / ASTRM on (keep comparable to existing tables), plus any long-horizon multi-event subset.

### 3.2 Annotation format (JSON, one object per video)
```json
{
  "video_id": "vid_0001",
  "fps": 25,
  "num_frames": 1500,
  "events": [
    {"type": "touch",   "hand": "left",  "frame": 312, "question": "when does the left hand first contact the object"},
    {"type": "untouch", "hand": "left",  "frame": 468, "question": "when does the left hand release the object"},
    {"type": "touch",   "hand": "right", "frame": 502, "question": "..."}
  ]
}
```
- `frame`: GT event frame (integer).
- Phase 1 sample = (video, single question, single GT frame).
- Phase 2 sample = (video, question(s), multiple GT frames), including "no-event" negatives (for `[REJ]`).

### 3.3 Token layout (`data/token_layout.py`)
```python
# pseudocode
def build_sequence(video, question):
    vjepa_tokens = vjepa_interleave(video)   # §3.4: [N_frames, d_vjepa], ONE token per frame
    vjepa_tokens = vjepa_adaptor(vjepa_tokens)  # trainable (1): d_vjepa -> d_llm
    seq, time_index = [], []
    for sec in range(num_seconds):
        seq.append(("vit", qwen_vit(frame_at(sec)), sec))     # frozen 1fps anchor
        for f in frames_in_second(sec):                        # F frames
            seq.append(("vjepa", vjepa_tokens[f], f))
            time_index.append(f)                               # frame id of this VJEPA token
    seq.append(("question", tokenize(question)))
    seq.append(("loc", loc_embedding))                         # trainable (2)
    return assemble(seq), time_index
```
**Critical check:** `len(time_index) == N_frames`. Pointer argmax index → frame id via `time_index`. **Write a unit test; if wrong the whole localization is offset.**

### 3.4 V-JEPA even/odd interleave — per-frame real features (`data/vjepa_interleave.py`) **[CORE, implement exactly]**

**Problem.** V-JEPA 2.1 patch embedding is a `Conv3d` with time `kernel = stride = TUBELET = 2`. So 16 frames → **8** temporal tokens, each covering 2 adjacent frames. A single pass gives neither per-frame features nor a clean "this feature belongs to frame t".

**Solution — run twice with a 1-frame offset, then interleave:**
```
even pass: frames [0,1,2,3,...]  -> tubelets (0,1)(2,3)... -> assign to first frame -> frames 0,2,4,...
odd  pass: frames [1,2,3,4,...]  -> tubelets (1,2)(3,4)... -> assign to first frame -> frames 1,3,5,...
interleave: out[2k] = even[k];  out[2k+1] = odd[k]
```
Implementation: `_run_pass(proc)` and `_run_pass(proc[1:])`; the adaptor's `interleave` mode does `out[0::2]=even; out[1::2]=odd`.

**Why this is frame-consistent (agent should preserve all four properties):**
1. **Every frame is a real feature, not interpolated.** even pass computes true features for even frames, odd pass for odd frames — each frame is genuinely forwarded once. This is the essential difference from single-stream `even`/`odd` interpolation (where the other half of frames are blurry averages of neighbors).
2. **Semantics are uniform (key).** Mapping:
   - feature of even frame `2k` = tubelet `(2k, 2k+1)`
   - feature of odd frame `2k+1` = tubelet `(2k+1, 2k+2)`
   → feature of **any** frame `t` = tubelet `(t, t+1)` = the forward 2-frame motion feature of "this frame + its next". Every frame carries the identical semantic — no odd/even mismatch — so adjacent-frame features are directly comparable and temporally coherent. This is the root of "frame-consistent".
3. **Exact, deterministic alignment.** Conv3d output tokens are time-major (`reshape (T', H', W', D)`); token `k` strictly covers window `(2k, 2k+1)`. Offsetting the two passes by 1 frame covers every `(t, t+1)` pair densely with no gaps — frame→feature is exact, not approximate.
4. **Boundary via replication.** The last frame `N-1` pairs with a pad (replicate last frame), so it has a well-defined tubelet (motion ≈ 0 there, acceptable).

**V-JEPA 2.1 helps:** its Dense Predictive Loss recipe is designed for high-quality, temporally-consistent dense features — the encoder output is already temporally coherent; even/odd interleave just spreads that coherence losslessly onto every frame.

**Result:** `vjepa_interleave(video)` returns exactly one real motion token per frame, semantics uniform across frames. No interpolation anywhere in the main path.

---

## 4. Phase 0 — Feasibility Diagnostic [GATE 1]

> **Purpose:** verify A1 before any training. **Half a day to 1 day.** Fail → fallback (§10), do NOT enter Phase 1.

### 4.1 Inputs
- ~200 GT contact / separation frames.
- Frozen Qwen3-VL + frozen V-JEPA + even/odd interleave (§3.4) + **untrained** adaptor (identity or random init).

### 4.2 Task list
1. Per GT event, build sequence, forward frozen Qwen, take output `hidden_states` (all layers).
2. Temporary query (last question-token hidden, or random `[LOC]`): `s(t) = cos(h_query, h_vjepa(t))`.
3. Plot `s(t)` vs frame, align to GT, average.
4. **Sweep layers** {6, 12, 18, 24, last}: compute FWHM, find sharpest.
5. Compare: raw V-JEPA feature space (pre-LLM) vs LLM output latent (post-LLM) — does the frozen LLM preserve sharpness.

### 4.3 Code skeleton
```python
# scripts/10_phase0_diagnostic.py
for layer in [6, 12, 18, 24, -1]:
    curves = []
    for ev in sampled_events:
        seq, time_index = build_sequence(ev.video, ev.question)
        hs = qwen_forward(seq, output_hidden_states=True)   # frozen
        h_query = hs[layer][loc_pos]
        h_vjepa = hs[layer][vjepa_positions]                # [T, d]
        s = cosine(h_query, h_vjepa)                        # [T]
        curves.append(align_to_gt(s, ev.gt_frame, window=15))
    mean_curve = np.mean(curves, axis=0)
    log(layer, fwhm=compute_fwhm(mean_curve), offset=argmax(mean_curve)-center)
# Output: results/phase0_fwhm_by_layer.csv + plots
```

### 4.4 Acceptance GATE 1
- **GO:** some layer has average `s(t)` with **FWHM ≤ 4 frames** and peak offset ≤ 2 frames → record `configs/train.yaml: sim_layer`.
- **WEAK GO:** FWHM ∈ (4, 8] → may proceed, must enable neighbor hard-neg; expect limited precision.
- **NO-GO:** all layers FWHM > 8 → STOP, trigger R1 (§10), re-run.

---

## 5. Phase 1 — Single-Event Localization Baseline [GATE 2]

> Single `[LOC]`, train only adaptor + `[LOC]` embedding. **1–1.5 weeks.**

### 5.1 Trainable components (only two)
```python
# models/vjepa_adaptor.py: lightweight MLP/Transformer, d_vjepa -> d_llm, with 'interleave' assembly (§3.4)
# models/loc_tokens.py:    nn.Parameter([1, d_llm]) as [LOC] embedding
# All else requires_grad=False. Assert trainable params < 5% of total before training.
```

### 5.2 Loss (`train/loss.py`)
```python
def loc_loss(s, gt_index, k=4):
    # s: [T]; softmax-CE over positions OR InfoNCE
    pos = gt_index
    hard_neg = [gt_index-k .. gt_index+k] \ {gt_index}   # neighbor hard negatives
    return info_nce(s, pos, hard_neg, other_video_frames_as_easy_neg)
```

### 5.3 Training config (`configs/train.yaml`)
```yaml
sim_layer: <from_phase0>
batch_size: 4            # grad accumulation if OOM
lr_adaptor: 1e-4
lr_loc_embed: 1e-3
epochs: 20
neighbor_k: 4
vjepa_mode: interleave   # §3.4 even/odd
optimizer: adamw
freeze: [qwen_vit, qwen_llm, vjepa]
```

### 5.4 Evaluation (`eval/metrics.py`, aligned with existing tables)
- Implement **AP@tIoU {0,1,2,4}**, **mAP**, **MAE (frames)**, touch/untouch separately.
- Reproduce the user's existing table conventions; validate the metric implementation against a known baseline's output first.

### 5.5 Acceptance GATE 2
- **GO:** single-event **AP@2 ≥ user's V-JEPA+det-head baseline AP@2** (LLM conditioning does not hurt). Ideal: near/above TSP+Soft-NMS AP@2.
- **NO-GO:** clearly below V-JEPA baseline → check (a) time_index offset, (b) sim_layer, (c) adaptor/interleave assembly (§3.4 ordering `out[0::2]/out[1::2]`), then retrain; else R3.

---

## 6. Phase 2 — Multi-Event + Rejection

> Long-horizon multi-contact. Multiple `[LOC]` + `[REJ]` (GSVA multi-SEG+REJ, VRS-HQ two-level tokens). **1 week.**

### 6.1 Changes
- k `[LOC]` tokens (k = max events, e.g. 8) + `[REJ]`.
- Each `[LOC]_i` → `s_i(t)` → soft-argmax → candidate `t*_i`.
- **Matching** (`localize/matching.py`): Hungarian, k candidates ↔ multiple GTs (DETR-style set prediction). Unmatched → `[REJ]` "empty", no output.

### 6.2 Loss
```python
L = L_match(loc_loss) + lambda_rej * L_rej(bce)   # matched -> loc_loss; unmatched -> reject
```

### 6.3 Acceptance
- Multi-event **mAP@012** (touch/untouch) ≥ reasonable extrapolation of Phase 1.
- Report `[REJ]` accuracy on no-event queries.
- Compare MAE vs EgoLoc (zero-shot, iterative multi-event) — expected better; selling point.

---

## 7. Phase 3 — Ablations

> Each dimension independent, rest fixed at Phase 2 best. Parallelize on 4090/HPC. **1 week.**

| Dimension | Variants | Verifies | Assumption |
|---|---|---|---|
| sim layer | {6,12,18,24,last} | fine-grained layer location | Phase 0 recheck |
| V-JEPA frame handling | even/odd interleave (§3.4) / single-pass half-rate / single-stream interpolation | per-frame real feature vs approximation | A1 |
| training objective | InfoNCE / softmax-CE / MSE | necessity of contrastive training | A1 |
| neighbor hard-neg k | {2,4,8,16} | discrimination granularity | K-homogeneity |
| **LLM semantic conditioning** | with ViT anchor / without (pure V-JEPA+LOC) | LLM contribution | **A3 (critical)** |
| token count | single [LOC] / multi [LOC] | is a single token enough | VRS-HQ point |
| retrieval space | V-JEPA / Qwen ViT@fps | cross-space alignment necessity | R3 |

**The V-JEPA frame-handling ablation directly validates §3.4:** even/odd interleave (every frame a real feature) should beat single-stream interpolation (half the frames are neighbor averages). This isolates the value of the two-pass design.

**A3 is the story fork:** if "without ViT anchor" matches "with", LLM semantics add nothing → reframe from "VLM-assisted" to "V-JEPA contrastive spotting", drop LLM. Acceptable negative result.

---

## 8. Evaluation protocol

### 8.1 Metric notes (`eval/metrics.py`)
- **AP@tIoU:** predicted timestamp → interval with tolerance τ frames → tIoU vs GT → standard AP. Thresholds {0,1,2,4}.
- **mAP:** mean over @0/1/2 (excludes @4). **Label clearly vs @4-inclusive convention.**
- **MAE (frames):** mean |pred_frame − gt_frame|.
- Report touch / untouch **separately**.

### 8.2 Baseline stratification (report by stratum, do NOT mix)
| Category | Methods | Property |
|---|---|---|
| supervised (existing) | V-JEPA+det, TSP-MSTCN, TSP-ASFormer, ASTRM | supervised |
| zero-shot | EgoLoc, GreedyVLM | training-free |
| ours | this pipeline (±LLM, ±multi-token) | supervised + LLM semantics |

---

## 9. Timeline (~4.5 weeks)

| Stage | Duration | Gate |
|---|---|---|
| env check §1 | 0.5 day | model_dims generated |
| **Phase 0 §4** | 0.5–1 day | **GATE 1 (FWHM)** |
| Phase 1 §5 | 1–1.5 weeks | **GATE 2 (AP@2)** |
| Phase 2 §6 | 1 week | multi-event mAP + REJ |
| Phase 3 §7 | 1 week | ablations complete |
| writing / extra exp | 0.5 week | — |

**Phase 0 fails → stop, run fallback, do NOT enter Phase 1.**

---

## 10. Risks & Fallbacks

| ID | Risk | Trigger | Fallback |
|---|---|---|---|
| R1 | V-JEPA output latents not sharp post-LLM | Phase 0 NO-GO | ① sim on V-JEPA difference space; ② switch mid-layer; ③ sim on **pre-LLM** V-JEPA feature (bypass LLM smoothing) with LLM only gating. Re-run Phase 0. |
| R2 | LLM semantics contribute nothing | Phase 3 A3 flat | Drop LLM, fall back to pure V-JEPA contrastive spotting. Publishable, adjust story. |
| R3 | Cross-space alignment hard | Phase 1 AP stuck + offset | Use Qwen ViT@fps as retrieval space too; no cross-space projection. |
| R4 | Context too long / OOM | long video exceeds VRAM | Coarse-to-fine: 1fps locates candidate window, inject per-frame V-JEPA only inside window. |
| R5 | Time mapping offset | systematic shift | Check `time_index` unit test; verify §3.4 interleave ordering `out[0::2]=even`; use Qwen3-VL text-timestamp alignment. |
| R6 | interleave assembly bug | doubled/halved token count, or even/odd swapped | Assert `len(vjepa_tokens)==N_frames`; unit test that token t ≈ tubelet(t,t+1) semantics (adjacent tokens smoothly varying). |

---

## 11. Execution order (agent checklist)

```
[ ] 1. Run scripts/00_env_check.py -> model_dims.yaml (confirm TUBELET=2, fps F)
[ ] 2. Implement data/vjepa_interleave.py (§3.4) + unit test: len==N_frames, token t ~ tubelet(t,t+1)
[ ] 3. Implement data/token_layout.py + time_index unit test (MUST pass)
[ ] 4. Run scripts/10_phase0_diagnostic.py -> check GATE 1
        └─ NO-GO -> R1, re-run until GO or infeasible
[ ] 5. Implement models/ (adaptor with 'interleave' mode + loc_tokens + wrapper); assert trainable <5%
[ ] 6. Implement train/loss.py + eval/metrics.py (validate metrics on a known baseline first)
[ ] 7. Run scripts/20 -> 21 -> check GATE 2
        └─ NO-GO -> check time_index / sim_layer / interleave ordering, or R3
[ ] 8. Extend to multi [LOC] + [REJ] + matching, run Phase 2
[ ] 9. Run all Phase 3 ablations (A3 first, then V-JEPA frame-handling)
[ ] 10. Compile stratified baseline table + ablation table
```

---

## 12. Invariants (agent must obey)

1. **Qwen3-VL (ViT+LLM) and V-JEPA are frozen at all times.** Assert before training.
2. **Only 2 trainable components:** V-JEPA adaptor, `[LOC]`/`[REJ]` embeddings.
3. **V-JEPA motion tokens come from §3.4 even/odd interleave — one real feature per frame, NO interpolation in the main path.** Ordering: `out[0::2]=even; out[1::2]=odd`. Semantics: token t = tubelet(t, t+1).
4. **`time_index` maps VJEPA token position ↔ frame id; unit-tested.**
5. **sim uses the Phase-0 layer**, not last layer by default.
6. **Metric conventions comparable to existing V-JEPA/TSP/ASTRM tables; state whether mAP includes @4.**
7. **No Phase proceeds past a failed gate.**
8. **Backbone is Qwen3-VL, never Qwen3.5** (fallback Qwen2.5-VL).
9. **Scope is action spotting only** — no robotics/downstream code.