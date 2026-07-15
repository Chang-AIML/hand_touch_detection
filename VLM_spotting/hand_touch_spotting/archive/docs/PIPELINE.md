# VLM Action Spotting — Pipeline & Training Spec (for optimization review)

Frozen-VLM emergent action spotting. Given an egocentric video + a natural-language
question ("when does the hand first make contact with the object"), predict the
**frame** of the event. Sibling to a supervised V-JEPA→MS-TCN baseline (same data/metrics).

## 1. Task & data
- **Dataset:** HOI4D-v3. 2288 / 138 / 424 videos (train/val/test), each **300 frames @ 15 fps**,
  two event types: `touch`, `untouch`. One video has multiple events.
- **Samples:** one per event = (video, question, gt_frame, type). train 10153 / val 647 / test 1945.
- **Metrics:** frame-tolerance **mAP@{0,1,2,4}** + **MAE(frames)**, touch/untouch separately, via the
  shared E2E-Spot scorer (`hand_touch_detection/common/score.py`).
- **Baseline to beat (supervised):** V-JEPA→MS-TCN (interleave) **mAP@2 = 67.84 (no-NMS) / 72.14 (soft-NMS)**.

## 2. Backbones — ALL FROZEN
- **Qwen3-VL-8B-Instruct**: ViT (vision tower) + LLM (36 layers, hidden **d=4096**). Both frozen.
- **V-JEPA 2.1 ViT-B/16-384** (dim **768**), frozen. Per-frame features via **even/odd interleave**
  (patch tubelet=2 → run encoder twice with a 1-frame offset → one *real* motion token per frame,
  token t = tubelet(t,t+1)). Pre-extracted `[300,768]` fp16 per video, spatially mean-pooled.

## 3. Trainable components — only these (~10M, 0.11% of total)
1. **Motion Bridge adaptor** (input-side): `LayerNorm(768) → Linear(768,2048) → GELU → Linear(2048,4096)`,
   output RMS-scaled to Qwen's input-embedding scale. Job = translate V-JEPA motion → Qwen token space.
2. **Shared `[LOC]` token** embedding: `1×4096` (a single temporal-pointer soft-prompt; its meaning is
   set by the question, NOT per-event tokens).
3. **(optional) FiLM** head: `MLP(4096→1024→2·4096)` → (γ,β), zero-init (starts as identity). +12.6M.
- **No decode head / no SimHead** (deliberately — to avoid a task-specific LISA-style decoder).

## 4. Token sequence (built in embedding space, injected as `inputs_embeds`)
Per second (fps=15):
```
[ViT-anchor(sec0)] [VJEPA f0..f14] [ViT-anchor(sec1)] [VJEPA f15..f29] ... [question toks] [LOC]
```
- **ViT anchor** = frozen Qwen-ViT tokens of the 1-fps frame, images downsized to 252px (~49 tokens/anchor),
  **cached per video** (frozen → computed once). Makes the LLM query video-aware.
- **VJEPA token** = `adaptor(vjepa_feat[f])`, one per frame (all 300 injected; 150 in a training crop).
- Aligned exactly `ViT(0)|VJEPA 0..14 | ViT(15)|VJEPA 15..29 | ...`.

## 5. Forward
- Inject the assembled `(L,d)` embeds; run **only the first `sim_layer=12` of 36 LLM layers**
  (truncated forward, gradient-checkpointed) → hidden at layer 12. Backprop flows *through* the frozen
  LLM to the adaptor + LOC (frozen params, but grad passes to inputs).
- **Position ids (KNOWN ISSUE):** currently **flat sequential arange**, all M-RoPE channels equal
  (text-like). We do NOT use Qwen's native video M-RoPE (temporal t / spatial h,w). Planned fix (below).

## 6. Readout — `align_target = adaptor` (the key finding)
- **query** `q = h_loc` = hidden at the `[LOC]` slot, layer 12 (post-LLM).
- **keys** `z_t = adaptor(vjepa_t)` — the **pre-LLM** bridge output (position-free), NOT the post-LLM
  V-JEPA hidden. (Aligning to the pre-LLM adaptor output gives the bridge a *direct* gradient instead of
  one attenuated through 12 frozen layers — this was the difference between "learns" and "doesn't".)
- (FiLM: `z'_t = LayerNorm(z_t)·(1+γ(q)) + β(q)`.)
- **score** `s(t) = cos(q, z_t) / temp`, `temp=0.1`. Plain cosine retrieval, no head.

## 7. Loss (per training window)
**Data augmentation:** random temporal crop = **10-second (150-frame)** window, fps-aligned, random
offset that contains the event (breaks the absolute event-position prior that the "first/second" ordinal
leaks). **40% of crops are NEGATIVE** (exclude the event, gt=−1).

- **Positive window** (`loss_mode=dist`, distance-weighted contrastive — current):
  ```
  L = CE(s, GaussianSoftTarget(gt, σ=4))                                   # smooth single peak
    + far_weight · mean_t relu( far_margin·|t-gt|/N − (s_gt − s_t) )       # far frames pushed below peak
    + λ_mae · |softargmax(s) − gt| / N
  far_margin=5.0, far_weight=1.0, λ_mae=0.5   (units: s = cos/temp)
  ```
  (Alt `loss_mode=ce`: softmax-CE with Gaussian label-smoothing σ=1 — the earlier version.)
- **Negative window** (reject, scale-free): `L = λ_neg · relu( max_t softmax(s)_t − 3/W )`, `λ_neg=2.0`
  → flattens non-event windows so they emit no confident peak (event presence/absence).

## 8. Training
- **Dual-GPU DDP**: backbone replicated frozen on each GPU; only the ~10M trainable grads are
  all-reduced (manual average). Per-GPU batch **24** → effective **48**.
- AdamW, `lr_adaptor=1e-3`, `lr_loc=3e-3`, cosine schedule + 5% warmup, **25 epochs**,
  gradient checkpointing on, bf16 backbone / fp32 trainable. wandb logging.
- ~20 min/epoch (epoch 0 slower: one-time ViT-anchor precompute, then cached).

## 9. Inference (full 300-frame or irregular video)
- Model trains on 150-frame crops → at test we **slide 10s windows (stride 5s)** over the whole video.
- Each window → `s(t)` over its frames. **Dense confidence-weighted aggregation**: for each *global*
  frame, average `s(t)` across all covering windows, weighted by each window's peak softmax prob
  (flat/non-event windows contribute ~uniformly). `argmax` of the global curve → predicted frame.
- Handles any video length; no need to feed 300 frames at once.

## 10. Current results & diagnostics (best checkpoint so far)
- **Local-window (GT-containing crop):** MAE 27, **hit@2 = 51%** (touch & untouch). → bridge retrieval works.
- **Language conditioning** (same video, q_touch vs q_untouch ranking): **93–100%**. → query is
  question-specific, not generic motion.
- **Full-video (honest sliding):** **mAP@2 ≈ 13**, MAE ≈ 74. ← the bottleneck.
- **Failure mode:** score curves are **multi-peak** (38% of videos have a 2nd peak >0.8× the top);
  full-video argmax hit@2 is actually 30%, no systematic position bias — a spurious *far* peak wins ~38%
  of the time → huge MAE.
- **Calibration:** event-present vs event-absent windows barely separate (this is being addressed by the
  scale-free reject + distance loss).

## 11. Open problems / levers under test (what to optimize)
1. **Full-video mAP@2 plateaus ~13** (vs 67.84 supervised). Multi-peak / global window selection is the gap.
2. **Distance-weighted contrastive loss** (§7) — under test, targets the far spurious peaks.
3. **Query-conditioned FiLM** (§3) — de-entangle contact vs release motion.
4. **RoPE fix (planned):** give injected tokens proper video M-RoPE — V-JEPA frame f → temporal `t=f`;
   ViT anchor(sec s) → `t=s·15` (shares time with V-JEPA); V-JEPA has no spatial so `h=w=t` (flattened);
   anchors keep real h/w grid. Currently flat sequential (anchors are OOD to the frozen LLM).
5. If all the above still plateau at 10–15: the frozen-VLM + tiny-bridge may be capacity-limited; next
   controlled step is a small **temporal-routing adapter / LoRA**, not more epochs.

## 12. Code map (`VLM_spotting/hand_touch_spotting/`)
- `data/vjepa_interleave.py` (§2 extraction), `data/token_layout.py` (§4), `data/dataset.py` (§7 crop+neg)
- `models/wrapper.py` (frozen Qwen inject + truncated forward + RoPE), `models/vjepa_adaptor.py`,
  `models/loc_tokens.py`, `models/film.py`, `models/localizer.py` (§6 readout)
- `train/loss.py` (§7), `train/train_loop.py` (§8, DDP), `eval/metrics.py` (§9 sliding+aggregation)
- `scripts/12_report.py` (4-metric diagnostic), `scripts/13_curves.py` (curve/distribution plots)
- `configs/train.yaml` (all hyperparams)
