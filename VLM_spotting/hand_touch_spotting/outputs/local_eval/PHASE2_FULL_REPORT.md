# `p2_slide_a70` — Experimental Record

## 1. Setup

**Model.** V-JEPA 2.1 ViT-B (frozen) → **FrameCompress connector (27.5M, the only trainable module)**
→ Qwen3-VL-8B (frozen). Output = frame indices, generated directly (no chain-of-thought).

**Task.** Natural-language-queried frame-level point-event spotting.
- In-domain (trained): `touchmoment`, `tennis`, `finegym`, `fs_perf`
- Held-out, never trained (zero-shot): `finediving`

**Training.** 4×A100-40GB, FSDP over the frozen decoder. eff_batch 64 (bs 1 × 4 ranks × grad_accum 16),
lr 3e-4, warmup 150, **1500 steps**, window 600 frames (non-overlapping), Qwen-ViT anchor every 5 s.
Query = general NL description of the action, paraphrase-sampled per step.
Sampling = temperature shuffle-epoch (without replacement), α = 0.70.
Negatives = within-clip type-absent (0.15) + in-domain other-type (1.5/window) + cross-dataset (0.15),
capped at 40 % per dataset, **finegym capped at 5 %**. jitter = 30. Total train pool = 53,923 samples.
Run crashed at step 840 (NCCL all-gather timeout, shared-PVC I/O), resumed losslessly at step 841,
completed 1500.

**Evaluation.** Local RTX 5090, deterministic (seeded stratified subsample, no shuffling per run).
- in-domain: 640 (window,type) samples for finegym, 200 each for tennis / fs_perf / touchmoment
- **finediving OOD: full validation set, 1018 samples**
- Metric: mAP at frame tolerances {0, 1, 2, 4, 8, 16}. `eval_max_tokens` 64.
- s450 and s600 were evaluated before the tolerance list was widened → only {0,1,2,4} available.

---

## 2. In-domain results

### 2.1 Aggregate (`_all`, over the 4 in-domain datasets)

| step | @0 | @1 | @2 | @4 | @8 | @16 | ndet | ngt |
|---|---|---|---|---|---|---|---|---|
| 150 | 0.03 | 0.40 | 0.52 | 0.87 | 2.29 | 6.41 | 1525 | 1772 |
| 300 | 0.58 | 1.98 | 4.22 | 9.30 | 14.48 | 21.94 | 1765 | 1772 |
| 450 | 5.30 | 14.74 | 20.73 | 28.97 | — | — | 1540 | 1772 |
| 600 | 6.84 | 21.12 | 27.59 | 35.20 | — | — | 1666 | 1772 |
| 750 | 8.11 | 28.38 | 39.93 | 48.82 | 55.50 | 59.61 | 1756 | 1772 |
| 900 | 9.29 | 30.80 | 42.50 | 54.67 | 60.56 | 64.93 | 1840 | 1772 |
| 1050 | 10.61 | 34.06 | 45.94 | 57.00 | 63.07 | 66.64 | 1761 | 1772 |
| 1200 | 11.24 | 37.59 | 50.18 | 59.13 | 66.40 | 70.16 | 1746 | 1772 |
| 1350 | 13.07 | 39.13 | 50.72 | 61.03 | 67.83 | 71.41 | 1704 | 1772 |
| 1500 | 12.47 | 39.19 | 50.77 | 61.18 | 68.23 | 71.91 | 1709 | 1772 |

Step-to-step change in `_all` @2: +6.86, +12.34, +2.57, +3.44, +4.24, +0.54, **+0.05**.

### 2.2 Per-dataset, mAP@2

| step | finegym | fs_perf | tennis | touchmoment |
|---|---|---|---|---|
| 150 | 0.56 | 0.03 | 0.64 | 0.50 |
| 300 | 2.87 | 9.26 | 5.95 | 9.96 |
| 450 | 16.15 | 27.61 | 33.72 | 38.99 |
| 600 | 25.75 | 27.96 | 29.82 | 48.59 |
| 750 | 33.03 | 54.33 | 62.86 | 49.14 |
| 900 | 33.43 | 60.54 | 72.08 | 58.16 |
| 1050 | 36.37 | 69.24 | 74.59 | 61.72 |
| 1200 | 40.91 | 77.39 | 75.06 | 64.82 |
| 1350 | 40.96 | 77.96 | 78.13 | 65.42 |
| 1500 | 41.50 | 73.60 | 79.45 | 62.81 |

### 2.3 Per-dataset, mAP@4

| step | finegym | fs_perf | tennis | touchmoment |
|---|---|---|---|---|
| 450 | 23.08 | 42.70 | 41.18 | 56.25 |
| 600 | 32.38 | 42.06 | 35.32 | 64.82 |
| 750 | 40.99 | 66.18 | 70.87 | 69.35 |
| 900 | 46.14 | 76.52 | 78.26 | 72.42 |
| 1050 | 48.59 | 75.81 | 81.22 | 76.94 |
| 1200 | 50.76 | 82.46 | 80.60 | 77.77 |
| 1350 | 51.85 | 86.62 | 85.50 | 78.91 |
| 1500 | 52.80 | 80.65 | 85.78 | 78.35 |

### 2.4 Per-dataset, mAP@8 and @16 (available from s750)

| step | fg @8 | fs @8 | tn @8 | tm @8 | fg @16 | fs @16 | tn @16 | tm @16 |
|---|---|---|---|---|---|---|---|---|
| 750 | 48.80 | 66.45 | 75.16 | 78.42 | 53.42 | 70.73 | 76.90 | 81.52 |
| 900 | 53.19 | 77.13 | 81.14 | 79.94 | 58.50 | 77.82 | 83.65 | 82.52 |
| 1050 | 56.09 | 75.81 | 84.34 | 82.01 | 60.51 | 76.09 | 86.02 | 84.70 |
| 1200 | 60.16 | 82.46 | 82.66 | 82.11 | 65.02 | 82.71 | 83.80 | 83.86 |
| 1350 | 60.78 | 86.62 | 86.82 | 82.53 | 65.29 | 86.88 | 87.96 | 85.50 |
| 1500 | 62.04 | 80.65 | 86.90 | 83.29 | 66.67 | 80.92 | 88.26 | 86.15 |

### 2.5 Per-dataset, mAP@0 (frame-exact)

| step | finegym | fs_perf | tennis | touchmoment |
|---|---|---|---|---|
| 450 | 5.38 | 3.20 | 7.12 | 2.71 |
| 600 | 7.19 | 6.87 | 5.60 | 5.18 |
| 750 | 5.20 | 12.75 | 20.70 | 6.25 |
| 900 | 6.82 | 10.79 | 21.98 | 6.66 |
| 1050 | 7.08 | 13.72 | 27.40 | 8.76 |
| 1200 | 7.79 | 14.76 | 27.68 | 8.40 |
| 1350 | 9.04 | 19.43 | 31.34 | 8.07 |
| 1500 | 8.06 | 18.22 | 32.26 | 9.80 |

### 2.6 Full per-dataset detail at the final step (s1500)

| dataset | @0 | @1 | @2 | @4 | @8 | @16 | ndet | ngt |
|---|---|---|---|---|---|---|---|---|
| finegym | 8.06 | 29.57 | 41.50 | 52.80 | 62.04 | 66.67 | 761 | 786 |
| fs_perf | 18.22 | 61.58 | 73.60 | 80.65 | 80.65 | 80.92 | 177 | 182 |
| tennis | 32.26 | 71.71 | 79.45 | 85.78 | 86.90 | 88.26 | 317 | 336 |
| touchmoment | 9.80 | 46.00 | 62.81 | 78.35 | 83.29 | 86.15 | 454 | 468 |

---

## 3. OOD results — `finediving` (zero-shot, full val, ngt = 1063)

| step | @0 | @1 | @2 | @4 | @8 | @16 | ndet |
|---|---|---|---|---|---|---|---|
| 150 | 0.00 | 0.01 | 0.11 | 0.28 | 0.99 | 4.74 | 926 |
| 300 | 0.17 | 2.09 | 7.94 | 18.39 | 40.81 | 65.18 | 1015 |
| 450 | 0.23 | 1.60 | 5.77 | 24.85 | 44.47 | 60.87 | 1007 |
| 600 | 0.53 | 2.01 | 5.23 | 24.08 | — | — | 964 |
| 750 | 0.24 | 3.01 | 7.68 | 26.45 | 48.17 | 65.81 | 1006 |
| 900 | 0.41 | 2.49 | 5.58 | 20.40 | 38.62 | 46.94 | 985 |
| 1050 | 0.12 | 1.16 | 3.51 | 13.28 | 30.47 | 37.67 | 841 |
| 1200 | 0.20 | 1.73 | 4.43 | 15.44 | 31.36 | 38.62 | 926 |
| 1350 | 0.29 | 1.79 | 4.65 | 15.11 | 30.77 | 36.51 | 926 |
| 1500 | 0.29 | 2.02 | 4.45 | 15.12 | 30.86 | 36.61 | 917 |

---

## 4. Observed phenomena

1. **In-domain `_all` rises monotonically across every tolerance and flattens.**
   @2 goes 20.73 → 50.77; the per-step increment falls from +12.34 (s600→s750) to +0.54 (s1200→s1350)
   and **+0.05** (s1350→s1500). Per-dataset at s1350→s1500, `fs_perf` @2 drops 77.96 → 73.60 while the
   others rise or hold; `_all` is flat.

2. **Every in-domain dataset improves monotonically**, except `tennis` at s600 (33.72 → 29.82 → 62.86)
   and small non-monotonic wobbles in `fs_perf` @8/@16 at s1050.

3. **mAP@0 differs strongly by dataset.** At s1350: `tennis` 31.34, `fs_perf` 19.43,
   `finegym` 9.04, `touchmoment` 8.07.

4. **OOD emerges very fast, then peaks at ~s750.** @16 jumps from 4.74 (s150) to 65.18 (s300) —
   i.e. it reaches its plateau level by s300, while in-domain @2 is still only 4.22. But the two
   tolerances behave differently: **@16 is already at s750-level by s300** (65.18 vs 65.81), whereas
   **@8 keeps building to s750** (s300 40.81 → s750 48.17). The position-prior baseline (§9) explains
   this: @16 is saturated by position prior (baseline 70.03), which is learned fast; the genuine
   above-baseline transfer (clearest at @4/@8) builds more slowly and peaks at s750. After s750, both
   fall through s900 and s1050.

5. **OOD stabilizes from s1050 onward.** Four consecutive points (s1050/1200/1350/1500):
   @2 = 3.51 / 4.43 / 4.65 / 4.45, @8 = 30.47 / 31.36 / 30.77 / 30.86, @16 = 37.67 / 38.62 / 36.51 / 36.61.

6. **The checkpoint maximizing OOD (s750) is not the one maximizing in-domain (s1350).**
   From s750 → s1350: in-domain @2 +26.9 %-pt, OOD @16 −29.3 %-pt.

7. **OOD `ndet` dips at s1050** (841 vs ngt 1063) and recovers to 926 at s1200/s1350;
   in-domain `ndet` stays close to `ngt` (1704–1840 vs 1772) at all steps.

8. **The OOD tolerance curve is far steeper than the in-domain one.**
   At s750, OOD goes 0.24 (@0) → 65.81 (@16); in-domain goes 8.11 → 59.61.

---

## 5. GRPO rollout-diversity diagnostic (checkpoint **s750**)

**Method.** For N positive windows, sample **G = 8 rollouts** with `do_sample=True` at temperature T,
parse the predicted frame set from each, score each rollout with **F1@2** (greedy bipartite match,
a prediction counts as a hit iff |p − g| ≤ 2 and g is its nearest unmatched GT).
GRPO's advantage is `A_i = r_i − mean(r)`, so a group with zero reward variance contributes no gradient.

| split | T | reward_std | frac(std>0) | reward_mean | uniq_preds / 8 | N |
|---|---|---|---|---|---|---|
| in-domain | 0.7 | 0.231 | 0.78 | 0.513 | 5.74 | 27 |
| in-domain | 1.0 | 0.272 | 0.93 | 0.501 | 6.63 | 27 |
| in-domain | 1.3 | 0.316 | 0.96 | 0.420 | 7.00 | 27 |
| finediving OOD | 0.7 | 0.120 | 0.30 | 0.150 | 5.40 | 30 |
| finediving OOD | 1.0 | 0.182 | 0.43 | 0.188 | 6.13 | 30 |
| finediving OOD | 1.3 | 0.167 | 0.40 | 0.129 | 6.53 | 30 |

### Measured facts
- **Predictions are diverse in both splits** (unique frame-sets: in-domain 6.63/8, OOD 6.13/8 at T = 1.0).
- **In-domain**: 93 % of windows have non-zero within-group reward variance at T = 1.0, with
  `reward_mean` ≈ 0.50. Raising T to 1.3 raises variance but lowers `reward_mean` to 0.42.
- **OOD**: only **43 %** of windows have non-zero reward variance at T = 1.0, with `reward_mean` 0.19.
  Increasing T to 1.3 does **not** help (0.40) and lowers `reward_mean` to 0.13.
- Therefore, under an **F1@2** reward, the OOD signal is limited by **reward sparsity**, not by
  prediction diversity, and temperature cannot recover it.

---

## 6. RLVR reward design

### 6.1 Constraint imposed by §5
A hard-tolerance reward (`F1@2`) leaves 57 % of OOD windows with zero gradient. Under GRPO, only the
**spread** of rewards inside a group matters, so the reward must assign **different, non-zero scores to
predictions that are close-but-not-exact**. Span grounding does not face this (IoU is continuous);
point-event F1@tol is a step function.

### 6.2 Candidate: soft-F1 with distance decay

```
① soft credit:      c(p, g) = exp(−|p − g| / τ)
② bipartite match:  each prediction ↔ at most one GT (greedy, by largest c)
③ soft-F1:
      soft_TP = Σ c(matched pairs)
      FP = # unmatched predictions,  FN = # unmatched GT
      R_task = 2·soft_TP / (2·soft_TP + FP + FN)          ∈ [0, 1]
   edge cases:  GT ∅ and pred ∅ → R = 1 ;  GT ∅ and pred ≠ ∅ → R = 0
④ optional drift control:
      R_total = R_task − β · KL(π_θ ‖ π_ref),   π_ref = the s750 policy
```

Credit values by τ:

| τ | Δ=2 | Δ=4 | Δ=8 | Δ=12 | Δ=16 |
|---|---|---|---|---|---|
| 4 | 0.61 | 0.37 | 0.14 | 0.05 | 0.02 |
| **8** | 0.78 | 0.61 | **0.37** | 0.22 | 0.14 |

Observed OOD prediction errors concentrate in the 4–12 frame range (consistent with @8 = 48, @16 = 66
at s750), so τ = 8 keeps them in a discriminative band; τ = 4 compresses them toward zero.

### 6.3 Notes on other reward forms
| form | note |
|---|---|
| `F1@2` | Measured: 43 % OOD frac(std>0). |
| staircase `1 / 0.5 / 0` at ≤4 / ≤8 / else | Zero beyond 8 frames — the band where OOD errors sit. |
| multi-tolerance `F1@{2,4,8}` | Denser than `F1@2`; still zero beyond 8 frames. |
| `R_format` term | The model is thinking-free and emits a bare frame list; format is valid in nearly every rollout, so this term does not discriminate between rollouts. |
| unbounded `−λ·|Δ|` | Unbounded negative values inflate the within-group variance used by GRPO. Distance is already carried by `c`. |
| per-sample `AP@4` | AP is a dataset-level ranking statistic; it is not defined on a single window. |
| single-event rewards `R = exp(−|t̂ − t*|/τ)` | Defined for one event; our windows contain **sets** of events, so a matching step plus FP/FN accounting is required. |

### 6.4 Not yet determined by data
- τ, the decay shape (linear vs exponential vs convex), and the weighting between a dense term and an
  explicit `F1@2` term. A rerun of the §5 diagnostic that scores the **same rollouts** under several
  reward definitions would settle this (~20 min, no retraining).
- Whether the trainable set should be connector-only (LLM frozen) or connector + LoRA.

---

## 7. External reference numbers (E2E-Spot, published)

| dataset | E2E-Spot δ=1 | E2E-Spot δ=2 |
|---|---|---|
| Tennis | 96.1 | 97.7 |
| FS-Comp | 81.0 | 93.5 |
| FS-Perf | 85.1 | 95.7 |
| FineDiving (supervised) | 68.4 | 85.3 |
| FG-Full | 47.9 | 65.2 |
| FG-Start | 61.0 | 78.4 |

Note: their FineGym is reported as FG-Full / FG-Start; our `finegym` uses a 32-type set that mixes
start and end events, so the two are not directly comparable. Their FineDiving figure is **supervised**;
ours is zero-shot.

---

## 8. Missing data and caveats

1. **s150 and s300** were evaluated on a separate 2×A100 HPC host (torch cu118 due to a CUDA-11.6-era
   driver; A100-PCIE, ~2.5 s/window for finediving) and are included above. Full curve = 10 steps
   (s150…s1500). s1500 evaluated on the local 5090.
2. **s450 and s600 lack @8 / @16** (the tolerance list was widened afterwards).
3. **No ablations.** This run changed six things at once relative to the previous run
   (dataset list, query representation, sampling scheme, α, negative design, jitter), so no single
   change can be credited with any observed difference.
4. **Single seed.**
5. **The OOD evaluation contains no background windows.** `finediving` clips are 111 frames, i.e. one
   window each, and every evaluated window contains the queried event. The measurement is therefore
   "localize the event given the clip", not "detect events and reject background".
   Measured empty-window fractions in the evaluation sets: `fs_perf` 31.5 %, `finegym` 8.6 %,
   `tennis` 0.5 %, `touchmoment` 0 %, `finediving` **0 %**.
6. **Evaluation uses non-overlapping 600-frame tiling**, not an overlapping sliding window with NMS.
7. Two class-level NL descriptions are known to be inaccurate: `touchmoment` (generated with a
   gymnastics context; unused at train time because touch/untouch fall back to the curated HOI4D bank)
   and `fs_perf | jump_landing` (says "gymnast … ground" instead of "skater … ice").
8. 15 of 812 instance-level descriptions failed generation (long vault comments) and fall back to
   class-level descriptions.
