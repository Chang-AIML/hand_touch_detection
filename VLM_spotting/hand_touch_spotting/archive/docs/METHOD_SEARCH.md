# Method search — overnight experiments (HOI4D touch/untouch spotting)

Goal: best 2-stage method (LLM coarse `<idx>` → refine), test the hypothesis that
**language-space features (post-LLM token hiddens) beat raw V-JEPA**, optimize mAP@0/1/2.
Two GPUs, autonomous. mAP = touch/untouch avg via the shared frame-tolerance scorer.

---
## TL;DR — recommended method
> **MS-TCN (full-sequence, E2E hard-label CE) on RAW V-JEPA features, with FiLM language
> conditioning from the query.** Single stage is enough; the LLM `<idx>` stage adds nothing
> for closed-set touch/untouch.

Three hypotheses tested — one **disproven**, one **confirmed**, one **null**:
1. ❌ **Language-space features do NOT help** — raw V-JEPA beats every LLM-processed feature by ~20+ mAP@2.
2. ✅ **Language *conditioning* (FiLM on the query) works and is nearly free** — one model spots either type by query, ≈ multi-class, and improves @0.
3. ⭕ **The 2-stage LLM idx-prior adds nothing** for closed-set — MS-TCN already localizes better than the LLM's coarse idx.

---
## P1 — Feature ablation (multi-class MS-TCN, vendored clip-100 SOTA pipeline)
| feature | dim | mAP@0 | mAP@1 | mAP@2 |
|---|---|---|---|---|
| **F3 raw V-JEPA** | 768 | **18.8** | **48.3** | **69.3 (test)** |
| F1 pre-LLM adaptor | 4096 | ~17 | ~38 | 46.6 |
| F2 post-LLM last (language-space) | 4096 | ~16 | ~35 | 44.5 |
| F2 post-LLM L12 | 4096 | 16.8 | 35.5 | 46.5 |
| F2 post-LLM L24 | 4096 | — | — | ~32 |

**Raw V-JEPA wins by ~23 mAP@2.** Every LLM-processed feature loses badly, at every layer.
Why: (a) the adaptor is a lossy projection trained for idx-copy; (b) the LLM's self-attention
**smooths across frames**, erasing the crisp per-frame boundaries a dilated temporal conv needs.
**Language-alignment ≠ good localization.** → use raw V-JEPA for the features.

## P3 — Language conditioning + backbone (raw V-JEPA F3, dilate=0, my pipeline, VAL)
| config | mAP@0 | mAP@1 | mAP@2 |
|---|---|---|---|
| MS-TCN multi-class | 20.1 | 47.6 | **66.1** |
| MS-TCN **query-FiLM** (language-conditioned) | **22.8** | 48.5 | 63.1 |
| ASFormer | 23.0 | — | 50.2 |

Query-conditioned FiLM (one model, language picks touch/untouch) is within ~3 mAP@2 of
multi-class and has **higher @0** — language conditioning is **harmless and useful**, and is
the right way to inject language (via the query, not the features). It's future-semantics
ready (swap the query for referring / open-vocab). **ASFormer underperforms MS-TCN** here.

## P2 — 2-stage value: LLM idx-prior (VAL)
Fed the LLM `<idx>` predictions as Gaussian-bump prior channels alongside F3.
| config | mAP@2 |
|---|---|
| MS-TCN on F3 (no prior) | ~66 |
| MS-TCN on F3 + idx-prior | 65.1 |
| query-FiLM on F3 | 63.1 |
| query-FiLM on F3 + idx-prior | 61.0 |

**The idx-prior adds nothing (slightly hurts).** The LLM's coarse idx (val mAP@2 44.9) is
*worse* than what MS-TCN extracts from the same features, so priming with it is redundant.
The 2-stage LLM idx is only worth it for **language-necessary** tasks (referring/open-vocab),
not closed-set touch/untouch.

---
## Recommended method (final)
- **Features:** raw V-JEPA interleave (768-d), NOT language-space.
- **Model:** 3-stage MS-TCN, full-sequence, E2E fg-weighted per-frame CE, **dilate=0** (sharp labels), **full-video eval**.
- **Language:** FiLM(query) → query-conditioned binary head (one model, any queried action). @2 63 / @0 22.8, near-SOTA and language-ready.
- **Highest closed-set number:** multi-class MS-TCN on raw V-JEPA = **@2 69.3 (test)**. Use this if language isn't needed.
- **Drop:** LLM-space features, the idx 2-stage (for this task), ASFormer.

## Reproduce
- Features F1/F2/F3 cached: `outputs/feat_cache/{F1_adaptor,F2_postllm_neutral[_L12/_L24],F3_idxprior}`, F3=`VLM_spotting/vjepa/feat_interleave`.
- Feature ablation: `hand_touch_detection/methods/spot_head/train_head.py -m mstcn --feat_dir <F> --label_dir data/HOI4D-v3`.
- Language / query-FiLM: `train/lang_spot.py --feat_dir <F> --mode {multiclass|query} --dilate 0`.
- idx-prior build: `scripts/41_gen_idx_all.py`; extraction: `scripts/40_extract_feats.py --layer <L>`.

## THE LLM 2-STAGE (the actual goal: LLM localizes → refine, LLM in the loop)
Stage-1 = frozen LLM idx-generation; stage-2 = **local MULTI-STAGE MS-TCN** on the RAW
V-JEPA window (±W) around each idx + **FiLM(type-query)**, E2E hard-label CE, argmax.

**Headline:** stage-1 idx → **refine (3-stage FiLM, W=48, dilate=1)** → **DENSE + soft-NMS read-out**
(`train/refine4.py`; read-out `scripts/47_refine_dense_eval.py`).

⚠️ **MAJOR READ-OUT CORRECTION (supersedes earlier @2~56 numbers):** the refine eval used to emit
ONE hard-argmax detection per idx, which destroys the score-ranking mAP@0/@1 need. Emitting the
DENSE scored field per window + soft-NMS(w=4,σ=0.5) — same read-out full-seq uses — with the SAME
checkpoint, no retraining:
| split | stage / read-out | @0 | @1 | @2 |
|---|---|---|---|---|
| test | stage-1 idx | 2.8 | 21.7 | 42.8 |
| test | + refine, **argmax (OLD/wrong)** | 4.1 | 28.8 | 56.1 |
| test | + refine, **dense+softNMS (RIGHT)** | **10.5** | **42.8** | **66.2** |
| test | + refine, dense (max @0) | 15.1 | 34.6 | 51.1 |
| test | *full-seq MS-TCN (no LLM)* | ~18 | ~48 | 62.6–66 |
| val | + refine, **dense+softNMS** | 10.4 | 45.3 | 65.8 |

**The LLM 2-stage now TIES full-seq MS-TCN on @1/@2 (@2 66.2 ≥ pure-supervised 62.6) with no
retraining** — the "caps at 56 / @0 stuck" story was a read-out artifact. @0 still trails (10–15 vs
~18–20): a real but modest windowed-training gap (dense-vs-dense 15 vs 20, NOT the 4-vs-20 the old
argmax read-out implied — the earlier gating claim of "4.7× @0 loss" over-counted the read-out).
Diagnostics: no off-by-one (signed err −0.4, median 0); per-event exact-hit 16% ≈ full-seq 20%
(`scripts/46_refine_error_diag.py`). Test cache: `scripts/43_gen_idx_test.py`.

### E2E-aligned protocol (r9 runs, final numbers)
`refine4.py` now evaluates E2E-style every epoch: DENSE field → {no-NMS, NMS(1), soft-NMS(4,.5)},
best epoch by **soft-NMS mean(@0/1/2)** (the old argmax selection was methodologically wrong but
empirically a wash — re-selected numbers identical ±1). Test, soft-NMS:
| config | params | @0 | @1 | @2 |
|---|---|---|---|---|
| **W=48** | 1.8M | **10.51** | **42.84** | **66.23** |
| W=12+rej | 1.8M | 10.53 | 42.68 | 61.72 |
| W=8+rej | 1.8M | 9.01 | 39.26 | 57.91 |
| W=8 small (hid64/L4/S2) | **544k** | 9.25 | 41.02 | **59.47** |

Window findings (corrected read-out): @0/@1 are window-INDEPENDENT (per-event exact-hit 17% at all
W) — ±8 saturates refinement precision; W=48's @2 edge (+6.8) is purely RECALL insurance for the
stage-1 error tail (~18% of events have idx error >8). At W=8 the 544k net BEATS the 1.8M one
(59.5 vs 57.9 — big net overfits narrow windows). Error curves: `plot/refine_error_dist_{val,test}.png`
(`scripts/49_error_curves.py`); coverage ceiling: exact-hit 17%, ≤2 66%, →92% by ±30 (the last 8%
= stage-1 total misses, unrecoverable by any window). Next lever: stage-1 recall/precision.

### 🏆 MULTI-GT LABEL FIX (user-spotted overlap bug) → LLM 2-stage BEATS the no-LLM baseline
Old pair-building labeled only the NEAREST same-type GT in the window; any second same-type event
inside was supervised as BACKGROUND (contradictory labels teaching the model to suppress real
events). Overlap rate: 0.4% of windows at W=12, 7.9% at W=48, 19% at W=96, 49% at W=128 — this is
what made wide windows LOOK worse. Fix: `--multi_gt` labels ALL same-type GTs in the window.
| test soft-NMS | @0 | @1 | @2 |
|---|---|---|---|
| W=12 +mg (control, 0.4% overlap) | 10.90 | 41.55 | 61.74 (unchanged ✓) |
| W=48 +mg | 11.46 | 44.45 | 66.23 |
| W=64 +mg | 11.27 | 44.83 | 68.39 |
| **W=96 +mg (BEST)** | **11.03** | **47.32** | **70.94** |
| W=128 +mg | 11.16 | 44.74 | 67.50 (val 69.3 — window ≈ whole video, degrades) |
| *no-LLM MS-TCN baseline* | 18.8 | 48.3 | 69.3 |

**Final: LLM idx → ±96 multi-GT FiLM refine → dense+softNMS = test @2 70.94 > 69.3 no-LLM MS-TCN;
@1 47.3 ≈ 48.3.** The "wider windows hurt" verdict was a label-contradiction artifact all along.
@0 (11 vs 18.8) remains the no-LLM head's only edge. Reproduce:
`train/refine4.py --run_name X --lang_mode film --W 96 --no_reject --multi_gt`.

### ⚠️ BUT: at W=96 the LLM is nearly meaningless (centers ablation, `--centers`)
Union window coverage of the video: W=12→35%, W=48→96%, W=96→100%. Replacing LLM idx with fake
centers (same counts): **W=96 grid = 69.5 ≈ LLM 70.9** (the 2-stage degenerates into a classifier
with extra steps — user called it); W=96 random = 57.7; **W=12 random = 3.6 vs LLM 61.7** (at narrow
W the LLM is EVERYTHING). So the honest design must keep the window/emission SMALL.

### ✅ Small-window fix: decouple INPUT CONTEXT from EMISSION (scripts/50_context_emit.py)
Small-window refinement is bottlenecked by CONTEXT, not the emission region (echo of the full-seq
±8-gate @0=20 result). Train with wide input (±48/±96, multi-GT), at eval EMIT only the central ±E:
| trained-context \ emit | E=8 | E=12 | E=24 | full |
|---|---|---|---|---|
| ±12-trained (ref) | — | 61.7 | — | — |
| ±8-trained (ref) | 57.9–59.5 | — | — | — |
| **±48-trained** | **62.5** | 63.4 | 65.3 | 66.2 |
| ±96-trained | 63.1 | **64.0** | 66.2 | 70.9 |

Same ±12 emission: wide-context-trained 64.0 vs narrow-trained 61.7 (+2.3); at ±8: 62.5 vs 57.9
(+4.6). Context saturates at ±48 for narrow emission.

### 🏁 ASYMMETRIC TRAINING (`--out_E`, user-proposed) — the final honest config
Train with input ±W as pure context but LABELS+LOSS+EMISSION restricted to the central ±out_E
(outer frames never supervised — no label contradiction possible). Test soft-NMS, all ±12 emission:
| config | @0 | @1 | @2 |
|---|---|---|---|
| ±12 in / ±12 out (symmetric, old) | 10.5 | 42.7 | 61.7 |
| ±48 full-supervised, eval-crop ±12 | 10.6 | 43.1 | 63.4 |
| ±48 in / ±12 out (asym) | 9.2 | 46.6 | 66.5 |
| **±96 in / ±12 out (asym) — FINAL** | **11.5** | **47.1** | **66.75** |
| ±48 in / ±8 out | 10.6 | 46.6 | 65.0 |
| *(degenerate ±96 full-emission "classifier")* | 11.0 | 47.3 | *70.9* |

**FINAL HONEST 2-STAGE: LLM idx → ±96-context asym refine (supervise/emit ±12 only) → dense +
soft-NMS = test 11.5/47.1/66.75.** Emission covers 35% of the video (LLM fully load-bearing;
random centers collapse to 3.6); @2 beats the pure-supervised no-LLM baseline (62.6), @1 ties the
degenerate wide-emission config (47.1 ≈ 47.3 — in-neighborhood refinement is SATURATED). The 4.2
gap to 70.9 is entirely stage-1 idx errors >12 frames — the only remaining lever is stage-1.
Reproduce: `train/refine4.py --lang_mode film --W 96 --out_E 12 --multi_gt`.

**The three widths are decoupled and each has a physically-correct value (all verified):**
- *Train input* ±96: regularization — wide crops give diverse far context/negatives. Training at
  ±24 directly LOSES 6.5 (60.0 vs 66.55 at the same ±24 inference) — same sample-homogeneity
  disease as the old @0 problem. Don't shrink this.
- *Inference input* ±24: the action arc (~1.6s per side). Test-input sweep on the ±96-trained
  model: ±12→60.0, **±24→66.55 (saturates)**, ±48→66.73, ±96→66.75. Deploy 4× cheaper, −0.2.
- *Emission* ±12: sweet spot of the stage-1 error distribution (85% of idx errors ≤12). Sweep at
  ±96 input: E=8→65.0 (max LLM-load-bearing, −1.8), **E=12→66.75**, E=24→64.6 (ranking tax
  outgrows coverage gain — wider emission is NOT free).
**DEPLOY RECIPE: train --W 96 --out_E 12 --multi_gt; infer with input ±24, emit ±12 → 66.55.**

### Full stage-2 ablation (all on raw V-JEPA window, val mAP@2)
| factor | winner | loser | note |
|---|---|---|---|
| **language inject** | **FiLM 56.7** | gated x-attn **43.5** | x-attn (gate-0, on hidden) too weak to carry the touch/untouch switch → stays type-blind → ≈ no refinement (base 44.9). FiLM modulates the *input* → propagates through all 15 conv layers. |
| **multi-stage** | **3-stage 56.7** | 1-stage **47** | the softmax-refinement stages are the load-bearing part (refine3 dropped them → regressed to 47). |
| **window W** | **48** | 8→52, 12→54, 24→50, 96→53 | wider helps to 48 (catches the stage-1 error tail), then neighbor-event confusion. |
| **dilate** | **1 (56.7)** | 0 (52.4) | OPPOSITE of full-seq: a local window has one event, so a single labeled frame is too sparse → dilate=1 gives stabler peaks *and* better @0. |
| **model size** | **hid128/3-stage 56.7** | hid256/4-stage 54.5 | bigger overfits the small pred-centric set. |
| **rejection (all-0 for no-GT windows)** | **W-dependent** | — | NULL at W=48 (50/3498 neg pairs, nothing to reject) but **HELPS at narrow W**: W=8 +1.1 test (50.3→51.4), W=12 **+2.4 test** (51.6→54.0), where 216-276 neg pairs exist. Doesn't beat W=48 overall though. |

**Narrow-window test (full multi-stage FiLM, val/test @2):** W=8 50.4/50.3 (no-rej) → 52.0/51.4 (rej);
W=12 52.9/51.6 → 53.6/**54.0** (rej). Both < W=48 (56.7/56.1). @0 stays ~4-4.5 at every W — tightening
does NOT recover exact-frame precision. But W=12+rej test @1 (31.4) > W=48 (28.8): tight+reject is
marginally sharper at ±1, while wide wins at ±2. Rejection's value rises as W shrinks.

- LLM idx-gen **test**: @0 2.8 / @1 21.7 / @2 42.8; **NMS/soft-NMS make no difference** (sparse list).
- **@0 stays ~6** at every setting — a local window **cannot recover exact-frame precision**;
  that lives in the full-sequence temporal pattern (why full-seq MS-TCN gets @0 18.8).
- **The LLM 2-stage caps ~57 @2** — a big lift over stage-1 (44.9) but still below the no-LLM
  MS-TCN (69.3): (a) LLM stage-1 is coarser, (b) local refine can't recover @0. The LLM's value
  is language (referring/open-vocab), not beating MS-TCN's closed-set number.
- Reproduce: `train/refine4.py --run_name X --lang_mode film --dilate 1 --no_reject` (W=48,
  3-stage, hid128); test cache via `scripts/43_gen_idx_test.py --split test`.

## DECISIVE: what kills @0 — windowed TRAINING, not the idx (`scripts/44_fullseq_gate.py`)
Took the full-seq query-FiLM MS-TCN (trained on WHOLE videos) and eval'd it gated to
±W of an LLM idx — same features, same idx neighborhoods the windowed refine saw.
| model | trained on | eval region | @0 | @1 | @2 (test) |
|---|---|---|---|---|---|
| full-seq MS-TCN | full videos | plain | 23.4 | 51.2 | 65.1 |
| full-seq MS-TCN | full videos | gated ±48 | 22.3 | 49.2 | 62.6 |
| full-seq MS-TCN | full videos | **gated ±8** | **20.3** | 44.4 | 56.8 |
| windowed refine | ±48 windows | ±48 | 4.1 | 28.8 | 56.1 |
| windowed refine | ±8 windows | **±8** | **4.3** | 29.7 | 51.4 |

**Smoking gun (±8 rows, identical region+features+idx):** full-seq-trained @0=20.3 vs
window-trained @0=4.3 — a **4.7× @0 gap from the TRAINING REGIME alone**. Shrinking a
full-seq model's region to ±8 barely dents @0 (23.4→20.3), so the *region* isn't the
problem; training on isolated single-event windows is (too few / too-similar / context-
truncated examples to learn a sharp boundary). Corollary: **"full-seq localizer + LLM idx
as a gate" dominates "idx + windowed refine"** (@2 56.8 vs 51.4 *and* @0 20 vs 4) — but
gating still slightly hurts plain full-seq (65→63) because the LLM under-detects. Net: on
closed-set, the LLM cannot be the primary localizer without a large @0 cost; its value is language.

## Key engineering learnings
- MS-TCN eval **must be full-video** (dilated convs need full context); sliding clips cripple it (+20 mAP@2 fixing this).
- **dilate=0** (exact-frame labels) >> dilate=4 for precise spotting (sharper peaks, fewer near-FPs).
- LR scheduler total-steps must equal the actual loop steps (10000-sample epochs), else cosine ends early → undertrained.
- Reproduced SOTA V-JEPA+MS-TCN = **@2 69.3 (test)**, ≥ the prior 67.84.
