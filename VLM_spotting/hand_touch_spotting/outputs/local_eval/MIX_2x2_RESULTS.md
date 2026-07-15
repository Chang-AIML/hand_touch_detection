# Connector weight-surgery 2×2 — ID + OOD + temporal-reversal (for external analysis)

## Setup
- Base run `p2_slide_a70`: FrameCompress connector (27.55M, only trainable) mapping frozen V-JEPA-2.1 → frozen Qwen3-VL-8B; output = frame indices. In-domain = {finegym, tennis, fs_perf, touchmoment}; OOD (zero-shot) = finediving.
- Metric: point-event mAP at frame tolerances {0,1,2,4,8,16}. In-domain eval balance = 640 finegym / 200 others; OOD = full val (finediving, 1018 windows, ngt 1063).
- Two checkpoints are grafted: **s750** (OOD-peak) and **s1200** (in-domain-high).

### The connector's 4 parameter blocks
`q` (8×768 learnable queries, 6K) · `in_ln` (LN, 1.5K) · `a3` (cross-attn "Q-Former", 2.36M) · `out` = `out.0` LN(6144) + `out.1` Linear 6144→4096 (25.2M readout). Total 27.55M.

### Two 2×2 splits (grafting s750 ↔ s1200)
- **Split A — q vs rest:** `MIX-1` = q@s750 + {in_ln,a3,out}@s1200 · `MIX-2` = q@s1200 + rest@s750.
- **Split B — Q-Former{q,in_ln,a3} vs readout{out}:** `MIX-3` = QF@s1200 + out@s750 · `MIX-4` = QF@s750 + out@s1200.

### ⚠️ Data-quality notes
1. **ID numbers below are from DPC** (verified: ndet≈ngt, healthy per-dataset). An earlier 5090 ID run was **invalid** (a launch-path bug left the connector un-loaded → random init → ~0.3) and is discarded. OOD numbers are from the 5090 and are **valid** (verified: each ckpt's OOD differs → connector was loaded).
2. **MIX-2 ID is re-running** (its first DPC run hit a temp-file race). Marked TBD.
3. **@16 is saturated by a position prior** for everyone (finediving position-prior baseline @16 = 60.75; see last section). Treat **@8 as the genuine OOD-transfer metric**; @16 is not discriminative.
4. Single seed. touchmoment eval questions are `PYTHONHASHSEED`-dependent (≤ a few points on that one dataset).

---

## 1. In-domain mAP (aggregate `_all` over the 4 datasets), by tolerance
| checkpoint | @0 | @1 | @2 | @4 | @8 | @16 | ndet (ngt 1772) |
|---|---|---|---|---|---|---|---|
| **s750** (orig) | 8.11 | 28.38 | 39.93 | 48.82 | 55.50 | 59.61 | 1756 |
| **s1200** (orig) | 11.24 | 37.59 | 50.18 | 59.13 | 66.40 | 70.16 | 1746 |
| s1350 (orig, ID-opt) | 13.07 | 39.13 | 50.72 | 61.03 | 67.83 | 71.41 | 1704 |
| **MIX-1** q750+rest1200 | 11.32 | 37.93 | 49.02 | 58.80 | 65.39 | 69.44 | 1775 |
| **MIX-2** q1200+rest750 | 7.64 | 27.79 | 38.66 | 48.69 | 54.80 | 59.60 | 1741 |
| **MIX-3** QF1200+out750 | 8.41 | 31.38 | 42.56 | 51.89 | 57.57 | 62.23 | 1833 |
| **MIX-4** QF750+out1200 | 11.08 | 32.56 | 43.67 | 56.08 | 61.33 | 65.41 | 1733 |

## 2. OOD mAP (finediving, zero-shot, full val), by tolerance
| checkpoint | @0 | @1 | @2 | @4 | @8 | @16 | ndet (ngt 1063) |
|---|---|---|---|---|---|---|---|
| **s750** (orig) | 0.24 | 3.01 | 7.68 | 26.45 | 48.17 | 65.81 | 1006 |
| **s1200** (orig) | 0.20 | 1.73 | 4.43 | 15.44 | 31.36 | 38.62 | 926 |
| s1350 (orig) | 0.29 | 1.79 | 4.65 | 15.11 | 30.77 | 36.51 | 926 |
| **MIX-1** q750+rest1200 | 0.27 | 1.79 | 5.05 | 15.78 | 31.37 | 38.35 | 928 |
| **MIX-2** q1200+rest750 | 0.21 | 2.85 | 7.69 | 27.10 | 48.03 | 66.99 | 1005 |
| **MIX-3** QF1200+out750 | 0.44 | 4.17 | 9.32 | 26.20 | 43.83 | 59.24 | 989 |
| **MIX-4** QF750+out1200 | 0.14 | 1.15 | 3.83 | 16.77 | 36.87 | 48.52 | 950 |
| *position-prior baseline* | 0.24 | 0.75 | 2.17 | 6.44 | 21.76 | 60.75 | 1018 |

## 3. Compact view — ID@2 vs OOD@8 (the two discriminative numbers)
| checkpoint | ID@2 | OOD@8 |
|---|---|---|
| s750 (QF750,out750) | 39.9 | 48.2 |
| s1200 (QF1200,out1200) | 50.2 | 31.4 |
| MIX-1 (q750, rest1200) | 49.0 | 31.4 |
| MIX-2 (q1200, rest750) | 38.7 | 48.0 |
| MIX-3 (QF1200, **out750**) | 42.6 | 43.8 |
| MIX-4 (QF750, **out1200**) | 43.7 | 36.9 |

Observations (for the analyst to check): OOD@8 tracks **`out`** (out750→44–48, out1200→31–37) and is **independent of `q`** (MIX-1). ID@2 rises toward s1200 with both `out` and `QF` at s1200 (roughly additive). The ID/OOD tension co-locates in **`out.1`**.

## 4. Per-dataset in-domain mAP (DPC), by tolerance — for the loaded MIX ckpts
**MIX-1 (q750+rest1200)** ndet 1775
| ds | @0 | @1 | @2 | @4 | @8 | @16 |
|---|---|---|---|---|---|---|
| finegym | 7.67 | 28.99 | 39.17 | 50.34 | 58.79 | 63.96 |
| fs_perf | 15.67 | 60.07 | 78.36 | 81.45 | 81.45 | 81.70 |
| tennis | 28.01 | 67.03 | 74.83 | 80.89 | 82.86 | 84.33 |
| touchmoment | 9.04 | 44.97 | 65.56 | 78.27 | 83.05 | 85.22 |

**MIX-3 (QF1200+out750)** ndet 1833
| ds | @0 | @1 | @2 | @4 | @8 | @16 |
|---|---|---|---|---|---|---|
| finegym | 4.95 | 24.05 | 34.53 | 44.00 | 50.29 | 55.78 |
| fs_perf | 12.11 | 49.34 | 62.69 | 68.00 | 71.39 | 74.35 |
| tennis | 25.15 | 56.43 | 67.31 | 76.56 | 80.27 | 82.04 |
| touchmoment | 4.51 | 33.91 | 52.65 | 67.95 | 74.66 | 78.54 |

**MIX-4 (QF750+out1200)** ndet 1733
| ds | @0 | @1 | @2 | @4 | @8 | @16 |
|---|---|---|---|---|---|---|
| finegym | 7.94 | 25.05 | 34.75 | 48.51 | 54.84 | 59.96 |
| fs_perf | 18.78 | 55.40 | 69.89 | 79.39 | 79.39 | 79.65 |
| tennis | 24.24 | 56.86 | 68.79 | 73.70 | 76.27 | 77.94 |
| touchmoment | 4.75 | 30.44 | 54.13 | 73.92 | 80.95 | 83.83 |

## 5. Temporal-reversal probe (checkpoint s750, finegym, 60 paired start/end windows)
Method: each finegym element has a paired start (f_s) and end (f_e), f_s<f_e. Run forward AND with frames reversed (frame p → N-1-p), with the start-query and the end-query. On reversed video the original landing looks like a takeoff; if the connector encodes **direction**, the reversed start-query should fire at the reversed-end position `rev_e = N-1-f_e` (a "swap"); if only **saliency/position**, it tracks the mirrored start position or a fixed clip position.

Results (N=60) — **must be stratified by whether the model localizes the event FORWARD**, because the
aggregate is diluted by windows where nothing localizes:

| subset | n | start→rev_e swap | end→rev_s swap | start **within 30f** of rev_e (accurate tracking) |
|---|---|---|---|---|
| **forward-GOOD** (fwd start≈f_s AND end≈f_e, tol 30f) | **18/60** | **100%** | **100%** | **100%** |
| forward-bad (fwd fails to localize) | 42/60 | 81% | 17% | 17% |
| ALL 60 (diluted aggregate) | 60 | 87% | 42% | 42% |

Aggregate context: forward start-error ~66.8f, end-error ~127.3f, start<end in only 37% of windows;
reversed absolute position ~0.23 for both queries.

**Read: DIRECTION IS ENCODED (temporal).** On every window where the model demonstrably localizes the
event forward (18/60), reversing the frames produces a **perfect, accurate swap** — the start-query lands
within 30 frames of the reversed-END (`rev_e = N-1-f_e`), the end-query at the reversed-START, 100%. This
is not a position prior: a prior would fire at the same *absolute* position (= `rev_s`), not `rev_e`; the
model is tracking the reversed *motion* (the reversed landing looks like a takeoff). The diluted 87%/42%
aggregate comes from the 42 forward-FAIL windows (finegym forward localization is only ~30% reliable — a
separate weakness), where reversed behavior is noise. Per-window rows: `outputs/local_eval/reversal_rows.json`.

## 6. Position-prior baseline (finediving OOD) — how it was computed
For each (window, type), predict the per-type median within-window event position (score 1.0), score with the same mAP. Values by tolerance: **@0 0.24 · @1 0.75 · @2 2.17 · @4 6.44 · @8 21.76 · @16 60.75** (ndet 1018, ngt 1063). Use to separate genuine transfer (models ≫ baseline at @2/@4/@8) from position-prior saturation (@16 ≈ baseline for all models).
