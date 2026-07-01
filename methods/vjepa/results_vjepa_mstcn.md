# V-JEPA → MS-TCN results (HOI4D-touch, test set)

Feature: V-JEPA 2.1 ViT-B/16-384 per-frame (768-d), 4 adapter modes. Head: MS-TCN (3-stage),
clip_len=100, 50 epochs, best epoch by val mAP (interleave=34, even=39, odd=36, stack=49).
Post-processing: no-NMS (dense high-recall), hard NMS (window=1, E2E-Spot default),
Gaussian temporal Soft-NMS (window=4, σ=0.5). mAP in %.

| mode | method | mAP@0 | mAP@1 | mAP@2 | mAP@4 | Avg |
|---|---|---|---|---|---|---|
| **interleave** | without NMS | 19.66 | 49.83 | 67.84 | 81.28 | 54.65 |
| **interleave** | NMS (w=1) | 7.40 | 45.95 | 69.52 | 86.22 | 52.27 |
| **interleave** | Soft-NMS | 17.76 | 52.59 | 72.14 | 86.48 | **57.24** |
| **even** | without NMS | 19.02 | 52.05 | 69.54 | 81.60 | 55.55 |
| **even** | NMS (w=1) | 7.76 | 46.70 | 70.68 | 86.39 | 52.88 |
| **even** | Soft-NMS | 17.79 | 53.99 | 73.27 | 86.28 | **57.83** |
| **odd** | without NMS | 19.90 | 49.21 | 68.90 | 82.04 | 55.01 |
| **odd** | NMS (w=1) | 8.00 | 43.45 | 69.67 | 86.18 | 51.83 |
| **odd** | Soft-NMS | 18.44 | 51.35 | 72.59 | 86.37 | **57.19** |
| **stack** | without NMS | 20.24 | 52.03 | 71.06 | 81.65 | 56.25 |
| **stack** | NMS (w=1) | 7.42 | 45.49 | 72.13 | 88.34 | 53.35 |
| **stack** | Soft-NMS | 18.87 | 54.92 | 76.17 | 88.38 | **59.58** |

## Takeaways

- **Post-processing**: Soft-NMS > no-NMS > hard-NMS on Avg for every mode. Hard NMS (w=1) badly
  hurts the strict @0 tolerance (19→7) but helps the loose @4; Soft-NMS keeps @0 high *and* wins @1/2/4.
- **Mode**: `stack` is best (Soft-NMS Avg **59.58**); `even` (57.83) ≈ `interleave` (57.24) ≈ `odd` (57.19).
  The "exact per-frame" `interleave` does **not** dominate on test — the single-stream/stacked variants
  are competitive or slightly better, suggesting the half-rate streams already carry the signal and the
  training-time even/odd augmentation (`stack`) helps most.

Reproduce: `PARALLEL=1 bash methods/vjepa/run_vjepa_mstcn.sh` then `python methods/spot_head/eval_nms.py --split test`.

## Per-class AP (%) — touch

| mode | method | @0 | @1 | @2 | @4 | Avg |
|---|---|---|---|---|---|---|
| interleave | without NMS | 18.10 | 44.03 | 62.57 | 79.68 | 51.09 |
| interleave | NMS (w=1) | 6.23 | 40.79 | 64.28 | 85.43 | 49.18 |
| interleave | Soft-NMS | 15.41 | 47.29 | 67.63 | 86.11 | 54.11 |
| even | without NMS | 16.53 | 44.57 | 63.14 | 79.92 | 51.04 |
| even | NMS (w=1) | 5.44 | 39.68 | 65.05 | 85.39 | 48.89 |
| even | Soft-NMS | 14.57 | 47.00 | 67.88 | 85.54 | 53.75 |
| odd | without NMS | 17.86 | 45.03 | 65.95 | 82.35 | 52.80 |
| odd | NMS (w=1) | 6.76 | 39.21 | 67.05 | 85.96 | 49.74 |
| odd | Soft-NMS | 15.68 | 47.34 | 69.94 | 86.75 | 54.93 |
| stack | without NMS | 17.83 | 46.34 | 67.11 | 80.93 | 53.05 |
| stack | NMS (w=1) | 5.54 | 39.56 | 67.66 | 88.08 | 50.21 |
| stack | Soft-NMS | 15.95 | 49.52 | 72.46 | 88.47 | **56.60** |

## Per-class AP (%) — untouch

| mode | method | @0 | @1 | @2 | @4 | Avg |
|---|---|---|---|---|---|---|
| interleave | without NMS | 21.22 | 55.62 | 73.11 | 82.88 | 58.21 |
| interleave | NMS (w=1) | 8.56 | 51.10 | 74.75 | 87.01 | 55.36 |
| interleave | Soft-NMS | 20.12 | 57.89 | 76.65 | 86.86 | 60.38 |
| even | without NMS | 21.50 | 59.54 | 75.93 | 83.27 | 60.06 |
| even | NMS (w=1) | 10.08 | 53.72 | 76.31 | 87.38 | 56.87 |
| even | Soft-NMS | 21.01 | 60.98 | 78.66 | 87.03 | 61.92 |
| odd | without NMS | 21.93 | 53.38 | 71.84 | 81.73 | 57.22 |
| odd | NMS (w=1) | 9.25 | 47.70 | 72.28 | 86.40 | 53.91 |
| odd | Soft-NMS | 21.20 | 55.36 | 75.24 | 85.99 | 59.45 |
| stack | without NMS | 22.65 | 57.73 | 75.01 | 82.37 | 59.44 |
| stack | NMS (w=1) | 9.31 | 51.42 | 76.60 | 88.59 | 56.48 |
| stack | Soft-NMS | 21.78 | 60.32 | 79.88 | 88.29 | **62.57** |

**untouch > touch by ~5–8 AP across all modes/methods** (touch — contact onset — is harder to spot than untouch — release). Run `python methods/spot_head/eval_nms.py --split test --per-class`.
