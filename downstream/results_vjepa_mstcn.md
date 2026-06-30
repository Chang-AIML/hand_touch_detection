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

Reproduce: `PARALLEL=1 bash downstream/run_vjepa_mstcn.sh` then `python downstream/eval_nms.py --split test`.
