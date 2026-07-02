# Test-set results — Touch Onset (touch) / Touch Release (untouch)

Per-class AP@{0,1,2} and per-class **mAP = mean(AP@0,AP@1,AP@2)**, on the **test** split.
NMS / Soft-NMS are the **hice `util/eval.py`** kernels (hard greedy `w=1`; parabolic
soft-NMS `score *= |d|²/w²`, `w=4`), scored identically for every model via
`methods/spot_head/eval_nms.py`. V-JEPA + MS-TCN uses the `interleave` (exact per-frame) mode.

> `V-JEPA + HiCE (Ours)` rows are left blank — that model is external; drop in its
> `pred-test.*.recall.json.gz` and re-run `eval_nms.py` to fill.

## HOI4D

### without NMS
| Models | Onset mAP | @0 | @1 | @2 | Release mAP | @0 | @1 | @2 |
|---|---|---|---|---|---|---|---|---|
| TSP + MS-TCN | 39.63 | 21.17 | 43.06 | 54.66 | 52.39 | 28.75 | 58.63 | 69.80 |
| TSP + ASFormer | 37.38 | 20.74 | 40.09 | 51.31 | 48.70 | 27.58 | 54.06 | 64.45 |
| V-JEPA + MS-TCN | 41.56 | 18.10 | 44.03 | 62.57 | 49.99 | 21.22 | 55.62 | 73.11 |
| ASTRM | 26.05 | 18.21 | 27.75 | 32.19 | 31.09 | 23.45 | 33.60 | 36.23 |
| V-JEPA + HiCE (Ours) | — | — | — | — | — | — | — | — |

### NMS (hice, w=1)
| Models | Onset mAP | @0 | @1 | @2 | Release mAP | @0 | @1 | @2 |
|---|---|---|---|---|---|---|---|---|
| TSP + MS-TCN | 46.66 | 10.98 | 55.13 | 73.86 | 57.07 | 17.08 | 68.80 | 85.32 |
| TSP + ASFormer | 46.26 | 11.80 | 53.09 | 73.89 | 56.50 | 15.52 | 68.35 | 85.63 |
| V-JEPA + MS-TCN | 41.52 | 8.54 | 48.41 | 67.60 | 48.39 | 10.69 | 57.81 | 76.66 |
| ASTRM | 37.78 | 11.05 | 45.23 | 57.05 | 45.31 | 13.36 | 56.17 | 66.40 |
| V-JEPA + HiCE (Ours) | — | — | — | — | — | — | — | — |

### Soft-NMS (hice, w=4)
| Models | Onset mAP | @0 | @1 | @2 | Release mAP | @0 | @1 | @2 |
|---|---|---|---|---|---|---|---|---|
| TSP + MS-TCN | 47.21 | 9.51 | 55.86 | 76.26 | 57.15 | 16.43 | 68.87 | 86.14 |
| TSP + ASFormer | 46.72 | 9.51 | 53.40 | 77.25 | 56.07 | 13.43 | 68.24 | 86.54 |
| V-JEPA + MS-TCN | 40.89 | 9.05 | 45.53 | 68.09 | 48.67 | 12.98 | 56.36 | 76.66 |
| ASTRM | 43.57 | 8.99 | 47.50 | 74.22 | 51.95 | 11.88 | 60.24 | 83.72 |
| V-JEPA + HiCE (Ours) | — | — | — | — | — | — | — | — |

## touchmoment

> Note: `data/touchmoment/val.json` is the shared 15fps val (138 videos); train/test add
> 30fps videos. Test measurement is valid (disjoint splits); epoch selection is on 15fps val.

### without NMS
| Models | Onset mAP | @0 | @1 | @2 | Release mAP | @0 | @1 | @2 |
|---|---|---|---|---|---|---|---|---|
| TSP + MS-TCN | 31.19 | 15.34 | 31.77 | 46.44 | 51.09 | 27.19 | 57.82 | 68.26 |
| TSP + ASFormer | 29.40 | 17.16 | 30.65 | 40.40 | 49.38 | 26.32 | 54.41 | 67.40 |
| V-JEPA + MS-TCN | 39.45 | 17.86 | 41.20 | 59.30 | 48.41 | 22.11 | 53.95 | 69.16 |
| ASTRM | 18.29 | 13.56 | 19.09 | 22.22 | 24.67 | 18.51 | 25.96 | 29.54 |
| V-JEPA + HiCE (Ours) | — | — | — | — | — | — | — | — |

### NMS (hice, w=1)
| Models | Onset mAP | @0 | @1 | @2 | Release mAP | @0 | @1 | @2 |
|---|---|---|---|---|---|---|---|---|
| TSP + MS-TCN | 37.35 | 7.14 | 41.82 | 63.09 | 55.89 | 15.10 | 68.15 | 84.41 |
| TSP + ASFormer | 39.26 | 9.08 | 44.59 | 64.10 | 54.10 | 14.69 | 64.87 | 82.75 |
| V-JEPA + MS-TCN | 41.66 | 9.08 | 47.14 | 68.78 | 49.65 | 11.86 | 60.12 | 76.97 |
| ASTRM | 26.56 | 6.55 | 32.29 | 40.83 | 36.15 | 11.41 | 43.51 | 53.55 |
| V-JEPA + HiCE (Ours) | — | — | — | — | — | — | — | — |

### Soft-NMS (hice, w=4)
| Models | Onset mAP | @0 | @1 | @2 | Release mAP | @0 | @1 | @2 |
|---|---|---|---|---|---|---|---|---|
| TSP + MS-TCN | 37.53 | 5.28 | 40.18 | 67.13 | 55.92 | 13.54 | 67.72 | 86.51 |
| TSP + ASFormer | 39.90 | 6.89 | 43.57 | 69.24 | 53.82 | 12.35 | 65.09 | 84.00 |
| V-JEPA + MS-TCN | 41.45 | 8.43 | 45.44 | 70.46 | 49.53 | 13.26 | 58.17 | 77.15 |
| ASTRM | 32.66 | 4.57 | 34.20 | 59.21 | 42.24 | 9.53 | 45.88 | 71.30 |
| V-JEPA + HiCE (Ours) | — | — | — | — | — | — | — | — |
