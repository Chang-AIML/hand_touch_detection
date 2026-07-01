# HOI4D performance (test set, 424 videos)

Onset = touch, Release = untouch. mAP = mean of that class's AP@{0,1,2}.
"+HiCE" = hice post-processing (Soft-NMS w=4); rows without it = raw (no suppression).
V-JEPA rows use the best adapter mode (stack).

| Models | Onset mAP | Onset AP@0 | Onset AP@1 | Onset AP@2 | Release mAP | Release AP@0 | Release AP@1 | Release AP@2 |
|---|---|---|---|---|---|---|---|---|
| TSP + MS-TCN          | 39.63 | 21.17 | 43.06 | 54.66 | 52.39 | 28.75 | 58.63 | 69.80 |
| TSP + ASFormer        | 37.38 | 20.74 | 40.09 | 51.31 | 48.70 | 27.58 | 54.06 | 64.45 |
| V-JEPA + MS-TCN       | 43.76 | 17.83 | 46.34 | 67.11 | 51.80 | 22.65 | 57.73 | 75.01 |
| V-JEPA + HiCE (Ours)  | 43.53 |  9.49 | 47.86 | 73.24 | 50.92 | 14.75 | 58.27 | 79.74 |
| T-DEED + HiCE (Ours)  | 40.57 |  6.47 | 43.82 | 71.42 | 50.54 | 11.58 | 61.10 | 78.96 |
