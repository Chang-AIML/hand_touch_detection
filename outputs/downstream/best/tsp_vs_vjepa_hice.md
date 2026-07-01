# TSP vs V-JEPA — unified comparison (hice NMS/Soft-NMS)

Test set (424 videos). All models evaluated with the SAME method: bbvisual/hice
`non_maximum_supression` (greedy, window=1) and `soft_non_maximum_supression`
(parabolic `score*=(d/w)^2`, window=4). mAP in %.

TSP = R(2+1)D-34 (TSP-pretrained) features + head. V-JEPA = V-JEPA 2.1 ViT-B/16 per-frame
features + MS-TCN, 4 adapter modes. Raw predictions in this folder (`*.recall.json.gz`).

model             method           @0     @1     @2     @4    mAP(0-2)    Avg(0-2-4)
----------------  ------------  -----  -----  -----  -----  ----------  ------------
TSP-MSTCN         none          24.96  50.84  62.23  66.89       46.01         51.23
TSP-MSTCN         NMS(w=1)      14.03  61.97  79.59  88.38       51.86         60.99
TSP-MSTCN         SoftNMS(w=4)  12.97  62.36  81.20  89.98       52.18         61.63
TSP-ASFormer      none          24.16  47.07  57.88  62.03       43.04         47.79
TSP-ASFormer      NMS(w=1)      13.66  60.72  79.76  89.00       51.38         60.78
TSP-ASFormer      SoftNMS(w=4)  11.47  60.82  81.89  91.65       51.40         61.46
VJEPA-interleave  none          19.66  49.83  67.84  81.28       45.77         54.65
VJEPA-interleave  NMS(w=1)       9.61  53.11  72.13  86.28       44.95         55.28
VJEPA-interleave  SoftNMS(w=4)  11.02  50.95  72.37  87.02       44.78         55.34
VJEPA-even        none          19.02  52.05  69.54  81.60       46.87         55.55
VJEPA-even        NMS(w=1)       9.78  53.88  73.97  86.38       45.88         56.00
VJEPA-even        SoftNMS(w=4)  11.76  51.79  73.71  86.87       45.75         56.03
VJEPA-odd         none          19.90  49.21  68.90  82.04       46.00         55.01
VJEPA-odd         NMS(w=1)      10.71  52.00  73.33  86.61       45.35         55.67
VJEPA-odd         SoftNMS(w=4)  12.38  49.44  73.13  86.84       44.98         55.45
VJEPA-stack       none          20.24  52.03  71.06  81.65       47.78         56.25
VJEPA-stack       NMS(w=1)      10.56  55.65  76.78  88.71       47.66         57.93
VJEPA-stack       SoftNMS(w=4)  12.12  53.06  76.49  89.04       47.22         57.68

## Takeaways
- Same-method comparison: **TSP-MSTCN is best** on both mAP(0-2) and Avg, ahead of every V-JEPA mode.
- Post-processing (NMS/Soft-NMS) >> none; Soft-NMS ~ NMS (Soft slightly better at loose tol).
- Among V-JEPA modes, `stack` is strongest.
