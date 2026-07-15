# InfoNCE contrast — fix-at-extraction (FAIL) vs prevent-collapse-at-readout (WORKS)

## Control: nce_mix3 (MIX-3 warm-start | tap z_t PRE-out.1 | out.1 FROZEN | train Q-Former+heads)
Killed at step85 loss as a concluded negative (mean_cos climbs = z_t collapses MORE; L_gen's generation solution IS the shared key, beats λ0.2 InfoNCE).
```
step5 loss 1.2260 nce 5.357 eff_rank 232.32 mean_cos 0.313
step10 loss 1.2734 nce 5.602 eff_rank 232.65 mean_cos 0.321
step15 loss 1.7299 nce 5.691 eff_rank 203.19 mean_cos 0.370
step20 loss 1.4417 nce 5.789 eff_rank 223.97 mean_cos 0.408
step25 loss 1.5604 nce 5.250 eff_rank 220.95 mean_cos 0.404
step30 loss 1.4822 nce 5.424 eff_rank 235.92 mean_cos 0.381
step35 loss 1.3805 nce 4.724 eff_rank 236.57 mean_cos 0.465
step40 loss 1.3215 nce 5.329 eff_rank 196.76 mean_cos 0.530
step45 loss 1.6588 nce 5.839 eff_rank 206.92 mean_cos 0.652
step50 loss 1.6471 nce 5.749 eff_rank 170.17 mean_cos 0.698
step55 loss 1.8108 nce 5.977 eff_rank 170.35 mean_cos 0.749
step60 loss 1.7181 nce 5.961 eff_rank 159.88 mean_cos 0.790
step65 loss 1.6745 nce 5.918 eff_rank 155.91 mean_cos 0.779
step70 loss 1.3972 nce 6.199 eff_rank 179.82 mean_cos 0.754
step75 loss 1.7244 nce 6.088 eff_rank 158.25 mean_cos 0.761
step80 loss 1.5115 nce 5.848 eff_rank 153.52 mean_cos 0.740
step85 loss 1.4157 nce 5.346 eff_rank 159.64 mean_cos 0.690
```
