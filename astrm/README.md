# ASTRM — Precise Event Spotting (HOI4D touch / untouch)

Reproduction of the **ASTRM** precise-event-spotting method, applied end-to-end to
**HOI4D-v3** hand touch/untouch moment detection.

> ASTRM paper: *Precise Event Spotting in Sports Videos: Solving Long-Range
> Dependency and Class Imbalance* (CVPR), Santra, Chudasama, Wasnik,
> Balasubramanian.

The code is built on the **E2E-Spot** framework (Hong et al., ECCV 2022) — the
backbone/temporal/dataset scaffolding is derived from
[jhong93/spot](https://jhong93.github.io/projects/spot.html); see `LICENSE`.

---

## What this is

An end-to-end RGB event spotter:

```
frames (B,T,3,H,W)
  └─ RegNetY-200MF backbone (ImageNet pretrained)
       └─ ASTRM inserted after conv1 of every bottleneck  (model/astrm.py)
            ├─ Local Spatial   F_s : CBAM-style, Conv7x7 → sigmoid          (eq.2)
            ├─ Local Temporal  F_t : Conv3x1x1→BN/ReLU→Conv1x1x1 → sigmoid  (eq.3)
            └─ Global Temporal G_t : GAP → FC → FC → softmax dynamic kernel (eq.4)
       Ψ(x) = ((x·(1+F_s))·(1+F_t)) * G_t          (Hadamard, then temporal conv)
  └─ spatial avg-pool → single Bi-GRU temporal block
  └─ heads: BCE event head (K logits) + 128-d projection head (Soft-IC)
```

~6.4M parameters (paper reports 6.46M).

**Losses** (`L = L_BCE + 0.001 · L_SoftIC`):
- BCE over the **K event classes only** (background is implicit; no bg logit).
- **Soft-IC** (`model/soft_ic.py`): InfoNCE-style instance-contrastive loss with a
  per-event-class FIFO memory bank (feat dim 128, bank 256), foreground-only,
  mixup-aware. Background never enters the bank.

**Optimizer**: AdamW + **ASAM** (ρ=2, `model/asam.py`), cosine LR with 3-epoch
linear warmup (`SequentialLR`: warmup → cosine), mixup α=0.1.

A `--cls_loss ce` softmax variant (K+1 logits) is kept as a non-paper option.

---

## Data — HOI4D-v3

`data/hoi4d_v3/` (E2E-Spot json format):

| split | videos |
|-------|--------|
| train | 2288 |
| val   | 138 |
| test  | 424 |

- `class.txt` = `touch`, `untouch` (**K=2**; everything else is background).
- 300 frames per video @ **15 fps**, 854×480. Events sparse (~2% of frames),
  touch ≈ untouch (~1:1).
- Frames are read from `--frame_dir` as `<video>/000000.jpg ...`
  (default: `/data/chang/data2/huyanh/Workspace/dataset/hoi4d/frames`).

---

## Train

```bash
bash run_train.sh           # GPU0, paper-faithful config, into runs/astrm_hoi4d_v3/
```

or directly:

```bash
python train_astrm.py hoi4d_v3 <FRAME_DIR> \
  -m rny002_astrm --clip_len 128 --batch_size 8 \
  --num_epochs 50 --warm_up_epochs 3 -lr 1e-3 \
  --cls_loss bce --fg_weight 1 \
  --use_asam 1 --asam_rho 2 --use_soft_ic 1 --lambda_sic 0.001 \
  --mixup_alpha 0.1 --amp_dtype bf16 -s runs/astrm_hoi4d_v3
```

Conda env: `astrm` (torch 2.10+cu128, timm 1.0.27).

## Eval

Done automatically each epoch (val) and on the best checkpoint (test). Clips use a
**sliding window with 50% overlap**, scores are averaged, then sparse events are
extracted. **mAP@δ** is reported at δ = {0, 1, 2} frames.

NMS is **optional post-processing** applied only at eval — the metric is reported
three ways and the best epoch is chosen by `--nms_mode`:

| mode | meaning |
|------|---------|
| `none` | raw dense high-recall predictions |
| `hard` | hard NMS, window `--nms_window` |
| `soft` | Gaussian Soft-NMS, window `--nms_window` (default selection) |

Saved per-video predictions (`pred-*.recall.json.gz`) are the **pre-NMS**
high-recall set with scores, so any NMS can be re-applied offline.

---

## Faithfulness notes

**Matches the paper**: ASTRM module (eq.1–4, incl. softmax dynamic kernel), Bi-GRU
temporal block, BCE + Soft-IC (λ=0.001, feat 128, bank 256), ASAM ρ=2, mixup
α=0.1, augmentations (blur k5 p0.25, brightness/contrast/saturation [0.7,1.2]
p0.5), clip_len 128, total batch 8, 3-epoch warmup (init 1e-5), RegNetY-200MF.

**Left to sensible defaults (paper does not specify a value)**: Global-Temporal
kernel length K (= clip_len; `--astrm_temporal_kernel`), the inter-FC activation
(ReLU, matching TAM), AdamW weight decay (0.01 default), epochs / clips-per-epoch
(`--num_epochs`, `--epoch_num_frames`), crop (224), NMS window/type.

**Not a paper experiment**: the paper evaluates on SoccerNet-V2 / Tennis / Figure
Skating / FineGym. Here the method is applied to **HOI4D-v3**, so the paper's
reported numbers are not directly reproducible — this is a faithful *method*
re-implementation plus a HOI4D adaptation, not a reproduction of the paper's
benchmarks.
```
