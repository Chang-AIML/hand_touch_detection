# Handoff — HOI4D-touch spotting (TSP + V-JEPA → MS-TCN/ASFormer)

Everything needed to reproduce the full project on another server. Three pipelines:
**A** = TSP pretrain + feature extraction, **C** = V-JEPA 2.1 feature extraction + adapter,
**B** = downstream spotting heads (shared by A and C).

Source machine paths are all under `/data/dong/project/Workspace/` (`repos/`, `dataset/`).

---

## 0. Current state (what's done vs TODO)

| Artifact | State | Location (source machine) | Size |
|---|---|---|---|
| Label splits (train/val/test.json + class.txt) | ✅ done, in-repo | `hand_touch_detection/data/HOI4D-v3/` | small |
| Dual-head TSP segment CSVs | ✅ done, in-repo | `hand_touch_detection/data/hoi4d_*_tsp.csv` | small |
| MViT-B GVF (768-d/video) | ✅ computed | `hand_touch_detection/outputs/global_video_features/mvit_v1_b-max_gvf.h5` | 9.4 MB |
| V-JEPA 2.1 raw per-frame features (even/odd) | ✅ computed (2850 vids ×2) | `feature_extraction/VJEPA_feature/` | 1.6 GB |
| V-JEPA 2.1 checkpoints (vitb 1.6G + vitl 4.8G) | ✅ downloaded | `feature_extraction/ckpts/` | 6.4 GB |
| HOI4D JPG frames (2850 H-videos + 1171 unused C) | ✅ exist | `dataset/hoi4d/frames/` | 49 GB |
| **TSP checkpoint (dual-head, 8 ep)** | ❌ **TODO — train on new server** | — | — |
| **TSP per-frame features [N,512]** | ❌ TODO (needs TSP ckpt) | — | — |
| **Downstream MS-TCN / ASFormer** | ❌ TODO | — | — |

A 1-epoch local smoke confirmed Stage 2 trains correctly: dual head
(`num_classes [2,2] num_heads 2`), both `loss_action-label` + `loss_temporal-region-label`
decreasing, steady **~35 clips/s ≈ 1 h/epoch** on one L40S (was abandoned only because that
GPU was shared with another user).

---

## 1. What to copy to the new server

Repos (git or rsync). `hand_touch_detection` is the main repo; the others are dependencies:

```
repos/hand_touch_detection/     # TSP train/extract + downstream + V-JEPA adapter (this repo)
repos/feature_extraction/       # V-JEPA extraction (extract_vjepa21.py) + frame_io.py
repos/vjepa2/                    # V-JEPA 2.1 model code (imported by extract_vjepa21.py)
```

Data / artifacts:

```
dataset/hoi4d/frames/                       # 49 GB — REQUIRED for TSP train + extract (pipeline A)
feature_extraction/VJEPA_feature/           # 1.6 GB — V-JEPA raw even/odd; copy to SKIP V-JEPA re-extraction
feature_extraction/ckpts/*.pt               # 6.4 GB — only if you re-extract V-JEPA (else skip)
hand_touch_detection/outputs/global_video_features/mvit_v1_b-max_gvf.h5   # 9.4 MB — optional (recompute in ~4 min)
```

**Minimal sets:**
- Downstream on V-JEPA only → `hand_touch_detection` + `VJEPA_feature/` (1.6 G). No frames, no big ckpts.
- Full TSP pipeline → `hand_touch_detection` + `frames/` (49 G). GVF recomputes itself.

---

## 2. Environment

Conda env (the source machine used `vjepa21`). Install torch/torchvision **matching the new
server's CUDA** first, then the rest:

```bash
conda create -n vjepa21 python=3.11 -y && conda activate vjepa21
# torch/torchvision for your CUDA (source used cu128):
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install timm transformers h5py pandas tqdm matplotlib tabulate einops beartype pillow numpy
conda install -y -c conda-forge bc        # train_tsp_on_hoi4d.sh uses `bc` (NOT in base image)
```

Versions used on source: python 3.11.15, torch 2.10.0+cu128, torchvision 0.25.0+cu128,
timm 1.0.27, transformers 4.57.3, h5py 3.16.0, pandas 3.0.1, numpy 2.4.3.

> `transformers`/`timm` are only needed for the V-JEPA path (`extract_vjepa21.py`); the TSP +
> downstream path needs just torch/torchvision/h5py/pandas/tqdm/matplotlib/tabulate/pillow/numpy + bc.

---

## 3. Paths to update on the new server

These are absolute and must be repointed (env var or edit):

- `hand_touch_detection/config.py`: `FRAMES_DIR` default → set `TOUCH_FRAMES_DIR=<new frames path>`.
  (`LABEL_DIR` already defaults to the in-repo `data/HOI4D-v3`; outputs default to `outputs/`.)
- `feature_extraction/extract_vjepa21.py`: `VJEPA_REPO` and `CKPT_DIR` constants (top of file) →
  point at the new `repos/vjepa2` and `feature_extraction/ckpts`.
- Env name: scripts call `source activate ${CONDA_ENV:-base}` and the driver hardcodes
  `/data/dong/miniconda3/...` — edit `run_full_pipeline.sh` (PY/conda paths) and run with `CONDA_ENV=<env>`.

---

## 4. Run — Pipeline A (TSP) + B (downstream)

One driver does all 5 stages; **edit the conda/repo paths at the top of `run_full_pipeline.sh` first**:

```bash
cd hand_touch_detection
export TOUCH_FRAMES_DIR=/NEW/path/to/hoi4d/frames
GPU=0 bash run_full_pipeline.sh        # tee's to outputs/pipeline_full.log
```

Stages (also runnable individually, all resumable/skip-existing):

```bash
# (0) CSVs already in data/; regenerate only if labels change:
python scripts/step0_make_tsp_csv.py
# (1) MViT-B GVF -> outputs/global_video_features/mvit_v1_b-max_gvf.h5   (~4 min, downloads MViT weights)
python scripts/step1_extract_mvit_gvf.py
# (2) train TSP DUAL-head (action touch/untouch + GVF-fed FG/BG, 8 ep)   (~1 h/epoch on a free L40S; downloads R(2+1)D ig65m init)
CONDA_ENV=vjepa21 bash train/train_tsp_on_hoi4d.sh
# (3) pick best epoch by Foreground-F1 (region head)  -> best_by_f1.json
python scripts/step3_select_best_f1.py
# (4) dense per-frame features [N,512]  -> outputs/TSP_features/<video>.npy
python scripts/step4_extract_features.py --ckpt <outputs/.../epoch_*.pth from step3>
# (B) downstream heads on TSP features  -> outputs/downstream/{mstcn,asformer}/
python downstream/train_head.py -m mstcn    --feat_dir outputs/TSP_features
python downstream/train_head.py -m asformer --feat_dir outputs/TSP_features
```

---

## 5. Run — Pipeline C (V-JEPA) + B (downstream)

If you copied `VJEPA_feature/` (the even/odd raw), **skip extraction** and go straight to the adapter:

```bash
# adapter: even/odd half-rate streams -> per-video frame-aligned [N,768]
python hand_touch_detection/scripts/adapters/vjepa_to_features.py \
    --raw-dir   ../feature_extraction/VJEPA_feature \
    --out-dir   outputs/VJEPA_features \
    --label-dir data/HOI4D-v3 \
    --mode      interleave        # | even | odd | stack  (see §6)
# downstream on V-JEPA features (feature_dim auto-detected = 768; same code as TSP)
python downstream/train_head.py -m mstcn --feat_dir outputs/VJEPA_features
```

To **re-extract** V-JEPA instead (needs vjepa2 repo + ckpts + frames; uses `dev/vjepa21` env;
fp32 due to a RoPE dtype quirk; one GPU per stream):

```bash
cd feature_extraction       # after fixing VJEPA_REPO/CKPT_DIR in extract_vjepa21.py
GPU0 even-stream, GPU1 odd-stream:  bash run_dual_gpu.sh    # ~1.5 h, model=base default
```

---

## 6. Conventions & gotchas (respect these)

- **TSP dual head** (TSP-native): head0 = action (`touch`/`untouch`, classified on Foreground
  clips only; background → label `-1`, ignored by `CrossEntropyLoss(ignore_index=-1)`),
  head1 = temporal-region (`Foreground`/`Background`), **GVF-fed**. Loss α = 1.0 : 1.0.
  Foreground segments = ±`FG_RADIUS`(=8)-frame windows centered on each event (`step0`).
- **TSP feature window (E2E-Spot §B.2/B.3)**: dense stride-1, 12-frame window, **replicate-pad**
  both ends. Frame `t`'s window = real frames `[t-6 … t+5]` (target at window position 6).
  Verified against the spec: frame 1 → `[0,0,0,0,0,0,1,2,3,4,5,6]`. Output `[N,512]`, one per frame
  (NOT downsampled). Controlled by `HALF=CLIP_LEN//2` in `step4_extract_features.py`.
- **V-JEPA per-frame**: tubelet=2 halves the temporal rate → two 1-frame-offset passes (`even`
  covers frames 0,2,4…; `odd` covers 1,3,5…). Adapter modes: `interleave` (exact per-frame, default),
  `even`/`odd` (single stream + linear interp), `stack` (`[N,2,D]`, both as augmentation versions).
- **Select TSP epoch by Foreground-F1**, not accuracy (FG:BG ≈ 1:1.5).
- `torch.load(..., weights_only=False)` is required everywhere (checkpoints embed argparse args).
- `bc` must be installed (train launcher arithmetic). `source activate base` was changed to
  `${CONDA_ENV:-base}` — set `CONDA_ENV`.
- First runs download weights (internet needed): MViT-B (~140 MB), R(2+1)D-34 ig65m (~), and
  for V-JEPA the .pt checkpoints (already in `ckpts/`).
- TSP train batch-16 R(2+1)D-34 peaks ~**30 GB**; needs a ≥32 GB GPU (or lower `DOWNSCALE_FACTOR`).

---

## 7. Repo structure (post-restructure)

`hand_touch_detection` was cleaned up (git history preserved; baseline commit `792f650`):
junk purged, dead code removed, `feature.py` modules disambiguated, dual-head step3/step4 fixed,
config self-contained. See `README.md` for the layout and per-stage details.
