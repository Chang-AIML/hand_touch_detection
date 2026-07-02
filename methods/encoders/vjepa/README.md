# V-JEPA 2.1 feature extraction (Pipeline C, stage 1)

Extracts per-frame V-JEPA 2.1 features from the HOI4D JPG frames. tubelet_size=2 halves the
temporal rate, so we run two 1-frame-offset passes (`even`/`odd`) and save both streams; the
adapter (`../scripts/adapters/vjepa_to_features.py`) then merges/interpolates them into the
per-video `[N, D]` arrays the spot_head consumes.

## Files

- `extract_vjepa21.py` — main: V-JEPA 2.1 (ViT-B/16-384, 768-d), one GPU per even/odd stream.
- `run_dual_gpu.sh` — launcher (GPU0=even, GPU1=odd). `progress.sh` — progress monitor.
- `extract_vjepa.py` — older V-JEPA 2.0 (HF transformers) variant.
- `extract_tsp.py` — R(2+1)D-34 TSP per-frame feature extractor (dense stride-1).
- `frame_io.py` — shared frame IO + pooling helpers.

## Paths to set on a new machine (env-overridable; defaults point at the Pro6000 setup)

- `TOUCH_VJEPA_REPO` → the `facebookresearch/vjepa2` clone (provides `src.hub.backbones`).
- `TOUCH_VJEPA_CKPT_DIR` → dir with `vjepa2_1_vit{b,l}_dist_vitG_384.pt` (download from
  `dl.fbaipublicfiles.com/vjepa2/…`; NOT in git — they are weights).

## Run

```bash
# conda env: vjepa21 (torch 2.10 + transformers + timm). Extraction is dataset-agnostic
# (keyed by frame folders) — extract ONCE per frame set, reuse across datasets.
PY=/path/to/envs/vjepa21/bin/python \
TOUCH_FRAMES_DIR=/path/to/frames  VJEPA_RAW=/path/to/VJEPA_feature \
TOUCH_VJEPA_REPO=/path/to/vjepa2  TOUCH_VJEPA_CKPT_DIR=/path/to/ckpts \
  bash methods/encoders/vjepa/run_dual_gpu.sh     # writes <clip>_even.npy / <clip>_odd.npy
```

## touchmoment on Pro6000 (end-to-end V-JEPA -> MS-TCN)

```bash
# 1) extract raw even/odd features (all frame folders; resumable). Skip if already done.
PY=.../vjepa21/bin/python TOUCH_FRAMES_DIR=/path/to/hoi4d/frames \
VJEPA_RAW=/path/to/VJEPA_feature bash methods/encoders/vjepa/run_dual_gpu.sh

# 2) adapt -> train MS-TCN -> score test mAP, writing under outputs/touchmoment/...
TOUCH_DATASET=touchmoment TOUCH_LABEL_DIR=data/touchmoment \
VJEPA_RAW=/path/to/VJEPA_feature GPU=0 MODES="interleave even odd stack" \
  bash methods/encoders/vjepa/run_vjepa_mstcn.sh
```

Output raw features (`VJEPA_feature/`) and checkpoints (`ckpts/`) are **NOT tracked** (regenerable /
weights). This directory is the version-controlled *code*; run it against whatever data path you set.
