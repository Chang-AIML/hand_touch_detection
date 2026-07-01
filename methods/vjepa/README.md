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

## Paths to set on a new machine (top of `extract_vjepa21.py`)

- `VJEPA_REPO` → the `facebookresearch/vjepa2` clone (provides `src.hub.backbones`).
- `CKPT_DIR`   → dir with `vjepa2_1_vit{b,l}_dist_vitG_384.pt` (download from
  `dl.fbaipublicfiles.com/vjepa2/…`; NOT in git — they are weights).

## Run

```bash
# conda env: dev/ifpv/kv/vjepa21 (torch 2.10 + transformers + timm)
GPU pair -> even/odd streams:
bash vjepa_extraction/run_dual_gpu.sh        # writes <clip>_even.npy / <clip>_odd.npy
```

Output raw features (`VJEPA_feature/`, ~1.6 GB) and the checkpoints (`ckpts/`, ~6.4 GB) are
**NOT tracked** (regenerable / weights). This directory is the version-controlled *code*; run it
against whatever data path you set.
