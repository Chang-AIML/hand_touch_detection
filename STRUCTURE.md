# Repository structure

Four spotting **methods** share one **evaluation library** and one **dataset**.
Training is per-method (different paradigms); everything after predictions
(NMS, Soft-NMS, mAP) is unified in `common/` so all methods are scored identically.

```
hand_touch_detection/
├── config.py            # central paths + hyperparams (TOUCH_* env vars override)
├── common/              # SHARED library — imported by every method
├── data/                # HOI4D-v3 labels + TSP segment CSVs (single source of truth)
├── methods/             # the four methods (see below)
├── scripts/             # cross-method drivers (unified eval + pipeline runners)
└── outputs/             # gitignored, regenerable; only outputs/spot_head/best/ is tracked
```

## `common/` — the shared library

The E2E-Spot–derived evaluation infra, plus TSP training helpers. Import as
`from common.<module> import ...`.

| module | purpose |
|---|---|
| `common/eval.py` | `non_maximum_supression` (hice greedy), `soft_non_maximum_supression` (hice parabolic `(d/w)²`), `ForegroundF1`, `ErrorStat`, `process_frame_predictions` |
| `common/score.py` | `compute_mAPs` / `compute_average_precision` (frame-tolerance mAP) |
| `common/io.py` | json / gz-json / pickle / text helpers |
| `common/spot_dataset.py` | `load_classes`, `DATASETS`, `read_fps`, `get_num_frames` |
| `common/transforms.py` `scheduler.py` `utils.py` | TSP training helpers (R(2+1)D transforms, LR schedule, dist utils) |

> All methods evaluate with **these** functions — so TSP-MSTCN / TSP-ASFormer /
> V-JEPA-MSTCN / ASTRM numbers are directly comparable (hice NMS `w=1`, Soft-NMS `w=4`).

## `methods/`

| method | paradigm | entrypoints |
|---|---|---|
| `methods/tsp/` | **Stage A** — pretrain R(2+1)D-34 (dual-head + GVF), extract dense `[N,512]` per-frame features | `train.py` / `train_tsp_on_hoi4d.sh`, then `step0/1/3/4_*.py` |
| `methods/spot_head/` | **Stage B** — MS-TCN / ASFormer / GRU / GCN heads on per-frame features (TSP **or** V-JEPA) | `train_head.py -m <arch> --feat_dir <...>`, `eval_nms.py` |
| `methods/vjepa/` | V-JEPA 2.1 per-frame feature extraction + adapter (alt features that feed Stage B) | `extract_*.py`, `adapters/vjepa_to_features.py`, `run_vjepa_mstcn.sh` |
| `methods/astrm/` | **end-to-end** RGB spotter (RegNetY-200MF + ASTRM module + Bi-GRU + BCE/Soft-IC) | `train_astrm.py`, `eval.py`, `run_train.sh` |

Each method keeps its own `model/` and `dataset/` (method-specific; import names
`model` / `dataset` / `models` resolve to the method's own dir via `sys.path`).
`methods/astrm/data/hoi4d_v3` is a symlink to `data/HOI4D-v3` (shared labels).

## `scripts/`

| file | purpose |
|---|---|
| `step6_eval_nms.py` | unified test **mAP@{0,1,2,4} × {none, NMS, SoftNMS}** for the spot_head heads |
| `run_full_pipeline.sh` | TSP stages 1→5 end to end |
| `run_stage6_after_spot_head.sh` | waits for spot_head preds, then runs `step6_eval_nms.py` |

## Imports & paths (how it stays wired after the move)

- **Shared code**: `from common.<mod> import ...` — every entrypoint puts the repo
  root on `sys.path`.
- **Method-local code**: `from model...`, `from dataset...`, `from models...` resolve
  to the running script's own directory (`sys.path[0]`).
- **Central config**: `import config` (repo-root `config.py`); `config.add_proj_to_path()`
  adds `methods/tsp/` for the TSP `models`/dataset modules.

## `outputs/` (gitignored)

Regenerable: TSP checkpoints (`*.pth`), features (`*.npy`), per-epoch predictions,
logs, ASTRM `runs/`. **Only `outputs/spot_head/best/`** (curated raw test predictions
+ comparison tables) is version-controlled. Trained weights are backed up separately
outside the repo.
