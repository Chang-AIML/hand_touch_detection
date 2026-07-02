# hand_touch_detection

Precise **temporal spotting** of `touch` / `untouch` point-events in egocentric video
(rest = background), aligned to the **E2E-Spot** recipe. Reads pre-extracted JPG frames
directly (no mp4). Four model families are trained and scored under one evaluation library
so their numbers are directly comparable:

| model | kind | front-end → head |
|---|---|---|
| **TSP + MS-TCN** | two-stage | R(2+1)D-34 dual-head encoder → dense `[N,512]` → MS-TCN |
| **TSP + ASFormer** | two-stage | …same TSP features → ASFormer |
| **V-JEPA + MS-TCN** | two-stage | V-JEPA 2.1 `[N,768]` per-frame features → MS-TCN |
| **ASTRM** | end-to-end | RegNetY-200MF + ASTRM + Bi-GRU (RGB in, preds out) |

The TSP encoder is pretrained with **two heads** (TSP-native): an **action** head
(`touch` vs `untouch`, on Foreground clips) and a **temporal-region** head (`Foreground`
vs `Background`, fed an MViT-B global video feature / GVF, training-only).

## Datasets (per-dataset, env-selected)

Everything is parameterized by `TOUCH_DATASET` + label dir; outputs never collide
(`outputs/<DATASET>/…`). Frames are shared across datasets (keyed by video id).

| dataset | classes | fps | notes |
|---|---|---|---|
| `hoi4d` (`data/HOI4D-v3/`) | touch/untouch | 15 | default |
| `touchmoment` (`data/touchmoment/`) | touch/untouch | **mixed 15 & 30** | larger; native-frame sampling |
| `fs_perf` | 4 (figure-skating) | 25 | code-quality cross-check |

Select a dataset by exporting env vars consumed by `config.py`:

```bash
export TOUCH_DATASET=touchmoment
export TOUCH_LABEL_DIR=data/touchmoment
export TOUCH_FRAMES_DIR=/path/to/frames        # <FRAMES_DIR>/<video>/000000.jpg ...
```

> **Mixed fps** is handled by **native-frame** sampling: each clip is `clip_len` *consecutive
> native frames* of its own video (`step=1`, no fps resampling), so 15 & 30 fps videos coexist
> without a global frame-rate. For single-fps data this is identical to the old behavior.

## Layout

```
config.py              # all paths + hyperparams; TOUCH_* env vars override
common/                # SHARED lib: eval.py score.py io.py spot_dataset.py (hice NMS/Soft-NMS, mAP@tol)
                       #             transforms.py scheduler.py utils.py (TSP training helpers)
data/<dataset>/        # {train,val,test}.json + class.txt ; plus <dataset>_{split}_tsp.csv
methods/
  encoders/tsp/        # R(2+1)D-34 dual-head + GVF → dense [N,512]  (train.py, step0/1/3/4)
  encoders/vjepa/      # V-JEPA 2.1 even/odd extract + adapter → [N,768]
  spot_head/           # SHARED head: features → per-frame preds  (train_head.py, eval_nms.py)
  astrm/               # end-to-end RGB spotter (self-contained model/ + dataset/)
scripts/run_full_pipeline.sh   # TSP → spot_head, stages 1→5
outputs/<dataset>/     # encoders/{tsp,vjepa}/, spot_head/<head>/, astrm/, logs/
```

Data flow: `tsp` / `vjepa` front-ends → `spot_head` → preds; `astrm` is end-to-end. **All**
predictions are scored by `methods/spot_head/eval_nms.py` (same NMS/Soft-NMS/mAP), so every
row of the results tables is apples-to-apples.

## Workflows

### 1 & 2 — TSP train + feature extraction (→ MS-TCN / ASFormer)

```bash
# full pipeline: GVF → train TSP → select best Foreground-F1 → extract [N,512] → MS-TCN + ASFormer
TOUCH_DATASET=touchmoment TOUCH_LABEL_DIR=data/touchmoment \
TOUCH_FRAMES_DIR=/path/to/frames  GPU=0  bash scripts/run_full_pipeline.sh
```

Or step by step: `step0_make_tsp_csv.py` (dual-head CSVs) → `step1_extract_mvit_gvf.py` (GVF)
→ `train_tsp_on_hoi4d.sh` (`EPOCHS=N` overridable) → `step3_select_best_f1.py` (best epoch by
**Foreground-F1** on val) → `step4_extract_features.py --ckpt <best>` (dense stride-1, window=12
native frames, one 512-d feature per frame) → `spot_head/train_head.py -m {mstcn,asformer}`.

### 3 — V-JEPA feature extraction (→ MS-TCN)

V-JEPA 2.1 uses tubelet=2, so we run two 1-frame-offset passes and save `<clip>_even.npy` /
`<clip>_odd.npy`; the adapter merges them into frame-aligned per-video arrays.

```bash
# extract raw even/odd (needs the vjepa21 env + local vjepa2 repo + ckpts; dataset-agnostic, resumable)
PY=.../vjepa21/bin/python TOUCH_FRAMES_DIR=/path/to/frames \
VJEPA_RAW=/path/to/VJEPA_feature bash methods/encoders/vjepa/run_dual_gpu.sh

# adapt → train MS-TCN → score test mAP, under outputs/<dataset>/...
TOUCH_DATASET=touchmoment TOUCH_LABEL_DIR=data/touchmoment \
VJEPA_RAW=/path/to/VJEPA_feature GPU=0 MODES="interleave even odd stack" \
  bash methods/encoders/vjepa/run_vjepa_mstcn.sh
```

### 4 — spot_head heads (shared, on TSP **or** V-JEPA features)

Feature dim is auto-detected, so the same head trains on TSP (512-d) or V-JEPA (768-d) —
just point `--feat_dir` at the directory:

```bash
python methods/spot_head/train_head.py -m mstcn \
  --feat_dir outputs/<dataset>/encoders/tsp/features --label_dir data/<dataset> \
  --save_dir outputs/<dataset>/spot_head/mstcn
```

### 5 — ASTRM end-to-end

```bash
# HOI4D (default). Auto-resumes; batch 8 x acc_grad 2 = effective 16 (32GB OOMs at 16).
bash methods/astrm/run_train.sh
# touchmoment (data/touchmoment resolves from repo root):
ASTRM_DATASET=touchmoment ASTRM_RUN_DIR=$PWD \
ASTRM_SAVE_DIR=$PWD/outputs/touchmoment/astrm/astrm_touchmoment \
  bash methods/astrm/run_train.sh
```

## Evaluation (unified)

```bash
python methods/spot_head/eval_nms.py --prefix '' --modes mstcn asformer \
  --spot-head-dir outputs/<dataset>/spot_head --label-dir data/<dataset> \
  --per-class --tolerances 0 1 2        # mAP + per-class touch/untouch AP
```

Reports test **mAP@{0,1,2,4}** × {without-NMS, hard-NMS `w=1`, Soft-NMS `w=4`} — NMS and
Soft-NMS are the **hice `util/eval.py`** kernels (greedy hard NMS; parabolic soft-NMS
`score *= |d|²/w²`), from `common/eval.py`, used identically by every model.
Every model is selected on **val**, evaluated on **test** exactly once — no test-set peeking,
splits are disjoint, NMS windows are fixed constants (not fit on test).

## `outputs/` — what is tracked

Version-controlled: predictions (`pred-*.json`, `*.recall.json.gz`), eval tables, loss curves,
logs, `config.json`. **Ignored** (`.gitignore`, regenerable): weights `*.pt`/`*.pth`, features
`*.npy`, MViT GVF `*.h5`. Trained weights are backed up outside the repo.

## Key notes

- **GVF (768-d)** feeds the TSP region head *during training only*; the extracted TSP feature is
  the R(2+1)D-34 backbone output = **512-d**.
- **Select TSP by Foreground-F1**, not accuracy (FG:BG imbalance hides recall).
- **V-JEPA interleave** is the exact per-frame mode (even/odd merged); `even`/`odd`/`stack`
  interpolate (lossy by design).
- `torch.load(..., weights_only=False)` is required (checkpoints embed argparse args).
