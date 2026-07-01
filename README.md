# hand_touch_detection

Self-contained pipeline to pretrain a **TSP** R(2+1)D-34 encoder on the HOI4D-touch
dataset and extract **per-frame features** for spot_head precise-spotting heads
(MS-TCN / ASFormer / GRU / GCN). Aligned to the **E2E-Spot paper §B.3** (clip-len 12,
MViT-B GVF). Reads JPG frames directly — no mp4.

Events are point events `touch` / `untouch` (rest = background). The TSP encoder is
trained with **two heads / two losses** (TSP-native):

- **action** head — `touch` vs `untouch` (classified on Foreground clips only)
- **temporal-region** head — `Foreground` vs `Background`, fed the GVF

It also plugs in **V-JEPA 2.1** per-frame features (extracted in the sibling
`../feature_extraction` repo) through an adapter, feeding the same spot_head heads.

## Layout

```
hand_touch_detection/                 # see STRUCTURE.md for the full per-directory guide
├── config.py            # ALL paths/hyperparams (edit here or set TOUCH_* env vars)
├── requirements.txt
├── common/                          # SHARED library imported by every method
│   ├── eval.py score.py io.py spot_dataset.py   # E2E-Spot eval infra (hice NMS/Soft-NMS, mAP@tol)
│   └── transforms.py scheduler.py utils.py       # TSP training helpers (R(2+1)D transforms etc.)
├── data/
│   ├── HOI4D-v3/{train,val,test}.json + class.txt   # label splits (default LABEL_DIR)
│   ├── hoi4d_{train,val,test}_tsp.csv                # TSP dual-head segment CSVs (touch/untouch + FG/BG)
│   └── {temporal_region,action}_label_mapping.json
├── methods/
│   ├── encoders/                   # feature-extraction front-ends (parallel) — feed spot_head
│   │   ├── tsp/                     # R(2+1)D-34 dual-head + GVF -> dense [N,512] features
│   │   │   ├── train.py opts.py frame_untrimmed_video_dataset.py train_tsp_on_hoi4d.sh
│   │   │   ├── models/{model,backbone}.py
│   │   │   └── step0_make_tsp_csv.py step1_extract_mvit_gvf.py step3_select_best_f1.py step4_extract_features.py
│   │   └── vjepa/                   # V-JEPA 2.1 extraction + adapter -> [N,768] features
│   │       ├── extract_*.py frame_io.py run_dual_gpu.sh run_vjepa_mstcn.sh
│   │       └── adapters/vjepa_to_features.py
│   ├── spot_head/                  # SHARED head: features (TSP or V-JEPA) -> per-frame preds
│   │   ├── train_head.py eval_nms.py train_spot_head.sh
│   │   ├── model/{common,feature_heads}.py + model/impl/{asformer,gtad,calf}.py
│   │   └── dataset/feature_dataset.py
│   └── astrm/                       # end-to-end RGB spotter (self-contained front+back)
│       ├── train_astrm.py eval.py run_train.sh  model/  dataset/
│       └── data/hoi4d_v3 -> ../../../data/HOI4D-v3
├── scripts/                        # cross-method drivers
│   ├── step6_eval_nms.py           # unified test mAP@{0,1,2,4} x {none,NMS,SoftNMS}
│   ├── run_full_pipeline.sh        # TSP encoder -> spot_head, stages 1-5
│   └── run_stage6_after_spot_head.sh
└── outputs/                        # mirrors methods/: encoders/{tsp,vjepa}, spot_head, astrm
    ├── encoders/{tsp,vjepa}/       # features + checkpoints + gvf (gitignored, regenerable)
    ├── spot_head/                  # head runs + preds; only spot_head/best/ tracked
    └── astrm/                      # astrm run
```

## External input (NOT shipped — set in config.py)

- `FRAMES_DIR` — JPG frames, one dir per video: `<FRAMES_DIR>/<video>/000000.jpg ...`

`LABEL_DIR` defaults to the in-repo `data/HOI4D-v3/` (splits + `class.txt`); outputs default
to `outputs/` (override via `TOUCH_OUT_DIR` etc.).

## Pipeline A — TSP pretrain + feature extraction

```bash
# (0) regenerate the dual-head segment CSVs from HOI4D json   [optional; CSVs already in data/]
python methods/encoders/tsp/step0_make_tsp_csv.py

# (1) extract MViT-B GVF  -> outputs/encoders/tsp/gvf/mvit_v1_b-max_gvf.h5
python methods/encoders/tsp/step1_extract_mvit_gvf.py

# (2) train TSP (clip-12, DUAL head: action + GVF-fed region)
bash methods/encoders/tsp/train_tsp_on_hoi4d.sh          # -> outputs/encoders/tsp/train/

# (3) pick best checkpoint by Foreground-F1 (region head; classes imbalanced)
python methods/encoders/tsp/step3_select_best_f1.py    # writes best_by_f1.json in the train output dir

# (4) extract dense per-frame features  -> outputs/encoders/tsp/features/<video>.npy  [T,512]
python methods/encoders/tsp/step4_extract_features.py --ckpt <best epoch_*.pth>
```

## Pipeline C — V-JEPA 2.1 features

V-JEPA is extracted in the sibling repo `../feature_extraction` (tubelet=2 → two
half-rate `even`/`odd` streams per clip). The adapter merges them into frame-aligned
per-video arrays the spot_head consumes:

```bash
python methods/encoders/vjepa/adapters/vjepa_to_features.py \
    --raw-dir   ../feature_extraction/VJEPA_feature \
    --out-dir   outputs/encoders/vjepa/features \
    --label-dir data/HOI4D-v3 \
    --mode      interleave        # or: even | odd (interpolated) | stack ([N,2,D])
```

## Pipeline B — spot_head spotting heads (MS-TCN / ASFormer)

Self-contained — head models, `FeatureDataset`, and mAP eval are **vendored** under
`common/` + `methods/spot_head/`. The feature dim is auto-detected, so the same code trains on TSP
(512-d) or V-JEPA (768-d) features — just point `--feat_dir` at the right directory.

```bash
# MS-TCN then ASFormer, evals mAP @ delta=[0,1,2,4]  -> outputs/spot_head/{mstcn,asformer}/
bash methods/spot_head/train_spot_head.sh
# or one arch on a chosen feature set:
python methods/spot_head/train_head.py -m mstcn    --feat_dir outputs/encoders/tsp/features
python methods/spot_head/train_head.py -m asformer --feat_dir outputs/encoders/vjepa/features
```

- Spotting classes come from `class.txt` (`touch` / `untouch`); +1 implicit background.
- Reads `class.txt` + `{train,val,test}.json` from `config.LABEL_DIR`. Saves
  `best_epoch.pt`, `loss.json`, `pred-{val,test}.*` under `--save_dir`.

## Notes

- **GVF (768-d)** feeds the region head during *training only*; it is NOT part of the
  extracted spot_head feature. TSP spot_head feature = R(2+1)D-34 backbone = **512-d**.
- **TSP feature extraction is dense stride-1, window=12** (= training clip-len): one
  feature per frame, `[num_frames, 512]`, frame-aligned (the spotting loader pads).
- **V-JEPA per-frame**: tubelet=2 halves the temporal rate, so we run two 1-frame-offset
  passes (`even`/`odd`) and interleave → one feature per frame `[num_frames, 768]`.
- **Select by Foreground-F1**, not accuracy (FG:BG imbalance hides recall).
- `torch.load(..., weights_only=False)` is required (checkpoints embed argparse args).
