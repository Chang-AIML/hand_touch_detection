# hand_touch_detection

Self-contained pipeline to pretrain a **TSP** R(2+1)D-34 encoder on the HOI4D-touch
dataset and extract **per-frame features** for downstream precise-spotting heads
(MS-TCN / ASFormer / GRU / GCN). Aligned to the **E2E-Spot paper §B.3** (clip-len 12,
MViT-B GVF). Reads JPG frames directly — no mp4.

Events are point events `touch` / `untouch` (rest = background). The TSP encoder is
trained with **two heads / two losses** (TSP-native):

- **action** head — `touch` vs `untouch` (classified on Foreground clips only)
- **temporal-region** head — `Foreground` vs `Background`, fed the GVF

It also plugs in **V-JEPA 2.1** per-frame features (extracted in the sibling
`../feature_extraction` repo) through an adapter, feeding the same downstream heads.

## Layout

```
hand_touch_detection/
├── config.py            # ALL paths/hyperparams (edit here or set TOUCH_* env vars)
├── requirements.txt
├── common/  models/     # vendored TSP code (transforms, R(2+1)D-34, dual-head Model w/ GVF)
├── train/
│   ├── train.py opts.py                         # TSP trainer (requires --frames-dir; reads JPGs)
│   ├── frame_untrimmed_video_dataset.py         # JPG clip dataset
│   └── train_tsp_on_hoi4d.sh                     # step (2) launcher (pulls paths from config.py)
├── scripts/
│   ├── step0_make_tsp_csv.py        # HOI4D json -> dual-head segment CSVs (touch/untouch + FG/BG)
│   ├── step1_extract_mvit_gvf.py    # MViT-B (16x4, Kinetics) GVF -> 768-d/video h5
│   ├── step3_select_best_f1.py      # eval every epoch on val, pick best Foreground-F1
│   ├── step4_extract_features.py    # dense stride-1, window-12 -> [T,512] npy per video
│   └── adapters/
│       └── vjepa_to_features.py     # V-JEPA even/odd half-rate streams -> per-video [N,768] npy
├── downstream/                      # spotting heads on per-frame features (self-contained)
│   ├── train_head.py                # MS-TCN / ASFormer / GRU / GCN trainer + mAP eval
│   ├── train_downstream.sh          # trains mstcn + asformer
│   └── lib/                         # vendored from spot (numpy-2 patched)
│       ├── dataset/feature_dataset.py
│       ├── model/{common,feature_heads}.py  model/impl/{asformer,gtad,calf}.py
│       └── util/{io,eval,dataset,score}.py
└── data/
    ├── hoi4d_{train,val,test}_tsp.csv
    ├── temporal_region_label_mapping.json   # {"Background":0,"Foreground":1}
    ├── action_label_mapping.json            # {"touch":0,"untouch":1}
    └── HOI4D-v3/{train,val,test}.json + class.txt   # shipped label splits (default LABEL_DIR)
```

## External input (NOT shipped — set in config.py)

- `FRAMES_DIR` — JPG frames, one dir per video: `<FRAMES_DIR>/<video>/000000.jpg ...`

`LABEL_DIR` defaults to the in-repo `data/HOI4D-v3/` (splits + `class.txt`); outputs default
to `outputs/` (override via `TOUCH_OUT_DIR` etc.).

## Pipeline A — TSP pretrain + feature extraction

```bash
# (0) regenerate the dual-head segment CSVs from HOI4D json   [optional; CSVs already in data/]
python scripts/step0_make_tsp_csv.py

# (1) extract MViT-B GVF  -> outputs/global_video_features/mvit_v1_b-max_gvf.h5
python scripts/step1_extract_mvit_gvf.py

# (2) train TSP (clip-12, DUAL head: action + GVF-fed region)
bash train/train_tsp_on_hoi4d.sh          # -> outputs/r2plus1d_34-tsp_on_hoi4d-mvitgvf_clip12/

# (3) pick best checkpoint by Foreground-F1 (region head; classes imbalanced)
python scripts/step3_select_best_f1.py    # writes best_by_f1.json in the train output dir

# (4) extract dense per-frame features  -> outputs/TSP_features/<video>.npy  [T,512]
python scripts/step4_extract_features.py --ckpt <best epoch_*.pth>
```

## Pipeline C — V-JEPA 2.1 features

V-JEPA is extracted in the sibling repo `../feature_extraction` (tubelet=2 → two
half-rate `even`/`odd` streams per clip). The adapter merges them into frame-aligned
per-video arrays the downstream consumes:

```bash
python scripts/adapters/vjepa_to_features.py \
    --raw-dir   ../feature_extraction/VJEPA_feature \
    --out-dir   outputs/VJEPA_features \
    --label-dir data/HOI4D-v3 \
    --mode      interleave        # or: even | odd (interpolated) | stack ([N,2,D])
```

## Pipeline B — downstream spotting heads (MS-TCN / ASFormer)

Self-contained — head models, `FeatureDataset`, and mAP eval are **vendored** under
`downstream/lib/`. The feature dim is auto-detected, so the same code trains on TSP
(512-d) or V-JEPA (768-d) features — just point `--feat_dir` at the right directory.

```bash
# MS-TCN then ASFormer, evals mAP @ delta=[0,1,2,4]  -> outputs/downstream/{mstcn,asformer}/
bash downstream/train_downstream.sh
# or one arch on a chosen feature set:
python downstream/train_head.py -m mstcn    --feat_dir outputs/TSP_features
python downstream/train_head.py -m asformer --feat_dir outputs/VJEPA_features
```

- Spotting classes come from `class.txt` (`touch` / `untouch`); +1 implicit background.
- Reads `class.txt` + `{train,val,test}.json` from `config.LABEL_DIR`. Saves
  `best_epoch.pt`, `loss.json`, `pred-{val,test}.*` under `--save_dir`.

## Notes

- **GVF (768-d)** feeds the region head during *training only*; it is NOT part of the
  extracted downstream feature. TSP downstream feature = R(2+1)D-34 backbone = **512-d**.
- **TSP feature extraction is dense stride-1, window=12** (= training clip-len): one
  feature per frame, `[num_frames, 512]`, frame-aligned (the spotting loader pads).
- **V-JEPA per-frame**: tubelet=2 halves the temporal rate, so we run two 1-frame-offset
  passes (`even`/`odd`) and interleave → one feature per frame `[num_frames, 768]`.
- **Select by Foreground-F1**, not accuracy (FG:BG imbalance hides recall).
- `torch.load(..., weights_only=False)` is required (checkpoints embed argparse args).
