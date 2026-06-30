# touch_tsp

Self-contained pipeline to train a **TSP** R(2+1)D-34 encoder on the HOI4D-touch
dataset and extract **per-frame 512-d features** for a downstream precise-spotting
head (MS-TCN / GRU / ASFormer). Aligned to the **E2E-Spot paper §B.3** (clip-len 12,
MViT-B GVF, single GVF-fed temporal-region head). Reads JPG frames directly — no mp4.

## Layout

```
touch_tsp/
├── config.py            # ALL paths/hyperparams (edit here or set env vars)
├── requirements.txt
├── common/  models/     # vendored TSP code (transforms, R(2+1)D-34, Model w/ flexible GVF)
├── train/
│   ├── train.py opts.py                         # TSP trainer (--frames-dir reads JPGs)
│   ├── frame_untrimmed_video_dataset.py         # JPG clip dataset
│   ├── untrimmed_video_dataset.py               # (mp4 variant, unused)
│   └── train_tsp_on_hoi4d.sh                     # launch script (pulls paths from config.py)
├── scripts/
│   ├── step0_make_tsp_csv.py        # HOI4D json -> TSP segment CSVs (contact / background)
│   ├── step1_extract_mvit_gvf.py    # MViT-B (16x4, Kinetics) GVF -> 768-d/video h5
│   ├── step3_select_best_f1.py      # eval every epoch on val, pick best Foreground-F1
│   └── step4_extract_features.py    # dense stride-1, window-12 -> [T,512] npy per video
├── downstream/                      # spotting heads on TSP features (self-contained)
│   ├── train_head.py                # MS-TCN / ASFormer / GRU / GCN trainer + mAP eval
│   ├── train_downstream.sh          # trains mstcn + asformer
│   └── lib/                         # vendored model/ dataset/ util/ (from spot, np2-patched)
└── data/
    ├── hoi4d_{train,val,test}_tsp.csv
    ├── temporal_region_label_mapping.json   # {"Background":0,"Foreground":1}
    └── action_label_mapping.json            # {"contact":0}
```

## External inputs (NOT shipped — set in config.py)

- `FRAMES_DIR` — JPG frames, one dir per video: `<FRAMES_DIR>/<video>/000000.jpg ...`
- `LABEL_DIR`  — holds the E2E-Spot splits `train.json` / `val.json` / `test.json`
  (only needed for step0 and the video list in step1/step4)

Outputs default to `touch_tsp/outputs/` (override via `TOUCH_OUT_DIR` etc.).

## Run order

```bash
conda activate base          # torch>=2.6, torchvision, h5py, pillow, pandas, numpy

# (0) regenerate the segment CSVs from HOI4D json   [optional; CSVs already in data/]
python scripts/step0_make_tsp_csv.py

# (1) extract MViT-B GVF  -> outputs/global_video_features/mvit_v1_b-max_gvf.h5
python scripts/step1_extract_mvit_gvf.py

# (2) train TSP (clip-12, single head)  -> outputs/r2plus1d_34-tsp_on_hoi4d-mvitgvf_clip12/
bash train/train_tsp_on_hoi4d.sh          # ~1.5h on a 32G GPU, 8 epochs

# (3) pick best checkpoint by Foreground-F1 (NOT accuracy; classes imbalanced)
python scripts/step3_select_best_f1.py    # writes best_by_f1.json in the train output dir

# (4) extract dense per-frame features  -> outputs/TSP_features/<video>.npy  [T,512]
python scripts/step4_extract_features.py --ckpt <best epoch_*.pth>
```

## Downstream: precise-spotting heads (MS-TCN / ASFormer)

Self-contained — the head models, FeatureDataset, and mAP eval are **vendored**
under `downstream/lib/` (from the spot repo, numpy-2 patched). No external spot
repo needed.

```bash
# trains MS-TCN then ASFormer on the TSP [T,512] features, evals mAP @ delta=[0,1,2,4]
bash downstream/train_downstream.sh                 # -> outputs/downstream/{mstcn,asformer}/
# or one arch / custom:
python downstream/train_head.py -m mstcn --num_epochs 50 --clip_len 100
python downstream/train_head.py -m asformer
```

- Spotting classes = the event labels in the json (`touch` / `untouch`); +1 background.
- Reads `class.txt` + `{train,val,test}.json` from `config.LABEL_DIR`, features from
  `config.FEATURES_OUT`. Saves `best_epoch.pt`, `loss.json`, `pred-{val,test}.*`.
- `downstream/prepare_spot_dataset.py` is an OPTIONAL alternative that instead drives a
  real external spot repo's `baseline.py` (set `TOUCH_SPOT_REPO`); not needed for the
  vendored path above.
- `downstream/lib/` contains only what the feature heads need (model/{common,feature},
  model/impl/{asformer,gtad,calf}, dataset/feature, util/{io,eval,dataset,score}).

## Notes

- **GVF (768-d)** feeds the head during *training only*; it is NOT part of the
  extracted downstream feature. Downstream feature = R(2+1)D-34 backbone = **512-d**.
- **Feature extraction is dense stride-1, window=12** (= training clip-len), i.e.
  overlapping windows / one feature per frame — exactly E2E-Spot §3.1/§B.2. Output is
  `[num_frames, 512]`, frame-aligned, no padding baked in (the spotting loader pads).
- **Select by Foreground-F1**, not accuracy: FG:BG ≈ 1:1.5, so accuracy hides recall.
- `torch.load(..., weights_only=False)` is required (checkpoints embed argparse args).
- The reference run selected **epoch 4** (FG-F1 0.9768, P 0.9712, R 0.9825).

## Provenance

Vendored from the modified TSP repo at `/home/huyanh/Workspace/repos/TSP`
(adds `--frames-dir` JPG dataset + flexible `gvf_size` for the 768-d MViT GVF).
See `/home/huyanh/Workspace/dataset/hoi4d/CLAUDE.md` for full project context.
