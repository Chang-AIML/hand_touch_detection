#!/bin/bash
# Supervisor: (re)launch ASTRM end-to-end training, auto-resuming from the last
# checkpoint if the process dies (e.g. transient OOM-kill). Writes to the shared
# per-dataset outputs tree: outputs/<dataset>/astrm/... (mirrors the other methods).
#
# Dataset-parameterized (defaults to HOI4D). train_astrm reads data/<DATASET>
# relative to CWD, so RUN_DIR selects where that resolves:
#   HOI4D (default):   data/hoi4d_v3 symlink lives in methods/astrm/  -> RUN_DIR=methods/astrm
#   touchmoment:       data/touchmoment lives at repo root            -> RUN_DIR=repo root
# Example touchmoment:
#   ASTRM_DATASET=touchmoment ASTRM_RUN_DIR=<repo> \
#   ASTRM_SAVE_DIR=<repo>/outputs/touchmoment/astrm/astrm_touchmoment \
#     bash methods/astrm/run_train.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"          # methods/astrm
REPO="$(cd "$HERE/../.." && pwd)"              # repo root
PY=${PY:-python}
DATASET=${ASTRM_DATASET:-hoi4d_v3}             # train_astrm 'data/<DATASET>' arg (must be in common.spot_dataset.DATASETS)
RUN_DIR=${ASTRM_RUN_DIR:-$HERE}                # cd here so data/<DATASET> resolves
SD=${ASTRM_SAVE_DIR:-$REPO/outputs/hoi4d/astrm/astrm_hoi4d_v3}
FRAMES=${TOUCH_FRAMES_DIR:-/home/huyanh/Workspace/dataset/hoi4d/frames}
BATCH=${ASTRM_BATCH:-8}; AG=${ASTRM_AG:-2}     # batch 8 x acc_grad 2 = effective 16 (32GB OOMs at 16)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$SD"; cd "$RUN_DIR"

# Events are ~1.5% of frames; with fg_weight=1 & dilate_len=0 the model collapses
# to all-background. dilate_len=2 (label +-2 frames per event, matches the eval
# delta) + fg_weight=5 fix the foreground/background imbalance. These only affect
# TRAINING labels; eval mAP still scores the true single event frame.
COMMON="$DATASET $FRAMES -m rny002_astrm --clip_len 128 --batch_size $BATCH -ag $AG \
  --num_epochs 50 --warm_up_epochs 3 -lr 1e-3 --mixup_alpha 0.1 --cls_loss bce \
  --use_asam 1 --asam_rho 2 --use_soft_ic 1 --lambda_sic 0.001 \
  --dilate_len 2 --fg_weight 5 --amp_dtype bf16 --start_val_epoch 0 -j 3 -s $SD"

for attempt in $(seq 1 40); do
  RESUME=""
  if ls "$SD"/checkpoint_*.pt >/dev/null 2>&1; then RESUME="--resume"; fi
  echo "===== [supervisor] attempt $attempt $(date) resume='$RESUME' ====="
  $PY -u "$HERE/train_astrm.py" $COMMON $RESUME
  code=$?
  echo "===== [supervisor] exited code=$code $(date) ====="
  if [ $code -eq 0 ]; then echo "[supervisor] training finished cleanly."; break; fi
  sleep 20
done
