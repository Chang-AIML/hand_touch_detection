#!/bin/bash
# Supervisor: (re)launch ASTRM training, auto-resuming from the last checkpoint
# if the process dies (e.g. transient OOM-kill while co-tenant jobs run).
set -u
cd /data/chang/data2/huyanh/Workspace/repos/astrm
PY=/home/chang_noroot/miniconda3/envs/astrm/bin/python
# Paper-faithful run on the real HOI4D-v3 dataset (2 classes: touch + untouch,
# everything else background). Fresh dir so we never resume stale checkpoints.
SD=runs/astrm_hoi4d_v3
FRAMES=/data/chang/data2/huyanh/Workspace/dataset/hoi4d/frames
# Single GPU. Measured bottleneck is the per-step GPU pipeline (per-clip
# augmentation loop + ASAM's 2x fwd/bwd), NOT data loading -- more workers did
# not speed it up and blew up RAM, so keep workers small.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# HOI4D events are ~1.5% of frames; with fg_weight=1 & dilate_len=0 the model
# collapses to all-background. dilate_len=2 (label +-2 frames per event, matches
# the eval delta) + fg_weight=5 fix the foreground/background imbalance. These
# only affect TRAINING labels; eval mAP still scores the true single event frame.
COMMON="hoi4d_v3 $FRAMES -m rny002_astrm --clip_len 128 --batch_size 16 \
  --num_epochs 50 --warm_up_epochs 3 -lr 1e-3 --mixup_alpha 0.1 --cls_loss bce \
  --use_asam 1 --asam_rho 2 --use_soft_ic 1 --lambda_sic 0.001 \
  --dilate_len 2 --fg_weight 5 --amp_dtype bf16 --start_val_epoch 0 -j 3 -s $SD"

for attempt in $(seq 1 40); do
  RESUME=""
  if ls $SD/checkpoint_*.pt >/dev/null 2>&1; then RESUME="--resume"; fi
  echo "===== [supervisor] attempt $attempt $(date) resume='$RESUME' ====="
  $PY -u train_astrm.py $COMMON $RESUME
  code=$?
  echo "===== [supervisor] exited code=$code $(date) ====="
  if [ $code -eq 0 ]; then echo "[supervisor] training finished cleanly."; break; fi
  sleep 20
done
