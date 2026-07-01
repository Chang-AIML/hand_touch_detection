#!/bin/bash
# V-JEPA 2.1 ViT-B (lightest) per-frame feature extraction, one GPU per offset stream.
#   GPU0 -> even pass (frames 0,2,4,...)  -> <clip>_even.npy
#   GPU1 -> odd  pass (frames 1,3,5,...)  -> <clip>_odd.npy
# Each card processes ALL clips for its stream. Temporal-only (N,768) features.
set -e

PY=/data/dong/miniconda3/envs/vjepa21/bin/python
HERE=/data/dong/project/Workspace/repos/feature_extraction
FRAMES=/data/dong/project/Workspace/dataset/hoi4d/frames
OUT=$HERE/VJEPA_feature
MODEL=${MODEL:-base}
BW=${BW:-48}            # cross-clip GPU batch (windows); sized to free VRAM on a shared card
WORKERS=${WORKERS:-6}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $OUT $HERE/logs

cd $HERE
CUDA_VISIBLE_DEVICES=0 nohup $PY extract_vjepa21.py --model $MODEL --pass even \
    --frames-dir $FRAMES --out-dir $OUT --batch-windows $BW --workers $WORKERS \
    > $HERE/logs/even_gpu0.log 2>&1 &
echo "even (GPU0) PID $!"

CUDA_VISIBLE_DEVICES=1 nohup $PY extract_vjepa21.py --model $MODEL --pass odd \
    --frames-dir $FRAMES --out-dir $OUT --batch-windows $BW --workers $WORKERS \
    > $HERE/logs/odd_gpu1.log 2>&1 &
echo "odd  (GPU1) PID $!"
echo "logs: $HERE/logs/{even_gpu0,odd_gpu1}.log"
