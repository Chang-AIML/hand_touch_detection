#!/bin/bash
# V-JEPA 2.1 per-frame feature extraction, one GPU per offset stream.
#   GPU0 -> even pass (frames 0,2,4,...)  -> <clip>_even.npy
#   GPU1 -> odd  pass (frames 1,3,5,...)  -> <clip>_odd.npy
# Each card processes ALL clips for its stream (dataset-agnostic: keyed by frame
# folders, resumable via already_done). Temporal-only features (N,768) for ViT-B.
#
# Needs the V-JEPA 2.1 conda env + local vjepa2 repo + checkpoints (set via env below).
# Runs extract_vjepa21.py straight from this repo checkout.
#   PY=/path/to/envs/vjepa21/bin/python \
#   TOUCH_FRAMES_DIR=/path/to/frames  VJEPA_RAW=/path/to/VJEPA_feature \
#   TOUCH_VJEPA_REPO=/path/to/vjepa2  TOUCH_VJEPA_CKPT_DIR=/path/to/ckpts \
#     bash methods/encoders/vjepa/run_dual_gpu.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"                          # methods/encoders/vjepa
PY=${PY:-python}
FRAMES=${TOUCH_FRAMES_DIR:-/data/dong/project/Workspace/dataset/hoi4d/frames}
OUT=${VJEPA_RAW:-/data/dong/project/Workspace/repos/feature_extraction/VJEPA_feature}
MODEL=${MODEL:-base}
BW=${BW:-48}            # cross-clip GPU batch (windows); sized to free VRAM on a shared card
WORKERS=${WORKERS:-6}
GPU_EVEN=${GPU_EVEN:-0}
GPU_ODD=${GPU_ODD:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$OUT/logs"; mkdir -p "$OUT" "$LOG"
cd "$HERE"
echo "== V-JEPA 2.1 extract | model=$MODEL | frames=$FRAMES | out=$OUT =="

CUDA_VISIBLE_DEVICES=$GPU_EVEN nohup $PY extract_vjepa21.py --model $MODEL --pass even \
    --frames-dir "$FRAMES" --out-dir "$OUT" --batch-windows $BW --workers $WORKERS \
    > "$LOG/even_gpu${GPU_EVEN}.log" 2>&1 &
echo "even (GPU$GPU_EVEN) PID $!"

CUDA_VISIBLE_DEVICES=$GPU_ODD nohup $PY extract_vjepa21.py --model $MODEL --pass odd \
    --frames-dir "$FRAMES" --out-dir "$OUT" --batch-windows $BW --workers $WORKERS \
    > "$LOG/odd_gpu${GPU_ODD}.log" 2>&1 &
echo "odd  (GPU$GPU_ODD) PID $!"
echo "logs: $LOG/{even_gpu${GPU_EVEN},odd_gpu${GPU_ODD}}.log"
