#!/bin/bash -i
# TSP on HOI4D-touch, ALIGNED TO E2E-Spot paper sec. B.3:
#   - R(2+1)D-34 backbone, Kinetics init
#   - clip length = 12
#   - GVF = pre-extracted MViT-B (16x4, Kinetics) features (768-d)   [run step1 first]
#   - DUAL head (TSP-native): head1 = action (touch/untouch), head2 = temporal-region
#     (Foreground/Background, GVF-fed). 2 losses, alpha 1.0 : 1.0.
# Reads JPG frames directly (no ffmpeg/mp4). Run from the project's train/ dir.

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"           # .../touch_tsp/train
PROJ="$(dirname "$HERE")"                        # .../touch_tsp

# paths come from config.py (env-overridable) so this stays in sync with the python steps
eval "$(python - <<PY
import sys; sys.path.insert(0, "$PROJ"); import config as c
print(f'FRAMES_DIR={c.FRAMES_DIR}')
print(f'TRAIN_CSV={c.TRAIN_CSV}')
print(f'VAL_CSV={c.VAL_CSV}')
print(f'GVF={c.GVF_PATH}')
print(f'TR_MAP={c.TR_LABEL_MAP}')
print(f'ACTION_MAP={c.ACTION_LABEL_MAP}')
print(f'OUTDIR={c.TRAIN_OUT}')
print(f'CLIP_LEN={c.CLIP_LEN}')
print(f'FRAME_RATE={c.FRAME_RATE}')
PY
)"

# ===== machine knobs =====
NUM_GPUS=${NUM_GPUS:-1}
DOWNSCALE_FACTOR=${DOWNSCALE_FACTOR:-1}          # 32G->1, 16G->2, 8G->4
# =========================

# head order MUST be [action, temporal-region]: Model puts GVF on the 2nd head (fc2).
LABEL_COLUMNS="action-label temporal-region-label"   # head1=action(touch/untouch), head2=region+GVF
LABEL_MAPPING_JSONS="$ACTION_MAP $TR_MAP"
LOSS_ALPHAS="1.0 1.0"
BACKBONE=r2plus1d_34
BATCH_SIZE=16
BACKBONE_LR=0.0001
FC_LR=0.004
CLIPS_PER_SEGMENT=5
EPOCHS=8

BATCH_SIZE=$(bc <<< $BATCH_SIZE/$DOWNSCALE_FACTOR)
BACKBONE_LR=$(bc -l <<< $BACKBONE_LR/$DOWNSCALE_FACTOR)
FC_LR=$(bc -l <<< $FC_LR/$DOWNSCALE_FACTOR)

source activate ${CONDA_ENV:-base}
mkdir -p "$OUTDIR"
export OMP_NUM_THREADS=6
cd "$HERE"

CMD="train.py \
--frames-dir $FRAMES_DIR \
--train-csv-filename $TRAIN_CSV \
--valid-csv-filename $VAL_CSV \
--label-mapping-jsons $LABEL_MAPPING_JSONS \
--label-columns $LABEL_COLUMNS \
--loss-alphas $LOSS_ALPHAS \
--global-video-features $GVF \
--backbone $BACKBONE \
--clip-len $CLIP_LEN --frame-rate $FRAME_RATE \
--clips-per-segment $CLIPS_PER_SEGMENT \
--epochs $EPOCHS \
--batch-size $BATCH_SIZE \
--backbone-lr $BACKBONE_LR --fc-lr $FC_LR \
--output-dir $OUTDIR"

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node=$NUM_GPUS --master_port $(shuf -i 30000-60000 -n 1) $CMD
else
    python $CMD
fi
