#!/bin/bash
# V-JEPA -> MS-TCN: adapt raw V-JEPA 2.1 even/odd features into per-frame feature
# sets (4 adapter modes), train an MS-TCN head per mode, then score test mAP under
# {no-NMS, NMS, Soft-NMS}. The raw features must already be extracted with
# run_dual_gpu.sh (writes <clip>_even.npy / <clip>_odd.npy into $VJEPA_RAW).
#
#   modes:  interleave (even/odd merged, exact per-frame) | even | odd | stack ([N,2,D])
#   GPUs:   sequential by default (spot_head is CPU-bound, so 4-way parallel oversubscribes
#           the CPU and is SLOWER per-job); set PARALLEL=1 to run 2-per-GPU concurrently.
#
# Dataset-parameterized (defaults to HOI4D; outputs land under outputs/<DATASET>/...):
#   TOUCH_DATASET=touchmoment TOUCH_LABEL_DIR=data/touchmoment \
#   VJEPA_RAW=/path/to/VJEPA_feature  GPU=0 MODES=interleave \
#     bash methods/encoders/vjepa/run_vjepa_mstcn.sh
set -eo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"                 # methods/encoders/vjepa -> repo root
PY=${PY:-python}
DATASET=${TOUCH_DATASET:-hoi4d}
LABEL_DIR=${TOUCH_LABEL_DIR:-data/HOI4D-v3}
RAW=${VJEPA_RAW:-$REPO/../feature_extraction/VJEPA_feature}    # dir of <clip>_even.npy / <clip>_odd.npy
GPU=${GPU:-0}
MODES=${MODES:-"interleave even odd stack"}
# MS-TCN is tiny; cap torch intra-op threads so concurrent jobs don't thrash the CPU.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
# export so config.py resolves outputs/<DATASET>/... consistently inside the Python steps.
export TOUCH_DATASET="$DATASET" TOUCH_LABEL_DIR="$LABEL_DIR"
cd "$REPO"

FEAT_ROOT="outputs/$DATASET/encoders/vjepa"                    # mirrors config.VJEPA_OUT
SPOT_ROOT="outputs/$DATASET/spot_head"                         # mirrors config.SPOT_HEAD_OUT
LOG="$FEAT_ROOT/logs"; mkdir -p "$LOG"
echo "== dataset=$DATASET | label_dir=$LABEL_DIR | raw=$RAW | modes=$MODES =="

echo "== generating feature sets (adapter, resumable) =="
for m in $MODES; do
  $PY methods/encoders/vjepa/adapters/vjepa_to_features.py --raw-dir "$RAW" \
    --out-dir "$FEAT_ROOT/feat_$m" --label-dir "$LABEL_DIR" --mode "$m" > "$LOG/adapter_$m.log" 2>&1
  echo "  adapted $m -> $FEAT_ROOT/feat_$m"
done

train(){ local gpu=$1 m=$2
  CUDA_VISIBLE_DEVICES=$gpu $PY methods/spot_head/train_head.py -m mstcn \
    --feat_dir "$FEAT_ROOT/feat_$m" --label_dir "$LABEL_DIR" \
    --save_dir "$SPOT_ROOT/vjepa_mstcn_$m" > "$LOG/mstcn_$m.log" 2>&1
  echo "  trained $m -> $SPOT_ROOT/vjepa_mstcn_$m"
}

echo "== training MS-TCN per mode =="
if [ "${PARALLEL:-0}" = "1" ]; then
  gpu_a=$GPU; gpu_b=$((GPU+1)); i=0
  for m in $MODES; do
    if [ $((i % 2)) -eq 0 ]; then train "$gpu_a" "$m" & else train "$gpu_b" "$m" & fi
    i=$((i+1))
  done
  wait
else
  for m in $MODES; do train "$GPU" "$m"; done
fi

echo "== test-set mAP under no-NMS / NMS / Soft-NMS =="
$PY methods/spot_head/eval_nms.py --modes $MODES --prefix vjepa_mstcn_ --split test \
  --label-dir "$LABEL_DIR" --spot-head-dir "$SPOT_ROOT" | tee "$LOG/results_test.txt"
