#!/bin/bash
# V-JEPA -> MS-TCN ablation across the 4 adapter modes, then score test mAP under
# {no-NMS, NMS, Soft-NMS}. Reproduces the V-JEPA downstream results.
#
#   modes:  interleave (even/odd merged, exact per-frame) | even | odd | stack ([N,2,D])
#   env:    vjepa21 (set CONDA_ENV / PY to override)
#   GPUs:   sequential by default (downstream is CPU-bound, so 4-way parallel oversubscribes
#           the CPU and is SLOWER per-job); set PARALLEL=1 to run 2-per-GPU concurrently.
#
# Usage:  GPU=0 bash downstream/run_vjepa_mstcn.sh
#         PARALLEL=1 bash downstream/run_vjepa_mstcn.sh
set -eo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PY:-/data/dong/miniconda3/envs/vjepa21/bin/python}
RAW=${VJEPA_RAW:-$REPO/../feature_extraction/VJEPA_feature}
GPU=${GPU:-0}
cd "$REPO"
LOG="$REPO/outputs/vjepa_mstcn_logs"; mkdir -p "$LOG"
MODES="interleave even odd stack"

echo "== generating feature sets (adapter, resumable) =="
for m in $MODES; do
  $PY scripts/adapters/vjepa_to_features.py --raw-dir "$RAW" \
    --out-dir "outputs/VJEPA_feat_$m" --label-dir data/HOI4D-v3 --mode "$m" > "$LOG/adapter_$m.log" 2>&1
done

train(){ local gpu=$1 m=$2
  CUDA_VISIBLE_DEVICES=$gpu $PY downstream/train_head.py -m mstcn \
    --feat_dir "outputs/VJEPA_feat_$m" --label_dir data/HOI4D-v3 \
    --save_dir "outputs/downstream/vjepa_mstcn_$m" > "$LOG/mstcn_$m.log" 2>&1
  echo "  trained $m"
}

echo "== training MS-TCN per mode =="
if [ "${PARALLEL:-0}" = "1" ]; then
  train 1 interleave & train 0 even & wait
  train 1 odd        & train 0 stack & wait
else
  for m in $MODES; do train "$GPU" "$m"; done
fi

echo "== test-set mAP under no-NMS / NMS / Soft-NMS =="
$PY downstream/eval_nms.py --modes $MODES --split test | tee "$LOG/results_test.txt"
