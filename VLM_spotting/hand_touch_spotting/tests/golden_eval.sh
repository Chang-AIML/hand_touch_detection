#!/bin/bash
# Tier-3 golden eval: run the REAL entry (dpc/train_mixed.py --eval_only) on a fixed connector
# (conn_s750) over a tiny FIXED in-domain + OOD window set, greedy (RNG-free), seed 0.
# Emits the per-dataset `[eval:...] mAP@(...)=[...] ndet=.. ngt=..` lines — the numbers that
# must stay identical across the refactor. Usage:
#   tests/golden_eval.sh capture   -> tests/golden/golden.txt   (run on CURRENT code, once)
#   tests/golden_eval.sh check     -> tests/golden/after.txt + diff vs golden.txt (must be empty)
set -uo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-check}"
CKPT="${2:-outputs/local_eval/conn_s750.pt}"
OUT=tests/golden; mkdir -p "$OUT"
export VJEPA_REPO=/home/chang/Project/vlm_deps/vjepa2
export VJEPA_CKPT=/home/chang/Project/vlm_deps/vjepa2_1_vitb_dist_vitG_384.pt
export QWEN_PATH=/home/chang/models/Qwen3-VL-8B-Instruct
export AS_DATA_ROOT=/home/chang/Dataset/Action_Spotting
export AS_ANN_DIR=/home/chang/Project/vlm_deps/annotations_dpc
export HF_HOME=/home/chang/.cache/huggingface
export AS_OUT=/home/chang/Project/VLM_spotting/hand_touch_spotting/tests/golden/runs
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0   # touchmoment eval questions are hash-seeded -> pin for a stable golden
PY=/home/chang/miniconda3/bin/python3
run(){ $PY dpc/train_mixed.py --run_name golden_eval --wandb_mode offline \
    --init_fc "$CKPT" --fsdp 0 --eval_only 1 --seed 0 \
    --local_frames /home/chang/Dataset/Action_Spotting \
    --window_frames 600 --batch_size 1 --n_q 8 --stain 0 --use_text 0 --gate_lang 0 \
    --use_anchor 1 --anchor_stride 5 --anchor_max_side 252 \
    --vjepa_bf16 1 --eval_max_tokens 64 --tols "0,1,2,4,8,16" \
    --eval_balance 4 --eval_balance_finegym 4 --eval_balance_ood 16 --eval_windows 4000 \
    2>&1 | grep -E "eval:in:|eval:ood:"; }
if [ "$MODE" = capture ]; then
  run | tee "$OUT/golden.txt"
  echo "== captured golden -> $OUT/golden.txt =="
else
  run | tee "$OUT/after.txt"
  echo "== diff (empty = PASS) =="
  if diff -u "$OUT/golden.txt" "$OUT/after.txt"; then echo "GOLDEN PASS: eval numbers identical"; else echo "GOLDEN FAIL: numbers changed"; exit 1; fi
fi
