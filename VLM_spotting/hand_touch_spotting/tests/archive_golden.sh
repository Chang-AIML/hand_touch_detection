#!/bin/bash
# Run the ORIGINAL (pre-refactor) code from archive/p2_pre_refactor under PYTHONHASHSEED=0,
# to produce a hash-STABLE oracle of original behavior. Writes tests/golden/golden.txt.
set -uo pipefail
cd "$(dirname "$0")/../archive/p2_pre_refactor"
export VJEPA_REPO=/home/chang/Project/vlm_deps/vjepa2
export VJEPA_CKPT=/home/chang/Project/vlm_deps/vjepa2_1_vitb_dist_vitG_384.pt
export QWEN_PATH=/home/chang/models/Qwen3-VL-8B-Instruct
export AS_DATA_ROOT=/home/chang/Dataset/Action_Spotting
export AS_ANN_DIR=/home/chang/Project/vlm_deps/annotations_dpc
export HF_HOME=/home/chang/.cache/huggingface
export AS_OUT=/home/chang/Project/VLM_spotting/hand_touch_spotting/tests/golden/runs
export WANDB_MODE=offline PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
/home/chang/miniconda3/bin/python3 dpc/train_mixed.py --run_name golden_eval --wandb_mode offline \
  --init_fc /home/chang/Project/VLM_spotting/hand_touch_spotting/outputs/local_eval/conn_s750.pt \
  --fsdp 0 --eval_only 1 --seed 0 --local_frames /home/chang/Dataset/Action_Spotting \
  --window_frames 600 --batch_size 1 --n_q 8 --stain 0 --use_text 0 --gate_lang 0 \
  --use_anchor 1 --anchor_stride 5 --anchor_max_side 252 \
  --vjepa_bf16 1 --eval_max_tokens 64 --tols "0,1,2,4,8,16" \
  --eval_balance 4 --eval_balance_finegym 4 --eval_balance_ood 16 --eval_windows 4000 \
  2>&1 | grep -E "eval:in:|eval:ood:" | tee /home/chang/Project/VLM_spotting/hand_touch_spotting/tests/golden/golden.txt
echo "== ORIGINAL(hash0) oracle -> tests/golden/golden.txt =="
