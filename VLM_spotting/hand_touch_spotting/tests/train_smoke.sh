#!/bin/bash
# 1-step training smoke via the root train.py shim: exercises the training loop the eval-golden does
# NOT (dataset build over all 4 datasets, loss_batch -> backward -> optim step, resume save). Single
# GPU, no FSDP. Success = it completes and writes a resume/conn checkpoint without crashing.
set -uo pipefail
cd "$(dirname "$0")/.."
export VJEPA_REPO=/home/chang/Project/vlm_deps/vjepa2
export VJEPA_CKPT=/home/chang/Project/vlm_deps/vjepa2_1_vitb_dist_vitG_384.pt
export QWEN_PATH=/home/chang/models/Qwen3-VL-8B-Instruct
export AS_DATA_ROOT=/home/chang/Dataset/Action_Spotting
export AS_ANN_DIR=/home/chang/Project/vlm_deps/annotations_dpc
export HF_HOME=/home/chang/.cache/huggingface
export AS_OUT=/home/chang/Project/VLM_spotting/hand_touch_spotting/tests/golden/runs
export WANDB_MODE=offline PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
/home/chang/miniconda3/bin/python3 train.py --run_name smoke_train --wandb_mode offline \
  --init_fc outputs/local_eval/conn_s750.pt --fsdp 0 \
  --local_frames /home/chang/Dataset/Action_Spotting \
  --balance 0 --window_frames 600 --jitter 30 \
  --negative_rate 0.15 --cross_neg_rate 0.15 --type2_rate 1.5 --neg_cap 0.40 --neg_cap_finegym 0.05 \
  --temp_alpha 0.70 --batch_size 1 --grad_accum 1 --lr 3e-4 --warmup 150 \
  --n_q 8 --stain 0 --use_text 0 --gate_lang 0 \
  --use_anchor 1 --anchor_stride 5 --anchor_max_side 252 --vjepa_bf16 1 --num_workers 0 \
  --max_steps 1 --eval_every 999 --eval_windows 0 --eval_max_tokens 64 --ckpt_secs 999999 \
  2>&1 | grep -E "step |loss|\[done\]|\[ckpt\]|\[cfg\]|Traceback|Error|assert" | tail -15
echo "=== resume ckpt written? ==="
ls -la tests/golden/runs/mixed/smoke_train_resume.pt 2>&1 && echo "TRAIN SMOKE PASS: 1 step + checkpoint OK" || echo "TRAIN SMOKE: no ckpt (check log)"
