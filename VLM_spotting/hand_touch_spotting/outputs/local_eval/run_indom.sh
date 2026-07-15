#!/bin/bash
cd /home/chang/Project/VLM_spotting/hand_touch_spotting
export VJEPA_REPO=/home/chang/Project/vlm_deps/vjepa2
export VJEPA_CKPT=/home/chang/Project/vlm_deps/vjepa2_1_vitb_dist_vitG_384.pt
export QWEN_PATH=/home/chang/models/Qwen3-VL-8B-Instruct
export AS_DATA_ROOT=/home/chang/Dataset/Action_Spotting
export AS_ANN_DIR=/home/chang/Project/vlm_deps/annotations_dpc
export HF_HOME=/home/chang/.cache/huggingface
export AS_OUT=/home/chang/Project/VLM_spotting/hand_touch_spotting/outputs/local_eval/runs
export WANDB_MODE=offline PYTHONUNBUFFERED=1
CKPT="$1"
/home/chang/miniconda3/bin/python3 dpc/train_mixed.py --run_name p2_rig_eval --wandb_mode offline \
  --init_fc "$CKPT" --fsdp 0 --eval_only 1 \
  --local_frames /home/chang/Dataset/Action_Spotting \
  --window_frames 600 --batch_size 1 --n_q 8 --stain 0 --use_text 0 --gate_lang 0 \
  --use_anchor 1 --anchor_stride 5 --anchor_max_side 252 \
  --vjepa_bf16 1 --eval_max_tokens 64 --tols "0,1,2,4,8,16" \
  --eval_balance 200 --eval_balance_finegym 640 --eval_balance_ood 1 --eval_windows 4000
