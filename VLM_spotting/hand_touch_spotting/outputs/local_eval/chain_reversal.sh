#!/bin/bash
# After MIX-4 ID frees the GPU, run the temporal-reversal probe: first a 3-window validation, then
# if clean the full 60-window run. Refactored local code, s750.
cd /home/chang/Project/VLM_spotting/hand_touch_spotting
export VJEPA_REPO=/home/chang/Project/vlm_deps/vjepa2 VJEPA_CKPT=/home/chang/Project/vlm_deps/vjepa2_1_vitb_dist_vitG_384.pt
export QWEN_PATH=/home/chang/models/Qwen3-VL-8B-Instruct AS_DATA_ROOT=/home/chang/Dataset/Action_Spotting
export AS_ANN_DIR=/home/chang/Project/vlm_deps/annotations_dpc HF_HOME=/home/chang/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY=/home/chang/miniconda3/bin/python3
gpu_busy(){ [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; }
# wait until BOTH MIX-3 and MIX-4 ID are done (their result files written) and the GPU is free
while [ ! -s outputs/local_eval/res2x2_MIX4_qf750_out1200_indom.txt ] || gpu_busy; do sleep 60; done
echo "### REVERSAL validate (3 windows) $(date +%H:%M) ###"
$PY outputs/local_eval/reversal_eval.py outputs/local_eval/conn_s750.pt 3 > outputs/local_eval/reversal_validate.log 2>&1
if grep -q "RESULTS" outputs/local_eval/reversal_validate.log; then
  echo "### validate OK -> full 60-window run ###"
  $PY outputs/local_eval/reversal_eval.py outputs/local_eval/conn_s750.pt 60 > outputs/local_eval/reversal_full.log 2>&1
  echo "### REVERSAL DONE ###"; grep -A6 "RESULTS" outputs/local_eval/reversal_full.log
else
  echo "### REVERSAL VALIDATE FAILED — see reversal_validate.log ###"; tail -8 outputs/local_eval/reversal_validate.log
fi
