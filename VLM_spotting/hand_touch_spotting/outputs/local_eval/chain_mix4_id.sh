#!/bin/bash
# After MIX-3 ID frees the GPU, run MIX-4 ID (priority pair on the reliable 5090).
cd /home/chang/Project/VLM_spotting/hand_touch_spotting/outputs/local_eval
gpu_busy(){ [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; }
while gpu_busy; do sleep 60; done
echo "### MIX-4 ID start $(date +%H:%M) ###"
bash run_indom.sh conn_MIX4_qf750_out1200.pt 2>&1 | grep -E "eval:in:" > res2x2_MIX4_qf750_out1200_indom.txt
echo "### MIX-4 ID DONE: $(grep eval:in:_all res2x2_MIX4_qf750_out1200_indom.txt) ###"
