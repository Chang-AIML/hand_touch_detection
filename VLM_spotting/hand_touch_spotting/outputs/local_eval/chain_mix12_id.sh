#!/bin/bash
# After the reversal probe frees the GPU, run MIX-1 + MIX-2 in-domain ID (query-vs-adaptor column).
cd /home/chang/Project/VLM_spotting/hand_touch_spotting/outputs/local_eval
gpu_busy(){ [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; }
# wait until the reversal full run has produced its results AND the gpu is free
while [ ! -s reversal_full.log ] || gpu_busy; do sleep 60; done
for M in MIX1_q750_adap1200 MIX2_q1200_adap750; do
  echo "### $M ID start $(date +%H:%M) ###"
  bash run_indom.sh conn_$M.pt 2>&1 | grep -E "eval:in:" > res_${M}_indom.txt
  echo "### $M ID DONE: $(grep eval:in:_all res_${M}_indom.txt) ###"
done
echo "### MIX1/MIX2 ID DONE ###"
