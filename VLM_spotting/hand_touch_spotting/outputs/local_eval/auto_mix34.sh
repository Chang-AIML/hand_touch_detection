#!/bin/bash
# After the GPU frees (MIX-ID chain done), run OOD (finediving full 1018, tols 0-16) for the finer
# swaps MIX3 (s750 out + s1200 rest) and MIX4 (s750 a3/Q-Former + s1200 rest).
cd /home/chang/Project/VLM_spotting/hand_touch_spotting
DST=outputs/local_eval
gpu_busy(){ [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; }
while gpu_busy; do sleep 120; done
for M in MIX3_out750_rest1200 MIX4_a3750_rest1200; do
  echo "### $M OOD start $(date +%H:%M) ###"
  bash "$DST/run_ood.sh" "$DST/conn_$M.pt" 2>&1 \
    | grep -E "eval:ood:finediving|Error|Traceback" > "$DST/res_${M}_ood.txt"
  echo "### $M OOD: $(cat "$DST/res_${M}_ood.txt") ###"
done
echo "=== MIX3/MIX4 OOD DONE ==="
