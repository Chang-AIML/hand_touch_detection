#!/bin/bash
# Auto-trace finediving OOD (0-16 tolerances) across checkpoints as they are produced on DPC.
# GPU-free check uses nvidia-smi (NOT pgrep, to avoid self-matching this script's own text).
cd /home/chang/Project/VLM_spotting/hand_touch_spotting
export PATH="$HOME/.krew/bin:$PATH"
NS=chang-dong
DST=outputs/local_eval
getpod(){ kubectl get pods -n "$NS" -l job-name=dpc-train-scratch -o jsonpath='{.items[0].metadata.name}' 2>/dev/null; }
gpu_busy(){ [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; }

for S in 750 900 1050 1200 1350 1500; do
  CK="$DST/conn_s${S}.pt"
  if [ ! -f "$CK" ]; then
    until POD=$(getpod); [ -n "$POD" ] && kubectl exec -n "$NS" "$POD" -- test -f "/data/runs/mixed/p2_slide_a70_conn_s${S}.pt" 2>/dev/null; do sleep 60; done
    kubectl cp -n "$NS" "$POD:/data/runs/mixed/p2_slide_a70_conn_s${S}.pt" "$CK" 2>/dev/null
  fi
  while gpu_busy; do sleep 20; done
  echo "### s${S} OOD-fast start $(date +%H:%M) ###"
  bash "$DST/run_ood.sh" "$CK" 2>&1 | grep -E "eval:ood:finediving|Error|Traceback" > "$DST/res_s${S}_ood.txt"
  echo "### s${S} DONE: $(cat "$DST/res_s${S}_ood.txt") ###"
done
echo "=== ALL OOD-FAST DONE ==="
