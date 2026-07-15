#!/bin/bash
# Sequentially eval IN-DOMAIN (finegym 640 / others 200, tols 0-16) for s900..s1500.
# s900/s1050 already have OOD -> in-domain only. s1200+ -> full (in-domain + OOD).
# GPU-free gate via nvidia-smi (not pgrep -> no self-match).
cd /home/chang/Project/VLM_spotting/hand_touch_spotting
export PATH="$HOME/.krew/bin:$PATH"
NS=chang-dong
DST=outputs/local_eval
getpod(){ kubectl get pods -n "$NS" -l job-name=dpc-train-scratch -o jsonpath='{.items[0].metadata.name}' 2>/dev/null; }
gpu_busy(){ [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; }

for S in 900 1050 1200 1350 1500; do
  CK="$DST/conn_s${S}.pt"
  if [ ! -f "$CK" ]; then
    until POD=$(getpod); [ -n "$POD" ] && kubectl exec -n "$NS" "$POD" -- test -f "/data/runs/mixed/p2_slide_a70_conn_s${S}.pt" 2>/dev/null; do sleep 60; done
    kubectl cp -n "$NS" "$POD:/data/runs/mixed/p2_slide_a70_conn_s${S}.pt" "$CK" 2>/dev/null
  fi
  while gpu_busy; do sleep 20; done
  echo "### s${S} in-domain start $(date +%H:%M) ###"
  if [ "$S" -le 1050 ]; then
    bash "$DST/run_indom.sh" "$CK" 2>&1 | grep -E "eval:in:|Error|Traceback" > "$DST/res_s${S}_indom.txt"
  else
    bash "$DST/run_rigorous.sh" "$CK" 2>&1 | grep -E "eval:in:|eval:ood:|Error|Traceback" > "$DST/res_s${S}_indom.txt"
  fi
  echo "### s${S} DONE: $(grep 'eval:in:_all' "$DST/res_s${S}_indom.txt") ###"
done
echo "=== ALL IN-DOMAIN DONE ==="
