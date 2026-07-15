#!/bin/bash
cd /home/chang/Project/VLM_spotting/hand_touch_spotting
export PATH="$HOME/.krew/bin:$PATH"
NS=chang-dong; DST=outputs/local_eval
getpod(){ kubectl get pods -n $NS -l job-name=dpc-train-scratch -o jsonpath='{.items[0].metadata.name}' 2>/dev/null; }
for S in 750 900 1050 1200 1350 1500; do
  # 1) wait until DPC has this ckpt (re-fetch pod each poll; pod may churn)
  until POD=$(getpod); [ -n "$POD" ] && kubectl exec -n $NS "$POD" -- test -f /data/runs/mixed/p2_slide_a70_conn_s${S}.pt 2>/dev/null; do sleep 60; done
  # 2) pull it (skip if already local)
  [ -f "$DST/conn_s${S}.pt" ] || kubectl cp -n $NS "$POD:/data/runs/mixed/p2_slide_a70_conn_s${S}.pt" "$DST/conn_s${S}.pt" 2>/dev/null
  # 3) wait for the 5090 to be free
  while pgrep -f "train_mixed.py.*eval_only" >/dev/null 2>&1; do sleep 20; done
  echo "### s${S} OOD-fast start $(date +%H:%M) ###"
  bash $DST/run_ood.sh $DST/conn_s${S}.pt 2>&1 | grep -E "eval:ood:finediving|Error|Traceback" > $DST/res_s${S}_ood.txt
  echo "### s${S} DONE: $(cat $DST/res_s${S}_ood.txt) ###"
done
echo "=== ALL OOD-FAST DONE ==="
