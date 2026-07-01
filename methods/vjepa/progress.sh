#!/bin/bash
# Live progress for the V-JEPA 2.1 dual-GPU extraction.
#   ./progress.sh        one-shot snapshot
#   ./progress.sh -w     refresh every 10s (Ctrl-C to quit)
HERE=/data/dong/project/Workspace/repos/feature_extraction
TOTAL=4021

snap() {
  ne=$(ls $HERE/VJEPA_feature/*_even.npy 2>/dev/null | wc -l)
  no=$(ls $HERE/VJEPA_feature/*_odd.npy  2>/dev/null | wc -l)
  echo "== V-JEPA 2.1 extraction =="
  printf "even (GPU0): %4d/%d  %s\n" "$ne" "$TOTAL" "$(grep s/clip $HERE/logs/even_gpu0.log 2>/dev/null | tail -1 | sed 's/^ *//')"
  printf "odd  (GPU1): %4d/%d  %s\n" "$no" "$TOTAL" "$(grep s/clip $HERE/logs/odd_gpu1.log  2>/dev/null | tail -1 | sed 's/^ *//')"
  err=$(grep -iE "error|traceback|out of memory" $HERE/logs/*.log 2>/dev/null | tail -1)
  [ -n "$err" ] && echo "!! $err"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/gpu /'
  pgrep -af extract_vjepa21.py | grep -v pgrep >/dev/null && echo "status: RUNNING" || echo "status: NOT running"
}

if [ "$1" = "-w" ]; then
  while true; do clear; snap; sleep 10; done
else
  snap
fi
