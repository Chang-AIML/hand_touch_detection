#!/bin/bash
cd /home/chang/Project/VLM_spotting/hand_touch_spotting
# wait until the s600 eval process is gone (frees the 5090)
while pgrep -f "train_mixed.py.*conn_s600" >/dev/null 2>&1; do sleep 30; done
echo "s600 done -> launching s450 OOD 0-16"
bash outputs/local_eval/run_ood.sh outputs/local_eval/conn_s450.pt 2>&1 \
  | grep -E "\[data\]|eval:ood:|Error|Traceback" > outputs/local_eval/res_s450_ood016.txt
echo "=== s450 OOD 0-16 DONE ==="; cat outputs/local_eval/res_s450_ood016.txt
