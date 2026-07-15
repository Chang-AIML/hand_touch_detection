#!/bin/bash
for S in 150 300 450; do
  echo "############ EVAL conn_s${S} (balance 60) ############"
  bash outputs/local_eval/run_eval.sh outputs/local_eval/conn_s${S}.pt 60 300 2>&1 \
    | grep -E "eval:in:|eval:ood:|win  |best in-mAP|train-step"
done
echo "############ ALL DONE ############"
