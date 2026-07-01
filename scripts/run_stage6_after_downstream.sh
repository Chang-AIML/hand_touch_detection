#!/bin/bash
# Waits for Stage 5 (downstream MS-TCN + ASFormer) to finish producing their
# test predictions, then runs Stage 6 (NMS/SNMS test-mAP eval). Robust to the
# main pipeline dying: if no relevant process is alive AND the artifacts never
# appeared for several consecutive checks, it bails with an error.
REPO=/home/huyanh/Workspace/hand_touch_detection
source /home/huyanh/miniconda/etc/profile.d/conda.sh
conda activate base
export TOUCH_FRAMES_DIR=${TOUCH_FRAMES_DIR:-/home/huyanh/Workspace/dataset/hoi4d/frames}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$REPO"
LOG="$REPO/outputs/pipeline_logs"; mkdir -p "$LOG"

MSTCN=outputs/downstream/mstcn
ASF=outputs/downstream/asformer
have_preds(){ ls "$MSTCN"/pred-test.*.recall.json.gz >/dev/null 2>&1 && \
              ls "$ASF"/pred-test.*.recall.json.gz   >/dev/null 2>&1; }
pipeline_alive(){ pgrep -f "run_full_pipeline.sh|train.py|step4_extract_features|train_head.py" >/dev/null 2>&1; }

echo "[stage6-waiter] waiting for downstream test predictions..."
dead=0
until have_preds; do
    if pipeline_alive; then dead=0; else dead=$((dead+1)); fi
    if [ "$dead" -ge 5 ]; then
        echo "[stage6-waiter] ERROR: pipeline not running and predictions absent (gave up)."
        exit 1
    fi
    sleep 60
done

echo "[stage6-waiter] predictions present -> running Stage 6 eval"
python scripts/step6_eval_nms.py 2>&1 | tee "$LOG/7_nms_eval.log"
echo "[stage6-waiter] STAGE6 DONE"
