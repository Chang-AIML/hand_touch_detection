#!/bin/bash
# Full TSP pipeline on HOI4D-touch (E2E-Spot §B.3 aligned):
#   1) MViT-B GVF  2) train TSP dual-head  3) select best Foreground-F1
#   4) extract per-frame [N,512] features  5) downstream MS-TCN + ASFormer
# Env: vjepa21. GPU: $GPU (default 0). Logs: outputs/pipeline_logs/.
REPO=/home/huyanh/Workspace/hand_touch_detection
source /home/huyanh/miniconda/etc/profile.d/conda.sh
conda activate base
set -eo pipefail
export CONDA_ENV=base
export TOUCH_FRAMES_DIR=${TOUCH_FRAMES_DIR:-/home/huyanh/Workspace/dataset/hoi4d/frames}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$REPO"
LOG="$REPO/outputs/pipeline_logs"; mkdir -p "$LOG"
say(){ echo "===== [$(date '+%F %T')] $* ====="; }

say "STAGE 1/5: MViT-B GVF extraction"
python methods/tsp/step1_extract_mvit_gvf.py 2>&1 | tee "$LOG/1_gvf.log"

say "STAGE 2/5: train TSP (dual head: action touch/untouch + GVF-fed FG/BG, 8 epochs)"
bash methods/tsp/train_tsp_on_hoi4d.sh 2>&1 | tee "$LOG/2_train.log"

say "STAGE 3/5: select best epoch by Foreground-F1"
python methods/tsp/step3_select_best_f1.py 2>&1 | tee "$LOG/3_selectf1.log"
BEST=$(python -c "import json,os,config; print(json.load(open(os.path.join(config.TRAIN_OUT,'best_by_f1.json')))['best_by_f1']['ckpt'])")
say "best checkpoint = $BEST"

say "STAGE 4/5: extract per-frame TSP features [N,512] (dense window-12, replicate-pad)"
python methods/tsp/step4_extract_features.py --ckpt "$BEST" 2>&1 | tee "$LOG/4_extract.log"

say "STAGE 5/5: downstream MS-TCN then ASFormer on TSP features"
python methods/spot_head/train_head.py -m mstcn    2>&1 | tee "$LOG/5_mstcn.log"
python methods/spot_head/train_head.py -m asformer 2>&1 | tee "$LOG/6_asformer.log"

say "ALL DONE"
