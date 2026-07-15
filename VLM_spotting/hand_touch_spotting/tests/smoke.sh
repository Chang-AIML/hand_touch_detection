#!/bin/bash
# Tier-1: import every live module + --help + verbatim-literal guard. Seconds, no GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/chang/miniconda3/bin/python3
export VJEPA_REPO=/home/chang/Project/vlm_deps/vjepa2 QWEN_PATH=/home/chang/models/Qwen3-VL-8B-Instruct
export AS_DATA_ROOT=/home/chang/Dataset/Action_Spotting AS_ANN_DIR=/home/chang/Project/vlm_deps/annotations_dpc HF_HOME=/home/chang/.cache/huggingface
fail=0
echo "== import every live module =="
$PY -c "import dpc.train_mixed, dpc.windowed_dataset, dpc.questions, dpc.vjepa, dpc.frame_source, dpc.paths, dpc.eval_vendor.score, dpc.eval_vendor.eval_nms, models.frame_compress, models.localizer, models.wrapper" \
  && echo "  imports OK" || { echo "  IMPORT FAIL"; fail=1; }
echo "== entry --help =="
$PY dpc/train_mixed.py --help >/dev/null 2>&1 && echo "  --help OK" || { echo "  --help FAIL"; fail=1; }
echo "== root train.py shim =="
$PY train.py --help >/dev/null 2>&1 && echo "  train.py OK" || { echo "  train.py FAIL"; fail=1; }
echo "== verbatim prompt literals preserved (R2 guard) =="
L=models/localizer.py
for s in \
  "Below are per-frame motion tokens, each preceded by its frame index." \
  " Answer with the single frame index. Answer:" \
  "there is no frame related to the action" ; do
  grep -qF -- "$s" "$L" && echo "  literal OK: ${s:0:40}..." || { echo "  LITERAL MISSING: $s"; fail=1; }
done
[ $fail = 0 ] && echo "SMOKE PASS" || { echo "SMOKE FAIL"; exit 1; }
