#!/bin/bash -i
# Train spot_head precise-spotting heads (MS-TCN + ASFormer) on the TSP features.
# SELF-CONTAINED: uses the vendored trainer methods/spot_head/train_head.py + common/
# (no external spot repo needed). Paths/epochs come from config.py.
#
# Usage:  bash spot_head/train_spot_head.sh ["mstcn asformer"]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

source activate base
ARCHS="${1:-mstcn asformer}"
for ARCH in $ARCHS; do
    echo "================  training $ARCH  ================"
    python "$HERE/train_head.py" -m "$ARCH"
done
echo "DONE. results + best_epoch.pt under <SPOT_HEAD_OUT>/{mstcn,asformer}"
