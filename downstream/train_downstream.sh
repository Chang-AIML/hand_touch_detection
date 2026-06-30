#!/bin/bash -i
# Train downstream precise-spotting heads (MS-TCN + ASFormer) on the TSP features.
# SELF-CONTAINED: uses the vendored trainer downstream/train_head.py + downstream/lib/
# (no external spot repo needed). Paths/epochs come from config.py.
#
# Usage:  bash downstream/train_downstream.sh ["mstcn asformer"]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

source activate base
ARCHS="${1:-mstcn asformer}"
for ARCH in $ARCHS; do
    echo "================  training $ARCH  ================"
    python "$HERE/train_head.py" -m "$ARCH"
done
echo "DONE. results + best_epoch.pt under <DOWNSTREAM_OUT>/{mstcn,asformer}"
