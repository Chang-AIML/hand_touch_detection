#!/bin/bash
# Full verification gate for the refactor. Run after each risky step; all must pass.
#   tests/run_all.sh          -> smoke + state_dict + cli + golden check
#   tests/run_all.sh nogpu    -> skip the GPU golden eval (Tier-1/2 only)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/chang/miniconda3/bin/python3
echo "###### Tier-1 smoke ######";        bash tests/smoke.sh            || exit 1
echo "###### Tier-2 state_dict ######";   $PY  tests/check_state_dict.py || exit 1
echo "###### Tier-2 cli back-compat ######"; $PY tests/check_cli.py      || exit 1
if [ "${1:-}" = nogpu ]; then echo "== skipped GPU golden =="; exit 0; fi
echo "###### Tier-3 golden eval ######";  bash tests/golden_eval.sh check || exit 1
echo "ALL CHECKS PASS"
