#!/bin/bash
# Decisive 2x2 swap: Group A = {q,in_ln,a3} (Q-Former), Group B = {out.*} (readout).
# MIX-3 = QF s1200 + out s750 ; MIX-4 = QF s750 + out s1200.
# Per cell record OOD@16, ID@2, OOD ndet. OOD first (decisive), then ID.
cd /home/chang/Project/VLM_spotting/hand_touch_spotting
DST=outputs/local_eval
declare -A CK=( [MIX3_qf1200_out750]=conn_MIX3_qf1200_out750.pt [MIX4_qf750_out1200]=conn_MIX4_qf750_out1200.pt )

echo "===== 2x2 START $(date +%H:%M:%S) ====="
# ---- Phase 1: OOD (finediving full 1018) for both, decisive out-vs-QF ----
for M in MIX3_qf1200_out750 MIX4_qf750_out1200; do
  echo "### $M OOD start $(date +%H:%M:%S) ###"
  bash "$DST/run_ood.sh" "$DST/${CK[$M]}" 2>&1 \
    | grep -E "eval:ood:finediving|Traceback|Error|assert" > "$DST/res2x2_${M}_ood.txt"
  echo "### $M OOD: $(cat "$DST/res2x2_${M}_ood.txt") ###"
done
# ---- Phase 2: in-domain (200/ds, finegym 640) for both, ID@2 ----
for M in MIX3_qf1200_out750 MIX4_qf750_out1200; do
  echo "### $M ID start $(date +%H:%M:%S) ###"
  bash "$DST/run_indom.sh" "$DST/${CK[$M]}" 2>&1 \
    | grep -E "eval:in:|Traceback|Error|assert" > "$DST/res2x2_${M}_indom.txt"
  echo "### $M ID: $(grep eval:in:_all "$DST/res2x2_${M}_indom.txt") ###"
done
echo "===== 2x2 DONE $(date +%H:%M:%S) ====="
