#!/bin/bash
# Run all (dataset, variant) combos as separate subprocesses (memory safety).
set -u
PY=/users/hjiang/.conda/envs/hongkai/bin/python
SCRIPT=/users/hjiang/GenoDistance/code/claude/sd_param_sweep_one.py
LOG=/tmp/sd_sweep.log
> $LOG

for ds in Lutea Retina Heart ENCODE; do
    for variant in baseline cw_0.30 cw_1.00 cw_2.00 fixed_w K_finer K_coarser A1_soft; do
        echo "=================================================" | tee -a $LOG
        echo "  $ds  /  $variant" | tee -a $LOG
        echo "=================================================" | tee -a $LOG
        $PY -u "$SCRIPT" "$ds" "$variant" 2>&1 | tee -a $LOG
    done
done
echo "ALL DONE" | tee -a $LOG
