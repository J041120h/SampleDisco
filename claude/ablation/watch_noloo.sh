#!/bin/bash
# Overnight monitor for the no_loo ablation. Purely observational -- the compute
# is SLURM-managed and completes whether or not this is running.
LOG=/dcs07/hongkai/data/harry/result/ablation_noloo/logs/watch.log
mkdir -p "$(dirname "$LOG")"
echo "=== no_loo ablation monitor started $(date) ===" | tee -a "$LOG"
while true; do
    N=$(squeue -u "$USER" -h -n abl_covid_noloo,abl_mo_noloo,abl_ha_noloo,abl_merge_noloo 2>/dev/null | wc -l)
    echo "[$(date '+%F %T')] $N job(s) still queued/running" | tee -a "$LOG"
    squeue -u "$USER" -h -o "   %.12i %.18j %.10T %.8M %R" \
        -n abl_covid_noloo,abl_mo_noloo,abl_ha_noloo,abl_merge_noloo 2>/dev/null | tee -a "$LOG"
    if [ "$N" -eq 0 ]; then
        echo "[$(date '+%F %T')] ALL JOBS FINISHED" | tee -a "$LOG"
        echo "--- results ---" | tee -a "$LOG"
        find /dcs07/hongkai/data/harry/result/ablation_noloo -name "ablation_summary_*.csv" 2>/dev/null | tee -a "$LOG"
        echo "--- any failures ---" | tee -a "$LOG"
        grep -il "error\|traceback" /dcs07/hongkai/data/harry/result/ablation_noloo/*/logs/*.err 2>/dev/null | tee -a "$LOG"
        break
    fi
    sleep 600
done
