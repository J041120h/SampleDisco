#!/bin/bash
# Wrap each default-α sample-embedding run with monitor_wrapper.py to capture
# CPU/RAM/GPU/time per task. Only the sample embedding step is timed
# (preprocess is already done; autotune and downstream not included).

set -u

MON=/users/hjiang/GenoDistance/code/Benchmark_covid/monitor_wrapper.py
WORK=/users/hjiang/GenoDistance/code
INNER=$WORK/se_one_task.py

run_task() {
    local task=$1
    local outdir=$2
    echo
    echo "================================================================="
    echo "TASK: $task   →   $outdir"
    echo "================================================================="
    mkdir -p "$outdir"
    python -u "$MON" \
        --outdir "$outdir" \
        --label sampledisco \
        --interval 1 \
        --perf \
        --nsys \
        --workdir "$WORK" \
        --cmd "python -u $INNER --task $task --outdir $outdir"
}

# ----- COVID RNA (6 sample sizes) -----
for s in 25 50 100 200 279 400; do
    run_task "covid_rna_${s}" \
        "/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_${s}_sample/rna/sampledisco_default"
done

# ----- COVID ATAC -----
run_task "covid_atac" \
    "/dcs07/hongkai/data/harry/result/Benchmark_covid/ATAC/sampledisco_default"

# ----- Multi-omics (default-α only) -----
run_task "mo_encode" \
    "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/sampledisco_default"
run_task "mo_lutea" \
    "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/sampledisco_default"
run_task "mo_retina" \
    "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/sampledisco_default"
run_task "mo_heart" \
    "/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/sampledisco_default"

# ----- Unpaired (default-α; previously only tuned was generated) -----
run_task "unpaired" \
    "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_paper/multiomics/sampledisco_default"

echo
echo "================================================================="
echo "ALL MONITORED TASKS COMPLETE"
echo "================================================================="
