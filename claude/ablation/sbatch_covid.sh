#!/bin/bash
#SBATCH --job-name=abl_covid
#SBATCH --output=/dcs07/hongkai/data/harry/result/ablation/covid/logs/covid_%a.out
#SBATCH --error=/dcs07/hongkai/data/harry/result/ablation/covid/logs/covid_%a.err
#SBATCH --partition=shared
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --array=0-4
set -euo pipefail
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

SIZES=(50 100 200 279 400)
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}
R=/dcs07/hongkai/data/harry/result
PY=/users/hjiang/.conda/envs/hongkai/bin/python

cd /users/hjiang/GenoDistance/code/Benchmark_covid
echo "[$(date)] starting covid ablation size=$SIZE on $(hostname)"
$PY /users/hjiang/GenoDistance/code/claude/ablation/run_ablation_covid.py \
    --size "$SIZE" \
    --adata "$R/Benchmark_covid/covid_${SIZE}_sample/rna/preprocess/adata_cell.h5ad" \
    --meta /dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv \
    --outroot "$R/ablation/covid"
echo "[$(date)] finished size=$SIZE"
