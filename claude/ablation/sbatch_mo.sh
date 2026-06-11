#!/bin/bash
#SBATCH --job-name=abl_mo
#SBATCH --output=/dcs07/hongkai/data/harry/result/ablation/multiomics/logs/mo_%a.out
#SBATCH --error=/dcs07/hongkai/data/harry/result/ablation/multiomics/logs/mo_%a.err
#SBATCH --partition=shared
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --array=0-2
set -euo pipefail
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
DATASETS=(ENCODE heart lutea)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
cd /users/hjiang/GenoDistance/code/Benchmark_multiomics
echo "[$(date)] mo ablation dataset=$DS on $(hostname)"
/users/hjiang/.conda/envs/hongkai/bin/python \
    /users/hjiang/GenoDistance/code/claude/ablation/run_ablation_mo.py \
    --dataset "$DS" --outroot /dcs07/hongkai/data/harry/result/ablation/multiomics
echo "[$(date)] done $DS"
