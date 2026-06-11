#!/bin/bash
#SBATCH --job-name=abl_ha
#SBATCH --output=/dcs07/hongkai/data/harry/result/ablation/health_aging/logs/ha.out
#SBATCH --error=/dcs07/hongkai/data/harry/result/ablation/health_aging/logs/ha.err
#SBATCH --partition=shared
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
set -euo pipefail
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONNOUSERSITE=1
/users/hjiang/.conda/envs/hongkai/bin/python \
    /users/hjiang/GenoDistance/code/claude/ablation/run_ablation_health_aging.py \
    --outroot /dcs07/hongkai/data/harry/result/ablation/health_aging
