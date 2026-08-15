#!/bin/bash
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/tune3_%j.out
#SBATCH --error=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/tune3_%j.err
#SBATCH --partition=shared
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --exclude=compute-103,compute-058
set -euo pipefail
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
echo "[$(date)] 3-alpha tune+render $1 on $(hostname)"
/dcs07/hongkai/data/harry/conda/envs/hongkai/bin/python -u \
    /users/hjiang/GenoDistance/code/claude/fig2d_similarity/tune_and_render.py \
    --dataset "$1" "${@:2}"
echo "[$(date)] done $1"
