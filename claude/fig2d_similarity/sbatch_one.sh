#!/bin/bash
# Build the similarity panels for ONE dataset. Submit with:
#   sbatch --job-name=f2d_<ds> sbatch_one.sh <dataset>
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/one_%j.out
#SBATCH --error=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/one_%j.err
#SBATCH --partition=shared
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --exclude=compute-103,compute-058
set -euo pipefail
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
DS="$1"
echo "[$(date)] fig2d similarity panel dataset=$DS on $(hostname)"
/dcs07/hongkai/data/harry/conda/envs/hongkai/bin/python -u \
    /users/hjiang/GenoDistance/code/claude/fig2d_similarity/build_similarity_panel.py \
    --dataset "$DS" \
    --outroot "${2:-/users/hjiang/GenoDistance/figure/figure2}"
echo "[$(date)] done $DS"
