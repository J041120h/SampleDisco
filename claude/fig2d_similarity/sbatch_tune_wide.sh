#!/bin/bash
#SBATCH --job-name=f2d_wide
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/wide_%j.out
#SBATCH --error=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/wide_%j.err
#SBATCH --partition=shared
#SBATCH --mem=48G --cpus-per-task=4 --time=6:00:00
#SBATCH --exclude=compute-103,compute-058
set -euo pipefail
export MPLBACKEND=Agg
/dcs07/hongkai/data/harry/conda/envs/hongkai/bin/python -u \
    /users/hjiang/GenoDistance/code/claude/fig2d_similarity/tune_and_render.py \
    --dataset scBloodNL_V1 --alpha-bounds 0.1 200 \
    --outroot /dcs07/hongkai/data/claude/f2d_widebound
