#!/bin/bash
#SBATCH --job-name=f2d_render
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/render_%j.out
#SBATCH --error=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/render_%j.err
#SBATCH --partition=shared
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
set -euo pipefail
export MPLBACKEND=Agg
/dcs07/hongkai/data/harry/conda/envs/hongkai/bin/python -u \
    /users/hjiang/GenoDistance/code/claude/fig2d_similarity/render_only.py --dataset "$1"
