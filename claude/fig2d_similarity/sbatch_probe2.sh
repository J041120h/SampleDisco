#!/bin/bash
#SBATCH --job-name=f2d_probe2
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/probe2.out
#SBATCH --error=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/probe2.err
#SBATCH --partition=shared
#SBATCH --mem=48G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
set -euo pipefail
export MPLBACKEND=Agg
/dcs07/hongkai/data/harry/conda/envs/hongkai/bin/python -u \
    /users/hjiang/GenoDistance/code/claude/fig2d_similarity/probe_more.py
