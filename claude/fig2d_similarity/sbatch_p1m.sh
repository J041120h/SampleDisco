#!/bin/bash
#SBATCH --job-name=f2d_1m
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/p1m.out
#SBATCH --error=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/p1m.err
#SBATCH --partition=shared
#SBATCH --mem=48G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --exclude=compute-103,compute-058
set -euo pipefail
export MPLBACKEND=Agg
/dcs07/hongkai/data/harry/conda/envs/hongkai/bin/python -u \
    /users/hjiang/GenoDistance/code/claude/fig2d_similarity/probe_1m.py
