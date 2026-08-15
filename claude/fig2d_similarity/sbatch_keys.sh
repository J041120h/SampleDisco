#!/bin/bash
#SBATCH --job-name=f2d_keys
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/keys.out
#SBATCH --error=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/keys.err
#SBATCH --partition=shared
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00
#SBATCH --exclude=compute-103
set -euo pipefail
export MPLBACKEND=Agg
/dcs07/hongkai/data/harry/conda/envs/hongkai/bin/python -u \
    /users/hjiang/GenoDistance/code/claude/fig2d_similarity/validate_keys.py
