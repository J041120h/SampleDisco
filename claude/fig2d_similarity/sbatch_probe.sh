#!/bin/bash
#SBATCH --job-name=f2d_probe
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/probe.out
#SBATCH --error=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/probe.err
#SBATCH --partition=shared
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
set -euo pipefail
export MPLBACKEND=Agg
/dcs07/hongkai/data/harry/conda/envs/hongkai/bin/python -u \
    /users/hjiang/GenoDistance/code/claude/fig2d_similarity/probe_inputs.py
