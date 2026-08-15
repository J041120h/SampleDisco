#!/bin/bash
#SBATCH --job-name=abl_merge_noloo
#SBATCH --output=/dcs07/hongkai/data/harry/result/ablation_noloo/logs/merge.out
#SBATCH --error=/dcs07/hongkai/data/harry/result/ablation_noloo/logs/merge.err
#SBATCH --partition=shared
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
set -euo pipefail
export MPLBACKEND=Agg
PY=/dcs07/hongkai/data/harry/conda/envs/hongkai/bin/python
echo "[$(date)] merging no_loo results"
$PY /users/hjiang/GenoDistance/code/claude/ablation/merge_and_plot_noloo.py
echo "[$(date)] merge done"
