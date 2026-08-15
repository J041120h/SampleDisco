#!/bin/bash
#SBATCH --job-name=f2d_verify
#SBATCH --output=/dcs07/hongkai/data/claude/_f2d_verify/v.out
#SBATCH --error=/dcs07/hongkai/data/claude/_f2d_verify/v.err
#SBATCH --partition=shared
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:20:00
#SBATCH --exclude=compute-103
set -euo pipefail
export MPLBACKEND=Agg
/dcs07/hongkai/data/harry/conda/envs/hongkai/bin/python -u /dcs07/hongkai/data/claude/_f2d_verify/make_plots.py
