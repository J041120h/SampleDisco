#!/bin/bash
#SBATCH --job-name=f2d_trimtest
#SBATCH --output=/dcs07/hongkai/data/claude/_f2d_trimtest/test.out
#SBATCH --error=/dcs07/hongkai/data/claude/_f2d_trimtest/test.err
#SBATCH --partition=shared
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:20:00
#SBATCH --exclude=compute-103
set -euo pipefail
export MPLBACKEND=Agg
/dcs07/hongkai/data/harry/conda/envs/hongkai/bin/python -u /dcs07/hongkai/data/claude/_f2d_trimtest/make_plots.py
