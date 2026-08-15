#!/bin/bash
#SBATCH --job-name=f2d_panel
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/panel_%a.out
#SBATCH --error=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/panel_%a.err
#SBATCH --partition=shared
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --array=0-6
set -euo pipefail
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
DATASETS=(retina lutea heart ENCODE covid long_covid health_aging)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
echo "[$(date)] fig2d similarity panel dataset=$DS on $(hostname)"
/dcs07/hongkai/data/harry/conda/envs/hongkai/bin/python -u \
    /users/hjiang/GenoDistance/code/claude/fig2d_similarity/build_similarity_panel.py \
    --dataset "$DS" \
    --outroot /dcs07/hongkai/data/claude/fig2d_similarity
echo "[$(date)] done $DS"
