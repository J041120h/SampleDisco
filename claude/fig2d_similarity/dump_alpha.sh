#!/bin/bash
#SBATCH --job-name=f2d_da
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/dumpalpha.out
#SBATCH --error=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/dumpalpha.err
#SBATCH --partition=shared
#SBATCH --mem=8G --cpus-per-task=2 --time=0:30:00
#SBATCH --exclude=compute-103,compute-058
set -uo pipefail
R=/dcs07/hongkai/data/harry/result
find $R -name "autotune_record.txt" 2>/dev/null | sort | while read f; do
  echo "############ ${f#$R/}"
  sed -n '1,40p' "$f"
done
