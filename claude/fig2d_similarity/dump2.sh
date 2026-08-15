#!/bin/bash
#SBATCH --job-name=f2d_d2
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/dump2.out
#SBATCH --error=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/dump2.err
#SBATCH --partition=shared
#SBATCH --mem=8G --cpus-per-task=2 --time=0:30:00
#SBATCH --exclude=compute-103,compute-058
set -uo pipefail
R=/dcs07/hongkai/data/harry/result
find $R -name "autotune_record.txt" 2>/dev/null | sort | while read f; do
  echo "############ ${f#$R/}"
  grep -A4 -iE "best param|chosen|^ *1 " "$f" | head -14
done
