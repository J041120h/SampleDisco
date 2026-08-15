#!/bin/bash
#SBATCH --job-name=f2d_alpha
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/alpha.out
#SBATCH --error=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/alpha.err
#SBATCH --partition=shared
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --exclude=compute-103,compute-058
set -uo pipefail
R=/dcs07/hongkai/data/harry/result
echo "############ autotune_record.txt ############"
find $R -name "autotune_record.txt" 2>/dev/null | sort | while read f; do
  echo "===== ${f#$R/}"
  grep -iE "rmd_weight|alpha|score|scoring|objective" "$f" 2>/dev/null | head -12
done
echo
echo "############ alpha_sweep.csv ############"
find $R -name "alpha_sweep.csv" 2>/dev/null | sort | while read f; do
  echo "===== ${f#$R/}"; head -25 "$f"
done
