#!/bin/bash
#SBATCH --job-name=f2d_rr
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/rr_%a.out
#SBATCH --error=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/rr_%a.err
#SBATCH --partition=shared
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --exclude=compute-103
#SBATCH --array=0-28
set -euo pipefail
export MPLBACKEND=Agg
F2=/users/hjiang/GenoDistance/figure/figure2
DS=(ENCODE ENCODE_rna ENCODE_rna_atac heart heart_rna retina lutea long_covid long_covid_fine health_aging unpaired_diemb unpaired_paper unpaired_test \
    covid_25 covid_50 covid_100 covid_200 covid_279 covid_400 \
    covid_study_Su covid_study_SS2 covid_study_SS1 covid_study_Lee covid_study_Zhu covid_study_Wilk covid_study_Aruna covid_study_Yu covid_study_Wen covid_study_Mudd)
D=${DS[$SLURM_ARRAY_TASK_ID]}
case "$D" in
  covid_study_*) OUT=$F2/covid_series/by_study ;;
  covid_*)       OUT=$F2/covid_series ;;
  *)             OUT=$F2 ;;
esac
echo "== $D -> $OUT"
/dcs07/hongkai/data/harry/conda/envs/hongkai/bin/python -u \
    /users/hjiang/GenoDistance/code/claude/fig2d_similarity/render_only.py \
    --dataset "$D" --outroot "$OUT"
