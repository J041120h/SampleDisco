#!/bin/bash
#SBATCH --job-name=f2d_rerender
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/rerender.out
#SBATCH --error=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/rerender.err
#SBATCH --partition=shared
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --exclude=compute-103
set -euo pipefail
export MPLBACKEND=Agg
PY=/dcs07/hongkai/data/harry/conda/envs/hongkai/bin/python
F2=/users/hjiang/GenoDistance/figure/figure2
S=/users/hjiang/GenoDistance/code/claude/fig2d_similarity
for ds in ENCODE ENCODE_rna ENCODE_rna_atac heart heart_rna retina lutea long_covid long_covid_fine health_aging unpaired_diemb unpaired_paper unpaired_test; do
    echo "== $ds"; $PY -u $S/render_only.py --dataset $ds --outroot $F2 || echo "FAILED $ds"
done
for n in 25 50 100 200 279 400; do
    echo "== covid_$n"; $PY -u $S/render_only.py --dataset covid_$n --outroot $F2/covid_series || echo "FAILED covid_$n"
done
for s in Su SS2 SS1 Lee Zhu Wilk Aruna Yu Wen Mudd; do
    echo "== covid_study_$s"; $PY -u $S/render_only.py --dataset covid_study_$s --outroot $F2/covid_series/by_study || echo "FAILED $s"
done
