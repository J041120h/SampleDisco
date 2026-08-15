#!/bin/bash
#SBATCH --job-name=f2d_d3
#SBATCH --output=/users/hjiang/GenoDistance/code/claude/fig2d_similarity/logs/dump3.out
#SBATCH --partition=shared
#SBATCH --mem=8G --cpus-per-task=1 --time=0:20:00
#SBATCH --exclude=compute-103,compute-058
cat -A "/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_400_sample/rna/sampledisco_tuned/sample_embedding/autotune_record.txt" | sed -n '25,70p' | sed 's/\$$//'
