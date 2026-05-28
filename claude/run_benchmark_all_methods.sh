#!/usr/bin/env bash
# Run the health-aging-PBMC benchmark on all 7 available competing methods.
# GloScope is skipped (no embedding produced yet).
set -uo pipefail

BENCH=/dcs07/hongkai/data/harry/result/health_aging_PBMC/benchmark
RUNNER="${BENCH}/scripts/run_one_method.sh"
OM=/dcs07/hongkai/data/harry/result/health_aging_PBMC/other_methods

declare -A EMB=(
  [naive_pseudobulk]="${OM}/naive_pseudobulk/pseudobulk/pca_embeddings.csv"
  [PILOT]="${OM}/pilot/wasserstein_distance_mds_10d.csv"
  [QOT]="${OM}/QOT/316_qot_distance_matrix_mds_10d.csv"
  [scPoli]="${OM}/scPoli/sample_embeddings_full.csv"
  [MFA]="${OM}/MFA/sample_embeddings.csv"
  [GEDI]="${OM}/GEDI_modifies/gedi_sample_embedding.csv"
  [MUSTARD]="${OM}/MUSTARD/sample_embedding.csv"
)

for method in naive_pseudobulk PILOT QOT scPoli MFA GEDI MUSTARD; do
  emb="${EMB[$method]}"
  echo
  echo "==================== $method ===================="
  if [ ! -f "$emb" ]; then
    echo "  SKIP: embedding not found: $emb"
    continue
  fi
  bash "$RUNNER" "$method" "$emb"
done

echo
echo "==================== AGGREGATE ===================="
/users/hjiang/.conda/envs/hongkai/bin/python -u "${BENCH}/scripts/99_aggregate_to_summary.py"
