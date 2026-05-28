"""Run merge on the two remaining files (preprocessed + benchmark_ready)."""
import sys
sys.path.insert(0, "/users/hjiang/GenoDistance/code/claude")
from merge_fine_names_to_h5ads import build_per_cell_lookup, annotate_file

print("Building lookup ...", flush=True)
fine, helper = build_per_cell_lookup()

for p in [
    "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/preprocess/adata_preprocessed.h5ad",
    "/dcs07/hongkai/data/harry/result/health_aging_PBMC/data/benchmark_ready/pbmc_benchmark_ready.h5ad",
]:
    annotate_file(p, fine, helper)
print("\nALL DONE.")
