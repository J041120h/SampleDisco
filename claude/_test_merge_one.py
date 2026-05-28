"""Test the merge on just adata_preprocessed_hvg.h5ad first."""
import sys
sys.path.insert(0, "/users/hjiang/GenoDistance/code/claude")
from merge_fine_names_to_h5ads import build_per_cell_lookup, annotate_file

print("Building lookup ...")
fine, helper = build_per_cell_lookup()
print(f"fine entries: {len(fine):,}; helper entries: {len(helper):,}")

annotate_file(
    "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/preprocess/adata_preprocessed_hvg.h5ad",
    fine,
    helper,
)
print("\nVERIFY (full anndata load):")
import anndata as ad
a = ad.read_h5ad("/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/preprocess/adata_preprocessed_hvg.h5ad")
print("shape:", a.shape)
print("obs columns:", list(a.obs.columns))
print()
print("Cluster_fine_names value counts (all):")
print(a.obs["Cluster_fine_names"].value_counts().to_string())
print()
print("Cluster_helper_memory_names value counts:")
print(a.obs["Cluster_helper_memory_names"].value_counts(dropna=False).to_string())
print()
print("Cluster_fine_numbers dtype:", a.obs["Cluster_fine_numbers"].dtype)
print("Cluster_fine_numbers value counts (top 10):")
print(a.obs["Cluster_fine_numbers"].value_counts().head(10).to_string())
