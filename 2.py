#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scanpy as sc

# ================================================================
# USER EDIT SECTION
# ================================================================
ADATA_PATH = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired/multiomics/preprocess/adata_sample.h5ad"
SAMPLE_COL = "sample"
BATCH_COL = "batch"
# ================================================================

print(f"[Inspector] Loading AnnData from: {ADATA_PATH}")
adata = sc.read_h5ad(ADATA_PATH)
obs = adata.obs.copy()

print(f"[Inspector] adata: {adata.n_obs} cells")
print(f"[Inspector] obs columns: {list(obs.columns)}")

if SAMPLE_COL not in obs.columns:
    raise ValueError(f"Sample column '{SAMPLE_COL}' not found.")

if BATCH_COL not in obs.columns:
    raise ValueError(f"Batch column '{BATCH_COL}' not found.")

grouped = obs.groupby(SAMPLE_COL)

print("\n===============================================================")
print("CHECKING: batch consistency within sample")
print("===============================================================")

# compute unique batch values per sample
unique_batches = grouped[BATCH_COL].apply(lambda x: np.unique(x.dropna().values))
n_unique = unique_batches.apply(len)

n_samples = len(n_unique)
n_all_na = (n_unique == 0).sum()
n_one = (n_unique == 1).sum()
n_multi = (n_unique > 1).sum()

print(f"Total samples        : {n_samples}")
print(f"All-NA batches       : {n_all_na}")
print(f"Single batch samples : {n_one}")
print(f"Multi-batch samples (!): {n_multi}")

# summary on why it fails
constant = (n_multi == 0)
print(f"\n=> batch constant within each sample? : {constant}")
if not constant:
    print("=> This means batch will NOT be promoted as sample metadata.\n")

# print problematic samples
if n_multi > 0:
    print("Conflicting samples (showing first 20):")
    conflict_samples = n_unique[n_unique > 1].index.tolist()
    for s in conflict_samples[:20]:
        vals = unique_batches.loc[s]
        print(f"  sample={s}  batches={vals}")

# show global batch distribution for sanity
print("\n===============================================================")
print("GLOBAL BATCH COUNTS (cell-level)")
print("===============================================================")
print(obs[BATCH_COL].value_counts(dropna=False))

print("\nDONE.")
