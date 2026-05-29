"""Propagate the merged biological cell-type names onto the embedding union
(adata_sample.h5ad) so BOTH RNA and ATAC cells carry the named cell_type.

The union's cell_type is the cross-omics Leiden clustering (1-17) computed on
the shared Z_clust embedding -- present for both modalities. CellTypist naming
(RNA-derived majority vote per Leiden cluster, then merged) maps 1:1 onto those
Leiden ids, so the names transfer to ATAC through the shared clustering. The
numeric Leiden is preserved as obs['leiden'].
"""
import sys

import pandas as pd
import scanpy as sc

sys.path.insert(0, "/users/hjiang/GenoDistance/code")
from utils.safe_save import safe_h5ad_write

BASE = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics"
UNION = f"{BASE}/preprocess/adata_sample.h5ad"
RNA = f"{BASE}/preprocess/adata_rna_preprocessed.h5ad"

# Leiden -> merged-name map, taken from the RNA file (has both columns).
r = sc.read_h5ad(RNA, backed="r")
name_map = (r.obs[["leiden", "cell_type"]].astype(str).drop_duplicates()
            .set_index("leiden")["cell_type"].to_dict())
print("leiden -> name map:", name_map, flush=True)

u = sc.read_h5ad(UNION)
u.obs["leiden"] = u.obs["cell_type"].astype(str)            # preserve numeric Leiden
u.obs["cell_type"] = (u.obs["leiden"].map(name_map)
                      .astype("category"))                  # named, both modalities
n_na = int(u.obs["cell_type"].isna().sum())
print(f"unmapped cells: {n_na}", flush=True)
print("named cell_type x modality:", flush=True)
print(pd.crosstab(u.obs["cell_type"], u.obs["modality"]), flush=True)

safe_h5ad_write(u, UNION)
print("UNION_NAMED_DONE", flush=True)
