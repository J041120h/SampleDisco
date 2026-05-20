"""Diagnose where integrate_preprocess drops obsm.

Reproduce the steps in preparation/multi_omics_preprocess.integrate_preprocess
on the saved adata_sample.h5ad and check obsm after each transformation."""
import sys, os
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import scanpy as sc
import pandas as pd, numpy as np

P = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics/preprocess/adata_sample.h5ad"

def show(tag, a):
    obsm_shapes = {k: tuple(a.obsm[k].shape) for k in a.obsm}
    print(f"[{tag}] shape={a.shape}  obsm={obsm_shapes}", flush=True)

a = sc.read_h5ad(P)
show("after sc.read_h5ad", a)

# _store_original_sample_ids
a.obs["original_sample"] = a.obs["sample"].astype(str)
show("after _store_original_sample_ids", a)

# _maybe_append_modality_to_duplicates — no-op if sample IDs unique
sample_col, modality_col = "sample", "modality"
s = a.obs[sample_col].astype(str)
if s.duplicated(keep=False).any():
    a.obs[sample_col] = a.obs[sample_col].astype(str)
    modality_labels = a.obs[modality_col].astype(str)
    dup_mask = s.duplicated(keep=False)
    a.obs.loc[dup_mask, sample_col] = (s[dup_mask] + "_" + modality_labels[dup_mask])
show("after _maybe_append_modality_to_duplicates", a)

# var_names_make_unique
a.var_names_make_unique()
show("after var_names_make_unique", a)

# var dropna
if isinstance(a.var, pd.DataFrame):
    a.var = a.var.dropna(axis=1, how="all")
show("after var.dropna", a)

# add mt / MT
a.var["mt"] = a.var_names.str.upper().str.startswith("MT-")
a.var["MT"] = a.var["mt"]
show("after add mt/MT", a)

# QC metrics
sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], log1p=False, inplace=True)
show("after calculate_qc_metrics", a)

# Filter cells
sc.pp.filter_cells(a, min_genes=500)
show("after filter_cells min_genes=500", a)

# Filter genes
sc.pp.filter_genes(a, min_cells=10)
show("after filter_genes min_cells=10", a)

# MT cutoff (uses .copy())
a = a[a.obs["pct_counts_mt"] < 20].copy()
show("after mt cutoff + copy", a)

# Drop MT genes (column subset + copy)
mt_genes = a.var_names[a.var_names.str.upper().str.startswith("MT-")]
a = a[:, ~a.var_names.isin(list(mt_genes))].copy()
show("after drop MT genes + copy", a)

# Sample size cutoff (uses .copy())
cell_counts = a.obs.groupby(sample_col).size()
patients_to_keep = cell_counts[cell_counts >= 30].index
a = a[a.obs[sample_col].isin(patients_to_keep)].copy()
show("after sample size cutoff + copy", a)

# Final filter_genes
min_cells_for_gene = max(1, int(0.001 * a.n_obs))
sc.pp.filter_genes(a, min_cells=min_cells_for_gene)
show(f"after final filter_genes min_cells={min_cells_for_gene}", a)

# fill_missing_metadata_with_placeholder — only modifies obs columns + DataFrame obsm
from preparation.multi_omics_preprocess import fill_missing_metadata_with_placeholder
a = fill_missing_metadata_with_placeholder(a, verbose=False)
show("after fill_missing_metadata", a)

# slim
from utils.slim_adata import slim_adata_drop_expression
slim_adata_drop_expression(a)
show("after slim_adata_drop_expression", a)

# safe write to tmp + reread
from utils.safe_save import safe_h5ad_write
tmp_path = "/tmp/diag_adata_preprocessed.h5ad"
safe_h5ad_write(a, tmp_path)
b = sc.read_h5ad(tmp_path)
show("after safe_h5ad_write + reread", b)
print("DONE")
