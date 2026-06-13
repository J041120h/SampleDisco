"""Tier-2 unit tests for the new preparation/multi_omics_merge.py functions.

Runs on cached glue-{rna,atac}-emb.h5ad from /dcs07/.../result/test (Mode A
data). Writes outputs to a tmp dir so the real test/ dir is untouched —
Tier 3 will exercise the full wrapper into /dcs07/.../result/test.

Pass criteria for each function are asserted; failures raise.
"""
from __future__ import annotations
import os, sys, shutil, tempfile, time
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import numpy as np, pandas as pd, scanpy as sc
from scipy import sparse

from sampledisco.preparation.multi_omics_merge import (
    build_embedding_union,
    preprocess_rna_for_downstream,
    preprocess_atac_for_downstream,
    propagate_cell_type,
)

RNA_EMB  = "/dcs07/hongkai/data/harry/result/test/multiomics/integration/glue/glue-rna-emb.h5ad"
ATAC_EMB = "/dcs07/hongkai/data/harry/result/test/multiomics/integration/glue/glue-atac-emb.h5ad"
TMP      = tempfile.mkdtemp(prefix="merge_units_test_")
print(f"[setup] tmp dir: {TMP}")

# ── T2.1: build_embedding_union ───────────────────────────────────────────
print("\n=== T2.1: build_embedding_union ===")
t = time.time()
union_path = f"{TMP}/adata_sample.h5ad"
union = build_embedding_union(
    rna_emb_path=RNA_EMB, atac_emb_path=ATAC_EMB,
    output_path=union_path, verbose=True,
)
print(f"  shape={union.shape}  obsm={list(union.obsm.keys())}  "
      f"obs cols={list(union.obs.columns)[:8]}")
print(f"  modality counts: {union.obs['modality'].value_counts().to_dict()}")

assert union.n_obs == 29989 + 29021, f"expected 59010 obs, got {union.n_obs}"
assert union.shape[1] == 0, f"expected X with 0 columns, got {union.shape[1]}"
assert "X_glue" in union.obsm, "X_glue missing from union obsm"
assert "modality" in union.obs.columns, "modality column missing"
assert "original_barcode" in union.obs.columns, "original_barcode missing"
assert "sample" in union.obs.columns, "sample column missing (ATAC inference failed?)"
assert union.obs["sample"].notna().all(), "some sample values are NaN"
assert union.obs.index.is_unique, "duplicate obs indices"
assert all(idx.endswith("_RNA") or idx.endswith("_ATAC") for idx in union.obs.index), \
    "obs index missing _RNA/_ATAC suffix"
print(f"  ✓ all assertions passed ({time.time()-t:.1f}s)")


# ── T2.2: preprocess_rna_for_downstream ───────────────────────────────────
print("\n=== T2.2: preprocess_rna_for_downstream ===")
t = time.time()
rna_pre_path = f"{TMP}/adata_rna_preprocessed.h5ad"
rna_pre = preprocess_rna_for_downstream(
    rna_emb_path=RNA_EMB, output_path=rna_pre_path, verbose=True,
)
print(f"  shape={rna_pre.shape}  obs cols={list(rna_pre.obs.columns)[:6]}")
print(f"  layers: {list(rna_pre.layers.keys())}")
print(f"  X stats: nnz={rna_pre.X.nnz if sparse.issparse(rna_pre.X) else (rna_pre.X!=0).sum()}  "
      f"max={float(rna_pre.X.max()):.3f}")

assert rna_pre.n_obs > 0 and rna_pre.n_vars > 0, "empty RNA preprocessed"
assert rna_pre.n_obs <= 29989, "QC should not add cells"
assert "counts" in rna_pre.layers, "layers['counts'] missing (raw counts not preserved)"
assert (rna_pre.X.nnz if sparse.issparse(rna_pre.X) else (rna_pre.X != 0).sum()) > 0, \
    "X is all zeros (normalize+log1p failed)"
# Normalized data: log1p caps max well below raw count scale
assert float(rna_pre.X.max()) < 20, \
    f"X max={float(rna_pre.X.max())} suggests no log1p applied"
print(f"  ✓ all assertions passed ({time.time()-t:.1f}s)")


# ── T2.3: preprocess_atac_for_downstream ──────────────────────────────────
print("\n=== T2.3: preprocess_atac_for_downstream ===")
t = time.time()
atac_pre_path = f"{TMP}/adata_atac_preprocessed.h5ad"
# scrublet on 29k cells × 230k peaks is slow + sometimes flaky; disable for
# the unit test (the QC path itself is the smoke target here).
atac_pre = preprocess_atac_for_downstream(
    atac_emb_path=ATAC_EMB, output_path=atac_pre_path,
    doublet_detection=False, verbose=True,
)
print(f"  shape={atac_pre.shape}  obs cols={list(atac_pre.obs.columns)[:6]}")
print(f"  layers: {list(atac_pre.layers.keys())}")
print(f"  X stats: nnz={atac_pre.X.nnz if sparse.issparse(atac_pre.X) else (atac_pre.X!=0).sum()}  "
      f"max={float(atac_pre.X.max()):.3f}")

assert atac_pre.n_obs > 0 and atac_pre.n_vars > 0, "empty ATAC preprocessed"
assert "counts" in atac_pre.layers, "layers['counts'] missing"
assert (atac_pre.X.nnz if sparse.issparse(atac_pre.X) else (atac_pre.X != 0).sum()) > 0, \
    "ATAC X is all zeros (TF-IDF failed)"
# TF-IDF + log1p: values are continuous, max well below raw count scale
assert float(atac_pre.X.max()) < 20, \
    f"ATAC X max={float(atac_pre.X.max())} unexpected for TF-IDF+log1p"
print(f"  ✓ all assertions passed ({time.time()-t:.1f}s)")


# ── T2.4: propagate_cell_type ──────────────────────────────────────────────
print("\n=== T2.4: propagate_cell_type ===")
t = time.time()
# Stage a fake cell_type on the union
union_check = sc.read_h5ad(union_path)
union_check.obs["cell_type"] = (np.arange(union_check.n_obs) % 5).astype(str)
union_check.obs["cell_type"] = union_check.obs["cell_type"].astype("category")
union_check.write_h5ad(union_path, compression="gzip")
print(f"  staged fake cell_type with 5 levels onto union")

propagate_cell_type(
    union_path=union_path,
    per_modality_paths=[rna_pre_path, atac_pre_path],
    verbose=True,
)

# Verify both per-modality h5ads got the label
rna_after  = sc.read_h5ad(rna_pre_path)
atac_after = sc.read_h5ad(atac_pre_path)
assert "cell_type" in rna_after.obs.columns,  "RNA cell_type missing"
assert "cell_type" in atac_after.obs.columns, "ATAC cell_type missing"
assert rna_after.obs["cell_type"].notna().all(),  "some RNA cells have NaN cell_type"
assert atac_after.obs["cell_type"].notna().all(), "some ATAC cells have NaN cell_type"
print(f"  RNA  cell_type levels: {sorted(rna_after.obs['cell_type'].unique())}")
print(f"  ATAC cell_type levels: {sorted(atac_after.obs['cell_type'].unique())}")
print(f"  ✓ all assertions passed ({time.time()-t:.1f}s)")


print("\n=== Tier 2 ALL PASSED ===")
print(f"tmp outputs in {TMP} — will be cleaned up below")
shutil.rmtree(TMP)
print("[cleanup] tmp removed")
