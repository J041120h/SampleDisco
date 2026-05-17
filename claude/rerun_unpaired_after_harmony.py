"""After cell-level Harmony has populated adata.obsm['X_glue_harmony'] in the
unpaired GLUE-integrated h5ad, re-derive the SampleDisco sample embedding
using THE HARMONIZED cell embedding, both default-α and autotuned. Then drop
new SE files into the expected sampledisco_default / sampledisco_tuned dirs
and regenerate the figure3/embedding/sampledisco/ visualizations.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import numpy as np, pandas as pd, scanpy as sc, anndata as ann
from sample_embedding import compute_sample_embedding
from parameter_selection.autotune import run_autotune

H5     = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_paper/multiomics/preprocess/atac_rna_integrated.h5ad'
ROOT   = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_paper/multiomics'

def load_minimal(h5: str) -> sc.AnnData:
    """Load adata WITHOUT X (saves >10 GB)."""
    a = ann.read_h5ad(h5, backed='r')
    n = a.shape[0]
    obs_df = a.obs.copy()
    obsm = {k: np.asarray(a.obsm[k]) for k in a.obsm.keys()}
    a.file.close()
    new = sc.AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs_df)
    for k, v in obsm.items():
        new.obsm[k] = v
    return new

print(f"[{time.strftime('%H:%M:%S')}] loading {H5} (X dropped for memory)", flush=True)
ad = load_minimal(H5)
print(f"  shape={ad.shape}  obsm={list(ad.obsm.keys())}", flush=True)
if 'X_glue_harmony' not in ad.obsm:
    raise SystemExit("X_glue_harmony missing — run /tmp/run_harmony_xglue.py first")

# ----- Default-α -----
out_default = f"{ROOT}/sampledisco_default_xglueharmony"
os.makedirs(out_default, exist_ok=True)
print(f"\n[{time.strftime('%H:%M:%S')}] default-α SE → {out_default}", flush=True)
compute_sample_embedding(
    ad, out_default,
    sample_col="sample", celltype_col="cell_type",
    cluster_emb_key="X_glue_harmony",
    cmd_emb_key="X_glue_harmony",
    modality_col="modality",
    batch_col="batch",
    save=True, verbose=True,
)

# ----- Tuned -----
out_tuned = f"{ROOT}/sampledisco_tuned_xglueharmony"
os.makedirs(out_tuned, exist_ok=True)
print(f"\n[{time.strftime('%H:%M:%S')}] autotuned SE → {out_tuned}", flush=True)
run_autotune(
    ad, out_tuned,
    sample_col="sample", celltype_col="cell_type",
    cluster_emb_key="X_glue_harmony",
    cmd_emb_key="X_glue_harmony",
    modality_col="modality",
    batch_col="batch",
    grouping_col="sev.level",
    save=True, verbose=True,
)
print(f"\n[{time.strftime('%H:%M:%S')}] DONE", flush=True)
