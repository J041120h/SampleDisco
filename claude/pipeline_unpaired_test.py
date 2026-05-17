"""End-to-end pipeline for /dcs07/hongkai/data/harry/result/multi_omics_unpaired_test
mirroring the unpaired_paper run, but memory-safe (does not load the X matrix).

  1. Read obs + obsm['X_glue'] only (via anndata backed='r' / h5py).
  2. Derive 'batch' column from sample names (RNA suffix; ATAC SRR* → 'Su').
  3. GPU cell-level Harmony on X_glue → X_glue_harmony.
  4. Write 'batch' and 'X_glue_harmony' back into the existing h5ad via h5py
     (no rewriting of the heavy X matrix).
  5. Sample embedding (default-α + autotuned) using X_glue_harmony.
"""
from __future__ import annotations
import os, sys, re, gc, time
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import h5py

H5   = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/preprocess/atac_rna_integrated.h5ad'
ROOT = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics'
SRR_PATTERN = re.compile(r'^SRR\d+')


def derive_batch(sample_name: str, modality: str) -> str:
    if modality == 'ATAC' and SRR_PATTERN.match(sample_name):
        return 'Su'              # SRR144664* = Su et al. 2020 PBMC ATAC
    m = re.match(r'.*-(\w+)$', sample_name)
    return m.group(1) if m else 'UNK'


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_minimal(h5: str) -> sc.AnnData:
    """Load adata WITHOUT X (saves 30+ GB on this dataset)."""
    a = ad.read_h5ad(h5, backed='r')
    n = a.shape[0]
    obs_df = a.obs.copy()
    obsm = {k: np.asarray(a.obsm[k]) for k in a.obsm.keys()}
    a.file.close()
    out = sc.AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs_df)
    for k, v in obsm.items():
        out.obsm[k] = v
    return out


def h5_replace_obsm(h5: str, key: str, mat: np.ndarray) -> None:
    """Write/replace adata.obsm[key] in-place via h5py — does not touch X."""
    with h5py.File(h5, 'a') as f:
        if f"obsm/{key}" in f:
            del f[f"obsm/{key}"]
        f.create_dataset(f"obsm/{key}", data=mat.astype(np.float32))


def h5_write_obs_string_column(h5: str, col: str, values: np.ndarray) -> None:
    """Write a string-valued obs column (categorical) in-place via h5py.
    Removes any pre-existing column at obs/<col> first."""
    cats = np.array(sorted(set(values)))
    code_of = {c: i for i, c in enumerate(cats)}
    codes = np.array([code_of[v] for v in values], dtype='int8')
    with h5py.File(h5, 'a') as f:
        target = f"obs/{col}"
        if target in f:
            del f[target]
        grp = f.create_group(target)
        grp.attrs['encoding-type'] = 'categorical'
        grp.attrs['encoding-version'] = '0.2.0'
        grp.attrs['ordered'] = False
        grp.create_dataset('codes', data=codes)
        grp.create_dataset('categories', data=np.array(cats, dtype='S'))
        # Add new col to obs column-order list
        obs_grp = f['obs']
        order = list(obs_grp.attrs.get('column-order', []))
        if col not in order:
            order.append(col)
            obs_grp.attrs['column-order'] = np.array(order, dtype='O')


# ---------------------------------------------------------------------------
log(f"loading {H5} (X dropped for memory)")
adata = load_minimal(H5)
log(f"  shape={adata.shape}; obsm={list(adata.obsm.keys())}")

log("deriving 'batch' from sample names (RNA: suffix; ATAC SRR*: 'Su')")
samples  = adata.obs['sample'].astype(str).values
modality = adata.obs['modality'].astype(str).values
batch_vals = np.array([derive_batch(s, m) for s, m in zip(samples, modality)])
adata.obs['batch'] = pd.Categorical(batch_vals)
log(f"  batch distribution:\n{adata.obs['batch'].value_counts().to_string()}")

from preparation.multi_omics_batch_correction import (
    harmonize_xglue, XGLUE_HARMONY_KEY, XGLUE_HARMONY_NOSAMP,
)
log("running DUAL GPU Harmony on X_glue (batch_col='batch', sample_col='sample', max_iter=50)")
adata = harmonize_xglue(
    adata, batch_col='batch', sample_col='sample',
    use_gpu=True, max_iter=50, verbose=True,
)

# Sanity ASW for both passes
from sklearn.metrics import silhouette_score
np.random.seed(0)
idx = np.random.choice(adata.n_obs, 20000, replace=False)
asw_b_orig    = silhouette_score(adata.obsm['X_glue'][idx], batch_vals[idx])
asw_b_harm    = silhouette_score(adata.obsm[XGLUE_HARMONY_KEY][idx], batch_vals[idx])
asw_b_nosamp  = silhouette_score(adata.obsm[XGLUE_HARMONY_NOSAMP][idx], batch_vals[idx])
asw_m_orig    = silhouette_score(adata.obsm['X_glue'][idx], modality[idx])
asw_m_harm    = silhouette_score(adata.obsm[XGLUE_HARMONY_KEY][idx], modality[idx])
asw_m_nosamp  = silhouette_score(adata.obsm[XGLUE_HARMONY_NOSAMP][idx], modality[idx])
log(f"  ASW(batch)    X_glue={asw_b_orig:+.4f}  X_glue_harmony={asw_b_harm:+.4f}  X_glue_harmony_nosamp={asw_b_nosamp:+.4f}")
log(f"  ASW(modality) X_glue={asw_m_orig:+.4f}  X_glue_harmony={asw_m_harm:+.4f}  X_glue_harmony_nosamp={asw_m_nosamp:+.4f}")

log(f"writing 'batch' + both Harmony obsm back to {H5} via h5py (no X rewrite)")
h5_write_obs_string_column(H5, 'batch', batch_vals)
h5_replace_obsm(H5, XGLUE_HARMONY_KEY,    adata.obsm[XGLUE_HARMONY_KEY])
h5_replace_obsm(H5, XGLUE_HARMONY_NOSAMP, adata.obsm[XGLUE_HARMONY_NOSAMP])
log("  saved")

# Free memory before SE (we only need obs + obsm which adata already has)
gc.collect()

from sample_embedding import compute_sample_embedding
from parameter_selection.autotune import run_autotune

out_default = f"{ROOT}/sampledisco_default_dualharmony"
log(f"default-α SE → {out_default}  (cluster={XGLUE_HARMONY_KEY}, cmd={XGLUE_HARMONY_NOSAMP})")
os.makedirs(out_default, exist_ok=True)
compute_sample_embedding(
    adata, out_default,
    sample_col="sample", celltype_col="cell_type",
    cluster_emb_key=XGLUE_HARMONY_KEY,
    cmd_emb_key=XGLUE_HARMONY_NOSAMP,
    modality_col="modality", batch_col="batch",
    save=True, verbose=True,
)

out_tuned = f"{ROOT}/sampledisco_tuned_dualharmony"
log(f"autotuned SE → {out_tuned}")
os.makedirs(out_tuned, exist_ok=True)
run_autotune(
    adata, out_tuned,
    sample_col="sample", celltype_col="cell_type",
    cluster_emb_key=XGLUE_HARMONY_KEY,
    cmd_emb_key=XGLUE_HARMONY_NOSAMP,
    modality_col="modality", batch_col="batch",
    grouping_col="sev.level",
    save=True, verbose=True,
)
log("DONE")
