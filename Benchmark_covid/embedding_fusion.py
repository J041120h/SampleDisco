#!/usr/bin/env python3
"""
Fuse pseudobulk sample embeddings from expression + cell proportion.

Inputs (from adata.obsm):
  - X_DR_expression
  - X_DR_proportion

Outputs (CSV):
  - fusion_concat_embedding.csv
  - fusion_mfa_embedding.csv

User edits: set H5AD_PATH and OUTPUT_DIR below.
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================
# USER EDITS HERE
# ============================
H5AD_PATH = "/dcs07/hongkai/data/harry/result/multi_omics_SD/multiomics/pseudobulk/pseudobulk_sample.h5ad"
OUTPUT_DIR = "/dcs07/hongkai/data/harry/result/multi_omics_SD/multiomics/rna/fusion_embeddings"

# how many fused PCs to output
N_COMPONENTS = 10

# optional: extra weighting for concat fusion after standardization
# (keeps both blocks comparable even if one has many more dims)
# if None: auto weights by sqrt(dim) so each block has similar total energy
CONCAT_BLOCK_WEIGHTS = None  # e.g., (1.0, 1.0) or (0.7, 1.3)
# ============================


def _require_obsm(adata: sc.AnnData, key: str) -> np.ndarray:
    if key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm['{key}']. Available keys: {list(adata.obsm.keys())}")
    X = adata.obsm[key]
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"adata.obsm['{key}'] must be 2D, got shape={X.shape}")
    if np.any(~np.isfinite(X)):
        bad = np.sum(~np.isfinite(X))
        raise ValueError(f"adata.obsm['{key}'] contains non-finite values (NaN/Inf). Count={bad}")
    return X


def _standardize_block(X: np.ndarray) -> np.ndarray:
    """Column-wise standardization (mean 0, var 1)."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    # Guard: if a column was constant, StandardScaler sets it to 0; that's fine.
    return Xs


def _safe_pca_scores(X: np.ndarray, n_components: int, prefix: str) -> pd.DataFrame:
    n_samples, n_feats = X.shape
    k = int(min(n_components, n_samples, n_feats))
    if k < 1:
        raise ValueError(f"Cannot compute PCA with shape={X.shape} and n_components={n_components}")
    pca = PCA(n_components=k, svd_solver="auto", random_state=0)
    Z = pca.fit_transform(X)
    cols = [f"{prefix}{i+1}" for i in range(Z.shape[1])]
    return pd.DataFrame(Z, columns=cols)


def fuse_concat(
    X_expr: np.ndarray,
    X_prop: np.ndarray,
    n_components: int = 10,
    block_weights=None,
) -> pd.DataFrame:
    """
    Feature-level fusion:
      1) standardize each block
      2) optionally apply block weights
      3) concatenate
      4) PCA -> fused embedding
    """
    Xe = _standardize_block(X_expr)
    Xp = _standardize_block(X_prop)

    if block_weights is None:
        # default: equalize block influence by scaling inversely with sqrt(dim)
        # so a higher-dim block doesn't automatically dominate.
        we = 1.0 / np.sqrt(max(Xe.shape[1], 1))
        wp = 1.0 / np.sqrt(max(Xp.shape[1], 1))
    else:
        if not (isinstance(block_weights, (tuple, list)) and len(block_weights) == 2):
            raise ValueError("block_weights must be a 2-tuple/list like (w_expr, w_prop)")
        we, wp = float(block_weights[0]), float(block_weights[1])

    Xcat = np.concatenate([we * Xe, wp * Xp], axis=1)
    return _safe_pca_scores(Xcat, n_components=n_components, prefix="CONCAT_PC")


def fuse_mfa(
    X_expr: np.ndarray,
    X_prop: np.ndarray,
    n_components: int = 10,
) -> pd.DataFrame:
    """
    MFA (2-block version):
      1) standardize each block
      2) for each block, compute top eigenvalue via PCA(1): lambda1
      3) scale block by 1/sqrt(lambda1)
      4) concatenate scaled blocks
      5) PCA -> fused embedding
    """
    Xe = _standardize_block(X_expr)
    Xp = _standardize_block(X_prop)

    # PCA(1) on each block to get the leading variance (eigenvalue)
    pca_e = PCA(n_components=1, svd_solver="auto", random_state=0).fit(Xe)
    pca_p = PCA(n_components=1, svd_solver="auto", random_state=0).fit(Xp)

    # explained_variance_ gives eigenvalues of covariance matrix
    lam_e = float(pca_e.explained_variance_[0])
    lam_p = float(pca_p.explained_variance_[0])

    if lam_e <= 0 or lam_p <= 0:
        raise ValueError(f"Non-positive lambda1 detected: lam_expr={lam_e}, lam_prop={lam_p}")

    Xe_mfa = Xe / np.sqrt(lam_e)
    Xp_mfa = Xp / np.sqrt(lam_p)

    X_mfa = np.concatenate([Xe_mfa, Xp_mfa], axis=1)
    return _safe_pca_scores(X_mfa, n_components=n_components, prefix="MFA_PC")


def run_fusion(h5ad_path: str, output_dir: str, n_components: int = 10) -> None:
    print(f"🔍 Loading AnnData from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    X_expr = _require_obsm(adata, "X_DR_expression")
    X_prop = _require_obsm(adata, "X_DR_proportion")

    if X_expr.shape[0] != X_prop.shape[0]:
        raise ValueError(f"Sample count mismatch: expr {X_expr.shape[0]} vs prop {X_prop.shape[0]}")

    sample_ids = (
        adata.obs_names.to_list()
        if getattr(adata, "obs_names", None) is not None
        else [f"sample_{i}" for i in range(X_expr.shape[0])]
    )

    os.makedirs(output_dir, exist_ok=True)

    # ---- concat fusion ----
    df_concat = fuse_concat(
        X_expr, X_prop,
        n_components=n_components,
        block_weights=CONCAT_BLOCK_WEIGHTS
    )
    df_concat.index = sample_ids
    concat_path = os.path.join(output_dir, "fusion_concat_embedding.csv")
    df_concat.to_csv(concat_path)
    print(f"✅ Wrote: {concat_path}  (shape={df_concat.shape})")

    # ---- MFA fusion ----
    df_mfa = fuse_mfa(
        X_expr, X_prop,
        n_components=n_components
    )
    df_mfa.index = sample_ids
    mfa_path = os.path.join(output_dir, "fusion_mfa_embedding.csv")
    df_mfa.to_csv(mfa_path)
    print(f"✅ Wrote: {mfa_path}  (shape={df_mfa.shape})")

    print("🎉 Done.")


if __name__ == "__main__":
    try:
        run_fusion(H5AD_PATH, OUTPUT_DIR, n_components=N_COMPONENTS)
    except Exception as e:
        print(f"❌ ERROR: {e}", file=sys.stderr)
        sys.exit(1)
