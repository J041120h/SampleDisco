#!/usr/bin/env python3
"""
Comprehensive Gene Activity Validation Function.

Combines:
- Per-cell RNA-ATAC correlation
- Per-gene RNA-ATAC correlation  
- Pseudobulk/KNN aggregation correlation analysis (k=1 to 50)
- Detailed overlap statistics

Memory-efficient implementation with optional GPU acceleration.
"""

import os
import re
import json
import gc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy import stats as sp_stats
from typing import Optional, List, Dict, Any
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# ID UTILITIES (for gene name harmonization)
# =============================================================================

_ENSEMBL_RE = re.compile(r"^ENSG\d+(?:\.\d+)?$", re.IGNORECASE)

def _looks_like_ensembl(x: str) -> bool:
    return bool(_ENSEMBL_RE.match(x or ""))

def _strip_ens_version(x: str) -> str:
    if x is None:
        return x
    return x.split(".", 1)[0] if x.upper().startswith("ENSG") else x

def _normalize_symbol(x: str) -> str:
    return (x or "").upper()

def _candidate_cols(var_df: pd.DataFrame) -> dict:
    cols = {c.lower(): c for c in var_df.columns}
    ens_cols = [cols[k] for k in ["gene_id", "ensembl", "ensembl_id", "gene_ids"] if k in cols]
    sym_cols = [cols[k] for k in ["gene_name", "symbol", "gene_symbol", "genesymbol"] if k in cols]
    return {"ensembl": ens_cols, "symbol": sym_cols}

def _derive_ids(var_names: pd.Index, var_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=var_names)
    df["ens_from_varname"] = pd.Series(
        [_strip_ens_version(v) if _looks_like_ensembl(str(v)) else np.nan for v in var_names],
        index=var_names, dtype="object"
    )
    df["sym_from_varname"] = pd.Series(
        [_normalize_symbol(v) if not _looks_like_ensembl(str(v)) else np.nan for v in var_names],
        index=var_names, dtype="object"
    )
    cand = _candidate_cols(var_df)
    
    ens_col_val = None
    for c in cand["ensembl"]:
        if c in var_df.columns:
            ens_col_val = var_df[c].astype(str).map(_strip_ens_version)
            break
    
    sym_col_val = None
    for c in cand["symbol"]:
        if c in var_df.columns:
            sym_col_val = var_df[c].astype(str).map(_normalize_symbol)
            break
    
    df["ens_from_cols"] = ens_col_val.reindex(var_names) if ens_col_val is not None else np.nan
    df["sym_from_cols"] = sym_col_val.reindex(var_names) if sym_col_val is not None else np.nan
    return df

def _choose_unified_key(rna_ids: pd.DataFrame, atac_ids: pd.DataFrame, prefer: str = "auto") -> str:
    rna_ens = pd.Series(rna_ids["ens_from_cols"]).fillna(rna_ids["ens_from_varname"])
    rna_sym = pd.Series(rna_ids["sym_from_cols"]).fillna(rna_ids["sym_from_varname"])
    atac_ens = pd.Series(atac_ids["ens_from_cols"]).fillna(atac_ids["ens_from_varname"])
    atac_sym = pd.Series(atac_ids["sym_from_cols"]).fillna(atac_ids["sym_from_varname"])
    
    ens_overlap = len(set(rna_ens.dropna()) & set(atac_ens.dropna()))
    sym_overlap = len(set(rna_sym.dropna()) & set(atac_sym.dropna()))
    
    if prefer in ("ensembl", "symbol"):
        return prefer
    return "ensembl" if ens_overlap >= sym_overlap else "symbol"


def unify_and_align_genes(
    adata_rna: sc.AnnData,
    adata_atac: sc.AnnData,
    output_dir: str,
    prefer: str = "auto",
    mapping_csv: str | None = None,
    atac_layer: str | None = "GeneActivity",
    verbose: bool = True,
):
    """Unify gene IDs between RNA and ATAC modalities."""
    os.makedirs(output_dir, exist_ok=True)

    if atac_layer and atac_layer in (adata_atac.layers.keys() if hasattr(adata_atac.layers, "keys") else {}):
        if verbose:
            print(f"[unify] Using ATAC layer '{atac_layer}' as X")
        X = adata_atac.layers[atac_layer]
        adata_atac = sc.AnnData(X=X, obs=adata_atac.obs.copy(), var=adata_atac.var.copy())

    rna_ids = _derive_ids(adata_rna.var_names, adata_rna.var)
    atac_ids = _derive_ids(adata_atac.var_names, adata_atac.var)

    sym2ens = {}
    ens2sym = {}
    if mapping_csv and os.path.exists(mapping_csv):
        mdf = pd.read_csv(mapping_csv)
        if {"gene_id", "gene_name"}.issubset(set(mdf.columns)):
            mdf = mdf.dropna(subset=["gene_id", "gene_name"]).copy()
            mdf["gene_id"] = mdf["gene_id"].astype(str).map(_strip_ens_version)
            mdf["gene_name"] = mdf["gene_name"].astype(str).map(_normalize_symbol)
            mdf = mdf.drop_duplicates(subset=["gene_id"], keep="first")
            ens2sym = pd.Series(mdf["gene_name"].values, index=mdf["gene_id"].values).to_dict()
            sym2ens = pd.Series(mdf["gene_id"].values, index=mdf["gene_name"].values).to_dict()
            if verbose:
                print(f"[unify] Loaded mapping_csv with {len(mdf)} rows")

    target = _choose_unified_key(rna_ids, atac_ids, prefer=prefer)
    if verbose:
        print(f"[unify] Unifying on: {target.upper()}")

    if target == "ensembl":
        rna_id = rna_ids["ens_from_cols"].fillna(rna_ids["ens_from_varname"])
        atac_id = atac_ids["ens_from_cols"].fillna(atac_ids["ens_from_varname"])
        if mapping_csv:
            rna_missing = rna_id.isna()
            atac_missing = atac_id.isna()
            if rna_missing.any():
                sym = rna_ids.loc[rna_missing, "sym_from_cols"].fillna(rna_ids.loc[rna_missing, "sym_from_varname"])
                rna_id.loc[rna_missing] = sym.map(sym2ens)
            if atac_missing.any():
                sym = atac_ids.loc[atac_missing, "sym_from_cols"].fillna(atac_ids.loc[atac_missing, "sym_from_varname"])
                atac_id.loc[atac_missing] = sym.map(sym2ens)
        rna_id = rna_id.dropna().astype(str).map(_strip_ens_version)
        atac_id = atac_id.dropna().astype(str).map(_strip_ens_version)
    else:
        rna_id = rna_ids["sym_from_cols"].fillna(rna_ids["sym_from_varname"])
        atac_id = atac_ids["sym_from_cols"].fillna(atac_ids["sym_from_varname"])
        if mapping_csv:
            rna_missing = rna_id.isna()
            atac_missing = atac_id.isna()
            if rna_missing.any():
                ens = rna_ids.loc[rna_missing, "ens_from_cols"].fillna(rna_ids.loc[rna_missing, "ens_from_varname"])
                rna_id.loc[rna_missing] = ens.map(ens2sym)
            if atac_missing.any():
                ens = atac_ids.loc[atac_missing, "ens_from_cols"].fillna(atac_ids.loc[atac_missing, "ens_from_varname"])
                atac_id.loc[atac_missing] = ens.map(ens2sym)
        rna_id = rna_id.dropna().astype(str).map(_normalize_symbol)
        atac_id = atac_id.dropna().astype(str).map(_normalize_symbol)

    map_rna = pd.DataFrame({"orig": adata_rna.var_names, "unified_id": rna_id.reindex(adata_rna.var_names).values})
    map_atac = pd.DataFrame({"orig": adata_atac.var_names, "unified_id": atac_id.reindex(adata_atac.var_names).values})

    shared = sorted(set(map_rna["unified_id"].dropna()) & set(map_atac["unified_id"].dropna()))
    if verbose:
        print(f"[unify] Shared unified IDs: {len(shared)}")

    if len(shared) == 0:
        raise ValueError("[unify] No overlap after ID harmonization.")

    rna_idx = pd.Index(map_rna["unified_id"]).get_indexer(shared)
    atac_idx = pd.Index(map_atac["unified_id"]).get_indexer(shared)

    rna_aligned = adata_rna[:, rna_idx].copy()
    atac_aligned = adata_atac[:, atac_idx].copy()
    rna_aligned.var_names = pd.Index(shared)
    atac_aligned.var_names = pd.Index(shared)

    md = pd.DataFrame({
        "unified_id": shared,
        "rna_orig": map_rna.set_index("unified_id").reindex(shared)["orig"].values,
        "atac_orig": map_atac.set_index("unified_id").reindex(shared)["orig"].values,
        "target_space": target
    })
    md_path = os.path.join(output_dir, "gene_id_mapping_unified.csv")
    md.to_csv(md_path, index=False)
    if verbose:
        print(f"[unify] Saved mapping → {md_path}")

    return rna_aligned, atac_aligned, shared, md


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _to_array(X):
    """Convert sparse matrix to dense array."""
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _compute_pca_cpu(X: np.ndarray, n_pcs: int = 30, n_hvgs: int = 2000) -> np.ndarray:
    """Compute PCA embedding on CPU (memory-efficient for smaller datasets)."""
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    
    n_cells, n_genes = X.shape
    
    # Normalize (CPM + log1p)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    X = X / row_sums * 1e4
    np.log1p(X, out=X)
    
    # Select HVGs by variance
    gene_vars = X.var(axis=0)
    n_hvgs = min(n_hvgs, n_genes)
    hvg_indices = np.argsort(gene_vars)[-n_hvgs:]
    X_hvg = X[:, hvg_indices]
    
    # Standardize
    means = X_hvg.mean(axis=0)
    stds = X_hvg.std(axis=0)
    stds = np.where(stds == 0, 1, stds)
    X_hvg = (X_hvg - means) / stds
    
    # PCA via SVD
    n_components = min(n_pcs, n_hvgs - 1, n_cells - 1)
    U, S, Vt = np.linalg.svd(X_hvg, full_matrices=False)
    pca_coords = U[:, :n_components] * S[:n_components]
    
    return pca_coords.astype(np.float32)


def _compute_knn_cpu(pca_coords: np.ndarray, k: int, batch_size: int = 5000) -> np.ndarray:
    """Compute KNN indices on CPU using batched distance computation."""
    n_samples = pca_coords.shape[0]
    knn_indices = np.zeros((n_samples, k), dtype=np.int64)
    
    X_norm_sq = (pca_coords ** 2).sum(axis=1)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = pca_coords[start:end]
        batch_norm_sq = X_norm_sq[start:end]
        
        # Squared distances
        dists_sq = batch_norm_sq[:, None] + X_norm_sq[None, :] - 2 * (batch @ pca_coords.T)
        
        # Get k smallest
        if k < n_samples:
            partition_idx = np.argpartition(dists_sq, k, axis=1)[:, :k]
            batch_indices = np.arange(end - start)[:, None]
            k_dists = dists_sq[batch_indices, partition_idx]
            sort_idx = np.argsort(k_dists, axis=1)
            knn_indices[start:end] = partition_idx[batch_indices, sort_idx]
        else:
            knn_indices[start:end] = np.argsort(dists_sq, axis=1)[:, :k]
    
    return knn_indices


def _aggregate_and_correlate_cpu(
    rna_X: np.ndarray,
    atac_X: np.ndarray,
    knn_indices: np.ndarray,
    k: int,
    min_nonzero: int = 10,
    batch_size: int = 5000,
) -> tuple:
    """Aggregate cells by KNN and compute correlations (CPU version)."""
    n_cells = rna_X.shape[0]
    knn_k = knn_indices[:, :k]
    
    all_corrs = []
    rna_zero_count = 0
    atac_zero_count = 0
    total_elements = 0
    
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        batch_knn = knn_k[start:end]
        
        # Gather and aggregate
        rna_neighbors = rna_X[batch_knn]
        atac_neighbors = atac_X[batch_knn]
        rna_agg = rna_neighbors.mean(axis=1).astype(np.float32)
        atac_agg = atac_neighbors.mean(axis=1).astype(np.float32)
        
        del rna_neighbors, atac_neighbors
        
        # Track sparsity
        rna_zero_count += (rna_agg == 0).sum()
        atac_zero_count += (atac_agg == 0).sum()
        total_elements += rna_agg.size
        
        # Compute correlations
        batch_corrs = np.full(end - start, np.nan, dtype=np.float64)
        
        for i in range(rna_agg.shape[0]):
            r = rna_agg[i]
            a = atac_agg[i]
            mask = (r != 0) | (a != 0)
            n_valid = mask.sum()
            
            if n_valid >= min_nonzero:
                r_m = r[mask]
                a_m = a[mask]
                r_mean = r_m.mean()
                a_mean = a_m.mean()
                r_c = r_m - r_mean
                a_c = a_m - a_mean
                cov = (r_c * a_c).sum()
                r_std = np.sqrt((r_c ** 2).sum())
                a_std = np.sqrt((a_c ** 2).sum())
                if r_std > 1e-10 and a_std > 1e-10:
                    batch_corrs[i] = cov / (r_std * a_std)
        
        all_corrs.append(batch_corrs)
    
    corrs = np.concatenate(all_corrs)
    rna_sparsity = rna_zero_count / total_elements * 100 if total_elements > 0 else 0
    atac_sparsity = atac_zero_count / total_elements * 100 if total_elements > 0 else 0
    
    return corrs, rna_sparsity, atac_sparsity


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_gene_activity(
    adata_rna: sc.AnnData,
    adata_atac: sc.AnnData,
    output_dir: str,
    # Correlation parameters
    min_cells_for_gene_corr: int = 3,
    sample_genes: int | None = None,
    # Pseudobulk parameters
    run_pseudobulk: bool = True,
    k_values: Optional[List[int]] = None,
    max_k: int = 50,
    n_pcs: int = 30,
    n_hvgs: int = 2000,
    use_rna_for_knn: bool = True,
    # ID unification
    unify_if_needed: bool = True,
    unify_prefer: str = "auto",
    unify_mapping_csv: str | None = None,
    atac_layer: str | None = "GeneActivity",
    # Other
    verbose: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Comprehensive gene activity validation function.
    
    Computes:
    1. Overlap statistics (cells and genes)
    2. Per-cell RNA-ATAC correlation (Pearson across genes)
    3. Per-gene RNA-ATAC correlation (Spearman across cells)
    4. Pseudobulk/KNN aggregation correlation (k=1 to max_k)
    
    Parameters
    ----------
    adata_rna : AnnData
        RNA expression data
    adata_atac : AnnData
        ATAC gene activity data
    output_dir : str
        Directory to save results
    min_cells_for_gene_corr : int
        Minimum co-expressing cells for per-gene correlation
    sample_genes : int or None
        If set, randomly sample this many genes for analysis
    run_pseudobulk : bool
        Whether to run KNN aggregation analysis
    k_values : list or None
        K values for pseudobulk. If None, auto-generated up to max_k
    max_k : int
        Maximum k value for pseudobulk (default 50)
    n_pcs : int
        Number of PCs for KNN graph
    n_hvgs : int
        Number of HVGs for PCA
    use_rna_for_knn : bool
        Use RNA (True) or ATAC (False) for KNN graph
    unify_if_needed : bool
        Attempt ID unification if no gene overlap
    unify_prefer : str
        'auto', 'ensembl', or 'symbol'
    unify_mapping_csv : str or None
        Optional mapping file for ID conversion
    atac_layer : str or None
        ATAC layer to use (e.g., 'GeneActivity')
    verbose : bool
        Print progress
    random_state : int
        Random seed
        
    Returns
    -------
    dict with keys:
        - overlap_stats: dict with cell/gene overlap info
        - per_cell_corr: DataFrame with per-cell correlations
        - per_gene_corr: DataFrame with per-gene correlations
        - pseudobulk_results: DataFrame with k vs correlation (if run_pseudobulk)
        - summary: dict with summary statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if k_values is None:
        # Generate k values: 1, 2, 3, 5, 10, 15, 20, 30, 40, 50 (up to max_k)
        k_values = [k for k in [1, 2, 3, 5, 10, 15, 20, 30, 40, 50] if k <= max_k]
    
    # =========================================================================
    # 1. OVERLAP STATISTICS
    # =========================================================================
    if verbose:
        print("\n" + "="*60)
        print("GENE ACTIVITY VALIDATION")
        print("="*60)
        print("\n[1/4] Computing overlap statistics...")
    
    # Cell overlap
    rna_cells = set(map(str, adata_rna.obs_names))
    atac_cells = set(map(str, adata_atac.obs_names))
    common_cells = sorted(rna_cells & atac_cells)
    
    overlap_stats = {
        "rna_total_cells": len(rna_cells),
        "atac_total_cells": len(atac_cells),
        "common_cells": len(common_cells),
        "cell_overlap_pct_rna": len(common_cells) / len(rna_cells) * 100 if rna_cells else 0,
        "cell_overlap_pct_atac": len(common_cells) / len(atac_cells) * 100 if atac_cells else 0,
        "rna_total_genes": adata_rna.n_vars,
        "atac_total_genes": adata_atac.n_vars,
    }
    
    if verbose:
        print(f"    RNA cells: {overlap_stats['rna_total_cells']}")
        print(f"    ATAC cells: {overlap_stats['atac_total_cells']}")
        print(f"    Common cells: {overlap_stats['common_cells']} "
              f"({overlap_stats['cell_overlap_pct_rna']:.1f}% of RNA, "
              f"{overlap_stats['cell_overlap_pct_atac']:.1f}% of ATAC)")
    
    if len(common_cells) == 0:
        raise ValueError("No paired cells found (no overlap in obs_names).")
    
    # Subset to common cells
    rna_sub = adata_rna[common_cells, :].copy()
    atac_sub = adata_atac[common_cells, :].copy()
    
    # Handle ATAC layer
    if atac_layer and atac_layer in (atac_sub.layers.keys() if hasattr(atac_sub.layers, "keys") else {}):
        if verbose:
            print(f"    Using ATAC layer: '{atac_layer}'")
        atac_sub = sc.AnnData(
            X=atac_sub.layers[atac_layer],
            obs=atac_sub.obs.copy(),
            var=atac_sub.var.copy()
        )
    
    # Gene overlap
    rna_genes = set(map(str, rna_sub.var_names))
    atac_genes = set(map(str, atac_sub.var_names))
    shared_genes = sorted(rna_genes & atac_genes)
    
    overlap_stats["shared_genes_direct"] = len(shared_genes)
    
    if verbose:
        print(f"    RNA genes: {overlap_stats['rna_total_genes']}")
        print(f"    ATAC genes: {overlap_stats['atac_total_genes']}")
        print(f"    Direct gene overlap: {len(shared_genes)}")
    
    # Attempt unification if needed
    if len(shared_genes) == 0 and unify_if_needed:
        if verbose:
            print("    No direct gene overlap. Attempting ID unification...")
        rna_sub, atac_sub, shared_genes, mapping_df = unify_and_align_genes(
            rna_sub, atac_sub,
            output_dir=output_dir,
            prefer=unify_prefer,
            mapping_csv=unify_mapping_csv,
            atac_layer=None,  # Already handled above
            verbose=verbose,
        )
        overlap_stats["shared_genes_after_unification"] = len(shared_genes)
        overlap_stats["unification_performed"] = True
    else:
        overlap_stats["shared_genes_after_unification"] = len(shared_genes)
        overlap_stats["unification_performed"] = False
    
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between RNA and ATAC.")
    
    # Optional gene sampling
    if sample_genes is not None and sample_genes < len(shared_genes):
        rng = np.random.default_rng(random_state)
        shared_genes = sorted(rng.choice(shared_genes, size=sample_genes, replace=False).tolist())
        if verbose:
            print(f"    Sampled {len(shared_genes)} genes for analysis")
    
    overlap_stats["genes_used_for_analysis"] = len(shared_genes)
    overlap_stats["gene_overlap_pct_rna"] = len(shared_genes) / overlap_stats["rna_total_genes"] * 100
    overlap_stats["gene_overlap_pct_atac"] = len(shared_genes) / overlap_stats["atac_total_genes"] * 100
    
    if verbose:
        print(f"    Final genes for analysis: {len(shared_genes)} "
              f"({overlap_stats['gene_overlap_pct_rna']:.1f}% of RNA, "
              f"{overlap_stats['gene_overlap_pct_atac']:.1f}% of ATAC)")
    
    # Align genes
    rna_sub = rna_sub[:, shared_genes].copy()
    atac_sub = atac_sub[:, shared_genes].copy()
    
    # Extract and normalize matrices
    rna_X = _to_array(rna_sub.X).astype(np.float32)
    atac_X = _to_array(atac_sub.X).astype(np.float32)
    
    # Normalize (CPM + log1p)
    rna_sums = rna_X.sum(axis=1, keepdims=True)
    rna_sums = np.maximum(rna_sums, 1e-10)
    rna_X = rna_X / rna_sums * 1e4
    np.log1p(rna_X, out=rna_X)
    
    atac_sums = atac_X.sum(axis=1, keepdims=True)
    atac_sums = np.maximum(atac_sums, 1e-10)
    atac_X = atac_X / atac_sums * 1e4
    np.log1p(atac_X, out=atac_X)
    
    n_cells, n_genes = rna_X.shape
    
    if verbose:
        print(f"\n    Final matrix: {n_cells} cells × {n_genes} genes")
        print(f"    Memory per matrix: {rna_X.nbytes / 1e6:.1f} MB")
    
    # Save overlap stats
    overlap_df = pd.DataFrame([overlap_stats])
    overlap_df.to_csv(os.path.join(output_dir, "overlap_statistics.csv"), index=False)
    
    # =========================================================================
    # 2. PER-CELL CORRELATIONS
    # =========================================================================
    if verbose:
        print("\n[2/4] Computing per-cell correlations (Pearson across genes)...")
    
    per_cell_corr = np.full(n_cells, np.nan, dtype=np.float64)
    
    for i in tqdm(range(n_cells), desc="    Per-cell", disable=not verbose, leave=False):
        r = rna_X[i, :]
        a = atac_X[i, :]
        mask = (r != 0) | (a != 0)
        if mask.sum() >= 3:
            r_m, a_m = r[mask], a[mask]
            if np.std(r_m) > 0 and np.std(a_m) > 0:
                per_cell_corr[i] = np.corrcoef(r_m, a_m)[0, 1]
    
    per_cell_df = pd.DataFrame({
        "cell": common_cells,
        "pearson_corr": per_cell_corr
    })
    
    valid_cell_corr = np.isfinite(per_cell_corr)
    per_cell_stats = {
        "n_valid": int(valid_cell_corr.sum()),
        "mean": float(np.nanmean(per_cell_corr)) if valid_cell_corr.any() else np.nan,
        "median": float(np.nanmedian(per_cell_corr)) if valid_cell_corr.any() else np.nan,
        "std": float(np.nanstd(per_cell_corr)) if valid_cell_corr.any() else np.nan,
        "pct_positive": float((per_cell_corr[valid_cell_corr] > 0).mean() * 100) if valid_cell_corr.any() else 0,
    }
    
    if verbose:
        print(f"    Valid cells: {per_cell_stats['n_valid']}/{n_cells}")
        print(f"    Mean: {per_cell_stats['mean']:.4f}, Median: {per_cell_stats['median']:.4f}")
        print(f"    % Positive: {per_cell_stats['pct_positive']:.1f}%")
    
    per_cell_df.to_csv(os.path.join(output_dir, "per_cell_correlations.csv"), index=False)
    
    # =========================================================================
    # 3. PER-GENE CORRELATIONS
    # =========================================================================
    if verbose:
        print("\n[3/4] Computing per-gene correlations (Spearman across cells)...")
    
    per_gene_corr = np.full(n_genes, np.nan, dtype=np.float64)
    per_gene_nco = np.zeros(n_genes, dtype=np.int32)
    
    rna_T = rna_X.T
    atac_T = atac_X.T
    
    for j in tqdm(range(n_genes), desc="    Per-gene", disable=not verbose, leave=False):
        r = rna_T[j, :]
        a = atac_T[j, :]
        co_mask = (r != 0) & (a != 0)
        nco = int(co_mask.sum())
        per_gene_nco[j] = nco
        
        if nco >= min_cells_for_gene_corr:
            r_m, a_m = r[co_mask], a[co_mask]
            if np.std(r_m) > 0 and np.std(a_m) > 0:
                try:
                    per_gene_corr[j], _ = sp_stats.spearmanr(r_m, a_m)
                except Exception:
                    pass
    
    per_gene_df = pd.DataFrame({
        "gene": shared_genes,
        "spearman_corr": per_gene_corr,
        "n_coexpressing_cells": per_gene_nco
    })
    
    valid_gene_corr = np.isfinite(per_gene_corr)
    per_gene_stats = {
        "n_valid": int(valid_gene_corr.sum()),
        "mean": float(np.nanmean(per_gene_corr)) if valid_gene_corr.any() else np.nan,
        "median": float(np.nanmedian(per_gene_corr)) if valid_gene_corr.any() else np.nan,
        "std": float(np.nanstd(per_gene_corr)) if valid_gene_corr.any() else np.nan,
        "pct_positive": float((per_gene_corr[valid_gene_corr] > 0).mean() * 100) if valid_gene_corr.any() else 0,
    }
    
    if verbose:
        print(f"    Valid genes: {per_gene_stats['n_valid']}/{n_genes}")
        print(f"    Mean: {per_gene_stats['mean']:.4f}, Median: {per_gene_stats['median']:.4f}")
        print(f"    % Positive: {per_gene_stats['pct_positive']:.1f}%")
    
    per_gene_df.to_csv(os.path.join(output_dir, "per_gene_correlations.csv"), index=False)
    
    # =========================================================================
    # 4. PSEUDOBULK / KNN AGGREGATION CORRELATIONS
    # =========================================================================
    pseudobulk_results = None
    per_k_correlations = {}
    
    if run_pseudobulk:
        if verbose:
            print(f"\n[4/4] Computing pseudobulk correlations (k=1 to {max_k})...")
        
        # Compute PCA for KNN
        if verbose:
            print("    Computing PCA for KNN graph...")
        
        if use_rna_for_knn:
            X_for_pca = rna_sub.X
        else:
            X_for_pca = atac_sub.X
        
        pca_coords = _compute_pca_cpu(X_for_pca, n_pcs=n_pcs, n_hvgs=n_hvgs)
        
        # Build KNN graph
        actual_max_k = min(max(k_values), n_cells - 1)
        k_values = [k for k in k_values if k < n_cells]
        
        if verbose:
            print(f"    Building KNN graph (max_k={actual_max_k})...")
        
        knn_indices = _compute_knn_cpu(pca_coords, k=actual_max_k)
        
        del pca_coords
        gc.collect()
        
        # Compute correlations for each k
        pseudobulk_rows = []
        
        for k in tqdm(k_values, desc="    Pseudobulk k", disable=not verbose):
            corrs, rna_sparsity, atac_sparsity = _aggregate_and_correlate_cpu(
                rna_X, atac_X, knn_indices, k, min_nonzero=10
            )
            
            per_k_correlations[k] = corrs
            
            valid = np.isfinite(corrs)
            n_valid = valid.sum()
            
            if n_valid > 0:
                mean_corr = float(np.nanmean(corrs))
                median_corr = float(np.nanmedian(corrs))
                std_corr = float(np.nanstd(corrs))
                pct_positive = float((corrs[valid] > 0).mean() * 100)
                pct_negative = float((corrs[valid] < 0).mean() * 100)
            else:
                mean_corr = median_corr = std_corr = np.nan
                pct_positive = pct_negative = 0
            
            pseudobulk_rows.append({
                'k': k,
                'n_valid': n_valid,
                'mean_corr': mean_corr,
                'median_corr': median_corr,
                'std_corr': std_corr,
                'pct_positive': pct_positive,
                'pct_negative': pct_negative,
                'rna_sparsity_pct': rna_sparsity,
                'atac_sparsity_pct': atac_sparsity,
            })
        
        pseudobulk_results = pd.DataFrame(pseudobulk_rows)
        pseudobulk_results.to_csv(os.path.join(output_dir, "pseudobulk_correlations.csv"), index=False)
        
        if verbose:
            print("\n    Pseudobulk results:")
            print(f"    k=1:  mean_r={pseudobulk_rows[0]['mean_corr']:.4f}")
            print(f"    k={k_values[-1]}: mean_r={pseudobulk_rows[-1]['mean_corr']:.4f}")
    
    # =========================================================================
    # 5. VISUALIZATION
    # =========================================================================
    if verbose:
        print("\n[5/5] Creating visualizations...")
    
    n_plots = 4 if run_pseudobulk else 3
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Plot 1: Per-cell correlation histogram
    ax = axes[0]
    if valid_cell_corr.any():
        ax.hist(per_cell_corr[valid_cell_corr], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(per_cell_stats['mean'], ls='--', color='red', lw=2, 
                   label=f"Mean={per_cell_stats['mean']:.3f}")
        ax.axvline(0, ls='-', color='gray', alpha=0.5)
    ax.set_xlabel("Pearson Correlation", fontsize=12)
    ax.set_ylabel("Number of Cells", fontsize=12)
    ax.set_title(f"Per-cell RNA-ATAC Correlation\n(n={per_cell_stats['n_valid']} cells)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Per-gene correlation histogram
    ax = axes[1]
    if valid_gene_corr.any():
        ax.hist(per_gene_corr[valid_gene_corr], bins=50, edgecolor='black', alpha=0.7, color='coral')
        ax.axvline(per_gene_stats['mean'], ls='--', color='red', lw=2,
                   label=f"Mean={per_gene_stats['mean']:.3f}")
        ax.axvline(0, ls='-', color='gray', alpha=0.5)
    ax.set_xlabel("Spearman Correlation", fontsize=12)
    ax.set_ylabel("Number of Genes", fontsize=12)
    ax.set_title(f"Per-gene RNA-ATAC Correlation\n(n={per_gene_stats['n_valid']} genes)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 3: Co-expressing cells vs correlation
    ax = axes[2]
    if valid_gene_corr.any():
        ax.scatter(per_gene_nco[valid_gene_corr], per_gene_corr[valid_gene_corr], 
                   s=8, alpha=0.5, c='coral')
        ax.axhline(0, ls='-', color='gray', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel("Number of Co-expressing Cells (log)", fontsize=12)
    ax.set_ylabel("Gene Correlation (Spearman)", fontsize=12)
    ax.set_title("Gene Correlation vs Co-expression", fontsize=14)
    ax.grid(alpha=0.3)
    
    # Plot 4: Pseudobulk correlation vs k
    ax = axes[3]
    if run_pseudobulk and pseudobulk_results is not None:
        ax.plot(pseudobulk_results['k'], pseudobulk_results['mean_corr'], 
                'b-o', label='Mean', linewidth=2, markersize=8)
        ax.plot(pseudobulk_results['k'], pseudobulk_results['median_corr'],
                'g-s', label='Median', linewidth=2, markersize=6)
        ax.fill_between(pseudobulk_results['k'],
                        pseudobulk_results['mean_corr'] - pseudobulk_results['std_corr'],
                        pseudobulk_results['mean_corr'] + pseudobulk_results['std_corr'],
                        alpha=0.2, color='blue')
        ax.axhline(0, ls='--', color='gray', alpha=0.5)
        ax.set_xlabel("Number of Neighbors (k)", fontsize=12)
        ax.set_ylabel("Pearson Correlation", fontsize=12)
        ax.set_title("Pseudobulk Correlation vs Aggregation Size", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Pseudobulk analysis\nnot run", ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
        ax.set_axis_off()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "validation_summary.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Additional pseudobulk plot
    if run_pseudobulk and pseudobulk_results is not None:
        fig2, ax = plt.subplots(figsize=(10, 7))
        
        ax.plot(pseudobulk_results['k'], pseudobulk_results['mean_corr'], 
                'b-o', linewidth=2.5, markersize=10, label='Mean Correlation')
        ax.fill_between(pseudobulk_results['k'],
                        pseudobulk_results['mean_corr'] - pseudobulk_results['std_corr'],
                        pseudobulk_results['mean_corr'] + pseudobulk_results['std_corr'],
                        alpha=0.2, color='blue')
        ax.axhline(0, ls='--', color='red', lw=1.5, alpha=0.7, label='Zero correlation')
        
        # Annotate key points
        k1_row = pseudobulk_results[pseudobulk_results['k'] == 1]
        if len(k1_row) > 0:
            ax.annotate(f"Single-cell: {k1_row['mean_corr'].values[0]:.3f}",
                        xy=(1, k1_row['mean_corr'].values[0]),
                        xytext=(3, k1_row['mean_corr'].values[0] - 0.05),
                        fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
        
        max_row = pseudobulk_results.iloc[-1]
        ax.annotate(f"k={int(max_row['k'])}: {max_row['mean_corr']:.3f}",
                    xy=(max_row['k'], max_row['mean_corr']),
                    xytext=(max_row['k'] * 0.7, max_row['mean_corr'] + 0.05),
                    fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
        
        ax.set_xlabel("Number of Neighbors (k) in Aggregation", fontsize=14)
        ax.set_ylabel("Pearson Correlation (RNA vs ATAC)", fontsize=14)
        ax.set_title("RNA-ATAC Correlation Increases with Cell Aggregation\n(KNN-based pseudobulk)", fontsize=14)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        pseudobulk_plot_path = os.path.join(output_dir, "pseudobulk_correlation_curve.png")
        plt.savefig(pseudobulk_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
    
    # =========================================================================
    # 6. SUMMARY
    # =========================================================================
    summary = {
        "overlap": overlap_stats,
        "per_cell": per_cell_stats,
        "per_gene": per_gene_stats,
        "pseudobulk": {
            "k_values": k_values if run_pseudobulk else None,
            "k1_mean_corr": float(pseudobulk_results.iloc[0]['mean_corr']) if run_pseudobulk else None,
            "max_k_mean_corr": float(pseudobulk_results.iloc[-1]['mean_corr']) if run_pseudobulk else None,
        } if run_pseudobulk else None,
        "paths": {
            "overlap_csv": os.path.join(output_dir, "overlap_statistics.csv"),
            "per_cell_csv": os.path.join(output_dir, "per_cell_correlations.csv"),
            "per_gene_csv": os.path.join(output_dir, "per_gene_correlations.csv"),
            "pseudobulk_csv": os.path.join(output_dir, "pseudobulk_correlations.csv") if run_pseudobulk else None,
            "summary_plot": plot_path,
        }
    }
    
    with open(os.path.join(output_dir, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    if verbose:
        print("\n" + "="*60)
        print("VALIDATION COMPLETE")
        print("="*60)
        print(f"\nResults saved to: {output_dir}")
        print("\nKey findings:")
        print(f"  Cell overlap: {overlap_stats['common_cells']} cells")
        print(f"  Gene overlap: {overlap_stats['genes_used_for_analysis']} genes")
        print(f"  Per-cell correlation: mean={per_cell_stats['mean']:.4f}, {per_cell_stats['pct_positive']:.1f}% positive")
        print(f"  Per-gene correlation: mean={per_gene_stats['mean']:.4f}, {per_gene_stats['pct_positive']:.1f}% positive")
        if run_pseudobulk:
            print(f"  Pseudobulk (k=1): mean={pseudobulk_results.iloc[0]['mean_corr']:.4f}")
            print(f"  Pseudobulk (k={k_values[-1]}): mean={pseudobulk_results.iloc[-1]['mean_corr']:.4f}")
    
    return {
        "overlap_stats": overlap_stats,
        "per_cell_corr": per_cell_df,
        "per_gene_corr": per_gene_df,
        "pseudobulk_results": pseudobulk_results,
        "per_k_correlations": per_k_correlations if run_pseudobulk else None,
        "summary": summary,
    }


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENE ACTIVITY VALIDATION - DEMO")
    print("="*60 + "\n")
    
    # Example usage
    rna_path = '/dcs07/hongkai/data/harry/result/multi_omics_heart/data/rna_raw.h5ad'
    atac_path = '/dcs07/hongkai/data/harry/result/multi_omics_heart/data/heart_gene_activity.h5ad'
    out_dir = "/dcs07/hongkai/data/harry/result/multi_omics_heart/data/gene_activity_validation"
    
    print("[Main] Loading data...")
    adata_rna = sc.read_h5ad(rna_path)
    adata_atac = sc.read_h5ad(atac_path)
    
    print(f"[Main] RNA shape: {adata_rna.shape}")
    print(f"[Main] ATAC shape: {adata_atac.shape}")
    
    result = validate_gene_activity(
        adata_rna=adata_rna,
        adata_atac=adata_atac,
        output_dir=out_dir,
        run_pseudobulk=True,
        max_k=50,
        k_values=[1, 2, 3, 5, 10, 15, 20, 30, 40, 50],
        n_pcs=30,
        n_hvgs=2000,
        use_rna_for_knn=True,
        atac_layer="GeneActivity",
        unify_if_needed=True,
        unify_prefer="auto",
        verbose=True,
    )
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60 + "\n")