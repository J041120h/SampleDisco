#!/usr/bin/env python3
"""
GPU-accelerated pseudobulk computation using rapids_singlecell.

Cleaner refactoring that leverages rapids_singlecell's native GPU functions
for normalization, log1p, and HVG selection.
"""

import os
import gc
import signal
import io
import warnings
import contextlib
from typing import Tuple, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse, csr_matrix

import cupy as cp
import rapids_singlecell as rsc

from tf_idf import tfidf_memory_efficient
from utils.random_seed import set_global_seed


# =============================================================================
# GPU Utilities
# =============================================================================

def clear_gpu_memory():
    """Clear GPU memory (CuPy + PyTorch if available)."""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def to_gpu(adata: sc.AnnData) -> sc.AnnData:
    """Move AnnData to GPU (CuPy sparse array)."""
    rsc.get.anndata_to_GPU(adata)
    return adata


def to_cpu(adata: sc.AnnData) -> sc.AnnData:
    """Move AnnData back to CPU."""
    rsc.get.anndata_to_CPU(adata)
    return adata


# =============================================================================
# Helper Utilities
# =============================================================================

def _as_list(x: Optional[Union[str, List[str]]]) -> List[str]:
    """Convert None/str/list to list."""
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def _extract_sample_metadata(
    cell_adata: sc.AnnData,
    sample_adata: sc.AnnData,
    sample_col: str,
    exclude_cols: Optional[List[str]] = None,
) -> sc.AnnData:
    """Extract sample-level metadata from cell-level obs."""
    exclude = set(exclude_cols or []) | {sample_col}
    grouped = cell_adata.obs.groupby(sample_col)
    
    meta = {}
    for col in cell_adata.obs.columns:
        if col in exclude:
            continue
        uniques = grouped[col].apply(lambda x: x.dropna().unique())
        if uniques.apply(lambda u: len(u) <= 1).all():
            meta[col] = uniques.apply(lambda u: u[0] if len(u) else np.nan)
    
    if meta:
        sample_adata.obs = sample_adata.obs.join(pd.DataFrame(meta), how="left")
    
    return sample_adata


# =============================================================================
# Batch Correction
# =============================================================================

def _try_combat(
    adata: sc.AnnData,
    batch_col: str,
    preserve_cols: List[str],
    timeout: float,
    verbose: bool
) -> bool:
    """Attempt ComBat correction with timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutError("ComBat timed out")
    
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler) if timeout else None
        if timeout:
            signal.alarm(int(timeout))
        
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                sc.pp.combat(
                    adata,
                    key=batch_col,
                    covariates=preserve_cols or None,
                    inplace=True
                )
            if verbose:
                print(f"    Applied ComBat" + (f" (kept={preserve_cols})" if preserve_cols else ""))
            return True
        finally:
            if timeout:
                signal.alarm(0)
                if old_handler:
                    signal.signal(signal.SIGALRM, old_handler)
    
    except (TimeoutError, Exception) as e:
        if verbose:
            print(f"    ComBat failed: {type(e).__name__}: {e}")
        return False


def _try_limma(
    adata: sc.AnnData,
    batch_col: str,
    preserve_cols: List[str],
    verbose: bool
) -> bool:
    """Attempt limma-style regression correction."""
    try:
        from utils.limma import limma
        
        X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
        
        if preserve_cols:
            terms = [f'Q("{c}")' for c in preserve_cols]
            keep_formula = "~ " + " + ".join(terms)
        else:
            keep_formula = "1"
        remove_formula = f'~ Q("{batch_col}")'
        
        adata.X = limma(
            pheno=adata.obs,
            exprs=X,
            covariate_formula=keep_formula,
            design_formula=remove_formula,
            rcond=1e-8,
            verbose=False
        )
        
        if verbose:
            print(f"    Applied limma" + (f" (kept={preserve_cols})" if preserve_cols else ""))
        return True
    
    except Exception as e:
        if verbose:
            print(f"    Limma failed: {type(e).__name__}: {e}")
        return False


def apply_batch_correction(
    adata: sc.AnnData,
    batch_col: str,
    preserve_cols: List[str],
    combat_timeout: float,
    verbose: bool
) -> bool:
    """Apply batch correction, trying ComBat first, then limma."""
    batch_counts = adata.obs[batch_col].value_counts()
    n_batches = len(batch_counts)
    
    if verbose:
        print(f"    Batches: {n_batches}, sizes: {batch_counts.min()}-{batch_counts.max()}")
    
    if n_batches <= 1:
        if verbose:
            print("    Skipping: only 1 batch")
        return False
    
    if batch_counts.min() < 2:
        if verbose:
            print(f"    Skipping ComBat: batches <2 samples")
        return _try_limma(adata, batch_col, preserve_cols, verbose)
    
    if _try_combat(adata, batch_col, preserve_cols, combat_timeout, verbose):
        return True
    return _try_limma(adata, batch_col, preserve_cols, verbose)


# =============================================================================
# Pseudobulk Aggregation
# =============================================================================

def aggregate_pseudobulk(
    adata: sc.AnnData,
    sample_col: str,
    celltype_col: str,
    batch_cols: List[str],
    verbose: bool
) -> sc.AnnData:
    """Aggregate cells into pseudobulk samples with cell-type layers."""
    samples = sorted(adata.obs[sample_col].unique())
    cell_types = sorted(adata.obs[celltype_col].unique())
    n_samples, n_genes = len(samples), adata.n_vars
    
    if verbose:
        print(f"  Aggregating {adata.n_obs} cells -> {n_samples} samples x {len(cell_types)} cell types")
    
    sample_idx = {s: i for i, s in enumerate(samples)}
    ct_idx = {ct: i for i, ct in enumerate(cell_types)}
    
    # Build obs with batch columns
    obs = pd.DataFrame(index=samples)
    obs.index.name = 'sample'
    for bc in batch_cols:
        mapping = adata.obs[[sample_col, bc]].drop_duplicates().set_index(sample_col)[bc].to_dict()
        obs[bc] = [mapping.get(s, 'Unknown') for s in samples]
    
    pb = sc.AnnData(
        X=np.zeros((n_samples, n_genes), dtype=np.float32),
        obs=obs,
        var=adata.var.copy()
    )
    
    # Pre-compute indices
    cell_sample = adata.obs[sample_col].map(sample_idx).values
    cell_ct = adata.obs[celltype_col].map(ct_idx).values
    valid = ~(pd.isna(cell_sample) | pd.isna(cell_ct))
    valid_idx = np.where(valid)[0]
    cell_sample_valid = cell_sample[valid].astype(int)
    cell_ct_valid = cell_ct[valid].astype(int)
    
    X = adata.X.tocsr() if issparse(adata.X) else adata.X
    
    # Aggregate per cell type using sparse indicator matrix
    for i, ct in enumerate(cell_types):
        mask = cell_ct_valid == i
        ct_cells = valid_idx[mask]
        ct_samples = cell_sample_valid[mask]
        
        if len(ct_cells) == 0:
            pb.layers[ct] = np.zeros((n_samples, n_genes), dtype=np.float32)
            continue
        
        indicator = csr_matrix(
            (np.ones(len(ct_cells), dtype=np.float32), (ct_samples, np.arange(len(ct_cells)))),
            shape=(n_samples, len(ct_cells))
        )
        
        counts = np.array(indicator.sum(axis=1)).flatten()
        counts[counts == 0] = 1
        
        X_ct = X[ct_cells, :]
        sums = indicator @ X_ct
        if issparse(sums):
            sums = np.asarray(sums.todense())
        
        pb.layers[ct] = (sums / counts[:, None]).astype(np.float32)
    
    if verbose:
        print(f"  Created {len(cell_types)} cell type layers")
    
    return pb


# =============================================================================
# GPU Processing Pipeline
# =============================================================================

def process_celltype_layer(
    layer_data: np.ndarray,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    cell_type: str,
    batch_col: Optional[str],
    preserve_cols: List[str],
    n_features: int,
    normalize: bool,
    target_sum: float,
    atac: bool,
    combat_timeout: float,
    verbose: bool
) -> Optional[Tuple[List[str], np.ndarray]]:
    """Process a single cell-type layer on GPU."""
    
    temp = sc.AnnData(X=layer_data.copy(), obs=obs.copy(), var=var.copy())
    
    # Filter empty genes
    sc.pp.filter_genes(temp, min_cells=1)
    if temp.n_vars == 0:
        if verbose:
            print(f"    Skipping: no genes after filtering")
        return None
    
    # Normalize on GPU
    if normalize:
        if atac:
            tfidf_memory_efficient(temp, scale_factor=target_sum)
        else:
            to_gpu(temp)
            rsc.pp.normalize_total(temp, target_sum=target_sum)
            rsc.pp.log1p(temp)
            to_cpu(temp)
    
    # Remove NaN genes
    nan_mask = _get_nan_mask(temp.X)
    if nan_mask.any():
        temp = temp[:, ~nan_mask].copy()
    
    # Batch correction (CPU - ComBat/limma don't have GPU versions)
    if batch_col and batch_col in temp.obs.columns:
        corrected = apply_batch_correction(temp, batch_col, preserve_cols, combat_timeout, verbose)
        
        if corrected:
            nan_mask = _get_nan_mask(temp.X)
            if nan_mask.any():
                if verbose:
                    print(f"    Removing {nan_mask.sum()} NaN genes post-correction")
                temp = temp[:, ~nan_mask].copy()
    
    # HVG selection on GPU
    n_hvgs = min(n_features, temp.n_vars)
    to_gpu(temp)
    rsc.pp.highly_variable_genes(temp, n_top_genes=n_hvgs)
    to_cpu(temp)
    
    hvg_mask = temp.var["highly_variable"].values
    hvg_genes = temp.var.index[hvg_mask].tolist()
    
    if not hvg_genes:
        if verbose:
            print(f"    Skipping: no HVGs found")
        return None
    
    hvg_expr = temp[:, hvg_mask].X
    if issparse(hvg_expr):
        hvg_expr = hvg_expr.toarray()
    
    prefixed = [f"{cell_type} - {g}" for g in hvg_genes]
    
    if verbose:
        print(f"    Selected {len(hvg_genes)} HVGs")
    
    return prefixed, hvg_expr


def _get_nan_mask(X) -> np.ndarray:
    """Get mask of genes with any NaN values."""
    if issparse(X):
        return np.array(np.isnan(X.toarray()).any(axis=0)).flatten()
    return np.isnan(X).any(axis=0)


# =============================================================================
# Cell Proportions
# =============================================================================

def compute_proportions(
    adata: sc.AnnData,
    samples: List[str],
    cell_types: List[str],
    sample_col: str,
    celltype_col: str
) -> pd.DataFrame:
    """Compute cell type proportions per sample."""
    props = pd.DataFrame(index=cell_types, columns=samples, dtype=float)
    
    for sample in samples:
        mask = adata.obs[sample_col] == sample
        total = mask.sum()
        for ct in cell_types:
            ct_count = (mask & (adata.obs[celltype_col] == ct)).sum()
            props.loc[ct, sample] = ct_count / total if total > 0 else 0.0
    
    return props


# =============================================================================
# Main Entry Points
# =============================================================================

def compute_pseudobulk_gpu(
    adata: sc.AnnData,
    batch_col: Optional[Union[str, List[str]]] = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './',
    n_features: int = 2000,
    normalize: bool = True,
    target_sum: float = 1e4,
    atac: bool = False,
    verbose: bool = False,
    combat_timeout: float = 20.0,
    preserve_cols: Optional[Union[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, sc.AnnData]:
    """
    GPU-accelerated pseudobulk computation.
    
    Parameters
    ----------
    adata : AnnData
        Input single-cell data
    batch_col : str or list, optional
        Column(s) for batch correction
    sample_col : str
        Column for sample IDs
    celltype_col : str
        Column for cell type labels
    output_dir : str
        Output directory
    n_features : int
        Number of HVGs per cell type
    normalize : bool
        Whether to normalize
    target_sum : float
        Target sum for normalization
    atac : bool
        Use TF-IDF for ATAC data
    verbose : bool
        Print progress
    combat_timeout : float
        Timeout for ComBat in seconds
    preserve_cols : str or list, optional
        Columns to preserve during batch correction
    
    Returns
    -------
    expression_df : DataFrame
        Cell type x sample expression (Series per cell)
    proportion_df : DataFrame
        Cell type proportions per sample
    concat_adata : AnnData
        Concatenated pseudobulk AnnData
    """
    if verbose:
        print(f"[Pseudobulk] Using GPU (rapids_singlecell)")
    clear_gpu_memory()
    
    os.makedirs(os.path.join(output_dir, "pseudobulk"), exist_ok=True)
    
    # Normalize batch columns
    batch_cols = [b for b in _as_list(batch_col) 
                  if b and b in adata.obs.columns and not adata.obs[b].isnull().all()]
    
    preserve_list = [c for c in _as_list(preserve_cols) if c in adata.obs.columns]
    
    # Fill NaNs in batch/preserve columns
    for col in batch_cols + preserve_list:
        if adata.obs[col].isnull().any():
            adata.obs[col] = adata.obs[col].fillna("Unknown")
    
    samples = sorted(adata.obs[sample_col].unique())
    cell_types = sorted(adata.obs[celltype_col].unique())
    
    if verbose:
        print(f"[Pseudobulk] {len(samples)} samples, {len(cell_types)} cell types")
        print(f"[Pseudobulk] Batch correction: {batch_cols if batch_cols else 'disabled'}")
    
    # Phase 1: Aggregate
    if verbose:
        print("[Pseudobulk] Phase 1: Aggregating...")
    pb = aggregate_pseudobulk(adata, sample_col, celltype_col, batch_cols, verbose)
    
    # Create combined batch column if needed
    combined_batch = None
    if batch_cols:
        combined_batch = "_combined_batch_"
        if len(batch_cols) == 1:
            pb.obs[combined_batch] = pb.obs[batch_cols[0]].astype(str)
        else:
            pb.obs[combined_batch] = pb.obs[batch_cols].astype(str).agg("|".join, axis=1)
    
    # Phase 2: Process cell types
    if verbose:
        print("[Pseudobulk] Phase 2: Processing cell types...")
    
    hvg_data: Dict[str, Dict[str, float]] = {}
    all_genes: List[str] = []
    valid_cts: List[str] = []
    
    for i, ct in enumerate(cell_types):
        if verbose:
            print(f"  [{i+1}/{len(cell_types)}] {ct}")
        
        try:
            result = process_celltype_layer(
                pb.layers[ct], pb.obs, pb.var, ct,
                combined_batch, preserve_list, n_features,
                normalize, target_sum, atac, combat_timeout, verbose
            )
            
            if result is None:
                continue
            
            genes, expr = result
            all_genes.extend(genes)
            valid_cts.append(ct)
            
            for j, sample in enumerate(pb.obs.index):
                hvg_data.setdefault(sample, {}).update(
                    {g: float(expr[j, k]) for k, g in enumerate(genes)}
                )
            
            clear_gpu_memory()
            
        except Exception as e:
            if verbose:
                print(f"    Error: {type(e).__name__}: {e}")
    
    cell_types = valid_cts
    if verbose:
        print(f"[Pseudobulk] Retained {len(cell_types)} cell types")
    
    # Phase 3: Final concatenation
    if verbose:
        print("[Pseudobulk] Phase 3: Building final matrix...")
    
    unique_genes = sorted(set(all_genes))
    gene_idx = {g: j for j, g in enumerate(unique_genes)}
    
    concat_matrix = np.zeros((len(samples), len(unique_genes)), dtype=np.float32)
    for i, sample in enumerate(samples):
        if sample in hvg_data:
            for g, v in hvg_data[sample].items():
                if g in gene_idx:
                    concat_matrix[i, gene_idx[g]] = v
    
    concat = sc.AnnData(
        X=concat_matrix,
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=unique_genes)
    )
    
    # Final HVG selection on GPU
    sc.pp.filter_genes(concat, min_cells=1)
    n_final = min(n_features, concat.n_vars)
    to_gpu(concat)
    rsc.pp.highly_variable_genes(concat, n_top_genes=n_final)
    to_cpu(concat)
    
    if verbose:
        print(f"[Pseudobulk] Final features: {concat.n_vars}")
    
    # Build expression DataFrame
    final_genes = set(concat.var.index)
    ct_groups = {}
    for g in final_genes:
        ct = g.split(' - ')[0]
        ct_groups.setdefault(ct, []).append(g)
    for ct in ct_groups:
        ct_groups[ct].sort()
    
    expr_df = pd.DataFrame(index=sorted(cell_types), columns=samples, dtype=object)
    for ct in expr_df.index:
        genes = ct_groups.get(ct, [])
        for sample in expr_df.columns:
            vals, names = [], []
            for g in genes:
                if sample in hvg_data and g in hvg_data[sample]:
                    vals.append(hvg_data[sample][g])
                    names.append(g)
            expr_df.at[ct, sample] = pd.Series(vals, index=names) if vals else pd.Series(dtype=float)
    
    # Phase 4: Proportions
    if verbose:
        print("[Pseudobulk] Phase 4: Computing proportions...")
    props = compute_proportions(adata, samples, cell_types, sample_col, celltype_col)
    concat.uns["cell_proportions"] = props
    
    clear_gpu_memory()
    
    if verbose:
        print("[Pseudobulk] Complete")
    
    return expr_df, props, concat


def compute_pseudobulk_adata_linux(
    adata: sc.AnnData,
    batch_col: Optional[Union[str, List[str]]] = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './',
    save: bool = True,
    n_features: int = 2000,
    normalize: bool = True,
    target_sum: float = 1e4,
    atac: bool = False,
    verbose: bool = False,
    combat_timeout: float = 20.0,
    preserve_cols: Optional[Union[str, List[str]]] = None,
) -> Tuple[Dict, sc.AnnData]:
    """
    Wrapper returning backward-compatible dict plus final AnnData.
    """
    if verbose:
        print(f"[Pseudobulk] Input: {adata.n_obs} cells, {adata.n_vars} genes")
    
    set_global_seed(seed=42, verbose=verbose)
    
    expr_df, props, final = compute_pseudobulk_gpu(
        adata=adata,
        batch_col=batch_col,
        sample_col=sample_col,
        celltype_col=celltype_col,
        output_dir=output_dir,
        n_features=n_features,
        normalize=normalize,
        target_sum=target_sum,
        atac=atac,
        verbose=verbose,
        combat_timeout=combat_timeout,
        preserve_cols=preserve_cols,
    )
    
    if verbose:
        print("[Pseudobulk] Extracting sample metadata...")
    
    final = _extract_sample_metadata(adata, final, sample_col)
    final.uns["cell_proportions"] = props
    
    if save:
        path = os.path.join(output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
        if verbose:
            print(f"[Pseudobulk] Saving to {path}")
        sc.write(path, final)
    
    result = {
        "cell_expression_corrected": expr_df,
        "cell_proportion": props.T
    }
    
    if verbose:
        print(f"[Pseudobulk] Output: {final.n_obs} samples, {final.n_vars} features")
    
    return result, final