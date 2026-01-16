#!/usr/bin/env python3
"""
GPU-accelerated pseudobulk computation using rapids_singlecell.

Modified version: HVG selection uses only RNA cells (both rounds).
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
        meta_df = pd.DataFrame(meta)

        overlap = sample_adata.obs.columns.intersection(meta_df.columns)
        if len(overlap) > 0:
            meta_df = meta_df.drop(columns=list(overlap))

        if meta_df.shape[1] > 0:
            sample_adata.obs = sample_adata.obs.join(meta_df, how="left")

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
    verbose: bool,
    modality_col: Optional[str] = None,
    modality_filter: Optional[str] = None
) -> sc.AnnData:
    """
    Aggregate cells into pseudobulk samples with cell-type layers.
    
    Parameters
    ----------
    modality_col : str, optional
        Column name containing modality info (e.g., 'modality')
    modality_filter : str, optional
        If provided, only aggregate cells from this modality (e.g., 'RNA')
    """
    # Apply modality filter if specified
    if modality_col and modality_filter:
        if modality_col in adata.obs.columns:
            mask = adata.obs[modality_col] == modality_filter
            adata_filtered = adata[mask].copy()
            if verbose:
                print(f"  Filtered to {modality_filter} cells: {adata_filtered.n_obs} / {adata.n_obs}")
        else:
            if verbose:
                print(f"  Warning: modality_col '{modality_col}' not found, using all cells")
            adata_filtered = adata
    else:
        adata_filtered = adata

    samples = sorted(adata_filtered.obs[sample_col].unique())
    cell_types = sorted(adata_filtered.obs[celltype_col].unique())
    n_samples, n_genes = len(samples), adata_filtered.n_vars

    if verbose:
        print(f"  Aggregating {adata_filtered.n_obs} cells -> {n_samples} samples x {len(cell_types)} cell types")

    sample_idx = {s: i for i, s in enumerate(samples)}
    ct_idx = {ct: i for i, ct in enumerate(cell_types)}

    # Build obs with batch columns
    obs = pd.DataFrame(index=samples)
    obs.index.name = 'sample'
    for bc in batch_cols:
        mapping = adata_filtered.obs[[sample_col, bc]].drop_duplicates().set_index(sample_col)[bc].to_dict()
        obs[bc] = [mapping.get(s, 'Unknown') for s in samples]

    pb = sc.AnnData(
        X=np.zeros((n_samples, n_genes), dtype=np.float32),
        obs=obs,
        var=adata_filtered.var.copy()
    )

    # Pre-compute indices
    cell_sample = adata_filtered.obs[sample_col].map(sample_idx).values
    cell_ct = adata_filtered.obs[celltype_col].map(ct_idx).values
    valid = ~(pd.isna(cell_sample) | pd.isna(cell_ct))
    valid_idx = np.where(valid)[0]
    cell_sample_valid = cell_sample[valid].astype(int)
    cell_ct_valid = cell_ct[valid].astype(int)

    X = adata_filtered.X.tocsr() if issparse(adata_filtered.X) else adata_filtered.X

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

def select_hvgs_from_layer(
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
) -> Optional[List[str]]:
    """
    Select HVGs from a single cell-type layer (RNA-only).
    Returns only the list of selected gene names.
    """
    temp = sc.AnnData(X=layer_data.copy(), obs=obs.copy(), var=var.copy())

    # Filter empty genes
    sc.pp.filter_genes(temp, min_cells=1)
    if temp.n_vars == 0:
        if verbose:
            print(f"    Skipping HVG selection: no genes after filtering")
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

    # Batch correction
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

    if verbose:
        print(f"    Selected {len(hvg_genes)} HVGs from RNA cells")

    return hvg_genes


def process_celltype_layer_with_hvg_list(
    layer_data: np.ndarray,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    cell_type: str,
    hvg_genes: List[str],
    batch_col: Optional[str],
    preserve_cols: List[str],
    normalize: bool,
    target_sum: float,
    atac: bool,
    combat_timeout: float,
    verbose: bool
) -> Optional[Tuple[List[str], np.ndarray]]:
    """
    Process a single cell-type layer using pre-selected HVG list.
    """
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

    # Batch correction
    if batch_col and batch_col in temp.obs.columns:
        corrected = apply_batch_correction(temp, batch_col, preserve_cols, combat_timeout, verbose)
        if corrected:
            nan_mask = _get_nan_mask(temp.X)
            if nan_mask.any():
                if verbose:
                    print(f"    Removing {nan_mask.sum()} NaN genes post-correction")
                temp = temp[:, ~nan_mask].copy()

    # Filter to HVG genes that exist in this data
    available_hvgs = [g for g in hvg_genes if g in temp.var.index]
    
    if not available_hvgs:
        if verbose:
            print(f"    Skipping: no HVGs available in processed data")
        return None

    hvg_expr = temp[:, available_hvgs].X
    if issparse(hvg_expr):
        hvg_expr = hvg_expr.toarray()

    prefixed = [f"{cell_type} - {g}" for g in available_hvgs]

    if verbose:
        print(f"    Using {len(available_hvgs)} HVGs")

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
    hvg_modality: str = 'RNA',  # NEW: modality to use for HVG selection
    modality_col: str = 'modality',  # NEW: column name for modality info
) -> Tuple[pd.DataFrame, pd.DataFrame, sc.AnnData]:
    """
    GPU-accelerated pseudobulk computation.
    
    Modified: HVG selection uses only RNA cells (both Round 1 and Round 2).
    
    Parameters
    ----------
    hvg_modality : str
        Modality to use for HVG selection (default: 'RNA')
    modality_col : str
        Column name in adata.obs containing modality information (default: 'modality')
    """
    if verbose:
        print(f"[Pseudobulk] Using GPU (rapids_singlecell)")
        print(f"[Pseudobulk] HVG selection will use {hvg_modality} cells only (both rounds)")
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

    # Check if modality column exists
    has_modality = modality_col in adata.obs.columns
    if has_modality:
        modality_counts = adata.obs[modality_col].value_counts()
        if verbose:
            print(f"[Pseudobulk] Modality distribution: {dict(modality_counts)}")
    else:
        if verbose:
            print(f"[Pseudobulk] Warning: '{modality_col}' column not found, using all cells for HVG")

    # Phase 1a: Aggregate ALL cells (for final expression values)
    if verbose:
        print("[Pseudobulk] Phase 1a: Aggregating ALL cells...")
    pb_all = aggregate_pseudobulk(adata, sample_col, celltype_col, batch_cols, verbose)

    # Phase 1b: Aggregate RNA-only cells (for HVG selection)
    if has_modality:
        if verbose:
            print(f"[Pseudobulk] Phase 1b: Aggregating {hvg_modality} cells for HVG selection...")
        pb_rna = aggregate_pseudobulk(
            adata, sample_col, celltype_col, batch_cols, verbose,
            modality_col=modality_col, modality_filter=hvg_modality
        )
    else:
        pb_rna = pb_all

    # Attach preserve covariates to pb_all.obs
    if preserve_list:
        pb_all = _extract_sample_metadata(
            cell_adata=adata,
            sample_adata=pb_all,
            sample_col=sample_col,
            exclude_cols=None,
        )
        preserve_list = [c for c in preserve_list if c in pb_all.obs.columns]

    # Also attach to pb_rna if different
    if has_modality and preserve_list:
        pb_rna = _extract_sample_metadata(
            cell_adata=adata[adata.obs[modality_col] == hvg_modality],
            sample_adata=pb_rna,
            sample_col=sample_col,
            exclude_cols=None,
        )

    # Create combined batch column if needed
    combined_batch = None
    if batch_cols:
        combined_batch = "_combined_batch_"
        if len(batch_cols) == 1:
            pb_all.obs[combined_batch] = pb_all.obs[batch_cols[0]].astype(str)
            pb_rna.obs[combined_batch] = pb_rna.obs[batch_cols[0]].astype(str)
        else:
            pb_all.obs[combined_batch] = pb_all.obs[batch_cols].astype(str).agg("|".join, axis=1)
            pb_rna.obs[combined_batch] = pb_rna.obs[batch_cols].astype(str).agg("|".join, axis=1)

    # Phase 2: Process cell types (Round 1 HVG selection)
    if verbose:
        print("[Pseudobulk] Phase 2: Processing cell types (Round 1 HVG)...")
        print(f"  Step 2a: Selecting HVGs from {hvg_modality} cells")
        print(f"  Step 2b: Extracting expression from ALL cells")

    hvg_data: Dict[str, Dict[str, float]] = {}
    all_genes: List[str] = []
    valid_cts: List[str] = []

    # Get cell types present in RNA data
    rna_cell_types = set(pb_rna.layers.keys())

    for i, ct in enumerate(cell_types):
        if verbose:
            print(f"  [{i+1}/{len(cell_types)}] {ct}")

        try:
            # Step 2a: Select HVGs from RNA-only pseudobulk
            if ct in rna_cell_types and ct in pb_rna.layers:
                if verbose:
                    print(f"    Selecting HVGs from {hvg_modality} cells...")
                
                hvg_genes = select_hvgs_from_layer(
                    pb_rna.layers[ct], pb_rna.obs, pb_rna.var, ct,
                    combined_batch, preserve_list, n_features,
                    normalize, target_sum, atac, combat_timeout, verbose
                )
            else:
                if verbose:
                    print(f"    Warning: {ct} not in {hvg_modality} data, skipping HVG selection")
                hvg_genes = None

            if hvg_genes is None or len(hvg_genes) == 0:
                if verbose:
                    print(f"    Skipping: no HVGs selected")
                continue

            # Step 2b: Extract expression from ALL cells using selected HVGs
            if verbose:
                print(f"    Extracting expression from ALL cells...")
            
            result = process_celltype_layer_with_hvg_list(
                pb_all.layers[ct], pb_all.obs, pb_all.var, ct,
                hvg_genes, combined_batch, preserve_list,
                normalize, target_sum, atac, combat_timeout, verbose
            )

            if result is None:
                continue

            genes, expr = result
            all_genes.extend(genes)
            valid_cts.append(ct)

            for j, sample in enumerate(pb_all.obs.index):
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

    # Phase 3: Build concatenated matrix from ALL cells
    if verbose:
        print("[Pseudobulk] Phase 3: Building concatenated matrix from ALL cells...")

    unique_genes = sorted(set(all_genes))
    gene_idx = {g: j for j, g in enumerate(unique_genes)}

    concat_matrix_all = np.zeros((len(samples), len(unique_genes)), dtype=np.float32)
    for i, sample in enumerate(samples):
        if sample in hvg_data:
            for g, v in hvg_data[sample].items():
                if g in gene_idx:
                    concat_matrix_all[i, gene_idx[g]] = v

    concat_all = sc.AnnData(
        X=concat_matrix_all,
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=unique_genes)
    )

    # ============ NEW: Build concatenated matrix from RNA cells for Round 2 HVG ============
    if verbose:
        print(f"[Pseudobulk] Phase 3b: Building concatenated matrix from {hvg_modality} cells for Round 2 HVG...")

    # Build RNA-only expression data for Round 2 HVG selection
    hvg_data_rna: Dict[str, Dict[str, float]] = {}
    
    for i, ct in enumerate(cell_types):
        if ct in rna_cell_types and ct in pb_rna.layers:
            try:
                # Get HVGs from Round 1
                ct_genes = [g for g in unique_genes if g.startswith(f"{ct} - ")]
                if not ct_genes:
                    continue
                
                # Extract base gene names (without cell type prefix)
                base_genes = [g.split(' - ', 1)[1] for g in ct_genes]
                
                # Process RNA layer with these genes
                result_rna = process_celltype_layer_with_hvg_list(
                    pb_rna.layers[ct], pb_rna.obs, pb_rna.var, ct,
                    base_genes, combined_batch, preserve_list,
                    normalize, target_sum, atac, combat_timeout, verbose=False
                )
                
                if result_rna is not None:
                    genes_rna, expr_rna = result_rna
                    for j, sample in enumerate(pb_rna.obs.index):
                        hvg_data_rna.setdefault(sample, {}).update(
                            {g: float(expr_rna[j, k]) for k, g in enumerate(genes_rna)}
                        )
            except Exception as e:
                if verbose:
                    print(f"    Warning: Could not process {ct} for Round 2 HVG: {e}")

    # Build RNA-only matrix
    concat_matrix_rna = np.zeros((len(samples), len(unique_genes)), dtype=np.float32)
    for i, sample in enumerate(samples):
        if sample in hvg_data_rna:
            for g, v in hvg_data_rna[sample].items():
                if g in gene_idx:
                    concat_matrix_rna[i, gene_idx[g]] = v

    concat_rna = sc.AnnData(
        X=concat_matrix_rna,
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=unique_genes)
    )
    # ========================================================================================

    # Round 2 HVG selection on RNA-only concatenated matrix
    if verbose:
        print(f"[Pseudobulk] Phase 4: Round 2 HVG selection from {hvg_modality} cells...")
    
    sc.pp.filter_genes(concat_rna, min_cells=1)
    n_final = min(n_features, concat_rna.n_vars)
    to_gpu(concat_rna)
    rsc.pp.highly_variable_genes(concat_rna, n_top_genes=n_final)
    to_cpu(concat_rna)

    # Get HVG mask from RNA and apply to ALL cells matrix
    hvg_mask = concat_rna.var["highly_variable"].values
    hvg_genes_round2 = concat_rna.var.index[hvg_mask].tolist()
    
    # Subset the ALL cells matrix to these HVGs
    concat_final = concat_all[:, hvg_genes_round2].copy()

    if verbose:
        print(f"[Pseudobulk] Final features: {concat_final.n_vars} (selected from {hvg_modality} cells, applied to ALL cells)")

    # Build expression DataFrame using final HVGs
    final_genes = set(concat_final.var.index)
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

    # Phase 5: Proportions
    if verbose:
        print("[Pseudobulk] Phase 5: Computing proportions...")
    props = compute_proportions(adata, samples, cell_types, sample_col, celltype_col)
    concat_final.uns["cell_proportions"] = props

    clear_gpu_memory()

    if verbose:
        print("[Pseudobulk] Complete")

    return expr_df, props, concat_final


def compute_pseudobulk_adata_linux(
    adata: sc.AnnData,
    batch_col: Optional[Union[str, List[str]]] = None,
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
    hvg_modality: str = 'RNA',  # NEW parameter
    modality_col: str = 'modality',  # NEW parameter
) -> Tuple[Dict, sc.AnnData]:
    """
    Wrapper returning backward-compatible dict plus final AnnData.
    
    Parameters
    ----------
    hvg_modality : str
        Modality to use for HVG selection (default: 'RNA')
    modality_col : str
        Column name in adata.obs containing modality information (default: 'modality')
    """
    if verbose:
        print(f"[Pseudobulk] Input: {adata.n_obs} cells, {adata.n_vars} genes")
        print(f"[Pseudobulk] HVG selection modality: {hvg_modality} (both rounds)")

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
        hvg_modality=hvg_modality,
        modality_col=modality_col,
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