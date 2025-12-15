#!/usr/bin/env python3
"""
GPU-accelerated pseudobulk computation using CELL EMBEDDINGS instead of expression.

Embedding mode:
  - Pseudobulks embeddings per cell type (mean over cells within each sample + cell type)
  - Applies ComBat batch correction (scanpy.pp.combat) per cell type layer
  - Concatenates corrected layers into one sample-by-feature AnnData
  - Saves in the same format as expression-based pseudobulk for downstream compatibility

Notes
-----
- ComBat runs on CPU (scanpy implementation). We keep GPU memory clearing utilities for your pipeline style.
- preserve_cols is now USED as ComBat covariates to preserve (if provided).
"""

import os
import gc
import warnings
import signal
from typing import Tuple, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

import cupy as cp

from utils.random_seed import set_global_seed


# =============================================================================
# USER CONFIGURATION - Modify embedding name here
# =============================================================================
EMBEDDING_KEY = "X_glue"  # Change this to your embedding key in adata.obsm
# =============================================================================


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
# Batch Correction with ComBat (scanpy)
# =============================================================================

class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout("ComBat timed out")


def _try_combat(
    embeddings: np.ndarray,
    obs: pd.DataFrame,
    batch_col: str,
    preserve_cols: Optional[Union[str, List[str]]],
    combat_timeout: float,
    verbose: bool
) -> Tuple[np.ndarray, bool]:
    """
    Attempt ComBat batch correction on embeddings using scanpy.pp.combat (CPU).

    Parameters
    ----------
    embeddings : np.ndarray
        (n_samples, n_dims) matrix
    obs : pd.DataFrame
        Sample metadata; must contain batch_col and any covariates you want to preserve
    batch_col : str
        Column name in obs defining batch
    preserve_cols : str|List[str]|None
        Covariates to preserve (scanpy combat covariates)
    combat_timeout : float
        Timeout (seconds); if <= 0, no timeout enforced
    verbose : bool
        Print progress

    Returns
    -------
    corrected : np.ndarray
        ComBat-corrected embeddings
    success : bool
        Whether correction was applied
    """
    try:
        if batch_col not in obs.columns:
            if verbose:
                print(f"    Skipping ComBat: batch_col '{batch_col}' not in obs")
            return embeddings, False

        batch_labels = obs[batch_col]
        n_batches = batch_labels.nunique(dropna=False)
        if n_batches <= 1:
            if verbose:
                print("    Skipping ComBat: only 1 batch")
            return embeddings, False

        batch_counts = batch_labels.value_counts(dropna=False)
        if batch_counts.min() < 2:
            if verbose:
                print("    Skipping ComBat: batch with <2 samples")
            return embeddings, False

        covariates = _as_list(preserve_cols)
        covariates = [c for c in covariates if c in obs.columns]
        if verbose:
            if covariates:
                print(f"    Running ComBat ({n_batches} batches, sizes: {batch_counts.min()}-{batch_counts.max()}), covariates={covariates}")
            else:
                print(f"    Running ComBat ({n_batches} batches, sizes: {batch_counts.min()}-{batch_counts.max()}), covariates=None")

        # Build temp AnnData for scanpy.pp.combat
        tmp = sc.AnnData(
            X=np.asarray(embeddings, dtype=np.float32),
            obs=obs.copy()
        )

        # Optional timeout (Linux signal-based)
        old_handler = None
        if combat_timeout is not None and combat_timeout > 0:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, float(combat_timeout))

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # scanpy.pp.combat modifies tmp.X in place
                sc.pp.combat(tmp, key=batch_col, covariates=covariates if covariates else None)
        finally:
            if combat_timeout is not None and combat_timeout > 0:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

        corrected = np.asarray(tmp.X)
        if verbose:
            print("    Applied ComBat correction")

        return corrected, True

    except _Timeout as e:
        if verbose:
            print(f"    ComBat timed out: {e}")
        return embeddings, False
    except Exception as e:
        if verbose:
            print(f"    ComBat failed: {type(e).__name__}: {e}")
        return embeddings, False


# =============================================================================
# Pseudobulk Embedding Aggregation
# =============================================================================

def aggregate_pseudobulk(
    adata: sc.AnnData,
    sample_col: str,
    celltype_col: str,
    batch_cols: List[str],
    embedding_key: str,
    verbose: bool
) -> Tuple[sc.AnnData, Dict[str, np.ndarray]]:
    """
    Aggregate cell embeddings into pseudobulk samples with cell-type layers.

    Returns
    -------
    pb : sc.AnnData
        Pseudobulk AnnData with batch info in obs
    ct_embeddings : Dict[str, np.ndarray]
        Dict mapping cell_type -> (n_samples, n_dims) embedding matrix
    """
    if embedding_key not in adata.obsm:
        raise ValueError(
            f"Embedding key '{embedding_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    embeddings = adata.obsm[embedding_key]
    if issparse(embeddings):
        embeddings = embeddings.toarray()
    embeddings = np.asarray(embeddings)

    samples = sorted(adata.obs[sample_col].unique())
    cell_types = sorted(adata.obs[celltype_col].unique())
    n_samples = len(samples)
    n_dims = embeddings.shape[1]

    if verbose:
        print(f"  Aggregating {adata.n_obs} cells -> {n_samples} samples x {len(cell_types)} cell types")
        print(f"  Embedding dimension: {n_dims}")

    sample_idx = {s: i for i, s in enumerate(samples)}
    ct_idx = {ct: i for i, ct in enumerate(cell_types)}

    # Build obs with batch columns
    obs = pd.DataFrame(index=samples)
    obs.index.name = "sample"
    for bc in batch_cols:
        mapping = (
            adata.obs[[sample_col, bc]]
            .drop_duplicates()
            .set_index(sample_col)[bc]
            .to_dict()
        )
        obs[bc] = [mapping.get(s, "Unknown") for s in samples]

    # Create placeholder AnnData (X filled later)
    pb = sc.AnnData(
        X=np.zeros((n_samples, 1), dtype=np.float32),
        obs=obs,
    )

    # Pre-compute indices
    cell_sample = adata.obs[sample_col].map(sample_idx).values
    cell_ct = adata.obs[celltype_col].map(ct_idx).values
    valid = ~(pd.isna(cell_sample) | pd.isna(cell_ct))
    valid_idx = np.where(valid)[0]
    cell_sample_valid = cell_sample[valid].astype(int)
    cell_ct_valid = cell_ct[valid].astype(int)

    # Aggregate per cell type
    ct_embeddings: Dict[str, np.ndarray] = {}

    for i, ct in enumerate(cell_types):
        mask = cell_ct_valid == i
        ct_cells = valid_idx[mask]
        ct_samples = cell_sample_valid[mask]

        if len(ct_cells) == 0:
            ct_embeddings[ct] = np.zeros((n_samples, n_dims), dtype=np.float32)
            continue

        ct_emb = np.zeros((n_samples, n_dims), dtype=np.float32)
        counts = np.zeros(n_samples, dtype=np.float32)

        for cell_i, sample_i in zip(ct_cells, ct_samples):
            ct_emb[sample_i] += embeddings[cell_i]
            counts[sample_i] += 1

        counts[counts == 0] = 1
        ct_emb = ct_emb / counts[:, None]
        ct_embeddings[ct] = ct_emb

    if verbose:
        print(f"  Created embeddings for {len(cell_types)} cell types")

    return pb, ct_embeddings


# =============================================================================
# Process Cell Type Embeddings (ComBat)
# =============================================================================

def process_celltype_layer(
    embedding_matrix: np.ndarray,
    obs: pd.DataFrame,
    cell_type: str,
    batch_col: Optional[str],
    preserve_cols: Optional[Union[str, List[str]]],
    combat_timeout: float,
    verbose: bool
) -> Optional[Tuple[List[str], np.ndarray]]:
    """
    Process a single cell-type embedding layer:
      - validate / drop NaN dims
      - apply ComBat (if enabled)
      - return feature names + processed matrix
    """
    n_samples, n_dims = embedding_matrix.shape

    sample_sums = np.abs(embedding_matrix).sum(axis=1)
    n_zero = (sample_sums == 0).sum()

    if n_zero == n_samples:
        if verbose:
            print(f"    Skipping: all samples have zero embeddings")
        return None

    if n_zero > 0 and verbose:
        print(f"    Warning: {n_zero} samples have zero embeddings (cell type absent)")

    # drop NaN dims
    nan_mask = np.isnan(embedding_matrix).any(axis=0)
    if nan_mask.any():
        if verbose:
            print(f"    Removing {int(nan_mask.sum())} NaN dimensions")
        embedding_matrix = embedding_matrix[:, ~nan_mask]
        n_dims = embedding_matrix.shape[1]
        if n_dims == 0:
            if verbose:
                print(f"    Skipping: no valid dimensions")
            return None

    # apply ComBat
    if batch_col and batch_col in obs.columns:
        embedding_matrix, corrected = _try_combat(
            embedding_matrix,
            obs=obs,
            batch_col=batch_col,
            preserve_cols=preserve_cols,
            combat_timeout=combat_timeout,
            verbose=verbose
        )

        # drop NaN dims post-correction (defensive)
        if corrected:
            nan_mask = np.isnan(embedding_matrix).any(axis=0)
            if nan_mask.any():
                if verbose:
                    print(f"    Removing {int(nan_mask.sum())} NaN dimensions post-correction")
                embedding_matrix = embedding_matrix[:, ~nan_mask]
                n_dims = embedding_matrix.shape[1]
                if n_dims == 0:
                    if verbose:
                        print(f"    Skipping: no valid dimensions post-correction")
                    return None

    feature_names = [f"{cell_type} - emb_{i}" for i in range(n_dims)]
    if verbose:
        print(f"    Output: {n_dims} dimensions")

    return feature_names, embedding_matrix


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
    batch_col: Optional[Union[str, List[str]]] = "batch",
    sample_col: str = "sample",
    celltype_col: str = "cell_type",
    output_dir: str = "./",
    n_features: int = 2000,
    normalize: bool = True,
    target_sum: float = 1e4,
    atac: bool = False,
    verbose: bool = False,
    combat_timeout: float = 20.0,
    preserve_cols: Optional[Union[str, List[str]]] = None,
    embedding_key: str = EMBEDDING_KEY,
) -> Tuple[pd.DataFrame, pd.DataFrame, sc.AnnData]:
    """
    GPU-accelerated pseudobulk computation using cell EMBEDDINGS.

    Note: n_features, normalize, target_sum, atac are kept for API compatibility but not used in embedding mode.
    ComBat is applied per cell-type layer if batch columns exist.
    """
    if verbose:
        print(f"[Pseudobulk-Embedding] Using embedding key: {embedding_key}")
        if preserve_cols is not None:
            print(f"[Pseudobulk-Embedding] preserve_cols (ComBat covariates): {preserve_cols}")
    clear_gpu_memory()

    os.makedirs(os.path.join(output_dir, "pseudobulk"), exist_ok=True)

    # Normalize batch columns
    batch_cols = [
        b for b in _as_list(batch_col)
        if b and b in adata.obs.columns and not adata.obs[b].isnull().all()
    ]

    # Fill NaNs in batch columns
    for col in batch_cols:
        if adata.obs[col].isnull().any():
            adata.obs[col] = adata.obs[col].fillna("Unknown")

    samples = sorted(adata.obs[sample_col].unique())
    cell_types = sorted(adata.obs[celltype_col].unique())

    if verbose:
        print(f"[Pseudobulk-Embedding] {len(samples)} samples, {len(cell_types)} cell types")
        print(f"[Pseudobulk-Embedding] Batch correction (ComBat): {batch_cols if batch_cols else 'disabled'}")

    # Phase 1: Aggregate embeddings
    if verbose:
        print("[Pseudobulk-Embedding] Phase 1: Aggregating embeddings...")
    pb, ct_embeddings = aggregate_pseudobulk(
        adata, sample_col, celltype_col, batch_cols, embedding_key, verbose
    )

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
        print("[Pseudobulk-Embedding] Phase 2: Processing cell types...")

    all_features: List[str] = []
    all_embeddings: List[np.ndarray] = []
    valid_cts: List[str] = []

    emb_data: Dict[str, Dict[str, float]] = {}

    for i, ct in enumerate(cell_types):
        if verbose:
            print(f"  [{i+1}/{len(cell_types)}] {ct}")

        try:
            result = process_celltype_layer(
                ct_embeddings[ct],
                pb.obs,
                ct,
                combined_batch,
                preserve_cols=preserve_cols,
                combat_timeout=combat_timeout,
                verbose=verbose,
            )

            if result is None:
                continue

            features, emb = result
            all_features.extend(features)
            all_embeddings.append(emb)
            valid_cts.append(ct)

            # Store for expr_df compatibility
            for j, sample in enumerate(pb.obs.index):
                emb_data.setdefault(sample, {}).update(
                    {f: float(emb[j, k]) for k, f in enumerate(features)}
                )

            clear_gpu_memory()

        except Exception as e:
            if verbose:
                print(f"    Error: {type(e).__name__}: {e}")

    cell_types = valid_cts
    if verbose:
        print(f"[Pseudobulk-Embedding] Retained {len(cell_types)} cell types")

    # Phase 3: Concatenate all embeddings
    if verbose:
        print("[Pseudobulk-Embedding] Phase 3: Concatenating...")

    if len(all_embeddings) == 0:
        raise ValueError("No valid cell type embeddings found")

    concat_matrix = np.hstack(all_embeddings)

    concat = sc.AnnData(
        X=concat_matrix,
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=all_features)
    )

    if verbose:
        print(f"[Pseudobulk-Embedding] Final features: {concat.n_vars}")

    # Build expression-style DataFrame for compatibility
    ct_groups: Dict[str, List[str]] = {}
    for f in all_features:
        ct_name = f.split(" - ")[0]
        ct_groups.setdefault(ct_name, []).append(f)
    for ct_name in ct_groups:
        ct_groups[ct_name].sort()

    expr_df = pd.DataFrame(index=sorted(cell_types), columns=samples, dtype=object)
    for ct_name in expr_df.index:
        feats = ct_groups.get(ct_name, [])
        for sample in expr_df.columns:
            vals, names = [], []
            for f in feats:
                if sample in emb_data and f in emb_data[sample]:
                    vals.append(emb_data[sample][f])
                    names.append(f)
            expr_df.at[ct_name, sample] = pd.Series(vals, index=names) if vals else pd.Series(dtype=float)

    # Phase 4: Proportions
    if verbose:
        print("[Pseudobulk-Embedding] Phase 4: Computing proportions...")
    props = compute_proportions(adata, samples, cell_types, sample_col, celltype_col)
    concat.uns["cell_proportions"] = props

    clear_gpu_memory()

    if verbose:
        print("[Pseudobulk-Embedding] Complete")

    return expr_df, props, concat


def compute_pseudobulk_adata_linux(
    adata: sc.AnnData,
    batch_col: Optional[Union[str, List[str]]] = "batch",
    sample_col: str = "sample",
    celltype_col: str = "cell_type",
    output_dir: str = "./",
    save: bool = True,
    n_features: int = 2000,
    normalize: bool = True,
    target_sum: float = 1e4,
    atac: bool = False,
    verbose: bool = False,
    Save = False,
    combat_timeout: float = 20.0,
    preserve_cols: Optional[Union[str, List[str]]] = None,
    embedding_key: str = EMBEDDING_KEY,
) -> Tuple[Dict, sc.AnnData]:
    """
    Wrapper returning backward-compatible dict plus final AnnData.

    Saves to same location as expression-based pseudobulk for downstream compatibility.

    Note: n_features, normalize, target_sum, atac are kept for API compatibility but not used in embedding mode.
    """
    if verbose:
        print(f"[Pseudobulk-Embedding] Input: {adata.n_obs} cells")
        if embedding_key in adata.obsm:
            print(f"[Pseudobulk-Embedding] Embedding: {embedding_key} with shape {adata.obsm[embedding_key].shape}")

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
        embedding_key=embedding_key,
    )

    if verbose:
        print("[Pseudobulk-Embedding] Extracting sample metadata...")

    final = _extract_sample_metadata(adata, final, sample_col)
    final.uns["cell_proportions"] = props

    # Store embedding key used for reference
    final.uns["embedding_key"] = embedding_key

    if save:
        path = os.path.join(output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
        if verbose:
            print(f"[Pseudobulk-Embedding] Saving to {path}")
        final.write(path)

    result = {
        "cell_expression_corrected": expr_df,  # Keep same key name for compatibility
        "cell_proportion": props.T
    }

    if verbose:
        print(f"[Pseudobulk-Embedding] Output: {final.n_obs} samples, {final.n_vars} features")

    return result, final
