import os
import time
import gc
import signal
import io
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from typing import Tuple, Dict, List
import contextlib
import patsy
import sys
import traceback

# GPU imports
import torch

# Import the TF-IDF function (keep as in your original setup)
from tf_idf import tfidf_memory_efficient
from utils.random_seed import set_global_seed
from utils.limma import limma

def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0


def _extract_sample_metadata(
    cell_adata: sc.AnnData,
    sample_adata: sc.AnnData,
    sample_col: str,
    exclude_cols: List[str] | None = None,
) -> sc.AnnData:
    """Detect and copy sample-level metadata (identical to CPU version)."""
    if exclude_cols is None:
        exclude_cols = []
    exclude_cols = set(exclude_cols) | {sample_col}

    grouped = cell_adata.obs.groupby(sample_col)
    meta_dict: Dict[str, pd.Series] = {}

    for col in cell_adata.obs.columns:
        if col in exclude_cols:
            continue
        uniques_per_sample = grouped[col].apply(lambda x: x.dropna().unique())
        # Keep if every sample shows ≤1 unique value
        if uniques_per_sample.apply(lambda u: len(u) <= 1).all():
            meta_dict[col] = uniques_per_sample.apply(lambda u: u[0] if len(u) else np.nan)

    if meta_dict:
        meta_df = pd.DataFrame(meta_dict)
        meta_df.index.name = "sample"
        sample_adata.obs = sample_adata.obs.join(meta_df, how="left")

    return sample_adata


def compute_pseudobulk_layers_gpu(
    adata: sc.AnnData,
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './',
    n_features: int = 2000,
    normalize: bool = True,
    target_sum: float = 1e4,
    atac: bool = False,
    verbose: bool = False,
    batch_size: int = 100,
    max_cells_per_batch: int = 50000,
    combat_timeout: float = 1800.0  # timeout for ComBat (seconds)
) -> Tuple[pd.DataFrame, pd.DataFrame, sc.AnnData]:
    """
    GPU-accelerated pseudobulk computation matching CPU version output exactly.

    Parameters
    ----------
    combat_timeout : float
        Maximum time in seconds to wait for ComBat before falling back to limma.
        Set to None to disable timeout.
    """
    start_time = time.time()
    timing_report = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    t0 = time.time()
    clear_gpu_memory()
    timing_report['initial_gpu_clear'] = time.time() - t0

    if verbose and torch.cuda.is_available():
        alloc, reserved = get_gpu_memory_info()
        print(f"[pseudobulk] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[pseudobulk] Initial GPU memory: {alloc:.2f}/{reserved:.2f} GB (alloc/reserved)")

    # Create output directory
    t0 = time.time()
    pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
    os.makedirs(pseudobulk_dir, exist_ok=True)
    timing_report['create_output_dir'] = time.time() - t0

    # Check if batch correction should be applied (based on original cell-level adata)
    t0 = time.time()
    batch_correction = (
        batch_col is not None and
        batch_col in adata.obs.columns and
        not adata.obs[batch_col].isnull().all()
    )
    
    if batch_correction and adata.obs[batch_col].isnull().any():
        adata.obs[batch_col] = adata.obs[batch_col].fillna("Unknown")
    timing_report['batch_col_setup'] = time.time() - t0

    if verbose:
        print(f"[pseudobulk] Batch correction enabled: {batch_correction} (col='{batch_col}')")

    # Get unique samples and cell types
    t0 = time.time()
    samples = sorted(adata.obs[sample_col].unique())
    cell_types = sorted(adata.obs[celltype_col].unique())
    timing_report['get_unique_samples_celltypes'] = time.time() - t0

    if verbose:
        print(f"[pseudobulk] {len(cell_types)} cell types, {len(samples)} samples")

    # Phase 1: Create pseudobulk AnnData with layers
    t0 = time.time()
    pseudobulk_adata = _create_pseudobulk_layers_gpu(
        adata, samples, cell_types, sample_col, celltype_col,
        batch_col, batch_size, max_cells_per_batch, verbose
    )
    timing_report['phase1_create_pseudobulk_layers'] = time.time() - t0
    print(f"[TIMING] Phase 1 (create pseudobulk layers): {timing_report['phase1_create_pseudobulk_layers']:.2f}s")

    # Phase 2: Process each cell type layer
    t0_phase2 = time.time()
    all_hvg_data = {}
    all_gene_names = []
    cell_types_to_remove = []
    
    # Timing accumulators for phase 2
    phase2_timing = {
        'layer_extraction': 0,
        'filter_genes': 0,
        'normalization': 0,
        'nan_removal_pre': 0,
        'batch_correction_combat': 0,
        'batch_correction_limma': 0,
        'nan_removal_post': 0,
        'hvg_selection': 0,
        'data_collection': 0,
    }

    for ct_idx, cell_type in enumerate(cell_types):
        if verbose:
            print(f"\n[pseudobulk] Processing cell type {ct_idx+1}/{len(cell_types)}: {cell_type}")

        try:
            # Temporary AnnData for this cell type
            t0 = time.time()
            layer_data = pseudobulk_adata.layers[cell_type]
            temp_adata = sc.AnnData(
                X=layer_data.copy(),
                obs=pseudobulk_adata.obs.copy(),
                var=pseudobulk_adata.var.copy()
            )
            phase2_timing['layer_extraction'] += time.time() - t0

            # Filter out genes with zero expression
            t0 = time.time()
            sc.pp.filter_genes(temp_adata, min_cells=1)
            phase2_timing['filter_genes'] += time.time() - t0
            
            if temp_adata.n_vars == 0:
                if verbose:
                    print(f"  - No expressed genes, skipping.")
                cell_types_to_remove.append(cell_type)
                continue

            # Normalization
            if normalize:
                t0 = time.time()
                if atac:
                    tfidf_memory_efficient(temp_adata, scale_factor=target_sum)
                else:
                    _normalize_gpu(temp_adata, target_sum, device, max_cells_per_batch)
                norm_time = time.time() - t0
                phase2_timing['normalization'] += norm_time
                if verbose:
                    print(f"  - Normalization: {norm_time:.2f}s")

            # Remove NaN genes if any
            t0 = time.time()
            if issparse(temp_adata.X):
                nan_genes = np.array(np.isnan(temp_adata.X.toarray()).any(axis=0)).flatten()
            else:
                nan_genes = np.isnan(temp_adata.X).any(axis=0)

            if nan_genes.any():
                if verbose:
                    print(f"  - Removing {nan_genes.sum()} NaN genes before batch correction.")
                temp_adata = temp_adata[:, ~nan_genes].copy()
            phase2_timing['nan_removal_pre'] += time.time() - t0

            # Batch correction
            batch_correction_applied = False
            batch_correction_method = None

            if batch_correction and batch_col in temp_adata.obs.columns and len(temp_adata.obs[batch_col].unique()) > 1:
                min_batch_size = temp_adata.obs[batch_col].value_counts().min()

                # Try ComBat first if possible
                if min_batch_size >= 2:
                    try:
                        if verbose:
                            print(f"  - Trying ComBat (min batch size={min_batch_size})")

                        def timeout_handler(signum, frame):
                            raise TimeoutError("ComBat timed out")

                        old_handler = None
                        if combat_timeout is not None:
                            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(int(combat_timeout))

                        t0 = time.time()
                        try:
                            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                sc.pp.combat(temp_adata, key=batch_col)
                            batch_correction_applied = True
                            batch_correction_method = "ComBat"
                        finally:
                            if combat_timeout is not None:
                                signal.alarm(0)
                                if old_handler is not None:
                                    signal.signal(signal.SIGALRM, old_handler)
                        
                        combat_time = time.time() - t0
                        phase2_timing['batch_correction_combat'] += combat_time
                        if verbose:
                            print(f"  - ComBat time: {combat_time:.2f}s")

                        # Remove NaNs generated by ComBat
                        t0 = time.time()
                        if issparse(temp_adata.X):
                            has_nan = np.any(np.isnan(temp_adata.X.data))
                            if has_nan:
                                nan_genes_post = np.array(
                                    np.isnan(temp_adata.X.toarray()).any(axis=0)
                                ).flatten()
                            else:
                                nan_genes_post = np.zeros(temp_adata.n_vars, dtype=bool)
                        else:
                            nan_genes_post = np.isnan(temp_adata.X).any(axis=0)

                        if nan_genes_post.any():
                            temp_adata = temp_adata[:, ~nan_genes_post].copy()
                        phase2_timing['nan_removal_post'] += time.time() - t0

                        if verbose:
                            print(f"  - ComBat applied successfully.")

                    except (TimeoutError, Exception):
                        if verbose:
                            print(f"  - ComBat failed or timed out; will try limma.")
                else:
                    if verbose:
                        print(f"  - Skipping ComBat (min batch size={min_batch_size} < 2).")

                # limma fallback / alternative
                if not batch_correction_applied:
                    try:
                        t0 = time.time()
                        # Ensure dense
                        if issparse(temp_adata.X):
                            X_dense = temp_adata.X.toarray()
                        else:
                            X_dense = temp_adata.X.copy()
                        covariate_formula = f'~ Q("{batch_col}")'

                        X_corrected = limma(
                            pheno=temp_adata.obs,
                            exprs=X_dense,
                            covariate_formula=covariate_formula,
                            design_formula='1',
                            rcond=1e-8,
                            verbose=False  # keep limma quiet here
                        )
                    
                        temp_adata.X = X_corrected
                        batch_correction_applied = True
                        batch_correction_method = "limma"
                        
                        limma_time = time.time() - t0
                        phase2_timing['batch_correction_limma'] += limma_time
                        if verbose:
                            print(f"  - limma time: {limma_time:.2f}s")

                        # Remove NaNs if any
                        t0 = time.time()
                        if issparse(temp_adata.X):
                            nan_genes_post = np.array(np.isnan(temp_adata.X.toarray()).any(axis=0)).flatten()
                        else:
                            nan_genes_post = np.isnan(temp_adata.X).any(axis=0)
                        if nan_genes_post.any():
                            temp_adata = temp_adata[:, ~nan_genes_post].copy()
                        phase2_timing['nan_removal_post'] += time.time() - t0

                        if verbose:
                            print("  - limma batch regression applied.")

                    except Exception as e:
                        if verbose:
                            print(f"  - limma failed ({type(e).__name__}); continuing without batch correction.")
                            print(f"    DEBUG: Exception details: {str(e)}")
                            print(f"    DEBUG: Traceback:")
                            traceback.print_exc()

            elif verbose:
                print("  - Batch correction skipped (conditions not met).")

            if verbose and batch_correction_applied:
                print(f"  - Batch correction method: {batch_correction_method}")

            # HVG selection
            t0 = time.time()
            n_hvgs = min(n_features, temp_adata.n_vars)
            sc.pp.highly_variable_genes(
                temp_adata,
                n_top_genes=n_hvgs,
                subset=False
            )

            hvg_mask = temp_adata.var['highly_variable']
            hvg_genes = temp_adata.var.index[hvg_mask].tolist()
            hvg_time = time.time() - t0
            phase2_timing['hvg_selection'] += hvg_time
            if verbose:
                print(f"  - HVG selection: {hvg_time:.2f}s")

            if len(hvg_genes) == 0:
                if verbose:
                    print("  - No HVGs found, skipping cell type.")
                cell_types_to_remove.append(cell_type)
                continue

            t0 = time.time()
            hvg_expr = temp_adata[:, hvg_mask].X
            if issparse(hvg_expr):
                hvg_expr = hvg_expr.toarray()

            prefixed_genes = [f"{cell_type} - {g}" for g in hvg_genes]
            all_gene_names.extend(prefixed_genes)

            for i, sample in enumerate(temp_adata.obs.index):
                if sample not in all_hvg_data:
                    all_hvg_data[sample] = {}
                for j, gene in enumerate(prefixed_genes):
                    all_hvg_data[sample][gene] = float(hvg_expr[i, j])
            phase2_timing['data_collection'] += time.time() - t0

            if verbose:
                print(f"  - Selected {len(hvg_genes)} HVGs.")

        except Exception as e:
            if verbose:
                print(f"  - Failed to process {cell_type}: {e}")
            cell_types_to_remove.append(cell_type)

    timing_report['phase2_total'] = time.time() - t0_phase2
    timing_report['phase2_details'] = phase2_timing
    
    print(f"\n[TIMING] Phase 2 (process cell types) total: {timing_report['phase2_total']:.2f}s")
    print(f"  - Layer extraction: {phase2_timing['layer_extraction']:.2f}s")
    print(f"  - Filter genes: {phase2_timing['filter_genes']:.2f}s")
    print(f"  - Normalization: {phase2_timing['normalization']:.2f}s")
    print(f"  - NaN removal (pre-batch): {phase2_timing['nan_removal_pre']:.2f}s")
    print(f"  - ComBat batch correction: {phase2_timing['batch_correction_combat']:.2f}s")
    print(f"  - limma batch correction: {phase2_timing['batch_correction_limma']:.2f}s")
    print(f"  - NaN removal (post-batch): {phase2_timing['nan_removal_post']:.2f}s")
    print(f"  - HVG selection: {phase2_timing['hvg_selection']:.2f}s")
    print(f"  - Data collection: {phase2_timing['data_collection']:.2f}s")

    # Remove failed cell types from list
    cell_types = [ct for ct in cell_types if ct not in cell_types_to_remove]

    if verbose:
        print("\n[pseudobulk] Summary:")
        print(f"  - Successful cell types: {len(cell_types)}")
        print(f"  - Failed/empty cell types: {len(cell_types_to_remove)}")

    # Phase 3: Create concatenated AnnData with all cell type HVGs
    t0_phase3 = time.time()
    if verbose:
        print("\n[pseudobulk] Concatenating HVGs across cell types.")

    t0 = time.time()
    all_unique_genes = sorted(list(set(all_gene_names)))
    concat_matrix = np.zeros((len(samples), len(all_unique_genes)), dtype=float)

    for i, sample in enumerate(samples):
        for j, gene in enumerate(all_unique_genes):
            if sample in all_hvg_data and gene in all_hvg_data[sample]:
                concat_matrix[i, j] = all_hvg_data[sample][gene]
    timing_report['phase3_build_matrix'] = time.time() - t0

    t0 = time.time()
    concat_adata = sc.AnnData(
        X=concat_matrix,
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=all_unique_genes)
    )
    timing_report['phase3_create_adata'] = time.time() - t0

    # Final HVG selection on concatenated data
    if verbose:
        print(f"[pseudobulk] Final HVG selection on concatenated matrix (genes={concat_adata.n_vars}).")

    t0 = time.time()
    sc.pp.filter_genes(concat_adata, min_cells=1)
    final_n_hvgs = min(n_features, concat_adata.n_vars)
    sc.pp.highly_variable_genes(
        concat_adata,
        n_top_genes=final_n_hvgs,
        subset=True
    )
    timing_report['phase3_final_hvg'] = time.time() - t0

    timing_report['phase3_total'] = time.time() - t0_phase3
    print(f"\n[TIMING] Phase 3 (concatenate HVGs) total: {timing_report['phase3_total']:.2f}s")
    print(f"  - Build matrix: {timing_report['phase3_build_matrix']:.2f}s")
    print(f"  - Create AnnData: {timing_report['phase3_create_adata']:.2f}s")
    print(f"  - Final HVG selection: {timing_report['phase3_final_hvg']:.2f}s")

    if verbose:
        print(f"[pseudobulk] Final HVGs retained: {concat_adata.n_vars}")

    # Create final expression DataFrame in original format
    t0 = time.time()
    final_expression_df = _create_final_expression_matrix(
        all_hvg_data,
        concat_adata.var.index.tolist(),
        samples,
        cell_types,
        verbose
    )
    timing_report['create_final_expression_matrix'] = time.time() - t0
    print(f"\n[TIMING] Create final expression matrix: {timing_report['create_final_expression_matrix']:.2f}s")

    # Phase 4: Compute cell proportions
    t0 = time.time()
    cell_proportion_df = _compute_cell_proportions(
        adata, samples, cell_types, sample_col, celltype_col
    )
    timing_report['phase4_cell_proportions'] = time.time() - t0
    print(f"[TIMING] Phase 4 (cell proportions): {timing_report['phase4_cell_proportions']:.2f}s")

    # Add cell proportions to concat_adata
    concat_adata.uns['cell_proportions'] = cell_proportion_df

    # Save final outputs
    t0 = time.time()
    final_expression_df.to_csv(
        os.path.join(pseudobulk_dir, "expression_hvg.csv")
    )
    cell_proportion_df.to_csv(
        os.path.join(pseudobulk_dir, "proportion.csv")
    )
    timing_report['save_outputs'] = time.time() - t0
    print(f"[TIMING] Save outputs: {timing_report['save_outputs']:.2f}s")

    t0 = time.time()
    clear_gpu_memory()
    timing_report['final_gpu_clear'] = time.time() - t0

    total_time = time.time() - start_time
    timing_report['total'] = total_time

    print(f"\n{'='*60}")
    print(f"[TIMING] TOTAL RUNTIME: {total_time:.2f}s")
    print(f"{'='*60}")
    print(f"  Phase 1 (create pseudobulk layers): {timing_report['phase1_create_pseudobulk_layers']:.2f}s ({100*timing_report['phase1_create_pseudobulk_layers']/total_time:.1f}%)")
    print(f"  Phase 2 (process cell types):       {timing_report['phase2_total']:.2f}s ({100*timing_report['phase2_total']/total_time:.1f}%)")
    print(f"  Phase 3 (concatenate HVGs):         {timing_report['phase3_total']:.2f}s ({100*timing_report['phase3_total']/total_time:.1f}%)")
    print(f"  Create final expression matrix:     {timing_report['create_final_expression_matrix']:.2f}s ({100*timing_report['create_final_expression_matrix']/total_time:.1f}%)")
    print(f"  Phase 4 (cell proportions):         {timing_report['phase4_cell_proportions']:.2f}s ({100*timing_report['phase4_cell_proportions']/total_time:.1f}%)")
    print(f"  Save outputs:                       {timing_report['save_outputs']:.2f}s ({100*timing_report['save_outputs']/total_time:.1f}%)")
    print(f"{'='*60}")

    if verbose:
        print(f"\n[pseudobulk] Done. Runtime: {total_time:.2f} s")
        print(f"[pseudobulk] Final HVG matrix shape: {final_expression_df.shape}")
        print(f"[pseudobulk] Final AnnData shape: {concat_adata.shape}")

    return final_expression_df, cell_proportion_df, concat_adata

def _create_pseudobulk_layers_gpu(
    adata: sc.AnnData,
    samples: list,
    cell_types: list,
    sample_col: str,
    celltype_col: str,
    batch_col: str,
    batch_size: int,
    max_cells_per_batch: int,
    verbose: bool
) -> sc.AnnData:
    """Optimized pseudobulk creation using sparse indicator matrix multiplication."""
    from scipy.sparse import csr_matrix, csc_matrix
    
    t0_total = time.time()
    
    n_samples = len(samples)
    n_celltypes = len(cell_types)
    n_genes = adata.n_vars
    n_cells = adata.n_obs
    
    # Create index mappings
    sample_to_idx = {s: i for i, s in enumerate(samples)}
    ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    
    # Setup obs dataframe
    obs_df = pd.DataFrame(index=samples)
    obs_df.index.name = 'sample'
    if batch_col is not None and batch_col in adata.obs.columns:
        sample_batch_map = (
            adata.obs[[sample_col, batch_col]]
            .drop_duplicates()
            .set_index(sample_col)[batch_col]
            .to_dict()
        )
        obs_df[batch_col] = [sample_batch_map.get(s, 'Unknown') for s in samples]
    
    var_df = adata.var.copy()
    
    # Create base AnnData
    X_main = np.zeros((n_samples, n_genes), dtype=np.float32)
    pseudobulk_adata = sc.AnnData(X=X_main, obs=obs_df, var=var_df)
    
    # Get cell assignments as indices
    t0 = time.time()
    cell_sample_idx = adata.obs[sample_col].map(sample_to_idx).values
    cell_ct_idx = adata.obs[celltype_col].map(ct_to_idx).values
    
    # Handle cells not in our sample/celltype lists
    valid_mask = ~(pd.isna(cell_sample_idx) | pd.isna(cell_ct_idx))
    cell_sample_idx = cell_sample_idx.astype(int)
    cell_ct_idx = cell_ct_idx.astype(int)
    
    if verbose:
        print(f"  Index mapping: {time.time()-t0:.2f}s")
    
    # Process each cell type using sparse indicator matrices
    t0 = time.time()
    
    # Ensure X is in efficient format for row slicing
    if issparse(adata.X):
        X = adata.X.tocsr() if not isinstance(adata.X, csr_matrix) else adata.X
    else:
        X = adata.X
    
    for ct_idx, cell_type in enumerate(cell_types):
        t0_ct = time.time()
        
        # Get cells belonging to this cell type
        ct_mask = (cell_ct_idx == ct_idx) & valid_mask
        ct_cell_indices = np.where(ct_mask)[0]
        
        if len(ct_cell_indices) == 0:
            pseudobulk_adata.layers[cell_type] = np.zeros((n_samples, n_genes), dtype=np.float32)
            continue
        
        # Get sample assignments for these cells
        ct_sample_idx = cell_sample_idx[ct_cell_indices]
        
        # Build sparse indicator matrix (n_samples x n_ct_cells)
        # indicator[sample, cell] = 1 if cell belongs to sample
        row_idx = ct_sample_idx
        col_idx = np.arange(len(ct_cell_indices))
        data = np.ones(len(ct_cell_indices), dtype=np.float32)
        
        indicator = csr_matrix(
            (data, (row_idx, col_idx)),
            shape=(n_samples, len(ct_cell_indices))
        )
        
        # Count cells per sample for this cell type
        counts = np.array(indicator.sum(axis=1)).flatten()
        counts[counts == 0] = 1  # Avoid division by zero
        
        # Extract expression for this cell type's cells
        if issparse(X):
            X_ct = X[ct_cell_indices, :]
        else:
            X_ct = X[ct_cell_indices, :]
        
        # Compute sums: (n_samples x n_ct_cells) @ (n_ct_cells x n_genes) = (n_samples x n_genes)
        if issparse(X_ct):
            sums = indicator @ X_ct
            sums = np.asarray(sums.todense())
        else:
            sums = indicator @ X_ct
        
        # Compute means
        layer_matrix = sums / counts[:, np.newaxis]
        
        pseudobulk_adata.layers[cell_type] = layer_matrix.astype(np.float32)
        
        if verbose:
            print(f"  Layer {ct_idx+1}/{n_celltypes} ({cell_type}): {time.time()-t0_ct:.2f}s, {len(ct_cell_indices)} cells")
    
    if verbose:
        print(f"  Total layer creation: {time.time()-t0:.2f}s")
    
    print(f"\n[TIMING] Phase 1 total: {time.time()-t0_total:.2f}s")
    
    return pseudobulk_adata

def _normalize_gpu(
    temp_adata: sc.AnnData,
    target_sum: float,
    device: torch.device,
    max_cells_per_batch: int
):
    """GPU-accelerated normalization matching scanpy's normalize_total and log1p."""
    n_obs = temp_adata.n_obs

    if issparse(temp_adata.X):
        temp_adata.X = temp_adata.X.toarray()

    for start_idx in range(0, n_obs, max_cells_per_batch):
        end_idx = min(start_idx + max_cells_per_batch, n_obs)
        chunk = np.asarray(temp_adata.X[start_idx:end_idx])
        X_chunk = torch.from_numpy(chunk).float().to(device)

        row_sums = X_chunk.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1
        X_chunk = X_chunk * (target_sum / row_sums)

        X_chunk = torch.log1p(X_chunk)

        temp_adata.X[start_idx:end_idx] = X_chunk.cpu().numpy()

        del X_chunk
        clear_gpu_memory()

def _create_final_expression_matrix(
    all_hvg_data: dict,
    all_gene_names: list,
    samples: list,
    cell_types: list,
    verbose: bool
) -> pd.DataFrame:
    """Create final expression matrix in original format (identical to CPU version)."""
    if verbose:
        print("[pseudobulk] Building final HVG expression matrix (CT x sample).")

    cell_type_groups = {}
    for gene in set(all_gene_names):
        cell_type = gene.split(' - ')[0]
        cell_type_groups.setdefault(cell_type, []).append(gene)

    for ct in cell_type_groups:
        cell_type_groups[ct].sort()

    final_expression_df = pd.DataFrame(
        index=sorted(cell_types),
        columns=samples,
        dtype=object
    )

    for cell_type in final_expression_df.index:
        ct_genes = cell_type_groups.get(cell_type, [])
        for sample in final_expression_df.columns:
            values = []
            gene_names = []
            for gene in ct_genes:
                if sample in all_hvg_data and gene in all_hvg_data[sample]:
                    values.append(all_hvg_data[sample][gene])
                    gene_names.append(gene)
            if values:
                final_expression_df.at[cell_type, sample] = pd.Series(values, index=gene_names)
            else:
                final_expression_df.at[cell_type, sample] = pd.Series(dtype=float)

    return final_expression_df


def _compute_cell_proportions(
    adata: sc.AnnData,
    samples: list,
    cell_types: list,
    sample_col: str,
    celltype_col: str
) -> pd.DataFrame:
    """Compute cell type proportions for each sample (identical to CPU version)."""
    proportion_df = pd.DataFrame(index=cell_types, columns=samples, dtype=float)

    for sample in samples:
        sample_mask = (adata.obs[sample_col] == sample)
        total_cells = int(sample_mask.sum())
        for cell_type in cell_types:
            ct_mask = sample_mask & (adata.obs[celltype_col] == cell_type)
            n_cells = int(ct_mask.sum())
            proportion = n_cells / total_cells if total_cells > 0 else 0.0
            proportion_df.loc[cell_type, sample] = proportion

    return proportion_df

def compute_pseudobulk_adata_linux(
    adata: sc.AnnData,
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './',
    Save: bool = True,
    n_features: int = 2000,
    normalize: bool = True,
    target_sum: float = 1e4,
    atac: bool = False,
    verbose: bool = False,
    batch_size: int = 100,
    max_cells_per_batch: int = 50000,
    combat_timeout: float = 1800.0
) -> Tuple[Dict, sc.AnnData]:
    """
    Explicit GPU version with additional control parameters.
    """
    wrapper_start = time.time()
    
    t0 = time.time()
    set_global_seed(seed=42, verbose=verbose)
    print(f"[TIMING] set_global_seed: {time.time() - t0:.2f}s")

    cell_expression_hvg_df, cell_proportion_df, final_adata = compute_pseudobulk_layers_gpu(
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
        batch_size=batch_size,
        max_cells_per_batch=max_cells_per_batch,
        combat_timeout=combat_timeout
    )

    # Attach sample metadata
    t0 = time.time()
    final_adata = _extract_sample_metadata(
        cell_adata=adata,
        sample_adata=final_adata,
        sample_col=sample_col,
    )
    print(f"[TIMING] Extract sample metadata: {time.time() - t0:.2f}s")

    # Attach proportions
    final_adata.uns['cell_proportions'] = cell_proportion_df

    if Save:
        t0 = time.time()
        pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
        sc.write(os.path.join(pseudobulk_dir, "pseudobulk_sample.h5ad"), final_adata)
        print(f"[TIMING] Save h5ad: {time.time() - t0:.2f}s")

    pseudobulk = {
        "cell_expression_corrected": cell_expression_hvg_df,
        "cell_proportion": cell_proportion_df.T
    }

    print(f"\n[TIMING] compute_pseudobulk_adata_linux total: {time.time() - wrapper_start:.2f}s")

    return pseudobulk, final_adata