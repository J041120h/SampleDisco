import os
import time
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from typing import Tuple, Dict, List
from tf_idf import tfidf_memory_efficient
import contextlib
import io
import signal
import patsy


from utils.limma import limma

def _extract_sample_metadata(
    cell_adata: sc.AnnData,
    sample_adata: sc.AnnData,
    sample_col: str,
    exclude_cols: List[str] | None = None,
) -> sc.AnnData:
    """Detect and copy sample-level metadata (identical to GPU version)."""
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

def compute_pseudobulk_layers(
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
    combat_timeout: int = 1800  # 30 minutes in seconds
) -> Tuple[pd.DataFrame, pd.DataFrame, sc.AnnData]:
    """
    CPU pseudobulk computation matching the GPU version's logic.

    - Pseudobulk by sample × cell type into layers
    - Normalize (RNA: normalize_total+log1p; ATAC: TF-IDF)
    - Remove NaNs
    - Batch correction:
        * Try ComBat if ≥2 samples per batch
        * If ComBat fails/times out, fall back to limma (~ batch)
    - HVGs per cell type, then global HVGs
    - Final sample × gene AnnData + cell proportions
    """
    start_time = time.time() if verbose else None

    # Create output directory
    pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
    os.makedirs(pseudobulk_dir, exist_ok=True)

    # Check if batch correction should be applied (based on original cell-level adata)
    batch_correction = (
        batch_col is not None and
        batch_col in adata.obs.columns and
        not adata.obs[batch_col].isnull().all()
    )

    if batch_correction and adata.obs[batch_col].isnull().any():
        adata.obs[batch_col] = adata.obs[batch_col].fillna("Unknown")

    if verbose:
        print(f"[pseudobulk-CPU] Batch correction enabled: {batch_correction} (col='{batch_col}')")

    # Get unique samples and cell types
    samples = sorted(adata.obs[sample_col].unique())
    cell_types = sorted(adata.obs[celltype_col].unique())

    if verbose:
        print(f"[pseudobulk-CPU] {len(cell_types)} cell types, {len(samples)} samples")
        print(f"[pseudobulk-CPU] ComBat timeout: {combat_timeout}s")

    # Phase 1: Create pseudobulk AnnData with layers
    pseudobulk_adata = _create_pseudobulk_layers(
        adata, samples, cell_types, sample_col, celltype_col, batch_col, verbose
    )

    # Phase 2: Process each cell type layer
    all_hvg_data = {}
    all_gene_names = []
    cell_types_to_remove = []

    for cell_type in cell_types:
        if verbose:
            print(f"\n[pseudobulk-CPU] Processing cell type: {cell_type}")

        try:
            # Temporary AnnData for this cell type
            layer_data = pseudobulk_adata.layers[cell_type]
            temp_adata = sc.AnnData(
                X=layer_data.copy(),
                obs=pseudobulk_adata.obs.copy(),
                var=pseudobulk_adata.var.copy()
            )

            # Filter out genes with zero expression
            sc.pp.filter_genes(temp_adata, min_cells=1)
            if temp_adata.n_vars == 0:
                if verbose:
                    print("  - No expressed genes, skipping.")
                cell_types_to_remove.append(cell_type)
                continue

            # Normalization (mirror GPU semantics)
            if normalize:
                if atac:
                    tfidf_memory_efficient(temp_adata, scale_factor=target_sum)
                else:
                    sc.pp.normalize_total(temp_adata, target_sum=target_sum)
                    sc.pp.log1p(temp_adata)

            # Remove NaN genes before batch correction
            if issparse(temp_adata.X):
                nan_genes = np.array(np.isnan(temp_adata.X.toarray()).any(axis=0)).flatten()
            else:
                nan_genes = np.isnan(temp_adata.X).any(axis=0)

            if nan_genes.any():
                if verbose:
                    print(f"  - Removing {nan_genes.sum()} NaN genes before batch correction.")
                temp_adata = temp_adata[:, ~nan_genes].copy()

            # Batch correction (ComBat → limma)
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

                        # Remove NaNs generated by ComBat
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

                        if verbose:
                            print("  - ComBat applied successfully.")

                    except (TimeoutError, Exception):
                        if verbose:
                            print("  - ComBat failed or timed out; will try limma.")

                else:
                    if verbose:
                        print(f"  - Skipping ComBat (min batch size={min_batch_size} < 2).")

                # limma fallback / alternative
                if not batch_correction_applied:
                    try:
                        if verbose:
                            print("  - Applying limma batch regression (~ batch).")

                        # Ensure dense
                        if issparse(temp_adata.X):
                            X_dense = temp_adata.X.toarray()
                        else:
                            X_dense = temp_adata.X.copy()

                        covariate_formula = f'~ {batch_col}'
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

                        # Remove NaNs if any
                        if issparse(temp_adata.X):
                            nan_genes_post = np.array(np.isnan(temp_adata.X.toarray()).any(axis=0)).flatten()
                        else:
                            nan_genes_post = np.isnan(temp_adata.X).any(axis=0)
                        if nan_genes_post.any():
                            temp_adata = temp_adata[:, ~nan_genes_post].copy()

                        if verbose:
                            print("  - limma batch regression applied.")

                    except Exception as e:
                        if verbose:
                            print(f"  - limma failed ({type(e).__name__}); continuing without batch correction.")

            elif verbose:
                print("  - Batch correction skipped (conditions not met).")

            if verbose and batch_correction_applied:
                print(f"  - Batch correction method: {batch_correction_method}")

            # HVG selection
            if verbose:
                print(f"  - Selecting top {n_features} HVGs")

            n_hvgs = min(n_features, temp_adata.n_vars)
            sc.pp.highly_variable_genes(
                temp_adata,
                n_top_genes=n_hvgs,
                subset=False
            )

            hvg_mask = temp_adata.var['highly_variable']
            hvg_genes = temp_adata.var.index[hvg_mask].tolist()

            if len(hvg_genes) == 0:
                if verbose:
                    print("  - No HVGs found, skipping cell type.")
                cell_types_to_remove.append(cell_type)
                continue

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

            if verbose:
                print(f"  - Selected {len(hvg_genes)} HVGs.")

        except Exception as e:
            if verbose:
                print(f"  - Failed to process {cell_type}: {e}")
            cell_types_to_remove.append(cell_type)

    # Remove failed cell types from list
    cell_types = [ct for ct in cell_types if ct not in cell_types_to_remove]

    if verbose:
        print("\n[pseudobulk-CPU] Summary:")
        print(f"  - Successful cell types: {len(cell_types)}")
        print(f"  - Failed/empty cell types: {len(cell_types_to_remove)}")

    # Phase 3: Create concatenated AnnData with all cell type HVGs
    if verbose:
        print("\n[pseudobulk-CPU] Concatenating HVGs across cell types.")

    all_unique_genes = sorted(list(set(all_gene_names)))
    concat_matrix = np.zeros((len(samples), len(all_unique_genes)), dtype=float)

    for i, sample in enumerate(samples):
        for j, gene in enumerate(all_unique_genes):
            if sample in all_hvg_data and gene in all_hvg_data[sample]:
                concat_matrix[i, j] = all_hvg_data[sample][gene]

    concat_adata = sc.AnnData(
        X=concat_matrix,
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=all_unique_genes)
    )

    # Final HVG selection on concatenated data
    if verbose:
        print(f"[pseudobulk-CPU] Final HVG selection on concatenated matrix (genes={concat_adata.n_vars}).")

    sc.pp.filter_genes(concat_adata, min_cells=1)
    final_n_hvgs = min(n_features, concat_adata.n_vars)
    sc.pp.highly_variable_genes(
        concat_adata,
        n_top_genes=final_n_hvgs,
        subset=True
    )

    if verbose:
        print(f"[pseudobulk-CPU] Final HVGs retained: {concat_adata.n_vars}")

    # Create final expression DataFrame in original format
    final_expression_df = _create_final_expression_matrix(
        all_hvg_data,
        concat_adata.var.index.tolist(),  # Only final HVGs
        samples,
        cell_types,
        verbose
    )

    # Phase 4: Compute cell proportions
    cell_proportion_df = _compute_cell_proportions(
        adata, samples, cell_types, sample_col, celltype_col
    )

    concat_adata.uns['cell_proportions'] = cell_proportion_df

    # Save final outputs
    final_expression_df.to_csv(
        os.path.join(pseudobulk_dir, "expression_hvg.csv")
    )
    cell_proportion_df.to_csv(
        os.path.join(pseudobulk_dir, "proportion.csv")
    )

    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\n[pseudobulk-CPU] Done. Runtime: {elapsed_time:.2f} s")
        print(f"[pseudobulk-CPU] Final HVG matrix shape: {final_expression_df.shape}")
        print(f"[pseudobulk-CPU] Final AnnData shape: {concat_adata.shape}")

    return final_expression_df, cell_proportion_df, concat_adata

def _create_pseudobulk_layers(
    adata: sc.AnnData,
    samples: list,
    cell_types: list,
    sample_col: str,
    celltype_col: str,
    batch_col: str,
    verbose: bool
) -> sc.AnnData:
    """Optimized pseudobulk creation using sparse indicator matrix multiplication."""
    from scipy.sparse import csr_matrix
    
    n_samples = len(samples)
    n_celltypes = len(cell_types)
    n_genes = adata.n_vars
    
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
    
    # Map cells to sample/celltype indices (vectorized)
    cell_sample_idx = adata.obs[sample_col].map(sample_to_idx).values
    cell_ct_idx = adata.obs[celltype_col].map(ct_to_idx).values
    
    # Handle unmapped cells
    valid_mask = ~(pd.isna(cell_sample_idx) | pd.isna(cell_ct_idx))
    cell_sample_idx = np.where(valid_mask, cell_sample_idx, -1).astype(int)
    cell_ct_idx = np.where(valid_mask, cell_ct_idx, -1).astype(int)
    
    # Ensure X is in CSR format for efficient row slicing
    if issparse(adata.X):
        X = adata.X.tocsr() if not isinstance(adata.X, csr_matrix) else adata.X
    else:
        X = adata.X
    
    for ct_idx, cell_type in enumerate(cell_types):
        if verbose:
            print(f"[pseudobulk-CPU] Creating layer for {cell_type}")
        
        # Get cells belonging to this cell type
        ct_mask = (cell_ct_idx == ct_idx) & valid_mask
        ct_cell_indices = np.where(ct_mask)[0]
        
        if len(ct_cell_indices) == 0:
            pseudobulk_adata.layers[cell_type] = np.zeros((n_samples, n_genes), dtype=np.float32)
            continue
        
        # Get sample assignments for these cells
        ct_sample_idx = cell_sample_idx[ct_cell_indices]
        
        # Build sparse indicator matrix (n_samples x n_ct_cells)
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
        X_ct = X[ct_cell_indices, :]
        
        # Compute sums via sparse matmul
        if issparse(X_ct):
            sums = indicator @ X_ct
            sums = np.asarray(sums.todense())
        else:
            sums = indicator @ X_ct
        
        # Compute means
        layer_matrix = sums / counts[:, np.newaxis]
        
        pseudobulk_adata.layers[cell_type] = layer_matrix.astype(np.float32)
    
    return pseudobulk_adata


def _compute_cell_proportions(
    adata: sc.AnnData,
    samples: list,
    cell_types: list,
    sample_col: str,
    celltype_col: str
) -> pd.DataFrame:
    """Optimized cell proportions using crosstab."""
    # Use pandas crosstab for vectorized counting
    counts = pd.crosstab(
        adata.obs[celltype_col],
        adata.obs[sample_col]
    )
    
    # Reindex to ensure all cell types and samples are present
    counts = counts.reindex(index=cell_types, columns=samples, fill_value=0)
    
    # Compute proportions (normalize by column sums)
    totals = counts.sum(axis=0)
    totals[totals == 0] = 1  # Avoid division by zero
    proportion_df = counts / totals
    
    return proportion_df.astype(float)


# ---------------------------------------------------------------------
# Final expression matrix + proportions (same as GPU version)
# ---------------------------------------------------------------------
def _create_final_expression_matrix(
    all_hvg_data: dict,
    all_gene_names: list,
    samples: list,
    cell_types: list,
    verbose: bool
) -> pd.DataFrame:
    """Create final expression matrix in original format (identical to GPU version)."""
    if verbose:
        print("[pseudobulk-CPU] Building final HVG expression matrix (CT x sample).")

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
    """Compute cell type proportions for each sample (identical to GPU version)."""
    proportion_df = pd.DataFrame(
        index=cell_types,
        columns=samples,
        dtype=float
    )

    for sample in samples:
        sample_mask = (adata.obs[sample_col] == sample)
        total_cells = int(sample_mask.sum())
        for cell_type in cell_types:
            ct_mask = sample_mask & (adata.obs[celltype_col] == cell_type)
            n_cells = int(ct_mask.sum())
            proportion = n_cells / total_cells if total_cells > 0 else 0.0
            proportion_df.loc[cell_type, sample] = proportion

    return proportion_df

def compute_pseudobulk_adata(
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
    combat_timeout: int = 1800  # 30 minutes default
) -> Tuple[Dict, sc.AnnData]:
    """
    Backward compatibility wrapper for the original function signature.

    Returns
    -------
    pseudobulk : dict
        Dictionary containing expression and proportion DataFrames
    final_adata : sc.AnnData
        Final AnnData object with samples x genes (final HVGs only)
    """
    from utils.random_seed import set_global_seed
    set_global_seed(seed=42, verbose=verbose)

    cell_expression_hvg_df, cell_proportion_df, final_adata = compute_pseudobulk_layers(
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
        combat_timeout=combat_timeout
    )

    # Attach sample metadata
    final_adata = _extract_sample_metadata(
        cell_adata=adata,
        sample_adata=final_adata,
        sample_col=sample_col,
    )

    # Attach proportions
    final_adata.uns['cell_proportions'] = cell_proportion_df

    if Save:
        pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
        os.makedirs(pseudobulk_dir, exist_ok=True)
        sc.write(os.path.join(pseudobulk_dir, "pseudobulk_sample.h5ad"), final_adata)

    # Create backward-compatible dictionary matching GPU output
    pseudobulk = {
        "cell_expression_corrected": cell_expression_hvg_df,
        "cell_proportion": cell_proportion_df.T
    }

    return pseudobulk, final_adata
