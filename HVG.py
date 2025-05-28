import warnings
import numpy as np
import pandas as pd
from skmisc.loess import loess
from anndata import AnnData
from typing import Optional
import numpy as np
import pandas as pd
import time

def find_hvgs(
    adata: AnnData,
    sample_column: str,
    num_features: Optional[int] = None,
    batch_key: Optional[str] = None,
    check_values: bool = True,
    inplace: bool = True,
    span: float = 0.3,
    threshold: float = 1.0,
    verbose: bool = False
) -> pd.DataFrame | None:
    warnings.filterwarnings("ignore", category=UserWarning, module="skmisc.loess")

    gene_expr = adata.to_df()

    if verbose:
        print("Computing means and variability metrics...")

    if batch_key is None:
        sample_means = gene_expr.groupby(adata.obs[sample_column]).mean()
        gene_mean = sample_means.mean(axis=0)
        if num_features is not None:
            gene_var = sample_means.var(axis=0)
            variability_metric = gene_var
            metric_name = 'variance'
        else:
            gene_sd = sample_means.std(axis=0)
            variability_metric = gene_sd
            metric_name = 'sd'

        not_const = variability_metric > 0

        log_mean = np.log10(gene_mean[not_const] + 1e-8)
        log_variability = np.log10(variability_metric[not_const] + 1e-8)

        if verbose:
            print("Fitting LOESS model...")
        try:
            loess_model = loess(x=log_mean.values, y=log_variability.values, span=span, degree=2)
            loess_model.fit()
            fitted_log_variability = loess_model.outputs.fitted_values
        except Exception as e:
            raise RuntimeError(f"LOESS fitting failed: {e}")

        smoothed_variability = np.zeros_like(variability_metric)
        smoothed_variability[not_const] = 10 ** fitted_log_variability
        smoothed_variability[~not_const] = variability_metric[~not_const]

        residuals = variability_metric - smoothed_variability
        residual_std = np.std(residuals)
        clipped_residuals = np.clip(residuals, -2 * residual_std, 2 * residual_std)
        normalized_var = clipped_residuals / residual_std

        if num_features is not None:
            top_hvg_genes = normalized_var.nlargest(num_features).index
            highly_variable = normalized_var.index.isin(top_hvg_genes)
        else:
            highly_variable = normalized_var > threshold

        if inplace:
            adata.var['gene_mean'] = gene_mean
            adata.var[f'gene_{metric_name}'] = variability_metric
            adata.var[f'smoothed_{metric_name}'] = smoothed_variability
            adata.var['normalized_variance'] = normalized_var
            adata.var['hvg_rank'] = normalized_var.rank(method='dense', ascending=False)
            adata.var['highly_variable'] = highly_variable
            if verbose:
                print("HVGs updated in AnnData object.")
        else:
            hvgs_df = pd.DataFrame({
                'gene_mean': gene_mean,
                f'gene_{metric_name}': variability_metric,
                f'smoothed_{metric_name}': smoothed_variability,
                'normalized_variance': normalized_var,
                'hvg_rank': normalized_var.rank(method='dense', ascending=False),
                'highly_variable': highly_variable
            })
            if verbose:
                print("Returning HVG DataFrame.")
            return hvgs_df
    return None

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse
import time

def select_hvf_loess(pseudobulk, n_features=2000, min_mean=0.0125, max_mean=3, 
                     min_disp=0.5, verbose=False):
    """
    Select highly variable features (HVFs) from pseudobulk data using Scanpy.
    
    Parameters
    ----------
    pseudobulk : pd.DataFrame
        DataFrame indexed by cell type and with columns as sample names.
        Each cell contains a pd.Series with gene expression values (indexed by gene names).
    n_features : int, default 2000
        Number of top HVFs to select.
    min_mean : float, default 0.0125
        Minimum mean expression for gene filtering.
    max_mean : float, default 3
        Maximum mean expression for gene filtering.
    min_disp : float, default 0.5
        Minimum dispersion for gene filtering.
    verbose : bool, default False
        If True, prints progress information.
        
    Returns
    -------
    adata : anndata.AnnData
        AnnData object with samples as observations and genes as variables.
        Contains only HVG after subsetting.
    expression_df : pd.DataFrame
        Sample-by-gene expression matrix as a DataFrame (only HVGs).
    """
    start_time = time.time() if verbose else None
    
    if verbose:
        print("Constructing sparse sample-by-feature matrix from pseudobulk...")
    
    # Step 1: Build unified gene index efficiently
    all_genes = set()
    for sample in pseudobulk.columns:
        for cell_type in pseudobulk.index:
            s = pseudobulk.loc[cell_type, sample]
            if isinstance(s, pd.Series) and len(s) > 0:
                all_genes.update(s.index)
    
    gene_index = pd.Index(sorted(all_genes))
    n_genes = len(gene_index)
    n_samples = len(pseudobulk.columns)
    
    if verbose:
        print(f"Found {n_genes} unique genes across {n_samples} samples")
    
    # Step 2: Build sparse matrix using COO format for efficient construction
    row_indices = []
    col_indices = []
    data_values = []
    
    for sample_idx, sample in enumerate(pseudobulk.columns):
        for cell_type in pseudobulk.index:
            s = pseudobulk.loc[cell_type, sample]
            if isinstance(s, pd.Series) and len(s) > 0:
                # Use get_indexer for batch lookup - much faster than individual lookups
                gene_positions = gene_index.get_indexer(s.index)
                valid_mask = gene_positions >= 0
                
                if valid_mask.any():
                    n_valid = valid_mask.sum()
                    row_indices.extend([sample_idx] * n_valid)
                    col_indices.extend(gene_positions[valid_mask])
                    data_values.extend(s.values[valid_mask])
    
    # Convert to CSR for efficient arithmetic operations
    sparse_matrix = sparse.coo_matrix(
        (data_values, (row_indices, col_indices)), 
        shape=(n_samples, n_genes)
    ).tocsr()
    
    # Step 3: Create AnnData object
    if verbose:
        print("Creating AnnData object...")
    
    # Create observation (sample) metadata
    obs_df = pd.DataFrame(index=pseudobulk.columns)
    obs_df.index.name = 'sample'
    
    # Create variable (gene) metadata
    var_df = pd.DataFrame(index=gene_index)
    var_df.index.name = 'gene'
    
    # Create AnnData object with sparse matrix
    adata = ad.AnnData(X=sparse_matrix, obs=obs_df, var=var_df)
    
    if verbose:
        print(f"Created AnnData object: {adata}")
    
    # Step 4: Use Scanpy to identify highly variable genes
    if verbose:
        print("Identifying highly variable genes using Scanpy...")
    
    # Store raw counts
    adata.raw = adata
    
    # Identify highly variable genes and subset the data
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_features,
        min_mean=min_mean,
        max_mean=max_mean,
        min_disp=min_disp,
        subset=True  # Subset to keep only HVGs
    )
    
    if verbose:
        print(f"Subsetted to {adata.n_vars} highly variable genes")
    
    if verbose:
        print("Normalizing and log-transforming data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Check for NaN values and remove samples with NaNs
    if verbose:
        print("Checking for NaN values after normalization and log transformation...")
    
    # Convert to dense array if sparse for NaN checking
    X_dense = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    
    # Check for NaN values per sample (row)
    nan_mask = np.isnan(X_dense).any(axis=1)
    n_nan_samples = nan_mask.sum()
    
    if n_nan_samples > 0:
        if verbose:
            print(f"Found {n_nan_samples} genes with NaN values. Removing these samples...")
        
        # Remove samples with NaN values
        adata = adata[~nan_mask, :].copy()
        
        if verbose:
            print(f"Remaining samples after NaN removal: {adata.n_obs}")
    else:
        if verbose:
            print("No NaN values found in the data.")

    # Step 5: Create expression DataFrame
    if verbose:
        print("Creating expression DataFrame...")
    
    # Expression matrix as DataFrame (already contains only HVGs)
    expression_df = pd.DataFrame(
        adata.X.toarray() if sparse.issparse(adata.X) else adata.X,
        index=adata.obs.index,
        columns=adata.var.index
    )
    
    if verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n[select_hvf_loess] Completed in {elapsed_time:.2f} seconds")
        print(f"HVG expression matrix shape: {expression_df.shape}")
    
    return expression_df, adata

def highly_variable_gene_selection(
    cell_expression_corrected_df: pd.DataFrame,
    n_top_genes: int = None,
    loess_span: float = 0.3,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Identify Highly Variable Genes (HVGs) for each cell type independently using LOESS regression
    of log(variance) vs. log(mean). Truncate each expression Series to only include HVGs.
    Gene names are prefixed with cell type at the end.
    """
    start_time = time.time() if verbose else None
    hvg_truncated_df = cell_expression_corrected_df.copy(deep=True)

    if verbose:
        print("Starting HVG selection process...")

    for ctype in hvg_truncated_df.index:
        if verbose:
            print(f"\nProcessing cell type: {ctype}")

        # Collect sample expression Series for this cell type
        sample_series = [
            hvg_truncated_df.loc[ctype, sample_id]
            for sample_id in hvg_truncated_df.columns
            if isinstance(hvg_truncated_df.loc[ctype, sample_id], pd.Series)
        ]
        if len(sample_series) == 0:
            if verbose:
                print(f"Skipping {ctype} due to lack of data.")
            continue

        # Extract gene names from the first sample
        gene_names = sample_series[0].index

        # Confirm all Series have the same index
        sample_series = [
            s for s in sample_series if s.index.equals(gene_names)
        ]
        if len(sample_series) < 2:
            if verbose:
                print(f"Skipping {ctype} due to inconsistent gene names or insufficient samples.")
            continue

        # Stack expression vectors
        expr_matrix = np.vstack([s.values for s in sample_series])
        n_samples, n_genes = expr_matrix.shape

        if verbose:
            print(f"Computing mean and variance for {n_genes} genes across {n_samples} samples...")

        gene_means = expr_matrix.mean(axis=0)
        gene_vars = expr_matrix.var(axis=0)

        # Check for problematic data before LOESS fitting
        if n_genes < 3:
            if verbose:
                print(f"Skipping {ctype}: insufficient genes ({n_genes}) for LOESS fitting.")
            continue

        # Remove genes with zero variance or zero mean
        valid_mask = (gene_vars > 0) & (gene_means > 0) & np.isfinite(gene_vars) & np.isfinite(gene_means)
        
        if np.sum(valid_mask) < 3:
            if verbose:
                print(f"Skipping {ctype}: insufficient valid genes ({np.sum(valid_mask)}) for LOESS fitting.")
            continue

        valid_means = gene_means[valid_mask]
        valid_vars = gene_vars[valid_mask]
        valid_gene_names = gene_names[valid_mask]

        epsilon = 1e-8
        log_means = np.log(valid_means + epsilon)
        log_vars = np.log(valid_vars + epsilon)

        # Additional check for variance in log values
        if np.var(log_means) == 0 or np.var(log_vars) == 0:
            if verbose:
                print(f"Skipping {ctype}: no variance in log-transformed values.")
            continue

        if verbose:
            print(f"Fitting LOESS model with span={loess_span} on {len(valid_means)} valid genes...")

        try:
            loess_model = loess(x=log_means, y=log_vars, span=loess_span, degree=2)
            loess_model.fit()
            fitted_vals = loess_model.outputs.fitted_values
        except Exception as e:
            if verbose:
                print(f"LOESS fitting failed for {ctype}: {e}. Using variance-based selection instead.")
            # Fallback: select genes with highest variance
            if n_top_genes is not None:
                n_select = min(n_top_genes, len(valid_gene_names))
                hvg_indices = np.argsort(valid_vars)[-n_select:]
            else:
                # Select top 50% by variance as fallback
                n_select = max(1, len(valid_gene_names) // 2)
                hvg_indices = np.argsort(valid_vars)[-n_select:]
            
            selected_genes = valid_gene_names[hvg_indices]
            prefixed_genes = [f"{ctype} - {g}" for g in selected_genes]

            if verbose:
                print(f"Selected {len(selected_genes)} genes using variance-based fallback for {ctype}.")

            # Update the dataframe with fallback selection
            for sample_id in hvg_truncated_df.columns:
                orig_series = hvg_truncated_df.loc[ctype, sample_id]
                if isinstance(orig_series, pd.Series):
                    truncated_values = orig_series.loc[selected_genes].values
                    hvg_truncated_df.at[ctype, sample_id] = pd.Series(truncated_values, index=prefixed_genes)
            continue

        residuals = log_vars - fitted_vals

        if verbose:
            print("Selecting highly variable genes...")

        if n_top_genes is not None:
            positive_mask = residuals > 0
            candidate_genes_idx = np.where(positive_mask)[0]
            if len(candidate_genes_idx) == 0:
                # If no positive residuals, select top genes by variance
                if verbose:
                    print(f"No positive residuals for {ctype}, using variance-based selection.")
                n_select = min(n_top_genes, len(valid_gene_names))
                hvg_genes_idx = np.argsort(valid_vars)[-n_select:]
            else:
                candidate_sorted = candidate_genes_idx[np.argsort(residuals[candidate_genes_idx])[::-1]]
                hvg_genes_idx = candidate_sorted[:n_top_genes]
                if verbose:
                    print(f"Selected top {len(hvg_genes_idx)} HVGs for {ctype}.")
        else:
            hvg_genes_idx = np.where(residuals > 0)[0]
            if len(hvg_genes_idx) == 0:
                # Fallback: select top 50% by variance
                n_select = max(1, len(valid_gene_names) // 2)
                hvg_genes_idx = np.argsort(valid_vars)[-n_select:]
                if verbose:
                    print(f"No positive residuals for {ctype}, selected top {len(hvg_genes_idx)} by variance.")
            else:
                if verbose:
                    print(f"Selected {len(hvg_genes_idx)} HVGs for {ctype} (using positive residuals).")

        selected_genes = valid_gene_names[hvg_genes_idx]
        prefixed_genes = [f"{ctype} - {g}" for g in selected_genes]

        if verbose:
            print("Updating truncated expression matrix with prefixed gene names...")

        for sample_id in hvg_truncated_df.columns:
            orig_series = hvg_truncated_df.loc[ctype, sample_id]
            if isinstance(orig_series, pd.Series):
                truncated_values = orig_series.loc[selected_genes].values
                hvg_truncated_df.at[ctype, sample_id] = pd.Series(truncated_values, index=prefixed_genes)

    if verbose:
        print("\nHVG selection process completed.")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\n[HVG for each cell type] Total execution time: {elapsed_time:.2f} seconds\n\n")

    return hvg_truncated_df