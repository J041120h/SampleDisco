import warnings
import numpy as np
import pandas as pd
from skmisc.loess import loess
from anndata import AnnData
from typing import Optional
import scipy.sparse as sp_sparse

def find_hvgs(
    adata: AnnData,
    sample_column: str,
    num_features: Optional[int] = None,
    batch_key: Optional[str] = None,
    check_values: bool = True,
    inplace: bool = True,
    span: float = 0.3,
    threshold: float = 1.0,
) -> pd.DataFrame | None:
    """
    Identify highly variable genes (HVGs) across samples and optionally across batches in an AnnData object.

    This function computes HVGs by fitting a LOESS model to regress variability (variance or standard deviation)
    against mean expression. If `num_features` is specified, the top genes are selected; otherwise, a threshold on
    normalized residuals is used.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cells as rows and genes as columns.
    sample_column : str
        Column name in `adata.obs` identifying sample labels for grouping.
    num_features : Optional[int], optional
        Number of highly variable genes to select. If None, a threshold on normalized residuals is used.
    batch_key : Optional[str], optional
        Column name in `adata.obs` for batch information. If provided, HVGs are computed per batch and aggregated.
    check_values : bool, optional
        Check if gene expression values are non-negative integers. Default is True.
    inplace : bool, optional
        Whether to modify `adata` in place or return a DataFrame. Default is True.
    span : float, optional
        LOESS smoothing parameter. Default is 0.3.
    threshold : float, optional
        Threshold for normalized residuals to determine HVGs when `num_features` is None. Default is 1.0.

    Returns
    -------
    pd.DataFrame | None
        If `inplace` is False, returns a DataFrame with HVG metrics. Otherwise, modifies `adata` in place.

    Raises
    ------
    ValueError
        If `sample_column` or `batch_key` are not found in `adata.obs`, or if data checks fail.
    RuntimeError
        If LOESS fitting fails.
    """
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="skmisc.loess")

    # Step 1: Prepare Data
    gene_expr = adata.to_df()

    # if check_values:
    #     if not np.all(gene_expr.values >= 0):
    #         raise ValueError("Gene expression data contains negative values.")
    #     if not np.all(np.floor(gene_expr.values) == gene_expr.values):
    #         raise ValueError("Gene expression data contains non-integer values.")

    # Step 2: Compute Means and Variability Metrics
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

        # Log-transform the means and variability metric
        log_mean = np.log10(gene_mean[not_const] + 1e-8)
        log_variability = np.log10(variability_metric[not_const] + 1e-8)

        # Step 3: LOESS Smoothing
        try:
            loess_model = loess(x=log_mean.values, y=log_variability.values, span=span, degree=2)
            loess_model.fit()
            fitted_log_variability = loess_model.outputs.fitted_values
        except Exception as e:
            raise RuntimeError(f"LOESS fitting failed: {e}")

        # Initialize smoothed_variability
        smoothed_variability = np.zeros_like(variability_metric)
        smoothed_variability[not_const] = 10 ** fitted_log_variability
        smoothed_variability[~not_const] = variability_metric[~not_const]

        # Step 4: Clipping
        residuals = variability_metric - smoothed_variability
        residual_std = np.std(residuals)
        clipped_residuals = np.clip(residuals, -2 * residual_std, 2 * residual_std)

        # Step 5: Calculate Normalized Variance
        normalized_var = clipped_residuals / residual_std

        # Step 6: Select HVGs
        if num_features is not None:
            top_hvg_genes = normalized_var.nlargest(num_features).index
            highly_variable = normalized_var.index.isin(top_hvg_genes)
        else:
            highly_variable = normalized_var > threshold

        # Step 7: Update or Return Data
        if inplace:
            adata.var['gene_mean'] = gene_mean
            adata.var[f'gene_{metric_name}'] = variability_metric
            adata.var[f'smoothed_{metric_name}'] = smoothed_variability
            adata.var['normalized_variance'] = normalized_var
            adata.var['hvg_rank'] = normalized_var.rank(method='dense', ascending=False)
            adata.var['highly_variable'] = highly_variable
        else:
            hvgs_df = pd.DataFrame({
                'gene_mean': gene_mean,
                f'gene_{metric_name}': variability_metric,
                f'smoothed_{metric_name}': smoothed_variability,
                'normalized_variance': normalized_var,
                'hvg_rank': normalized_var.rank(method='dense', ascending=False),
                'highly_variable': highly_variable
            })
            return hvgs_df

    else:
        if batch_key not in adata.obs:
            raise ValueError(f"Batch key '{batch_key}' not found in adata.obs.")

        batch_info = adata.obs[batch_key].astype(str)
        unique_batches = batch_info.unique()
        rank_lists = []

        for batch in unique_batches:
            batch_mask = batch_info == batch
            data_batch = gene_expr[batch_mask]

            sample_means = data_batch.groupby(adata.obs[sample_column]).mean()
            gene_mean_batch = sample_means.mean(axis=0)
            if num_features is not None:
                gene_var_batch = sample_means.var(axis=0)
                variability_metric_batch = gene_var_batch
                metric_name = 'variance'
            else:
                gene_sd_batch = sample_means.std(axis=0)
                variability_metric_batch = gene_sd_batch
                metric_name = 'sd'

            not_const_batch = variability_metric_batch > 0

            log_mean_batch = np.log10(gene_mean_batch[not_const_batch] + 1e-8)
            log_variability_batch = np.log10(variability_metric_batch[not_const_batch] + 1e-8)

            # LOESS Smoothing
            try:
                loess_model_batch = loess(x=log_mean_batch.values, y=log_variability_batch.values, span=span, degree=2)
                loess_model_batch.fit()
                fitted_log_variability_batch = loess_model_batch.outputs.fitted_values
            except Exception as e:
                print(f"LOESS fitting failed for batch '{batch}' with degree=2. Error: {e}")
                try:
                    loess_model_batch = loess(x=log_mean_batch.values, y=log_variability_batch.values, span=span, degree=1)
                    loess_model_batch.fit()
                    fitted_log_variability_batch = loess_model_batch.outputs.fitted_values
                except Exception as e1:
                    print(f"LOESS fitting failed for batch '{batch}' with degree=1 as well. Error: {e1}")
                    fitted_log_variability_batch = np.zeros_like(log_variability_batch.values)

            smoothed_variability_batch = np.zeros_like(variability_metric_batch)
            smoothed_variability_batch[not_const_batch] = 10 ** fitted_log_variability_batch
            smoothed_variability_batch[~not_const_batch] = variability_metric_batch[~not_const_batch]

            residuals_batch = variability_metric_batch - smoothed_variability_batch
            residual_std_batch = np.std(residuals_batch)
            clipped_residuals_batch = np.clip(residuals_batch, -2 * residual_std_batch, 2 * residual_std_batch)

            normalized_var_batch = clipped_residuals_batch / residual_std_batch

            ranks_batch = normalized_var_batch.rank(method='dense', ascending=False)
            rank_lists.append(ranks_batch)

        rank_df = pd.concat(rank_lists, axis=1)
        median_rank = rank_df.median(axis=1)

        if num_features is not None:
            top_hvg_genes = median_rank.nsmallest(num_features).index
            highly_variable = median_rank.index.isin(top_hvg_genes)
        else:
            normalized_var = (median_rank - median_rank.min()) / (median_rank.max() - median_rank.min())
            highly_variable = normalized_var > threshold

        if inplace:
            adata.var['gene_mean'] = gene_expr.mean(axis=0)
            adata.var[f'gene_{metric_name}'] = gene_expr.var(axis=0) if num_features is not None else gene_expr.std(axis=0)
            adata.var['median_hvg_rank'] = median_rank
            adata.var['hvg_rank'] = median_rank.rank(method='dense', ascending=True)
            adata.var['highly_variable'] = highly_variable
            if num_features is not None:
                adata.var['highly_variable_nbatches'] = (rank_df <= np.percentile(rank_df.values, (num_features / gene_expr.shape[1]) * 100, axis=0)).sum(axis=1)
        else:
            hvgs_df = pd.DataFrame({
                'gene_mean': gene_expr.mean(axis=0),
                f'gene_{metric_name}': gene_expr.var(axis=0) if num_features is not None else gene_expr.std(axis=0),
                'median_hvg_rank': median_rank,
                'hvg_rank': median_rank.rank(method='dense', ascending=True),
                'highly_variable': highly_variable
            })
            if num_features is not None:
                hvgs_df['highly_variable_nbatches'] = (rank_df <= np.percentile(rank_df.values, (num_features / gene_expr.shape[1]) * 100, axis=0)).sum(axis=1)
            return hvgs_df

    return None

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

def select_hvf_loess(pseudobulk, n_features=2000, frac=0.3):
    """
    Select highly variable features (HVFs) from pseudobulk data using LOESS.
    
    Parameters:
    -----------
    pseudobulk : dict
        A dictionary that contains 'cell_expression_corrected'.
    n_features : int, default 2000
        Number of top HVFs to select (only used if the total feature count is > n_features).
    frac : float, default 0.3
        Fraction parameter passed to LOESS. Controls the degree of smoothing.
    
    Returns:
    --------
    sample_df : pd.DataFrame
        Sample-by-feature matrix (rows = samples, columns = selected features).
    top_features : pd.Index
        Index of the selected features.
    """

    cell_expr = pseudobulk['cell_expression_corrected']

    # Construct the sample-by-feature matrix
    sample_df = pd.DataFrame({
        sample: np.concatenate([cell_expr.loc[ct, sample] for ct in cell_expr.index])
        for sample in cell_expr.columns
    }).T
    sample_df.index.name = 'sample'
    sample_df.columns = [f"feature_{i}" for i in range(sample_df.shape[1])]

    # Compute per-feature mean and variance
    means = sample_df.mean(axis=0)
    variances = sample_df.var(axis=0)

    # Fit LOESS of variance vs. mean
    loess_fit = lowess(variances, means, frac=frac)

    # Interpolate to get the LOESS-fitted variance for each feature's mean
    fitted_var = np.interp(means, loess_fit[:, 0], loess_fit[:, 1])

    # Residual = observed variance - LOESS-fitted variance
    residuals = variances - fitted_var

    # Select the top n_features by largest positive residual
    if sample_df.shape[1] > n_features:
        top_features = residuals.nlargest(n_features).index
        sample_df = sample_df[top_features]
    else:
        top_features = sample_df.columns

    return sample_df, top_features

import numpy as np
import pandas as pd
from skmisc.loess import loess

def highly_variable_gene_selection(
    cell_expression_corrected_df: pd.DataFrame,
    n_top_genes: int = None,
    loess_span: float = 0.3
) -> pd.DataFrame:
    """
    Identify Highly Variable Genes (HVGs) for each cell type independently using LOESS regression
    of log(variance) vs. log(mean). Then truncate the original expression vectors so that only
    the HVGs are kept for each cell type.

    Parameters
    ----------
    cell_expression_corrected_df : pd.DataFrame
        DataFrame indexed by cell_type and columns by sample. Each cell is a vector (np.array)
        of length n_genes, representing corrected expression values for that cell type in that sample.
    n_top_genes : int, optional
        If specified, retain the top N HVGs with the largest positive residuals. If None,
        retain all genes that have a positive residual.
    loess_span : float, optional
        The span parameter used by LOESS for smoothing. Default is 0.3.

    Returns
    -------
    hvg_truncated_df : pd.DataFrame
        A new DataFrame with the same shape (same cell_type index, same sample columns),
        but each cell is a truncated vector containing only the HVG expression values.
    """

    hvg_truncated_df = cell_expression_corrected_df.copy(deep=True)

    print("Starting HVG selection process...")
    
    for ctype in hvg_truncated_df.index:
        print(f"\nProcessing cell type: {ctype}")

        sample_vectors = [
            hvg_truncated_df.loc[ctype, sample_id]
            for sample_id in hvg_truncated_df.columns
            if (hvg_truncated_df.loc[ctype, sample_id] is not None
                and len(hvg_truncated_df.loc[ctype, sample_id]) > 0)
        ]
        if len(sample_vectors) == 0:
            print(f"Skipping {ctype} due to lack of data.")
            continue

        expr_matrix = np.vstack(sample_vectors)
        n_samples, n_genes = expr_matrix.shape

        if n_samples < 2:
            print(f"Skipping {ctype} due to insufficient samples (n_samples={n_samples}).")
            continue

        print(f"Computing mean and variance for {n_genes} genes across {n_samples} samples...")
        gene_means = expr_matrix.mean(axis=0)
        gene_vars = expr_matrix.var(axis=0)

        epsilon = 1e-8
        log_means = np.log(gene_means + epsilon)
        log_vars = np.log(gene_vars + epsilon)

        print(f"Fitting LOESS model with span={loess_span}...")
        loess_model = loess(x=log_means, y=log_vars, span=loess_span, degree=2)
        loess_model.fit()
        fitted_vals = loess_model.outputs.fitted_values

        residuals = log_vars - fitted_vals

        print("Selecting highly variable genes...")
        if n_top_genes is not None:
            positive_mask = residuals > 0
            candidate_genes_idx = np.where(positive_mask)[0]
            if len(candidate_genes_idx) == 0:
                hvg_genes_idx = []
                print(f"No HVGs found for {ctype}.")
            else:
                candidate_sorted = candidate_genes_idx[np.argsort(residuals[candidate_genes_idx])[::-1]]
                hvg_genes_idx = candidate_sorted[:n_top_genes]
                print(f"Selected top {len(hvg_genes_idx)} HVGs for {ctype}.")
        else:
            hvg_genes_idx = np.where(residuals > 0)[0]
            print(f"Selected {len(hvg_genes_idx)} HVGs for {ctype} (using positive residuals).")

        print("Updating truncated expression matrix...")
        for sample_id in hvg_truncated_df.columns:
            orig_vector = hvg_truncated_df.loc[ctype, sample_id]
            if (orig_vector is not None) and (len(orig_vector) == n_genes):
                hvg_truncated_df.loc[ctype, sample_id] = orig_vector[hvg_genes_idx]

    print("\nHVG selection process completed.")
    return hvg_truncated_df
