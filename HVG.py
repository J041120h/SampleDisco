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
    num_features: int,
    batch_key: Optional[str] = None,
    check_values: bool = True,
    inplace: bool = True,
    span: float = 0.3,
) -> pd.DataFrame | None:
    """
    Identify highly variable genes (HVGs) across samples and optionally across batches in an AnnData object.

    ... [Docstring remains unchanged] ...
    """

    # Step 1: Prepare Data
    gene_expr = adata.to_df()

    # if check_values:
    #     if not np.all(gene_expr.values >= 0):
    #         raise ValueError("Gene expression data contains negative values.")
    #     if not np.all(np.floor(gene_expr.values) == gene_expr.values):
    #         raise ValueError("Gene expression data contains non-integer values.")

    # Step 2: Compute Means and Variances
    if batch_key is None:
        # Single batch scenario
        sample_means = gene_expr.groupby(adata.obs[sample_column]).mean()
        gene_mean = sample_means.mean(axis=0)
        gene_var = sample_means.var(axis=0)

        # Identify non-constant genes (variance > 0)
        not_const = gene_var > 0

        # Log-transform the means and variances to stabilize variance
        log_mean = np.log10(gene_mean[not_const] + 1e-8)  # Add small value to avoid log(0)
        log_var = np.log10(gene_var[not_const] + 1e-8)

        # Step 3: LOESS Smoothing
        try:
            loess_model = loess(x=log_mean.values, y=log_var.values, span=span, degree=2)
            loess_model.fit()
            fitted_log_var = loess_model.outputs.fitted_values
        except Exception as e:
            raise RuntimeError(f"LOESS fitting failed: {e}")

        # Initialize smoothed_var with zeros
        smoothed_var = np.zeros_like(gene_var)
        smoothed_var[not_const] = 10 ** fitted_log_var  # Convert back from log10 scale
        smoothed_var[~not_const] = gene_var[~not_const]

        # Step 4: Clipping
        residuals = gene_var - smoothed_var
        residual_std = np.std(residuals)
        clipped_residuals = np.clip(residuals, -2 * residual_std, 2 * residual_std)

        # Step 5: Calculate Normalized Variance
        normalized_var = clipped_residuals / residual_std

        # Step 6: Rank Genes
        top_hvg_genes = normalized_var.nlargest(num_features).index

        # Step 7: Update or Return Data
        if inplace:
            adata.var['gene_mean'] = gene_mean
            adata.var['gene_variance'] = gene_var
            adata.var['smoothed_variance'] = smoothed_var
            adata.var['normalized_variance'] = normalized_var
            adata.var['hvg_rank'] = normalized_var.rank(method='dense', ascending=False)
            adata.var['highly_variable'] = adata.var.index.isin(top_hvg_genes)
        else:
            hvgs_df = pd.DataFrame({
                'gene_mean': gene_mean,
                'gene_variance': gene_var,
                'smoothed_variance': smoothed_var,
                'normalized_variance': normalized_var,
                'hvg_rank': normalized_var.rank(method='dense', ascending=False),
                'highly_variable': normalized_var.index.isin(top_hvg_genes)
            })
            return hvgs_df

    else:
        # Multiple batches scenario
        if batch_key not in adata.obs:
            raise ValueError(f"Batch key '{batch_key}' not found in adata.obs.")

        batch_info = adata.obs[batch_key].astype(str)
        unique_batches = batch_info.unique()
        rank_lists = []  # To store ranks from each batch

        for batch in unique_batches:
            batch_mask = batch_info == batch
            data_batch = gene_expr[batch_mask]

            # Compute mean and variance within the batch
            sample_means = data_batch.groupby(adata.obs[sample_column]).mean()
            gene_mean_batch = sample_means.mean(axis=0)
            gene_var_batch = sample_means.var(axis=0)

            # Identify non-constant genes (variance > 0)
            not_const_batch = gene_var_batch > 0

            # Log-transform the means and variances
            log_mean_batch = np.log10(gene_mean_batch[not_const_batch] + 1e-8)
            log_var_batch = np.log10(gene_var_batch[not_const_batch] + 1e-8)

            # LOESS Smoothing
            try:
                # Attempt LOESS fitting with degree=2 (quadratic)
                loess_model_batch = loess(x=log_mean_batch.values, y=log_var_batch.values, span=span, degree=2)
                loess_model_batch.fit()
                fitted_log_var_batch = loess_model_batch.outputs.fitted_values
            except Exception as e:
                # Handle failure of degree=2 LOESS fitting
                print(f"LOESS fitting failed for batch '{batch}' with degree=2. Error: {e}")
                print("Data used for LOESS fitting:")
                print("log_mean_batch:", log_mean_batch.values)
                print("log_var_batch:", log_var_batch.values)
                
                # Retry LOESS fitting with degree=1 (linear)
                try:
                    loess_model_batch = loess(x=log_mean_batch.values, y=log_var_batch.values, span=span, degree=1)
                    loess_model_batch.fit()
                    fitted_log_var_batch = loess_model_batch.outputs.fitted_values
                except Exception as e1:
                    # Handle failure of degree=1 LOESS fitting
                    print(f"LOESS fitting failed for batch '{batch}' with degree=1 as well. Error: {e1}")
                    fitted_log_var_batch = np.zeros_like(log_var_batch.values)
                    # raise RuntimeError(f"LOESS fitting failed for batch '{batch}' with both degrees.")

            # Initialize smoothed_var for the batch
            smoothed_var_batch = np.zeros_like(gene_var_batch)
            smoothed_var_batch[not_const_batch] = 10 ** fitted_log_var_batch
            smoothed_var_batch[~not_const_batch] = gene_var_batch[~not_const_batch]

            # Clipping
            residuals_batch = gene_var_batch - smoothed_var_batch
            residual_std_batch = np.std(residuals_batch)
            clipped_residuals_batch = np.clip(residuals_batch, -2 * residual_std_batch, 2 * residual_std_batch)

            # Calculate Normalized Variance for the batch
            normalized_var_batch = clipped_residuals_batch / residual_std_batch

            # Rank genes within the batch
            ranks_batch = normalized_var_batch.rank(method='dense', ascending=False)
            rank_lists.append(ranks_batch)

        # Combine ranks from all batches
        rank_df = pd.concat(rank_lists, axis=1)

        # Compute median rank across batches for each gene
        median_rank = rank_df.median(axis=1)

        # Select top HVGs based on median rank
        top_hvg_genes = median_rank.nsmallest(num_features).index

        # Step 7: Update or Return Data
        if inplace:
            # Store aggregated metrics
            adata.var['gene_mean'] = gene_expr.mean(axis=0)
            adata.var['gene_variance'] = gene_expr.var(axis=0)
            adata.var['median_hvg_rank'] = median_rank
            adata.var['hvg_rank'] = median_rank.rank(method='dense', ascending=True)
            adata.var['highly_variable'] = adata.var.index.isin(top_hvg_genes)
            adata.var['highly_variable_nbatches'] = (rank_df <= np.percentile(rank_df.values, (num_features / gene_expr.shape[1]) * 100, axis=0)).sum(axis=1)
        else:
            hvgs_df = pd.DataFrame({
                'gene_mean': gene_expr.mean(axis=0),
                'gene_variance': gene_expr.var(axis=0),
                'median_hvg_rank': median_rank,
                'hvg_rank': median_rank.rank(method='dense', ascending=True),
                'highly_variable': median_rank.index.isin(top_hvg_genes)
            })
            hvgs_df['highly_variable_nbatches'] = (rank_df <= np.percentile(rank_df.values, (num_features / gene_expr.shape[1]) * 100, axis=0)).sum(axis=1)
            return hvgs_df

    return None  # When inplace=True, nothing is returned
