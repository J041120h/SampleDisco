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

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

def select_hvf_loess(pseudobulk, n_features=2000, frac=0.3, verbose=False):
    """
    Select highly variable features (HVFs) from pseudobulk data using LOESS.

    Parameters
    ----------
    pseudobulk : pd.DataFrame
        DataFrame indexed by cell type and with columns as sample names.
        Each cell contains a pd.Series with gene expression values (indexed by gene names).
    n_features : int, default 2000
        Number of top HVFs to select.
    frac : float, default 0.3
        Fraction parameter for LOESS smoothing.
    verbose : bool, default False
        If True, prints progress information.

    Returns
    -------
    sample_df : pd.DataFrame
        Sample-by-feature matrix (rows = samples, columns = selected features).
    top_features : pd.Index
        Index of the selected features.
    """
    start_time = time.time() if verbose else None

    # Step 1: Flatten into sample-by-feature matrix with gene names
    if verbose:
        print("Constructing sample-by-feature matrix from pseudobulk...")

    sample_dict = {}

    for sample in pseudobulk.columns:
        combined = []
        for cell_type in pseudobulk.index:
            s = pseudobulk.loc[cell_type, sample]
            if isinstance(s, pd.Series):
                combined.append(s)
        if len(combined) > 0:
            full_series = pd.concat(combined)
            sample_dict[sample] = full_series

    sample_df = pd.DataFrame(sample_dict).T
    sample_df.index.name = "sample"

    # Step 2: Compute per-feature mean and variance
    means = sample_df.mean(axis=0)
    variances = sample_df.var(axis=0)

    # Step 3: Fit LOESS of variance vs. mean
    loess_fit = lowess(variances, means, frac=frac)

    # Step 4: Predict LOESS-smoothed variance
    fitted_var = np.interp(means, loess_fit[:, 0], loess_fit[:, 1])

    # Step 5: Calculate residuals
    residuals = variances - fitted_var

    # Step 6: Select top features by residuals
    if sample_df.shape[1] > n_features:
        top_features = residuals.nlargest(n_features).index
        sample_df = sample_df[top_features]
    else:
        top_features = sample_df.columns

    if verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\n[select top features after concatenation] Total runtime: {elapsed_time:.2f} seconds\n\n")

    return sample_df, top_features


def highly_variable_gene_selection(
    cell_expression_corrected_df: pd.DataFrame,
    n_top_genes: int = None,
    loess_span: float = 0.3,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Identify Highly Variable Genes (HVGs) for each cell type independently using LOESS regression
    of log(variance) vs. log(mean). Truncate each expression Series to only include HVGs.
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

        # Stack all Series into a matrix and get gene names
        gene_names = sample_series[0].index
        expr_matrix = np.vstack([s.loc[gene_names].values for s in sample_series])
        n_samples, n_genes = expr_matrix.shape

        if n_samples < 2:
            if verbose:
                print(f"Skipping {ctype} due to insufficient samples (n_samples={n_samples}).")
            continue

        if verbose:
            print(f"Computing mean and variance for {n_genes} genes across {n_samples} samples...")

        gene_means = expr_matrix.mean(axis=0)
        gene_vars = expr_matrix.var(axis=0)

        epsilon = 1e-8
        log_means = np.log(gene_means + epsilon)
        log_vars = np.log(gene_vars + epsilon)

        if verbose:
            print(f"Fitting LOESS model with span={loess_span}...")

        loess_model = loess(x=log_means, y=log_vars, span=loess_span, degree=2)
        loess_model.fit()
        fitted_vals = loess_model.outputs.fitted_values

        residuals = log_vars - fitted_vals

        if verbose:
            print("Selecting highly variable genes...")

        if n_top_genes is not None:
            positive_mask = residuals > 0
            candidate_genes_idx = np.where(positive_mask)[0]
            if len(candidate_genes_idx) == 0:
                hvg_genes_idx = []
                if verbose:
                    print(f"No HVGs found for {ctype}.")
            else:
                candidate_sorted = candidate_genes_idx[np.argsort(residuals[candidate_genes_idx])[::-1]]
                hvg_genes_idx = candidate_sorted[:n_top_genes]
                if verbose:
                    print(f"Selected top {len(hvg_genes_idx)} HVGs for {ctype}.")
        else:
            hvg_genes_idx = np.where(residuals > 0)[0]
            if verbose:
                print(f"Selected {len(hvg_genes_idx)} HVGs for {ctype} (using positive residuals).")

        selected_genes = gene_names[hvg_genes_idx]

        if verbose:
            print("Updating truncated expression matrix...")

        for sample_id in hvg_truncated_df.columns:
            orig_series = hvg_truncated_df.loc[ctype, sample_id]
            if isinstance(orig_series, pd.Series):
                hvg_truncated_df.loc[ctype, sample_id] = orig_series.loc[selected_genes]

    if verbose:
        print("\nHVG selection process completed.")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\n[HVG for each cell type] Total execution time: {elapsed_time:.2f} seconds\n\n")

    return hvg_truncated_df