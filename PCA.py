import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import scanpy as sc
from Grouping import find_sample_grouping
from HVG import select_hvf_loess

def run_pca(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    grouping_columns: list = ['batch'], 
    age_bin_size: int = None, 
    n_components: int = 10, 
    n_features: int = 2000, 
    frac: float = 0.3, 
    verbose: bool = False
) -> None:
    """
    Performs PCA on pseudobulk-corrected expression data and stores the principal components in the AnnData object.
    
    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object containing sample metadata in adata.obs.
    pseudobulk : dict
        A dictionary containing 'cell_expression_corrected' with expression data.
    grouping_columns : list, default ['batch']
        List of columns in adata.obs to use for grouping.
    age_bin_size : int, optional
        Integer for age binning if required.
    n_components : int, default 10
        Number of principal components to compute.
    n_features : int, default 2000
        Number of top HVFs to select.
    frac : float, default 0.3
        Fraction parameter for LOESS smoothing in HVG selection.
    verbose : bool, default False
        If True, prints additional information.

    Returns:
    --------
    None
        The function modifies the `adata` object in place by adding PCA results to `adata.obsm['X_pca']`.
    """

    # Ensure the corrected data is in pseudobulk
    if 'cell_expression_corrected' not in pseudobulk:
        raise KeyError("Missing 'cell_expression_corrected' key in pseudobulk dictionary.")

    # Select highly variable genes and construct the sample-by-feature matrix
    sample_df, top_features = select_hvf_loess(pseudobulk, n_features=n_features, frac=frac)

    # Retrieve grouping info
    diff_groups = find_sample_grouping(adata, adata.obs['sample'].unique(), grouping_columns, age_bin_size)
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])
    if 'plot_group' not in diff_groups.columns:
        raise KeyError("Column 'plot_group' is missing in diff_groups.")

    # Format indices for merging
    sample_df.index = sample_df.index.astype(str).str.strip().str.lower()
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()

    # Convert sample index into a column for merging
    sample_df = sample_df.reset_index().rename(columns={'index': 'sample'})
    diff_groups = diff_groups.reset_index().rename(columns={'index': 'sample'})

    # Merge grouping info
    sample_df = sample_df.merge(diff_groups, on='sample', how='left')
    if sample_df['plot_group'].isna().any():
        raise ValueError("Some samples could not be matched to grouping information.")

    # Perform PCA on numeric columns only
    pca = PCA(n_components=n_components)
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    pca_coords = pca.fit_transform(sample_df[numeric_cols])

    # Store PCA results in the AnnData object
    adata.obsm["X_pca"] = pca_coords

    # Optionally print progress
    if verbose:
        print(f"PCA completed. Stored {n_components} components in `adata.obsm['X_pca']`.")
