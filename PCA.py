import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import scanpy as sc
from Grouping import find_sample_grouping
from HVG import select_hvf_loess

def run_pca_expression(
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
    sample_df = pseudobulk["cell_expression_corrected"]

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
    adata.uns["X_pca_expression"] = pca_coords

    # Optionally print progress
    if verbose:
        print(f"PCA completed. Stored {n_components} components in `adata.obsm['X_pca']`.")

def run_pca_proportion(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    n_components: int = 10, 
    verbose: bool = False
) -> None:
    """
    Performs PCA on cell proportion data and stores the principal components in the AnnData object.
    
    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object to store PCA results.
    pseudobulk : dict
        A dictionary containing 'cell_proportion' with cell-type proportions per sample.
    n_components : int, default 10
        Number of principal components to compute.
    verbose : bool, default False
        If True, prints additional information.

    Returns:
    --------
    None
        The function modifies the `adata` object in place by adding PCA results to `adata.obsm['X_PCA_proportion']`.
    """

    # Ensure the proportion data is in pseudobulk
    if 'cell_proportion' not in pseudobulk:
        raise KeyError("Missing 'cell_proportion' key in pseudobulk dictionary.")

    # Extract the sample-by-cell-type proportion matrix
    proportion_df = pseudobulk["cell_proportion"]  # Transpose to get (samples x cell types)

    # Ensure no NaNs exist in the data
    proportion_df = proportion_df.fillna(0)

    # Perform PCA on cell proportion data
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(proportion_df)

    # Store PCA results in the AnnData object
    adata.uns["X_pca_proportion"] = pca_coords

    # Optionally print progress
    if verbose:
        print(f"PCA on cell proportions completed. Stored {n_components} components in `adata.obsm['X_PCA_proportion']`.")

def process_anndata_with_pca(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    n_expression_pcs: int = 10, 
    n_proportion_pcs: int = 10, 
    output_dir: str = "./", 
    adata_path: str = None,
    verbose: bool = True
) -> None:
    """
    Computes PCA for both cell expression and cell proportion data and stores the results in an AnnData object.
    
    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object that will be modified with PCA results.
    pseudobulk : dict
        A dictionary containing:
        - 'cell_expression_corrected': Batch-corrected expression data.
        - 'cell_proportion': Cell type proportions.
    n_expression_pcs : int, default 10
        Number of principal components to compute for cell expression.
    n_proportion_pcs : int, default 10
        Number of principal components to compute for cell proportion.
    output_dir : str, default "./"
        Directory to save PCA results if needed.
    verbose : bool, default False
        If True, prints additional progress messages.

    Returns:
    --------
    None
        Modifies `adata` in place by adding PCA results to:
        - `adata.obsm["X_pca"]` for cell expression PCA.
        - `adata.obsm["X_PCA_proportion"]` for cell proportion PCA.
    """

    # Ensure the necessary data is available
    if "cell_expression_corrected" not in pseudobulk or "cell_proportion" not in pseudobulk:
        raise KeyError("Missing required keys ('cell_expression_corrected' or 'cell_proportion') in pseudobulk dictionary.")

    if verbose:
        print("Starting PCA computation for cell expression...")

    # 0. Create output directories if not present
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating output directory")

    # Append 'harmony' subdirectory
    output_dir = os.path.join(output_dir, 'harmony')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating harmony subdirectory")

    # Retrieve the sample and feature dimensions
    sample_expression_df = pseudobulk["cell_expression_corrected"]
    sample_proportion_df = pseudobulk["cell_proportion"].T  # Transposed to (samples x cell types)

    # Adjust n_components dynamically based on available samples and features
    n_expression_pcs = min(n_expression_pcs, min(sample_expression_df.shape))
    n_proportion_pcs = min(n_proportion_pcs, min(sample_proportion_df.shape))

    if verbose:
        print(f"Adjusted n_expression_pcs: {n_expression_pcs}")
        print(f"Adjusted n_proportion_pcs: {n_proportion_pcs}")

    # Run PCA on cell expression and store in `adata.obsm["X_pca"]`
    run_pca_expression(
        adata=adata, 
        pseudobulk=pseudobulk, 
        n_components=n_expression_pcs, 
        verbose=verbose
    )

    if verbose:
        print("Cell expression PCA completed. Now processing cell proportions...")

    # Run PCA on cell proportions and store in `adata.obsm["X_PCA_proportion"]`
    run_pca_proportion(
        adata=adata, 
        pseudobulk=pseudobulk, 
        n_components=n_proportion_pcs, 
        verbose=verbose
    )

    if verbose:
        print("PCA on cell proportions completed.")

    sc.write(adata_path, adata)
    if verbose:
        print(f"Modified AnnData object saved to original position with PCA results.")