import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA

from Grouping import find_sample_grouping
from HVG import select_hvf_loess

def run_pca_expression(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    grouping_columns: list = ["batch"], 
    age_bin_size: int = None, 
    n_components: int = 10, 
    n_features: int = 2000, 
    frac: float = 0.3, 
    verbose: bool = False
) -> None:
    """
    Performs PCA on pseudobulk-corrected expression data and stores the principal components in the AnnData object 
    (adata.uns["X_pca_expression"]). The resulting DataFrame will have each sample as the row index.

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
        Number of top HVGs to select (not fully shown here, but you can integrate HVG selection as needed).
    frac : float, default 0.3
        Fraction parameter for LOESS smoothing in HVG selection (not fully shown, but included for clarity).
    verbose : bool, default False
        If True, prints additional information.

    Returns:
    --------
    None
        The function modifies `adata` in place by adding the PCA results to `adata.uns["X_pca_expression"]`,
        which will be a DataFrame indexed by sample name.
    """

    # ------------------------------------------------------------------------
    # 1. Prepare and check data
    # ------------------------------------------------------------------------
    # Ensure the corrected data is present
    if "cell_expression_corrected" not in pseudobulk:
        raise KeyError("Missing 'cell_expression_corrected' key in pseudobulk dictionary.")

    # (Optional) If you're doing HVG selection, do it here before further steps.
    # For example: genes_to_keep = select_hvf_loess(...), etc.

    # We'll proceed with all features for demonstration
    sample_df = pseudobulk["cell_expression_corrected"].copy()

    # ------------------------------------------------------------------------
    # 2. Retrieve grouping info
    # ------------------------------------------------------------------------
    diff_groups = find_sample_grouping(adata, adata.obs["sample"].unique(), grouping_columns, age_bin_size)
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient="index", columns=["plot_group"])
    if "plot_group" not in diff_groups.columns:
        raise KeyError("Column 'plot_group' is missing in diff_groups.")

    # Format indices
    sample_df.index = sample_df.index.astype(str).str.strip().str.lower()
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()

    # Put sample index into a column for merging
    sample_df = sample_df.reset_index().rename(columns={"index": "sample"})
    diff_groups = diff_groups.reset_index().rename(columns={"index": "sample"})

    # Merge grouping info
    sample_df = sample_df.merge(diff_groups, on="sample", how="left")
    if sample_df["plot_group"].isna().any():
        raise ValueError("Some samples could not be matched to grouping information in 'diff_groups'.")

    # ------------------------------------------------------------------------
    # 3. Run PCA on numeric columns
    # ------------------------------------------------------------------------
    pca = PCA(n_components=n_components)

    # Keep only the numeric columns for PCA
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    pca_coords = pca.fit_transform(sample_df[numeric_cols])  # shape: (num_samples, n_components)

    # Create a DataFrame with 'sample' as index and PC1..PCn as columns
    pca_df = pd.DataFrame(
        data=pca_coords,
        index=sample_df["sample"], 
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    # ------------------------------------------------------------------------
    # 4. Store PCA results in adata (with row index = sample name)
    # ------------------------------------------------------------------------
    # This ensures that each row in adata.uns["X_pca_expression"] corresponds 
    # exactly to a sample. 
    adata.uns["X_pca_expression"] = pca_df

    # Optionally print progress
    if verbose:
        print(f"[run_pca_expression] PCA completed and stored in adata.uns['X_pca_expression'].")
        print(f"Shape of PCA DataFrame: {pca_df.shape}")


def run_pca_proportion(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    n_components: int = 10, 
    verbose: bool = False
) -> None:
    """
    Performs PCA on cell proportion data and stores the principal components in the AnnData object
    (adata.uns["X_pca_proportion"]). The resulting DataFrame will have each sample as the row index.

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
        The function modifies `adata` in place by adding the PCA results to `adata.uns["X_pca_proportion"]`,
        which will be a DataFrame indexed by sample name.
    """

    if "cell_proportion" not in pseudobulk:
        raise KeyError("Missing 'cell_proportion' key in pseudobulk dictionary.")

    # Extract proportion data. 
    # If you originally had it transposed, confirm the dimension alignment 
    # (samples as rows, cell types as columns).
    proportion_df = pseudobulk["cell_proportion"].copy()

    # Ensure samples are string-lowered, etc., if needed
    proportion_df.index = proportion_df.index.astype(str).str.strip().str.lower()

    # Fill any NaNs
    proportion_df = proportion_df.fillna(0)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(proportion_df)

    # Create DataFrame with sample names as index
    pca_df = pd.DataFrame(
        data=pca_coords,
        index=proportion_df.index, 
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    # Store in adata
    adata.uns["X_pca_proportion"] = pca_df

    if verbose:
        print(f"[run_pca_proportion] PCA on cell proportions completed.")
        print(f"Stored results in adata.uns['X_pca_proportion'] with shape: {pca_df.shape}")


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
    adata_path : str, optional
        If provided, the modified AnnData will be saved to this path.
    verbose : bool, default True
        If True, prints additional progress messages.

    Returns:
    --------
    None
        Modifies `adata` in place by adding DataFrame PCA results to:
          - `adata.uns["X_pca_expression"]` (indexed by sample)
          - `adata.uns["X_pca_proportion"]` (indexed by sample)
    """

    start_time = time.time() if verbose else None

    if "cell_expression_corrected" not in pseudobulk or "cell_proportion" not in pseudobulk:
        raise KeyError("Missing required keys ('cell_expression_corrected' or 'cell_proportion') in pseudobulk.")

    if verbose:
        print("[process_anndata_with_pca] Starting PCA computation...")

    # Create output directories if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")

    # Make a "harmony" subdir for storing results if that’s part of your pipeline
    output_dir = os.path.join(output_dir, "harmony")
    os.makedirs(output_dir, exist_ok=True)

    # Adjust number of PCs if sample count or feature count is less than requested
    sample_expression_df = pseudobulk["cell_expression_corrected"]
    sample_proportion_df = pseudobulk["cell_proportion"]  # Already oriented as (sample x cell_type) ideally

    n_expression_pcs = min(n_expression_pcs, min(sample_expression_df.shape))
    n_proportion_pcs = min(n_proportion_pcs, min(sample_proportion_df.shape))

    if verbose:
        print(f"[process_anndata_with_pca] Using n_expression_pcs={n_expression_pcs}, n_proportion_pcs={n_proportion_pcs}")

    # ---------------------------------------
    # 1. Run PCA on expression
    # ---------------------------------------
    run_pca_expression(
        adata=adata, 
        pseudobulk=pseudobulk, 
        n_components=n_expression_pcs, 
        verbose=verbose
    )

    # ---------------------------------------
    # 2. Run PCA on proportions
    # ---------------------------------------
    run_pca_proportion(
        adata=adata, 
        pseudobulk=pseudobulk, 
        n_components=n_proportion_pcs, 
        verbose=verbose
    )

    # Optionally save AnnData
    if adata_path is not None:
        sc.write(adata_path, adata)
        if verbose:
            print(f"[process_anndata_with_pca] AnnData with PCA results saved to: {adata_path}")

    if verbose and start_time is not None:
        elapsed_time = time.time() - start_time
        print(f"[process_anndata_with_pca] Total runtime for PCA processing: {elapsed_time:.2f} seconds")
