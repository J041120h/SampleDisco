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
    sample_col: str = 'sample',
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
    """
    if "cell_expression_corrected" not in pseudobulk:
        raise KeyError("Missing 'cell_expression_corrected' key in pseudobulk dictionary.")
    
    sample_df = pseudobulk["cell_expression_corrected"].copy()
    diff_groups = find_sample_grouping(adata, adata.obs[sample_col].unique(), grouping_columns, age_bin_size)
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient="index", columns=["plot_group"])
    
    if "plot_group" not in diff_groups.columns:
        raise KeyError("Column 'plot_group' is missing in diff_groups.")
    # Normalize formatting for consistent merge
    sample_df.index = sample_df.index.astype(str).str.strip().str.lower()
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
    # Reset index using sample_col
    sample_df = sample_df.reset_index().rename(columns={"index": sample_col})
    diff_groups = diff_groups.reset_index().rename(columns={"index": sample_col})
    sample_df = sample_df.merge(diff_groups, on=sample_col, how="left")
    if sample_df["plot_group"].isna().any():
        raise ValueError("Some samples could not be matched to grouping information in 'diff_groups'.")
    pca = PCA(n_components=n_components)
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    pca_coords = pca.fit_transform(sample_df[numeric_cols])
    pca_df = pd.DataFrame(
        data=pca_coords,
        index=sample_df[sample_col], 
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    adata.uns["X_pca_expression"] = pca_df
    if verbose:
        print(f"[run_pca_expression] PCA completed and stored in adata.uns['X_pca_expression'].")
        print(f"Shape of PCA DataFrame: {pca_df.shape}")

def run_pca_proportion(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    sample_col: str = 'sample',
    n_components: int = 10, 
    verbose: bool = False
) -> None:
    """
    Performs PCA on cell proportion data and stores the principal components in the AnnData object
    (adata.uns["X_pca_proportion"]). The resulting DataFrame will have each sample as the row index.
    """
    if "cell_proportion" not in pseudobulk:
        raise KeyError("Missing 'cell_proportion' key in pseudobulk dictionary.")
    proportion_df = pseudobulk["cell_proportion"].copy()
    # Normalize sample index formatting
    proportion_df.index = proportion_df.index.astype(str).str.strip().str.lower()
    proportion_df = proportion_df.fillna(0)
    
    # Check if there are enough dimensions for PCA
    n_samples, n_features = proportion_df.shape
    max_components = min(n_samples, n_features)
    if n_components > max_components:
        raise ValueError(
            f"Cannot perform PCA with n_components={n_components}. "
            f"Maximum possible components is {max_components} "
            f"(min of n_samples={n_samples} and n_features={n_features})."
        )
    
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(proportion_df)
    pca_df = pd.DataFrame(
        data=pca_coords,
        index=proportion_df.index, 
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    adata.uns["X_pca_proportion"] = pca_df
    if verbose:
        print(f"[run_pca_proportion] PCA on cell proportions completed.")
        print(f"Stored results in adata.uns['X_pca_proportion'] with shape: {pca_df.shape}")

def process_anndata_with_pca(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    sample_col: str = 'sample',
    grouping_columns: list = ["batch"], 
    age_bin_size: int = None,
    n_expression_pcs: int = 10, 
    n_proportion_pcs: int = 10, 
    output_dir: str = "./", 
    not_save: bool = False,
    atac: bool = False,
    verbose: bool = True
) -> None:
    """
    Computes PCA for both cell expression and cell proportion data and stores the results in an AnnData object.
    """
    start_time = time.time() if verbose else None
    if "cell_expression_corrected" not in pseudobulk or "cell_proportion" not in pseudobulk:
        raise KeyError("Missing required keys ('cell_expression_corrected' or 'cell_proportion') in pseudobulk.")
    if verbose:
        print("[process_anndata_with_pca] Starting PCA computation...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")
    output_dir = os.path.join(output_dir, "harmony")
    os.makedirs(output_dir, exist_ok=True)
    sample_expression_df = pseudobulk["cell_expression_corrected"]
    sample_proportion_df = pseudobulk["cell_proportion"]
    n_expression_pcs = min(n_expression_pcs, min(sample_expression_df.shape))
    n_proportion_pcs = min(n_proportion_pcs, min(sample_proportion_df.shape))
    if verbose:
        print(f"[process_anndata_with_pca] Using n_expression_pcs={n_expression_pcs}, n_proportion_pcs={n_proportion_pcs}")
    run_pca_expression(
        adata=adata, 
        pseudobulk=pseudobulk, 
        sample_col=sample_col,
        grouping_columns=grouping_columns,
        age_bin_size=age_bin_size,
        n_components=n_expression_pcs, 
        verbose=verbose
    )
    run_pca_proportion(
        adata=adata, 
        pseudobulk=pseudobulk, 
        sample_col=sample_col,
        n_components=n_proportion_pcs, 
        verbose=verbose
    )
    
    # Save data unless not_save is True
    if not not_save:
        if atac:
            adata_path = os.path.join(output_dir, 'atac_sample.h5ad')
        else:
            adata_path = os.path.join(output_dir, 'adata_sample.h5ad')
        sc.write(adata_path, adata)
        if verbose:
            print(f"[process_anndata_with_pca] AnnData with PCA results saved to: {adata_path}")
    
    if verbose and start_time is not None:
        elapsed_time = time.time() - start_time
        print(f"[process_anndata_with_pca] Total runtime for PCA processing: {elapsed_time:.2f} seconds")