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
    grouping_columns: list = None, 
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
    grouping_columns : list, optional
        List of columns to use for sample grouping. If None or empty, no grouping is performed.
    """
    if "cell_expression_corrected" not in pseudobulk:
        raise KeyError("Missing 'cell_expression_corrected' key in pseudobulk dictionary.")
    
    sample_df = pseudobulk["cell_expression_corrected"].copy()
    
    # Handle optional grouping
    plot_group_info = None
    if grouping_columns and len(grouping_columns) > 0:
        try:
            # Check if grouping columns exist in the data
            available_samples = adata.obs[sample_col].unique()
            
            # Validate that grouping columns exist in adata.obs
            missing_cols = [col for col in grouping_columns if col not in adata.obs.columns]
            if missing_cols:
                if verbose:
                    print(f"[run_pca_expression] Warning: Grouping columns {missing_cols} not found in adata.obs. Skipping grouping.")
            else:
                diff_groups = find_sample_grouping(adata, available_samples, grouping_columns, age_bin_size)
                if isinstance(diff_groups, dict):
                    diff_groups = pd.DataFrame.from_dict(diff_groups, orient="index", columns=["plot_group"])
                
                if "plot_group" in diff_groups.columns:
                    # Normalize formatting for consistent merge
                    sample_df.index = sample_df.index.astype(str).str.strip().str.lower()
                    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
                    
                    # Reset index using sample_col
                    sample_df_with_sample = sample_df.reset_index().rename(columns={"index": sample_col})
                    diff_groups_with_sample = diff_groups.reset_index().rename(columns={"index": sample_col})
                    
                    # Try to merge grouping information
                    merged_df = sample_df_with_sample.merge(diff_groups_with_sample, on=sample_col, how="left")
                    
                    # Check if merge was successful
                    if not merged_df["plot_group"].isna().all():
                        sample_df = merged_df
                        plot_group_info = True
                        if verbose:
                            n_matched = (~merged_df["plot_group"].isna()).sum()
                            print(f"[run_pca_expression] Successfully matched {n_matched}/{len(merged_df)} samples to grouping information.")
                    else:
                        if verbose:
                            print("[run_pca_expression] Warning: Could not match samples to grouping information. Proceeding without grouping.")
                else:
                    if verbose:
                        print("[run_pca_expression] Warning: 'plot_group' column missing in grouping results. Proceeding without grouping.")
        except Exception as e:
            if verbose:
                print(f"[run_pca_expression] Warning: Error during grouping ({str(e)}). Proceeding without grouping.")
    else:
        if verbose:
            print("[run_pca_expression] No grouping columns specified. Proceeding without grouping.")
    
    # Ensure sample_df has the correct format for PCA
    if sample_col not in sample_df.columns:
        sample_df = sample_df.reset_index().rename(columns={"index": sample_col})
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for PCA computation.")
    
    pca_coords = pca.fit_transform(sample_df[numeric_cols])
    pca_df = pd.DataFrame(
        data=pca_coords,
        index=sample_df[sample_col], 
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    
    # Store results
    adata.uns["X_pca_expression"] = pca_df
    
    # Store grouping information if available
    if plot_group_info and "plot_group" in sample_df.columns:
        grouping_df = sample_df[[sample_col, "plot_group"]].set_index(sample_col)
        adata.uns["X_pca_expression_groups"] = grouping_df
    
    if verbose:
        print(f"[run_pca_expression] PCA completed and stored in adata.uns['X_pca_expression'].")
        print(f"Shape of PCA DataFrame: {pca_df.shape}")
        if plot_group_info:
            print(f"Grouping information stored in adata.uns['X_pca_expression_groups'].")

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
        if verbose:
            print(f"[run_pca_proportion] Warning: Requested n_components={n_components} exceeds maximum possible ({max_components}). Using {max_components} components.")
        n_components = max_components
    
    if n_components <= 0:
        raise ValueError(f"Cannot perform PCA: insufficient data dimensions (samples={n_samples}, features={n_features}).")
    
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
    grouping_columns: list = None, 
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
    
    Parameters:
    -----------
    grouping_columns : list, optional
        List of columns to use for sample grouping (e.g., ["batch"]). If None, no grouping is performed.
        Default: None (changed from ["batch"] to make batch optional)
    """
    start_time = time.time() if verbose else None
    
    if "cell_expression_corrected" not in pseudobulk or "cell_proportion" not in pseudobulk:
        raise KeyError("Missing required keys ('cell_expression_corrected' or 'cell_proportion') in pseudobulk.")
    
    if verbose:
        print("[process_anndata_with_pca] Starting PCA computation...")
        if grouping_columns:
            print(f"Using grouping columns: {grouping_columns}")
        else:
            print("No grouping columns specified - running without sample grouping.")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")
    
    output_dir = os.path.join(output_dir, "harmony")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data dimensions and adjust n_components accordingly
    sample_expression_df = pseudobulk["cell_expression_corrected"]
    sample_proportion_df = pseudobulk["cell_proportion"]
    
    n_expression_pcs = min(n_expression_pcs, min(sample_expression_df.shape))
    n_proportion_pcs = min(n_proportion_pcs, min(sample_proportion_df.shape))
    
    if verbose:
        print(f"[process_anndata_with_pca] Using n_expression_pcs={n_expression_pcs}, n_proportion_pcs={n_proportion_pcs}")
    
    # Run PCA on expression data
    try:
        run_pca_expression(
            adata=adata, 
            pseudobulk=pseudobulk, 
            sample_col=sample_col,
            grouping_columns=grouping_columns,
            age_bin_size=age_bin_size,
            n_components=n_expression_pcs, 
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"[process_anndata_with_pca] Warning: Expression PCA failed ({str(e)}). Continuing with proportion PCA.")
    
    # Run PCA on proportion data
    try:
        run_pca_proportion(
            adata=adata, 
            pseudobulk=pseudobulk, 
            sample_col=sample_col,
            n_components=n_proportion_pcs, 
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"[process_anndata_with_pca] Warning: Proportion PCA failed ({str(e)}).")
    
    # Save data unless not_save is True
    if not not_save:
        if atac:
            adata_path = os.path.join(output_dir, 'ATAC_sample.h5ad')
        else:
            adata_path = os.path.join(output_dir, 'adata_sample.h5ad')
        
        try:
            sc.write(adata_path, adata)
            if verbose:
                print(f"[process_anndata_with_pca] AnnData with PCA results saved to: {adata_path}")
        except Exception as e:
            if verbose:
                print(f"[process_anndata_with_pca] Warning: Could not save file ({str(e)}).")
    
    if verbose and start_time is not None:
        elapsed_time = time.time() - start_time
        print(f"[process_anndata_with_pca] Total runtime for PCA processing: {elapsed_time:.2f} seconds")

# Utility function to check available grouping columns
def check_available_grouping_columns(adata: sc.AnnData, sample_col: str = 'sample') -> list:
    """
    Check which potential grouping columns are available in the AnnData object.
    
    Returns:
    --------
    list : Available columns that could be used for grouping
    """
    potential_columns = ['batch', 'condition', 'treatment', 'group', 'cohort', 'study']
    available_columns = []
    
    for col in potential_columns:
        if col in adata.obs.columns:
            # Check if column has meaningful variation
            unique_values = adata.obs[col].nunique()
            if unique_values > 1:
                available_columns.append(col)
    
    return available_columns