import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import snapatac2 as snap
from scipy.sparse import csr_matrix
from Grouping import find_sample_grouping

def run_spectral_expression(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    sample_col: str = 'sample',
    grouping_columns: list = ["batch"], 
    age_bin_size: int = None, 
    n_components: int = 10,
    verbose: bool = False
) -> None:
    """
    Performs spectral dimension reduction on pseudobulk-corrected expression data using snapATAC2
    and stores the results in the AnnData object (adata.uns["X_spectral_expression"]).
    """
    if "cell_expression_corrected" not in pseudobulk:
        raise KeyError("Missing 'cell_expression_corrected' key in pseudobulk dictionary.")
    
    sample_df = pseudobulk["cell_expression_corrected"].copy()
    
    # Get sample grouping information
    diff_groups = find_sample_grouping(adata, adata.obs[sample_col].unique(), grouping_columns, age_bin_size)
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient="index", columns=["plot_group"])
    
    if "plot_group" not in diff_groups.columns:
        raise KeyError("Column 'plot_group' is missing in diff_groups.")
    
    # Normalize formatting for consistent merge
    sample_df.index = sample_df.index.astype(str).str.strip().str.lower()
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
    
    # Create temporary AnnData object for snapATAC2 with sparse matrix
    # Convert to sparse matrix format
    sparse_matrix = csr_matrix(sample_df.values)
    sample_adata = sc.AnnData(X=sparse_matrix, obs=pd.DataFrame(index=sample_df.index))
    sample_adata.var_names = sample_df.columns
    
    # Add grouping information
    sample_adata.obs = sample_adata.obs.merge(
        diff_groups, left_index=True, right_index=True, how="left"
    )
    
    if verbose:
        print(f"[run_spectral_expression] Starting spectral dimension reduction...")
        print(f"[run_spectral_expression] Data shape: {sample_adata.shape}")
        print(f"[run_spectral_expression] Data type: {type(sample_adata.X)}")
    
    # Perform spectral embedding directly (data already preprocessed with TF-IDF)
    snap.tl.spectral(
        sample_adata,
        n_comps=n_components,
        features=None,  # Use all features
        random_state=0
    )
    
    # Extract the embedding and create DataFrame
    embedding = sample_adata.obsm['X_spectral']
    spectral_df = pd.DataFrame(
        data=embedding,
        index=sample_df.index,
        columns=[f"Spectral{i+1}" for i in range(n_components)]
    )
    
    # Store in original adata
    adata.uns["X_spectral_expression"] = spectral_df
    
    if verbose:
        print(f"[run_spectral_expression] Spectral embedding completed.")
        print(f"Stored in adata.uns['X_spectral_expression'] with shape: {spectral_df.shape}")


def run_spectral_proportion(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    sample_col: str = 'sample',
    n_components: int = 10,
    verbose: bool = False
) -> None:
    """
    Performs spectral dimension reduction on cell proportion data using snapATAC2
    and stores the results in the AnnData object (adata.uns["X_spectral_proportion"]).
    """
    if "cell_proportion" not in pseudobulk:
        raise KeyError("Missing 'cell_proportion' key in pseudobulk dictionary.")
    
    proportion_df = pseudobulk["cell_proportion"].copy()
    
    # Normalize sample index formatting
    proportion_df.index = proportion_df.index.astype(str).str.strip().str.lower()
    proportion_df = proportion_df.fillna(0)
    
    # Check dimensions
    n_samples, n_features = proportion_df.shape
    max_components = min(n_samples, n_features)
    if n_components > max_components:
        raise ValueError(
            f"Cannot perform dimension reduction with n_components={n_components}. "
            f"Maximum possible components is {max_components}."
        )
    
    # Create temporary AnnData object with sparse matrix
    sparse_matrix = csr_matrix(proportion_df.values)
    prop_adata = sc.AnnData(X=sparse_matrix, obs=pd.DataFrame(index=proportion_df.index))
    prop_adata.var_names = proportion_df.columns
    
    if verbose:
        print(f"[run_spectral_proportion] Starting spectral dimension reduction...")
        print(f"[run_spectral_proportion] Data shape: {prop_adata.shape}")
        print(f"[run_spectral_proportion] Data type: {type(prop_adata.X)}")
    
    # Perform spectral embedding
    snap.tl.spectral(
        prop_adata,
        n_comps=n_components,
        features=None,
        random_state=0
    )
    
    # Extract embedding and create DataFrame
    embedding = prop_adata.obsm['X_spectral']
    spectral_df = pd.DataFrame(
        data=embedding,
        index=proportion_df.index,
        columns=[f"Spectral{i+1}" for i in range(n_components)]
    )
    
    # Store in original adata
    adata.uns["X_spectral_proportion"] = spectral_df
    
    if verbose:
        print(f"[run_spectral_proportion] Spectral embedding completed.")
        print(f"Stored in adata.uns['X_spectral_proportion'] with shape: {spectral_df.shape}")


def process_anndata_with_spectral(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    sample_col: str = 'sample',
    grouping_columns: list = ["batch"], 
    age_bin_size: int = None,
    n_expression_components: int = 10, 
    n_proportion_components: int = 10,
    output_dir: str = "./", 
    not_save: bool = False,
    verbose: bool = True
) -> None:
    """
    Computes spectral dimension reduction for both cell expression and cell proportion data
    using snapATAC2 and stores the results in an AnnData object.
    """
    start_time = time.time() if verbose else None
    
    if "cell_expression_corrected" not in pseudobulk or "cell_proportion" not in pseudobulk:
        raise KeyError("Missing required keys ('cell_expression_corrected' or 'cell_proportion') in pseudobulk.")
    
    if verbose:
        print("[process_anndata_with_spectral] Starting snapATAC2 spectral dimension reduction...")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")
    
    output_dir = os.path.join(output_dir, "spectral")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data shapes
    sample_expression_df = pseudobulk["cell_expression_corrected"]
    sample_proportion_df = pseudobulk["cell_proportion"]
    
    # Adjust components if necessary
    n_expression_components = min(n_expression_components, min(sample_expression_df.shape))
    n_proportion_components = min(n_proportion_components, min(sample_proportion_df.shape))
    
    if verbose:
        print(f"[process_anndata_with_spectral] Using n_expression_components={n_expression_components}, n_proportion_components={n_proportion_components}")
    
    # Run spectral dimension reduction for expression data
    run_spectral_expression(
        adata=adata, 
        pseudobulk=pseudobulk, 
        sample_col=sample_col,
        grouping_columns=grouping_columns,
        age_bin_size=age_bin_size,
        n_components=n_expression_components,
        verbose=verbose
    )
    
    # Run spectral dimension reduction for proportion data
    run_spectral_proportion(
        adata=adata, 
        pseudobulk=pseudobulk, 
        sample_col=sample_col,
        n_components=n_proportion_components,
        verbose=verbose
    )
    
    # Save data unless not_save is True
    if not not_save:
        adata_path = os.path.join(output_dir, 'adata_spectral.h5ad')
        sc.write(adata_path, adata)
        if verbose:
            print(f"[process_anndata_with_spectral] AnnData with spectral results saved to: {adata_path}")
    
    if verbose and start_time is not None:
        elapsed_time = time.time() - start_time
        print(f"[process_anndata_with_spectral] Total runtime: {elapsed_time:.2f} seconds")