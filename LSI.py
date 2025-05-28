import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse
from Grouping import find_sample_grouping
from HVG import select_hvf_loess

def run_lsi_expression(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    sample_col: str = 'sample',
    grouping_columns: list = None, 
    age_bin_size: int = None, 
    n_components: int = 10, 
    n_features: int = 2000, 
    frac: float = 0.3,
    scale_data: bool = True,
    verbose: bool = False
) -> None:
    """
    Performs LSI (Latent Semantic Indexing) on pseudobulk-corrected expression data and stores the 
    latent components in the AnnData object (adata.uns["X_lsi_expression"]). 
    The resulting DataFrame will have each sample as the row index.
    
    Parameters:
    -----------
    grouping_columns : list, optional
        List of columns to use for sample grouping. If None or empty, no grouping is performed.
    scale_data : bool, default=True
        Whether to scale the data before LSI. For sparse data, this might be skipped.
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
                    print(f"[run_lsi_expression] Warning: Grouping columns {missing_cols} not found in adata.obs. Skipping grouping.")
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
                            print(f"[run_lsi_expression] Successfully matched {n_matched}/{len(merged_df)} samples to grouping information.")
                    else:
                        if verbose:
                            print("[run_lsi_expression] Warning: Could not match samples to grouping information. Proceeding without grouping.")
                else:
                    if verbose:
                        print("[run_lsi_expression] Warning: 'plot_group' column missing in grouping results. Proceeding without grouping.")
        except Exception as e:
            if verbose:
                print(f"[run_lsi_expression] Warning: Error during grouping ({str(e)}). Proceeding without grouping.")
    else:
        if verbose:
            print("[run_lsi_expression] No grouping columns specified. Proceeding without grouping.")
    
    # Ensure sample_df has the correct format for LSI
    if sample_col not in sample_df.columns:
        sample_df = sample_df.reset_index().rename(columns={"index": sample_col})
    
    # Perform LSI
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for LSI computation.")
    
    # Prepare data for LSI
    X = sample_df[numeric_cols].values
    
    # Optional scaling (usually not done for LSI on count data)
    if scale_data and not issparse(X):
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X)
    
    # Apply TruncatedSVD for LSI
    lsi = TruncatedSVD(n_components=n_components, random_state=42)
    lsi_coords = lsi.fit_transform(X)
    
    # Create DataFrame with LSI components
    lsi_df = pd.DataFrame(
        data=lsi_coords,
        index=sample_df[sample_col], 
        columns=[f"LSI{i+1}" for i in range(n_components)]
    )
    
    # Store results
    adata.uns["X_lsi_expression"] = lsi_df
    adata.uns["lsi_expression_explained_variance_ratio"] = lsi.explained_variance_ratio_
    adata.uns["lsi_expression_singular_values"] = lsi.singular_values_
    
    # Store grouping information if available
    if plot_group_info and "plot_group" in sample_df.columns:
        grouping_df = sample_df[[sample_col, "plot_group"]].set_index(sample_col)
        adata.uns["X_lsi_expression_groups"] = grouping_df
    
    if verbose:
        print(f"[run_lsi_expression] LSI completed and stored in adata.uns['X_lsi_expression'].")
        print(f"Shape of LSI DataFrame: {lsi_df.shape}")
        print(f"Explained variance ratio: {lsi.explained_variance_ratio_[:5]}...")  # Show first 5
        if plot_group_info:
            print(f"Grouping information stored in adata.uns['X_lsi_expression_groups'].")

def run_lsi_proportion(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    sample_col: str = 'sample',
    n_components: int = 10,
    scale_data: bool = False,
    verbose: bool = False
) -> None:
    """
    Performs LSI on cell proportion data and stores the latent components in the AnnData object
    (adata.uns["X_lsi_proportion"]). The resulting DataFrame will have each sample as the row index.
    
    Parameters:
    -----------
    scale_data : bool, default=False
        Whether to scale the data before LSI. For proportion data, this is usually not needed.
    """
    if "cell_proportion" not in pseudobulk:
        raise KeyError("Missing 'cell_proportion' key in pseudobulk dictionary.")
    
    proportion_df = pseudobulk["cell_proportion"].copy()
    
    # Normalize sample index formatting
    proportion_df.index = proportion_df.index.astype(str).str.strip().str.lower()
    proportion_df = proportion_df.fillna(0)
    
    # Check if there are enough dimensions for LSI
    n_samples, n_features = proportion_df.shape
    max_components = min(n_samples, n_features) - 1  # TruncatedSVD needs n_components < min(n_samples, n_features)
    if n_components > max_components:
        if verbose:
            print(f"[run_lsi_proportion] Warning: Requested n_components={n_components} exceeds maximum possible ({max_components}). Using {max_components} components.")
        n_components = max_components
    
    if n_components <= 0:
        raise ValueError(f"Cannot perform LSI: insufficient data dimensions (samples={n_samples}, features={n_features}).")
    
    # Prepare data
    X = proportion_df.values
    
    # Optional scaling (usually not for proportion data)
    if scale_data:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X)
    
    # Apply TruncatedSVD for LSI
    lsi = TruncatedSVD(n_components=n_components, random_state=42)
    lsi_coords = lsi.fit_transform(X)
    
    lsi_df = pd.DataFrame(
        data=lsi_coords,
        index=proportion_df.index, 
        columns=[f"LSI{i+1}" for i in range(n_components)]
    )
    
    adata.uns["X_lsi_proportion"] = lsi_df
    adata.uns["lsi_proportion_explained_variance_ratio"] = lsi.explained_variance_ratio_
    adata.uns["lsi_proportion_singular_values"] = lsi.singular_values_
    
    if verbose:
        print(f"[run_lsi_proportion] LSI on cell proportions completed.")
        print(f"Stored results in adata.uns['X_lsi_proportion'] with shape: {lsi_df.shape}")
        print(f"Explained variance ratio: {lsi.explained_variance_ratio_[:5]}...")  # Show first 5

def process_anndata_with_lsi(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    sample_col: str = 'sample',
    grouping_columns: list = None, 
    age_bin_size: int = None,
    n_expression_components: int = 10, 
    n_proportion_components: int = 10,
    scale_expression: bool = True,
    scale_proportion: bool = False,
    output_dir: str = "./", 
    not_save: bool = False,
    atac: bool = False,
    verbose: bool = True
) -> None:
    """
    Computes LSI for both cell expression and cell proportion data and stores the results in an AnnData object.
    LSI is particularly useful for sparse data like scRNA-seq or scATAC-seq.
    
    Parameters:
    -----------
    grouping_columns : list, optional
        List of columns to use for sample grouping (e.g., ["batch"]). If None, no grouping is performed.
        Default: None (changed from ["batch"] to make batch optional)
    n_expression_components : int, default=10
        Number of LSI components for expression data
    n_proportion_components : int, default=10
        Number of LSI components for proportion data
    scale_expression : bool, default=True
        Whether to scale expression data before LSI
    scale_proportion : bool, default=False
        Whether to scale proportion data before LSI
    """
    start_time = time.time() if verbose else None
    
    if "cell_expression_corrected" not in pseudobulk or "cell_proportion" not in pseudobulk:
        raise KeyError("Missing required keys ('cell_expression_corrected' or 'cell_proportion') in pseudobulk.")
    
    if verbose:
        print("[process_anndata_with_lsi] Starting LSI computation...")
        if grouping_columns:
            print(f"Using grouping columns: {grouping_columns}")
        else:
            print("No grouping columns specified - running without sample grouping.")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")
    
    output_dir = os.path.join(output_dir, "lsi_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data dimensions and adjust n_components accordingly
    sample_expression_df = pseudobulk["cell_expression_corrected"]
    sample_proportion_df = pseudobulk["cell_proportion"]
    
    # TruncatedSVD requires n_components < min(n_samples, n_features)
    n_expression_components = min(n_expression_components, min(sample_expression_df.shape) - 1)
    n_proportion_components = min(n_proportion_components, min(sample_proportion_df.shape) - 1)
    
    if verbose:
        print(f"[process_anndata_with_lsi] Using n_expression_components={n_expression_components}, n_proportion_components={n_proportion_components}")
    
    # Run LSI on expression data
    try:
        run_lsi_expression(
            adata=adata, 
            pseudobulk=pseudobulk, 
            sample_col=sample_col,
            grouping_columns=grouping_columns,
            age_bin_size=age_bin_size,
            n_components=n_expression_components,
            scale_data=scale_expression,
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"[process_anndata_with_lsi] Warning: Expression LSI failed ({str(e)}). Continuing with proportion LSI.")
    
    # Run LSI on proportion data
    try:
        run_lsi_proportion(
            adata=adata, 
            pseudobulk=pseudobulk, 
            sample_col=sample_col,
            n_components=n_proportion_components,
            scale_data=scale_proportion,
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"[process_anndata_with_lsi] Warning: Proportion LSI failed ({str(e)}).")
    
    # Save data unless not_save is True
    if not not_save:
        if atac:
            adata_path = os.path.join(output_dir, 'ATAC_sample_lsi.h5ad')
        else:
            adata_path = os.path.join(output_dir, 'adata_sample_lsi.h5ad')
        
        try:
            sc.write(adata_path, adata)
            if verbose:
                print(f"[process_anndata_with_lsi] AnnData with LSI results saved to: {adata_path}")
        except Exception as e:
            if verbose:
                print(f"[process_anndata_with_lsi] Warning: Could not save file ({str(e)}).")
    
    if verbose and start_time is not None:
        elapsed_time = time.time() - start_time
        print(f"[process_anndata_with_lsi] Total runtime for LSI processing: {elapsed_time:.2f} seconds")

# Additional utility function for LSI-specific operations
def compute_lsi_with_tf_idf(
    adata: sc.AnnData,
    pseudobulk: dict,
    sample_col: str = 'sample',
    n_components: int = 10,
    use_tf_idf: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Compute LSI with optional TF-IDF transformation, which is common for text-like sparse data.
    This is particularly useful for ATAC-seq data where peaks can be treated like "words".
    
    Parameters:
    -----------
    use_tf_idf : bool, default=True
        Whether to apply TF-IDF transformation before LSI
    
    Returns:
    --------
    pd.DataFrame : LSI components
    """
    from sklearn.feature_extraction.text import TfidfTransformer
    
    if "cell_expression_corrected" not in pseudobulk:
        raise KeyError("Missing 'cell_expression_corrected' key in pseudobulk dictionary.")
    
    sample_df = pseudobulk["cell_expression_corrected"].copy()
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    X = sample_df[numeric_cols].values
    
    if use_tf_idf:
        # Apply TF-IDF transformation
        tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
        X_tfidf = tfidf.fit_transform(X)
        X = X_tfidf.toarray() if hasattr(X_tfidf, 'toarray') else X_tfidf
    
    # Apply LSI
    lsi = TruncatedSVD(n_components=n_components, random_state=42)
    lsi_coords = lsi.fit_transform(X)
    
    lsi_df = pd.DataFrame(
        data=lsi_coords,
        index=sample_df.index,
        columns=[f"LSI{i+1}" for i in range(n_components)]
    )
    
    if verbose:
        print(f"[compute_lsi_with_tf_idf] LSI with TF-IDF={'enabled' if use_tf_idf else 'disabled'} completed.")
        print(f"Shape: {lsi_df.shape}, Explained variance: {lsi.explained_variance_ratio_[:3]}...")
    
    return lsi_df

# Keep the original utility function
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