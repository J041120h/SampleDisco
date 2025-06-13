import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from Grouping import find_sample_grouping
from HVG import select_hvf_loess
import scipy.sparse as sparse
from muon import atac as ac
import snapatac2 as snap

def run_lsi_expression(
    pseudobulk_anndata: sc.AnnData,
    n_components: int = 10,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Performs LSI (Latent Semantic Indexing) on pseudobulk expression data for ATAC-seq using scanpy.
    Parameters:
    -----------
    pseudobulk_anndata : sc.AnnData
        AnnData object with samples as observations and genes as variables (sample * gene)
    n_components : int, default 10
        Number of LSI components to compute
    verbose : bool, default False
        Whether to print verbose output
    Returns:
    --------
    pd.DataFrame
        LSI coordinates with samples as rows and LSI components as columns
    """
    if verbose:
        print(f"[LSI] Computing LSI with {n_components} components on {pseudobulk_anndata.shape} data")
    
    pb_adata = pseudobulk_anndata.copy()
    n_samples, n_genes = pb_adata.shape
    max_components = min(n_samples - 1, n_genes - 1)
    
    if n_components > max_components:
        n_components = max_components
    if n_components <= 0:
        raise ValueError(f"Cannot perform LSI: insufficient data dimensions (samples={n_samples}, genes={n_genes}).")
    
    try:
        if sparse.issparse(pb_adata.X):
            pb_adata.X = pb_adata.X.tocsr()
        
        ac.pp.tfidf(pb_adata, scale_factor=1e4)
        sc.tl.lsi(pb_adata, n_comps=n_components)
        lsi_coords = pb_adata.obsm['X_lsi']
        
        lsi_df = pd.DataFrame(
            data=lsi_coords,
            index=pb_adata.obs_names,
            columns=[f"LSI{i+1}" for i in range(lsi_coords.shape[1])]
        )
        
        if verbose:
            print(f"[LSI] Success. Shape: {lsi_df.shape}")
        return lsi_df
        
    except Exception as e:
        if verbose:
            print(f"[run_lsi_expression] Scanpy LSI failed: {str(e)}")
        
        # Method 2: Manual LSI implementation using TruncatedSVD with TF-IDF
        try:
            if verbose:
                print("[run_lsi_expression] Attempting manual LSI with TF-IDF")
            
            # Get the data matrix
            X = pb_adata.X
            if sparse.issparse(X):
                X = X.toarray()
            
            # Check for NaN values in genes (columns) and drop them
            if verbose:
                print(f"[run_lsi_expression] Original shape: {X.shape}")
            
            # Check for NaN values in each gene (column)
            nan_genes = np.isnan(X).any(axis=0)
            n_nan_genes = np.sum(nan_genes)
            
            if n_nan_genes > 0:
                if verbose:
                    print(f"[run_lsi_expression] Found {n_nan_genes} genes with NaN values, dropping them")
                
                # Keep only genes without NaN values
                X = X[:, ~nan_genes]
                
                # Update the AnnData object to reflect dropped genes
                pb_adata = pb_adata[:, ~nan_genes].copy()
                
                if verbose:
                    print(f"[run_lsi_expression] Shape after dropping NaN genes: {X.shape}")
                
                # Update max_components after potentially reducing gene count
                n_samples, n_genes = X.shape
                max_components = min(n_samples - 1, n_genes - 1)
                
                if n_components > max_components:
                    n_components = max_components
                    if verbose:
                        print(f"[run_lsi_expression] Reduced n_components to {n_components} due to data dimensions")
                
                if n_components <= 0:
                    if verbose:
                        print("[run_lsi_expression] Insufficient data dimensions after dropping NaN genes")
                    return None
            
            # Apply TF-IDF transformation (common for ATAC-seq LSI)
            tfidf = TfidfTransformer(norm='l2', use_idf=True, sublinear_tf=True)
            X_tfidf = tfidf.fit_transform(X)
            
            # Perform SVD (LSI is essentially SVD on term-document matrix)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            lsi_coords = svd.fit_transform(X_tfidf)
            
            lsi_df = pd.DataFrame(
                data=lsi_coords,
                index=pb_adata.obs_names,
                columns=[f"LSI{i+1}" for i in range(lsi_coords.shape[1])]
            )
            
            if verbose:
                print(f"[run_lsi_expression] Manual LSI computation successful. Shape: {lsi_df.shape}")
            
            return lsi_df
            
        except Exception as e2:
            if verbose:
                print(f"[run_lsi_expression] Manual LSI also failed: {str(e2)}")
            return None

def run_snapatac2_spectral(
    pseudobulk_anndata: sc.AnnData,
    n_components: int = 10,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Performs snapATAC2 spectral embedding on pseudobulk ATAC-seq data.
    
    Parameters:
    -----------
    pseudobulk_anndata : sc.AnnData
        AnnData object with samples as observations and genes as variables (sample * gene)
    n_components : int, default 10
        Number of spectral components to compute
    verbose : bool, default False
        Whether to print verbose output
        
    Returns:
    --------
    pd.DataFrame
        Spectral embedding coordinates with samples as rows and components as columns
    """
    if verbose:
        print(f"[snapATAC2] Computing spectral embedding with {n_components} components on {pseudobulk_anndata.shape} data")
    
    pb_adata = pseudobulk_anndata.copy()
    
    try:
        # Select features (keeping all since they're already filtered)
        n_features_to_select = min(50000, pb_adata.shape[1])
        snap.pp.select_features(pb_adata, n_features=n_features_to_select)
        
        # Compute spectral embedding
        snap.tl.spectral(pb_adata, n_comps=n_components)
        
        # Extract spectral coordinates
        spectral_coords = pb_adata.obsm['X_spectral']
        spectral_df = pd.DataFrame(
            data=spectral_coords,
            index=pb_adata.obs_names,
            columns=[f"Spectral{i+1}" for i in range(spectral_coords.shape[1])]
        )
        
        if verbose:
            print(f"[snapATAC2] Success. Shape: {spectral_df.shape}")
            
        return spectral_df
        
    except Exception as e:
        if verbose:
            print(f"[snapATAC2] Failed: {str(e)}")
        return None

def _store_results_in_both_objects(adata, pseudobulk_anndata, key, df_result, obsm_key=None, verbose=False):
    """
    Helper function to store results consistently in both adata and pseudobulk_anndata objects.
    
    Parameters:
    -----------
    adata : sc.AnnData
        Main AnnData object
    pseudobulk_anndata : sc.AnnData
        Pseudobulk AnnData object
    key : str
        Key for storing in .uns
    df_result : pd.DataFrame
        Results DataFrame to store
    obsm_key : str, optional
        Key for storing in .obsm (if provided)
    verbose : bool
        Whether to print verbose output
    """
    if df_result is not None:
        # Store in both .uns dictionaries
        adata.uns[key] = df_result
        pseudobulk_anndata.uns[key] = df_result
        
        # Store in .obsm if requested
        if obsm_key is not None:
            pseudobulk_anndata.obsm[obsm_key] = df_result.values
            
        if verbose:
            print(f"[Storage] Stored {key} in both adata and pseudobulk_anndata (shape: {df_result.shape})")
    else:
        if verbose:
            print(f"[Storage] Skipped storing {key} - result was None")

def run_pca_expression(
    adata: sc.AnnData, 
    pseudobulk_anndata: sc.AnnData,
    n_components: int = 10, 
    atac: bool = False,
    use_snapatac2_dimred: bool = False,
    verbose: bool = False
) -> None:
    """
    Performs dimension reduction on pseudobulk-corrected expression data.
    All results are stored under the unified key 'X_DR_expression'.
    
    For RNA-seq data (atac=False):
    - Performs PCA and stores results as X_DR_expression
    
    For ATAC-seq data (atac=True):
    - Performs LSI and stores results as X_DR_expression (default)
    - If use_snapatac2_dimred=True, performs snapATAC2 spectral embedding instead
    
    Additional method results (if computed) are stored with method-specific keys for reference:
    - X_pca_expression_method: PCA results
    - X_lsi_expression_method: LSI results  
    - X_spectral_expression_method: snapATAC2 spectral results
    
    Parameters:
    -----------
    adata : sc.AnnData
        Original AnnData object
    pseudobulk : dict
        Dictionary containing pseudobulk data (kept for backward compatibility)
    pseudobulk_anndata : sc.AnnData
        AnnData object with samples as observations and genes as variables (sample * gene)
    sample_col : str, default 'sample'
        Column name for sample identification
    n_components : int, default 10
        Number of components to compute
    n_features : int, default 2000
        Number of highly variable features to use (if feature selection is needed)
    atac : bool, default False
        If True, use ATAC-seq appropriate methods (LSI or spectral)
    use_snapatac2_dimred : bool, default False
        If True and atac=True, use snapATAC2 spectral embedding instead of LSI
    verbose : bool, default False
        Whether to print verbose output
    """
    
    if pseudobulk_anndata is None:
        raise ValueError("pseudobulk_anndata parameter is required.")
    
    if verbose:
        if atac:
            method = "snapATAC2 spectral" if use_snapatac2_dimred else "LSI"
            print(f"[DimRed] Computing {method} for ATAC data with {n_components} components on {pseudobulk_anndata.shape} data")
        else:
            print(f"[DimRed] Computing PCA for RNA data with {n_components} components on {pseudobulk_anndata.shape} data")
    
    pb_adata = pseudobulk_anndata.copy()
    
    n_samples, n_genes = pb_adata.shape
    max_components = min(n_samples - 1, n_genes)
    if n_components > max_components:
        n_components = max_components
    
    if n_components <= 0:
        raise ValueError(f"Cannot perform dimension reduction: insufficient data dimensions (samples={n_samples}, genes={n_genes}).")
    
    # For ATAC data
    if atac:
        if use_snapatac2_dimred:
            # Use snapATAC2 spectral embedding
            spectral_df = run_snapatac2_spectral(
                pseudobulk_anndata=pseudobulk_anndata,
                n_components=n_components,
                verbose=verbose
            )
            
            # Store as unified X_DR_expression
            _store_results_in_both_objects(
                adata, pseudobulk_anndata, 
                "X_DR_expression", spectral_df, 
                obsm_key="X_DR_expression", 
                verbose=verbose
            )
            
            # Also store with method-specific key for reference
            _store_results_in_both_objects(
                adata, pseudobulk_anndata, 
                "X_spectral_expression_method", spectral_df, 
                obsm_key="X_spectral_expression_method", 
                verbose=verbose
            )
        else:
            # Use LSI (default for ATAC)
            lsi_df = run_lsi_expression(
                pseudobulk_anndata=pseudobulk_anndata,
                n_components=n_components,
                verbose=verbose
            )
            
            # Store as unified X_DR_expression
            _store_results_in_both_objects(
                adata, pseudobulk_anndata, 
                "X_DR_expression", lsi_df, 
                obsm_key="X_DR_expression", 
                verbose=verbose
            )
            
            # Also store with method-specific key for reference
            _store_results_in_both_objects(
                adata, pseudobulk_anndata, 
                "X_lsi_expression_method", lsi_df, 
                obsm_key="X_lsi_expression_method", 
                verbose=verbose
            )
    
    # For RNA data (or always compute PCA for reference)
    else:
        try:
            if pb_adata.X.max() > 100:
                sc.pp.log1p(pb_adata)
            
            sc.tl.pca(pb_adata, n_comps=n_components, svd_solver='arpack')
            
            pca_coords = pb_adata.obsm['X_pca']
            pca_df = pd.DataFrame(
                data=pca_coords,
                index=pb_adata.obs_names,
                columns=[f"PC{i+1}" for i in range(pca_coords.shape[1])]
            )
            
            if verbose:
                print(f"[PCA] Success. Shape: {pca_df.shape}")
            
            # Store as unified X_DR_expression
            _store_results_in_both_objects(
                adata, pseudobulk_anndata, 
                "X_DR_expression", pca_df, 
                obsm_key="X_DR_expression", 
                verbose=verbose
            )
            
            # Also store with method-specific key for reference
            _store_results_in_both_objects(
                adata, pseudobulk_anndata, 
                "X_pca_expression_method", pca_df, 
                obsm_key="X_pca_expression_method", 
                verbose=verbose
            )
            
            # Store variance information
            if 'pca' in pb_adata.uns:
                adata.uns["X_DR_expression_variance"] = pb_adata.uns['pca']['variance']
                adata.uns["X_DR_expression_variance_ratio"] = pb_adata.uns['pca']['variance_ratio']
                pseudobulk_anndata.uns["X_DR_expression_variance"] = pb_adata.uns['pca']['variance']
                pseudobulk_anndata.uns["X_DR_expression_variance_ratio"] = pb_adata.uns['pca']['variance_ratio']
                
                if verbose:
                    print(f"[PCA] Stored variance information in both objects")
            
        except Exception as e:
            raise RuntimeError(f"PCA computation failed: {str(e)}")
    
    if verbose:
        print(f"[DimRed] Completed - results stored as X_DR_expression in both adata and pseudobulk_anndata")


def run_pca_proportion(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    pseudobulk_anndata: sc.AnnData = None,
    sample_col: str = 'sample',
    n_components: int = 10, 
    verbose: bool = False
) -> None:
    """
    Performs PCA on cell proportion data and stores the principal components under the unified key
    'X_DR_proportion' in both AnnData objects.
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
    
    # Store in main adata object with unified key
    adata.uns["X_DR_proportion"] = pca_df
    
    # Store in pseudobulk_anndata if provided
    if pseudobulk_anndata is not None:
        pseudobulk_anndata.uns["X_DR_proportion"] = pca_df
        # Also store in obsm for convenience
        pseudobulk_anndata.obsm["X_DR_proportion"] = pca_df.values
        
        if verbose:
            print(f"[run_pca_proportion] PCA on cell proportions completed.")
            print(f"Stored results as X_DR_proportion in both adata and pseudobulk_anndata with shape: {pca_df.shape}")
    else:
        if verbose:
            print(f"[run_pca_proportion] PCA on cell proportions completed.")
            print(f"Stored results in adata.uns['X_DR_proportion'] with shape: {pca_df.shape}")


def process_anndata_with_pca(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    pseudobulk_anndata: sc.AnnData,
    sample_col: str = 'sample',
    n_expression_pcs: int = 10, 
    n_proportion_pcs: int = 10, 
    output_dir: str = "./", 
    integrated_data: bool = False,
    not_save: bool = False,
    atac: bool = False,
    use_snapatac2_dimred: bool = False,
    verbose: bool = True
) -> None:
    """
    Computes dimension reduction for both cell expression and cell proportion data.
    All results are stored under unified keys:
    - X_DR_expression: dimension reduction results for expression data
    - X_DR_proportion: dimension reduction results for proportion data
    
    The specific method used depends on the data type:
    - RNA-seq (atac=False): PCA
    - ATAC-seq (atac=True): LSI (default) or snapATAC2 spectral (if use_snapatac2_dimred=True)
    
    Parameters:
    -----------
    adata : sc.AnnData
        Original AnnData object
    pseudobulk : dict
        Dictionary containing pseudobulk data
    pseudobulk_anndata : sc.AnnData
        AnnData object with samples as observations and genes as variables (sample * gene)
    sample_col : str, default 'sample'
        Column name for sample identification
    n_expression_pcs : int, default 10
        Number of components for expression dimension reduction
    n_proportion_pcs : int, default 10
        Number of components for proportion dimension reduction
    output_dir : str, default "./"
        Output directory for saving results
    not_save : bool, default False
        If True, skip saving files
    atac : bool, default False
        If True, use ATAC-seq appropriate methods
    use_snapatac2_dimred : bool, default False
        If True and atac=True, use snapATAC2 spectral embedding instead of LSI
    verbose : bool, default True
        Whether to print verbose output
    """
    start_time = time.time() if verbose else None
    
    if "cell_expression_corrected" not in pseudobulk or "cell_proportion" not in pseudobulk:
        raise KeyError("Missing required keys ('cell_expression_corrected' or 'cell_proportion') in pseudobulk.")
    
    if pseudobulk_anndata is None:
        raise ValueError("pseudobulk_anndata parameter is required.")
    
    if verbose:
        print("[process_anndata_with_pca] Starting dimension reduction computation...")
        print("[process_anndata_with_pca] Results will be stored under unified keys:")
        print("  - X_DR_expression: for expression data")
        print("  - X_DR_proportion: for proportion data")
        if atac:
            method = "snapATAC2 spectral" if use_snapatac2_dimred else "LSI"
            print(f"[process_anndata_with_pca] ATAC mode: Will use {method} for expression data")
        else:
            print("[process_anndata_with_pca] RNA mode: Will use PCA for expression data")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")
    
    if not integrated_data:
        count_output_dir = os.path.join(output_dir, "harmony")
        pseudobulk_output_dir = os.path.join(output_dir, "pseudobulk")
        os.makedirs(output_dir, exist_ok=True)
    else:
        count_output_dir = os.path.join(output_dir, "preprocess")
        pseudobulk_output_dir = os.path.join(output_dir, "pseudobulk")
        os.makedirs(output_dir, exist_ok=True)
    
    # Get data dimensions and adjust n_components accordingly
    sample_proportion_df = pseudobulk["cell_proportion"]
    
    n_expression_pcs = min(n_expression_pcs, min(pseudobulk_anndata.shape) - 1)
    n_proportion_pcs = min(n_proportion_pcs, min(sample_proportion_df.shape))
    
    if verbose:
        print(f"[process_anndata_with_pca] Using n_expression_pcs={n_expression_pcs}, n_proportion_pcs={n_proportion_pcs}")
    
    # Run dimension reduction on expression data
    try:
        run_pca_expression(
            adata=adata, 
            pseudobulk=pseudobulk, 
            pseudobulk_anndata=pseudobulk_anndata,
            sample_col=sample_col,
            n_components=n_expression_pcs,
            atac=atac,
            use_snapatac2_dimred=use_snapatac2_dimred,
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"[process_anndata_with_pca] Warning: Expression dimension reduction failed ({str(e)}). Continuing with proportion dimension reduction.")
    
    # Run dimension reduction on proportion data
    try:
        run_pca_proportion(
            adata=adata, 
            pseudobulk=pseudobulk, 
            pseudobulk_anndata=pseudobulk_anndata,
            sample_col=sample_col,
            n_components=n_proportion_pcs, 
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"[process_anndata_with_pca] Warning: Proportion dimension reduction failed ({str(e)}).")
    
    # Save data unless not_save is True
    if not not_save:
        if atac:
            adata_path = os.path.join(count_output_dir, 'ATAC_sample.h5ad')
            pb_adata_path = os.path.join(pseudobulk_output_dir, 'pseudobulk_adata.h5ad')
        elif integrated_data:
            adata_path = os.path.join(count_output_dir, 'atac_rna_integrated.h5ad')
            pb_adata_path = os.path.join(pseudobulk_output_dir, 'pseudobulk_sample.h5ad')
        else:
            adata_path = os.path.join(count_output_dir, 'adata_sample.h5ad')
            pb_adata_path = os.path.join(pseudobulk_output_dir, 'pseudobulk_sample.h5ad')
        
        try:
            sc.write(adata_path, adata)
            sc.write(pb_adata_path, pseudobulk_anndata)
            if verbose:
                print(f"[process_anndata_with_pca] AnnData with dimension reduction results saved to: {adata_path}")
                print(f"[process_anndata_with_pca] Pseudobulk AnnData with dimension reduction results saved to: {pb_adata_path}")
        except Exception as e:
            if verbose:
                print(f"[process_anndata_with_pca] Warning: Could not save file ({str(e)}).")
    
    if verbose and start_time is not None:
        elapsed_time = time.time() - start_time
        print(f"[process_anndata_with_pca] Total runtime for dimension reduction: {elapsed_time:.2f} seconds")
        print(f"[process_anndata_with_pca] All results are now available under unified keys:")
        print(f"  - X_DR_expression and X_DR_proportion in both adata.uns and pseudobulk_anndata.uns")

    return pseudobulk_anndata