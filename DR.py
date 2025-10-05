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
        sc.tl.lsi(pb_adata, n_comps=n_components, random_state=42)
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
            
            # CRITICAL FIX: Check for NaN/inf values AFTER TF-IDF transformation
            if sparse.issparse(X_tfidf):
                # For sparse matrices, check the data array
                nan_mask = np.isnan(X_tfidf.data) | np.isinf(X_tfidf.data)
                if np.any(nan_mask):
                    if verbose:
                        print(f"[run_lsi_expression] Found {np.sum(nan_mask)} NaN/inf values in TF-IDF data, replacing with 0")
                    X_tfidf.data[nan_mask] = 0.0
                    X_tfidf.eliminate_zeros()  # Remove explicit zeros
                
                # Convert to dense for final check and SVD
                X_tfidf_dense = X_tfidf.toarray()
            else:
                X_tfidf_dense = X_tfidf
            
            # Final check for NaN/inf values in dense matrix
            nan_mask = np.isnan(X_tfidf_dense) | np.isinf(X_tfidf_dense)
            if np.any(nan_mask):
                if verbose:
                    print(f"[run_lsi_expression] Found {np.sum(nan_mask)} NaN/inf values after TF-IDF, replacing with 0")
                X_tfidf_dense[nan_mask] = 0.0
            
            # Additional safety check: if any sample has all zeros, this can cause issues
            sample_sums = np.sum(X_tfidf_dense, axis=1)
            zero_samples = sample_sums == 0
            if np.any(zero_samples):
                if verbose:
                    print(f"[run_lsi_expression] Found {np.sum(zero_samples)} samples with all-zero TF-IDF values")
                    print("[run_lsi_expression] Adding small noise to prevent SVD issues")
                # Add very small random noise to zero samples
                noise_scale = 1e-10
                for i in np.where(zero_samples)[0]:
                    X_tfidf_dense[i, :] = np.random.normal(0, noise_scale, X_tfidf_dense.shape[1])
            
            if verbose:
                print(f"[run_lsi_expression] TF-IDF matrix stats: shape={X_tfidf_dense.shape}, "
                      f"min={X_tfidf_dense.min():.6f}, max={X_tfidf_dense.max():.6f}, "
                      f"mean={X_tfidf_dense.mean():.6f}")
            
            # Perform SVD (LSI is essentially SVD on term-document matrix)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            lsi_coords = svd.fit_transform(X_tfidf_dense)
            
            # Final check for NaN in results
            if np.any(np.isnan(lsi_coords)):
                if verbose:
                    print("[run_lsi_expression] NaN values found in LSI results, attempting cleanup")
                lsi_coords = np.nan_to_num(lsi_coords, nan=0.0, posinf=0.0, neginf=0.0)
            
            lsi_df = pd.DataFrame(
                data=lsi_coords,
                index=pb_adata.obs_names,
                columns=[f"LSI{i+1}" for i in range(lsi_coords.shape[1])]
            )
            
            if verbose:
                print(f"[run_lsi_expression] Manual LSI computation successful. Shape: {lsi_df.shape}")
                print(f"[run_lsi_expression] LSI result stats: min={lsi_coords.min():.6f}, max={lsi_coords.max():.6f}")
            
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
        # Fix data type issues - ensure data is in correct format for snapATAC2
        if sparse.issparse(pb_adata.X):
            # Convert to CSR format and ensure correct dtype
            pb_adata.X = pb_adata.X.tocsr().astype(np.float32)
        else:
            # Ensure dense matrix has correct dtype
            pb_adata.X = pb_adata.X.astype(np.float32)
        
        # Handle any infinite or NaN values
        if sparse.issparse(pb_adata.X):
            pb_adata.X.data = np.nan_to_num(pb_adata.X.data, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            pb_adata.X = np.nan_to_num(pb_adata.X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if verbose:
            print(f"[snapATAC2] Data preprocessing complete. Matrix type: {type(pb_adata.X)}, dtype: {pb_adata.X.dtype}")
        
        # Select features (keeping all since they're already filtered)
        n_features_to_select = min(50000, pb_adata.shape[1])
        snap.pp.select_features(pb_adata, n_features=n_features_to_select)
        
        # Compute spectral embedding
        snap.tl.spectral(pb_adata, n_comps=n_components, random_state = 42)
        
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

def run_dimension_reduction_expression(
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
    - Tries snapATAC2 spectral embedding if use_snapatac2_dimred=True, falls back to LSI on failure
    - Performs LSI if use_snapatac2_dimred=False (default)
    
    Additional method results (if computed) are stored with method-specific keys for reference:
    - X_pca_expression_method: PCA results
    - X_lsi_expression_method: LSI results  
    - X_spectral_expression_method: snapATAC2 spectral results
    
    Parameters:
    -----------
    adata : sc.AnnData
        Original AnnData object
    pseudobulk_anndata : sc.AnnData
        AnnData object with samples as observations and genes as variables (sample * gene)
    n_components : int, default 10
        Number of components to compute
    atac : bool, default False
        If True, use ATAC-seq appropriate methods (LSI or spectral)
    use_snapatac2_dimred : bool, default False
        If True and atac=True, try snapATAC2 spectral embedding first, fallback to LSI if it fails
    verbose : bool, default False
        Whether to print verbose output
    """
    
    if pseudobulk_anndata is None:
        raise ValueError("pseudobulk_anndata parameter is required.")
    
    if verbose:
        if atac:
            if use_snapatac2_dimred:
                print(f"[DimRed] Computing snapATAC2 spectral (with LSI fallback) for ATAC data with {n_components} components on {pseudobulk_anndata.shape} data")
            else:
                print(f"[DimRed] Computing LSI for ATAC data with {n_components} components on {pseudobulk_anndata.shape} data")
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
        primary_result = None
        method_used = None
        
        if use_snapatac2_dimred:
            # Try snapATAC2 spectral embedding first
            spectral_df = run_snapatac2_spectral(
                pseudobulk_anndata=pseudobulk_anndata,
                n_components=n_components,
                verbose=verbose
            )
            
            if spectral_df is not None:
                primary_result = spectral_df
                method_used = "snapATAC2_spectral"
                
                # Store with method-specific key for reference
                _store_results_in_both_objects(
                    adata, pseudobulk_anndata, 
                    "X_spectral_expression_method", spectral_df, 
                    obsm_key="X_spectral_expression_method", 
                    verbose=verbose
                )
            else:
                if verbose:
                    print("[DimRed] snapATAC2 spectral failed, falling back to LSI...")
        
        # If snapATAC2 failed or wasn't requested, use LSI
        if primary_result is None:
            lsi_df = run_lsi_expression(
                pseudobulk_anndata=pseudobulk_anndata,
                n_components=n_components,
                verbose=verbose
            )
            
            if lsi_df is not None:
                primary_result = lsi_df
                method_used = "LSI"
                
                # Store with method-specific key for reference
                _store_results_in_both_objects(
                    adata, pseudobulk_anndata, 
                    "X_lsi_expression_method", lsi_df, 
                    obsm_key="X_lsi_expression_method", 
                    verbose=verbose
                )
            else:
                raise RuntimeError("Both snapATAC2 spectral and LSI methods failed for ATAC data")
        
        # Store the successful result as unified X_DR_expression
        if primary_result is not None:
            _store_results_in_both_objects(
                adata, pseudobulk_anndata, 
                "X_DR_expression", primary_result, 
                obsm_key="X_DR_expression", 
                verbose=verbose
            )
            
            if verbose:
                print(f"[DimRed] Successfully used {method_used} for ATAC dimension reduction")
    
    # For RNA data (or always compute PCA for reference)
    else:
        try:
            if pb_adata.X.max() > 100:
                sc.pp.log1p(pb_adata)
            
            sc.tl.pca(pb_adata, n_comps=n_components, svd_solver='arpack', random_state = 42)
            
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


def run_dimension_reduction_proportion(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    pseudobulk_anndata: sc.AnnData = None,
    sample_col: str = 'sample',
    batch_col: str = None,
    harmony_for_proportion: bool = False,
    n_components: int = 10, 
    verbose: bool = False
) -> None:
    """
    Performs dimension reduction on cell proportion data with small dataset handling.
    """
    if "cell_proportion" not in pseudobulk:
        raise KeyError("Missing 'cell_proportion' key in pseudobulk dictionary.")
    
    proportion_df = pseudobulk["cell_proportion"].copy()
    proportion_df.index = proportion_df.index.astype(str).str.strip().str.lower()
    proportion_df = proportion_df.fillna(0)
    
    # Validate components
    n_samples, n_features = proportion_df.shape
    max_components = min(n_samples - 1, n_features)  # n_samples - 1 for PCA
    
    if n_components > max_components:
        if verbose:
            print(f"[run_dimension_reduction_proportion] Adjusting n_components from {n_components} to {max_components}")
        n_components = max_components
    
    if n_components <= 0:
        raise ValueError(f"Insufficient data dimensions (samples={n_samples}, features={n_features})")
    
    # Check if dataset is too small for Harmony
    MIN_SAMPLES_FOR_HARMONY = 10
    if harmony_for_proportion and n_samples < MIN_SAMPLES_FOR_HARMONY:
        if verbose:
            print(f"[run_dimension_reduction_proportion] Warning: Only {n_samples} samples. "
                  f"Skipping Harmony (requires >={MIN_SAMPLES_FOR_HARMONY} samples)")
        harmony_for_proportion = False
    
    # Step 1: PCA
    if verbose:
        print(f"[run_dimension_reduction_proportion] Computing PCA with {n_components} components...")
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state =42)
    pca_coords = pca.fit_transform(proportion_df)
    
    if pseudobulk_anndata is not None:
        pseudobulk_anndata.obsm['X_pca_proportion'] = pca_coords
        pseudobulk_anndata.uns['pca_proportion_variance_ratio'] = pca.explained_variance_ratio_
    
    # Step 2: Harmony (if applicable)
    final_coords = pca_coords
    
    if harmony_for_proportion and batch_col and pseudobulk_anndata is not None:
        if batch_col not in pseudobulk_anndata.obs.columns:
            if verbose:
                print(f"[run_dimension_reduction_proportion] Warning: batch column '{batch_col}' not found")
        else:
            try:
                import harmonypy as hm
                
                if verbose:
                    print(f"[run_dimension_reduction_proportion] Applying Harmony with batch: {batch_col}")
                
                # Check unique batches
                n_batches = pseudobulk_anndata.obs[batch_col].nunique()
                if n_batches < 2:
                    if verbose:
                        print(f"[run_dimension_reduction_proportion] Skipping Harmony: only {n_batches} batch(es)")
                else:
                    harmony_out = hm.run_harmony(
                        pca_coords.T,
                        pseudobulk_anndata.obs,
                        batch_col,
                        max_iter_harmony=30,
                        nclust=max(2, min(n_batches, n_samples // 2))  # Ensure valid cluster number
                    )
                    final_coords = harmony_out.Z_corr.T
                    
                    if verbose:
                        print(f"[run_dimension_reduction_proportion] Harmony completed. Shape: {final_coords.shape}")
                        
            except Exception as e:
                if verbose:
                    print(f"[run_dimension_reduction_proportion] Harmony failed: {e}. Using PCA only.")
                final_coords = pca_coords
    
    # Step 3: Create DataFrame
    import pandas as pd
    final_df = pd.DataFrame(
        data=final_coords,
        index=proportion_df.index,
        columns=[f"PC{i+1}" for i in range(final_coords.shape[1])]
    )
    
    # Step 4: Store results
    adata.uns["X_DR_proportion"] = final_df
    adata.uns["X_DR_proportion_variance_ratio"] = pca.explained_variance_ratio_
    
    if pseudobulk_anndata is not None:
        pseudobulk_anndata.uns["X_DR_proportion"] = final_df
        pseudobulk_anndata.obsm["X_DR_proportion"] = final_coords
        pseudobulk_anndata.uns["X_DR_proportion_variance_ratio"] = pca.explained_variance_ratio_
    
    if verbose:
        method = "Harmony-integrated PCA" if (final_coords is not pca_coords) else "PCA"
        print(f"[run_dimension_reduction_proportion] Completed using {method}. Shape: {final_df.shape}")

def _save_anndata_with_detailed_error_handling(file_path, adata, object_name, verbose=False):
    """
    Save an AnnData object with detailed error handling and reporting.
    
    Parameters:
    -----------
    file_path : str
        Full path where to save the file
    adata : sc.AnnData
        AnnData object to save
    object_name : str
        Name of the object for error reporting
    verbose : bool
        Whether to print detailed information
        
    Returns:
    --------
    bool
        True if save was successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            if verbose:
                print(f"[Save] Created directory: {directory}")
        
        # Attempt to save
        if verbose:
            print(f"[Save] Saving {object_name} to: {file_path}")
        
        sc.write(file_path, adata)
        
        # Verify the file was created and has reasonable size
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > 0:
                if verbose:
                    print(f"[Save] ✓ Successfully saved {object_name} ({file_size / (1024*1024):.1f} MB)")
                return True
            else:
                if verbose:
                    print(f"[Save] ✗ File was created but has zero size: {file_path}")
                return False
        else:
            if verbose:
                print(f"[Save] ✗ File was not created: {file_path}")
            return False
        
    except Exception as e:
        if verbose:
            print(f"[Save] ✗ Unexpected error saving {object_name}: {str(e)}")
        return False


def dimension_reduction(
    adata: sc.AnnData, 
    pseudobulk: dict, 
    pseudobulk_anndata: sc.AnnData,
    sample_col: str = 'sample',
    n_expression_components: int = 10, 
    n_proportion_components: int = 10, 
    batch_col: str = None,
    harmony_for_proportion: bool = False,
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
    - ATAC-seq (atac=True): snapATAC2 spectral (with LSI fallback) or LSI (default)
    
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
    n_expression_components : int, default 10
        Number of components for expression dimension reduction
    n_proportion_components : int, default 10
        Number of components for proportion dimension reduction
    output_dir : str, default "./"
        Output directory for saving results
    integrated_data : bool, default False
        Whether this is integrated multimodal data
    not_save : bool, default False
        If True, skip saving files
    atac : bool, default False
        If True, use ATAC-seq appropriate methods
    use_snapatac2_dimred : bool, default False
        If True and atac=True, try snapATAC2 spectral embedding first, fallback to LSI if it fails
    verbose : bool, default True
        Whether to print verbose output
    """
    start_time = time.time() if verbose else None
    
    if "cell_expression_corrected" not in pseudobulk or "cell_proportion" not in pseudobulk:
        raise KeyError("Missing required keys ('cell_expression_corrected' or 'cell_proportion') in pseudobulk.")
    
    if pseudobulk_anndata is None:
        raise ValueError("pseudobulk_anndata parameter is required.")
    
    if verbose:
        print("[process_anndata_with_dimension_reduction] Starting dimension reduction computation...")
        print("[process_anndata_with_dimension_reduction] Results will be stored under unified keys:")
        print("  - X_DR_expression: for expression data")
        print("  - X_DR_proportion: for proportion data")
        if atac:
            if use_snapatac2_dimred:
                print("[process_anndata_with_dimension_reduction] ATAC mode: Will try snapATAC2 spectral first, fallback to LSI if needed")
            else:
                print("[process_anndata_with_dimension_reduction] ATAC mode: Will use LSI for expression data")
        else:
            print("[process_anndata_with_dimension_reduction] RNA mode: Will use PCA for expression data")
    
    # Validate and create output directory structure
    output_dir = os.path.abspath(output_dir)  # Convert to absolute path
    
    count_output_dir = os.path.join(output_dir, "preprocess")
    pseudobulk_output_dir = os.path.join(output_dir, "pseudobulk")

    
    # Create directories early to catch permission issues
    if not not_save:
        try:
            os.makedirs(count_output_dir, exist_ok=True)
            os.makedirs(pseudobulk_output_dir, exist_ok=True)
            if verbose:
                print(f"[process_anndata_with_dimension_reduction] ✓ Created output directories:")
                print(f"  - count_output_dir: {count_output_dir}")
                print(f"  - pseudobulk_output_dir: {pseudobulk_output_dir}")
        except Exception as e:
            if verbose:
                print(f"[process_anndata_with_dimension_reduction] ✗ Failed to create output directories: {str(e)}")
            if not verbose:  # Always show critical errors
                print(f"ERROR: Cannot create output directories: {str(e)}")
            raise
    
    # Get data dimensions and adjust n_components accordingly
    sample_proportion_df = pseudobulk["cell_proportion"]
    
    n_expression_components = min(n_expression_components, min(pseudobulk_anndata.shape) - 1)
    n_proportion_components = min(n_proportion_components, min(sample_proportion_df.shape))
    
    if verbose:
        print(f"[process_anndata_with_dimension_reduction] Using n_expression_components={n_expression_components}, n_proportion_components={n_proportion_components}")
        print(f"[process_anndata_with_dimension_reduction] Data dimensions: expression={pseudobulk_anndata.shape}, proportion={sample_proportion_df.shape}")
    
    # Track what succeeded for better error reporting
    expression_dr_successful = False
    proportion_dr_successful = False
    expression_error = None
    proportion_error = None
    
    # Run dimension reduction on expression data
    try:
        run_dimension_reduction_expression(
            adata=adata, 
            pseudobulk_anndata=pseudobulk_anndata,
            n_components=n_expression_components,
            atac=atac,
            use_snapatac2_dimred=use_snapatac2_dimred,
            verbose=verbose
        )
        expression_dr_successful = True
        if verbose:
            print("[process_anndata_with_dimension_reduction] ✓ Expression dimension reduction completed successfully")
    except Exception as e:
        expression_error = str(e)
        if verbose:
            print(f"[process_anndata_with_dimension_reduction] ✗ Expression dimension reduction failed: {expression_error}")
    
    # Run dimension reduction on proportion data
    try:
        run_dimension_reduction_proportion(
            adata=adata, 
            pseudobulk=pseudobulk, 
            pseudobulk_anndata=pseudobulk_anndata,
            sample_col=sample_col,
            n_components=n_proportion_components, 
            batch_col = batch_col,
            harmony_for_proportion = harmony_for_proportion,
            verbose=verbose
        )
        proportion_dr_successful = True
        if verbose:
            print("[process_anndata_with_dimension_reduction] ✓ Proportion dimension reduction completed successfully")
    except Exception as e:
        proportion_error = str(e)
        if verbose:
            print(f"[process_anndata_with_dimension_reduction] ✗ Proportion dimension reduction failed: {proportion_error}")
    
    # Check if at least one dimension reduction succeeded
    if not expression_dr_successful and not proportion_dr_successful:
        error_msg = "Both expression and proportion dimension reduction failed.\n"
        if expression_error:
            error_msg += f"Expression error: {expression_error}\n"
        if proportion_error:
            error_msg += f"Proportion error: {proportion_error}"
        raise RuntimeError(error_msg)
    
    # Save results if requested and at least one dimension reduction succeeded
    if not not_save:
        if verbose:
            print("[process_anndata_with_dimension_reduction] Preparing to save results...")
        
        # Determine file names based on data type
        if integrated_data:
            adata_filename = 'atac_rna_integrated.h5ad'
        else:
            adata_filename = 'adata_sample.h5ad'
        
        pb_filename = 'pseudobulk_sample.h5ad'
        
        adata_path = os.path.join(count_output_dir, adata_filename)
        pb_adata_path = os.path.join(pseudobulk_output_dir, pb_filename)
        
        if verbose:
            print(f"[process_anndata_with_dimension_reduction] Target file paths:")
            print(f"  - adata: {adata_path}")
            print(f"  - pseudobulk_anndata: {pb_adata_path}")
        
        # Save with detailed error handling
        adata_save_success = _save_anndata_with_detailed_error_handling(
            adata_path, adata, "adata", verbose
        )
        
        pb_save_success = _save_anndata_with_detailed_error_handling(
            pb_adata_path, pseudobulk_anndata, "pseudobulk_anndata", verbose
        )
        
        # Report save results
        if adata_save_success and pb_save_success:
            if verbose:
                print("[process_anndata_with_dimension_reduction] ✓ All files saved successfully")
        elif adata_save_success or pb_save_success:
            if verbose:
                print(f"[process_anndata_with_dimension_reduction] ⚠ Partial save success:")
                print(f"  - adata: {'✓' if adata_save_success else '✗'}")
                print(f"  - pseudobulk_anndata: {'✓' if pb_save_success else '✗'}")
        else:
            if verbose:
                print("[process_anndata_with_dimension_reduction] ✗ All file saves failed")
            else:
                print("WARNING: Failed to save processed data files")
    
    # Final summary
    if verbose and start_time is not None:
        elapsed_time = time.time() - start_time
        print(f"\n[process_anndata_with_dimension_reduction] === SUMMARY ===")
        print(f"Total runtime: {elapsed_time:.2f} seconds")
        print(f"Expression dimension reduction: {'✓ SUCCESS' if expression_dr_successful else '✗ FAILED'}")
        print(f"Proportion dimension reduction: {'✓ SUCCESS' if proportion_dr_successful else '✗ FAILED'}")
        
        if expression_dr_successful or proportion_dr_successful:
            print(f"Results available under unified keys:")
            if expression_dr_successful:
                print(f"  - X_DR_expression in both adata.uns and pseudobulk_anndata.uns")
            if proportion_dr_successful:
                print(f"  - X_DR_proportion in both adata.uns and pseudobulk_anndata.uns")
        
        if not not_save:
            save_attempted = True
            print(f"File saving: {'✓ ATTEMPTED' if save_attempted else 'SKIPPED'}")
    
    return pseudobulk_anndata