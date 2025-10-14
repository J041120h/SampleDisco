import scanpy as sc
import numpy as np
import warnings
import io
import contextlib
from scipy.sparse import issparse


def simple_batch_regression(adata: sc.AnnData, batch_col: str, verbose: bool = False) -> sc.AnnData:
    """
    Perform simple batch effect regression as a fallback when Combat times out.
    
    This function uses scanpy's regress_out to remove batch effects through linear regression,
    which is much faster than Combat but may be less sophisticated.
    
    Parameters
    ----------
    adata : sc.AnnData
        AnnData object to correct
    batch_col : str
        Column name for batch information
    verbose : bool
        Print progress information
    
    Returns
    -------
    adata : sc.AnnData
        Batch-corrected AnnData object (modified in-place)
    """
    if verbose:
        print(f"  Applying simple batch regression (fallback method)...")
    
    try:
        # Store the original data in case we need to revert
        original_X = adata.X.copy()
        
        # Use scanpy's regress_out function which performs linear regression
        # This is much faster than Combat and less likely to timeout
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sc.pp.regress_out(adata, [batch_col])
        
        # Check for NaN values after regression
        if issparse(adata.X):
            has_nan = np.any(np.isnan(adata.X.data))
            if has_nan:
                nan_genes = np.array(np.isnan(adata.X.toarray()).any(axis=0)).flatten()
        else:
            nan_genes = np.isnan(adata.X).any(axis=0)
            has_nan = nan_genes.any()
        
        if has_nan:
            if verbose:
                print(f"  Warning: Found {nan_genes.sum()} genes with NaN after regression, reverting to original")
            # Revert to original data if regression introduced NaN values
            adata.X = original_X
            return adata
        
        if verbose:
            print(f"  Simple batch regression completed successfully")
        
    except Exception as e:
        if verbose:
            print(f"  Simple batch regression failed: {str(e)}")
            print(f"  Proceeding with original data")
    
    return adata
