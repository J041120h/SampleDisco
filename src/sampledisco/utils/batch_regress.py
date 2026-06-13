import scanpy as sc
import numpy as np
import warnings
import io
import contextlib
from scipy.sparse import issparse


def simple_batch_regression(adata: sc.AnnData, batch_col: str, verbose: bool = False) -> sc.AnnData:
    """Regress out ``batch_col`` via ``sc.pp.regress_out`` (fallback when Combat times out).

    If regression introduces NaN (can happen when a batch column has rank-deficient
    design), the original X is restored and the function returns unchanged.

    Modifies ``adata.X`` in-place; returns ``adata``.
    """
    if verbose:
        print(f"  Applying simple batch regression (fallback method)...")

    try:
        original_X = adata.X.copy()

        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sc.pp.regress_out(adata, [batch_col])

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
            adata.X = original_X
            return adata
        
        if verbose:
            print(f"  Simple batch regression completed successfully")

    except Exception as e:
        if verbose:
            print(f"  Simple batch regression failed: {str(e)}")
            print(f"  Proceeding with original data")

    return adata
