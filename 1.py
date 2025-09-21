import anndata as ad
import numpy as np
import scipy.sparse as sp
from typing import Dict, Any, Optional

def _is_sparseish(X) -> bool:
    """True if X is SciPy sparse or AnnData SparseDataset (has .format like 'csr'/'csc')."""
    if sp.issparse(X):
        return True
    fmt = getattr(X, "format", None)
    return fmt in ("csr", "csc")

def inspect_X(
    filepath: str,
    verbose: bool = True,
    chunk_rows: int = 10_000,
    backed: Optional[str] = "r",
) -> Dict[str, Any]:
    """
    Robust inspection of .X: counts zeros, checks NaN/Inf (floats only), and gives basic stats.
    Safe for dense, SciPy sparse, and AnnData backed SparseDataset.
    """
    res: Dict[str, Any] = {}
    adata = ad.read_h5ad(filepath, backed=backed)
    X = adata.X

    n_cells, n_genes = X.shape
    total = n_cells * n_genes
    res["shape"] = (n_cells, n_genes)
    res["dtype"] = getattr(X, "dtype", str(type(X)))
    res["sparse"] = _is_sparseish(X)
    res["format"] = (getattr(X, "format", None) if res["sparse"] else "ndarray")

    # Defaults
    res.update(dict(
        has_nan=False, has_inf=False,
        num_zeros=None, nnz=None, density=None,
        min=None, max=None, mean=None
    ))

    # ======= Global sparse path (fast) =======
    if res["sparse"]:
        # Handle CSRDataset/CSCDataset (backed AnnData) specially
        is_backed_sparse = hasattr(X, 'group')  # AnnData backed datasets have 'group' attribute
        
        if is_backed_sparse:
            # For backed sparse datasets, we need to compute everything in chunks
            nnz = 0
            cur_min, cur_max = None, None
            data_sum = 0.0
            has_nan = False
            has_inf = False
            
            for i0 in range(0, n_cells, chunk_rows):
                i1 = min(i0 + chunk_rows, n_cells)
                try:
                    # Access the chunk - this returns a scipy sparse matrix
                    s = X[i0:i1, :]
                    
                    # Convert to csr if needed and ensure it's scipy sparse
                    if not sp.issparse(s):
                        s = sp.csr_matrix(s)
                    
                    # Update nnz count
                    nnz += int(s.nnz)
                    
                    # Process non-zero data
                    if s.nnz > 0:
                        d = s.data
                        kind = d.dtype.kind
                        
                        if kind == 'f':
                            if not has_nan and np.isnan(d).any(): 
                                has_nan = True
                            if not has_inf and np.isinf(d).any(): 
                                has_inf = True
                        
                        # Update min/max from nonzeros
                        dmin, dmax = float(d.min()), float(d.max())
                        cur_min = dmin if cur_min is None else min(cur_min, dmin)
                        cur_max = dmax if cur_max is None else max(cur_max, dmax)
                        data_sum += float(d.sum())
                    
                    # Check if there are implicit zeros (sparse format)
                    if s.nnz < s.shape[0] * s.shape[1]:
                        cur_min = 0.0 if (cur_min is None or cur_min > 0.0) else cur_min
                        
                except Exception as e:
                    print(f"Warning: Error processing chunk {i0}:{i1}: {e}")
                    # Try to continue with remaining chunks
                    continue
            
            res["nnz"] = nnz
            res["num_zeros"] = int(total - nnz)
            res["density"] = nnz / total if total else 0.0
            res["has_nan"], res["has_inf"] = bool(has_nan), bool(has_inf)
            res["mean"] = float(data_sum / total) if total else None
            res["min"] = cur_min
            res["max"] = cur_max
            
        else:
            # Regular SciPy sparse matrix - can use nnz directly
            nnz = int(X.nnz)
            res["nnz"] = nnz
            res["num_zeros"] = int(total - nnz)
            res["density"] = nnz / total if total else 0.0
            
            # Stats & NaN/Inf from nonzero data
            cur_min, cur_max = None, None
            data_sum = 0.0
            has_nan = False
            has_inf = False
            
            for i0 in range(0, n_cells, chunk_rows):
                i1 = min(i0 + chunk_rows, n_cells)
                s = X[i0:i1, :]
                
                if s.nnz > 0:
                    d = s.data
                    kind = d.dtype.kind
                    
                    if kind == 'f':
                        if not has_nan and np.isnan(d).any(): has_nan = True
                        if not has_inf and np.isinf(d).any(): has_inf = True
                    
                    dmin, dmax = float(d.min()), float(d.max())
                    cur_min = dmin if cur_min is None else min(cur_min, dmin)
                    cur_max = dmax if cur_max is None else max(cur_max, dmax)
                    data_sum += float(d.sum())
                
                if s.nnz < s.shape[0] * s.shape[1]:
                    cur_min = 0.0 if (cur_min is None or cur_min > 0.0) else cur_min
            
            res["has_nan"], res["has_inf"] = bool(has_nan), bool(has_inf)
            res["mean"] = float(data_sum / total) if total else None
            res["min"] = cur_min
            res["max"] = cur_max

    # ======= Dense path (chunked) =======
    else:
        dtype = getattr(X, "dtype", None)
        kind = getattr(dtype, "kind", None)
        check_nan_inf = (kind == 'f')

        zeros = 0
        cur_min, cur_max = None, None
        data_sum = 0.0
        has_nan = False
        has_inf = False

        for i0 in range(0, n_cells, chunk_rows):
            i1 = min(i0 + chunk_rows, n_cells)
            arr = np.asarray(X[i0:i1, :])   # materialize the slice

            zeros += int(np.count_nonzero(arr == 0))
            vmin, vmax = float(arr.min()), float(arr.max())
            cur_min = vmin if cur_min is None else min(cur_min, vmin)
            cur_max = vmax if cur_max is None else max(cur_max, vmax)
            data_sum += float(arr.sum())

            if check_nan_inf:
                if not has_nan and np.isnan(arr).any(): has_nan = True
                if not has_inf and np.isinf(arr).any(): has_inf = True

        res["num_zeros"] = int(zeros)
        res["has_nan"], res["has_inf"] = bool(has_nan), bool(has_inf)
        res["mean"] = float(data_sum / total) if total else None
        res["min"], res["max"] = cur_min, cur_max

    if verbose:
        print("=" * 70)
        print(f"File: {filepath}")
        print(f"Shape: {res['shape']}   dtype: {res['dtype']}")
        print(f"sparse: {res['sparse']}   format: {res['format']}")
        if res["sparse"]:
            print(f"nnz: {res['nnz']}   density: {res['density']:.6f}")
        print(f"num zeros: {res['num_zeros']}")
        print(f"has NaN: {res['has_nan']}   has Inf: {res['has_inf']}")
        print(f"min: {res['min']}   max: {res['max']}   mean: {res['mean']}")
        print("=" * 70)

    # close backed file handle (no-op if not backed)
    try:
        adata.file.close()
    except Exception:
        pass

    return res

import anndata as ad
import numpy as np
from scipy import sparse
import gc

def validate_and_fix_csr_matrix(filepath, output_path=None, chunk_size=10000):
    """
    Load a corrupted h5ad file, validate and fix the CSR matrix structure.
    
    Parameters:
    -----------
    filepath : str
        Path to the corrupted h5ad file
    output_path : str, optional
        Path to save the fixed file. If None, will overwrite the original
    chunk_size : int
        Number of rows to process at a time
    """
    print(f"Loading file: {filepath}")
    
    # Load the file without backing mode to avoid CSR corruption issues
    try:
        adata = ad.read_h5ad(filepath, backed=None)
    except Exception as e:
        print(f"Error loading file directly: {e}")
        print("Attempting to load with backing mode and reconstruct...")
        
        # If direct loading fails, try backed mode and reconstruct
        adata = ad.read_h5ad(filepath, backed='r')
        
        print(f"Original shape: {adata.shape}")
        print(f"Processing in chunks of {chunk_size} rows...")
        
        # Process in chunks to avoid memory issues
        n_chunks = (adata.n_obs + chunk_size - 1) // chunk_size
        chunks = []
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, adata.n_obs)
            
            try:
                # Try to read chunk
                chunk_data = adata[start_idx:end_idx, :].X
                
                # Convert to dense array if sparse
                if sparse.issparse(chunk_data):
                    chunk_data = chunk_data.toarray()
                
                chunks.append(chunk_data)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{n_chunks} chunks")
                    
            except Exception as e:
                print(f"  Error reading chunk {start_idx}:{end_idx}: {e}")
                # Create zero array for failed chunk
                chunks.append(np.zeros((end_idx - start_idx, adata.n_vars)))
        
        # Reconstruct the matrix
        print("Reconstructing matrix from chunks...")
        full_matrix = np.vstack(chunks)
        
        # Create new AnnData object with properly structured sparse matrix
        print("Creating new AnnData with valid CSR matrix...")
        X_sparse = sparse.csr_matrix(full_matrix, dtype=np.float64)
        
        # Validate the CSR structure
        if not _validate_csr_structure(X_sparse):
            raise ValueError("Failed to create valid CSR matrix")
        
        # Create new AnnData object
        adata_fixed = ad.AnnData(
            X=X_sparse,
            obs=adata.obs.copy(),
            var=adata.var.copy()
        )
        
        # Copy over other attributes
        if hasattr(adata, 'obsm'):
            for key in adata.obsm.keys():
                adata_fixed.obsm[key] = adata.obsm[key].copy()
        
        if hasattr(adata, 'varm'):
            for key in adata.varm.keys():
                adata_fixed.varm[key] = adata.varm[key].copy()
        
        if hasattr(adata, 'layers'):
            for key in adata.layers.keys():
                try:
                    layer_data = adata.layers[key]
                    if sparse.issparse(layer_data):
                        adata_fixed.layers[key] = sparse.csr_matrix(layer_data.toarray())
                    else:
                        adata_fixed.layers[key] = layer_data.copy()
                except:
                    print(f"  Warning: Could not copy layer '{key}'")
        
        adata = adata_fixed
        
        # Clean up
        del chunks, full_matrix
        gc.collect()
    
    # Validate and fix the CSR matrix structure
    print("\nValidating CSR matrix structure...")
    
    if sparse.issparse(adata.X):
        print(f"Matrix format: {type(adata.X).__name__}")
        print(f"Matrix shape: {adata.X.shape}")
        print(f"Matrix dtype: {adata.X.dtype}")
        print(f"nnz: {adata.X.nnz}")
        
        # Ensure it's CSR
        if not isinstance(adata.X, sparse.csr_matrix):
            print("Converting to CSR format...")
            adata.X = adata.X.tocsr()
        
        # Validate structure
        is_valid = _validate_csr_structure(adata.X)
        
        if not is_valid:
            print("CSR structure is invalid. Reconstructing...")
            # Reconstruct by converting to/from dense
            dense_matrix = adata.X.toarray()
            adata.X = sparse.csr_matrix(dense_matrix, dtype=adata.X.dtype)
            
            # Re-validate
            if not _validate_csr_structure(adata.X):
                raise ValueError("Failed to fix CSR structure")
            
            print("✓ CSR structure fixed")
        else:
            print("✓ CSR structure is valid")
        
        # Clean up indices and pointers
        print("Optimizing CSR structure...")
        adata.X.sum_duplicates()
        adata.X.sort_indices()
        
        # Set appropriate dtype for indices
        max_nnz = adata.X.shape[0] * adata.X.shape[1]
        if adata.X.nnz > np.iinfo(np.int32).max or max_nnz > np.iinfo(np.int32).max:
            print("Using int64 for indices (large matrix)")
            adata.X.indices = adata.X.indices.astype(np.int64, copy=False)
            adata.X.indptr = adata.X.indptr.astype(np.int64, copy=False)
        else:
            print("Using int32 for indices")
            adata.X.indices = adata.X.indices.astype(np.int32, copy=False)
            adata.X.indptr = adata.X.indptr.astype(np.int32, copy=False)
    
    # Save the fixed file
    if output_path is None:
        output_path = filepath.replace('.h5ad', '_fixed.h5ad')
    
    print(f"\nSaving fixed file to: {output_path}")
    adata.write(output_path, compression='gzip')
    
    print("✓ File saved successfully")
    
    # Verify the saved file can be read
    print("\nVerifying saved file...")
    adata_test = ad.read_h5ad(output_path, backed='r')
    
    # Try to read a chunk
    try:
        test_chunk = adata_test[:1000, :].X
        print("✓ File verification successful")
    except Exception as e:
        print(f"✗ File verification failed: {e}")
        raise
    
    return adata

def _validate_csr_structure(matrix):
    """Validate CSR matrix structure"""
    if not isinstance(matrix, sparse.csr_matrix):
        return False
    
    # Check basic structure
    n_rows, n_cols = matrix.shape
    
    # indptr should have length n_rows + 1
    if len(matrix.indptr) != n_rows + 1:
        print(f"  ✗ Invalid indptr length: {len(matrix.indptr)} (expected {n_rows + 1})")
        return False
    
    # First element of indptr should be 0
    if matrix.indptr[0] != 0:
        print(f"  ✗ indptr[0] is not 0: {matrix.indptr[0]}")
        return False
    
    # Last element of indptr should equal nnz
    if matrix.indptr[-1] != len(matrix.data):
        print(f"  ✗ indptr[-1] != len(data): {matrix.indptr[-1]} != {len(matrix.data)}")
        return False
    
    # indptr should be monotonically increasing
    if not np.all(np.diff(matrix.indptr) >= 0):
        print("  ✗ indptr is not monotonically increasing")
        return False
    
    # indices and data should have the same length
    if len(matrix.indices) != len(matrix.data):
        print(f"  ✗ indices and data length mismatch: {len(matrix.indices)} != {len(matrix.data)}")
        return False
    
    # All indices should be valid column indices
    if len(matrix.indices) > 0:
        if np.min(matrix.indices) < 0 or np.max(matrix.indices) >= n_cols:
            print(f"  ✗ Invalid indices: min={np.min(matrix.indices)}, max={np.max(matrix.indices)}, n_cols={n_cols}")
            return False
    
    return True

# # Usage example
# if __name__ == "__main__":
#     filepath = '/dcs07/hongkai/data/harry/result/Benchmark/multiomics/preprocess/atac_rna_integrated.h5ad'
    
#     # Fix the corrupted file
#     adata_fixed = validate_and_fix_csr_matrix(
#         filepath,
#         output_path=filepath.replace('.h5ad', '_fixed.h5ad')
#     )
    
#     print(f"\nFixed AnnData shape: {adata_fixed.shape}")

if __name__ == "__main__":
    inspect_X('/dcl01/hongkai/data/data/hjiang/Data/paired/atac/all.h5ad')