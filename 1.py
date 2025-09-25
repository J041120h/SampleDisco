import anndata as ad
import numpy as np
import pandas as pd
import os
import gc
from scipy import sparse

def clean_anndata_for_saving(adata, verbose=True):
    """
    Clean AnnData object to ensure it can be saved to HDF5 format.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object to clean
    verbose : bool
        Whether to print cleaning statistics
    
    Returns:
    --------
    adata : AnnData
        Cleaned AnnData object
    """
    import pandas as pd
    import numpy as np
    
    if verbose:
        print("🧹 Cleaning AnnData object for HDF5 compatibility...")
    
    # Clean obs dataframe
    for col in adata.obs.columns:
        if verbose:
            print(f"   Processing column: {col}")
        
        # Convert object columns to string, handling NaN values
        if adata.obs[col].dtype == 'object':
            # Fill NaN values with 'Unknown' or appropriate default
            adata.obs[col] = adata.obs[col].fillna('Unknown')
            # Convert to string
            adata.obs[col] = adata.obs[col].astype(str)
            # Convert to category for memory efficiency
            adata.obs[col] = adata.obs[col].astype('category')
        
        # Handle numeric columns with NaN
        elif adata.obs[col].dtype in ['float64', 'float32']:
            # Fill NaN values with appropriate defaults
            if adata.obs[col].isna().any():
                adata.obs[col] = adata.obs[col].fillna(0.0)
        
        # Handle integer columns
        elif adata.obs[col].dtype in ['int64', 'int32']:
            # Ensure no NaN values in integer columns
            if adata.obs[col].isna().any():
                adata.obs[col] = adata.obs[col].fillna(0).astype('int64')
    
    # Clean var dataframe
    for col in adata.var.columns:
        if adata.var[col].dtype == 'object':
            # Fill NaN values and convert to string
            adata.var[col] = adata.var[col].fillna('Unknown').astype(str)
            # Convert to category for memory efficiency
            adata.var[col] = adata.var[col].astype('category')
        elif adata.var[col].dtype in ['float64', 'float32']:
            if adata.var[col].isna().any():
                adata.var[col] = adata.var[col].fillna(0.0)
        elif adata.var[col].dtype in ['int64', 'int32']:
            if adata.var[col].isna().any():
                adata.var[col] = adata.var[col].fillna(0).astype('int64')
    
    if verbose:
        print("✅ AnnData cleaning complete")
    
    return adata

def test_metadata_merging(
    glue_dir: str,
    output_path: str,
    raw_rna_path: str,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Improved test version with sparse matrix dtype fixes for large data.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import os
    import gc
    from scipy import sparse
    import scanpy as sc
    
    def fix_sparse_matrix_dtype(X, verbose=True):
        """Fix sparse matrix by converting to int64 indices"""
        if not sparse.issparse(X):
            return X
            
        if verbose:
            print(f"   Converting sparse matrix indices to int64...")
            print(f"   Current dtypes - indices: {X.indices.dtype}, indptr: {X.indptr.dtype}")
        
        # Convert to COO then back to CSR with int64
        coo = X.tocoo()
        
        # Create new CSR matrix with int64 indices
        X_fixed = sparse.csr_matrix(
            (coo.data.astype(np.float64), 
             (coo.row.astype(np.int64), coo.col.astype(np.int64))),
            shape=X.shape,
            dtype=np.float64
        )
        
        # Clean up
        X_fixed.eliminate_zeros()
        X_fixed.sort_indices()
        
        if verbose:
            print(f"   Fixed dtypes - indices: {X_fixed.indices.dtype}, indptr: {X_fixed.indptr.dtype}")
        
        return X_fixed
    
    print("=" * 80)
    print("IMPROVED TEST VERSION: METADATA MERGING WITH DTYPE FIX")
    print("=" * 80)
    
    # Construct file paths
    rna_processed_path = os.path.join(glue_dir, "glue-rna-emb.h5ad")
    atac_path = os.path.join(glue_dir, "glue-atac-emb.h5ad")
    
    # 1. Load processed RNA embeddings
    print("\n1. Loading processed RNA embeddings...")
    rna_processed = ad.read_h5ad(rna_processed_path)
    processed_rna_cells = rna_processed.obs.index.copy()
    rna_obsm_dict = {k: v.copy() for k, v in rna_processed.obsm.items()}
    print(f"   Processed RNA cells: {len(processed_rna_cells)}")
    del rna_processed
    gc.collect()
    
    # 2. Load ATAC
    print("\n2. Loading ATAC embeddings...")
    atac = ad.read_h5ad(atac_path)
    atac_obs = atac.obs.copy()
    atac_obsm_dict = {k: v.copy() for k, v in atac.obsm.items()}
    n_atac_cells = atac.n_obs
    print(f"   ATAC cells: {n_atac_cells}")
    del atac
    gc.collect()
    
    # 3. Load raw RNA
    print("\n3. Loading raw RNA counts...")
    rna_raw = ad.read_h5ad(raw_rna_path, backed='r')
    raw_rna_obs = rna_raw.obs.copy()
    raw_rna_var = rna_raw.var.copy()
    raw_rna_varm_dict = {k: v.copy() for k, v in rna_raw.varm.items()} if hasattr(rna_raw, 'varm') else {}
    
    # 4. Align cells
    print("\n4. Aligning cells between processed and raw RNA...")
    common_cells = processed_rna_cells.intersection(raw_rna_obs.index)
    print(f"   Common cells: {len(common_cells)}")
    
    embedding_mask = np.isin(processed_rna_cells, common_cells)
    for key in rna_obsm_dict:
        rna_obsm_dict[key] = rna_obsm_dict[key][embedding_mask]
    
    rna_obs = raw_rna_obs.loc[common_cells].copy()
    n_rna_cells = len(common_cells)
    n_genes = rna_raw.n_vars
    
    print(f"   Final RNA cells: {n_rna_cells}, Genes: {n_genes}")
    
    # 5. Load ACTUAL RNA data (not dummy) but use subset for testing
    print("\n5. Loading actual RNA data subset for realistic testing...")
    test_size = min(1000, n_rna_cells)
    test_cells = common_cells[:test_size]
    
    # Load actual RNA data
    rna_X = rna_raw[test_cells, :].X
    if sparse.issparse(rna_X):
        rna_X_sparse = rna_X.tocsr().astype(np.float64)
    else:
        rna_X_sparse = sparse.csr_matrix(rna_X[:], dtype=np.float64)
    
    # FIX: Convert to int64 indices
    rna_X_sparse = fix_sparse_matrix_dtype(rna_X_sparse, verbose=True)
    
    print(f"   Loaded RNA subset: shape={rna_X_sparse.shape}, nnz={rna_X_sparse.nnz}")
    
    # 6. Create realistic gene activity matrix (sparse but with proper dtype)
    print("\n6. Creating gene activity matrix with proper dtypes...")
    density = 0.01
    nnz = int(n_atac_cells * n_genes * density)
    
    # Generate with int64 from the start
    row_indices = np.random.randint(0, n_atac_cells, nnz, dtype=np.int64)
    col_indices = np.random.randint(0, n_genes, nnz, dtype=np.int64)
    data = np.random.random(nnz).astype(np.float64) * 10
    
    gene_activity_sparse = sparse.csr_matrix(
        (data, (row_indices, col_indices)), 
        shape=(n_atac_cells, n_genes),
        dtype=np.float64
    )
    
    # Ensure proper dtypes
    gene_activity_sparse = fix_sparse_matrix_dtype(gene_activity_sparse, verbose=True)
    
    print(f"   Created gene activity: shape={gene_activity_sparse.shape}, nnz={gene_activity_sparse.nnz}")
    
    # 7. Create AnnData objects
    print("\n7. Creating AnnData objects...")
    
    # RNA AnnData
    rna_for_merge = ad.AnnData(
        X=rna_X_sparse,
        obs=rna_obs.loc[test_cells].copy(),
        var=raw_rna_var.copy()
    )
    rna_for_merge.obs['modality'] = 'RNA'
    
    for key, value in rna_obsm_dict.items():
        rna_for_merge.obsm[key] = value[:test_size]
    
    # ATAC AnnData
    gene_activity_adata = ad.AnnData(
        X=gene_activity_sparse,
        obs=atac_obs.copy(),
        var=raw_rna_var.copy()
    )
    gene_activity_adata.obs['modality'] = 'ATAC'
    
    for key, value in atac_obsm_dict.items():
        gene_activity_adata.obsm[key] = value
    
    print(f"   RNA: {rna_for_merge.shape}")
    print(f"   ATAC: {gene_activity_adata.shape}")
    
    # Close raw file
    rna_raw.file.close()
    del rna_raw
    gc.collect()
    
    # 8. IMPROVED MERGING WITH INDEX HANDLING
    print("\n" + "=" * 80)
    print("IMPROVED MERGING APPROACH")
    print("=" * 80)
    
    # Check for overlapping indices
    print("\n8a. Checking for index overlaps...")
    rna_indices = set(rna_for_merge.obs.index)
    atac_indices = set(gene_activity_adata.obs.index)
    overlap = rna_indices.intersection(atac_indices)
    
    if overlap:
        print(f"   ⚠️ Found {len(overlap)} overlapping indices - will add modality suffix")
    else:
        print(f"   ✓ No overlapping indices found")
    
    # ALWAYS add modality suffix for safety
    print("\n8b. Adding modality suffix to ensure unique indices...")
    
    # Store original barcodes
    rna_for_merge.obs['original_barcode'] = rna_for_merge.obs.index
    gene_activity_adata.obs['original_barcode'] = gene_activity_adata.obs.index
    
    # Create unique indices
    rna_for_merge.obs.index = pd.Index([f"{idx}_RNA" for idx in rna_for_merge.obs.index])
    gene_activity_adata.obs.index = pd.Index([f"{idx}_ATAC" for idx in gene_activity_adata.obs.index])
    
    print(f"   RNA indices updated: {list(rna_for_merge.obs.index[:3])}")
    print(f"   ATAC indices updated: {list(gene_activity_adata.obs.index[:3])}")
    
    # 9. Merge with fixed indices
    print("\n9. Merging with fixed indices and dtypes...")
    try:
        merged = ad.concat(
            [rna_for_merge, gene_activity_adata], 
            axis=0, 
            join='inner',
            merge='same',
            label=None,
            keys=None,
            index_unique=None
        )
        
        print(f"   ✓ Merge successful: {merged.shape}")
        print(f"   Unique indices: {merged.obs.index.is_unique}")
        print(f"   RNA cells: {(merged.obs['modality'] == 'RNA').sum()}")
        print(f"   ATAC cells: {(merged.obs['modality'] == 'ATAC').sum()}")
        
        # Verify sparse matrix dtypes
        if sparse.issparse(merged.X):
            print(f"   Matrix dtypes - indices: {merged.X.indices.dtype}, indptr: {merged.X.indptr.dtype}")
        
    except Exception as e:
        print(f"   ✗ Merge failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 10. Test scanpy operations on merged data
    print("\n10. Testing scanpy operations on merged data...")
    try:
        # Ensure CSR format
        if sparse.issparse(merged.X):
            merged.X = merged.X.tocsr()
            merged.X.sort_indices()
            merged.X.eliminate_zeros()
        
        # Test filtering on subset
        test_subset = merged[:100, :].copy()
        sc.pp.filter_cells(test_subset, min_genes=10)
        print(f"   ✓ Scanpy filter works on subset: {test_subset.shape}")
        
        # Test on full merged data
        initial_shape = merged.shape
        sc.pp.filter_cells(merged, min_genes=10)
        print(f"   ✓ Scanpy filter works on full data: {initial_shape} -> {merged.shape}")
        
    except Exception as e:
        print(f"   ✗ Scanpy operation failed: {e}")
        
        # Manual filtering as fallback
        print("   Attempting manual filtering...")
        if sparse.issparse(merged.X):
            genes_per_cell = np.array((merged.X > 0).sum(axis=1)).flatten()
        else:
            genes_per_cell = (merged.X > 0).sum(axis=1)
        
        keep_cells = genes_per_cell >= 10
        merged = merged[keep_cells, :].copy()
        print(f"   ✓ Manual filtering successful: {merged.shape}")
    
    # 11. Clean and save
    print("\n11. Cleaning and saving...")
    try:
        # Clean metadata
        merged = clean_anndata_for_saving(merged, verbose=False)
        
        # Ensure proper sparse format
        if sparse.issparse(merged.X):
            merged.X = fix_sparse_matrix_dtype(merged.X, verbose=False)
        
        # Save
        test_output_dir = os.path.join(output_path, 'test_merge_fixed')
        os.makedirs(test_output_dir, exist_ok=True)
        test_output_path = os.path.join(test_output_dir, 'test_merged_fixed.h5ad')
        
        merged.write(test_output_path, compression='gzip', compression_opts=4)
        print(f"   ✓ Save successful: {test_output_path}")
        
        # Verify by reloading
        reloaded = ad.read_h5ad(test_output_path)
        print(f"   ✓ Reload successful: {reloaded.shape}")
        
        # Final test on reloaded data
        sc.pp.filter_cells(reloaded, min_genes=10)
        print(f"   ✓ Scanpy works on reloaded data: {reloaded.shape}")
        
    except Exception as e:
        print(f"   ✗ Save/reload failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE - Data should now work with scanpy")
    print("=" * 80)
    
    return merged

import anndata as ad
import numpy as np
from scipy import sparse
import scanpy as sc

def diagnose_sparse_matrix_issue(h5ad_path):
    """Deep diagnostic of sparse matrix issues"""
    
    print("Loading data...")
    adata = ad.read_h5ad(h5ad_path)
    
    print(f"\n1. Basic info:")
    print(f"   Shape: {adata.shape}")
    print(f"   Matrix type: {type(adata.X)}")
    
    if sparse.issparse(adata.X):
        print(f"\n2. Sparse matrix internals:")
        print(f"   Format: {adata.X.format}")
        print(f"   dtype: {adata.X.dtype}")
        print(f"   nnz: {adata.X.nnz}")
        
        # Check internal arrays
        if hasattr(adata.X, 'data'):
            print(f"   data.shape: {adata.X.data.shape}")
            print(f"   data.dtype: {adata.X.data.dtype}")
            
        if hasattr(adata.X, 'indices'):
            print(f"   indices.shape: {adata.X.indices.shape}")
            print(f"   indices.dtype: {adata.X.indices.dtype}")
            print(f"   indices.max(): {adata.X.indices.max()}")
            print(f"   indices.min(): {adata.X.indices.min()}")
            
        if hasattr(adata.X, 'indptr'):
            print(f"   indptr.shape: {adata.X.indptr.shape}")
            print(f"   indptr.dtype: {adata.X.indptr.dtype}")
            print(f"   indptr[-1]: {adata.X.indptr[-1]}")
            
        # Check for consistency
        print(f"\n3. Consistency checks:")
        
        # For CSR matrix
        if isinstance(adata.X, sparse.csr_matrix):
            expected_indptr_len = adata.X.shape[0] + 1
            actual_indptr_len = len(adata.X.indptr)
            print(f"   CSR indptr length: {actual_indptr_len} (expected: {expected_indptr_len})")
            
            if adata.X.indptr[-1] != len(adata.X.indices):
                print(f"   ⚠️ WARNING: indptr[-1] ({adata.X.indptr[-1]}) != len(indices) ({len(adata.X.indices)})")
        
        # For CSC matrix  
        elif isinstance(adata.X, sparse.csc_matrix):
            expected_indptr_len = adata.X.shape[1] + 1
            actual_indptr_len = len(adata.X.indptr)
            print(f"   CSC indptr length: {actual_indptr_len} (expected: {expected_indptr_len})")
            
        # Check if indices are within bounds
        if hasattr(adata.X, 'indices'):
            max_allowed = adata.X.shape[1] - 1 if isinstance(adata.X, sparse.csr_matrix) else adata.X.shape[0] - 1
            if adata.X.indices.max() > max_allowed:
                print(f"   ⚠️ WARNING: indices.max() ({adata.X.indices.max()}) > max_allowed ({max_allowed})")
    
    print(f"\n4. Testing operations:")
    
    # Test 1: Simple slicing
    try:
        test_slice = adata[:10, :10]
        print(f"   ✓ Simple slicing works: {test_slice.shape}")
    except Exception as e:
        print(f"   ✗ Simple slicing failed: {e}")
    
    # Test 2: Manual gene counting (avoiding scanpy)
    try:
        if sparse.issparse(adata.X):
            n_genes = np.array((adata.X > 0).sum(axis=1)).flatten()
        else:
            n_genes = (adata.X > 0).sum(axis=1)
        print(f"   ✓ Manual gene counting works: min={n_genes.min()}, max={n_genes.max()}")
    except Exception as e:
        print(f"   ✗ Manual gene counting failed: {e}")
    
    # Test 3: The problematic scanpy operation
    try:
        test_copy = adata[:100, :].copy()
        sc.pp.filter_cells(test_copy, min_genes=10)
        print(f"   ✓ Scanpy filter works on small subset")
    except Exception as e:
        print(f"   ✗ Scanpy filter failed even on small subset: {e}")
    
    return adata

def diagnose_metadata_corruption(h5ad_path):
    """
    Diagnose metadata issues that cause segfaults in scanpy
    """
    import anndata as ad
    import pandas as pd
    import numpy as np
    from scipy import sparse
    
    print("DEEP DIAGNOSIS OF METADATA CORRUPTION")
    print("=" * 80)
    
    adata = ad.read_h5ad(h5ad_path)
    
    # 1. Check dimension consistency
    print("\n1. DIMENSION CONSISTENCY CHECK:")
    print(f"   X shape: {adata.X.shape}")
    print(f"   obs length: {len(adata.obs)}")
    print(f"   var length: {len(adata.var)}")
    
    if adata.X.shape[0] != len(adata.obs):
        print(f"   ❌ CRITICAL: X rows ({adata.X.shape[0]}) != obs rows ({len(adata.obs)})")
    if adata.X.shape[1] != len(adata.var):
        print(f"   ❌ CRITICAL: X cols ({adata.X.shape[1]}) != var rows ({len(adata.var)})")
    
    # 2. Check for corrupted categorical columns
    print("\n2. CATEGORICAL COLUMN CORRUPTION CHECK:")
    
    for col in adata.obs.columns:
        if pd.api.types.is_categorical_dtype(adata.obs[col]):
            cat_col = adata.obs[col]
            print(f"\n   Checking obs['{col}']:")
            
            # Check for invalid category codes
            if hasattr(cat_col.cat, 'codes'):
                codes = cat_col.cat.codes
                n_categories = len(cat_col.cat.categories)
                
                # Check if any code exceeds the number of categories
                invalid_codes = codes[(codes >= n_categories) | (codes < -1)]
                if len(invalid_codes) > 0:
                    print(f"      ❌ CORRUPTED: Found {len(invalid_codes)} invalid category codes!")
                    print(f"         Max code: {codes.max()}, but only {n_categories} categories")
                
                # Check for -1 codes (NaN representation)
                nan_codes = (codes == -1).sum()
                if nan_codes > 0:
                    print(f"      ⚠️  {nan_codes} NaN values in categorical")
            
            # Check for empty categories
            value_counts = cat_col.value_counts()
            unused_cats = set(cat_col.cat.categories) - set(value_counts.index)
            if unused_cats:
                print(f"      ⚠️  {len(unused_cats)} unused categories")
    
    # 3. Check for memory view issues
    print("\n3. CHECKING FOR MEMORY VIEW ISSUES:")
    
    # Test if obs/var are views or copies
    try:
        # Try modifying a copy - if it affects original, it's a view
        obs_copy = adata.obs.copy()
        if len(adata.obs.columns) > 0:
            first_col = adata.obs.columns[0]
            original_val = adata.obs[first_col].iloc[0] if len(adata.obs) > 0 else None
            obs_copy[first_col].iloc[0] = 'TEST_MODIFICATION'
            if len(adata.obs) > 0 and adata.obs[first_col].iloc[0] == 'TEST_MODIFICATION':
                print("   ❌ CRITICAL: obs is a memory view, not a copy!")
    except:
        pass
    
    # 4. Test scanpy operation in isolation
    print("\n4. TESTING SCANPY OPERATIONS:")
    
    # Test on tiny subset first
    try:
        tiny = adata[:10, :10].copy()
        genes_per_cell = np.array((tiny.X > 0).sum(axis=1)).flatten()
        print(f"   ✓ Manual gene counting works on 10x10 subset")
    except Exception as e:
        print(f"   ❌ Even manual counting fails on subset: {e}")
    
    # Try filter_cells on subset
    try:
        import scanpy as sc
        tiny = adata[:100, :100].copy()
        sc.pp.filter_cells(tiny, min_genes=1)
        print(f"   ✓ scanpy filter_cells works on 100x100 subset")
    except Exception as e:
        print(f"   ❌ scanpy filter_cells fails on subset: {e}")
    
    return adata

# Fix categorical corruption
def fix_categorical_columns(adata, verbose=True):
    """
    Fix corrupted categorical columns that cause segfaults
    """
    import pandas as pd
    
    if verbose:
        print("FIXING CATEGORICAL COLUMN CORRUPTION")
        print("=" * 80)
    
    # Fix obs
    for col in adata.obs.columns:
        if pd.api.types.is_categorical_dtype(adata.obs[col]):
            if verbose:
                print(f"Fixing obs['{col}']...")
            
            # Convert to string, then back to categorical
            # This rebuilds the category codes properly
            values = adata.obs[col].astype(str)
            adata.obs[col] = pd.Categorical(values)
    
    # Fix var
    for col in adata.var.columns:
        if pd.api.types.is_categorical_dtype(adata.var[col]):
            if verbose:
                print(f"Fixing var['{col}']...")
            
            values = adata.var[col].astype(str)
            adata.var[col] = pd.Categorical(values)
    
    # Ensure obs and var are true copies, not views
    adata.obs = adata.obs.copy()
    adata.var = adata.var.copy()
    
    # Ensure matrix format consistency
    if sparse.issparse(adata.X):
        # Force rebuild of sparse matrix structure
        adata.X = sparse.csr_matrix(adata.X.tocsr())
        adata.X.sort_indices()
        adata.X.eliminate_zeros()
    
    return adata

from integration.integration_preprocess import integrate_preprocess

# Run the test
if __name__ == "__main__":
    # print("Running metadata merging test...")
    # # test_metadata_merging(
    # #     glue_dir="/dcs07/hongkai/data/harry/result/Benchmark/multiomics/integration/glue",
    # #     output_path="/dcs07/hongkai/data/harry/result/Benchmark/multiomics/preprocess",
    # #     raw_rna_path= '/dcl01/hongkai/data/data/hjiang/Data/paired/rna/all.h5ad'
    # # )

    # integrate_preprocess(
    #     output_dir = '/dcs07/hongkai/data/harry/result/Benchmark/multiomics/preprocess/test_merge',
    #     h5ad_path = '/dcs07/hongkai/data/harry/result/Benchmark/multiomics/preprocess/test_merge/test_merged.h5ad',
    #     sample_column = 'sample',
    #     modality_col = 'modality',
    #     min_cells_sample=1,
    #     min_cell_gene=10,
    #     min_features=500,
    #     pct_mito_cutoff=20,
    #     exclude_genes=None,
    #     doublet=True,
    #     verbose=True
    # )

    # fix_categorical_columns(ad.read_h5ad("/dcs07/hongkai/data/harry/result/Benchmark/multiomics/preprocess/test_merge/test_merged.h5ad"))
    diagnose_sparse_matrix_issue("/dcs07/hongkai/data/harry/result/heart/multiomics/preprocess/atac_rna_integrated.h5ad")