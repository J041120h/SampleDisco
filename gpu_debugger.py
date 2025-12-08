"""
Data Inspector for GPU Single-Cell Processing
Diagnoses data issues and memory limits before full pipeline run.

=== MODIFY THESE PARAMETERS ===
"""

# =============================================================================
# USER INPUT - MODIFY HERE
# =============================================================================
H5AD_PATH = "/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/paired_rna_atac_merged.h5ad"
SAMPLE_COLUMN = "sample"
SAMPLE_META_PATH = None  # or "/path/to/sample_meta.csv"
CELL_META_PATH = None    # or "/path/to/cell_meta.csv"

RUN_GPU_TEST = True
TEST_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
NUM_FEATURES = 2000

# =============================================================================
# CODE BELOW - DO NOT MODIFY
# =============================================================================

import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse import issparse, csr_matrix
import warnings


def inspect_h5ad(h5ad_path, sample_column='sample', sample_meta_path=None, cell_meta_path=None):
    """Comprehensive data inspection for h5ad files before GPU processing."""
    print("=" * 70)
    print("DATA INSPECTOR FOR GPU SINGLE-CELL PROCESSING")
    print("=" * 70)
    
    results = {"passed": [], "warnings": [], "errors": []}
    
    # =========================================================================
    # 1. BASIC FILE AND LOADING CHECKS
    # =========================================================================
    print("\n[1/8] BASIC FILE CHECKS")
    print("-" * 40)
    
    if not os.path.exists(h5ad_path):
        results["errors"].append(f"File not found: {h5ad_path}")
        print(f"  ❌ File not found: {h5ad_path}")
        return results, None
    
    file_size_gb = os.path.getsize(h5ad_path) / (1024**3)
    print(f"  File size: {file_size_gb:.2f} GB")
    
    try:
        adata = sc.read_h5ad(h5ad_path)
        print(f"  ✓ Successfully loaded h5ad")
        results["passed"].append("File loading")
    except Exception as e:
        results["errors"].append(f"Failed to load h5ad: {e}")
        print(f"  ❌ Failed to load: {e}")
        return results, None
    
    print(f"  Shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    
    # =========================================================================
    # 2. MATRIX FORMAT AND DTYPE CHECKS
    # =========================================================================
    print("\n[2/8] MATRIX FORMAT CHECKS")
    print("-" * 40)
    
    X = adata.X
    print(f"  Matrix type: {type(X).__name__}")
    print(f"  Data dtype: {X.dtype}")
    print(f"  Is sparse: {issparse(X)}")
    
    if issparse(X):
        print(f"  Sparse format: {X.format}")
        print(f"  NNZ (non-zeros): {X.nnz:,}")
        sparsity = 1 - (X.nnz / (X.shape[0] * X.shape[1]))
        print(f"  Sparsity: {sparsity*100:.2f}%")
        
        if X.format != 'csr':
            results["warnings"].append(f"Matrix is {X.format}, not CSR")
            print(f"  ⚠ Matrix is {X.format}, CSR format preferred for GPU ops")
    
    if X.dtype not in [np.float32, np.float64]:
        if np.issubdtype(X.dtype, np.integer):
            results["warnings"].append(f"Integer dtype {X.dtype} - will convert to float32")
            print(f"  ⚠ Integer dtype detected - conversion to float32 needed")
        else:
            results["errors"].append(f"Unsupported dtype: {X.dtype}")
            print(f"  ❌ Unsupported dtype: {X.dtype}")
    else:
        results["passed"].append("Matrix dtype compatible")
        print(f"  ✓ Dtype compatible")
    
    # =========================================================================
    # 3. DATA VALUE CHECKS
    # =========================================================================
    print("\n[3/8] DATA VALUE CHECKS")
    print("-" * 40)
    
    if adata.n_obs > 10000:
        idx = np.random.choice(adata.n_obs, 10000, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
    
    if issparse(X_sample):
        data_vals = X_sample.data
    else:
        data_vals = X_sample.flatten()
    
    has_nan = np.any(np.isnan(data_vals))
    has_inf = np.any(np.isinf(data_vals))
    
    if has_nan:
        nan_count = np.sum(np.isnan(data_vals))
        results["errors"].append(f"Matrix contains {nan_count:,} NaN values")
        print(f"  ❌ Contains NaN values: {nan_count:,}")
    else:
        print(f"  ✓ No NaN values")
        results["passed"].append("No NaN values")
    
    if has_inf:
        inf_count = np.sum(np.isinf(data_vals))
        results["errors"].append(f"Matrix contains {inf_count:,} Inf values")
        print(f"  ❌ Contains Inf values: {inf_count:,}")
    else:
        print(f"  ✓ No Inf values")
        results["passed"].append("No Inf values")
    
    has_neg = np.any(data_vals < 0)
    if has_neg:
        neg_count = np.sum(data_vals < 0)
        results["warnings"].append(f"Matrix contains {neg_count:,} negative values")
        print(f"  ⚠ Contains negative values: {neg_count:,}")
    else:
        print(f"  ✓ No negative values")
        results["passed"].append("No negative values")
    
    print(f"  Value range: [{data_vals.min():.4f}, {data_vals.max():.4f}]")
    print(f"  Mean: {data_vals.mean():.4f}, Std: {data_vals.std():.4f}")
    
    # =========================================================================
    # 4. OBSERVATION (CELL) METADATA CHECKS
    # =========================================================================
    print("\n[4/8] CELL METADATA CHECKS")
    print("-" * 40)
    
    print(f"  Columns: {list(adata.obs.columns)}")
    
    if sample_column in adata.obs.columns:
        n_samples = adata.obs[sample_column].nunique()
        print(f"  ✓ Sample column '{sample_column}' found: {n_samples} unique samples")
        results["passed"].append(f"Sample column '{sample_column}' present")
        
        nan_samples = adata.obs[sample_column].isna().sum()
        if nan_samples > 0:
            results["errors"].append(f"{nan_samples:,} cells have NaN in sample column")
            print(f"  ❌ {nan_samples:,} cells have NaN in '{sample_column}'")
        
        sample_sizes = adata.obs[sample_column].value_counts()
        print(f"  Sample sizes: min={sample_sizes.min()}, max={sample_sizes.max()}, median={sample_sizes.median():.0f}")
        
        small_samples = (sample_sizes < 100).sum()
        if small_samples > 0:
            results["warnings"].append(f"{small_samples} samples have <100 cells")
            print(f"  ⚠ {small_samples} samples have <100 cells")
    else:
        results["warnings"].append(f"Sample column '{sample_column}' not found")
        print(f"  ⚠ Sample column '{sample_column}' not found in obs")
    
    obj_cols = adata.obs.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        print(f"  Object dtype columns: {obj_cols}")
        for col in obj_cols:
            nan_count = adata.obs[col].isna().sum()
            if nan_count > 0:
                print(f"    - '{col}': {nan_count:,} NaN values")
    
    # =========================================================================
    # 5. VARIABLE (GENE) METADATA CHECKS
    # =========================================================================
    print("\n[5/8] GENE METADATA CHECKS")
    print("-" * 40)
    
    gene_names = adata.var_names
    print(f"  Gene name dtype: {gene_names.dtype}")
    
    n_dups = gene_names.duplicated().sum()
    if n_dups > 0:
        results["errors"].append(f"{n_dups} duplicate gene names")
        print(f"  ❌ {n_dups} duplicate gene names found")
    else:
        print(f"  ✓ No duplicate gene names")
        results["passed"].append("No duplicate genes")
    
    empty_names = (gene_names == '').sum() + gene_names.isna().sum()
    if empty_names > 0:
        results["errors"].append(f"{empty_names} empty/NaN gene names")
        print(f"  ❌ {empty_names} empty/NaN gene names")
    else:
        print(f"  ✓ No empty gene names")
    
    mt_genes = gene_names.str.startswith('MT-').sum()
    print(f"  Mitochondrial genes (MT-): {mt_genes}")
    
    # =========================================================================
    # 6. MEMORY ESTIMATION
    # =========================================================================
    print("\n[6/8] MEMORY ESTIMATION")
    print("-" * 40)
    
    n_cells, n_genes = adata.shape
    
    if issparse(X):
        nnz = X.nnz
        sparse_mem_gb = (nnz * 4 + nnz * 4 + (n_cells + 1) * 4) / (1024**3)
        print(f"  Sparse matrix memory (float32): ~{sparse_mem_gb:.2f} GB")
    
    dense_mem_gb = (n_cells * n_genes * 4) / (1024**3)
    print(f"  Dense matrix memory (float32): ~{dense_mem_gb:.2f} GB")
    
    hvg_overhead_gb = (n_cells * 2000 * 8) / (1024**3)
    print(f"  Estimated HVG overhead: ~{hvg_overhead_gb:.2f} GB")
    
    total_estimate = (sparse_mem_gb if issparse(X) else dense_mem_gb) + hvg_overhead_gb
    print(f"  Estimated total GPU memory needed: ~{total_estimate:.2f} GB")
    
    if total_estimate > 40:
        results["warnings"].append(f"High memory estimate: {total_estimate:.1f} GB")
        print(f"  ⚠ May exceed GPU memory on most devices")
    
    # =========================================================================
    # 7. EXTERNAL METADATA CHECKS
    # =========================================================================
    print("\n[7/8] EXTERNAL METADATA CHECKS")
    print("-" * 40)
    
    if sample_meta_path and os.path.exists(sample_meta_path):
        try:
            sample_meta = pd.read_csv(sample_meta_path)
            print(f"  ✓ Sample metadata loaded: {sample_meta.shape}")
            print(f"    Columns: {list(sample_meta.columns)}")
            
            if sample_column in sample_meta.columns:
                meta_samples = set(sample_meta[sample_column].unique())
                data_samples = set(adata.obs[sample_column].unique()) if sample_column in adata.obs.columns else set()
                
                missing_in_meta = data_samples - meta_samples
                if missing_in_meta:
                    results["warnings"].append(f"{len(missing_in_meta)} samples in data not in metadata")
                    print(f"  ⚠ {len(missing_in_meta)} samples in data not found in metadata")
        except Exception as e:
            results["warnings"].append(f"Could not load sample metadata: {e}")
            print(f"  ⚠ Could not load sample metadata: {e}")
    else:
        print(f"  No sample metadata path provided or file not found")
    
    if cell_meta_path and os.path.exists(cell_meta_path):
        try:
            cell_meta = pd.read_csv(cell_meta_path)
            print(f"  ✓ Cell metadata loaded: {cell_meta.shape}")
        except Exception as e:
            results["warnings"].append(f"Could not load cell metadata: {e}")
            print(f"  ⚠ Could not load cell metadata: {e}")
    else:
        print(f"  No cell metadata path provided or file not found")
    
    # =========================================================================
    # 8. SUMMARY
    # =========================================================================
    print("\n[8/8] INSPECTION SUMMARY")
    print("=" * 70)
    print(f"  ✓ Passed: {len(results['passed'])}")
    print(f"  ⚠ Warnings: {len(results['warnings'])}")
    print(f"  ❌ Errors: {len(results['errors'])}")
    
    if results['errors']:
        print("\n  ERRORS (must fix):")
        for e in results['errors']:
            print(f"    - {e}")
    
    if results['warnings']:
        print("\n  WARNINGS (review):")
        for w in results['warnings']:
            print(f"    - {w}")
    
    return results, adata


def gpu_memory_stress_test(adata, sample_column='sample', fractions=None, num_features=2000):
    """Incrementally test GPU memory by running HVG on increasing data subsets."""
    if fractions is None:
        fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    
    print("\n" + "=" * 70)
    print("GPU MEMORY STRESS TEST")
    print("=" * 70)
    
    try:
        import cupy as cp
        import rapids_singlecell as rsc
        from rmm.allocators.cupy import rmm_cupy_allocator
        import rmm
    except ImportError as e:
        print(f"❌ GPU libraries not available: {e}")
        return None
    
    try:
        rmm.reinitialize(managed_memory=True)
        cp.cuda.set_allocator(rmm_cupy_allocator)
        print("✓ RMM initialized with managed memory")
    except Exception as e:
        print(f"⚠ RMM initialization warning: {e}")
    
    try:
        device = cp.cuda.Device()
        total_mem = device.mem_info[1] / (1024**3)
        print(f"GPU: {device.id}, Total memory: {total_mem:.1f} GB")
    except:
        print("Could not get GPU info")
    
    results = []
    n_cells = adata.n_obs
    
    print("\nPreparing data for GPU test...")
    if issparse(adata.X):
        if adata.X.format != 'csr':
            print("  Converting to CSR format...")
            adata.X = csr_matrix(adata.X)
    
    if adata.X.dtype != np.float32:
        print(f"  Converting from {adata.X.dtype} to float32...")
        if issparse(adata.X):
            adata.X = adata.X.astype(np.float32)
        else:
            adata.X = np.asarray(adata.X, dtype=np.float32)
    
    print(f"\nTesting fractions: {fractions}")
    print("-" * 70)
    
    for frac in fractions:
        n_subset = max(100, int(n_cells * frac))
        if n_subset > n_cells:
            n_subset = n_cells
        
        print(f"\n[{frac*100:.0f}%] Testing with {n_subset:,} cells...")
        
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        
        try:
            mem_before = cp.cuda.Device().mem_info
            free_before = mem_before[0] / (1024**3)
            
            idx = np.random.choice(n_cells, n_subset, replace=False)
            adata_sub = adata[idx].copy()
            
            rsc.get.anndata_to_GPU(adata_sub)
            
            mem_after_transfer = cp.cuda.Device().mem_info
            free_after_transfer = mem_after_transfer[0] / (1024**3)
            
            print(f"  GPU memory after transfer: {free_before - free_after_transfer:.2f} GB used")
            
            import time
            start = time.time()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rsc.pp.highly_variable_genes(
                    adata_sub,
                    n_top_genes=num_features,
                    flavor='seurat_v3',
                    batch_key=sample_column if sample_column in adata_sub.obs.columns else None
                )
            
            elapsed = time.time() - start
            
            mem_after_hvg = cp.cuda.Device().mem_info
            free_after_hvg = mem_after_hvg[0] / (1024**3)
            peak_usage = free_before - free_after_hvg
            
            print(f"  ✓ HVG completed in {elapsed:.1f}s")
            print(f"  Peak GPU memory used: ~{peak_usage:.2f} GB")
            
            results.append({
                'fraction': frac,
                'n_cells': n_subset,
                'success': True,
                'time_sec': elapsed,
                'peak_mem_gb': peak_usage,
                'error': None
            })
            
            del adata_sub
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ FAILED: {error_msg[:100]}...")
            
            results.append({
                'fraction': frac,
                'n_cells': n_subset,
                'success': False,
                'time_sec': None,
                'peak_mem_gb': None,
                'error': error_msg
            })
            
            if 'MemoryError' in error_msg or 'CUDA' in error_msg:
                print(f"\n  ⚠ Memory error detected - stopping test")
                break
        
        finally:
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
    
    # Summary
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)
    
    df_results = pd.DataFrame(results)
    
    successful = df_results[df_results['success']]
    failed = df_results[~df_results['success']]
    
    if len(successful) > 0:
        max_success = successful.iloc[-1]
        print(f"\n✓ Maximum successful: {max_success['fraction']*100:.0f}% ({max_success['n_cells']:,} cells)")
        print(f"  Time: {max_success['time_sec']:.1f}s, Peak memory: {max_success['peak_mem_gb']:.2f} GB")
    
    if len(failed) > 0:
        first_fail = failed.iloc[0]
        print(f"\n❌ First failure: {first_fail['fraction']*100:.0f}% ({first_fail['n_cells']:,} cells)")
        print(f"  Error: {first_fail['error'][:200]}")
    
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS:")
    
    if len(failed) == 0:
        print("  ✓ All tests passed - your data should work on GPU")
    elif len(successful) == 0:
        print("  ❌ All tests failed - likely GPU/driver issue or data corruption")
        print("  Try: 1) Check nvidia-smi for GPU health")
        print("       2) Restart Python/clear GPU memory")
        print("       3) Check data for corruption (NaN/Inf)")
    else:
        max_cells = successful['n_cells'].max()
        print(f"  ⚠ Data too large for GPU memory")
        print(f"  Maximum cells that work: ~{max_cells:,}")
        print(f"  Options:")
        print(f"    1) Subsample to {max_cells:,} cells before GPU processing")
        print(f"    2) Process in batches by sample")
        print(f"    3) Use CPU-based HVG (scanpy) instead of GPU")
        print(f"    4) Use a GPU with more memory")
    
    return df_results


def check_cuda_health():
    """Quick CUDA health check."""
    print("\n" + "=" * 70)
    print("CUDA HEALTH CHECK")
    print("=" * 70)
    
    try:
        import cupy as cp
        
        print("\n[1] Basic allocation test...")
        try:
            x = cp.zeros(1000)
            del x
            print("  ✓ Basic allocation works")
        except Exception as e:
            print(f"  ❌ Basic allocation failed: {e}")
            return False
        
        print("\n[2] GPU memory info...")
        try:
            device = cp.cuda.Device()
            free, total = device.mem_info
            print(f"  Device: {device.id}")
            print(f"  Total: {total/(1024**3):.1f} GB")
            print(f"  Free: {free/(1024**3):.1f} GB")
            print(f"  Used: {(total-free)/(1024**3):.1f} GB")
        except Exception as e:
            print(f"  ⚠ Could not get memory info: {e}")
        
        print("\n[3] Simple compute test...")
        try:
            a = cp.random.rand(1000, 1000, dtype=cp.float32)
            b = cp.matmul(a, a.T)
            result = float(cp.mean(b))
            del a, b
            print(f"  ✓ Compute test passed (result: {result:.4f})")
        except Exception as e:
            print(f"  ❌ Compute test failed: {e}")
            return False
        
        print("\n[4] Synchronization test...")
        try:
            cp.cuda.Stream.null.synchronize()
            print("  ✓ Synchronization works")
        except Exception as e:
            print(f"  ❌ Synchronization failed: {e}")
            return False
        
        print("\n✓ CUDA appears healthy")
        return True
        
    except ImportError:
        print("❌ CuPy not installed")
        return False
    except Exception as e:
        print(f"❌ CUDA health check failed: {e}")
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# FULL DATA DIAGNOSTIC FOR GPU SINGLE-CELL PROCESSING")
    print("#" * 70)
    print(f"\nInput file: {H5AD_PATH}")
    print(f"Sample column: {SAMPLE_COLUMN}")
    
    # 1. CUDA health check
    cuda_ok = check_cuda_health()
    
    if not cuda_ok:
        print("\n⚠ CUDA issues detected - GPU tests may fail")
    
    # 2. Data inspection
    inspection_results, adata = inspect_h5ad(
        H5AD_PATH,
        sample_column=SAMPLE_COLUMN,
        sample_meta_path=SAMPLE_META_PATH,
        cell_meta_path=CELL_META_PATH
    )
    
    if inspection_results['errors']:
        print("\n❌ Data has errors - fix before GPU testing")
    elif RUN_GPU_TEST and cuda_ok and adata is not None:
        # 3. GPU stress test
        stress_results = gpu_memory_stress_test(
            adata,
            sample_column=SAMPLE_COLUMN,
            fractions=TEST_FRACTIONS,
            num_features=NUM_FEATURES
        )
    
    print("\n" + "#" * 70)
    print("# DIAGNOSTIC COMPLETE")
    print("#" * 70)