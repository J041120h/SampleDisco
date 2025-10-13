import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse, stats
import os
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import gc
from tqdm import tqdm
import anndata as ad


def safe_load_batch_from_hdf5(
    f: h5py.File,
    indices: np.ndarray,
    n_features: int,
    verbose: bool = False
) -> np.ndarray:
    """
    Safely load a batch of data from HDF5, handling both sparse and dense formats.
    """
    if 'X' not in f:
        raise ValueError("No 'X' matrix found in HDF5 file")
    
    X_data = f['X']
    
    # Check if this is a sparse matrix (stored as a group with data/indices/indptr)
    is_sparse = isinstance(X_data, h5py.Group)
    
    if is_sparse:
        # CSR sparse matrix format
        if 'data' not in X_data or 'indices' not in X_data or 'indptr' not in X_data:
            raise ValueError("Sparse matrix missing required components")
        
        # Load only the required rows from sparse matrix
        data_arr = X_data['data']
        indices_arr = X_data['indices']
        indptr_arr = X_data['indptr']
        
        # Initialize output
        batch_data = np.zeros((len(indices), n_features), dtype=np.float32)
        
        # Load each row separately to avoid memory issues
        for i, idx in enumerate(indices):
            start = indptr_arr[idx]
            end = indptr_arr[idx + 1]
            
            if end > start:
                col_indices = indices_arr[start:end]
                values = data_arr[start:end]
                batch_data[i, col_indices] = values
        
        return batch_data
    
    else:
        # Dense matrix
        if not isinstance(X_data, h5py.Dataset):
            raise ValueError("X is neither a sparse matrix group nor a dense dataset")
        
        # Check shape
        if len(X_data.shape) != 2:
            raise ValueError(f"Expected 2D matrix, got shape {X_data.shape}")
        
        # For dense matrices, we need to load rows one by one or in small chunks
        # because fancy indexing with arrays may not work well
        batch_data = np.zeros((len(indices), n_features), dtype=np.float32)
        
        # Process in small chunks to avoid memory issues
        chunk_size = min(10, len(indices))  # Process 10 rows at a time
        
        for i in range(0, len(indices), chunk_size):
            chunk_end = min(i + chunk_size, len(indices))
            chunk_indices = indices[i:chunk_end]
            
            # Load each row individually if chunk loading fails
            try:
                # Try to load chunk at once
                for j, idx in enumerate(chunk_indices):
                    batch_data[i + j, :] = X_data[idx, :]
            except:
                # Fallback to loading one row at a time
                if verbose:
                    print(f"Warning: Falling back to single-row loading for indices {i} to {chunk_end}")
                for j, idx in enumerate(chunk_indices):
                    try:
                        batch_data[i + j, :] = X_data[idx, :]
                    except Exception as e:
                        print(f"Error loading row {idx}: {e}")
                        # Keep as zeros if we can't load
        
        return batch_data


def compute_metrics_direct_hdf5_robust(
    integrated_path: str,
    output_dir: str = "./validation_results",
    batch_size: int = 100,
    verbose: bool = True,
    subsample_ratio: float = 1.0,
    max_memory_gb: float = 8.0  # Maximum memory to use for sparse matrix
) -> Dict:
    """
    Robust HDF5 processing that handles both sparse and dense matrices safely.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("=" * 80)
        print("DIRECT HDF5 PROCESSING (Robust Version)")
        print("=" * 80)
        print(f"\n📂 Processing: {integrated_path}")
        print(f"   Batch size: {batch_size}")
        print(f"   Subsample ratio: {subsample_ratio}")
        print(f"   Max memory: {max_memory_gb:.1f} GB")
    
    # Step 1: Get metadata to find paired cells
    if verbose:
        print("\n📊 Loading metadata...")
    
    # Use AnnData just to get metadata
    try:
        adata_meta = ad.read_h5ad(integrated_path, backed='r')
        obs_df = adata_meta.obs.copy()
        var_names = adata_meta.var_names.copy()
        n_genes = len(var_names)
        
        # Get matrix shape
        matrix_shape = adata_meta.shape
        
        # Close the metadata object
        del adata_meta
        gc.collect()
        
    except Exception as e:
        print(f"Error reading metadata: {e}")
        raise
    
    # Find paired cells
    rna_mask = obs_df['modality'] == 'RNA'
    atac_mask = obs_df['modality'] == 'ATAC'
    
    rna_indices = np.where(rna_mask)[0]
    atac_indices = np.where(atac_mask)[0]
    
    if verbose:
        print(f"   Matrix shape: {matrix_shape}")
        print(f"   RNA cells: {len(rna_indices)}")
        print(f"   ATAC cells: {len(atac_indices)}")
    
    # Match paired cells
    rna_obs = obs_df[rna_mask]
    atac_obs = obs_df[atac_mask]
    
    # Try original barcode matching
    if 'original_barcode' in obs_df.columns:
        rna_barcodes = rna_obs['original_barcode'].values
        atac_barcodes = atac_obs['original_barcode'].values
    else:
        # Use index-based matching
        rna_barcodes = np.array([idx.replace('_RNA', '') for idx in rna_obs.index])
        atac_barcodes = np.array([idx.replace('_ATAC', '') for idx in atac_obs.index])
    
    # Find common barcodes
    common_barcodes = list(set(rna_barcodes) & set(atac_barcodes))
    
    if len(common_barcodes) == 0:
        rna_barcodes = np.array([idx.replace('_RNA', '') for idx in rna_obs.index])
        atac_barcodes = np.array([idx.replace('_ATAC', '') for idx in atac_obs.index])
        common_barcodes = list(set(rna_barcodes) & set(atac_barcodes))
    
    n_paired = len(common_barcodes)
    if verbose:
        print(f"\n🔗 Found {n_paired} paired cells")
    
    if n_paired == 0:
        raise ValueError("No paired cells found!")
    
    # Subsample if requested
    if subsample_ratio < 1.0:
        n_subsample = max(1, int(n_paired * subsample_ratio))
        np.random.seed(42)  # For reproducibility
        common_barcodes = list(np.random.choice(common_barcodes, n_subsample, replace=False))
        n_paired = len(common_barcodes)
        if verbose:
            print(f"   Subsampled to {n_paired} cells")
    
    # Create mapping
    rna_bc_to_idx = {bc: i for i, bc in enumerate(rna_barcodes)}
    atac_bc_to_idx = {bc: i for i, bc in enumerate(atac_barcodes)}
    
    # Get paired indices
    paired_rna_local = np.array([rna_bc_to_idx[bc] for bc in common_barcodes])
    paired_atac_local = np.array([atac_bc_to_idx[bc] for bc in common_barcodes])
    
    paired_rna_global = rna_indices[paired_rna_local]
    paired_atac_global = atac_indices[paired_atac_local]
    
    # Step 2: Detect data format
    if verbose:
        print(f"\n🔍 Detecting data format...")
    
    with h5py.File(integrated_path, 'r') as f:
        if 'X' not in f:
            raise ValueError("No 'X' matrix found in HDF5 file")
        
        X_data = f['X']
        is_sparse = isinstance(X_data, h5py.Group)
        
        if is_sparse:
            if verbose:
                print("   Data format: Sparse CSR matrix")
                # Check sparse matrix size
                data_size = X_data['data'].size
                dtype_size = X_data['data'].dtype.itemsize
                estimated_gb = (data_size * dtype_size * 3) / (1024**3)  # Rough estimate
                print(f"   Estimated sparse matrix size: {estimated_gb:.2f} GB")
        else:
            if verbose:
                print("   Data format: Dense matrix")
                print(f"   Matrix dtype: {X_data.dtype}")
                print(f"   Matrix shape: {X_data.shape}")
    
    # Step 3: Process data in batches
    if verbose:
        print(f"\n📈 Processing paired cells in batches...")
    
    # Initialize storage
    per_cell_corr = []
    gene_sums_rna = np.zeros(n_genes, dtype=np.float64)
    gene_sums_atac = np.zeros(n_genes, dtype=np.float64)
    gene_sums_sq_rna = np.zeros(n_genes, dtype=np.float64)
    gene_sums_sq_atac = np.zeros(n_genes, dtype=np.float64)
    gene_counts = np.zeros(n_genes, dtype=np.int64)
    
    n_processed = 0
    
    # Process in batches
    with h5py.File(integrated_path, 'r') as f:
        for batch_start in tqdm(range(0, n_paired, batch_size), 
                               desc="Processing cells", disable=not verbose):
            batch_end = min(batch_start + batch_size, n_paired)
            
            rna_idx_batch = paired_rna_global[batch_start:batch_end]
            atac_idx_batch = paired_atac_global[batch_start:batch_end]
            
            try:
                # Load batch data safely
                rna_batch = safe_load_batch_from_hdf5(f, rna_idx_batch, n_genes, verbose=False)
                atac_batch = safe_load_batch_from_hdf5(f, atac_idx_batch, n_genes, verbose=False)
                
                # Compute per-cell correlations
                for i in range(len(rna_idx_batch)):
                    rna_vec = rna_batch[i, :]
                    atac_vec = atac_batch[i, :]
                    
                    # Only correlate non-zero values
                    mask = (rna_vec != 0) | (atac_vec != 0)
                    if mask.sum() >= 10:  # Need at least 10 non-zero values
                        try:
                            corr, _ = stats.pearsonr(rna_vec[mask], atac_vec[mask])
                            if not np.isnan(corr) and not np.isinf(corr):
                                per_cell_corr.append(corr)
                        except:
                            pass
                
                # Update gene statistics
                # Use float64 for accumulation to avoid overflow
                gene_sums_rna += np.sum(rna_batch, axis=0, dtype=np.float64)
                gene_sums_atac += np.sum(atac_batch, axis=0, dtype=np.float64)
                gene_sums_sq_rna += np.sum(rna_batch.astype(np.float64) ** 2, axis=0)
                gene_sums_sq_atac += np.sum(atac_batch.astype(np.float64) ** 2, axis=0)
                gene_counts += len(rna_idx_batch)
                
                n_processed += len(rna_idx_batch)
                
                # Clear batch data
                del rna_batch, atac_batch
                gc.collect()
                
            except Exception as e:
                print(f"\nError processing batch {batch_start}-{batch_end}: {e}")
                continue
    
    # Compute gene statistics
    with np.errstate(divide='ignore', invalid='ignore'):
        gene_means_rna = np.where(gene_counts > 0, gene_sums_rna / gene_counts, 0)
        gene_means_atac = np.where(gene_counts > 0, gene_sums_atac / gene_counts, 0)
        gene_vars_rna = np.where(gene_counts > 0, 
                                 (gene_sums_sq_rna / gene_counts) - (gene_means_rna ** 2), 
                                 0)
        gene_vars_atac = np.where(gene_counts > 0,
                                  (gene_sums_sq_atac / gene_counts) - (gene_means_atac ** 2),
                                  0)
        gene_stds_rna = np.sqrt(np.maximum(gene_vars_rna, 0))
        gene_stds_atac = np.sqrt(np.maximum(gene_vars_atac, 0))
    
    per_cell_corr = np.array(per_cell_corr)
    
    if verbose:
        print(f"\n✅ Processed {n_processed} paired cells")
        print(f"   Computed {len(per_cell_corr)} cell correlations")
    
    # Step 4: Compute per-gene correlations (sampling)
    if verbose:
        print(f"\n📈 Computing per-gene correlations (sampling)...")
    
    n_sample_genes = min(1000, n_genes)
    np.random.seed(42)
    sample_gene_idx = np.random.choice(n_genes, n_sample_genes, replace=False)
    
    gene_correlations = np.full(n_genes, np.nan)
    
    with h5py.File(integrated_path, 'r') as f:
        # Process genes in small batches to avoid memory issues
        gene_batch_size = 50
        
        for gb_start in tqdm(range(0, len(sample_gene_idx), gene_batch_size),
                            desc="Gene correlations", disable=not verbose):
            gb_end = min(gb_start + gene_batch_size, len(sample_gene_idx))
            gene_batch = sample_gene_idx[gb_start:gb_end]
            
            # Load all paired cells for these genes
            try:
                rna_data = safe_load_batch_from_hdf5(f, paired_rna_global, n_genes)
                atac_data = safe_load_batch_from_hdf5(f, paired_atac_global, n_genes)
                
                for gene_idx in gene_batch:
                    rna_vals = rna_data[:, gene_idx]
                    atac_vals = atac_data[:, gene_idx]
                    
                    if np.std(rna_vals) > 0 and np.std(atac_vals) > 0:
                        try:
                            corr, _ = stats.pearsonr(rna_vals, atac_vals)
                            if not np.isnan(corr) and not np.isinf(corr):
                                gene_correlations[gene_idx] = corr
                        except:
                            pass
                
                del rna_data, atac_data
                gc.collect()
                
            except Exception as e:
                print(f"\nError processing gene batch: {e}")
                continue
    
    # Create results
    gene_results = pd.DataFrame({
        'gene': var_names,
        'pearson_corr': gene_correlations,
        'mean_rna': gene_means_rna,
        'mean_atac': gene_means_atac,
        'std_rna': gene_stds_rna,
        'std_atac': gene_stds_atac
    })
    
    # Visualization
    if verbose:
        print("\n🎨 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cell correlations
    if len(per_cell_corr) > 0:
        axes[0, 0].hist(per_cell_corr, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(per_cell_corr.mean(), color='red', linestyle='--',
                           label=f'Mean: {per_cell_corr.mean():.3f}')
        axes[0, 0].set_xlabel('Pearson Correlation')
        axes[0, 0].set_ylabel('Number of Cells')
        axes[0, 0].set_title('Per-Cell Correlations')
        axes[0, 0].legend()
    
    # Gene correlations
    valid_gene_corr = gene_correlations[~np.isnan(gene_correlations)]
    if len(valid_gene_corr) > 0:
        axes[0, 1].hist(valid_gene_corr, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].axvline(valid_gene_corr.mean(), color='red', linestyle='--',
                           label=f'Mean: {valid_gene_corr.mean():.3f}')
        axes[0, 1].set_xlabel('Pearson Correlation')
        axes[0, 1].set_ylabel('Number of Genes')
        axes[0, 1].set_title(f'Gene Correlations (n={len(valid_gene_corr)} sampled)')
        axes[0, 1].legend()
    
    # Mean expression
    # Filter out zeros for better visualization
    mask_nonzero = (gene_means_rna > 0) | (gene_means_atac > 0)
    axes[1, 0].scatter(gene_means_rna[mask_nonzero], gene_means_atac[mask_nonzero], 
                      alpha=0.3, s=5)
    if mask_nonzero.any():
        max_val = max(gene_means_rna[mask_nonzero].max(), gene_means_atac[mask_nonzero].max())
        axes[1, 0].plot([0, max_val], [0, max_val], 'r--', label='Identity')
    axes[1, 0].set_xlabel('Mean RNA Expression')
    axes[1, 0].set_ylabel('Mean ATAC Gene Activity')
    axes[1, 0].set_title('Mean Expression Comparison')
    axes[1, 0].legend()
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    
    # Expression vs std
    axes[1, 1].scatter(gene_means_rna[mask_nonzero], gene_stds_rna[mask_nonzero], 
                      alpha=0.3, s=5, label='RNA')
    axes[1, 1].scatter(gene_means_atac[mask_nonzero], gene_stds_atac[mask_nonzero], 
                      alpha=0.3, s=5, label='ATAC')
    axes[1, 1].set_xlabel('Mean Expression')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].set_title('Mean-Variance Relationship')
    axes[1, 1].legend()
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_results_robust.png'), dpi=150)
    plt.close()
    
    # Save results
    if verbose:
        print("\n💾 Saving results...")
    
    if len(per_cell_corr) > 0:
        cell_df = pd.DataFrame({
            'barcode': common_barcodes[:len(per_cell_corr)],
            'correlation': per_cell_corr
        })
        cell_df.to_csv(os.path.join(output_dir, 'cell_correlations_robust.csv'), index=False)
    
    gene_results.to_csv(os.path.join(output_dir, 'gene_metrics_robust.csv'), index=False)
    
    summary = {
        'n_paired_cells': n_paired,
        'n_processed': n_processed,
        'n_genes': n_genes,
        'n_genes_sampled': n_sample_genes,
        'mean_cell_corr': float(per_cell_corr.mean()) if len(per_cell_corr) > 0 else np.nan,
        'median_cell_corr': float(np.median(per_cell_corr)) if len(per_cell_corr) > 0 else np.nan,
        'std_cell_corr': float(per_cell_corr.std()) if len(per_cell_corr) > 0 else np.nan,
        'mean_gene_corr': float(valid_gene_corr.mean()) if len(valid_gene_corr) > 0 else np.nan,
        'subsample_ratio': subsample_ratio
    }
    
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, 'summary_robust.csv'), index=False)
    
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Paired cells: {n_paired}")
        print(f"Processed cells: {n_processed}")
        print(f"Cell correlations: {len(per_cell_corr)}")
        if len(per_cell_corr) > 0:
            print(f"Mean cell correlation: {summary['mean_cell_corr']:.4f}")
        if len(valid_gene_corr) > 0:
            print(f"Mean gene correlation (sampled): {summary['mean_gene_corr']:.4f}")
        print(f"\nResults saved to: {output_dir}")
    
    return summary


if __name__ == "__main__":
    try:
        results = compute_metrics_direct_hdf5_robust(
            integrated_path="/dcs07/hongkai/data/harry/result/all/multiomics/preprocess/atac_rna_integrated.h5ad",
            output_dir="/dcs07/hongkai/data/harry/result/all/multiomics",
            batch_size=100,  # Smaller batch size for safety
            verbose=True,
            subsample_ratio=0.1,
            max_memory_gb=400.0  # Limit memory usage
        )
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()