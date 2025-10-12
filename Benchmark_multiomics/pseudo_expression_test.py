import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse, stats
import os
from typing import Dict
import warnings
warnings.filterwarnings('ignore')
import gc
from tqdm import tqdm
import anndata as ad


def compute_metrics_direct_hdf5(
    integrated_path: str,
    output_dir: str = "./validation_results",
    batch_size: int = 100,
    verbose: bool = True,
    subsample_ratio: float = 1.0  # Use 0.1 for 10% subsample
) -> Dict:
    """
    Direct HDF5 processing to avoid AnnData backed mode issues.
    
    This version reads directly from the HDF5 file to avoid the
    indexing problems with backed sparse matrices.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("=" * 80)
        print("DIRECT HDF5 PROCESSING (Most Robust)")
        print("=" * 80)
        print(f"\n📂 Processing: {integrated_path}")
        print(f"   Batch size: {batch_size}")
        print(f"   Subsample ratio: {subsample_ratio}")
    
    # Step 1: Get metadata to find paired cells
    if verbose:
        print("\n📊 Loading metadata...")
    
    # First get obs data to find paired cells
    with h5py.File(integrated_path, 'r') as f:
        # Check structure
        if verbose:
            print(f"   HDF5 keys: {list(f.keys())}")
    
    # Use AnnData just to get metadata
    adata_meta = ad.read_h5ad(integrated_path, backed='r')
    obs_df = adata_meta.obs.copy()
    var_names = adata_meta.var_names.copy()
    n_genes = len(var_names)
    
    # Find paired cells
    rna_mask = obs_df['modality'] == 'RNA'
    atac_mask = obs_df['modality'] == 'ATAC'
    
    rna_indices = np.where(rna_mask)[0]
    atac_indices = np.where(atac_mask)[0]
    
    if verbose:
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
        # Try alternative
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
        n_subsample = int(n_paired * subsample_ratio)
        common_barcodes = np.random.choice(common_barcodes, n_subsample, replace=False)
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
    
    # Sort for sequential access
    sort_idx = np.argsort(paired_rna_global)
    paired_rna_global = paired_rna_global[sort_idx]
    paired_atac_global = paired_atac_global[sort_idx]
    common_barcodes = [common_barcodes[i] for i in sort_idx]
    
    # Close the metadata object
    del adata_meta
    gc.collect()
    
    # Step 2: Process data in batches using a simpler approach
    if verbose:
        print(f"\n📈 Processing paired cells in batches...")
    
    # Initialize storage
    per_cell_corr = []
    gene_sums_rna = np.zeros(n_genes)
    gene_sums_atac = np.zeros(n_genes)
    gene_sums_sq_rna = np.zeros(n_genes)
    gene_sums_sq_atac = np.zeros(n_genes)
    gene_counts = np.zeros(n_genes)
    
    # Process using direct reading - one batch at a time
    n_processed = 0
    
    # Open the file in read mode
    with h5py.File(integrated_path, 'r') as f:
        # Try to access the data matrix
        if 'X' in f:
            X_data = f['X']
            
            # Check if it's sparse
            if 'data' in X_data and 'indices' in X_data and 'indptr' in X_data:
                # It's a sparse matrix in CSR format
                if verbose:
                    print("   Data is stored as sparse CSR matrix")
                    print("   Loading full sparse matrix (this may take time)...")
                
                # Load the sparse matrix components
                data = X_data['data'][:]
                indices = X_data['indices'][:]
                indptr = X_data['indptr'][:]
                
                # Create sparse matrix
                from scipy.sparse import csr_matrix
                shape = (len(indptr) - 1, n_genes)
                X_sparse = csr_matrix((data, indices, indptr), shape=shape)
                
                # Process in batches
                for batch_start in tqdm(range(0, n_paired, batch_size), 
                                       desc="Processing cells", disable=not verbose):
                    batch_end = min(batch_start + batch_size, n_paired)
                    
                    # Get indices for this batch
                    rna_idx_batch = paired_rna_global[batch_start:batch_end]
                    atac_idx_batch = paired_atac_global[batch_start:batch_end]
                    
                    # Extract data
                    rna_batch = X_sparse[rna_idx_batch, :].toarray()
                    atac_batch = X_sparse[atac_idx_batch, :].toarray()
                    
                    # Compute per-cell correlations
                    for i in range(len(rna_idx_batch)):
                        rna_vec = rna_batch[i, :]
                        atac_vec = atac_batch[i, :]
                        
                        mask = (rna_vec != 0) | (atac_vec != 0)
                        if mask.sum() >= 10:
                            try:
                                corr, _ = stats.pearsonr(rna_vec[mask], atac_vec[mask])
                                if not np.isnan(corr):
                                    per_cell_corr.append(corr)
                            except:
                                pass
                    
                    # Update gene statistics
                    for j in range(n_genes):
                        rna_col = rna_batch[:, j]
                        atac_col = atac_batch[:, j]
                        
                        gene_sums_rna[j] += np.sum(rna_col)
                        gene_sums_atac[j] += np.sum(atac_col)
                        gene_sums_sq_rna[j] += np.sum(rna_col ** 2)
                        gene_sums_sq_atac[j] += np.sum(atac_col ** 2)
                        gene_counts[j] += len(rna_col)
                    
                    n_processed += len(rna_idx_batch)
                
            else:
                # It's a dense matrix
                if verbose:
                    print("   Data is stored as dense matrix")
                    print("   Processing in chunks...")
                
                # Process in row chunks
                for batch_start in tqdm(range(0, n_paired, batch_size), 
                                       desc="Processing cells", disable=not verbose):
                    batch_end = min(batch_start + batch_size, n_paired)
                    
                    # Get indices for this batch
                    rna_idx_batch = paired_rna_global[batch_start:batch_end]
                    atac_idx_batch = paired_atac_global[batch_start:batch_end]
                    
                    # Read data for these rows
                    rna_batch = X_data[rna_idx_batch, :]
                    atac_batch = X_data[atac_idx_batch, :]
                    
                    # Compute per-cell correlations
                    for i in range(len(rna_idx_batch)):
                        rna_vec = rna_batch[i, :]
                        atac_vec = atac_batch[i, :]
                        
                        mask = (rna_vec != 0) | (atac_vec != 0)
                        if mask.sum() >= 10:
                            try:
                                corr, _ = stats.pearsonr(rna_vec[mask], atac_vec[mask])
                                if not np.isnan(corr):
                                    per_cell_corr.append(corr)
                            except:
                                pass
                    
                    # Update gene statistics
                    for j in range(n_genes):
                        rna_col = rna_batch[:, j]
                        atac_col = atac_batch[:, j]
                        
                        gene_sums_rna[j] += np.sum(rna_col)
                        gene_sums_atac[j] += np.sum(atac_col)
                        gene_sums_sq_rna[j] += np.sum(rna_col ** 2)
                        gene_sums_sq_atac[j] += np.sum(atac_col ** 2)
                        gene_counts[j] += len(rna_col)
                    
                    n_processed += len(rna_idx_batch)
        else:
            raise ValueError("Could not find expression matrix in HDF5 file")
    
    # Compute gene statistics
    gene_means_rna = gene_sums_rna / gene_counts
    gene_means_atac = gene_sums_atac / gene_counts
    gene_vars_rna = (gene_sums_sq_rna / gene_counts) - (gene_means_rna ** 2)
    gene_vars_atac = (gene_sums_sq_atac / gene_counts) - (gene_means_atac ** 2)
    gene_stds_rna = np.sqrt(np.maximum(gene_vars_rna, 0))
    gene_stds_atac = np.sqrt(np.maximum(gene_vars_atac, 0))
    
    # Convert to array
    per_cell_corr = np.array(per_cell_corr)
    
    if verbose:
        print(f"\n✅ Processed {n_processed} paired cells")
        print(f"   Computed {len(per_cell_corr)} cell correlations")
    
    # Step 3: Compute per-gene correlations (simplified)
    if verbose:
        print(f"\n📈 Computing per-gene correlations (sampling)...")
    
    # Sample genes for correlation computation
    n_sample_genes = min(1000, n_genes)
    sample_gene_idx = np.random.choice(n_genes, n_sample_genes, replace=False)
    
    gene_correlations = np.full(n_genes, np.nan)
    
    # Reload data for gene correlations (sampling approach)
    with h5py.File(integrated_path, 'r') as f:
        X_data = f['X']
        
        if 'data' in X_data:
            # Sparse
            data = X_data['data'][:]
            indices = X_data['indices'][:]
            indptr = X_data['indptr'][:]
            from scipy.sparse import csr_matrix
            shape = (len(indptr) - 1, n_genes)
            X_sparse = csr_matrix((data, indices, indptr), shape=shape)
            
            for gene_idx in tqdm(sample_gene_idx, desc="Gene correlations", disable=not verbose):
                rna_vals = X_sparse[paired_rna_global, gene_idx].toarray().flatten()
                atac_vals = X_sparse[paired_atac_global, gene_idx].toarray().flatten()
                
                if np.std(rna_vals) > 0 and np.std(atac_vals) > 0:
                    try:
                        corr, _ = stats.pearsonr(rna_vals, atac_vals)
                        gene_correlations[gene_idx] = corr
                    except:
                        pass
        else:
            # Dense
            for gene_idx in tqdm(sample_gene_idx, desc="Gene correlations", disable=not verbose):
                rna_vals = X_data[paired_rna_global, gene_idx]
                atac_vals = X_data[paired_atac_global, gene_idx]
                
                if np.std(rna_vals) > 0 and np.std(atac_vals) > 0:
                    try:
                        corr, _ = stats.pearsonr(rna_vals, atac_vals)
                        gene_correlations[gene_idx] = corr
                    except:
                        pass
    
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
    
    # Gene correlations (sampled)
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
    axes[1, 0].scatter(gene_means_rna, gene_means_atac, alpha=0.3, s=5)
    max_val = max(gene_means_rna.max(), gene_means_atac.max())
    axes[1, 0].plot([0, max_val], [0, max_val], 'r--', label='Identity')
    axes[1, 0].set_xlabel('Mean RNA Expression')
    axes[1, 0].set_ylabel('Mean ATAC Gene Activity')
    axes[1, 0].set_title('Mean Expression Comparison')
    axes[1, 0].legend()
    
    # Expression vs std
    axes[1, 1].scatter(gene_means_rna, gene_stds_rna, alpha=0.3, s=5, label='RNA')
    axes[1, 1].scatter(gene_means_atac, gene_stds_atac, alpha=0.3, s=5, label='ATAC')
    axes[1, 1].set_xlabel('Mean Expression')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].set_title('Mean-Variance Relationship')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_results.png'), dpi=150)
    plt.close()
    
    # Save results
    if verbose:
        print("\n💾 Saving results...")
    
    # Cell results
    cell_df = pd.DataFrame({
        'barcode': common_barcodes[:len(per_cell_corr)],
        'correlation': per_cell_corr
    })
    cell_df.to_csv(os.path.join(output_dir, 'cell_correlations.csv'), index=False)
    
    # Gene results
    gene_results.to_csv(os.path.join(output_dir, 'gene_metrics.csv'), index=False)
    
    # Summary
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
    
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Paired cells: {n_paired}")
        print(f"Processed cells: {n_processed}")
        print(f"Cell correlations: {len(per_cell_corr)}")
        print(f"Mean cell correlation: {summary['mean_cell_corr']:.4f}")
        if len(valid_gene_corr) > 0:
            print(f"Mean gene correlation (sampled): {summary['mean_gene_corr']:.4f}")
        print(f"\nResults saved to: {output_dir}")
    
    return summary


if __name__ == "__main__":
    # Start with a small subsample to test
    results = compute_metrics_direct_hdf5(
        integrated_path="/dcs07/hongkai/data/harry/result/Benchmark/multiomics/preprocess/atac_rna_integrated.h5ad",
        output_dir="/dcs07/hongkai/data/harry/result/Benchmark/pseudo_expression",
        batch_size=100,
        verbose=True,
        subsample_ratio=0.01  # Start with 1% of cells to test
    )
    