#!/usr/bin/env python3
"""
KNN-based aggregation analysis for RNA-ATAC correlation (GPU-accelerated).
Memory-efficient version - NEVER loads full matrix to GPU.
All operations batched.
"""

import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

# Initialize RMM BEFORE importing cupy operations
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

rmm.reinitialize(
    managed_memory=False,
    pool_allocator=True,
)

import cupy as cp
cp.cuda.set_allocator(rmm_cupy_allocator)


def _get_gpu_memory():
    """Get current GPU memory usage."""
    mem_free, mem_total = cp.cuda.Device().mem_info
    return mem_free / 1e9, mem_total / 1e9


def _clear_gpu():
    """Clear GPU memory."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()


def _compute_pca_embedding_gpu_batched(
    X_cpu,  # scipy sparse or numpy array - STAYS ON CPU
    n_pcs: int = 30,
    n_hvgs: int = 2000,
    batch_size: int = 20000,
) -> cp.ndarray:
    """
    Compute PCA embedding with fully batched approach.
    Never loads full matrix to GPU.
    """
    # Convert sparse to dense on CPU if needed
    if sparse.issparse(X_cpu):
        X_cpu = X_cpu.toarray()
    X_cpu = np.asarray(X_cpu, dtype=np.float32)
    
    n_cells, n_genes = X_cpu.shape
    print(f"    [PCA] Matrix size: {n_cells} x {n_genes}")
    
    # --- Pass 1: Compute row sums and gene statistics ---
    print("    [PCA] Pass 1: Computing gene statistics...")
    gene_sums = np.zeros(n_genes, dtype=np.float64)
    gene_sq_sums = np.zeros(n_genes, dtype=np.float64)
    
    for start in tqdm(range(0, n_cells, batch_size), desc="    [PCA] Stats", leave=False):
        end = min(start + batch_size, n_cells)
        batch = X_cpu[start:end].astype(np.float32)
        
        # Normalize on CPU
        row_sums = batch.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        batch = batch / row_sums * 1e4
        np.log1p(batch, out=batch)
        
        gene_sums += batch.sum(axis=0).astype(np.float64)
        gene_sq_sums += (batch ** 2).sum(axis=0).astype(np.float64)
    
    gene_means = gene_sums / n_cells
    gene_vars = gene_sq_sums / n_cells - gene_means ** 2
    
    # Select HVGs
    n_hvgs = min(n_hvgs, n_genes)
    hvg_indices = np.argsort(gene_vars)[-n_hvgs:]
    hvg_means = gene_means[hvg_indices].astype(np.float32)
    hvg_stds = np.sqrt(gene_vars[hvg_indices]).astype(np.float32)
    hvg_stds = np.where(hvg_stds == 0, 1, hvg_stds)
    
    print(f"    [PCA] Selected {n_hvgs} HVGs")
    
    # --- Pass 2: Compute covariance matrix ---
    print("    [PCA] Pass 2: Computing covariance matrix...")
    n_hvgs_actual = len(hvg_indices)
    cov_matrix = np.zeros((n_hvgs_actual, n_hvgs_actual), dtype=np.float64)
    
    for start in tqdm(range(0, n_cells, batch_size), desc="    [PCA] Cov", leave=False):
        end = min(start + batch_size, n_cells)
        batch = X_cpu[start:end].astype(np.float32)
        
        # Normalize
        row_sums = batch.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        batch = batch / row_sums * 1e4
        np.log1p(batch, out=batch)
        
        # Select HVGs and standardize
        batch = batch[:, hvg_indices]
        batch = (batch - hvg_means) / hvg_stds
        
        # Accumulate covariance on GPU (small matrix multiply)
        batch_gpu = cp.asarray(batch, dtype=cp.float32)
        cov_gpu = batch_gpu.T @ batch_gpu
        cov_matrix += cp.asnumpy(cov_gpu).astype(np.float64)
        
        del batch_gpu, cov_gpu
        _clear_gpu()
    
    cov_matrix /= n_cells
    
    # --- Eigendecomposition on GPU (small matrix) ---
    print("    [PCA] Computing eigendecomposition...")
    cov_gpu = cp.asarray(cov_matrix, dtype=cp.float32)
    eigenvalues, eigenvectors = cp.linalg.eigh(cov_gpu)
    
    # Sort descending and select top components
    n_components = min(n_pcs, n_hvgs_actual - 1)
    idx = cp.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx[:n_components]]
    eigenvectors_cpu = cp.asnumpy(eigenvectors).astype(np.float32)
    
    del cov_gpu, eigenvalues, eigenvectors
    _clear_gpu()
    
    # --- Pass 3: Project data onto PCs ---
    print("    [PCA] Pass 3: Projecting onto PCs...")
    pca_coords = np.zeros((n_cells, n_components), dtype=np.float32)
    
    for start in tqdm(range(0, n_cells, batch_size), desc="    [PCA] Project", leave=False):
        end = min(start + batch_size, n_cells)
        batch = X_cpu[start:end].astype(np.float32)
        
        # Normalize
        row_sums = batch.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        batch = batch / row_sums * 1e4
        np.log1p(batch, out=batch)
        
        # Select HVGs and standardize
        batch = batch[:, hvg_indices]
        batch = (batch - hvg_means) / hvg_stds
        
        # Project on GPU
        batch_gpu = cp.asarray(batch, dtype=cp.float32)
        eigenvectors_gpu = cp.asarray(eigenvectors_cpu, dtype=cp.float32)
        proj = batch_gpu @ eigenvectors_gpu
        pca_coords[start:end] = cp.asnumpy(proj)
        
        del batch_gpu, eigenvectors_gpu, proj
        _clear_gpu()
    
    print("    [PCA] Complete.")
    return cp.asarray(pca_coords, dtype=cp.float32)


def _compute_knn_gpu_batched(
    pca_coords: cp.ndarray,  # This is small enough to fit on GPU
    k: int,
    batch_size: int = 5000,
) -> cp.ndarray:
    """Compute KNN indices on GPU using batched distance computation."""
    n_samples = pca_coords.shape[0]
    knn_indices = cp.zeros((n_samples, k), dtype=cp.int64)
    
    # Precompute norms (small - just n_cells floats)
    X_norm_sq = (pca_coords ** 2).sum(axis=1)
    
    for start in tqdm(range(0, n_samples, batch_size), desc="    [KNN] Building graph", leave=False):
        end = min(start + batch_size, n_samples)
        batch = pca_coords[start:end]
        batch_norm_sq = X_norm_sq[start:end]
        
        # Squared distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        dists_sq = batch_norm_sq[:, None] + X_norm_sq[None, :] - 2 * (batch @ pca_coords.T)
        
        # Get k smallest
        if k < n_samples:
            partition_idx = cp.argpartition(dists_sq, k, axis=1)[:, :k]
            batch_indices = cp.arange(end - start)[:, None]
            k_dists = dists_sq[batch_indices, partition_idx]
            sort_idx = cp.argsort(k_dists, axis=1)
            knn_indices[start:end] = partition_idx[batch_indices, sort_idx]
        else:
            knn_indices[start:end] = cp.argsort(dists_sq, axis=1)[:, :k]
        
        del dists_sq
        _clear_gpu()
    
    return knn_indices


def _aggregate_and_correlate_batched(
    rna_X_cpu: np.ndarray,  # Normalized, on CPU
    atac_X_cpu: np.ndarray,  # Normalized, on CPU
    knn_indices_cpu: np.ndarray,  # On CPU
    k: int,
    min_nonzero: int = 10,
    batch_size: int = 5000,
) -> tuple:
    """
    Aggregate and compute correlation in batches.
    Everything stays on CPU except small batch computations.
    """
    n_cells = rna_X_cpu.shape[0]
    knn_k = knn_indices_cpu[:, :k]  # (n_cells, k)
    
    all_corrs = []
    rna_zero_count = 0
    atac_zero_count = 0
    total_elements = 0
    
    for start in tqdm(range(0, n_cells, batch_size), desc=f"    [k={k}] Correlating", leave=False):
        end = min(start + batch_size, n_cells)
        
        # Get neighbor indices for this batch
        batch_knn = knn_k[start:end]  # (batch, k)
        
        # Gather neighbors on CPU: (batch, k, n_genes)
        rna_neighbors = rna_X_cpu[batch_knn]
        atac_neighbors = atac_X_cpu[batch_knn]
        
        # Aggregate on CPU (mean over k neighbors)
        rna_agg = rna_neighbors.mean(axis=1).astype(np.float32)  # (batch, n_genes)
        atac_agg = atac_neighbors.mean(axis=1).astype(np.float32)
        
        del rna_neighbors, atac_neighbors
        
        # Track sparsity
        rna_zero_count += (rna_agg == 0).sum()
        atac_zero_count += (atac_agg == 0).sum()
        total_elements += rna_agg.size
        
        # Transfer to GPU for correlation
        rna_gpu = cp.asarray(rna_agg, dtype=cp.float64)
        atac_gpu = cp.asarray(atac_agg, dtype=cp.float64)
        
        del rna_agg, atac_agg
        
        # Mask for valid genes
        mask = (rna_gpu != 0) | (atac_gpu != 0)
        n_valid_genes = mask.sum(axis=1)
        
        # Masked values
        rna_masked = cp.where(mask, rna_gpu, 0.0)
        atac_masked = cp.where(mask, atac_gpu, 0.0)
        
        # Means
        n_valid_safe = cp.maximum(n_valid_genes, 1).astype(cp.float64)
        rna_mean = rna_masked.sum(axis=1) / n_valid_safe
        atac_mean = atac_masked.sum(axis=1) / n_valid_safe
        
        # Center
        rna_centered = cp.where(mask, rna_gpu - rna_mean[:, None], 0.0)
        atac_centered = cp.where(mask, atac_gpu - atac_mean[:, None], 0.0)
        
        # Correlation components
        cov = (rna_centered * atac_centered).sum(axis=1)
        rna_var = (rna_centered ** 2).sum(axis=1)
        atac_var = (atac_centered ** 2).sum(axis=1)
        denom = cp.sqrt(rna_var * atac_var)
        
        # Compute correlations
        batch_corrs = cp.full(end - start, cp.nan, dtype=cp.float64)
        valid = (n_valid_genes >= min_nonzero) & (denom > 1e-10)
        batch_corrs[valid] = cov[valid] / denom[valid]
        
        all_corrs.append(cp.asnumpy(batch_corrs))
        
        del rna_gpu, atac_gpu, mask, rna_masked, atac_masked
        del rna_centered, atac_centered, cov, rna_var, atac_var, denom, batch_corrs
        _clear_gpu()
    
    corrs = np.concatenate(all_corrs)
    rna_sparsity = rna_zero_count / total_elements * 100
    atac_sparsity = atac_zero_count / total_elements * 100
    
    return corrs, rna_sparsity, atac_sparsity


def analyze_knn_aggregation_correlation_gpu(
    adata_rna: sc.AnnData,
    adata_atac: sc.AnnData,
    output_dir: str,
    k_values: Optional[List[int]] = None,
    n_pcs: int = 30,
    n_hvgs: int = 2000,
    corr_method: str = "pearson",
    min_nonzero: int = 10,
    use_rna_for_knn: bool = True,
    atac_layer: Optional[str] = "GeneActivity",
    sample_cells: Optional[int] = None,
    random_state: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Analyze how RNA-ATAC correlation changes with KNN-based cell aggregation.
    Fully memory-efficient GPU implementation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 50, 100, 200, 500]
    
    # -------------------------
    # 1. Align cells
    # -------------------------
    if verbose:
        print("[KNN-Agg-GPU] Aligning cells between RNA and ATAC...")
        free, total = _get_gpu_memory()
        print(f"[KNN-Agg-GPU] GPU memory: {free:.2f} GB free / {total:.2f} GB total")
    
    common_cells = sorted(set(adata_rna.obs_names) & set(adata_atac.obs_names))
    if len(common_cells) == 0:
        raise ValueError("No common cells between RNA and ATAC")
    
    if verbose:
        print(f"[KNN-Agg-GPU] Found {len(common_cells)} paired cells")
    
    if sample_cells is not None and sample_cells < len(common_cells):
        rng = np.random.default_rng(random_state)
        common_cells = sorted(rng.choice(common_cells, size=sample_cells, replace=False))
        if verbose:
            print(f"[KNN-Agg-GPU] Subsampled to {len(common_cells)} cells")
    
    adata_rna = adata_rna[common_cells].copy()
    adata_atac = adata_atac[common_cells].copy()
    
    # -------------------------
    # 2. Align genes
    # -------------------------
    if verbose:
        print("[KNN-Agg-GPU] Aligning genes...")
    
    if atac_layer and atac_layer in adata_atac.layers:
        atac_X_raw = adata_atac.layers[atac_layer]
        if verbose:
            print(f"[KNN-Agg-GPU] Using ATAC layer: {atac_layer}")
    else:
        atac_X_raw = adata_atac.X
        if verbose:
            print("[KNN-Agg-GPU] Using ATAC .X")
    
    shared_genes = sorted(set(adata_rna.var_names) & set(adata_atac.var_names))
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between RNA and ATAC.")
    
    if verbose:
        print(f"[KNN-Agg-GPU] Shared genes: {len(shared_genes)}")
    
    rna_gene_idx = adata_rna.var_names.get_indexer(shared_genes)
    atac_gene_idx = adata_atac.var_names.get_indexer(shared_genes)
    
    # -------------------------
    # 3. Extract and normalize data ON CPU
    # -------------------------
    if verbose:
        print("[KNN-Agg-GPU] Extracting and normalizing data on CPU...")
    
    # Extract subset
    rna_X_cpu = adata_rna.X[:, rna_gene_idx]
    if sparse.issparse(rna_X_cpu):
        rna_X_cpu = rna_X_cpu.toarray()
    rna_X_cpu = np.asarray(rna_X_cpu, dtype=np.float32)
    
    atac_X_cpu = atac_X_raw[:, atac_gene_idx]
    if sparse.issparse(atac_X_cpu):
        atac_X_cpu = atac_X_cpu.toarray()
    atac_X_cpu = np.asarray(atac_X_cpu, dtype=np.float32)
    
    n_cells, n_genes = rna_X_cpu.shape
    if verbose:
        print(f"[KNN-Agg-GPU] Final matrix: {n_cells} cells × {n_genes} genes")
        print(f"[KNN-Agg-GPU] Data size: {rna_X_cpu.nbytes / 1e9:.2f} GB per matrix")
    
    # Normalize on CPU
    if verbose:
        print("[KNN-Agg-GPU] Normalizing RNA (CPM + log1p)...")
    rna_sums = rna_X_cpu.sum(axis=1, keepdims=True)
    rna_sums = np.maximum(rna_sums, 1e-10)
    rna_X_cpu /= rna_sums
    rna_X_cpu *= 1e4
    np.log1p(rna_X_cpu, out=rna_X_cpu)
    del rna_sums
    
    if verbose:
        print("[KNN-Agg-GPU] Normalizing ATAC (CPM + log1p)...")
    atac_sums = atac_X_cpu.sum(axis=1, keepdims=True)
    atac_sums = np.maximum(atac_sums, 1e-10)
    atac_X_cpu /= atac_sums
    atac_X_cpu *= 1e4
    np.log1p(atac_X_cpu, out=atac_X_cpu)
    del atac_sums
    
    gc.collect()
    
    # -------------------------
    # 4. Compute PCA for KNN (batched)
    # -------------------------
    if verbose:
        print("[KNN-Agg-GPU] Computing PCA for KNN graph (batched)...")
        free, total = _get_gpu_memory()
        print(f"[KNN-Agg-GPU] GPU memory before PCA: {free:.2f} GB free")
    
    if use_rna_for_knn:
        X_for_pca = adata_rna.X
    else:
        X_for_pca = atac_X_raw
    
    pca_coords = _compute_pca_embedding_gpu_batched(
        X_for_pca, n_pcs=n_pcs, n_hvgs=n_hvgs, batch_size=20000
    )
    
    # Free adata objects - no longer needed
    del adata_rna, adata_atac, X_for_pca, atac_X_raw
    gc.collect()
    _clear_gpu()
    
    if verbose:
        free, total = _get_gpu_memory()
        print(f"[KNN-Agg-GPU] GPU memory after PCA: {free:.2f} GB free")
        print(f"[KNN-Agg-GPU] PCA coords shape: {pca_coords.shape}")
    
    # -------------------------
    # 5. Build KNN graph
    # -------------------------
    max_k = min(max(k_values), n_cells)
    k_values = [k for k in k_values if k <= n_cells]
    
    if verbose:
        print(f"[KNN-Agg-GPU] Building KNN graph with max_k={max_k}...")
    
    knn_indices = _compute_knn_gpu_batched(pca_coords, k=max_k, batch_size=5000)
    
    # Move KNN indices to CPU for correlation step
    knn_indices_cpu = cp.asnumpy(knn_indices)
    
    del pca_coords, knn_indices
    _clear_gpu()
    gc.collect()
    
    if verbose:
        print("[KNN-Agg-GPU] KNN graph complete.")
        free, total = _get_gpu_memory()
        print(f"[KNN-Agg-GPU] GPU memory after KNN: {free:.2f} GB free")
    
    # -------------------------
    # 6. Compute correlations for each k
    # -------------------------
    results = []
    per_k_correlations = {}
    
    for k in k_values:
        if verbose:
            print(f"\n[KNN-Agg-GPU] Processing k={k}...")
        
        corrs, rna_sparsity, atac_sparsity = _aggregate_and_correlate_batched(
            rna_X_cpu, atac_X_cpu, knn_indices_cpu, k,
            min_nonzero=min_nonzero, batch_size=5000
        )
        
        per_k_correlations[k] = corrs
        
        # Statistics
        valid = np.isfinite(corrs)
        n_valid = valid.sum()
        
        if n_valid > 0:
            mean_corr = np.nanmean(corrs)
            median_corr = np.nanmedian(corrs)
            std_corr = np.nanstd(corrs)
            pct_positive = (corrs[valid] > 0).mean() * 100
            pct_negative = (corrs[valid] < 0).mean() * 100
        else:
            mean_corr = median_corr = std_corr = np.nan
            pct_positive = pct_negative = 0
        
        results.append({
            'k': k,
            'n_valid': n_valid,
            'mean_corr': mean_corr,
            'median_corr': median_corr,
            'std_corr': std_corr,
            'pct_positive': pct_positive,
            'pct_negative': pct_negative,
            'rna_sparsity_pct': rna_sparsity,
            'atac_sparsity_pct': atac_sparsity,
        })
        
        if verbose:
            print(f"    k={k}: mean_r={mean_corr:.4f}, median_r={median_corr:.4f}, "
                  f"positive={pct_positive:.1f}%, RNA_sparse={rna_sparsity:.1f}%")
    
    results_df = pd.DataFrame(results)
    
    # Clean up
    del knn_indices_cpu, rna_X_cpu, atac_X_cpu
    gc.collect()
    _clear_gpu()
    
    # -------------------------
    # 7. Save results
    # -------------------------
    results_csv = os.path.join(output_dir, "knn_aggregation_results.csv")
    results_df.to_csv(results_csv, index=False)
    if verbose:
        print(f"\n[KNN-Agg-GPU] Saved results to: {results_csv}")
    
    # -------------------------
    # 8. Create visualization
    # -------------------------
    if verbose:
        print("[KNN-Agg-GPU] Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(results_df['k'], results_df['mean_corr'], 'b-o', label='Mean', linewidth=2, markersize=8)
    ax1.plot(results_df['k'], results_df['median_corr'], 'g-s', label='Median', linewidth=2, markersize=8)
    ax1.fill_between(results_df['k'], 
                     results_df['mean_corr'] - results_df['std_corr'],
                     results_df['mean_corr'] + results_df['std_corr'],
                     alpha=0.2, color='blue')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of neighbors (k)', fontsize=12)
    ax1.set_ylabel(f'{corr_method.capitalize()} Correlation', fontsize=12)
    ax1.set_title('RNA-ATAC Correlation vs. Aggregation Size', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(results_df['k'], results_df['pct_positive'], 'g-o', label='% Positive', linewidth=2)
    ax2.plot(results_df['k'], results_df['pct_negative'], 'r-s', label='% Negative', linewidth=2)
    ax2.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of neighbors (k)', fontsize=12)
    ax2.set_ylabel('Percentage of cells', fontsize=12)
    ax2.set_title('Proportion of Positive/Negative Correlations', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    ax3 = axes[1, 0]
    ax3.plot(results_df['k'], results_df['rna_sparsity_pct'], 'b-o', label='RNA', linewidth=2)
    ax3.plot(results_df['k'], results_df['atac_sparsity_pct'], 'orange', marker='s', label='ATAC', linewidth=2)
    ax3.set_xscale('log')
    ax3.set_xlabel('Number of neighbors (k)', fontsize=12)
    ax3.set_ylabel('Sparsity (%)', fontsize=12)
    ax3.set_title('Data Sparsity vs. Aggregation Size', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    selected_ks = [k for k in [1, 10, 50, 200] if k in per_k_correlations]
    if not selected_ks:
        selected_ks = list(per_k_correlations.keys())[:4]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_ks)))
    for k, color in zip(selected_ks, colors):
        corrs = per_k_correlations[k]
        valid = np.isfinite(corrs)
        if valid.any():
            ax4.hist(corrs[valid], bins=50, alpha=0.5, label=f'k={k}', 
                    color=color, density=True)
    
    ax4.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel(f'{corr_method.capitalize()} Correlation', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Correlation Distribution at Different k', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "knn_aggregation_correlation_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"[KNN-Agg-GPU] Saved plot to: {plot_path}")
    
    # Summary plot
    fig2, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(results_df['k'], results_df['mean_corr'], 'b-o', 
            label='Mean Correlation', linewidth=2.5, markersize=10)
    ax.fill_between(results_df['k'], 
                   results_df['mean_corr'] - results_df['std_corr'],
                   results_df['mean_corr'] + results_df['std_corr'],
                   alpha=0.2, color='blue')
    
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero correlation')
    
    ax.set_xscale('log')
    ax.set_xlabel('Number of neighbors (k) in aggregation', fontsize=14)
    ax.set_ylabel(f'{corr_method.capitalize()} Correlation (RNA vs ATAC)', fontsize=14)
    ax.set_title('RNA-ATAC Correlation Increases with Cell Aggregation\n(KNN-based pseudobulk, GPU-accelerated)', fontsize=14)
    
    k_1_corr = results_df[results_df['k'] == 1]['mean_corr'].values
    if len(k_1_corr) > 0:
        ax.annotate(f'Single-cell: {k_1_corr[0]:.3f}', 
                   xy=(1, k_1_corr[0]), xytext=(2, k_1_corr[0] - 0.1),
                   fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    
    max_k_row = results_df.iloc[-1]
    ax.annotate(f'k={int(max_k_row["k"])}: {max_k_row["mean_corr"]:.3f}', 
               xy=(max_k_row['k'], max_k_row['mean_corr']), 
               xytext=(max_k_row['k']/2, max_k_row['mean_corr'] + 0.05),
               fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    summary_plot_path = os.path.join(output_dir, "knn_correlation_summary.png")
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    if verbose:
        print(f"[KNN-Agg-GPU] Saved summary plot to: {summary_plot_path}")
        print("\n[KNN-Agg-GPU] Analysis complete!")
        k1_val = results_df[results_df['k']==1]['mean_corr'].values
        print(f"[KNN-Agg-GPU] Key finding: Correlation at k=1: {k1_val[0] if len(k1_val) > 0 else 'N/A':.4f}")
        print(f"[KNN-Agg-GPU] Key finding: Correlation at max k={int(max_k_row['k'])}: {max_k_row['mean_corr']:.4f}")
    
    return {
        'results_df': results_df,
        'k_values': k_values,
        'per_k_correlations': per_k_correlations,
        'plot_path': plot_path,
        'summary_plot_path': summary_plot_path,
    }


# =====================================================================
#                             ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("KNN AGGREGATION CORRELATION ANALYSIS (GPU - Memory Efficient)")
    print("="*60 + "\n")
    
    print(f"[Main] CUDA Device: {cp.cuda.Device().id}")
    free, total = _get_gpu_memory()
    print(f"[Main] GPU Memory: {free:.2f} GB free / {total:.2f} GB total")
    print(f"[Main] RMM pool allocator enabled")
    
    rna_path = '/dcl01/hongkai/data/data/hjiang/Data/paired/rna/placenta.h5ad'
    atac_path = '/dcs07/hongkai/data/harry/result/gene_activity/h5ad/placenta.h5ad'
    out_dir = "/dcs07/hongkai/data/harry/result/gene_activity/correlation/true_signac/knn_aggregation"
    
    print("[Main] Loading data...")
    adata_rna = sc.read_h5ad(rna_path)
    adata_atac = sc.read_h5ad(atac_path)
    
    print(f"[Main] RNA shape: {adata_rna.shape}")
    print(f"[Main] ATAC shape: {adata_atac.shape}")
    
    result = analyze_knn_aggregation_correlation_gpu(
        adata_rna=adata_rna,
        adata_atac=adata_atac,
        output_dir=out_dir,
        k_values=[1, 3, 5, 10, 20, 50, 100, 200, 500],
        n_pcs=30,
        n_hvgs=2000,
        corr_method="pearson",
        use_rna_for_knn=True,
        atac_layer="GeneActivity",
        verbose=True,
    )
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(result['results_df'].to_string(index=False))
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60 + "\n")