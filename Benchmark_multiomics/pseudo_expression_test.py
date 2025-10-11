import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse, stats
from scipy.spatial.distance import correlation as corr_distance
import os
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import gc


def compute_paired_cell_metrics(
    integrated_path: str,
    output_dir: str = "./validation_results",
    top_n_genes: int = 2000,
    n_sample_genes: int = 100,
    n_workers: int = 8,
    verbose: bool = True
) -> Dict:
    """
    Compare inferred ATAC gene activity with ground truth RNA expression for paired cells.
    
    **Important**: This function expects data processed by the integration pipeline where:
    - Cell indices have _RNA or _ATAC suffixes added (e.g., "AAACCCAAGAAACACT_RNA")
    - The 'original_barcode' column stores the true barcodes WITHOUT suffixes
    - Cells are matched using the 'original_barcode' column to find RNA-ATAC pairs
    
    Parameters:
    -----------
    integrated_path : str
        Path to integrated h5ad file with both RNA and ATAC cells
    output_dir : str
        Directory to save results and visualizations
    top_n_genes : int
        Number of top variable genes to focus analysis on (currently not used)
    n_sample_genes : int
        Number of genes to sample for detailed scatter plots (50 best + 50 worst)
    n_workers : int
        Number of parallel workers for computation
    verbose : bool
        Print progress information
        
    Returns:
    --------
    Dict containing all computed metrics and statistics
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("=" * 80)
        print("ATAC Gene Activity Validation Analysis")
        print("=" * 80)
        print(f"\n📂 Loading integrated data from: {integrated_path}")
    
    # Load integrated data
    adata = ad.read_h5ad(integrated_path)
    
    if verbose:
        print(f"   Total cells: {adata.n_obs}")
        print(f"   Total genes: {adata.n_vars}")
    
    # Check for required columns
    if 'modality' not in adata.obs.columns:
        raise ValueError("'modality' column not found in obs. Cannot separate RNA and ATAC cells.")
    
    if 'original_barcode' not in adata.obs.columns:
        raise ValueError("'original_barcode' column not found. Cannot match paired cells.")
    
    # Verify that indices have suffixes (as expected from integration)
    if verbose:
        print(f"\n🔍 Verifying cell index structure...")
        sample_indices = adata.obs.index[:5].tolist()
        print(f"   Sample indices: {sample_indices}")
        has_rna_suffix = any('_RNA' in str(idx) for idx in adata.obs.index)
        has_atac_suffix = any('_ATAC' in str(idx) for idx in adata.obs.index)
        print(f"   Has _RNA suffix: {has_rna_suffix}")
        print(f"   Has _ATAC suffix: {has_atac_suffix}")
        
        if not (has_rna_suffix or has_atac_suffix):
            print(f"   ⚠️ Warning: Expected suffixes (_RNA/_ATAC) not found in indices")
    
    # Separate RNA and ATAC cells
    rna_mask = adata.obs['modality'] == 'RNA'
    atac_mask = adata.obs['modality'] == 'ATAC'
    
    rna_cells = adata[rna_mask].copy()
    atac_cells = adata[atac_mask].copy()
    
    if verbose:
        print(f"\n🧬 Separated modalities:")
        print(f"   RNA cells: {rna_cells.n_obs}")
        print(f"   ATAC cells: {atac_cells.n_obs}")
    
    # Match paired cells using original barcodes
    # Note: Integration process adds _RNA/_ATAC suffixes to indices
    # but 'original_barcode' column stores the true barcodes for matching
    rna_barcodes = rna_cells.obs['original_barcode'].values
    atac_barcodes = atac_cells.obs['original_barcode'].values
    
    if verbose:
        print(f"\n🔗 Matching cells using original barcodes...")
        print(f"   Sample RNA barcodes: {rna_barcodes[:3].tolist()}")
        print(f"   Sample ATAC barcodes: {atac_barcodes[:3].tolist()}")
        print(f"   Sample RNA indices: {rna_cells.obs.index[:3].tolist()}")
        print(f"   Sample ATAC indices: {atac_cells.obs.index[:3].tolist()}")
    
    # Find common barcodes (paired cells)
    common_barcodes = np.intersect1d(rna_barcodes, atac_barcodes)
    n_paired = len(common_barcodes)
    
    # Alternative matching strategy if original_barcode doesn't work well
    # Try stripping suffixes from indices
    if n_paired == 0:
        if verbose:
            print(f"   ⚠️ No matches found using original_barcode column")
            print(f"   Attempting alternative: stripping suffixes from indices...")
        
        rna_barcodes_alt = [str(idx).replace('_RNA', '') for idx in rna_cells.obs.index]
        atac_barcodes_alt = [str(idx).replace('_ATAC', '') for idx in atac_cells.obs.index]
        common_barcodes = np.intersect1d(rna_barcodes_alt, atac_barcodes_alt)
        n_paired = len(common_barcodes)
        
        if n_paired > 0:
            # Update the barcode arrays for subsequent matching
            rna_barcodes = np.array(rna_barcodes_alt)
            atac_barcodes = np.array(atac_barcodes_alt)
            if verbose:
                print(f"   ✅ Found {n_paired} matches using suffix-stripped indices")
    
    if verbose:
        print(f"\n🔗 Paired cell matching:")
        print(f"   Paired cells found: {n_paired}")
        print(f"   RNA-only cells: {len(rna_barcodes) - n_paired}")
        print(f"   ATAC-only cells: {len(atac_barcodes) - n_paired}")
        
        # Show example matches
        if n_paired > 0:
            print(f"\n   Example paired cell matches:")
            for i in range(min(3, n_paired)):
                bc = common_barcodes[i]
                rna_idx = rna_cells.obs.index[rna_barcode_to_idx[bc]]
                atac_idx = atac_cells.obs.index[atac_barcode_to_idx[bc]]
                print(f"     {bc:20s} -> RNA: {rna_idx:30s} | ATAC: {atac_idx:30s}")
    
    if n_paired == 0:
        raise ValueError("No paired cells found. Check original_barcode matching.")
    
    # Create mapping for efficient indexing
    rna_barcode_to_idx = {bc: idx for idx, bc in enumerate(rna_barcodes)}
    atac_barcode_to_idx = {bc: idx for idx, bc in enumerate(atac_barcodes)}
    
    # Get indices for paired cells
    rna_paired_idx = np.array([rna_barcode_to_idx[bc] for bc in common_barcodes])
    atac_paired_idx = np.array([atac_barcode_to_idx[bc] for bc in common_barcodes])
    
    # Extract paired cell data
    rna_paired = rna_cells[rna_paired_idx, :].copy()
    atac_paired = atac_cells[atac_paired_idx, :].copy()
    
    # Ensure genes are aligned
    assert np.all(rna_paired.var_names == atac_paired.var_names), "Gene names mismatch between RNA and ATAC"
    
    if verbose:
        print(f"\n📊 Extracting expression matrices...")
    
    # Get expression matrices (convert to dense for easier computation)
    if sparse.issparse(rna_paired.X):
        rna_expr = rna_paired.X.toarray()
    else:
        rna_expr = rna_paired.X
    
    if sparse.issparse(atac_paired.X):
        atac_expr = atac_paired.X.toarray()
    else:
        atac_paired_X
    
    n_cells, n_genes = rna_expr.shape
    
    if verbose:
        print(f"   Matrix shape: {n_cells} cells × {n_genes} genes")
        print(f"   RNA expression range: [{rna_expr.min():.2f}, {rna_expr.max():.2f}]")
        print(f"   ATAC gene activity range: [{atac_expr.min():.2f}, {atac_expr.max():.2f}]")
    
    # ============================================================================
    # METRIC 1: Per-cell correlation
    # ============================================================================
    if verbose:
        print(f"\n📈 Computing per-cell correlations (using {n_workers} workers)...")
    
    def compute_cell_correlation(i):
        """Compute Pearson correlation for a single cell"""
        rna_cell = rna_expr[i, :]
        atac_cell = atac_expr[i, :]
        
        # Remove zero-variance genes for this cell pair
        mask = (rna_cell != rna_cell[0]) | (atac_cell != atac_cell[0])
        if mask.sum() < 10:  # Need at least 10 genes
            return np.nan
        
        corr, _ = stats.pearsonr(rna_cell[mask], atac_cell[mask])
        return corr
    
    # Parallel computation of per-cell correlations
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        per_cell_corr = list(executor.map(compute_cell_correlation, range(n_cells)))
    
    per_cell_corr = np.array(per_cell_corr)
    valid_corr = per_cell_corr[~np.isnan(per_cell_corr)]
    
    if verbose:
        print(f"   Valid correlations: {len(valid_corr)}/{n_cells}")
        print(f"   Mean correlation: {valid_corr.mean():.4f}")
        print(f"   Median correlation: {np.median(valid_corr):.4f}")
        print(f"   Std correlation: {valid_corr.std():.4f}")
    
    # ============================================================================
    # METRIC 2: Per-gene correlation
    # ============================================================================
    if verbose:
        print(f"\n📈 Computing per-gene correlations...")
    
    def compute_gene_correlation(j):
        """Compute Pearson correlation for a single gene"""
        rna_gene = rna_expr[:, j]
        atac_gene = atac_expr[:, j]
        
        # Skip genes with no variance
        if rna_gene.std() == 0 or atac_gene.std() == 0:
            return np.nan, np.nan, np.nan
        
        corr, pval = stats.pearsonr(rna_gene, atac_gene)
        spearman_corr, _ = stats.spearmanr(rna_gene, atac_gene)
        
        # Compute RMSE
        rmse = np.sqrt(np.mean((rna_gene - atac_gene) ** 2))
        
        return corr, spearman_corr, rmse
    
    # Parallel computation
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        gene_metrics = list(executor.map(compute_gene_correlation, range(n_genes)))
    
    per_gene_corr = np.array([m[0] for m in gene_metrics])
    per_gene_spearman = np.array([m[1] for m in gene_metrics])
    per_gene_rmse = np.array([m[2] for m in gene_metrics])
    
    # Create gene-level results DataFrame
    gene_results = pd.DataFrame({
        'gene': rna_paired.var_names,
        'pearson_corr': per_gene_corr,
        'spearman_corr': per_gene_spearman,
        'rmse': per_gene_rmse,
        'mean_rna': rna_expr.mean(axis=0),
        'mean_atac': atac_expr.mean(axis=0),
        'std_rna': rna_expr.std(axis=0),
        'std_atac': atac_expr.std(axis=0)
    })
    
    # Remove genes with NaN correlations
    gene_results_valid = gene_results.dropna(subset=['pearson_corr'])
    
    if verbose:
        print(f"   Valid gene correlations: {len(gene_results_valid)}/{n_genes}")
        print(f"   Mean Pearson correlation: {gene_results_valid['pearson_corr'].mean():.4f}")
        print(f"   Mean Spearman correlation: {gene_results_valid['spearman_corr'].mean():.4f}")
        print(f"   Mean RMSE: {gene_results_valid['rmse'].mean():.4f}")
    
    # ============================================================================
    # METRIC 3: Overall statistics
    # ============================================================================
    if verbose:
        print(f"\n📊 Computing overall statistics...")
    
    # Flatten arrays for overall correlation
    rna_flat = rna_expr.flatten()
    atac_flat = atac_expr.flatten()
    
    overall_corr, _ = stats.pearsonr(rna_flat, atac_flat)
    overall_spearman, _ = stats.spearmanr(rna_flat, atac_flat)
    overall_rmse = np.sqrt(np.mean((rna_flat - atac_flat) ** 2))
    overall_mae = np.mean(np.abs(rna_flat - atac_flat))
    
    # R-squared
    ss_res = np.sum((rna_flat - atac_flat) ** 2)
    ss_tot = np.sum((rna_flat - rna_flat.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    if verbose:
        print(f"   Overall Pearson correlation: {overall_corr:.4f}")
        print(f"   Overall Spearman correlation: {overall_spearman:.4f}")
        print(f"   Overall RMSE: {overall_rmse:.4f}")
        print(f"   Overall MAE: {overall_mae:.4f}")
        print(f"   Overall R²: {r_squared:.4f}")
    
    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    if verbose:
        print(f"\n🎨 Generating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # 1. Per-cell correlation distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Per-cell correlation histogram
    axes[0, 0].hist(valid_corr, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(valid_corr.mean(), color='red', linestyle='--', 
                       label=f'Mean: {valid_corr.mean():.3f}')
    axes[0, 0].axvline(np.median(valid_corr), color='green', linestyle='--',
                       label=f'Median: {np.median(valid_corr):.3f}')
    axes[0, 0].set_xlabel('Pearson Correlation')
    axes[0, 0].set_ylabel('Number of Cells')
    axes[0, 0].set_title('Per-Cell Correlation Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Per-gene correlation histogram
    axes[0, 1].hist(gene_results_valid['pearson_corr'], bins=50, 
                    edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(gene_results_valid['pearson_corr'].mean(), 
                       color='red', linestyle='--',
                       label=f'Mean: {gene_results_valid["pearson_corr"].mean():.3f}')
    axes[0, 1].set_xlabel('Pearson Correlation')
    axes[0, 1].set_ylabel('Number of Genes')
    axes[0, 1].set_title('Per-Gene Correlation Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot of mean expression
    axes[1, 0].scatter(gene_results_valid['mean_rna'], 
                      gene_results_valid['mean_atac'],
                      alpha=0.5, s=10)
    axes[1, 0].plot([gene_results_valid['mean_rna'].min(), 
                     gene_results_valid['mean_rna'].max()],
                    [gene_results_valid['mean_rna'].min(), 
                     gene_results_valid['mean_rna'].max()],
                    'r--', label='Identity line')
    axes[1, 0].set_xlabel('Mean RNA Expression')
    axes[1, 0].set_ylabel('Mean ATAC Gene Activity')
    axes[1, 0].set_title('Mean Expression: RNA vs ATAC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Correlation vs expression level
    axes[1, 1].scatter(gene_results_valid['mean_rna'], 
                      gene_results_valid['pearson_corr'],
                      alpha=0.5, s=10, c=gene_results_valid['std_rna'],
                      cmap='viridis')
    axes[1, 1].set_xlabel('Mean RNA Expression')
    axes[1, 1].set_ylabel('Pearson Correlation')
    axes[1, 1].set_title('Gene Correlation vs Expression Level')
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('RNA Std Dev')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_overview.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed scatter plots for top correlated genes
    if verbose:
        print(f"   Creating detailed gene scatter plots...")
    
    # Select genes for detailed analysis
    top_genes_idx = gene_results_valid.nlargest(n_sample_genes // 2, 'pearson_corr').index
    bottom_genes_idx = gene_results_valid.nsmallest(n_sample_genes // 2, 'pearson_corr').index
    sample_genes_idx = np.concatenate([top_genes_idx, bottom_genes_idx])
    
    n_plots = min(12, len(sample_genes_idx))
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_genes_idx[:n_plots]):
        gene_name = gene_results.loc[idx, 'gene']
        corr_val = gene_results.loc[idx, 'pearson_corr']
        
        gene_idx = rna_paired.var_names.get_loc(gene_name)
        rna_gene_expr = rna_expr[:, gene_idx]
        atac_gene_expr = atac_expr[:, gene_idx]
        
        axes[i].scatter(rna_gene_expr, atac_gene_expr, alpha=0.5, s=20)
        axes[i].set_xlabel('RNA Expression')
        axes[i].set_ylabel('ATAC Gene Activity')
        axes[i].set_title(f'{gene_name}\n(r={corr_val:.3f})')
        axes[i].grid(True, alpha=0.3)
        
        # Add regression line
        if corr_val == corr_val:  # Check not NaN
            z = np.polyfit(rna_gene_expr, atac_gene_expr, 1)
            p = np.poly1d(z)
            x_line = np.linspace(rna_gene_expr.min(), rna_gene_expr.max(), 100)
            axes[i].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gene_scatter_plots.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of correlation matrix (sample of cells)
    if verbose:
        print(f"   Creating correlation heatmap...")
    
    n_sample_cells = min(100, n_cells)
    sample_idx = np.random.choice(n_cells, n_sample_cells, replace=False)
    sample_genes = min(50, n_genes)
    
    # Select top variable genes for heatmap
    gene_var = rna_expr.var(axis=0)
    top_var_genes_idx = np.argsort(gene_var)[-sample_genes:]
    
    rna_sample = rna_expr[np.ix_(sample_idx, top_var_genes_idx)]
    atac_sample = atac_expr[np.ix_(sample_idx, top_var_genes_idx)]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # RNA heatmap
    im1 = axes[0].imshow(rna_sample.T, aspect='auto', cmap='YlOrRd')
    axes[0].set_title('RNA Expression')
    axes[0].set_xlabel('Cells')
    axes[0].set_ylabel('Genes')
    plt.colorbar(im1, ax=axes[0])
    
    # ATAC heatmap
    im2 = axes[1].imshow(atac_sample.T, aspect='auto', cmap='YlOrRd')
    axes[1].set_title('ATAC Gene Activity')
    axes[1].set_xlabel('Cells')
    axes[1].set_ylabel('Genes')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference heatmap
    diff = rna_sample - atac_sample
    im3 = axes[2].imshow(diff.T, aspect='auto', cmap='RdBu_r', 
                        vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[2].set_title('Difference (RNA - ATAC)')
    axes[2].set_xlabel('Cells')
    axes[2].set_ylabel('Genes')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expression_heatmaps.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    if verbose:
        print(f"\n💾 Saving results...")
    
    # Save gene-level results
    gene_results.to_csv(os.path.join(output_dir, 'per_gene_metrics.csv'), 
                       index=False)
    
    # Save cell-level results
    cell_results = pd.DataFrame({
        'barcode': common_barcodes,
        'pearson_corr': per_cell_corr
    })
    cell_results.to_csv(os.path.join(output_dir, 'per_cell_metrics.csv'), 
                       index=False)
    
    # Save summary statistics
    summary_stats = {
        'n_paired_cells': n_paired,
        'n_genes': n_genes,
        'mean_cell_correlation': float(valid_corr.mean()),
        'median_cell_correlation': float(np.median(valid_corr)),
        'std_cell_correlation': float(valid_corr.std()),
        'mean_gene_pearson': float(gene_results_valid['pearson_corr'].mean()),
        'mean_gene_spearman': float(gene_results_valid['spearman_corr'].mean()),
        'mean_gene_rmse': float(gene_results_valid['rmse'].mean()),
        'overall_pearson': float(overall_corr),
        'overall_spearman': float(overall_spearman),
        'overall_rmse': float(overall_rmse),
        'overall_mae': float(overall_mae),
        'overall_r_squared': float(r_squared)
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), 
                     index=False)
    
    # Print summary
    if verbose:
        print(f"\n" + "=" * 80)
        print("SUMMARY OF VALIDATION RESULTS")
        print("=" * 80)
        print(f"\n📊 Sample Information:")
        print(f"   Paired cells analyzed: {n_paired}")
        print(f"   Genes analyzed: {n_genes}")
        
        print(f"\n📈 Per-Cell Metrics:")
        print(f"   Mean correlation: {valid_corr.mean():.4f} ± {valid_corr.std():.4f}")
        print(f"   Median correlation: {np.median(valid_corr):.4f}")
        print(f"   Min/Max correlation: [{valid_corr.min():.4f}, {valid_corr.max():.4f}]")
        
        print(f"\n📈 Per-Gene Metrics:")
        print(f"   Mean Pearson correlation: {gene_results_valid['pearson_corr'].mean():.4f}")
        print(f"   Mean Spearman correlation: {gene_results_valid['spearman_corr'].mean():.4f}")
        print(f"   Mean RMSE: {gene_results_valid['rmse'].mean():.4f}")
        
        print(f"\n📈 Overall Metrics:")
        print(f"   Overall Pearson correlation: {overall_corr:.4f}")
        print(f"   Overall R²: {r_squared:.4f}")
        print(f"   Overall RMSE: {overall_rmse:.4f}")
        print(f"   Overall MAE: {overall_mae:.4f}")
        
        print(f"\n💾 Results saved to: {output_dir}")
        print("=" * 80)
    
    # Clean up
    del rna_expr, atac_expr, rna_paired, atac_paired
    gc.collect()
    
    return summary_stats

if __name__ == "__main__":
    # Run validation
    results = compute_paired_cell_metrics(
        integrated_path="/dcs07/hongkai/data/harry/result/Benchmark/multiomics/preprocess/atac_rna_integrated.h5ad",
        output_dir="/dcs07/hongkai/data/harry/result/Benchmark/pseudo_expression",
        top_n_genes=2000,
        n_sample_genes=100,
        n_workers=8,
        verbose=True
    )