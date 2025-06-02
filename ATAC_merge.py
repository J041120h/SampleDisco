#!/usr/bin/env python3
"""
Script to merge ATAC-seq (gene activity) and RNA-seq data with batch effect correction.
Selects HVGs separately for each modality before merging.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def merge_atac_gene_rna(
    adata_atac_gene,
    adata_rna,
    output_dir,
    RNA_preprocess=True,
    ATAC_GENE_preprocess=True,
    target_sum=1e4,
    n_highly_variable_genes_rna=3000,
    n_highly_variable_genes_atac_gene=3000,
    min_shared_hvgs=500,
    n_pcs=50,
    n_neighbors=30,
    n_pcs_neighbors=30,
    leiden_resolution=0.5,
    umap_min_dist=0.3,
    combat_covariates=None,
    scale_max_value=10,
    min_genes=200,
    min_cells=3,
    save_raw=True,
    random_state=42,
    hvg_flavor='seurat_v3',
    hvg_batch_key=None
):
    """
    Merge ATAC-seq gene activity and RNA-seq data, perform batch correction, and visualize.
    HVGs are selected separately for each modality, then only shared HVGs are kept.
    
    Parameters:
    -----------
    adata_atac_gene : anndata.AnnData
        ATAC-seq gene activity matrix as AnnData object
    adata_rna : anndata.AnnData
        RNA-seq data as AnnData object
    output_dir : str
        Output directory for results
    RNA_preprocess : bool, default=True
        Whether to preprocess RNA data (normalize, log transform, find HVGs)
    ATAC_GENE_preprocess : bool, default=True
        Whether to preprocess ATAC gene activity data (normalize, log transform, find HVGs)
    target_sum : float, default=1e4
        Target sum for normalization
    n_highly_variable_genes_rna : int, default=3000
        Number of highly variable genes to select from RNA data
    n_highly_variable_genes_atac_gene : int, default=3000
        Number of highly variable genes to select from ATAC gene activity data
    min_shared_hvgs : int, default=500
        Minimum number of shared HVGs required between modalities
    n_pcs : int, default=50
        Number of principal components to compute
    n_neighbors : int, default=30
        Number of neighbors for neighborhood graph
    n_pcs_neighbors : int, default=30
        Number of PCs to use for neighborhood graph
    leiden_resolution : float, default=0.5
        Resolution parameter for Leiden clustering
    umap_min_dist : float, default=0.3
        Minimum distance parameter for UMAP
    combat_covariates : list, default=None
        Additional covariates to preserve during ComBat correction
    scale_max_value : float, default=10
        Maximum value for scaling
    min_genes : int, default=200
        Minimum number of genes per cell (for filtering)
    min_cells : int, default=3
        Minimum number of cells per gene (for filtering)
    save_raw : bool, default=True
        Whether to save raw counts before preprocessing
    random_state : int, default=42
        Random seed for reproducibility
    hvg_flavor : str, default='seurat_v3'
        Method for HVG selection ('seurat', 'seurat_v3', 'cell_ranger')
    hvg_batch_key : str, default=None
        Batch key for HVG selection within each modality
    """
    
    # Set random seed
    sc.settings.seed = random_state
    np.random.seed(random_state)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Make copies to avoid modifying the original objects
    adata_atac_gene = adata_atac_gene.copy()
    adata_rna = adata_rna.copy()
    
    print("Data loaded successfully")
    print(f"Initial ATAC gene activity data shape: {adata_atac_gene.shape}")
    print(f"Initial RNA data shape: {adata_rna.shape}")
    
    # Process ATAC gene activity data
    if ATAC_GENE_preprocess:
        print("\nPreprocessing ATAC gene activity data...")
        # Basic quality control
        sc.pp.filter_cells(adata_atac_gene, min_genes=min_genes)
        sc.pp.filter_genes(adata_atac_gene, min_cells=min_cells)
        
        # Calculate QC metrics
        adata_atac_gene.var['mt'] = adata_atac_gene.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata_atac_gene, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        
        # Save raw counts if requested
        if save_raw:
            adata_atac_gene.raw = adata_atac_gene.copy()
        
        # Normalize and log transform
        sc.pp.normalize_total(adata_atac_gene, target_sum=target_sum)
        sc.pp.log1p(adata_atac_gene)
        
        # Find highly variable genes for ATAC gene activity
        print(f"Finding top {n_highly_variable_genes_atac_gene} HVGs in ATAC gene activity data...")
        sc.pp.highly_variable_genes(
            adata_atac_gene, 
            n_top_genes=n_highly_variable_genes_atac_gene,
            flavor=hvg_flavor,
            batch_key=hvg_batch_key
        )
        atac_gene_hvgs = set(adata_atac_gene.var_names[adata_atac_gene.var.highly_variable])
        print(f"Found {len(atac_gene_hvgs)} HVGs in ATAC gene activity data")
    else:
        print("ATAC gene activity preprocessing skipped - assuming HVGs are already marked")
        if 'highly_variable' not in adata_atac_gene.var.columns:
            raise ValueError("ATAC gene activity data doesn't have 'highly_variable' column. Please preprocess or set ATAC_GENE_preprocess=True")
        atac_gene_hvgs = set(adata_atac_gene.var_names[adata_atac_gene.var.highly_variable])
        print(f"Found {len(atac_gene_hvgs)} pre-selected HVGs in ATAC gene activity data")
    
    # Process RNA data
    if RNA_preprocess:
        print("\nPreprocessing RNA data...")
        # Basic quality control
        sc.pp.filter_cells(adata_rna, min_genes=min_genes)
        sc.pp.filter_genes(adata_rna, min_cells=min_cells)
        
        # Calculate QC metrics
        adata_rna.var['mt'] = adata_rna.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata_rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        
        # Save raw counts if requested
        if save_raw:
            adata_rna.raw = adata_rna.copy()
        
        # Normalize and log transform
        sc.pp.normalize_total(adata_rna, target_sum=target_sum)
        sc.pp.log1p(adata_rna)
        
        # Find highly variable genes for RNA
        print(f"Finding top {n_highly_variable_genes_rna} HVGs in RNA data...")
        sc.pp.highly_variable_genes(
            adata_rna, 
            n_top_genes=n_highly_variable_genes_rna,
            flavor=hvg_flavor,
            batch_key=hvg_batch_key
        )
        rna_hvgs = set(adata_rna.var_names[adata_rna.var.highly_variable])
        print(f"Found {len(rna_hvgs)} HVGs in RNA data")
    else:
        print("RNA preprocessing skipped - assuming HVGs are already marked")
        if 'highly_variable' not in adata_rna.var.columns:
            raise ValueError("RNA data doesn't have 'highly_variable' column. Please preprocess or set RNA_preprocess=True")
        rna_hvgs = set(adata_rna.var_names[adata_rna.var.highly_variable])
        print(f"Found {len(rna_hvgs)} pre-selected HVGs in RNA data")
    
    # Find shared HVGs
    shared_hvgs = list(atac_gene_hvgs.intersection(rna_hvgs))
    print(f"\nNumber of shared HVGs: {len(shared_hvgs)}")
    
    # Also find shared genes (not just HVGs) for comparison
    all_shared_genes = list(set(adata_atac_gene.var_names) & set(adata_rna.var_names))
    print(f"Total number of shared genes: {len(all_shared_genes)}")
    
    if len(shared_hvgs) < min_shared_hvgs:
        print(f"Warning: Only {len(shared_hvgs)} shared HVGs found, which is less than the minimum {min_shared_hvgs}")
        print("Consider adjusting the number of HVGs selected or using all shared genes")
        
        # Option to use all shared genes if not enough shared HVGs
        use_all_shared = input("Use all shared genes instead of just HVGs? (y/n): ").lower() == 'y'
        if use_all_shared:
            shared_genes_to_use = all_shared_genes
            print(f"Using all {len(shared_genes_to_use)} shared genes")
        else:
            shared_genes_to_use = shared_hvgs
            print(f"Proceeding with {len(shared_genes_to_use)} shared HVGs")
    else:
        shared_genes_to_use = shared_hvgs
    
    # Sort genes for consistency
    shared_genes_to_use = sorted(shared_genes_to_use)
    
    # Subset to shared HVGs
    print(f"\nSubsetting to {len(shared_genes_to_use)} shared genes...")
    adata_atac_gene_subset = adata_atac_gene[:, shared_genes_to_use].copy()
    adata_rna_subset = adata_rna[:, shared_genes_to_use].copy()
    
    # Add batch labels
    adata_atac_gene_subset.obs['_batch'] = 'ATAC_GENE'
    adata_rna_subset.obs['_batch'] = 'RNA'
    
    # Concatenate the datasets
    print("Merging datasets...")
    adata_merged = ad.concat([adata_atac_gene_subset, adata_rna_subset], 
                            axis=0, 
                            join='outer',
                            merge='same')
    
    print(f"Merged data shape: {adata_merged.shape}")
    
    # Apply ComBat batch correction
    print("Applying ComBat batch correction...")
    sc.pp.combat(adata_merged, key='_batch', covariates=combat_covariates)
    
    # Scale the data
    sc.pp.scale(adata_merged, max_value=scale_max_value)
    
    # PCA
    print("Running PCA...")
    sc.tl.pca(adata_merged, svd_solver='arpack', n_comps=min(n_pcs, len(shared_genes_to_use)-1))
    
    # Compute neighborhood graph
    print("Computing neighborhood graph...")
    sc.pp.neighbors(adata_merged, n_neighbors=n_neighbors, n_pcs=min(n_pcs_neighbors, len(shared_genes_to_use)-1))
    
    # UMAP embedding
    print("Computing UMAP...")
    sc.tl.umap(adata_merged, min_dist=umap_min_dist)
    
    # Leiden clustering for visualization
    sc.tl.leiden(adata_merged, resolution=leiden_resolution)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Set up the plotting style
    sc.settings.figdir = str(output_path)
    sc.set_figure_params(dpi=300, fontsize=12, figsize=(6, 6))
    
    # Create comprehensive UMAP visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Plot 1: Color by batch
    sc.pl.umap(adata_merged, 
               color='_batch', 
               title='UMAP colored by batch (after ComBat)',
               ax=axes[0, 0],
               show=False,
               legend_loc='on data',
               frameon=False)
    
    # Plot 2: Color by leiden clusters
    sc.pl.umap(adata_merged, 
               color='leiden', 
               title='UMAP colored by Leiden clusters',
               ax=axes[0, 1],
               show=False,
               frameon=False,
               palette='tab20')
    
    # Plot 3: Color by total counts
    sc.pl.umap(adata_merged,
               color='n_genes_by_counts',
               title='UMAP colored by number of genes',
               ax=axes[1, 0],
               show=False,
               frameon=False,
               cmap='viridis')
    
    # Plot 4: Color by total UMI counts
    sc.pl.umap(adata_merged,
               color='total_counts',
               title='UMAP colored by total counts',
               ax=axes[1, 1],
               show=False,
               frameon=False,
               cmap='viridis')
    
    plt.tight_layout()
    plt.savefig(output_path / 'umap_visualization.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'umap_visualization.pdf', bbox_inches='tight')
    plt.close()
    
    # Batch distribution per cluster
    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_batch_counts = pd.crosstab(adata_merged.obs['leiden'], 
                                      adata_merged.obs['_batch'])
    cluster_batch_prop = cluster_batch_counts.div(cluster_batch_counts.sum(axis=1), axis=0)
    
    cluster_batch_prop.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_xlabel('Leiden Cluster')
    ax.set_ylabel('Proportion')
    ax.set_title('Batch distribution across clusters')
    ax.legend(title='Batch', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'batch_distribution_per_cluster.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Heatmap of cluster composition
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cluster_batch_prop.T, 
                annot=True, 
                fmt='.2f', 
                cmap='RdBu_r', 
                center=0.5,
                ax=ax,
                cbar_kws={'label': 'Proportion'})
    ax.set_xlabel('Leiden Cluster')
    ax.set_ylabel('Batch')
    ax.set_title('Batch composition heatmap')
    plt.tight_layout()
    plt.savefig(output_path / 'batch_composition_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Venn diagram of HVGs
    try:
        from matplotlib_venn import venn2
        fig, ax = plt.subplots(figsize=(8, 8))
        venn = venn2([atac_gene_hvgs, rna_hvgs], ('ATAC Gene Activity HVGs', 'RNA HVGs'))
        ax.set_title(f'Overlap of Highly Variable Genes\n(ATAC Gene Activity: {len(atac_gene_hvgs)}, RNA: {len(rna_hvgs)}, Shared: {len(shared_hvgs)})')
        plt.savefig(output_path / 'hvg_overlap_venn.png', dpi=300, bbox_inches='tight')
        plt.close()
    except ImportError:
        print("matplotlib_venn not installed. Skipping Venn diagram. Install with: pip install matplotlib-venn")
    
    # Save quality metrics
    print("Calculating quality metrics...")
    metrics = {
        'n_cells_total': adata_merged.n_obs,
        'n_cells_atac_gene': (adata_merged.obs['_batch'] == 'ATAC_GENE').sum(),
        'n_cells_rna': (adata_merged.obs['_batch'] == 'RNA').sum(),
        'n_genes_total_shared': len(all_shared_genes),
        'n_hvgs_atac_gene': len(atac_gene_hvgs),
        'n_hvgs_rna': len(rna_hvgs),
        'n_hvgs_shared': len(shared_hvgs),
        'n_genes_used_in_merge': len(shared_genes_to_use),
        'n_clusters': len(adata_merged.obs['leiden'].unique()),
        'preprocessing_params': {
            'RNA_preprocess': RNA_preprocess,
            'ATAC_GENE_preprocess': ATAC_GENE_preprocess,
            'target_sum': target_sum,
            'n_highly_variable_genes_rna': n_highly_variable_genes_rna,
            'n_highly_variable_genes_atac_gene': n_highly_variable_genes_atac_gene,
            'hvg_flavor': hvg_flavor,
            'n_pcs': n_pcs,
            'n_neighbors': n_neighbors,
            'leiden_resolution': leiden_resolution,
            'umap_min_dist': umap_min_dist
        }
    }
    
    # Save detailed metrics
    with open(output_path / 'merge_metrics.txt', 'w') as f:
        f.write("Merge and Batch Correction Metrics\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Cell Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total cells: {metrics['n_cells_total']:,}\n")
        f.write(f"ATAC gene activity cells: {metrics['n_cells_atac_gene']:,}\n")
        f.write(f"RNA cells: {metrics['n_cells_rna']:,}\n")
        f.write(f"Number of clusters: {metrics['n_clusters']}\n\n")
        
        f.write("Gene Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total shared genes: {metrics['n_genes_total_shared']:,}\n")
        f.write(f"HVGs in ATAC gene activity: {metrics['n_hvgs_atac_gene']:,}\n")
        f.write(f"HVGs in RNA: {metrics['n_hvgs_rna']:,}\n")
        f.write(f"Shared HVGs: {metrics['n_hvgs_shared']:,}\n")
        f.write(f"Genes used in merge: {metrics['n_genes_used_in_merge']:,}\n\n")
        
        f.write("Preprocessing Parameters:\n")
        f.write("-" * 30 + "\n")
        for key, value in metrics['preprocessing_params'].items():
            f.write(f"{key}: {value}\n")
    
    # Save list of shared HVGs
    with open(output_path / 'shared_hvgs.txt', 'w') as f:
        for gene in sorted(shared_hvgs):
            f.write(f"{gene}\n")
    
    # Save the merged object
    print("Saving merged data...")
    adata_merged.write_h5ad(output_path / 'merged_batch_corrected.h5ad')
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"- Merged data: merged_batch_corrected.h5ad")
    print(f"- UMAP visualizations: umap_visualization.png/pdf")
    print(f"- Batch distribution: batch_distribution_per_cluster.png")
    print(f"- Batch composition heatmap: batch_composition_heatmap.png")
    print(f"- HVG overlap: hvg_overlap_venn.png")
    print(f"- Metrics: merge_metrics.txt")
    print(f"- Shared HVGs list: shared_hvgs.txt")
    
    return adata_merged


# Example usage with manual parameter input
if __name__ == "__main__":
    # Load your anndata objects
    import scanpy as sc
    
    # Load the data
    adata_atac_gene = sc.read_h5ad("/Users/harry/Desktop/GenoDistance/result/gene_activity/gene_activity.h5ad")
    adata_rna = sc.read_h5ad("/Users/harry/Desktop/GenoDistance/Data/count_data.h5ad")
    
    # Define output directory
    output_directory = "/Users/harry/Desktop/GenoDistance/result/integration_atac_rna"
    
    # Call the function with custom parameters
    merged_data = merge_atac_gene_rna(
        adata_atac_gene=adata_atac_gene,
        adata_rna=adata_rna,
        output_dir=output_directory,
        RNA_preprocess=True,  # Set to False if RNA data is already preprocessed
        ATAC_GENE_preprocess=True,  # Set to False if ATAC gene activity data is already preprocessed
        target_sum=1e4,
        n_highly_variable_genes_rna=6000,  # HVGs for RNA
        n_highly_variable_genes_atac_gene=6000,  # HVGs for ATAC gene activity
        min_shared_hvgs=500,  # Minimum shared HVGs required
        n_pcs=50,
        n_neighbors=30,
        n_pcs_neighbors=30,
        leiden_resolution=0.5,
        umap_min_dist=0.3,
        combat_covariates=None,  # Add list of covariate column names if needed
        scale_max_value=10,
        min_genes=200,
        min_cells=3,
        save_raw=True,
        random_state=42,
        hvg_flavor='seurat_v3',  # or 'seurat', 'cell_ranger'
        hvg_batch_key=None  # Set if you have batches within each modality
    )