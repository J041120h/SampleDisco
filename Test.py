#!/usr/bin/env python
"""
Script to explore and provide an overview of an scATAC-seq h5ad file
This example examines SRR14466459.h5ad
"""

import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_h5ad(file_path):
    """
    Explore and analyze a single h5ad file to provide an overview
    
    Parameters:
    -----------
    file_path : str
        Path to the h5ad file to analyze
    """
    # Set up figure aesthetics
    sc.settings.set_figure_params(dpi=100, facecolor='white')
    sc.settings.verbosity = 1  # Verbosity: errors (0), warnings (1), info (2), hints (3)
    
    print(f"Reading file: {file_path}")
    
    # Load the h5ad file
    adata = sc.read_h5ad(file_path)
    
    # Basic information about the dataset
    print("\n=== BASIC DATASET INFORMATION ===")
    print(f"AnnData object: {adata.shape[0]} features × {adata.shape[1]} cells")
    
    # Sample information
    if 'sample' in adata.obs.columns:
        print(f"Sample ID: {adata.obs['sample'].unique()[0]}")
    
    # Examine the observation (cell) metadata
    print("\n=== CELL METADATA OVERVIEW ===")
    print(f"Number of cells: {adata.n_obs}")
    print(f"Cell metadata fields: {list(adata.obs.columns)}")
    
    # Display first few rows of cell metadata
    print("\nSample of cell metadata:")
    print(adata.obs.head())
    
    # Examine the variable (feature/peak) metadata
    print("\n=== FEATURE/PEAK METADATA OVERVIEW ===")
    print(f"Number of features: {adata.n_vars}")
    if adata.var.shape[1] > 0:
        print(f"Feature metadata fields: {list(adata.var.columns)}")
        
        # Display first few rows of feature metadata
        print("\nSample of feature metadata:")
        print(adata.var.head())
    else:
        print("No feature metadata available")
    
    # Check feature names to understand the peak format
    print("\n=== PEAK FORMAT ===")
    print("First 5 peak names:")
    print(list(adata.var_names[:5]))
    
    # Examine the count matrix
    print("\n=== COUNT MATRIX PROPERTIES ===")
    print(f"Count matrix shape: {adata.X.shape}")
    print(f"Count matrix type: {type(adata.X)}")
    print(f"Count matrix sparsity: {adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.4f} (fraction of non-zero values)")
    
    # Summary statistics of counts
    if hasattr(adata.X, 'toarray'):
        counts_per_cell = adata.X.sum(axis=1).A1
        counts_per_gene = adata.X.sum(axis=0).A1
    else:
        counts_per_cell = adata.X.sum(axis=1)
        counts_per_gene = adata.X.sum(axis=0)
    
    print(f"\nMean counts per cell: {np.mean(counts_per_cell):.2f}")
    print(f"Median counts per cell: {np.median(counts_per_cell):.2f}")
    print(f"Min counts per cell: {np.min(counts_per_cell):.2f}")
    print(f"Max counts per cell: {np.max(counts_per_cell):.2f}")
    
    print(f"\nMean counts per peak: {np.mean(counts_per_gene):.2f}")
    print(f"Median counts per peak: {np.median(counts_per_gene):.2f}")
    print(f"Min counts per peak: {np.min(counts_per_gene):.2f}")
    print(f"Max counts per peak: {np.max(counts_per_gene):.2f}")
    
    # Check for layers (e.g., normalized data)
    print("\n=== AVAILABLE LAYERS ===")
    if adata.layers:
        print(f"Available layers: {list(adata.layers.keys())}")
        for layer_name, layer in adata.layers.items():
            print(f"  Layer '{layer_name}': {layer.shape}, type: {type(layer)}")
    else:
        print("No additional layers found")
    
    # Check for existing dimensionality reduction results
    print("\n=== DIMENSIONALITY REDUCTIONS ===")
    if adata.obsm:
        print(f"Available dimensionality reductions: {list(adata.obsm.keys())}")
        for key, value in adata.obsm.items():
            print(f"  {key}: {value.shape}")
    else:
        print("No dimensionality reductions found")
    
    # Check for existing cluster assignments
    print("\n=== CLUSTER ASSIGNMENTS ===")
    cluster_columns = [col for col in adata.obs.columns if 'cluster' in col.lower() or 'leiden' in col.lower() or 'louvain' in col.lower()]
    if cluster_columns:
        print(f"Potential cluster assignments: {cluster_columns}")
        for col in cluster_columns:
            print(f"  {col} values: {adata.obs[col].unique()}")
    else:
        print("No obvious cluster assignments found")
    
    # Generate diagnostic plots
    print("\n=== GENERATING DIAGNOSTIC PLOTS ===")
    
    # Create a figure directory
    output_dir = os.path.dirname(file_path)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Histogram of counts per cell
    plt.figure(figsize=(8, 6))
    plt.hist(counts_per_cell, bins=100, log=True)
    plt.xlabel('Counts per cell')
    plt.ylabel('Number of cells')
    plt.title('Distribution of counts per cell')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'counts_per_cell_hist.png'))
    plt.close()
    
    # 2. Histogram of counts per peak
    plt.figure(figsize=(8, 6))
    plt.hist(counts_per_gene[counts_per_gene > 0], bins=100, log=True)
    plt.xlabel('Counts per peak')
    plt.ylabel('Number of peaks')
    plt.title('Distribution of counts per peak')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'counts_per_peak_hist.png'))
    plt.close()
    
    # 3. Scatter plot of counts vs features
    plt.figure(figsize=(8, 6))
    plt.scatter(counts_per_cell, np.sum(adata.X > 0, axis=1), alpha=0.1, s=1)
    plt.xlabel('Total counts per cell')
    plt.ylabel('Number of peaks detected per cell')
    plt.title('Counts vs. Features')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'counts_vs_features.png'))
    plt.close()
    
    print(f"Diagnostic plots saved to: {plot_dir}")
    
    # Check if any preprocessed data is available (e.g., PCA, UMAP)
    if 'X_pca' in adata.obsm and 'X_umap' in adata.obsm:
        print("\nGenerating PCA and UMAP plots...")
        
        sc.pl.pca(adata, color='sample', title='PCA colored by sample')
        plt.savefig(os.path.join(plot_dir, 'pca_plot.png'))
        
        sc.pl.umap(adata, color='sample', title='UMAP colored by sample')
        plt.savefig(os.path.join(plot_dir, 'umap_plot.png'))
    else:
        print("\nNo PCA/UMAP found. Would need to run preprocessing to generate these.")
    
    print("\nData exploration complete!")
    
    return adata

if __name__ == "__main__":
    # Use the path that matches the output from the R script
    file_path = "/users/hjiang/GenoDistance/Data/ATAC.h5ad"
    adata = explore_h5ad(file_path)
    
    # After exploration, you can perform additional analysis if needed
    print("\nData is now available in the 'adata' variable for further analysis")
    print("Example commands for further analysis:")
    print("  sc.pp.normalize_total(adata)")
    print("  sc.pp.log1p(adata)")
    print("  sc.pp.highly_variable_genes(adata)")
    print("  sc.tl.pca(adata)")
    print("  sc.pp.neighbors(adata)")
    print("  sc.tl.umap(adata)")
    print("  sc.tl.leiden(adata)")
