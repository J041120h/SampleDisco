#!/usr/bin/env python3
"""
Script to visualize scGLUE integrated RNA and ATAC data
Generates UMAP embeddings and saves visualizations
"""

import os
import sys
import argparse
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_visualize(rna_path, atac_path, output_dir=None):
    """
    Load processed RNA and ATAC data, create joint UMAP visualization
    
    Parameters:
    -----------
    rna_path : str
        Path to processed RNA h5ad file
    atac_path : str
        Path to processed ATAC h5ad file
    output_dir : str, optional
        Output directory. If None, uses directory of RNA file
    """
    
    # Load the processed data
    print("Loading RNA data...")
    rna = ad.read_h5ad(rna_path)
    
    print("Loading ATAC data...")
    atac = ad.read_h5ad(atac_path)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(rna_path)
    
    # Check if scGLUE embeddings exist
    if "X_glue" not in rna.obsm:
        print("Error: X_glue embeddings not found in RNA data. Run scGLUE integration first.")
        return
    
    if "X_glue" not in atac.obsm:
        print("Error: X_glue embeddings not found in ATAC data. Run scGLUE integration first.")
        return
    
    print("Creating combined dataset...")
    # Combine the datasets for joint visualization
    combined = ad.concat([rna, atac])
    
    # Add modality information
    combined.obs['modality'] = ['RNA'] * rna.n_obs + ['ATAC'] * atac.n_obs
    
    print("Computing UMAP from scGLUE embeddings...")
    # Compute neighbors and UMAP using the scGLUE embeddings
    sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")
    sc.tl.umap(combined)
    
    # Set up plotting parameters
    sc.settings.set_figure_params(dpi=80, facecolor='white', figsize=(8, 6))
    
    print("Generating visualizations...")
    
    # Create visualization by modality
    plt.figure(figsize=(10, 8))
    sc.pl.umap(combined, color="modality", 
               title="scGLUE Integration: RNA vs ATAC",
               save=False, show=False)
    plt.tight_layout()
    modality_plot_path = os.path.join(output_dir, "scglue_umap_modality.png")
    plt.savefig(modality_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved modality plot: {modality_plot_path}")
    
    # Create visualization by cell type (if available)
    if "cell_type" in combined.obs.columns:
        plt.figure(figsize=(12, 8))
        sc.pl.umap(combined, color="cell_type", 
                   title="scGLUE Integration: Cell Types",
                   save=False, show=False, wspace=0.65)
        plt.tight_layout()
        celltype_plot_path = os.path.join(output_dir, "scglue_umap_celltype.png")
        plt.savefig(celltype_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved cell type plot: {celltype_plot_path}")
    
    # Create visualization by domain (if available)
    if "domain" in combined.obs.columns:
        plt.figure(figsize=(12, 8))
        sc.pl.umap(combined, color="domain", 
                   title="scGLUE Integration: Domains",
                   save=False, show=False, wspace=0.65)
        plt.tight_layout()
        domain_plot_path = os.path.join(output_dir, "scglue_umap_domain.png")
        plt.savefig(domain_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved domain plot: {domain_plot_path}")
    
    # Create visualization by sample/batch (if available)
    if "sample" in combined.obs.columns:
        plt.figure(figsize=(12, 8))
        sc.pl.umap(combined, color="sample", 
                   title="scGLUE Integration: Samples/Batches",
                   save=False, show=False, wspace=0.65)
        plt.tight_layout()
        sample_plot_path = os.path.join(output_dir, "scglue_umap_sample.png")
        plt.savefig(sample_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved sample plot: {sample_plot_path}")
    
    # Save the combined dataset with UMAP coordinates
    combined_output_path = os.path.join(output_dir, "scglue_combined_with_umap.h5ad")
    combined.write(combined_output_path)
    print(f"Saved combined dataset: {combined_output_path}")
    
    # Generate summary statistics
    print("\n=== Integration Summary ===")
    print(f"RNA cells: {rna.n_obs}")
    print(f"ATAC cells: {atac.n_obs}")
    print(f"Total cells: {combined.n_obs}")
    print(f"Available metadata columns: {list(combined.obs.columns)}")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, "scglue_integration_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("scGLUE Integration Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"RNA cells: {rna.n_obs}\n")
        f.write(f"ATAC cells: {atac.n_obs}\n")
        f.write(f"Total cells: {combined.n_obs}\n")
        f.write(f"Available metadata: {', '.join(combined.obs.columns)}\n")
        f.write(f"Files generated:\n")
        f.write(f"  - {os.path.basename(modality_plot_path)}\n")
        if "cell_type" in combined.obs.columns:
            f.write(f"  - {os.path.basename(celltype_plot_path)}\n")
        if "domain" in combined.obs.columns:
            f.write(f"  - {os.path.basename(domain_plot_path)}\n")
        if "sample" in combined.obs.columns:
            f.write(f"  - {os.path.basename(sample_plot_path)}\n")
        f.write(f"  - {os.path.basename(combined_output_path)}\n")
    
    print(f"Saved summary: {summary_path}")
    print("\nVisualization complete!")

def main():

    # Run visualization
    load_and_visualize("/users/hjiang/GenoDistance/result/glue/glue-rna-emb.h5ad", "/users/hjiang/GenoDistance/result/glue/glue-atac-emb.h5ad", "/users/hjiang/GenoDistance/result/glue")

if __name__ == "__main__":
    main()