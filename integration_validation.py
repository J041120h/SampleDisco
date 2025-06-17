import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np
import os

def integration_validation(adata_path, n_genes=5, output_dir='./'):
    """
    Validate integration results by finding marker genes for each cell type in RNA modality.
    Follows standard scanpy preprocessing and differential expression workflow.
    """
    
    # Load and extract RNA cells
    adata = ad.read_h5ad(adata_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    adata_rna = adata[adata.obs['modality'] == 'RNA'].copy()
    print(f"Found {adata_rna.n_obs} RNA cells, {len(adata_rna.obs['cell_type'].unique())} cell types")
    
    # Set up scanpy settings
    sc.settings.figdir = output_dir
    sc.settings.set_figure_params(dpi=80, facecolor='white')
    
    # === MISSING STEP 1: Data validation and cleaning ===
    print("Validating and cleaning data...")
    # Check for and handle problematic values
    print(f"Data shape before cleaning: {adata_rna.shape}")
    print(f"Data type: {adata_rna.X.dtype}")
    print(f"Contains inf values: {np.isinf(adata_rna.X.data if hasattr(adata_rna.X, 'data') else adata_rna.X).any()}")
    print(f"Contains nan values: {np.isnan(adata_rna.X.data if hasattr(adata_rna.X, 'data') else adata_rna.X).any()}")
    
    # Replace inf and nan values with 0
    if hasattr(adata_rna.X, 'data'):  # sparse matrix
        adata_rna.X.data = np.nan_to_num(adata_rna.X.data, nan=0.0, posinf=0.0, neginf=0.0)
    else:  # dense matrix
        adata_rna.X = np.nan_to_num(adata_rna.X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Additional filtering to remove problematic genes
    # Remove genes that are expressed in very few cells (< 3 cells)
    sc.pp.filter_genes(adata_rna, min_cells=3)
    
    # Remove cells with very few genes (< 200 genes)
    sc.pp.filter_cells(adata_rna, min_genes=200)
    
    print(f"Data shape after cleaning: {adata_rna.shape}")
    
    # === MISSING STEP 2: Store raw counts ===
    # This is crucial for proper differential expression analysis
    adata_rna.raw = adata_rna.copy()
    
    # === MISSING STEP 3: Feature selection ===
    # Find highly variable genes before normalization for better marker gene detection
    try:
        sc.pp.highly_variable_genes(
            adata_rna, 
            n_top_genes=2000, 
            batch_key='sample' if 'sample' in adata_rna.obs.columns else None,
            flavor='seurat_v3'  # More robust method
        )
    except Exception as e:
        print(f"HVG detection with batch_key failed: {e}")
        print("Trying without batch correction...")
        sc.pp.highly_variable_genes(
            adata_rna, 
            n_top_genes=2000,
            flavor='seurat_v3'
        )
    
    # Preprocessing (normalization and log transformation)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    
    # === MISSING STEP 4: Use only highly variable genes for analysis ===
    # Check if we have highly variable genes
    if 'highly_variable' not in adata_rna.var.columns:
        print("Warning: No highly variable genes detected, using all genes")
        adata_rna.var['highly_variable'] = True
    
    print(f"Using {adata_rna.var['highly_variable'].sum()} highly variable genes out of {adata_rna.n_vars} total genes")
    
    # === MISSING STEP 5: Scale data ===
    # Scale data to unit variance for better marker gene detection
    sc.pp.scale(adata_rna, max_value=10)
    
    # Find marker genes using the processed data but raw counts for statistics
    print("Finding marker genes for each cell type...")
    sc.tl.rank_genes_groups(
        adata_rna, 
        'cell_type', 
        use_raw = False,
        method='wilcoxon',
        pts=True,      # Calculate fraction of cells expressing each gene
        tie_correct=True  # Apply tie correction for better p-values
    )
    adata_rna.raw = adata_rna.copy() 
    
    # 1. Dot plot (most informative for marker genes)
    print("Generating dot plot...")
    try:
        sc.pl.rank_genes_groups_dotplot(
            adata_rna,
            n_genes=n_genes,
            show=False,
            save='_marker_genes_dotplot.pdf'
        )
    except Exception as e:
        print(f"Dot plot generation failed: {e}")
    
    # 2. Heatmap
    print("Generating heatmap...")
    try:
        sc.pl.rank_genes_groups_heatmap(
            adata_rna,
            n_genes=n_genes,
            show=False,
            save='_marker_genes_heatmap.pdf',
            standard_scale='var'  # Standardize across genes
        )
    except Exception as e:
        print(f"Heatmap generation failed: {e}")
    
    # === MISSING STEP 7: Enhanced marker gene table with more statistics ===
    print("Creating comprehensive marker gene table...")
    
    # Extract all information including pts (percentage of cells expressing)
    result = adata_rna.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    # Create a more comprehensive dataframe
    marker_data = []
    for group in groups:
        for i in range(len(result['names'][group])):
            marker_data.append({
                'cell_type': group,
                'gene': result['names'][group][i],
                'score': result['scores'][group][i],
                'logfoldchange': result['logfoldchanges'][group][i],
                'pval': result['pvals'][group][i],
                'pval_adj': result['pvals_adj'][group][i],
                'pts': result['pts'][group][i],  # Fraction of cells in group expressing gene
                'pts_rest': result['pts_rest'][group][i] if 'pts_rest' in result else None  # Fraction in other groups
            })
    
    marker_genes_df = pd.DataFrame(marker_data)
    
    # === MISSING STEP 8: Filter for significant markers ===
    # Filter for significant and meaningful markers
    significant_markers = marker_genes_df[
        (marker_genes_df['pval_adj'] < 0.05) & 
        (marker_genes_df['logfoldchange'] > 0.5) &
        (marker_genes_df['pts'] > 0.25)  # Expressed in at least 25% of cells in the group
    ].copy()
    
    # Save both full and filtered tables
    marker_genes_df.to_csv(f"{output_dir}/marker_genes_full.csv", index=False)
    significant_markers.to_csv(f"{output_dir}/marker_genes_significant.csv", index=False)
    
    # === MISSING STEP 9: Top markers summary ===
    # Create a summary of top markers per cell type
    top_markers_per_type = (significant_markers
                           .groupby('cell_type')
                           .apply(lambda x: x.nlargest(n_genes, 'score'))
                           .reset_index(drop=True))
    top_markers_per_type.to_csv(f"{output_dir}/top_markers_per_celltype.csv", index=False)
    
    # === MISSING STEP 10: Quality metrics ===
    print("\n=== Marker Gene Quality Summary ===")
    print(f"Total cell types analyzed: {len(groups)}")
    print(f"Total significant markers found: {len(significant_markers)}")
    print(f"Average markers per cell type: {len(significant_markers) / len(groups):.1f}")
    print(f"\nAll results saved to: {output_dir}")
    
    return adata_rna, marker_genes_df, significant_markers


# Example usage
if __name__ == "__main__":
    adata_path = "/dcl01/hongkai/data/data/hjiang/result/integration/glue/atac_rna_integrated_test.h5ad"
    adata_rna, marker_genes, significant_markers = integration_validation(
        adata_path=adata_path,
        n_genes=10,
        output_dir="/dcl01/hongkai/data/data/hjiang/result/integration/validation"
    )