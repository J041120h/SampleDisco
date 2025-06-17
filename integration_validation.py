import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad

def integration_validation(adata_path, n_genes=5, save_plot=True, output_dir='./', 
                          figsize=(12, 8), **kwargs):
    """
    Validate integration results by finding marker genes for RNA modality clusters.
    
    Parameters:
    -----------
    adata_path : str
        Path to the integrated AnnData object
    n_genes : int, default=5
        Number of top marker genes per cluster to include in heatmap
    save_plot : bool, default=True
        Whether to save the heatmap plot
    output_dir : str, default='./'
        Directory to save output files
    figsize : tuple, default=(12, 8)
        Figure size for the heatmap
    **kwargs : additional arguments
        Additional arguments passed to sc.pl.rank_genes_groups_heatmap
    
    Returns:
    --------
    adata_rna : AnnData
        Subset AnnData object containing only RNA modality cells
    marker_genes_df : DataFrame
        DataFrame containing marker genes for each cluster
    """
    
    # Load the integrated AnnData object
    print("Loading integrated AnnData object...")
    adata = ad.read_h5ad(adata_path)
    
    # Extract only RNA modality cells
    print("Extracting RNA modality cells...")
    if 'modality' not in adata.obs.columns:
        raise ValueError("'modality' column not found in adata.obs")
    
    adata_rna = adata[adata.obs['modality'] == 'RNA'].copy()
    print(f"Found {adata_rna.n_obs} RNA cells out of {adata.n_obs} total cells")
    
    # Check if leiden clustering exists
    if 'leiden' not in adata_rna.obs.columns:
        raise ValueError("'leiden' column not found in adata.obs. Please run leiden clustering first.")
    
    print(f"Found {len(adata_rna.obs['leiden'].unique())} leiden clusters")
    
    # Set up scanpy settings
    sc.settings.verbosity = 3
    sc.settings.figdir = output_dir
    
    # Find marker genes for each cluster
    print("Finding marker genes for each cluster...")
    sc.tl.rank_genes_groups(
        adata_rna, 
        'leiden', 
        method='wilcoxon',
        key_added='rank_genes_groups'
    )
    
    # Create a DataFrame of marker genes
    result = adata_rna.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    marker_genes_df = pd.DataFrame({
        group + '_' + key: result[key][group]
        for group in groups for key in ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']
    })
    
    # Save marker genes to CSV
    marker_genes_file = f"{output_dir}/marker_genes_by_cluster.csv"
    marker_genes_df.to_csv(marker_genes_file, index=False)
    print(f"Marker genes saved to: {marker_genes_file}")
    
    # Generate heatmap of top marker genes
    print(f"Generating heatmap of top {n_genes} marker genes per cluster...")
    
    # Set default kwargs for heatmap
    heatmap_kwargs = {
        'n_genes': n_genes,
        'show_gene_labels': True,
        'figsize': figsize,
        'save': 'marker_genes_heatmap.pdf' if save_plot else None,
        'dendrogram': True,
        'var_group_rotation': 90
    }
    heatmap_kwargs.update(kwargs)
    
    # Create the heatmap
    sc.pl.rank_genes_groups_heatmap(
        adata_rna,
        **heatmap_kwargs
    )
    
    # Also create a dot plot for additional visualization
    print("Generating dot plot of marker genes...")
    sc.pl.rank_genes_groups_dotplot(
        adata_rna,
        n_genes=n_genes,
        save='_marker_genes_dotplot.pdf' if save_plot else None,
        figsize=figsize
    )
    
    # Print summary statistics
    print("\n=== Integration Validation Summary ===")
    print(f"Total RNA cells: {adata_rna.n_obs}")
    print(f"Number of genes: {adata_rna.n_vars}")
    print(f"Number of clusters: {len(adata_rna.obs['leiden'].unique())}")
    print(f"Clusters found: {sorted(adata_rna.obs['leiden'].unique())}")
    
    # Print cluster sizes
    cluster_counts = adata_rna.obs['leiden'].value_counts().sort_index()
    print("\nCluster sizes:")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} cells")
    
    if save_plot:
        print(f"\nPlots saved to: {output_dir}")
    
    return adata_rna, marker_genes_df


# Example usage:
if __name__ == "__main__":
    # Example usage of the function
    adata_path = "path/to/your/integrated_data.h5ad"
    
    # Basic usage
    adata_rna, marker_genes = integration_validation(adata_path)
    
    # Advanced usage with custom parameters
    adata_rna, marker_genes = integration_validation(
        adata_path=adata_path,
        n_genes=10,  # Show top 10 genes per cluster
        save_plot=True,
        output_dir="./validation_results/",
        figsize=(15, 10),
        # Additional scanpy heatmap parameters
        show_gene_labels=True,
        dendrogram=True,
        var_group_rotation=45
    )