import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np
import os

def integration_validation(adata_path, n_genes=5, output_dir='./'):
    """
    Validate integration results by finding marker genes for each cell type in RNA modality.
    Follows standard scanpy preprocessing and differential expression workflow.
    Enhanced with proper sorting for integer-labeled clusters.
    """
    
    # Load and extract RNA cells
    adata = ad.read_h5ad(adata_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # adata_rna = adata[adata.obs['modality'] == 'RNA'].copy()
    adata_rna = adata
    print(f"Found {adata_rna.n_obs} RNA cells, {len(adata_rna.obs['cell_type'].unique())} cell types")
    
    # === FIXED: Proper numerical sorting ===
    def sort_clusters_numerically(cluster_labels):
        """Sort cluster labels numerically if they can be converted to integers"""
        unique_labels = list(cluster_labels.unique())
        print(f"Original cluster labels: {unique_labels}")
        
        try:
            # Convert to integers, sort, then back to original string format
            label_pairs = []
            for label in unique_labels:
                # Handle both string and numeric labels
                if isinstance(label, str):
                    label_pairs.append((int(label), label))
                else:
                    label_pairs.append((int(label), str(label)))
            
            # Sort by integer value
            label_pairs.sort(key=lambda x: x[0])
            sorted_labels = [original_label for _, original_label in label_pairs]
            return sorted_labels
            
        except ValueError as e:
            print(f"Warning: Cannot convert cluster labels to integers: {e}")
            print("Using alphabetical sort instead")
            return sorted(unique_labels)
    
    # Get properly sorted cluster order
    sorted_clusters = sort_clusters_numerically(adata_rna.obs['cell_type'])
    print(f"Sorted cluster order: {sorted_clusters}")
    
    # Convert cell_type to categorical with proper order
    adata_rna.obs['cell_type'] = pd.Categorical(
        adata_rna.obs['cell_type'], 
        categories=sorted_clusters, 
        ordered=True
    )
    
    # Set up scanpy settings
    sc.settings.figdir = output_dir
    sc.settings.set_figure_params(dpi=80, facecolor='white')
    
    # === Data validation and cleaning ===
    print("Validating and cleaning data...")
    print(f"Data shape before cleaning: {adata_rna.shape}")
    print(f"Data type: {adata_rna.X.dtype}")
    print(f"Contains inf values: {np.isinf(adata_rna.X.data if hasattr(adata_rna.X, 'data') else adata_rna.X).any()}")
    print(f"Contains nan values: {np.isnan(adata_rna.X.data if hasattr(adata_rna.X, 'data') else adata_rna.X).any()}")
    
    # Replace inf and nan values with 0
    if hasattr(adata_rna.X, 'data'):  # sparse matrix
        adata_rna.X.data = np.nan_to_num(adata_rna.X.data, nan=0.0, posinf=0.0, neginf=0.0)
    else:  # dense matrix
        adata_rna.X = np.nan_to_num(adata_rna.X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Additional filtering
    sc.pp.filter_genes(adata_rna, min_cells=3)
    sc.pp.filter_cells(adata_rna, min_genes=200)
    print(f"Data shape after cleaning: {adata_rna.shape}")
    
    # === Store raw counts ===
    adata_rna.raw = adata_rna.copy()
    
    # === Feature selection ===
    try:
        sc.pp.highly_variable_genes(
            adata_rna, 
            n_top_genes=2000, 
            batch_key='sample' if 'sample' in adata_rna.obs.columns else None,
            flavor='seurat_v3'
        )
    except Exception as e:
        print(f"HVG detection with batch_key failed: {e}")
        print("Trying without batch correction...")
        sc.pp.highly_variable_genes(
            adata_rna, 
            n_top_genes=2000,
            flavor='seurat_v3'
        )
    
    # Preprocessing
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    
    # Check highly variable genes
    if 'highly_variable' not in adata_rna.var.columns:
        print("Warning: No highly variable genes detected, using all genes")
        adata_rna.var['highly_variable'] = True
    
    print(f"Using {adata_rna.var['highly_variable'].sum()} highly variable genes out of {adata_rna.n_vars} total genes")
    
    # === Scale data ===
    sc.pp.scale(adata_rna, max_value=10)
    
    # === Find marker genes with proper cluster ordering ===
    print("Finding marker genes for each cell type...")
    sc.tl.rank_genes_groups(
        adata_rna, 
        'cell_type', 
        groups=sorted_clusters,  # Specify the order explicitly
        use_raw=False,
        method='wilcoxon',
        pts=True,
        tie_correct=True
    )

    adata_rna.raw = adata_rna.copy()
    # === Generate visualizations with sorted clusters ===
    sc.pl.rank_genes_groups_dotplot(
        adata_rna,
        n_genes=n_genes,
        groupby='cell_type',  # Use groupby instead
        show=False,
        save='_marker_genes_dotplot.pdf'
    )
    
    # 2. Heatmap (FIXED: remove unsupported parameter)
    print("Generating heatmap...")
    try:
        # Create heatmap without categories_order (not supported in newer versions)
        sc.pl.rank_genes_groups_heatmap(
            adata_rna,
            n_genes=n_genes,
            show=False,
            save='_marker_genes_heatmap.pdf',
            standard_scale='var',
            show_gene_labels=True  # Show gene labels
        )
    except Exception as e:
        print(f"Heatmap generation failed: {e}")
    
    # === FIXED: Create sorted marker gene tables with proper indexing ===
    print("Creating comprehensive marker gene table...")
    
    result = adata_rna.uns['rank_genes_groups']
    
    # Create marker data with proper cluster ordering and fixed indexing
    marker_data = []
    for group in sorted_clusters:  # Use sorted order
        if group in result['names'].dtype.names:  # Make sure the group exists in results
            group_names = result['names'][group]
            group_scores = result['scores'][group]
            group_logfc = result['logfoldchanges'][group]
            group_pvals = result['pvals'][group]
            group_pvals_adj = result['pvals_adj'][group]
            group_pts = result['pts'][group]
            group_pts_rest = result['pts_rest'][group] if 'pts_rest' in result else None
            
            # Use iloc for position-based indexing to avoid deprecation warning
            for i in range(len(group_names)):
                marker_data.append({
                    'cell_type': group,
                    'gene': group_names.iloc[i] if hasattr(group_names, 'iloc') else group_names[i],
                    'score': group_scores.iloc[i] if hasattr(group_scores, 'iloc') else group_scores[i],
                    'logfoldchange': group_logfc.iloc[i] if hasattr(group_logfc, 'iloc') else group_logfc[i],
                    'pval': group_pvals.iloc[i] if hasattr(group_pvals, 'iloc') else group_pvals[i],
                    'pval_adj': group_pvals_adj.iloc[i] if hasattr(group_pvals_adj, 'iloc') else group_pvals_adj[i],
                    'pts': group_pts.iloc[i] if hasattr(group_pts, 'iloc') else group_pts[i],
                    'pts_rest': group_pts_rest.iloc[i] if hasattr(group_pts_rest, 'iloc') and group_pts_rest is not None else None
                })
    
    marker_genes_df = pd.DataFrame(marker_data)
    
    # Convert cell_type to categorical to maintain order in outputs
    marker_genes_df['cell_type'] = pd.Categorical(
        marker_genes_df['cell_type'], 
        categories=sorted_clusters, 
        ordered=True
    )
    
    # Sort the dataframe by cell_type (maintaining numerical order) and then by score
    marker_genes_df = marker_genes_df.sort_values(['cell_type', 'score'], ascending=[True, False])
    
    # === Filter for significant markers ===
    significant_markers = marker_genes_df[
        (marker_genes_df['pval_adj'] < 0.05) & 
        (marker_genes_df['logfoldchange'] > 0.5) &
        (marker_genes_df['pts'] > 0.25)
    ].copy()
    
    # Save tables (they will be in proper numerical order)
    marker_genes_df.to_csv(f"{output_dir}/marker_genes_full.csv", index=False)
    significant_markers.to_csv(f"{output_dir}/marker_genes_significant.csv", index=False)
    
    # === Top markers summary with proper ordering ===
    top_markers_per_type = []
    for cluster in sorted_clusters:
        cluster_markers = significant_markers[significant_markers['cell_type'] == cluster]
        if len(cluster_markers) > 0:
            top_markers = cluster_markers.nlargest(n_genes, 'score')
            top_markers_per_type.append(top_markers)
    
    if top_markers_per_type:
        top_markers_per_type = pd.concat(top_markers_per_type, ignore_index=True)
    else:
        top_markers_per_type = pd.DataFrame()
    
    top_markers_per_type.to_csv(f"{output_dir}/top_markers_per_celltype.csv", index=False)
    
    # === ENHANCED: Create a properly ordered gene list for manual heatmap ===
    # Extract top genes in cluster order for potential manual plotting
    ordered_genes = []
    for cluster in sorted_clusters:
        cluster_markers = significant_markers[significant_markers['cell_type'] == cluster]
        if len(cluster_markers) > 0:
            top_genes = cluster_markers.nlargest(n_genes, 'score')['gene'].tolist()
            ordered_genes.extend(top_genes)
    
    # Save ordered gene list
    with open(f"{output_dir}/ordered_marker_genes.txt", 'w') as f:
        f.write('\n'.join(ordered_genes))
    
    # === Quality metrics ===
    print("\n=== Marker Gene Quality Summary ===")
    print(f"Total cell types analyzed: {len(sorted_clusters)}")
    print(f"Cluster order: {' -> '.join(map(str, sorted_clusters))}")
    print(f"Total significant markers found: {len(significant_markers)}")
    print(f"Average markers per cell type: {len(significant_markers) / len(sorted_clusters):.1f}")
    
    # Show markers per cluster in numerical order
    print(f"\nMarkers per cluster (in numerical order):")
    for cluster in sorted_clusters:
        cluster_count = len(significant_markers[significant_markers['cell_type'] == cluster])
        print(f"  Cluster {cluster}: {cluster_count} significant markers")
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"Ordered gene list saved to: {output_dir}/ordered_marker_genes.txt")
    
    return adata_rna, marker_genes_df, significant_markers

# Example usage
if __name__ == "__main__":
    # adata_path = "/dcl01/hongkai/data/data/hjiang/Test/result/harmony/adata_sample.h5ad"
    # adata_rna, marker_genes, significant_markers = integration_validation(
    #     adata_path=adata_path,
    #     n_genes=10,
    #     output_dir="/dcl01/hongkai/data/data/hjiang/Test/result"
    # )
    
    adata_path = "/dcl01/hongkai/data/data/hjiang/result/integration/glue/atac_rna_integrated_test.h5ad"
    adata_rna, marker_genes, significant_markers = integration_validation(
        adata_path=adata_path,
        n_genes=10,
        output_dir="/dcl01/hongkai/data/data/hjiang/result/integration/validation"
    )
    