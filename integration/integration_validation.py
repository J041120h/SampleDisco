import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np
import pyensembl
import os
import matplotlib.pyplot as plt
from scipy.sparse import issparse

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
    
    adata_rna = [adata.obs['modality'] == 'RNA'].copy()
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
    
    # Remove duplicates while preserving order
    seen = set()
    ordered_genes_unique = []
    for gene in ordered_genes:
        if gene not in seen:
            seen.add(gene)
            ordered_genes_unique.append(gene)
    
    # Save ordered gene list
    with open(f"{output_dir}/ordered_marker_genes.txt", 'w') as f:
        f.write('\n'.join(ordered_genes_unique))
    
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
    
    # === Create ordered heatmap as complementary visualization ===
    print("Creating complementary ordered heatmap...")
    create_ordered_heatmap(adata_rna, marker_genes_df, n_genes=n_genes, 
                          output_dir=output_dir, ordered_genes=ordered_genes_unique)
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"Ordered gene list saved to: {output_dir}/ordered_marker_genes.txt")
    
    return adata_rna, marker_genes_df, significant_markers


def create_ordered_heatmap(adata_rna, marker_genes_df, n_genes=5, output_dir='./', 
                          ordered_genes=None, suffix=''):
    """
    Create a properly ordered heatmap using matplotlib/seaborn for full control over ordering
    Returns the ordered list of marker genes for further use
    
    Parameters:
    -----------
    adata_rna : AnnData
        RNA expression data
    marker_genes_df : DataFrame
        Marker genes dataframe
    n_genes : int
        Number of genes per cluster
    output_dir : str
        Output directory
    ordered_genes : list
        Pre-defined gene order (if None, will be calculated)
    suffix : str
        Suffix for output filenames (e.g., '_atac' for ATAC data)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    # Get sorted clusters
    sorted_clusters = sorted(adata_rna.obs['cell_type'].cat.categories, 
                           key=lambda x: int(x) if str(x).isdigit() else float('inf'))
    
    # Use provided gene order or calculate it
    if ordered_genes is None:
        # Get top genes per cluster in order
        ordered_genes = []
        for cluster in sorted_clusters:
            cluster_markers = marker_genes_df[marker_genes_df['cell_type'] == cluster]
            if len(cluster_markers) > 0:
                top_genes = cluster_markers.nlargest(n_genes, 'score')['gene'].tolist()
                ordered_genes.extend(top_genes)
        
        # Remove duplicates while preserving order
        unique_ordered_genes = []
        seen = set()
        for gene in ordered_genes:
            if gene not in seen:
                unique_ordered_genes.append(gene)
                seen.add(gene)
    else:
        unique_ordered_genes = ordered_genes
    
    # Create expression matrix
    gene_mask = adata_rna.var.index.isin(unique_ordered_genes)
    available_genes = [g for g in unique_ordered_genes if g in adata_rna.var.index]
    
    if len(available_genes) == 0:
        print("No marker genes found in the data")
        return unique_ordered_genes
    
    # Get expression data
    expr_data = adata_rna[:, available_genes].X
    if hasattr(expr_data, 'toarray'):
        expr_data = expr_data.toarray()
    
    # Calculate mean expression per cluster
    cluster_means = []
    for cluster in sorted_clusters:
        cluster_mask = adata_rna.obs['cell_type'] == cluster
        if cluster_mask.sum() > 0:
            cluster_mean = expr_data[cluster_mask, :].mean(axis=0)
            cluster_means.append(cluster_mean)
    
    cluster_means = np.array(cluster_means)
    
    # Create DataFrame for plotting with genes in the exact order we want
    heatmap_df = pd.DataFrame(
        cluster_means.T,
        index=available_genes,
        columns=sorted_clusters
    )
    
    # Create heatmap
    plt.figure(figsize=(len(sorted_clusters) * 0.8, len(available_genes) * 0.3))
    sns.heatmap(heatmap_df, 
                cmap='RdYlBu_r', 
                center=0, 
                cbar_kws={'label': 'Mean Expression'},
                xticklabels=True,
                yticklabels=True)
    
    plt.title(f'Marker Genes Heatmap (Numerically Ordered Clusters){" - " + suffix if suffix else ""}')
    plt.xlabel('Clusters')
    plt.ylabel('Genes')
    plt.tight_layout()
    
    # Save with suffix
    plt.savefig(f'{output_dir}/ordered_marker_genes_heatmap{suffix}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/ordered_marker_genes_heatmap{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Custom ordered heatmap saved to {output_dir}")
    
    # Return the ordered genes for use in subsequent steps
    return available_genes


def atac_integration_validation(integrated_adata_path, rna_validation_output_dir, gene_activity_path, 
                                output_dir='./atac_validation', ensembl_version=98,
                                create_ordered_heatmap_func=None):
    """
    Validate ATAC integration results by visualizing gene activity scores for the same marker genes 
    identified in RNA modality.
    
    Parameters:
    -----------
    integrated_adata_path : str
        Path to the integrated h5ad file containing both RNA and ATAC modalities
    rna_validation_output_dir : str
        Directory containing RNA validation outputs (to get marker genes and order)
    gene_activity_path : str
        Path to the gene activity matrix h5ad file
    output_dir : str
        Output directory for ATAC validation results
    ensembl_version : int
        Ensembl version for gene name conversion (default: 98)
    create_ordered_heatmap_func : function
        The create_ordered_heatmap function from RNA validation (optional)
    
    Returns:
    --------
    tuple : (adata_atac, gene_activity, marker_genes_used)
    """
    import os
    import pandas as pd
    import anndata as ad
    import scanpy as sc
    import numpy as np
    from scipy.sparse import issparse
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=== ATAC Integration Validation ===")
    
    # Step 1: Load integrated data and extract ATAC cells
    print("Loading integrated data...")
    adata_integrated = ad.read_h5ad(integrated_adata_path)
    adata_atac = adata_integrated[adata_integrated.obs['modality'] == 'ATAC'].copy()
    print(f"Found {adata_atac.n_obs} ATAC cells")
    
    # Step 2: Load gene activity matrix
    print("Loading gene activity matrix...")
    gene_activity = ad.read_h5ad(gene_activity_path)
    print(f"Gene activity shape: {gene_activity.shape}")
    
    # Step 3: Load marker genes from RNA validation with exact order
    print("Loading RNA marker genes...")
    marker_genes_df = pd.read_csv(f"{rna_validation_output_dir}/marker_genes_significant.csv")
    ordered_genes_file = f"{rna_validation_output_dir}/ordered_marker_genes.txt"
    
    if os.path.exists(ordered_genes_file):
        with open(ordered_genes_file, 'r') as f:
            ordered_marker_genes = [gene.strip() for gene in f.readlines()]
        print(f"Loaded exact gene order from RNA analysis: {len(ordered_marker_genes)} genes")
    else:
        # Fallback: get genes from marker_genes_df
        sorted_clusters = get_sorted_clusters(marker_genes_df['cell_type'])
        ordered_marker_genes = []
        for cluster in sorted_clusters:
            cluster_markers = marker_genes_df[marker_genes_df['cell_type'] == cluster]
            if len(cluster_markers) > 0:
                top_genes = cluster_markers.nlargest(5, 'score')['gene'].tolist()
                ordered_marker_genes.extend(top_genes)
        # Remove duplicates while preserving order
        seen = set()
        ordered_marker_genes = [g for g in ordered_marker_genes if not (g in seen or seen.add(g))]
    
    print(f"Found {len(ordered_marker_genes)} ordered marker genes from RNA analysis")
    
    # Step 4: Check if gene names need conversion
    print("Checking gene naming convention...")
    sample_gene_names = gene_activity.var.index[:10].tolist()
    needs_conversion = any(name.startswith('ENSG') for name in sample_gene_names)
    
    if needs_conversion:
        print("Gene activity matrix uses Ensembl IDs. Converting to gene symbols...")
        gene_activity = convert_ensembl_to_symbol(gene_activity, ensembl_version)
    else:
        print("Gene activity matrix already uses gene symbols")
    
    # Step 5: Align cells between integrated ATAC and gene activity
    print("Aligning cells between integrated data and gene activity matrix...")
    common_cells = list(set(adata_atac.obs.index) & set(gene_activity.obs.index))
    print(f"Found {len(common_cells)} common cells")
    
    if len(common_cells) == 0:
        raise ValueError("No common cells found between integrated ATAC and gene activity matrix!")
    
    # Subset both objects to common cells
    adata_atac = adata_atac[common_cells].copy()
    gene_activity = gene_activity[common_cells].copy()
    
    # Transfer cell type annotations
    gene_activity.obs['cell_type'] = adata_atac.obs['cell_type'].copy()
    
    # Step 6: Get cluster order from RNA analysis
    sorted_clusters = get_sorted_clusters(adata_atac.obs['cell_type'])
    print(f"Cluster order: {sorted_clusters}")
    
    # Convert cell_type to categorical with proper order
    gene_activity.obs['cell_type'] = pd.Categorical(
        gene_activity.obs['cell_type'], 
        categories=sorted_clusters, 
        ordered=True
    )
    
    # Step 7: Filter genes to match RNA marker genes IN THE EXACT SAME ORDER
    # Check which genes from the ordered list are available in gene activity
    available_genes_ordered = []
    for gene in ordered_marker_genes:
        if gene in gene_activity.var.index:
            available_genes_ordered.append(gene)
    
    print(f"Found {len(available_genes_ordered)} marker genes in gene activity matrix (in RNA order)")
    
    if len(available_genes_ordered) == 0:
        print("WARNING: No marker genes found in gene activity matrix!")
        print("Sample of ordered marker genes:", ordered_marker_genes[:5])
        print("Sample of gene activity genes:", gene_activity.var.index[:5].tolist())
        raise ValueError("No matching genes found!")
    
    # Step 8: Preprocess gene activity data for visualization
    print("Preprocessing gene activity data...")
    gene_activity_subset = gene_activity[:, available_genes_ordered].copy()
    
    # Normalize and log transform if needed
    if gene_activity_subset.X.min() >= 0:  # Check if already log-transformed
        sc.pp.normalize_total(gene_activity_subset, target_sum=1e4)
        sc.pp.log1p(gene_activity_subset)
    
    print("Creating visualizations...")
    
    # Set up scanpy settings
    sc.settings.figdir = output_dir
    sc.settings.set_figure_params(dpi=80, facecolor='white')
    
    # 9a. Dotplot
    try:
        create_gene_activity_dotplot(gene_activity_subset, available_genes_ordered, available_genes_ordered,
                                    sorted_clusters, output_dir)
    except Exception as e:
        print(f"WARNING: Dotplot creation failed: {e}")
        # Basic dotplot as fallback
        sc.pl.dotplot(gene_activity_subset, available_genes_ordered, groupby='cell_type', 
                     save='_gene_activity_dotplot.png')
    
    # 9b. Heatmap
    try:
        create_gene_activity_heatmap(gene_activity_subset, available_genes_ordered, 
                                    sorted_clusters, output_dir)
    except Exception as e:
        print(f"WARNING: Heatmap creation failed: {e}")
        # Basic heatmap as fallback
        sc.pl.heatmap(gene_activity_subset, available_genes_ordered, groupby='cell_type',
                     save='_gene_activity_heatmap.png')
    
    # 9c. Custom ordered heatmap - using exact same gene order as RNA
    if create_ordered_heatmap_func is not None:
        # Create a mock marker_genes_df for compatibility
        mock_marker_df = pd.DataFrame()
        for i, cluster in enumerate(sorted_clusters):
            for j, gene in enumerate(available_genes_ordered):
                mock_marker_df = pd.concat([mock_marker_df, pd.DataFrame({
                    'cell_type': [cluster],
                    'gene': [gene],
                    'score': [len(available_genes_ordered) - j]  # Higher score for genes that appear earlier
                })], ignore_index=True)
        
        print("Creating ordered heatmap with exact RNA gene order...")
        try:
            # Call with the exact same gene order as RNA
            create_ordered_heatmap_func(gene_activity_subset, mock_marker_df, 
                                       n_genes=len(available_genes_ordered), 
                                       output_dir=output_dir,
                                       ordered_genes=available_genes_ordered,  # Pass exact gene order
                                       suffix='_atac')
        except Exception as e:
            print(f"Error creating ordered heatmap: {e}")
            print("Creating basic heatmap instead...")
            sc.pl.heatmap(gene_activity_subset, available_genes_ordered, groupby='cell_type',
                         save='_gene_activity_heatmap_basic.png')
    else:
        print("create_ordered_heatmap function not provided, skipping custom heatmap")
    
    # Step 10: Summary statistics
    print("\n=== ATAC Gene Activity Summary ===")
    print(f"Total ATAC cells analyzed: {gene_activity_subset.n_obs}")
    print(f"Total marker genes visualized: {len(available_genes_ordered)}")
    print(f"Cell types: {len(sorted_clusters)}")
    
    # Calculate mean gene activity per cell type
    mean_activity = {}
    for ct in sorted_clusters:
        ct_mask = gene_activity_subset.obs['cell_type'] == ct
        if ct_mask.sum() > 0:
            ct_data = gene_activity_subset[ct_mask].X
            if issparse(ct_data):
                ct_data = ct_data.toarray()
            mean_activity[ct] = np.mean(ct_data)
    
    print("\nMean gene activity per cell type:")
    for ct in sorted_clusters:
        if ct in mean_activity:
            print(f"  {ct}: {mean_activity[ct]:.3f}")
    
    # Save gene list used (in the exact order used)
    with open(f"{output_dir}/genes_used_in_atac_validation.txt", 'w') as f:
        for gene in available_genes_ordered:
            f.write(f"{gene}\n")
    
    # Also save a comparison file showing which genes were missing
    missing_genes = [g for g in ordered_marker_genes if g not in available_genes_ordered]
    if missing_genes:
        with open(f"{output_dir}/genes_missing_from_atac.txt", 'w') as f:
            for gene in missing_genes:
                f.write(f"{gene}\n")
        print(f"\n{len(missing_genes)} genes from RNA were not found in ATAC (saved to genes_missing_from_atac.txt)")
    
    print(f"\nAll ATAC validation results saved to: {output_dir}")
    print(f"Gene order is preserved from RNA analysis for direct comparison")
    
    return adata_atac, gene_activity_subset, available_genes_ordered


def convert_ensembl_to_symbol(adata, ensembl_version=98):
    """
    Convert Ensembl IDs to gene symbols using pyensembl
    """
    print(f"Initializing Ensembl database (version {ensembl_version})...")
    
    # Initialize Ensembl database for human
    ensembl = pyensembl.EnsemblRelease(ensembl_version, species='human')
    
    # Download if needed (this might take a while the first time)
    try:
        ensembl.gene_ids()
    except:
        print("Downloading Ensembl database... This may take a few minutes.")
        ensembl.download()
        ensembl.index()
    
    # Convert gene IDs
    print("Converting Ensembl IDs to symbols...")
    gene_id_to_symbol = {}
    unmapped_genes = []
    
    for gene_id in adata.var.index:
        try:
            if gene_id.startswith('ENSG'):
                # Remove version number if present (e.g., ENSG00000000003.14 -> ENSG00000000003)
                gene_id_clean = gene_id.split('.')[0]
                gene_symbol = ensembl.gene_name_of_gene_id(gene_id_clean)
                gene_id_to_symbol[gene_id] = gene_symbol
            else:
                # Already a symbol
                gene_id_to_symbol[gene_id] = gene_id
        except:
            unmapped_genes.append(gene_id)
            gene_id_to_symbol[gene_id] = gene_id  # Keep original if can't convert
    
    print(f"Successfully converted {len(gene_id_to_symbol) - len(unmapped_genes)} genes")
    if unmapped_genes:
        print(f"Could not convert {len(unmapped_genes)} genes")
    
    # Update gene names
    adata.var['ensembl_id'] = adata.var.index
    adata.var.index = [gene_id_to_symbol[gid] for gid in adata.var.index]
    
    # Make gene names unique by appending a suffix to duplicates
    adata.var_names_make_unique()
    
    print(f"Made gene names unique. Total genes: {adata.n_vars}")
    
    return adata


def get_sorted_clusters(cell_type_series):
    """
    Get numerically sorted cluster labels
    """
    unique_labels = list(cell_type_series.unique())
    
    try:
        # Try to sort numerically
        label_pairs = []
        for label in unique_labels:
            if isinstance(label, str):
                label_pairs.append((int(label), label))
            else:
                label_pairs.append((int(label), str(label)))
        
        label_pairs.sort(key=lambda x: x[0])
        sorted_labels = [original_label for _, original_label in label_pairs]
        return sorted_labels
        
    except ValueError:
        # Fall back to alphabetical sort
        return sorted(unique_labels)


def create_gene_activity_dotplot(gene_activity, ordered_genes, var_names, sorted_clusters, output_dir):
    """
    Create dotplot for gene activity matching RNA visualization
    
    Parameters:
    -----------
    gene_activity : AnnData
        Gene activity data
    ordered_genes : list
        Original gene names in order
    var_names : list
        Actual variable names in gene_activity (may have suffixes for duplicates)
    sorted_clusters : list
        Sorted cluster names
    output_dir : str
        Output directory
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.sparse import issparse
    
    print("Creating gene activity dotplot...")
    
    # Calculate statistics for dotplot
    dotplot_data = []
    
    for cluster in sorted_clusters:
        cluster_mask = gene_activity.obs['cell_type'] == cluster
        if cluster_mask.sum() == 0:
            continue
            
        cluster_data = gene_activity[cluster_mask]
        
        for gene_orig, gene_var in zip(ordered_genes, var_names):
            if gene_var in gene_activity.var_names:
                gene_idx = gene_activity.var_names.get_loc(gene_var)
                gene_expr = cluster_data.X[:, gene_idx]
                
                if issparse(gene_expr):
                    gene_expr = gene_expr.toarray().flatten()
                else:
                    gene_expr = gene_expr.flatten()
                
                # Calculate percentage of cells expressing
                pct_expr = (gene_expr > 0).sum() / len(gene_expr) * 100
                
                # Calculate mean expression in expressing cells
                expressing_cells = gene_expr[gene_expr > 0]
                mean_expr = np.mean(expressing_cells) if len(expressing_cells) > 0 else 0
                
                dotplot_data.append({
                    'cell_type': cluster,
                    'gene': gene_orig,  # Use original gene name for display
                    'pct_expr': pct_expr,
                    'mean_expr': mean_expr
                })
    
    dotplot_df = pd.DataFrame(dotplot_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(len(ordered_genes) * 0.5, len(sorted_clusters) * 0.5))
    
    # Pivot data for visualization
    pct_pivot = dotplot_df.pivot(index='cell_type', columns='gene', values='pct_expr')
    expr_pivot = dotplot_df.pivot(index='cell_type', columns='gene', values='mean_expr')
    
    # Ensure correct order
    pct_pivot = pct_pivot.reindex(index=sorted_clusters, columns=ordered_genes)
    expr_pivot = expr_pivot.reindex(index=sorted_clusters, columns=ordered_genes)
    
    # Create scatter plot
    for i, cluster in enumerate(sorted_clusters):
        for j, gene in enumerate(ordered_genes):
            if cluster in pct_pivot.index and gene in pct_pivot.columns:
                size = pct_pivot.loc[cluster, gene]
                color = expr_pivot.loc[cluster, gene]
                
                if not np.isnan(size) and not np.isnan(color):
                    scatter = ax.scatter(j, i, s=size*5, c=color, 
                                       cmap='Reds', vmin=0, vmax=expr_pivot.max().max(),
                                       edgecolors='black', linewidth=0.5)
    
    ax.set_xticks(range(len(ordered_genes)))
    ax.set_xticklabels(ordered_genes, rotation=90, ha='right')
    ax.set_yticks(range(len(sorted_clusters)))
    ax.set_yticklabels(sorted_clusters)
    ax.set_xlabel('Genes')
    ax.set_ylabel('Cell Types')
    ax.set_title('Gene Activity Dotplot (ATAC)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Gene Activity', rotation=270, labelpad=20)
    
    # Add size legend
    for size_pct in [25, 50, 75]:
        ax.scatter([], [], s=size_pct*5, c='gray', edgecolors='black', 
                  linewidth=0.5, label=f'{size_pct}%')
    ax.legend(title='% Expressing', bbox_to_anchor=(1.2, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gene_activity_dotplot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/gene_activity_dotplot.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_gene_activity_heatmap(gene_activity, ordered_genes, sorted_clusters, output_dir):
    """
    Create heatmap for gene activity using the exact gene order from RNA
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.sparse import issparse
    
    print("Creating gene activity heatmap with RNA gene order...")
    
    # Calculate mean expression per cluster
    cluster_means = []
    for cluster in sorted_clusters:
        cluster_mask = gene_activity.obs['cell_type'] == cluster
        if cluster_mask.sum() > 0:
            cluster_data = gene_activity[cluster_mask, ordered_genes].X
            if issparse(cluster_data):
                cluster_data = cluster_data.toarray()
            cluster_mean = cluster_data.mean(axis=0)
            cluster_means.append(cluster_mean)
    
    cluster_means = np.array(cluster_means)
    
    # Create DataFrame for plotting
    heatmap_df = pd.DataFrame(
        cluster_means.T,
        index=ordered_genes,
        columns=sorted_clusters
    )
    
    # Standardize by gene (z-score)
    heatmap_df_zscore = heatmap_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    # Create figure
    plt.figure(figsize=(len(sorted_clusters) * 0.8, len(ordered_genes) * 0.3))
    
    # Create heatmap
    sns.heatmap(heatmap_df_zscore, 
                cmap='RdBu_r', 
                center=0, 
                cbar_kws={'label': 'Z-score'},
                xticklabels=True,
                yticklabels=True,
                vmin=-2, vmax=2)  # Cap at +/- 2 std devs for better visualization
    
    plt.title('Gene Activity Heatmap (ATAC) - Standardized')
    plt.xlabel('Clusters')
    plt.ylabel('Genes (in RNA marker order)')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/gene_activity_heatmap_zscore.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/gene_activity_heatmap_zscore.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create non-standardized version
    plt.figure(figsize=(len(sorted_clusters) * 0.8, len(ordered_genes) * 0.3))
    sns.heatmap(heatmap_df, 
                cmap='RdBu_r', 
                center=0, 
                cbar_kws={'label': 'Mean Expression'},
                xticklabels=True,
                yticklabels=True)
    
    plt.title('Gene Activity Heatmap (ATAC) - Raw Values')
    plt.xlabel('Clusters')
    plt.ylabel('Genes (in RNA marker order)')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/gene_activity_heatmap_raw.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/gene_activity_heatmap_raw.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Gene activity heatmaps created with exact RNA gene order")

if __name__ == "__main__":
    # adata_path = "/dcl01/hongkai/data/data/hjiang/Test/result/harmony/adata_sample.h5ad"
    # adata_rna, marker_genes, significant_markers = integration_validation(
    #     adata_path=adata_path,
    #     n_genes=10,
    #     output_dir="/dcl01/hongkai/data/data/hjiang/Test/result"
    # )
    
    adata_path = "/dcl01/hongkai/data/data/hjiang/result/integration/glue/atac_rna_integrated.h5ad"
    adata_rna, marker_genes, significant_markers = integration_validation(
        adata_path=adata_path,
        n_genes=10,
        output_dir="/dcl01/hongkai/data/data/hjiang/result/integration/validation"
    )

    # Example paths
    integrated_path = "/dcl01/hongkai/data/data/hjiang/result/integration/glue/atac_rna_integrated.h5ad"
    rna_output = "/dcl01/hongkai/data/data/hjiang/result/integration/validation"
    gene_activity_path = "/dcl01/hongkai/data/data/hjiang/result/gene_activity/gene_activity_weighted_gpu.h5ad"
    
    # Run ATAC validation
    adata_atac, gene_activity, genes_used = atac_integration_validation(
        integrated_adata_path=integrated_path,
        rna_validation_output_dir=rna_output,
        gene_activity_path=gene_activity_path,
        output_dir="/dcl01/hongkai/data/data/hjiang/result/integration/validation/gene_activity",
        ensembl_version=98,
        create_ordered_heatmap_func=create_ordered_heatmap  # Pass the function if available
    )