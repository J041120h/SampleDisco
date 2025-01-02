import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from HierarchicalConstruction import cell_type_dendrogram
from HVG import find_hvgs

def treecor_harmony(count_path, sample_meta_path, output_dir, cell_meta_path=None, markers=None, cluster_resolution=1, num_PCs=20, num_harmony=5, num_features=2000, min_cells=0, min_features=0, pct_mito_cutoff=20, exclude_genes=None, method='average', metric='euclidean', distance_mode='centroid', vars_to_regress=[], verbose=True):
    """
    Harmony Integration with proportional HVG selection by cell type.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating output directory")

    # Append 'harmony' to the output directory path
    output_dir = os.path.join(output_dir, 'harmony')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating harmony subdirectory")

    # 1. Input data
    if verbose:
        print('=== Read input dataset ===')
    count = pd.read_csv(count_path, index_col=0)
    
    if verbose:
        print(f'Dimension of raw data: {count.shape[0]} genes x {count.shape[1]} cells')
    
    # 2. Filter genes with zero expression
    count = count[count.sum(axis=1) > 0]
    
    # 3. Create AnnData object
    if verbose:
        print('=== Creating AnnData object ===')
    adata = sc.AnnData(count.T)
    adata.var_names = count.index.astype(str)
    adata.obs_names = count.columns.astype(str)
    
    # Filter by min_cells and min_features if provided
    if min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if min_features > 0:
        sc.pp.filter_cells(adata, min_genes=min_features)
    
    # Calculate pct mito
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()
    
    # Exclude genes if requested
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()
    
    if verbose:
        print(f'Dimension of processed data: {adata.shape[0]} cells x {adata.shape[1]} genes')

    # Cell metadata
    if cell_meta_path is None:
        adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path)
        cell_meta.set_index('barcode', inplace=True)
        adata.obs = adata.obs.join(cell_meta, how='left')
    
    # Sample metadata
    sample_meta = pd.read_csv(sample_meta_path)
    adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')

    # Split the data for two analyses
    adata_cluster = adata.copy()      # with batch effect correction
    adata_sample_diff = adata.copy()  # without batch effect correction

    # === Clustering Data with Harmony ===
    if verbose:
        print('=== Processing data for clustering (mediating batch effects) ===')
    # Normalize and log transform for clustering
    sc.pp.normalize_total(adata_cluster, target_sum=1e4)
    sc.pp.log1p(adata_cluster)
    adata_cluster.raw = adata_cluster.copy()

    # HVGs for clustering
    sc.pp.highly_variable_genes(adata_cluster, n_top_genes=num_features, flavor='seurat_v3', batch_key='sample')
    adata_cluster = adata_cluster[:, adata_cluster.var['highly_variable']].copy()
    sc.pp.scale(adata_cluster, max_value=10)

    # PCA and Harmony
    sc.tl.pca(adata_cluster, n_comps=num_PCs, svd_solver='arpack')
    if verbose:
        print('=== Running Harmony integration for clustering ===')
        print('Variables to be regressed out: ', ','.join(vars_to_regress))
        print(f'Clustering cluster_resolution: {cluster_resolution}')

    vars_to_regress_for_harmony = vars_to_regress.copy()
    if "sample" not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append("sample")

    ho = hm.run_harmony(adata_cluster.obsm['X_pca'], adata_cluster.obs, vars_to_regress_for_harmony)
    adata_cluster.obsm['X_pca_harmony'] = ho.Z_corr.T

    # Clustering
    if 'celltype' in adata_cluster.obs.columns:
        adata_cluster.obs['cell_type'] = adata_cluster.obs['celltype'].astype('category')
        if markers is not None:
            marker_dict = {i: markers[i - 1] for i in range(1, len(markers) + 1)}
            adata_cluster.obs['cell_type'] = adata_cluster.obs['cell_type'].map(marker_dict)
    else:
        sc.tl.leiden(adata_cluster, resolution=cluster_resolution, flavor='igraph', n_iterations=2, directed=False, key_added='cell_type')
        adata_cluster.obs['cell_type'] = (adata_cluster.obs['cell_type'].astype(int) + 1).astype(str)

    # Marker genes for dendrogram
    sc.tl.rank_genes_groups(adata_cluster, groupby='cell_type', method='wilcoxon', n_genes=100)
    rank_results = adata_cluster.uns['rank_genes_groups']
    groups = rank_results['names'].dtype.names
    all_marker_genes = []
    for group in groups:
        all_marker_genes.extend(rank_results['names'][group])  
    all_marker_genes = list(set(all_marker_genes))

    # Construct dendrogram
    adata_cluster = cell_type_dendrogram(
        adata=adata_cluster,
        resolution=cluster_resolution,
        groupby='cell_type',
        method='average',
        metric='euclidean',
        distance_mode='centroid',
        marker_genes=all_marker_genes,
        verbose=True
    )

    # Neighbors and UMAP
    sc.pp.neighbors(adata_cluster, use_rep='X_pca_harmony', n_pcs=num_harmony)
    sc.tl.umap(adata_cluster, min_dist=0.5)

    # Dendrogram plot
    if verbose:
        print('=== Build Tree ===')
    sc.tl.dendrogram(adata_cluster, groupby='cell_type')
    sc.pl.dendrogram(adata_cluster, groupby='cell_type', show=False)
    plt.savefig(os.path.join(output_dir, 'phylo_tree.pdf'))
    plt.close()

    # Cluster plots
    if verbose:
        print('=== Generate 2D cluster plot ===')
    plt.figure(figsize=(15, 12))
    sc.pl.umap(
        adata_cluster,
        color='cell_type',
        legend_loc='right margin',
        frameon=False,
        size=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cell_cluster_umap_clusters.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 12))
    sc.pl.umap(
        adata_cluster,
        color='sample',
        legend_loc='right margin',
        frameon=False,
        size=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cell_cluster_sample_clusters.pdf'), bbox_inches='tight')
    plt.close()

    adata_cluster.write(os.path.join(output_dir, 'adata_cell.h5ad'))

    # === Sample Differences (No batch correction) ========================================================
    if verbose:
        print('=== Processing data for sample differences (without batch effect correction) ===')
    # Ensure cell_type is carried over
    if 'cell_type' not in adata_cluster.obs.columns or adata_cluster.obs['cell_type'].nunique() == 0:
        adata_cluster.obs['cell_type'] = '1'
    adata_sample_diff.obs['cell_type'] = adata_cluster.obs['cell_type']

    # At this point, adata_sample_diff should still have raw counts. Perfect for HVG detection.

    # Run HVG detection on raw counts
    find_hvgs(
        adata=adata_sample_diff,
        sample_column='sample',
        num_features = num_features,
        batch_key = "cell_type",
        check_values = True,
        inplace = True
    )
    adata_sample_diff.raw = adata_sample_diff.copy()

    # After HVGs are found, now we normalize and log-transform
    sc.pp.normalize_total(adata_sample_diff, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata_sample_diff)
    adata_sample_diff = adata_sample_diff[:, adata_sample_diff.var['highly_variable']].copy()
    sc.pp.scale(adata_sample_diff, max_value=10)

    # PCA and neighbors/UMAP
    sc.tl.pca(adata_sample_diff, n_comps=num_PCs, svd_solver='arpack', zero_center=True)
    ha = hm.run_harmony(adata_sample_diff.obsm['X_pca'], adata_sample_diff.obs, vars_to_regress_for_harmony)
    # adata_sample_diff.obsm['X_pca_harmony'] = adata_sample_diff.obsm['X_pca']
    adata_sample_diff.obsm['X_pca_harmony'] = ha.Z_corr.T
    sc.pp.neighbors(adata_sample_diff, use_rep='X_pca_harmony', n_pcs=num_harmony, n_neighbors=15, metric='cosine')
    sc.tl.umap(adata_sample_diff, min_dist=0.3, spread=1.0)

    # Visualization
    if verbose:
        print('=== Visualizing sample differences ===')
    plt.figure(figsize=(15, 12))
    sc.pl.umap(
        adata_sample_diff,
        color='sample',
        legend_loc='right margin',
        frameon=False,
        size=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_umap_by_sample.pdf'), bbox_inches='tight')
    plt.close()

    adata_sample_diff.obs['group'] = adata_sample_diff.obs['sample'].apply(
        lambda x: 'HD' if x.startswith('HD') else ('Se' if x.startswith('Se') else 'Other')
    )
    if verbose:
        print('=== Visualizing group differences ===')
    plt.figure(figsize=(15, 12))
    sc.pl.umap(
        adata_sample_diff,
        color='group',
        legend_loc='right margin',
        frameon=False,
        size=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_umap_by_group.pdf'), bbox_inches='tight')
    plt.close()

    adata_cluster.obs['group'] = adata_cluster.obs['sample'].apply(
        lambda x: 'HD' if x.startswith('HD') else ('Se' if x.startswith('Se') else 'Other')
    )
    if verbose:
        print('=== Visualizing cell differences by group ===')
    plt.figure(figsize=(15, 12))
    sc.pl.umap(
        adata_cluster,
        color='group',
        legend_loc='right margin',
        frameon=False,
        size=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cell_umap_by_group.pdf'), bbox_inches='tight')
    plt.close()

    adata_sample_diff.write(os.path.join(output_dir, 'adata_sample.h5ad'))

    # Visualizing sample relationship
    if verbose:
        print('=== Computing average HVG expression per sample ===')
    hvg_genes = adata_sample_diff.var_names[adata_sample_diff.var['highly_variable']]
    hvg_data = adata_sample_diff[:, hvg_genes].to_df()
    hvg_data['sample'] = adata_sample_diff.obs['sample'].values
    sample_means = hvg_data.groupby('sample').mean()

    if verbose:
        print(f'Computed average expression for {sample_means.shape[0]} samples and {sample_means.shape[1]} HVGs.')

    if verbose:
        print('=== Performing PCA to reduce dimensions to 2D ===')
    pca = PCA(n_components=2)
    sample_pca = pca.fit_transform(sample_means)
    pca_df = pd.DataFrame(sample_pca, index=sample_means.index, columns=['PC1', 'PC2'])

    if verbose:
        print('=== Visualizing samples in 2D space ===')
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], s=100)
    for i, sample_name in enumerate(pca_df.index):
        plt.text(
            pca_df.iloc[i]['PC1'],
            pca_df.iloc[i]['PC2'],
            sample_name,
            fontsize=9,
            ha='right'
        )
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Average HVG Expression per Sample')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_relationship_pca.pdf'))
    plt.close()

    if verbose:
        print('=== Sample relationship visualization saved as sample_relationship_pca.pdf ===')
        print('=== End of processing ===')

    return adata_cluster, adata_sample_diff