import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt

def treecor_harmony(count_path, sample_meta_path, output_dir, cell_meta_path=None, markers=None, num_PCs=50, num_harmony=20, num_features=2000, min_cells=0, min_features=0, pct_mito_cutoff=20, exclude_genes=None, vars_to_regress=[], resolution=0.5, verbose=True):
    """
    Harmony Integration

    This function is developed using 'Scanpy'. It takes in a raw count gene expression matrix and a sample meta data frame and performs harmony integration.

    Parameters:
    count_path : str
        Path to the raw count gene expression matrix CSV file with genes on rows and cells on columns. Note that cell barcode shall use ':' to separate sample name and barcode (i.e. "sample:barcode")
    sample_meta_path : str
        Path to the sample metadata CSV file. Must contain a column named as 'sample'.
    output_dir : str
        Output directory
    cell_meta_path : str, optional
        Path to the cell metadata CSV file that contains both 'barcode' (cell barcode) and 'sample' columns (its corresponding sample). By default, the sample information is contained in cell barcode with "sample:barcode" format. If your data is not in this format, you should specify this parameter.
    markers : list, optional
        List that matches the cell cluster to specific cell types. Only considered when user inputs cell type.
    num_PCs : int, optional
        Number of PCs used in integration (default: 20)
    num_harmony : int, optional
        Number of harmony embedding dimensions used in integration (default: 20)
    num_features : int, optional
        Number of features used in integration (default: 2000)
    min_cells : int, optional
        Include features detected in at least this many cells (default: 0)
    min_features : int, optional
        Include cells where at least this many features are detected (default: 0)
    pct_mito_cutoff : float, optional
        Include cells with less than this percentage of mitochondrial counts (default: 20). Ranges from 0 to 100. Genes starting with 'MT-' are defined as mitochondrial genes.
    exclude_genes : list, optional
        Additional genes to be excluded from integration. Will subset the count matrix.
    vars_to_regress : list, optional
        Variables to be regressed out during scaling (default: [])
    resolution : float, optional
        A clustering resolution (default: 0.5). A higher (lower) value indicates a larger (smaller) number of cell subclusters.
    verbose : bool, optional
        Show progress

    Returns:
    adata_cluster, adata_sample_diff : AnnData objects
        The first is used for clustering with batch effects mediated.
        The second focuses on finding differences between samples without correcting for sample effects.
    """

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Automatically generating output directory")

    # Append 'harmony' to the output directory path
    output_dir = os.path.join(output_dir, 'harmony')

    # Create the new subdirectory if it doesn’t exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Automatically generating harmony subdirectory")

    # 1. Input data
    if verbose:
        print('=== Read input dataset ===')
    # Read count data
    count = pd.read_csv(count_path, index_col=0)
    
    if verbose:
        print(f'Dimension of raw data: {count.shape[0]} genes x {count.shape[1]} cells')
    
    # 2. Filter genes with zero expression across all cells
    count = count[count.sum(axis=1) > 0]
    
    # 3. Create AnnData object
    if verbose:
        print('=== Creating AnnData object ===')
    adata = sc.AnnData(count.T)
    adata.var_names = count.index.astype(str)  # Access gene names
    adata.obs_names = count.columns.astype(str)  # Access cell names
    
    # Apply min_cells and min_features filters
    if min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if min_features > 0:
        sc.pp.filter_cells(adata, min_genes=min_features)
    
    # Calculate percentage of mitochondrial genes
    # Subset cells based on mitochondrial gene percentage
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()
    
    # Exclude specified genes and mitochondrial genes
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()
    
    if verbose:
        print(f'Dimension of processed data after filtering: {adata.shape[0]} cells x {adata.shape[1]} genes')
    
    # Handle cell metadata
    if cell_meta_path is None:
        adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path)
        cell_meta.set_index('barcode', inplace=True)
        adata.obs = adata.obs.join(cell_meta, how='left')
    
    # Read sample metadata and merge
    sample_meta = pd.read_csv(sample_meta_path)
    adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')

    # --- Split the data processing for two separate analyses ---
    # Copy adata for cluster analysis (with batch effect correction)
    adata_cluster = adata.copy()
    # Copy adata for sample difference analysis (without batch effect correction)
    adata_sample_diff = adata.copy()

    # 4. Process adata_cluster (mediating sample batch effects)
    if verbose:
        print('=== Processing data for clustering (mediating batch effects) ===')
    # Normalize data
    sc.pp.normalize_total(adata_cluster, target_sum=1e4)
    sc.pp.log1p(adata_cluster)
    adata_cluster.raw = adata_cluster.copy()
    # Find HVGs
    sc.pp.highly_variable_genes(adata_cluster, n_top_genes=num_features, flavor='seurat_v3',batch_key='sample')
    adata_cluster = adata_cluster[:, adata_cluster.var['highly_variable']].copy()
    sc.pp.scale(adata_cluster, max_value=10)
    # PCA
    sc.tl.pca(adata_cluster, n_comps=num_PCs, svd_solver='arpack')
    # Harmony integration
    if verbose:
        print('=== Running Harmony integration for clustering ===')
        print('Variables to be regressed out: ', ','.join(vars_to_regress))
        print(f'Clustering resolution: {resolution}')
    vars_to_regress.append("sample")
    ho = hm.run_harmony(adata_cluster.obsm['X_pca'], adata_cluster.obs, vars_to_regress)
    adata_cluster.obsm['X_pca_harmony'] = ho.Z_corr.T
    # Neighbors and UMAP
    sc.pp.neighbors(adata_cluster, use_rep='X_pca_harmony', n_pcs=num_harmony)
    sc.tl.umap(adata_cluster, min_dist=0.5)
    # Cluster cells
    if 'celltype' in adata_cluster.obs.columns:
        adata_cluster.obs['leiden'] = adata_cluster.obs['celltype'].astype('category')
        if markers != None: 
            marker_dict = {i: markers[i - 1] for i in range(1, len(markers) + 1)}
            adata_cluster.obs['leiden'] = adata_cluster.obs['leiden'].map(marker_dict)
    else:
        sc.tl.leiden(adata_cluster, resolution=resolution, flavor='igraph', n_iterations=2, directed=False)
        adata_cluster.obs['leiden'] = (adata_cluster.obs['leiden'].astype(int) + 1).astype(str)
    
    # Build dendrogram (phylogenetic tree)
    if verbose:
        print('=== Build Tree ===')
    adata_cluster.obs['leiden'] = adata_cluster.obs['leiden'].astype('category')
    sc.tl.dendrogram(adata_cluster, groupby='leiden')
    sc.pl.dendrogram(adata_cluster, groupby='leiden', show=False)
    plt.savefig(os.path.join(output_dir, 'phylo_tree.pdf'))
    plt.close()
    
    if verbose:
        print('=== Generate 2D cluster plot ===')
    plt.figure(figsize=(15, 12))
    sc.pl.umap(
        adata_cluster,
        color='leiden',
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
    # Save results
    adata_cluster.write(os.path.join(output_dir, 'adata_cell.h5ad'))

    """
    ==============================================================================================================
    """

    # 5. Process adata_sample_diff (focusing on sample differences)
    if verbose:
        print('=== Processing data for sample differences (without batch effect correction) ===')
    # Normalize data
    sc.pp.normalize_total(adata_sample_diff, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata_sample_diff)
    adata_sample_diff.raw = adata_sample_diff.copy()
    # Calculate HVGs between samples
    # sc.tl.rank_genes_groups(adata_sample_diff, 'sample', method='t-test')
    # deg_genes = adata_sample_diff.uns['rank_genes_groups']['names']
    # deg_genes = pd.DataFrame(deg_genes).stack().reset_index(level=1, drop=True)
    # deg_genes = deg_genes.unique()

    # # Select top DEGs as highly variable genes
    # top_deg_genes = deg_genes[:num_features]
    # adata_sample_diff = adata_sample_diff[:, top_deg_genes].copy()
    # adata_sample_diff.var['highly_variable'] = True
    sample_means = adata_sample_diff.to_df().groupby(adata_sample_diff.obs['sample']).mean()
    gene_variance = sample_means.var(axis=0)
    top_hvg_genes = gene_variance.nlargest(num_features).index
    adata_sample_diff = adata_sample_diff[:, top_hvg_genes].copy()

    sc.pp.scale(adata_sample_diff, max_value=10)
    # PCA
    sc.tl.pca(adata_sample_diff, n_comps=num_PCs, svd_solver='arpack', zero_center=True)
    # Neighbors and UMAP
    ho = hm.run_harmony(adata_sample_diff.obsm['X_pca'], adata_sample_diff.obs, vars_to_regress)
    adata_sample_diff.obsm['X_pca_harmony'] = ho.Z_corr.T
    sc.pp.neighbors(adata_sample_diff, use_rep='X_pca_harmony', n_pcs=num_harmony,n_neighbors=15, metric='cosine')
    sc.tl.umap(adata_sample_diff, min_dist=0.3, spread=1.0)
    # Cluster cells
    adata_sample_diff.obs['leiden'] = adata_cluster.obs['leiden']

    # Visualization for sample differences
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

    # Build dendrogram (phylogenetic tree) for sample differences
    if verbose:
        print('=== Building dendrogram for sample differences ===')
    adata_sample_diff.obs['leiden'] = adata_sample_diff.obs['leiden'].astype('category')
    sc.tl.dendrogram(adata_sample_diff, groupby='leiden')
    sc.pl.dendrogram(adata_sample_diff, groupby='leiden', show=False)
    plt.savefig(os.path.join(output_dir, 'phylo_tree_sample_diff.pdf'))
    plt.close()

    # Save results
    adata_sample_diff.write(os.path.join(output_dir, 'adata_sample.h5ad'))

    if verbose:
        print('=== End of processing ===')

    return adata_cluster, adata_sample_diff
