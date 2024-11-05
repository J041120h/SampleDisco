import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from scipy.sparse import issparse

def treecor_harmony(count_path, sample_meta_path, output_dir, cell_meta_path = None, markers = None, num_PCs=20, num_harmony=20, num_features=2000, min_cells=0, min_features=0, pct_mito_cutoff=20, exclude_genes=None, vars_to_regress=['sample'], resolution=0.5, verbose=True):
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
    marker : list, optional
        list that match the cell cluster to specifc cell type. Only be considered when user input cell type
    num_PCs : int, optional
        Number of PCs used in integration (default: 20)
    num_harmony : int, optional
        Number of harmony embedding used in integration (default: 20)
    num_features : int, optional
        Number of features used in integration (default: 2000)
    min_cells : int, optional
        Include features detected in at least this many cells (default: 0)
    min_features : int, optional
        Include cells where at least this many features are detected (default: 0)
    pct_mito_cutoff : float, optional
        Include cells with less than this many percent of mitochondrial percent are detected (default: 20). Ranges from 0 to 100. Will be used as a QC metric to subset the count matrix. Genes starting with 'MT-' are defined as a set of mitochondrial genes.
    exclude_genes : list, optional
        Additional genes to be excluded from integration. Will subset the count matrix.
    vars_to_regress : list, optional
        Variables to be regressed out during Harmony integration (default: ['sample'])
    resolution : float, optional
        A clustering resolution (default: 0.5). A higher (lower) value indicates larger (smaller) number of cell subclusters.
    verbose : bool, optional
        Show progress

    Returns:
    An AnnData object
    """

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Autiomatically generating output directory")

    # 1. Input data
    if verbose:
        print('=== Read input dataset ===')
    # Read count data
    count = pd.read_csv(count_path, index_col=0)
    
    #potential file combination

    if verbose:
        print(f'Dimension of raw data: {count.shape[0]} genes x {count.shape[1]} cells')
    
    # 2. Filter genes with zero expression across all cells
    count = count[count.sum(axis=1) > 0]
    
    # 3. Harmony Integration
    if verbose:
        print('=== Harmony Integration ===')
    
    # Create AnnData object
    adata = sc.AnnData(count.T)
    adata.var_names = count.index.astype(str) #access gene barcode
    adata.obs_names = count.columns.astype(str) #access cell name
    
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
    
    # Compute highly variable genes per batch
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=num_features, batch_key='sample')
    
    # Normalize data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata_raw = adata.copy()
    # Scale data
    sc.pp.scale(adata, max_value=10)
    
    # Run PCA
    sc.tl.pca(adata, n_comps=num_PCs, svd_solver='arpack', mask_var='highly_variable')
    
    if verbose:
        print('Variables to be regressed out: ', ','.join(vars_to_regress))
        print(f'Clustering resolution: {resolution}')
    
    # Run Harmony integration
    pca_embeddings = adata.obsm['X_pca']
    ho = hm.run_harmony(pca_embeddings, adata.obs, vars_to_regress)
    adata.obsm['X_pca_harmony'] = ho.Z_corr.T
    
    # Run UMAP on Harmony embeddings
    sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_pcs=num_harmony)
    sc.tl.umap(adata)
    
    # Cluster cells
    if 'celltype' in adata.obs.columns:
        adata.obs['leiden'] = adata.obs['celltype'].astype('category')
        if markers != None: 
            marker_dict = {i: markers[i - 1] for i in range(1, len(markers) + 1)}
            adata.obs['leiden'] = adata.obs['leiden'].map(marker_dict)
    else:
        sc.tl.leiden(adata, resolution=resolution, flavor='igraph', n_iterations=2, directed=False)
        adata.obs['leiden'] = (adata.obs['leiden'].astype(int) + 1).astype(str)
    
    # Build dendrogram (phylogenetic tree)
    if verbose:
        print('=== Build Tree ===')
    adata.obs['leiden'] = adata.obs['leiden'].astype('category')
    sc.tl.dendrogram(adata, groupby='leiden')
    sc.pl.dendrogram(adata, groupby='leiden', show=False)
    plt.savefig(os.path.join(output_dir, 'phylo_tree.pdf'))
    plt.close()
    
    if verbose:
        print('=== Generate 2D cluster plot ===')
    plt.figure(figsize=(15, 12))
    sc.pl.umap(
        adata,
        color='leiden',
        legend_loc='right margin',
        frameon=False,
        size=50,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_clusters.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 12))
    sc.pl.umap(
        adata,
        color='sample',
        legend_loc='right margin',
        frameon=False,
        size=10,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_clusters.pdf'), bbox_inches='tight')
    plt.close()
    # Save AnnData object
    adata.write(os.path.join(output_dir, 'integrate.h5ad'))

    #potential marker gene
    
    if verbose:
        print('=== End of preprocessing ===')
    
    return adata