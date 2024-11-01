import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from scipy.sparse import issparse

def treecor_harmony(count_path, sample_meta_path, output_dir, cell_meta_path=None, num_PCs=20, num_harmony=20, num_features=2000, min_cells=0, min_features=0, pct_mito_cutoff=20, exclude_genes=None, vars_to_regress=['sample'], resolution=0.5, verbose=True):
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
    
    # Set random seed
    np.random.seed(12345)
    
    # 1. Input data
    if verbose:
        print('=== Read input dataset ===')
    # Read count data
    count = pd.read_csv(count_path, index_col=0)
    # for count_path, sample_meta_path in zip(counts_path, sample_meta_paths):
    #     # Process count data
    #     temp_count = pd.read_csv(count_path, index_col=0)
    #     temp_count = temp_count.sort_index()
        
    #     # Extract sample name from the count file path
    #     sample_name = extract_sample_name_from_path(count_path)
        
    #     # Prefix cell barcodes with sample name
    #     temp_count.columns = [f"{sample_name}:{cell_barcode}" for cell_barcode in temp_count.columns]
        
    #     # Initialize or concatenate counts
    #     if count is None:
    #         count = temp_count
    #     else:
    #         count = count.sort_index()
    #         if not temp_count.index.equals(count.index):
    #             raise ValueError(f"Gene names do not match between files: {count_path}")
    #         else:
    #             count = pd.concat([count, temp_count], axis=1)
        
    #     # Process sample metadata
    #     temp_meta = pd.read_csv(sample_meta_path)
    #     temp_meta['sample'] = sample_name
    #     sample_meta_list.append(temp_meta)

    # # Combine all sample metadata into a single DataFrame
    # sample_meta = pd.concat(sample_meta_list, ignore_index=True)

    # # Optionally save the combined count matrix and sample metadata
    # count.to_csv('combined_counts.csv')
    # sample_meta.to_csv('combined_sample_meta.csv', index=False)

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
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Subset cells based on mitochondrial gene percentage
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
    else:
        # Cluster cells as usual
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
    sc.pl.umap(adata, color='leiden', legend_loc='right margin', figsize=(12, 8), show=False)
    plt.savefig(os.path.join(output_dir, 'umap_clusters.pdf'))
    plt.close()

    # Save AnnData object
    adata.write(os.path.join(output_dir, 'integrate.h5ad'))
    
    if verbose:
        print('=== End of preprocessing ===')
    # # Find marker genes for each cluster
    # if verbose:
    #     print('=== Find gene markers for each cell cluster ===')
    
    # if issparse(adata.X):
    #     adata.X.data += 1e-6
    # else:
    #     adata.X += 1e-6

    # if issparse(adata.X):
    #     has_nan = np.isnan(adata.X.data).any()
    #     has_zero = np.any(adata.X.data == 0)
    # else:
    #     has_nan = np.isnan(adata.X).any()
    #     has_zero = np.any(adata.X == 0)

    # print(f"Contains NaNs: {has_nan}, Contains Zeros: {has_zero}")

    # sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    # markers = sc.get.rank_genes_groups_df(adata, group=None)
    # markers.to_csv(os.path.join(output_dir, 'markers.csv'), index=False)
    
    # # Get top 10 markers per cluster
    # top10 = markers.groupby('group', observed=True).head(10)
    # top10.to_csv(os.path.join(output_dir, 'markers_top10.csv'), index=False)
    
    return adata