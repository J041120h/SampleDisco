import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Visualization import visualization_harmony
from combat.pycombat import pycombat
from pseudobulk import compute_pseudobulk_dataframes

# Local imports from your project
from HierarchicalConstruction import cell_type_dendrogram
from HVG import find_hvgs
from Grouping import find_sample_grouping

def treecor_harmony(h5ad_path,
                    sample_meta_path,
                    output_dir,
                    cell_meta_path=None,
                    markers=None,
                    cluster_resolution=0.8,
                    num_PCs=20,
                    num_harmony=30,
                    num_features=2000,
                    min_cells=500,
                    min_features=500,
                    pct_mito_cutoff=20,
                    exclude_genes=None,
                    doublet = True,
                    combat = True,
                    method='average',
                    metric='euclidean',
                    distance_mode='centroid',
                    vars_to_regress=[],
                    verbose=True):
    """
    Harmony Integration with proportional HVG selection by cell type,
    now reading an existing H5AD file that only contains raw counts (no meta).
    """

    # 0. Create output directories if not present
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating output directory")

    # Append 'harmony' subdirectory
    output_dir = os.path.join(output_dir, 'harmony')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating harmony subdirectory")

    # 1. Read the raw count data from an existing H5AD
    if verbose:
        print('=== Read input dataset ===')
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f'Dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    vars_to_regress_for_harmony = vars_to_regress.copy()
    if "sample" not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append("sample")

    if cell_meta_path is None:
        adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path)
        cell_meta.set_index('barcode', inplace=True)
        adata.obs = adata.obs.join(cell_meta, how='left')

    sample_meta = pd.read_csv(sample_meta_path)
    adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_features)

    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()

    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()

    cell_counts_per_patient = adata.obs.groupby('sample').size()
    patients_to_keep = cell_counts_per_patient[cell_counts_per_patient >= min_cells].index
    adata = adata[adata.obs['sample'].isin(patients_to_keep)].copy()

    min_cells_for_gene = int(0.01 * adata.n_obs)
    sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)

    if verbose:
        print(f'Dimension of processed data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # Doublet detection is now optional
    if doublet:
        sc.pp.scrublet(adata, batch_key="sample")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if verbose:
        print("Preprocessing complete!")
    # ----------------------------------------------------------------------
    # We now split the data for two analyses:
    #    (a) Clustering with Harmony (adata_cluster)
    #    (b) Sample differences (adata_sample_diff)
    # ----------------------------------------------------------------------

    adata_cluster = adata.copy()      # with batch effect correction
    adata_sample_diff = adata.copy()  # without batch effect correction

    # ==================== (a) Clustering with Harmony ====================
    if verbose:
        print('=== Processing data for clustering (mediating batch effects) ===')

    adata_cluster.raw = adata_cluster.copy()

    # Find highly variable genes (HVGs)
    sc.pp.highly_variable_genes(
        adata_cluster,
        n_top_genes=num_features,
        flavor='seurat_v3',
        batch_key='sample'
    )

    adata_cluster = adata_cluster[:, adata_cluster.var['highly_variable']].copy()  
    sc.tl.pca(adata_cluster, n_comps=num_PCs, svd_solver='arpack')
    # Harmony batch correction
    if verbose:
        print('=== Running Harmony integration for clustering ===')
        print('Variables to be regressed out: ', ','.join(vars_to_regress))
        print(f'Clustering cluster_resolution: {cluster_resolution}')

    ho = hm.run_harmony(adata_cluster.obsm['X_pca'], adata_cluster.obs, vars_to_regress_for_harmony, max_iter_harmony=num_harmony, max_iter_kmeans=50)
    adata_cluster.obsm['X_pca_harmony'] = ho.Z_corr.T

    if verbose:
        print("End of harmony for adata_cluster.")

    if 'celltype' in adata_cluster.obs.columns:
        adata_cluster.obs['cell_type'] = adata_cluster.obs['celltype'].astype(str)
        if markers is not None:
            marker_dict = {i: markers[i - 1] for i in range(1, len(markers) + 1)}
            adata_cluster.obs['cell_type'] = adata_cluster.obs['cell_type'].map(marker_dict)
    else:
        sc.tl.leiden(
            adata_cluster,
            resolution=cluster_resolution,
            flavor='igraph',
            n_iterations=1,
            directed=False,
            key_added='cell_type'
        )
        # Convert cluster IDs to "1, 2, 3..."
        adata_cluster.obs['cell_type'] = (adata_cluster.obs['leiden'].astype(int) + 1).astype('category')
    if verbose:
        print("Finish find cell type")

    sc.tl.rank_genes_groups(adata_cluster, groupby='cell_type', method='logreg', n_genes=100)
    rank_results = adata_cluster.uns['rank_genes_groups']
    groups = rank_results['names'].dtype.names
    all_marker_genes = []
    for group in groups:
        all_marker_genes.extend(rank_results['names'][group])
    all_marker_genes = list(set(all_marker_genes))

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
    sc.pp.neighbors(adata_cluster, use_rep='X_pca_harmony', n_pcs=num_PCs)
    sc.tl.umap(adata_cluster, min_dist=0.5)

    # Save results
    adata_cluster.write(os.path.join(output_dir, 'adata_cell.h5ad'))

    # ============== (b) Sample Differences (No batch correction) ==========
    if verbose:
        print('=== Processing data for sample differences (without batch effect correction) ===')

    # Ensure 'cell_type' is carried over
    if 'cell_type' not in adata_cluster.obs.columns or adata_cluster.obs['cell_type'].nunique() == 0:
        adata_cluster.obs['cell_type'] = '1'
    adata_sample_diff.obs['cell_type'] = adata_cluster.obs['cell_type']

    if verbose:
        print('=== Preprocessing ===')
    find_hvgs(
        adata=adata_sample_diff,
        sample_column='sample',
        num_features=num_features,
        batch_key='cell_type',
        check_values=True,
        inplace=True
    )
    adata_sample_diff = adata_sample_diff[:, adata_sample_diff.var['highly_variable']].copy()

    if verbose:
        print('=== HVG ===')

    sc.tl.pca(adata_sample_diff, n_comps=num_PCs, svd_solver='arpack', zero_center=True)

    if verbose:
        print('=== Begin Harmony ===')
    ho = hm.run_harmony(adata_sample_diff.obsm['X_pca'], adata_sample_diff.obs, ['batch'], max_iter_harmony=num_harmony, max_iter_kmeans=50)
    adata_sample_diff.obsm['X_pca_harmony'] = ho.Z_corr.T
    
    sc.pp.neighbors(adata_sample_diff, use_rep='X_pca_harmony', n_pcs=num_PCs, n_neighbors=15, metric='cosine')
    sc.tl.umap(adata_sample_diff, min_dist=0.3, spread=1.0)

    # At this point, we have create our comparison group as well as find the correct grouping of cell. 
    # Now all our operation should be based on a new round of operation of our new sample_diff adata.
    sc.write(os.path.join(output_dir, 'adata_sample.h5ad'), adata_sample_diff)
    return adata_cluster, adata_sample_diff