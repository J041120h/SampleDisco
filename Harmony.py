import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from harmony import harmonize
import time

# Local imports from your project
from HierarchicalConstruction import cell_type_dendrogram

def anndata_cluster(
    adata_cluster,
    output_dir,
    cluster_resolution=0.8,
    markers=None,
    num_features=2000,
    num_PCs=20,
    num_harmony=30,
    vars_to_regress_for_harmony=None,
    method='average',
    metric='euclidean',
    distance_mode='centroid',
    verbose=True
):
    """
    Given an AnnData object, perform:
      - HVG selection (Seurat v3),
      - PCA,
      - Harmony batch correction for clustering,
      - Leiden clustering or label transfer to 'cell_type',
      - Ranking marker genes,
      - Hierarchical dendrogram,
      - UMAP projection.

    Results and the final adata are written to 'adata_cell.h5ad' under output_dir/harmony.
    """
    if verbose:
        print('=== Processing data for clustering (mediating batch effects) ===')

    # Keep a copy of raw
    adata_cluster.raw = adata_cluster.copy()

    # Step A1: HVG selection
    sc.pp.highly_variable_genes(
        adata_cluster,
        n_top_genes=num_features,
        flavor='seurat_v3',
        batch_key='sample'
    )
    adata_cluster = adata_cluster[:, adata_cluster.var['highly_variable']].copy()

    # Step A2: PCA
    sc.tl.pca(adata_cluster, n_comps=num_PCs, svd_solver='arpack')

    if verbose:
        print('=== Running Harmony integration for clustering ===')
        print('Variables to be regressed out: ', ','.join(vars_to_regress_for_harmony or []))
        print(f'Clustering cluster_resolution: {cluster_resolution}')

    # Step A3: Harmony
    Z = harmonize(
        adata_cluster.obsm['X_pca'],
        adata_cluster.obs,
        batch_key = vars_to_regress_for_harmony,
        max_iter_harmony=num_harmony
    )
    adata_cluster.obsm['X_pca_harmony'] = Z

    if verbose:
        print("End of harmony for adata_cluster.")

    # Step A4: Clustering / cell_type annotation
    if 'celltype' in adata_cluster.obs.columns:
        # If there's an existing "celltype" annotation, rename it
        adata_cluster.obs['cell_type'] = adata_cluster.obs['celltype'].astype(str)
        if markers is not None:
            # Optionally map numeric IDs to markers
            marker_dict = {i: markers[i - 1] for i in range(1, len(markers) + 1)}
            adata_cluster.obs['cell_type'] = adata_cluster.obs['cell_type'].map(marker_dict)
    else:
        # Otherwise, run Leiden clustering
        sc.tl.leiden(
            adata_cluster,
            resolution=cluster_resolution,
            flavor='igraph',
            n_iterations=1,
            directed=False,
            key_added='cell_type'
        )
        # Convert cluster IDs to "1, 2, 3, ..."
        adata_cluster.obs['cell_type'] = (adata_cluster.obs['cell_type'].astype(int) + 1).astype('category')

    if verbose:
        print("Finished assigning cell types.")

    # Step A5: Rank marker genes
    sc.tl.rank_genes_groups(adata_cluster, groupby='cell_type', method='logreg', n_genes=100)
    rank_results = adata_cluster.uns['rank_genes_groups']
    groups = rank_results['names'].dtype.names
    all_marker_genes = set()
    for group in groups:
        all_marker_genes.update(rank_results['names'][group])

    # Step A6: Construct dendrogram
    adata_cluster = cell_type_dendrogram(
        adata=adata_cluster,
        resolution=cluster_resolution,
        groupby='cell_type',
        method=method,
        metric=metric,
        distance_mode=distance_mode,
        marker_genes=list(all_marker_genes),
        verbose=verbose
    )

    # Step A7: Neighbors + UMAP using Harmony embedding
    sc.pp.neighbors(adata_cluster, use_rep='X_pca_harmony', n_pcs=num_PCs)
    sc.tl.umap(adata_cluster, min_dist=0.5)

    # Step A8: Write out final
    sc.write(os.path.join(output_dir, 'adata_cell.h5ad'), adata_cluster)
    return adata_cluster


def anndata_sample(
    adata_sample_diff,
    output_dir,
    sample_column='sample',
    num_features=2000,
    num_PCs=20,
    num_harmony=30,
    verbose=True
):
    """
    Given an AnnData object, perform:
      - HVG selection by cell_type,
      - PCA,
      - Harmony integration (here by default on 'batch'),
      - UMAP projection,
      - and writes out 'adata_sample.h5ad'.

    This version does minimal or no batch correction on the front-end so that
    the sample differences remain interpretable.
    """
    if verbose:
        print('=== Processing data for sample differences (without batch effect correction) ===')

    # We are not finding HVGs here, as we are looking for HVG within each cell type after combat
    # find_hvgs(
    #     adata=adata_sample_diff,
    #     sample_column=sample_column,
    #     num_features=num_features,
    #     batch_key='cell_type',    # or whichever key you want
    #     check_values=True,
    #     inplace=True
    # )
    # adata_sample_diff = adata_sample_diff[:, adata_sample_diff.var['highly_variable']].copy()

    # if verbose:
    #     print('=== HVG selected. Performing PCA. ===')

    # # Step B1: PCA
    # sc.tl.pca(adata_sample_diff, n_comps=num_PCs, svd_solver='arpack', zero_center=True)

    # # Step B2: Harmony on 'batch'
    # if verbose:
    #     print('=== Begin Harmony ===')
    
    # Z = harmonize(
    #     adata_sample_diff.obsm['X_pca'],
    #     adata_sample_diff.obs,
    #     batch_key = ['batch'],
    #     max_iter_harmony=num_harmony
    # )
    # adata_sample_diff.obsm['X_pca_harmony'] = Z

    # Step B3: Neighbors + UMAP using Harmony embedding
    # sc.pp.neighbors(adata_sample_diff, use_rep='X_pca_harmony', n_pcs=num_PCs, n_neighbors=15, metric='cosine')
    # sc.tl.umap(adata_sample_diff, min_dist=0.3, spread=1.0)

    # Write out final
    sc.write(os.path.join(output_dir, 'adata_sample.h5ad'), adata_sample_diff)
    return adata_sample_diff

def harmony(
    h5ad_path,
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
    doublet=True,
    combat=True,  # Note: not fully implemented in example
    method='average',
    metric='euclidean',
    distance_mode='centroid',
    vars_to_regress=[],
    verbose=True
):
    """
    Harmony Integration with proportional HVG selection by cell type,
    now reading an existing H5AD file that only contains raw counts (no meta).

    This function:
      1. Reads and preprocesses the data (filter genes/cells, remove MT genes, etc.).
      2. Splits into two branches for:
         (a) adata_cluster used for clustering with Harmony
         (b) adata_sample_diff used for sample-level analysis (minimal batch correction).
      3. Returns both AnnData objects.
    """

    # Start timing
    start_time = time.time()

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

    # We want to regress "sample" plus anything else the user included
    vars_to_regress_for_harmony = vars_to_regress.copy()
    if "sample" not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append("sample")

    # Attach sample info
    if cell_meta_path is None:
        # If no cell metadata provided, assume the obs_names "barcode" has "sample" in front
        adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path)
        cell_meta.set_index('barcode', inplace=True)
        adata.obs = adata.obs.join(cell_meta, how='left')

    sample_meta = pd.read_csv(sample_meta_path)
    adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')

    # Basic filtering
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_features)

    # Mito QC
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()

    # Exclude genes if needed
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()

    # Only keep samples with enough cells
    cell_counts_per_patient = adata.obs.groupby('sample').size()
    patients_to_keep = cell_counts_per_patient[cell_counts_per_patient >= min_cells].index
    adata = adata[adata.obs['sample'].isin(patients_to_keep)].copy()

    # Drop genes that are too rare in these final cells
    min_cells_for_gene = int(0.01 * adata.n_obs)  # e.g., gene must appear in 1% of cells
    sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)

    if verbose:
        print(f'Dimension of processed data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # Optional doublet detection
    if doublet:
        sc.pp.scrublet(adata, batch_key="sample")

    # Normalization & log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if verbose:
        print("Preprocessing complete!")

    # Split data for two separate analyses
    adata_cluster = adata.copy()      # used for clustering (with batch effect correction)
    adata_sample_diff = adata.copy()  # used for sample-level analysis

    # 2(a). Clustering and cell-type annotation
    adata_cluster = anndata_cluster(
        adata_cluster=adata_cluster,
        output_dir=output_dir,
        cluster_resolution=cluster_resolution,
        markers=markers,
        num_features=num_features,
        num_PCs=num_PCs,
        num_harmony=num_harmony,
        vars_to_regress_for_harmony=vars_to_regress_for_harmony,
        method=method,
        metric=metric,
        distance_mode=distance_mode,
        verbose=verbose
    )

    # 2(b). Sample-level analysis
    # Carry over the cell_type from the cluster object if needed
    if 'cell_type' not in adata_cluster.obs.columns or adata_cluster.obs['cell_type'].nunique() == 0:
        adata_cluster.obs['cell_type'] = '1'
    adata_sample_diff.obs['cell_type'] = adata_cluster.obs['cell_type']

    adata_sample_diff = anndata_sample(
        adata_sample_diff=adata_sample_diff,
        output_dir=output_dir,
        sample_column='sample',
        num_features=num_features,
        num_PCs=num_PCs,
        num_harmony=num_harmony,
        verbose=verbose
    )

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print execution time
    if verbose:
        print(f"Function execution time: {elapsed_time:.2f} seconds")

    return adata_cluster, adata_sample_diff