import os
import contextlib
import io
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from harmony import harmonize
import rapids_singlecell as rsc
from scipy import sparse
import time

def anndata_cluster(
    adata_cluster,
    output_dir,
    sample_column = 'sample',
    cell_column='cell_type',
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
    if verbose:
        print('=== [GPU] Processing data for clustering ===')

    # Step A1: HVG selection (run on CPU before GPU conversion)
    rsc.pp.highly_variable_genes(
        adata_cluster,
        n_top_genes=num_features,
        flavor='seurat_v3',
        batch_key=sample_column
    )
    adata_cluster = adata_cluster[:, adata_cluster.var['highly_variable']].copy()
    rsc.pp.normalize_total(adata_cluster, target_sum=1e4)
    rsc.pp.log1p(adata_cluster)

    # Step A2: GPU PCA
    rsc.pp.pca(adata_cluster, n_comps=num_PCs)

    if verbose:
        print('=== [GPU] Running Harmony integration ===')
        print('Variables to regress:', ', '.join(vars_to_regress_for_harmony or []))

    # Step A3: GPU Harmony
    Z = harmonize(
        adata_cluster.obsm['X_pca'],
        adata_cluster.obs,
        batch_key=vars_to_regress_for_harmony,
        max_iter_harmony=num_harmony,
        use_gpu=True
    )
    adata_cluster.obsm['X_pca_harmony'] = Z

    if verbose:
        print("End of harmony for adata_cluster.")

    # Back to CPU before saving
    rsc.get.anndata_to_CPU(adata_cluster)
    sc.write(os.path.join(output_dir, 'adata_cell.h5ad'), adata_cluster)
    return adata_cluster

import os
import scanpy as sc
import rapids_singlecell as rsc
from harmony import harmonize

def anndata_sample(
    adata_sample_diff,
    output_dir,
    batch_key,
    sample_column='sample',
    num_features=2000,
    num_PCs=20,
    num_harmony=30,
    verbose=True
):
    if verbose:
        print('=== [GPU] Processing data for sample differences ===')

    # Store original counts
    adata_sample_diff.layers["counts"] = adata_sample_diff.X.copy()

    # Move to GPU and preprocess
    rsc.get.anndata_to_GPU(adata_sample_diff)
    rsc.pp.normalize_total(adata_sample_diff, target_sum=1e4)
    rsc.pp.log1p(adata_sample_diff)
    rsc.pp.pca(adata_sample_diff, n_comps=num_PCs)

    if verbose:
        print('=== [GPU] Begin Harmony ===')

    # Harmony batch correction on GPU
    Z = harmonize(
        adata_sample_diff.obsm['X_pca'],
        adata_sample_diff.obs,
        batch_key=[batch_key],
        max_iter_harmony=num_harmony,
        use_gpu=True
    )
    adata_sample_diff.obsm['X_pca_harmony'] = Z

    # Neighbors and UMAP using Harmony representation
    rsc.pp.neighbors(adata_sample_diff, use_rep='X_pca_harmony', n_neighbors=15, metric='cosine')
    rsc.tl.umap(adata_sample_diff, min_dist=0.3, spread=1.0)

    # Back to CPU before saving
    rsc.get.anndata_to_CPU(adata_sample_diff)
    adata_sample_diff.X = adata_sample_diff.layers["counts"].copy()
    del adata_sample_diff.layers["counts"]
    sc.write(os.path.join(output_dir, 'adata_sample.h5ad'), adata_sample_diff)

    return adata_sample_diff


def harmony_linux(
    h5ad_path,
    sample_meta_path,
    output_dir,
    sample_column='sample',
    cell_column='cell_type',
    cell_meta_path=None,
    batch_key='batch',
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
    combat=True,
    method='average',
    metric='euclidean',
    distance_mode='centroid',
    vars_to_regress=[],
    verbose=True
):
    start_time = time.time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Created output directory")

    output_dir = os.path.join(output_dir, 'harmony')
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print('=== Reading input dataset ===')

    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f'Raw shape: {adata.shape[0]} cells × {adata.shape[1]} genes')

    vars_to_regress_for_harmony = vars_to_regress.copy()
    if sample_column not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append(sample_column)

    if cell_meta_path is None:
        if sample_column not in adata.obs.columns: 
            adata.obs[sample_column] = adata.obs_names.str.split(':').str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path)
        cell_meta.set_index('barcode', inplace=True)
        adata.obs = adata.obs.join(cell_meta, how='left')

    if sample_meta_path is not None:
        sample_meta = pd.read_csv(sample_meta_path)
        adata.obs = adata.obs.merge(sample_meta, on=sample_column, how='left')

    all_required_columns = vars_to_regress + [batch_key]
    missing_vars = [col for col in all_required_columns if col not in adata.obs.columns]
    if missing_vars:
        raise KeyError(f"The following variables are missing from adata.obs: {missing_vars}")
    else:
        print("All required columns are present in adata.obs.")

    print("Before GPU conversion. Type of adata.X:", type(adata.X))
    print("adata.X dtype:", adata.X.dtype)
    print("adata.X is sparse:", sparse.issparse(adata.X))

    if adata.X.dtype != np.float32:
        print(f"Converting adata.X from {adata.X.dtype} to float32")
        adata.X = adata.X.astype(np.float32)
    rsc.get.anndata_to_GPU(adata)
    print("Converted to GPU. Type of adata.X:", type(adata.X))

    # Filter genes and cells
    rsc.pp.filter_genes(adata, min_cells=min_cells)
    rsc.pp.filter_cells(adata, min_genes=min_features)
    if verbose:
        print(f'After initial filtering: {adata.shape[0]} cells × {adata.shape[1]} genes')

    # Calculate mitochondrial gene percentage
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    rsc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], log1p=False)
    
    # Filter based on mitochondrial percentage
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()
    if verbose:
        print(f'After mitochondrial filtering: {adata.shape[0]} cells × {adata.shape[1]} genes')

    # Filter MT genes and excluded genes
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    genes_to_exclude = set(mt_genes) | set(exclude_genes or [])
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()
    if verbose:
        print(f'After gene exclusion filtering: {adata.shape[0]} cells × {adata.shape[1]} genes')

    # Filter samples with too few cells
    cell_counts = adata.obs.groupby(sample_column).size()
    keep = cell_counts[cell_counts >= min_cells].index
    adata = adata[adata.obs[sample_column].isin(keep)].copy()
    if verbose:
        print(f'After sample filtering: {adata.shape[0]} cells × {adata.shape[1]} genes')
        print(f'Number of samples remaining: {len(keep)}')

    # Additional gene filtering based on cell percentage
    rsc.pp.filter_genes(adata, min_cells=int(0.01 * adata.n_obs))
    if verbose:
        print(f'After final gene filtering: {adata.shape[0]} cells × {adata.shape[1]} genes')

    if verbose:
        print(f'Processed shape: {adata.shape[0]} cells × {adata.shape[1]} genes')

    if doublet:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            rsc.pp.scrublet(adata, batch_key=sample_column)
            adata[~adata.obs['predicted_doublet']].copy()

    adata.raw = adata.copy()
    if verbose:
        print("Preprocessing complete!")

    adata_cluster = adata.copy()
    adata_sample_diff = adata.copy()

    adata_cluster = anndata_cluster(
        adata_cluster=adata_cluster,
        output_dir=output_dir,
        sample_column = sample_column,
        cell_column=cell_column,
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

    adata_sample_diff = anndata_sample(
        adata_sample_diff=adata_sample_diff,
        output_dir=output_dir,
        sample_column=sample_column, 
        num_features=num_features,
        num_PCs=num_PCs,
        num_harmony=num_harmony,
        batch_key=batch_key,
        verbose=verbose
    )

    if verbose:
        print(f"Total runtime: {time.time() - start_time:.2f} seconds")

    return adata_cluster, adata_sample_diff