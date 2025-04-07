import numpy as np
import pandas as pd
import os
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import rapids_singlecell as rsc 
import scanpy as sc
from sklearn.neighbors import KNeighborsTransformer
import time

def cell_type_dendrogram_linux(
    adata,
    resolution,
    groupby='cell_type',
    method='average',
    metric='euclidean',
    distance_mode='centroid',
    marker_genes=None,
    verbose=True
):
    """
    Constructs a dendrogram of cell types based on selected marker genes rather than PCA.

    Parameters:
    - adata : AnnData object
        Annotated data matrix.
    - resolution : float
        Clustering resolution as before.
    - groupby : str, optional
        The observation key to group cells by. Default is 'cell_type'.
    - method : str, optional
        The linkage algorithm (e.g. 'average', 'complete', 'ward').
    - metric : str, optional
        The distance metric (e.g. 'euclidean', 'cosine').
    - distance_mode : str, optional
        How to compute the distance:
        - 'centroid': Compute distances between centroids of groups in marker gene space.
    - marker_genes : list, optional
        A list of marker genes to use for dendrogram construction.
    - verbose : bool, optional
        Print progress messages.
    """
    start_time = time.time()
    if verbose:
        print('=== Preparing data for dendrogram (using marker genes) ===')

    if groupby not in adata.obs.columns:
        raise ValueError(f"The groupby key '{groupby}' is not present in adata.obs.")

    if marker_genes is None or len(marker_genes) == 0:
        raise ValueError("No marker genes provided. Please supply a non-empty list of marker genes.")

    marker_genes = [g for g in marker_genes if g in adata.var_names]
    if len(marker_genes) == 0:
        raise ValueError("None of the provided marker genes are found in adata.var_names.")

    marker_data = adata[:, marker_genes].X

    df_markers = pd.DataFrame(
        marker_data.toarray() if hasattr(marker_data, 'toarray') else marker_data,
        index=adata.obs_names,
        columns=marker_genes
    )
    df_markers[groupby] = adata.obs[groupby].values

    if distance_mode == 'centroid':
        if verbose:
            print('=== Computing centroids of cell types in marker gene space ===')
        centroids = df_markers.groupby(groupby).mean()
        if verbose:
            print(f'Calculated centroids for {centroids.shape[0]} cell types.')
            print(f'=== Computing distance matrix between centroids using {metric} distance ===')
        dist_matrix = pdist(centroids.values, metric=metric)
        labels = centroids.index.tolist()
    else:
        raise ValueError(f"Unsupported distance_mode '{distance_mode}' for marker gene approach.")

    if verbose:
        print('=== Performing hierarchical clustering on marker gene centroids ===')
        print(f'Linkage method: {method}, Distance metric: {metric}')
    Z = sch.linkage(dist_matrix, method=method)

    adata.uns['cell_type_linkage'] = Z

    if verbose:
        print(f'=== Reclustering cell types with resolution {resolution} ===')
    max_height = np.max(Z[:, 2])
    threshold = (1 - resolution) * max_height
    if verbose:
        print(f'Using threshold {threshold} to cut the dendrogram (max height: {max_height})')

    cluster_labels = fcluster(Z, t=threshold, criterion='distance')
    if verbose:
        print(f'Formed {len(np.unique(cluster_labels))} clusters at resolution {resolution}')

    celltype_to_cluster = dict(zip(centroids.index, cluster_labels))
    adata.obs[groupby] = adata.obs[groupby].map(celltype_to_cluster).astype('category')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Function execution time: {elapsed_time:.2f} seconds")

    return adata


def cell_types_linux(
    adata, 
    cell_column='cell_type', 
    existing_cell_types=False,
    umap=False,
    Save=False,
    output_dir=None,
    cluster_resolution=0.8,
    markers=None, 
    method='average', 
    metric='euclidean', 
    distance_mode='centroid',
    num_PCs=20, 
    verbose=True
):
    """
    Assigns cell types based on existing annotations or performs Leiden clustering if no annotation exists.

    Parameters:
    - adata: AnnData object
    - cell_column: Column name containing cell type annotations
    - existing_cell_types: bool, whether to use existing cell types
    - umap: bool, whether to compute UMAP
    - Save: Boolean, whether to save the output
    - output_dir: Directory to save the output if Save=True
    - cluster_resolution: Resolution for Leiden clustering
    - markers: List of markers for mapping numeric IDs to names
    - method, metric, distance_mode: Parameters for hierarchical clustering
    - num_PCs: Number of principal components for neighborhood graph
    - verbose: Whether to print progress messages

    Returns:
    - Updated AnnData object with assigned cell types
    """
    start_time = time.time() if verbose else None
    rsc.get.anndata_to_GPU(adata)
    if cell_column in adata.obs.columns and existing_cell_types:
        if verbose:
            print("[cell_types] Found existing cell type annotation.")
        adata.obs['cell_type'] = adata.obs[cell_column].astype(str)

        if markers is not None:
            # if numeric cluster labels and want to map them to known cell type names
            marker_dict = {i: markers[i - 1] for i in range(1, len(markers) + 1)}
            adata.obs['cell_type'] = adata.obs['cell_type'].map(marker_dict)

        # rank genes for each cell type
        rsc.tl.rank_genes_groups(adata, groupby='cell_type', method='logreg', n_genes=100)
        rank_results = adata.uns['rank_genes_groups']
        groups = rank_results['names'].dtype.names
        all_marker_genes = set()
        for group in groups:
            all_marker_genes.update(rank_results['names'][group])

        adata = cell_type_dendrogram_linux(
            adata=adata,
            resolution=cluster_resolution,
            groupby='cell_type',
            method=method,
            metric=metric,
            distance_mode=distance_mode,
            marker_genes=list(all_marker_genes),
            verbose=verbose
        )

        # rebuild neighbors using 'X_pca_harmony'
        rsc.pp.neighbors(adata, use_rep='X_pca_harmony', n_pcs=num_PCs)
    else:
        if verbose:
            print("[cell_types] No cell type annotation found. Performing clustering.")

        start_time_internal = time.time() if verbose else None
        # build the neighbor graph
        rsc.pp.neighbors(adata, use_rep='X_pca_harmony', n_pcs=num_PCs)
        if verbose:
            end_time_neighbor = time.time()
            elapsed_time = end_time_neighbor - start_time_internal
            print(f"\n\n[rsc.neighbors] Total runtime: {elapsed_time:.2f} seconds\n\n")

        # perform Leiden clustering
        rsc.tl.leiden(
            adata,
            resolution=cluster_resolution,
            key_added='cell_type'
        )
        if verbose:
            end_time_leiden = time.time()
            elapsed_time = end_time_leiden - start_time_internal
            print(f"\n\n[Leiden] Total runtime: {elapsed_time:.2f} seconds\n\n")

        # convert numeric labels to categories (1-based)
        adata.obs['cell_type'] = (adata.obs['cell_type'].astype(int) + 1).astype('category')

        num_clusters = adata.obs['cell_type'].nunique()
        if verbose:
            print(f"\n[cell_types] Found {num_clusters} clusters after Leiden clustering.\n")

    if verbose:
        print("[cell_types] Finished assigning cell types.")
    
    if umap:
        if verbose:
            print("[cell_types] Computing UMAP.")
        rsc.tl.umap(adata, min_dist=0.5)
    
    
    rsc.get.anndata_to_CPU(adata)
    if Save and output_dir:
        if verbose:
            print(f"saving the data to {output_dir}")
        output_dir = os.path.join(output_dir, 'harmony')
        save_path = os.path.join(output_dir, 'adata_cell.h5ad')
        # if os.path.exists(save_path):
        #     if verbose:
        #         print(f"[cell_types] Removing existing file at {save_path}")
        #     os.remove(save_path)
        adata.write(save_path)  # saving in CPU-based .h5ad format
        if verbose:
            print(f"[cell_types] Saved AnnData object to {save_path}")
    
    if verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\n[cell_types] Total runtime: {elapsed_time:.2f} seconds\n\n")

    return adata


def cell_type_assign_linux(adata_cluster, adata, Save=False, output_dir=None, verbose=True):
    """
    Assign cell type labels from one AnnData object to another and optionally save the result.

    Parameters
    ----------
    adata_cluster : AnnData
        AnnData object containing a 'cell_type' column in `.obs` to be used for assignment.
    adata : AnnData
        Target AnnData object to receive the 'cell_type' labels.
    Save : bool, optional
        If True, saves the modified `adata` object to disk.
    output_dir : str, optional
        Directory to save the `adata` object if `Save` is True.
    verbose : bool, optional
        If True and saving is enabled, prints the save location.
    """
    if 'cell_type' not in adata_cluster.obs.columns or adata_cluster.obs['cell_type'].nunique() == 0:
        adata_cluster.obs['cell_type'] = '1'

    adata.obs['cell_type'] = adata_cluster.obs['cell_type']

    if Save and output_dir:
        output_dir = os.path.join(output_dir, 'harmony')
        save_path = os.path.join(output_dir, 'adata_sample.h5ad')
        adata.write(save_path)  # saving in CPU-based .h5ad format
        if verbose:
            print(f"[cell_types] Saved AnnData object to {save_path}")