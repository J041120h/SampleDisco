import numpy as np
import pandas as pd
import os
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import scanpy as sc
from sklearn.neighbors import KNeighborsTransformer

def cell_type_dendrogram(
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

    if verbose:
        print('=== Preparing data for dendrogram (using marker genes) ===')

    # Check that the groupby column exists
    if groupby not in adata.obs.columns:
        raise ValueError(f"The groupby key '{groupby}' is not present in adata.obs.")

    # Check marker genes
    if marker_genes is None or len(marker_genes) == 0:
        raise ValueError("No marker genes provided. Please supply a non-empty list of marker genes.")

    # Ensure marker genes are in adata
    marker_genes = [g for g in marker_genes if g in adata.var_names]
    if len(marker_genes) == 0:
        raise ValueError("None of the provided marker genes are found in adata.var_names.")

    # Extract marker gene expression data
    # adata.X is typically log-normalized if following standard Scanpy workflow
    marker_data = adata[:, marker_genes].X

    # Create a DataFrame for convenience
    df_markers = pd.DataFrame(marker_data.toarray() if hasattr(marker_data, 'toarray') else marker_data,
                              index=adata.obs_names,
                              columns=marker_genes)
    df_markers[groupby] = adata.obs[groupby].values

    if distance_mode == 'centroid':
        if verbose:
            print('=== Computing centroids of cell types in marker gene space ===')
        # Calculate centroids for each cell type using marker genes
        centroids = df_markers.groupby(groupby).mean()
        if verbose:
            print(f'Calculated centroids for {centroids.shape[0]} cell types.')
            print(f'=== Computing distance matrix between centroids using {metric} distance ===')
        dist_matrix = pdist(centroids.values, metric=metric)
        labels = centroids.index.tolist()
    else:
        raise ValueError(f"Unsupported distance_mode '{distance_mode}' for marker gene approach.")

    # Perform hierarchical clustering
    if verbose:
        print('=== Performing hierarchical clustering on marker gene centroids ===')
        print(f'Linkage method: {method}, Distance metric: {metric}')
    Z = sch.linkage(dist_matrix, method=method)

    # Store the linkage matrix in adata
    adata.uns['cell_type_linkage'] = Z

    # Reclustering cell types based on resolution
    if verbose:
        print(f'=== Reclustering cell types with resolution {resolution} ===')
    max_height = np.max(Z[:, 2])
    threshold = (1 - resolution) * max_height
    if verbose:
        print(f'Using threshold {threshold} to cut the dendrogram (max height: {max_height})')

    # Get new cluster labels
    cluster_labels = fcluster(Z, t=threshold, criterion='distance')
    if verbose:
        print(f'Formed {len(np.unique(cluster_labels))} clusters at resolution {resolution}')

    # Map original cell types to new cluster labels
    celltype_to_cluster = dict(zip(centroids.index, cluster_labels))
    adata.obs[groupby] = adata.obs[groupby].map(celltype_to_cluster).astype('category')

    return adata

def cell_types(
    adata, 
    cell_column='cell_type', 
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
    if cell_column in adata.obs.columns:
        if verbose:
            print("[cell_types] Found existing cell type annotation.")
        adata.obs['cell_type'] = adata.obs[cell_column].astype(str)

        if markers is not None:
            marker_dict = {i: markers[i - 1] for i in range(1, len(markers) + 1)}
            adata.obs['cell_type'] = adata.obs['cell_type'].map(marker_dict)

        sc.tl.rank_genes_groups(adata, groupby='cell_type', method='logreg', n_genes=100)
        rank_results = adata.uns['rank_genes_groups']
        groups = rank_results['names'].dtype.names
        all_marker_genes = set()
        for group in groups:
            all_marker_genes.update(rank_results['names'][group])

        adata = cell_type_dendrogram(
            adata=adata,
            resolution=cluster_resolution,
            groupby='cell_type',
            method=method,
            metric=metric,
            distance_mode=distance_mode,
            marker_genes=list(all_marker_genes),
            verbose=verbose
        )

        sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_pcs=num_PCs)

    else:
        if verbose:
            print("[cell_types] No cell type annotation found. Performing clustering.")
        transformer = KNeighborsTransformer(n_neighbors=10, metric='manhattan', algorithm='kd_tree')
        sc.pp.neighbors(adata, use_rep='X_pca_harmony', transformer=transformer)

        sc.tl.leiden(
            adata,
            resolution=cluster_resolution,
            flavor='igraph',
            n_iterations=1,
            directed=False,
            key_added='cell_type'
        )

        adata.obs['cell_type'] = (adata.obs['cell_type'].astype(int) + 1).astype('category')

    if verbose:
        print("[cell_types] Finished assigning cell types.")
    
    sc.tl.umap(adata, min_dist=0.5)
    if Save and output_dir:
        save_path = os.path.join(output_dir, 'adata_cell.h5ad')
        sc.write(save_path, adata)
        if verbose:
            print(f"[cell_types] Saved AnnData object to {save_path}")

    return adata

def cell_type_assign(adata_cluster, adata, Save=False, output_dir=None,verbose = True):
    if 'cell_type' not in adata_cluster.obs.columns or adata_cluster.obs['cell_type'].nunique() == 0:
        adata_cluster.obs['cell_type'] = '1'
    adata.obs['cell_type'] = adata_cluster.obs['cell_type']
    if Save and output_dir:
        save_path = os.path.join(output_dir, 'adata_sample.h5ad')
        sc.write(save_path, adata)
        if verbose:
            print(f"[cell_types] Saved AnnData object to {save_path}")