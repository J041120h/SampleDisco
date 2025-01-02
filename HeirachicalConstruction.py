import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster

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
