import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
import os

def cell_type_dendrogram(
    adata,
    resolution,
    pca_key='X_pca',
    groupby='cell_type',
    method='average',
    metric='euclidean',
    distance_mode='centroid',
    verbose=True
):
    """
    Constructs a dendrogram of cell types based on PCA data.

    Parameters:
    - adata : AnnData object
        The annotated data matrix.
    - pca_key : str, optional
        The key in `adata.obsm` where PCA coordinates are stored. Default is 'X_pca'.
    - groupby : str, optional
        The observation key to group cells (e.g., 'celltype'). Default is 'celltype'.
    - method : str, optional
        The linkage algorithm to use for hierarchical clustering. Options include 'single', 'complete', 'average', 'ward', etc. Default is 'average'.
    - metric : str, optional
        The distance metric to use. Options include 'euclidean', 'cosine', etc. Default is 'euclidean'.
    - distance_mode : str, optional
        How to compute the distance matrix. Options are:
        - 'centroid': Compute distances between centroids of groups.
        Default is 'centroid'.
    - output_dir : str, optional
        Directory to save the dendrogram plot. If None, the plot is shown but not saved.
    - verbose : bool, optional
        If True, prints progress messages.
    """

    if verbose:
        print('=== Preparing data for dendrogram ===')

    # Check if groupby column exists
    if groupby not in adata.obs.columns:
        raise ValueError(f"The groupby key '{groupby}' is not present in adata.obs.")

    # Extract PCA coordinates
    pca_coords = adata.obsm[pca_key]

    # Create a DataFrame with PCA coordinates and cell types
    df_pca = pd.DataFrame(pca_coords, index=adata.obs_names)
    df_pca[groupby] = adata.obs[groupby].values
 
    if distance_mode == 'centroid':
        if verbose:
            print('=== Computing centroids of cell types in PCA space ===')
        # Calculate centroids for each cell type
        centroids = df_pca.groupby(groupby).mean()
        if verbose:
            print(f'Calculated centroids for {centroids.shape[0]} cell types.')
            print(f'=== Computing distance matrix between centroids using {metric} distance ===')
        dist_matrix = pdist(centroids.values, metric=metric)
        labels = centroids.index.tolist()
    elif distance_mode in ['euclidean', 'cosine']:
        if verbose:
            print(f'=== Computing pairwise distances between cells using {distance_mode} distance ===')
        # Map cell types to integers
        celltype_to_int = {ctype: idx for idx, ctype in enumerate(df_pca[groupby].unique())}
        labels = [celltype_to_int[ctype] for ctype in df_pca[groupby]]
        # Compute pairwise distances
        dist_matrix = pdist(df_pca.drop(columns=[groupby]).values, metric=distance_mode)
    else:
        raise ValueError(f"Invalid distance_mode '{distance_mode}'. Choose from 'centroid', 'euclidean', 'cosine'.")

    # Perform hierarchical clustering
    if verbose:
        print('=== Performing hierarchical clustering ===')
        print(f'Linkage method: {method}, Distance metric: {metric}')
    Z = sch.linkage(dist_matrix, method=method)

    # Store the linkage matrix in adata for later use (e.g., plotting)
    adata.uns['cell_type_linkage'] = Z

    # Recluster cell types based on the resolution parameter
    if verbose:
        print(f'=== Reclustering cell types with resolution {resolution} ===')
    max_height = np.max(Z[:, 2])
    threshold = (1 - resolution) * max_height  # Smaller resolution leads to higher threshold
    if verbose:
        print(f'Using threshold {threshold} to cut the dendrogram (max height: {max_height})')
    # Get new cluster labels for cell types
    cluster_labels = fcluster(Z, t=threshold, criterion='distance')
    if verbose:
        print(f'Formed {len(np.unique(cluster_labels))} clusters at resolution {resolution}')

    # Map original cell types to new cluster labels
    celltype_to_cluster = dict(zip(centroids.index, cluster_labels))
    # Update the 'cell_type' column in adata.obs
    adata.obs[groupby] = adata.obs[groupby].map(celltype_to_cluster).astype('category')
    return adata