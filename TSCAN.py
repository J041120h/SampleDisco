from sklearn.cluster import KMeans
import scanpy as sc
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import os

def cluster_samples_by_pca(
    adata: sc.AnnData,
    column: str,
    k: int = 0,
    key_added: str = None,
    random_state: int = 0,
    verbose: bool = False
) -> tuple[dict, np.ndarray]:
    """
    Performs k-means clustering on sample-level PCA data stored in `adata.uns`.

    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object containing PCA results.
    column : str
        One of "expression" or "proportion" to specify the PCA data source.
    k : int
        Number of clusters for k-means.
    key_added : str, optional
        Key to store the cluster labels in `adata.obs`. Defaults to 'kmeans_<column>_<k>'.
    random_state : int, default 0
        Random state for reproducibility.
    verbose : bool, default False
        Whether to print progress info.

    Returns:
    --------
    tuple
        sample_cluster dictionary and PCA data array.
    """
    if column not in ["expression", "proportion"]:
        raise ValueError("`column` must be either 'expression' or 'proportion'.")

    pca_key = f"X_pca_{column}"
    if pca_key not in adata.uns:
        raise KeyError(f"PCA data not found in `adata.uns['{pca_key}']`.")

    pca_data = adata.uns[pca_key]
    n_samples = pca_data.shape[0]
    if k > n_samples:
        raise ValueError(f"k ({k}) cannot be greater than the number of samples ({n_samples}).")
    if k == 0:
        k = max(1, int(n_samples * 0.05))

    kmeans = KMeans(n_clusters=k, random_state=random_state)
    cluster_labels = kmeans.fit_predict(pca_data)

    # Map numeric cluster labels to 'cluster_#'
    cluster_label_map = {i: f"cluster_{i + 1}" for i in range(k)}
    cluster_dict = defaultdict(list)

    sample_ids = adata.obs_names
    for sample, cluster in zip(sample_ids, cluster_labels):
        cluster_name = cluster_label_map[cluster]
        cluster_dict[cluster_name].append(sample)

    sample_cluster = dict(cluster_dict)

    if verbose:
        print(f"K-means clustering (k={k}) on '{column}' PCA data completed.")

    return sample_cluster, pca_data

def Cluster_distance(
    pca_data: np.ndarray,
    sample_cluster: dict,
    metric: str = "euclidean",
    output_file: str = "cluster_distances.csv",
    verbose: bool = False
) -> np.ndarray:
    """
    Calculates pairwise distances between cluster centroids in PCA space.

    Parameters:
    -----------
    pca_data : np.ndarray
        PCA coordinates for each sample. Rows must match sample IDs in `sample_cluster`.
    sample_cluster : dict
        Dictionary mapping cluster label to list of sample IDs.
    metric : str, default "euclidean"
        Distance metric to use with `pdist`.
    output_file : str
        Path to save the resulting distance matrix (CSV format).
    verbose : bool
        If True, prints additional info.

    Returns:
    --------
    np.ndarray
        Condensed distance matrix as returned by `pdist`.
    """
    # Invert sample_cluster to map sample -> cluster
    sample_to_cluster = {
        sample: cluster for cluster, samples in sample_cluster.items() for sample in samples
    }

    sample_ids = list(sample_to_cluster.keys())
    sample_idx = {sample: idx for idx, sample in enumerate(sample_ids)}

    cluster_centroids = []
    cluster_names = []

    # Use natural sort so cluster_10 comes after cluster_2
    for cluster, samples in sorted(sample_cluster.items()):
        indices = [sample_idx[s] for s in samples if s in sample_idx]
        cluster_data = pca_data[indices]
        centroid = cluster_data.mean(axis=0)
        cluster_centroids.append(centroid)
        cluster_names.append(cluster)

    cluster_centroids = np.vstack(cluster_centroids)

    pairwise_distances = pdist(cluster_centroids, metric=metric)
    distance_matrix = pd.DataFrame(
        squareform(pairwise_distances),
        index=cluster_names,
        columns=cluster_names
    )

    distance_matrix.to_csv(output_file)

    if verbose:
        print(f"Pairwise cluster distances saved to {output_file}")
        print(distance_matrix)

    return pairwise_distances

def construct_MST(pairwise_distances):
    """
    Constructs a Minimum Spanning Tree (MST) from pairwise distances.

    Parameters:
    -----------
    pairwise_distances : np.ndarray
        Condensed distance matrix.

    Returns:
    --------
    np.ndarray
        Adjacency matrix of the MST.
    """
    mst = minimum_spanning_tree(squareform(pairwise_distances))
    return mst.toarray()