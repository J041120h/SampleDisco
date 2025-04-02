import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc

from sklearn.cluster import KMeans
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

from Visualization import plot_clusters_by_cluster, plot_clusters_by_grouping
# (Assuming your existing visualize_TSCAN_paths is imported correctly)

def cluster_samples_by_pca(
    adata: sc.AnnData,
    column: str,
    k: int = 0,
    random_state: int = 0,
    verbose: bool = False
) -> tuple[dict, pd.DataFrame]:
    """
    Clusters samples by K-means on PCA coordinates stored in `adata.uns[column]`.

    Parameters
    ----------
    adata : AnnData
        Must have adata.uns[column] as a DataFrame (rows=samples, columns=PCs).
    column : str
        Key in adata.uns where PCA DataFrame is stored.
    k : int
        Number of clusters. If 0, defaults to 5% of samples (at least 1).
    random_state : int
        RNG seed for K-means.
    verbose : bool
        Print messages if True.

    Returns
    -------
    sample_cluster : dict
        Mapping of "cluster_i" -> list of sample IDs in that cluster.
    pca_df : DataFrame
        The PCA coordinates (unchanged, just returned for convenience).
    """
    # ----------------------------------------------------------------------
    # 1. Retrieve PCA DataFrame
    # ----------------------------------------------------------------------
    if column not in adata.uns:
        raise KeyError(f"[cluster_samples_by_pca] No PCA data found in adata.uns['{column}'].")
    pca_data = adata.uns[column]

    if not isinstance(pca_data, pd.DataFrame):
        raise TypeError(
            f"[cluster_samples_by_pca] Expected a DataFrame in adata.uns['{column}'], "
            f"but got {type(pca_data)}."
        )

    # Number of samples in the PCA DataFrame
    sample_ids = pca_data.index
    n_samples = pca_data.shape[0]

    if k > n_samples:
        raise ValueError(f"[cluster_samples_by_pca] k={k} cannot exceed n_samples={n_samples}.")
    if k == 0:
        k = max(1, int(n_samples * 0.05))  # default 5% of samples

    # ----------------------------------------------------------------------
    # 2. K-means clustering
    # ----------------------------------------------------------------------
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    cluster_labels = kmeans.fit_predict(pca_data.values)  # fit on the numeric array

    # Map from cluster index -> "cluster_1", "cluster_2", etc.
    cluster_label_map = {i: f"cluster_{i + 1}" for i in range(k)}
    cluster_dict = defaultdict(list)

    # ----------------------------------------------------------------------
    # 3. Build sample_cluster dictionary
    # ----------------------------------------------------------------------
    for sample, cluster_idx in zip(sample_ids, cluster_labels):
        cluster_name = cluster_label_map[cluster_idx]
        cluster_dict[cluster_name].append(sample)

    sample_cluster = dict(cluster_dict)

    if verbose:
        print(f"[cluster_samples_by_pca] K-means done (k={k}).")
        print(f"   => {len(sample_cluster)} clusters formed from {n_samples} samples.")

    # Return the cluster assignment and the PCA DataFrame
    return sample_cluster, pca_data


def Cluster_distance(
    pca_data: pd.DataFrame,
    sample_cluster: dict,
    metric: str = "euclidean",
    verbose: bool = False
) -> np.ndarray:
    """
    Computes pairwise distances between cluster centroids, then writes a distance matrix to CSV.

    Parameters
    ----------
    pca_data : DataFrame
        Rows are samples, columns are PCA coordinates (PC1..PCn).
    sample_cluster : dict
        Mapping from cluster_name -> list of samples.
    metric : str
        Distance metric for pdist.
    output_file : str
        CSV file path to save the distance matrix.
    verbose : bool
        Print messages if True.

    Returns
    -------
    pairwise_distances : np.ndarray
        Condensed distance matrix from pdist for the cluster centroids.
    """
    # ----------------------------------------------------------------------
    # 1. Compute cluster centroids from the DataFrame
    # ----------------------------------------------------------------------
    cluster_names = sorted(sample_cluster.keys())
    cluster_centroids = []

    for cluster_name in cluster_names:
        # Subset the pca_data for samples in this cluster
        cluster_samples = sample_cluster[cluster_name]
        coords = pca_data.loc[cluster_samples, :]  # shape: (num_cluster_samples, nPCs)
        centroid = coords.mean(axis=0).values
        cluster_centroids.append(centroid)

    # ----------------------------------------------------------------------
    # 2. Compute pairwise distances and build a full matrix
    # ----------------------------------------------------------------------
    cluster_centroids = np.vstack(cluster_centroids)  # shape: (k_clusters, nPCs)
    pairwise_dists = pdist(cluster_centroids, metric=metric)  # condensed
    dist_matrix = squareform(pairwise_dists)

    distance_df = pd.DataFrame(
        dist_matrix,
        index=cluster_names,
        columns=cluster_names
    )

    return pairwise_dists


def construct_MST(pairwise_distances, verbose=False):
    mst = minimum_spanning_tree(squareform(pairwise_distances))
    if verbose:
        print("[construct_MST] MST adjacency matrix:\n", mst.toarray())
    return mst.toarray()


def find_principal_path(mst_array, sample_cluster, verbose=False):
    """
    Creates a graph from the MST adjacency matrix and searches for the
    single longest 'shortest_path' in terms of number of edges (tie-break by cell count).
    Returns that path as a list of cluster indices (0..k-1 in sorted order).
    """
    G = nx.from_numpy_array(mst_array + mst_array.T)

    # cluster_names: sorted => cluster_1, cluster_2, ...
    cluster_list = sorted(sample_cluster.keys())

    def total_cells_in_path(path):
        # path is a list of cluster indices in the MST (0..k-1)
        # we convert each index to the actual cluster name
        # and sum the sizes
        return sum(len(sample_cluster[cluster_list[idx]]) for idx in path)

    max_path = []
    max_cell_count = 0

    for source in G.nodes:
        for target in G.nodes:
            if source >= target:
                continue
            try:
                path = nx.shortest_path(G, source=source, target=target)
                # longer path => update
                if len(path) > len(max_path):
                    max_path = path
                    max_cell_count = total_cells_in_path(path)
                elif len(path) == len(max_path):
                    # tie-break by total cell count
                    cell_count = total_cells_in_path(path)
                    if cell_count > max_cell_count:
                        max_path = path
                        max_cell_count = cell_count
            except nx.NetworkXNoPath:
                continue

    if verbose:
        print(f"[find_principal_path] Main path: length={len(max_path)}, total cell count={max_cell_count}")

    return max_path  # list of cluster indices (0..k-1)


def find_branching_paths(G, origin, main_path, verbose=False):
    """
    Identifies leaf nodes not in main_path, then gets shortest path from origin to each leaf.
    Returns a list of these branching paths (each a list of cluster indices).
    """
    branching_paths = []
    leaf_nodes = [n for n in G.nodes if G.degree[n] == 1 and n not in main_path]

    for leaf in leaf_nodes:
        try:
            path = nx.shortest_path(G, source=origin, target=leaf)
            # Only consider it a "branching" path if it diverges from main_path
            if not all(node in main_path for node in path):
                branching_paths.append(path)
        except nx.NetworkXNoPath:
            continue

    if verbose:
        print(f"[find_branching_paths] {len(branching_paths)} branching path(s) found.")
    return branching_paths


def project_cells_onto_edges(
    pca_data: pd.DataFrame,
    sample_cluster: dict[str, list[str]],
    main_path: list[int],
    verbose: bool = False
) -> dict[str, dict]:
    """
    Projects each cell onto the relevant edge in the cluster-level ordering (main_path).
    The cluster IDs in main_path are MST indices: 0..k-1. We'll map them to "cluster_x" via sorted order.

    Returns
    -------
    cell_projections : dict[sample_id -> dict]
        For each sample: the cluster, which edge it's projected on, the scalar projection, etc.
    """
    # cluster_list is sorted => [cluster_1, cluster_2, ...]
    cluster_list = sorted(sample_cluster.keys())

    # 1) Compute cluster centroids from the DF
    cluster_centroids = {}
    for clust in cluster_list:
        coords = pca_data.loc[sample_cluster[clust], :]  # (N, nPCs)
        centroid = coords.mean(axis=0).values
        cluster_centroids[clust] = centroid

    # 2) Helper for scalar projection
    def _projection(Ek, Ei, Ej):
        v_ij = Ej - Ei
        numerator = np.dot(Ek - Ei, v_ij)
        denom = np.linalg.norm(v_ij)
        if denom == 0:
            if verbose:
                print("[project_cells_onto_edges] WARNING: zero-length edge in MST.")
            return 0.0
        return numerator / denom

    # 3) Helper for distance
    def _dist_cell_to_centroid(Ek, Ec):
        return np.linalg.norm(Ek - Ec)

    # main_path is a list of indices in MST space. Convert them to cluster names
    # e.g., if main_path=[0,2,1], cluster_0 => cluster_list[0], cluster_2 => cluster_list[2], etc.
    # We'll re-use the integer => cluster_name approach
    path_cluster_names = [cluster_list[idx] for idx in main_path]
    M = len(path_cluster_names)

    cell_projections = {}

    # Go step by step along the cluster sequence in main_path
    for i, cluster_name in enumerate(path_cluster_names):
        cells_in_cluster = sample_cluster[cluster_name]
        Ei = cluster_centroids[cluster_name]

        if i == 0 and M > 1:
            # all cells from the first cluster project onto edge (C0->C1)
            next_cluster = path_cluster_names[i+1]
            Ej = cluster_centroids[next_cluster]
            for c in cells_in_cluster:
                Ek = pca_data.loc[c].values
                val = _projection(Ek, Ei, Ej)
                cell_projections[c] = {
                    "cluster": cluster_name,
                    "edge": (cluster_name, next_cluster),
                    "projection": val,
                    "cluster_index": i,
                    "edge_index": 1  # forward
                }

        elif i == M - 1 and M > 1:
            # last cluster => project onto edge ((M-1)->M)
            prev_cluster = path_cluster_names[i-1]
            Eprev = cluster_centroids[prev_cluster]
            for c in cells_in_cluster:
                Ek = pca_data.loc[c].values
                val = _projection(Ek, Eprev, Ei)
                cell_projections[c] = {
                    "cluster": cluster_name,
                    "edge": (prev_cluster, cluster_name),
                    "projection": val,
                    "cluster_index": i,
                    "edge_index": 0  # backward
                }

        else:
            # intermediate cluster => decide if each cell is "closer" to the previous or next
            # before we do the projection
            if M == 1:
                # only one cluster in path => no edges
                for c in cells_in_cluster:
                    cell_projections[c] = {
                        "cluster": cluster_name,
                        "edge": None,
                        "projection": 0.0,
                        "cluster_index": i,
                        "edge_index": None
                    }
            else:
                prev_cluster = path_cluster_names[i-1]
                next_cluster = path_cluster_names[i+1]
                Eprev = cluster_centroids[prev_cluster]
                Enext = cluster_centroids[next_cluster]
                for c in cells_in_cluster:
                    Ek = pca_data.loc[c].values

                    dist2prev = _dist_cell_to_centroid(Ek, Eprev)
                    dist2next = _dist_cell_to_centroid(Ek, Enext)

                    if dist2prev <= dist2next:
                        # project onto edge (prev->this)
                        val = _projection(Ek, Eprev, Ei)
                        cell_projections[c] = {
                            "cluster": cluster_name,
                            "edge": (prev_cluster, cluster_name),
                            "projection": val,
                            "cluster_index": i,
                            "edge_index": 0
                        }
                    else:
                        # project onto edge (this->next)
                        val = _projection(Ek, Ei, Enext)
                        cell_projections[c] = {
                            "cluster": cluster_name,
                            "edge": (cluster_name, next_cluster),
                            "projection": val,
                            "cluster_index": i,
                            "edge_index": 1
                        }

    if verbose:
        print("[project_cells_onto_edges] Projection complete. "
              f"Projected {len(cell_projections)} cells.")

    return cell_projections


def order_cells_along_paths(
    cell_projections: dict[str, dict],
    main_path: list[int],
    verbose: bool = False
) -> list[str]:
    """
    Sorts cells along the path by (cluster_index, edge_index, projection).
    main_path is in MST indices, but we only need cell_projections' fields:
        cluster_index, edge_index, projection

    Returns
    -------
    ordered_cells : list[str]
        Final linear ordering of sample IDs.
    """
    # Build a sortable tuple for each cell
    sortable = []
    for cell_id, info in cell_projections.items():
        c_idx = info["cluster_index"]
        e_idx = info["edge_index"] if info["edge_index"] is not None else 0
        proj = info["projection"]
        sortable.append((c_idx, e_idx, proj, cell_id))

    # Sort
    sortable.sort(key=lambda x: (x[0], x[1], x[2]))

    ordered_cells = [tup[3] for tup in sortable]

    if verbose:
        print("[order_cells_along_paths] Final ordering computed with "
              f"{len(ordered_cells)} cells total.")
    return ordered_cells


def TSCAN(
    AnnData_sample: sc.AnnData,
    column: str,
    n_clusters: int,
    output_dir: str,
    grouping_columns = None,
    verbose: bool = False,
    origin: int = None
):
    """
    Orchestrates TSCAN steps: K-means on PCA -> MST -> principal path + branching paths -> 
    optional plotting and final ordering of cells.

    Returns
    -------
    results : dict
        Contains main path, branching paths, sample_cluster, and final cell order, etc.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"[TSCAN] Output directory created: {output_dir}")

    # subdirectory for TSCAN results
    output_path = os.path.join(output_dir, "TSCAN")
    os.makedirs(output_path, exist_ok=True)

    # --------------------------------------------------
    # 1. Cluster samples by PCA
    # --------------------------------------------------
    sample_cluster, pca_df = cluster_samples_by_pca(
        AnnData_sample,
        column=column,
        k=n_clusters,
        random_state=0,
        verbose=verbose
    )
    if verbose:
        print("[TSCAN] sample_cluster:", sample_cluster)

    # --------------------------------------------------
    # 2. MST on cluster centroids
    # --------------------------------------------------
    pairwise_dists = Cluster_distance(
        pca_df, 
        sample_cluster, 
        metric="euclidean",
        verbose=verbose
    )

    mst = construct_MST(pairwise_dists, verbose=verbose)
    main_path = find_principal_path(mst, sample_cluster, verbose=verbose)

    # 'main_path' is a list of cluster indices. The first and last are 'ends'.
    if verbose:
        print(f"[TSCAN] main_path (cluster indices) => {main_path}")

    ends = [main_path[0], main_path[-1]]
    if origin is None:
        origin = random.choice(ends)  # pick either end
        if verbose:
            print(f"[TSCAN] No origin specified. Using random end: {origin}")
    elif origin not in ends:
        raise ValueError(f"Provided origin {origin} is not an endpoint in {ends}")
    else:
        if verbose:
            print(f"[TSCAN] origin set to: {origin}")

    # --------------------------------------------------
    # 3. Branching paths
    # --------------------------------------------------
    G = nx.from_numpy_array(mst + mst.T)
    branching_paths = find_branching_paths(G, origin, main_path, verbose=verbose)

    for i, bp in enumerate(branching_paths):
        if verbose:
            print(f"[TSCAN] Branch {i+1}: {bp}")

    # --------------------------------------------------
    # 4. Project cells along main path, then order them
    # --------------------------------------------------
    cell_projections = project_cells_onto_edges(
        pca_data=pca_df,
        sample_cluster=sample_cluster,
        main_path=main_path,
        verbose=verbose
    )
    ordered_cells = order_cells_along_paths(
        cell_projections,
        main_path=main_path,
        verbose=verbose
    )

    # --------------------------------------------------
    # 5. Handle branching paths
    # --------------------------------------------------
    cell_projections_branching_paths = {}
    ordered_cells_branching_paths = {}

    for index, path in enumerate(branching_paths):
        cp = project_cells_onto_edges(
            pca_data=pca_df,
            sample_cluster=sample_cluster,
            main_path=path,
            verbose=verbose
        )
        oc = order_cells_along_paths(cp, main_path=path, verbose=verbose)

        cell_projections_branching_paths[index] = cp
        ordered_cells_branching_paths[index] = oc

    # --------------------------------------------------
    # 6. Visualization
    # --------------------------------------------------
    # (Your existing function that expects sample_cluster, main_path, branching_paths, etc.)
    plot_clusters_by_cluster(
        adata = AnnData_sample,
        sample_cluster = sample_cluster,
        main_path = main_path,
        branching_paths = branching_paths,
        output_path = output_path,
        pca_key = column,
        verbose = verbose
    )

    if grouping_columns is not None:
        plot_clusters_by_grouping(
            adata = AnnData_sample,
            sample_cluster = sample_cluster,
            main_path = main_path,
            branching_paths = branching_paths,
            output_path = output_path,
            pca_key = column,
            grouping_columns = grouping_columns,
            verbose = verbose
        )

    # --------------------------------------------------
    # 7. Return everything
    # --------------------------------------------------
    results = {
        "main_path": main_path,
        "origin": origin,
        "branching_paths": branching_paths,
        "graph": G,
        "sample_cluster": sample_cluster,
        "pca_data": pca_df,
        "cell_projections": cell_projections,
        "ordered_cells": ordered_cells,
        "cell_projections_branching_paths": cell_projections_branching_paths,
        "ordered_cells_branching_paths": ordered_cells_branching_paths
    }

    if verbose:
        print("[TSCAN] Completed. Main path cell order length:", len(ordered_cells))
        for idx, bp in ordered_cells_branching_paths.items():
            print(f"   Branch {idx+1} order length:", len(bp))

    return results