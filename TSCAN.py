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
    # 1. Retrieve PCA DataFrame
    if column not in adata.uns:
        raise KeyError(f"[cluster_samples_by_pca] No PCA data found in adata.uns['{column}'].")
    pca_data = adata.uns[column]

    if not isinstance(pca_data, pd.DataFrame):
        raise TypeError(
            f"[cluster_samples_by_pca] Expected a DataFrame in adata.uns['{column}'], "
            f"but got {type(pca_data)}."
        )

    sample_ids = pca_data.index
    n_samples = pca_data.shape[0]

    if k > n_samples:
        raise ValueError(f"[cluster_samples_by_pca] k={k} cannot exceed n_samples={n_samples}.")
    if k == 0:
        k = max(1, int(n_samples * 0.05))  # default 5% of samples

    # 2. K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    cluster_labels = kmeans.fit_predict(pca_data.values)

    # Build sample_cluster dictionary
    cluster_label_map = {i: f"cluster_{i + 1}" for i in range(k)}
    cluster_dict = defaultdict(list)
    for sample, cluster_idx in zip(sample_ids, cluster_labels):
        cluster_name = cluster_label_map[cluster_idx]
        cluster_dict[cluster_name].append(sample)

    sample_cluster = dict(cluster_dict)

    if verbose:
        print(f"[cluster_samples_by_pca] K-means done (k={k}).")
        print(f"   => {len(sample_cluster)} clusters formed from {n_samples} samples.")

    return sample_cluster, pca_data


def Cluster_distance(
    pca_data: pd.DataFrame,
    sample_cluster: dict,
    metric: str = "euclidean",
    verbose: bool = False
) -> np.ndarray:
    """
    Computes pairwise distances between cluster centroids, returns a condensed distance array.

    Parameters
    ----------
    pca_data : DataFrame
        Rows are samples, columns are PCA coordinates (PC1..PCn).
    sample_cluster : dict
        Mapping from cluster_name -> list of samples.
    metric : str
        Distance metric for pdist.
    verbose : bool
        Print messages if True.

    Returns
    -------
    pairwise_distances : np.ndarray
        Condensed distance matrix from pdist for the cluster centroids.
    """
    cluster_names = sorted(sample_cluster.keys())
    cluster_centroids = []

    for cluster_name in cluster_names:
        cluster_samples = sample_cluster[cluster_name]
        coords = pca_data.loc[cluster_samples, :]
        centroid = coords.mean(axis=0).values
        cluster_centroids.append(centroid)

    # shape: (k_clusters, nPCs)
    cluster_centroids = np.vstack(cluster_centroids)
    pairwise_dists = pdist(cluster_centroids, metric=metric)

    if verbose:
        print(f"[Cluster_distance] Computed pairwise {metric} distances among cluster centroids.")
    return pairwise_dists


def construct_MST(pairwise_distances, verbose=False):
    """
    Builds an MST from the condensed distance matrix and returns
    the MST as a dense adjacency matrix.
    """
    mst = minimum_spanning_tree(squareform(pairwise_distances))
    if verbose:
        print("[construct_MST] MST adjacency matrix:\n", mst.toarray())
    return mst.toarray()


def find_principal_path(mst_array, sample_cluster, verbose=False):
    """
    Creates a graph from the MST adjacency matrix and searches for
    the single longest 'shortest_path' (in # of edges). Ties are broken
    by total cell count. Returns that path as a list of cluster indices.
    """
    G = nx.from_numpy_array(mst_array + mst_array.T)
    cluster_list = sorted(sample_cluster.keys())

    def total_cells_in_path(path):
        return sum(len(sample_cluster[cluster_list[idx]]) for idx in path)

    max_path = []
    max_cell_count = 0

    for source in G.nodes:
        for target in G.nodes:
            if source >= target:
                continue
            try:
                path = nx.shortest_path(G, source=source, target=target)
                if len(path) > len(max_path):
                    max_path = path
                    max_cell_count = total_cells_in_path(path)
                elif len(path) == len(max_path):
                    cell_count = total_cells_in_path(path)
                    if cell_count > max_cell_count:
                        max_path = path
                        max_cell_count = cell_count
            except nx.NetworkXNoPath:
                continue

    if verbose:
        print(f"[find_principal_path] Main path: length={len(max_path)}, total cell count={max_cell_count}")

    return max_path


def find_branching_paths(G, origin, main_path, verbose=False):
    """
    Identifies leaf nodes not in main_path, then gets the shortest path
    from origin to each leaf. Returns a list of these branching paths.
    """
    branching_paths = []
    leaf_nodes = [n for n in G.nodes if G.degree[n] == 1 and n not in main_path]

    for leaf in leaf_nodes:
        try:
            path = nx.shortest_path(G, source=origin, target=leaf)
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
    For each cell, projects it onto the relevant edge in the cluster ordering (main_path).
    Returns {cell_id -> {cluster, edge, projection, cluster_index, edge_index}}
    """
    cluster_list = sorted(sample_cluster.keys())

    # Precompute centroids
    cluster_centroids = {}
    for clust in cluster_list:
        coords = pca_data.loc[sample_cluster[clust], :]
        cluster_centroids[clust] = coords.mean(axis=0).values

    def _projection(Ek, Ei, Ej):
        v_ij = Ej - Ei
        numerator = np.dot(Ek - Ei, v_ij)
        denom = np.linalg.norm(v_ij)
        if denom == 0:
            if verbose:
                print("[project_cells_onto_edges] WARNING: zero-length edge.")
            return 0.0
        return numerator / denom

    def _dist_cell_to_centroid(Ek, Ec):
        return np.linalg.norm(Ek - Ec)

    path_cluster_names = [cluster_list[idx] for idx in main_path]
    M = len(path_cluster_names)

    cell_projections = {}

    for i, cluster_name in enumerate(path_cluster_names):
        cells_in_cluster = sample_cluster[cluster_name]
        Ei = cluster_centroids[cluster_name]

        # First cluster
        if i == 0 and M > 1:
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

        # Last cluster
        elif i == M - 1 and M > 1:
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

        # Intermediate cluster(s)
        else:
            if M == 1:
                # Only one cluster => no edges
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
                        val = _projection(Ek, Eprev, Ei)
                        cell_projections[c] = {
                            "cluster": cluster_name,
                            "edge": (prev_cluster, cluster_name),
                            "projection": val,
                            "cluster_index": i,
                            "edge_index": 0
                        }
                    else:
                        val = _projection(Ek, Ei, Enext)
                        cell_projections[c] = {
                            "cluster": cluster_name,
                            "edge": (cluster_name, next_cluster),
                            "projection": val,
                            "cluster_index": i,
                            "edge_index": 1
                        }

    if verbose:
        print("[project_cells_onto_edges] Projection complete for "
              f"{len(cell_projections)} cells.")
    return cell_projections


def order_cells_along_paths(
    cell_projections: dict[str, dict],
    main_path: list[int],
    verbose: bool = False
) -> list[str]:
    """
    Sorts cells by (cluster_index, edge_index, projection). 
    Returns a list of cell IDs in the final ordering.
    """
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
    Orchestrates TSCAN steps: 
    1) K-means on PCA -> 
    2) MST -> 
    3) principal path + branching paths -> 
    4) final ordering of cells -> 
    5) Pseudotime calculation ->
    6) optional plotting

    Returns
    -------
    results : dict
        Contains MST, main path, branching paths, sample_cluster, 
        final cell order, pseudotime, etc.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"[TSCAN] Output directory created: {output_dir}")

    output_path = os.path.join(output_dir, "TSCAN")
    os.makedirs(output_path, exist_ok=True)

    # 1. Cluster samples by PCA
    sample_cluster, pca_df = cluster_samples_by_pca(
        AnnData_sample,
        column=column,
        k=n_clusters,
        random_state=0,
        verbose=verbose
    )

    # 2. MST on cluster centroids
    pairwise_dists = Cluster_distance(
        pca_df, 
        sample_cluster, 
        metric="euclidean",
        verbose=verbose
    )
    mst = construct_MST(pairwise_dists, verbose=verbose)

    # 3. Find main path and pick origin
    main_path = find_principal_path(mst, sample_cluster, verbose=verbose)
    ends = [main_path[0], main_path[-1]]
    if origin is None:
        origin = random.choice(ends)
        if verbose:
            print(f"[TSCAN] No origin specified. Using random end: {origin}")
    elif origin not in ends:
        raise ValueError(f"Provided origin {origin} is not an endpoint in {ends}")
    else:
        if verbose:
            print(f"[TSCAN] origin set to: {origin}")

    # Build graph for branching analysis
    G = nx.from_numpy_array(mst + mst.T)
    branching_paths = find_branching_paths(G, origin, main_path, verbose=verbose)

    # 4. Project + order cells along main path
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

    # 5. Handle branching paths: project + order
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

    # --- NEW STEP: Compute pseudo‐time along each path ---
    #   We'll assign integer positions along the main path and each branch
    pseudo_main = {cell_id: i for i, cell_id in enumerate(ordered_cells, start=1)}
    pseudo_branches = {}
    for branch_idx, oc in ordered_cells_branching_paths.items():
        pseudo_branches[branch_idx] = {cell_id: i for i, cell_id in enumerate(oc, start=1)}

    # 6. Visualization
    plot_clusters_by_cluster(
        adata=AnnData_sample,
        sample_cluster=sample_cluster,
        main_path=main_path,
        branching_paths=branching_paths,
        output_path=output_path,
        pca_key=column,
        verbose=verbose
    )

    if grouping_columns is not None:
        plot_clusters_by_grouping(
            adata=AnnData_sample,
            sample_cluster=sample_cluster,
            main_path=main_path,
            branching_paths=branching_paths,
            output_path=output_path,
            pca_key=column,
            grouping_columns=grouping_columns,
            verbose=verbose
        )

    # 7. Prepare and return final results
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
        "ordered_cells_branching_paths": ordered_cells_branching_paths,
        "pseudotime": {
            "main_path": pseudo_main,
            "branching_paths": pseudo_branches
        }
    }

    if verbose:
        print("[TSCAN] Completed. Main path cell order length:", len(ordered_cells))
        for idx, bp in ordered_cells_branching_paths.items():
            print(f"   Branch {idx+1} order length:", len(bp))
        print("   Pseudotime computation done.")

    return results