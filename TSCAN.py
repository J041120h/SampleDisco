from sklearn.cluster import KMeans
import scanpy as sc
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import random
import os
from Visualization import visualize_TSCAN_paths

def cluster_samples_by_pca(
    adata: sc.AnnData,
    column: str,
    k: int = 0,
    random_state: int = 0,
    verbose: bool = False
) -> tuple[dict, np.ndarray]:
    pca_key = column
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

    cluster_label_map = {i: f"cluster_{i + 1}" for i in range(k)}
    cluster_dict = defaultdict(list)

    sample_ids = adata.obs_names
    for sample, cluster in zip(sample_ids, cluster_labels):
        cluster_name = cluster_label_map[cluster]
        cluster_dict[cluster_name].append(sample)

    sample_cluster = dict(cluster_dict)

    if verbose:
        print(f"[cluster_samples_by_pca] K-means clustering (k={k}) on '{column}' PCA data completed.")

    return sample_cluster, pca_data

def Cluster_distance(
    pca_data: np.ndarray,
    sample_cluster: dict,
    metric: str = "euclidean",
    output_file: str = "cluster_distances.csv",
    verbose: bool = False
) -> np.ndarray:
    sample_to_cluster = {
        sample: cluster for cluster, samples in sample_cluster.items() for sample in samples
    }

    sample_ids = list(sample_to_cluster.keys())
    sample_idx = {sample: idx for idx, sample in enumerate(sample_ids)}

    cluster_centroids = []
    cluster_names = []

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
        print(f"[Cluster_distance] Pairwise cluster distances saved to {output_file}")
        print(distance_matrix)

    return pairwise_distances

def construct_MST(pairwise_distances, verbose=False):
    mst = minimum_spanning_tree(squareform(pairwise_distances))
    if verbose:
        print(mst.toarray())
    return mst.toarray()

def find_principal_path(mst_array, sample_cluster, verbose=False):
    G = nx.from_numpy_array(mst_array + mst_array.T)

    def total_cells_in_path(path):
        return sum(len(sample_cluster.get(node, [])) for node in path)

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
        print(f"[find_principal_path] Main path determined with length {len(max_path)} and total cell count {max_cell_count}")

    return max_path

def find_branching_paths(G, origin, main_path, verbose=False):
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
        print(f"[find_branching_paths] Found {len(branching_paths)} branching path(s)")

    return branching_paths

def project_cells_onto_edges(
    pca_data: np.ndarray,
    sample_cluster: dict[str, list[str]],
    main_path: list[str],
    verbose: bool = False
) -> dict[str, dict]:
    """
    Project each cell onto the appropriate edge in the cluster-level ordering (main_path).
    
    For each cluster Ci in the main_path:
      - If Ci is the first cluster, project all its cells onto the edge (C1 -> C2).
      - If Ci is the last cluster, project all its cells onto the edge (C(M-1) -> CM).
      - Otherwise (1 < i < M), divide the cells of Ci based on whether they are closer
        to C(i-1) or C(i+1). Then project onto the corresponding edge.

    The projection for cell k onto edge (Ci -> Cj) is given by
        proj_k = ( (Ek - E(i)) · v_ij ) / ||v_ij||
    where 
        Ek       = PCA coordinate vector for cell k,
        E(i)     = centroid of cluster Ci,
        v_ij     = E(j) - E(i).

    Returns:
      A dictionary cell_projections keyed by cell ID, where each value is a dict:
          {
            "cluster":       name of the cluster the cell belongs to,
            "edge":          (cluster_i, cluster_j),
            "projection":    scalar projection distance along that edge,
            "cluster_index": i (the index of the cluster in main_path),
            "edge_index":    0 or 1 (0 = the edge from C(i-1) to Ci, 1 = the edge from Ci to C(i+1))
          }
    """

    #simply get the unique sample ids
    sample_ids = []
    for clust, cells in sample_cluster.items():
        sample_ids.extend(cells)
    sample_ids = list(set(sample_ids))  # unique list
    sample_ids.sort()
    
    # Map sample -> row index in pca_data
    #   (Assuming that adata.obs_names are in the same order as rows in pca_data,
    #    you need to adjust if your ordering is different.)
    sample_index_map = {s: i for i, s in enumerate(sample_ids)}

    # Compute cluster centroids
    cluster_centroids = {}
    for clust, cells in sample_cluster.items():
        idxs = [sample_index_map[c] for c in cells if c in sample_index_map]
        if idxs:
            coords = pca_data[idxs, :]
            centroid = coords.mean(axis=0)
            cluster_centroids[clust] = centroid
        else:
            raise ValueError(f"Cluster '{clust}' has no valid samples in PCA data.")

    # ------------------------------------------------
    # 2. Assign cells to edges and compute projections
    # ------------------------------------------------
    cell_projections = {}
    
    def _projection(Ek, Ei, Ej):
        """Compute scalar projection of Ek onto the line from Ei to Ej."""
        v_ij = Ej - Ei
        numerator = np.dot(Ek - Ei, v_ij)
        denom = np.linalg.norm(v_ij)
        if denom == 0:
            print(f"[project_cells_onto_edges] Warning: Zero length edge between {Ei} and {Ej}.")
            return 0.0
        return numerator / denom
    
    # Helper to get Euclidean distance from a cell to a cluster centroid
    def _dist_cell_to_centroid(Ek, Ec):
        return np.linalg.norm(Ek - Ec)

    M = len(main_path)
    for i, clust in enumerate(main_path):
        clust = f"cluster_{clust + 1}"
        cells = sample_cluster[clust]
        Ei = cluster_centroids[clust]
        
        if i == 0:
            # All cells in first cluster project onto the edge (C0->C1)
            next_clust = main_path[i+1]
            next_clust = f"cluster_{next_clust + 1}"
            Ej = cluster_centroids[next_clust]
            for c in cells:
                row_idx = sample_index_map[c]
                Ek = pca_data[row_idx, :]
                val = _projection(Ek, Ei, Ej)
                cell_projections[c] = {
                    "cluster": clust,
                    "edge": (clust, next_clust),
                    "projection": val,
                    "cluster_index": i,
                    "edge_index": 1  # 'forward' edge
                }
                
        elif i == M - 1:
            # All cells in last cluster project onto the edge (C(M-1)->CM)
            prev_clust = main_path[i-1]
            prev_clust = f"cluster_{prev_clust + 1}"
            Eprev = cluster_centroids[prev_clust]
            for c in cells:
                row_idx = sample_index_map[c]
                Ek = pca_data[row_idx, :]
                val = _projection(Ek, Eprev, Ei)
                cell_projections[c] = {
                    "cluster": clust,
                    "edge": (prev_clust, clust),
                    "projection": val,
                    "cluster_index": i,
                    "edge_index": 0  # 'backward' edge
                }
        else:
            # Intermediate cluster: decide if each cell is closer to the previous or next centroid
            prev_clust = main_path[i-1]
            next_clust = main_path[i+1]
            prev_clust = f"cluster_{prev_clust + 1}"
            next_clust = f"cluster_{next_clust + 1}"
            Eprev = cluster_centroids[prev_clust]
            Enext = cluster_centroids[next_clust]
            for c in cells:
                row_idx = sample_index_map[c]
                Ek = pca_data[row_idx, :]

                # Distances from cell to the two neighboring centroids
                dist2prev = _dist_cell_to_centroid(Ek, Eprev)
                dist2next = _dist_cell_to_centroid(Ek, Enext)
                
                if dist2prev <= dist2next:
                    # Project onto edge (C(i-1)->C(i))
                    val = _projection(Ek, Eprev, Ei)
                    cell_projections[c] = {
                        "cluster": clust,
                        "edge": (prev_clust, clust),
                        "projection": val,
                        "cluster_index": i,
                        "edge_index": 0
                    }
                else:
                    # Project onto edge (C(i)->C(i+1))
                    val = _projection(Ek, Ei, Enext)
                    cell_projections[c] = {
                        "cluster": clust,
                        "edge": (clust, next_clust),
                        "projection": val,
                        "cluster_index": i,
                        "edge_index": 1
                    }

    if verbose:
        print("[project_cells_onto_edges] Finished projecting cells onto edges.")
    
    return cell_projections


def order_cells_along_paths(
    cell_projections: dict[str, dict],
    main_path: list[str],
    verbose: bool = False
) -> list[str]:
    """
    Order all cells along the main path according to the three-step procedure:

    1. For cells which are in the same cluster and are projected onto the same edge,
       their order is determined by the (scalar) 'projection' value on that edge.
    2. Within each cluster, the order of cells projected onto different edges is
       determined by the order of edges (which is given by the cluster-level path).
       In other words, an edge_index=0 (previous->this) is ordered before edge_index=1 (this->next).
    3. The order of cells in different clusters is determined by the order of clusters
       in the main_path.

    Returns:
      A list of cell IDs (strings) giving the final ordering.
    """

    # We will create a sortable tuple for each cell:
    #   (cluster_path_index, edge_index, projection_value, cell_id)
    # Then we sort by the first three fields and keep cell_id as a tiebreak or stable sort.
    sortable = []

    for cell_id, info in cell_projections.items():
        c_index = info["cluster_index"]      # position in main_path
        e_index = info["edge_index"]         # 0 or 1
        val     = info["projection"]         # scalar projection
        sortable.append((c_index, e_index, val, cell_id))

    # Sort by (cluster_index ascending, edge_index ascending, projection ascending)
    sortable.sort(key=lambda x: (x[0], x[1], x[2]))

    # Extract the cell IDs in sorted order
    ordered_cells = [x[3] for x in sortable]

    if verbose:
        print("[order_cells_along_paths] Final ordering of cells computed.")
    
    return ordered_cells

def TSCAN(AnnData_sample, column, n_clusters, output_dir, verbose=False, origin=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating output directory")

    # Append 'harmony' subdirectory
    output_dir = os.path.join(output_dir, 'TSCAN')

    sample_cluster, pca_data = cluster_samples_by_pca(
        AnnData_sample,
        column=column,
        k=n_clusters,
        random_state=0,
        verbose=verbose
    )
    print(sample_cluster)

    pairwise_distances = Cluster_distance(
        pca_data, 
        sample_cluster, 
        verbose=verbose
    )

    mst = construct_MST(pairwise_distances, verbose = verbose)
    main_path = find_principal_path(mst, sample_cluster, verbose=verbose)

    if verbose:
        print(f"[TSCAN] Main path (clusters): {main_path}")

    ends = [main_path[0], main_path[-1]]
    if origin is None:
        origin = random.choice(ends)
        if verbose:
            print(f"[TSCAN] No origin specified. Randomly selected origin: {origin}")
    elif origin not in ends:
        raise ValueError(f"Provided origin {origin} is not an endpoint of the main path: {ends}")
    else:
        if verbose:
            print(f"[TSCAN] Using user-specified origin: {origin}")

    G = nx.from_numpy_array(mst + mst.T)
    branching_paths = find_branching_paths(G, origin, main_path, verbose=verbose)

    if verbose:
        for i, path in enumerate(branching_paths):
            print(f"[TSCAN] Branch {i+1}: {path}")

    cell_projections = dict()
    ordered_cells = dict()
    cell_projections= project_cells_onto_edges(
        pca_data = pca_data,
        sample_cluster = sample_cluster,
        main_path = main_path,
        verbose = verbose
    )
        
    ordered_cells = order_cells_along_paths(
        cell_projections = cell_projections,
        main_path = main_path,
        verbose = verbose,
    ) 

    for index, path in enumerate(branching_paths):
        # project branching paths as well
        cell_projections_branching_paths = dict() # reset for branching paths
        ordered_cells_branching_paths = dict()
        cell_projections_branching_paths[index] = project_cells_onto_edges(
            pca_data = pca_data,
            sample_cluster = sample_cluster,
            main_path = path,
            verbose = verbose
        )

        # Order cells in the branching path
        ordered_cells_branching_paths[index] = order_cells_along_paths(
            cell_projections = cell_projections_branching_paths[index],
            main_path = path,
            verbose = verbose
        )
    
    visualize_TSCAN_paths(
        AnnData_sample,
        sample_cluster = sample_cluster,
        main_path = main_path,
        branching_paths = branching_paths,
        output_dir = output_dir,
        pca_key = column,
        verbose = verbose
    )
        
    if verbose:
        print(f"[TSCAN] Ordered cells: {ordered_cells}")
    return {
        "main_path": main_path,
        "origin": origin,
        "branching_paths": branching_paths,
        "graph": G,
        "sample_cluster": sample_cluster,
        "pca_data": pca_data,
        "cell_projections": cell_projections,
        "ordered_cells": ordered_cells,
        "cell_projections_branching_paths": cell_projections_branching_paths,
        "ordered_cells_branching_paths": ordered_cells_branching_paths
    }
