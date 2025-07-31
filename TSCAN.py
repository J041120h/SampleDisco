import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import Dict, List, Tuple, Optional, Union


def find_sample_grouping(adata: sc.AnnData, samples: List[str], grouping_columns: List[str]) -> Dict[str, str]:
    """
    Extract grouping information for samples from adata.obs.
    """
    grouping_dict = {}
    
    # Normalize sample names for comparison
    samples_normalized = [str(s).strip().lower() for s in samples]
    obs_index_normalized = [str(idx).strip().lower() for idx in adata.obs.index]
    
    for sample in samples:
        sample_norm = str(sample).strip().lower()
        if sample_norm in obs_index_normalized:
            # Find the original index
            orig_idx = adata.obs.index[obs_index_normalized.index(sample_norm)]
            
            # Combine grouping columns
            group_values = []
            for col in grouping_columns:
                if col in adata.obs.columns:
                    group_values.append(str(adata.obs.loc[orig_idx, col]))
            
            grouping_dict[sample] = "_".join(group_values) if group_values else "unknown"
        else:
            grouping_dict[sample] = "unknown"
    
    return grouping_dict


def cluster_samples_by_pca(
    adata: sc.AnnData,
    column: str,
    k: int = 0,
    random_state: int = 0,
    verbose: bool = False
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Cluster samples based on PCA coordinates using K-means.
    """
    # 1. Retrieve PCA DataFrame
    if column not in adata.uns:
        raise KeyError(f"No PCA data found in adata.uns['{column}'].")
    pca_data = adata.uns[column]

    if not isinstance(pca_data, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame in adata.uns['{column}'], but got {type(pca_data)}.")
    
    # Enhanced sample alignment
    if not pca_data.index.equals(adata.obs.index):
        common_samples = pca_data.index.intersection(adata.obs.index)
        
        if len(common_samples) == 0:
            # Try normalized comparison
            pca_normalized = pd.Index([str(s).strip().lower() for s in pca_data.index])
            obs_normalized = pd.Index([str(s).strip().lower() for s in adata.obs.index])
            
            # Create mapping from normalized to original
            pca_norm_to_orig = dict(zip(pca_normalized, pca_data.index))
            obs_norm_to_orig = dict(zip(obs_normalized, adata.obs.index))
            
            common_normalized = pca_normalized.intersection(obs_normalized)
            
            if len(common_normalized) > 0:
                # Map back to original sample names
                common_pca_orig = [pca_norm_to_orig[norm] for norm in common_normalized]
                common_obs_orig = [obs_norm_to_orig[norm] for norm in common_normalized]
                
                # Reindex PCA data to match obs
                pca_data = pca_data.loc[common_pca_orig]
                pca_data.index = common_obs_orig  # Use obs naming convention
            else:
                raise ValueError("No common samples found between PCA data and adata.obs")
        else:
            pca_data = pca_data.loc[common_samples]

    sample_ids = pca_data.index
    n_samples = pca_data.shape[0]

    if k > n_samples:
        raise ValueError(f"k={k} cannot exceed n_samples={n_samples}.")
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
        cluster_dict[cluster_name].append(str(sample))

    sample_cluster = dict(cluster_dict)

    if verbose:
        print(f"K-means clustering complete: {len(sample_cluster)} clusters from {n_samples} samples")

    return sample_cluster, pca_data


def Cluster_distance(
    pca_data: pd.DataFrame,
    sample_cluster: Dict[str, List[str]],
    metric: str = "euclidean",
    verbose: bool = False
) -> np.ndarray:
    """
    Computes pairwise distances between cluster centroids.
    """
    cluster_names = sorted(sample_cluster.keys())
    cluster_centroids = []

    for cluster_name in cluster_names:
        cluster_samples = sample_cluster[cluster_name]
        available_samples = [s for s in cluster_samples if s in pca_data.index]
        if len(available_samples) == 0:
            raise ValueError(f"No samples from cluster {cluster_name} found in PCA data")
        
        coords = pca_data.loc[available_samples, :]
        centroid = coords.mean(axis=0).values
        cluster_centroids.append(centroid)

    cluster_centroids = np.vstack(cluster_centroids)
    pairwise_dists = pdist(cluster_centroids, metric=metric)

    if verbose:
        print(f"Computed pairwise {metric} distances among {len(cluster_names)} cluster centroids")
    
    return pairwise_dists


def construct_MST(pairwise_distances: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Builds an MST from the condensed distance matrix.
    """
    mst = minimum_spanning_tree(squareform(pairwise_distances))
    if verbose:
        print(f"MST construction complete with shape: {mst.toarray().shape}")
    return mst.toarray()


def find_principal_path(mst_array: np.ndarray, sample_cluster: Dict[str, List[str]], verbose: bool = False) -> List[int]:
    """
    Finds the longest path in the MST (by number of edges), with ties broken by total sample count.
    """
    G = nx.from_numpy_array(mst_array + mst_array.T)
    cluster_list = sorted(sample_cluster.keys())

    def total_samples_in_path(path):
        return sum(len(sample_cluster[cluster_list[idx]]) for idx in path)

    max_path = []
    max_sample_count = 0

    for source in G.nodes:
        for target in G.nodes:
            if source >= target:
                continue
            try:
                path = nx.shortest_path(G, source=source, target=target)
                if len(path) > len(max_path):
                    max_path = path
                    max_sample_count = total_samples_in_path(path)
                elif len(path) == len(max_path):
                    sample_count = total_samples_in_path(path)
                    if sample_count > max_sample_count:
                        max_path = path
                        max_sample_count = sample_count
            except nx.NetworkXNoPath:
                continue

    if verbose:
        cluster_names = [f"cluster_{i+1}" for i in max_path]
        print(f"Principal path found: {len(max_path)} clusters, {max_sample_count} total samples")
        print(f"Path: {' -> '.join(cluster_names)}")

    return max_path


def find_branching_paths(G: nx.Graph, origin: int, main_path: List[int], verbose: bool = False) -> List[List[int]]:
    """
    Identifies branching paths from the main trajectory.
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
        print(f"Found {len(branching_paths)} branching paths")
    
    return branching_paths


def project_samples_onto_edges(
    pca_data: pd.DataFrame,
    sample_cluster: Dict[str, List[str]],
    main_path: List[int],
    verbose: bool = False
) -> Dict[str, Dict]:
    """
    Projects samples onto edges in the cluster ordering.
    """
    cluster_list = sorted(sample_cluster.keys())

    # Precompute centroids
    cluster_centroids = {}
    for clust in cluster_list:
        available_samples = [s for s in sample_cluster[clust] if s in pca_data.index]
        if len(available_samples) == 0:
            raise ValueError(f"No available samples for cluster {clust}")
        coords = pca_data.loc[available_samples, :]
        cluster_centroids[clust] = coords.mean(axis=0).values

    def _projection(Ek, Ei, Ej):
        v_ij = Ej - Ei
        numerator = np.dot(Ek - Ei, v_ij)
        denom = np.linalg.norm(v_ij)
        return numerator / denom if denom != 0 else 0.0

    def _dist_sample_to_centroid(Ek, Ec):
        return np.linalg.norm(Ek - Ec)

    path_cluster_names = [cluster_list[idx] for idx in main_path]
    M = len(path_cluster_names)
    sample_projections = {}

    for i, cluster_name in enumerate(path_cluster_names):
        samples_in_cluster = [s for s in sample_cluster[cluster_name] if s in pca_data.index]
        Ei = cluster_centroids[cluster_name]

        # First cluster
        if i == 0 and M > 1:
            next_cluster = path_cluster_names[i+1]
            Ej = cluster_centroids[next_cluster]
            for s in samples_in_cluster:
                Ek = pca_data.loc[s].values
                val = _projection(Ek, Ei, Ej)
                sample_projections[s] = {
                    "cluster": cluster_name,
                    "edge": (cluster_name, next_cluster),
                    "projection": val,
                    "cluster_index": i,
                    "edge_index": 1
                }

        # Last cluster
        elif i == M - 1 and M > 1:
            prev_cluster = path_cluster_names[i-1]
            Eprev = cluster_centroids[prev_cluster]
            for s in samples_in_cluster:
                Ek = pca_data.loc[s].values
                val = _projection(Ek, Eprev, Ei)
                sample_projections[s] = {
                    "cluster": cluster_name,
                    "edge": (prev_cluster, cluster_name),
                    "projection": val,
                    "cluster_index": i,
                    "edge_index": 0
                }

        # Intermediate clusters
        else:
            if M == 1:
                for s in samples_in_cluster:
                    sample_projections[s] = {
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
                for s in samples_in_cluster:
                    Ek = pca_data.loc[s].values
                    dist2prev = _dist_sample_to_centroid(Ek, Eprev)
                    dist2next = _dist_sample_to_centroid(Ek, Enext)
                    if dist2prev <= dist2next:
                        val = _projection(Ek, Eprev, Ei)
                        sample_projections[s] = {
                            "cluster": cluster_name,
                            "edge": (prev_cluster, cluster_name),
                            "projection": val,
                            "cluster_index": i,
                            "edge_index": 0
                        }
                    else:
                        val = _projection(Ek, Ei, Enext)
                        sample_projections[s] = {
                            "cluster": cluster_name,
                            "edge": (cluster_name, next_cluster),
                            "projection": val,
                            "cluster_index": i,
                            "edge_index": 1
                        }

    if verbose:
        print(f"Projection complete for {len(sample_projections)} samples")
    
    return sample_projections


def order_samples_along_paths(
    sample_projections: Dict[str, Dict],
    main_path: List[int],
    verbose: bool = False
) -> List[str]:
    """
    Orders samples along the trajectory path.
    """
    sortable = []
    for sample_id, info in sample_projections.items():
        c_idx = info["cluster_index"]
        e_idx = info["edge_index"] if info["edge_index"] is not None else 0
        proj = info["projection"]
        sortable.append((c_idx, e_idx, proj, sample_id))

    sortable.sort(key=lambda x: (x[0], x[1], x[2]))
    ordered_samples = [tup[3] for tup in sortable]

    if verbose:
        print(f"Sample ordering complete: {len(ordered_samples)} samples")
    
    return ordered_samples


def plot_clusters_by_cluster(
    adata: sc.AnnData,
    sample_cluster: Dict[str, List[str]],
    main_path: List[int],
    branching_paths: List[List[int]],
    output_path: str,
    pca_key: str = "X_pca_expression",
    verbose: bool = False
):
    """Plot PCA with samples colored by cluster assignment."""
    if pca_key not in adata.uns:
        raise KeyError(f"Missing PCA data in adata.uns['{pca_key}'].")
    pca_df = adata.uns[pca_key]
    
    if not isinstance(pca_df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame in adata.uns['{pca_key}'], but got {type(pca_df)}.")

    required_pcs = {"PC1", "PC2"}
    if not required_pcs.issubset(pca_df.columns):
        raise ValueError(f"PCA DataFrame must contain at least {required_pcs}. Found columns: {pca_df.columns.tolist()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cluster_names = sorted(sample_cluster.keys())
    cluster_centroids = {}
    for clust in cluster_names:
        available_samples = [s for s in sample_cluster[clust] if s in pca_df.index]
        if len(available_samples) > 0:
            subset = pca_df.loc[available_samples, ["PC1", "PC2"]]
            centroid = subset.mean(axis=0).values
            cluster_centroids[clust] = centroid

    def _connect_clusters(c1, c2, style="-", color="k", linewidth=2):
        if c1 in cluster_centroids and c2 in cluster_centroids:
            p1 = cluster_centroids[c1]
            p2 = cluster_centroids[c2]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle=style, color=color, linewidth=linewidth, zorder=1)

    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab20", len(cluster_names))

    for i, clust in enumerate(cluster_names):
        available_samples = [s for s in sample_cluster[clust] if s in pca_df.index]
        if len(available_samples) > 0:
            subset = pca_df.loc[available_samples, ["PC1", "PC2"]]
            plt.scatter(subset["PC1"], subset["PC2"], color=cmap(i), label=clust, s=60, alpha=0.8, edgecolors="k")

    # Draw main path
    for i in range(len(main_path) - 1):
        _connect_clusters(f"cluster_{main_path[i] + 1}", f"cluster_{main_path[i + 1] + 1}", style="-", color="red", linewidth=3)

    # Draw branching paths
    for path in branching_paths:
        for j in range(len(path) - 1):
            _connect_clusters(f"cluster_{path[j] + 1}", f"cluster_{path[j + 1] + 1}", style="--", color="blue", linewidth=2)

    # Label clusters
    for clust in cluster_names:
        if clust in cluster_centroids:
            cx, cy = cluster_centroids[clust]
            plt.text(cx, cy, clust.replace("cluster_", ""), fontsize=12, ha="center", va="center", 
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))

    plt.title("PCA (2D) - Samples Colored by Cluster", fontsize=14)
    plt.xlabel("PC1", fontsize=12)
    plt.ylabel("PC2", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", bbox_to_anchor=(1.05, 1), fontsize=10)
    plt.tight_layout()
    
    plot_path = os.path.join(output_path, "clusters_by_cluster.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Cluster plot saved to {plot_path}")


def plot_clusters_by_grouping(
    adata: sc.AnnData,
    sample_cluster: Dict[str, List[str]],
    main_path: List[int],
    branching_paths: List[List[int]],
    output_path: str,
    pca_key: str = "X_pca_expression",
    grouping_columns: List[str] = ["sev.level"],
    verbose: bool = False
):
    """Plot PCA with samples colored by grouping/metadata."""
    pca_df = adata.uns[pca_key].copy()
    
    # Reset index to make sample names a column for easier merging
    pca_df = pca_df[["PC1", "PC2"]].copy().reset_index().rename(columns={"index": "sample"})
    pca_df["sample"] = pca_df["sample"].astype(str).str.strip()

    # Get grouping info
    sample_list = pca_df["sample"].tolist()
    diff_groups = find_sample_grouping(adata, sample_list, grouping_columns)
    
    # Convert to DataFrame
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient="index", columns=["plot_group"])
    diff_groups.index = diff_groups.index.astype(str).str.strip()
    diff_groups = diff_groups.reset_index().rename(columns={"index": "sample"})

    # Merge and process grouping info
    pca_df = pca_df.merge(diff_groups, how="left", on="sample")
    
    # Try to extract numeric values for coloring
    pca_df["numeric_value"] = pd.to_numeric(pca_df["plot_group"].str.extract(r"(\d+\.?\d*)")[0], errors='coerce')
    
    if pca_df["numeric_value"].notnull().any():
        color_values = pca_df["numeric_value"].fillna(pca_df["numeric_value"].median())
        cmap = "viridis_r"
        colorbar_label = f"{'/'.join(grouping_columns)} (numeric)"
    else:
        unique_groups = pca_df["plot_group"].unique()
        color_map = {group: i for i, group in enumerate(unique_groups)}
        color_values = pca_df["plot_group"].map(color_map)
        cmap = "tab10"
        colorbar_label = f"{'/'.join(grouping_columns)} (categorical)"

    # Compute cluster centroids
    cluster_names = sorted(sample_cluster.keys())
    cluster_centroids = {}
    full_pca_df = adata.uns[pca_key]
    for clust in cluster_names:
        available_samples = [s for s in sample_cluster[clust] if s in full_pca_df.index]
        if len(available_samples) > 0:
            subset = full_pca_df.loc[available_samples, ["PC1", "PC2"]]
            centroid = subset.mean(axis=0).values
            cluster_centroids[clust] = centroid

    def _connect_clusters(c1, c2, style="-", color="k", linewidth=2):
        if c1 in cluster_centroids and c2 in cluster_centroids:
            p1 = cluster_centroids[c1]
            p2 = cluster_centroids[c2]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle=style, color=color, linewidth=linewidth, zorder=1)

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter_obj = plt.scatter(
        pca_df["PC1"], pca_df["PC2"], c=color_values, cmap=cmap, 
        s=80, alpha=0.8, edgecolors="k", zorder=2
    )

    # Draw main path
    for i in range(len(main_path) - 1):
        _connect_clusters(f"cluster_{main_path[i] + 1}", f"cluster_{main_path[i + 1] + 1}", 
                         style="-", color="red", linewidth=3)

    # Draw branching paths
    for path in branching_paths:
        for j in range(len(path) - 1):
            _connect_clusters(f"cluster_{path[j] + 1}", f"cluster_{path[j + 1] + 1}", 
                             style="--", color="blue", linewidth=2)

    # Label clusters
    for clust in cluster_names:
        if clust in cluster_centroids:
            cx, cy = cluster_centroids[clust]
            plt.text(cx, cy, clust.replace("cluster_", ""), fontsize=10, ha="center", va="center", 
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))

    plt.colorbar(scatter_obj, label=colorbar_label)
    plt.title("PCA (2D) - Samples Colored by Grouping", fontsize=14)
    plt.xlabel("PC1", fontsize=12)
    plt.ylabel("PC2", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_path, "clusters_by_grouping.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Grouping plot saved to {plot_path}")


def TSCAN(
    AnnData_sample: sc.AnnData,
    column: str,
    n_clusters: int,
    output_dir: str,
    grouping_columns: Optional[List[str]] = None,
    verbose: bool = False,
    origin: Optional[int] = None
) -> Dict:
    """
    Trajectory analysis using TSCAN algorithm for pseudobulk data.
    
    Parameters:
    -----------
    AnnData_sample : sc.AnnData
        AnnData object containing sample-level data
    column : str
        Key for PCA data in adata.uns
    n_clusters : int
        Number of clusters for K-means
    output_dir : str
        Directory to save results
    grouping_columns : List[str], optional
        Columns from adata.obs to use for grouping visualization
    verbose : bool
        Whether to print detailed progress information
    origin : int, optional
        Cluster index to use as trajectory origin
    
    Returns:
    --------
    Dict containing trajectory analysis results
    """
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tscan_output_path = os.path.join(output_dir, "TSCAN")
    os.makedirs(tscan_output_path, exist_ok=True)

    if verbose:
        print(f"Starting TSCAN analysis:")
        print(f"  Samples: {AnnData_sample.n_obs}")
        print(f"  Genes: {AnnData_sample.n_vars}")
        print(f"  Clusters: {n_clusters}")
    
    # 1. Cluster samples by PCA
    sample_cluster, pca_df = cluster_samples_by_pca(
        AnnData_sample,
        column=column,
        k=n_clusters,
        random_state=0,
        verbose=verbose
    )

    # 2. MST on cluster centroids
    pairwise_dists = Cluster_distance(pca_df, sample_cluster, metric="euclidean", verbose=verbose)
    mst = construct_MST(pairwise_dists, verbose=verbose)

    # 3. Find main path and determine origin
    main_path = find_principal_path(mst, sample_cluster, verbose=verbose)
    ends = [main_path[0], main_path[-1]]
    if origin is None:
        origin = random.choice(ends)
        if verbose:
            print(f"Using random endpoint as origin: cluster_{origin + 1}")
    elif origin not in ends:
        raise ValueError(f"Provided origin {origin} is not an endpoint. Available endpoints: {ends}")

    # Build graph for branching analysis
    G = nx.from_numpy_array(mst + mst.T)
    branching_paths = find_branching_paths(G, origin, main_path, verbose=verbose)

    # 4. Project + order samples along main path
    sample_projections = project_samples_onto_edges(
        pca_data=pca_df,
        sample_cluster=sample_cluster,
        main_path=main_path,
        verbose=verbose
    )
    ordered_samples = order_samples_along_paths(sample_projections, main_path=main_path, verbose=verbose)

    # 5. Handle branching paths
    sample_projections_branching_paths = {}
    ordered_samples_branching_paths = {}
    for index, path in enumerate(branching_paths):
        sp = project_samples_onto_edges(
            pca_data=pca_df,
            sample_cluster=sample_cluster,
            main_path=path,
            verbose=verbose
        )
        os = order_samples_along_paths(sp, main_path=path, verbose=verbose)

        sample_projections_branching_paths[index] = sp
        ordered_samples_branching_paths[index] = os

    # 6. Compute pseudotime
    # Main path pseudotime (normalized 0-1)
    pseudo_main = {}
    if len(ordered_samples) > 1:
        for i, sample_id in enumerate(ordered_samples):
            pseudo_main[sample_id] = i / (len(ordered_samples) - 1)
    elif len(ordered_samples) == 1:
        pseudo_main[ordered_samples[0]] = 0.0

    # Branching paths pseudotime
    pseudo_branches = {}
    for branch_idx, os in ordered_samples_branching_paths.items():
        if len(os) > 1:
            pseudo_branches[branch_idx] = {sample_id: i / (len(os) - 1) for i, sample_id in enumerate(os)}
        elif len(os) == 1:
            pseudo_branches[branch_idx] = {os[0]: 0.0}
        else:
            pseudo_branches[branch_idx] = {}

    # 7. Add pseudotime to AnnData object
    # Main path pseudotime
    main_pseudotime = np.full(AnnData_sample.n_obs, np.nan)
    for i, sample_id in enumerate(AnnData_sample.obs.index):
        if str(sample_id) in pseudo_main:
            main_pseudotime[i] = pseudo_main[str(sample_id)]
    
    AnnData_sample.obs['tscan_pseudotime_main'] = main_pseudotime
    
    # Cluster assignments
    cluster_assignment = np.full(AnnData_sample.n_obs, 'unassigned', dtype=object)
    for cluster_name, sample_list in sample_cluster.items():
        for sample_id in sample_list:
            try:
                idx = AnnData_sample.obs.index.get_loc(sample_id)
                cluster_assignment[idx] = cluster_name
            except KeyError:
                if verbose:
                    print(f"Warning: Sample {sample_id} not found in AnnData obs index")
    
    AnnData_sample.obs['tscan_cluster'] = cluster_assignment

    # 8. Visualization
    try:
        plot_clusters_by_cluster(
            adata=AnnData_sample,
            sample_cluster=sample_cluster,
            main_path=main_path,
            branching_paths=branching_paths,
            output_path=tscan_output_path,
            pca_key=column,
            verbose=verbose
        )
        if verbose:
            print("Cluster plot created successfully")
    except Exception as e:
        if verbose:
            print(f"Warning: Cluster plot failed - {e}")

    if grouping_columns is not None:
        try:
            # Handle string input for grouping columns
            if isinstance(grouping_columns, str):
                actual_grouping_columns = [grouping_columns]
            else:
                actual_grouping_columns = grouping_columns
            
            plot_clusters_by_grouping(
                adata=AnnData_sample,
                sample_cluster=sample_cluster,
                main_path=main_path,
                branching_paths=branching_paths,
                output_path=tscan_output_path,
                pca_key=column,
                grouping_columns=actual_grouping_columns,
                verbose=verbose
            )
            if verbose:
                print("Grouping plot created successfully")
        except Exception as e:
            if verbose:
                print(f"Warning: Grouping plot failed - {e}")

    # 9. Prepare final results
    results = {
        "main_path": main_path,
        "origin": origin,
        "branching_paths": branching_paths,
        "graph": G,
        "sample_cluster": sample_cluster,
        "pca_data": pca_df,
        "sample_projections": sample_projections,
        "ordered_samples": ordered_samples,
        "sample_projections_branching_paths": sample_projections_branching_paths,
        "ordered_samples_branching_paths": ordered_samples_branching_paths,
        "pseudotime": {
            "main_path": pseudo_main,
            "branching_paths": pseudo_branches
        },
        "cluster_names": sorted(sample_cluster.keys()),
        "n_samples_total": sum(len(samples) for samples in sample_cluster.values())
    }

    if verbose:
        print(f"\nTSCAN analysis completed successfully!")
        print(f"  Main path: {len(ordered_samples)} samples across {len(main_path)} clusters")
        for idx, bp in ordered_samples_branching_paths.items():
            print(f"  Branch {idx+1}: {len(bp)} samples across {len(branching_paths[idx])} clusters")
        print(f"  Total samples processed: {results['n_samples_total']}")
        print(f"  Results saved to: {tscan_output_path}")

    return results