import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import Dict, List, Tuple, Optional, Union
from visualization.visualization_helper import plot_clusters_by_cluster, plot_clusters_by_grouping


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


def cluster_samples_gmm(
    pca_data: pd.DataFrame,
    n_clusters: Optional[int] = None,
    max_clusters: int = 20,
    random_state: int = 0,
    verbose: bool = False
) -> Tuple[Dict[str, List[str]], np.ndarray]:
    """
    Cluster samples using Gaussian Mixture Model (GMM), following TSCAN's mclust approach.
    
    Uses BIC to select optimal number of clusters if not specified.
    GMM uses covariance_type='full' for ellipsoidal clusters with varying 
    volume, shape, and orientation (matching mclust's 'VVV' model).
    
    Parameters:
    -----------
    pca_data : pd.DataFrame
        PCA coordinates (samples x PCs)
    n_clusters : int, optional
        Number of clusters. If None, determined by BIC.
    max_clusters : int
        Maximum clusters to try when using BIC (default: 20)
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    Tuple[Dict[str, List[str]], np.ndarray]
        - Dictionary mapping cluster names to sample lists
        - Cluster labels for each sample
    """
    X = pca_data.values
    n_samples = X.shape[0]
    
    if n_clusters is None:
        # Use BIC to select optimal number of clusters (as in mclust)
        max_k = min(max_clusters, n_samples - 1)
        best_bic = np.inf
        best_k = 1
        
        if verbose:
            print("Determining optimal cluster number using BIC...")
        
        for k in range(1, max_k + 1):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type='full',  # Ellipsoidal, varying volume/shape/orientation
                    random_state=random_state,
                    n_init=3
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                
                if bic < best_bic:
                    best_bic = bic
                    best_k = k
            except Exception:
                continue
        
        n_clusters = best_k
        if verbose:
            print(f"  Optimal clusters by BIC: {n_clusters}")
    
    # Fit final GMM with selected number of clusters
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        random_state=random_state,
        n_init=3
    )
    cluster_labels = gmm.fit_predict(X)
    
    # Build sample_cluster dictionary
    cluster_dict = defaultdict(list)
    for sample, cluster_idx in zip(pca_data.index, cluster_labels):
        cluster_name = f"cluster_{cluster_idx + 1}"
        cluster_dict[cluster_name].append(str(sample))
    
    sample_cluster = dict(cluster_dict)
    
    if verbose:
        print(f"GMM clustering complete:")
        print(f"  Clusters: {len(sample_cluster)}")
        for name, samples in sorted(sample_cluster.items()):
            print(f"    {name}: {len(samples)} samples")
    
    return sample_cluster, cluster_labels


def cluster_samples_by_pca(
    adata: sc.AnnData,
    column: str,
    n_clusters: Optional[int] = None,
    random_state: int = 0,
    verbose: bool = False
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Cluster samples based on PCA coordinates using GMM with BIC.
    
    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object containing PCA data
    column : str
        Key for PCA data in adata.uns
    n_clusters : int, optional
        Number of clusters. If None, determined automatically by BIC.
    random_state : int
        Random seed
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    Tuple[Dict[str, List[str]], pd.DataFrame]
        - Dictionary mapping cluster names to sample lists
        - PCA DataFrame
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

    n_samples = pca_data.shape[0]

    if n_clusters is not None and n_clusters > n_samples:
        raise ValueError(f"n_clusters={n_clusters} cannot exceed n_samples={n_samples}.")

    # 2. GMM clustering with BIC
    sample_cluster, _ = cluster_samples_gmm(
        pca_data,
        n_clusters=n_clusters,
        random_state=random_state,
        verbose=verbose
    )

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
    
    Following TSCAN paper:
    - Projection of sample k onto edge (i,j): v_ij^T * E_k / ||v_ij||
    - where v_ij = E(j) - E(i) and E(i), E(j) are cluster centers
    - Samples in first cluster -> project to edge (C1, C2)
    - Samples in last cluster -> project to edge (C_{M-1}, C_M)
    - Samples in intermediate cluster C_m:
      - If closer to C_{m-1}: project to edge (C_{m-1}, C_m)
      - If closer to C_{m+1}: project to edge (C_m, C_{m+1})
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
        """
        Project sample onto edge following TSCAN formula:
        v_ij^T * (E_k - E_i) / ||v_ij||
        """
        v_ij = Ej - Ei
        norm_v = np.linalg.norm(v_ij)
        if norm_v == 0:
            return 0.0
        return np.dot(v_ij, Ek - Ei) / norm_v

    def _dist_sample_to_centroid(Ek, Ec):
        return np.linalg.norm(Ek - Ec)

    path_cluster_names = [cluster_list[idx] for idx in main_path]
    M = len(path_cluster_names)
    sample_projections = {}

    for i, cluster_name in enumerate(path_cluster_names):
        samples_in_cluster = [s for s in sample_cluster[cluster_name] if s in pca_data.index]
        Ei = cluster_centroids[cluster_name]

        # First cluster: all samples project onto edge (C1, C2)
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

        # Last cluster: all samples project onto edge (C_{M-1}, C_M)
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

        # Intermediate clusters: assign based on distance to neighboring cluster centers
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
                        # Closer to previous cluster -> project onto edge (C_{m-1}, C_m)
                        val = _projection(Ek, Eprev, Ei)
                        sample_projections[s] = {
                            "cluster": cluster_name,
                            "edge": (prev_cluster, cluster_name),
                            "projection": val,
                            "cluster_index": i,
                            "edge_index": 0
                        }
                    else:
                        # Closer to next cluster -> project onto edge (C_m, C_{m+1})
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
    
    Following TSCAN paper, ordering is determined in three steps:
    1. Within same cluster and same edge: order by projection value
    2. Within same cluster but different edges: order by edge order
    3. Between different clusters: order by cluster order
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


def TSCAN(
    AnnData_sample: sc.AnnData,
    column: str,
    n_clusters: Optional[int] = None,
    output_dir: str = "./",
    grouping_columns: Optional[List[str]] = None,
    verbose: bool = False,
    origin: Optional[int] = None,
) -> Dict:
    """
    Trajectory analysis using TSCAN algorithm for pseudobulk data.
    
    This implementation follows the original TSCAN paper (Ji & Ji, NAR 2016):
    1. Sample clustering using GMM (mclust-like) with BIC for automatic cluster selection
    2. MST construction on cluster centroids
    3. Sample projection onto MST edges
    4. Pseudotime ordering based on sample rank
    
    Parameters:
    -----------
    AnnData_sample : sc.AnnData
        AnnData object containing sample-level data with pre-computed PCA
    column : str
        Key for PCA data in adata.uns
    n_clusters : int, optional
        Number of clusters for sample clustering.
        If None, determined automatically using BIC (TSCAN default)
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tscan_output_path = os.path.join(output_dir, "TSCAN")
    os.makedirs(tscan_output_path, exist_ok=True)

    # Safe version of column name for filenames
    safe_column = str(column).replace(os.sep, "_")

    if verbose:
        print(f"Starting TSCAN analysis:")
        print(f"  Samples: {AnnData_sample.n_obs}")
        print(f"  Genes: {AnnData_sample.n_vars}")
        print(f"  N clusters: {'auto (BIC)' if n_clusters is None else n_clusters}")
        print(f"  Embedding key: {column}")

    # 1. Cluster samples by PCA using GMM with BIC
    sample_cluster, pca_df = cluster_samples_by_pca(
        AnnData_sample,
        column=column,
        n_clusters=n_clusters,
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

    # Ensure main_path starts from origin
    if main_path[0] != origin:
        main_path = main_path[::-1]

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
        os_ordered = order_samples_along_paths(sp, main_path=path, verbose=verbose)

        sample_projections_branching_paths[index] = sp
        ordered_samples_branching_paths[index] = os_ordered

    # 6. Compute pseudotime (rank-based as in original TSCAN, normalized to 0-1)
    # Main path pseudotime
    pseudo_main = {}
    if len(ordered_samples) > 1:
        for i, sample_id in enumerate(ordered_samples):
            pseudo_main[sample_id] = i / (len(ordered_samples) - 1)
    elif len(ordered_samples) == 1:
        pseudo_main[ordered_samples[0]] = 0.0

    # Branching paths pseudotime
    pseudo_branches = {}
    for branch_idx, os_ordered in ordered_samples_branching_paths.items():
        if len(os_ordered) > 1:
            pseudo_branches[branch_idx] = {sample_id: i / (len(os_ordered) - 1) for i, sample_id in enumerate(os_ordered)}
        elif len(os_ordered) == 1:
            pseudo_branches[branch_idx] = {os_ordered[0]: 0.0}
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
                # Find the sample in the obs index
                if sample_id in AnnData_sample.obs.index:
                    idx = AnnData_sample.obs.index.get_loc(sample_id)
                    cluster_assignment[idx] = cluster_name
                else:
                    # Try string conversion
                    sample_id_str = str(sample_id)
                    if sample_id_str in AnnData_sample.obs.index:
                        idx = AnnData_sample.obs.index.get_loc(sample_id_str)
                        cluster_assignment[idx] = cluster_name
            except KeyError:
                if verbose:
                    print(f"Warning: Sample {sample_id} not found in AnnData obs index")
    
    AnnData_sample.obs['tscan_cluster'] = cluster_assignment

    # 8. Visualization
    try:
        plot_clusters_by_cluster(
            adata=AnnData_sample,
            main_path=main_path,
            branching_paths=branching_paths,
            output_path=tscan_output_path,
            pca_key=column,
            verbose=verbose
        )
        # Rename cluster plot to include embedding name
        cluster_plot_default = os.path.join(tscan_output_path, "clusters_by_cluster.png")
        cluster_plot_named = os.path.join(tscan_output_path, f"clusters_by_cluster_{safe_column}.png")
        if os.path.exists(cluster_plot_default):
            os.replace(cluster_plot_default, cluster_plot_named)
            if verbose:
                print(f"Cluster plot saved as: {cluster_plot_named}")
        elif verbose:
            print("Warning: Expected cluster plot file not found to rename.")
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
                main_path=main_path,
                branching_paths=branching_paths,
                output_path=tscan_output_path,
                pca_key=column,
                grouping_columns=actual_grouping_columns,
                verbose=verbose
            )

            # Rename grouping plot to include embedding name
            grouping_plot_default = os.path.join(tscan_output_path, "clusters_by_grouping.png")
            grouping_plot_named = os.path.join(tscan_output_path, f"clusters_by_grouping_{safe_column}.png")
            if os.path.exists(grouping_plot_default):
                os.replace(grouping_plot_default, grouping_plot_named)
                if verbose:
                    print(f"Grouping plot saved as: {grouping_plot_named}")
            elif verbose:
                print("Warning: Expected grouping plot file not found to rename.")
        except Exception as e:
            if verbose:
                print(f"Warning: Grouping plot failed - {e}")

    # 9. Save pseudotime data to a single CSV file
    try:
        pseudotime_data = []
        
        # Main path samples
        for sample_id, pseudotime_val in pseudo_main.items():
            pseudotime_data.append({
                'sample_id': sample_id,
                'trajectory_type': 'main_path',
                'branch_id': 'main',
                'pseudotime': pseudotime_val,
                'cluster': sample_projections[sample_id]['cluster'] if sample_id in sample_projections else 'unknown'
            })
        
        # Branching path samples
        for branch_idx, branch_pseudo in pseudo_branches.items():
            for sample_id, pseudotime_val in branch_pseudo.items():
                pseudotime_data.append({
                    'sample_id': sample_id,
                    'trajectory_type': 'branch',
                    'branch_id': f'branch_{branch_idx}',
                    'pseudotime': pseudotime_val,
                    'cluster': sample_projections_branching_paths[branch_idx][sample_id]['cluster'] if sample_id in sample_projections_branching_paths[branch_idx] else 'unknown'
                })
        
        pseudotime_df = pd.DataFrame(pseudotime_data)

        pseudotime_path = os.path.join(
            tscan_output_path,
            f"{safe_column}_pseudotime.csv"
        )
        pseudotime_df.to_csv(pseudotime_path, index=False)
        
        if verbose:
            print(f"Pseudotime data saved: {pseudotime_path}")
            print(f"  Total samples in CSV: {len(pseudotime_df)}")
    
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to save pseudotime CSV file - {e}")

    # 10. Prepare final results
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
        "n_samples_total": sum(len(samples) for samples in sample_cluster.values()),
    }

    if verbose:
        print(f"\nTSCAN analysis completed successfully!")
        print(f"  Main path: {len(ordered_samples)} samples across {len(main_path)} clusters")
        for idx, bp in ordered_samples_branching_paths.items():
            print(f"  Branch {idx+1}: {len(bp)} samples across {len(branching_paths[idx])} clusters")
        print(f"  Total samples processed: {results['n_samples_total']}")
        print(f"  Results saved to: {tscan_output_path}")

    return results