import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from typing import List, Tuple

import matplotlib.pyplot as plt
from typing import Tuple

def generate_umap_visualizations(
    adata: sc.AnnData,
    output_dir: str,
    groupby: str = "cell_type",
    figsize: Tuple[int, int] = (12, 8),
    point_size: float = 20,
    dpi: int = 300,
    palette: str = "tab20",
    verbose: bool = True,
) -> sc.AnnData:
    """
    Save UMAP coloured by `groupby` to `<output_dir>/preprocess/`.

    Requires that neighbors and UMAP coordinates already exist in `adata`.

    Parameters
    ----------
    adata
        AnnData with existing neighbors graph and X_umap.
    output_dir
        Parent folder; figure goes in a `preprocess/` subdirectory.
    groupby
        .obs column used for colouring.
    figsize, point_size, dpi
        Plot appearance parameters.
    palette
        Matplotlib/seaborn colour palette name.
    verbose
        Print progress messages.

    Returns
    -------
    AnnData
        The same object (unchanged).
    """
    if verbose:
        print("[generate_umap_visualizations] Generating UMAP plots...")

    if groupby not in adata.obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")

    if "X_umap" not in adata.obsm:
        raise ValueError("UMAP coordinates not found. Please run UMAP computation first.")

    output_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(output_dir, exist_ok=True)

    sc.settings.set_figure_params(dpi=dpi, facecolor="white", figsize=figsize)

    if verbose:
        n_categories = adata.obs[groupby].nunique()
        print(f"[generate_umap_visualizations] → Plotting UMAP colored by '{groupby}' ({n_categories} categories)")

    fig = sc.pl.umap(
        adata,
        color=groupby,
        palette=palette,
        size=point_size,
        alpha=0.8,
        legend_loc="right margin",
        legend_fontsize=10,
        title=f"UMAP – {groupby.replace('_', ' ').title()}",
        show=False,
        return_fig=True,
    )

    outfile = os.path.join(output_dir, f"umap_{groupby}.png")
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if verbose:
        print(f"[generate_umap_visualizations] ✓ Saved: {outfile}")

    return adata


def plot_cell_type_abundances(proportions: pd.DataFrame, output_dir: str):
    """
    Stacked bar plot of cell-type proportions across samples.

    Parameters
    ----------
    proportions : pd.DataFrame
        Samples × cell-types proportion matrix (rows = samples, columns = cell types).
    output_dir : str
        Directory to save the output plot.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Automatically generating output directory")

    proportions = proportions.sort_index()
    cell_types = proportions.columns.tolist()

    num_cell_types = len(cell_types)
    colors = sns.color_palette('tab20', n_colors=num_cell_types)

    plt.figure(figsize=(12, 8))

    bottom = np.zeros(len(proportions))
    sample_indices = np.arange(len(proportions))

    for idx, cell_type in enumerate(cell_types):
        values = proportions[cell_type].values
        plt.bar(
            sample_indices,
            values,
            bottom=bottom,
            color=colors[idx],
            edgecolor='white',
            width=0.8,
            label=cell_type
        )
        bottom += values

    plt.ylabel('Proportion', fontsize=14)
    plt.title('Cell Type Proportions Across Samples', fontsize=16)
    plt.xticks(sample_indices, proportions.index, rotation=90, fontsize=10)
    plt.yticks(fontsize=12)
    plt.legend(title='Cell Types', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'cell_type_abundances.pdf')
    plt.savefig(plot_path)
    plt.close()
    print(f"Cell type abundance plot saved to {plot_path}")

def visualizeDistanceMatrix(sample_distance_matrix, heatmap_path):
    """
    Save a clustered heatmap of a square sample-distance matrix.

    Parameters
    ----------
    sample_distance_matrix : pd.DataFrame
        Square distance matrix (samples × samples).
    heatmap_path : str
        File path to save the heatmap.
    """
    condensed_distances = squareform(sample_distance_matrix.values)
    linkage_matrix = linkage(condensed_distances, method='average')
    sns.clustermap(
        sample_distance_matrix,
        cmap='viridis',
        linewidths=0.5,
        annot=True,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix
    )
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Sample distance heatmap saved to {heatmap_path}")


def plot_clusters_by_cluster(
    adata: sc.AnnData,
    main_path: List[int],
    branching_paths: List[List[int]],
    output_path: str,
    pca_key: str = "X_DR_sample",
    cluster_col: str = "tscan_cluster",
    verbose: bool = False
):
    """
    Plot the TSCAN trajectory with samples coloured by cluster assignment.

    Centroids of each cluster are connected by solid red lines (main path)
    or dashed blue lines (branching paths).
    """
    if pca_key not in adata.uns:
        raise KeyError(f"Missing PCA data in adata.uns['{pca_key}'].")

    pca_df = adata.uns[pca_key]
    if not isinstance(pca_df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame in adata.uns['{pca_key}'], but got {type(pca_df)}.")

    # Detect which embedding columns to use (PCA, LSI, or generic first-two).
    dim_columns = pca_df.columns.tolist()
    if "PC1" in dim_columns and "PC2" in dim_columns:
        dim1, dim2 = "PC1", "PC2"
    elif "LSI1" in dim_columns and "LSI2" in dim_columns:
        dim1, dim2 = "LSI1", "LSI2"
    elif len(dim_columns) >= 2:
        dim1, dim2 = dim_columns[0], dim_columns[1]
    else:
        raise ValueError(f"Need at least 2 dimensions for plotting. Found columns: {dim_columns}")

    if cluster_col not in adata.obs.columns:
        raise KeyError(f"Cluster column '{cluster_col}' not found in adata.obs")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cluster_assignments = adata.obs[cluster_col].copy()
    valid_samples = cluster_assignments != 'unassigned'
    cluster_assignments = cluster_assignments[valid_samples]

    # Case-insensitive index matching to tolerate capitalisation differences.
    pca_index_lower = pd.Index([str(idx).lower() for idx in pca_df.index])
    obs_index_lower = pd.Index([str(idx).lower() for idx in cluster_assignments.index])

    common_samples_lower = pca_index_lower.intersection(obs_index_lower)

    if len(common_samples_lower) > 0:
        pca_to_lower = dict(zip(pca_df.index, pca_index_lower))
        obs_to_lower = dict(zip(cluster_assignments.index, obs_index_lower))

        lower_to_pca = {v: k for k, v in pca_to_lower.items()}
        lower_to_obs = {v: k for k, v in obs_to_lower.items()}

        common_pca = [lower_to_pca[s] for s in common_samples_lower]
        common_obs = [lower_to_obs[s] for s in common_samples_lower]

        pca_subset = pca_df.loc[common_pca, [dim1, dim2]].copy()
        pca_subset.index = common_obs
        cluster_subset = cluster_assignments.loc[common_obs]

        common_samples = common_obs
    else:
        common_samples = pca_df.index.intersection(cluster_assignments.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples found between PCA data and cluster assignments")
        pca_subset = pca_df.loc[common_samples, [dim1, dim2]]
        cluster_subset = cluster_assignments.loc[common_samples]

    unique_clusters = sorted([c for c in cluster_subset.unique() if c != 'unassigned'])
    cluster_centroids = {}

    for cluster_name in unique_clusters:
        cluster_samples = cluster_subset[cluster_subset == cluster_name].index
        if len(cluster_samples) > 0:
            subset_coords = pca_subset.loc[cluster_samples, [dim1, dim2]]
            centroid = subset_coords.mean(axis=0).values
            cluster_centroids[cluster_name] = centroid

    def _connect_clusters(c1, c2, style="-", color="k", linewidth=2):
        if c1 in cluster_centroids and c2 in cluster_centroids:
            p1 = cluster_centroids[c1]
            p2 = cluster_centroids[c2]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    linestyle=style, color=color, linewidth=linewidth, zorder=1)

    plt.figure(figsize=(12, 9))

    n_clusters = len(unique_clusters)
    cmap = plt.get_cmap("tab20", n_clusters)
    cluster_to_color = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}

    for i, cluster_name in enumerate(unique_clusters):
        cluster_samples = cluster_subset[cluster_subset == cluster_name].index
        if len(cluster_samples) > 0:
            subset_coords = pca_subset.loc[cluster_samples, [dim1, dim2]]
            plt.scatter(subset_coords[dim1], subset_coords[dim2],
                       color=cluster_to_color[cluster_name],
                       label=cluster_name, s=60, alpha=0.8,
                       edgecolors="k", linewidth=0.5, zorder=2)

    if len(main_path) > 1:
        for i in range(len(main_path) - 1):
            cluster1 = f"cluster_{main_path[i] + 1}"
            cluster2 = f"cluster_{main_path[i + 1] + 1}"
            _connect_clusters(cluster1, cluster2, style="-", color="red", linewidth=4)

    for path in branching_paths:
        if len(path) > 1:
            for j in range(len(path) - 1):
                cluster1 = f"cluster_{path[j] + 1}"
                cluster2 = f"cluster_{path[j + 1] + 1}"
                _connect_clusters(cluster1, cluster2, style="--", color="blue", linewidth=3)

    for cluster_name in unique_clusters:
        if cluster_name in cluster_centroids:
            cx, cy = cluster_centroids[cluster_name]
            cluster_num = cluster_name.replace("cluster_", "")
            plt.text(cx, cy, cluster_num, fontsize=12, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.4"),
                    zorder=3)

    plt.title("TSCAN Trajectory - Samples Colored by Cluster", fontsize=16, pad=20)
    plt.xlabel(dim1, fontsize=14)
    plt.ylabel(dim2, fontsize=14)
    plt.grid(True, alpha=0.3)

    legend = plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                       fontsize=10, title="Clusters", title_fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()

    plot_path = os.path.join(output_path, "clusters_by_cluster.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    if verbose:
        print(f"Cluster plot saved to {plot_path}")
        print(f"Plotted {len(common_samples)} samples across {len(unique_clusters)} clusters")


def plot_clusters_by_grouping(
    adata: sc.AnnData,
    main_path: List[int],
    branching_paths: List[List[int]],
    output_path: str,
    pca_key: str = "X_DR_sample",
    grouping_columns: List[str] = ["sev.level"],
    verbose: bool = False
):
    """
    Plot the TSCAN trajectory with samples coloured by a metadata grouping column.

    Multiple grouping columns are concatenated with '_'. Numeric groupings get a
    viridis_r continuous colormap; categorical groupings get tab10.
    """
    pca_df = adata.uns[pca_key]

    # Detect embedding axis names (PCA, LSI, or generic first-two).
    dim_columns = pca_df.columns.tolist()
    if "PC1" in dim_columns and "PC2" in dim_columns:
        dim1, dim2 = "PC1", "PC2"
    elif "LSI1" in dim_columns and "LSI2" in dim_columns:
        dim1, dim2 = "LSI1", "LSI2"
    elif len(dim_columns) >= 2:
        dim1, dim2 = dim_columns[0], dim_columns[1]
    else:
        raise ValueError(f"Need at least 2 dimensions for plotting. Found columns: {dim_columns}")

    cluster_assignments = adata.obs['tscan_cluster'].copy()
    valid_samples = cluster_assignments != 'unassigned'
    cluster_assignments = cluster_assignments[valid_samples]

    # Case-insensitive index matching.
    pca_index_lower = pd.Index([str(idx).lower() for idx in pca_df.index])
    obs_index_lower = pd.Index([str(idx).lower() for idx in cluster_assignments.index])

    common_samples_lower = pca_index_lower.intersection(obs_index_lower)

    if len(common_samples_lower) > 0:
        pca_to_lower = dict(zip(pca_df.index, pca_index_lower))
        obs_to_lower = dict(zip(cluster_assignments.index, obs_index_lower))

        lower_to_pca = {v: k for k, v in pca_to_lower.items()}
        lower_to_obs = {v: k for k, v in obs_to_lower.items()}

        common_pca = [lower_to_pca[s] for s in common_samples_lower]
        common_obs = [lower_to_obs[s] for s in common_samples_lower]

        pca_subset = pca_df.loc[common_pca, [dim1, dim2]].copy()
        pca_subset.index = common_obs
        cluster_subset = cluster_assignments.loc[common_obs]

        common_samples = common_obs
    else:
        common_samples = pca_df.index.intersection(cluster_assignments.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples found between PCA data and cluster assignments")
        pca_subset = pca_df.loc[common_samples, [dim1, dim2]]
        cluster_subset = cluster_assignments.loc[common_samples]

    if len(grouping_columns) == 1:
        grouping_values = adata.obs.loc[common_samples, grouping_columns[0]].astype(str)
    else:
        grouping_values = adata.obs.loc[common_samples, grouping_columns].astype(str).agg('_'.join, axis=1)

    # Use a continuous scale for numeric groups, discrete for categorical.
    numeric_values = pd.to_numeric(grouping_values.str.extract(r"(\d+\.?\d*)")[0], errors='coerce')

    if numeric_values.notnull().any():
        color_values = numeric_values.fillna(numeric_values.median())
        cmap = "viridis_r"
        colorbar_label = f"{'/'.join(grouping_columns)} (numeric)"
    else:
        unique_groups = grouping_values.unique()
        color_map = {group: i for i, group in enumerate(unique_groups)}
        color_values = grouping_values.map(color_map)
        cmap = "tab10"
        colorbar_label = f"{'/'.join(grouping_columns)} (categorical)"

    cluster_names = sorted([c for c in cluster_subset.unique() if c != 'unassigned'])
    cluster_centroids = {}

    for cluster_name in cluster_names:
        cluster_samples = cluster_subset[cluster_subset == cluster_name].index
        if len(cluster_samples) > 0:
            subset_coords = pca_subset.loc[cluster_samples, [dim1, dim2]]
            centroid = subset_coords.mean(axis=0).values
            cluster_centroids[cluster_name] = centroid

    def _connect_clusters(c1, c2, style="-", color="k", linewidth=2):
        if c1 in cluster_centroids and c2 in cluster_centroids:
            p1 = cluster_centroids[c1]
            p2 = cluster_centroids[c2]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle=style, color=color, linewidth=linewidth, zorder=1)

    plt.figure(figsize=(10, 8))
    scatter_obj = plt.scatter(
        pca_subset[dim1], pca_subset[dim2], c=color_values, cmap=cmap,
        s=80, alpha=0.8, edgecolors="k", zorder=2
    )

    for i in range(len(main_path) - 1):
        _connect_clusters(f"cluster_{main_path[i] + 1}", f"cluster_{main_path[i + 1] + 1}",
                         style="-", color="red", linewidth=3)

    for path in branching_paths:
        for j in range(len(path) - 1):
            _connect_clusters(f"cluster_{path[j] + 1}", f"cluster_{path[j + 1] + 1}",
                             style="--", color="blue", linewidth=2)

    for clust in cluster_names:
        if clust in cluster_centroids:
            cx, cy = cluster_centroids[clust]
            plt.text(cx, cy, clust.replace("cluster_", ""), fontsize=10, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))

    plt.colorbar(scatter_obj, label=colorbar_label)
    plt.title("PCA/LSI (2D) - Samples Colored by Grouping", fontsize=14)
    plt.xlabel(dim1, fontsize=12)
    plt.ylabel(dim2, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_path, "clusters_by_grouping.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Grouping plot saved to {plot_path}")