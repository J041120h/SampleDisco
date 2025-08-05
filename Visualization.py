import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from anndata import AnnData
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.sparse import issparse
import plotly.express as px
import plotly.io as pio
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import List, Dict
from Grouping import find_sample_grouping
from visualization_emebedding import plot_sample_cell_proportions_embedding, plot_sample_cell_expression_embedding


#------------------- VISUALIZATION FOR SAMPLE DISTANCE -------------------

def plot_cell_type_abundances(proportions: pd.DataFrame, output_dir: str):
    """
    Generate a stacked bar plot to visualize the cell type proportions across samples.

    Parameters:
    ----------
    proportions : pd.DataFrame
        DataFrame containing cell type proportions for each sample.
        Rows represent samples, and columns represent cell types.
    output_dir : str
        Directory to save the output plot.

    Returns:
    -------
    None
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
    Generate and save a clustered heatmap from a sample distance matrix.

    Parameters
    ----------
    sample_distance_matrix : pd.DataFrame
        A square distance matrix (samples x samples) with numerical values.
    heatmap_path : str
        File path to save the resulting heatmap figure.
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

#------------------- VISUALIZATION FOR TSCAN -------------------


def plot_clusters_by_cluster(
    adata: sc.AnnData,
    main_path: List[int],
    branching_paths: List[List[int]],
    output_path: str,
    pca_key: str = "X_DR_expression",
    cluster_col: str = "tscan_cluster",
    verbose: bool = False
):
    """
    Plot PCA with samples colored by cluster assignment using adata.obs.
    """
    if pca_key not in adata.uns:
        raise KeyError(f"Missing PCA data in adata.uns['{pca_key}'].")
    
    pca_df = adata.uns[pca_key]
    if not isinstance(pca_df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame in adata.uns['{pca_key}'], but got {type(pca_df)}.")

    # Handle different dimensionality reduction methods (PCA, LSI, etc.)
    dim_columns = pca_df.columns.tolist()
    if "PC1" in dim_columns and "PC2" in dim_columns:
        dim1, dim2 = "PC1", "PC2"
    elif "LSI1" in dim_columns and "LSI2" in dim_columns:
        dim1, dim2 = "LSI1", "LSI2"
    elif len(dim_columns) >= 2:
        # Use first two dimensions if named differently
        dim1, dim2 = dim_columns[0], dim_columns[1]
    else:
        raise ValueError(f"Need at least 2 dimensions for plotting. Found columns: {dim_columns}")

    if cluster_col not in adata.obs.columns:
        raise KeyError(f"Cluster column '{cluster_col}' not found in adata.obs")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get cluster assignments from adata.obs
    cluster_assignments = adata.obs[cluster_col].copy()
    
    # Filter out unassigned samples
    valid_samples = cluster_assignments != 'unassigned'
    cluster_assignments = cluster_assignments[valid_samples]
    
    # Ensure we only use samples that exist in both PCA data and obs
    # Handle case-insensitive matching
    pca_index_lower = pd.Index([str(idx).lower() for idx in pca_df.index])
    obs_index_lower = pd.Index([str(idx).lower() for idx in cluster_assignments.index])
    
    common_samples_lower = pca_index_lower.intersection(obs_index_lower)
    
    if len(common_samples_lower) > 0:
        # Map back to original indices
        pca_to_lower = dict(zip(pca_df.index, pca_index_lower))
        obs_to_lower = dict(zip(cluster_assignments.index, obs_index_lower))
        
        lower_to_pca = {v: k for k, v in pca_to_lower.items()}
        lower_to_obs = {v: k for k, v in obs_to_lower.items()}
        
        # Get original indices for common samples
        common_pca = [lower_to_pca[s] for s in common_samples_lower]
        common_obs = [lower_to_obs[s] for s in common_samples_lower]
        
        # Create aligned data
        pca_subset = pca_df.loc[common_pca, [dim1, dim2]].copy()
        pca_subset.index = common_obs  # Align indices
        cluster_subset = cluster_assignments.loc[common_obs]
        
        common_samples = common_obs
    else:
        # Try regular intersection
        common_samples = pca_df.index.intersection(cluster_assignments.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples found between PCA data and cluster assignments")
        pca_subset = pca_df.loc[common_samples, [dim1, dim2]]
        cluster_subset = cluster_assignments.loc[common_samples]
    
    # Get unique clusters and compute centroids
    unique_clusters = sorted([c for c in cluster_subset.unique() if c != 'unassigned'])
    cluster_centroids = {}
    
    for cluster_name in unique_clusters:
        cluster_samples = cluster_subset[cluster_subset == cluster_name].index
        if len(cluster_samples) > 0:
            subset_coords = pca_subset.loc[cluster_samples, [dim1, dim2]]
            centroid = subset_coords.mean(axis=0).values
            cluster_centroids[cluster_name] = centroid

    def _connect_clusters(c1, c2, style="-", color="k", linewidth=2):
        """Connect two clusters with a line."""
        if c1 in cluster_centroids and c2 in cluster_centroids:
            p1 = cluster_centroids[c1]
            p2 = cluster_centroids[c2]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                    linestyle=style, color=color, linewidth=linewidth, zorder=1)

    # Create the plot
    plt.figure(figsize=(12, 9))
    
    # Color map for clusters
    n_clusters = len(unique_clusters)
    cmap = plt.cm.get_cmap("tab20", n_clusters)
    cluster_to_color = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}

    # Plot samples colored by cluster
    for i, cluster_name in enumerate(unique_clusters):
        cluster_samples = cluster_subset[cluster_subset == cluster_name].index
        if len(cluster_samples) > 0:
            subset_coords = pca_subset.loc[cluster_samples, [dim1, dim2]]
            plt.scatter(subset_coords[dim1], subset_coords[dim2], 
                       color=cluster_to_color[cluster_name], 
                       label=cluster_name, s=60, alpha=0.8, 
                       edgecolors="k", linewidth=0.5, zorder=2)

    # Draw main path connections
    if len(main_path) > 1:
        for i in range(len(main_path) - 1):
            cluster1 = f"cluster_{main_path[i] + 1}"
            cluster2 = f"cluster_{main_path[i + 1] + 1}"
            _connect_clusters(cluster1, cluster2, style="-", color="red", linewidth=4)

    # Draw branching path connections
    for path in branching_paths:
        if len(path) > 1:
            for j in range(len(path) - 1):
                cluster1 = f"cluster_{path[j] + 1}"
                cluster2 = f"cluster_{path[j + 1] + 1}"
                _connect_clusters(cluster1, cluster2, style="--", color="blue", linewidth=3)

    # Add cluster labels at centroids
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
    
    # Improved legend
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
    pca_key: str = "X_DR_expression",
    grouping_columns: List[str] = ["sev.level"],
    verbose: bool = False
):
    """Plot PCA with samples colored by grouping/metadata."""
    pca_df = adata.uns[pca_key]
    
    # Handle different dimensionality reduction methods
    dim_columns = pca_df.columns.tolist()
    if "PC1" in dim_columns and "PC2" in dim_columns:
        dim1, dim2 = "PC1", "PC2"
    elif "LSI1" in dim_columns and "LSI2" in dim_columns:
        dim1, dim2 = "LSI1", "LSI2"
    elif len(dim_columns) >= 2:
        dim1, dim2 = dim_columns[0], dim_columns[1]
    else:
        raise ValueError(f"Need at least 2 dimensions for plotting. Found columns: {dim_columns}")
    
    # Get cluster assignments from adata.obs
    cluster_assignments = adata.obs['tscan_cluster'].copy()
    
    # Filter out unassigned samples
    valid_samples = cluster_assignments != 'unassigned'
    cluster_assignments = cluster_assignments[valid_samples]
    
    # Ensure we only use samples that exist in both PCA data and obs
    # Handle case-insensitive matching
    pca_index_lower = pd.Index([str(idx).lower() for idx in pca_df.index])
    obs_index_lower = pd.Index([str(idx).lower() for idx in cluster_assignments.index])
    
    common_samples_lower = pca_index_lower.intersection(obs_index_lower)
    
    if len(common_samples_lower) > 0:
        # Map back to original indices
        pca_to_lower = dict(zip(pca_df.index, pca_index_lower))
        obs_to_lower = dict(zip(cluster_assignments.index, obs_index_lower))
        
        lower_to_pca = {v: k for k, v in pca_to_lower.items()}
        lower_to_obs = {v: k for k, v in obs_to_lower.items()}
        
        # Get original indices for common samples
        common_pca = [lower_to_pca[s] for s in common_samples_lower]
        common_obs = [lower_to_obs[s] for s in common_samples_lower]
        
        # Create aligned data
        pca_subset = pca_df.loc[common_pca, [dim1, dim2]].copy()
        pca_subset.index = common_obs  # Align indices
        cluster_subset = cluster_assignments.loc[common_obs]
        
        common_samples = common_obs
    else:
        # Try regular intersection
        common_samples = pca_df.index.intersection(cluster_assignments.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples found between PCA data and cluster assignments")
        pca_subset = pca_df.loc[common_samples, [dim1, dim2]]
        cluster_subset = cluster_assignments.loc[common_samples]
    
    # Create combined grouping column
    if len(grouping_columns) == 1:
        grouping_values = adata.obs.loc[common_samples, grouping_columns[0]].astype(str)
    else:
        grouping_values = adata.obs.loc[common_samples, grouping_columns].astype(str).agg('_'.join, axis=1)
    
    # Try to extract numeric values for coloring
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

    # Compute cluster centroids using cluster assignments from obs
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

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter_obj = plt.scatter(
        pca_subset[dim1], pca_subset[dim2], c=color_values, cmap=cmap, 
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

def _preprocessing(
    adata_sample_diff,
    output_dir,
    grouping_columns,
    age_bin_size,
    verbose
):
    """
    Create output directory, add 'plot_group' to adata_sample_diff.obs based on grouping columns, etc.
    """
    # Create output sub-directory
    output_dir = os.path.join(output_dir, "harmony")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Find grouping
    diff_samples = adata_sample_diff.obs['sample'].unique().tolist()
    diff_groups = find_sample_grouping(
        adata_sample_diff, diff_samples, grouping_columns, age_bin_size
    )
    adata_sample_diff.obs['plot_group'] = adata_sample_diff.obs['sample'].map(diff_groups)

    if verbose:
        print("[_preprocessing] 'plot_group' assigned via find_sample_grouping.")

    return output_dir


def _plot_dendrogram(adata_sample_diff, output_dir, verbose):
    """
    Plot dendrogram (by cell_type).
    """
    if verbose:
        print("[_plot_dendrogram] Plotting dendrogram by cell_type.")
        
    sc.pl.dendrogram(adata_sample_diff, groupby='cell_type', show=False)
    plt.savefig(os.path.join(output_dir, 'phylo_tree.pdf'))
    plt.close()

def _plot_umap_by_cell_type(adata_sample_diff, output_dir, dot_size, verbose):
    """
    UMAP colored by 'cell_type' with cell type labels on cluster centroids.
    """
    if verbose:
        print("[plot_umap_by_cell_type] UMAP colored by 'cell_type'.")
    
    plt.figure(figsize=(12, 10))
    
    # Create the UMAP plot
    sc.pl.umap(
        adata_sample_diff,
        color='cell_type',
        legend_loc=None,
        frameon=False,
        size=dot_size,
        show=False
    )
    
    # Get UMAP coordinates
    umap_coords = adata_sample_diff.obsm['X_umap']
    
    # Get cell type labels
    cell_types = adata_sample_diff.obs['cell_type']
    
    # Calculate centroids for each cell type and add labels
    unique_cell_types = cell_types.unique()
    
    for cell_type in unique_cell_types:
        # Get indices for this cell type
        mask = cell_types == cell_type
        
        # Calculate centroid coordinates
        centroid_x = umap_coords[mask, 0].mean()
        centroid_y = umap_coords[mask, 1].mean()
        
        # Add text label at centroid
        plt.text(
            centroid_x, centroid_y, 
            str(cell_type),
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor='black',
                alpha=0.8
            )
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_umap_by_cell_type.pdf'), bbox_inches='tight')
    plt.close()

def visualization(
    adata_sample_diff,
    output_dir,
    grouping_columns=['sev.level'],
    age_bin_size=None,
    verbose=True,
    dot_size=3,

    plot_dendrogram_flag=True,
    plot_umap_by_cell_type_flag=True,
    plot_cell_type_proportions_pca_flag=False,
    plot_cell_type_expression_umap_flag=False,
):
    """
    Main function to handle all steps. Sub-functions are called conditionally based on flags.
    """
    # 1. Preprocessing
    if grouping_columns:
        output_dir = _preprocessing(
            adata_sample_diff,
            output_dir,
            grouping_columns,
            age_bin_size,
            verbose
        )

    # 2. Dendrogram
    if plot_dendrogram_flag:
        _plot_dendrogram(adata_sample_diff, output_dir, verbose)

    # 3. UMAP by cell type
    if plot_umap_by_cell_type_flag:
        _plot_umap_by_cell_type(adata_sample_diff, output_dir, dot_size, verbose)

    # 4. Cell type proportions PCA embedding
    if plot_cell_type_proportions_pca_flag:
        plot_sample_cell_proportions_embedding(
            adata_sample_diff,
            os.path.dirname(output_dir),  # pass parent (function itself appends 'harmony')
            grouping_columns=grouping_columns,
            verbose=verbose
        )

    # 5. Cell expression UMAP embedding
    if plot_cell_type_expression_umap_flag:
        plot_sample_cell_expression_embedding(
            adata_sample_diff, 
            os.path.dirname(output_dir),  # pass parent (function itself appends 'harmony')
            grouping_columns=grouping_columns,
            verbose=verbose
        )

    if verbose:
        print("[visualization] All requested visualizations saved.")