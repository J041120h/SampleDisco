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

def plot_cell_type_expression_heatmap(
    avg_expression: dict,
    output_dir: str,
    cell_type_order: list = None,
    sample_order: list = None,
    figsize: tuple = (10, 8),
    cmap: str = 'viridis_r',
    annot: bool = False
):
    """
    Generate a heatmap showing the average expression of each cell type across samples.
    
    Parameters:
    ----------
    avg_expression : dict
        Nested dictionary where avg_expression[sample][cell_type] = average_expression_array
    output_dir : str
        Directory to save the heatmap.
    cell_type_order : list, optional
        Order of cell types in the heatmap. If None, uses the order in the dictionary.
    sample_order : list, optional
        Order of samples in the heatmap. If None, uses the order in the dictionary.
    figsize : tuple, optional
        Size of the heatmap figure.
    cmap : str, optional
        Colormap for the heatmap.
    annot : bool, optional
        Whether to annotate the heatmap cells with their values.
    
    Returns:
    -------
    None
    """
    
    samples = list(avg_expression.keys())
    cell_types = list(next(iter(avg_expression.values())).keys()) if samples else []
    
    expression_matrix = pd.DataFrame(index=cell_types, columns=samples, dtype=np.float64)
    
    for sample in samples:
        for cell_type in cell_types:
            expression_value = avg_expression[sample].get(cell_type, np.zeros(avg_expression[sample][list(avg_expression[sample].keys())[0]].shape)[0].astype(np.float64)).mean()
            expression_matrix.loc[cell_type, sample] = expression_value
    # Replace NaN with 0 (in case some cell types are missing in certain samples)
    expression_matrix.fillna(0, inplace=True)
    
    if cell_type_order:
        expression_matrix = expression_matrix.reindex(cell_type_order)
    if sample_order:
        expression_matrix = expression_matrix[sample_order]
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        expression_matrix,
        cmap=cmap,
        linewidths=0.5,
        linecolor='grey',
        annot=annot,
        fmt=".2f"
    )
    plt.title('Average Expression of Cell Types Across Samples')
    plt.xlabel('Samples')
    plt.ylabel('Cell Types')
    
    heatmap_path = os.path.join(output_dir, 'cell_type_expression_heatmap.pdf')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Cell type expression heatmap saved to {heatmap_path}")
    
def visualizeGroupRelationship(
    sample_distance_matrix,
    outputDir,
    adata,
    grouping_columns=['sev.level'],
    age_bin_size=None,
    heatmap_path=None
):
    """
    Generates 2D MDS and PCA plots from a sample distance matrix, coloring points
    according to values from the specified grouping columns.
    
    - For continuous numeric values: Uses a continuous color scale
    - For categorical values: Uses discrete colors with a legend
    
    Arguments:
    ----------
    sample_distance_matrix : pd.DataFrame
        A square, symmetric distance matrix with samples as both rows and columns.
    outputDir : str
        Directory where the plot will be saved.
    adata : anndata.AnnData
        An AnnData object (or any structure) needed by find_sample_grouping
        to determine group assignments per sample.
    grouping_columns : list, optional
        Which columns in `adata.obs` to use for grouping (default: ['sev.level']).
    age_bin_size : int or None, optional
        If grouping by age, specify the bin size here (optional).
    heatmap_path : str or None, optional
        If provided, the final figure will be saved to this path. Otherwise,
        filenames are derived automatically.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import re
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    import pandas as pd
    from matplotlib.colors import ListedColormap
    
    os.makedirs(outputDir, exist_ok=True)
    samples = sample_distance_matrix.index.tolist()
    
    # Create symmetric matrix
    sym_matrix = (sample_distance_matrix + sample_distance_matrix.T) / 2
    np.fill_diagonal(sym_matrix.values, 0)
    
    # Generate MDS and PCA coordinates
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    points_mds = mds.fit_transform(sym_matrix)
    
    pca = PCA(n_components=2)
    points_pca = pca.fit_transform(sym_matrix)
    
    # Get group mappings for samples
    group_mapping = find_sample_grouping(
        adata,
        samples,
        grouping_columns=grouping_columns,
        age_bin_size=age_bin_size
    )
    
    group_labels = [group_mapping[sample] for sample in samples]
    
    # Determine if the grouping is numeric or categorical
    is_numeric = True
    numeric_values = []
    
    # Try to extract numeric values from labels
    for lbl in group_labels:
        m = re.search(r'(\d+\.?\d*)', lbl)
        if m:
            numeric_values.append(float(m.group(1)))
        else:
            is_numeric = False
            break
    
    # Setup visualization parameters based on data type
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    if is_numeric:
        # Handle numeric/continuous data with a color gradient
        numeric_values = np.array(numeric_values)
        value_min, value_max = numeric_values.min(), numeric_values.max()
        norm_values = (numeric_values - value_min) / (value_max - value_min) if value_max > value_min else np.zeros_like(numeric_values)
        
        cmap = plt.cm.coolwarm
        color_values = norm_values
        
        for ax, method, points in zip(axes, ['MDS', 'PCA'], [points_mds, points_pca]):
            sc = ax.scatter(
                points[:, 0], points[:, 1],
                s=100, c=color_values, cmap=cmap, alpha=0.8, edgecolors='k'
            )
            ax.set_xlabel(f"{method} Dimension 1")
            ax.set_ylabel(f"{method} Dimension 2")
            ax.set_title(f"2D {method} Visualization of Sample Distance Matrix")
            ax.grid(True)
        
        # Add a colorbar for continuous data
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist())
        cbar.set_label(f"Value ({value_min:.2f} - {value_max:.2f})")
    
    else:
        # Handle categorical data with discrete colors and a legend
        unique_labels = sorted(set(group_labels))
        color_map = plt.cm.get_cmap('tab20', len(unique_labels))
        label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}
        
        for ax, method, points in zip(axes, ['MDS', 'PCA'], [points_mds, points_pca]):
            for label in unique_labels:
                # Plot each category separately to build the legend
                indices = [i for i, l in enumerate(group_labels) if l == label]
                if indices:
                    ax.scatter(
                        points[indices, 0], points[indices, 1],
                        s=100, c=[label_to_color[label]], alpha=0.8, edgecolors='k', 
                        label=label
                    )
            
            ax.set_xlabel(f"{method} Dimension 1")
            ax.set_ylabel(f"{method} Dimension 2")
            ax.set_title(f"2D {method} Visualization of Sample Distance Matrix")
            ax.grid(True)
            
            # Add legend for categorical data
            if method == 'MDS':  # Only add legend to one plot to avoid redundancy
                ax.legend(title=grouping_columns[0], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plots
    if heatmap_path is None:
        mds_path = os.path.join(outputDir, f"sample_distance_matrix_MDS_{grouping_columns[0]}.png")
        pca_path = os.path.join(outputDir, f"sample_distance_matrix_PCA_{grouping_columns[0]}.png")
    else:
        mds_path = heatmap_path.replace(".png", f"_MDS_{grouping_columns[0]}.png")
        pca_path = heatmap_path.replace(".png", f"_PCA_{grouping_columns[0]}.png")
    
    plt.tight_layout()
    plt.savefig(mds_path, dpi=300, bbox_inches='tight')
    
    # Save PCA plot separately
    plt.figure(figsize=(7, 6))
    if is_numeric:
        plt.scatter(
            points_pca[:, 0], points_pca[:, 1],
            s=100, c=color_values, cmap=cmap, alpha=0.8, edgecolors='k'
        )
        plt.colorbar(label=f"Value ({value_min:.2f} - {value_max:.2f})")
    else:
        for label in unique_labels:
            indices = [i for i, l in enumerate(group_labels) if l == label]
            if indices:
                plt.scatter(
                    points_pca[indices, 0], points_pca[indices, 1],
                    s=100, c=[label_to_color[label]], alpha=0.8, edgecolors='k',
                    label=label
                )
        plt.legend(title=grouping_columns[0], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.title("2D PCA Visualization of Sample Distance Matrix")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    
    plt.close('all')
    print(f"MDS plot saved to {mds_path}")
    print(f"PCA plot saved to {pca_path}")


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
    adata,
    sample_cluster: Dict[str, List[str]],
    main_path: List[int],
    branching_paths: List[List[int]],
    output_path: str,
    pca_key: str = "X_pca_expression",
    verbose: bool = False
):
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
        subset = pca_df.loc[sample_cluster[clust], ["PC1", "PC2"]]
        centroid = subset.mean(axis=0).values
        cluster_centroids[clust] = centroid

    def _connect_clusters(c1, c2, style="-", color="k", linewidth=2):
        p1 = cluster_centroids[c1]
        p2 = cluster_centroids[c2]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle=style, color=color, linewidth=linewidth, zorder=1)

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap("tab20", len(cluster_names))

    for i, clust in enumerate(cluster_names):
        subset = pca_df.loc[sample_cluster[clust], ["PC1", "PC2"]]
        plt.scatter(subset["PC1"], subset["PC2"], color=cmap(i), label=clust, s=60, alpha=0.8, edgecolors="k")

    for i in range(len(main_path) - 1):
        _connect_clusters(f"cluster_{main_path[i] + 1}", f"cluster_{main_path[i + 1] + 1}", style="-")

    for path in branching_paths:
        for j in range(len(path) - 1):
            _connect_clusters(f"cluster_{path[j] + 1}", f"cluster_{path[j + 1] + 1}", style="--")

    for clust in cluster_names:
        cx, cy = cluster_centroids[clust]
        plt.text(cx, cy, clust.replace("cluster_", ""), fontsize=12, ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"))

    plt.title("PCA (2D) - Colored by Cluster")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    output_path = os.path.join(output_path, "clusters_by_cluster.png")
    plt.savefig(output_path)

    if verbose:
        print(f"[plot_clusters_by_cluster] Plot saved to {output_path}")

def plot_clusters_by_grouping(
    adata,
    sample_cluster: Dict[str, List[str]],
    main_path: List[int],
    branching_paths: List[List[int]],
    output_path: str,
    pca_key: str = "X_pca_expression",
    grouping_columns: List[str] = ["sev.level"],
    verbose: bool = False
):
    pca_df = adata.uns[pca_key]
    pca_df = pca_df[["PC1", "PC2"]].copy().reset_index().rename(columns={"index": "sample"})
    pca_df["sample"] = pca_df["sample"].astype(str).str.strip().str.lower()

    # Get grouping info
    diff_groups = find_sample_grouping(adata, list(pca_df["sample"]), grouping_columns)
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient="index", columns=["plot_group"])
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
    diff_groups = diff_groups.reset_index().rename(columns={"index": "sample"})

    # Merge and process severity
    pca_df = pca_df.merge(diff_groups, how="left", on="sample")
    pca_df["severity"] = pca_df["plot_group"].str.extract(r"(\d+\.\d+)").astype(float)

    # Normalize severity with improved handling
    if pca_df["severity"].notnull().all():
        norm_severity = (pca_df["severity"] - pca_df["severity"].min()) / (pca_df["severity"].max() - pca_df["severity"].min())
    else:
        pca_df["severity"].fillna(1.0, inplace=True)
        norm_severity = pca_df["severity"]

    # Compute cluster centroids
    cluster_names = sorted(sample_cluster.keys())
    cluster_centroids = {}
    full_pca_df = adata.uns[pca_key]
    for clust in cluster_names:
        subset = full_pca_df.loc[sample_cluster[clust], ["PC1", "PC2"]]
        centroid = subset.mean(axis=0).values
        cluster_centroids[clust] = centroid

    def _connect_clusters(c1, c2, style="-", color="k", linewidth=2):
        p1 = cluster_centroids[c1]
        p2 = cluster_centroids[c2]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle=style, color=color, linewidth=linewidth, zorder=1)

    # Plotting
    plt.figure(figsize=(8, 6))
    scatter_obj = plt.scatter(
        pca_df["PC1"], pca_df["PC2"], c=norm_severity, cmap="viridis_r", s=80, alpha=0.8, edgecolors="k", zorder=2
    )

    # Draw main and branching paths
    for i in range(len(main_path) - 1):
        _connect_clusters(f"cluster_{main_path[i] + 1}", f"cluster_{main_path[i + 1] + 1}", style="-")

    for path in branching_paths:
        for j in range(len(path) - 1):
            _connect_clusters(f"cluster_{path[j] + 1}", f"cluster_{path[j + 1] + 1}", style="--")

    # Label clusters
    for clust in cluster_names:
        cx, cy = cluster_centroids[clust]
        plt.text(cx, cy, clust.replace("cluster_", ""), fontsize = 7, ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"))

    plt.colorbar(scatter_obj, label="Severity Level (normalized)")
    plt.title("PCA (2D) - Colored by Sample Grouping")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_path, "clusters_by_grouping.png")
    plt.savefig(output_path)
    plt.close("all")

    if verbose:
        print(f"[plot_clusters_by_grouping] Plot saved to {output_path}")


def plot_cell_type_proportions_pca(
    adata: AnnData, 
    output_dir: str, 
    grouping_columns: list = ['sev.level'], 
    age_bin_size: int = None,
    verbose: bool = False
) -> None:
    """
    Reads precomputed PCA results for cell type proportions from adata.uns["X_pca_proportion"]
    and visualizes PC1 vs. PC2, coloring samples by severity.

    Parameters:
    - adata: AnnData object containing PCA results in `adata.uns["X_pca_proportion"]`
    - output_dir: Directory to save the PCA plot.
    - grouping_columns: Columns used for grouping. Default is ['sev.level'].
    - age_bin_size: Integer for age binning if required. Default is None.
    """
    output_dir = os.path.join(output_dir, 'harmony')
    os.makedirs(output_dir, exist_ok=True)

    if "X_pca_proportion" not in adata.uns:
        raise KeyError("Missing 'X_pca_proportion' in adata.uns. Ensure PCA was run on cell proportions.")

    pca_coords = adata.uns["X_pca_proportion"].values
    samples = adata.obs['sample'].unique()
    
    # Construct PCA DataFrame
    pca_df = pd.DataFrame(pca_coords[:, :2], index=samples, columns=['PC1', 'PC2'])
    
    # Retrieve grouping info
    diff_groups = find_sample_grouping(adata, samples, grouping_columns, age_bin_size)
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
    
    # Merge grouping info
    pca_df.index = pca_df.index.astype(str).str.strip().str.lower()
    diff_groups = diff_groups.reset_index().rename(columns={'index': 'sample'})
    pca_df = pca_df.reset_index().rename(columns={'index': 'sample'})
    pca_df = pca_df.merge(diff_groups, on='sample', how='left')

    # Extract and normalize severity levels for color mapping
    pca_df['severity'] = pca_df['plot_group'].str.extract(r'(\d+\.\d+)').astype(float)
    norm_severity = (pca_df['severity'] - pca_df['severity'].min()) / (pca_df['severity'].max() - pca_df['severity'].min())

    # Plot PCA
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=norm_severity, cmap='viridis_r', s=80, alpha=0.8, edgecolors='k')
    plt.colorbar(sc, label='Severity Level')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA of Cell Type Proportions (Severity Gradient)')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'sample_relationship_pca_2D_sample_proportion.pdf')
    plt.savefig(plot_path)
    print(f"PCA plot saved to: {plot_path}")

def plot_pseudobulk_expression_pca(adata: AnnData, output_dir: str) -> None:
    """
    Reads precomputed PCA results for batch-corrected pseudobulk expression data from
    adata.uns["X_pca_expression"] and visualizes PC1 vs. PC2, coloring samples by severity.

    Parameters:
    - adata: AnnData object containing PCA results in `adata.uns["X_pca_expression"]`
    - output_dir: Directory to save the PCA plot.
    """
    if "X_pca_expression" not in adata.uns:
        raise KeyError("Missing 'X_pca_expression' in adata.uns. Ensure PCA was run on cell expression.")
    
    output_dir = os.path.join(output_dir, 'harmony')
    os.makedirs(output_dir, exist_ok=True)

    pca_coords = adata.uns["X_pca_expression"]
    samples = adata.obs['sample'].unique()

    # Convert pca_coords to numpy array if it's a DataFrame
    if isinstance(pca_coords, pd.DataFrame):
        pca_coords = pca_coords.values

    # Sanity check: number of rows must match number of samples
    if pca_coords.shape[0] != len(samples):
        raise ValueError(f"Mismatch: PCA data has {pca_coords.shape[0]} rows but found {len(samples)} unique samples in adata.obs.")

    # Construct PCA DataFrame
    pca_df = pd.DataFrame(pca_coords[:, :2], index=samples, columns=['PC1', 'PC2'])

    # Retrieve grouping info
    grouping_columns = ['sev.level']
    diff_groups = find_sample_grouping(adata, samples, grouping_columns)
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
    
    # Merge grouping info
    pca_df.index = pca_df.index.astype(str).str.strip().str.lower()
    diff_groups = diff_groups.reset_index().rename(columns={'index': 'sample'})
    pca_df = pca_df.reset_index().rename(columns={'index': 'sample'})
    pca_df = pca_df.merge(diff_groups, on='sample', how='left')

    # Extract and normalize severity levels for color mapping
    pca_df['severity'] = pca_df['plot_group'].str.extract(r'(\d+\.\d+)').astype(float)
    norm_severity = (pca_df['severity'] - pca_df['severity'].min()) / (pca_df['severity'].max() - pca_df['severity'].min())

    # Plot PCA
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        pca_df['PC1'], pca_df['PC2'],
        c=norm_severity,
        cmap='viridis_r',
        s=80, alpha=0.8, edgecolors='k'
    )
    plt.colorbar(sc, label='Severity Level')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA of HVG Expression')
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'sample_relationship_pca_2D_sample.pdf')
    plt.savefig(plot_path)
    print(f"PCA plot saved to: {plot_path}")
def plot_pseudobulk_batch_test_expression(adata: AnnData, output_dir: str) -> None:
    """
    Reads precomputed PCA results for pseudobulk expression from `adata.uns["X_pca_expression"]`
    and visualizes PC1 vs. PC2, coloring samples by batch.
    """
    if "X_pca_expression" not in adata.uns:
        raise KeyError("Missing 'X_pca_expression' in adata.uns.")
    
    output_dir = os.path.join(output_dir, 'harmony')
    os.makedirs(output_dir, exist_ok=True)

    pca_coords = adata.uns["X_pca_expression"]
    samples = adata.obs['sample'].unique()

    # Convert to numpy array if necessary
    if isinstance(pca_coords, pd.DataFrame):
        pca_coords = pca_coords.values

    if pca_coords.shape[0] != len(samples):
        raise ValueError(f"Mismatch: PCA data has {pca_coords.shape[0]} rows but found {len(samples)} unique samples in adata.obs.")

    pca_df = pd.DataFrame(pca_coords[:, :2], index=samples, columns=['PC1', 'PC2'])

    grouping_columns = ['batch']
    diff_groups = find_sample_grouping(adata, samples, grouping_columns)
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()

    pca_df.index = pca_df.index.astype(str).str.strip().str.lower()
    diff_groups = diff_groups.reset_index().rename(columns={'index': 'sample'})
    pca_df = pca_df.reset_index().rename(columns={'index': 'sample'})
    pca_df = pca_df.merge(diff_groups, on='sample', how='left')

    plt.figure(figsize=(8, 6))
    batches = pca_df['plot_group'].unique()
    for i, batch in enumerate(batches):
        subset = pca_df[pca_df['plot_group'] == batch]
        plt.scatter(subset['PC1'], subset['PC2'], label=batch, s=80, alpha=0.8, edgecolors='k')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA of HVG Expression (Colored by Batch)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'sample_relationship_batch_expression.pdf')
    plt.savefig(plot_path)
    print(f"PCA plot saved to: {plot_path}")

def plot_pseudobulk_batch_test_proportion(adata: AnnData, output_dir: str) -> None:
    """
    Reads precomputed PCA results for pseudobulk proportions from `adata.uns["X_pca_proportion"]`
    and visualizes PC1 vs. PC2, coloring samples by batch.
    """
    if "X_pca_proportion" not in adata.uns:
        raise KeyError("Missing 'X_pca_proportion' in adata.uns.")
    
    output_dir = os.path.join(output_dir, 'harmony')
    os.makedirs(output_dir, exist_ok=True)

    pca_coords = adata.uns["X_pca_proportion"]
    samples = adata.obs['sample'].unique()

    # Convert to numpy array if necessary
    if isinstance(pca_coords, pd.DataFrame):
        pca_coords = pca_coords.values

    if pca_coords.shape[0] != len(samples):
        raise ValueError(f"Mismatch: PCA data has {pca_coords.shape[0]} rows but found {len(samples)} unique samples in adata.obs.")

    pca_df = pd.DataFrame(pca_coords[:, :2], index=samples, columns=['PC1', 'PC2'])

    grouping_columns = ['batch']
    diff_groups = find_sample_grouping(adata, samples, grouping_columns)
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()

    pca_df.index = pca_df.index.astype(str).str.strip().str.lower()
    diff_groups = diff_groups.reset_index().rename(columns={'index': 'sample'})
    pca_df = pca_df.reset_index().rename(columns={'index': 'sample'})
    pca_df = pca_df.merge(diff_groups, on='sample', how='left')

    plt.figure(figsize=(8, 6))
    batches = pca_df['plot_group'].unique()
    for i, batch in enumerate(batches):
        subset = pca_df[pca_df['plot_group'] == batch]
        plt.scatter(subset['PC1'], subset['PC2'], label=batch, s=80, alpha=0.8, edgecolors='k')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA of HVG Proportion (Colored by Batch)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'sample_relationship_pca_batch_proportion.pdf')
    plt.savefig(plot_path)
    print(f"PCA plot saved to: {plot_path}")


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


def _plot_umap_by_plot_group(adata_sample_diff, output_dir, dot_size, verbose):
    """
    UMAP colored by 'plot_group'.
    """
    if verbose:
        print("[_plot_umap_by_plot_group] UMAP colored by 'plot_group'.")
    
    plt.figure(figsize=(12, 10))
    sc.pl.umap(
        adata_sample_diff,
        color='plot_group',
        legend_loc=None,
        frameon=False,
        size=dot_size,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_umap_by_plot_group.pdf'), bbox_inches='tight')
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


def _pca_sample_level_2d(adata_sample_diff, output_dir, verbose):
    """
    Compute sample-level PCA (2D) from average HVG expression and save plot.
    """
    if verbose:
        print("[_pca_sample_level_2d] Computing sample-level PCA from average HVG expression.")

    # Check if the data matrix is sparse
    if issparse(adata_sample_diff.X):
        df = pd.DataFrame(
            adata_sample_diff.X.toarray(),
            index=adata_sample_diff.obs_names,
            columns=adata_sample_diff.var_names
        )
    else:
        df = pd.DataFrame(
            adata_sample_diff.X,
            index=adata_sample_diff.obs_names,
            columns=adata_sample_diff.var_names
        )

    # Compute mean expression per sample
    df['sample'] = adata_sample_diff.obs['sample']
    sample_means = df.groupby('sample').mean()

    # Map back to "plot_group"
    sample_to_group = adata_sample_diff.obs[['sample', 'plot_group']].drop_duplicates().set_index('sample')
    
    # PCA
    pca_2d = PCA(n_components=2)
    pca_coords_2d = pca_2d.fit_transform(sample_means)
    pca_2d_df = pd.DataFrame(pca_coords_2d, index=sample_means.index, columns=['PC1', 'PC2'])
    pca_2d_df = pca_2d_df.join(sample_to_group, how='left')

    # Extract severity level (assuming plot_group has a pattern like "x.x_...")
    pca_2d_df['sev_level'] = pca_2d_df['plot_group'].str.extract(r'(\d\.\d+)').astype(float)
    if pca_2d_df['sev_level'].isna().sum() > 0:
        raise ValueError("Some plot_group values could not be parsed for severity levels.")

    # Create colormap
    norm = mcolors.Normalize(vmin=1.00, vmax=4.00)
    cmap = cm.get_cmap('coolwarm')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in pca_2d_df.iterrows():
        color = cmap(norm(row['sev_level']))
        ax.scatter(row['PC1'], row['PC2'], color=color, s=80, alpha=0.8)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('2D PCA of Avg HVG Expression')
    ax.grid(True)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Severity Level')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_relationship_pca_2D_sample.pdf'))
    plt.close()


def _pca_sample_level_3d(adata_sample_diff, output_dir, verbose):
    """
    Compute sample-level PCA (3D) from average HVG expression and save interactive HTML.
    """
    if verbose:
        print("[_pca_sample_level_3d] Computing 3D sample-level PCA from average HVG expression.")

    # Check if the data matrix is sparse
    if issparse(adata_sample_diff.X):
        df = pd.DataFrame(
            adata_sample_diff.X.toarray(),
            index=adata_sample_diff.obs_names,
            columns=adata_sample_diff.var_names
        )
    else:
        df = pd.DataFrame(
            adata_sample_diff.X,
            index=adata_sample_diff.obs_names,
            columns=adata_sample_diff.var_names
        )

    # Compute mean expression per sample
    df['sample'] = adata_sample_diff.obs['sample']
    sample_means = df.groupby('sample').mean()

    # Map back to "plot_group"
    sample_to_group = adata_sample_diff.obs[['sample', 'plot_group']].drop_duplicates().set_index('sample')

    # PCA
    pca_3d = PCA(n_components=3)
    pca_coords_3d = pca_3d.fit_transform(sample_means)
    pca_3d_df = pd.DataFrame(pca_coords_3d, index=sample_means.index, columns=['PC1', 'PC2', 'PC3'])
    pca_3d_df = pca_3d_df.join(sample_to_group, how='left')

    # Create 3D scatter using plotly
    fig_3d = px.scatter_3d(
        pca_3d_df,
        x='PC1', y='PC2', z='PC3',
        color='plot_group',
        hover_data={'plot_group': False}
    )
    fig_3d.update_layout(showlegend=False)
    fig_3d.update_traces(marker=dict(size=5), hovertemplate='<extra></extra>')

    # Save interactive plot
    output_html_path = os.path.join(output_dir, 'sample_relationship_pca_3D.html')
    pio.write_html(fig_3d, file=output_html_path, auto_open=False)


def _plot_3d_harmony_cells(adata_sample_diff, output_dir, verbose):
    """
    3D visualization at the cell-level using Harmony PCA (X_pca_harmony).
    """
    if verbose:
        print("[_plot_3d_harmony_cells] Generating 3D cell-level Harmony PCA visualization.")
    
    # Extract the first 3 harmony components
    harmony_coords = adata_sample_diff.obsm['X_pca_harmony'][:, :3]
    pca_cell_df = pd.DataFrame(
        harmony_coords,
        columns=['PC1', 'PC2', 'PC3'],
        index=adata_sample_diff.obs.index
    )
    pca_cell_df['plot_group'] = adata_sample_diff.obs['plot_group']

    fig_cell_3d = px.scatter_3d(
        pca_cell_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='plot_group',
        hover_data={'plot_group': False}
    )
    fig_cell_3d.update_layout(showlegend=False)
    fig_cell_3d.update_traces(marker=dict(size=2), hovertemplate='<extra></extra>')

    cell_3d_path = os.path.join(output_dir, 'cell_pca_sample.html')
    pio.write_html(fig_cell_3d, file=cell_3d_path, auto_open=False)
    
    if verbose:
        print(f"[_plot_3d_harmony_cells] 3D cell-level PCA saved to {cell_3d_path}")

def visualization(
    adata_sample_diff,
    output_dir,
    grouping_columns=['sev.level'],
    age_bin_size=None,
    verbose=True,
    dot_size=3,

    plot_dendrogram_flag=True,
    plot_umap_by_plot_group_flag=True,
    plot_umap_by_cell_type_flag=True,
    plot_pca_2d_flag=True,
    plot_pca_3d_flag=True,
    plot_3d_cells_flag=True,

    plot_cell_type_proportions_pca_flag=False,
    plot_cell_type_expression_pca_flag=False,
    plot_pseudobulk_batch_test_expression_flag=False,
    plot_pseudobulk_batch_test_proportion_flag=False,
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

    # 3. UMAP (Sample Differences)
    if plot_umap_by_plot_group_flag:
        _plot_umap_by_plot_group(adata_sample_diff, output_dir, dot_size, verbose)

    if plot_umap_by_cell_type_flag:
        _plot_umap_by_cell_type(adata_sample_diff, output_dir, dot_size, verbose)

    # 4. 2D PCA of Average HVG Expression
    if plot_pca_2d_flag:
        _pca_sample_level_2d(adata_sample_diff, output_dir, verbose)

    # 5. 3D PCA (interactive) of Average HVG Expression
    if plot_pca_3d_flag:
        _pca_sample_level_3d(adata_sample_diff, output_dir, verbose)

    # 6. 3D Cell-level Harmony Visualization
    if plot_3d_cells_flag:
        _plot_3d_harmony_cells(adata_sample_diff, output_dir, verbose)

    if plot_cell_type_proportions_pca_flag:
        plot_cell_type_proportions_pca(
            adata_sample_diff,
            os.path.dirname(output_dir),  # pass parent (function itself appends 'harmony')
            grouping_columns=grouping_columns,
            age_bin_size=age_bin_size,
            verbose=verbose
        )

    if plot_cell_type_expression_pca_flag:
        plot_pseudobulk_expression_pca(adata_sample_diff, os.path.dirname(output_dir))

    if plot_pseudobulk_batch_test_expression_flag:
        plot_pseudobulk_batch_test_expression(adata_sample_diff, os.path.dirname(output_dir))
    
    if plot_pseudobulk_batch_test_expression_flag:
        plot_pseudobulk_batch_test_proportion(adata_sample_diff, os.path.dirname(output_dir))

    if verbose:
        print("[visualization_harmony] All requested visualizations saved.")