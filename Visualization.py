import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from Grouping import find_sample_grouping

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
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Automatically generating output directory")

    # Sort the samples for consistent plotting
    proportions = proportions.sort_index()

    # Define the order of cell types (optional: you can sort or specify a custom order)
    cell_types = proportions.columns.tolist()

    # Define a color palette with enough colors for all cell types
    num_cell_types = len(cell_types)
    colors = sns.color_palette('tab20', n_colors=num_cell_types)

    # Create a figure and axis
    plt.figure(figsize=(12, 8))

    # Plot stacked bar chart
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

    # Customize the plot
    plt.ylabel('Proportion', fontsize=14)
    plt.title('Cell Type Proportions Across Samples', fontsize=16)
    plt.xticks(sample_indices, proportions.index, rotation=90, fontsize=10)
    plt.yticks(fontsize=12)
    plt.legend(title='Cell Types', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the figure
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
    cmap: str = 'viridis',
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
    
    # Extract unique cell types and samples
    samples = list(avg_expression.keys())
    cell_types = list(next(iter(avg_expression.values())).keys()) if samples else []
    
    # Initialize a DataFrame with cell types as rows and samples as columns
    expression_matrix = pd.DataFrame(index=cell_types, columns=samples, dtype=np.float64)
    
    for sample in samples:
        for cell_type in cell_types:
            # Sum the average expression array to get a single scalar value
            # If the cell type is not present, it should already be 0
            # expression_value = avg_expression[sample].get(cell_type, np.zeros(1))[0] if avg_expression[sample].get(cell_type, np.zeros(1)).size > 0 else 0
            # Alternatively, sum across genes if avg_expression[sample][cell_type] is a vector
            expression_value = avg_expression[sample].get(cell_type, np.zeros(avg_expression[sample][list(avg_expression[sample].keys())[0]].shape)[0].astype(np.float64)).mean()
            expression_matrix.loc[cell_type, sample] = expression_value
    # Replace NaN with 0 (in case some cell types are missing in certain samples)
    expression_matrix.fillna(0, inplace=True)
    
    if cell_type_order:
        expression_matrix = expression_matrix.reindex(cell_type_order)
    if sample_order:
        expression_matrix = expression_matrix[sample_order]
    
    # Create the heatmap
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
    
    # Save the heatmap
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
    Generates a 2D MDS plot from a sample distance matrix, coloring points
    according to groups identified by `find_sample_grouping`. 
    Does not print sample names or include a legend.
    
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
        Which columns in `adata.obs` to use for grouping (example: ['sev.level']).
    age_bin_size : int or None, optional
        If grouping by age, specify the bin size here (optional).
    heatmap_path : str or None, optional
        If provided, the final figure will be saved to this path. Otherwise,
        the filename is derived automatically.
    """
    # Ensure output directory exists
    os.makedirs(outputDir, exist_ok=True)

    # Retrieve the sample names from the distance matrix index
    samples = sample_distance_matrix.index.tolist()

    # Make the matrix symmetric and set diagonal to zero
    sym_matrix = (sample_distance_matrix + sample_distance_matrix.T) / 2
    np.fill_diagonal(sym_matrix.values, 0)

    # Perform MDS on the distance matrix to get 2D coordinates
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    points_2d = mds.fit_transform(sym_matrix)

    # Determine group assignments for each sample
    # (Adjust find_sample_grouping calls as needed for your project.)
    group_mapping = find_sample_grouping(
        adata,
        samples,
        grouping_columns=grouping_columns,
        age_bin_size=age_bin_size
    )
    group_labels = [group_mapping[sample] for sample in samples]
    unique_groups = sorted(set(group_labels))

    # Choose colors for each group (one color per group, no legend, no text labels)
    color_map = plt.cm.get_cmap("tab10", len(unique_groups))

    plt.figure(figsize=(8, 6))

    # Plot each group separately, with its own color
    for i, group in enumerate(unique_groups):
        idx = [j for j, lbl in enumerate(group_labels) if lbl == group]
        plt.scatter(points_2d[idx, 0], points_2d[idx, 1],
                    s=100, c=[color_map(i)], alpha=0.8)

    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.title("2D MDS Visualization of Sample Distance Matrix")
    plt.grid(True)

    # Determine output path if not provided
    if heatmap_path is None:
        heatmap_path = os.path.join(outputDir, "sample_distance_matrix_MDS.png")

    plt.savefig(heatmap_path)
    plt.close()
    print(f"Plot saved to {heatmap_path}")

def visualizeDistanceMatrix(sample_distance_matrix, heatmap_path):
    # Convert the square distance matrix to condensed form for linkage
    condensed_distances = squareform(sample_distance_matrix.values)
    # Compute the linkage matrix using the condensed distance matrix
    linkage_matrix = linkage(condensed_distances, method='average')
    # Generate the clustermap
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

import os
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.sparse import issparse
# For interactive 3D plot
import plotly.express as px
import plotly.io as pio

def visualization_harmony(
    adata_cluster,
    adata_sample_diff,
    output_dir,
    grouping_columns=['sev.level'],
    age_bin_size=None,
    verbose=True,
    dot_size = 3
):
    # -----------------------------
    # 1. Ensure output directory
    # -----------------------------
    output_dir = os.path.join(output_dir, 'harmony')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group assignments for both AnnData objects
    cluster_samples = adata_cluster.obs['sample'].unique().tolist()
    cluster_groups = find_sample_grouping(
        adata_cluster, cluster_samples, grouping_columns, age_bin_size
    )
    adata_cluster.obs['plot_group'] = adata_cluster.obs['sample'].map(cluster_groups)

    diff_samples = adata_sample_diff.obs['sample'].unique().tolist()
    diff_groups = find_sample_grouping(
        adata_sample_diff, diff_samples, grouping_columns, age_bin_size
    )
    adata_sample_diff.obs['plot_group'] = adata_sample_diff.obs['sample'].map(diff_groups)

    if verbose:
        print("[visualization_harmony] 'plot_group' assigned via find_sample_grouping.")

    # --------------------------------
    # 3. Dendrogram (by cell_type)
    # --------------------------------
    sc.pl.dendrogram(adata_cluster, groupby='cell_type', show=False)
    plt.savefig(os.path.join(output_dir, 'phylo_tree.pdf'))
    plt.close()

    # --------------------------------
    # 4. UMAP colored by plot_group (Clusters)
    # --------------------------------
    plt.figure(figsize=(12, 10))
    sc.pl.umap(
        adata_cluster,
        color='plot_group',
        legend_loc=None,
        frameon=False,
        size=dot_size,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_umap_by_plot_group.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 10))
    sc.pl.umap(
        adata_cluster,
        color='cell_type',
        legend_loc=None,
        frameon=False,
        size=dot_size,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_umap_cell_type.pdf'), bbox_inches='tight')
    plt.close()

    # --------------------------------
    # 5. UMAP colored by plot_group (Sample Differences)
    # --------------------------------
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

    plt.figure(figsize=(12, 10))
    sc.pl.umap(
        adata_sample_diff,
        color='cell_type',
        legend_loc=None,
        frameon=False,
        size=dot_size,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_umap_by_cell_type.pdf'), bbox_inches='tight')
    plt.close()

    # --------------------------------
    # 6. PCA of Average HVG Expression
    # --------------------------------
    if verbose:
        print("[visualization_harmony] Computing sample-level PCA from average HVG expression.")

    print("adata_sample_diff shape:", adata_sample_diff.shape)
    print("adata_sample_diff.X shape:", adata_sample_diff.X.shape)
    print("Is adata_sample_diff.X sparse?", issparse(adata_sample_diff.X))
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
    df['sample'] = adata_sample_diff.obs['sample']
    sample_means = df.groupby('sample').mean()
    sample_to_group = adata_sample_diff.obs[['sample', 'plot_group']].drop_duplicates().set_index('sample')

    pca_2d = PCA(n_components=2)
    pca_coords_2d = pca_2d.fit_transform(sample_means)
    pca_2d_df = pd.DataFrame(pca_coords_2d, index=sample_means.index, columns=['PC1', 'PC2'])
    pca_2d_df = pca_2d_df.join(sample_to_group, how='left')
    

    plt.figure(figsize=(8, 6))
    unique_groups = pca_2d_df['plot_group'].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_groups))
    for i, grp in enumerate(unique_groups):
        mask = (pca_2d_df['plot_group'] == grp)
        plt.scatter(pca_2d_df.loc[mask, 'PC1'], pca_2d_df.loc[mask, 'PC2'], color=colors(i), s=80, alpha=0.8)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA of Avg HVG Expression ')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_relationship_pca_2D_sample.pdf'))
    plt.close()

    if verbose:
        print("[visualization_harmony] Computing sample-level PCA from average HVG expression.")

    print("adata_cluster shape:", adata_cluster.shape)
    print("adata_cluster.X shape:", adata_cluster.X.shape)
    print("Is adata_cluster.X sparse?", issparse(adata_cluster.X))
    if issparse(adata_cluster.X):
        df = pd.DataFrame(
            adata_cluster.X.toarray(),
            index=adata_cluster.obs_names,
            columns=adata_cluster.var_names
        )
    else:
        df = pd.DataFrame(
            adata_cluster.X,
            index=adata_cluster.obs_names,
            columns=adata_cluster.var_names
        )
    df['sample'] = adata_cluster.obs['sample']
    sample_means = df.groupby('sample').mean()
    sample_to_group = adata_cluster.obs[['sample', 'plot_group']].drop_duplicates().set_index('sample')

    pca_2d = PCA(n_components=2)
    pca_coords_2d = pca_2d.fit_transform(sample_means)
    pca_2d_df = pd.DataFrame(pca_coords_2d, index=sample_means.index, columns=['PC1', 'PC2'])
    pca_2d_df = pca_2d_df.join(sample_to_group, how='left')

    plt.figure(figsize=(8, 6))
    unique_groups = pca_2d_df['plot_group'].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_groups))
    for i, grp in enumerate(unique_groups):
        mask = (pca_2d_df['plot_group'] == grp)
        plt.scatter(pca_2d_df.loc[mask, 'PC1'], pca_2d_df.loc[mask, 'PC2'], color=colors(i), s=80, alpha=0.8)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA of Avg HVG Expression ')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_relationship_pca_cluster.pdf'))
    plt.close()


    # 3D Interactive PCA
    pca_3d = PCA(n_components=3)
    pca_coords_3d = pca_3d.fit_transform(sample_means)
    pca_3d_df = pd.DataFrame(pca_coords_3d, index=sample_means.index, columns=['PC1', 'PC2', 'PC3'])
    pca_3d_df = pca_3d_df.join(sample_to_group, how='left')

    fig_3d = px.scatter_3d(
        pca_3d_df,
        x='PC1', y='PC2', z='PC3',
        color='plot_group',
        hover_data={'plot_group': False}
    )
    fig_3d.update_layout(showlegend=False)
    fig_3d.update_traces(marker=dict(size=5), hovertemplate='<extra></extra>')
    output_html_path = os.path.join(output_dir, 'sample_relationship_pca_3D.html')
    pio.write_html(fig_3d, file=output_html_path, auto_open=False)

    # --------------------------------
    # 7. 3D Visualization of Cell-level Harmony PCA
    # --------------------------------
    if verbose:
        print("[visualization_harmony] Generating 3D cell-level Harmony PCA visualization.")

    # If using obsm (standard):
    harmony_coords = adata_sample_diff.obsm['X_pca_harmony'][:, :3]
    pca_cell_df = pd.DataFrame(
        harmony_coords,
        columns=['PC1', 'PC2', 'PC3'],
        index=adata_sample_diff.obs.index
    )
    pca_cell_df['plot_group'] = adata_sample_diff.obs['plot_group']

    # Create interactive plot
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

    # Save plot
    cell_3d_path = os.path.join(output_dir, 'cell_pca_sample.html')
    pio.write_html(fig_cell_3d, file=cell_3d_path, auto_open=False)

    if verbose:
        print(f"[visualization_harmony] 3D cell-level PCA saved to {cell_3d_path}")

    # --------------------------------
    # 8. 3D Visualization of Cell-level Harmony PCA from cluster
    # --------------------------------
    if verbose:
        print("[visualization_harmony] Generating 3D cell-level Harmony PCA visualization.")

    # If using obsm (standard):
    harmony_coords = adata_cluster.obsm['X_pca_harmony'][:, :3]
    pca_cell_df = pd.DataFrame(
        harmony_coords,
        columns=['PC1', 'PC2', 'PC3'],
        index=adata_cluster.obs.index
    )
    pca_cell_df['plot_group'] = adata_cluster.obs['plot_group']

    # Create interactive plot
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

    # Save plot
    cell_3d_path = os.path.join(output_dir, 'cell_pca_cluster.html')
    pio.write_html(fig_cell_3d, file=cell_3d_path, auto_open=False)

    if verbose:
        print(f"[visualization_harmony] 3D cell-level PCA saved to {cell_3d_path}")

    # --------------------------------
    # Done
    # --------------------------------
    if verbose:
        print("[visualization_harmony] All visualizations saved.")