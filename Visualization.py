import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from Grouping import find_sample_grouping
from HVG import select_hvf_loess

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

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

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
    according to severity levels extracted from group assignments determined by
    find_sample_grouping. A continuous color scale is used based on the observed
    range of severity values.

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
    # Ensure output directory exists
    os.makedirs(outputDir, exist_ok=True)

    # Retrieve the sample names from the distance matrix index
    samples = sample_distance_matrix.index.tolist()

    # Make the matrix symmetric and set diagonal to zero
    sym_matrix = (sample_distance_matrix + sample_distance_matrix.T) / 2
    np.fill_diagonal(sym_matrix.values, 0)

    # Perform MDS on the distance matrix to get 2D coordinates
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    points_mds = mds.fit_transform(sym_matrix)

    # Perform PCA for comparison (PCA requires a square numerical matrix)
    pca = PCA(n_components=2)
    points_pca = pca.fit_transform(sym_matrix)

    # Determine group assignments for each sample via find_sample_grouping
    group_mapping = find_sample_grouping(
        adata,
        samples,
        grouping_columns=grouping_columns,
        age_bin_size=age_bin_size
    )
    
    # Assume group_mapping returns a mapping from sample to a string like "sev.level_X.XX"
    group_labels = [group_mapping[sample] for sample in samples]

    # Extract numeric severity levels from the group labels using regex
    sev_levels = []
    for lbl in group_labels:
        m = re.search(r'(\d+\.\d+)', lbl)
        if m:
            sev_levels.append(float(m.group(1)))
        else:
            sev_levels.append(np.nan)
    sev_levels = np.array(sev_levels)
    if np.isnan(sev_levels).any():
        raise ValueError("Some plot_group values could not be parsed for severity levels.")

    # Normalize severity values based on observed min and max
    sev_min, sev_max = sev_levels.min(), sev_levels.max()
    norm_sev = (sev_levels - sev_min) / (sev_max - sev_min)

    # Use a continuous colormap (blue for low, red for high severity)
    cmap = plt.cm.coolwarm

    # Create figure with two subplots (MDS and PCA)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, method, points in zip(axes, ['MDS', 'PCA'], [points_mds, points_pca]):
        sc = ax.scatter(
            points[:, 0], points[:, 1],
            s=100, c=norm_sev, cmap=cmap, alpha=0.8, edgecolors='k'
        )
        ax.set_xlabel(f"{method} Dimension 1")
        ax.set_ylabel(f"{method} Dimension 2")
        ax.set_title(f"2D {method} Visualization of Sample Distance Matrix")
        ax.grid(True)

    # Add a common colorbar for the entire figure showing the severity scale
    # cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), label="Severity Level")

    # Determine output filenames
    if heatmap_path is None:
        mds_path = os.path.join(outputDir, "sample_distance_matrix_MDS.png")
        pca_path = os.path.join(outputDir, "sample_distance_matrix_PCA.png")
    else:
        mds_path = heatmap_path.replace(".png", "_MDS.png")
        pca_path = heatmap_path.replace(".png", "_PCA.png")

    # Save the figure (both subplots are saved together)
    plt.tight_layout()
    plt.savefig(mds_path)
    plt.savefig(pca_path)
    plt.close()

    print(f"MDS plot saved to {mds_path}")
    print(f"PCA plot saved to {pca_path}")


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
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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

    # Map samples to severity levels
    sample_to_group = adata_sample_diff.obs[['sample', 'plot_group']].drop_duplicates().set_index('sample')

    # Perform PCA
    pca_2d = PCA(n_components=2)
    pca_coords_2d = pca_2d.fit_transform(sample_means)
    pca_2d_df = pd.DataFrame(pca_coords_2d, index=sample_means.index, columns=['PC1', 'PC2'])
    pca_2d_df = pca_2d_df.join(sample_to_group, how='left')

    # Extract severity level and convert to numeric
    pca_2d_df['sev_level'] = pca_2d_df['plot_group'].str.extract(r'(\d\.\d+)').astype(float)

    # Check for missing severity levels
    if pca_2d_df['sev_level'].isna().sum() > 0:
        raise ValueError("Some plot_group values could not be parsed for severity levels.")

    # Normalize severity levels for colormap mapping
    norm = mcolors.Normalize(vmin=1.00, vmax=4.00)
    cmap = cm.get_cmap('coolwarm')

    # Plot PCA with color mapping
    plt.figure(figsize=(8, 6))
    for _, row in pca_2d_df.iterrows():
        color = cmap(norm(row['sev_level']))
        plt.scatter(row['PC1'], row['PC2'], color=color, s=80, alpha=0.8, label=f"sev.level_{row['sev_level']:.2f}")

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA of Avg HVG Expression')
    plt.grid(True)
    plt.tight_layout()

    # Create a colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Severity Level')

    # Save and close figure
    plt.savefig(os.path.join(output_dir, 'sample_relationship_pca_2D_sample.pdf'))
    plt.close()

    if verbose:
        print("[visualization_harmony] Computing sample-level PCA from average HVG expression.")

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
    plt.legend()
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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_cell_type_proportions_pca(adata, output_dir, grouping_columns=['sev.level'], age_bin_size=None):
    output_dir = os.path.join(output_dir, 'harmony')
    os.makedirs(output_dir, exist_ok=True)

    samples = adata.obs['sample'].unique()
    cell_types = list(adata.obs['cell_type'].unique())  # Ensure it's a list

    # Compute cell type proportions per sample
    proportions = pd.DataFrame(0, index=samples, columns=cell_types, dtype=np.float64)
    for sample in samples:
        sample_data = adata.obs[adata.obs['sample'] == sample]
        total_cells = sample_data.shape[0]
        counts = sample_data['cell_type'].value_counts()
        proportions.loc[sample, counts.index] = counts.values / total_cells

    diff_groups = find_sample_grouping(adata, samples, grouping_columns, age_bin_size)

    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])

    if not isinstance(diff_groups, pd.DataFrame):
        raise TypeError(f"Expected diff_groups to be a DataFrame, but got {type(diff_groups)}")

    if 'plot_group' not in diff_groups.columns:
        raise KeyError("Column 'plot_group' is missing in diff_groups.")

    # Standardize index formats
    proportions.index = proportions.index.astype(str).str.strip().str.lower()
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()

    print("Proportions Index Samples:", proportions.index.tolist()[:5])
    print("Diff Groups Index Samples:", diff_groups.index.tolist()[:5])

    # Merge proportions with grouping information
    diff_groups = diff_groups.reset_index().rename(columns={'index': 'sample'})
    proportions = proportions.reset_index().rename(columns={'index': 'sample'})
    proportions = proportions.merge(diff_groups, on='sample', how='left').set_index('sample')

    proportions[cell_types] = proportions[cell_types].apply(pd.to_numeric, errors='coerce')
    proportions.fillna(0, inplace=True)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(proportions[cell_types])
    
    pca_df = pd.DataFrame(pca_coords, index=proportions.index, columns=['PC1', 'PC2'])
    pca_df = pca_df.join(proportions[['plot_group']])

    # Extract severity levels from 'plot_group' (assuming format "sev.level_X.XX")
    pca_df['severity'] = pca_df['plot_group'].str.extract(r'(\d+\.\d+)').astype(float)

    # Normalize severity values between 0 and 1 for color mapping
    norm_severity = (pca_df['severity'] - pca_df['severity'].min()) / (
            pca_df['severity'].max() - pca_df['severity'].min()
    )

    # Define colormap from blue (low severity) to red (high severity)
    colormap = plt.cm.coolwarm

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        pca_df['PC1'], pca_df['PC2'],
        c=norm_severity, cmap=colormap, s=80, alpha=0.8, edgecolors='k'
    )

    # Add colorbar for severity levels
    cbar = plt.colorbar(sc)
    cbar.set_label('Severity Level')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA of Cell Type Proportions (Severity Gradient)')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, 'sample_relationship_pca_2D_sample_proportion.pdf')
    plt.savefig(plot_path)
    plt.show()

    print(f"PCA plot saved to: {plot_path}")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from sklearn.decomposition import PCA

# Note: Ensure that find_sample_grouping is imported or defined in your environment

def plot_avg_hvg_expression_pca(adata_sample_diff, output_dir, grouping_columns=['sev.level'], age_bin_size=None, verbose=False):
    """
    Computes and plots a 2D PCA of the average HVG expression per sample.
    The PCA points are colored based on severity levels extracted from the 'plot_group'
    field, which is derived by calling find_sample_grouping on the input data.

    Parameters:
    -----------
    adata_sample_diff : AnnData-like object
        Object with attributes .X (expression data) and .obs (observation metadata).
    output_dir : str
        Directory where the output plot will be saved.
    grouping_columns : list, optional
        List of columns to use for grouping samples (default is ['sev.level']).
    age_bin_size : int or None, optional
        Age bin size for grouping, if applicable.
    verbose : bool, optional
        If True, prints additional debugging information.
    """
    if verbose:
        print("[visualization_harmony] Computing sample-level PCA from average HVG expression.")
    
    # Debug prints for data shapes
    print("adata_sample_diff shape:", adata_sample_diff.shape)
    print("adata_sample_diff.X shape:", adata_sample_diff.X.shape)
    print("Is adata_sample_diff.X sparse?", issparse(adata_sample_diff.X))
    
    # Convert expression data to a DataFrame (handling sparse and dense cases)
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
    
    # Add sample information and compute average HVG expression per sample
    df['sample'] = adata_sample_diff.obs['sample']
    sample_means = df.groupby('sample').mean()
    
    # Derive the plot grouping using the find_sample_grouping function
    samples = adata_sample_diff.obs['sample'].unique()
    diff_groups = find_sample_grouping(adata_sample_diff, samples, grouping_columns, age_bin_size)
    
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])
    
    if not isinstance(diff_groups, pd.DataFrame):
        raise TypeError(f"Expected diff_groups to be a DataFrame, but got {type(diff_groups)}")
    if 'plot_group' not in diff_groups.columns:
        raise KeyError("Column 'plot_group' is missing in diff_groups.")
    
    # Standardize index formats for merging
    sample_means.index = sample_means.index.astype(str).str.strip().str.lower()
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
    
    # Reset index and merge grouping information with sample_means
    diff_groups = diff_groups.reset_index().rename(columns={'index': 'sample'})
    sample_means = sample_means.reset_index().rename(columns={'index': 'sample'})
    sample_means = sample_means.merge(diff_groups, on='sample', how='left').set_index('sample')
    
    # Perform PCA on the sample-averaged data (excluding the plot_group column)
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(sample_means.drop(columns=['plot_group']))
    pca_df = pd.DataFrame(pca_coords, index=sample_means.index, columns=['PC1', 'PC2'])
    pca_df = pca_df.join(sample_means[['plot_group']])
    
    # Extract severity level as a numeric value from the plot_group string (e.g., "sev.level_X.XX")
    pca_df['sev_level'] = pca_df['plot_group'].str.extract(r'(\d+\.\d+)').astype(float)
    
    if pca_df['sev_level'].isna().sum() > 0:
        raise ValueError("Some plot_group values could not be parsed for severity levels.")
    
    # Normalize severity values based on the observed range
    sev_min = pca_df['sev_level'].min()
    sev_max = pca_df['sev_level'].max()
    norm_severity = (pca_df['sev_level'] - sev_min) / (sev_max - sev_min)
    
    # Define the colormap (blue for low severity, red for high)
    colormap = plt.cm.coolwarm
    
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        pca_df['PC1'], pca_df['PC2'],
        c=norm_severity, cmap=colormap, s=80, alpha=0.8, edgecolors='k'
    )
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA of Avg HVG Expression')
    plt.grid(True)
    plt.tight_layout()
    
    # Add a colorbar to show severity levels
    cbar = plt.colorbar(sc)
    cbar.set_label('Severity Level')
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'sample_relationship_pca_2D_sample.pdf')
    plt.savefig(plot_path)
    plt.close()
    
    if verbose:
        print(f"[visualization_harmony] PCA plot saved to: {plot_path}")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
from umap import UMAP

def plot_pseudobulk_umap(
    adata: AnnData,
    output_dir: str,
    pseudobulk: dict,
    grouping_columns: list = ['sev.level'],
    age_bin_size: int = None,
    verbose: bool = False
) -> None:
    """
    Constructs a sample-by-feature dataframe from pseudobulk-corrected expression data,
    selects the top 2000 highly variable genes (HVGs), and performs UMAP to embed samples in 2D.
    Samples are colored by severity level.

    Parameters:
    - adata: AnnData object containing sample metadata in adata.obs.
    - output_dir: Directory to save the UMAP plot.
    - pseudobulk: Dictionary containing 'cell_expression_corrected' with expression data.
    - grouping_columns: Columns to use for grouping. Default is ['sev.level'].
    - age_bin_size: Integer for age binning if required. Default is None.
    - verbose: If True, prints additional information.
    """
    # Extract corrected cell expression data
    cell_expr = pseudobulk['cell_expression']
    
    # Construct sample-by-feature matrix by concatenating expression across cell types
    sample_df = pd.DataFrame({
        sample: np.concatenate([cell_expr.loc[ct, sample] for ct in cell_expr.index])
        for sample in cell_expr.columns
    }).T
    sample_df.index.name = 'sample'
    sample_df.columns = [f"feature_{i}" for i in range(sample_df.shape[1])]
    
    # Select top 2000 most variable features
    top_features = sample_df.var(axis=0).nlargest(2000).index
    sample_df = sample_df[top_features]
    
    # Get sample grouping information
    diff_groups = find_sample_grouping(adata, adata.obs['sample'].unique(), grouping_columns, age_bin_size)
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])
    if 'plot_group' not in diff_groups.columns:
        raise KeyError("Column 'plot_group' is missing in diff_groups.")
    
    # Ensure consistent formatting and merge grouping information
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
    sample_df.index = sample_df.index.astype(str).str.strip().str.lower()
    sample_df = sample_df.merge(diff_groups.reset_index().rename(columns={'index': 'sample'}), on='sample', how='left')
    
    # Identify numeric columns (excluding the 'plot_group' column)
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    
    # Perform UMAP on top HVGs
    umap_model = UMAP(n_components=2, random_state=42)
    umap_coords = umap_model.fit_transform(sample_df[numeric_cols])
    umap_df = pd.DataFrame(umap_coords, index=sample_df.index, columns=['UMAP1', 'UMAP2'])
    umap_df['plot_group'] = sample_df['plot_group']
    
    # Extract numeric severity levels for coloring
    umap_df['sev_level'] = umap_df['plot_group'].str.extract(r'(\d+\.\d+)').astype(float)
    if umap_df['sev_level'].isna().sum() > 0:
        raise ValueError("Some plot_group values could not be parsed for severity levels.")
    
    # Normalize severity levels for color mapping
    norm_severity = (
        (umap_df['sev_level'] - umap_df['sev_level'].min()) 
        / (umap_df['sev_level'].max() - umap_df['sev_level'].min())
    )
    
    # Plot UMAP results
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        umap_df['UMAP1'], 
        umap_df['UMAP2'], 
        c=norm_severity, 
        cmap='coolwarm', 
        s=80, 
        alpha=0.8, 
        edgecolors='k'
    )
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('2D UMAP of HVG Expression')
    plt.colorbar(sc, label='Severity Level')
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sample_relationship_umap_2D_sample.pdf'))
    plt.close()
    
    if verbose:
        print(f"UMAP plot saved to {output_dir}/sample_relationship_umap_2D_sample.pdf")


def plot_pseudobulk_batch_umap(
    adata: AnnData,
    output_dir: str,
    pseudobulk: dict,
    grouping_columns: list = ['batch'],
    age_bin_size: int = None,
    verbose: bool = False
) -> None:
    """
    Constructs a sample-by-feature dataframe from pseudobulk-corrected expression data,
    selects the top 2000 highly variable genes (HVGs), and performs UMAP to embed samples in 2D.
    Samples are colored by batch.

    Parameters:
    - adata: AnnData object containing sample metadata in adata.obs.
    - output_dir: Directory to save the UMAP plot.
    - pseudobulk: Dictionary containing 'cell_expression_corrected' with expression data.
    - grouping_columns: List of columns in adata.obs to use for grouping. Default is ['batch'].
    - age_bin_size: Integer for age binning if required. Default is None.
    - verbose: If True, prints additional information.
    """
    # Ensure required data exists
    if 'cell_expression_corrected' not in pseudobulk:
        raise KeyError("Missing 'cell_expression_corrected' key in pseudobulk dictionary.")

    # Extract corrected cell expression data
    cell_expr = pseudobulk['cell_expression_corrected']

    # Construct sample-by-feature matrix
    sample_df = pd.DataFrame({
        sample: np.concatenate([cell_expr.loc[ct, sample] for ct in cell_expr.index])
        for sample in cell_expr.columns
    }).T
    sample_df.index.name = 'sample'
    sample_df.columns = [f"feature_{i}" for i in range(sample_df.shape[1])]

    # Select top 2000 most variable features (if applicable)
    if sample_df.shape[1] > 2000:
        top_features = sample_df.var(axis=0).nlargest(2000).index
        sample_df = sample_df[top_features]

    # Retrieve batch (or grouping) information
    diff_groups = find_sample_grouping(adata, adata.obs['sample'].unique(), grouping_columns, age_bin_size)

    # Convert to DataFrame if necessary
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])

    if 'plot_group' not in diff_groups.columns:
        raise KeyError("Column 'plot_group' is missing in diff_groups.")

    # Format index for merging
    sample_df.index = sample_df.index.astype(str).str.strip().str.lower()
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
    diff_groups = diff_groups.reset_index().rename(columns={'index': 'sample'})

    # Merge grouping information
    sample_df = sample_df.merge(diff_groups, on='sample', how='left')

    # Identify numeric columns (excluding 'plot_group')
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns

    # Perform UMAP
    umap_model = UMAP(n_components=2, random_state=42)
    umap_coords = umap_model.fit_transform(sample_df[numeric_cols])
    umap_df = pd.DataFrame(umap_coords, index=sample_df.index, columns=['UMAP1', 'UMAP2'])
    umap_df['batch'] = sample_df['plot_group']

    # Drop NaN values from batch column
    umap_df = umap_df.dropna(subset=['batch'])

    # Plot UMAP results with batch coloring
    plt.figure(figsize=(8, 6))

    unique_batches = umap_df['batch'].unique()
    cmap = plt.cm.get_cmap('tab10', min(len(unique_batches), 10))  # Up to 10 unique colors

    has_legend = False

    for i, batch in enumerate(unique_batches):
        subset = umap_df[umap_df['batch'] == batch]
        if not subset.empty:
            plt.scatter(
                subset['UMAP1'],
                subset['UMAP2'],
                label=str(batch),
                color=cmap(i % 10),
                s=80,
                alpha=0.8,
                edgecolors='k'
            )
            has_legend = True

    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('2D UMAP of HVG Expression (Colored by Batch)')

    if has_legend:
        plt.legend(title='Batch', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'sample_relationship_umap_2D_batch.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"UMAP plot saved to {save_path}")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from anndata import AnnData

# Be sure to have your select_hvf_loess function in scope
# from statsmodels.nonparametric.smoothers_lowess import lowess
# def select_hvf_loess(pseudobulk, n_features=2000, frac=0.3):
#     ...
#     return sample_df, top_features

def plot_pseudobulk_pca(
    adata: AnnData,
    output_dir: str,
    pseudobulk: dict,
    grouping_columns: list = ['sev.level'],
    age_bin_size: int = None,
    verbose: bool = False
) -> None:
    """
    Constructs a sample-by-feature dataframe from pseudobulk-corrected expression data,
    selects the top 2000 highly variable genes (HVGs), and performs PCA to embed samples in 2D.
    Samples are colored by severity level.

    Parameters:
    - adata: AnnData object containing sample metadata in adata.obs.
    - output_dir: Directory to save the PCA plot.
    - pseudobulk: Dictionary containing 'cell_expression_corrected' with expression data.
    - grouping_columns: Columns to use for grouping. Default is ['sev.level'].
    - age_bin_size: Integer for age binning if required. Default is None.
    - verbose: If True, prints additional information.
    """
    # Use the new HVF selection function to get the filtered sample-by-feature DataFrame
    sample_df, top_features = select_hvf_loess(pseudobulk, n_features=2000, frac=0.3)

    # Retrieve grouping info
    diff_groups = find_sample_grouping(adata, adata.obs['sample'].unique(), grouping_columns, age_bin_size)
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])
    if 'plot_group' not in diff_groups.columns:
        raise KeyError("Column 'plot_group' is missing in diff_groups.")

    # Format indices for merging
    sample_df.index = sample_df.index.astype(str).str.strip().str.lower()
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()

    # Convert the sample index into a column so we can merge on='sample'
    sample_df = sample_df.reset_index().rename(columns={'index': 'sample'})
    diff_groups = diff_groups.reset_index().rename(columns={'index': 'sample'})

    # Merge grouping info
    sample_df = sample_df.merge(diff_groups, on='sample', how='left')
    if sample_df['plot_group'].isna().any():
        raise ValueError("Some samples could not be matched to grouping information.")

    # Perform PCA on numeric columns only
    pca = PCA(n_components=2)
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    pca_coords = pca.fit_transform(sample_df[numeric_cols])
    pca_df = pd.DataFrame(pca_coords, index=sample_df.index, columns=['PC1', 'PC2'])

    # Store grouping info (severity levels) for plotting
    pca_df['plot_group'] = sample_df['plot_group']

    # Attempt to extract numeric severity levels from group labels
    pca_df['sev_level'] = pca_df['plot_group'].str.extract(r'(\d+\.\d+)').astype(float)
    if pca_df['sev_level'].isna().sum() > 0:
        raise ValueError("Some plot_group values could not be parsed for severity levels.")

    # Normalize severity levels for color mapping
    norm_severity = (
        (pca_df['sev_level'] - pca_df['sev_level'].min()) /
        (pca_df['sev_level'].max() - pca_df['sev_level'].min())
    )

    # Plot PCA results
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=norm_severity, cmap='coolwarm',
                     s=80, alpha=0.8, edgecolors='k')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA of HVG Expression')
    plt.colorbar(sc, label='Severity Level')

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sample_relationship_pca_2D_sample.pdf'))
    plt.close()

    if verbose:
        print(f"PCA plot saved to {output_dir}/sample_relationship_pca_2D_sample.pdf")


def plot_pseudobulk_batch_test_pca(
    adata: AnnData,
    output_dir: str,
    pseudobulk: dict,
    grouping_columns: list = ['batch'],
    age_bin_size: int = None,
    verbose: bool = False
) -> None:
    """
    Constructs a sample-by-feature dataframe from pseudobulk-corrected expression data,
    selects the top 2000 highly variable genes (HVGs), and performs PCA to embed samples in 2D.
    Samples are colored by batch.

    Parameters:
    - adata: AnnData object containing sample metadata in adata.obs.
    - output_dir: Directory to save the PCA plot.
    - pseudobulk: Dictionary containing 'cell_expression_corrected' with expression data.
    - grouping_columns: List of columns in adata.obs to use for grouping. Default is ['batch'].
    - age_bin_size: Integer for age binning if required. Default is None.
    - verbose: If True, prints additional information.
    """
    # Ensure the corrected data is in pseudobulk
    if 'cell_expression_corrected' not in pseudobulk:
        raise KeyError("Missing 'cell_expression_corrected' key in pseudobulk dictionary.")

    # Use the new HVF selection function to get the filtered sample-by-feature DataFrame
    sample_df, top_features = select_hvf_loess(pseudobulk, n_features=2000, frac=0.3)

    # Retrieve grouping info
    diff_groups = find_sample_grouping(adata, adata.obs['sample'].unique(), grouping_columns, age_bin_size)
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])
    if 'plot_group' not in diff_groups.columns:
        raise KeyError("Column 'plot_group' is missing in diff_groups.")

    # Format indices for merging
    sample_df.index = sample_df.index.astype(str).str.strip().str.lower()
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()

    # Convert the sample index into a column so we can merge on='sample'
    sample_df = sample_df.reset_index().rename(columns={'index': 'sample'})
    diff_groups = diff_groups.reset_index().rename(columns={'index': 'sample'})

    # Merge grouping info
    sample_df = sample_df.merge(diff_groups, on='sample', how='left')
    if sample_df['plot_group'].isna().any():
        raise ValueError("Some samples could not be matched to grouping information.")

    # Perform PCA on numeric columns only
    pca = PCA(n_components=2)
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    pca_coords = pca.fit_transform(sample_df[numeric_cols])
    pca_df = pd.DataFrame(pca_coords, index=sample_df.index, columns=['PC1', 'PC2'])

    # Batch labels
    pca_df['batch'] = sample_df['plot_group']

    # Drop any samples with missing batch info
    pca_df.dropna(subset=['batch'], inplace=True)

    # Plot PCA, coloring by batch
    plt.figure(figsize=(8, 6))
    unique_batches = pca_df['batch'].unique()

    # Basic colormap limiting to 10 distinct colors
    cmap = plt.cm.get_cmap('tab10', min(len(unique_batches), 10))

    has_legend = False
    for i, batch in enumerate(unique_batches):
        subset = pca_df[pca_df['batch'] == batch]
        if not subset.empty:
            plt.scatter(subset['PC1'], subset['PC2'],
                        label=str(batch),
                        color=cmap(i % 10),
                        s=80, alpha=0.8, edgecolors='k')
            has_legend = True

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA of HVG Expression (Colored by Batch)')
    if has_legend:
        plt.legend(title='Batch', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'sample_relationship_pca_2D_batch.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"PCA plot saved to {save_path}")
