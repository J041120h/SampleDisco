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
    os.makedirs(outputDir, exist_ok=True)
    samples = sample_distance_matrix.index.tolist()
    sym_matrix = (sample_distance_matrix + sample_distance_matrix.T) / 2
    np.fill_diagonal(sym_matrix.values, 0)

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    points_mds = mds.fit_transform(sym_matrix)

    pca = PCA(n_components=2)
    points_pca = pca.fit_transform(sym_matrix)

    group_mapping = find_sample_grouping(
        adata,
        samples,
        grouping_columns=grouping_columns,
        age_bin_size=age_bin_size
    )
    
    group_labels = [group_mapping[sample] for sample in samples]
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

    sev_min, sev_max = sev_levels.min(), sev_levels.max()
    norm_sev = (sev_levels - sev_min) / (sev_max - sev_min)
    cmap = plt.cm.coolwarm
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
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), label="Severity Level")
    if heatmap_path is None:
        mds_path = os.path.join(outputDir, "sample_distance_matrix_MDS.png")
        pca_path = os.path.join(outputDir, "sample_distance_matrix_PCA.png")
    else:
        mds_path = heatmap_path.replace(".png", "_MDS.png")
    plt.tight_layout()
    plt.savefig(mds_path)
    plt.savefig(pca_path)
    plt.close()
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

    pca_coords = adata.uns["X_pca_proportion"]
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
    sc = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=norm_severity, cmap='viridis_r', s=80, alpha=0.8, edgecolors='k')
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

def plot_pseudobulk_batch_test_pca(adata: AnnData, output_dir: str) -> None:
    """
    Reads precomputed PCA results for pseudobulk expression from `adata.uns["X_pca_expression"]`
    and visualizes PC1 vs. PC2, coloring samples by batch.

    Parameters:
    - adata: AnnData object containing PCA results in `adata.uns["X_pca_expression"]`
    - output_dir: Directory to save the PCA plot.
    """
    if "X_pca_expression" not in adata.uns:
        raise KeyError("Missing 'X_pca_expression' in adata.uns.")
    
    output_dir = os.path.join(output_dir, 'harmony')
    os.makedirs(output_dir, exist_ok=True)

    pca_coords = adata.uns["X_pca_expression"]
    samples = adata.obs['sample'].unique()
    
    # Construct PCA DataFrame
    pca_df = pd.DataFrame(pca_coords[:, :2], index=samples, columns=['PC1', 'PC2'])
    
    # Retrieve batch grouping info
    grouping_columns = ['batch']
    diff_groups = find_sample_grouping(adata, samples, grouping_columns)
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
    
    # Merge grouping info
    pca_df.index = pca_df.index.astype(str).str.strip().str.lower()
    diff_groups = diff_groups.reset_index().rename(columns={'index': 'sample'})
    pca_df = pca_df.reset_index().rename(columns={'index': 'sample'})
    pca_df = pca_df.merge(diff_groups, on='sample', how='left')

    # Plot PCA colored by batch
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

    plot_path = os.path.join(output_dir, 'sample_relationship_pca_2D_batch.pdf')
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
    UMAP colored by 'cell_type'.
    """
    if verbose:
        print("[_plot_umap_by_cell_type] UMAP colored by 'cell_type'.")
    
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
    plot_pseudobulk_expression_pca_flag=False,
    plot_pseudobulk_batch_test_pca_flag=False
):
    """
    Main function to handle all steps. Sub-functions are called conditionally based on flags.
    """
    # 1. Preprocessing
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

    if plot_pseudobulk_expression_pca_flag:
        plot_pseudobulk_expression_pca(adata_sample_diff, os.path.dirname(output_dir))

    if plot_pseudobulk_batch_test_pca_flag:
        plot_pseudobulk_batch_test_pca(adata_sample_diff, os.path.dirname(output_dir))

    if verbose:
        print("[visualization_harmony] All requested visualizations saved.")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

from Grouping import find_sample_grouping

def visualize_TSCAN_paths(
    adata,
    sample_cluster: Dict[str, List[str]],
    main_path: List[int],
    branching_paths: List[List[int]],
    output_dir: str,
    pca_key: str = "X_pca_expression",
    grouping_columns: List[str] = ["sev.level"],
    verbose: bool = False
):
    """
    Generates two 2D PCA plots (and saves them locally) visualizing:
      1) Clusters returned by `cluster_samples_by_pca` (colored by cluster).
      2) The same lines but colored by a user-defined grouping (e.g. severity).

    Arguments
    ---------
    adata : AnnData
        Annotated data object with PCA results in `adata.uns[pca_key]` as a DataFrame.
        The DataFrame must have row index = sample_name, columns = ["PC1", "PC2", ...].
    sample_cluster : dict
        Mapping from cluster_name (e.g. "cluster_1") -> list of sample IDs in that cluster.
    main_path : list
        List of cluster indices (0-based) forming the principal path (e.g. [0,1,2]).
    branching_paths : list of lists
        Each inner list is a path that branches from `main_path`.
    output_dir : str
        Directory to which plots will be saved.
    pca_key : str
        Key in `adata.uns` where the PCA DataFrame is stored.
    grouping_columns : list
        Columns used by `find_sample_grouping` (or similar) to color the second plot.
    verbose : bool
        If True, prints logging info.

    Returns
    -------
    None
        The function saves two PDF plots to `output_dir`.
    """

    # ----------------------------------------------------------
    # 1. Validate and gather PCA data
    # ----------------------------------------------------------
    if pca_key not in adata.uns:
        raise KeyError(f"Missing PCA data in adata.uns['{pca_key}'].")
    pca_df = adata.uns[pca_key]
    if not isinstance(pca_df, pd.DataFrame):
        raise TypeError(
            f"Expected a DataFrame in adata.uns['{pca_key}'], but got {type(pca_df)}."
        )
    if pca_df.shape[1] < 2:
        raise ValueError("PCA DataFrame must have at least 2 components (e.g., PC1 and PC2).")

    # Ensure the columns we need exist
    required_pcs = {"PC1", "PC2"}
    if not required_pcs.issubset(pca_df.columns):
        raise ValueError(
            f"PCA DataFrame must contain at least {required_pcs}. Found columns: {pca_df.columns.tolist()}"
        )

    # Make sure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 2. Check sample alignment and compute cluster centroids
    # ----------------------------------------------------------
    cluster_names = sorted(sample_cluster.keys())
    all_samples_in_clusters = set()
    for clust_name, samples in sample_cluster.items():
        all_samples_in_clusters.update(samples)

    # Check for missing samples in the PCA index
    missing_in_pca = [s for s in all_samples_in_clusters if s not in pca_df.index]
    if missing_in_pca:
        raise ValueError(
            "Some sample(s) in `sample_cluster` are not present in the PCA DataFrame index:\n"
            f"{missing_in_pca}\n"
            "Ensure `pca_df.index` includes all these samples."
        )

    # Compute centroids (mean of PC1, PC2) for each cluster
    cluster_centroids = {}
    for clust in cluster_names:
        # Subset the PCA data to only the samples in this cluster
        subset = pca_df.loc[sample_cluster[clust], ["PC1", "PC2"]]
        centroid = subset.mean(axis=0).values  # shape (2,)
        cluster_centroids[clust] = centroid

    if verbose:
        print("[visualize_TSCAN_paths] Computed cluster centroids for each cluster.")

    # A helper function to connect two cluster centroids in the plot
    def _connect_clusters(c1, c2, style="-", color="k", linewidth=2):
        """
        Draws a line between the centroids of clusters c1 and c2 on the active plot.
        """
        p1 = cluster_centroids[c1]
        p2 = cluster_centroids[c2]
        plt.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            linestyle=style,
            color=color,
            linewidth=linewidth,
            zorder=1
        )

    # ----------------------------------------------------------
    # 3. Plot #1: color by cluster
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 6))
    n_clusters = len(cluster_names)
    cmap = plt.cm.get_cmap("tab20", n_clusters)  # color map with enough distinct colors

    # Plot each cluster
    for i, clust in enumerate(cluster_names):
        subset = pca_df.loc[sample_cluster[clust], ["PC1", "PC2"]]
        plt.scatter(
            subset["PC1"], subset["PC2"],
            color=cmap(i),
            label=clust,
            s=60,
            alpha=0.8,
            edgecolors="k"
        )

    # Connect main_path with solid lines
    for i in range(len(main_path) - 1):
        c1 = f"cluster_{main_path[i] + 1}"
        c2 = f"cluster_{main_path[i + 1] + 1}"
        _connect_clusters(c1, c2, style="-")

    # Connect branching paths with dashed lines
    for path in branching_paths:
        for j in range(len(path) - 1):
            c1 = f"cluster_{path[j] + 1}"
            c2 = f"cluster_{path[j + 1] + 1}"
            _connect_clusters(c1, c2, style="--")

    # Label each centroid
    for clust in cluster_names:
        cx, cy = cluster_centroids[clust]
        plt.text(cx, cy, clust.replace("cluster_", ""),  # e.g. "1", "2", etc.
                 fontsize=12, ha="center", va="center",
                 bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"))

    plt.title("PCA (2D) - Colored by Cluster")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()

    outpath_1 = os.path.join(output_dir, "pca_cluster_paths_by_cluster.pdf")
    plt.savefig(outpath_1)
    if verbose:
        print(f"[visualize_TSCAN_paths] Plot #1 saved to {outpath_1}")

    # ----------------------------------------------------------
    # 4. Plot #2: color by user-defined grouping (e.g. severity)
    # ----------------------------------------------------------
    # Build a fresh DataFrame with sample, PC1, PC2
    pca_plot_df = pca_df[["PC1", "PC2"]].copy()
    pca_plot_df = pca_plot_df.reset_index().rename(columns={"index": "sample"})
    pca_plot_df["sample"] = pca_plot_df["sample"].astype(str).str.strip().str.lower()

    # Retrieve grouping info (depends on your function 'find_sample_grouping')
    diff_groups = find_sample_grouping(adata, list(pca_plot_df["sample"]), grouping_columns)
    if isinstance(diff_groups, dict):
        diff_groups = pd.DataFrame.from_dict(diff_groups, orient="index", columns=["plot_group"])
    diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
    diff_groups = diff_groups.reset_index().rename(columns={"index": "sample"})

    # Merge into pca_plot_df
    pca_plot_df = pca_plot_df.merge(diff_groups, how="left", on="sample")

    # Extract numeric severity for color mapping if it exists
    pca_plot_df["severity"] = pca_plot_df["plot_group"].str.extract(r"(\d+\.\d+)").astype(float)
    min_sev, max_sev = pca_plot_df["severity"].min(), pca_plot_df["severity"].max()

    if pd.isnull(min_sev) or pd.isnull(max_sev) or min_sev == max_sev:
        # fallback if no numeric severity is found
        pca_plot_df["severity"] = 1.0
        norm_severity = pca_plot_df["severity"]
    else:
        norm_severity = (pca_plot_df["severity"] - min_sev) / (max_sev - min_sev)

    # Create second plot
    plt.figure(figsize=(8, 6))
    scatter_obj = plt.scatter(
        pca_plot_df["PC1"],
        pca_plot_df["PC2"],
        c=norm_severity,
        cmap="viridis_r",
        s=80,
        alpha=0.8,
        edgecolors="k",
        zorder=2
    )

    # Connect cluster centroids as before
    for i in range(len(main_path) - 1):
        c1 = f"cluster_{main_path[i] + 1}"
        c2 = f"cluster_{main_path[i + 1] + 1}"
        _connect_clusters(c1, c2, style="-")

    for path in branching_paths:
        for j in range(len(path) - 1):
            c1 = f"cluster_{path[j] + 1}"
            c2 = f"cluster_{path[j + 1] + 1}"
            _connect_clusters(c1, c2, style="--")

    # Label each centroid
    for clust in cluster_names:
        cx, cy = cluster_centroids[clust]
        plt.text(cx, cy, clust.replace("cluster_", ""), fontsize=12, ha="center", va="center",
                 bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"))

    plt.colorbar(scatter_obj, label="Severity Level (normalized)")
    plt.title("PCA (2D) - Colored by Sample Grouping")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()

    outpath_2 = os.path.join(output_dir, "pca_cluster_paths_by_severity.pdf")
    plt.savefig(outpath_2)
    plt.close("all")

    if verbose:
        print(f"[visualize_TSCAN_paths] Plot #2 saved to {outpath_2}")
        print("[visualize_TSCAN_paths] Done. Two plots saved.")
