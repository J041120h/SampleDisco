import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, squareform
import seaborn as sns
from pyemd import emd
from anndata import AnnData
from scipy.cluster.hierarchy import linkage
import warnings
from anndata._core.aligned_df import ImplicitModificationWarning

def calculate_sample_distances_cell_expression(
    adata: AnnData,
    output_dir: str,
    cell_type_column: str = 'leiden',
    sample_column: str = 'sample'
) -> pd.DataFrame:
    """
    Calculate distances between samples based on the expression levels of each cell type using Earth Mover's Distance (EMD).

    This function computes the EMD between each pair of samples by considering the distributions of cell types and their average expression profiles within each sample.
    The ground distance between cell types is defined based on the Euclidean distances between their average expression profiles of highly variable genes.

    Parameters:
    ----------
    adata : AnnData
        The integrated single-cell dataset with highly variable genes calculated.
    output_dir : str
        Directory to save the output files.
    cell_type_column : str, optional
        Column name in `adata.obs` that contains the cell type assignments (default: 'leiden').
    sample_column : str, optional
        Column name in `adata.obs` that contains the sample information (default: 'sample').

    Returns:
    -------
    sample_distance_matrix : pandas.DataFrame
        A symmetric matrix of distances between samples.
    """

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Automatically generating output directory")

    # Check if highly variable genes are present
    if 'highly_variable' not in adata.var.columns:
        raise ValueError("Highly variable genes have not been calculated. Please run `sc.pp.highly_variable_genes` on the AnnData object.")

    # Filter the data to include only highly variable genes
    hvg = adata[:, adata.var['highly_variable']]

    # 1. Compute average expression profiles for each cell type in each sample
    samples = hvg.obs[sample_column].unique()
    cell_types = hvg.obs[cell_type_column].unique()

    # Initialize a dictionary to hold average expression profiles
    avg_expression = {}

    for sample in samples:
        avg_expression[sample] = {}
        sample_data = hvg[hvg.obs[sample_column] == sample]

        for cell_type in cell_types:
            cell_type_data = sample_data[sample_data.obs[cell_type_column] == cell_type]
            if cell_type_data.shape[0] > 0:
                avg_expr = cell_type_data.X.mean(axis=0).A1 if issparse(cell_type_data.X) else cell_type_data.X.mean(axis=0)
                avg_expression[sample][cell_type] = avg_expr
            else:
                # If a cell type is not present in a sample, set average expression to zeros
                avg_expression[sample][cell_type] = np.zeros(hvg.shape[1])

    # 2. Compute cell type proportions in each sample
    proportions = pd.DataFrame(0, index=samples, columns=cell_types, dtype=np.float64)

    for sample in samples:
        sample_data = hvg.obs[hvg.obs[sample_column] == sample]
        total_cells = sample_data.shape[0]
        counts = sample_data[cell_type_column].value_counts()
        proportions.loc[sample, counts.index] = counts.values / total_cells

    # 3. Compute ground distance matrix between cell types based on average expression profiles
    # Compute global average expression profiles for each cell type across all samples
    global_avg_expression = {}
    for cell_type in cell_types:
        cell_type_data = hvg[hvg.obs[cell_type_column] == cell_type]
        if cell_type_data.shape[0] > 0:
            avg_expr = cell_type_data.X.mean(axis=0).A1 if issparse(cell_type_data.X) else cell_type_data.X.mean(axis=0)
            global_avg_expression[cell_type] = avg_expr
        else:
            global_avg_expression[cell_type] = np.zeros(hvg.shape[1])

    # Now compute pairwise distances between cell types
    cell_type_list = list(cell_types)
    num_cell_types = len(cell_type_list)
    ground_distance = np.zeros((num_cell_types, num_cell_types), dtype=np.float64)

    for i in range(num_cell_types):
        for j in range(num_cell_types):
            expr_i = global_avg_expression[cell_type_list[i]]
            expr_j = global_avg_expression[cell_type_list[j]]
            distance = np.linalg.norm(expr_i - expr_j)
            ground_distance[i, j] = distance

    # 4. Compute EMD between each pair of samples
    num_samples = len(samples)
    sample_distance_matrix = pd.DataFrame(0, index=samples, columns=samples, dtype=np.float64)

    for i, sample_i in enumerate(samples):
        for j, sample_j in enumerate(samples):
            if i < j:
                # Histograms (masses) for the two samples
                hist_i = proportions.loc[sample_i].values
                hist_j = proportions.loc[sample_j].values

                # EMD requires histograms to sum to the same value
                # Since they are proportions, they sum to 1
                # Ground distance matrix between cell types (bins)
                # Already computed based on global average expressions

                # Compute EMD
                distance = emd(hist_i, hist_j, ground_distance)
                sample_distance_matrix.loc[sample_i, sample_j] = distance
                sample_distance_matrix.loc[sample_j, sample_i] = distance

    # Save the distance matrix
    distance_matrix_path = os.path.join(output_dir, 'sample_distance_matrix_expression.csv')
    sample_distance_matrix.to_csv(distance_matrix_path)
    print(f"Sample distance matrix saved to {distance_matrix_path}")

    # Optionally, generate a heatmap
    heatmap_path = os.path.join(output_dir, 'sample_distance_heatmap_expression.pdf')

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

    # Optionally, plot cell type expression profiles
    # This function can be implemented separately
    # plot_cell_type_expression_profiles(avg_expression, output_dir)

    return sample_distance_matrix