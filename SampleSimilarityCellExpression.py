import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage
from pyemd import emd
import seaborn as sns
from anndata import AnnData
from scipy.sparse import issparse
from Visualization import plot_cell_type_expression_heatmap

def calculate_sample_distances_cell_expression(
    adata: AnnData,
    output_dir: str,
    cell_type_column: str = 'leiden',
    sample_column: str = 'sample',
    cell_group_weight = 0.8
) -> pd.DataFrame:
    """
    Calculate distances between samples based on the expression levels of each cell type using Earth Mover's Distance (EMD).

    This function computes the EMD between each pair of samples by considering the distribution of cell type expression levels within each sample.
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
        print("Automatically generated output directory.")

    # Check if highly variable genes are present
    if 'highly_variable' not in adata.var.columns:
        raise ValueError("Highly variable genes have not been calculated. Please run `sc.pp.highly_variable_genes` on the AnnData object.")

    # Filter the data to include only highly variable genes
    hvg = adata[:, adata.var['highly_variable']]

    # 1. Compute average expression profiles for each cell type in each sample
    samples = hvg.obs[sample_column].unique()
    cell_types = hvg.obs[cell_type_column].unique()

    # Initialize a dictionary to hold average expression profiles
    avg_expression = {sample: {} for sample in samples}

    for sample in samples:
        sample_data = hvg[hvg.obs[sample_column] == sample]
        for cell_type in cell_types:
            cell_type_data = sample_data[sample_data.obs[cell_type_column] == cell_type]
            if cell_type_data.shape[0] > 0:
                # Handle sparse and dense matrices
                if issparse(cell_type_data.X):
                    avg_expr = cell_type_data.X.mean(axis=0).A1.astype(np.float64)
                else:
                    avg_expr = cell_type_data.X.mean(axis=0).astype(np.float64)
                avg_expression[sample][cell_type] = avg_expr
            else:
                # If a cell type is not present in a sample, set average expression to zeros
                avg_expression[sample][cell_type] = np.zeros(hvg.shape[1], dtype=np.float64)

    # 2. Compute ground distance matrix between cell types based on average expression profiles
    # Compute global average expression profiles for each cell type across all samples
    global_avg_expression = {}
    for cell_type in cell_types:
        cell_type_data = hvg[hvg.obs[cell_type_column] == cell_type]
        if cell_type_data.shape[0] > 0:
            if issparse(cell_type_data.X):
                avg_expr = cell_type_data.X.mean(axis=0).A1.astype(np.float64)
            else:
                avg_expr = cell_type_data.X.mean(axis=0).astype(np.float64)
            global_avg_expression[cell_type] = avg_expr
        else:
            global_avg_expression[cell_type] = np.zeros(hvg.shape[1], dtype=np.float64)

    # Create a list of cell types to maintain order
    cell_type_list = list(cell_types)
    num_cell_types = len(cell_type_list)

    # Initialize the ground distance matrix
    ground_distance = np.zeros((num_cell_types, num_cell_types), dtype=np.float64)

    # Populate the ground distance matrix with Euclidean distances between cell type centroids
    for i in range(num_cell_types):
        for j in range(num_cell_types):
            expr_i = global_avg_expression[cell_type_list[i]]
            expr_j = global_avg_expression[cell_type_list[j]]
            distance = np.linalg.norm(expr_i - expr_j)
            ground_distance[i, j] = distance

    # 3. Normalize the ground distance matrix (optional but recommended)
    # This ensures that the distances are scaled appropriately for EMD
    max_distance = ground_distance.max()
    if max_distance > 0:
        ground_distance /= max_distance

    # Ensure ground_distance is float64
    ground_distance = ground_distance.astype(np.float64)

    # 4. Compute EMD between each pair of samples based on expression levels
    sample_distance_matrix = pd.DataFrame(0, index=samples, columns=samples, dtype=np.float64)

    for i, sample_i in enumerate(samples):
        for j, sample_j in enumerate(samples):
            if i < j:
                # Histograms (masses) for the two samples based on average expression per cell type
                # For each cell type, calculate the total expression in the sample
                # This assumes that higher expression indicates more "mass"
                hist_i = np.array([avg_expression[sample_i][ct].mean() for ct in cell_type_list], dtype=np.float64)
                hist_j = np.array([avg_expression[sample_j][ct].mean() for ct in cell_type_list], dtype=np.float64)

                # Normalize histograms to ensure they sum to the same value (e.g., 1)
                # This is necessary for EMD
                sum_i = hist_i.sum()
                sum_j = hist_j.sum()

                if sum_i == 0 or sum_j == 0:
                    # Handle cases where a sample has zero expression across all cell types
                    distance = np.inf
                else:
                    hist_i_normalized = (hist_i / sum_i).astype(np.float64)
                    hist_j_normalized = (hist_j / sum_j).astype(np.float64)
                    distance = emd(hist_i_normalized, hist_j_normalized, ground_distance)

                sample_distance_matrix.loc[sample_i, sample_j] = distance
                sample_distance_matrix.loc[sample_j, sample_i] = distance

    # Save the distance matrix
    distance_matrix_path = os.path.join(output_dir, 'sample_distance_matrix_expression.csv')
    sample_distance_matrix.to_csv(distance_matrix_path)
    avrg_expr_matrix_path = os.path.join(output_dir, 'avarage_expression.csv')

    data_rows = []
    for sample in samples:
        row = {'Sample': sample}
        # Iterate over each cell type
        for ct in cell_type_list:
            expression_array = avg_expression[sample][ct]
            # Iterate over each expression index
            for idx, expr_value in enumerate(expression_array):
                # Create a column name like 'T_cells_0', 'T_cells_1', etc.
                column_name = f"{ct}_{idx}"
                row[column_name] = expr_value
        data_rows.append(row)
    df_wide = pd.DataFrame(data_rows)
    df_wide.to_csv(avrg_expr_matrix_path)
    print(f"Sample distance matrix based on expression levels saved to {distance_matrix_path}")

    # 5. Generate a heatmap of the distance matrix
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
    print(f"Sample distance heatmap based on expression levels saved to {heatmap_path}")

    # 6. Plot cell type expression abundances (optional)
    # Implement this function based on your specific visualization needs
    # plot_cell_type_expression_profiles(avg_expression, output_dir)
    cell_type_distribution_map = os.path.join(output_dir, 'cell_type_expression_distribution.pdf')
    # Example placeholder for plotting function
    # plot_cell_type_expression_profiles(avg_expression, cell_type_distribution_map)
    print(f"Cell type expression distribution in samples should be saved to {cell_type_distribution_map}")

    plot_cell_type_expression_heatmap(
        avg_expression=avg_expression,
        output_dir=output_dir,
        figsize=(12, 10),
        cmap='viridis',
        annot=False  # Set to True if you want to annotate the heatmap with expression values
    )
    
    return sample_distance_matrix
