import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import warnings
from anndata._core.aligned_df import ImplicitModificationWarning
from Visualization import plot_cell_type_expression_heatmap, plot_cell_type_abundances, visualizeGroupRelationship, visualizeDistanceMatrix
from distanceTest import distanceCheck
from scipy.sparse import issparse

warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

def calculate_sample_distances_cell_proportion_chi_square(
    adata: AnnData,
    output_dir: str,
    summary_csv_path: str = "/users/harry/desktop/GenoDistance/result/summary.csv",
    cell_type_column: str = 'cell_type',
    sample_column: str = 'sample',
) -> pd.DataFrame:
    """
    Calculate distances between samples based on the proportions of each cell type using Chi-Square Distance.

    This function computes the Chi-Square distance between each pair of samples by considering the distribution of cell types within each sample.

    Parameters:
    ----------
    adata : AnnData
        The integrated single-cell dataset obtained from the previous analysis.
    output_dir : str
        Directory to save the output files.
    cell_type_column : str, optional
        Column name in `adata.obs` that contains the cell type assignments (default: 'cell_type').
    sample_column : str, optional
        Column name in `adata.obs` that contains the sample information (default: 'sample').
    summary_csv_path : str, optional
        Path to the summary CSV file to record distance checks.

    Returns:
    -------
    sample_distance_matrix : pandas.DataFrame
        A symmetric matrix of distances between samples.
    """

    # Check if output directory exists and create if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Automatically generating output directory")

    # Append 'cell_proportion' to the output directory path
    output_dir = os.path.join(output_dir, 'cell_proportion')

    # Create the new subdirectory if it doesn’t exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Automatically generating cell_proportion subdirectory")

    # 1. Compute cell type proportions in each sample
    samples = adata.obs[sample_column].unique()
    cell_types = adata.obs[cell_type_column].unique()

    # Create a DataFrame to hold proportions
    proportions = pd.DataFrame(0, index=samples, columns=cell_types, dtype=np.float64)

    for sample in samples:
        sample_data = adata.obs[adata.obs[sample_column] == sample]
        total_cells = sample_data.shape[0]
        counts = sample_data[cell_type_column].value_counts()
        proportions.loc[sample, counts.index] = counts.values / total_cells

    # 2. Compute Chi-Square distance between each pair of samples
    num_samples = len(samples)
    sample_distance_matrix = pd.DataFrame(0, index=samples, columns=samples, dtype=np.float64)

    # Replace zeros to avoid division by zero in denominator
    epsilon = 1e-10
    proportions = proportions.replace(0, epsilon)

    for i, sample_i in enumerate(samples):
        hist_i = proportions.loc[sample_i].values
        for j, sample_j in enumerate(samples):
            if i < j:
                hist_j = proportions.loc[sample_j].values

                # Compute Chi-Square distance
                chi_square = 0.5 * np.sum(((hist_i - hist_j) ** 2) / (hist_i + hist_j))
                sample_distance_matrix.loc[sample_i, sample_j] = chi_square
                sample_distance_matrix.loc[sample_j, sample_i] = chi_square

    # Save the distance matrix
    distance_matrix_path = os.path.join(output_dir, 'sample_distance_proportion_matrix.csv')
    sample_distance_matrix.to_csv(distance_matrix_path)
    distanceCheck(distance_matrix_path, "cell_proportion", "Chi-Square", summary_csv_path, adata)
    print(f"Sample distance proportion matrix saved to {distance_matrix_path}")

    # Save the cell type distribution map
    cell_type_distribution_map = os.path.join(output_dir, 'cell_type_distribution.pdf')
    plot_cell_type_abundances(proportions, output_dir)
    print(f"Cell type distribution in Sample saved to {cell_type_distribution_map}")

    # Generate a heatmap for sample distance
    heatmap_path = os.path.join(output_dir, 'sample_distance_proportion_heatmap.pdf')
    visualizeDistanceMatrix(sample_distance_matrix, heatmap_path)
    visualizeGroupRelationship(
        sample_distance_matrix,
        outputDir=output_dir,
        adata = adata
    )

    return sample_distance_matrix

# def calculate_sample_distances_cell_expression_chi_square(
#     adata: AnnData,
#     output_dir: str,
#     summary_csv_path: str = "/users/harry/desktop/GenoDistance/result/summary.csv",
#     cell_type_column: str = 'cell_type',
#     sample_column: str = 'sample',
# ) -> pd.DataFrame:
#     """
#     Calculate distances between samples based on the expression levels of each cell type using Chi-Square Distance.

#     This function computes the Chi-Square distance between each pair of samples by considering the distribution of cell type expression levels within each sample.

#     Parameters:
#     ----------
#     adata : AnnData
#         The integrated single-cell dataset with highly variable genes calculated.
#     output_dir : str
#         Directory to save the output files.
#     cell_type_column : str, optional
#         Column name in `adata.obs` that contains the cell type assignments (default: 'cell_type').
#     sample_column : str, optional
#         Column name in `adata.obs` that contains the sample information (default: 'sample').
#     summary_csv_path : str, optional
#         Path to the summary CSV file to record distance checks.

#     Returns:
#     -------
#     sample_distance_matrix : pandas.DataFrame
#         A symmetric matrix of distances between samples.
#     """

#     # Ensure output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print("Automatically generated output directory.")

#     # Append 'cell_expression' to the output directory path
#     output_dir = os.path.join(output_dir, 'cell_expression')

#     # Create the new subdirectory if it doesn’t exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print("Automatically generating cell_expression subdirectory")

#     # Check if highly variable genes are present
#     if 'highly_variable' not in adata.var.columns:
#         raise ValueError("Highly variable genes have not been calculated. Please run `sc.pp.highly_variable_genes` on the AnnData object.")

#     # Filter the data to include only highly variable genes
#     hvg = adata[:, adata.var['highly_variable']]

#     # 1. Compute average expression profiles for each cell type in each sample
#     samples = hvg.obs[sample_column].unique()
#     cell_types = hvg.obs[cell_type_column].unique()

#     # Initialize a dictionary to hold average expression profiles
#     avg_expression = {sample: {} for sample in samples}
#     for sample in samples:
#         sample_data = hvg[hvg.obs[sample_column] == sample]
#         for cell_type in cell_types:
#             cell_type_data = sample_data[sample_data.obs[cell_type_column] == cell_type]
#             if cell_type_data.shape[0] > 0:
#                 # Handle sparse and dense matrices
#                 if issparse(cell_type_data.X):
#                     avg_expr = cell_type_data.X.mean(axis=0).A1.astype(np.float64)
#                 else:
#                     avg_expr = cell_type_data.X.mean(axis=0).astype(np.float64)
#                 avg_expression[sample][cell_type] = avg_expr
#             else:
#                 # If a cell type is not present in a sample, set average expression to zeros
#                 avg_expression[sample][cell_type] = np.zeros(hvg.shape[1], dtype=np.float64)

#     # Find the global min and max values
#     all_values = [value for sample_dict in avg_expression.values() for value in sample_dict.values()]
#     min_val = np.min([np.min(v) for v in all_values])
#     max_val = np.max([np.max(v) for v in all_values])

#     # Apply normalization
#     if max_val > min_val:
#         avg_expression = {
#             sample: {
#                 cell_type: (value - min_val) / (max_val - min_val)
#                 for cell_type, value in sample_dict.items()
#             }
#             for sample, sample_dict in avg_expression.items()
#         }
#     else:
#         # If all values are the same, set expressions to zero
#         avg_expression = {
#             sample: {
#                 cell_type: np.zeros_like(value)
#                 for cell_type, value in sample_dict.items()
#             }
#             for sample, sample_dict in avg_expression.items()
#         }

#     cell_type_list = list(cell_types)

#     # 2. Compute Chi-Square distance between each pair of samples based on expression levels
#     sample_distance_matrix = pd.DataFrame(0, index=samples, columns=samples, dtype=np.float64)

#     epsilon = 1e-10  # Small value to avoid division by zero

#     for i, sample_i in enumerate(samples):
#         for j, sample_j in enumerate(samples):
#             if i < j:
#                 # Histograms for the two samples based on average expression per cell type
#                 hist_i = np.array([avg_expression[sample_i][ct].mean() for ct in cell_type_list], dtype=np.float64)
#                 hist_j = np.array([avg_expression[sample_j][ct].mean() for ct in cell_type_list], dtype=np.float64)

#                 # Replace zeros to avoid division by zero
#                 hist_i_nonzero = np.where(hist_i == 0, epsilon, hist_i)
#                 hist_j_nonzero = np.where(hist_j == 0, epsilon, hist_j)

#                 # Compute Chi-Square distance
#                 numerator = (hist_i_nonzero - hist_j_nonzero) ** 2
#                 denominator = hist_i_nonzero + hist_j_nonzero + epsilon  # Avoid division by zero
#                 chi_square_distance = 0.5 * np.sum(numerator / denominator)

#                 sample_distance_matrix.loc[sample_i, sample_j] = chi_square_distance
#                 sample_distance_matrix.loc[sample_j, sample_i] = chi_square_distance

#     # Save the distance matrix
#     distance_matrix_path = os.path.join(output_dir, 'sample_distance_matrix_expression.csv')
#     sample_distance_matrix.to_csv(distance_matrix_path)
#     distanceCheck(distance_matrix_path, "average_expression", "Chi-Square", summary_csv_path, adata)
#     avg_expr_matrix_path = os.path.join(output_dir, 'average_expression.csv')

#     # Save the average expression data
#     data_rows = []
#     for sample in samples:
#         row = {'Sample': sample}
#         # Iterate over each cell type
#         for ct in cell_type_list:
#             expression_array = avg_expression[sample][ct]
#             # Iterate over each gene index
#             for idx, expr_value in enumerate(expression_array):
#                 # Create a column name like 'CellType_GeneIndex'
#                 column_name = f"{ct}_{idx}"
#                 row[column_name] = expr_value
#         data_rows.append(row)
#     df_wide = pd.DataFrame(data_rows)
#     df_wide.to_csv(avg_expr_matrix_path)
#     print(f"Sample distance matrix based on expression levels saved to {distance_matrix_path}")

#     # 3. Generate a heatmap of the distance matrix
#     heatmap_path = os.path.join(output_dir, 'sample_distance_heatmap_expression.pdf')
#     visualizeDistanceMatrix(sample_distance_matrix, heatmap_path)
    
#     # 4. Plot cell type expression abundances
#     cell_type_distribution_map = os.path.join(output_dir, 'cell_type_expression_distribution.pdf')
#     print(f"Cell type expression distribution in samples saved to {cell_type_distribution_map}")

#     plot_cell_type_expression_heatmap(
#         avg_expression=avg_expression,
#         output_dir=output_dir,
#         figsize=(12, 10),
#         cmap='viridis',
#         annot=False  # Set to True if you want to annotate the heatmap with expression values
#     )

#     # 5. Visualize group relationships
#     visualizeGroupRelationship(
#         sample_distance_matrix,
#         outputDir=output_dir,
#         adata = adata,
#         heatmap_path=os.path.join(output_dir, 'sample_expression_relationship.pdf')
#     )

#     return sample_distance_matrix

def chi_square_distance(
    adata: AnnData,
    output_dir: str,
    summary_csv_path: str = "/users/harry/desktop/GenoDistance/result/summary.csv",
    sample_column: str = 'sample',
    normalize: bool = True,
    log_transform: bool = True
) -> pd.DataFrame:
    method = "Chi_Square"
    calculate_sample_distances_cell_proportion_chi_square(adata, output_dir, summary_csv_path)