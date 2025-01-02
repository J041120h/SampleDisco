import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform
from Visualization import visualizeGroupRelationship, visualizeDistanceMatrix
from distanceTest import distanceCheck
from scipy.sparse import issparse

def calculate_sample_distances_cell_proportion(
    adata: AnnData,
    output_dir: str,
    method: str,
    summary_csv_path: str,
    cell_type_column: str = 'cell_type',
    sample_column: str = 'sample'
) -> pd.DataFrame:
    # Ensure the directory exists
    output_dir = os.path.join(output_dir, 'cell_proportion')
    os.makedirs(output_dir, exist_ok=True)

    # Calculate cell type proportions
    cell_counts = adata.obs.groupby([sample_column, cell_type_column]).size().unstack(fill_value=0)
    cell_proportions = cell_counts.div(cell_counts.sum(axis=1), axis=0)

    # Calculate distance matrix
    distance_matrix = pdist(cell_proportions.values, metric=method)
    distance_df = pd.DataFrame(
        squareform(distance_matrix),
        index=cell_proportions.index,
        columns=cell_proportions.index
    )
    distance_df = np.log1p(np.maximum(distance_df, 0))
    distance_df = distance_df / distance_df.max().max()
    # Save the distance matrix
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_average_expression.csv')
    distance_df.to_csv(distance_matrix_path)
    distanceCheck(distance_matrix_path, 'cell_proportion', method, summary_csv_path)
    print(f"Sample distance proportion matrix saved to {distance_matrix_path}")

    # generate a heatmap for sample distance
    heatmap_path = os.path.join(output_dir, 'sample_distance_proportion_heatmap.pdf')
    visualizeDistanceMatrix(distance_df, heatmap_path)
    visualizeGroupRelationship(distance_df, outputDir=output_dir, heatmap_path=os.path.join(output_dir, 'sample_proportion_relationship.pdf'))
    print("Distance matrix based on average expression per cell type saved to 'distance_matrix_proportion.csv'.")
    return distance_df

def calculate_sample_distances_gene_expression(
    adata: AnnData,
    output_dir: str,
    method: str,
    summary_csv_path: str,
    sample_column: str = 'sample',
    normalize: bool = True,
    log_transform: bool = True
) -> pd.DataFrame:
    """
    Calculate distance matrix based on average gene expression for each sample.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - output_dir: Directory to save the distance matrix and related plots.
    - sample_column: Column in adata.obs indicating sample identifiers.
    - normalize: Whether to perform normalization (e.g., total count normalization).
    - log_transform: Whether to perform log1p transformation.

    Returns:
    - distance_df: DataFrame containing the pairwise distances between samples.
    """
    output_dir = os.path.join(output_dir, 'gene_expression')
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute the average expression of each gene per sample
    gene_expression = adata.to_df().groupby(adata.obs[sample_column]).mean()
    gene_expression.fillna(0, inplace=True)
    
    # Save the average expression to a CSV file
    gene_expression.to_csv(os.path.join(output_dir, 'average_gene_expression_per_sample.csv'))
    print("Average gene expression per sample saved to 'average_gene_expression_per_sample.csv'.")
    
    distance_matrix = pdist(gene_expression.values, metric=method)
    distance_df = pd.DataFrame(
        squareform(distance_matrix),
        index = gene_expression.index,
        columns = gene_expression.index
    )
    
    distance_df = np.log1p(np.maximum(distance_df, 0))
    distance_df = distance_df / distance_df.max().max()
    
    # Save the distance matrix
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_gene_expression.csv')
    distance_df.to_csv(distance_matrix_path)
    distanceCheck(distance_matrix_path, 'gene_expression', method, summary_csv_path)
    print(f"Sample distance gene expresission matrix saved to {distance_matrix_path}")

    # generate a heatmap for sample distance
    heatmap_path = os.path.join(output_dir, 'sample_distance_gene_expression_heatmap.pdf')
    visualizeDistanceMatrix(distance_df, heatmap_path)
    visualizeGroupRelationship(distance_df, outputDir=output_dir, heatmap_path=os.path.join(output_dir, 'sample_gene_expression_relationship.pdf'))
    print("Distance matrix based on gene expression per sample saved to 'distance_matrix_gene_expression.csv'.")
    return distance_df

def calculate_sample_distances_weighted_expression(
    adata: AnnData,
    output_dir: str,
    method: str,
    summary_csv_path: str,
    cell_type_column: str = 'cell_type',
    sample_column: str = 'sample'
) -> pd.DataFrame:
    """
    Calculate distance matrix based on average gene expression per cell type multiplied by their abundance
    (cell type proportions) for each sample.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - output_dir: Directory to save the distance matrix and related plots.
    - method: Distance metric to use (e.g., 'euclidean', 'cityblock').
    - summary_csv_path: Path to save the summary CSV.
    - cell_type_column: Column in adata.obs indicating cell types.
    - sample_column: Column in adata.obs indicating sample identifiers.

    Returns:
    - distance_df: DataFrame containing the pairwise distances between samples.
    """
    output_dir = os.path.join(output_dir, 'weighted_expression')
    os.makedirs(output_dir, exist_ok=True)

    samples = adata.obs[sample_column].unique()
    cell_types = adata.obs[cell_type_column].unique()

    # Compute cell type proportions in each sample
    proportions = pd.DataFrame(0, index=samples, columns=cell_types, dtype=np.float64)

    for sample in samples:
        sample_mask = adata.obs[sample_column] == sample
        sample_data = adata.obs.loc[sample_mask]
        total_cells = sample_data.shape[0]
        counts = sample_data[cell_type_column].value_counts()
        proportions.loc[sample, counts.index] = counts.values / total_cells

    # Compute average expression profiles for each cell type in each sample
    avg_expression = {sample: {} for sample in samples}
    for sample in samples:
        sample_mask = adata.obs[sample_column] == sample
        sample_data = adata[sample_mask]
        for cell_type in cell_types:
            cell_type_mask = sample_data.obs[cell_type_column] == cell_type
            cell_type_data = sample_data[cell_type_mask]
            if cell_type_data.shape[0] > 0:
                # Handle sparse and dense matrices
                if issparse(cell_type_data.X):
                    avg_expr = cell_type_data.X.mean(axis=0).A1.astype(np.float64)
                else:
                    avg_expr = cell_type_data.X.mean(axis=0).astype(np.float64)
                avg_expression[sample][cell_type] = avg_expr
            else:
                # If a cell type is not present in a sample, set average expression to zeros
                avg_expression[sample][cell_type] = np.zeros(adata.shape[1], dtype=np.float64)

    # Find the global min and max values across all avg_expression
    all_values = np.concatenate([
        np.concatenate(list(cell_dict.values()))
        for cell_dict in avg_expression.values()
    ])
    min_val = np.min(all_values)
    max_val = np.max(all_values)

    # Apply normalization
    for sample in samples:
        for cell_type in cell_types:
            value = avg_expression[sample][cell_type]
            if max_val > min_val:
                avg_expression[sample][cell_type] = (value - min_val) / (max_val - min_val)
            else:
                avg_expression[sample][cell_type] = np.zeros_like(value)

    # Multiply cell type proportions with their average expression profiles to get weighted expression
    weighted_expression = pd.DataFrame(index=samples, columns=adata.var_names, dtype=np.float64)

    for sample in samples:
        total_weighted_expr = np.zeros(adata.shape[1], dtype=np.float64)
        for cell_type in cell_types:
            proportion = proportions.loc[sample, cell_type]
            weighted_expr = proportion * avg_expression[sample][cell_type]
            total_weighted_expr += weighted_expr
        weighted_expression.loc[sample] = total_weighted_expr

    # Calculate distance matrix
    distance_matrix = pdist(weighted_expression.values, metric=method)
    distance_df = pd.DataFrame(
        squareform(distance_matrix),
        index=weighted_expression.index,
        columns=weighted_expression.index
    )
    # Normalize the distance matrix
    distance_df = np.log1p(np.maximum(distance_df, 0))
    distance_df = distance_df / distance_df.max().max()

    # Save the distance matrix
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_weighted_expression.csv')
    distance_df.to_csv(distance_matrix_path)
    # Assuming distanceCheck is a function that processes the distance matrix
    distanceCheck(distance_matrix_path, 'weighted_expression', method, summary_csv_path)
    print(f"Sample distance weighted expression matrix saved to {distance_matrix_path}")

    # Generate heatmaps for sample distance
    heatmap_path = os.path.join(output_dir, 'sample_distance_weighted_expression_heatmap.pdf')
    visualizeDistanceMatrix(distance_df, heatmap_path)
    visualizeGroupRelationship(
        distance_df,
        outputDir=output_dir,
        heatmap_path=os.path.join(output_dir, 'sample_weighted_expression_relationship.pdf')
    )
    print("Distance matrix based on weighted expression per sample saved.")

    return distance_df

def sample_distance(
    adata: AnnData,
    output_dir: str,
    method: str,
    summary_csv_path: str = "/users/harry/desktop/GenoDistance/result/summary.csv",
    sample_column: str = 'sample',
    normalize: bool = True,
    log_transform: bool = True
) -> pd.DataFrame:
    output_dir = os.path.join(os.path.join(output_dir, method))
    calculate_sample_distances_cell_proportion(adata, output_dir, method, summary_csv_path)
    calculate_sample_distances_gene_expression(adata, output_dir, method, summary_csv_path)
    # calculate_sample_distances_weighted_expression(adata, output_dir, method, summary_csv_path)
