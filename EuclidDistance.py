import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform
from Visualization import plot_cell_type_abundances, visualizeGroupRelationship, visualizeDistanceMatrix
from distanceTest import distanceCheck

def calculate_sample_distances_cell_proportion(
    adata: AnnData,
    output_dir: str,
    method: str,
    cell_type_column: str = 'leiden',
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
    distance_df = distance_df / distance_df.max().max()
    # Save the distance matrix
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_average_expression.csv')
    distance_df.to_csv(distance_matrix_path)
    distanceCheck(distance_matrix_path)
    print(f"Sample distance proportion matrix saved to {distance_matrix_path}")

    # generate a heatmap for sample distance
    heatmap_path = os.path.join(output_dir, 'sample_distance_proportion_heatmap.pdf')
    visualizeDistanceMatrix(distance_df, heatmap_path)
    visualizeGroupRelationship(distance_df, outputDir=output_dir, heatmap_path=os.path.join(output_dir, 'sample_proportion_relationship.pdf'))
    print("Distance matrix based on average expression per cell type saved to 'distance_matrix_proportion.csv'.")
    return distance_df

def calculate_sample_distances_average_expression(
    adata: AnnData,
    output_dir: str,
    method: str,
    cell_type_column: str = 'leiden',
    sample_column: str = 'sample'
) -> pd.DataFrame:
    """
    Calculate distance matrix based on average gene expression per cell type for each sample.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - output_dir: Directory to save the distance matrix and related plots.
    - cell_type_column: Column in adata.obs indicating cell types.
    - sample_column: Column in adata.obs indicating sample identifiers.

    Returns:
    - distance_df: DataFrame containing the pairwise distances between samples.
    """
    output_dir = os.path.join(output_dir, 'avarage_expression')
    os.makedirs(output_dir, exist_ok=True)

    avg_expression = adata.to_df().groupby([adata.obs[sample_column], adata.obs[cell_type_column]]).mean()
    # Reshape to have samples as rows and (cell_type, gene) as columns
    avg_expression = avg_expression.unstack(level=1)
    # Flatten the MultiIndex columns
    avg_expression.columns = ['{}_{}'.format(cell_type, gene) for cell_type, gene in avg_expression.columns]
    avg_expression.fillna(0, inplace=True)  # Handle any missing values

    # Save average expression to a CSV file
    avg_expression.to_csv(os.path.join(output_dir, 'average_expression_per_cell_type.csv'))
    print("Average expression per cell type saved to 'average_expression_per_cell_type.csv'.")

    # Calculate distance matrix
    distance_matrix = pdist(avg_expression.values, metric = method)
    distance_df = pd.DataFrame(
        squareform(distance_matrix),
        index=avg_expression.index,
        columns=avg_expression.index
    )
    distance_df = distance_df / distance_df.max().max()

    # Save the distance matrix
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_average_expression.csv')
    distance_df.to_csv(distance_matrix_path)
    distanceCheck(distance_matrix_path)
    print(f"Sample distance avarage expresission matrix saved to {distance_matrix_path}")

    # generate a heatmap for sample distance
    heatmap_path = os.path.join(output_dir, 'sample_distance_average_expression_heatmap.pdf')
    visualizeDistanceMatrix(distance_df, heatmap_path)
    visualizeGroupRelationship(distance_df, outputDir=output_dir, heatmap_path=os.path.join(output_dir, 'sample_avarage_expression_relationship.pdf'))
    print("Distance matrix based on average expression per cell type saved to 'distance_matrix_average_expression.csv'.")
    return distance_df

def calculate_sample_distances_gene_expression(
    adata: AnnData,
    output_dir: str,
    method: str,
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
    
    # Step 3: Calculate Distance Matrix
    distance_matrix = pdist(gene_expression.values, metric=method)
    distance_df = pd.DataFrame(
        squareform(distance_matrix),
        index = gene_expression.index,
        columns = gene_expression.index
    )
    
    distance_df = distance_df / distance_df.max().max()
     # Save the distance matrix
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_gene_expression.csv')
    distance_df.to_csv(distance_matrix_path)
    distanceCheck(distance_matrix_path)
    print(f"Sample distance gene expresission matrix saved to {distance_matrix_path}")

    # generate a heatmap for sample distance
    heatmap_path = os.path.join(output_dir, 'sample_distance_gene_expression_heatmap.pdf')
    visualizeDistanceMatrix(distance_df, heatmap_path)
    visualizeGroupRelationship(distance_df, outputDir=output_dir, heatmap_path=os.path.join(output_dir, 'sample_gene_expression_relationship.pdf'))
    print("Distance matrix based on gene expression per sample saved to 'distance_matrix_gene_expression.csv'.")
    return distance_df

def sample_distance(
    adata: AnnData,
    output_dir: str,
    method: str,
    sample_column: str = 'sample',
    normalize: bool = True,
    log_transform: bool = True
) -> pd.DataFrame:
    calculate_sample_distances_cell_proportion(adata, output_dir, method)
    calculate_sample_distances_average_expression(adata, output_dir, method)
    calculate_sample_distances_gene_expression(adata, output_dir, method)
