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
from typing import Optional
from HVG import select_hvf_loess
import logging

def calculate_sample_distances_cell_proportion(
    adata: AnnData,
    output_dir: str,
    method: str,
    summary_csv_path: str,
    pseudobulk: dict
) -> pd.DataFrame:
    """
    Compute a sample distance matrix using cell proportions.

    Parameters
    ----------
    adata : AnnData
        The single-cell data object (not directly used here, but needed for logging/metadata).
    output_dir : str
        Directory to save outputs (distance matrix, heatmaps, etc.).
    method : str
        Distance metric (e.g., 'euclidean', 'cityblock').
    summary_csv_path : str
        Path to summary CSV, for logging distanceCheck results.
    pseudobulk : dict
        Dictionary storing pseudobulk data, including "cell_proportion".

    Returns
    -------
    distance_df : pd.DataFrame
        Pairwise distance matrix (samples x samples).
    """
    # Create subdirectory for results
    output_dir = os.path.join(output_dir, 'cell_proportion')
    os.makedirs(output_dir, exist_ok=True)

    # cell_proportions is assumed to be a DataFrame where:
    #   rows = cell types, columns = samples, each value = proportion
    cell_proportions = pseudobulk["cell_proportion"]

    # Transpose so that rows = samples, columns = cell types
    # Now each row is a sample vector of cell-type proportions

    # Compute the sample-sample distance matrix and normalize it
    distance_matrix = pdist(cell_proportions.values, metric=method)
    distance_df = pd.DataFrame(
        squareform(distance_matrix),
        index=cell_proportions.index,
        columns=cell_proportions.index
    )

    # Example post-processing: log1p transform & scale to [0, 1]
    distance_df = np.log1p(np.maximum(distance_df, 0))
    distance_df = distance_df / distance_df.max().max()

    # Save distance matrix to CSV
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_proportion.csv')
    distance_df.to_csv(distance_matrix_path)

    # If you have a custom distanceCheck function for logging
    distanceCheck(distance_matrix_path, 'cell_proportion', method, summary_csv_path, adata)

    # Visualizations
    visualizeDistanceMatrix(distance_df, os.path.join(output_dir, 'sample_distance_proportion_heatmap.pdf'))
    visualizeGroupRelationship(
        distance_df,
        outputDir=output_dir,
        adata = adata
    )

    print(f"Cell proportion-based sample distance matrix saved to: {distance_matrix_path}")
    return distance_df

def calculate_sample_distances_pca(
    adata: AnnData,
    output_dir: str,
    method: str = 'euclidean',
    summary_csv_path: Optional[str] = None,
    sample_column: str = 'sample'
) -> pd.DataFrame:
    """
    Compute a sample distance matrix using PCA-transformed data.

    Parameters:
    - adata: AnnData object with PCA stored in .obsm['X_pca_harmony'].
    - output_dir: Directory for output files.
    - method: Distance metric.
    - summary_csv_path: Path for logging.
    - sample_column: Column in adata.obs indicating sample IDs.

    Returns:
    - distance_df: Sample distance matrix (samples x samples).
    """
    if 'X_pca_harmony' not in adata.obsm:
        raise KeyError("PCA data 'X_pca_harmony' not found in adata.obsm.")

    output_dir = os.path.join(output_dir, 'pca_distance_harmony')
    os.makedirs(output_dir, exist_ok=True)

    # Compute PCA sample averages
    pca_data = pd.DataFrame(adata.obsm['X_pca_harmony'], index=adata.obs_names)
    pca_data['sample'] = adata.obs[sample_column].values
    average_pca = pca_data.groupby('sample').mean().fillna(0)

    # Save PCA data
    average_pca.to_csv(os.path.join(output_dir, 'average_pca_per_sample.csv'))

    # Compute distances
    distance_matrix = pdist(average_pca.values, metric=method)
    distance_df = pd.DataFrame(squareform(distance_matrix), index=average_pca.index, columns=average_pca.index)
    distance_df = np.log1p(np.maximum(distance_df, 0)) / distance_df.max().max()

    # Save and visualize
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_gene_expression.csv')
    distance_df.to_csv(distance_matrix_path)
    distanceCheck(distance_matrix_path, 'pca_harmony', method, summary_csv_path, adata)
    visualizeDistanceMatrix(distance_df, os.path.join(output_dir, 'sample_distance_pca_harmony_heatmap.pdf'))
    visualizeGroupRelationship(distance_df, outputDir=output_dir, adata = adata, heatmap_path=os.path.join(output_dir, 'sample_pca_harmony_relationship.pdf'))

    print(f"PCA-based distance matrix saved to: {output_dir}")
    return distance_df

def calculate_sample_distances_gene_pseudobulk(
    adata: AnnData,
    output_dir: str,
    method: str,
    summary_csv_path: str,
    pseudobulk: dict,
    sample_column: str = 'sample',
    celltype_column: str = 'cell_type'
) -> pd.DataFrame:
    """
    Compute a distance matrix based on concatenated average gene expressions 
    from different cell types for each sample.

    Parameters:
    - adata: AnnData object.
    - output_dir: Directory to save results.
    - method: Distance metric.
    - summary_csv_path: Path for logging.
    - pseudobulk: Dictionary with 'cell_expression_corrected'.
    - sample_column: Column name for sample IDs.
    - celltype_column: Column name for cell type labels.

    Returns:
    - distance_df: Pairwise distance matrix.
    """
    output_dir = os.path.join(output_dir, 'cell_expression')
    os.makedirs(output_dir, exist_ok=True)

    # cell_proportions is assumed to be a DataFrame where:
    #   rows = cell types, columns = samples, each value = proportion
    cell_expression_corrected = pseudobulk["cell_expression_corrected"]

    distance_matrix = pdist(cell_expression_corrected.values, metric=method)
    distance_df = pd.DataFrame(
        squareform(distance_matrix),
        index=cell_expression_corrected.index,
        columns=cell_expression_corrected.index
    )

    distance_df = np.log1p(np.maximum(distance_df, 0))
    distance_df = distance_df / distance_df.max().max()

    # Save distance matrix to CSV
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_expression.csv')
    distance_df.to_csv(distance_matrix_path)

    # If you have a custom distanceCheck function for logging
    distanceCheck(distance_matrix_path, 'cell_expression', method, summary_csv_path, adata)

    # Visualizations
    visualizeDistanceMatrix(distance_df, os.path.join(output_dir, 'sample_distance_expression_heatmap.pdf'))
    visualizeGroupRelationship(
        distance_df,
        outputDir=output_dir,
        adata = adata
    )

    print(f"Cell expression-based sample distance matrix saved to: {distance_matrix_path}")
    return distance_df

def sample_distance(
    adata: AnnData,
    output_dir: str,
    method: str,
    summary_csv_path: str = "/users/harry/desktop/GenoDistance/result/summary.csv",
    pseudobulk: dict = None,
    sample_column: str = 'sample'
) -> None:
    """
    Compute and save sample distance matrices using different features.

    Parameters:
    - adata: AnnData object.
    - output_dir: Directory for outputs.
    - method: Distance metric.
    - summary_csv_path: Path to summary CSV.
    - pseudobulk: Dictionary containing pseudobulk data.
    - sample_column: Sample ID column.

    Returns:
    - None (results are saved to disk).
    """
    output_dir = os.path.join(output_dir, method)
    os.makedirs(output_dir, exist_ok=True)

    # Compute distances using different methods
    calculate_sample_distances_cell_proportion(adata, output_dir, method, summary_csv_path, pseudobulk)
    # calculate_sample_distances_gene_expression(adata, output_dir, method, summary_csv_path, sample_column)
    # calculate_sample_distances_pca(adata, output_dir, method, summary_csv_path, sample_column)
    calculate_sample_distances_gene_pseudobulk(adata, output_dir, method, summary_csv_path, pseudobulk, sample_column)

    print("Sample distance computations completed.")
