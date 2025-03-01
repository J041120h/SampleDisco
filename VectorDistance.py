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

    Parameters:
    - adata: AnnData object containing cell proportion data.
    - output_dir: Directory to save outputs.
    - method: Distance metric (e.g., 'euclidean', 'cityblock').
    - summary_csv_path: Path to summary CSV.
    - pseudobulk: Dictionary storing pseudobulk data.

    Returns:
    - distance_df: Pairwise distance matrix (samples x samples).
    """
    output_dir = os.path.join(output_dir, 'cell_proportion')
    os.makedirs(output_dir, exist_ok=True)

    cell_proportions = pseudobulk["cell_proportion"]

    # Compute distance matrix and normalize
    distance_matrix = pdist(cell_proportions.values, metric=method)
    distance_df = pd.DataFrame(squareform(distance_matrix), index=cell_proportions.index, columns=cell_proportions.index)
    distance_df = np.log1p(np.maximum(distance_df, 0)) / distance_df.max().max()

    # Save results
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_proportion.csv')
    distance_df.to_csv(distance_matrix_path)
    distanceCheck(distance_matrix_path, 'cell_proportion', method, summary_csv_path, adata)

    # Generate visualizations
    visualizeDistanceMatrix(distance_df, os.path.join(output_dir, 'sample_distance_proportion_heatmap.pdf'))
    visualizeGroupRelationship(distance_df, outputDir=output_dir, heatmap_path=os.path.join(output_dir, 'sample_proportion_relationship.pdf'))

    print(f"Cell proportion-based distance matrix saved to: {distance_matrix_path}")
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
    Compute a distance matrix based on average gene expression per sample.

    Parameters:
    - adata: AnnData object with sample labels in adata.obs[sample_column].
    - output_dir: Directory for output files.
    - method: Distance metric.
    - summary_csv_path: Path to summary CSV.
    - sample_column: Column in adata.obs indicating sample IDs.
    - normalize: If True, Z-score normalize gene expression.
    - log_transform: If True, apply log1p transformation.

    Returns:
    - distance_df: Sample distance matrix (samples x samples).
    """
    output_dir = os.path.join(output_dir, 'gene_expression')
    os.makedirs(output_dir, exist_ok=True)

    # Compute average gene expression per sample
    gene_expression = adata.to_df().groupby(adata.obs[sample_column]).mean().fillna(0)

    # Select top 2000 variable genes
    top_2000_genes = gene_expression.var(axis=0).nlargest(2000).index
    gene_expression_top2000 = gene_expression[top_2000_genes]

    # Save results
    gene_expression.to_csv(os.path.join(output_dir, 'average_gene_expression_per_sample.csv'))
    gene_expression_top2000.to_csv(os.path.join(output_dir, 'average_gene_expression_per_sample_top2000.csv'))

    # Compute pairwise distances and normalize
    distance_matrix = pdist(gene_expression_top2000.values, metric=method)
    distance_df = pd.DataFrame(squareform(distance_matrix), index=gene_expression_top2000.index, columns=gene_expression_top2000.index)
    distance_df = np.log1p(np.maximum(distance_df, 0)) / distance_df.max().max()

    # Save and visualize
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_gene_expression.csv')
    distance_df.to_csv(distance_matrix_path)
    visualizeDistanceMatrix(distance_df, os.path.join(output_dir, 'sample_distance_gene_expression_heatmap.pdf'))
    visualizeGroupRelationship(distance_df, outputDir=output_dir, heatmap_path=os.path.join(output_dir, 'sample_gene_expression_relationship.pdf'))

    print(f"Gene expression-based distance matrix saved to: {distance_matrix_path}")
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
    distance_df.to_csv(os.path.join(output_dir, 'distance_matrix_pca_harmony.csv'))
    visualizeDistanceMatrix(distance_df, os.path.join(output_dir, 'sample_distance_pca_harmony_heatmap.pdf'))
    visualizeGroupRelationship(distance_df, outputDir=output_dir, heatmap_path=os.path.join(output_dir, 'sample_pca_harmony_relationship.pdf'))

    print(f"PCA-based distance matrix saved to: {output_dir}")
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

    adata = adata[:, adata.var['highly_variable']].copy()

    # Compute distances using different methods
    calculate_sample_distances_cell_proportion(adata, output_dir, method, summary_csv_path, pseudobulk)
    calculate_sample_distances_gene_expression(adata, output_dir, method, summary_csv_path, sample_column)
    calculate_sample_distances_pca(adata, output_dir, method, summary_csv_path, sample_column)

    print("Sample distance computations completed.")
