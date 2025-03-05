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
    cell_proportions = cell_proportions.T

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
        adata = adata,
        heatmap_path=os.path.join(output_dir, 'sample_proportion_relationship.pdf')
    )

    print(f"Cell proportion-based sample distance matrix saved to: {distance_matrix_path}")
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
    distanceCheck(distance_matrix_path, 'cell_expression', method, summary_csv_path, adata)
    visualizeDistanceMatrix(distance_df, os.path.join(output_dir, 'sample_distance_gene_expression_heatmap.pdf'))
    visualizeGroupRelationship(distance_df, outputDir=output_dir, adata = adata, heatmap_path=os.path.join(output_dir, 'sample_gene_expression_relationship.pdf'))

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
    output_subdir = os.path.join(output_dir, 'gene_pseudobulk')
    os.makedirs(output_subdir, exist_ok=True)

    cell_expr = pseudobulk['cell_expression_corrected']
    samples, cell_types = cell_expr.columns, cell_expr.index

    # Build concatenated expression vectors per sample
    sample_vectors = {
        sample: np.concatenate([cell_expr.loc[ct, sample] for ct in cell_types])
        for sample in samples
    }

    sample_df = pd.DataFrame.from_dict(sample_vectors, orient='index')
    sample_df.index.name = 'sample'
    sample_df.columns = [f"feature_{i}" for i in range(sample_df.shape[1])]

    # Select top 2000 most variable features
    top_2000_features = sample_df.var(axis=0).nlargest(2000).index
    sample_df_top2000 = sample_df[top_2000_features]

    # Save transformed data
    sample_df_top2000.to_csv(os.path.join(output_subdir, 'average_gene_pseudobulk_per_sample_top2000.csv'))

    # Compute pairwise distances
    distance_matrix = pdist(sample_df_top2000.values, metric=method)
    distance_df = pd.DataFrame(squareform(distance_matrix), index=samples, columns=samples)

    # Save results
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_gene_expression.csv')
    distance_df.to_csv(distance_matrix_path)
    distanceCheck(distance_matrix_path, 'cell_pseudobulk', method, summary_csv_path, adata)
    visualizeDistanceMatrix(distance_df, os.path.join(output_subdir, 'sample_distance_pseudobulk_heatmap.pdf'))
    visualizeGroupRelationship(distance_df, outputDir=output_subdir, adata = adata, heatmap_path=os.path.join(output_subdir, 'sample_pseudobulk_relationship.pdf'))
    print(f"Sample distance matrix (top 2000 HVG) saved to {distance_matrix_path}")

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
    Compute a distance matrix using weighted gene expression, where each 
    sample's cell type expression is scaled by its proportion.

    Parameters:
    - adata: AnnData object.
    - output_dir: Directory to save results.
    - method: Distance metric.
    - summary_csv_path: Path for logging.
    - cell_type_column: Column for cell types.
    - sample_column: Column for sample IDs.

    Returns:
    - distance_df: Pairwise distance matrix.
    """
    output_dir = os.path.join(output_dir, 'weighted_expression')
    os.makedirs(output_dir, exist_ok=True)

    samples, cell_types = adata.obs[sample_column].unique(), adata.obs[cell_type_column].unique()

    # Compute cell type proportions per sample
    proportions = pd.DataFrame(0, index=samples, columns=cell_types, dtype=np.float64)
    for sample in samples:
        sample_mask = adata.obs[sample_column] == sample
        sample_data = adata.obs.loc[sample_mask]
        total_cells = sample_data.shape[0]
        counts = sample_data[cell_type_column].value_counts()
        proportions.loc[sample, counts.index] = counts.values / total_cells

    # Compute average expression profiles per cell type
    avg_expression = {sample: {} for sample in samples}
    for sample in samples:
        sample_mask = adata.obs[sample_column] == sample
        sample_data = adata[sample_mask]
        for cell_type in cell_types:
            cell_mask = sample_data.obs[cell_type_column] == cell_type
            cell_data = sample_data[cell_mask]

            if cell_data.shape[0] > 0:
                avg_expr = cell_data.X.mean(axis=0).A1 if issparse(cell_data.X) else cell_data.X.mean(axis=0)
            else:
                avg_expr = np.zeros(adata.shape[1], dtype=np.float64)

            avg_expression[sample][cell_type] = avg_expr

    # Normalize values
    all_values = np.concatenate([np.concatenate(list(cell_dict.values())) for cell_dict in avg_expression.values()])
    min_val, max_val = np.min(all_values), np.max(all_values)

    for sample in samples:
        for cell_type in cell_types:
            value = avg_expression[sample][cell_type]
            avg_expression[sample][cell_type] = (value - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(value)

    # Compute weighted expression per sample
    weighted_expression = pd.DataFrame(index=samples, columns=adata.var_names, dtype=np.float64)
    for sample in samples:
        total_weighted_expr = np.zeros(adata.shape[1], dtype=np.float64)
        for cell_type in cell_types:
            proportion = proportions.loc[sample, cell_type]
            total_weighted_expr += proportion * avg_expression[sample][cell_type]
        weighted_expression.loc[sample] = total_weighted_expr

    # Compute pairwise distances
    distance_matrix = pdist(weighted_expression.values, metric=method)
    distance_df = pd.DataFrame(squareform(distance_matrix), index=weighted_expression.index, columns=weighted_expression.index)
    distance_df = np.log1p(np.maximum(distance_df, 0)) / distance_df.max().max()

    # Save results
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_weighted_expression.csv')
    distance_df.to_csv(distance_matrix_path)
    print(f"Sample distance weighted expression matrix saved to {distance_matrix_path}")

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
    calculate_sample_distances_gene_pseudobulk(adata, output_dir, method, summary_csv_path, pseudobulk, sample_column)

    print("Sample distance computations completed.")
