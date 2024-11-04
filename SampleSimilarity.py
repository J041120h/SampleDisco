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
from SampleSimilarityCellExpression import calculate_sample_distances_cell_expression
from SampleSimilarityCellProportion import calculate_sample_distances_cell_proprotion
from SampleSimilarityWeighted import calculate_sample_distances_weighted_expression

def Sample_distances(
    adata: AnnData,
    output_dir: str,
    cell_type_column: str = 'leiden',
    sample_column: str = 'sample',
    proportion_weight: float = 1.0,
    expression_weight: float = 1.0,
    cell_group_weight = 0.8
) -> pd.DataFrame:
    """
    Calculate combined distances between samples based on cell type proportions and gene expression.

    Parameters:
    ----------
    adata : AnnData
        The integrated single-cell dataset obtained from the previous analysis.
    output_dir : str
        Directory to save the output files.
    cell_type_column : str, optional
        Column name in `adata.obs` that contains the cell type assignments (default: 'leiden').
    sample_column : str, optional
        Column name in `adata.obs` that contains the sample information (default: 'sample').
    proportion_weight : float, optional
        Weight for the proportion distance matrix (default: 1.0).
    expression_weight : float, optional
        Weight for the expression distance matrix (default: 1.0).

    Returns:
    -------
    combined_matrix : pd.DataFrame
        A symmetric matrix of combined distances between samples.
    """

    # Calculate the proportion distance matrix
    proportion_matrix = calculate_sample_distances_cell_proprotion(
        adata=adata,
        output_dir=output_dir,
        cell_type_column=cell_type_column,
        sample_column=sample_column
    )
    
    # Calculate the expression distance matrix
    expression_matrix = calculate_sample_distances_cell_expression(
        adata=adata,
        output_dir=output_dir,
        cell_type_column=cell_type_column,
        sample_column=sample_column
    )

    calculate_sample_distances_weighted_expression (
        adata=adata,
        output_dir=output_dir,
        cell_type_column=cell_type_column,
        sample_column=sample_column
    )
    
    # Ensure that both matrices have the same order of samples
    if not proportion_matrix.index.equals(expression_matrix.index):
        raise ValueError("The indices of proportion_matrix and expression_matrix do not match.")
    if not proportion_matrix.columns.equals(expression_matrix.columns):
        raise ValueError("The columns of proportion_matrix and expression_matrix do not match.")
    
    # Combine the two distance matrices with the specified weights
    combined_matrix = (proportion_weight * proportion_matrix) + (expression_weight * expression_matrix)
    
    # Optionally, save the combined distance matrix to a CSV file
    combined_matrix_path = os.path.join(output_dir, 'combined_distance_matrix.csv')
    combined_matrix.to_csv(combined_matrix_path)
    print(f"Combined distance matrix saved to {combined_matrix_path}")

    heatmap_path = os.path.join(output_dir, 'sample_distance_heatmap.pdf')
    # Convert the square distance matrix to condensed form for linkage
    condensed_distances = squareform(combined_matrix.values)
    # Compute the linkage matrix using the condensed distance matrix
    linkage_matrix = linkage(condensed_distances, method='average')
    # Generate the clustermap
    sns.clustermap(
        combined_matrix,
        cmap='viridis',
        linewidths=0.5,
        annot=True,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix
    )
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Sample distance heatmap saved to {heatmap_path}")

    return combined_matrix