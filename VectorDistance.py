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
    pseudobulk: dict,
    grouping_columns: list
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
    print("Correctly calculate raw distance")

    # Example post-processing: log1p transform & scale to [0, 1]
    distance_df = np.log1p(np.maximum(distance_df, 0))
    distance_df = distance_df / distance_df.max().max()

    # Save distance matrix to CSV
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_proportion.csv')
    distance_df.to_csv(distance_matrix_path)

    # If you have a custom distanceCheck function for logging
    distanceCheck(distance_matrix_path, 'cell_proportion', method, summary_csv_path, adata, grouping_columns = grouping_columns)

    # Visualizations
    visualizeDistanceMatrix(distance_df, os.path.join(output_dir, 'sample_distance_proportion_heatmap.pdf'))
    visualizeGroupRelationship(
        distance_df,
        outputDir=output_dir,
        adata = adata,
        grouping_columns = grouping_columns
    )

    print(f"Cell proportion-based sample distance matrix saved to: {distance_matrix_path}")
    return distance_df

def calculate_sample_distances_pca(
    adata: AnnData,
    output_dir: str,
    method: str = 'euclidean',
    summary_csv_path: Optional[str] = None,
    sample_column: str = 'sample',
    grouping_columns: Optional[list] = None
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
    distanceCheck(distance_matrix_path, 'pca_harmony', method, summary_csv_path, adata, grouping_columns = grouping_columns)
    visualizeDistanceMatrix(distance_df, os.path.join(output_dir, 'sample_distance_pca_harmony_heatmap.pdf'))
    visualizeGroupRelationship(distance_df, outputDir=output_dir, adata = adata, heatmap_path=os.path.join(output_dir, 'sample_pca_harmony_relationship.pdf'))

    print(f"PCA-based distance matrix saved to: {output_dir}")
    return distance_df

def calculate_sample_distances_DR(
    adata: AnnData,
    DR: pd.DataFrame,
    output_dir: str,
    method: str = 'euclidean',
    summary_csv_path: Optional[str] = None,
    sample_column: str = 'sample',
    grouping_columns: Optional[list] = None,
    dr_name: str = 'DR'
) -> pd.DataFrame:
    """
    Compute a sample distance matrix using dimensionality reduction-transformed data.
    
    Parameters:
    - adata: AnnData object containing sample metadata.
    - DR: pd.DataFrame with dimensionality reduction data (samples x components).
           Index should match sample names in adata.obs[sample_column].
    - output_dir: Directory for output files.
    - method: Distance metric for pdist (default: 'euclidean').
    - summary_csv_path: Path for logging.
    - sample_column: Column in adata.obs indicating sample IDs.
    - grouping_columns: Optional list of columns for grouping analysis.
    - dr_name: Name of the dimensionality reduction method for file naming.
    
    Returns:
    - distance_df: Sample distance matrix (samples x samples).
    """
    
    # Validate inputs
    if DR is None or DR.empty:
        raise ValueError("DR DataFrame is empty or None.")
    
    if sample_column not in adata.obs.columns:
        raise KeyError(f"Sample column '{sample_column}' not found in adata.obs.")
    
    # Get unique samples from adata
    unique_samples = adata.obs[sample_column].unique()
    
    # Handle case sensitivity issues between DR index and sample names
    # Check if samples exist in DR index (exact match first)
    missing_samples = [s for s in unique_samples if s not in DR.index]
    
    # If we have missing samples, try case-insensitive matching
    if missing_samples:
        # Create mapping from lowercase DR index to original DR index
        dr_index_lower_to_original = {idx.lower(): idx for idx in DR.index}
        
        # Create mapping from original samples to their lowercase versions
        sample_to_lower = {sample: sample.lower() for sample in unique_samples}
        
        # Check if all samples exist in lowercase in DR index
        missing_after_case_check = []
        for sample in unique_samples:
            if sample_to_lower[sample] not in dr_index_lower_to_original:
                missing_after_case_check.append(sample)
        
        if missing_after_case_check:
            raise ValueError(f"DR DataFrame missing data for samples: {missing_after_case_check}")
        
        # Create a mapping from original sample names to DR index names
        sample_to_dr_index = {}
        for sample in unique_samples:
            lowercase_sample = sample_to_lower[sample]
            if lowercase_sample in dr_index_lower_to_original:
                sample_to_dr_index[sample] = dr_index_lower_to_original[lowercase_sample]
            else:
                sample_to_dr_index[sample] = sample  # fallback to original
        
        # Get the corresponding DR index names for our samples
        dr_sample_names = [sample_to_dr_index[sample] for sample in unique_samples]
    else:
        # No missing samples, use original sample names
        dr_sample_names = unique_samples
    
    # Create output directory
    output_dir = os.path.join(output_dir, "Sample", f'{dr_name}_distance')
    os.makedirs(output_dir, exist_ok=True)
    
    # Use DR data directly (it should already be averaged per sample)
    average_dr = DR.loc[dr_sample_names].fillna(0)
    
    # Rename the index back to original sample names for consistency
    if missing_samples:  # Only if we had case issues
        average_dr.index = unique_samples
    
    # Save DR data
    average_dr.to_csv(os.path.join(output_dir, f'average_{dr_name}_per_sample.csv'))
    
    # Compute distances
    distance_matrix = pdist(average_dr.values, metric=method)
    distance_df = pd.DataFrame(
        squareform(distance_matrix), 
        index=average_dr.index, 
        columns=average_dr.index
    )
    
    # Apply log transformation and normalization
    distance_df = np.log1p(np.maximum(distance_df, 0)) / distance_df.max().max()
    
    # Save distance matrix
    distance_matrix_path = os.path.join(output_dir, f'distance_matrix_{dr_name}.csv')
    distance_df.to_csv(distance_matrix_path)
    
    # Perform quality checks and visualizations
    if summary_csv_path is not None:
        distanceCheck(
            distance_matrix_path, 
            dr_name, 
            method, 
            summary_csv_path, 
            adata, 
            grouping_columns=grouping_columns
        )
    
    visualizeDistanceMatrix(
        distance_df, 
        os.path.join(output_dir, f'sample_distance_{dr_name}_heatmap.pdf')
    )
    
    visualizeGroupRelationship(
        distance_df, 
        outputDir=output_dir, 
        adata=adata, 
        grouping_columns = grouping_columns,
        heatmap_path=os.path.join(output_dir, f'sample_{dr_name}_relationship.pdf')
    )
    
    print(f"{dr_name}-based distance matrix saved to: {output_dir}")
    return distance_df

def calculate_sample_distances_gene_pseudobulk(
    adata: AnnData,
    output_dir: str,
    method: str,
    summary_csv_path: str,
    pseudobulk: dict,
    sample_column: str = 'sample',
    grouping_columns: str = 'cell_type'
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
    distanceCheck(distance_matrix_path, 'cell_expression', method, summary_csv_path, adata, grouping_columns = grouping_columns)

    # Visualizations
    visualizeDistanceMatrix(distance_df, os.path.join(output_dir, 'sample_distance_expression_heatmap.pdf'))
    visualizeGroupRelationship(
        distance_df,
        outputDir=output_dir,
        adata = adata,
        grouping_columns = grouping_columns
    )

    print(f"Cell expression-based sample distance matrix saved to: {distance_matrix_path}")
    return distance_df

def sample_distance(
    adata: AnnData,
    output_dir: str,
    method: str,
    summary_csv_path: str = "/users/harry/desktop/GenoDistance/result/summary.csv",
    pseudobulk: dict = None,
    sample_column: str = 'sample',
    grouping_columns = ["sev.level"]
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
    calculate_sample_distances_cell_proportion(adata, output_dir, method, summary_csv_path, pseudobulk, grouping_columns)
    # calculate_sample_distances_gene_expression(adata, output_dir, method, summary_csv_path, sample_column)
    # calculate_sample_distances_pca(adata, output_dir, method, summary_csv_path, sample_column)
    calculate_sample_distances_gene_pseudobulk(adata, output_dir, method, summary_csv_path, pseudobulk, sample_column, grouping_columns)

    print("Sample distance computations completed.")
