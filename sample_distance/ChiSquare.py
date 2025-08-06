import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import warnings
from anndata._core.aligned_df import ImplicitModificationWarning
from visualization_helper import plot_cell_type_abundances, visualizeDistanceMatrix
from distance_test import distanceCheck

warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

def calculate_sample_distances_cell_proportion_chi_square(
    adata: AnnData,
    output_dir: str,
    cell_type_column: str = 'cell_type',
    sample_column: str = 'sample',
    summary_csv_path: str = "/users/harry/desktop/GenoDistance/result/summary.csv",
    pseudobulk_adata: AnnData = None,
    grouping_columns: list = None
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
        Path to save the summary CSV file.
    pseudobulk_adata : AnnData, optional
        Pseudobulk AnnData object where observations are samples (not cells). 
        If provided, this will be used for sample metadata in distanceCheck.
    grouping_columns : list, optional
        List of columns for grouping analysis.

    Returns:
    -------
    sample_distance_matrix : pandas.DataFrame
        A symmetric matrix of distances between samples.
    """

    # Create standardized output directory structure: proportion_DR_distance
    proportion_output_dir = os.path.join(output_dir, 'proportion_DR_distance')
    os.makedirs(proportion_output_dir, exist_ok=True)

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

    # Save proportions (equivalent to coordinates in vector distance)
    proportions.to_csv(os.path.join(proportion_output_dir, 'proportion_DR_coordinates.csv'))

    # 2. Compute Chi-Square distance between each pair of samples
    num_samples = len(samples)
    sample_distance_matrix = pd.DataFrame(0, index=samples, columns=samples, dtype=np.float64)

    # Replace zeros with a small epsilon to avoid division by zero in denominator
    epsilon = 1e-10
    proportions_safe = proportions.replace(0, epsilon)

    for i, sample_i in enumerate(samples):
        for j, sample_j in enumerate(samples):
            if i < j:
                hist_i = proportions_safe.loc[sample_i].values
                hist_j = proportions_safe.loc[sample_j].values

                # Compute Chi-Square distance
                chi_square = 0.5 * np.sum(((hist_i - hist_j) ** 2) / (hist_i + hist_j))
                sample_distance_matrix.loc[sample_i, sample_j] = chi_square
                sample_distance_matrix.loc[sample_j, sample_i] = chi_square

    # Apply log transformation and normalization (same as vector distance)
    sample_distance_matrix = np.log1p(np.maximum(sample_distance_matrix, 0))
    if sample_distance_matrix.max().max() > 0:  # Avoid division by zero
        sample_distance_matrix = sample_distance_matrix / sample_distance_matrix.max().max()

    # Save the distance matrix with standardized naming
    distance_matrix_path = os.path.join(proportion_output_dir, 'distance_matrix_proportion_DR.csv')
    sample_distance_matrix.to_csv(distance_matrix_path)
    
    # Perform distance check using the improved distanceCheck function
    try:
        score = distanceCheck(
            distance_df=sample_distance_matrix,
            row="proportion_DR",
            method="chi_square",
            output_dir=proportion_output_dir,
            adata=pseudobulk_adata,
            grouping_columns=grouping_columns,
            summary_csv_path=summary_csv_path
        )
        print(f"Distance check completed for proportion_DR: score = {score:.6f}")
    except Exception as e:
        print(f"Warning: Distance check failed for proportion_DR: {e}")

    print(f"Sample distance proportion matrix saved to {distance_matrix_path}")

    # Save the cell type distribution map
    plot_cell_type_abundances(proportions, proportion_output_dir)
    print(f"Cell type distribution in Sample saved to {proportion_output_dir}")

    # Generate visualizations with standardized naming
    try:
        heatmap_path = os.path.join(proportion_output_dir, 'sample_distance_proportion_DR_heatmap.pdf')
        visualizeDistanceMatrix(sample_distance_matrix, heatmap_path)
    except Exception as e:
        print(f"Warning: Failed to create distance heatmap for proportion_DR: {e}")

    print(f"Chi-square-based distance matrix saved to: {proportion_output_dir}")
    return sample_distance_matrix

def chi_square_distance(
    adata: AnnData,
    output_dir: str,
    summary_csv_path: str,
    cell_type_column: str = 'cell_type',
    sample_column: str = 'sample',
    pseudobulk_adata: AnnData = None,
    grouping_columns: list = None
) -> pd.DataFrame:
    """
    Calculate combined distances between samples based on cell type proportions using Chi-Square Distance.

    Parameters:
    ----------
    adata : AnnData
        The integrated single-cell dataset obtained from the previous analysis.
    output_dir : str
        Directory to save the output files. Should already include method name (chi_square).
    summary_csv_path : str
        Path to save the summary CSV file.
    cell_type_column : str, optional
        Column name in `adata.obs` that contains the cell type assignments (default: 'cell_type').
    sample_column : str, optional
        Column name in `adata.obs` that contains the sample information (default: 'sample').
    pseudobulk_adata : AnnData, optional
        Pseudobulk AnnData object where observations are samples (not cells). 
    grouping_columns : list, optional
        List of columns for grouping analysis.

    Returns:
    -------
    proportion_matrix : pd.DataFrame
        A symmetric matrix of Chi-Square distances between samples.
    """

    # Check if output directory exists and create if necessary
    os.makedirs(output_dir, exist_ok=True)

    # Calculate the proportion distance matrix using Chi-Square distance
    proportion_matrix = calculate_sample_distances_cell_proportion_chi_square(
        adata=adata,
        output_dir=output_dir,
        cell_type_column=cell_type_column,
        sample_column=sample_column,
        summary_csv_path=summary_csv_path,
        pseudobulk_adata=pseudobulk_adata,
        grouping_columns=grouping_columns
    )

    # Create basic statistics summary (matching vector distance structure)
    try:
        dist_values = proportion_matrix.values[np.triu_indices_from(proportion_matrix.values, k=1)]
        summary_stats = {
            "proportion_DR_mean": np.mean(dist_values),
            "proportion_DR_std": np.std(dist_values),
            "proportion_DR_min": np.min(dist_values),
            "proportion_DR_max": np.max(dist_values),
            "proportion_DR_median": np.median(dist_values)
        }
        
        # Save basic statistics summary
        stats_file = os.path.join(output_dir, 'distance_statistics_summary_chi_square.csv')
        stats_df = pd.DataFrame([summary_stats])
        stats_df.to_csv(stats_file)
        print(f"Distance statistics summary saved to: {stats_file}")
        
    except Exception as e:
        print(f"Warning: Failed to create distance statistics summary: {e}")

    return proportion_matrix