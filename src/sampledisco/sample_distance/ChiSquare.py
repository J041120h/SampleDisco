import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import warnings
from typing import Optional
from anndata._core.aligned_df import ImplicitModificationWarning

import sys
from sampledisco.visualization.visualization_helper import visualizeDistanceMatrix
from sampledisco.sample_distance.distance_test import distanceCheck

warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

def calculate_sample_distances_cell_proportion_chi_square(
    adata: AnnData,
    output_dir: str,
    cell_type_column: str = 'cell_type',
    sample_column: str = 'sample',
    summary_csv_path: Optional[str] = None,
    pseudobulk_adata: AnnData = None,
    grouping_columns: list = None
) -> pd.DataFrame:
    """Compute pairwise Chi-Square distances between samples on cell-type proportions.

    Formula: 0.5 * sum((p_i - p_j)^2 / (p_i + p_j)) with eps=1e-10 to avoid zero
    denominators. Normalizes the matrix to [0, 1].
    Returns a symmetric distance DataFrame (samples × samples).
    """
    proportion_output_dir = os.path.join(output_dir, 'proportion_DR_distance')
    os.makedirs(proportion_output_dir, exist_ok=True)

    samples = adata.obs[sample_column].unique()
    cell_types = adata.obs[cell_type_column].unique()

    proportions = pd.DataFrame(0, index=samples, columns=cell_types, dtype=np.float64)

    for sample in samples:
        sample_data = adata.obs[adata.obs[sample_column] == sample]
        total_cells = sample_data.shape[0]
        counts = sample_data[cell_type_column].value_counts()
        proportions.loc[sample, counts.index] = counts.values / total_cells

    proportions.to_csv(os.path.join(proportion_output_dir, 'proportion_DR_coordinates.csv'))

    num_samples = len(samples)
    sample_distance_matrix = pd.DataFrame(0, index=samples, columns=samples, dtype=np.float64)

    epsilon = 1e-10  # prevents zero denominator in Chi-Square formula
    proportions_safe = proportions.replace(0, epsilon)

    for i, sample_i in enumerate(samples):
        for j, sample_j in enumerate(samples):
            if i < j:
                hist_i = proportions_safe.loc[sample_i].values
                hist_j = proportions_safe.loc[sample_j].values

                chi_square = 0.5 * np.sum(((hist_i - hist_j) ** 2) / (hist_i + hist_j))
                sample_distance_matrix.loc[sample_i, sample_j] = chi_square
                sample_distance_matrix.loc[sample_j, sample_i] = chi_square

    if sample_distance_matrix.max().max() > 0:
        sample_distance_matrix = sample_distance_matrix / sample_distance_matrix.max().max()

    distance_matrix_path = os.path.join(proportion_output_dir, 'distance_matrix_proportion_DR.csv')
    sample_distance_matrix.to_csv(distance_matrix_path)

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

    plot_cell_type_abundances(proportions, proportion_output_dir)
    print(f"Cell type distribution in Sample saved to {proportion_output_dir}")

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
    """Compute Chi-Square proportion distances and save statistics summary.

    Delegates to `calculate_sample_distances_cell_proportion_chi_square`,
    then writes distance_statistics_summary_chi_square.csv under output_dir.
    Returns the symmetric distance DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)

    proportion_matrix = calculate_sample_distances_cell_proportion_chi_square(
        adata=adata,
        output_dir=output_dir,
        cell_type_column=cell_type_column,
        sample_column=sample_column,
        summary_csv_path=summary_csv_path,
        pseudobulk_adata=pseudobulk_adata,
        grouping_columns=grouping_columns
    )

    try:
        dist_values = proportion_matrix.values[np.triu_indices_from(proportion_matrix.values, k=1)]
        summary_stats = {
            "proportion_DR_mean": np.mean(dist_values),
            "proportion_DR_std": np.std(dist_values),
            "proportion_DR_min": np.min(dist_values),
            "proportion_DR_max": np.max(dist_values),
            "proportion_DR_median": np.median(dist_values)
        }
        
        stats_file = os.path.join(output_dir, 'distance_statistics_summary_chi_square.csv')
        stats_df = pd.DataFrame([summary_stats])
        stats_df.to_csv(stats_file)
        print(f"Distance statistics summary saved to: {stats_file}")
        
    except Exception as e:
        print(f"Warning: Failed to create distance statistics summary: {e}")

    return proportion_matrix