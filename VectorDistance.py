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
    distanceCheck(distance_matrix_path, 'cell_proportion', method, summary_csv_path, adata)
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
    distanceCheck(distance_matrix_path, 'gene_expression', method, summary_csv_path, adata)
    print(f"Sample distance gene expresission matrix saved to {distance_matrix_path}")

    # generate a heatmap for sample distance
    heatmap_path = os.path.join(output_dir, 'sample_distance_gene_expression_heatmap.pdf')
    visualizeDistanceMatrix(distance_df, heatmap_path)
    visualizeGroupRelationship(distance_df, outputDir=output_dir, heatmap_path=os.path.join(output_dir, 'sample_gene_expression_relationship.pdf'))
    print("Distance matrix based on gene expression per sample saved to 'distance_matrix_gene_expression.csv'.")
    return distance_df


def calculate_sample_distances_gene_pseudobulk(
    adata: AnnData,
    output_dir: str,
    method: str,
    summary_csv_path: str,
    sample_column: str = 'sample',
    celltype_column: str = 'cell_type',
    normalize: bool = True,
    log_transform: bool = True
) -> pd.DataFrame:
    """
    1) Compute the average gene expression for each (sample, cell_type).
    2) Concatenate these gene expressions by unstacking/pivoting so each sample
       is a single row containing (gene x cell_type) columns.
    3) Compute sample-sample distance using the chosen method (e.g., euclidean).
    """

    # Create sub-directory for outputs
    output_dir = os.path.join(output_dir, 'gene_pseudobulk')
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------
    # 1) Build a dataframe with expression + sample + cell_type
    # ---------------------------
    # adata.to_df() returns a DataFrame of shape [n_cells, n_genes],
    # with column names as gene symbols/features in adata.var_names.
    expr_df = adata.to_df().copy()

    # Add columns for sample & cell_type from .obs
    expr_df[sample_column] = adata.obs[sample_column].values
    expr_df[celltype_column] = adata.obs[celltype_column].values

    # ---------------------------
    # 2) Group by (sample, cell_type) and compute mean
    # ---------------------------
    # This yields a multi-index DataFrame: Index levels = sample, cell_type
    avg_expr = expr_df.groupby([sample_column, celltype_column]).mean()

    # ---------------------------
    # 3) "Concatenate" these means into a single vector for each sample
    #    by unstacking cell_type. That way each row = 1 sample
    #    and columns = gene x cell_type
    # ---------------------------
    avg_expr = avg_expr.unstack(level=celltype_column).fillna(0)

    # Flatten the multi-level columns: (gene, cell_type) -> gene_celltype
    avg_expr.columns = [f"{gene}_{ct}" for gene, ct in avg_expr.columns]

    # Optionally, save the average expression to CSV
    avg_expr_path = os.path.join(output_dir, 'average_gene_pseudobulk_per_sample.csv')
    avg_expr.to_csv(avg_expr_path)
    print(f"Average gene pseudobulk per sample saved to {avg_expr_path}")

    # ---------------------------
    # 4) Compute the sample distance
    # ---------------------------
    # avg_expr.index are the samples, each row is the concatenated expression vector
    distance_matrix = pdist(avg_expr.values, metric=method)

    # Convert to a square distance matrix
    distance_df = pd.DataFrame(
        squareform(distance_matrix),
        index=avg_expr.index,
        columns=avg_expr.index
    )

    # Save distance matrix
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_gene_pseudobulk.csv')
    distance_df.to_csv(distance_matrix_path)
    distanceCheck(distance_matrix_path, 'gene_pseudobulk', method, summary_csv_path, adata)
    print(f"Sample distance matrix saved to {distance_matrix_path}")

    return distance_df

def calculate_sample_distances_pca(
    adata: AnnData,
    output_dir: str,
    method: str = 'euclidean',
    summary_csv_path: Optional[str] = None,
    sample_column: str = 'sample',
    normalize: bool = True,
    log_transform: bool = True
) -> pd.DataFrame:
    """
    Calculate a distance matrix based on precomputed PCA-transformed data for each sample.
    
    Parameters:
    - adata: AnnData object containing single-cell data with PCA already computed.
    - output_dir: Directory to save the distance matrix and related plots.
    - method: Distance metric to use (e.g., 'euclidean', 'cosine').
    - summary_csv_path: Optional path to save a summary CSV of distance metrics.
    - sample_column: Column in adata.obs indicating sample identifiers.
    - normalize: Whether to perform normalization (e.g., total count normalization).
    - log_transform: Whether to perform log1p transformation.
    
    Returns:
    - distance_df: DataFrame containing the pairwise distances between samples.
    
    Raises:
    - KeyError: If 'X_pca_harmony' is not found in adata.obsm.
    - ValueError: If the specified sample_column does not exist in adata.obs.
    """
    
    # Validate input data
    if 'X_pca_harmony' not in adata.obsm:
        print("ERROR: PCA data 'X_pca_harmony' not found in adata.obsm.")
        raise KeyError("PCA data 'X_pca_harmony' not found in adata.obsm.")

    if sample_column not in adata.obs.columns:
        print(f"ERROR: Sample column '{sample_column}' not found in adata.obs.")
        raise ValueError(f"Sample column '{sample_column}' not found in adata.obs.")

    # Create output directory
    pca_output_dir = os.path.join(output_dir, 'pca_distance_harmony')
    os.makedirs(pca_output_dir, exist_ok=True)
    print(f"Output directory created at: {pca_output_dir}")

    # Extract PCA data
    print("Extracting PCA data from adata.obsm['X_pca_harmony']...")
    pca_data = adata.obsm['X_pca_harmony']
    if not isinstance(pca_data, np.ndarray):
        pca_data = np.array(pca_data)
    pca_df = pd.DataFrame(pca_data, index=adata.obs_names,
                          columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
    print(f"PCA data extracted with shape: {pca_df.shape}")

    # Associate samples with PCA data
    pca_df['sample'] = adata.obs[sample_column].values

    # Compute average PCA components per sample
    print(f"Calculating average PCA components per sample based on column '{sample_column}'...")
    average_pca = pca_df.groupby('sample').mean()
    average_pca.fillna(0, inplace=True)
    print(f"Average PCA components per sample calculated with shape: {average_pca.shape}")

    # Save the average PCA components to a CSV file
    average_pca_csv_path = os.path.join(pca_output_dir, 'average_pca_per_sample.csv')
    average_pca.to_csv(average_pca_csv_path)
    print(f"Average PCA components per sample saved to: {average_pca_csv_path}")

    # Calculate pairwise distance matrix
    print(f"Calculating pairwise distances using '{method}' metric...")
    distance_matrix = pdist(average_pca.values, metric=method)
    distance_df = pd.DataFrame(
        squareform(distance_matrix),
        index=average_pca.index,
        columns=average_pca.index
    )
    print("Pairwise distance matrix calculated.")

    # Optional: Log transform and normalize the distance matrix
    if log_transform:
        print("Applying log1p transformation to the distance matrix...")
        distance_df = np.log1p(np.maximum(distance_df, 0))
    if normalize:
        print("Normalizing the distance matrix...")
        max_val = distance_df.max().max()
        if max_val > 0:
            distance_df = distance_df / max_val
        else:
            print("WARNING: Maximum value of distance matrix is 0. Skipping normalization.")

    # Save the distance matrix
    distance_matrix_path = os.path.join(pca_output_dir, 'distance_matrix_pca_harmony.csv')
    distance_df.to_csv(distance_matrix_path)
    print(f"Sample distance PCA matrix saved to: {distance_matrix_path}")

    # Perform distance checks or summaries if required
    if summary_csv_path:
        try:
            distanceCheck(distance_matrix_path, 'pca_harmony', method, summary_csv_path, adata)
            print(f"Distance checks saved to: {summary_csv_path}")
        except Exception as e:
            print(f"ERROR: Failed to perform distance checks: {e}")

    # Generate a heatmap for sample distances
    heatmap_path = os.path.join(pca_output_dir, 'sample_distance_pca_harmony_heatmap.pdf')
    try:
        visualizeDistanceMatrix(distance_df, heatmap_path)
        print(f"PCA distance heatmap saved to: {heatmap_path}")
    except Exception as e:
        print(f"ERROR: Failed to generate distance heatmap: {e}")

    # Visualize group relationships
    relationship_path = os.path.join(pca_output_dir, 'sample_pca_harmony_relationship.pdf')
    try:
        visualizeGroupRelationship(distance_df, outputDir=pca_output_dir, heatmap_path=heatmap_path)
        print(f"PCA group relationship plot saved to: {relationship_path}")
    except Exception as e:
        print(f"ERROR: Failed to visualize group relationships: {e}")

    print("Distance calculation based on PCA-harmony completed successfully.")
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
    distanceCheck(distance_matrix_path, 'weighted_expression', method, summary_csv_path, adata)
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
    adata = adata[:, adata.var['highly_variable']].copy()
    calculate_sample_distances_cell_proportion(adata, output_dir, method, summary_csv_path)
    calculate_sample_distances_gene_expression(adata, output_dir, method, summary_csv_path)
    calculate_sample_distances_pca(adata, output_dir, method, summary_csv_path)
    calculate_sample_distances_gene_pseudobulk(adata, output_dir, method, summary_csv_path)
    # calculate_sample_distances_weighted_expression(adata, output_dir, method, summary_csv_path)
