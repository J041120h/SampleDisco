import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

def load_severity_levels(summary_sample_csv_path: str, sample_index: pd.Index) -> np.ndarray:
    """
    Load severity levels from a CSV file and align them with the provided sample index.

    Parameters:
    - summary_sample_csv_path (str): Path to the CSV file containing severity levels.
    - sample_index (pd.Index): Index of the samples in the provided DataFrame.

    Returns:
    - sev_levels (np.ndarray): Aligned 2D array of severity levels for the samples.
    """
    # Load CSV
    summary_df = pd.read_csv(summary_sample_csv_path)

    # Ensure required columns exist
    if 'sample' not in summary_df.columns or 'sev.level' not in summary_df.columns:
        raise ValueError("CSV must contain 'sample' and 'sev.level' columns.")

    # Convert severity levels to numeric
    summary_df['sev.level'] = pd.to_numeric(summary_df['sev.level'], errors='coerce')

    # Map severity levels for the provided samples
    sample_to_sev = summary_df.set_index('sample')['sev.level'].to_dict()
    sev_levels = np.array([sample_to_sev.get(sample, np.nan) for sample in sample_index])

    # Handle missing severity levels
    missing_samples = np.isnan(sev_levels).sum()
    if missing_samples > 0:
        print(f"\nWarning: {missing_samples} samples have missing severity levels! These will be imputed with the mean.\n")
        sev_levels[np.isnan(sev_levels)] = np.nanmean(sev_levels)  # Impute missing values

    # Add small random noise to prevent singularity issues
    noise = np.random.normal(scale=0.01, size=sev_levels.shape)
    sev_levels_2D = np.hstack([sev_levels.reshape(-1, 1), sev_levels.reshape(-1, 1) + noise])

    print("\nGenerated 2D sev.level values:\n", sev_levels_2D)

    return sev_levels_2D

def cca_trajectory_analysis_expression(cell_expression_corrected_df: pd.DataFrame, summary_sample_csv_path: str):
    """
    Performs CCA trajectory analysis on batch-corrected cell expression data to find
    the direction most related to severity level.

    Parameters:
    - cell_expression_corrected_df (pd.DataFrame): DataFrame with samples as rows and genes as columns.
    - summary_sample_csv_path (str): Path to the CSV file containing severity levels.

    Returns:
    - cca (CCA): Trained CCA model.
    - transformed_X (np.ndarray): 2D projected corrected expression data in the direction of 'sev.level'.
    - sev_levels (np.ndarray): Aligned 2D severity levels.
    """

    sev_levels = load_severity_levels(summary_sample_csv_path, cell_expression_corrected_df.index)

    X = cell_expression_corrected_df.to_numpy()

    # Ensure matching number of samples
    assert X.shape[0] == sev_levels.shape[0], "Mismatch between sample numbers in data and severity labels"

    # Use `n_components=2` since `sev_levels` is now 2D
    cca = CCA(n_components=2)
    transformed_X, _ = cca.fit_transform(X, sev_levels)

    return cca, transformed_X, sev_levels

def cca_trajectory_analysis_proportion(cell_proportion_df: pd.DataFrame, summary_sample_csv_path: str):
    """
    Performs CCA trajectory analysis on cell proportion data to find the direction
    most related to severity level.

    Parameters:
    - cell_proportion_df (pd.DataFrame): DataFrame with samples as rows and cell types as columns.
    - summary_sample_csv_path (str): Path to the CSV file containing severity levels.

    Returns:
    - cca (CCA): Trained CCA model.
    - transformed_X (np.ndarray): 2D projected cell proportion data in the direction of 'sev.level'.
    - sev_levels (np.ndarray): Aligned 2D severity levels.
    """

    sev_levels = load_severity_levels(summary_sample_csv_path, cell_proportion_df.index)

    X = cell_proportion_df.to_numpy()

    # Ensure matching number of samples
    assert X.shape[0] == sev_levels.shape[0], "Mismatch between sample numbers in data and severity labels"

    # Use `n_components=2` since `sev_levels` is now 2D
    cca = CCA(n_components=2)
    transformed_X, _ = cca.fit_transform(X, sev_levels)

    return cca, transformed_X, sev_levels

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

def project_cca_on_pca(cca, pca, transformed_cca):
    """
    Projects the first CCA component onto PCA space.

    Parameters:
    - cca: Trained CCA model.
    - pca: Trained PCA model.
    - transformed_cca: The transformed data in CCA space.

    Returns:
    - cca_pca_coords: Projected CCA trajectory onto PCA space (2D).
    """
    # Ensure transformed_cca is 2D (n_samples, 2)
    if transformed_cca.shape[1] != 2:
        raise ValueError(f"transformed_cca should have shape (n_samples, 2), got {transformed_cca.shape}")

    # Project CCA trajectory onto PCA space
    cca_pca_coords_full = pca.transform(transformed_cca)

    # Extract only the first two PCA components (PC1 and PC2)
    cca_pca_coords = cca_pca_coords_full[:, :2]  # Keep only the first two principal components

    return cca_pca_coords



def plot_pca_with_cca(
    adata: AnnData, 
    cca, 
    transformed_cca, 
    sev_levels, 
    output_dir: str, 
    pca_key: str, 
    title: str
):
    """
    Visualizes PCA with the projected CCA trajectory.

    Parameters:
    - adata: AnnData object containing PCA results.
    - cca: Trained CCA model.
    - transformed_cca: 2D projected CCA coordinates.
    - sev_levels: Severity levels.
    - output_dir: Directory to save the plot.
    - pca_key: Key in `adata.uns` where PCA results are stored.
    - title: Title of the plot.
    """
    if pca_key not in adata.uns:
        raise KeyError(f"Missing '{pca_key}' in adata.uns. Ensure PCA was run.")

    output_dir = os.path.join(output_dir, 'harmony')
    os.makedirs(output_dir, exist_ok=True)

    # Extract PCA coordinates
    pca_coords = adata.uns[pca_key]
    samples = adata.obs['sample'].unique()
    pca_df = pd.DataFrame(pca_coords[:, :2], index=samples, columns=['PC1', 'PC2'])

    # Normalize severity levels
    norm_severity = (sev_levels - sev_levels.min()) / (sev_levels.max() - sev_levels.min())

    # Compute PCA model for reference
    pca = PCA(n_components=2)
    pca.fit(pca_coords)  # Fit PCA to original PCA data

    # Project CCA onto PCA space
    cca_pca_coords = project_cca_on_pca(cca, pca, transformed_cca)

    # Compute centroid for arrow start
    centroid = pca_df[['PC1', 'PC2']].mean().values

    # Compute CCA direction in PCA space
    cca_vector = cca_pca_coords.mean(axis=0) - centroid  # Direction from centroid

    # Plot PCA
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=norm_severity, cmap='viridis_r', s=80, alpha=0.8, edgecolors='k')
    plt.colorbar(sc, label='Severity Level')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    # Plot CCA trajectory as an arrow
    plt.arrow(centroid[0], centroid[1], cca_vector[0], cca_vector[1], color='red', width=0.1, head_width=0.3, label="CCA Direction")
    plt.legend()

    # Save plot
    plot_path = os.path.join(output_dir, f"{pca_key}_with_cca.pdf")
    plt.savefig(plot_path)
    print(f"PCA plot with CCA direction saved to: {plot_path}")

def plot_cell_type_proportions_pca_with_cca(
    adata: AnnData, 
    cca, 
    transformed_cca, 
    sev_levels, 
    output_dir: str
):
    """
    Wrapper for PCA with CCA trajectory visualization on cell type proportions.
    """
    plot_pca_with_cca(
        adata, cca, transformed_cca, sev_levels, output_dir, 
        "X_pca_proportion", "2D PCA of Cell Type Proportions (CCA Overlay)"
    )

def plot_pseudobulk_pca_with_cca(
    adata: AnnData, 
    cca, 
    transformed_cca, 
    sev_levels, 
    output_dir: str
):
    """
    Wrapper for PCA with CCA trajectory visualization on pseudobulk expression.
    """
    plot_pca_with_cca(
        adata, cca, transformed_cca, sev_levels, output_dir, 
        "X_pca_expression", "2D PCA of HVG Expression (CCA Overlay)"
    )