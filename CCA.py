import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def load_severity_levels(summary_sample_csv_path: str, sample_index: pd.Index) -> np.ndarray:
    """
    Load severity levels from a CSV file and align them with the provided sample index.

    Parameters:
    - summary_sample_csv_path (str): Path to the CSV file containing severity levels.
    - sample_index (pd.Index): Index of the samples in the provided DataFrame.

    Returns:
    - sev_levels (np.ndarray): Aligned 1D array of severity levels for the samples.
    """
    summary_df = pd.read_csv(summary_sample_csv_path)

    if 'sample' not in summary_df.columns or 'sev.level' not in summary_df.columns:
        raise ValueError("CSV must contain 'sample' and 'sev.level' columns.")

    summary_df['sev.level'] = pd.to_numeric(summary_df['sev.level'], errors='coerce')
    sample_to_sev = summary_df.set_index('sample')['sev.level'].to_dict()
    sev_levels = np.array([sample_to_sev.get(sample, np.nan) for sample in sample_index])

    missing_samples = np.isnan(sev_levels).sum()
    if missing_samples > 0:
        print(f"\nWarning: {missing_samples} samples have missing severity levels! Imputing with mean.\n")
        sev_levels[np.isnan(sev_levels)] = np.nanmean(sev_levels)

    return sev_levels.reshape(-1, 1)  # Return as 2D array for CCA compatibility


def cca_trajectory_analysis_proportion(proprtion: pd.DataFrame, summary_sample_csv_path: str):
    """
    Performs CCA trajectory analysis on batch-corrected cell expression data.

    Parameters:
    - cell_expression_corrected_df (pd.DataFrame): DataFrame with samples as rows and genes as columns.
    - summary_sample_csv_path (str): Path to the CSV file containing severity levels.

    Returns:
    - cca (CCA): Trained CCA model.
    - transformed_CA1 (np.ndarray): 1D projected expression data in the direction of 'sev.level'.
    - sev_levels (np.ndarray): Aligned 1D severity levels.
    """
    sev_levels = load_severity_levels(summary_sample_csv_path, proprtion.index)
    X = proprtion.to_numpy()

    assert X.shape[0] == sev_levels.shape[0], "Mismatch between sample numbers in data and severity labels"

    cca = CCA(n_components=1)  # Use only 1 component
    transformed_CA1, _ = cca.fit_transform(X, sev_levels)

    return cca, transformed_CA1.flatten(), sev_levels.flatten()  # Return 1D CA1


def run_pca_proportion(adata: sc.AnnData, pseudobulk: dict, n_components: int = 10, verbose: bool = False) -> None:
    """
    Performs PCA on cell proportion data and stores the principal components in the AnnData object.
    """
    if 'cell_proportion' not in pseudobulk:
        raise KeyError("Missing 'cell_proportion' key in pseudobulk dictionary.")

    proportion_df = pseudobulk["cell_proportion"]
    proportion_df = proportion_df.fillna(0)

    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(proportion_df)

    adata.uns["X_pca_proportion"] = pca_coords  # Store PCA in AnnData

    if verbose:
        print(f"PCA on cell proportions completed. Stored {n_components} components in `adata.uns['X_pca_proportion']`.")

def plot_pca_with_cca_trajectory(pca_results: np.ndarray, cca_trajectory: np.ndarray, title: str = "PCA Projection with CCA Trajectory"):
    """
    Overlays the CCA trajectory direction onto a PCA scatter plot.

    Parameters:
    - pca_results (np.ndarray): 2D array (n_samples, 2) of PCA coordinates.
    - cca_trajectory (np.ndarray): 1D array of CCA's first component (CA1).
    - title (str): Title of the plot.
    """
    
    # Ensure PCA results have exactly 2 columns
    if pca_results.shape[1] != 2:
        raise ValueError(f"Expected pca_results to have shape (n_samples, 2), but got {pca_results.shape}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=cca_trajectory, cmap='coolwarm', alpha=0.7, edgecolors='k')

    cbar = plt.colorbar(scatter)
    cbar.set_label('CA1 (Canonical Component 1)')

    # Fit a regression line for the trajectory
    reg = LinearRegression()
    reg.fit(pca_results[:, 0].reshape(-1, 1), pca_results[:, 1])
    x_vals = np.linspace(min(pca_results[:, 0]), max(pca_results[:, 0]), 100)
    y_vals = reg.predict(x_vals.reshape(-1, 1))

    plt.plot(x_vals, y_vals, color='red', linestyle='--', linewidth=2, label="CCA Trajectory Direction")

    mean_x, mean_y = np.mean(pca_results[:, :2], axis=0)  # Ensure we only take two values
    arrow_dx = x_vals[-1] - x_vals[0]
    arrow_dy = y_vals[-1] - y_vals[0]
    plt.arrow(mean_x, mean_y, arrow_dx * 0.3, arrow_dy * 0.3, color='red', head_width=0.002, head_length=0.2)

    plt.xlabel("PC1 (Principal Component 1)")
    plt.ylabel("PC2 (Principal Component 2)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
