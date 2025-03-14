import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
from sklearn.cross_decomposition import CCA

def load_severity_levels(summary_sample_csv_path: str, sample_index: pd.Index) -> np.ndarray:
    """
    Load severity levels from a CSV file and align them with the provided sample index (which
    must match adata.obs["sample"]).
    
    Parameters
    ----------
    summary_sample_csv_path : str
        Path to the CSV file containing severity levels.
        Must have columns ['sample', 'sev.level'].
    sample_index : pd.Index
        Index (or array of sample names) that we need to align to the CSV data.
    
    Returns
    -------
    np.ndarray
        A 2D array (n_samples, 1) of severity levels aligned to `sample_index`.
        Missing values are replaced with the mean severity.
    """
    summary_df = pd.read_csv(summary_sample_csv_path)
    if 'sample' not in summary_df.columns or 'sev.level' not in summary_df.columns:
        raise ValueError("CSV must contain columns: 'sample' and 'sev.level'.")

    # Make sure severity is numeric
    summary_df['sev.level'] = pd.to_numeric(summary_df['sev.level'], errors='coerce')
    sample_to_sev = summary_df.set_index('sample')['sev.level'].to_dict()

    # Align to our sample_index
    sev_levels = np.array([sample_to_sev.get(sample, np.nan) for sample in sample_index])

    # Handle missing
    missing_samples = np.isnan(sev_levels).sum()
    if missing_samples > 0:
        print(f"Warning: {missing_samples} samples have missing severity levels. Imputing with mean.\n")
        sev_levels[np.isnan(sev_levels)] = np.nanmean(sev_levels)

    return sev_levels.reshape(-1, 1)  # shape: (n_samples, 1)


def run_cca_on_2d_pca_from_adata(
    adata: AnnData,
    summary_sample_csv_path: str
):
    """
    1) Reads the 2D PCA coordinates from `adata.uns["X_pca_proportion"]`.
    2) Loads and aligns severity levels based on `adata.obs["sample"]`.
    3) Performs single-component CCA on the (PC1, PC2) vs. severity.

    Parameters
    ----------
    adata : AnnData
        Must have:
            - adata.uns["X_pca_proportion"] of shape (n_samples, >=2)
            - adata.obs["sample"] containing sample names
    summary_sample_csv_path : str
        Path to CSV with columns ['sample', 'sev.level'].

    Returns
    -------
    pca_coords_2d : np.ndarray
        The first 2 columns from `adata.uns["X_pca_proportion"]`.
    sev_levels : np.ndarray
        1D severity levels, aligned to `adata.obs["sample"]`.
    cca : CCA
        The fitted CCA model (n_components=1).
    """
    pca_coords = adata.uns["X_pca_proportion"]
    if pca_coords.shape[1] < 2:
        raise ValueError("X_pca_proportion must have at least 2 components for 2D plotting.")

    # Extract the first two PC coordinates
    pca_coords_2d = pca_coords[:, :2]  # shape: (n_samples, 2)

    # Align severity to our samples
    if "sample" not in adata.obs.columns:
        raise KeyError("adata.obs must have a 'sample' column to match the CSV severity data.")
    samples = adata.obs["sample"].values.unique()
    if len(samples) != pca_coords_2d.shape[0]:
        raise ValueError("The number of PCA rows does not match the number of samples in adata.obs['sample'].")

    sev_levels_2d = load_severity_levels(summary_sample_csv_path, samples)
    sev_levels = sev_levels_2d.flatten()  # make it 1D

    # CCA on the 2D PCA coordinates vs severity
    cca = CCA(n_components=1)
    cca.fit(pca_coords_2d, sev_levels_2d)

    return pca_coords_2d, sev_levels, cca

def plot_cca_on_2d_pca(
    pca_coords_2d: np.ndarray,
    sev_levels: np.ndarray,
    cca: CCA,
    output_path: str = None,
    sample_labels=None
):
    """
    Plots a scatter of the first two PCA coordinates (colored by severity) with a dashed line
    extending in both directions to represent the CCA direction in the 2D plane.

    Parameters
    ----------
    pca_coords_2d : np.ndarray
        (n_samples, 2) PCA coordinates.
    sev_levels : np.ndarray
        (n_samples,) severity levels (for color mapping).
    cca : CCA
        Fitted CCA model with n_components=1.
    output_path : str, optional
        If provided, saves the figure to the given path. Otherwise, it shows interactively.
    sample_labels : array-like, optional
        If provided, text labels for the points (useful for debugging).
    """
    plt.figure(figsize=(7, 6))

    # Normalize severity for color mapping
    min_sev, max_sev = np.min(sev_levels), np.max(sev_levels)
    norm_sev = (sev_levels - min_sev) / (max_sev - min_sev + 1e-16)

    # Scatter: PC1 vs PC2
    sc = plt.scatter(
        pca_coords_2d[:, 0], 
        pca_coords_2d[:, 1], 
        c=norm_sev, 
        cmap='viridis_r',
        edgecolors='k',
        alpha=0.8
    )
    plt.colorbar(sc, label='Severity')

    # CCA direction vector
    dx, dy = cca.x_weights_[:, 0]

    # Get PCA range for scaling
    pc1_min, pc1_max = np.min(pca_coords_2d[:, 0]), np.max(pca_coords_2d[:, 0])
    pc2_min, pc2_max = np.min(pca_coords_2d[:, 1]), np.max(pca_coords_2d[:, 1])

    max_range = max(pc1_max - pc1_min, pc2_max - pc2_min)  # Maximum PCA span

    # Extend the line fully to the min/max range
    scale_factor = 0.5 * max_range  # Extends symmetrically in both directions
    x_start, x_end = -scale_factor * dx, scale_factor * dx
    y_start, y_end = -scale_factor * dy, scale_factor * dy

    # Dashed line representing the CCA direction
    plt.plot(
        [x_start, x_end], 
        [y_start, y_end], 
        linestyle="dashed", 
        color="red", 
        linewidth=2, 
        label="CCA Direction"
    )

    # Optionally label each point (sample name)
    if sample_labels is not None:
        for i, label in enumerate(sample_labels):
            plt.text(pca_coords_2d[i, 0], pca_coords_2d[i, 1], label, fontsize=8)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D PCA of Cell Type Proportions with CCA Direction")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved CCA direction plot to {output_path}")
    else:
        plt.show()


def CCA_Call(adata: AnnData, summary_sample_csv_path: str, output_dir: str):
    pca_coords_2d, sev_levels, cca_model = run_cca_on_2d_pca_from_adata(
        adata,
        summary_sample_csv_path
    )

    # Step 3: Plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, 'CCA')
    output_path = os.path.join(output_dir, "pca_2d_cca_direction.pdf")

    plot_cca_on_2d_pca(
        pca_coords_2d=pca_coords_2d,
        sev_levels=sev_levels,
        cca=cca_model,
        output_path=output_path
    )
