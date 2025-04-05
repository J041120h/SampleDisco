import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
from sklearn.cross_decomposition import CCA
import time

def load_severity_levels(sample_meta_path: str, sample_index: pd.Index, sev_col: str = "sev.level") -> np.ndarray:
    """
    Load severity levels from a CSV file and align them with the provided sample index.

    Parameters
    ----------
    sample_meta_path : str
        Path to the CSV file containing severity levels.
    sample_index : pd.Index
        Index of sample names to align to.
    sev_col : str
        Column name in the CSV that contains severity levels.

    Returns
    -------
    np.ndarray
        A 2D array (n_samples, 1) of severity levels aligned to `sample_index`.
    """
    summary_df = pd.read_csv(sample_meta_path)
    if 'sample' not in summary_df.columns or sev_col not in summary_df.columns:
        raise ValueError(f"CSV must contain columns: 'sample' and '{sev_col}'.")

    summary_df[sev_col] = pd.to_numeric(summary_df[sev_col], errors='coerce')
    sample_to_sev = summary_df.set_index('sample')[sev_col].to_dict()

    sev_levels = np.array([sample_to_sev.get(sample, np.nan) for sample in sample_index])
    missing_samples = np.isnan(sev_levels).sum()

    if missing_samples > 0:
        print(f"Warning: {missing_samples} samples have missing severity levels. Imputing with mean.\n")
        sev_levels[np.isnan(sev_levels)] = np.nanmean(sev_levels)

    return sev_levels.reshape(-1, 1)


def run_cca_on_2d_pca_from_adata(
    adata: AnnData,
    sample_meta_path: str,
    column: str,
    sev_col: str = "sev.level"
):
    """
    Run CCA on 2D PCA coordinates from adata.uns[column] vs. severity levels from CSV.

    Parameters
    ----------
    adata : AnnData
    sample_meta_path : str
    column : str
        Key in adata.uns containing PCA coordinates.
    sev_col : str
        Column name for severity levels in the metadata CSV.

    Returns
    -------
    pca_coords_2d : np.ndarray
    sev_levels : np.ndarray
    cca : CCA
    first_component_score : float
    """
    pca_coords = adata.uns[column]
    if pca_coords.shape[1] < 2:
        raise ValueError("X_pca must have at least 2 components for 2D plotting.")
    pca_coords = pca_coords.values
    pca_coords_2d = pca_coords[:, :2]

    if "sample" not in adata.obs.columns:
        raise KeyError("adata.obs must have a 'sample' column to match the CSV severity data.")
    samples = adata.obs["sample"].values.unique()
    if len(samples) != pca_coords_2d.shape[0]:
        raise ValueError("Mismatch between PCA rows and number of unique samples in adata.obs['sample'].")

    sev_levels_2d = load_severity_levels(sample_meta_path, samples, sev_col=sev_col)
    sev_levels = sev_levels_2d.flatten()

    cca = CCA(n_components=1)
    cca.fit(pca_coords_2d, sev_levels_2d)
    U, V = cca.transform(pca_coords_2d, sev_levels_2d)
    first_component_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

    print(f"\n\nThe CCA score for {column} is {first_component_score}\n\n")
    return pca_coords_2d, sev_levels, cca, first_component_score


def plot_cca_on_2d_pca(
    pca_coords_2d: np.ndarray,
    sev_levels: np.ndarray,
    cca: CCA,
    output_path: str = None,
    sample_labels=None
):
    """
    Plot PCA 2D scatter colored by severity and show CCA direction.

    Parameters
    ----------
    pca_coords_2d : np.ndarray
    sev_levels : np.ndarray
    cca : CCA
    output_path : str, optional
    sample_labels : array-like, optional
    """
    plt.figure(figsize=(7, 6))

    # Normalize severity for color mapping
    min_sev, max_sev = np.min(sev_levels), np.max(sev_levels)
    norm_sev = (sev_levels - min_sev) / (max_sev - min_sev + 1e-16)

    sc = plt.scatter(
        pca_coords_2d[:, 0],
        pca_coords_2d[:, 1],
        c=norm_sev,
        cmap='viridis_r',
        edgecolors='k',
        alpha=0.8
    )
    plt.colorbar(sc, label='Severity')

    # CCA direction
    dx, dy = cca.x_weights_[:, 0]
    max_range = max(
        np.ptp(pca_coords_2d[:, 0]),
        np.ptp(pca_coords_2d[:, 1])
    )
    scale = 0.5 * max_range
    x_start, x_end = -scale * dx, scale * dx
    y_start, y_end = -scale * dy, scale * dy

    plt.plot([x_start, x_end], [y_start, y_end],
             linestyle="dashed", color="red", linewidth=2, label="CCA Direction")

    if sample_labels is not None:
        for i, label in enumerate(sample_labels):
            plt.text(pca_coords_2d[i, 0], pca_coords_2d[i, 1], label, fontsize=8)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D PCA with CCA Direction")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Saved CCA direction plot to {output_path}")
    else:
        plt.show()


def CCA_Call(
    adata: AnnData,
    sample_meta_path: str,
    output_dir=None,
    verbose=False,
    sev_col: str = "sev.level"
):
    """
    Run CCA analysis on PCA projections in AnnData object and plot results.

    Parameters
    ----------
    adata : AnnData
    sample_meta_path : str
    output_dir : str, optional
    verbose : bool
    sev_col : str
        Column name in CSV for severity levels.
    """
    start_time = time.time() if verbose else None

    if output_dir:
        output_dir = os.path.join(output_dir, 'CCA')
        os.makedirs(output_dir, exist_ok=True)

    output_path_proportion = os.path.join(output_dir, "pca_2d_cca_proportion.pdf") if output_dir else None
    output_path_expression = os.path.join(output_dir, "pca_2d_cca_expression.pdf") if output_dir else None

    pca_coords_2d, sev_levels, cca_model, first_component_score_proportion = run_cca_on_2d_pca_from_adata(
        adata,
        sample_meta_path,
        "X_pca_proportion",
        sev_col=sev_col
    )

    plot_cca_on_2d_pca(
        pca_coords_2d=pca_coords_2d,
        sev_levels=sev_levels,
        cca=cca_model,
        output_path=output_path_proportion
    )

    pca_coords_2d, sev_levels, cca_model, first_component_score_expression = run_cca_on_2d_pca_from_adata(
        adata,
        sample_meta_path,
        "X_pca_expression",
        sev_col=sev_col
    )

    plot_cca_on_2d_pca(
        pca_coords_2d=pca_coords_2d,
        sev_levels=sev_levels,
        cca=cca_model,
        output_path=output_path_expression
    )

    if verbose:
        print("CCA completed.")
        print(f"\n\n[CCA]Total runtime for CCA processing: {time.time() - start_time:.2f} seconds\n\n")

    return first_component_score_proportion, first_component_score_expression