import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
from sklearn.cross_decomposition import CCA
import time


def load_severity_levels(
    sample_meta_path: str, 
    sample_index: pd.Index, 
    sample_col: str = "sample", 
    sev_col: str = "sev.level"
) -> np.ndarray:
    """
    Load severity levels from a CSV and align them with the provided sample index.
    """
    summary_df = pd.read_csv(sample_meta_path)
    
    if sample_col not in summary_df.columns or sev_col not in summary_df.columns:
        raise ValueError(f"CSV must contain columns: '{sample_col}' and '{sev_col}'.")

    summary_df[sev_col] = pd.to_numeric(summary_df[sev_col], errors='coerce')
    summary_df[sample_col] = summary_df[sample_col].astype(str).str.strip().str.lower()
    
    # Convert sample_index to a pandas Series before applying string methods
    if isinstance(sample_index, np.ndarray):
        sample_index = pd.Series(sample_index).astype(str).str.strip().str.lower().values
    else:
        # If it's already a pandas Index/Series
        sample_index = sample_index.astype(str).str.strip().str.lower()

    sample_to_sev = summary_df.set_index(sample_col)[sev_col].to_dict()
    sev_levels = np.array([sample_to_sev.get(sample, np.nan) for sample in sample_index])

    missing = np.isnan(sev_levels).sum()
    if missing > 0:
        print(f"Warning: {missing} sample(s) missing severity level. Imputing with mean.")
        sev_levels[np.isnan(sev_levels)] = np.nanmean(sev_levels)

    return sev_levels.reshape(-1, 1)


def run_cca_on_2d_pca_from_adata(
    adata: AnnData,
    sample_meta_path: str,
    column: str,
    sample_col: str = "sample",
    sev_col: str = "sev.level"
):
    """
    Run CCA on 2D PCA coordinates from adata.uns[column] vs. severity levels from CSV.
    """
    if column not in adata.uns:
        raise KeyError(f"'{column}' not found in adata.uns.")

    pca_coords = adata.uns[column]
    if pca_coords.shape[1] < 2:
        raise ValueError("PCA must have at least 2 components for CCA.")

    # Extract first two PCs
    pca_coords_2d = pca_coords.iloc[:, :2].values if hasattr(pca_coords, 'iloc') else pca_coords[:, :2]

    # Ensure sample_col exists in adata.obs
    if sample_col not in adata.obs.columns:
        raise KeyError(f"'{sample_col}' column is missing in adata.obs.")

    # Get the unique samples (lowercased, stripped) in the same order as pca_coords_2d
    samples = adata.obs[sample_col].astype(str).str.strip().str.lower().unique()
    
    # Check dimension alignment
    if len(samples) != pca_coords_2d.shape[0]:
        raise ValueError("Mismatch between PCA rows and number of unique samples in adata.obs[sample_col].")

    # Load severity levels aligned to these samples
    sev_levels_2d = load_severity_levels(sample_meta_path, samples, sample_col=sample_col, sev_col=sev_col)
    sev_levels = sev_levels_2d.flatten()

    cca = CCA(n_components=1)
    cca.fit(pca_coords_2d, sev_levels_2d)
    U, V = cca.transform(pca_coords_2d, sev_levels_2d)
    first_component_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

    print(f"\nThe CCA score for {column} is {first_component_score:.4f}\n")
    return pca_coords_2d, sev_levels, cca, first_component_score, samples


def plot_cca_on_2d_pca(
    pca_coords_2d: np.ndarray,
    sev_levels: np.ndarray,
    cca: CCA,
    output_path: str = None,
    sample_labels=None
):
    """
    Plot 2D PCA colored by severity with CCA direction.
    """
    plt.figure(figsize=(7, 6))

    norm_sev = (sev_levels - np.min(sev_levels)) / (np.max(sev_levels) - np.min(sev_levels) + 1e-16)

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
    scale = 0.5 * max(np.ptp(pca_coords_2d[:, 0]), np.ptp(pca_coords_2d[:, 1]))
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
        print(f"Saved CCA plot to: {output_path}")
    else:
        plt.show()


def assign_pseudotime_from_cca(
    pca_coords_2d: np.ndarray, 
    cca: CCA, 
    sample_labels: np.ndarray,
    scale_to_unit: bool = True
) -> dict:
    """
    Assign pseudotime to each sample based on its projection onto the CCA direction.

    Parameters
    ----------
    pca_coords_2d : np.ndarray
        The 2D PCA coordinates for each sample.
    cca : sklearn.cross_decomposition.CCA
        A fitted CCA model. We use cca.x_weights_ to get the direction in PCA space.
    sample_labels : np.ndarray
        An array of sample labels corresponding to each row in pca_coords_2d.
    scale_to_unit : bool, default=True
        Whether to scale the pseudotime to the [0, 1] range.

    Returns
    -------
    dict
        A dictionary mapping each sample label to its pseudotime.
    """
    # Extract the direction (the first CCA weight vector)
    direction = cca.x_weights_[:, 0]

    # Project each point onto the CCA direction (dot product)
    # shape of pca_coords_2d: (n_samples, 2)
    # shape of direction: (2,)
    raw_projection = pca_coords_2d @ direction  # shape: (n_samples,)

    if scale_to_unit:
        min_proj, max_proj = np.min(raw_projection), np.max(raw_projection)
        denom = max_proj - min_proj
        # Avoid division by zero
        if denom < 1e-16:
            denom = 1e-16
        # Scale the projection to [0, 1]
        pseudotimes = (raw_projection - min_proj) / denom
    else:
        pseudotimes = raw_projection

    # Build and return dictionary
    return {sample_labels[i]: pseudotimes[i] for i in range(len(sample_labels))}

def CCA_Call(
    adata: AnnData,
    sample_meta_path: str,
    output_dir: str = None,
    sample_col: str = "sample",
    sev_col: str = "sev.level",
    ptime: bool = False,
    verbose: bool = False
):
    """
    Run CCA analysis on PCA projections stored in adata.uns and plot results.
    """
    start_time = time.time() if verbose else None
    if output_dir:
        output_dir = os.path.join(output_dir, 'CCA')
        os.makedirs(output_dir, exist_ok=True)

    paths = {
        "X_pca_proportion": os.path.join(output_dir, "pca_2d_cca_proportion.pdf") if output_dir else None,
        "X_pca_expression": os.path.join(output_dir, "pca_2d_cca_expression.pdf") if output_dir else None
    }

    results = {}
    # We'll also hold onto the sample labels to generate pseudotime at the end
    sample_dicts = {}

    for key in ["X_pca_proportion", "X_pca_expression"]:
        (pca_coords_2d, 
         sev_levels, 
         cca_model, 
         score, 
         samples) = run_cca_on_2d_pca_from_adata(
            adata=adata,
            sample_meta_path=sample_meta_path,
            column=key,
            sample_col=sample_col,
            sev_col=sev_col
        )

        # Plot and save
        plot_cca_on_2d_pca(
            pca_coords_2d=pca_coords_2d,
            sev_levels=sev_levels,
            cca=cca_model,
            output_path=paths[key],
            sample_labels=samples  # if you want text labels, else None
        )
        # Save results
        results[key] = score

        # Compute pseudotime for each sample (dictionary)
        sample_dicts[key] = assign_pseudotime_from_cca(
            pca_coords_2d=pca_coords_2d, 
            cca=cca_model, 
            sample_labels=samples
        )

    if verbose:
        print("CCA completed.")
        print(f"\n[CCA] Total runtime: {time.time() - start_time:.2f} seconds\n")

    return results["X_pca_proportion"], results["X_pca_expression"], sample_dicts["X_pca_proportion"], sample_dicts["X_pca_expression"]