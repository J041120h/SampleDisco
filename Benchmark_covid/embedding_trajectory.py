"""
Trajectory Analysis from Embedding Matrix

This module computes pseudotime/trajectory from an embedding matrix using CCA
against a severity score or other continuous metadata variable.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def _find_best_2d_for_cca(embedding: np.ndarray, severity_levels: np.ndarray, verbose: bool = False):
    """
    Search all 2-dimension pairs to find best CCA alignment with severity.
    
    Parameters
    ----------
    embedding : np.ndarray
        Embedding matrix (n_samples x n_dims)
    severity_levels : np.ndarray
        Severity levels for each sample
    verbose : bool
        Print progress
        
    Returns
    -------
    tuple : (best_dim_pair, best_score, fitted_cca, best_2d_coords)
    """
    n_dims = embedding.shape[1]
    if n_dims < 2:
        raise ValueError("Need at least 2 dimensions for trajectory CCA.")
    
    sev_2d = severity_levels.reshape(-1, 1)
    
    # If exactly 2 dimensions, just use them
    if n_dims == 2:
        cca = CCA(n_components=1)
        cca.fit(embedding, sev_2d)
        U, V = cca.transform(embedding, sev_2d)
        score = float(abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1]))
        return (0, 1), score, cca, embedding
    
    # Search all pairs
    best = (-1.0, None, None, None)
    for dim1, dim2 in combinations(range(n_dims), 2):
        sub = embedding[:, [dim1, dim2]]
        try:
            cca = CCA(n_components=1)
            cca.fit(sub, sev_2d)
            U, V = cca.transform(sub, sev_2d)
            score = float(abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1]))
            if score > best[0]:
                best = (score, (dim1, dim2), cca, sub)
            if verbose:
                print(f"Dim{dim1+1}+Dim{dim2+1} CCA score={score:.4f}")
        except Exception as e:
            if verbose:
                print(f"Skip Dim{dim1+1}+Dim{dim2+1}: {e}")
            continue
    
    if best[1] is None:
        raise RuntimeError("Failed to find a valid 2-dimension CCA combination.")
    
    return best[1], best[0], best[2], best[3]


def _assign_pseudotime_from_cca(coords_2d: np.ndarray, cca: CCA, scale_to_unit: bool = True) -> np.ndarray:
    """
    Project onto CCA x-weights to get pseudotime.
    
    Parameters
    ----------
    coords_2d : np.ndarray
        2D coordinates (n_samples x 2)
    cca : CCA
        Fitted CCA model
    scale_to_unit : bool
        Scale pseudotime to [0, 1]
        
    Returns
    -------
    np.ndarray : Pseudotime values for each sample
    """
    direction = cca.x_weights_[:, 0]  # shape (2,)
    proj = coords_2d @ direction      # shape (n_samples,)
    
    if not scale_to_unit:
        return proj
    
    lo, hi = float(np.min(proj)), float(np.max(proj))
    denom = max(hi - lo, 1e-16)
    return (proj - lo) / denom


def compute_trajectory_from_embedding(
    embedding_csv_path: str,
    sample_metadata_csv_path: str,
    severity_column: str = "sev.level",
    sample_column: str = "sample",
    auto_select_best_2d: bool = True,
    save_plot: bool = True,
    verbose: bool = True,
    plot_filename: str = "trajectory_embedding_space.png"
):
    """
    Compute pseudotime/trajectory from an embedding matrix CSV file using CCA.
    
    This function:
    1. Reads embedding matrix and sample metadata from CSV files
    2. Finds the best 2D projection that aligns with severity via CCA
    3. Computes pseudotime as projection along CCA direction
    4. Saves results in the same directory as the embedding file
    
    Parameters
    ----------
    embedding_csv_path : str
        Path to CSV file containing embedding matrix (samples as rows, dimensions as columns)
        Expected format: First column is sample IDs, remaining columns are embedding dimensions
    sample_metadata_csv_path : str
        Path to CSV file containing sample metadata
        Must have columns specified by sample_column and severity_column
    severity_column : str
        Column name in metadata containing severity or progression scores
    sample_column : str
        Column name in metadata containing sample identifiers
    auto_select_best_2d : bool
        If True, automatically select best 2-dimension pair for CCA
        If False, use first two dimensions
    save_plot : bool
        Whether to save trajectory visualization plot
    verbose : bool
        Print progress messages
    plot_filename : str
        Name for the trajectory plot file
        
    Returns
    -------
    pd.DataFrame : DataFrame with columns [sample, pseudotime, dim1, dim2, cca_score]
    
    Example
    -------
    >>> # Modify these paths for your data
    >>> embedding_path = "/path/to/your/embedding.csv"
    >>> metadata_path = "/path/to/your/metadata.csv"
    >>> results = compute_trajectory_from_embedding(
    ...     embedding_csv_path=embedding_path,
    ...     sample_metadata_csv_path=metadata_path,
    ...     severity_column="severity_score"
    ... )
    """
    
    # Determine output directory (same as embedding file)
    output_dir = os.path.dirname(embedding_csv_path)
    trajectory_dir = os.path.join(output_dir, "trajectory")
    os.makedirs(trajectory_dir, exist_ok=True)
    
    if verbose:
        print(f"=== Computing Trajectory from Embedding ===")
        print(f"Embedding file: {embedding_csv_path}")
        print(f"Metadata file: {sample_metadata_csv_path}")
        print(f"Output directory: {trajectory_dir}")
    
    # Read embedding matrix
    embedding_df = pd.read_csv(embedding_csv_path, index_col=0)
    samples = embedding_df.index.to_numpy()
    embedding_matrix = embedding_df.values
    
    if verbose:
        print(f"Embedding shape: {embedding_matrix.shape[0]} samples x {embedding_matrix.shape[1]} dimensions")
    
    # Read sample metadata
    metadata_df = pd.read_csv(sample_metadata_csv_path)
    
    # Check if severity column exists
    if severity_column not in metadata_df.columns:
        raise KeyError(f"'{severity_column}' not found in metadata. Available columns: {list(metadata_df.columns)}")
    
    # Align samples and get severity levels
    metadata_indexed = metadata_df.set_index(sample_column)
    
    # Get severity levels for samples in same order as embedding
    severity_levels = pd.to_numeric(
        metadata_indexed.loc[samples, severity_column],
        errors='coerce'
    ).to_numpy()
    
    # Handle missing values
    if np.isnan(severity_levels).any():
        n_missing = int(np.isnan(severity_levels).sum())
        if verbose:
            print(f"Warning: {n_missing} samples have missing {severity_column} values; imputing with mean.")
        mean_severity = np.nanmean(severity_levels)
        severity_levels = np.where(np.isnan(severity_levels), mean_severity, severity_levels)
    
    # Find best 2D projection and fit CCA
    if auto_select_best_2d and embedding_matrix.shape[1] > 2:
        if verbose:
            print("Searching for best 2-dimension pair...")
        (dim1, dim2), score, cca, coords_2d = _find_best_2d_for_cca(
            embedding_matrix, severity_levels, verbose
        )
        if verbose:
            print(f"Selected dimensions {dim1+1} and {dim2+1} (CCA score: {score:.4f})")
    else:
        dim1, dim2 = 0, 1
        coords_2d = embedding_matrix[:, [dim1, dim2]]
        cca = CCA(n_components=1)
        cca.fit(coords_2d, severity_levels.reshape(-1, 1))
        U, V = cca.transform(coords_2d, severity_levels.reshape(-1, 1))
        score = float(abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1]))
        if verbose:
            print(f"Using dimensions 1 and 2 (CCA score: {score:.4f})")
    
    # Compute pseudotime
    pseudotime = _assign_pseudotime_from_cca(coords_2d, cca, scale_to_unit=True)
    
    # Create trajectory plot
    if save_plot:
        fig_path = os.path.join(trajectory_dir, plot_filename)
        
        plt.figure(figsize=(10, 8))
        
        # Normalize severity for colormap
        sev_norm = (severity_levels - severity_levels.min()) / (severity_levels.max() - severity_levels.min() + 1e-16)
        
        # Scatter plot colored by severity
        sc = plt.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=sev_norm, cmap='viridis',
            edgecolors='black', alpha=0.8, s=100,
            linewidths=0.5
        )
        
        # Colorbar
        cbar = plt.colorbar(sc, label=f'Normalized {severity_column}')
        cbar.ax.tick_params(labelsize=10)
        
        # Add CCA direction arrow
        dx, dy = cca.x_weights_[:, 0]
        scale = 0.4 * max(np.ptp(coords_2d[:, 0]), np.ptp(coords_2d[:, 1]))
        
        # Draw arrow showing trajectory direction
        plt.arrow(
            0, 0, scale*dx, scale*dy,
            head_width=0.03*scale, head_length=0.05*scale,
            fc='red', ec='red', linewidth=2.5, alpha=0.7,
            label=f'Trajectory (CCA score={score:.3f})'
        )
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Labels and title
        dim_names = embedding_df.columns
        xlabel = dim_names[dim1] if dim1 < len(dim_names) else f"Dimension {dim1+1}"
        ylabel = dim_names[dim2] if dim2 < len(dim_names) else f"Dimension {dim2+1}"
        
        plt.xlabel(xlabel, fontsize=12, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12, fontweight='bold')
        plt.title(
            f'Trajectory Analysis in Embedding Space\n'
            f'CCA alignment with {severity_column}',
            fontsize=14, fontweight='bold', pad=15
        )
        plt.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved trajectory plot: {fig_path}")
    
    # Create results dataframe with only sample and pseudotime
    results_df = pd.DataFrame({
        "sample": samples,
        "pseudotime": pseudotime
    })
    
    # Sort by pseudotime for better readability
    results_df = results_df.sort_values('pseudotime')
    
    # Save results (only sample and pseudotime columns)
    csv_path = os.path.join(trajectory_dir, "pseudotime_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Store additional info for verbose output only (not saved to file)
    if verbose:
        full_info = pd.DataFrame({
            "sample": samples,
            "pseudotime": pseudotime,
            "dim1": dim1 + 1,
            "dim2": dim2 + 1,
            "cca_score": score,
            severity_column: severity_levels
        }).sort_values('pseudotime')
    
    if verbose:
        print(f"Saved pseudotime results: {csv_path}")
        print(f"\nTrajectory computation complete!")
        print(f"Pseudotime range: [{pseudotime.min():.4f}, {pseudotime.max():.4f}]")
        print(f"CCA correlation score: {score:.4f}")
    
    return results_df

# ============================================================================
# MAIN EXECUTION - MODIFY THESE PATHS FOR YOUR DATA
# ============================================================================
if __name__ == "__main__":
    # ========== CONSTANT PATHS ==========
    SAMPLE_METADATA_CSV_PATH = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv"
    SEVERITY_COLUMN = "sev.level"
    SAMPLE_COLUMN = "sample"
    # =====================================

    # Sample sizes to loop over
    sample_sizes = [25, 50, 100, 200, 279, 400]

    for n in sample_sizes:
        print(f"\n=======================")
        print(f" Running trajectory for {n} samples")
        print(f"=======================\n")

        # Construct the embedding path for this sample size
        EMBEDDING_CSV_PATH = (
            f'/dcs07/hongkai/data/harry/result/pilot/{n}_sample/wasserstein_distance_mds_10d.csv'
        )
        
        # Run trajectory
        results = compute_trajectory_from_embedding(
            embedding_csv_path=EMBEDDING_CSV_PATH,
            sample_metadata_csv_path=SAMPLE_METADATA_CSV_PATH,
            severity_column=SEVERITY_COLUMN,
            sample_column=SAMPLE_COLUMN,
            auto_select_best_2d=True,
            save_plot=True,
            verbose=True
        )
