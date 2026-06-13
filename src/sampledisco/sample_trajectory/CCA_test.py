import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from anndata import AnnData
import time


def generate_null_distribution(pseudobulk_adata, column, trajectory_col,
                                   n_permutations=1000, n_pcs=None,
                                   save_path=None, verbose=True):
    """
    Generate a permutation null distribution of CCA scores.

    Labels are shuffled (not coordinates) so that the null captures the
    distribution of |corr| when trajectory is unrelated to the embedding.
    abs() is taken because CCA canonical correlation is sign-invariant —
    both +r and -r are equally extreme under the null.

    Parameters:
        pseudobulk_adata: AnnData with DR coords in uns[column]
        column: key in uns for the coordinate matrix
        trajectory_col: obs column with trajectory labels
        n_permutations: number of label permutations
        n_pcs: number of DR dimensions to use (None = all)
        save_path: if given, save null array as .npy
        verbose: print success rate

    Returns:
        np.ndarray of length n_permutations (absolute CCA correlations)
    """
    if column not in pseudobulk_adata.uns:
        raise ValueError(f"Column '{column}' not found in pseudobulk_adata.uns")

    dr_coords_full = pseudobulk_adata.uns[column].copy()
    sev_levels = pseudobulk_adata.obs[trajectory_col].values

    if n_pcs is None:
        dr_coords = dr_coords_full
        n_dims_used = dr_coords_full.shape[1]
    else:
        n_pcs = min(n_pcs, dr_coords_full.shape[1])
        dr_coords = dr_coords_full.iloc[:, :n_pcs]
        n_dims_used = n_pcs

    if len(dr_coords) < 3:
        raise ValueError(f"Insufficient samples: {len(dr_coords)}")
    if len(np.unique(sev_levels)) < 2:
        raise ValueError("Insufficient severity level variance")

    X = dr_coords.values
    y_original = sev_levels.copy()

    null_scores = []
    failed_permutations = 0

    for perm in range(n_permutations):
        try:
            permuted_sev = np.random.permutation(y_original)

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_permuted_scaled = scaler_y.fit_transform(permuted_sev.reshape(-1, 1))

            cca_perm = CCA(n_components=1, max_iter=1000, tol=1e-6)
            cca_perm.fit(X_scaled, y_permuted_scaled)

            X_c_perm, y_c_perm = cca_perm.transform(X_scaled, y_permuted_scaled)
            perm_correlation = np.corrcoef(X_c_perm[:, 0], y_c_perm[:, 0])[0, 1]

            if np.isnan(perm_correlation) or np.isinf(perm_correlation):
                null_scores.append(0.0)
                failed_permutations += 1
            else:
                # abs() because CCA score is sign-invariant
                null_scores.append(abs(perm_correlation))

        except Exception:
            null_scores.append(0.0)
            failed_permutations += 1

    null_distribution = np.array(null_scores)

    if verbose:
        success_rate = (n_permutations - failed_permutations) / n_permutations * 100
        print(f"Null distribution generated using {n_dims_used} PC dimensions: {success_rate:.1f}% success rate")

    if save_path:
        np.save(save_path, null_distribution)

    return null_distribution


def generate_corrected_null_distribution(all_resolution_results, n_permutations=1000):
    """
    Build a corrected null that accounts for selecting the best-scoring resolution.

    For each permutation index, takes the maximum permuted CCA score across all
    resolutions — this mirrors the selection bias introduced when the observed
    score is also chosen as the best-resolution score.

    Parameters:
        all_resolution_results: list of dicts, each with key 'null_scores' (array)
        n_permutations: number of permutations (must match the arrays in all_resolution_results)

    Returns:
        np.ndarray: corrected null distribution of length n_permutations
    """
    corrected_null_scores = []

    for perm_idx in range(n_permutations):
        perm_scores_across_resolutions = []

        for resolution_result in all_resolution_results:
            if 'null_scores' in resolution_result and resolution_result['null_scores'] is not None:
                if len(resolution_result['null_scores']) > perm_idx:
                    perm_scores_across_resolutions.append(resolution_result['null_scores'][perm_idx])

        if perm_scores_across_resolutions:
            max_score_for_this_perm = max(perm_scores_across_resolutions)
            corrected_null_scores.append(max_score_for_this_perm)

    return np.array(corrected_null_scores)


def compute_corrected_pvalues(df_results, corrected_null_distribution, output_dir, column):
    """
    Compute corrected p-values for all CCA scores and save one diagnostic plot per resolution.

    Parameters:
        df_results: DataFrame with columns 'resolution' and 'cca_score'
        corrected_null_distribution: np.ndarray from generate_corrected_null_distribution
        output_dir: directory for output plots
        column: embedding key label (for titles/filenames)

    Returns:
        df_results with 'corrected_pvalue' column added
    """
    pvalue_dir = os.path.join(output_dir, "corrected_p_values")
    os.makedirs(pvalue_dir, exist_ok=True)

    df_results['corrected_pvalue'] = np.nan

    for idx, row in df_results.iterrows():
        resolution = row['resolution']
        cca_score = row['cca_score']

        if not np.isnan(cca_score):
            corrected_p_value = np.mean(corrected_null_distribution >= cca_score)
            df_results.loc[idx, 'corrected_pvalue'] = corrected_p_value

            plt.figure(figsize=(10, 6))

            plt.hist(corrected_null_distribution, bins=50, alpha=0.7, color='lightblue',
                    density=True, label='Corrected Null Distribution')

            plt.axvline(cca_score, color='red', linestyle='--', linewidth=2,
                       label=f'Observed CCA Score: {cca_score:.4f}')

            plt.text(0.05, 0.95, f'Corrected p-value: {corrected_p_value:.4f}',
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.xlabel('CCA Score')
            plt.ylabel('Density')
            plt.title(f'Corrected P-value Analysis\nResolution: {resolution:.3f}, {column}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plot_filename = f'corrected_pvalue_res_{resolution:.3f}.png'
            plot_path = os.path.join(pvalue_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

    return df_results


def create_comprehensive_summary(df_results, best_resolution, column, output_dir,
                                     has_corrected_pvalues=False):
    """
    Create summary visualizations and a text report for ATAC resolution optimization.

    Parameters:
        df_results: DataFrame with columns 'resolution', 'cca_score', 'pass'
        best_resolution: optimal resolution to highlight
        column: embedding key label
        output_dir: root output directory (summary/ subdirectory is created)
        has_corrected_pvalues: whether 'corrected_pvalue' column is present
    """
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    df_sorted = df_results.sort_values('resolution').copy()

    if has_corrected_pvalues:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    valid_df = df_sorted[~df_sorted['cca_score'].isna()]

    coarse_df = valid_df[valid_df['pass'] == 'coarse']
    fine_df = valid_df[valid_df['pass'] == 'fine']

    ax1.scatter(coarse_df['resolution'], coarse_df['cca_score'],
                color='blue', s=80, alpha=0.6, label='Coarse Search', zorder=2)
    ax1.scatter(fine_df['resolution'], fine_df['cca_score'],
                color='green', s=60, alpha=0.8, label='Fine Search', zorder=3)

    ax1.plot(valid_df['resolution'], valid_df['cca_score'],
             'k-', linewidth=1, alpha=0.4, zorder=1)

    ax1.axvline(x=best_resolution, color='red', linestyle='--', linewidth=2,
                label=f'Best Resolution: {best_resolution:.3f}', zorder=4)

    best_score = valid_df.loc[valid_df['resolution'] == best_resolution, 'cca_score'].iloc[0]
    ax1.annotate(f'Best Score: {best_score:.4f}',
                 xy=(best_resolution, best_score),
                 xytext=(best_resolution + 0.05, best_score + 0.01),
                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                 fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    ax1.set_xlabel('Resolution', fontsize=12)
    ax1.set_ylabel('CCA Score', fontsize=12)
    ax1.set_title(f'ATAC Resolution Optimization: {column}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    if has_corrected_pvalues:
        valid_pval_df = valid_df[~valid_df['corrected_pvalue'].isna()]

        coarse_pval = valid_pval_df[valid_pval_df['pass'] == 'coarse']
        fine_pval = valid_pval_df[valid_pval_df['pass'] == 'fine']

        ax2.scatter(coarse_pval['resolution'], coarse_pval['corrected_pvalue'],
                    color='blue', s=80, alpha=0.6, label='Coarse Search', zorder=2)
        ax2.scatter(fine_pval['resolution'], fine_pval['corrected_pvalue'],
                    color='green', s=60, alpha=0.8, label='Fine Search', zorder=3)

        ax2.plot(valid_pval_df['resolution'], valid_pval_df['corrected_pvalue'],
                 'k-', linewidth=1, alpha=0.4, zorder=1)

        ax2.axvline(x=best_resolution, color='red', linestyle='--', linewidth=2, zorder=4)
        ax2.axhline(y=0.05, color='orange', linestyle=':', linewidth=2,
                    label='p=0.05 threshold', zorder=4)

        best_pval = valid_pval_df.loc[valid_pval_df['resolution'] == best_resolution, 'corrected_pvalue'].iloc[0]
        ax2.annotate(f'p={best_pval:.4f}',
                     xy=(best_resolution, best_pval),
                     xytext=(best_resolution + 0.05, best_pval + 0.05),
                     arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                     fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

        ax2.set_xlabel('Resolution', fontsize=12)
        ax2.set_ylabel('Corrected P-value', fontsize=12)
        ax2.set_title('Corrected P-values (Accounting for Resolution Selection)', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    plot_path = os.path.join(summary_dir, f'resolution_optimization_summary_{column}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved to: {plot_path}")

    summary_path = os.path.join(summary_dir, f'optimization_results_{column}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"ATAC Resolution Optimization Results: {column}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Best Resolution: {best_resolution:.3f}\n")
        f.write(f"Best CCA Score: {best_score:.4f}\n")

        if has_corrected_pvalues:
            f.write(f"Corrected P-value at Best Resolution: {best_pval:.4f}\n")

        f.write(f"\nTotal Resolutions Tested: {len(valid_df)}\n")
        f.write(f"  - Coarse Search: {len(coarse_df)} resolutions\n")
        f.write(f"  - Fine Search: {len(fine_df)} resolutions\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("All Results (sorted by resolution):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Resolution':<12} {'CCA Score':<12} {'Pass Type':<12}")
        if has_corrected_pvalues:
            f.write(f" {'Corrected P-value':<18}")
        f.write("\n")

        for _, row in valid_df.iterrows():
            f.write(f"{row['resolution']:<12.3f} {row['cca_score']:<12.4f} {row['pass']:<12}")
            if has_corrected_pvalues and 'corrected_pvalue' in row:
                pval = row['corrected_pvalue']
                pval_str = f"{pval:.4f}" if not np.isnan(pval) else "N/A"
                f.write(f" {pval_str:<18}")
            f.write("\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("Summary Statistics:\n")
        f.write(f"CCA Score Range: [{valid_df['cca_score'].min():.4f}, {valid_df['cca_score'].max():.4f}]\n")
        f.write(f"Mean CCA Score: {valid_df['cca_score'].mean():.4f} ± {valid_df['cca_score'].std():.4f}\n")

        if has_corrected_pvalues:
            valid_pvals = valid_df['corrected_pvalue'].dropna()
            if len(valid_pvals) > 0:
                f.write(f"\nCorrected P-value Statistics:\n")
                f.write(f"Min P-value: {valid_pvals.min():.4f}\n")
                f.write(f"Resolutions with p < 0.05: {(valid_pvals < 0.05).sum()}\n")
                f.write(f"Resolutions with p < 0.01: {(valid_pvals < 0.01).sum()}\n")

    print(f"Summary report saved to: {summary_path}")

    detailed_csv_path = os.path.join(summary_dir, f'detailed_results_{column}.csv')
    df_sorted.to_csv(detailed_csv_path, index=False)
    print(f"Detailed results saved to: {detailed_csv_path}")


def compute_corrected_pvalues_rna(df_results, corrected_null_distribution, output_dir, column):
    """
    Compute corrected p-values and save diagnostic plots for RNA-seq resolution optimization.

    Identical logic to compute_corrected_pvalues but with RNA-seq labels in titles.

    Parameters:
        df_results: DataFrame with columns 'resolution' and 'cca_score'
        corrected_null_distribution: np.ndarray from generate_corrected_null_distribution
        output_dir: directory for output plots
        column: embedding key label

    Returns:
        df_results with 'corrected_pvalue' column added
    """
    pvalue_dir = os.path.join(output_dir, "corrected_p_values")
    os.makedirs(pvalue_dir, exist_ok=True)

    df_results['corrected_pvalue'] = np.nan

    for idx, row in df_results.iterrows():
        resolution = row['resolution']
        cca_score = row['cca_score']

        if not np.isnan(cca_score):
            corrected_p_value = np.mean(corrected_null_distribution >= cca_score)
            df_results.loc[idx, 'corrected_pvalue'] = corrected_p_value

            plt.figure(figsize=(10, 6))

            plt.hist(corrected_null_distribution, bins=50, alpha=0.7, color='lightblue',
                    density=True, label='Corrected Null Distribution')

            plt.axvline(cca_score, color='red', linestyle='--', linewidth=2,
                       label=f'Observed CCA Score: {cca_score:.4f}')

            plt.text(0.05, 0.95, f'Corrected p-value: {corrected_p_value:.4f}',
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.xlabel('CCA Score')
            plt.ylabel('Density')
            plt.title(f'Corrected P-value Analysis (RNA-seq)\nResolution: {resolution:.3f}, {column}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plot_filename = f'corrected_pvalue_res_{resolution:.3f}.png'
            plot_path = os.path.join(pvalue_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

    return df_results


def create_comprehensive_summary_rna(df_results, best_resolution, column, output_dir,
                                    has_corrected_pvalues=False):
    """
    Create summary visualizations and a text report for RNA-seq resolution optimization.

    Parameters:
        df_results: DataFrame with columns 'resolution', 'cca_score', 'pass'
        best_resolution: optimal resolution to highlight
        column: embedding key label
        output_dir: root output directory (summary/ subdirectory is created)
        has_corrected_pvalues: whether 'corrected_pvalue' column is present
    """
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    df_sorted = df_results.sort_values('resolution').copy()

    if has_corrected_pvalues:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    valid_df = df_sorted[~df_sorted['cca_score'].isna()]

    coarse_df = valid_df[valid_df['pass'] == 'coarse']
    fine_df = valid_df[valid_df['pass'] == 'fine']

    ax1.scatter(coarse_df['resolution'], coarse_df['cca_score'],
                color='blue', s=80, alpha=0.6, label='Coarse Search', zorder=2)
    ax1.scatter(fine_df['resolution'], fine_df['cca_score'],
                color='green', s=60, alpha=0.8, label='Fine Search', zorder=3)

    ax1.plot(valid_df['resolution'], valid_df['cca_score'],
             'k-', linewidth=1, alpha=0.4, zorder=1)

    ax1.axvline(x=best_resolution, color='red', linestyle='--', linewidth=2,
                label=f'Best Resolution: {best_resolution:.3f}', zorder=4)

    best_score = valid_df.loc[valid_df['resolution'] == best_resolution, 'cca_score'].iloc[0]
    ax1.annotate(
        f'Best Score: {best_score:.4f}',
        xy=(best_resolution, best_score),
        xytext=(best_resolution, best_score + 0.02),
        arrowprops=dict(arrowstyle='->', color='black'),
        fontsize=10,
        ha='center'
    )

    ax1.set_xlabel('Resolution', fontsize=12)
    ax1.set_ylabel('CCA Score', fontsize=12)
    ax1.set_title(f'RNA-seq Resolution Optimization: {column}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    if has_corrected_pvalues:
        valid_pval_df = valid_df[~valid_df['corrected_pvalue'].isna()]

        coarse_pval = valid_pval_df[valid_pval_df['pass'] == 'coarse']
        fine_pval = valid_pval_df[valid_pval_df['pass'] == 'fine']

        ax2.scatter(coarse_pval['resolution'], coarse_pval['corrected_pvalue'],
                    color='blue', s=80, alpha=0.6, label='Coarse Search', zorder=2)
        ax2.scatter(fine_pval['resolution'], fine_pval['corrected_pvalue'],
                    color='green', s=60, alpha=0.8, label='Fine Search', zorder=3)

        ax2.plot(valid_pval_df['resolution'], valid_pval_df['corrected_pvalue'],
                 'k-', linewidth=1, alpha=0.4, zorder=1)

        ax2.axvline(x=best_resolution, color='red', linestyle='--', linewidth=2, zorder=4)
        ax2.axhline(y=0.05, color='orange', linestyle=':', linewidth=2,
                    label='p=0.05 threshold', zorder=4)

        best_pval = valid_pval_df.loc[valid_pval_df['resolution'] == best_resolution, 'corrected_pvalue'].iloc[0]
        ax2.annotate(f'p={best_pval:.4f}',
                     xy=(best_resolution, best_pval),
                     xytext=(best_resolution + 0.05, best_pval + 0.05),
                     arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                     fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

        ax2.set_xlabel('Resolution', fontsize=12)
        ax2.set_ylabel('Corrected P-value', fontsize=12)
        ax2.set_title('Corrected P-values (Accounting for Resolution Selection)', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    plot_path = os.path.join(summary_dir, f'resolution_optimization_summary_{column}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved to: {plot_path}")

    # encoding='utf-8' required for the '±' character in the mean ± std line
    summary_path = os.path.join(summary_dir, f'optimization_results_{column}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"RNA-seq Resolution Optimization Results: {column}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Best Resolution: {best_resolution:.3f}\n")
        f.write(f"Best CCA Score: {best_score:.4f}\n")

        if has_corrected_pvalues:
            f.write(f"Corrected P-value at Best Resolution: {best_pval:.4f}\n")

        f.write(f"\nTotal Resolutions Tested: {len(valid_df)}\n")
        f.write(f"  - Coarse Search: {len(coarse_df)} resolutions\n")
        f.write(f"  - Fine Search: {len(fine_df)} resolutions\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("All Results (sorted by resolution):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Resolution':<12} {'CCA Score':<12} {'Pass Type':<12}")
        if has_corrected_pvalues:
            f.write(f" {'Corrected P-value':<18}")
        f.write("\n")

        for _, row in valid_df.iterrows():
            f.write(f"{row['resolution']:<12.3f} {row['cca_score']:<12.4f} {row['pass']:<12}")
            if has_corrected_pvalues and 'corrected_pvalue' in row:
                pval = row['corrected_pvalue']
                pval_str = f"{pval:.4f}" if not np.isnan(pval) else "N/A"
                f.write(f" {pval_str:<18}")
            f.write("\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("Summary Statistics:\n")
        f.write(f"CCA Score Range: [{valid_df['cca_score'].min():.4f}, {valid_df['cca_score'].max():.4f}]\n")
        f.write(f"Mean CCA Score: {valid_df['cca_score'].mean():.4f} ± {valid_df['cca_score'].std():.4f}\n")

        if has_corrected_pvalues:
            valid_pvals = valid_df['corrected_pvalue'].dropna()
            if len(valid_pvals) > 0:
                f.write(f"\nCorrected P-value Statistics:\n")
                f.write(f"Min P-value: {valid_pvals.min():.4f}\n")
                f.write(f"Resolutions with p < 0.05: {(valid_pvals < 0.05).sum()}\n")
                f.write(f"Resolutions with p < 0.01: {(valid_pvals < 0.01).sum()}\n")

    print(f"Summary report saved to: {summary_path}")

    detailed_csv_path = os.path.join(summary_dir, f'detailed_results_{column}.csv')
    df_sorted.to_csv(detailed_csv_path, index=False)
    print(f"Detailed results saved to: {detailed_csv_path}")


def cca_pvalue_test(
    pseudo_adata: AnnData,
    column: str,
    input_correlation: float,
    output_directory: str,
    num_simulations: int = 1000,
    trajectory_col: str = "sev.level",
    verbose: bool = True
):
    """
    Permutation p-value test for a CCA correlation on 2D coordinates.

    Shuffles trajectory labels to build a null distribution of |corr|,
    then computes the fraction of null scores >= input_correlation.
    abs() is used because CCA canonical correlation is sign-invariant.

    Parameters:
        pseudo_adata: AnnData where obs are samples; uns[column] holds coordinates
        column: key in uns for coordinate matrix (uses first 2 columns)
        input_correlation: observed CCA score to test
        output_directory: directory to write plot and result text (CCA_test/ subdirectory)
        num_simulations: number of label permutations
        trajectory_col: obs column with trajectory labels
        verbose: print runtime

    Returns:
        float: permutation p-value
    """
    from pandas.api.types import is_categorical_dtype

    start_time = time.time() if verbose else None

    output_directory = os.path.join(output_directory, "CCA_test")
    os.makedirs(output_directory, exist_ok=True)

    pca_coords = pseudo_adata.uns[column]
    if pca_coords.shape[1] < 2:
        raise ValueError("Coordinates must have at least 2 components for 2D analysis.")

    pca_coords_2d = pca_coords.iloc[:, :2].values if hasattr(pca_coords, "iloc") else pca_coords[:, :2]

    if trajectory_col not in pseudo_adata.obs.columns:
        raise KeyError(f"pseudo_adata.obs must have a '{trajectory_col}' column.")

    sev_levels = pseudo_adata.obs[trajectory_col]

    if is_categorical_dtype(sev_levels):
        sev_levels_numerical = sev_levels.cat.codes.values
    elif sev_levels.dtype == 'object':
        sev_levels_numerical = sev_levels.astype('category').cat.codes.values
    else:
        sev_levels_numerical = sev_levels.values

    if len(sev_levels_numerical) != pca_coords_2d.shape[0]:
        raise ValueError("Mismatch between number of coordinate rows and number of samples.")

    sev_levels_2d = sev_levels_numerical.reshape(-1, 1)

    simulated_scores = []
    for i in range(num_simulations):
        permuted = np.random.permutation(sev_levels_numerical).reshape(-1, 1)
        cca = CCA(n_components=1)
        cca.fit(pca_coords_2d, permuted)
        U, V = cca.transform(pca_coords_2d, permuted)
        # abs() because CCA canonical correlation is sign-invariant
        corr = abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1])
        simulated_scores.append(corr)

    simulated_scores = np.array(simulated_scores)
    p_value = np.mean(simulated_scores >= input_correlation)

    plt.figure(figsize=(8, 5))
    plt.hist(simulated_scores, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(input_correlation, color='red', linestyle='dashed', linewidth=2,
                label=f'Observed corr: {input_correlation:.3f} (p={p_value:.4f})')
    plt.xlabel('Simulated Correlation Scores')
    plt.ylabel('Frequency')
    plt.title('Permutation Test: CCA Correlations')
    plt.legend()
    plot_path = os.path.join(output_directory, f"cca_pvalue_distribution_{column}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    with open(os.path.join(output_directory, f"cca_pvalue_result_{column}.txt"), "w") as f:
        f.write(f"Observed correlation: {input_correlation}\n")
        f.write(f"P-value: {p_value}\n")

    print(f"P-value for observed correlation {input_correlation}: {p_value}")

    if verbose:
        print(f"[CCA p-test] Runtime: {time.time() - start_time:.2f} seconds")

    return p_value
