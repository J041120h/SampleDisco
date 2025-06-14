# test_resolution.py (Or you can define this inside your existing .py script)
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

from anndata import AnnData
from pseudobulk import compute_pseudobulk_dataframes
from DR import process_anndata_with_pca
from CCA import load_severity_levels
from CellType import cell_types, cell_type_assign

def _test_resolution(
    resolution: float,
    AnnData_cell: AnnData,
    AnnData_sample: AnnData,
    summary_sample_csv_path: str,
    output_dir: str,
    column: str,
    sev_col: str = "sev.level",
    sample_col: str = "sample",
    verbose: bool = False
) -> tuple[float, float]:
    """
    Runs the clustering+CCA pipeline on a copy of the AnnData objects for a single resolution.
    Returns (resolution, correlation_score).
    """
    cell_copy = AnnData_cell.copy()
    sample_copy = AnnData_sample.copy()

    # Remove prior annotations
    cell_copy.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
    sample_copy.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')

    # Cluster
    cell_types(
        cell_copy,
        cell_column='cell_type',
        Save=False,
        output_dir=output_dir,
        cluster_resolution=resolution,
        markers=None,
        method='average',
        metric='euclidean',
        distance_mode='centroid',
        num_PCs=20,
        verbose=verbose
    )

    # Assign
    cell_type_assign(cell_copy, sample_copy, Save=False, output_dir=output_dir, verbose=verbose)

    # Pseudobulk
    pseudobulk = compute_pseudobulk_dataframes(sample_copy, 'batch', sample_col, 'cell_type', output_dir)

    # PCA
    process_anndata_with_pca(
        adata=sample_copy,
        pseudobulk=pseudobulk,
        output_dir=output_dir,
        sample_col=sample_col,
        not_save=True,
        verbose=verbose
    )

    pca_coords = sample_copy.uns[column]
    pca_coords_2d = pca_coords.iloc[:, :2].values

    samples = sample_copy.obs[sample_col].unique()
    sev_levels_2d = load_severity_levels(summary_sample_csv_path, samples, sample_col=sample_col, sev_col=sev_col)

    cca = CCA(n_components=1)
    cca.fit(pca_coords_2d, sev_levels_2d)
    U, V = cca.transform(pca_coords_2d, sev_levels_2d)
    score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

    if verbose:
        print(f"Resolution {resolution:.2f} -> CCA Score: {score:.4f}")

    return resolution, score

# find_optimal_cell_resolution.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from anndata import AnnData
import time

def find_optimal_cell_resolution_parallel(
    AnnData_cell: AnnData,
    AnnData_sample: AnnData,
    output_dir: str,
    summary_sample_csv_path: str,
    column: str,
    sev_col: str = "sev.level",
    sample_col: str = "sample",
    n_jobs: int = -1,
    verbose: bool = True
):
    """
    Finds optimal cell clustering resolution by maximizing a CCA-based correlation with severity labels.
    """
    start_time = time.time()

    if verbose:
        print("---- First Pass: [0.1 .. 1.0] by 0.1 ----")

    coarse_resolutions = np.arange(0.1, 1.0 + 1e-9, 0.1)
    coarse_results = Parallel(n_jobs=n_jobs)(
        delayed(_test_resolution)(
            resolution=res,
            AnnData_cell=AnnData_cell,
            AnnData_sample=AnnData_sample,
            summary_sample_csv_path=summary_sample_csv_path,
            output_dir=output_dir,
            column=column,
            sev_col=sev_col,
            sample_col=sample_col,
            verbose=False
        )
        for res in coarse_resolutions
    )
    coarse_score_dict = dict(coarse_results)
    best_resolution = max(coarse_score_dict, key=coarse_score_dict.get)

    if verbose:
        print(f"Best resolution from first pass: {best_resolution:.2f}")
        print("\n---- Second Pass: [best_res - 0.05 .. best_res + 0.05] by 0.01 ----")

    lower = max(0.0, best_resolution - 0.05)
    upper = min(1.0, best_resolution + 0.05)
    fine_resolutions = np.arange(lower, upper + 1e-9, 0.01)
    fine_results = Parallel(n_jobs=n_jobs)(
        delayed(_test_resolution)(
            resolution=res,
            AnnData_cell=AnnData_cell,
            AnnData_sample=AnnData_sample,
            summary_sample_csv_path=summary_sample_csv_path,
            output_dir=output_dir,
            column=column,
            sev_col=sev_col,
            sample_col=sample_col,
            verbose=False
        )
        for res in fine_resolutions
    )
    fine_score_dict = dict(fine_results)
    final_best_resolution = max(fine_score_dict, key=fine_score_dict.get)

    if verbose:
        print(f"Final best resolution: {final_best_resolution:.2f}")

    df_coarse = pd.DataFrame(coarse_results, columns=["resolution", "score"])
    df_fine = pd.DataFrame(fine_results, columns=["resolution", "score"])
    df_results = pd.concat([df_coarse, df_fine], ignore_index=True)

    cca_output_dir = os.path.join(output_dir, "CCA_test")
    os.makedirs(cca_output_dir, exist_ok=True)

    csv_path = os.path.join(cca_output_dir, f"resolution_scores_{column}.csv")
    df_results.to_csv(csv_path, index=False)

    plot_path = os.path.join(cca_output_dir, f"resolution_vs_cca_score_{column}.png")
    plt.figure(figsize=(8, 6))
    plt.plot(df_results["resolution"], df_results["score"], marker='o', linestyle='None', label="CCA Score")
    plt.xlabel("Resolution")
    plt.ylabel("CCA Score")
    plt.title("Resolution vs. CCA Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path, dpi=300)
    plt.close()

    if verbose:
        print(f"\nPlot saved: {plot_path}")
        print(f"Scores saved: {csv_path}")
        print(f"\n[Find Optimal Resolution] Total runtime: {time.time() - start_time:.2f} seconds")

    return final_best_resolution