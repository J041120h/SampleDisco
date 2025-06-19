import anndata as ad
import pandas as pd
from sklearn.cross_decomposition import CCA
import numpy as np
import os
import matplotlib.pyplot as plt
from anndata import AnnData
import time
from pseudobulk import compute_pseudobulk_dataframes
from DR import process_anndata_with_pca
from CellType import cell_types, cell_type_assign
from pseudo_adata import compute_pseudobulk_adata 

def find_optimal_cell_resolution(
    AnnData_cell,
    AnnData_sample,
    output_dir,
    summary_sample_csv_path,
    column,
    sev_col: str = "sev.level",
    sample_col: str = "sample"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import CCA
    import os
    import time

    start_time = time.time()
    score_counter = dict()

    for resolution in np.arange(0.01, 1.01, 0.01):
        print(f"\n\nTesting resolution: {resolution}\n\n")
        cell_types(
            AnnData_cell,
            cell_column='cell_type',
            Save=False,
            output_dir=output_dir,
            cluster_resolution=resolution,
            markers=None,
            method='average',
            metric='euclidean',
            distance_mode='centroid',
            num_PCs=20,
            verbose=False
        )
        cell_type_assign(AnnData_cell, AnnData_sample, Save=False, output_dir=output_dir, verbose=False)
        pseudobulk = compute_pseudobulk_adata(AnnData_sample, 'batch', sample_col, 'cell_type', output_dir)
        process_anndata_with_pca(
            adata=AnnData_sample,
            pseudobulk=pseudobulk,
            output_dir=output_dir,
            sample_col=sample_col,
            not_save=True,
            verbose=False
        )

        pca_coords = AnnData_sample.uns[column]
        pca_coords_2d = pca_coords.iloc[:, :2].values
        samples = AnnData_sample.obs[sample_col].unique()

        sev_levels_2d = load_severity_levels(summary_sample_csv_path, samples, sample_col=sample_col, sev_col=sev_col)
        cca = CCA(n_components=1)
        cca.fit(pca_coords_2d, sev_levels_2d)
        U, V = cca.transform(pca_coords_2d, sev_levels_2d)
        first_component_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        print(first_component_score)
        score_counter[resolution] = first_component_score

        AnnData_cell.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
        AnnData_sample.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')

    best_resolution = max(score_counter, key=score_counter.get)
    print(f"Best resolution from first pass: {best_resolution}")
    fine_score_counter = dict()

    for resolution in np.arange(max(0.1, best_resolution - 0.05), min(1.0, best_resolution + 0.05) + 0.01, 0.01):
        cell_types(
            AnnData_cell,
            cell_column='cell_type',
            Save=False,
            output_dir=output_dir,
            cluster_resolution=resolution,
            markers=None,
            method='average',
            metric='euclidean',
            distance_mode='centroid',
            num_PCs=20,
            verbose=False
        )
        cell_type_assign(AnnData_cell, AnnData_sample, Save=False, output_dir=output_dir, verbose=False)
        pseudobulk = compute_pseudobulk_dataframes(AnnData_sample, 'batch', sample_col, 'cell_type', output_dir)
        process_anndata_with_pca(
            adata=AnnData_sample,
            pseudobulk=pseudobulk,
            output_dir=output_dir,
            sample_col=sample_col,
            not_save=True,
            verbose=False
        )

        pca_coords = AnnData_sample.uns[column]
        pca_coords_2d = pca_coords.iloc[:, :2].values
        samples = AnnData_sample.obs[sample_col].unique()

        sev_levels_2d = load_severity_levels(summary_sample_csv_path, samples, sample_col=sample_col, sev_col=sev_col)
        cca = CCA(n_components=1)
        cca.fit(pca_coords_2d, sev_levels_2d)
        U, V = cca.transform(pca_coords_2d, sev_levels_2d)
        first_component_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

        print(f"Fine-tuned Resolution {resolution:.2f}: Score {first_component_score}")
        fine_score_counter[resolution] = first_component_score

        AnnData_cell.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
        AnnData_sample.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')

    final_best_resolution = max(fine_score_counter, key=fine_score_counter.get)
    print(f"Final best resolution: {final_best_resolution}")

    df_coarse = pd.DataFrame(score_counter.items(), columns=["resolution", "score"])
    df_fine = pd.DataFrame(fine_score_counter.items(), columns=["resolution", "score"])
    df_results = pd.concat([df_coarse, df_fine], ignore_index=True)

    output_dir = os.path.join(output_dir, "CCA_test")
    os.makedirs(output_dir, exist_ok=True)
    to_csv_path = os.path.join(output_dir, f"resolution_scores_{column}.csv")
    df_results.to_csv(to_csv_path, index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(df_results["resolution"], df_results["score"], marker='o', linestyle='None', color='b', label="CCA Score")
    plt.xlabel("Resolution")
    plt.ylabel("CCA Score")
    plt.title("Resolution vs. CCA Score")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_dir, f"resolution_vs_cca_score_{column}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Plot saved to: {plot_path}")
    print(f"Resolution scores saved to: {to_csv_path}")

    print(f"\n[Find Optimal Resolution] Total runtime: {time.time() - start_time:.2f} seconds\n")

    return final_best_resolution

def cca_pvalue_test(
    pseudo_adata: AnnData,
    column: str,
    input_correlation: float,
    output_directory: str,
    num_simulations: int = 1000,
    sev_col: str = "sev.level",
    verbose: bool = True
):
    """
    Perform CCA p-value test using pseudo anndata (sample by gene).
    
    Parameters:
    -----------
    pseudo_adata : AnnData
        Pseudo anndata object where observations are samples and variables are genes.
        Must contain severity levels in pseudo_adata.obs[sev_col].
    column : str
        Key in pseudo_adata.uns containing the coordinates (e.g., PCA coordinates)
    input_correlation : float
        Observed correlation to test against
    output_directory : str
        Directory to save results
    num_simulations : int
        Number of permutation simulations (default: 1000)
    sev_col : str
        Column name for severity levels in pseudo_adata.obs (default: "sev.level")
    verbose : bool
        Whether to print timing information (default: True)
    
    Returns:
    --------
    float
        P-value from permutation test
    """
    import os
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import CCA
    
    start_time = time.time() if verbose else None
    output_directory = os.path.join(output_directory, "CCA_test")
    os.makedirs(output_directory, exist_ok=True)
    
    # Extract coordinates from pseudo_adata.uns
    pca_coords = pseudo_adata.uns[column]
    if pca_coords.shape[1] < 2:
        raise ValueError("Coordinates must have at least 2 components for 2D analysis.")
    
    # Get first 2 components
    pca_coords_2d = pca_coords.iloc[:, :2].values if hasattr(pca_coords, "iloc") else pca_coords[:, :2]
    
    # Check if severity column exists
    if sev_col not in pseudo_adata.obs.columns:
        raise KeyError(f"pseudo_adata.obs must have a '{sev_col}' column.")
    
    # Get severity levels directly from pseudo_adata.obs
    sev_levels = pseudo_adata.obs[sev_col].values
    
    if len(sev_levels) != pca_coords_2d.shape[0]:
        raise ValueError("Mismatch between number of coordinate rows and number of samples.")
    
    # Reshape for CCA (needs 2D array)
    sev_levels_1d = sev_levels.flatten()
    
    # Perform permutation test
    simulated_scores = []
    for _ in range(num_simulations):
        permuted = np.random.permutation(sev_levels_1d).reshape(-1, 1)
        cca = CCA(n_components=1)
        cca.fit(pca_coords_2d, permuted)
        U, V = cca.transform(pca_coords_2d, permuted)
        corr = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        simulated_scores.append(corr)
    
    simulated_scores = np.array(simulated_scores)
    p_value = np.mean(simulated_scores >= input_correlation)
    
    # Plot the permutation distribution
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
    
    # Save results to file
    with open(os.path.join(output_directory, f"cca_pvalue_result_{column}.txt"), "w") as f:
        f.write(f"Observed correlation: {input_correlation}\n")
        f.write(f"P-value: {p_value}\n")
    
    print(f"P-value for observed correlation {input_correlation}: {p_value}")
    
    if verbose:
        print(f"[CCA p-test] Runtime: {time.time() - start_time:.2f} seconds")
    
    return p_value