import anndata as ad
import pandas as pd
from sklearn.cross_decomposition import CCA
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from anndata import AnnData
import time
from pseudobulk import compute_pseudobulk_dataframes
from PCA import process_anndata_with_pca
from CCA import load_severity_levels
from CellType import cell_types, cell_type_assign

def find_optimal_cell_resolution(AnnData_cell, AnnData_sample, output_dir, summary_sample_csv_path, AnnData_sample_path, column):
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
        cell_type_assign(AnnData_cell, AnnData_sample, Save=False, output_dir=output_dir,verbose = False)
        pseudobulk = compute_pseudobulk_dataframes(AnnData_sample, 'batch', 'sample', 'cell_type', output_dir)
        process_anndata_with_pca(adata = AnnData_sample, pseudobulk = pseudobulk, output_dir = output_dir, adata_path=AnnData_sample_path, verbose = False)

        pca_coords = AnnData_sample.uns[column]
        pca_coords_2d = pca_coords[:, :2]  # shape: (n_samples, 2)

        samples = AnnData_sample.obs["sample"].values.unique()
        sev_levels_2d = load_severity_levels(summary_sample_csv_path, samples)

        # CCA on the 2D PCA coordinates vs severity
        cca = CCA(n_components=1)
        cca.fit(pca_coords_2d, sev_levels_2d)
        U, V = cca.transform(pca_coords_2d, sev_levels_2d)
        first_component_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        print(first_component_score)
        score_counter[resolution] = first_component_score
        if 'cell_type' in AnnData_cell.obs.columns:
            del AnnData_cell.obs['cell_type']
        if 'cell_type' in AnnData_sample.obs.columns:
            del AnnData_sample.obs['cell_type']
    
    best_resolution = max(score_counter, key=score_counter.get)
    print(f"Best resolution from first pass: {best_resolution}")
    fine_score_counter = dict()
    for resolution in np.arange(max(0.1, best_resolution - 0.1), min(1.0, best_resolution + 0.1) + 0.01, 0.01):
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
        cell_type_assign(AnnData_cell, AnnData_sample, Save=True, output_dir=output_dir, verbose=False)
        pseudobulk = compute_pseudobulk_dataframes(AnnData_sample, 'batch', 'sample', 'cell_type', output_dir)
        process_anndata_with_pca(adata=AnnData_sample, pseudobulk=pseudobulk, output_dir=output_dir, adata_path=AnnData_sample_path,verbose=False)

        pca_coords = AnnData_sample.uns[column]
        pca_coords_2d = pca_coords[:, :2]  # shape: (n_samples, 2)

        samples = AnnData_sample.obs["sample"].values.unique()
        sev_levels_2d = load_severity_levels(summary_sample_csv_path, samples)

        # CCA on the 2D PCA coordinates vs severity
        cca = CCA(n_components=1)
        cca.fit(pca_coords_2d, sev_levels_2d)
        U, V = cca.transform(pca_coords_2d, sev_levels_2d)
        first_component_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

        print(f"Fine-tuned Resolution {resolution}: Score {first_component_score}")
        fine_score_counter[resolution] = first_component_score
        if 'cell_type' in AnnData_cell.obs.columns:
            del AnnData_cell.obs['cell_type']
        if 'cell_type' in AnnData_sample.obs.columns:
            del AnnData_sample.obs['cell_type']

    # Find the best resolution from the finer search
    final_best_resolution = max(fine_score_counter, key=fine_score_counter.get)
    print(f"Final best resolution: {final_best_resolution}")

    df_coarse = pd.DataFrame(list(score_counter.items()), columns=["resolution", "score"])
    df_fine = pd.DataFrame(list(fine_score_counter.items()), columns=["resolution", "score"])
    df_results = pd.concat([df_coarse, df_fine], ignore_index=True)
    output_dir = os.path.join(output_dir, "CCA test")
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

    # Save plot locally
    plot_path = os.path.join(output_dir, f"resolution_vs_cca_score_{column}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Resolution vs. CCA Score plot saved as '{plot_path}'.")
    print("Resolution scores saved as 'resolution_scores.csv'.")
    print("All data saved locally.")

def cca_pvalue_test(
    adata: AnnData,
    summary_sample_csv_path: str,
    column: str,
    input_correlation: float,
    output_directory: str,
    num_simulations: int = 1000,
    verbose = False
):
    """
    Runs a statistical test to determine whether the observed correlation is significant.
    
    1) Reads the 2D PCA coordinates from `adata.uns[column]`.
    2) Loads and aligns severity levels based on `adata.obs["sample"]`.
    3) Performs a Monte Carlo simulation with randomly generated feature values.
    4) Plots the distribution of simulated correlation scores and saves it.
    5) Computes the p-value for a given user-input correlation and saves it in a text file.

    Parameters
    ----------
    adata : AnnData
        Must contain:
            - `adata.uns[column]` of shape (n_samples, >=2)
            - `adata.obs["sample"]` with sample names
    summary_sample_csv_path : str
        Path to CSV with columns ['sample', 'sev.level'].
    column : str
        The key in `adata.uns` that contains the 2D PCA coordinates.
    input_correlation : float
        The observed correlation to test against the null distribution.
    output_directory : str
        Directory where the plot and p-value text file will be saved.
    num_simulations : int, optional
        Number of Monte Carlo simulations (default is 1000).
    
    Returns
    -------
    p_value : float
        The p-value indicating how extreme the observed correlation is compared to the null distribution.
    """
    start_time = time.time() if verbose else None

    output_directory = os.path.join(output_directory, "CCA test")
    os.makedirs(output_directory, exist_ok=True)

    pca_coords = adata.uns[column]
    if pca_coords.shape[1] < 2:
        raise ValueError("X_pca must have at least 2 components for 2D plotting.")

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

    # Run Monte Carlo simulations
    simulated_scores = []
    for _ in range(num_simulations):
        random_features = np.random.rand(*pca_coords_2d.shape)  # Generate random features
        cca = CCA(n_components=1)
        cca.fit(random_features, sev_levels_2d)
        U, V = cca.transform(random_features, sev_levels_2d)
        first_component_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        simulated_scores.append(first_component_score)

    # Compute p-value
    simulated_scores = np.array(simulated_scores)
    p_value = np.mean(simulated_scores >= input_correlation)

    # Plot the histogram of simulated correlation scores
    plt.figure(figsize=(8, 5))
    plt.hist(simulated_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(input_correlation, color='red', linestyle='dashed', linewidth=2, label=f'Observed correlation: {input_correlation}')
    plt.xlabel('Simulated Correlation Scores')
    plt.ylabel('Frequency')
    plt.title('Monte Carlo Simulated Distribution of CCA Correlations')
    plt.legend()

    # Save plot
    plot_path = os.path.join(output_directory, f"cca_pvalue_distributio_{column}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Save p-value to text file
    p_value_path = os.path.join(output_directory, f"cca_pvalue_result_{column}.txt")
    with open(p_value_path, "w") as f:
        f.write(f"Observed correlation: {input_correlation}\n")
        f.write(f"P-value: {p_value}\n")

    # Print p-value
    print(f"P-value for observed correlation {input_correlation}: {p_value}")

    if verbose:
        print("CCA p-value test on cell proportions completed.")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\n[CCA p-test]Total runtime processing: {elapsed_time:.2f} seconds\n\n")

    return p_value