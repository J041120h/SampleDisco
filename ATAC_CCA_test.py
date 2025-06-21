import anndata as ad
import pandas as pd
from sklearn.cross_decomposition import CCA
import numpy as np
import os
import matplotlib.pyplot as plt
from anndata import AnnData
import time
from DR import process_anndata_with_pca
from ATAC_cell_type import cell_types_atac, cell_type_assign_atac
from pseudo_adata import compute_pseudobulk_adata

def find_optimal_cell_resolution_atac(
    AnnData_cell: AnnData,
    AnnData_sample: AnnData,
    output_dir: str,
    column: str,
    sev_col: str = "sev.level",
    batch_col: str = None,
    sample_col: str = "sample",
    use_rep: str = 'X_DM_harmony',
    num_DMs: int = 20
) -> float:
    """
    Find optimal clustering resolution by maximizing CCA correlation between 
    dimension reduction and severity levels for ATAC data.
    
    Parameters:
    -----------
    AnnData_cell : AnnData
        Cell-level AnnData object (ATAC)
    AnnData_sample : AnnData  
        Sample-level AnnData object (ATAC)
    output_dir : str
        Output directory for results
    column : str
        Column name in adata.uns for dimension reduction results
    sev_col : str
        Column name for severity levels in pseudobulk_anndata.obs
    sample_col : str
        Column name for sample identifiers
    use_rep : str
        Representation to use for neighborhood graph (default: 'X_DM_harmony')
    num_DMs : int
        Number of diffusion map components for neighborhood graph
        
    Returns:
    --------
    float
        Optimal resolution value
    """
    start_time = time.time()
    score_counter = dict()

    print(f"Starting ATAC resolution optimization for {column}...")
    print(f"Using representation: {use_rep} with {num_DMs} components")
    print(f"Testing resolutions from 0.01 to 1.00...")

    # First pass: coarse search
    for resolution in np.arange(0.1, 1.01, 0.1):
        print(f"\n\nTesting resolution: {resolution:.2f}\n")
        
        try:
            # Clean up previous cell type assignments
            if 'cell_type' in AnnData_cell.obs.columns:
                AnnData_cell.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            if 'cell_type' in AnnData_sample.obs.columns:
                AnnData_sample.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            
            # Perform clustering using ATAC-specific function
            cell_types_atac(
                AnnData_sample,
                cell_column='cell_type',
                Save=False,
                output_dir=output_dir,
                cluster_resolution=resolution,
                use_rep=use_rep,
                peaks=None,
                method='average',
                metric='euclidean',
                distance_mode='centroid',
                num_DMs=num_DMs,
                verbose=False
            )
            
            # Compute pseudobulk data using updated function
            pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata(
                adata=AnnData_sample, 
                batch_col=batch_col, 
                sample_col=sample_col, 
                celltype_col='cell_type', 
                output_dir=output_dir,
                Save = False,
                verbose=False
            )
            
            # Perform dimension reduction using updated function
            process_anndata_with_pca(
                adata=AnnData_sample,
                pseudobulk=pseudobulk_dict,
                pseudobulk_anndata=pseudobulk_adata,
                sample_col=sample_col,
                atac = True,
                output_dir=output_dir,
                not_save=True,
                verbose=False
            )

            # Get coordinates from the updated location
            if column not in AnnData_sample.uns:
                print(f"Warning: {column} not found in AnnData_sample.uns. Skipping resolution {resolution:.2f}")
                continue
                
            coords = AnnData_sample.uns[column]
            if hasattr(coords, 'iloc'):
                coords_2d = coords.iloc[:, :2].values
            else:
                coords_2d = coords[:, :2]
            
            # Get severity levels directly from pseudobulk_adata
            if sev_col not in pseudobulk_adata.obs.columns:
                print(f"Warning: {sev_col} not found in pseudobulk_adata.obs. Skipping resolution {resolution:.2f}")
                continue
                
            sev_levels = pd.to_numeric(pseudobulk_adata.obs[sev_col], errors='coerce').values
            missing = np.isnan(sev_levels).sum()
            if missing > 0:
                print(f"Warning: {missing} sample(s) missing severity level. Imputing with mean.")
                sev_levels[np.isnan(sev_levels)] = np.nanmean(sev_levels)
            
            sev_levels_2d = sev_levels.reshape(-1, 1)
            
            # Ensure matching dimensions
            if len(sev_levels_2d) != coords_2d.shape[0]:
                print(f"Warning: Dimension mismatch at resolution {resolution:.2f}. Skipping.")
                continue
            
            # Perform CCA
            cca = CCA(n_components=1)
            cca.fit(coords_2d, sev_levels_2d)
            U, V = cca.transform(coords_2d, sev_levels_2d)
            first_component_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
            
            print(f"Resolution {resolution:.2f}: CCA Score = {first_component_score:.4f}")
            score_counter[resolution] = first_component_score
            
        except Exception as e:
            print(f"Error at resolution {resolution:.2f}: {str(e)}")
            continue

    if not score_counter:
        raise ValueError("No valid CCA scores obtained. Check your ATAC data and parameters.")

    # Find best resolution from first pass
    best_resolution = max(score_counter, key=score_counter.get)
    print(f"\nBest resolution from first pass: {best_resolution:.2f}")
    print(f"Best CCA score: {score_counter[best_resolution]:.4f}")

    # Second pass: fine-tuned search around best resolution
    fine_score_counter = dict()
    search_range_start = max(0.01, best_resolution - 0.05)
    search_range_end = min(1.00, best_resolution + 0.05)
    
    print(f"\nFine-tuning search from {search_range_start:.2f} to {search_range_end:.2f}...")

    for resolution in np.arange(search_range_start, search_range_end + 0.001, 0.01):
        resolution = round(resolution, 3)  # Avoid floating point precision issues
        
        try:
            # Clean up previous cell type assignments
            if 'cell_type' in AnnData_cell.obs.columns:
                AnnData_cell.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            if 'cell_type' in AnnData_sample.obs.columns:
                AnnData_sample.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            
            # Perform clustering using ATAC-specific function
            cell_types_atac(
                AnnData_sample,
                cell_column='cell_type',
                Save=False,
                output_dir=output_dir,
                cluster_resolution=resolution,
                use_rep=use_rep,
                peaks=None,
                method='average',
                metric='euclidean',
                distance_mode='centroid',
                num_DMs=num_DMs,
                verbose=False
            )
            
            # Compute pseudobulk data using updated function
            pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata(
                adata=AnnData_sample, 
                batch_col='batch', 
                sample_col=sample_col, 
                celltype_col='cell_type', 
                output_dir=output_dir,
                Save = False,
                verbose=False
            )
            
            # Perform dimension reduction using updated function
            process_anndata_with_pca(
                adata=AnnData_sample,
                pseudobulk=pseudobulk_dict,
                pseudobulk_anndata=pseudobulk_adata,
                sample_col=sample_col,
                atac = True,
                output_dir=output_dir,
                not_save=True,
                verbose=False
            )

            # Get coordinates
            if column not in AnnData_sample.uns:
                continue
                
            coords = AnnData_sample.uns[column]
            if hasattr(coords, 'iloc'):
                coords_2d = coords.iloc[:, :2].values
            else:
                coords_2d = coords[:, :2]

            # Get severity levels directly from pseudobulk_adata
            if sev_col not in pseudobulk_adata.obs.columns:
                continue
                
            sev_levels = pd.to_numeric(pseudobulk_adata.obs[sev_col], errors='coerce').values
            missing = np.isnan(sev_levels).sum()
            if missing > 0:
                print(f"Warning: {missing} sample(s) missing severity level. Imputing with mean.")
                sev_levels[np.isnan(sev_levels)] = np.nanmean(sev_levels)
            
            sev_levels_2d = sev_levels.reshape(-1, 1)
            
            # Ensure matching dimensions
            if len(sev_levels_2d) != coords_2d.shape[0]:
                continue
            
            # Perform CCA
            cca = CCA(n_components=1)
            cca.fit(coords_2d, sev_levels_2d)
            U, V = cca.transform(coords_2d, sev_levels_2d)
            first_component_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

            print(f"Fine-tuned Resolution {resolution:.3f}: Score {first_component_score:.4f}")
            fine_score_counter[resolution] = first_component_score
            
        except Exception as e:
            print(f"Error at fine-tuned resolution {resolution:.3f}: {str(e)}")
            continue

    if not fine_score_counter:
        print("Warning: Fine-tuning failed. Using coarse search results.")
        final_best_resolution = best_resolution
        final_results = score_counter
    else:
        final_best_resolution = max(fine_score_counter, key=fine_score_counter.get)
        final_results = {**score_counter, **fine_score_counter}

    print(f"\nFinal best resolution: {final_best_resolution:.3f}")
    print(f"Final best CCA score: {final_results[final_best_resolution]:.4f}")

    # Save results
    df_results = pd.DataFrame(final_results.items(), columns=["resolution", "score"])
    df_results = df_results.sort_values("resolution")

    output_dir_results = os.path.join(output_dir, "CCA_test")
    os.makedirs(output_dir_results, exist_ok=True)
    
    to_csv_path = os.path.join(output_dir_results, f"resolution_scores_atac_{column}.csv")
    df_results.to_csv(to_csv_path, index=False)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["resolution"], df_results["score"], marker='o', markersize=3, 
             linestyle='-', linewidth=1, color='b', label="CCA Score")
    plt.axvline(x=final_best_resolution, color='r', linestyle='--', 
                label=f'Best Resolution: {final_best_resolution:.3f}')
    plt.xlabel("Resolution")
    plt.ylabel("CCA Score")
    plt.title(f"ATAC Resolution vs. CCA Score ({column})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir_results, f"resolution_vs_cca_score_atac_{column}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {plot_path}")
    print(f"Resolution scores saved to: {to_csv_path}")
    print(f"\n[Find Optimal Resolution ATAC] Total runtime: {time.time() - start_time:.2f} seconds\n")

    return final_best_resolution

def cca_pvalue_test_atac(
    pseudo_adata: AnnData,
    column: str,
    input_correlation: float,
    output_directory: str,
    num_simulations: int = 1000,
    sev_col: str = "sev.level",
    verbose: bool = True
):
    """
    Perform CCA p-value test using pseudo anndata (sample by gene) for ATAC data.
    
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
    coords = pseudo_adata.uns[column]
    if coords.shape[1] < 2:
        raise ValueError("Coordinates must have at least 2 components for 2D analysis.")
    
    # Get first 2 components
    coords_2d = coords.iloc[:, :2].values if hasattr(coords, "iloc") else coords[:, :2]
    
    # Check if severity column exists
    if sev_col not in pseudo_adata.obs.columns:
        raise KeyError(f"pseudo_adata.obs must have a '{sev_col}' column.")
    
    # Get severity levels directly from pseudo_adata.obs
    sev_levels = pseudo_adata.obs[sev_col].values
    
    if len(sev_levels) != coords_2d.shape[0]:
        raise ValueError("Mismatch between number of coordinate rows and number of samples.")
    
    # Reshape for CCA (needs 2D array)
    sev_levels_1d = sev_levels.flatten()
    
    # Perform permutation test
    simulated_scores = []
    for _ in range(num_simulations):
        permuted = np.random.permutation(sev_levels_1d).reshape(-1, 1)
        cca = CCA(n_components=1)
        cca.fit(coords_2d, permuted)
        U, V = cca.transform(coords_2d, permuted)
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
    plt.title('ATAC Permutation Test: CCA Correlations')
    plt.legend()
    plot_path = os.path.join(output_directory, f"cca_pvalue_distribution_atac_{column}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # Save results to file
    with open(os.path.join(output_directory, f"cca_pvalue_result_atac_{column}.txt"), "w") as f:
        f.write(f"Observed correlation: {input_correlation}\n")
        f.write(f"P-value: {p_value}\n")
    
    print(f"P-value for observed correlation {input_correlation}: {p_value}")
    
    if verbose:
        print(f"[CCA p-test ATAC] Runtime: {time.time() - start_time:.2f} seconds")
    
    return p_value