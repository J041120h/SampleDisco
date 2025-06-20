import anndata as ad
import pandas as pd
from sklearn.cross_decomposition import CCA
import numpy as np
import os
import matplotlib.pyplot as plt
from anndata import AnnData
import time
from DR import process_anndata_with_pca
from CellType import cell_types, cell_type_assign
from pseudo_adata import compute_pseudobulk_adata
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
import warnings
import copy
from typing import Dict, Tuple, Optional
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global lock for thread-safe operations
result_lock = Lock()

def process_single_resolution(
    resolution: float,
    AnnData_cell: AnnData,
    AnnData_sample: AnnData,
    output_dir: str,
    column: str,
    sev_col: str,
    sample_col: str,
    verbose: bool = False
) -> Tuple[float, Optional[float]]:
    """
    Process a single resolution value and return the CCA score.
    
    Parameters:
    -----------
    resolution : float
        Clustering resolution to test
    AnnData_cell : AnnData
        Cell-level AnnData object
    AnnData_sample : AnnData  
        Sample-level AnnData object
    output_dir : str
        Output directory for results
    column : str
        Column name in adata.uns for dimension reduction results
    sev_col : str
        Column name for severity levels in pseudobulk_anndata.obs
    sample_col : str
        Column name for sample identifiers
    verbose : bool
        Whether to print detailed progress
        
    Returns:
    --------
    Tuple[float, Optional[float]]
        Resolution value and CCA score (None if error)
    """
    try:
        # Create deep copies to avoid thread conflicts
        adata_cell_copy = AnnData_cell.copy()
        adata_sample_copy = AnnData_sample.copy()
        
        # Clean up previous cell type assignments
        if 'cell_type' in adata_cell_copy.obs.columns:
            adata_cell_copy.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
        if 'cell_type' in adata_sample_copy.obs.columns:
            adata_sample_copy.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
        
        # Perform clustering
        cell_types(
            adata_cell_copy,
            cell_column='cell_type',
            Save=False,
            output_dir=resolution_dir,
            cluster_resolution=resolution,
            markers=None,
            method='average',
            metric='euclidean',
            distance_mode='centroid',
            num_PCs=20,
            verbose=verbose
        )
        
        # Assign cell types to samples
        cell_type_assign(adata_cell_copy, adata_sample_copy, Save=False, 
                        output_dir=resolution_dir, verbose=verbose)
        
        # Compute pseudobulk data
        pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata(
            adata=adata_sample_copy, 
            batch_col='batch', 
            sample_col=sample_col, 
            celltype_col='cell_type', 
            output_dir=resolution_dir,
            Save=False,
            verbose=verbose
        )
        
        # Perform dimension reduction
        process_anndata_with_pca(
            adata=adata_sample_copy,
            pseudobulk=pseudobulk_dict,
            pseudobulk_anndata=pseudobulk_adata,
            sample_col=sample_col,
            output_dir=resolution_dir,
            not_save=True,
            verbose=verbose
        )

        # Get PCA coordinates
        if column not in adata_sample_copy.uns:
            logger.warning(f"Column {column} not found in AnnData.uns for resolution {resolution:.3f}")
            return resolution, None
            
        pca_coords = adata_sample_copy.uns[column]
        if hasattr(pca_coords, 'iloc'):
            pca_coords_2d = pca_coords.iloc[:, :2].values
        else:
            pca_coords_2d = pca_coords[:, :2]
        
        # Get severity levels
        if sev_col not in pseudobulk_adata.obs.columns:
            logger.warning(f"Severity column {sev_col} not found for resolution {resolution:.3f}")
            return resolution, None
            
        sev_levels = pd.to_numeric(pseudobulk_adata.obs[sev_col], errors='coerce').values
        missing = np.isnan(sev_levels).sum()
        if missing > 0:
            if verbose:
                logger.info(f"Imputing {missing} missing severity values for resolution {resolution:.3f}")
            sev_levels[np.isnan(sev_levels)] = np.nanmean(sev_levels)
        
        sev_levels_2d = sev_levels.reshape(-1, 1)
        
        # Ensure matching dimensions
        if len(sev_levels_2d) != pca_coords_2d.shape[0]:
            logger.warning(f"Dimension mismatch at resolution {resolution:.3f}")
            return resolution, None
        
        # Perform CCA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cca = CCA(n_components=1)
            cca.fit(pca_coords_2d, sev_levels_2d)
            U, V = cca.transform(pca_coords_2d, sev_levels_2d)
            first_component_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        
        # Clean up temporary directory
        try:
            import shutil
            shutil.rmtree(resolution_dir)
        except:
            pass
            
        return resolution, first_component_score
        
    except Exception as e:
        logger.error(f"Error processing resolution {resolution:.3f}: {str(e)}")
        return resolution, None

def find_optimal_cell_resolution(
    AnnData_cell: AnnData,
    AnnData_sample: AnnData,
    output_dir: str,
    column: str,
    sev_col: str = "sev.level",
    sample_col: str = "sample",
    n_threads: int = None,
    use_processes: bool = False,
    verbose: bool = True
) -> float:
    """
    Find optimal clustering resolution by maximizing CCA correlation between 
    dimension reduction and severity levels using multithreading.
    
    Parameters:
    -----------
    AnnData_cell : AnnData
        Cell-level AnnData object
    AnnData_sample : AnnData  
        Sample-level AnnData object
    output_dir : str
        Output directory for results
    column : str
        Column name in adata.uns for dimension reduction results
    sev_col : str
        Column name for severity levels in pseudobulk_anndata.obs
    sample_col : str
        Column name for sample identifiers
    n_threads : int, optional
        Number of threads to use. If None, uses CPU count
    use_processes : bool
        Whether to use processes instead of threads (for CPU-bound operations)
    verbose : bool
        Whether to print detailed progress
        
    Returns:
    --------
    float
        Optimal resolution value
    """
    start_time = time.time()
    score_counter = {}
    
    if n_threads is None:
        n_threads = os.cpu_count() or 4
    
    print(f"Starting resolution optimization for {column}...")
    print(f"Using {n_threads} {'processes' if use_processes else 'threads'}")
    print(f"Testing resolutions from 0.01 to 1.00...")

    # Create executor (process or thread based)
    ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    # First pass: coarse search
    coarse_resolutions = np.arange(0.1, 1.01, 0.1)
    
    with ExecutorClass(max_workers=n_threads) as executor:
        # Submit all tasks
        future_to_resolution = {
            executor.submit(
                process_single_resolution,
                resolution,
                AnnData_cell,
                AnnData_sample,
                output_dir,
                column,
                sev_col,
                sample_col,
                verbose
            ): resolution
            for resolution in coarse_resolutions
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_resolution):
            resolution = future_to_resolution[future]
            try:
                res, score = future.result()
                if score is not None:
                    with result_lock:
                        score_counter[res] = score
                    print(f"Resolution {res:.2f}: CCA Score = {score:.4f}")
                else:
                    print(f"Resolution {res:.2f}: Failed to compute score")
            except Exception as exc:
                print(f"Resolution {resolution:.2f} generated an exception: {exc}")

    if not score_counter:
        raise ValueError("No valid CCA scores obtained. Check your data and parameters.")

    # Find best resolution from first pass
    best_resolution = max(score_counter, key=score_counter.get)
    print(f"\nBest resolution from first pass: {best_resolution:.2f}")
    print(f"Best CCA score: {score_counter[best_resolution]:.4f}")

    # Second pass: fine-tuned search around best resolution
    fine_score_counter = {}
    search_range_start = max(0.01, best_resolution - 0.05)
    search_range_end = min(1.00, best_resolution + 0.05)
    
    print(f"\nFine-tuning search from {search_range_start:.2f} to {search_range_end:.2f}...")
    
    # Generate fine resolutions
    fine_resolutions = np.arange(search_range_start, search_range_end + 0.001, 0.01)
    fine_resolutions = [round(r, 3) for r in fine_resolutions]
    
    # Batch processing for fine-tuning to avoid overwhelming the system
    batch_size = min(n_threads * 2, len(fine_resolutions))
    
    with ExecutorClass(max_workers=n_threads) as executor:
        for i in range(0, len(fine_resolutions), batch_size):
            batch = fine_resolutions[i:i + batch_size]
            
            future_to_resolution = {
                executor.submit(
                    process_single_resolution,
                    resolution,
                    AnnData_cell,
                    AnnData_sample,
                    output_dir,
                    column,
                    sev_col,
                    sample_col,
                    False  # Less verbose for fine-tuning
                ): resolution
                for resolution in batch
            }
            
            for future in as_completed(future_to_resolution):
                resolution = future_to_resolution[future]
                try:
                    res, score = future.result()
                    if score is not None:
                        with result_lock:
                            fine_score_counter[res] = score
                        if verbose:
                            print(f"Fine-tuned Resolution {res:.3f}: Score {score:.4f}")
                except Exception as exc:
                    if verbose:
                        print(f"Fine-tuned resolution {resolution:.3f} generated an exception: {exc}")

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
    
    to_csv_path = os.path.join(output_dir_results, f"resolution_scores_{column}.csv")
    df_results.to_csv(to_csv_path, index=False)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["resolution"], df_results["score"], marker='o', markersize=3, 
             linestyle='-', linewidth=1, color='b', label="CCA Score")
    plt.axvline(x=final_best_resolution, color='r', linestyle='--', 
                label=f'Best Resolution: {final_best_resolution:.3f}')
    plt.xlabel("Resolution")
    plt.ylabel("CCA Score")
    plt.title(f"Resolution vs. CCA Score ({column})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir_results, f"resolution_vs_cca_score_{column}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {plot_path}")
    print(f"Resolution scores saved to: {to_csv_path}")
    print(f"\n[Find Optimal Resolution] Total runtime: {time.time() - start_time:.2f} seconds\n")

    return final_best_resolution


# Additional utility function for running multiple columns in parallel
def find_optimal_resolutions_multiple_columns(
    AnnData_cell: AnnData,
    AnnData_sample: AnnData,
    output_dir: str,
    columns: list,
    sev_col: str = "sev.level",
    sample_col: str = "sample",
    n_threads: int = None,
    use_processes: bool = False
) -> Dict[str, float]:
    """
    Find optimal resolutions for multiple columns in parallel.
    
    Parameters:
    -----------
    columns : list
        List of column names to process
    Other parameters same as find_optimal_cell_resolution
    
    Returns:
    --------
    Dict[str, float]
        Dictionary mapping column names to optimal resolutions
    """
    results = {}
    
    # Use half the threads for column-level parallelism, half for resolution-level
    if n_threads is None:
        n_threads = os.cpu_count() or 4
    
    column_threads = max(1, n_threads // 2)
    resolution_threads = max(1, n_threads // len(columns)) if len(columns) > 1 else n_threads
    
    with ThreadPoolExecutor(max_workers=column_threads) as executor:
        future_to_column = {
            executor.submit(
                find_optimal_cell_resolution,
                AnnData_cell,
                AnnData_sample,
                output_dir,
                column,
                sev_col,
                sample_col,
                resolution_threads,
                use_processes
            ): column
            for column in columns
        }
        
        for future in as_completed(future_to_column):
            column = future_to_column[future]
            try:
                optimal_res = future.result()
                results[column] = optimal_res
                print(f"Completed optimization for {column}: {optimal_res:.3f}")
            except Exception as exc:
                print(f"Column {column} generated an exception: {exc}")
                results[column] = None
    
    return results

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