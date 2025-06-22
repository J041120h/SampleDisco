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
from CCA import *
from CCA_test import *

def find_optimal_cell_resolution_atac(
    AnnData_cell: AnnData,
    AnnData_sample: AnnData,
    output_dir: str,
    column: str,
    n_features: int = 40000,
    sev_col: str = "sev.level",
    batch_col: str = None,
    sample_col: str = "sample",
    use_rep: str = 'X_DM_harmony',
    num_DR_components: int = 30,
    num_DMs: int = 20,
    num_pvalue_simulations: int = 100,
    compute_pvalues: bool = True
) -> tuple:
    """
    Find optimal clustering resolution by maximizing CCA correlation between 
    dimension reduction and severity levels for ATAC data, with p-value tracking.
    
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
    num_pvalue_simulations : int
        Number of simulations for p-value calculation (default: 100)
    compute_pvalues : bool
        Whether to compute p-values for each resolution (default: True)
        
    Returns:
    --------
    tuple: (optimal_resolution, results_dataframe)
    """
    start_time = time.time()

    try:
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator
        import cupy as cp
        
        rmm.reinitialize(
            managed_memory=True,
            pool_allocator=False,
        )
        cp.cuda.set_allocator(rmm_cupy_allocator)
    except:
        pass
    
    # Create subdirectories for different outputs
    main_output_dir = os.path.join(output_dir, "CCA_resolution_optimization")
    resolution_plots_dir = os.path.join(main_output_dir, "resolution_plots")
    pvalue_results_dir = os.path.join(main_output_dir, "pvalue_results")
    
    for dir_path in [main_output_dir, resolution_plots_dir, pvalue_results_dir]:
        os.makedirs(dir_path, exist_ok=True)

    print(f"Starting ATAC resolution optimization for {column}...")
    print(f"Using representation: {use_rep} with {num_DMs} components")
    print(f"Testing resolutions from 0.01 to 1.00...")
    if compute_pvalues:
        print(f"Computing p-values with {num_pvalue_simulations} simulations per resolution")

    all_results = []

    print("\n=== FIRST PASS: Coarse Search ===")
    for resolution in np.arange(0.1, 1.01, 0.1):
        print(f"\n\nTesting resolution: {resolution:.2f}\n")
        
        result_dict = {
            'resolution': resolution,
            'cca_score': np.nan,
            'p_value': np.nan,
            'pass': 'coarse'
        }
        
        try:
            # Clean up previous cell type assignments
            if 'cell_type' in AnnData_cell.obs.columns:
                AnnData_cell.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            if 'cell_type' in AnnData_sample.obs.columns:
                AnnData_sample.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            
            # Perform clustering using ATAC-specific function
            AnnData_sample = cell_types_atac(
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
            
            # Compute pseudobulk data
            pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata(
                adata=AnnData_sample, 
                batch_col=batch_col, 
                sample_col=sample_col, 
                celltype_col='cell_type', 
                n_features=n_features, 
                output_dir=output_dir,
                Save=False,
                verbose=False
            )
            
            # Perform dimension reduction
            process_anndata_with_pca(
                adata=AnnData_sample,
                pseudobulk=pseudobulk_dict,
                pseudobulk_anndata=pseudobulk_adata,
                sample_col=sample_col,
                n_expression_pcs=num_DR_components,
                n_proportion_pcs=num_DR_components,
                atac=True,
                output_dir=output_dir,
                not_save=True,
                verbose=False
            )

            # Check if column exists in AnnData_sample.uns
            if column not in AnnData_sample.uns:
                print(f"Warning: {column} not found in AnnData_sample.uns. Skipping resolution {resolution:.2f}")
                all_results.append(result_dict)
                continue
            
            # Use the existing CCA function to run analysis
            try:
                (pca_coords_2d, 
                 sev_levels, 
                 cca_model, 
                 cca_score, 
                 samples) = run_cca_on_2d_pca_from_adata(
                    adata=pseudobulk_adata,  # Use pseudobulk_adata which has the severity info
                    column=column,
                    sev_col=sev_col
                )
                
                result_dict['cca_score'] = cca_score
                print(f"Resolution {resolution:.2f}: CCA Score = {cca_score:.4f}")
                
                # Save CCA plot for this resolution
                plot_path = os.path.join(resolution_plots_dir, f"cca_plot_res_{resolution:.2f}_{column}.png")
                plot_cca_on_2d_pca(
                    pca_coords_2d=pca_coords_2d,
                    sev_levels=sev_levels,
                    cca=cca_model,
                    output_path=plot_path,
                    sample_labels=None,
                    title_suffix=f"Resolution {resolution:.2f}"
                )
                
                p_value = cca_pvalue_test(
                    pseudo_adata=pseudobulk_adata,
                    column=column,
                    input_correlation=cca_score,
                    output_directory=pvalue_results_dir,
                    num_simulations=num_pvalue_simulations,
                    sev_col=sev_col,
                    verbose=False
                )
                result_dict['p_value'] = p_value
                print(f"Resolution {resolution:.2f}: p-value = {p_value:.4f}")
                
            except Exception as e:
                print(f"Error in CCA analysis at resolution {resolution:.2f}: {str(e)}")
                all_results.append(result_dict)
                continue
                
        except Exception as e:
            print(f"Error at resolution {resolution:.2f}: {str(e)}")
        
        all_results.append(result_dict)

    # Find best resolution from first pass
    coarse_results = [r for r in all_results if not np.isnan(r['cca_score'])]
    if not coarse_results:
        raise ValueError("No valid CCA scores obtained in coarse search. Check your ATAC data and parameters.")
    
    best_coarse = max(coarse_results, key=lambda x: x['cca_score'])
    best_resolution = best_coarse['resolution']
    print(f"\nBest resolution from first pass: {best_resolution:.2f}")
    print(f"Best CCA score: {best_coarse['cca_score']:.4f}")

    # Second pass: fine-tuned search
    print("\n=== SECOND PASS: Fine-tuned Search ===")
    search_range_start = max(0.01, best_resolution - 0.05)
    search_range_end = min(1.00, best_resolution + 0.05)
    
    print(f"Fine-tuning search from {search_range_start:.2f} to {search_range_end:.2f}...")

    for resolution in np.arange(search_range_start, search_range_end + 0.001, 0.01):
        resolution = round(resolution, 3)
        
        # Skip if already tested in coarse search
        if any(abs(r['resolution'] - resolution) < 0.001 for r in all_results):
            continue
        
        print(f"\nTesting fine-tuned resolution: {resolution:.3f}")
        
        result_dict = {
            'resolution': resolution,
            'cca_score': np.nan,
            'p_value': np.nan,
            'pass': 'fine'
        }
        
        try:
            # Clean up previous cell type assignments
            if 'cell_type' in AnnData_cell.obs.columns:
                AnnData_cell.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            if 'cell_type' in AnnData_sample.obs.columns:
                AnnData_sample.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            
            # Perform clustering
            AnnData_sample = cell_types_atac(
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
            
            # Compute pseudobulk data
            pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata(
                adata=AnnData_sample, 
                batch_col=batch_col, 
                sample_col=sample_col, 
                celltype_col='cell_type', 
                n_features=n_features, 
                output_dir=output_dir,
                Save=False,
                verbose=False
            )
            
            # Perform dimension reduction
            process_anndata_with_pca(
                adata=AnnData_sample,
                pseudobulk=pseudobulk_dict,
                pseudobulk_anndata=pseudobulk_adata,
                sample_col=sample_col,
                n_expression_pcs=num_DR_components,
                n_proportion_pcs=num_DR_components,
                atac=True,
                output_dir=output_dir,
                not_save=True,
                verbose=False
            )

            # Check if column exists
            if column not in AnnData_sample.uns:
                all_results.append(result_dict)
                continue
            
            # Use the existing CCA function
            try:
                (pca_coords_2d, 
                 sev_levels, 
                 cca_model, 
                 cca_score, 
                 samples) = run_cca_on_2d_pca_from_adata(
                    adata=pseudobulk_adata,
                    column=column,
                    sev_col=sev_col
                )
                
                result_dict['cca_score'] = cca_score
                print(f"Fine-tuned Resolution {resolution:.3f}: Score {cca_score:.4f}")
                
                # Compute p-value if requested
                if compute_pvalues:
                    p_value = cca_pvalue_test(
                        pseudo_adata=pseudobulk_adata,
                        column=column,
                        input_correlation=cca_score,
                        output_directory=pvalue_results_dir,
                        num_simulations=num_pvalue_simulations,
                        sev_col=sev_col,
                        verbose=False
                    )
                    result_dict['p_value'] = p_value
                    print(f"Fine-tuned Resolution {resolution:.3f}: p-value = {p_value:.4f}")
                    
            except Exception as e:
                print(f"Error in CCA analysis at fine-tuned resolution {resolution:.3f}: {str(e)}")
                
        except Exception as e:
            print(f"Error at fine-tuned resolution {resolution:.3f}: {str(e)}")
        
        all_results.append(result_dict)

    # Create comprehensive results dataframe
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("resolution")
    
    # Find final best resolution
    valid_results = df_results[~df_results['cca_score'].isna()]
    if valid_results.empty:
        raise ValueError("No valid results obtained.")
    
    final_best_idx = valid_results['cca_score'].idxmax()
    final_best_resolution = valid_results.loc[final_best_idx, 'resolution']
    final_best_score = valid_results.loc[final_best_idx, 'cca_score']
    final_best_pvalue = valid_results.loc[final_best_idx, 'p_value'] if compute_pvalues else np.nan

    print(f"\n=== FINAL RESULTS ===")
    print(f"Best resolution: {final_best_resolution:.3f}")
    print(f"Best CCA score: {final_best_score:.4f}")
    if compute_pvalues:
        print(f"Best p-value: {final_best_pvalue:.4f}")

    # Save comprehensive results
    results_csv_path = os.path.join(main_output_dir, f"resolution_scores_comprehensive_atac_{column}.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nComprehensive results saved to: {results_csv_path}")

    # Create main visualization plot
    create_resolution_visualization(df_results, final_best_resolution, column, main_output_dir, compute_pvalues)

    # Save p-value summary if computed
    if compute_pvalues:
        pvalue_summary_path = os.path.join(pvalue_results_dir, f"pvalue_summary_{column}.txt")
        with open(pvalue_summary_path, "w") as f:
            f.write(f"Resolution Optimization P-value Summary for {column}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Best Resolution: {final_best_resolution:.3f}\n")
            f.write(f"Best CCA Score: {final_best_score:.4f}\n")
            f.write(f"Best P-value: {final_best_pvalue:.4f}\n\n")
            f.write("All Results:\n")
            f.write("-"*40 + "\n")
            for _, row in valid_results.iterrows():
                f.write(f"Resolution {row['resolution']:.3f}: CCA={row['cca_score']:.4f}, p={row['p_value']:.4f}\n")
        print(f"P-value summary saved to: {pvalue_summary_path}")

    print(f"\n[Find Optimal Resolution ATAC] Total runtime: {time.time() - start_time:.2f} seconds\n")

    return final_best_resolution, df_results


def create_resolution_visualization(df_results, best_resolution, column, output_dir, include_pvalues):
    """Create comprehensive visualization of resolution search results"""
    if include_pvalues:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot CCA scores
    valid_df = df_results[~df_results['cca_score'].isna()]
    
    # Color by pass type
    coarse_df = valid_df[valid_df['pass'] == 'coarse']
    fine_df = valid_df[valid_df['pass'] == 'fine']
    
    ax1.scatter(coarse_df['resolution'], coarse_df['cca_score'], 
                color='blue', s=60, alpha=0.6, label='Coarse Search')
    ax1.scatter(fine_df['resolution'], fine_df['cca_score'], 
                color='green', s=40, alpha=0.8, label='Fine Search')
    
    # Connect points with lines
    ax1.plot(valid_df['resolution'], valid_df['cca_score'], 
             'k-', linewidth=0.5, alpha=0.3)
    
    ax1.axvline(x=best_resolution, color='r', linestyle='--', 
                label=f'Best Resolution: {best_resolution:.3f}')
    ax1.set_xlabel("Resolution")
    ax1.set_ylabel("CCA Score")
    ax1.set_title(f"ATAC Resolution vs. CCA Score ({column})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot p-values if available
    if include_pvalues:
        valid_pval_df = valid_df[~valid_df['p_value'].isna()]
        
        coarse_pval = valid_pval_df[valid_pval_df['pass'] == 'coarse']
        fine_pval = valid_pval_df[valid_pval_df['pass'] == 'fine']
        
        ax2.scatter(coarse_pval['resolution'], coarse_pval['p_value'], 
                    color='blue', s=60, alpha=0.6, label='Coarse Search')
        ax2.scatter(fine_pval['resolution'], fine_pval['p_value'], 
                    color='green', s=40, alpha=0.8, label='Fine Search')
        
        ax2.plot(valid_pval_df['resolution'], valid_pval_df['p_value'], 
                 'k-', linewidth=0.5, alpha=0.3)
        
        ax2.axvline(x=best_resolution, color='r', linestyle='--')
        ax2.axhline(y=0.05, color='orange', linestyle=':', 
                    label='p=0.05 threshold')
        
        ax2.set_xlabel("Resolution")
        ax2.set_ylabel("P-value")
        ax2.set_title(f"ATAC Resolution vs. P-value ({column})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"resolution_optimization_comprehensive_{column}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive plot saved to: {plot_path}")