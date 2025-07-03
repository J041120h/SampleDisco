import anndata as ad
from anndata import AnnData
import pandas as pd
from sklearn.cross_decomposition import CCA
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from pseudo_adata import compute_pseudobulk_adata  # Updated import
from DR import process_anndata_with_pca
from CCA_test import * 
from linux.CellType_linux import cell_types_linux, cell_type_assign_linux
from pseudo_adata_linux import compute_pseudobulk_adata_linux

def find_optimal_cell_resolution_linux(
    AnnData_cell: AnnData,
    AnnData_sample: AnnData,
    output_dir: str,
    column: str,
    n_features: int = 2000,
    sev_col: str = "sev.level",
    batch_col: str = "batch",
    sample_col: str = "sample",
    use_rep: str = 'X_pca',
    num_PCs: int = 20,
    num_DR_components: int = 30,
    num_pvalue_simulations: int = 1000,
    n_pcs_for_null: int = 10,
    compute_corrected_pvalues: bool = True
) -> tuple:
    """
    PROFILED VERSION: Find optimal clustering resolution with detailed timing information.
    """
    import time
    
    # Helper function for timing
    def time_function(func_name, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️  {func_name}: {end - start:.2f} seconds")
        return result
    
    start_time = time.time()
    print("🚀 Using PROFILED GPU version of optimal resolution")
    
    # Setup directories
    setup_start = time.time()
    main_output_dir = os.path.join(output_dir, f"RNA_resolution_optimization_{column}")
    os.makedirs(main_output_dir, exist_ok=True)
    resolutions_dir = os.path.join(main_output_dir, "resolutions")
    os.makedirs(resolutions_dir, exist_ok=True)
    setup_time = time.time() - setup_start
    print(f"⏱️  Directory setup: {setup_time:.2f} seconds")

    print("\n📊 Starting RNA-seq resolution optimization with detailed timing...")
    print(f"Using representation: {use_rep} with {num_PCs} components")
    if compute_corrected_pvalues:
        print(f"Will compute corrected p-values with {num_pvalue_simulations} simulations per resolution")

    # Storage for all results
    all_results = []
    all_resolution_null_results = []

    # First pass: coarse search
    print("\n🔍 === FIRST PASS: Coarse Search ===")
    coarse_start = time.time()
    
    for resolution in np.arange(0.1, 1.01, 0.1):
        print(f"\n🎯 Testing resolution: {resolution:.2f}")
        resolution_start = time.time()
        
        # Create resolution-specific directory
        resolution_dir = os.path.join(resolutions_dir, f"resolution_{resolution:.2f}")
        os.makedirs(resolution_dir, exist_ok=True)
        
        result_dict = {
            'resolution': resolution,
            'cca_score': np.nan,
            'p_value': np.nan,
            'corrected_pvalue': np.nan,
            'pass': 'coarse',
            'n_clusters': 0,
            'n_samples': 0
        }
        
        resolution_null_result = {
            'resolution': resolution,
            'null_scores': None
        }
        
        try:
            # Clean up previous cell type assignments
            cleanup_start = time.time()
            if 'cell_type' in AnnData_cell.obs.columns:
                AnnData_cell.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            if 'cell_type' in AnnData_sample.obs.columns:
                AnnData_sample.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            cleanup_time = time.time() - cleanup_start
            print(f"  ⏱️  Cleanup: {cleanup_time:.3f} seconds")
            
            # Perform clustering using Linux GPU version
            clustering_result = time_function(
                "  🧬 Cell clustering",
                cell_types_linux,
                AnnData_cell,
                cell_column='cell_type',
                Save=False,
                output_dir=resolution_dir,
                cluster_resolution=resolution,
                markers=None,
                method='average',
                metric='euclidean',
                distance_mode='centroid',
                num_PCs=num_PCs,
                verbose=False
            )
            
            # Assign cell types to samples using Linux GPU version
            assignment_result = time_function(
                "  📋 Cell type assignment",
                cell_type_assign_linux,
                AnnData_cell, 
                AnnData_sample, 
                Save=False, 
                output_dir=resolution_dir, 
                verbose=False
            )
            
            # Record number of clusters
            n_clusters = AnnData_sample.obs['cell_type'].nunique()
            result_dict['n_clusters'] = n_clusters
            print(f"  📊 Number of clusters: {n_clusters}")
            
            # Compute pseudobulk data
            pseudobulk_dict, pseudobulk_adata = time_function(
                "  🧪 Pseudobulk computation",
                compute_pseudobulk_adata_linux,
                adata=AnnData_sample, 
                batch_col=batch_col, 
                sample_col=sample_col, 
                celltype_col='cell_type', 
                n_features=n_features,
                output_dir=resolution_dir,
                Save=False,
                verbose=False
            )
            
            result_dict['n_samples'] = len(pseudobulk_adata)
            
            # Perform dimension reduction
            dr_result = time_function(
                "  📐 Dimension reduction",
                process_anndata_with_pca,
                adata=AnnData_sample,
                pseudobulk=pseudobulk_dict,
                pseudobulk_anndata=pseudobulk_adata,
                sample_col=sample_col,
                n_expression_pcs=num_DR_components,
                n_proportion_pcs=num_DR_components,
                atac=False,
                output_dir=resolution_dir,
                not_save=True,
                verbose=False
            )

            # Check if column exists
            if column not in AnnData_sample.uns:
                print(f"  ⚠️  Warning: {column} not found in AnnData_sample.uns")
                all_results.append(result_dict)
                all_resolution_null_results.append(resolution_null_result)
                continue
            
            # Run CCA analysis
            try:
                cca_start = time.time()
                (pca_coords_2d, 
                 sev_levels, 
                 cca_model, 
                 cca_score, 
                 samples) = run_cca_on_2d_pca_from_adata(
                    adata=pseudobulk_adata,
                    column=column,
                    sev_col=sev_col
                )
                cca_time = time.time() - cca_start
                print(f"  ⏱️  CCA analysis: {cca_time:.3f} seconds")
                
                result_dict['cca_score'] = cca_score
                print(f"  🎯 CCA Score = {cca_score:.4f}")
                
                # Save CCA plot
                plot_start = time.time()
                plot_path = os.path.join(resolution_dir, f"cca_plot_res_{resolution:.2f}.png")
                plot_cca_on_2d_pca(
                    pca_coords_2d=pca_coords_2d,
                    sev_levels=sev_levels,
                    cca=cca_model,
                    output_path=plot_path,
                    sample_labels=None,
                    title_suffix=f"Resolution {resolution:.2f}"
                )
                plot_time = time.time() - plot_start
                print(f"  ⏱️  CCA plot generation: {plot_time:.3f} seconds")
                
                # Generate null distribution if computing corrected p-values
                if compute_corrected_pvalues:
                    try:
                        null_distribution = time_function(
                            f"  🎲 Null distribution ({num_pvalue_simulations} sims)",
                            generate_null_distribution,
                            pseudobulk_adata=pseudobulk_adata,
                            column=column,
                            sev_col=sev_col,
                            n_pcs=n_pcs_for_null,
                            n_permutations=num_pvalue_simulations,
                            save_path=os.path.join(resolution_dir, f'null_dist_{resolution:.2f}.npy'),
                            verbose=False
                        )
                        resolution_null_result['null_scores'] = null_distribution
                        
                        # Compute standard p-value for this resolution
                        pvalue_start = time.time()
                        p_value = np.mean(null_distribution >= cca_score)
                        pvalue_time = time.time() - pvalue_start
                        result_dict['p_value'] = p_value
                        print(f"  ⏱️  P-value computation: {pvalue_time:.3f} seconds")
                        print(f"  📈 Standard p-value = {p_value:.4f}")
                        
                    except Exception as e:
                        print(f"  ❌ Failed to generate null distribution: {str(e)}")
                
            except Exception as e:
                print(f"  ❌ Error in CCA analysis: {str(e)}")
                
        except Exception as e:
            print(f"  ❌ Error at resolution {resolution:.2f}: {str(e)}")
        
        resolution_time = time.time() - resolution_start
        print(f"  ⏱️  TOTAL for resolution {resolution:.2f}: {resolution_time:.2f} seconds")
        
        all_results.append(result_dict)
        all_resolution_null_results.append(resolution_null_result)

    coarse_time = time.time() - coarse_start
    print(f"\n⏱️  COARSE SEARCH TOTAL: {coarse_time:.2f} seconds")

    # Find best resolution from first pass
    coarse_results = [r for r in all_results if not np.isnan(r['cca_score'])]
    if not coarse_results:
        raise ValueError("No valid CCA scores obtained in coarse search.")
    
    best_coarse = max(coarse_results, key=lambda x: x['cca_score'])
    best_resolution = best_coarse['resolution']
    print(f"\n🏆 Best resolution from first pass: {best_resolution:.2f}")
    print(f"🎯 Best CCA score: {best_coarse['cca_score']:.4f}")

    # Second pass: fine-tuned search
    print("\n🔬 === SECOND PASS: Fine-tuned Search ===")
    fine_start = time.time()
    
    search_range_start = max(0.01, best_resolution - 0.02)
    search_range_end = min(1.00, best_resolution + 0.02)
    print(f"Fine-tuning search from {search_range_start:.2f} to {search_range_end:.2f}...")

    for resolution in np.arange(search_range_start, search_range_end + 0.001, 0.01):
        resolution = round(resolution, 3)
        
        # Skip if already tested in coarse search
        if any(abs(r['resolution'] - resolution) < 0.001 for r in all_results):
            continue
        
        print(f"\n🎯 Testing fine-tuned resolution: {resolution:.3f}")
        resolution_start = time.time()
        
        # Create resolution-specific directory
        resolution_dir = os.path.join(resolutions_dir, f"resolution_{resolution:.3f}")
        os.makedirs(resolution_dir, exist_ok=True)
        
        result_dict = {
            'resolution': resolution,
            'cca_score': np.nan,
            'p_value': np.nan,
            'corrected_pvalue': np.nan,
            'pass': 'fine',
            'n_clusters': 0,
            'n_samples': 0
        }
        
        resolution_null_result = {
            'resolution': resolution,
            'null_scores': None
        }
        
        try:
            # Clean up previous cell type assignments
            cleanup_start = time.time()
            if 'cell_type' in AnnData_cell.obs.columns:
                AnnData_cell.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            if 'cell_type' in AnnData_sample.obs.columns:
                AnnData_sample.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            cleanup_time = time.time() - cleanup_start
            print(f"  ⏱️  Cleanup: {cleanup_time:.3f} seconds")
            
            # Perform clustering using Linux GPU version
            clustering_result = time_function(
                "  🧬 Cell clustering",
                cell_types_linux,
                AnnData_cell,
                cell_column='cell_type',
                Save=False,
                output_dir=resolution_dir,
                cluster_resolution=resolution,
                markers=None,
                method='average',
                metric='euclidean',
                distance_mode='centroid',
                num_PCs=num_PCs,
                verbose=False
            )
            
            # Assign cell types to samples using Linux GPU version
            assignment_result = time_function(
                "  📋 Cell type assignment",
                cell_type_assign_linux,
                AnnData_cell, 
                AnnData_sample, 
                Save=False, 
                output_dir=resolution_dir, 
                verbose=False
            )
            
            # Record number of clusters
            n_clusters = AnnData_sample.obs['cell_type'].nunique()
            result_dict['n_clusters'] = n_clusters
            
            # Compute pseudobulk data
            pseudobulk_dict, pseudobulk_adata = time_function(
                "  🧪 Pseudobulk computation",
                compute_pseudobulk_adata,
                adata=AnnData_sample, 
                batch_col=batch_col, 
                sample_col=sample_col, 
                celltype_col='cell_type', 
                n_features=n_features,
                output_dir=resolution_dir,
                Save=False,
                verbose=False
            )
            
            result_dict['n_samples'] = len(pseudobulk_adata)
            
            # Perform dimension reduction
            dr_result = time_function(
                "  📐 Dimension reduction",
                process_anndata_with_pca,
                adata=AnnData_sample,
                pseudobulk=pseudobulk_dict,
                pseudobulk_anndata=pseudobulk_adata,
                sample_col=sample_col,
                n_expression_pcs=num_DR_components,
                n_proportion_pcs=num_DR_components,
                atac=False,
                output_dir=resolution_dir,
                not_save=True,
                verbose=False
            )

            # Check if column exists
            if column not in AnnData_sample.uns:
                all_results.append(result_dict)
                all_resolution_null_results.append(resolution_null_result)
                continue
            
            # Run CCA analysis
            try:
                cca_start = time.time()
                (pca_coords_2d, 
                 sev_levels, 
                 cca_model, 
                 cca_score, 
                 samples) = run_cca_on_2d_pca_from_adata(
                    adata=pseudobulk_adata,
                    column=column,
                    sev_col=sev_col
                )
                cca_time = time.time() - cca_start
                print(f"  ⏱️  CCA analysis: {cca_time:.3f} seconds")
                
                result_dict['cca_score'] = cca_score
                print(f"  🎯 Score {cca_score:.4f}")
                
                # Generate null distribution if computing corrected p-values
                if compute_corrected_pvalues:
                    try:
                        null_distribution = time_function(
                            f"  🎲 Null distribution ({num_pvalue_simulations} sims)",
                            generate_null_distribution,
                            pseudobulk_adata=pseudobulk_adata,
                            column=column,
                            sev_col=sev_col,
                            n_pcs=n_pcs_for_null,
                            n_permutations=num_pvalue_simulations,
                            save_path=os.path.join(resolution_dir, f'null_dist_{resolution:.3f}.npy'),
                            verbose=False
                        )
                        resolution_null_result['null_scores'] = null_distribution
                        
                        # Compute standard p-value
                        pvalue_start = time.time()
                        p_value = np.mean(null_distribution >= cca_score)
                        pvalue_time = time.time() - pvalue_start
                        result_dict['p_value'] = p_value
                        print(f"  ⏱️  P-value computation: {pvalue_time:.3f} seconds")
                        
                    except Exception as e:
                        print(f"  ❌ Failed to generate null distribution: {str(e)}")
                    
            except Exception as e:
                print(f"  ❌ Error in CCA analysis: {str(e)}")
                
        except Exception as e:
            print(f"  ❌ Error at fine-tuned resolution {resolution:.3f}: {str(e)}")
        
        resolution_time = time.time() - resolution_start
        print(f"  ⏱️  TOTAL for resolution {resolution:.3f}: {resolution_time:.2f} seconds")
        
        all_results.append(result_dict)
        all_resolution_null_results.append(resolution_null_result)

    fine_time = time.time() - fine_start
    print(f"\n⏱️  FINE SEARCH TOTAL: {fine_time:.2f} seconds")

    # Create comprehensive results dataframe
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("resolution")
    
    # Generate corrected null distribution if computing corrected p-values
    if compute_corrected_pvalues:
        print("\n🔧 === GENERATING CORRECTED NULL DISTRIBUTION ===")
        corrected_start = time.time()
        
        # Filter out null results that failed to generate
        valid_null_results = [r for r in all_resolution_null_results if r['null_scores'] is not None]
        
        if valid_null_results:
            # Use the same function from ATAC
            corrected_null_distribution = time_function(
                "📊 Corrected null distribution generation",
                generate_corrected_null_distribution,
                all_resolution_results=valid_null_results,
                n_permutations=num_pvalue_simulations
            )
            
            # Save corrected null distribution
            save_start = time.time()
            corrected_null_dir = os.path.join(main_output_dir, "corrected_null")
            os.makedirs(corrected_null_dir, exist_ok=True)
            corrected_null_path = os.path.join(corrected_null_dir, f'corrected_null_distribution_{column}.npy')
            np.save(corrected_null_path, corrected_null_distribution)
            save_time = time.time() - save_start
            print(f"⏱️  Saving corrected null: {save_time:.3f} seconds")
            
            # Compute corrected p-values for all resolutions
            print("\n📊 === COMPUTING CORRECTED P-VALUES ===")
            df_results = time_function(
                "📈 Corrected p-values computation",
                compute_corrected_pvalues_rna,
                df_results=df_results,
                corrected_null_distribution=corrected_null_distribution,
                output_dir=main_output_dir,
                column=column
            )
            
            # Create visualization of corrected null distribution
            viz_start = time.time()
            plt.figure(figsize=(10, 6))
            plt.hist(corrected_null_distribution, bins=50, alpha=0.7, color='lightblue', 
                    density=True, edgecolor='black')
            plt.xlabel('Maximum CCA Score (across resolutions)', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.title(f'Corrected Null Distribution\n{column} - Accounts for Resolution Selection', 
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            plt.text(0.02, 0.98, 
                    f'Mean: {np.mean(corrected_null_distribution):.4f}\n'
                    f'Std: {np.std(corrected_null_distribution):.4f}\n'
                    f'95th percentile: {np.percentile(corrected_null_distribution, 95):.4f}',
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            corrected_null_plot_path = os.path.join(corrected_null_dir, 'corrected_null_distribution.png')
            plt.savefig(corrected_null_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_time = time.time() - viz_start
            print(f"⏱️  Corrected null visualization: {viz_time:.3f} seconds")
            
        else:
            print("⚠️  Warning: No valid null distributions generated")
            compute_corrected_pvalues = False
        
        corrected_time = time.time() - corrected_start
        print(f"⏱️  CORRECTED P-VALUES TOTAL: {corrected_time:.2f} seconds")
    
    # Find final best resolution
    valid_results = df_results[~df_results['cca_score'].isna()]
    if valid_results.empty:
        raise ValueError("No valid results obtained.")
    
    final_best_idx = valid_results['cca_score'].idxmax()
    final_best_resolution = valid_results.loc[final_best_idx, 'resolution']
    final_best_score = valid_results.loc[final_best_idx, 'cca_score']
    final_best_pvalue = valid_results.loc[final_best_idx, 'p_value'] if 'p_value' in valid_results.columns else np.nan
    final_best_corrected_pvalue = valid_results.loc[final_best_idx, 'corrected_pvalue'] if compute_corrected_pvalues else np.nan

    print(f"\n🏆 === FINAL RESULTS ===")
    print(f"🎯 Best resolution: {final_best_resolution:.3f}")
    print(f"📊 Best CCA score: {final_best_score:.4f}")
    print(f"🧬 Number of clusters: {valid_results.loc[final_best_idx, 'n_clusters']}")
    if not np.isnan(final_best_pvalue):
        print(f"📈 Standard p-value: {final_best_pvalue:.4f}")
    if compute_corrected_pvalues and not np.isnan(final_best_corrected_pvalue):
        print(f"📊 Corrected p-value: {final_best_corrected_pvalue:.4f}")

    # Create comprehensive summary and save results
    final_processing_start = time.time()
    
    create_comprehensive_summary_rna(
        df_results=df_results,
        best_resolution=final_best_resolution,
        column=column,
        output_dir=main_output_dir,
        has_corrected_pvalues=compute_corrected_pvalues
    )

    # Save complete results
    results_csv_path = os.path.join(main_output_dir, f"all_resolution_results_{column}.csv")
    df_results.to_csv(results_csv_path, index=False)

    # Create a final summary report with timing information
    final_summary_path = os.path.join(main_output_dir, "FINAL_SUMMARY.txt")
    total_runtime = time.time() - start_time
    
    with open(final_summary_path, 'w') as f:
        f.write("RNA-SEQ RESOLUTION OPTIMIZATION FINAL SUMMARY (PROFILED GPU/Linux Version)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Analysis completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total runtime: {total_runtime:.2f} seconds\n\n")
        
        f.write("TIMING BREAKDOWN:\n")
        f.write(f"  - Setup: {setup_time:.2f} seconds\n")
        f.write(f"  - Coarse search: {coarse_time:.2f} seconds\n")
        f.write(f"  - Fine search: {fine_time:.2f} seconds\n")
        if compute_corrected_pvalues and 'corrected_time' in locals():
            f.write(f"  - Corrected p-values: {corrected_time:.2f} seconds\n")
        f.write(f"  - Final processing: {time.time() - final_processing_start:.2f} seconds\n\n")
        
        f.write("OPTIMIZATION PARAMETERS:\n")
        f.write(f"  - Column analyzed: {column}\n")
        f.write(f"  - Representation used: {use_rep}\n")
        f.write(f"  - Number of PCs for clustering: {num_PCs}\n")
        f.write(f"  - Number of DR components: {num_DR_components}\n")
        f.write(f"  - Number of features: {n_features}\n")
        f.write(f"  - Number of simulations: {num_pvalue_simulations}\n")
        f.write(f"  - PCs used for null distribution: {n_pcs_for_null}\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"  - Optimal resolution: {final_best_resolution:.3f}\n")
        f.write(f"  - Best CCA score: {final_best_score:.4f}\n")
        f.write(f"  - Number of clusters: {valid_results.loc[final_best_idx, 'n_clusters']}\n")
        if not np.isnan(final_best_pvalue):
            f.write(f"  - Standard p-value: {final_best_pvalue:.4f}\n")
        if compute_corrected_pvalues and not np.isnan(final_best_corrected_pvalue):
            f.write(f"  - Corrected p-value: {final_best_corrected_pvalue:.4f}\n")
        
        f.write(f"\nTotal resolutions tested: {len(valid_results)}\n")
        f.write(f"  - Coarse search: {len(valid_results[valid_results['pass'] == 'coarse'])} resolutions\n")
        f.write(f"  - Fine search: {len(valid_results[valid_results['pass'] == 'fine'])} resolutions\n")
    
    final_processing_time = time.time() - final_processing_start
    print(f"⏱️  Final processing: {final_processing_time:.3f} seconds")

    print(f"\n🚀 [PROFILED Find Optimal Resolution RNA-seq] TOTAL RUNTIME: {total_runtime:.2f} seconds")
    print("\n📊 PERFORMANCE SUMMARY:")
    print(f"  🔧 Setup: {setup_time:.1f}s")
    print(f"  🔍 Coarse search: {coarse_time:.1f}s ({coarse_time/total_runtime*100:.1f}%)")
    print(f"  🔬 Fine search: {fine_time:.1f}s ({fine_time/total_runtime*100:.1f}%)")
    if compute_corrected_pvalues and 'corrected_time' in locals():
        print(f"  📊 Corrected p-values: {corrected_time:.1f}s ({corrected_time/total_runtime*100:.1f}%)")
    print(f"  📝 Final processing: {final_processing_time:.1f}s")

    return final_best_resolution, df_results


def cca_pvalue_test_linux(
    pseudo_adata,
    column: str,
    input_correlation: float,
    output_directory: str,
    num_simulations: int = 1000,
    sev_col: str = "sev.level",
    verbose: bool = True
):
    """
    Perform CCA p-value test using pseudo anndata (sample by gene) - Linux GPU version.
    
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


# import anndata as ad
# from anndata import AnnData
# import pandas as pd
# from sklearn.cross_decomposition import CCA
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import time
# from pseudo_adata import compute_pseudobulk_adata  # Updated import
# from DR import process_anndata_with_pca
# from CCA_test import * 
# from linux.CellType_linux import cell_types_linux, cell_type_assign_linux

# def find_optimal_cell_resolution_linux(
#     AnnData_cell: AnnData,
#     AnnData_sample: AnnData,
#     output_dir: str,
#     column: str,
#     n_features: int = 2000,
#     sev_col: str = "sev.level",
#     batch_col: str = "batch",
#     sample_col: str = "sample",
#     use_rep: str = 'X_pca',
#     num_PCs: int = 20,
#     num_DR_components: int = 30,
#     num_pvalue_simulations: int = 1000,
#     n_pcs_for_null: int = 10,
#     compute_corrected_pvalues: bool = True
# ) -> tuple:
#     """
#     Find optimal clustering resolution by maximizing CCA correlation between 
#     dimension reduction and severity levels for RNA-seq data, with corrected p-value tracking.
#     GPU/Linux version with same functionality as CPU version.
    
#     Parameters:
#     -----------
#     AnnData_cell : AnnData
#         Cell-level AnnData object (RNA-seq)
#     AnnData_sample : AnnData  
#         Sample-level AnnData object (RNA-seq)
#     output_dir : str
#         Output directory for results
#     column : str
#         Column name in adata.uns for dimension reduction results
#     n_features : int
#         Number of features for pseudobulk
#     sev_col : str
#         Column name for severity levels
#     batch_col : str
#         Column name for batch information
#     sample_col : str
#         Column name for sample identifiers
#     use_rep : str
#         Representation to use for clustering (default: 'X_pca')
#     num_PCs : int
#         Number of PCs for clustering
#     num_DR_components : int
#         Number of dimension reduction components
#     num_pvalue_simulations : int
#         Number of simulations for null distribution (default: 1000)
#     n_pcs_for_null : int
#         Number of PCs to use for null distribution (default: 10)
#     compute_corrected_pvalues : bool
#         Whether to compute corrected p-values (default: True)
        
#     Returns:
#     --------
#     tuple: (optimal_resolution, results_dataframe)
#     """
#     start_time = time.time()
#     print("Using GPU version of optimal resolution")
#     main_output_dir = os.path.join(output_dir, f"RNA_resolution_optimization_{column}")
#     os.makedirs(main_output_dir, exist_ok=True)
    
#     resolutions_dir = os.path.join(main_output_dir, "resolutions")
#     os.makedirs(resolutions_dir, exist_ok=True)

#     print("\n\n Using Linux GPU version for cell type assignment and CCA analysis.\n\n")
#     print(f"Starting RNA-seq resolution optimization for {column}...")
#     print(f"Using representation: {use_rep} with {num_PCs} components")
#     print(f"Testing resolutions from 0.01 to 1.00...")
#     if compute_corrected_pvalues:
#         print(f"Will compute corrected p-values with {num_pvalue_simulations} simulations per resolution")

#     # Storage for all results
#     all_results = []
#     all_resolution_null_results = []

#     # First pass: coarse search
#     print("\n=== FIRST PASS: Coarse Search ===")
#     for resolution in np.arange(0.1, 1.01, 0.1):
#         print(f"\n\nTesting resolution: {resolution:.2f}\n")
        
#         # Create resolution-specific directory
#         resolution_dir = os.path.join(resolutions_dir, f"resolution_{resolution:.2f}")
#         os.makedirs(resolution_dir, exist_ok=True)
        
#         result_dict = {
#             'resolution': resolution,
#             'cca_score': np.nan,
#             'p_value': np.nan,
#             'corrected_pvalue': np.nan,
#             'pass': 'coarse',
#             'n_clusters': 0,
#             'n_samples': 0
#         }
        
#         # Initialize null results for this resolution
#         resolution_null_result = {
#             'resolution': resolution,
#             'null_scores': None
#         }
        
#         try:
#             # Clean up previous cell type assignments
#             if 'cell_type' in AnnData_cell.obs.columns:
#                 AnnData_cell.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
#             if 'cell_type' in AnnData_sample.obs.columns:
#                 AnnData_sample.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            
#             # Perform clustering using Linux GPU version
#             cell_types_linux(
#                 AnnData_cell,
#                 cell_column='cell_type',
#                 Save=False,
#                 output_dir=resolution_dir,
#                 cluster_resolution=resolution,
#                 markers=None,
#                 method='average',
#                 metric='euclidean',
#                 distance_mode='centroid',
#                 num_PCs=num_PCs,
#                 verbose=False
#             )
            
#             # Assign cell types to samples using Linux GPU version
#             cell_type_assign_linux(AnnData_cell, AnnData_sample, Save=False, output_dir=resolution_dir, verbose=False)
            
#             # Record number of clusters
#             n_clusters = AnnData_sample.obs['cell_type'].nunique()
#             result_dict['n_clusters'] = n_clusters
#             print(f"Number of clusters: {n_clusters}")
            
#             # Compute pseudobulk data
#             pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata(
#                 adata=AnnData_sample, 
#                 batch_col=batch_col, 
#                 sample_col=sample_col, 
#                 celltype_col='cell_type', 
#                 n_features=n_features,
#                 output_dir=resolution_dir,
#                 Save=False,
#                 verbose=False
#             )
            
#             result_dict['n_samples'] = len(pseudobulk_adata)
            
#             # Perform dimension reduction
#             process_anndata_with_pca(
#                 adata=AnnData_sample,
#                 pseudobulk=pseudobulk_dict,
#                 pseudobulk_anndata=pseudobulk_adata,
#                 sample_col=sample_col,
#                 n_expression_pcs=num_DR_components,
#                 n_proportion_pcs=num_DR_components,
#                 atac=False,  # RNA-seq data
#                 output_dir=resolution_dir,
#                 not_save=True,
#                 verbose=False
#             )

#             # Check if column exists
#             if column not in AnnData_sample.uns:
#                 print(f"Warning: {column} not found in AnnData_sample.uns. Skipping resolution {resolution:.2f}")
#                 all_results.append(result_dict)
#                 all_resolution_null_results.append(resolution_null_result)
#                 continue
            
#             # Run CCA analysis
#             try:
#                 (pca_coords_2d, 
#                  sev_levels, 
#                  cca_model, 
#                  cca_score, 
#                  samples) = run_cca_on_2d_pca_from_adata(
#                     adata=pseudobulk_adata,
#                     column=column,
#                     sev_col=sev_col
#                 )
                
#                 result_dict['cca_score'] = cca_score
#                 print(f"Resolution {resolution:.2f}: CCA Score = {cca_score:.4f}")
                
#                 # Save CCA plot
#                 plot_path = os.path.join(resolution_dir, f"cca_plot_res_{resolution:.2f}.png")
#                 plot_cca_on_2d_pca(
#                     pca_coords_2d=pca_coords_2d,
#                     sev_levels=sev_levels,
#                     cca=cca_model,
#                     output_path=plot_path,
#                     sample_labels=None,
#                     title_suffix=f"Resolution {resolution:.2f}"
#                 )
                
#                 # Generate null distribution if computing corrected p-values
#                 if compute_corrected_pvalues:
#                     try:
#                         null_distribution = generate_null_distribution(
#                             pseudobulk_adata=pseudobulk_adata,
#                             column=column,
#                             sev_col=sev_col,
#                             n_pcs=n_pcs_for_null,
#                             n_permutations=num_pvalue_simulations,
#                             save_path=os.path.join(resolution_dir, f'null_dist_{resolution:.2f}.npy'),
#                             verbose=False
#                         )
#                         resolution_null_result['null_scores'] = null_distribution
                        
#                         # Compute standard p-value for this resolution
#                         p_value = np.mean(null_distribution >= cca_score)
#                         result_dict['p_value'] = p_value
#                         print(f"Resolution {resolution:.2f}: Standard p-value = {p_value:.4f}")
                        
#                     except Exception as e:
#                         print(f"Warning: Failed to generate null distribution: {str(e)}")
                
#             except Exception as e:
#                 print(f"Error in CCA analysis at resolution {resolution:.2f}: {str(e)}")
                
#         except Exception as e:
#             print(f"Error at resolution {resolution:.2f}: {str(e)}")
        
#         all_results.append(result_dict)
#         all_resolution_null_results.append(resolution_null_result)

#     # Find best resolution from first pass
#     coarse_results = [r for r in all_results if not np.isnan(r['cca_score'])]
#     if not coarse_results:
#         raise ValueError("No valid CCA scores obtained in coarse search. Check your RNA data and parameters.")
    
#     best_coarse = max(coarse_results, key=lambda x: x['cca_score'])
#     best_resolution = best_coarse['resolution']
#     print(f"\nBest resolution from first pass: {best_resolution:.2f}")
#     print(f"Best CCA score: {best_coarse['cca_score']:.4f}")

#     # Second pass: fine-tuned search
#     print("\n=== SECOND PASS: Fine-tuned Search ===")
#     search_range_start = max(0.01, best_resolution - 0.01)
#     search_range_end = min(1.00, best_resolution + 0.01)
    
#     print(f"Fine-tuning search from {search_range_start:.2f} to {search_range_end:.2f}...")

#     for resolution in np.arange(search_range_start, search_range_end + 0.001, 0.01):
#         resolution = round(resolution, 3)
        
#         # Skip if already tested in coarse search
#         if any(abs(r['resolution'] - resolution) < 0.001 for r in all_results):
#             continue
        
#         print(f"\nTesting fine-tuned resolution: {resolution:.3f}")
        
#         # Create resolution-specific directory
#         resolution_dir = os.path.join(resolutions_dir, f"resolution_{resolution:.3f}")
#         os.makedirs(resolution_dir, exist_ok=True)
        
#         result_dict = {
#             'resolution': resolution,
#             'cca_score': np.nan,
#             'p_value': np.nan,
#             'corrected_pvalue': np.nan,
#             'pass': 'fine',
#             'n_clusters': 0,
#             'n_samples': 0
#         }
        
#         # Initialize null results for this resolution
#         resolution_null_result = {
#             'resolution': resolution,
#             'null_scores': None
#         }
        
#         try:
#             # Clean up previous cell type assignments
#             if 'cell_type' in AnnData_cell.obs.columns:
#                 AnnData_cell.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
#             if 'cell_type' in AnnData_sample.obs.columns:
#                 AnnData_sample.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            
#             # Perform clustering using Linux GPU version
#             cell_types_linux(
#                 AnnData_cell,
#                 cell_column='cell_type',
#                 Save=False,
#                 output_dir=resolution_dir,
#                 cluster_resolution=resolution,
#                 markers=None,
#                 method='average',
#                 metric='euclidean',
#                 distance_mode='centroid',
#                 num_PCs=num_PCs,
#                 verbose=False
#             )
            
#             # Assign cell types to samples using Linux GPU version
#             cell_type_assign_linux(AnnData_cell, AnnData_sample, Save=False, output_dir=resolution_dir, verbose=False)
            
#             # Record number of clusters
#             n_clusters = AnnData_sample.obs['cell_type'].nunique()
#             result_dict['n_clusters'] = n_clusters
            
#             # Compute pseudobulk data
#             pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata(
#                 adata=AnnData_sample, 
#                 batch_col=batch_col, 
#                 sample_col=sample_col, 
#                 celltype_col='cell_type', 
#                 n_features=n_features,
#                 output_dir=resolution_dir,
#                 Save=False,
#                 verbose=False
#             )
            
#             result_dict['n_samples'] = len(pseudobulk_adata)
            
#             # Perform dimension reduction
#             process_anndata_with_pca(
#                 adata=AnnData_sample,
#                 pseudobulk=pseudobulk_dict,
#                 pseudobulk_anndata=pseudobulk_adata,
#                 sample_col=sample_col,
#                 n_expression_pcs=num_DR_components,
#                 n_proportion_pcs=num_DR_components,
#                 atac=False,
#                 output_dir=resolution_dir,
#                 not_save=True,
#                 verbose=False
#             )

#             # Check if column exists
#             if column not in AnnData_sample.uns:
#                 all_results.append(result_dict)
#                 all_resolution_null_results.append(resolution_null_result)
#                 continue
            
#             # Run CCA analysis
#             try:
#                 (pca_coords_2d, 
#                  sev_levels, 
#                  cca_model, 
#                  cca_score, 
#                  samples) = run_cca_on_2d_pca_from_adata(
#                     adata=pseudobulk_adata,
#                     column=column,
#                     sev_col=sev_col
#                 )
                
#                 result_dict['cca_score'] = cca_score
#                 print(f"Fine-tuned Resolution {resolution:.3f}: Score {cca_score:.4f}")
                
#                 # Generate null distribution if computing corrected p-values
#                 if compute_corrected_pvalues:
#                     try:
#                         null_distribution = generate_null_distribution(
#                             pseudobulk_adata=pseudobulk_adata,
#                             column=column,
#                             sev_col=sev_col,
#                             n_pcs=n_pcs_for_null,
#                             n_permutations=num_pvalue_simulations,
#                             save_path=os.path.join(resolution_dir, f'null_dist_{resolution:.3f}.npy'),
#                             verbose=False
#                         )
#                         resolution_null_result['null_scores'] = null_distribution
                        
#                         # Compute standard p-value
#                         p_value = np.mean(null_distribution >= cca_score)
#                         result_dict['p_value'] = p_value
                        
#                     except Exception as e:
#                         print(f"Warning: Failed to generate null distribution: {str(e)}")
                    
#             except Exception as e:
#                 print(f"Error in CCA analysis at fine-tuned resolution {resolution:.3f}: {str(e)}")
                
#         except Exception as e:
#             print(f"Error at fine-tuned resolution {resolution:.3f}: {str(e)}")
        
#         all_results.append(result_dict)
#         all_resolution_null_results.append(resolution_null_result)

#     # Create comprehensive results dataframe
#     df_results = pd.DataFrame(all_results)
#     df_results = df_results.sort_values("resolution")
    
#     # Generate corrected null distribution if computing corrected p-values
#     if compute_corrected_pvalues:
#         print("\n=== GENERATING CORRECTED NULL DISTRIBUTION ===")
#         print("Accounting for resolution selection bias...")
        
#         # Filter out null results that failed to generate
#         valid_null_results = [r for r in all_resolution_null_results if r['null_scores'] is not None]
        
#         if valid_null_results:
#             # Use the same function from ATAC
#             corrected_null_distribution = generate_corrected_null_distribution(
#                 all_resolution_results=valid_null_results,
#                 n_permutations=num_pvalue_simulations
#             )
            
#             # Save corrected null distribution
#             corrected_null_dir = os.path.join(main_output_dir, "corrected_null")
#             os.makedirs(corrected_null_dir, exist_ok=True)
#             corrected_null_path = os.path.join(corrected_null_dir, f'corrected_null_distribution_{column}.npy')
#             np.save(corrected_null_path, corrected_null_distribution)
#             print(f"Corrected null distribution saved to: {corrected_null_path}")
            
#             # Compute corrected p-values for all resolutions
#             print("\n=== COMPUTING CORRECTED P-VALUES ===")
#             df_results = compute_corrected_pvalues_rna(
#                 df_results=df_results,
#                 corrected_null_distribution=corrected_null_distribution,
#                 output_dir=main_output_dir,
#                 column=column
#             )
            
#             # Create visualization of corrected null distribution
#             plt.figure(figsize=(10, 6))
#             plt.hist(corrected_null_distribution, bins=50, alpha=0.7, color='lightblue', 
#                     density=True, edgecolor='black')
#             plt.xlabel('Maximum CCA Score (across resolutions)', fontsize=12)
#             plt.ylabel('Density', fontsize=12)
#             plt.title(f'Corrected Null Distribution\n{column} - Accounts for Resolution Selection', 
#                      fontsize=14, fontweight='bold')
#             plt.grid(True, alpha=0.3)
            
#             # Add statistics
#             plt.text(0.02, 0.98, 
#                     f'Mean: {np.mean(corrected_null_distribution):.4f}\n'
#                     f'Std: {np.std(corrected_null_distribution):.4f}\n'
#                     f'95th percentile: {np.percentile(corrected_null_distribution, 95):.4f}',
#                     transform=plt.gca().transAxes, fontsize=10,
#                     verticalalignment='top',
#                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
#             corrected_null_plot_path = os.path.join(corrected_null_dir, 'corrected_null_distribution.png')
#             plt.savefig(corrected_null_plot_path, dpi=300, bbox_inches='tight')
#             plt.close()
#             print(f"Corrected null distribution plot saved to: {corrected_null_plot_path}")
#         else:
#             print("Warning: No valid null distributions generated, cannot compute corrected p-values")
#             compute_corrected_pvalues = False
    
#     # Find final best resolution
#     valid_results = df_results[~df_results['cca_score'].isna()]
#     if valid_results.empty:
#         raise ValueError("No valid results obtained.")
    
#     final_best_idx = valid_results['cca_score'].idxmax()
#     final_best_resolution = valid_results.loc[final_best_idx, 'resolution']
#     final_best_score = valid_results.loc[final_best_idx, 'cca_score']
#     final_best_pvalue = valid_results.loc[final_best_idx, 'p_value'] if 'p_value' in valid_results.columns else np.nan
#     final_best_corrected_pvalue = valid_results.loc[final_best_idx, 'corrected_pvalue'] if compute_corrected_pvalues else np.nan

#     print(f"\n=== FINAL RESULTS ===")
#     print(f"Best resolution: {final_best_resolution:.3f}")
#     print(f"Best CCA score: {final_best_score:.4f}")
#     print(f"Number of clusters at best resolution: {valid_results.loc[final_best_idx, 'n_clusters']}")
#     if not np.isnan(final_best_pvalue):
#         print(f"Standard p-value: {final_best_pvalue:.4f}")
#     if compute_corrected_pvalues and not np.isnan(final_best_corrected_pvalue):
#         print(f"Corrected p-value: {final_best_corrected_pvalue:.4f}")

#     # Create comprehensive summary
#     create_comprehensive_summary_rna(
#         df_results=df_results,
#         best_resolution=final_best_resolution,
#         column=column,
#         output_dir=main_output_dir,
#         has_corrected_pvalues=compute_corrected_pvalues
#     )

#     # Save complete results
#     results_csv_path = os.path.join(main_output_dir, f"all_resolution_results_{column}.csv")
#     df_results.to_csv(results_csv_path, index=False)
#     print(f"\nAll results saved to: {results_csv_path}")

#     # Create a final summary report
#     final_summary_path = os.path.join(main_output_dir, "FINAL_SUMMARY.txt")
#     with open(final_summary_path, 'w') as f:
#         f.write("RNA-SEQ RESOLUTION OPTIMIZATION FINAL SUMMARY (GPU/Linux Version)\n")
#         f.write("=" * 60 + "\n\n")
#         f.write(f"Analysis completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"Total runtime: {time.time() - start_time:.2f} seconds\n\n")
        
#         f.write("OPTIMIZATION PARAMETERS:\n")
#         f.write(f"  - Column analyzed: {column}\n")
#         f.write(f"  - Representation used: {use_rep}\n")
#         f.write(f"  - Number of PCs for clustering: {num_PCs}\n")
#         f.write(f"  - Number of DR components: {num_DR_components}\n")
#         f.write(f"  - Number of features: {n_features}\n")
#         f.write(f"  - Number of simulations: {num_pvalue_simulations}\n")
#         f.write(f"  - PCs used for null distribution: {n_pcs_for_null}\n\n")
        
#         f.write("RESULTS:\n")
#         f.write(f"  - Optimal resolution: {final_best_resolution:.3f}\n")
#         f.write(f"  - Best CCA score: {final_best_score:.4f}\n")
#         f.write(f"  - Number of clusters: {valid_results.loc[final_best_idx, 'n_clusters']}\n")
#         if not np.isnan(final_best_pvalue):
#             f.write(f"  - Standard p-value: {final_best_pvalue:.4f}\n")
#         if compute_corrected_pvalues and not np.isnan(final_best_corrected_pvalue):
#             f.write(f"  - Corrected p-value: {final_best_corrected_pvalue:.4f}\n")
        
#         f.write(f"\nTotal resolutions tested: {len(valid_results)}\n")
#         f.write(f"  - Coarse search: {len(valid_results[valid_results['pass'] == 'coarse'])} resolutions\n")
#         f.write(f"  - Fine search: {len(valid_results[valid_results['pass'] == 'fine'])} resolutions\n")
        
#         f.write("\nOUTPUT FILES:\n")
#         f.write(f"  - Main directory: {main_output_dir}\n")
#         f.write(f"  - Summary plots: {os.path.join(main_output_dir, 'summary')}\n")
#         f.write(f"  - Resolution results: {os.path.join(main_output_dir, 'resolutions')}\n")
#         if compute_corrected_pvalues:
#             f.write(f"  - Corrected p-values: {os.path.join(main_output_dir, 'corrected_p_values')}\n")
#             f.write(f"  - Corrected null distribution: {os.path.join(main_output_dir, 'corrected_null')}\n")
    
#     print(f"\nFinal summary saved to: {final_summary_path}")
#     print(f"\n[Find Optimal Resolution RNA-seq GPU/Linux] Total runtime: {time.time() - start_time:.2f} seconds\n")

#     return final_best_resolution, df_results