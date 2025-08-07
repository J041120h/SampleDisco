import os
import scanpy as sc
import sys
from ATAC_general_pipeline import run_scatac_pipeline
from ATAC_cell_type import *
from ATAC_visualization import DR_visualization_all
from ATAC_CCA_test import find_optimal_cell_resolution_atac
from pseudo_adata import compute_pseudobulk_adata
from DR import dimension_reduction
from CCA import CCA_Call
from CCA_test import cca_pvalue_test
from TSCAN import TSCAN
from sample_distance.sample_distance import sample_distance
from cluster import cluster
from trajectory_diff_gene import run_integrated_differential_analysis, summarize_results

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_clustering.RAISIN import *
from sample_clustering.RAISIN_TEST import *
def atac_wrapper(
    # ===== Required Parameters =====
    atac_count_data_path=None,
    atac_output_dir=None,
    atac_sample_col="sample",
    
    # ===== Process Control Flags =====
    atac_preprocessing=True,
    atac_cell_type_cluster=True,
    atac_pseudobulk_dimensionality_reduction=True,
    trajectory_analysis_atac=True,
    trajectory_DGE=True,
    sample_distance_calculation=True,
    atac_sample_cluster=True,
    cluster_DGE=True,
    atac_visualization_processing=True,
    
    # ===== Basic Parameters =====
    atac_batch_col=None,
    atac_cell_type_column="cell_type",
    atac_metadata_path=None,
    grouping_columns=['sev.level'],
    atac_pipeline_verbose=True,
    use_gpu=False,
    verbose=True,
    
    # ===== Preprocessing Parameters =====
    use_snapatac2_dimred=False,
    atac_min_cells=3,
    atac_min_genes=2000,
    atac_max_genes=15000,
    atac_min_cells_per_sample=10,
    atac_doublet=True,
    atac_tfidf_scale_factor=1e4,
    atac_num_features=40000,
    atac_n_lsi_components=30,
    atac_drop_first_lsi=True,
    atac_harmony_max_iter=30,
    atac_n_neighbors=15,
    atac_n_pcs=30,
    atac_umap_min_dist=0.3,
    atac_umap_spread=1.0,
    atac_umap_random_state=42,
    atac_plot_dpi=300,
    
    # ===== Cell Type Clustering Parameters =====
    atac_leiden_resolution=0.8,
    atac_existing_cell_types=False,
    atac_n_target_cell_clusters=None,
    
    # ===== Paths for Skipping Preprocessing =====
    atac_cell_path=None,
    atac_sample_path=None,
    
    # ===== Pseudobulk Parameters =====
    atac_pseudobulk_output_dir=None,
    atac_pseudobulk_n_features=50000,
    atac_pseudobulk_verbose=True,
    
    # ===== Dimensionality Reduction Parameters =====
    atac_dr_output_dir=None,
    atac_dr_n_expression_components=30,
    atac_dr_n_proportion_components=30,
    atac_dr_verbose=True,
    
    # ===== Trajectory Analysis Parameters =====
    trajectory_supervised_atac=True,
    n_components_for_cca_atac=2,
    atac_cca_output_dir=None,
    sev_col_cca="sev.level",
    trajectory_verbose=True,
    cca_pvalue=False,
    atac_cca_optimal_cell_resolution=False,
    n_pcs_for_null_atac=10,
    TSCAN_origin=None,
    trajectory_visualization_label=['sev.level'],
    
    # ===== Trajectory Differential Gene Parameters =====
    fdr_threshold=0.05,
    effect_size_threshold=1,
    top_n_genes=100,
    trajectory_diff_gene_covariate=None,
    num_splines=5,
    spline_order=3,
    atac_trajectory_diff_gene_output_dir=None,
    visualization_gene_list=None,
    visualize_all_deg=True,
    top_n_heatmap=50,
    trajectory_diff_gene_verbose=True,
    top_gene_number=30,
    
    # ===== Sample Distance Parameters =====
    sample_distance_methods=None,
    summary_sample_csv_path=None,
    
    # ===== Clustering Parameters =====
    Kmeans_based_cluster_flag=False,
    Tree_building_method=['HRA_VEC', 'HRC_VEC', 'NN', 'UPGMA'],
    proportion_test=False,
    RAISIN_analysis=False,
    cluster_distance_method='cosine',
    cluster_number=4,
    user_provided_sample_to_clade=None,
    
    # ===== Visualization Parameters =====
    atac_figsize=(10, 8),
    atac_point_size=50,
    atac_visualization_grouping_columns=['current_severity'],
    atac_show_sample_names=True,
    atac_visualization_age_size=None,
    age_bin_size=None,
    verbose_Visualization=True,
    dot_size=3,
    plot_dendrogram_flag=True,
    plot_cell_umap_by_plot_group_flag=True,
    plot_umap_by_cell_type_flag=True,
    plot_pca_2d_flag=True,
    plot_pca_3d_flag=True,
    plot_3d_cells_flag=True,
    plot_cell_type_proportions_pca_flag=True,
    plot_cell_type_expression_pca_flag=True,
    plot_pseudobulk_batch_test_expression_flag=False,
    plot_pseudobulk_batch_test_proportion_flag=False,
    
    # ===== System Parameters =====
    sample_col='sample',
    status_flags=None,
):
    # Validate input paths
    if any(data is None for data in (atac_count_data_path, atac_output_dir)):
        raise ValueError("Please provide valid paths for atac_count_data_path, atac_output_dir.")
    
    # Set default sample distance methods if not provided
    if sample_distance_methods is None:
        sample_distance_methods = ['cosine', 'correlation']
    if atac_pseudobulk_output_dir is None:
        atac_pseudobulk_output_dir = atac_output_dir
    if atac_dr_output_dir is None:
        atac_dr_output_dir = atac_output_dir
    if atac_cca_output_dir is None:
        atac_cca_output_dir = atac_output_dir
    if atac_trajectory_diff_gene_output_dir is None:
        atac_trajectory_diff_gene_output_dir = os.path.join(atac_output_dir, 'trajectoryDEG')
    
    os.makedirs(atac_output_dir, exist_ok=True)
    os.makedirs(atac_pseudobulk_output_dir, exist_ok=True)
    os.makedirs(atac_dr_output_dir, exist_ok=True)
    os.makedirs(atac_cca_output_dir, exist_ok=True)
    
    # Initialize status flags if not provided
    if status_flags is None:
        status_flags = {
            "atac": {
                "preprocessing": False,
                "cell_type_cluster": False,
                "dimensionality_reduction": False,
                "sample_distance_calculation": False,
                "trajectory_analysis": False,
                "trajectory_dge": False,
                "cluster_dge": False,
                "visualization": False
            }
        }
    
    # Ensure ATAC section exists in status_flags
    if "atac" not in status_flags:
        status_flags["atac"] = {
            "preprocessing": False,
            "cell_type_cluster": False,
            "dimensionality_reduction": False,
            "sample_distance_calculation": False,
            "trajectory_analysis": False,
            "trajectory_dge": False,
            "cluster_dge": False,
            "visualization": False
        }
    
    # Initialize variables
    atac_sample = None
    atac_cell = None
    pseudobulk_anndata = None
    atac_pseudobulk_df = None
    
    # Step 1: ATAC Preprocessing
    if atac_preprocessing:
        if atac_pipeline_verbose:
            print("Starting ATAC-seq preprocessing...")
            
        atac_sample, atac_cell = run_scatac_pipeline(
            filepath=atac_count_data_path,
            output_dir=atac_output_dir,
            metadata_path=atac_metadata_path,
            sample_column=atac_sample_col,
            batch_key=atac_batch_col,
            verbose=atac_pipeline_verbose,
            use_snapatac2_dimred=use_snapatac2_dimred,
            
            # QC and filtering parameters
            min_cells=atac_min_cells,
            min_genes=atac_min_genes,
            max_genes=atac_max_genes,
            min_cells_per_sample=atac_min_cells_per_sample,
            
            # Doublet detection
            doublet=atac_doublet,
            
            # TF-IDF parameters
            tfidf_scale_factor=atac_tfidf_scale_factor,
            
            # Highly variable features
            num_features=atac_num_features,
            
            # LSI / dimensionality reduction
            n_lsi_components=atac_n_lsi_components,
            drop_first_lsi=atac_drop_first_lsi,
            
            # Harmony integration
            harmony_max_iter=atac_harmony_max_iter,
            harmony_use_gpu=use_gpu,
            
            # Nearest neighbors
            n_neighbors=atac_n_neighbors,
            n_pcs=atac_n_pcs,
            
            # UMAP visualization
            umap_min_dist=atac_umap_min_dist,
            umap_spread=atac_umap_spread,
            umap_random_state=atac_umap_random_state,
        )
        status_flags["atac"]["preprocessing"] = True
    else:
        # Load preprocessed data
        if not atac_cell_path or not atac_sample_path:
            temp_cell_path = os.path.join(atac_output_dir, "preprocess", "adata_cell.h5ad")
            temp_sample_path = os.path.join(atac_output_dir, "preprocess", "adata_sample.h5ad")
            if not os.path.exists(temp_cell_path) or not os.path.exists(temp_sample_path):
                raise ValueError("Preprocessed ATAC data paths are not provided and default files do not exist.")
            atac_cell_path = temp_cell_path
            atac_sample_path = temp_sample_path
        
        if atac_pipeline_verbose:
            print(f"Loading preprocessed ATAC data from {atac_cell_path} and {atac_sample_path}")
            
        atac_cell = sc.read(atac_cell_path)
        atac_sample = sc.read(atac_sample_path)
    
    # Step 2: Cell Type Clustering
    if atac_pipeline_verbose:
        print("Performing cell type clustering for ATAC data...")
    
    if atac_cell_type_cluster:
        atac_sample = cell_types_atac(
            adata=atac_sample,
            cell_column=atac_cell_type_column,
            Save = True,
            existing_cell_types=atac_existing_cell_types,
            n_target_clusters=atac_n_target_cell_clusters,
            cluster_resolution=atac_leiden_resolution,
            use_rep='X_DM_harmony',
            method='average',
            metric='euclidean',
            distance_mode='centroid',
            num_DMs=atac_n_lsi_components,
            output_dir=atac_output_dir,
            verbose=verbose
        )
        status_flags["atac"]["cell_type_cluster"] = True
    
    # Step 3: Pseudobulk and Dimensionality Reduction
    if atac_pseudobulk_dimensionality_reduction:
        if atac_pipeline_verbose:
            print("Computing pseudobulk and performing dimensionality reduction...")
            
        atac_pseudobulk_df, pseudobulk_adata = compute_pseudobulk_adata(
            adata=atac_sample,
            batch_col=atac_batch_col,
            sample_col=atac_sample_col,
            celltype_col=atac_cell_type_column,
            output_dir=atac_pseudobulk_output_dir,
            n_features=atac_pseudobulk_n_features,
            atac=True,
            verbose=atac_pseudobulk_verbose
        )
        
        pseudobulk_anndata = dimension_reduction(
            adata=atac_sample,
            pseudobulk=atac_pseudobulk_df,
            pseudobulk_anndata=pseudobulk_adata,
            sample_col=sample_col,
            n_expression_components=atac_dr_n_expression_components,
            n_proportion_components=atac_dr_n_proportion_components,
            output_dir=atac_dr_output_dir,
            atac=True,
            use_snapatac2_dimred=use_snapatac2_dimred,
            verbose=atac_dr_verbose
        )
        status_flags["atac"]["dimensionality_reduction"] = True

    elif sample_distance_calculation or cluster_DGE or  trajectory_analysis_atac:
        temp_pseuobulk_path = os.path.join(atac_output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
        if not os.path.exists(temp_pseuobulk_path):
            raise ValueError("Pseudobulk data not found. Ensure dimensionality reduction is performed first.")
        pseudobulk_anndata = sc.read(temp_pseuobulk_path)
        status_flags["atac"]["dimensionality_reduction"] = True

    # Step 4: Trajectory Analysis
    if trajectory_analysis_atac:
        if not status_flags["atac"]["dimensionality_reduction"]:
            # Load pseudobulk data if not computed
            pseudobulk_path = os.path.join(atac_pseudobulk_output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
            if not os.path.exists(pseudobulk_path):
                raise ValueError(f"Pseudobulk data not found at {pseudobulk_path}. Please run pseudobulk dimensionality reduction first.")
            pseudobulk_anndata = sc.read(pseudobulk_path)
        
        if trajectory_supervised_atac:
            if atac_pipeline_verbose:
                print("Running supervised trajectory analysis...")
                
            if sev_col_cca not in pseudobulk_anndata.obs.columns:
                raise ValueError(f"Severity column '{sev_col_cca}' not found in pseudobulk data.")
                
            first_component_score_proportion, first_component_score_expression, ptime_proportion, ptime_expression = CCA_Call(
                adata=pseudobulk_anndata,
                n_components = n_components_for_cca_atac,
                output_dir=atac_cca_output_dir,
                sev_col=sev_col_cca,
                ptime=True,
                verbose=trajectory_verbose
            )
            
            if cca_pvalue:
                if trajectory_verbose:
                    print("Computing CCA p-values...")
                    
                cca_pvalue_test(
                    pseudo_adata=pseudobulk_anndata,
                    column="X_DR_expression",
                    input_correlation=first_component_score_expression,
                    output_directory=atac_cca_output_dir,
                    num_simulations=1000,
                    sev_col=sev_col_cca,
                    verbose=trajectory_verbose
                )
                
                cca_pvalue_test(
                    pseudo_adata=pseudobulk_anndata,
                    column="X_DR_proportion",
                    input_correlation=first_component_score_proportion,
                    output_directory=atac_cca_output_dir,
                    num_simulations=1000,
                    sev_col=sev_col_cca,
                    verbose=trajectory_verbose
                )
            
            if atac_cca_optimal_cell_resolution:
                if trajectory_verbose:
                    print("Finding optimal cell resolution...")
                    
                find_optimal_cell_resolution_atac(
                    AnnData_cell=atac_cell,
                    AnnData_sample=atac_sample,
                    output_dir=atac_cca_output_dir,
                    column="X_DR_expression",
                    n_features=atac_pseudobulk_n_features,
                    sample_col=atac_sample_col,
                    batch_col=atac_batch_col,
                    num_DR_components=atac_dr_n_expression_components,
                    num_DMs=atac_n_lsi_components,
                    n_pcs=n_pcs_for_null_atac
                )
                
                find_optimal_cell_resolution_atac(
                    AnnData_cell=atac_cell,
                    AnnData_sample=atac_sample,
                    output_dir=atac_cca_output_dir,
                    column="X_DR_proportion",
                    n_features=atac_pseudobulk_n_features,
                    sample_col=atac_sample_col,
                    batch_col=atac_batch_col,
                    num_DR_components=atac_dr_n_proportion_components,
                    num_DMs=atac_n_lsi_components,
                    n_pcs=n_pcs_for_null_atac
                )
        else:
            # Unsupervised trajectory analysis using TSCAN
            if atac_pipeline_verbose:
                print("Running unsupervised trajectory analysis with TSCAN...")
                
            TSCAN_result_expression = TSCAN(
                AnnData_sample=pseudobulk_anndata,
                column="X_DR_expression",
                n_clusters=8,
                output_dir=atac_output_dir,
                grouping_columns=trajectory_visualization_label,
                verbose=trajectory_verbose,
                origin=TSCAN_origin
            )
            
            TSCAN_result_proportion = TSCAN(
                AnnData_sample=pseudobulk_anndata,
                column="X_DR_proportion",
                n_clusters=8,
                output_dir=atac_output_dir,
                grouping_columns=trajectory_visualization_label,
                verbose=trajectory_verbose,
                origin=TSCAN_origin
            )
        
        status_flags["atac"]["trajectory_analysis"] = True
        
    if trajectory_DGE and trajectory_analysis_atac:
        if atac_pipeline_verbose:
            print("Starting trajectory differential analysis for ATAC data...")
        
        if trajectory_supervised_atac:
            # CCA-based trajectory analysis for ATAC
            print("Running CCA-based differential analysis for ATAC peaks...")
            
            # Assuming you have CCA_results_atac from running CCA on ATAC pseudobulk
            results = run_integrated_differential_analysis(
                trajectory_results=(first_component_score_proportion, first_component_score_expression, ptime_proportion, ptime_expression),
                pseudobulk_adata=pseudobulk_anndata,  # ATAC pseudobulk AnnData
                trajectory_type="CCA",
                sample_col=atac_sample_col,
                fdr_threshold=fdr_threshold,
                effect_size_threshold=effect_size_threshold,
                top_n_genes=top_n_genes, 
                covariate_columns=trajectory_diff_gene_covariate,
                num_splines=num_splines,
                spline_order=spline_order,
                base_output_dir=atac_trajectory_diff_gene_output_dir,
                visualization_gene_list=visualization_gene_list,
                visualize_all_deg=visualize_all_deg,
                top_n_heatmap=top_n_heatmap,
                verbose=trajectory_diff_gene_verbose
            )
            
            if trajectory_diff_gene_verbose and "main_path" in results:
                print("Generating summary for differential ATAC peaks...")
                summarize_results(
                    results=results["main_path"],
                    top_n=top_gene_number,
                    output_file=os.path.join(atac_trajectory_diff_gene_output_dir, "differential_peaks_summary.txt"),
                    verbose=trajectory_diff_gene_verbose
                )
        
        else:
            # TSCAN-based trajectory analysis for ATAC
            print("Running TSCAN-based differential analysis for ATAC peaks...")
            
            # Using the integrated function for TSCAN analysis
            all_path_results = run_integrated_differential_analysis(
                trajectory_results=TSCAN_result_expression,  # TSCAN results for ATAC data
                pseudobulk_adata=pseudobulk_anndata,  # ATAC pseudobulk AnnData
                trajectory_type="TSCAN",
                sample_col=atac_sample_col,
                fdr_threshold=fdr_threshold,
                effect_size_threshold=effect_size_threshold,
                top_n_genes=top_n_genes,  # This will be top peaks for ATAC
                covariate_columns=trajectory_diff_gene_covariate,
                num_splines=num_splines,
                spline_order=spline_order,
                base_output_dir=os.path.join(atac_output_dir, 'trajectoryDEG'),
                visualization_gene_list=visualization_gene_list,
                visualize_all_deg=visualize_all_deg,
                top_n_heatmap=top_n_heatmap,
                verbose=trajectory_diff_gene_verbose
            )
        
        # Set status flag
        status_flags["atac"]["trajectory_dge"] = True
        
        if atac_pipeline_verbose:
            print("ATAC trajectory differential analysis completed!")
        
    # Step 5: Sample Distance Calculation (if enabled)
    if sample_distance_calculation:
        if atac_pipeline_verbose:
            print("Starting sample distance calculation for ATAC data...")
            
        if not status_flags["atac"]["dimensionality_reduction"]:
            raise ValueError("Pseudobulk dimensionality reduction is required before sample distance calculation.")
        
        # Ensure summary CSV path is set
        if summary_sample_csv_path is None:
            summary_sample_csv_path = os.path.join(atac_output_dir, 'summary_sample_atac.csv')
        
        # Clean up existing summary file
        if os.path.exists(summary_sample_csv_path):
            os.remove(summary_sample_csv_path)
        for method in sample_distance_methods:
            print(f"\nRunning sample distance: {method}\n")
            sample_distance(
                adata= pseudobulk_anndata,
                output_dir=os.path.join(atac_output_dir, 'Sample_distance'),
                method=method,
                grouping_columns=grouping_columns,
                summary_csv_path=summary_sample_csv_path,
                cell_adata= atac_sample,
                cell_type_column='cell_type',
                sample_column=sample_col,
                pseudobulk_adata=pseudobulk_anndata
            )
        status_flags["atac"]["sample_distance_calculation"] = True
        
        if atac_pipeline_verbose:
            print(f"Sample distance calculation completed. Results saved in {os.path.join(atac_output_dir, 'Sample')}")
    
    # Step 6: Clustering and Differential Analysis (if enabled)
    if atac_sample_cluster:
        if atac_pipeline_verbose:
            print("Starting clustering and differential analysis for ATAC data...")
            
        if cluster_distance_method not in sample_distance_methods:
            raise ValueError(f"Distance method '{cluster_distance_method}' not found in sample distance methods.")
        
        for method in sample_distance_methods:
            expr_results, prop_results = cluster(
                generalFolder = atac_output_dir,
                Kmeans=Kmeans_based_cluster_flag,
                methods=Tree_building_method,
                prportion_test=proportion_test,
                distance_method=method,
                number_of_clusters=cluster_number,
                sample_to_clade_user=user_provided_sample_to_clade
            )
        
            if cluster_DGE:
                if atac_pipeline_verbose:
                    print("Running RAISIN analysis for ATAC data...")
                    
                # Expression-based RAISIN analysis
                if expr_results is not None:
                    unique_expr_clades = len(set(expr_results.values()))
                    if unique_expr_clades <= 1:
                        print("Only one clade found in ATAC expression results. Skipping RAISIN analysis.")
                    else:
                        fit = raisinfit(
                            adata_path=os.path.join(atac_output_dir, 'preprocess', 'adata_sample.h5ad'),
                            sample_col=atac_sample_col,
                            batch_key=atac_batch_col,
                            sample_to_clade=expr_results,
                            verbose=verbose,
                            intercept=True,
                            n_jobs=-1,
                        )
                        run_pairwise_raisin_analysis(
                            fit=fit,
                            output_dir=os.path.join(atac_output_dir, 'raisin_results_expression'),
                            min_samples=2,
                            fdrmethod='fdr_bh',
                            n_permutations=10,
                            fdr_threshold=0.05,
                            verbose=True
                        )
                else:
                    print("No ATAC expression results available. Skipping RAISIN analysis.")
                
                # Proportion-based RAISIN analysis
                if prop_results is not None:
                    unique_prop_clades = len(set(prop_results.values()))
                    if unique_prop_clades <= 1:
                        print("Only one clade found in ATAC proportion results. Skipping RAISIN analysis.")
                    else:
                        fit = raisinfit(
                            adata_path=os.path.join(atac_output_dir, 'preprocess', 'adata_sample.h5ad'),
                            sample_col=atac_sample_col,
                            batch_key=atac_batch_col,
                            sample_to_clade=prop_results,
                            intercept=True,
                            n_jobs=-1,
                        )
                        run_pairwise_raisin_analysis(
                            fit=fit,
                            output_dir=os.path.join(atac_output_dir, 'raisin_results_proportion'),
                            min_samples=2,
                            fdrmethod='fdr_bh',
                            n_permutations=10,
                            fdr_threshold=0.05,
                            verbose=True
                        )
                else:
                    print("No ATAC proportion results available. Skipping RAISIN analysis.")
        status_flags["atac"]["cluster_dge"] = True
    
    # Step 8: Visualization
    if atac_visualization_processing:
        if atac_pipeline_verbose:
            print("Creating ATAC visualizations...")
        # ATAC-specific visualization
        DR_visualization_all(
            atac_sample,
            figsize=atac_figsize,
            point_size=atac_point_size,
            alpha=0.7,
            output_dir=os.path.join(atac_output_dir, "visualization"),
            grouping_columns=atac_visualization_grouping_columns,
            age_bin_size=atac_visualization_age_size,
            show_sample_names=atac_show_sample_names,
            sample_col=atac_sample_col
        )

    if atac_pipeline_verbose:
        print("ATAC-seq analysis completed successfully!")
    
    return atac_sample, atac_cell, pseudobulk_anndata