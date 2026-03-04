import os
import scanpy as sc
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ATAC_general_pipeline import run_scatac_pipeline
from ATAC_cell_type import cell_types_atac
from visualization.ATAC_visualization import DR_visualization_all
from ATAC_CCA_test import find_optimal_cell_resolution_atac
from sample_embedding.calculate_sample_embedding import calculate_sample_embedding
from CCA import CCA_Call
from CCA_test import cca_pvalue_test
from TSCAN import TSCAN
from sample_distance.sample_distance import sample_distance
from cluster import cluster
from trajectory_diff_gene import run_trajectory_gam_differential_gene_analysis
from sample_clustering.RAISIN import raisinfit
from sample_clustering.RAISIN_TEST import run_pairwise_tests
from sample_clustering.proportion_test import proportion_test as run_proportion_test


def atac_wrapper(
    atac_count_data_path=None,
    atac_output_dir=None,
    atac_sample_col="sample",
    
    atac_preprocessing=True,
    atac_cell_type_cluster=True,
    atac_pseudobulk_dimensionality_reduction=True,
    trajectory_analysis_atac=True,
    trajectory_DGE=True,
    sample_distance_calculation=True,
    atac_sample_cluster=True,
    cluster_DGE=True,
    atac_visualization_processing=True,
    
    atac_cell_path=None,
    atac_sample_path=None,
    atac_pseudobulk_adata_path=None,
    
    atac_batch_col=None,
    atac_cell_type_column="cell_type",
    atac_metadata_path=None,
    grouping_columns=None,
    atac_pipeline_verbose=True,
    use_gpu=False,
    verbose=True,
    
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
    
    atac_leiden_resolution=0.8,
    atac_existing_cell_types=False,
    atac_n_target_cell_clusters=None,
    
    atac_sample_hvg_number=50000,
    preserve_cols_for_sample_embedding=None,
    atac_n_expression_components=30,
    atac_n_proportion_components=30,
    atac_harmony_for_proportion=True,
    atac_dr_verbose=True,
    
    trajectory_supervised_atac=True,
    n_components_for_cca_atac=2,
    sev_col_cca="sev.level",
    trajectory_verbose=True,
    cca_pvalue=False,
    atac_cca_optimal_cell_resolution=False,
    n_pcs_for_null_atac=10,
    TSCAN_origin=None,
    trajectory_visualization_label=None,
    
    fdr_threshold=0.05,
    effect_size_threshold=1.0,
    top_n_genes=100,
    trajectory_diff_gene_covariate=None,
    num_splines=5,
    spline_order=3,
    visualization_gene_list=None,
    generate_visualizations=True,
    n_clusters_heatmap=3,
    top_n_genes_for_curves=20,
    trajectory_diff_gene_verbose=True,
    
    sample_distance_methods=None,
    summary_sample_csv_path=None,
    
    Kmeans_based_cluster_flag=False,
    Tree_building_method=None,
    proportion_test_flag=False,
    RAISIN_analysis=False,
    cluster_distance_method='cosine',
    cluster_number=4,
    user_provided_sample_to_clade=None,
    cluster_differential_gene_group_col=None,
    
    atac_figsize=(10, 8),
    atac_point_size=50,
    atac_visualization_grouping_columns=None,
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
    
    status_flags=None,
):
    print("Starting ATAC wrapper function with provided parameters...")
    
    if atac_count_data_path is None or atac_output_dir is None:
        raise ValueError("Required parameters atac_count_data_path and atac_output_dir must be provided.")
    
    if grouping_columns is None:
        grouping_columns = ['sev.level']
    if sample_distance_methods is None:
        sample_distance_methods = ['cosine', 'correlation']
    if Tree_building_method is None:
        Tree_building_method = ['HRA_VEC', 'HRC_VEC', 'NN', 'UPGMA']
    if trajectory_visualization_label is None:
        trajectory_visualization_label = ['sev.level']
    if atac_visualization_grouping_columns is None:
        atac_visualization_grouping_columns = ['current_severity']
    if summary_sample_csv_path is None:
        summary_sample_csv_path = os.path.join(atac_output_dir, 'summary_sample.csv')
    
    trajectory_diff_gene_output_dir = os.path.join(atac_output_dir, 'trajectoryDEG')
    
    default_status = {
        "preprocessing": False,
        "cell_type_cluster": False,
        "dimensionality_reduction": False,
        "sample_distance_calculation": False,
        "trajectory_analysis": False,
        "trajectory_dge": False,
        "cluster_dge": False,
        "visualization": False
    }
    
    if status_flags is None:
        status_flags = {"atac": default_status.copy()}
    elif "atac" not in status_flags:
        status_flags["atac"] = default_status.copy()
    
    atac_cell = None
    atac_sample = None
    pseudobulk_df = None
    pseudobulk_adata = None
    ptime_expression = None
    ptime_proportion = None
    
    if atac_preprocessing:
        print("="*70)
        print("STEP 1: ATAC PREPROCESSING")
        print("="*70)
        
        atac_sample, atac_cell = run_scatac_pipeline(
            filepath=atac_count_data_path,
            output_dir=atac_output_dir,
            metadata_path=atac_metadata_path,
            sample_column=atac_sample_col,
            batch_key=atac_batch_col,
            verbose=atac_pipeline_verbose,
            min_cells=atac_min_cells,
            min_genes=atac_min_genes,
            max_genes=atac_max_genes,
            min_cells_per_sample=atac_min_cells_per_sample,
            doublet=atac_doublet,
            tfidf_scale_factor=atac_tfidf_scale_factor,
            num_features=atac_num_features,
            n_lsi_components=atac_n_lsi_components,
            drop_first_lsi=atac_drop_first_lsi,
            harmony_max_iter=atac_harmony_max_iter,
            harmony_use_gpu=use_gpu,
            n_neighbors=atac_n_neighbors,
            n_pcs=atac_n_pcs,
            umap_min_dist=atac_umap_min_dist,
            umap_spread=atac_umap_spread,
            umap_random_state=atac_umap_random_state,
        )
        status_flags["atac"]["preprocessing"] = True
        print("ATAC preprocessing completed.")
        
    else:
        cell_path = atac_cell_path or os.path.join(atac_output_dir, "preprocess", "adata_cell.h5ad")
        sample_path = atac_sample_path or os.path.join(atac_output_dir, "preprocess", "adata_sample.h5ad")
        
        if not os.path.exists(cell_path) or not os.path.exists(sample_path):
            raise ValueError(
                f"Preprocessed ATAC data not found. Expected:\n"
                f"  Cell data: {cell_path}\n"
                f"  Sample data: {sample_path}"
            )
        
        status_flags["atac"]["preprocessing"] = True
        status_flags["atac"]["cell_type_cluster"] = True
        
        if (atac_cell_type_cluster or atac_pseudobulk_dimensionality_reduction or 
            trajectory_analysis_atac or trajectory_DGE or atac_sample_cluster or 
            cluster_DGE or proportion_test_flag or atac_cca_optimal_cell_resolution or 
            RAISIN_analysis or atac_visualization_processing):
            atac_cell = sc.read(cell_path)
            atac_sample = sc.read(sample_path)
            print(f"Loaded preprocessed ATAC data from {atac_output_dir}")
    
    if not status_flags["atac"]["preprocessing"]:
        raise ValueError("ATAC preprocessing is skipped, but no preprocessed data found.")
    
    if atac_cell_type_cluster and not status_flags["atac"]["cell_type_cluster"]:
        print("\n" + "="*70)
        print("STEP 2: ATAC CELL TYPE CLUSTERING")
        print("="*70)
        print(f"Using Leiden resolution: {atac_leiden_resolution}")
        
        atac_sample = cell_types_atac(
            adata=atac_sample,
            cell_column=atac_cell_type_column,
            Save=True,
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
        print("Cell type clustering completed.")
    
    if atac_pseudobulk_dimensionality_reduction:
        print("\n" + "="*70)
        print("STEP 3: PSEUDOBULK AND DIMENSIONALITY REDUCTION")
        print("="*70)
        
        if not status_flags["atac"]["cell_type_cluster"]:
            raise ValueError("Cell type clustering is required before dimensionality reduction.")
        
        pseudobulk_df, pseudobulk_adata = calculate_sample_embedding(
            adata=atac_sample,
            sample_col=atac_sample_col,
            celltype_col=atac_cell_type_column,
            batch_col=atac_batch_col,
            output_dir=atac_output_dir,
            sample_hvg_number=atac_sample_hvg_number,
            n_expression_components=atac_n_expression_components,
            n_proportion_components=atac_n_proportion_components,
            harmony_for_proportion=atac_harmony_for_proportion,
            preserve_cols=preserve_cols_for_sample_embedding,
            use_gpu=use_gpu,
            atac=True,
            save=True,
            verbose=atac_dr_verbose,
        )
        status_flags["atac"]["dimensionality_reduction"] = True
        print("Dimensionality reduction completed.")
        
    else:
        pseudobulk_path = atac_pseudobulk_adata_path or os.path.join(
            atac_output_dir, "pseudobulk", "pseudobulk_sample.h5ad"
        )
        
        if not os.path.exists(pseudobulk_path):
            if (trajectory_analysis_atac or trajectory_DGE or atac_sample_cluster or 
                cluster_DGE or atac_visualization_processing):
                raise ValueError(
                    f"Dimensionality reduction data not found at {pseudobulk_path}. "
                    "Set atac_pseudobulk_dimensionality_reduction=True or provide valid path."
                )
        
        status_flags["atac"]["dimensionality_reduction"] = True
        print(f"Reading pseudobulk from provided or default path: {pseudobulk_path}")
        pseudobulk_adata = sc.read(pseudobulk_path)
    
    if atac_cca_optimal_cell_resolution:
        print("\n" + "-"*50)
        print("Finding optimal cell resolution for ATAC...")
        print("-"*50)
        
        for column, n_components in [
            ("X_DR_expression", atac_n_expression_components),
            ("X_DR_proportion", atac_n_proportion_components)
        ]:
            find_optimal_cell_resolution_atac(
                AnnData_cell=atac_cell,
                AnnData_sample=atac_sample,
                output_dir=atac_output_dir,
                column=column,
                n_features=atac_sample_hvg_number,
                sample_col=atac_sample_col,
                batch_col=atac_batch_col,
                num_DR_components=n_components,
                num_DMs=atac_n_lsi_components,
                n_pcs=n_pcs_for_null_atac,
                preserve_cols=preserve_cols_for_sample_embedding,
            )
        
        from utils.unify_optimal import replace_optimal_dimension_reduction
        pseudobulk_adata = replace_optimal_dimension_reduction(atac_output_dir)
        print("Optimal cell resolution analysis completed.")
    
    if sample_distance_calculation:
        print("\n" + "="*70)
        print("STEP 4: SAMPLE DISTANCE CALCULATION")
        print("="*70)
        
        distance_output_dir = os.path.join(atac_output_dir, 'Sample_distance')
        
        for method in sample_distance_methods:
            print(f"\nComputing {method} distance...")
            sample_distance(
                adata=pseudobulk_adata,
                output_dir=distance_output_dir,
                method=method,
                grouping_columns=grouping_columns,
                summary_csv_path=summary_sample_csv_path,
                cell_adata=atac_sample,
                cell_type_column=atac_cell_type_column,
                sample_column=atac_sample_col,
                pseudobulk_adata=pseudobulk_adata
            )
        
        status_flags["atac"]["sample_distance_calculation"] = True
        print(f"Sample distance calculation completed. Results saved in {distance_output_dir}")
    
    if trajectory_analysis_atac:
        print("\n" + "="*70)
        print("STEP 5: TRAJECTORY ANALYSIS")
        print("="*70)
        
        if not status_flags["atac"]["dimensionality_reduction"]:
            raise ValueError("Dimensionality reduction is required before trajectory analysis.")
        
        if trajectory_supervised_atac:
            print("\n[5a] Running supervised CCA-based trajectory analysis...")
            
            if sev_col_cca not in pseudobulk_adata.obs.columns:
                raise ValueError(f"Severity column '{sev_col_cca}' not found in pseudobulk data.")
            
            (first_component_score_proportion, 
             first_component_score_expression, 
             ptime_proportion, 
             ptime_expression) = CCA_Call(
                adata=pseudobulk_adata,
                n_components=n_components_for_cca_atac,
                output_dir=atac_output_dir,
                sev_col=sev_col_cca,
                ptime=True,
                verbose=trajectory_verbose
            )
            
            if cca_pvalue:
                print("\n[5b] Running CCA p-value tests...")
                cca_pvalue_test(
                    pseudo_adata=pseudobulk_adata,
                    column="X_DR_proportion",
                    input_correlation=first_component_score_proportion,
                    output_directory=atac_output_dir,
                    num_simulations=1000,
                    sev_col=sev_col_cca,
                    verbose=trajectory_verbose
                )
                cca_pvalue_test(
                    pseudo_adata=pseudobulk_adata,
                    column="X_DR_expression",
                    input_correlation=first_component_score_expression,
                    output_directory=atac_output_dir,
                    num_simulations=1000,
                    sev_col=sev_col_cca,
                    verbose=trajectory_verbose
                )
            
            status_flags["atac"]["trajectory_analysis"] = True
            
        else:
            print("\n[5a] Running unsupervised TSCAN-based trajectory analysis...")
            
            TSCAN_result_expression = TSCAN(
                AnnData_sample=pseudobulk_adata,
                column="X_DR_expression",
                n_clusters=8,
                output_dir=atac_output_dir,
                grouping_columns=trajectory_visualization_label,
                verbose=trajectory_verbose,
                origin=TSCAN_origin
            )
            TSCAN_result_proportion = TSCAN(
                AnnData_sample=pseudobulk_adata,
                column="X_DR_proportion",
                n_clusters=8,
                output_dir=atac_output_dir,
                grouping_columns=trajectory_visualization_label,
                verbose=trajectory_verbose,
                origin=TSCAN_origin
            )
            
            ptime_expression = pd.Series(
                TSCAN_result_expression["pseudotime"]["main_path"],
                name="tscan_pseudotime_expression"
            ).reindex(pseudobulk_adata.obs.index)
            
            ptime_proportion = pd.Series(
                TSCAN_result_proportion["pseudotime"]["main_path"],
                name="tscan_pseudotime_proportion"
            ).reindex(pseudobulk_adata.obs.index)
            
            status_flags["atac"]["trajectory_analysis"] = True
        
        if trajectory_DGE:
            print("\n[5c] Running trajectory differential gene expression analysis...")
            
            if ptime_expression is not None:
                print("\n  Running expression-based trajectory DEGs...")
                run_trajectory_gam_differential_gene_analysis(
                    pseudobulk_adata=pseudobulk_adata,
                    pseudotime_source=ptime_expression,
                    sample_col=atac_sample_col,
                    pseudotime_col="pseudotime",
                    covariate_columns=trajectory_diff_gene_covariate,
                    fdr_threshold=fdr_threshold,
                    effect_size_threshold=effect_size_threshold,
                    top_n_genes=top_n_genes,
                    num_splines=num_splines,
                    spline_order=spline_order,
                    output_dir=os.path.join(trajectory_diff_gene_output_dir, "expression"),
                    visualization_gene_list=visualization_gene_list,
                    generate_visualizations=generate_visualizations,
                    group_col=sev_col_cca if trajectory_supervised_atac else None,
                    n_clusters=n_clusters_heatmap,
                    top_n_genes_for_curves=top_n_genes_for_curves,
                    verbose=trajectory_diff_gene_verbose
                )
            
            if ptime_proportion is not None:
                print("\n  Running proportion-based trajectory DEGs...")
                run_trajectory_gam_differential_gene_analysis(
                    pseudobulk_adata=pseudobulk_adata,
                    pseudotime_source=ptime_proportion,
                    sample_col=atac_sample_col,
                    pseudotime_col="pseudotime",
                    covariate_columns=trajectory_diff_gene_covariate,
                    fdr_threshold=fdr_threshold,
                    effect_size_threshold=effect_size_threshold,
                    top_n_genes=top_n_genes,
                    num_splines=num_splines,
                    spline_order=spline_order,
                    output_dir=os.path.join(trajectory_diff_gene_output_dir, "proportion"),
                    visualization_gene_list=visualization_gene_list,
                    generate_visualizations=generate_visualizations,
                    group_col=sev_col_cca if trajectory_supervised_atac else None,
                    n_clusters=n_clusters_heatmap,
                    top_n_genes_for_curves=top_n_genes_for_curves,
                    verbose=trajectory_diff_gene_verbose
                )
            
            status_flags["atac"]["trajectory_dge"] = True
            print("Trajectory differential gene analysis completed!")
        
        print("Trajectory analysis completed!")
    
    if summary_sample_csv_path and os.path.exists(summary_sample_csv_path):
        os.remove(summary_sample_csv_path)
    
    expr_results, prop_results = {}, {}
    
    if atac_sample_cluster:
        print("\n" + "="*70)
        print("STEP 6: SAMPLE CLUSTERING")
        print("="*70)
        
        expr_results, prop_results = cluster(
            pseudobulk_adata=pseudobulk_adata,
            output_dir=atac_output_dir,
            number_of_clusters=cluster_number,
            use_expression=True,
            use_proportion=True,
            random_state=0,
        )
        
        print("Sample clustering completed.")
    
    if proportion_test_flag:
        print("\n[INFO] Running proportion tests...")
        try:
            if cluster_differential_gene_group_col is not None or len(expr_results) > 0:
                run_proportion_test(
                    adata=atac_sample,
                    sample_col=atac_sample_col,
                    sample_to_clade=expr_results,
                    group_col=cluster_differential_gene_group_col,
                    celltype_col=atac_cell_type_column,
                    output_dir=os.path.join(atac_output_dir, "sample_cluster", "expression", "proportion_test"),
                    verbose=True
                )
            
            if cluster_differential_gene_group_col is not None or len(prop_results) > 0:
                run_proportion_test(
                    adata=atac_sample,
                    sample_col=atac_sample_col,
                    sample_to_clade=prop_results,
                    group_col=cluster_differential_gene_group_col,
                    celltype_col=atac_cell_type_column,
                    output_dir=os.path.join(atac_output_dir, "sample_cluster", "proportion", "proportion_test"),
                    verbose=True
                )
            print("Proportion tests completed.")
        except Exception as e:
            print(f"[ERROR] Error in proportion test: {e}")
            import traceback
            traceback.print_exc()
    
    if RAISIN_analysis:
        print("\n[INFO] Running RAISIN analysis...")
        try:
            if cluster_differential_gene_group_col is not None or len(expr_results) > 0:
                fit = raisinfit(
                    adata=atac_sample,
                    sample_col=atac_sample_col,
                    testtype='unpaired',
                    batch_col=atac_batch_col,
                    sample_to_clade=expr_results,
                    group_col=cluster_differential_gene_group_col,
                    verbose=verbose,
                    intercept=True,
                    n_jobs=-1,
                )
                run_pairwise_tests(
                    fit=fit,
                    output_dir=os.path.join(atac_output_dir, 'raisin_results_expression', cluster_distance_method),
                    fdrmethod='fdr_bh',
                    fdr_threshold=0.05,
                    verbose=True
                )
            else:
                print("[INFO] No expression results available. Skipping RAISIN analysis for expression.")
            
            if cluster_differential_gene_group_col is not None or len(prop_results) > 0:
                fit = raisinfit(
                    adata=atac_sample,
                    sample_col=atac_sample_col,
                    testtype='unpaired',
                    batch_col=atac_batch_col,
                    sample_to_clade=prop_results,
                    group_col=cluster_differential_gene_group_col,
                    verbose=verbose,
                    intercept=True,
                    n_jobs=-1,
                )
                run_pairwise_tests(
                    fit=fit,
                    output_dir=os.path.join(atac_output_dir, 'raisin_results_proportion', cluster_distance_method),
                    fdrmethod='fdr_bh',
                    fdr_threshold=0.05,
                    verbose=True
                )
            else:
                print("[INFO] No proportion results available. Skipping RAISIN analysis for proportion.")
            
            print("RAISIN analysis completed.")
            
        except Exception as e:
            print(f"[ERROR] Error in RAISIN analysis: {e}")
            import traceback
            traceback.print_exc()
    
    status_flags["atac"]["cluster_dge"] = True
    
    if atac_visualization_processing:
        print("\n" + "="*70)
        print("STEP 7: VISUALIZATION")
        print("="*70)
        
        if plot_dendrogram_flag and not status_flags["atac"]["cell_type_cluster"]:
            raise ValueError("Cell type clustering is required before dendrogram visualization.")
        
        if (plot_cell_type_proportions_pca_flag or plot_cell_type_expression_pca_flag):
            if not status_flags["atac"]["dimensionality_reduction"]:
                raise ValueError("Dimensionality reduction is required before PCA visualization.")
        
        visualization_output_dir = os.path.join(atac_output_dir, "visualization")
        
        DR_visualization_all(
            atac_sample,
            figsize=atac_figsize,
            point_size=atac_point_size,
            alpha=0.7,
            output_dir=visualization_output_dir,
            grouping_columns=atac_visualization_grouping_columns,
            age_bin_size=atac_visualization_age_size,
            show_sample_names=atac_show_sample_names,
            sample_col=atac_sample_col
        )
        
        status_flags["atac"]["visualization"] = True
        print("ATAC visualization completed!")
    
    print("\n" + "="*60)
    print("ATAC ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return {
        'atac_cell': atac_cell,
        'atac_sample': atac_sample,
        'pseudobulk_df': pseudobulk_df,
        'pseudobulk_adata': pseudobulk_adata,
        'status_flags': status_flags
    }