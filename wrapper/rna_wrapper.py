# Import required modules based on system
import os
import json
import scanpy as sc
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pseudo_adata import compute_pseudobulk_adata
from preprocess import preprocess
from sample_distance.sample_distance import sample_distance
from visualization.visualization_other import visualization
from DR import dimension_reduction
from CCA import CCA_Call
from Cell_type import cell_types, cell_type_assign
from CCA_test import find_optimal_cell_resolution, cca_pvalue_test
from TSCAN import TSCAN
from trajectory_diff_gene import run_integrated_differential_analysis
from cluster import cluster
from sample_clustering.RAISIN import *
from sample_clustering.RAISIN_TEST import *

def rna_wrapper(
    # ===== Required Parameters =====
    rna_count_data_path=None,
    rna_output_dir=None,
    sample_col='sample',
    
    # ===== Process Control Flags =====
    preprocessing=True,
    cell_type_cluster=True,
    DimensionalityReduction=True,
    sample_distance_calculation=True,
    trajectory_analysis=True,
    trajectory_DGE=True,
    sample_cluster=True,
    cluster_DGE=True,
    visualize_data=True,
    
    # ===== Paths for Skipping Processes =====
    AnnData_cell_path=None,
    AnnData_sample_path=None,
    pseudobulk_adata_path=None,
    
    # ===== Basic Parameters =====
    rna_sample_meta_path=None,
    grouping_columns=['sev.level'],
    cell_type_column='cell_type',
    cell_meta_path=None,
    batch_col=None,
    markers=None,
    cluster_resolution=0.8,
    num_PCs=20,
    num_harmony=30,
    num_features=2000,
    min_cells=500,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
    method='average',
    metric='euclidean',
    distance_mode='centroid',
    vars_to_regress=None,
    verbose=True,
    
    # ===== Cell Type Clustering Parameters =====
    existing_cell_types=False,
    n_target_cell_clusters=None,
    umap=False,
    
    # ===== Cell Type Assignment Parameters =====
    assign_save=True,
    
    # ===== Cell Type Annotation Parameters =====
    cell_type_annotation=False,
    rna_cell_type_annotation_model_name=None,
    rna_cell_type_annotation_custom_model_path=None,
    
    # ===== Pseudobulk Parameters =====
    celltype_col='cell_type',
    pseudobulk_output_dir=None,
    n_features=2000,
    pseudobulk_verbose=True,
    
    # ===== PCA Parameters =====
    n_expression_components=10,
    n_proportion_components=10,
    rna_harmony_for_proportion = True,
    dr_output_dir=None,
    dr_verbose=True,
    
    # ===== Trajectory Analysis Parameters =====
    trajectory_supervised=False,
    n_components_for_cca_rna = 2,
    Listcca_output_dir=None,
    sev_col_cca="sev.level",
    cca_optimal_cell_resolution=False,
    cca_pvalue=False,
    trajectory_verbose=True,
    TSCAN_origin=None,
    
    # ===== Trajectory Differential Gene Parameters =====
    fdr_threshold=0.05,
    effect_size_threshold=1,
    top_n_genes=100,
    trajectory_diff_gene_covariate=None,
    num_splines=5,
    spline_order=3,
    trajectory_diff_gene_output_dir=None,
    visualization_gene_list=None,
    visualize_all_deg=True,
    top_n_heatmap=50,
    trajectory_diff_gene_verbose=True,
    n_pcs_for_null=10,
    
    # ===== Distance Methods =====
    sample_distance_methods=None,
    summary_sample_csv_path=None,
    
    # ===== Updated Visualization Parameters =====
    verbose_Visualization=True,
    trajectory_visualization_label=['sev.level'],
    age_bin_size=None,
    age_column='age',
    dot_size=3,
    
    # Updated visualization flags to match new function
    plot_dendrogram_flag=True,
    plot_umap_by_cell_type_flag=True,
    plot_cell_type_proportions_pca_flag=False,
    plot_cell_type_expression_umap_flag=False,
    
    # ===== Cluster Based DEG =====
    Kmeans_based_cluster_flag=False,
    Tree_building_method=['HRA_VEC', 'HRC_VEC', 'NN', 'UPGMA'],
    proportion_test=False,
    RAISIN_analysis=False,
    cluster_distance_method='cosine',
    cluster_number=4,
    user_provided_sample_to_clade=None,
    
    # ===== System Parameters (passed from main) =====
    linux_system=False,
    use_gpu=False,
    status_flags=None
):
    print("Starting RNA wrapper function with provided parameters...")
    
    if linux_system and use_gpu:
        from linux.preprocess_linux import preprocess_linux
        from linux.CellType_linux import cell_types_linux, cell_type_assign_linux
        from linux.CCA_test_linux import find_optimal_cell_resolution_linux
        from linux.pseudo_adata_linux import compute_pseudobulk_adata_linux
        from cell_type_annotation import annotate_cell_types_with_celltypist

    # Set default values for parameters
    if vars_to_regress is None:
        vars_to_regress = []
    
    if sample_distance_methods is None:
        sample_distance_methods = ['cosine', 'correlation']
    
    if pseudobulk_output_dir is None:
        pseudobulk_output_dir = rna_output_dir
    
    if dr_output_dir is None:
        dr_output_dir = rna_output_dir
    
    if cca_output_dir is None:
        cca_output_dir = rna_output_dir
    
    if summary_sample_csv_path is None:
        summary_sample_csv_path = os.path.join(rna_output_dir, 'summary_sample.csv')
    
    if trajectory_diff_gene_output_dir is None:
        trajectory_diff_gene_output_dir = os.path.join(rna_output_dir, 'trajectoryDEG')
    
    if any(x is None for x in (rna_count_data_path, rna_output_dir)):
        raise ValueError("Required parameters rna_count_data_path, and output_dir must be provided.")
    
    # Initialize status flags if not provided
    if status_flags is None:
        status_flags = {
            "rna": {
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
    
    # Ensure RNA section exists in status_flags
    if "rna" not in status_flags:
        status_flags["rna"] = {
            "preprocessing": False,
            "cell_type_cluster": False,
            "dimensionality_reduction": False,
            "sample_distance_calculation": False,
            "trajectory_analysis": False,
            "trajectory_dge": False,
            "cluster_dge": False,
            "visualization": False
        }

    # Initialize variables - UNIFIED NAMING
    AnnData_cell = None
    AnnData_sample = None
    pseudobulk_df = None
    pseudobulk_adata = None 

    # Step 1: Harmony Preprocessing
    if preprocessing:
        print("Starting preprocessing...")
        if linux_system and use_gpu:
            AnnData_cell, AnnData_sample = preprocess_linux(
                h5ad_path=rna_count_data_path,
                sample_meta_path=rna_sample_meta_path,
                output_dir=rna_output_dir,
                sample_column=sample_col,
                cell_meta_path=cell_meta_path,
                batch_key=batch_col,
                num_PCs=num_PCs,
                num_harmony=num_harmony,
                num_features=num_features,
                min_cells=min_cells,
                min_features=min_features,
                pct_mito_cutoff=pct_mito_cutoff,
                exclude_genes=exclude_genes,
                doublet=doublet,
                vars_to_regress=vars_to_regress,
                verbose=verbose
            )
        else:
            AnnData_cell, AnnData_sample = preprocess(
                h5ad_path=rna_count_data_path,
                sample_meta_path=rna_sample_meta_path,
                output_dir=rna_output_dir,
                sample_column=sample_col,
                cell_meta_path=cell_meta_path,
                batch_key=batch_col,
                num_PCs=num_PCs,
                num_harmony=num_harmony,
                num_features=num_features,
                min_cells=min_cells,
                min_features=min_features,
                pct_mito_cutoff=pct_mito_cutoff,
                exclude_genes=exclude_genes,
                doublet=doublet,
                vars_to_regress=vars_to_regress,
                verbose=verbose
            )

        status_flags["rna"]["preprocessing"] = True

    else:
        # We offer the option to skip preprocessing and use preprocessed data
        if not AnnData_cell_path or not AnnData_sample_path:
            temp_cell_path = os.path.join(rna_output_dir, "preprocess", "adata_cell.h5ad")
            temp_sample_path = os.path.join(rna_output_dir, "preprocess", "adata_sample.h5ad")
        else:
            temp_cell_path = AnnData_cell_path
            temp_sample_path = AnnData_sample_path
        if not os.path.exists(temp_cell_path) or not os.path.exists(temp_sample_path):
            raise ValueError("Preprocessed data paths are not provided and default files path do not exist.")

        status_flags["rna"]["preprocessing"] = True
        status_flags["rna"]["cell_type_cluster"] = True
        AnnData_cell_path = temp_cell_path
        AnnData_sample_path = temp_sample_path
        if cell_type_cluster or DimensionalityReduction or trajectory_analysis or trajectory_DGE or sample_cluster or cluster_DGE or cca_optimal_cell_resolution or visualize_data:
            AnnData_cell = sc.read(AnnData_cell_path)
            AnnData_sample = sc.read(AnnData_sample_path)
    
    if not status_flags["rna"]["preprocessing"]:
            raise ValueError("RNA preprocessing is skipped, but no preprocessed data found.")
    
    # Step 2: Cell Type Clustering
    if cell_type_cluster:
        print("Starting cell type clustering at resolution:", cluster_resolution)
        if linux_system and use_gpu:
            AnnData_cell = cell_types_linux(
                adata=AnnData_cell,
                cell_type_column=cell_type_column,
                existing_cell_types=existing_cell_types,
                n_target_clusters=n_target_cell_clusters,
                umap=umap,
                output_dir=rna_output_dir,
                cluster_resolution=cluster_resolution,
                markers=markers,
                method=method,
                metric=metric,
                Save = True,
                distance_mode=distance_mode,
                num_PCs=num_PCs,
                verbose=verbose
            )
            cell_type_assign_linux(
                adata_cluster=AnnData_cell,
                adata=AnnData_sample,
                Save=assign_save,
                output_dir=rna_output_dir,
                verbose=verbose
            )
        else:
            AnnData_cell = cell_types(
                adata = AnnData_cell,
                cell_type_column = cell_type_column,
                existing_cell_types = existing_cell_types,
                n_target_clusters = n_target_cell_clusters,
                umap = umap,
                Save = True,
                output_dir = rna_output_dir,
                cluster_resolution =cluster_resolution,
                markers=markers,
                method=method,
                metric=metric,
                distance_mode=distance_mode,
                num_PCs=num_PCs,
                verbose=verbose
            )
            cell_type_assign(
                adata_cluster=AnnData_cell,
                adata=AnnData_sample,
                Save=assign_save,
                output_dir=rna_output_dir,
                verbose=verbose
            )
        if cell_type_annotation:
            annotate_cell_types_with_celltypist(
                adata = AnnData_sample,
                output_dir = rna_output_dir,
                model_name= rna_cell_type_annotation_model_name,
                custom_model_path= rna_cell_type_annotation_custom_model_path
            )
        
        status_flags["rna"]["cell_type_cluster"] = True
    
    # Step 3: Pseudobulk and PCA
    if DimensionalityReduction:
        print("Starting dimensionality reduction...")
        if not status_flags["rna"]["cell_type_cluster"]:
            raise ValueError("Cell type clustering is required before dimension reduction.")
        
        if linux_system and use_gpu:
            pseudobulk_df, pseudobulk_adata = compute_pseudobulk_adata_linux(
                adata=AnnData_sample,
                batch_col=batch_col,
                sample_col=sample_col,
                celltype_col=celltype_col,
                output_dir=pseudobulk_output_dir,
                n_features=n_features,
                verbose=pseudobulk_verbose
            )
        else:
            pseudobulk_df, pseudobulk_adata = compute_pseudobulk_adata(
                adata=AnnData_sample,
                batch_col=batch_col,
                sample_col=sample_col,
                celltype_col=celltype_col,
                output_dir=pseudobulk_output_dir,
                n_features=n_features,
                verbose=pseudobulk_verbose
            )
        
        pseudobulk_adata = dimension_reduction(
            adata=AnnData_sample,
            pseudobulk=pseudobulk_df,
            pseudobulk_anndata=pseudobulk_adata,
            sample_col=sample_col,
            n_expression_components=n_expression_components,
            n_proportion_components=n_proportion_components,
            batch_col = batch_col,
            harmony_for_proportion = rna_harmony_for_proportion,
            output_dir=dr_output_dir,
            verbose=dr_verbose
        )
        status_flags["rna"]["dimensionality_reduction"] = True
    else:
        if not pseudobulk_adata_path:
            temp_pseudobulk_path = os.path.join(pseudobulk_output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
        else:
            temp_pseudobulk_path = pseudobulk_adata_path
        if not os.path.exists(temp_pseudobulk_path):
            raise ValueError("Dimensionality_reduction is skipped, but no dimensionality_reduction data found.")
        status_flags["rna"]["dimensionality_reduction"] = True
        print(f"Reading Pseudobulk from provided or default path: {temp_pseudobulk_path}")
        pseudobulk_adata = sc.read(temp_pseudobulk_path)
    
    if cca_optimal_cell_resolution:
        if linux_system and use_gpu:
            find_optimal_cell_resolution_linux(
                AnnData_cell = AnnData_cell,
                AnnData_sample = AnnData_sample,
                output_dir = cca_output_dir,
                column = "X_DR_expression",
                n_features = n_features,
                sev_col = sev_col_cca,
                batch_col = batch_col,
                sample_col = sample_col,
                num_PCs = num_PCs,
                num_DR_components = n_expression_components,
                n_pcs_for_null = n_components_for_cca_rna,
                verbose = trajectory_verbose
            )

            find_optimal_cell_resolution_linux(
                AnnData_cell = AnnData_cell,
                AnnData_sample = AnnData_sample,
                output_dir = cca_output_dir,
                column = "X_DR_proportion",
                n_features = n_features,
                sev_col = sev_col_cca,
                batch_col = batch_col,
                sample_col = sample_col,
                num_PCs = num_PCs,
                num_DR_components = n_proportion_components,
                n_pcs_for_null = n_components_for_cca_rna,
                verbose = trajectory_verbose
            )
        else:
            find_optimal_cell_resolution(
                AnnData_cell = AnnData_cell,
                AnnData_sample = AnnData_sample,
                output_dir = cca_output_dir,
                column = "X_DR_expression",
                n_features = n_features,
                sev_col = sev_col_cca,
                batch_col = batch_col,
                sample_col = sample_col,
                num_PCs = num_PCs,
                num_DR_components = n_expression_components,
                n_pcs_for_null = n_components_for_cca_rna,
                verbose = trajectory_verbose
            )
            find_optimal_cell_resolution(
                AnnData_cell = AnnData_cell,
                AnnData_sample = AnnData_sample,
                output_dir = cca_output_dir,
                column = "X_DR_proportion",
                n_features = n_features,
                sev_col = sev_col_cca,
                batch_col = batch_col,
                sample_col = sample_col,
                num_PCs = num_PCs,
                num_DR_components = n_proportion_components,
                n_pcs_for_null = n_components_for_cca_rna,
                verbose = trajectory_verbose
            )

        from utils.unify_optimal import replace_optimal_dimension_reduction
        pseudobulk_adata = replace_optimal_dimension_reduction(rna_output_dir)
    
    # Step 5: Sample Distance Calculation - UNIFIED NAMING
    if sample_distance_calculation:
        for method in sample_distance_methods:
            print(f"\nRunning sample distance: {method}\n")
            sample_distance(
                adata=pseudobulk_adata,
                output_dir=os.path.join(rna_output_dir, 'Sample_distance'),
                method=method,
                grouping_columns=grouping_columns,
                summary_csv_path=summary_sample_csv_path,
                cell_adata=AnnData_cell,
                cell_type_column='cell_type',
                sample_column=sample_col,
                pseudobulk_adata=pseudobulk_adata
            )
        status_flags["rna"]["sample_distance_calculation"] = True
        if verbose:
            print(f"Sample distance calculation completed. Results saved in {os.path.join(rna_output_dir, 'Sample_distance')}")

    # Step 4: Trajectory Analysis - UNIFIED NAMING
    if trajectory_analysis:
        print("Starting trajectory analysis...")
        if not status_flags["rna"]["dimensionality_reduction"]:
            raise ValueError("Dimensionality reduction is required before trajectory analysis.")
        
        if trajectory_supervised:
            if sev_col_cca not in pseudobulk_adata.obs.columns:
                raise ValueError(f"Severity column '{sev_col_cca}' not found in AnnData_sample.")
            
            first_component_score_proportion, first_component_score_expression, ptime_proportion, ptime_expression = CCA_Call(
                adata=pseudobulk_adata,
                n_components = n_components_for_cca_rna,
                output_dir=cca_output_dir,
                sev_col=sev_col_cca,
                ptime=True,
                verbose=trajectory_verbose
            )
            
            if cca_pvalue:
                cca_pvalue_test(
                    pseudo_adata=pseudobulk_adata,
                    column="X_DR_proportion",
                    input_correlation=first_component_score_proportion,
                    output_directory=cca_output_dir,
                    num_simulations=1000,
                    sev_col=sev_col_cca,
                    verbose=trajectory_verbose
                )
                cca_pvalue_test(
                    pseudo_adata=pseudobulk_adata,
                    column="X_DR_expression",
                    input_correlation=first_component_score_expression,
                    output_directory=cca_output_dir,
                    num_simulations=1000,
                    sev_col=sev_col_cca,
                    verbose=trajectory_verbose
                )
            
            status_flags["rna"]["trajectory_analysis"] = True
        else:
            # Unsupervised trajectory analysis
            TSCAN_result_expression = TSCAN(
                AnnData_sample=pseudobulk_adata,
                column="X_DR_expression",
                n_clusters=8,
                output_dir=rna_output_dir,
                grouping_columns=trajectory_visualization_label,
                verbose=trajectory_verbose,
                origin=TSCAN_origin
            )
            TSCAN_result_proportion = TSCAN(
                AnnData_sample=pseudobulk_adata,
                column="X_DR_proportion",
                n_clusters=8,
                output_dir=rna_output_dir,
                grouping_columns=trajectory_visualization_label,
                verbose=trajectory_verbose,
                origin=TSCAN_origin
            )
            status_flags["rna"]["trajectory_analysis"] = True

       # Trajectory differential gene analysis - UNIFIED NAMING
        if trajectory_DGE:
            print("Starting trajectory differential gene analysis...")
            
            if trajectory_supervised:
                # CCA-based trajectory analysis
                print("Running CCA-based differential analysis...")
                
                results = run_integrated_differential_analysis(
                    trajectory_results= (first_component_score_proportion, first_component_score_expression, ptime_proportion, ptime_expression),
                    pseudobulk_adata=pseudobulk_adata,
                    trajectory_type="CCA",
                    sample_col=sample_col,
                    fdr_threshold=fdr_threshold,
                    effect_size_threshold=effect_size_threshold,
                    top_n_genes=top_n_genes,
                    covariate_columns=trajectory_diff_gene_covariate,
                    num_splines=num_splines,
                    spline_order=spline_order,
                    base_output_dir=trajectory_diff_gene_output_dir,
                    visualization_gene_list=visualization_gene_list,
                    visualize_all_deg=visualize_all_deg,
                    top_n_heatmap=top_n_heatmap,
                    verbose=trajectory_diff_gene_verbose
                )
            
            else:
                # TSCAN-based trajectory analysis
                print("Running TSCAN-based differential analysis...")
                all_path_results = run_integrated_differential_analysis(
                    trajectory_results=TSCAN_result_expression,
                    pseudobulk_adata=pseudobulk_adata,
                    trajectory_type="TSCAN",
                    sample_col=sample_col,
                    fdr_threshold=fdr_threshold,
                    effect_size_threshold=effect_size_threshold,
                    top_n_genes=top_n_genes,
                    covariate_columns=trajectory_diff_gene_covariate,
                    num_splines=num_splines,
                    spline_order=spline_order,
                    base_output_dir=trajectory_diff_gene_output_dir,
                    visualization_gene_list=visualization_gene_list,
                    visualize_all_deg=visualize_all_deg,
                    top_n_heatmap=top_n_heatmap,
                    verbose=trajectory_diff_gene_verbose
                )
            
            print("Trajectory differential gene analysis completed!")
            status_flags["rna"]["trajectory_dge"] = True
    
    # Clean up summary file if exists
    if os.path.exists(summary_sample_csv_path):
        os.remove(summary_sample_csv_path)

    # Step 6: Clustering and Differential Gene Expression
    if sample_cluster:
        print("Starting clustering and differential gene expression...")
        if cluster_distance_method not in sample_distance_methods:
            raise ValueError(f"Distance method '{cluster_distance_method}' not found in sample distance methods.")
        
        for method in sample_distance_methods:
            expr_results, prop_results = cluster(
                generalFolder = rna_output_dir,
                Kmeans=Kmeans_based_cluster_flag,
                methods=Tree_building_method,
                prportion_test=proportion_test,
                distance_method=sample_distance_methods,
                number_of_clusters=cluster_number,
                sample_to_clade_user=user_provided_sample_to_clade
            )
            
            if cluster_DGE and RAISIN_analysis:
                print("Running RAISIN analysis...")
                if expr_results is not None:
                    unique_expr_clades = len(set(expr_results.values()))
                    if unique_expr_clades <= 1:
                        print("Only one clade found in expression results. Skipping RAISIN analysis.")
                    else:
                        fit = raisinfit(
                            adata_path=os.path.join(rna_output_dir, 'preprocess', 'adata_sample.h5ad'),
                            sample_col=sample_col,
                            batch_key=batch_col,
                            sample_to_clade=expr_results,
                            verbose=verbose,
                            intercept=True,
                            n_jobs=-1,
                        )
                        run_pairwise_raisin_analysis(
                            fit=fit,
                            output_dir=os.path.join(rna_output_dir, 'raisin_results_expression', method),
                            min_samples=2,
                            fdrmethod='fdr_bh',
                            n_permutations=10,
                            fdr_threshold=0.05,
                            verbose=True
                        )
                else:
                    print("No expression results available. Skipping RAISIN analysis.")
                
                if prop_results is not None:
                    unique_prop_clades = len(set(prop_results.values()))
                    if unique_prop_clades <= 1:
                        print("Only one clade found in proportion results. Skipping RAISIN analysis.")
                    else:
                        fit = raisinfit(
                            adata_path=os.path.join(rna_output_dir, 'preprocess', 'adata_sample.h5ad'),
                            sample_col=sample_col,
                            batch_key=batch_col,
                            sample_to_clade=prop_results,
                            intercept=True,
                            n_jobs=-1,
                        )
                        run_pairwise_raisin_analysis(
                            fit=fit,
                            output_dir=os.path.join(rna_output_dir, 'raisin_results_proportion', method),
                            min_samples=2,
                            fdrmethod='fdr_bh',
                            n_permutations=10,
                            fdr_threshold=0.05,
                            verbose=True
                        )
                else:
                    print("No proportion results available. Skipping RAISIN analysis.")
        
        status_flags["rna"]["cluster_dge"] = True
    
    # Step 7: Visualization - UNIFIED NAMING
    if visualize_data:
        print("Starting visualization...")
        if not status_flags["rna"]["cell_type_cluster"]:
            if plot_dendrogram_flag:
                raise ValueError("Cell type clustering is required before dendrogram visualization.")
        
        if not status_flags["rna"]["dimensionality_reduction"]:
            if (plot_cell_type_proportions_pca_flag or plot_cell_type_expression_umap_flag):
                raise ValueError("Dimensionality reduction is required before the requested visualization.")
        
        visualization(
            AnnData_cell=AnnData_cell,
            pseudobulk_anndata=pseudobulk_adata, 
            output_dir=rna_output_dir,
            grouping_columns=grouping_columns,
            age_bin_size=age_bin_size,
            age_column=age_column,
            verbose=verbose_Visualization,
            plot_dendrogram_flag=plot_dendrogram_flag,
            plot_cell_type_proportions_pca_flag=plot_cell_type_proportions_pca_flag,
            plot_cell_type_expression_umap_flag=plot_cell_type_expression_umap_flag
        )
        status_flags["rna"]["visualization"] = True
    
    print("RNA analysis completed successfully!")
    
    return {
        'AnnData_cell': AnnData_cell,
        'AnnData_sample': AnnData_sample,
        'pseudobulk_df': pseudobulk_df,
        'pseudobulk_adata': pseudobulk_adata,
        'status_flags': status_flags
    }