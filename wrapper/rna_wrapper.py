# Import required modules based on system
import os
import json
import scanpy as sc
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_embedding.calculate_sample_embedding import calculate_sample_embedding
from preparation.preprocess import preprocess
from sample_distance.sample_distance import sample_distance
from visualization.visualization_other import visualization
from sample_trajectory.CCA import CCA_Call
from preparation.Cell_type import cell_types
from sample_trajectory.CCA_test import cca_pvalue_test
from parameter_selection.cpu_optimal_resolution import find_optimal_cell_resolution
from sample_trajectory.TSCAN import TSCAN
from sample_trajectory.trajectory_diff_gene import run_trajectory_gam_differential_gene_analysis
from cluster import cluster
from sample_clustering.RAISIN import *
from sample_clustering.RAISIN_TEST import *
from sample_clustering.proportion_test import proportion_test as run_proportion_test

def rna_wrapper(
    # ===== Required Parameters =====
    rna_count_data_path=None,
    rna_output_dir=None,
    sample_col='sample',
    
    # ===== Process Control Flags =====
    preprocessing=True,
    cell_type_cluster=True,
    derive_sample_embedding=True,
    sample_distance_calculation=True,
    trajectory_analysis=True,
    trajectory_DGE=True,
    sample_cluster=True,
    cluster_DGE=True,
    visualize_data=True,
    
    # ===== Paths for Skipping Processes =====
    adata_cell_path=None,
    adata_sample_path=None,
    pseudo_adata_path=None,
    
    # ===== Basic Parameters =====
    rna_sample_meta_path=None,
    grouping_columns=['sev.level'],
    celltype_col='cell_type',
    cell_meta_path=None,
    batch_col=None,
    leiden_cluster_resolution=0.8,
    cell_embedding_column="X_pca_harmony",
    cell_embedding_num_pcs=20,
    num_harmony_iterations=30,
    num_cell_hvgs=2000,
    min_cells=500,
    min_genes=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
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
    sample_hvg_number=2000,
    preserve_cols_for_sample_embedding=None,
    
    # ===== PCA Parameters =====
    n_expression_components=10,
    n_proportion_components=10,
    rna_harmony_for_proportion=True,
    
    # ===== Trajectory Analysis Parameters =====
    trajectory_supervised=False,
    n_components_for_cca=2,
    trajectory_col_cca="sev.level",
    cca_optimal_cell_resolution=False,
    cca_pvalue=False,
    tscan_origin=None,
    
    # ===== Trajectory Differential Gene Parameters =====
    fdr_threshold=0.05,
    effect_size_threshold=1,
    top_n_genes=100,
    trajectory_diff_gene_covariate=None,
    num_splines=5,
    spline_order=3,
    visualization_gene_list=None,
    visualize_all_deg=True,
    top_n_heatmap=50,
    
    # ===== Distance Methods =====
    sample_distance_methods=None,
    summary_sample_csv_path=None,
    
    # ===== Updated Visualization Parameters =====
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
    kmeans_based_cluster_flag=False,
    tree_building_methods=['HRA_VEC', 'HRC_VEC', 'NN', 'UPGMA'],
    proportion_test=False,
    RAISIN_analysis=False,
    cluster_distance_method='cosine',
    cluster_number=4,
    user_provided_sample_to_clade=None,
    cluster_differential_gene_group_col=None,
    
    # ===== System Parameters (passed from main) =====
    linux_system=False,
    use_gpu=False,
    status_flags=None
):
    print("Starting RNA wrapper function with provided parameters...")
    
    if linux_system and use_gpu:
        from preparation.preprocess_linux import preprocess_linux
        from preparation.Cell_type_linux import cell_types_linux
        from parameter_selection.gpu_optimal_resolution import find_optimal_cell_resolution_linux

    # Set default values for parameters
    if vars_to_regress is None:
        vars_to_regress = []
    
    if sample_distance_methods is None:
        sample_distance_methods = ['cosine', 'correlation']
    
    if summary_sample_csv_path is None:
        summary_sample_csv_path = os.path.join(rna_output_dir, 'summary_sample.csv')
    
    trajectory_diff_gene_output_dir = os.path.join(rna_output_dir, 'trajectoryDEG')
    
    if any(x is None for x in (rna_count_data_path, rna_output_dir)):
        raise ValueError("Required parameters rna_count_data_path, and output_dir must be provided.")
    
    # Initialize status flags if not provided
    default_status = {
        "preprocessing": False,
        "cell_type_cluster": False,
        "derive_sample_embedding": False,
        "sample_distance_calculation": False,
        "trajectory_analysis": False,
        "trajectory_dge": False,
        "cluster_dge": False,
        "visualization": False
    }
    
    if status_flags is None:
        status_flags = {"rna": default_status.copy()}
    elif "rna" not in status_flags:
        status_flags["rna"] = default_status.copy()

    # Initialize variables
    adata_cell = None
    adata_sample = None
    pseudobulk_df = None
    pseudo_adata = None 

    # Step 1: Harmony Preprocessing
    if preprocessing:
        print("Starting preprocessing...")
        preprocess_func = preprocess_linux if (linux_system and use_gpu) else preprocess
        
        adata_cell, adata_sample = preprocess_func(
            h5ad_path=rna_count_data_path,
            sample_meta_path=rna_sample_meta_path,
            output_dir=rna_output_dir,
            sample_column=sample_col,
            cell_meta_path=cell_meta_path,
            batch_key=batch_col,
            cell_embedding_num_PCs=cell_embedding_num_pcs,
            num_harmony_iterations=num_harmony_iterations,
            num_cell_hvgs=num_cell_hvgs,
            min_cells=min_cells,
            min_genes=min_genes,
            pct_mito_cutoff=pct_mito_cutoff,
            exclude_genes=exclude_genes,
            vars_to_regress=vars_to_regress,
            verbose=verbose
        )

        status_flags["rna"]["preprocessing"] = True

    else:
        # We offer the option to skip preprocessing and use preprocessed data
        cell_path = adata_cell_path or os.path.join(rna_output_dir, "preprocess", "adata_cell.h5ad")
        sample_path = adata_sample_path or os.path.join(rna_output_dir, "preprocess", "adata_sample.h5ad")
        
        if not os.path.exists(cell_path) or not os.path.exists(sample_path):
            raise ValueError("Preprocessed data paths are not provided and default files path do not exist.")

        status_flags["rna"]["preprocessing"] = True
        status_flags["rna"]["cell_type_cluster"] = True
        
        if cell_type_cluster or derive_sample_embedding or trajectory_analysis or trajectory_DGE or sample_cluster or cluster_DGE or proportion_test or cca_optimal_cell_resolution or RAISIN_analysis or visualize_data:
            adata_cell = sc.read(cell_path)
            adata_sample = sc.read(sample_path)
    
    if not status_flags["rna"]["preprocessing"]:
        raise ValueError("RNA preprocessing is skipped, but no preprocessed data found.")
    
    # Step 2: Cell Type Clustering
    if cell_type_cluster:
        print("Starting cell type clustering at resolution:", leiden_cluster_resolution)

        cell_types_func = cell_types_linux if (linux_system and use_gpu) else cell_types
        adata_cell, adata_sample = cell_types_func(
            anndata_cell=adata_cell,
            anndata_sample=adata_sample,
            cell_type_column=celltype_col,
            existing_cell_types=existing_cell_types,
            n_target_clusters=n_target_cell_clusters,
            umap=umap,
            output_dir=rna_output_dir,
            leiden_cluster_resolution=leiden_cluster_resolution,
            cell_embedding_column=cell_embedding_column,
            cell_embedding_num_PCs=cell_embedding_num_pcs,
            verbose=verbose,
            umap_plots=True,
            save=True,
        )
        status_flags["rna"]["cell_type_cluster"] = True
    
    # Step 3: Pseudobulk and PCA (derive_sample_embedding)
    if derive_sample_embedding:
        print("Starting sample embedding derivation (dimensionality reduction)...")
        if not status_flags["rna"]["cell_type_cluster"]:
            raise ValueError("Cell type clustering is required before sample embedding derivation.")
        
        pseudobulk_df, pseudo_adata = calculate_sample_embedding(
            adata=adata_sample,
            sample_col=sample_col,
            celltype_col=celltype_col,
            batch_col=batch_col,
            output_dir=rna_output_dir,
            sample_hvg_number=sample_hvg_number,
            n_expression_components=n_expression_components,
            n_proportion_components=n_proportion_components,
            harmony_for_proportion=rna_harmony_for_proportion,
            preserve_cols=preserve_cols_for_sample_embedding,
            use_gpu=linux_system and use_gpu,
            atac=False,
            save=True,
            verbose=verbose,
        )
        
        status_flags["rna"]["derive_sample_embedding"] = True
    else:
        pseudobulk_path = pseudo_adata_path or os.path.join(rna_output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
        
        if not os.path.exists(pseudobulk_path):
            if trajectory_analysis or trajectory_DGE or sample_cluster or cluster_DGE or visualize_data:
                raise ValueError("Sample embedding derivation is skipped, but no sample embedding data found.")
        
        status_flags["rna"]["derive_sample_embedding"] = True
        print(f"Reading Pseudobulk from provided or default path: {pseudobulk_path}")
        pseudo_adata = sc.read(pseudobulk_path)
    
    if cca_optimal_cell_resolution:
        find_resolution_func = find_optimal_cell_resolution_linux if (linux_system and use_gpu) else find_optimal_cell_resolution
        
        for column, num_dr_components in [("X_DR_expression", n_expression_components), ("X_DR_proportion", n_proportion_components)]:
            find_resolution_func(
                AnnData_cell=adata_cell,
                AnnData_sample=adata_sample,
                output_dir=rna_output_dir,
                column=column,
                n_features=sample_hvg_number,
                trajectory_col=trajectory_col_cca,
                batch_col=batch_col,
                sample_col=sample_col,
                num_PCs=cell_embedding_num_pcs,
                num_DR_components=num_dr_components,
                n_pcs_for_null=n_components_for_cca,
                verbose=verbose,
                preserve_cols=preserve_cols_for_sample_embedding,
            )

        from utils.unify_optimal import replace_optimal_dimension_reduction
        pseudo_adata = replace_optimal_dimension_reduction(rna_output_dir)
    
    # Step 5: Sample Distance Calculation
    if sample_distance_calculation:
        for distance_method in sample_distance_methods:
            print(f"\nRunning sample distance: {distance_method}\n")
            sample_distance(
                adata=pseudo_adata,
                output_dir=os.path.join(rna_output_dir, 'Sample_distance'),
                method=distance_method,
                grouping_columns=grouping_columns,
                summary_csv_path=summary_sample_csv_path,
                cell_adata=adata_cell,
                cell_type_column='cell_type',
                sample_column=sample_col,
                pseudobulk_adata=pseudo_adata
            )
        status_flags["rna"]["sample_distance_calculation"] = True
        if verbose:
            print(f"Sample distance calculation completed. Results saved in {os.path.join(rna_output_dir, 'Sample_distance')}")

    # Step 4: Trajectory Analysis
    if trajectory_analysis:
        print("Starting trajectory analysis...")
        if not status_flags["rna"]["derive_sample_embedding"]:
            raise ValueError("Sample embedding derivation is required before trajectory analysis.")
        
        if trajectory_supervised:
            if trajectory_col_cca not in pseudo_adata.obs.columns:
                raise ValueError(f"Trajectory column '{trajectory_col_cca}' not found in adata_sample.")
            
            first_component_score_proportion, first_component_score_expression, ptime_proportion, ptime_expression = CCA_Call(
                adata=pseudo_adata,
                n_components=n_components_for_cca,
                output_dir=rna_output_dir,
                trajectory_col=trajectory_col_cca,
                verbose=verbose
            )
            
            if cca_pvalue:
                cca_pvalue_test(
                    pseudo_adata=pseudo_adata,
                    column="X_DR_proportion",
                    input_correlation=first_component_score_proportion,
                    output_directory=rna_output_dir,
                    trajectory_col=trajectory_col_cca,
                    verbose=verbose
                )
                cca_pvalue_test(
                    pseudo_adata=pseudo_adata,
                    column="X_DR_expression",
                    input_correlation=first_component_score_expression,
                    output_directory=rna_output_dir,
                    trajectory_col=trajectory_col_cca,
                    verbose=verbose
                )
            status_flags["rna"]["trajectory_analysis"] = True
        else:
            # Unsupervised trajectory analysis
            tscan_result_expression = TSCAN(
                AnnData_sample=pseudo_adata,
                column="X_DR_expression",
                output_dir=rna_output_dir,
                grouping_columns=trajectory_visualization_label,
                verbose=verbose,
                origin=tscan_origin
            )

            tscan_result_proportion = TSCAN(
                AnnData_sample=pseudo_adata,
                column="X_DR_proportion",
                output_dir=rna_output_dir,
                grouping_columns=trajectory_visualization_label,
                verbose=verbose,
                origin=tscan_origin
            )
            ptime_expression = pd.Series(
                tscan_result_expression["pseudotime"]["main_path"],
                name="tscan_pseudotime_expression"
            ).reindex(pseudo_adata.obs.index)

            ptime_proportion = pd.Series(
                tscan_result_proportion["pseudotime"]["main_path"],
                name="tscan_pseudotime_proportion"
            ).reindex(pseudo_adata.obs.index)
            status_flags["rna"]["trajectory_analysis"] = True

        # Trajectory differential gene analysis
        if trajectory_DGE:
            print("Running single-pseudotime GAM-based differential analysis...")

            run_trajectory_gam_differential_gene_analysis(
                pseudobulk_adata=pseudo_adata,
                pseudotime_source=ptime_expression,
                sample_col=sample_col,
                pseudotime_col="pseudotime",
                covariate_columns=trajectory_diff_gene_covariate,
                fdr_threshold=fdr_threshold,
                effect_size_threshold=effect_size_threshold,
                top_n_genes=top_n_genes,
                num_splines=num_splines,
                spline_order=spline_order,
                output_dir=os.path.join(trajectory_diff_gene_output_dir, "expression"),
                visualization_gene_list=visualization_gene_list,
                verbose=verbose
            )

            status_flags["rna"]["trajectory_analysis"] = True
            print("Trajectory differential gene analysis completed!")
            status_flags["rna"]["trajectory_dge"] = True

    
    # Clean up summary file if exists
    if os.path.exists(summary_sample_csv_path):
        os.remove(summary_sample_csv_path)

    # Step 6: Clustering and Differential Gene Expression
    expr_results, prop_results = {}, {}
    if sample_cluster:
        print("Starting clustering and differential gene expression...")

        expr_results, prop_results = cluster(
            pseudobulk_adata=pseudo_adata,      
            output_dir=rna_output_dir,            
            number_of_clusters=cluster_number,
            use_expression=True,
            use_proportion=True,
            random_state=0,
        )

            
    # Proportion test
    if proportion_test:
        print("[INFO] Starting proportion tests...")
        try:    
            if cluster_differential_gene_group_col is not None or len(expr_results) > 0:
                run_proportion_test(
                    adata=adata_sample,
                    sample_col=sample_col,
                    sample_to_clade=expr_results,
                    group_col=cluster_differential_gene_group_col,
                    celltype_col=celltype_col,
                    output_dir=os.path.join(rna_output_dir, "sample_cluster", "expression", "proportion_test"),
                    verbose=True
                )
                    
            if cluster_differential_gene_group_col is not None or len(prop_results) > 0:
                run_proportion_test(
                    adata=adata_sample,
                    sample_col=sample_col,
                    sample_to_clade=prop_results,
                    group_col=cluster_differential_gene_group_col,
                    celltype_col=celltype_col,
                    output_dir=os.path.join(rna_output_dir, "sample_cluster", "proportion", "proportion_test"),
                    verbose=True
                )
            print("[INFO] Proportion tests completed.")
        except Exception as e:
            print(f"[ERROR] Error in proportion test: {e}")
            import traceback
            traceback.print_exc()
    
    # RAISIN analysis
    if RAISIN_analysis:
        print("[INFO] Running RAISIN analysis...")
        try:
            if cluster_differential_gene_group_col is not None or len(expr_results) > 0:
                fit = raisinfit(
                    adata=adata_sample,
                    sample_col=sample_col,
                    testtype='unpaired',
                    batch_col=batch_col,
                    sample_to_clade=expr_results,
                    group_col=cluster_differential_gene_group_col,
                    verbose=verbose,
                    intercept=True,
                    n_jobs=-1,
                )
                run_pairwise_tests(
                    fit=fit,
                    output_dir=os.path.join(rna_output_dir, 'raisin_results_expression'),
                    fdrmethod='fdr_bh',
                    fdr_threshold=0.05,
                    verbose=True
                )
            else:
                print("[INFO] No expression results available. Skipping RAISIN analysis for expression.")
            
            print("[INFO] RAISIN analysis completed.")

        except Exception as e:
            print(f"[ERROR] Error in RAISIN analysis: {e}")
            import traceback
            traceback.print_exc()

    status_flags["rna"]["cluster_dge"] = True
    
    # Step 7: Visualization
    if visualize_data:
        print("Starting visualization...")
        if not status_flags["rna"]["cell_type_cluster"]:
            if plot_dendrogram_flag:
                raise ValueError("Cell type clustering is required before dendrogram visualization.")
        
        if not status_flags["rna"]["derive_sample_embedding"]:
            if plot_cell_type_proportions_pca_flag or plot_cell_type_expression_umap_flag:
                raise ValueError("Sample embedding derivation is required before the requested visualization.")
        
        visualization(
            AnnData_cell=adata_cell,
            pseudobulk_anndata=pseudo_adata, 
            output_dir=rna_output_dir,
            grouping_columns=grouping_columns,
            age_bin_size=age_bin_size,
            age_column=age_column,
            verbose=verbose,
            plot_dendrogram_flag=plot_dendrogram_flag,
            plot_cell_type_proportions_pca_flag=plot_cell_type_proportions_pca_flag,
            plot_cell_type_expression_umap_flag=plot_cell_type_expression_umap_flag
        )
        status_flags["rna"]["visualization"] = True
    
    print("RNA analysis completed successfully!")
    
    return {
        'adata_cell': adata_cell,
        'adata_sample': adata_sample,
        'pseudobulk_df': pseudobulk_df,
        'pseudo_adata': pseudo_adata,
        'status_flags': status_flags
    }