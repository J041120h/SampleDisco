#!/usr/bin/env python3
"""
RNA-seq analysis wrapper for single-cell data processing pipeline.
"""

import os
import sys
import scanpy as sc
import pandas as pd

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
from sample_clustering.RAISIN import raisinfit
from sample_clustering.RAISIN_TEST import run_pairwise_tests
from sample_clustering.proportion_test import proportion_test as run_proportion_test


def rna_wrapper(
    # ===== Required Parameters =====
    rna_count_data_path: str = None,
    rna_output_dir: str = None,
    sample_col: str = 'sample',
    
    # ===== Process Control Flags =====
    preprocessing: bool = True,
    cell_type_cluster: bool = True,
    derive_sample_embedding: bool = True,
    sample_distance_calculation: bool = True,
    trajectory_analysis: bool = True,
    trajectory_DGE: bool = True,
    sample_cluster: bool = True,
    cluster_DGE: bool = True,
    visualize_data: bool = True,
    
    # ===== Paths for Skipping Processes =====
    adata_cell_path: str = None,
    adata_sample_path: str = None,
    pseudo_adata_path: str = None,
    
    # ===== Basic Parameters =====
    rna_sample_meta_path: str = None,
    grouping_columns: list = None,
    celltype_col: str = 'cell_type',
    cell_meta_path: str = None,
    batch_col: str = None,
    leiden_cluster_resolution: float = 0.8,
    cell_embedding_column: str = "X_pca_harmony",
    cell_embedding_num_pcs: int = 20,
    num_harmony_iterations: int = 30,
    num_cell_hvgs: int = 2000,
    min_cells: int = 500,
    min_genes: int = 500,
    pct_mito_cutoff: float = 20,
    exclude_genes: list = None,
    doublet: bool = True,
    metric: str = 'euclidean',
    distance_mode: str = 'centroid',
    vars_to_regress: list = None,
    verbose: bool = True,
    
    # ===== Cell Type Clustering Parameters =====
    existing_cell_types: bool = False,
    n_target_cell_clusters: int = None,
    umap: bool = False,
    
    # ===== Cell Type Assignment Parameters =====
    assign_save: bool = True,
    
    # ===== Cell Type Annotation Parameters =====
    cell_type_annotation: bool = False,
    rna_cell_type_annotation_model_name: str = None,
    rna_cell_type_annotation_custom_model_path: str = None,
    
    # ===== Sample Embedding Parameters =====
    sample_hvg_number: int = 2000,
    preserve_cols_in_sample_embedding: list = None,
    sample_embedding_dimension: int = 10,
    harmony_for_proportion: bool = True,
    
    # ===== Trajectory Analysis Parameters =====
    trajectory_supervised: bool = False,
    n_cca_pcs: int = 2,
    trajectory_col: str = "sev.level",
    cca_optimal_cell_resolution: bool = False,
    cca_pvalue: bool = False,
    tscan_origin: str = None,
    
    # ===== Trajectory Differential Gene Parameters =====
    fdr_threshold: float = 0.05,
    effect_size_threshold: float = 1,
    top_n_genes: int = 100,
    trajectory_diff_gene_covariate: list = None,
    num_splines: int = 5,
    spline_order: int = 3,
    visualization_gene_list: list = None,
    visualize_all_deg: bool = True,
    top_n_heatmap: int = 50,
    
    # ===== Distance Methods =====
    sample_distance_methods: list = None,
    summary_sample_csv_path: str = None,
    
    # ===== Visualization Parameters =====
    trajectory_visualization_label: list = None,
    age_bin_size: int = None,
    age_column: str = 'age',
    dot_size: int = 3,
    plot_dendrogram_flag: bool = True,
    plot_umap_by_cell_type_flag: bool = True,
    plot_cell_type_proportions_pca_flag: bool = False,
    plot_cell_type_expression_umap_flag: bool = False,
    
    # ===== Cluster Based Analysis =====
    kmeans_based_cluster_flag: bool = False,
    tree_building_methods: list = None,
    proportion_test: bool = False,
    RAISIN_analysis: bool = False,
    cluster_distance_method: str = 'cosine',
    cluster_number: int = 4,
    user_provided_sample_to_clade: dict = None,
    cluster_differential_gene_group_col: str = None,
    
    # ===== System Parameters =====
    use_gpu: bool = False,
    status_flags: dict = None
) -> dict:
    """
    RNA-seq analysis wrapper for single-cell data processing pipeline.
    
    Parameters
    ----------
    rna_count_data_path : str
        Path to RNA count data (h5ad format)
    rna_output_dir : str
        Output directory for results
    sample_col : str
        Column name for sample identifiers
    preprocessing : bool
        Whether to run preprocessing
    cell_type_cluster : bool
        Whether to run cell type clustering
    derive_sample_embedding : bool
        Whether to derive sample embeddings
    sample_distance_calculation : bool
        Whether to calculate sample distances
    trajectory_analysis : bool
        Whether to run trajectory analysis
    trajectory_DGE : bool
        Whether to run trajectory differential gene expression
    sample_cluster : bool
        Whether to run sample clustering
    cluster_DGE : bool
        Whether to run cluster differential gene expression
    visualize_data : bool
        Whether to generate visualizations
    use_gpu : bool
        Whether to use GPU acceleration (Linux only)
    
    Returns
    -------
    dict
        Dictionary containing adata_cell, adata_sample, pseudobulk_df, 
        pseudo_adata, and status_flags
    """
    print("Starting RNA wrapper function...")
    
    # === SETUP AND VALIDATION ===
    if rna_count_data_path is None or rna_output_dir is None:
        raise ValueError("Required parameters rna_count_data_path and rna_output_dir must be provided.")
    
    # Import GPU versions if needed
    if use_gpu:
        from preparation.preprocess_linux import preprocess_linux
        from preparation.Cell_type_linux import cell_types_linux
        from parameter_selection.gpu_optimal_resolution import find_optimal_cell_resolution_linux
    
    # Set default values
    if grouping_columns is None:
        grouping_columns = ['sev.level']
    if vars_to_regress is None:
        vars_to_regress = []
    if sample_distance_methods is None:
        sample_distance_methods = ['cosine', 'correlation']
    if trajectory_visualization_label is None:
        trajectory_visualization_label = ['sev.level']
    if tree_building_methods is None:
        tree_building_methods = ['HRA_VEC', 'HRC_VEC', 'NN', 'UPGMA']
    if summary_sample_csv_path is None:
        summary_sample_csv_path = os.path.join(rna_output_dir, 'summary_sample.csv')
    
    trajectory_diff_gene_output_dir = os.path.join(rna_output_dir, 'trajectoryDEG')
    
    # Initialize status flags
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

    # Initialize data objects
    adata_cell = None
    adata_sample = None
    pseudobulk_df = None
    pseudo_adata = None

    # === STEP 1: PREPROCESSING ===
    if preprocessing:
        print("Starting preprocessing...")
        preprocess_func = preprocess_linux if use_gpu else preprocess
        
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
        # Load preprocessed data
        cell_path = adata_cell_path or os.path.join(rna_output_dir, "preprocess", "adata_cell.h5ad")
        sample_path = adata_sample_path or os.path.join(rna_output_dir, "preprocess", "adata_sample.h5ad")
        
        if not os.path.exists(cell_path) or not os.path.exists(sample_path):
            raise ValueError("Preprocessed data paths not provided and default files do not exist.")
        
        status_flags["rna"]["preprocessing"] = True
        status_flags["rna"]["cell_type_cluster"] = True
        
        needs_data = any([
            cell_type_cluster, derive_sample_embedding, trajectory_analysis,
            trajectory_DGE, sample_cluster, cluster_DGE, proportion_test,
            cca_optimal_cell_resolution, RAISIN_analysis, visualize_data
        ])
        
        if needs_data:
            adata_cell = sc.read(cell_path)
            adata_sample = sc.read(sample_path)
    
    if not status_flags["rna"]["preprocessing"]:
        raise ValueError("RNA preprocessing skipped but no preprocessed data found.")

    # === STEP 2: CELL TYPE CLUSTERING ===
    if cell_type_cluster:
        print(f"Starting cell type clustering at resolution: {leiden_cluster_resolution}")
        
        cell_types_func = cell_types_linux if use_gpu else cell_types
        
        adata_cell, adata_sample = cell_types_func(
            anndata_cell=adata_cell,
            anndata_sample=adata_sample,
            cell_type_column=celltype_col,
            existing_cell_types=existing_cell_types,
            n_target_clusters=n_target_cell_clusters,
            umap=umap,
            save=True,
            output_dir=rna_output_dir,
            leiden_cluster_resolution=leiden_cluster_resolution,
            cell_embedding_column=cell_embedding_column,
            cell_embedding_num_PCs=cell_embedding_num_pcs,
            verbose=verbose,
            umap_plots=True,
        )
        status_flags["rna"]["cell_type_cluster"] = True

    # === STEP 3: SAMPLE EMBEDDING ===
    if derive_sample_embedding:
        print("Starting sample embedding derivation...")
        
        if not status_flags["rna"]["cell_type_cluster"]:
            raise ValueError("Cell type clustering required before sample embedding derivation.")
        
        pseudobulk_df, pseudo_adata = calculate_sample_embedding(
            adata=adata_sample,
            sample_col=sample_col,
            celltype_col=celltype_col,
            batch_col=batch_col,
            output_dir=rna_output_dir,
            sample_hvg_number=sample_hvg_number,
            n_expression_components=sample_embedding_dimension,
            n_proportion_components=sample_embedding_dimension,
            harmony_for_proportion=harmony_for_proportion,
            preserve_cols_in_sample_embedding=preserve_cols_in_sample_embedding,
            use_gpu=use_gpu,
            atac=False,
            save=True,
            verbose=verbose,
        )
        status_flags["rna"]["derive_sample_embedding"] = True
    else:
        pseudobulk_path = pseudo_adata_path or os.path.join(rna_output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
        
        needs_pseudobulk = any([trajectory_analysis, trajectory_DGE, sample_cluster, cluster_DGE, visualize_data])
        
        if needs_pseudobulk and not os.path.exists(pseudobulk_path):
            raise ValueError("Sample embedding skipped but no sample embedding data found.")
        
        if os.path.exists(pseudobulk_path):
            print(f"Loading pseudobulk from: {pseudobulk_path}")
            pseudo_adata = sc.read(pseudobulk_path)
        
        status_flags["rna"]["derive_sample_embedding"] = True

    # === STEP 4: OPTIMAL RESOLUTION (OPTIONAL) ===
    if cca_optimal_cell_resolution:
        print("Finding optimal cell resolution...")
        
        find_resolution_func = find_optimal_cell_resolution_linux if use_gpu else find_optimal_cell_resolution
        
        for column in ["X_DR_expression", "X_DR_proportion"]:
            find_resolution_func(
                adata_cell=adata_cell,
                adata_sample=adata_sample,
                output_dir=rna_output_dir,
                column=column,
                trajectory_col=trajectory_col,
                batch_col=batch_col,
                sample_col=sample_col,
                cell_embedding_column=cell_embedding_column,
                cell_embedding_num_pcs=cell_embedding_num_pcs,
                n_hvg_features=sample_hvg_number,
                sample_embedding_dimension=sample_embedding_dimension,
                n_cca_pcs=n_cca_pcs,
                preserve_cols_in_sample_embedding=preserve_cols_in_sample_embedding,
                verbose=verbose,
            )
        
        from utils.unify_optimal import replace_optimal_dimension_reduction
        pseudo_adata = replace_optimal_dimension_reduction(rna_output_dir)

    # === STEP 5: SAMPLE DISTANCE CALCULATION ===
    if sample_distance_calculation:
        print("Starting sample distance calculation...")
        
        for distance_method in sample_distance_methods:
            print(f"Running sample distance: {distance_method}")
            sample_distance(
                adata=pseudo_adata,
                output_dir=os.path.join(rna_output_dir, 'Sample_distance'),
                method=distance_method,
                grouping_columns=grouping_columns,
                summary_csv_path=summary_sample_csv_path,
                cell_adata=adata_cell,
                cell_type_column=celltype_col,
                sample_column=sample_col,
                pseudobulk_adata=pseudo_adata
            )
        
        status_flags["rna"]["sample_distance_calculation"] = True
        if verbose:
            print(f"Sample distance calculation completed: {os.path.join(rna_output_dir, 'Sample_distance')}")

    # === STEP 6: TRAJECTORY ANALYSIS ===
    ptime_expression = None
    ptime_proportion = None
    
    if trajectory_analysis:
        print("Starting trajectory analysis...")
        
        if not status_flags["rna"]["derive_sample_embedding"]:
            raise ValueError("Sample embedding derivation required before trajectory analysis.")
        
        if trajectory_supervised:
            # Supervised trajectory (CCA)
            if trajectory_col not in pseudo_adata.obs.columns:
                raise ValueError(f"Trajectory column '{trajectory_col}' not found in pseudo_adata.obs.")
            
            cca_score_proportion, cca_score_expression, ptime_proportion, ptime_expression = CCA_Call(
                adata=pseudo_adata,
                n_components=n_cca_pcs,
                output_dir=rna_output_dir,
                trajectory_col=trajectory_col,
                verbose=verbose
            )
            
            if cca_pvalue:
                cca_pvalue_test(
                    pseudo_adata=pseudo_adata,
                    column="X_DR_proportion",
                    input_correlation=cca_score_proportion,
                    output_directory=rna_output_dir,
                    trajectory_col=trajectory_col,
                    verbose=verbose
                )
                cca_pvalue_test(
                    pseudo_adata=pseudo_adata,
                    column="X_DR_expression",
                    input_correlation=cca_score_expression,
                    output_directory=rna_output_dir,
                    trajectory_col=trajectory_col,
                    verbose=verbose
                )
        else:
            # Unsupervised trajectory (TSCAN)
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

        # Trajectory Differential Gene Expression
        if trajectory_DGE:
            print("Running trajectory differential gene analysis...")
            
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
            
            status_flags["rna"]["trajectory_dge"] = True
            print("Trajectory differential gene analysis completed!")

    # Clean up summary file if exists
    if os.path.exists(summary_sample_csv_path):
        os.remove(summary_sample_csv_path)

    # === STEP 7: SAMPLE CLUSTERING ===
    expr_results, prop_results = {}, {}
    
    if sample_cluster:
        print("Starting sample clustering...")
        
        expr_results, prop_results = cluster(
            pseudobulk_adata=pseudo_adata,
            output_dir=rna_output_dir,
            number_of_clusters=cluster_number,
            use_expression=True,
            use_proportion=True,
            random_state=0,
        )

    # === STEP 8: PROPORTION TEST ===
    if proportion_test:
        print("Starting proportion tests...")
        
        try:
            if cluster_differential_gene_group_col is not None or expr_results:
                run_proportion_test(
                    adata=adata_sample,
                    sample_col=sample_col,
                    sample_to_clade=expr_results,
                    group_col=cluster_differential_gene_group_col,
                    celltype_col=celltype_col,
                    output_dir=os.path.join(rna_output_dir, "sample_cluster", "expression", "proportion_test"),
                    verbose=True
                )
            
            if cluster_differential_gene_group_col is not None or prop_results:
                run_proportion_test(
                    adata=adata_sample,
                    sample_col=sample_col,
                    sample_to_clade=prop_results,
                    group_col=cluster_differential_gene_group_col,
                    celltype_col=celltype_col,
                    output_dir=os.path.join(rna_output_dir, "sample_cluster", "proportion", "proportion_test"),
                    verbose=True
                )
            
            print("Proportion tests completed.")
        except Exception as e:
            print(f"Error in proportion test: {e}")
            import traceback
            traceback.print_exc()

    # === STEP 9: RAISIN ANALYSIS ===
    if RAISIN_analysis:
        print("Running RAISIN analysis...")
        
        try:
            if cluster_differential_gene_group_col is not None or expr_results:
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
                print("No expression results available. Skipping RAISIN analysis.")
            
            print("RAISIN analysis completed.")
        except Exception as e:
            print(f"Error in RAISIN analysis: {e}")
            import traceback
            traceback.print_exc()

    status_flags["rna"]["cluster_dge"] = True

    # === STEP 10: VISUALIZATION ===
    if visualize_data:
        print("Starting visualization...")
        
        if plot_dendrogram_flag and not status_flags["rna"]["cell_type_cluster"]:
            raise ValueError("Cell type clustering required for dendrogram visualization.")
        
        if (plot_cell_type_proportions_pca_flag or plot_cell_type_expression_umap_flag) and not status_flags["rna"]["derive_sample_embedding"]:
            raise ValueError("Sample embedding derivation required for requested visualization.")
        
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