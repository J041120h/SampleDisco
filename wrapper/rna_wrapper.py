#!/usr/bin/env python3

import os
import sys
import scanpy as sc
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sample_embedding.calculate_sample_embedding import calculate_sample_embedding
from preparation.rna_preprocess_cpu import preprocess
from sample_distance.sample_distance import sample_distance
from visualization.visualization_other import visualization
from sample_trajectory.CCA import CCA_Call
from preparation.cell_type_cpu import cell_types
from sample_trajectory.CCA_test import cca_pvalue_test
from parameter_selection.cpu_optimal_resolution import find_optimal_cell_resolution
from sample_trajectory.TSCAN import TSCAN
from sample_trajectory.trajectory_diff_gene import run_trajectory_gam_differential_gene_analysis
from cluster import cluster
from sample_clustering.RAISIN import raisinfit
from sample_clustering.RAISIN_TEST import run_pairwise_tests
from sample_clustering.proportion_test import proportion_test as run_proportion_test


def rna_wrapper(
    rna_count_data_path: str = None,
    rna_output_dir: str = None,
    
    preprocessing: bool = True,
    cell_type_cluster: bool = True,
    derive_sample_embedding: bool = True,
    cca_based_cell_resolution_selection: bool = False,
    sample_distance_calculation: bool = True,
    trajectory_analysis: bool = True,
    trajectory_DGE: bool = True,
    sample_cluster: bool = True,
    proportion_test: bool = False,
    cluster_DGE: bool = False,
    visualize_data: bool = True,
    
    use_gpu: bool = False,
    verbose: bool = True,
    status_flags: dict = None,
    
    adata_cell_path: str = None,
    adata_sample_path: str = None,
    rna_sample_meta_path: str = None,
    cell_meta_path: str = None,
    sample_col: str = 'sample',
    batch_col: str = None,
    min_cells: int = 500,
    min_genes: int = 500,
    pct_mito_cutoff: float = 20,
    exclude_genes: list = None,
    num_cell_hvgs: int = 2000,
    cell_embedding_num_pcs: int = 20,
    num_harmony_iterations: int = 30,
    vars_to_regress: list = None,
    
    celltype_col: str = 'cell_type',
    leiden_cluster_resolution: float = 0.8,
    cell_embedding_column: str = "X_pca_harmony",
    existing_cell_types: bool = False,
    n_target_cell_clusters: int = None,
    umap: bool = False,
    
    pseudo_adata_path: str = None,
    sample_hvg_number: int = 2000,
    sample_embedding_dimension: int = 10,
    harmony_for_proportion: bool = True,
    preserve_cols_in_sample_embedding: list = None,
    
    n_cca_pcs: int = 2,
    trajectory_col: str = "sev.level",
    
    sample_distance_methods: list = None,
    grouping_columns: list = None,
    summary_sample_csv_path: str = None,
    
    trajectory_supervised: bool = False,
    trajectory_visualization_label: list = None,
    cca_pvalue: bool = False,
    tscan_origin: str = None,
    
    fdr_threshold: float = 0.05,
    effect_size_threshold: float = 1,
    top_n_genes: int = 100,
    trajectory_diff_gene_covariate: list = None,
    num_splines: int = 5,
    spline_order: int = 3,
    visualization_gene_list: list = None,
    
    cluster_number: int = 4,
    
    cluster_differential_gene_group_col: str = None,
    
    age_bin_size: int = None,
    age_column: str = 'age',
    plot_dendrogram_flag: bool = True,
    plot_cell_type_proportions_pca_flag: bool = False,
    plot_cell_type_expression_umap_flag: bool = False,
    
) -> dict:
    print("Starting RNA wrapper function...")
    
    if rna_count_data_path is None or rna_output_dir is None:
        raise ValueError("Required parameters rna_count_data_path and rna_output_dir must be provided.")
    
    if use_gpu:
        from preparation.rna_preprocess_gpu import preprocess_linux
        from preparation.cell_type_gpu import cell_types_linux
        from parameter_selection.gpu_optimal_resolution import find_optimal_cell_resolution_linux
    
    if grouping_columns is None:
        grouping_columns = ['sev.level']
    if vars_to_regress is None:
        vars_to_regress = []
    if sample_distance_methods is None:
        sample_distance_methods = ['cosine', 'correlation']
    if trajectory_visualization_label is None:
        trajectory_visualization_label = ['sev.level']
    if summary_sample_csv_path is None:
        summary_sample_csv_path = os.path.join(rna_output_dir, 'summary_sample.csv')
    
    trajectory_diff_gene_output_dir = os.path.join(rna_output_dir, 'trajectoryDEG')
    
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

    adata_cell = None
    adata_sample = None
    pseudobulk_df = None
    pseudo_adata = None

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
        cell_path = adata_cell_path or os.path.join(rna_output_dir, "preprocess", "adata_cell.h5ad")
        sample_path = adata_sample_path or os.path.join(rna_output_dir, "preprocess", "adata_sample.h5ad")
        
        if not os.path.exists(cell_path) or not os.path.exists(sample_path):
            raise ValueError("Preprocessed data paths not provided and default files do not exist.")
        
        status_flags["rna"]["preprocessing"] = True
        status_flags["rna"]["cell_type_cluster"] = True
        
        needs_data = any([
            cell_type_cluster, derive_sample_embedding, trajectory_analysis,
            trajectory_DGE, sample_cluster, cluster_DGE, proportion_test,
            cca_based_cell_resolution_selection, visualize_data
        ])
        
        if needs_data:
            adata_cell = sc.read(cell_path)
            adata_sample = sc.read(sample_path)
    
    if not status_flags["rna"]["preprocessing"]:
        raise ValueError("RNA preprocessing skipped but no preprocessed data found.")

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

    if cca_based_cell_resolution_selection:
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

    ptime_expression = None
    ptime_proportion = None
    
    if trajectory_analysis:
        print("Starting trajectory analysis...")
        
        if not status_flags["rna"]["derive_sample_embedding"]:
            raise ValueError("Sample embedding derivation required before trajectory analysis.")
        
        if trajectory_supervised:
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

    if os.path.exists(summary_sample_csv_path):
        os.remove(summary_sample_csv_path)

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

    if cluster_DGE:
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