import os
import sys
import json
import time
import shutil
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import pandas as pd

from .rna_wrapper import rna_wrapper
from .atac_wrapper import atac_wrapper


def _coerce_sample_level_batch_col_list(value: Optional[Any]) -> List[str]:
    """Normalize YAML string or list into a list of obs column names."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if x is not None and str(x).strip() != ""]
    return []


def _store_pseudotime_in_obs(pseudo_adata, ptime, col_name: str) -> None:
    """Write pseudotime values into pseudo_adata.obs so downstream modules can find them."""
    if ptime is None:
        return
    try:
        if isinstance(ptime, dict):
            series = pd.Series(ptime)
        elif isinstance(ptime, pd.DataFrame) and "pseudotime" in ptime.columns:
            series = ptime["pseudotime"]
        elif isinstance(ptime, pd.Series):
            series = ptime
        else:
            return
        pseudo_adata.obs[col_name] = pseudo_adata.obs_names.map(series)
    except Exception as e:
        print(f"Warning: could not store pseudotime column '{col_name}': {e}")


def _first_batch_col_for_raisin(
    batch_col: Optional[Union[str, List[str]]],
) -> Optional[str]:
    """RAISIN expects a single batch column name."""
    if batch_col is None:
        return None
    if isinstance(batch_col, str):
        return batch_col
    if isinstance(batch_col, (list, tuple)) and len(batch_col) > 0:
        return str(batch_col[0])
    return None


# =============================================================================
# SHARED DOWNSTREAM ANALYSIS
# =============================================================================

def downstream_analysis(
    # ===== Required =====
    pseudo_adata,
    output_dir: str,
    modality: str,              
    status_flags: dict,
    
    # ===== Data references (needed by some steps) =====
    adata_cell=None,
    adata_sample=None,
    
    # ===== Step control flags =====
    sample_distance_calculation: bool = True,
    trajectory_analysis: bool = True,
    trajectory_DGE: bool = True,
    sample_cluster: bool = True,
    proportion_test: bool = False,
    cluster_DGE: bool = False,
    visualize_data: bool = True,
    # Multiomics-only: embedding visualization
    visualize_embedding: bool = False,
    
    # ===== General settings =====
    use_gpu: bool = False,
    verbose: bool = True,
    
    # ===== Common column names =====
    sample_col: str = 'sample',
    batch_col: Optional[Union[str, List[str]]] = None,
    celltype_col: str = 'cell_type',

    # ===== Sample distance parameters =====
    sample_distance_methods: Optional[List[str]] = None,
    grouping_columns: Optional[List[str]] = None,
    summary_sample_csv_path: Optional[str] = None,
    
    # ===== Trajectory analysis parameters =====
    n_cca_pcs: int = 2,
    trajectory_col: str = "sev.level",
    trajectory_supervised: bool = False,
    trajectory_visualization_label: Optional[List[str]] = None,
    cca_pvalue: bool = False,
    tscan_origin: Optional[str] = None,
    tscan_n_clusters: Optional[int] = None,
    tscan_pseudotime_mode: str = "rank",

    # ===== Trajectory DGE parameters =====
    fdr_threshold: float = 0.05,
    effect_size_threshold: float = 1,
    top_n_genes: int = 100,
    trajectory_diff_gene_covariate: Optional[List] = None,
    num_splines: int = 5,
    spline_order: int = 3,
    visualization_gene_list: Optional[List] = None,
    
    # ===== Sample clustering parameters =====
    cluster_number: int = 4,
    cluster_differential_gene_group_col: Optional[str] = None,
    
    # ===== Visualization parameters =====
    age_bin_size: Optional[int] = None,
    age_column: str = 'age',
    plot_dendrogram_flag: bool = True,
    plot_cell_type_proportions_pca_flag: bool = False,
    plot_cell_type_expression_umap_flag: bool = False,

    # ===== Phenotype prediction parameters =====
    phenotype_prediction: bool = False,
    prediction_target_col: Optional[str] = None,
    prediction_feature_source: Union[str, List[str]] = "expression",
    prediction_task_type: str = "auto",
    prediction_cv: str = "auto",
    prediction_n_permutations: int = 0,

    # ===== Dimension association analysis parameters =====
    dimension_association_analysis: bool = False,
    association_continuous_cols: Optional[List[str]] = None,
    association_categorical_cols: Optional[List[str]] = None,
    association_n_permutations: int = 999,

    # ===== Multiomics embedding visualization parameters =====
    multiomics_modality_col: str = 'modality',
    multiomics_color_col: Optional[str] = None,
    multiomics_visualization_grouping_column: Optional[List[str]] = None,
    multiomics_target_modality: str = 'ATAC',
    multiomics_expression_key: str = 'X_DR_expression',
    multiomics_proportion_key: str = 'X_DR_proportion',
    multiomics_figsize: Tuple[int, int] = (20, 8),
    multiomics_point_size: int = 60,
    multiomics_alpha: float = 0.8,
    multiomics_colormap: str = 'viridis',
    multiomics_show_sample_names: bool = False,
    multiomics_force_data_type: Optional[str] = None,
    
) -> dict:
    """
    Shared downstream analysis after sample embedding derivation.
    
    Works for RNA, ATAC, and multiomics pipelines. Steps include:
    sample distance, trajectory analysis, trajectory DGE, sample clustering,
    proportion test, cluster DGE (RAISIN), and visualization.
    
    Note: CCA-based resolution selection is handled by each individual wrapper,
    not here, because it occurs at different pipeline positions per modality.
    """
    import scanpy as sc
    import pandas as pd
    
    is_multiomics = (modality == "multiomics")
    sf = status_flags[modality]
    
    # Defaults
    if grouping_columns is None:
        grouping_columns = ['sev.level']
    if sample_distance_methods is None:
        sample_distance_methods = ['cosine', 'correlation']
    if trajectory_visualization_label is None:
        trajectory_visualization_label = ['sev.level']
    if summary_sample_csv_path is None:
        summary_sample_csv_path = os.path.join(output_dir, 'summary_sample.csv')
    
    trajectory_diff_gene_output_dir = os.path.join(output_dir, 'trajectoryDEG')

    # ==================== SAMPLE DISTANCE CALCULATION ====================
    if sample_distance_calculation:
        print("Starting sample distance calculation...")
        from sample_distance.sample_distance import sample_distance

        data_type_map = {'rna': 'RNA', 'atac': 'ATAC', 'multiomics': 'multiomics'}
        data_type = data_type_map.get(modality.lower(), 'RNA')

        for distance_method in sample_distance_methods:
            print(f"Running sample distance: {distance_method}")
            sample_distance(
                adata=pseudo_adata,
                output_dir=os.path.join(output_dir, 'Sample_distance'),
                method=distance_method,
                data_type=data_type,
                grouping_columns=grouping_columns,
                summary_csv_path=summary_sample_csv_path,
                cell_adata=adata_cell,
                cell_type_column=celltype_col,
                sample_column=sample_col,
                pseudobulk_adata=pseudo_adata
            )
        
        sf["sample_distance_calculation"] = True
        if verbose:
            print(f"Sample distance calculation completed: {os.path.join(output_dir, 'Sample_distance')}")

    # ==================== TRAJECTORY ANALYSIS ====================
    ptime_expression = None
    ptime_proportion = None
    
    if trajectory_analysis:
        print("Starting trajectory analysis...")
        from sample_trajectory.CCA import CCA_Call
        from sample_trajectory.CCA_test import cca_pvalue_test
        from sample_trajectory.TSCAN import TSCAN
        
        if trajectory_supervised:
            if trajectory_col not in pseudo_adata.obs.columns:
                raise ValueError(f"Trajectory column '{trajectory_col}' not found in pseudo_adata.obs.")
            
            cca_score_proportion, cca_score_expression, ptime_proportion, ptime_expression = CCA_Call(
                adata=pseudo_adata,
                n_components=n_cca_pcs,
                output_dir=output_dir,
                trajectory_col=trajectory_col,
                verbose=verbose
            )
            
            if cca_pvalue:
                cca_pvalue_test(
                    pseudo_adata=pseudo_adata,
                    column="X_DR_proportion",
                    input_correlation=cca_score_proportion,
                    output_directory=output_dir,
                    trajectory_col=trajectory_col,
                    verbose=verbose
                )
                cca_pvalue_test(
                    pseudo_adata=pseudo_adata,
                    column="X_DR_expression",
                    input_correlation=cca_score_expression,
                    output_directory=output_dir,
                    trajectory_col=trajectory_col,
                    verbose=verbose
                )
        else:
            tscan_result_expression = TSCAN(
                AnnData_sample=pseudo_adata,
                column="X_DR_expression",
                n_clusters=tscan_n_clusters,
                output_dir=output_dir,
                grouping_columns=trajectory_visualization_label,
                verbose=verbose,
                origin=tscan_origin,
                pseudotime_mode=tscan_pseudotime_mode,
            )

            tscan_result_proportion = TSCAN(
                AnnData_sample=pseudo_adata,
                column="X_DR_proportion",
                n_clusters=tscan_n_clusters,
                output_dir=output_dir,
                grouping_columns=trajectory_visualization_label,
                verbose=verbose,
                origin=tscan_origin,
                pseudotime_mode=tscan_pseudotime_mode,
            )
            
        # Keep as dictionary - matching CCA output format
        ptime_expression = tscan_result_expression["pseudotime"]["main_path"]
        ptime_proportion = tscan_result_proportion["pseudotime"]["main_path"]
        
        sf["trajectory_analysis"] = True

        # Store pseudotime in obs so prediction/association modules can use it
        _store_pseudotime_in_obs(pseudo_adata, ptime_expression, "pseudotime_expression")
        _store_pseudotime_in_obs(pseudo_adata, ptime_proportion, "pseudotime_proportion")

        # ==================== TRAJECTORY DGE ====================
        if trajectory_DGE:
            print("Running trajectory differential gene analysis...")
            from sample_trajectory.trajectory_diff_gene import run_trajectory_gam_differential_gene_analysis
            
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
            
            run_trajectory_gam_differential_gene_analysis(
                pseudobulk_adata=pseudo_adata,
                pseudotime_source=ptime_proportion,
                sample_col=sample_col,
                pseudotime_col="pseudotime",
                covariate_columns=trajectory_diff_gene_covariate,
                fdr_threshold=fdr_threshold,
                effect_size_threshold=effect_size_threshold,
                top_n_genes=top_n_genes,
                num_splines=num_splines,
                spline_order=spline_order,
                output_dir=os.path.join(trajectory_diff_gene_output_dir, "proportion"),
                visualization_gene_list=visualization_gene_list,
                verbose=verbose
            )
            
            sf["trajectory_dge"] = True
            print("Trajectory differential gene analysis completed!")

    # Clean up summary file if exists
    if os.path.exists(summary_sample_csv_path):
        os.remove(summary_sample_csv_path)

    # ==================== SAMPLE CLUSTERING ====================
    expr_results, prop_results = {}, {}
    
    if sample_cluster:
        print("Starting sample clustering...")
        from cluster import cluster
        
        expr_results, prop_results = cluster(
            pseudobulk_adata=pseudo_adata,
            output_dir=output_dir,
            number_of_clusters=cluster_number,
            use_expression=True,
            use_proportion=True,
            random_state=0,
        )
        
        sf["sample_cluster"] = True

    # ==================== PROPORTION TEST ====================
    if proportion_test:
        print("Starting proportion tests...")
        from sample_clustering.proportion_test import proportion_test as run_proportion_test
        
        try:
            if cluster_differential_gene_group_col is not None or expr_results:
                run_proportion_test(
                    adata=adata_sample,
                    sample_col=sample_col,
                    sample_to_clade=expr_results,
                    group_col=cluster_differential_gene_group_col,
                    celltype_col=celltype_col,
                    output_dir=os.path.join(output_dir, "sample_cluster", "expression", "proportion_test"),
                    verbose=True
                )
            
            if cluster_differential_gene_group_col is not None or prop_results:
                run_proportion_test(
                    adata=adata_sample,
                    sample_col=sample_col,
                    sample_to_clade=prop_results,
                    group_col=cluster_differential_gene_group_col,
                    celltype_col=celltype_col,
                    output_dir=os.path.join(output_dir, "sample_cluster", "proportion", "proportion_test"),
                    verbose=True
                )
            
            sf["proportion_test"] = True
            print("Proportion tests completed.")
        except Exception as e:
            print(f"Error in proportion test: {e}")
            import traceback
            traceback.print_exc()

    # ==================== CLUSTER DGE (RAISIN) ====================
    if cluster_DGE:
        print("Running RAISIN analysis...")
        from sample_clustering.RAISIN import raisinfit
        from sample_clustering.RAISIN_TEST import run_pairwise_tests
        
        try:
            if cluster_differential_gene_group_col is not None or expr_results:
                fit = raisinfit(
                    adata=adata_sample,
                    sample_col=sample_col,
                    testtype='unpaired',
                    batch_col=_first_batch_col_for_raisin(batch_col),
                    sample_to_clade=expr_results,
                    group_col=cluster_differential_gene_group_col,
                    verbose=verbose,
                    intercept=True,
                    n_jobs=-1,
                )
                run_pairwise_tests(
                    fit=fit,
                    output_dir=os.path.join(output_dir, 'raisin_results_expression'),
                    fdrmethod='fdr_bh',
                    fdr_threshold=0.05,
                    verbose=True
                )
            else:
                print("No expression results available. Skipping RAISIN analysis.")
            
            sf["cluster_dge"] = True
            print("RAISIN analysis completed.")
        except Exception as e:
            print(f"Error in RAISIN analysis: {e}")
            import traceback
            traceback.print_exc()

    # ==================== VISUALIZATION ====================
    if visualize_data:
        print("Starting visualization...")
        from visualization.visualization_other import visualization
        
        if plot_dendrogram_flag:
            if adata_cell is None or celltype_col not in adata_cell.obs.columns:
                raise ValueError(
                    f"Cell type column '{celltype_col}' not found in adata_cell.obs; "
                    "dendrogram visualization requires a cell type column. "
                    "Provide an input with the column present, or enable cell type clustering to generate it."
                )
        
        if (plot_cell_type_proportions_pca_flag or plot_cell_type_expression_umap_flag) and not sf.get("derive_sample_embedding", False) and not sf.get("dimensionality_reduction", False):
            raise ValueError("Sample embedding derivation required for requested visualization.")
        
        visualization(
            AnnData_cell=adata_cell,
            pseudobulk_anndata=pseudo_adata,
            output_dir=output_dir,
            grouping_columns=grouping_columns,
            age_bin_size=age_bin_size,
            age_column=age_column,
            verbose=verbose,
            plot_dendrogram_flag=plot_dendrogram_flag,
            plot_cell_type_proportions_pca_flag=plot_cell_type_proportions_pca_flag,
            plot_cell_type_expression_umap_flag=plot_cell_type_expression_umap_flag
        )
        sf["visualization"] = True

    # ==================== MULTIOMICS EMBEDDING VISUALIZATION ====================
    if visualize_embedding and is_multiomics:
        print("Visualizing multimodal embedding...")
        from visualization.multi_omics_visualization import visualize_multimodal_embedding
        
        fig, axes = visualize_multimodal_embedding(
            adata=pseudo_adata,
            modality_col=multiomics_modality_col,
            color_col=multiomics_color_col,
            visualization_grouping_column=multiomics_visualization_grouping_column,
            target_modality=multiomics_target_modality,
            expression_key=multiomics_expression_key,
            proportion_key=multiomics_proportion_key,
            figsize=multiomics_figsize,
            point_size=multiomics_point_size,
            alpha=multiomics_alpha,
            colormap=multiomics_colormap,
            output_dir=output_dir,
            show_sample_names=multiomics_show_sample_names,
            force_data_type=multiomics_force_data_type,
            verbose=verbose,
        )
        sf["embedding_visualization"] = True
        print("Embedding visualization completed successfully")

    # ==================== PHENOTYPE PREDICTION ====================
    if phenotype_prediction and prediction_target_col:
        print("Starting sample prediction...")
        from sample_prediction.predict_sample_phenotype import predict_sample_phenotype

        pred_output_dir = os.path.join(output_dir, "sample_prediction")
        sources = (
            prediction_feature_source
            if isinstance(prediction_feature_source, list)
            else [prediction_feature_source]
        )
        for src in sources:
            try:
                predict_sample_phenotype(
                    pseudo_adata=pseudo_adata,
                    target_col=prediction_target_col,
                    feature_source=src,
                    task_type=prediction_task_type,
                    cv=prediction_cv,
                    n_permutations=prediction_n_permutations,
                    output_dir=pred_output_dir,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"[Prediction] Warning: failed for source={src}: {e}")
                if verbose:
                    import traceback; traceback.print_exc()

        sf["phenotype_prediction"] = True
        print(f"Phenotype prediction completed: {pred_output_dir}")

    # ==================== DIMENSION ASSOCIATION ANALYSIS ====================
    if dimension_association_analysis:
        print("Starting dimension association analysis...")
        from sample_association.association import run_dimension_association_analysis

        assoc_output_dir = os.path.join(output_dir, "sample_association")
        try:
            run_dimension_association_analysis(
                pseudo_adata=pseudo_adata,
                output_dir=assoc_output_dir,
                continuous_cols=association_continuous_cols,
                categorical_cols=association_categorical_cols,
                n_permutations=association_n_permutations,
                sample_col=sample_col,
                verbose=verbose,
            )
        except Exception as e:
            print(f"[Association] Warning: analysis failed: {e}")
            if verbose:
                import traceback; traceback.print_exc()

        sf["dimension_association_analysis"] = True
        print(f"Dimension association analysis completed: {assoc_output_dir}")

    print(f"{modality.upper()} downstream analysis completed!")
    return {'pseudo_adata': pseudo_adata, 'status_flags': status_flags}


# =============================================================================
# MAIN WRAPPER
# =============================================================================

def wrapper(
    output_dir: str,
    
    # Pipeline selection
    run_rna_pipeline: bool = True,
    run_atac_pipeline: bool = False,
    run_multiomics_pipeline: bool = False,
    
    # General settings
    use_gpu: bool = False,
    initialization: bool = True,
    verbose: bool = True,
    save_intermediate: bool = True,
    large_data_need_extra_memory: bool = False,
    
    # ==========================================================================
    # RNA PIPELINE PARAMETERS
    # ==========================================================================
    rna_count_data_path: Optional[str] = None,
    rna_output_dir: Optional[str] = None,
    
    # Pipeline control flags
    rna_preprocessing: bool = True,
    rna_cell_type_cluster: bool = True,
    rna_derive_sample_embedding: bool = True,
    rna_cca_based_cell_resolution_selection: bool = False,
    rna_sample_distance_calculation: bool = True,
    rna_trajectory_analysis: bool = True,
    rna_trajectory_dge: bool = True,
    rna_sample_cluster: bool = True,
    rna_proportion_test: bool = False,
    rna_cluster_dge: bool = False,
    rna_visualize_data: bool = True,
    rna_phenotype_prediction: bool = False,
    rna_dimension_association_analysis: bool = False,

    # Input data paths (for resuming)
    rna_adata_cell_path: Optional[str] = None,
    rna_adata_sample_path: Optional[str] = None,
    rna_sample_meta_path: Optional[str] = None,
    rna_cell_meta_path: Optional[str] = None,
    rna_pseudo_adata_path: Optional[str] = None,
    
    # Common column names
    rna_sample_col: str = 'sample',
    rna_sample_level_batch_col: Optional[Union[str, List[str]]] = None,
    rna_celltype_col: str = 'cell_type',
    
    # Preprocessing parameters
    rna_min_cells: int = 500,
    rna_min_genes: int = 500,
    rna_pct_mito_cutoff: float = 20,
    rna_exclude_genes: Optional[List] = None,
    rna_num_cell_hvgs: int = 2000,
    rna_cell_embedding_num_pcs: int = 20,
    rna_num_harmony_iterations: int = 30,
    rna_cell_level_batch_key: Optional[List] = None,
    
    # Cell type clustering parameters
    rna_leiden_cluster_resolution: float = 0.8,
    rna_cell_embedding_column: Optional[str] = None,
    rna_existing_cell_types: bool = False,
    rna_n_target_cell_clusters: Optional[int] = None,
    rna_umap: bool = False,
    
    # Sample embedding parameters
    rna_sample_hvg_number: int = 2000,
    rna_sample_embedding_dimension: int = 10,
    rna_harmony_for_proportion: bool = True,
    rna_preserve_cols_in_sample_embedding: Optional[List[str]] = None,
    
    # CCA-based resolution selection parameters
    rna_cca_compute_corrected_pvalues: bool = True,
    rna_cca_coarse_start: float = 0.1,
    rna_cca_coarse_end: float = 1.0,
    rna_cca_coarse_step: float = 0.1,
    rna_cca_fine_range: float = 0.02,
    rna_cca_fine_step: float = 0.01,
    
    # Trajectory analysis parameters
    rna_n_cca_pcs: int = 2,
    rna_trajectory_col: str = "sev.level",
    rna_trajectory_supervised: bool = False,
    rna_trajectory_visualization_label: Optional[List[str]] = None,
    rna_cca_pvalue: bool = False,
    rna_tscan_origin: Optional[int] = None,
    rna_tscan_n_clusters: Optional[int] = None,
    rna_tscan_pseudotime_mode: str = "rank",
    
    # Sample distance parameters
    rna_sample_distance_methods: Optional[List[str]] = None,
    rna_grouping_columns: Optional[List[str]] = None,
    rna_summary_sample_csv_path: Optional[str] = None,
    
    # Trajectory DGE parameters
    rna_fdr_threshold: float = 0.05,
    rna_effect_size_threshold: float = 1,
    rna_top_n_genes: int = 100,
    rna_trajectory_diff_gene_covariate: Optional[List] = None,
    rna_num_splines: int = 5,
    rna_spline_order: int = 3,
    rna_visualization_gene_list: Optional[List] = None,
    
    # Sample clustering parameters
    rna_cluster_number: int = 4,
    rna_cluster_differential_gene_group_col: Optional[str] = None,
    
    # Visualization parameters
    rna_age_bin_size: Optional[int] = None,
    rna_age_column: str = 'age',
    rna_plot_dendrogram_flag: bool = True,
    rna_plot_cell_type_proportions_pca_flag: bool = False,
    rna_plot_cell_type_expression_umap_flag: bool = False,

    # Phenotype prediction parameters
    rna_prediction_target_col: Optional[str] = None,
    rna_prediction_feature_source: Union[str, List[str]] = "expression",
    rna_prediction_task_type: str = "auto",
    rna_prediction_cv: str = "auto",
    rna_prediction_n_permutations: int = 0,

    # Dimension association analysis parameters
    rna_association_continuous_cols: Optional[List[str]] = None,
    rna_association_categorical_cols: Optional[List[str]] = None,
    rna_association_n_permutations: int = 999,

    # ==========================================================================
    # ATAC PIPELINE PARAMETERS
    # ==========================================================================
    atac_count_data_path: Optional[str] = None,
    atac_output_dir: Optional[str] = None,
    
    # Pipeline control flags
    atac_preprocessing: bool = True,
    atac_cell_type_cluster: bool = True,
    atac_derive_sample_embedding: bool = True,
    atac_cca_based_cell_resolution_selection: bool = False,
    atac_sample_distance_calculation: bool = True,
    atac_trajectory_analysis: bool = True,
    atac_trajectory_dge: bool = True,
    atac_sample_cluster: bool = True,
    atac_proportion_test: bool = False,
    atac_cluster_dge: bool = False,
    atac_visualize_data: bool = True,
    atac_phenotype_prediction: bool = False,
    atac_dimension_association_analysis: bool = False,

    # Input data paths (for resuming)
    atac_adata_cell_path: Optional[str] = None,
    atac_adata_sample_path: Optional[str] = None,
    atac_sample_meta_path: Optional[str] = None,
    atac_cell_meta_path: Optional[str] = None,
    atac_pseudo_adata_path: Optional[str] = None,
    
    # Common column names
    atac_sample_col: str = "sample",
    atac_sample_level_batch_col: Optional[Union[str, List[str]]] = None,
    atac_celltype_col: str = "cell_type",
    atac_cell_embedding_column: Optional[str] = None,
    
    # ATAC-specific preprocessing parameters
    atac_min_cells: int = 1,
    atac_min_features: int = 2000,
    atac_max_features: int = 15000,
    atac_min_cells_per_sample: int = 1,
    atac_exclude_features: Optional[List] = None,
    atac_cell_level_batch_key: Optional[List] = None,
    atac_doublet_detection: bool = True,
    atac_num_cell_hvfs: int = 50000,
    atac_cell_embedding_num_pcs: int = 50,
    atac_num_harmony_iterations: int = 30,
    atac_tfidf_scale_factor: float = 1e4,
    atac_log_transform: bool = True,
    atac_drop_first_lsi: bool = True,
    
    # Cell type clustering parameters
    atac_leiden_cluster_resolution: float = 0.8,
    atac_existing_cell_types: bool = False,
    atac_n_target_cell_clusters: Optional[int] = None,
    atac_umap: bool = False,
    
    # Sample embedding parameters
    atac_sample_hvg_number: int = 50000,
    atac_sample_embedding_dimension: int = 30,
    atac_harmony_for_proportion: bool = True,
    atac_preserve_cols_in_sample_embedding: Optional[List[str]] = None,
    
    # CCA-based resolution selection parameters
    atac_cca_compute_corrected_pvalues: bool = True,
    atac_cca_coarse_start: float = 0.1,
    atac_cca_coarse_end: float = 1.0,
    atac_cca_coarse_step: float = 0.1,
    atac_cca_fine_range: float = 0.02,
    atac_cca_fine_step: float = 0.01,
    
    # Trajectory analysis parameters
    atac_n_cca_pcs: int = 2,
    atac_trajectory_col: str = "sev.level",
    atac_trajectory_supervised: bool = True,
    atac_trajectory_visualization_label: Optional[List[str]] = None,
    atac_cca_pvalue: bool = False,
    atac_tscan_origin: Optional[str] = None,
    atac_tscan_n_clusters: Optional[int] = None,
    atac_tscan_pseudotime_mode: str = "rank",
    
    # Sample distance parameters
    atac_sample_distance_methods: Optional[List[str]] = None,
    atac_grouping_columns: Optional[List[str]] = None,
    atac_summary_sample_csv_path: Optional[str] = None,
    
    # Trajectory DGE parameters
    atac_fdr_threshold: float = 0.05,
    atac_effect_size_threshold: float = 1.0,
    atac_top_n_genes: int = 100,
    atac_trajectory_diff_gene_covariate: Optional[List] = None,
    atac_num_splines: int = 5,
    atac_spline_order: int = 3,
    atac_visualization_gene_list: Optional[List] = None,
    
    # Sample clustering parameters
    atac_cluster_number: int = 4,
    atac_cluster_differential_gene_group_col: Optional[str] = None,
    
    # Visualization parameters
    atac_age_bin_size: Optional[int] = None,
    atac_age_column: str = 'age',
    atac_plot_dendrogram_flag: bool = True,
    atac_plot_cell_type_proportions_pca_flag: bool = False,
    atac_plot_cell_type_expression_umap_flag: bool = False,

    # Phenotype prediction parameters
    atac_prediction_target_col: Optional[str] = None,
    atac_prediction_feature_source: Union[str, List[str]] = "expression",
    atac_prediction_task_type: str = "auto",
    atac_prediction_cv: str = "auto",
    atac_prediction_n_permutations: int = 0,

    # Dimension association analysis parameters
    atac_association_continuous_cols: Optional[List[str]] = None,
    atac_association_categorical_cols: Optional[List[str]] = None,
    atac_association_n_permutations: int = 999,

    # ==========================================================================
    # MULTIOMICS PIPELINE PARAMETERS
    # ==========================================================================
    multiomics_rna_file: Optional[str] = None,
    multiomics_atac_file: Optional[str] = None,
    multiomics_output_dir: Optional[str] = None,
    
    # Pipeline control flags -- preprocessing & resolution
    multiomics_integration: bool = True,
    multiomics_integration_preprocessing: bool = True,
    multiomics_dimensionality_reduction: bool = True,
    multiomics_find_optimal_resolution: bool = False,
    # Pipeline control flags -- downstream
    multiomics_sample_distance_calculation: bool = True,
    multiomics_trajectory_analysis: bool = True,
    multiomics_trajectory_dge: bool = True,
    multiomics_sample_cluster: bool = True,
    multiomics_proportion_test: bool = False,
    multiomics_cluster_dge: bool = False,
    multiomics_visualize_embedding: bool = True,
    multiomics_phenotype_prediction: bool = False,
    multiomics_dimension_association_analysis: bool = False,
    
    # GLUE sub-pipeline flags
    multiomics_run_glue_preprocessing: bool = True,
    multiomics_run_glue_training: bool = True,
    multiomics_run_glue_gene_activity: bool = True,
    multiomics_cell_type_cluster: bool = True,
    multiomics_run_glue_visualization: bool = True,
    
    # Input data paths (for resuming)
    multiomics_integrated_h5ad_path: Optional[str] = None,
    multiomics_pseudobulk_h5ad_path: Optional[str] = None,
    multiomics_rna_sample_meta_file: Optional[str] = None,
    multiomics_atac_sample_meta_file: Optional[str] = None,
    multiomics_additional_hvg_file: Optional[str] = None,
    
    # Common column names
    multiomics_rna_sample_column: str = "sample",
    multiomics_atac_sample_column: str = "sample",
    multiomics_sample_col: str = 'sample',
    multiomics_batch_col: Optional[str] = None,
    multiomics_celltype_col: str = 'cell_type',
    multiomics_modality_col: str = 'modality',
    
    # General multiomics settings
    multiomics_verbose: bool = True,
    multiomics_use_gpu: bool = True,
    multiomics_random_state: int = 42,
    
    # GLUE preprocessing parameters
    multiomics_ensembl_release: int = 98,
    multiomics_species: str = "homo_sapiens",
    multiomics_use_highly_variable: bool = True,
    multiomics_n_top_genes: int = 2000,
    multiomics_n_pca_comps: int = 50,
    multiomics_n_lsi_comps: int = 50,
    multiomics_lsi_n_iter: int = 15,
    multiomics_gtf_by: str = "gene_name",
    multiomics_flavor: str = "seurat_v3",
    multiomics_generate_umap: bool = False,
    multiomics_compression: str = "gzip",
    
    # GLUE training parameters
    multiomics_consistency_threshold: float = 0.05,
    multiomics_treat_sample_as_batch: bool = True,
    multiomics_save_prefix: str = "glue",
    
    # Neighbor/metric parameters
    multiomics_k_neighbors: int = 10,
    multiomics_use_rep: str = "X_glue",
    multiomics_metric: str = "cosine",
    
    # Cell type clustering parameters
    multiomics_existing_cell_types: bool = False,
    multiomics_n_target_clusters: int = 10,
    multiomics_cluster_resolution: float = 0.8,
    multiomics_use_rep_celltype: str = "X_glue",
    multiomics_markers: Optional[Dict] = None,
    multiomics_generate_umap_celltype: bool = True,
    
    # Visualization parameters
    multiomics_plot_columns: Optional[List[str]] = None,
    
    # Integration preprocessing parameters
    multiomics_min_cells_sample: int = 1,
    multiomics_min_cell_gene: int = 10,
    multiomics_min_features: int = 500,
    multiomics_pct_mito_cutoff: int = 20,
    multiomics_exclude_genes: Optional[List] = None,
    multiomics_doublet: bool = True,
    
    # Sample embedding parameters
    multiomics_sample_hvg_number: int = 2000,
    multiomics_preserve_cols_for_sample_embedding: Optional[List[str]] = None,
    multiomics_n_expression_components: int = 10,
    multiomics_n_proportion_components: int = 10,
    multiomics_harmony_for_proportion: bool = True,
    
    # Optimal resolution parameters
    multiomics_optimization_target: str = "rna",
    multiomics_sev_col: str = "sev.level",
    multiomics_resolution_use_rep: str = 'X_glue',
    multiomics_num_pcs: int = 20,
    multiomics_visualize_cell_types: bool = True,
    multiomics_cca_coarse_start: float = 0.1,
    multiomics_cca_coarse_end: float = 1.0,
    multiomics_cca_coarse_step: float = 0.1,
    multiomics_cca_fine_range: float = 0.05,
    multiomics_cca_fine_step: float = 0.01,
    multiomics_compute_corrected_pvalues: bool = False,
    multiomics_analyze_modality_alignment: bool = True,
    
    # Trajectory analysis parameters
    multiomics_trajectory_col: str = "sev.level",
    multiomics_trajectory_supervised: bool = False,
    multiomics_trajectory_visualization_label: Optional[List[str]] = None,
    multiomics_n_cca_pcs: int = 2,
    multiomics_cca_pvalue: bool = False,
    multiomics_tscan_origin: Optional[str] = None,
    multiomics_tscan_n_clusters: Optional[int] = None,
    multiomics_tscan_pseudotime_mode: str = "rank",
    
    # Sample distance parameters
    multiomics_sample_distance_methods: Optional[List[str]] = None,
    multiomics_grouping_columns: Optional[List[str]] = None,
    multiomics_summary_sample_csv_path: Optional[str] = None,
    
    # Trajectory DGE parameters
    multiomics_fdr_threshold: float = 0.05,
    multiomics_effect_size_threshold: float = 1.0,
    multiomics_top_n_genes: int = 100,
    multiomics_trajectory_diff_gene_covariate: Optional[List] = None,
    multiomics_num_splines: int = 5,
    multiomics_spline_order: int = 3,
    multiomics_visualization_gene_list: Optional[List] = None,
    
    # Sample clustering parameters
    multiomics_cluster_number: int = 4,
    multiomics_cluster_differential_gene_group_col: Optional[str] = None,
    
    # Visualization parameters (general)
    multiomics_age_bin_size: Optional[int] = None,
    multiomics_age_column: str = 'age',
    multiomics_plot_dendrogram_flag: bool = True,
    multiomics_plot_cell_type_proportions_pca_flag: bool = False,
    multiomics_plot_cell_type_expression_umap_flag: bool = False,
    
    # Embedding visualization parameters
    multiomics_color_col: Optional[str] = None,
    multiomics_visualization_grouping_column: Optional[List[str]] = None,
    multiomics_target_modality: str = 'ATAC',
    multiomics_expression_key: str = 'X_DR_expression',
    multiomics_proportion_key: str = 'X_DR_proportion',
    multiomics_figsize: Tuple[int, int] = (20, 8),
    multiomics_point_size: int = 60,
    multiomics_alpha: float = 0.8,
    multiomics_colormap: str = 'viridis',
    multiomics_show_sample_names: bool = False,
    multiomics_force_data_type: Optional[str] = None,

    # Phenotype prediction parameters
    multiomics_prediction_target_col: Optional[str] = None,
    multiomics_prediction_feature_source: Union[str, List[str]] = "expression",
    multiomics_prediction_task_type: str = "auto",
    multiomics_prediction_cv: str = "auto",
    multiomics_prediction_n_permutations: int = 0,

    # Dimension association analysis parameters
    multiomics_association_continuous_cols: Optional[List[str]] = None,
    multiomics_association_categorical_cols: Optional[List[str]] = None,
    multiomics_association_n_permutations: int = 999,

) -> Dict[str, Any]:
    """
    Main wrapper function that orchestrates RNA, ATAC, and Multiomics pipelines.
    
    Each pipeline runs in two phases:
    1. Modality-specific preprocessing + resolution selection (individual wrappers)
    2. Shared downstream analysis (downstream_analysis())
    
    Parameters
    ----------
    output_dir : str
        Base output directory for all pipelines.
    run_rna_pipeline : bool
        Whether to run the RNA pipeline.
    run_atac_pipeline : bool
        Whether to run the ATAC pipeline.
    run_multiomics_pipeline : bool
        Whether to run the Multiomics pipeline.
    use_gpu : bool
        Whether to use GPU acceleration (requires Linux and CUDA).
    initialization : bool
        Whether to initialize/reset the pipeline status.
    verbose : bool
        Whether to print verbose output.
    save_intermediate : bool
        Whether to save intermediate results.
    large_data_need_extra_memory : bool
        Whether to use managed memory for large datasets.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing results from all executed pipelines.
    """
    start_time = time.time()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check system capabilities
    is_linux = platform.system() == "Linux"
    gpu_available = is_linux and use_gpu
    
    system_info = {
        'platform': platform.system(),
        'python_version': sys.version,
        'use_gpu': use_gpu,
        'gpu_available': gpu_available
    }
    
    # Initialize GPU if available; fall back to CPU on any import or runtime error
    if gpu_available:
        try:
            import rmm
            import cupy as cp
            from rmm.allocators.cupy import rmm_cupy_allocator

            rmm.reinitialize(
                managed_memory=large_data_need_extra_memory,
                pool_allocator=not large_data_need_extra_memory,
            )
            cp.cuda.set_allocator(rmm_cupy_allocator)
        except Exception as e:
            print(f"Warning: GPU initialization failed ({type(e).__name__}: {e}). Falling back to CPU.")
            gpu_available = False
            system_info['gpu_available'] = False
    
    # Install GPU dependencies if needed
    if gpu_available and initialization:
        try:
            print("Installing GPU dependencies...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "rapids-singlecell[rapids12]",
                "--extra-index-url=https://pypi.nvidia.com"
            ])
            print("GPU dependencies installed successfully.")
        except Exception as e:
            print(f"Warning: Failed to install GPU dependencies: {e}")
            print("Continuing without GPU acceleration.")
            gpu_available = False
            system_info['gpu_available'] = False
    
    # Initialize status tracking
    status_file_path = os.path.join(output_dir, "sys_log", "main_process_status.json")
    os.makedirs(os.path.dirname(status_file_path), exist_ok=True)
    
    default_pipeline_status = {
        "preprocessing": False,
        "cell_type_cluster": False,
        "derive_sample_embedding": False,
        "cca_based_cell_resolution_selection": False,
        "sample_distance_calculation": False,
        "trajectory_analysis": False,
        "trajectory_dge": False,
        "sample_cluster": False,
        "proportion_test": False,
        "cluster_dge": False,
        "visualization": False,
        "phenotype_prediction": False,
        "dimension_association_analysis": False,
    }
    
    status_flags = {
        "rna": default_pipeline_status.copy(),
        "atac": default_pipeline_status.copy(),
        "multiomics": {
            "glue_integration": False,
            "glue_preprocessing": False,
            "glue_training": False,
            "glue_gene_activity": False,
            "glue_cell_types": False,
            "glue_visualization": False,
            "integration_preprocessing": False,
            "optimal_resolution": False,
            "dimensionality_reduction": False,
            "cca_based_cell_resolution_selection": False,
            "sample_distance_calculation": False,
            "trajectory_analysis": False,
            "trajectory_dge": False,
            "sample_cluster": False,
            "proportion_test": False,
            "cluster_dge": False,
            "embedding_visualization": False,
            "visualization": False,
            "phenotype_prediction": False,
            "dimension_association_analysis": False,
        },
        "system_info": system_info
    }
    
    # Load or initialize status
    if os.path.exists(status_file_path) and not initialization:
        try:
            with open(status_file_path, 'r') as f:
                saved_status = json.load(f)
                for key in ['rna', 'atac', 'multiomics']:
                    if key in saved_status:
                        status_flags[key].update(saved_status[key])
            print("Resuming from previous progress:")
            print(json.dumps(status_flags, indent=2))
        except Exception as e:
            print(f"Error reading status file: {e}. Reinitializing.")
    else:
        if initialization:
            for subdir in ['rna', 'atac', 'multiomics']:
                result_dir = os.path.join(output_dir, subdir, "result")
                if os.path.exists(result_dir):
                    try:
                        shutil.rmtree(result_dir)
                        print(f"Removed existing directory: {result_dir}")
                    except Exception as e:
                        print(f"Failed to remove directory: {e}")
        
        print("Initializing pipeline status.")
        with open(status_file_path, 'w') as f:
            json.dump(status_flags, f, indent=2)
    
    results = {'status_flags': status_flags, 'system_info': system_info}

    # ==================== RNA PIPELINE ====================
    if run_rna_pipeline:
        print("\n" + "=" * 60)
        print("RUNNING RNA PIPELINE")
        print("=" * 60)
        
        rna_output_dir = rna_output_dir or os.path.join(output_dir, 'rna')
        
        if rna_count_data_path is None:
            raise ValueError("RNA pipeline requires rna_count_data_path")
        
        try:
            rna_slb_list = _coerce_sample_level_batch_col_list(rna_sample_level_batch_col)
            # Phase 1: Preprocessing + resolution selection
            rna_results = rna_wrapper(
                rna_count_data_path=rna_count_data_path,
                rna_output_dir=rna_output_dir,
                # Pipeline control
                preprocessing=rna_preprocessing,
                cell_type_cluster=rna_cell_type_cluster,
                derive_sample_embedding=rna_derive_sample_embedding,
                cca_based_cell_resolution_selection=rna_cca_based_cell_resolution_selection,
                # General
                use_gpu=gpu_available,
                verbose=verbose,
                status_flags=status_flags,
                # Input paths
                adata_cell_path=rna_adata_cell_path,
                adata_sample_path=rna_adata_sample_path,
                rna_sample_meta_path=rna_sample_meta_path,
                cell_meta_path=rna_cell_meta_path,
                pseudo_adata_path=rna_pseudo_adata_path,
                # Column names
                sample_col=rna_sample_col,
                sample_level_batch_col=rna_slb_list,
                celltype_col=rna_celltype_col,
                # Preprocessing
                min_cells=rna_min_cells,
                min_genes=rna_min_genes,
                pct_mito_cutoff=rna_pct_mito_cutoff,
                exclude_genes=rna_exclude_genes,
                num_cell_hvgs=rna_num_cell_hvgs,
                cell_embedding_num_pcs=rna_cell_embedding_num_pcs,
                num_harmony_iterations=rna_num_harmony_iterations,
                cell_level_batch_key=rna_cell_level_batch_key,
                # Cell type clustering
                leiden_cluster_resolution=rna_leiden_cluster_resolution,
                cell_embedding_column=rna_cell_embedding_column,
                existing_cell_types=rna_existing_cell_types,
                n_target_cell_clusters=rna_n_target_cell_clusters,
                umap=rna_umap,
                # Sample embedding
                sample_hvg_number=rna_sample_hvg_number,
                sample_embedding_dimension=rna_sample_embedding_dimension,
                harmony_for_proportion=rna_harmony_for_proportion,
                preserve_cols_in_sample_embedding=rna_preserve_cols_in_sample_embedding,
                # CCA resolution selection
                trajectory_col=rna_trajectory_col,
                n_cca_pcs=rna_n_cca_pcs,
                cca_compute_corrected_pvalues=rna_cca_compute_corrected_pvalues,
                cca_coarse_start=rna_cca_coarse_start,
                cca_coarse_end=rna_cca_coarse_end,
                cca_coarse_step=rna_cca_coarse_step,
                cca_fine_range=rna_cca_fine_range,
                cca_fine_step=rna_cca_fine_step,
            )
            status_flags = rna_results['status_flags']
            
            # Phase 2: Downstream analysis
            downstream_results = downstream_analysis(
                pseudo_adata=rna_results['pseudo_adata'],
                output_dir=rna_output_dir,
                modality="rna",
                status_flags=status_flags,
                adata_cell=rna_results['adata_cell'],
                adata_sample=rna_results['adata_sample'],
                # Step control
                sample_distance_calculation=rna_sample_distance_calculation,
                trajectory_analysis=rna_trajectory_analysis,
                trajectory_DGE=rna_trajectory_dge,
                sample_cluster=rna_sample_cluster,
                proportion_test=rna_proportion_test,
                cluster_DGE=rna_cluster_dge,
                visualize_data=rna_visualize_data,
                # General
                use_gpu=gpu_available,
                verbose=verbose,
                # Column names
                sample_col=rna_sample_col,
                batch_col=rna_slb_list or None,
                celltype_col=rna_celltype_col,
                # Sample distance
                sample_distance_methods=rna_sample_distance_methods or ['cosine', 'correlation'],
                grouping_columns=rna_grouping_columns or ['sev.level'],
                summary_sample_csv_path=rna_summary_sample_csv_path,
                # Trajectory
                n_cca_pcs=rna_n_cca_pcs,
                trajectory_col=rna_trajectory_col,
                trajectory_supervised=rna_trajectory_supervised,
                trajectory_visualization_label=rna_trajectory_visualization_label or ['sev.level'],
                cca_pvalue=rna_cca_pvalue,
                tscan_origin=rna_tscan_origin,
                tscan_n_clusters=rna_tscan_n_clusters,
                tscan_pseudotime_mode=rna_tscan_pseudotime_mode,
                # Trajectory DGE
                fdr_threshold=rna_fdr_threshold,
                effect_size_threshold=rna_effect_size_threshold,
                top_n_genes=rna_top_n_genes,
                trajectory_diff_gene_covariate=rna_trajectory_diff_gene_covariate,
                num_splines=rna_num_splines,
                spline_order=rna_spline_order,
                visualization_gene_list=rna_visualization_gene_list,
                # Sample clustering
                cluster_number=rna_cluster_number,
                cluster_differential_gene_group_col=rna_cluster_differential_gene_group_col,
                # Visualization
                age_bin_size=rna_age_bin_size,
                age_column=rna_age_column,
                plot_dendrogram_flag=rna_plot_dendrogram_flag,
                plot_cell_type_proportions_pca_flag=rna_plot_cell_type_proportions_pca_flag,
                plot_cell_type_expression_umap_flag=rna_plot_cell_type_expression_umap_flag,
                # Phenotype prediction
                phenotype_prediction=rna_phenotype_prediction,
                prediction_target_col=rna_prediction_target_col,
                prediction_feature_source=rna_prediction_feature_source,
                prediction_task_type=rna_prediction_task_type,
                prediction_cv=rna_prediction_cv,
                prediction_n_permutations=rna_prediction_n_permutations,
                # Dimension association analysis
                dimension_association_analysis=rna_dimension_association_analysis,
                association_continuous_cols=rna_association_continuous_cols,
                association_categorical_cols=rna_association_categorical_cols,
                association_n_permutations=rna_association_n_permutations,
            )

            status_flags = downstream_results['status_flags']
            results['rna_results'] = {**rna_results, **downstream_results}
            _save_status(status_file_path, status_flags)
            print("\nRNA pipeline completed successfully!")
            
        except Exception as e:
            print(f"\nRNA pipeline failed: {e}")
            results['rna_error'] = str(e)
            if verbose:
                import traceback
                traceback.print_exc()

    # ==================== ATAC PIPELINE ====================
    if run_atac_pipeline:
        print("\n" + "=" * 60)
        print("RUNNING ATAC PIPELINE")
        print("=" * 60)
        
        atac_output_dir = atac_output_dir or os.path.join(output_dir, 'atac')
        
        if atac_count_data_path is None:
            raise ValueError("ATAC pipeline requires atac_count_data_path")
        
        try:
            atac_slb_list = _coerce_sample_level_batch_col_list(atac_sample_level_batch_col)
            # Phase 1: Preprocessing + resolution selection
            atac_results = atac_wrapper(
                atac_count_data_path=atac_count_data_path,
                atac_output_dir=atac_output_dir,
                # Pipeline control
                preprocessing=atac_preprocessing,
                cell_type_cluster=atac_cell_type_cluster,
                derive_sample_embedding=atac_derive_sample_embedding,
                cca_based_cell_resolution_selection=atac_cca_based_cell_resolution_selection,
                # General
                use_gpu=gpu_available,
                verbose=verbose,
                status_flags=status_flags,
                # Input paths
                adata_cell_path=atac_adata_cell_path,
                adata_sample_path=atac_adata_sample_path,
                atac_sample_meta_path=atac_sample_meta_path,
                cell_meta_path=atac_cell_meta_path,
                pseudo_adata_path=atac_pseudo_adata_path,
                # Column names
                sample_col=atac_sample_col,
                sample_level_batch_col=atac_slb_list,
                celltype_col=atac_celltype_col,
                cell_embedding_column=atac_cell_embedding_column,
                # Preprocessing
                min_cells=atac_min_cells,
                min_features=atac_min_features,
                max_features=atac_max_features,
                min_cells_per_sample=atac_min_cells_per_sample,
                exclude_features=atac_exclude_features,
                cell_level_batch_key=atac_cell_level_batch_key,
                doublet_detection=atac_doublet_detection,
                num_cell_hvfs=atac_num_cell_hvfs,
                cell_embedding_num_pcs=atac_cell_embedding_num_pcs,
                num_harmony_iterations=atac_num_harmony_iterations,
                tfidf_scale_factor=atac_tfidf_scale_factor,
                log_transform=atac_log_transform,
                drop_first_lsi=atac_drop_first_lsi,
                # Cell type clustering
                leiden_cluster_resolution=atac_leiden_cluster_resolution,
                existing_cell_types=atac_existing_cell_types,
                n_target_cell_clusters=atac_n_target_cell_clusters,
                umap=atac_umap,
                # Sample embedding
                sample_hvg_number=atac_sample_hvg_number,
                sample_embedding_dimension=atac_sample_embedding_dimension,
                harmony_for_proportion=atac_harmony_for_proportion,
                preserve_cols_in_sample_embedding=atac_preserve_cols_in_sample_embedding,
                # CCA resolution selection
                trajectory_col=atac_trajectory_col,
                n_cca_pcs=atac_n_cca_pcs,
                cca_compute_corrected_pvalues=atac_cca_compute_corrected_pvalues,
                cca_coarse_start=atac_cca_coarse_start,
                cca_coarse_end=atac_cca_coarse_end,
                cca_coarse_step=atac_cca_coarse_step,
                cca_fine_range=atac_cca_fine_range,
                cca_fine_step=atac_cca_fine_step,
            )
            status_flags = atac_results['status_flags']
            
            # Phase 2: Downstream analysis
            downstream_results = downstream_analysis(
                pseudo_adata=atac_results['pseudo_adata'],
                output_dir=atac_output_dir,
                modality="atac",
                status_flags=status_flags,
                adata_cell=atac_results['adata_cell'],
                adata_sample=atac_results['adata_sample'],
                # Step control
                sample_distance_calculation=atac_sample_distance_calculation,
                trajectory_analysis=atac_trajectory_analysis,
                trajectory_DGE=atac_trajectory_dge,
                sample_cluster=atac_sample_cluster,
                proportion_test=atac_proportion_test,
                cluster_DGE=atac_cluster_dge,
                visualize_data=atac_visualize_data,
                # General
                use_gpu=gpu_available,
                verbose=verbose,
                # Column names
                sample_col=atac_sample_col,
                batch_col=atac_slb_list or None,
                celltype_col=atac_celltype_col,
                # Sample distance
                sample_distance_methods=atac_sample_distance_methods or ['cosine', 'correlation'],
                grouping_columns=atac_grouping_columns or ['sev.level'],
                summary_sample_csv_path=atac_summary_sample_csv_path,
                # Trajectory
                n_cca_pcs=atac_n_cca_pcs,
                trajectory_col=atac_trajectory_col,
                trajectory_supervised=atac_trajectory_supervised,
                trajectory_visualization_label=atac_trajectory_visualization_label or ['sev.level'],
                cca_pvalue=atac_cca_pvalue,
                tscan_origin=atac_tscan_origin,
                tscan_n_clusters=atac_tscan_n_clusters,
                tscan_pseudotime_mode=atac_tscan_pseudotime_mode,
                # Trajectory DGE
                fdr_threshold=atac_fdr_threshold,
                effect_size_threshold=atac_effect_size_threshold,
                top_n_genes=atac_top_n_genes,
                trajectory_diff_gene_covariate=atac_trajectory_diff_gene_covariate,
                num_splines=atac_num_splines,
                spline_order=atac_spline_order,
                visualization_gene_list=atac_visualization_gene_list,
                # Sample clustering
                cluster_number=atac_cluster_number,
                cluster_differential_gene_group_col=atac_cluster_differential_gene_group_col,
                # Visualization
                age_bin_size=atac_age_bin_size,
                age_column=atac_age_column,
                plot_dendrogram_flag=atac_plot_dendrogram_flag,
                plot_cell_type_proportions_pca_flag=atac_plot_cell_type_proportions_pca_flag,
                plot_cell_type_expression_umap_flag=atac_plot_cell_type_expression_umap_flag,
                # Phenotype prediction
                phenotype_prediction=atac_phenotype_prediction,
                prediction_target_col=atac_prediction_target_col,
                prediction_feature_source=atac_prediction_feature_source,
                prediction_task_type=atac_prediction_task_type,
                prediction_cv=atac_prediction_cv,
                prediction_n_permutations=atac_prediction_n_permutations,
                # Dimension association analysis
                dimension_association_analysis=atac_dimension_association_analysis,
                association_continuous_cols=atac_association_continuous_cols,
                association_categorical_cols=atac_association_categorical_cols,
                association_n_permutations=atac_association_n_permutations,
            )

            status_flags = downstream_results['status_flags']
            results['atac_results'] = {**atac_results, **downstream_results}
            _save_status(status_file_path, status_flags)
            print("\nATAC pipeline completed successfully!")
            
        except Exception as e:
            print(f"\nATAC pipeline failed: {e}")
            results['atac_error'] = str(e)
            if verbose:
                import traceback
                traceback.print_exc()

    # ==================== MULTIOMICS PIPELINE ====================
    if run_multiomics_pipeline:
        print("\n" + "=" * 60)
        print("RUNNING MULTIOMICS PIPELINE")
        print("=" * 60)
        
        multiomics_output_dir = multiomics_output_dir or os.path.join(output_dir, 'multiomics')
        
        if multiomics_integration and (multiomics_rna_file is None or multiomics_atac_file is None):
            raise ValueError("Multiomics pipeline with GLUE requires multiomics_rna_file and multiomics_atac_file")
        
        try:
            from .multiomics_wrapper import multiomics_wrapper
            
            # Phase 1: Preprocessing + resolution selection + sample embedding
            multiomics_results = multiomics_wrapper(
                rna_file=multiomics_rna_file,
                atac_file=multiomics_atac_file,
                multiomics_output_dir=multiomics_output_dir,
                # Pipeline control
                integration=multiomics_integration,
                integration_preprocessing=multiomics_integration_preprocessing,
                dimensionality_reduction=multiomics_dimensionality_reduction,
                find_optimal_resolution=multiomics_find_optimal_resolution,
                # Basic parameters
                rna_sample_meta_file=multiomics_rna_sample_meta_file,
                atac_sample_meta_file=multiomics_atac_sample_meta_file,
                additional_hvg_file=multiomics_additional_hvg_file,
                rna_sample_column=multiomics_rna_sample_column,
                atac_sample_column=multiomics_atac_sample_column,
                sample_col=multiomics_sample_col,
                batch_col=multiomics_batch_col,
                celltype_col=multiomics_celltype_col,
                modality_col=multiomics_modality_col,
                multiomics_verbose=multiomics_verbose,
                save_intermediate=save_intermediate,
                use_gpu=multiomics_use_gpu and gpu_available,
                random_state=multiomics_random_state,
                # GLUE flags
                run_glue_preprocessing=multiomics_run_glue_preprocessing,
                run_glue_training=multiomics_run_glue_training,
                run_glue_gene_activity=multiomics_run_glue_gene_activity,
                cell_type_cluster=multiomics_cell_type_cluster,
                run_glue_visualization=multiomics_run_glue_visualization,
                # GLUE preprocessing
                ensembl_release=multiomics_ensembl_release,
                species=multiomics_species,
                use_highly_variable=multiomics_use_highly_variable,
                n_top_genes=multiomics_n_top_genes,
                n_pca_comps=multiomics_n_pca_comps,
                n_lsi_comps=multiomics_n_lsi_comps,
                lsi_n_iter=multiomics_lsi_n_iter,
                gtf_by=multiomics_gtf_by,
                flavor=multiomics_flavor,
                generate_umap=multiomics_generate_umap,
                compression=multiomics_compression,
                # GLUE training
                consistency_threshold=multiomics_consistency_threshold,
                treat_sample_as_batch=multiomics_treat_sample_as_batch,
                save_prefix=multiomics_save_prefix,
                # GLUE gene activity
                k_neighbors=multiomics_k_neighbors,
                use_rep=multiomics_use_rep,
                metric=multiomics_metric,
                # Cell type
                existing_cell_types=multiomics_existing_cell_types,
                n_target_clusters=multiomics_n_target_clusters,
                cluster_resolution=multiomics_cluster_resolution,
                use_rep_celltype=multiomics_use_rep_celltype,
                markers=multiomics_markers,
                generate_umap_celltype=multiomics_generate_umap_celltype,
                plot_columns=multiomics_plot_columns,
                # Integration preprocessing
                min_cells_sample=multiomics_min_cells_sample,
                min_cell_gene=multiomics_min_cell_gene,
                min_features=multiomics_min_features,
                pct_mito_cutoff=multiomics_pct_mito_cutoff,
                exclude_genes=multiomics_exclude_genes,
                doublet=multiomics_doublet,
                # Sample embedding
                sample_hvg_number=multiomics_sample_hvg_number,
                preserve_cols_for_sample_embedding=multiomics_preserve_cols_for_sample_embedding,
                n_expression_components=multiomics_n_expression_components,
                n_proportion_components=multiomics_n_proportion_components,
                multiomics_harmony_for_proportion=multiomics_harmony_for_proportion,
                # Optimal resolution
                optimization_target=multiomics_optimization_target,
                sev_col=multiomics_sev_col,
                resolution_use_rep=multiomics_resolution_use_rep,
                num_PCs=multiomics_num_pcs,
                visualize_cell_types=multiomics_visualize_cell_types,
                coarse_start=multiomics_cca_coarse_start,
                coarse_end=multiomics_cca_coarse_end,
                coarse_step=multiomics_cca_coarse_step,
                fine_range=multiomics_cca_fine_range,
                fine_step=multiomics_cca_fine_step,
                compute_corrected_pvalues=multiomics_compute_corrected_pvalues,
                analyze_modality_alignment=multiomics_analyze_modality_alignment,
                # Paths for skipping
                integrated_h5ad_path=multiomics_integrated_h5ad_path,
                pseudobulk_h5ad_path=multiomics_pseudobulk_h5ad_path,
                status_flags=status_flags,
            )
            status_flags = multiomics_results['status_flags']
            
            multiomics_adata_cell = multiomics_results.get("adata")
            if multiomics_adata_cell is None:
                import scanpy as sc
                _cell_paths = []
                if multiomics_integrated_h5ad_path:
                    _cell_paths.append(multiomics_integrated_h5ad_path)
                _cell_paths.append(
                    os.path.join(multiomics_output_dir, "preprocess", "adata_sample.h5ad")
                )
                for _p in _cell_paths:
                    if _p and os.path.exists(_p):
                        multiomics_adata_cell = sc.read(_p)
                        break
            
            # Phase 2: Downstream analysis
            downstream_results = downstream_analysis(
                pseudo_adata=multiomics_results['pseudo_adata'],
                output_dir=multiomics_output_dir,
                modality="multiomics",
                status_flags=status_flags,
                adata_cell=multiomics_adata_cell,
                adata_sample=multiomics_results.get('adata_sample'),
                # Step control
                sample_distance_calculation=multiomics_sample_distance_calculation,
                trajectory_analysis=multiomics_trajectory_analysis,
                trajectory_DGE=multiomics_trajectory_dge,
                sample_cluster=multiomics_sample_cluster,
                proportion_test=multiomics_proportion_test,
                cluster_DGE=multiomics_cluster_dge,
                visualize_data=False,
                visualize_embedding=multiomics_visualize_embedding,
                # General
                use_gpu=multiomics_use_gpu and gpu_available,
                verbose=multiomics_verbose,
                # Column names
                sample_col=multiomics_sample_col,
                batch_col=multiomics_batch_col,
                celltype_col=multiomics_celltype_col,
                # Sample distance
                sample_distance_methods=multiomics_sample_distance_methods or ['cosine', 'correlation'],
                grouping_columns=multiomics_grouping_columns or ['sev.level'],
                summary_sample_csv_path=multiomics_summary_sample_csv_path,
                # Trajectory
                n_cca_pcs=multiomics_n_cca_pcs,
                trajectory_col=multiomics_trajectory_col,
                trajectory_supervised=multiomics_trajectory_supervised,
                trajectory_visualization_label=multiomics_trajectory_visualization_label or ['sev.level'],
                cca_pvalue=multiomics_cca_pvalue,
                tscan_origin=multiomics_tscan_origin,
                tscan_n_clusters=multiomics_tscan_n_clusters,
                tscan_pseudotime_mode=multiomics_tscan_pseudotime_mode,
                # Trajectory DGE
                fdr_threshold=multiomics_fdr_threshold,
                effect_size_threshold=multiomics_effect_size_threshold,
                top_n_genes=multiomics_top_n_genes,
                trajectory_diff_gene_covariate=multiomics_trajectory_diff_gene_covariate,
                num_splines=multiomics_num_splines,
                spline_order=multiomics_spline_order,
                visualization_gene_list=multiomics_visualization_gene_list,
                # Sample clustering
                cluster_number=multiomics_cluster_number,
                cluster_differential_gene_group_col=multiomics_cluster_differential_gene_group_col,
                # Visualization
                age_bin_size=multiomics_age_bin_size,
                age_column=multiomics_age_column,
                plot_dendrogram_flag=multiomics_plot_dendrogram_flag,
                plot_cell_type_proportions_pca_flag=multiomics_plot_cell_type_proportions_pca_flag,
                plot_cell_type_expression_umap_flag=multiomics_plot_cell_type_expression_umap_flag,
                # Multiomics embedding visualization
                multiomics_modality_col=multiomics_modality_col,
                multiomics_color_col=multiomics_color_col,
                multiomics_visualization_grouping_column=multiomics_visualization_grouping_column,
                multiomics_target_modality=multiomics_target_modality,
                multiomics_expression_key=multiomics_expression_key,
                multiomics_proportion_key=multiomics_proportion_key,
                multiomics_figsize=multiomics_figsize,
                multiomics_point_size=multiomics_point_size,
                multiomics_alpha=multiomics_alpha,
                multiomics_colormap=multiomics_colormap,
                multiomics_show_sample_names=multiomics_show_sample_names,
                multiomics_force_data_type=multiomics_force_data_type,
                # Phenotype prediction
                phenotype_prediction=multiomics_phenotype_prediction,
                prediction_target_col=multiomics_prediction_target_col,
                prediction_feature_source=multiomics_prediction_feature_source,
                prediction_task_type=multiomics_prediction_task_type,
                prediction_cv=multiomics_prediction_cv,
                prediction_n_permutations=multiomics_prediction_n_permutations,
                # Dimension association analysis
                dimension_association_analysis=multiomics_dimension_association_analysis,
                association_continuous_cols=multiomics_association_continuous_cols,
                association_categorical_cols=multiomics_association_categorical_cols,
                association_n_permutations=multiomics_association_n_permutations,
            )

            status_flags = downstream_results['status_flags']
            results['multiomics_results'] = {**multiomics_results, **downstream_results}
            _save_status(status_file_path, status_flags)
            print("\nMultiomics pipeline completed successfully!")
            
        except Exception as e:
            print(f"\nMultiomics pipeline failed: {e}")
            results['multiomics_error'] = str(e)
            if verbose:
                import traceback
                traceback.print_exc()
    
    # Final summary
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    return results


def _save_status(status_file_path: str, status_flags: Dict) -> None:
    """Save status flags to JSON file."""
    with open(status_file_path, 'w') as f:
        json.dump(status_flags, f, indent=2)