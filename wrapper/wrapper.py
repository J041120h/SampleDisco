#!/usr/bin/env python3
"""
Comprehensive main wrapper for single-cell multi-modal analysis pipelines.

This wrapper integrates RNA-seq, ATAC-seq, and multiomics analysis pipelines
with proper status tracking, GPU support, and flexible parameter control.
"""

import os
import sys
import json
import time
import shutil
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .rna_wrapper import rna_wrapper
from .atac_wrapper import atac_wrapper


def wrapper(
    # ========================================
    # REQUIRED PARAMETERS
    # ========================================
    output_dir: str,
    
    # ========================================
    # PIPELINE CONTROL FLAGS
    # ========================================
    run_rna_pipeline: bool = True,
    run_atac_pipeline: bool = False,
    run_multiomics_pipeline: bool = False,
    
    # System configuration
    use_gpu: bool = False,
    initialization: bool = True,
    verbose: bool = True,
    save_intermediate: bool = True,
    large_data_need_extra_memory: bool = False,
    
    # ========================================
    # RNA PIPELINE PARAMETERS
    # ========================================
    # Required RNA parameters
    rna_count_data_path: Optional[str] = None,
    rna_sample_meta_path: Optional[str] = None,
    rna_output_dir: Optional[str] = None,
    
    # RNA Process Control Flags
    rna_preprocessing: bool = True,
    rna_cell_type_cluster: bool = True,
    rna_derive_sample_embedding: bool = True,
    rna_sample_distance_calculation: bool = True,
    rna_trajectory_analysis: bool = True,
    rna_trajectory_dge: bool = True,
    rna_sample_cluster: bool = True,
    rna_cluster_dge: bool = True,
    rna_visualize_data: bool = True,
    
    # RNA Paths for Skipping Processes
    rna_adata_cell_path: Optional[str] = None,
    rna_adata_sample_path: Optional[str] = None,
    rna_pseudo_adata_path: Optional[str] = None,
    
    # RNA Basic Parameters
    rna_sample_col: str = 'sample',
    rna_grouping_columns: Optional[List[str]] = None,
    rna_celltype_col: str = 'cell_type',
    rna_cell_meta_path: Optional[str] = None,
    rna_batch_col: Optional[str] = None,
    rna_leiden_cluster_resolution: float = 0.8,
    rna_cell_embedding_column: str = "X_pca_harmony",
    rna_cell_embedding_num_pcs: int = 20,
    rna_num_harmony_iterations: int = 30,
    rna_num_cell_hvgs: int = 2000,
    rna_min_cells: int = 500,
    rna_min_genes: int = 500,
    rna_pct_mito_cutoff: float = 20,
    rna_exclude_genes: Optional[List] = None,
    rna_doublet: bool = True,
    rna_metric: str = 'euclidean',
    rna_distance_mode: str = 'centroid',
    rna_vars_to_regress: Optional[List] = None,
    
    # RNA Cell Type Clustering Parameters
    rna_existing_cell_types: bool = False,
    rna_n_target_cell_clusters: Optional[int] = None,
    rna_umap: bool = False,
    rna_assign_save: bool = True,
    
    # RNA Cell Type Annotation Parameters
    rna_cell_type_annotation: bool = False,
    rna_cell_type_annotation_model_name: Optional[str] = None,
    rna_cell_type_annotation_custom_model_path: Optional[str] = None,
    
    # RNA Sample Embedding Parameters
    rna_sample_hvg_number: int = 2000,
    rna_preserve_cols_in_sample_embedding: Optional[List[str]] = None,
    rna_sample_embedding_dimension: int = 10,
    rna_harmony_for_proportion: bool = True,
    
    # RNA Trajectory Analysis Parameters
    rna_trajectory_supervised: bool = False,
    rna_n_cca_pcs: int = 2,
    rna_trajectory_col: str = "sev.level",
    rna_cca_optimal_cell_resolution: bool = False,
    rna_cca_pvalue: bool = False,
    rna_tscan_origin: Optional[int] = None,
    
    # RNA Trajectory Differential Gene Parameters
    rna_fdr_threshold: float = 0.05,
    rna_effect_size_threshold: float = 1,
    rna_top_n_genes: int = 100,
    rna_trajectory_diff_gene_covariate: Optional[List] = None,
    rna_num_splines: int = 5,
    rna_spline_order: int = 3,
    rna_visualization_gene_list: Optional[List] = None,
    rna_visualize_all_deg: bool = True,
    rna_top_n_heatmap: int = 50,
    
    # RNA Distance Methods
    rna_summary_sample_csv_path: Optional[str] = None,
    rna_sample_distance_methods: Optional[List[str]] = None,
    
    # RNA Visualization Parameters
    rna_trajectory_visualization_label: Optional[List[str]] = None,
    rna_age_bin_size: Optional[int] = None,
    rna_age_column: str = 'age',
    rna_dot_size: int = 3,
    rna_plot_dendrogram_flag: bool = True,
    rna_plot_umap_by_cell_type_flag: bool = True,
    rna_plot_cell_type_proportions_pca_flag: bool = False,
    rna_plot_cell_type_expression_umap_flag: bool = False,
    
    # RNA Cluster Based Analysis
    rna_kmeans_based_cluster_flag: bool = False,
    rna_tree_building_methods: Optional[List[str]] = None,
    rna_proportion_test: bool = False,
    rna_raisin_analysis: bool = False,
    rna_cluster_distance_method: str = 'cosine',
    rna_cluster_number: int = 4,
    rna_user_provided_sample_to_clade: Optional[Dict] = None,
    rna_cluster_differential_gene_group_col: Optional[str] = None,
    
    # ========================================
    # ATAC PIPELINE PARAMETERS
    # ========================================
    # Required ATAC parameters
    atac_count_data_path: Optional[str] = None,
    atac_output_dir: Optional[str] = None,
    atac_metadata_path: Optional[str] = None,
    
    # ATAC Process Control Flags
    atac_preprocessing: bool = True,
    atac_cell_type_cluster: bool = True,
    atac_pseudobulk_dimensionality_reduction: bool = True,
    atac_visualization_processing: bool = True,
    atac_trajectory_analysis: bool = True,
    atac_sample_distance_calculation: bool = True,
    atac_sample_cluster: bool = True,
    atac_cluster_dge: bool = True,
    atac_trajectory_dge: bool = True,
    
    # ATAC Paths for Skipping Processes
    atac_cell_path: Optional[str] = None,
    atac_sample_path: Optional[str] = None,
    atac_pseudobulk_adata_path: Optional[str] = None,
    
    # ATAC Column specifications
    atac_sample_col: str = "sample",
    atac_batch_col: Optional[List[str]] = None,
    atac_cell_type_column: str = "cell_type",
    atac_grouping_columns: Optional[List[str]] = None,
    
    # ATAC Pipeline configuration
    atac_pipeline_verbose: bool = True,
    use_snapatac2_dimred: bool = False,
    
    # ATAC QC and filtering parameters
    atac_min_cells: int = 3,
    atac_min_genes: int = 2000,
    atac_max_genes: int = 15000,
    atac_min_cells_per_sample: int = 10,
    
    # ATAC Processing parameters
    atac_doublet: bool = True,
    atac_tfidf_scale_factor: float = 1e4,
    atac_num_features: int = 40000,
    atac_n_lsi_components: int = 30,
    atac_drop_first_lsi: bool = True,
    atac_harmony_max_iter: int = 30,
    atac_n_neighbors: int = 15,
    atac_n_pcs: int = 30,
    
    # ATAC Leiden clustering parameters
    atac_leiden_resolution: float = 0.8,
    atac_existing_cell_types: bool = False,
    atac_n_target_cell_clusters: Optional[int] = None,
    atac_preserve_cols_in_sample_embedding: Optional[List[str]] = None,
    
    # ATAC UMAP parameters
    atac_umap_min_dist: float = 0.3,
    atac_umap_spread: float = 1.0,
    atac_umap_random_state: int = 42,
    atac_plot_dpi: int = 300,
    
    # ATAC Pseudobulk parameters
    atac_pseudobulk_output_dir: Optional[str] = None,
    atac_pseudobulk_n_features: int = 50000,
    atac_pseudobulk_verbose: bool = True,
    
    # ATAC Sample Embedding parameters
    atac_dr_output_dir: Optional[str] = None,
    atac_sample_embedding_dimension: int = 30,
    atac_harmony_for_proportion: bool = True,
    atac_dr_verbose: bool = True,
    
    # ATAC Trajectory analysis parameters
    atac_trajectory_supervised: bool = True,
    atac_n_cca_pcs: int = 2,
    atac_cca_output_dir: Optional[str] = None,
    atac_trajectory_col: str = "sev.level",
    atac_trajectory_verbose: bool = True,
    atac_cca_pvalue: bool = False,
    atac_cca_optimal_cell_resolution: bool = False,
    atac_tscan_origin: Optional[int] = None,
    atac_trajectory_visualization_label: Optional[List[str]] = None,
    
    # ATAC Trajectory differential analysis parameters
    atac_fdr_threshold: float = 0.05,
    atac_effect_size_threshold: float = 1,
    atac_top_n_genes: int = 100,
    atac_trajectory_diff_gene_covariate: Optional[List] = None,
    atac_num_splines: int = 5,
    atac_spline_order: int = 3,
    atac_visualization_gene_list: Optional[List] = None,
    atac_visualize_all_deg: bool = True,
    atac_top_n_heatmap: int = 50,
    atac_trajectory_diff_gene_verbose: bool = True,
    atac_top_gene_number: int = 30,
    atac_trajectory_diff_gene_output_dir: Optional[str] = None,
    
    # ATAC Sample distance parameters
    atac_sample_distance_methods: Optional[List[str]] = None,
    atac_summary_sample_csv_path: Optional[str] = None,
    
    # ATAC Clustering parameters
    atac_kmeans_based_cluster_flag: bool = False,
    atac_tree_building_methods: Optional[List[str]] = None,
    atac_proportion_test: bool = False,
    atac_raisin_analysis: bool = False,
    atac_cluster_distance_method: str = 'cosine',
    atac_cluster_number: int = 4,
    atac_user_provided_sample_to_clade: Optional[Dict] = None,
    atac_cluster_differential_gene_group_col: Optional[str] = None,
    
    # ATAC Visualization parameters
    atac_figsize: Tuple[int, int] = (10, 8),
    atac_point_size: int = 50,
    atac_visualization_grouping_columns: Optional[List[str]] = None,
    atac_show_sample_names: bool = True,
    atac_visualization_age_size: Optional[int] = None,
    atac_age_bin_size: Optional[int] = None,
    atac_verbose_visualization: bool = True,
    atac_dot_size: int = 3,
    atac_plot_dendrogram_flag: bool = True,
    atac_plot_cell_umap_by_plot_group_flag: bool = True,
    atac_plot_umap_by_cell_type_flag: bool = True,
    atac_plot_pca_2d_flag: bool = True,
    atac_plot_pca_3d_flag: bool = True,
    atac_plot_3d_cells_flag: bool = True,
    atac_plot_cell_type_proportions_pca_flag: bool = True,
    atac_plot_cell_type_expression_pca_flag: bool = True,
    atac_plot_pseudobulk_batch_test_expression_flag: bool = False,
    atac_plot_pseudobulk_batch_test_proportion_flag: bool = False,
    
    # ========================================
    # MULTIOMICS PIPELINE PARAMETERS
    # ========================================
    # Required Parameters
    multiomics_rna_file: Optional[str] = None,
    multiomics_atac_file: Optional[str] = None,
    multiomics_output_dir: Optional[str] = None,
    
    # Process Control Flags
    multiomics_run_glue: bool = True,
    multiomics_run_integrate_preprocess: bool = True,
    multiomics_run_dimensionality_reduction: bool = True,
    multiomics_run_visualize_embedding: bool = True,
    multiomics_run_find_optimal_resolution: bool = False,
    
    # GLUE Sub-Step Control Flags
    multiomics_run_glue_preprocessing: bool = True,
    multiomics_run_glue_training: bool = True,
    multiomics_run_glue_gene_activity: bool = True,
    multiomics_run_glue_cell_types: bool = True,
    multiomics_run_glue_visualization: bool = True,
    
    # Basic Parameters
    multiomics_rna_sample_meta_file: Optional[str] = None,
    multiomics_atac_sample_meta_file: Optional[str] = None,
    multiomics_additional_hvg_file: Optional[str] = None,
    multiomics_rna_sample_column: str = "sample",
    multiomics_atac_sample_column: str = "sample",
    multiomics_sample_col: str = 'sample',
    multiomics_batch_col: Optional[List[str]] = None,
    multiomics_celltype_col: str = 'cell_type',
    multiomics_modality_col: str = 'modality',
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
    multiomics_gtf_by: str = "gene_name",
    multiomics_flavor: str = "seurat_v3",
    multiomics_generate_umap: bool = False,
    multiomics_compression: str = "gzip",
    
    # GLUE training parameters
    multiomics_consistency_threshold: float = 0.05,
    multiomics_treat_sample_as_batch: bool = True,
    multiomics_save_prefix: str = "glue",
    
    # GLUE gene activity parameters
    multiomics_k_neighbors: int = 10,
    multiomics_use_rep: str = "X_glue",
    multiomics_metric: str = "cosine",
    
    # GLUE cell type parameters
    multiomics_existing_cell_types: bool = False,
    multiomics_n_target_clusters: int = 10,
    multiomics_cluster_resolution: float = 0.8,
    multiomics_use_rep_celltype: str = "X_glue",
    multiomics_markers: Optional[Dict] = None,
    multiomics_generate_umap_celltype: bool = True,
    
    # GLUE visualization parameters
    multiomics_plot_columns: Optional[List[str]] = None,
    
    # Integration Preprocessing Parameters
    multiomics_min_cells_sample: int = 1,
    multiomics_min_cell_gene: int = 10,
    multiomics_min_features: int = 500,
    multiomics_pct_mito_cutoff: int = 20,
    multiomics_exclude_genes: Optional[List] = None,
    multiomics_doublet: bool = True,
    
    # Dimensionality Reduction Parameters
    multiomics_sample_hvg_number: int = 2000,
    multiomics_preserve_cols_in_sample_embedding: Optional[List[str]] = None,
    multiomics_sample_embedding_dimension: int = 10,
    multiomics_harmony_for_proportion: bool = True,
    
    # Visualization Parameters
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
    
    # Optimal Resolution Parameters
    multiomics_optimization_target: str = "rna",
    multiomics_trajectory_col: str = "sev.level",
    multiomics_resolution_use_rep: str = 'X_glue',
    multiomics_num_pcs: int = 20,
    multiomics_visualize_cell_types: bool = True,
    
    # Paths for Skipping Steps
    multiomics_integrated_h5ad_path: Optional[str] = None,
    multiomics_pseudobulk_h5ad_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive main wrapper for single-cell multi-modal analysis pipelines.
    
    This wrapper integrates RNA-seq, ATAC-seq, and multiomics analysis pipelines
    with proper status tracking, GPU support, and flexible parameter control.
    
    Parameters
    ----------
    output_dir : str
        Main output directory for all pipeline results
    run_rna_pipeline : bool
        Whether to run the RNA-seq pipeline
    run_atac_pipeline : bool
        Whether to run the ATAC-seq pipeline
    run_multiomics_pipeline : bool
        Whether to run the multiomics integration pipeline
    use_gpu : bool
        Whether to use GPU acceleration (requires Linux and CUDA)
    initialization : bool
        Whether to initialize/reset the pipeline status
    verbose : bool
        Whether to print verbose output
    save_intermediate : bool
        Whether to save intermediate results
    large_data_need_extra_memory : bool
        Whether to use managed memory for large datasets
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing results from each pipeline and status flags
    """
    start_time = time.time()
    
    # === SETUP ===
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check system and GPU availability
    is_linux = platform.system() == "Linux"
    gpu_available = is_linux and use_gpu
    
    system_info = {
        'platform': platform.system(),
        'python_version': sys.version,
        'use_gpu': use_gpu,
        'gpu_available': gpu_available
    }
    
    # Import GPU-dependent modules
    if run_multiomics_pipeline and gpu_available:
        from .multiomics_wrapper import multiomics_wrapper
    
    # Configure GPU memory management
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
        except ImportError as e:
            print(f"Warning: GPU libraries not available: {e}")
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
    
    # === INITIALIZE STATUS FLAGS ===
    status_file_path = os.path.join(output_dir, "sys_log", "main_process_status.json")
    os.makedirs(os.path.dirname(status_file_path), exist_ok=True)
    
    default_pipeline_status = {
        "preprocessing": False,
        "cell_type_cluster": False,
        "derive_sample_embedding": False,
        "sample_distance_calculation": False,
        "trajectory_analysis": False,
        "trajectory_dge": False,
        "cluster_dge": False,
        "visualization": False
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
            "derive_sample_embedding": False,
            "embedding_visualization": False,
            "optimal_resolution": False
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
            # Clean up existing result directories
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
    
    results = {
        'status_flags': status_flags,
        'system_info': system_info
    }
    
    # === SET DEFAULT VALUES ===
    rna_grouping_columns = rna_grouping_columns or ['sev.level']
    rna_trajectory_visualization_label = rna_trajectory_visualization_label or ['sev.level']
    rna_tree_building_methods = rna_tree_building_methods or ['HRA_VEC', 'HRC_VEC', 'NN', 'UPGMA']
    rna_sample_distance_methods = rna_sample_distance_methods or ['cosine', 'correlation']
    
    atac_grouping_columns = atac_grouping_columns or ['sev.level']
    atac_visualization_grouping_columns = atac_visualization_grouping_columns or ['current_severity']
    atac_tree_building_methods = atac_tree_building_methods or ['HRA_VEC', 'HRC_VEC', 'NN', 'UPGMA']
    atac_sample_distance_methods = atac_sample_distance_methods or ['cosine', 'correlation']
    atac_trajectory_visualization_label = atac_trajectory_visualization_label or ['sev.level']
    
    # === RUN RNA PIPELINE ===
    if run_rna_pipeline:
        print("\n" + "=" * 60)
        print("RUNNING RNA PIPELINE")
        print("=" * 60)
        
        rna_output_dir = rna_output_dir or os.path.join(output_dir, 'rna')
        
        if rna_count_data_path is None:
            raise ValueError("RNA pipeline requires rna_count_data_path")
        
        try:
            rna_results = rna_wrapper(
                # Required
                rna_count_data_path=rna_count_data_path,
                rna_output_dir=rna_output_dir,
                sample_col=rna_sample_col,
                
                # Process Control
                preprocessing=rna_preprocessing,
                cell_type_cluster=rna_cell_type_cluster,
                derive_sample_embedding=rna_derive_sample_embedding,
                sample_distance_calculation=rna_sample_distance_calculation,
                trajectory_analysis=rna_trajectory_analysis,
                trajectory_DGE=rna_trajectory_dge,
                sample_cluster=rna_sample_cluster,
                cluster_DGE=rna_cluster_dge,
                visualize_data=rna_visualize_data,
                
                # Paths for Skipping
                adata_cell_path=rna_adata_cell_path,
                adata_sample_path=rna_adata_sample_path,
                pseudo_adata_path=rna_pseudo_adata_path,
                
                # Basic Parameters
                rna_sample_meta_path=rna_sample_meta_path,
                grouping_columns=rna_grouping_columns,
                celltype_col=rna_celltype_col,
                cell_meta_path=rna_cell_meta_path,
                batch_col=rna_batch_col,
                leiden_cluster_resolution=rna_leiden_cluster_resolution,
                cell_embedding_column=rna_cell_embedding_column,
                cell_embedding_num_pcs=rna_cell_embedding_num_pcs,
                num_harmony_iterations=rna_num_harmony_iterations,
                num_cell_hvgs=rna_num_cell_hvgs,
                min_cells=rna_min_cells,
                min_genes=rna_min_genes,
                pct_mito_cutoff=rna_pct_mito_cutoff,
                exclude_genes=rna_exclude_genes,
                doublet=rna_doublet,
                metric=rna_metric,
                distance_mode=rna_distance_mode,
                vars_to_regress=rna_vars_to_regress,
                verbose=verbose,
                
                # Cell Type Clustering
                existing_cell_types=rna_existing_cell_types,
                n_target_cell_clusters=rna_n_target_cell_clusters,
                umap=rna_umap,
                assign_save=rna_assign_save,
                
                # Cell Type Annotation
                cell_type_annotation=rna_cell_type_annotation,
                rna_cell_type_annotation_model_name=rna_cell_type_annotation_model_name,
                rna_cell_type_annotation_custom_model_path=rna_cell_type_annotation_custom_model_path,
                
                # Sample Embedding (updated parameter names)
                sample_hvg_number=rna_sample_hvg_number,
                preserve_cols_in_sample_embedding=rna_preserve_cols_in_sample_embedding,
                sample_embedding_dimension=rna_sample_embedding_dimension,
                harmony_for_proportion=rna_harmony_for_proportion,
                
                # Trajectory Analysis (updated parameter names)
                trajectory_supervised=rna_trajectory_supervised,
                n_cca_pcs=rna_n_cca_pcs,
                trajectory_col=rna_trajectory_col,
                cca_optimal_cell_resolution=rna_cca_optimal_cell_resolution,
                cca_pvalue=rna_cca_pvalue,
                tscan_origin=rna_tscan_origin,
                
                # Trajectory DGE
                fdr_threshold=rna_fdr_threshold,
                effect_size_threshold=rna_effect_size_threshold,
                top_n_genes=rna_top_n_genes,
                trajectory_diff_gene_covariate=rna_trajectory_diff_gene_covariate,
                num_splines=rna_num_splines,
                spline_order=rna_spline_order,
                visualization_gene_list=rna_visualization_gene_list,
                visualize_all_deg=rna_visualize_all_deg,
                top_n_heatmap=rna_top_n_heatmap,
                
                # Distance
                sample_distance_methods=rna_sample_distance_methods,
                summary_sample_csv_path=rna_summary_sample_csv_path,
                
                # Visualization
                trajectory_visualization_label=rna_trajectory_visualization_label,
                age_bin_size=rna_age_bin_size,
                age_column=rna_age_column,
                dot_size=rna_dot_size,
                plot_dendrogram_flag=rna_plot_dendrogram_flag,
                plot_umap_by_cell_type_flag=rna_plot_umap_by_cell_type_flag,
                plot_cell_type_proportions_pca_flag=rna_plot_cell_type_proportions_pca_flag,
                plot_cell_type_expression_umap_flag=rna_plot_cell_type_expression_umap_flag,
                
                # Cluster Analysis
                kmeans_based_cluster_flag=rna_kmeans_based_cluster_flag,
                tree_building_methods=rna_tree_building_methods,
                proportion_test=rna_proportion_test,
                RAISIN_analysis=rna_raisin_analysis,
                cluster_distance_method=rna_cluster_distance_method,
                cluster_number=rna_cluster_number,
                user_provided_sample_to_clade=rna_user_provided_sample_to_clade,
                cluster_differential_gene_group_col=rna_cluster_differential_gene_group_col,
                
                # System
                use_gpu=gpu_available,
                status_flags=status_flags,
            )
            
            results['rna_results'] = rna_results
            
            if 'status_flags' in rna_results:
                status_flags = rna_results['status_flags']
            
            _save_status(status_file_path, status_flags)
            print("\nRNA pipeline completed successfully!")
            
        except Exception as e:
            print(f"\nRNA pipeline failed: {e}")
            results['rna_error'] = str(e)
            if verbose:
                import traceback
                traceback.print_exc()
    
    # === RUN ATAC PIPELINE ===
    if run_atac_pipeline:
        print("\n" + "=" * 60)
        print("RUNNING ATAC PIPELINE")
        print("=" * 60)
        
        atac_output_dir = atac_output_dir or os.path.join(output_dir, 'atac')
        
        if atac_count_data_path is None:
            raise ValueError("ATAC pipeline requires atac_count_data_path")
        
        try:
            atac_results = atac_wrapper(
                # Required
                atac_count_data_path=atac_count_data_path,
                atac_output_dir=atac_output_dir,
                atac_sample_col=atac_sample_col,
                
                # Process Control
                atac_preprocessing=atac_preprocessing,
                atac_cell_type_cluster=atac_cell_type_cluster,
                atac_pseudobulk_dimensionality_reduction=atac_pseudobulk_dimensionality_reduction,
                trajectory_analysis_atac=atac_trajectory_analysis,
                trajectory_DGE=atac_trajectory_dge,
                sample_distance_calculation=atac_sample_distance_calculation,
                atac_sample_cluster=atac_sample_cluster,
                cluster_DGE=atac_cluster_dge,
                atac_visualization_processing=atac_visualization_processing,
                
                # Paths for Skipping
                atac_cell_path=atac_cell_path,
                atac_sample_path=atac_sample_path,
                atac_pseudobulk_adata_path=atac_pseudobulk_adata_path,
                
                # Basic Parameters
                atac_batch_col=atac_batch_col,
                atac_cell_type_column=atac_cell_type_column,
                atac_metadata_path=atac_metadata_path,
                grouping_columns=atac_grouping_columns,
                atac_pipeline_verbose=atac_pipeline_verbose,
                use_gpu=gpu_available,
                verbose=verbose,
                
                # Processing
                use_snapatac2_dimred=use_snapatac2_dimred,
                atac_min_cells=atac_min_cells,
                atac_min_genes=atac_min_genes,
                atac_max_genes=atac_max_genes,
                atac_min_cells_per_sample=atac_min_cells_per_sample,
                atac_doublet=atac_doublet,
                atac_tfidf_scale_factor=atac_tfidf_scale_factor,
                atac_num_features=atac_num_features,
                atac_n_lsi_components=atac_n_lsi_components,
                atac_drop_first_lsi=atac_drop_first_lsi,
                atac_harmony_max_iter=atac_harmony_max_iter,
                atac_n_neighbors=atac_n_neighbors,
                atac_n_pcs=atac_n_pcs,
                atac_umap_min_dist=atac_umap_min_dist,
                atac_umap_spread=atac_umap_spread,
                atac_umap_random_state=atac_umap_random_state,
                atac_plot_dpi=atac_plot_dpi,
                
                # Clustering
                atac_leiden_resolution=atac_leiden_resolution,
                atac_existing_cell_types=atac_existing_cell_types,
                atac_n_target_cell_clusters=atac_n_target_cell_clusters,
                preserve_cols_in_sample_embedding=atac_preserve_cols_in_sample_embedding,
                
                # Pseudobulk
                atac_pseudobulk_output_dir=atac_pseudobulk_output_dir,
                atac_pseudobulk_n_features=atac_pseudobulk_n_features,
                atac_pseudobulk_verbose=atac_pseudobulk_verbose,
                
                # Sample Embedding (updated parameter names)
                atac_dr_output_dir=atac_dr_output_dir,
                atac_sample_embedding_dimension=atac_sample_embedding_dimension,
                atac_harmony_for_proportion=atac_harmony_for_proportion,
                atac_dr_verbose=atac_dr_verbose,
                
                # Trajectory (updated parameter names)
                trajectory_supervised_atac=atac_trajectory_supervised,
                n_cca_pcs_atac=atac_n_cca_pcs,
                atac_cca_output_dir=atac_cca_output_dir,
                trajectory_col=atac_trajectory_col,
                trajectory_verbose=atac_trajectory_verbose,
                cca_pvalue=atac_cca_pvalue,
                atac_cca_optimal_cell_resolution=atac_cca_optimal_cell_resolution,
                TSCAN_origin=atac_tscan_origin,
                trajectory_visualization_label=atac_trajectory_visualization_label,
                
                # Trajectory DGE
                fdr_threshold=atac_fdr_threshold,
                effect_size_threshold=atac_effect_size_threshold,
                top_n_genes=atac_top_n_genes,
                trajectory_diff_gene_covariate=atac_trajectory_diff_gene_covariate,
                num_splines=atac_num_splines,
                spline_order=atac_spline_order,
                atac_trajectory_diff_gene_output_dir=atac_trajectory_diff_gene_output_dir,
                visualization_gene_list=atac_visualization_gene_list,
                visualize_all_deg=atac_visualize_all_deg,
                top_n_heatmap=atac_top_n_heatmap,
                trajectory_diff_gene_verbose=atac_trajectory_diff_gene_verbose,
                top_gene_number=atac_top_gene_number,
                
                # Distance
                sample_distance_methods=atac_sample_distance_methods,
                summary_sample_csv_path=atac_summary_sample_csv_path,
                
                # Cluster Analysis
                Kmeans_based_cluster_flag=atac_kmeans_based_cluster_flag,
                Tree_building_method=atac_tree_building_methods,
                proportion_test=atac_proportion_test,
                RAISIN_analysis=atac_raisin_analysis,
                cluster_distance_method=atac_cluster_distance_method,
                cluster_number=atac_cluster_number,
                user_provided_sample_to_clade=atac_user_provided_sample_to_clade,
                cluster_differential_gene_group_col=atac_cluster_differential_gene_group_col,
                
                # Visualization
                atac_figsize=atac_figsize,
                atac_point_size=atac_point_size,
                atac_visualization_grouping_columns=atac_visualization_grouping_columns,
                atac_show_sample_names=atac_show_sample_names,
                atac_visualization_age_size=atac_visualization_age_size,
                age_bin_size=atac_age_bin_size,
                verbose_Visualization=atac_verbose_visualization,
                dot_size=atac_dot_size,
                plot_dendrogram_flag=atac_plot_dendrogram_flag,
                plot_cell_umap_by_plot_group_flag=atac_plot_cell_umap_by_plot_group_flag,
                plot_umap_by_cell_type_flag=atac_plot_umap_by_cell_type_flag,
                plot_pca_2d_flag=atac_plot_pca_2d_flag,
                plot_pca_3d_flag=atac_plot_pca_3d_flag,
                plot_3d_cells_flag=atac_plot_3d_cells_flag,
                plot_cell_type_proportions_pca_flag=atac_plot_cell_type_proportions_pca_flag,
                plot_cell_type_expression_pca_flag=atac_plot_cell_type_expression_pca_flag,
                plot_pseudobulk_batch_test_expression_flag=atac_plot_pseudobulk_batch_test_expression_flag,
                plot_pseudobulk_batch_test_proportion_flag=atac_plot_pseudobulk_batch_test_proportion_flag,
                
                # System
                sample_col='sample',
                status_flags=status_flags,
            )
            
            results['atac_results'] = atac_results
            
            if isinstance(atac_results, dict) and 'status_flags' in atac_results:
                status_flags = atac_results['status_flags']
            elif isinstance(atac_results, tuple) and len(atac_results) > 3:
                results['atac_results'] = {
                    'atac_sample': atac_results[0],
                    'atac_cell': atac_results[1],
                    'pseudobulk_anndata': atac_results[2]
                }
            
            _save_status(status_file_path, status_flags)
            print("\nATAC pipeline completed successfully!")
            
        except Exception as e:
            print(f"\nATAC pipeline failed: {e}")
            results['atac_error'] = str(e)
            if verbose:
                import traceback
                traceback.print_exc()
    
    # === RUN MULTIOMICS PIPELINE ===
    if run_multiomics_pipeline:
        print("\n" + "=" * 60)
        print("RUNNING MULTIOMICS PIPELINE")
        print("=" * 60)
        
        multiomics_output_dir = multiomics_output_dir or os.path.join(output_dir, 'multiomics')
        
        if multiomics_run_glue and (multiomics_rna_file is None or multiomics_atac_file is None):
            raise ValueError("Multiomics pipeline with GLUE requires rna_file and atac_file")
        
        try:
            from .multiomics_wrapper import multiomics_wrapper
            
            multiomics_results = multiomics_wrapper(
                # Required
                rna_file=multiomics_rna_file,
                atac_file=multiomics_atac_file,
                multiomics_output_dir=multiomics_output_dir,
                
                # Process Control
                run_glue=multiomics_run_glue,
                run_integrate_preprocess=multiomics_run_integrate_preprocess,
                run_dimensionality_reduction=multiomics_run_dimensionality_reduction,
                run_visualize_embedding=multiomics_run_visualize_embedding,
                run_find_optimal_resolution=multiomics_run_find_optimal_resolution,
                
                # Basic Parameters
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
                use_gpu=multiomics_use_gpu,
                random_state=multiomics_random_state,
                
                # GLUE Sub-Steps
                run_glue_preprocessing=multiomics_run_glue_preprocessing,
                run_glue_training=multiomics_run_glue_training,
                run_glue_gene_activity=multiomics_run_glue_gene_activity,
                run_glue_cell_types=multiomics_run_glue_cell_types,
                run_glue_visualization=multiomics_run_glue_visualization,
                
                # GLUE Preprocessing
                ensembl_release=multiomics_ensembl_release,
                species=multiomics_species,
                use_highly_variable=multiomics_use_highly_variable,
                n_top_genes=multiomics_n_top_genes,
                n_pca_comps=multiomics_n_pca_comps,
                n_lsi_comps=multiomics_n_lsi_comps,
                gtf_by=multiomics_gtf_by,
                flavor=multiomics_flavor,
                generate_umap=multiomics_generate_umap,
                compression=multiomics_compression,
                
                # GLUE Training
                consistency_threshold=multiomics_consistency_threshold,
                treat_sample_as_batch=multiomics_treat_sample_as_batch,
                save_prefix=multiomics_save_prefix,
                
                # GLUE Gene Activity
                k_neighbors=multiomics_k_neighbors,
                use_rep=multiomics_use_rep,
                metric=multiomics_metric,
                
                # GLUE Cell Types
                existing_cell_types=multiomics_existing_cell_types,
                n_target_clusters=multiomics_n_target_clusters,
                cluster_resolution=multiomics_cluster_resolution,
                use_rep_celltype=multiomics_use_rep_celltype,
                markers=multiomics_markers,
                generate_umap_celltype=multiomics_generate_umap_celltype,
                
                # GLUE Visualization
                plot_columns=multiomics_plot_columns,
                
                # Integration Preprocessing
                min_cells_sample=multiomics_min_cells_sample,
                min_cell_gene=multiomics_min_cell_gene,
                min_features=multiomics_min_features,
                pct_mito_cutoff=multiomics_pct_mito_cutoff,
                exclude_genes=multiomics_exclude_genes,
                doublet=multiomics_doublet,
                
                # Sample Embedding (updated parameter names)
                sample_hvg_number=multiomics_sample_hvg_number,
                preserve_cols_in_sample_embedding=multiomics_preserve_cols_in_sample_embedding,
                sample_embedding_dimension=multiomics_sample_embedding_dimension,
                multiomics_harmony_for_proportion=multiomics_harmony_for_proportion,
                
                # Visualization
                color_col=multiomics_color_col,
                visualization_grouping_column=multiomics_visualization_grouping_column,
                target_modality=multiomics_target_modality,
                expression_key=multiomics_expression_key,
                proportion_key=multiomics_proportion_key,
                figsize=multiomics_figsize,
                point_size=multiomics_point_size,
                alpha=multiomics_alpha,
                colormap=multiomics_colormap,
                show_sample_names=multiomics_show_sample_names,
                force_data_type=multiomics_force_data_type,
                
                # Optimal Resolution (updated parameter names)
                optimization_target=multiomics_optimization_target,
                trajectory_col=multiomics_trajectory_col,
                resolution_use_rep=multiomics_resolution_use_rep,
                num_PCs=multiomics_num_pcs,
                visualize_cell_types=multiomics_visualize_cell_types,
                
                # Paths for Skipping
                integrated_h5ad_path=multiomics_integrated_h5ad_path,
                pseudobulk_h5ad_path=multiomics_pseudobulk_h5ad_path,
                
                # System
                status_flags=status_flags,
            )
            
            results['multiomics_results'] = multiomics_results
            
            if 'status_flags' in multiomics_results:
                status_flags = multiomics_results['status_flags']
            
            _save_status(status_file_path, status_flags)
            print("\nMultiomics pipeline completed successfully!")
            
        except Exception as e:
            print(f"\nMultiomics pipeline failed: {e}")
            results['multiomics_error'] = str(e)
            if verbose:
                import traceback
                traceback.print_exc()
    
    # === SUMMARY ===
    _print_summary(status_flags, run_rna_pipeline, run_atac_pipeline, run_multiomics_pipeline, verbose)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Status file: {status_file_path}")
    
    _save_status(status_file_path, status_flags)
    
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    return results


def _save_status(status_file_path: str, status_flags: Dict) -> None:
    """Save status flags to file."""
    with open(status_file_path, 'w') as f:
        json.dump(status_flags, f, indent=2)


def _print_summary(
    status_flags: Dict,
    run_rna: bool,
    run_atac: bool,
    run_multiomics: bool,
    verbose: bool
) -> None:
    """Print pipeline execution summary."""
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    
    pipelines = [
        ('rna', run_rna),
        ('atac', run_atac),
        ('multiomics', run_multiomics)
    ]
    
    for pipeline_name, should_run in pipelines:
        if should_run:
            pipeline_status = status_flags.get(pipeline_name, {})
            is_complete = all(pipeline_status.values()) if pipeline_status else False
            status_str = 'COMPLETE' if is_complete else 'PARTIAL'
            print(f"{pipeline_name.upper()} Pipeline: {status_str}")
            
            if verbose:
                for step, completed in pipeline_status.items():
                    status_icon = '✓' if completed else '○'
                    print(f"  {status_icon} {step}")