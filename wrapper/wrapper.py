import os
import sys
import json
import time
import shutil
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .rna_wrapper import rna_wrapper
from .atac_wrapper import atac_wrapper


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
    
    # Input data paths (for resuming)
    rna_adata_cell_path: Optional[str] = None,
    rna_adata_sample_path: Optional[str] = None,
    rna_sample_meta_path: Optional[str] = None,
    rna_cell_meta_path: Optional[str] = None,
    
    # Common column names
    rna_sample_col: str = 'sample',
    rna_batch_col: Optional[str] = None,
    rna_celltype_col: str = 'cell_type',
    
    # Preprocessing parameters
    rna_min_cells: int = 500,
    rna_min_genes: int = 500,
    rna_pct_mito_cutoff: float = 20,
    rna_exclude_genes: Optional[List] = None,
    rna_num_cell_hvgs: int = 2000,
    rna_cell_embedding_num_pcs: int = 20,
    rna_num_harmony_iterations: int = 30,
    rna_vars_to_regress: Optional[List] = None,
    
    # Cell type clustering parameters
    rna_leiden_cluster_resolution: float = 0.8,
    rna_cell_embedding_column: Optional[str] = None,
    rna_existing_cell_types: bool = False,
    rna_n_target_cell_clusters: Optional[int] = None,
    rna_umap: bool = False,
    
    # Sample embedding parameters
    rna_pseudo_adata_path: Optional[str] = None,
    rna_sample_hvg_number: int = 2000,
    rna_sample_embedding_dimension: int = 10,
    rna_harmony_for_proportion: bool = True,
    rna_preserve_cols_in_sample_embedding: Optional[List[str]] = None,
    
    # Trajectory analysis parameters
    rna_n_cca_pcs: int = 2,
    rna_trajectory_col: str = "sev.level",
    rna_trajectory_supervised: bool = False,
    rna_trajectory_visualization_label: Optional[List[str]] = None,
    rna_cca_pvalue: bool = False,
    rna_tscan_origin: Optional[int] = None,
    
    # CCA-based resolution selection parameters
    rna_cca_compute_corrected_pvalues: bool = True,
    rna_cca_coarse_start: float = 0.1,
    rna_cca_coarse_end: float = 1.0,
    rna_cca_coarse_step: float = 0.1,
    rna_cca_fine_range: float = 0.02,
    rna_cca_fine_step: float = 0.01,
    
    # Sample distance parameters
    rna_sample_distance_methods: Optional[List[str]] = None,
    rna_grouping_columns: Optional[List[str]] = None,
    rna_summary_sample_csv_path: Optional[str] = None,
    
    # Trajectory differential gene analysis parameters
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
    
    # Input data paths (for resuming)
    atac_adata_cell_path: Optional[str] = None,
    atac_adata_sample_path: Optional[str] = None,
    atac_sample_meta_path: Optional[str] = None,
    atac_cell_meta_path: Optional[str] = None,
    atac_pseudo_adata_path: Optional[str] = None,
    
    # Common column names
    atac_sample_col: str = "sample",
    atac_batch_col: Optional[str] = None,
    atac_celltype_col: str = "cell_type",
    atac_cell_embedding_column: Optional[str] = None,
    
    # ATAC-specific preprocessing parameters
    atac_min_cells: int = 1,
    atac_min_features: int = 2000,
    atac_max_features: int = 15000,
    atac_min_cells_per_sample: int = 1,
    atac_exclude_features: Optional[List] = None,
    atac_vars_to_regress: Optional[List] = None,
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
    
    # Trajectory analysis parameters
    atac_n_cca_pcs: int = 2,
    atac_trajectory_col: str = "sev.level",
    atac_trajectory_supervised: bool = True,
    atac_trajectory_visualization_label: Optional[List[str]] = None,
    atac_cca_pvalue: bool = False,
    atac_tscan_origin: Optional[str] = None,
    
    # CCA-based resolution selection parameters
    atac_cca_compute_corrected_pvalues: bool = True,
    atac_cca_coarse_start: float = 0.1,
    atac_cca_coarse_end: float = 1.0,
    atac_cca_coarse_step: float = 0.1,
    atac_cca_fine_range: float = 0.02,
    atac_cca_fine_step: float = 0.01,
    
    # Sample distance parameters
    atac_sample_distance_methods: Optional[List[str]] = None,
    atac_grouping_columns: Optional[List[str]] = None,
    atac_summary_sample_csv_path: Optional[str] = None,
    
    # Trajectory differential gene analysis parameters
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
    
    # ==========================================================================
    # MULTIOMICS PIPELINE PARAMETERS
    # ==========================================================================
    multiomics_rna_file: Optional[str] = None,
    multiomics_atac_file: Optional[str] = None,
    multiomics_output_dir: Optional[str] = None,
    
    # Pipeline control flags
    multiomics_run_glue: bool = True,
    multiomics_run_integrate_preprocess: bool = True,
    multiomics_run_dimensionality_reduction: bool = True,
    multiomics_run_visualize_embedding: bool = True,
    multiomics_run_find_optimal_resolution: bool = False,
    
    # GLUE sub-pipeline flags
    multiomics_run_glue_preprocessing: bool = True,
    multiomics_run_glue_training: bool = True,
    multiomics_run_glue_gene_activity: bool = True,
    multiomics_run_glue_cell_types: bool = True,
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
    
    # Optimal resolution parameters
    multiomics_optimization_target: str = "rna",
    multiomics_sev_col: str = "sev.level",
    multiomics_resolution_use_rep: str = 'X_glue',
    multiomics_num_pcs: int = 20,
    multiomics_visualize_cell_types: bool = True,
    
) -> Dict[str, Any]:
    """
    Main wrapper function that orchestrates RNA, ATAC, and Multiomics pipelines.
    
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
    
    # Import multiomics wrapper if needed
    if run_multiomics_pipeline and gpu_available:
        from .multiomics_wrapper import multiomics_wrapper
    
    # Initialize GPU if available
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
            "dimensionality_reduction": False,
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
    
    # Set default values for list parameters
    rna_grouping_columns = rna_grouping_columns or ['sev.level']
    rna_trajectory_visualization_label = rna_trajectory_visualization_label or ['sev.level']
    rna_sample_distance_methods = rna_sample_distance_methods or ['cosine', 'correlation']
    
    atac_grouping_columns = atac_grouping_columns or ['sev.level']
    atac_sample_distance_methods = atac_sample_distance_methods or ['cosine', 'correlation']
    atac_trajectory_visualization_label = atac_trajectory_visualization_label or ['sev.level']

    # ==================== RNA PIPELINE ====================
    if run_rna_pipeline:
        print("\n" + "=" * 60)
        print("RUNNING RNA PIPELINE")
        print("=" * 60)
        
        rna_output_dir = rna_output_dir or os.path.join(output_dir, 'rna')
        
        if rna_count_data_path is None:
            raise ValueError("RNA pipeline requires rna_count_data_path")
        
        try:
            rna_results = rna_wrapper(
                rna_count_data_path=rna_count_data_path,
                rna_output_dir=rna_output_dir,
                # Pipeline control flags
                preprocessing=rna_preprocessing,
                cell_type_cluster=rna_cell_type_cluster,
                derive_sample_embedding=rna_derive_sample_embedding,
                cca_based_cell_resolution_selection=rna_cca_based_cell_resolution_selection,
                sample_distance_calculation=rna_sample_distance_calculation,
                trajectory_analysis=rna_trajectory_analysis,
                trajectory_DGE=rna_trajectory_dge,
                sample_cluster=rna_sample_cluster,
                proportion_test=rna_proportion_test,
                cluster_DGE=rna_cluster_dge,
                visualize_data=rna_visualize_data,
                # General settings
                use_gpu=gpu_available,
                verbose=verbose,
                status_flags=status_flags,
                # Input data paths
                adata_cell_path=rna_adata_cell_path,
                adata_sample_path=rna_adata_sample_path,
                rna_sample_meta_path=rna_sample_meta_path,
                cell_meta_path=rna_cell_meta_path,
                # Common column names
                sample_col=rna_sample_col,
                batch_col=rna_batch_col,
                celltype_col=rna_celltype_col,
                # Preprocessing parameters
                min_cells=rna_min_cells,
                min_genes=rna_min_genes,
                pct_mito_cutoff=rna_pct_mito_cutoff,
                exclude_genes=rna_exclude_genes,
                num_cell_hvgs=rna_num_cell_hvgs,
                cell_embedding_num_pcs=rna_cell_embedding_num_pcs,
                num_harmony_iterations=rna_num_harmony_iterations,
                vars_to_regress=rna_vars_to_regress,
                # Cell type clustering parameters
                leiden_cluster_resolution=rna_leiden_cluster_resolution,
                cell_embedding_column=rna_cell_embedding_column,
                existing_cell_types=rna_existing_cell_types,
                n_target_cell_clusters=rna_n_target_cell_clusters,
                umap=rna_umap,
                # Sample embedding parameters
                pseudo_adata_path=rna_pseudo_adata_path,
                sample_hvg_number=rna_sample_hvg_number,
                sample_embedding_dimension=rna_sample_embedding_dimension,
                harmony_for_proportion=rna_harmony_for_proportion,
                preserve_cols_in_sample_embedding=rna_preserve_cols_in_sample_embedding,
                # Trajectory analysis parameters
                n_cca_pcs=rna_n_cca_pcs,
                trajectory_col=rna_trajectory_col,
                trajectory_supervised=rna_trajectory_supervised,
                trajectory_visualization_label=rna_trajectory_visualization_label,
                cca_pvalue=rna_cca_pvalue,
                tscan_origin=rna_tscan_origin,
                # CCA-based resolution selection parameters
                cca_compute_corrected_pvalues=rna_cca_compute_corrected_pvalues,
                cca_coarse_start=rna_cca_coarse_start,
                cca_coarse_end=rna_cca_coarse_end,
                cca_coarse_step=rna_cca_coarse_step,
                cca_fine_range=rna_cca_fine_range,
                cca_fine_step=rna_cca_fine_step,
                # Sample distance parameters
                sample_distance_methods=rna_sample_distance_methods,
                grouping_columns=rna_grouping_columns,
                summary_sample_csv_path=rna_summary_sample_csv_path,
                # Trajectory differential gene analysis parameters
                fdr_threshold=rna_fdr_threshold,
                effect_size_threshold=rna_effect_size_threshold,
                top_n_genes=rna_top_n_genes,
                trajectory_diff_gene_covariate=rna_trajectory_diff_gene_covariate,
                num_splines=rna_num_splines,
                spline_order=rna_spline_order,
                visualization_gene_list=rna_visualization_gene_list,
                # Sample clustering parameters
                cluster_number=rna_cluster_number,
                cluster_differential_gene_group_col=rna_cluster_differential_gene_group_col,
                # Visualization parameters
                age_bin_size=rna_age_bin_size,
                age_column=rna_age_column,
                plot_dendrogram_flag=rna_plot_dendrogram_flag,
                plot_cell_type_proportions_pca_flag=rna_plot_cell_type_proportions_pca_flag,
                plot_cell_type_expression_umap_flag=rna_plot_cell_type_expression_umap_flag,
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

    # ==================== ATAC PIPELINE ====================
    if run_atac_pipeline:
        print("\n" + "=" * 60)
        print("RUNNING ATAC PIPELINE")
        print("=" * 60)
        
        atac_output_dir = atac_output_dir or os.path.join(output_dir, 'atac')
        
        if atac_count_data_path is None:
            raise ValueError("ATAC pipeline requires atac_count_data_path")
        
        try:
            atac_results = atac_wrapper(
                atac_count_data_path=atac_count_data_path,
                atac_output_dir=atac_output_dir,
                # Pipeline control flags
                preprocessing=atac_preprocessing,
                cell_type_cluster=atac_cell_type_cluster,
                derive_sample_embedding=atac_derive_sample_embedding,
                cca_based_cell_resolution_selection=atac_cca_based_cell_resolution_selection,
                sample_distance_calculation=atac_sample_distance_calculation,
                trajectory_analysis=atac_trajectory_analysis,
                trajectory_DGE=atac_trajectory_dge,
                sample_cluster=atac_sample_cluster,
                proportion_test=atac_proportion_test,
                cluster_DGE=atac_cluster_dge,
                visualize_data=atac_visualize_data,
                # General settings
                use_gpu=gpu_available,
                verbose=verbose,
                status_flags=status_flags,
                # Input data paths
                adata_cell_path=atac_adata_cell_path,
                adata_sample_path=atac_adata_sample_path,
                atac_sample_meta_path=atac_sample_meta_path,
                cell_meta_path=atac_cell_meta_path,
                pseudo_adata_path=atac_pseudo_adata_path,
                # Common column names
                sample_col=atac_sample_col,
                batch_col=atac_batch_col,
                celltype_col=atac_celltype_col,
                cell_embedding_column=atac_cell_embedding_column,
                # ATAC-specific preprocessing parameters
                min_cells=atac_min_cells,
                min_features=atac_min_features,
                max_features=atac_max_features,
                min_cells_per_sample=atac_min_cells_per_sample,
                exclude_features=atac_exclude_features,
                vars_to_regress=atac_vars_to_regress,
                doublet_detection=atac_doublet_detection,
                num_cell_hvfs=atac_num_cell_hvfs,
                cell_embedding_num_pcs=atac_cell_embedding_num_pcs,
                num_harmony_iterations=atac_num_harmony_iterations,
                tfidf_scale_factor=atac_tfidf_scale_factor,
                log_transform=atac_log_transform,
                drop_first_lsi=atac_drop_first_lsi,
                # Cell type clustering parameters
                leiden_cluster_resolution=atac_leiden_cluster_resolution,
                existing_cell_types=atac_existing_cell_types,
                n_target_cell_clusters=atac_n_target_cell_clusters,
                umap=atac_umap,
                # Sample embedding parameters
                sample_hvg_number=atac_sample_hvg_number,
                sample_embedding_dimension=atac_sample_embedding_dimension,
                harmony_for_proportion=atac_harmony_for_proportion,
                preserve_cols_in_sample_embedding=atac_preserve_cols_in_sample_embedding,
                # Trajectory analysis parameters
                n_cca_pcs=atac_n_cca_pcs,
                trajectory_col=atac_trajectory_col,
                trajectory_supervised=atac_trajectory_supervised,
                trajectory_visualization_label=atac_trajectory_visualization_label,
                cca_pvalue=atac_cca_pvalue,
                tscan_origin=atac_tscan_origin,
                # CCA-based resolution selection parameters
                cca_compute_corrected_pvalues=atac_cca_compute_corrected_pvalues,
                cca_coarse_start=atac_cca_coarse_start,
                cca_coarse_end=atac_cca_coarse_end,
                cca_coarse_step=atac_cca_coarse_step,
                cca_fine_range=atac_cca_fine_range,
                cca_fine_step=atac_cca_fine_step,
                # Sample distance parameters
                sample_distance_methods=atac_sample_distance_methods,
                grouping_columns=atac_grouping_columns,
                summary_sample_csv_path=atac_summary_sample_csv_path,
                # Trajectory differential gene analysis parameters
                fdr_threshold=atac_fdr_threshold,
                effect_size_threshold=atac_effect_size_threshold,
                top_n_genes=atac_top_n_genes,
                trajectory_diff_gene_covariate=atac_trajectory_diff_gene_covariate,
                num_splines=atac_num_splines,
                spline_order=atac_spline_order,
                visualization_gene_list=atac_visualization_gene_list,
                # Sample clustering parameters
                cluster_number=atac_cluster_number,
                cluster_differential_gene_group_col=atac_cluster_differential_gene_group_col,
                # Visualization parameters
                age_bin_size=atac_age_bin_size,
                age_column=atac_age_column,
                plot_dendrogram_flag=atac_plot_dendrogram_flag,
                plot_cell_type_proportions_pca_flag=atac_plot_cell_type_proportions_pca_flag,
                plot_cell_type_expression_umap_flag=atac_plot_cell_type_expression_umap_flag,
            )
            
            results['atac_results'] = atac_results
            if isinstance(atac_results, dict) and 'status_flags' in atac_results:
                status_flags = atac_results['status_flags']
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
        
        if multiomics_run_glue and (multiomics_rna_file is None or multiomics_atac_file is None):
            raise ValueError("Multiomics pipeline with GLUE requires multiomics_rna_file and multiomics_atac_file")
        
        try:
            from .multiomics_wrapper import multiomics_wrapper
            
            multiomics_results = multiomics_wrapper(
                # ===== Required Parameters =====
                rna_file=multiomics_rna_file,
                atac_file=multiomics_atac_file,
                multiomics_output_dir=multiomics_output_dir,
                
                # ===== Process Control Flags =====
                run_glue=multiomics_run_glue,
                run_integrate_preprocess=multiomics_run_integrate_preprocess,
                run_dimensionality_reduction=multiomics_run_dimensionality_reduction,
                run_visualize_embedding=multiomics_run_visualize_embedding,
                run_find_optimal_resolution=multiomics_run_find_optimal_resolution,
                
                # ===== Basic Parameters =====
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
                
                # ===== GLUE Integration Parameters =====
                run_glue_preprocessing=multiomics_run_glue_preprocessing,
                run_glue_training=multiomics_run_glue_training,
                run_glue_gene_activity=multiomics_run_glue_gene_activity,
                run_glue_cell_types=multiomics_run_glue_cell_types,
                run_glue_visualization=multiomics_run_glue_visualization,
                
                # GLUE preprocessing parameters
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
                
                # GLUE training parameters
                consistency_threshold=multiomics_consistency_threshold,
                treat_sample_as_batch=multiomics_treat_sample_as_batch,
                save_prefix=multiomics_save_prefix,
                
                # GLUE gene activity parameters
                k_neighbors=multiomics_k_neighbors,
                use_rep=multiomics_use_rep,
                metric=multiomics_metric,
                
                # GLUE cell type parameters
                existing_cell_types=multiomics_existing_cell_types,
                n_target_clusters=multiomics_n_target_clusters,
                cluster_resolution=multiomics_cluster_resolution,
                use_rep_celltype=multiomics_use_rep_celltype,
                markers=multiomics_markers,
                generate_umap_celltype=multiomics_generate_umap_celltype,
                
                # GLUE visualization parameters
                plot_columns=multiomics_plot_columns,
                
                # ===== Integration Preprocessing Parameters =====
                min_cells_sample=multiomics_min_cells_sample,
                min_cell_gene=multiomics_min_cell_gene,
                min_features=multiomics_min_features,
                pct_mito_cutoff=multiomics_pct_mito_cutoff,
                exclude_genes=multiomics_exclude_genes,
                doublet=multiomics_doublet,
                
                # ===== Dimensionality Reduction Parameters =====
                sample_hvg_number=multiomics_sample_hvg_number,
                preserve_cols_for_sample_embedding=multiomics_preserve_cols_for_sample_embedding,
                n_expression_components=multiomics_n_expression_components,
                n_proportion_components=multiomics_n_proportion_components,
                multiomics_harmony_for_proportion=multiomics_harmony_for_proportion,
                
                # ===== Visualization Parameters =====
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
                
                # ===== Optimal Resolution Parameters =====
                optimization_target=multiomics_optimization_target,
                sev_col=multiomics_sev_col,
                resolution_use_rep=multiomics_resolution_use_rep,
                num_PCs=multiomics_num_pcs,
                visualize_cell_types=multiomics_visualize_cell_types,
                
                # ===== Paths for Skipping Steps =====
                integrated_h5ad_path=multiomics_integrated_h5ad_path,
                pseudobulk_h5ad_path=multiomics_pseudobulk_h5ad_path,
                
                # ===== System Parameters =====
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
    
    # Final summary
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    return results


def _save_status(status_file_path: str, status_flags: Dict) -> None:
    """Save status flags to JSON file."""
    with open(status_file_path, 'w') as f:
        json.dump(status_flags, f, indent=2)