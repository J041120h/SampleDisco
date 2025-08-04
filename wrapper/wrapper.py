import os
import json
import sys
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Import individual wrappers
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
    # Overall pipeline control
    run_rna_pipeline: bool = True,
    run_atac_pipeline: bool = False,
    run_multiomics_pipeline: bool = False,
    
    # System configuration
    use_gpu: bool = False,
    initialization: bool = True,
    verbose: bool = True,
    save_intermediate: bool = True,
    
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
    rna_dimensionality_reduction: bool = True,
    rna_sample_distance_calculation: bool = True,
    rna_trajectory_analysis: bool = True,
    rna_trajectory_dge: bool = True,
    rna_sample_cluster: bool = True,
    rna_cluster_dge: bool = True,
    rna_visualize_data: bool = True,
    
    # RNA Basic Parameters
    rna_sample_col: str = 'sample',
    rna_grouping_columns: List[str] = None,
    rna_cell_type_column: str = 'cell_type',
    rna_cell_meta_path: Optional[str] = None,
    rna_batch_col: List = None,
    rna_markers: Optional[List] = None,
    rna_cluster_resolution: float = 0.8,
    rna_num_pcs: int = 20,
    rna_num_harmony: int = 30,
    rna_num_features: int = 2000,
    rna_min_cells: int = 500,
    rna_min_features: int = 500,
    rna_pct_mito_cutoff: int = 20,
    rna_exclude_genes: Optional[List] = None,
    rna_doublet: bool = True,
    rna_combat: bool = True,
    rna_method: str = 'average',
    rna_metric: str = 'euclidean',
    rna_distance_mode: str = 'centroid',
    rna_vars_to_regress: Optional[List] = None,
    
    # RNA Cell Type Clustering Parameters
    rna_existing_cell_types: bool = False,
    rna_n_target_cell_clusters: Optional[int] = None,
    rna_umap: bool = False,
    rna_cell_type_save: bool = True,
    rna_assign_save: bool = True,
    
    # RNA Cell Type Annotation Parameters
    rna_cell_type_annotation: bool = False,
    rna_cell_type_annotation_model_name: Optional[str] = None,
    rna_cell_type_annotation_custom_model_path: Optional[str] = None,
    
    # RNA Pseudobulk Parameters
    rna_celltype_col: str = 'cell_type',
    rna_pseudobulk_output_dir: Optional[str] = None,
    rna_pseudobulk_n_features: int = 2000,
    rna_frac: float = 0.3,
    rna_pseudobulk_verbose: bool = True,
    
    # RNA PCA Parameters
    rna_n_expression_components: int = 10,
    rna_n_proportion_components: int = 10,
    rna_dr_output_dir: Optional[str] = None,
    rna_anndata_sample_path: Optional[str] = None,
    rna_dr_verbose: bool = True,
    
    # RNA Trajectory Analysis Parameters
    rna_trajectory_supervised: bool = False,
    n_components_for_cca_rna: int = 2,
    rna_cca_output_dir: Optional[str] = None,
    rna_sev_col_cca: str = "sev.level",
    rna_cca_optimal_cell_resolution: bool = False,
    rna_cca_pvalue: bool = False,
    rna_trajectory_verbose: bool = True,
    rna_tscan_origin: Optional[int] = None,
    
    # RNA Trajectory Differential Gene Parameters
    rna_fdr_threshold: float = 0.05,
    rna_effect_size_threshold: float = 1,
    rna_top_n_genes: int = 100,
    rna_trajectory_diff_gene_covariate: Optional[List] = None,
    rna_num_splines: int = 5,
    rna_spline_order: int = 3,
    rna_trajectory_diff_gene_output_dir: Optional[str] = None,
    rna_visualization_gene_list: Optional[List] = None,
    rna_visualize_all_deg: bool = True,
    rna_top_n_heatmap: int = 50,
    rna_trajectory_diff_gene_verbose: bool = True,
    rna_top_gene_number: int = 30,
    rna_n_pcs_for_null: int = 10,
    
    # RNA Paths for Skipping Preprocessing
    rna_anndata_cell_path: Optional[str] = None,
    
    # RNA Distance Methods
    rna_summary_sample_csv_path: Optional[str] = None,
    rna_sample_distance_methods: Optional[List[str]] = None,
    
    # RNA Visualization Parameters
    rna_verbose_visualization: bool = True,
    rna_trajectory_visualization_label: List[str] = None,
    rna_age_bin_size: Optional[int] = None,
    rna_dot_size: int = 3,
    rna_plot_dendrogram_flag: bool = True,
    rna_plot_cell_umap_by_plot_group_flag: bool = True,
    rna_plot_umap_by_cell_type_flag: bool = True,
    rna_plot_pca_2d_flag: bool = True,
    rna_plot_pca_3d_flag: bool = True,
    rna_plot_3d_cells_flag: bool = True,
    rna_plot_cell_type_proportions_pca_flag: bool = True,
    rna_plot_cell_type_expression_pca_flag: bool = True,
    rna_plot_pseudobulk_batch_test_expression_flag: bool = False,
    rna_plot_pseudobulk_batch_test_proportion_flag: bool = False,
    
    # RNA Cluster Based DEG
    rna_kmeans_based_cluster_flag: bool = False,
    rna_tree_building_method: List[str] = None,
    rna_proportion_test: bool = False,
    rna_raisin_analysis: bool = False,
    rna_cluster_distance_method: str = 'cosine',
    rna_cluster_number: int = 4,
    rna_user_provided_sample_to_clade: Optional[Dict] = None,
    
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
    atac_visualize_data: bool = True,
    
    # ATAC Column specifications
    atac_sample_col: str = "sample",
    atac_batch_col: Optional[str] = None,
    atac_cell_type_column: str = "cell_type",
    
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
    
    # ATAC Cell Type Annotation Parameters
    atac_cell_type_annotation: bool = False,
    atac_cell_type_annotation_model_name: Optional[str] = None,
    atac_cell_type_annotation_custom_model_path: Optional[str] = None,
    
    # ATAC UMAP parameters
    atac_umap_min_dist: float = 0.3,
    atac_umap_spread: float = 1.0,
    atac_umap_random_state: int = 42,
    atac_plot_dpi: int = 300,
    
    # ATAC Paths for skipping preprocessing
    atac_cell_path: Optional[str] = None,
    atac_sample_path: Optional[str] = None,
    
    # ATAC Pseudobulk parameters
    atac_pseudobulk_output_dir: Optional[str] = None,
    atac_pseudobulk_n_features: int = 50000,
    atac_pseudobulk_verbose: bool = True,
    
    # ATAC PCA parameters
    atac_dr_output_dir: Optional[str] = None,
    atac_dr_n_expression_components: int = 30,
    atac_dr_n_proportion_components: int = 30,
    atac_dr_verbose: bool = True,
    
    # ATAC Trajectory analysis parameters
    atac_trajectory_supervised: bool = True,
    n_components_for_cca_atac: int = 2,
    atac_cca_output_dir: Optional[str] = None,
    atac_sev_col_cca: str = "sev.level",
    atac_trajectory_verbose: bool = True,
    atac_cca_pvalue: bool = False,
    atac_cca_optimal_cell_resolution: bool = False,
    atac_n_pcs_for_null: int = 10,
    atac_tscan_origin: Optional[int] = None,
    atac_trajectory_visualization_label: List[str] = None,
    
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
    atac_grouping_columns: List[str] = None,
    
    # ATAC Clustering parameters
    atac_kmeans_based_cluster_flag: bool = False,
    atac_tree_building_method: List[str] = None,
    atac_proportion_test: bool = False,
    atac_raisin_analysis: bool = False,
    atac_cluster_distance_method: str = 'cosine',
    atac_cluster_number: int = 4,
    atac_user_provided_sample_to_clade: Optional[Dict] = None,
    
    # ATAC Visualization parameters
    atac_figsize: Tuple[int, int] = (10, 8),
    atac_point_size: int = 50,
    atac_visualization_grouping_columns: List[str] = None,
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
    # Multiomics Control Flags
    multiomics_run_glue: bool = True,
    multiomics_run_integrate_preprocess: bool = True,
    multiomics_run_compute_pseudobulk: bool = True,
    multiomics_run_process_pca: bool = True,
    multiomics_run_visualize_embedding: bool = True,
    multiomics_run_find_optimal_resolution: bool = False,
    
    # Multiomics Data files
    multiomics_rna_file: Optional[str] = None,
    multiomics_atac_file: Optional[str] = None,
    multiomics_rna_sample_meta_file: Optional[str] = None,
    multiomics_atac_sample_meta_file: Optional[str] = None,
    multiomics_output_dir: Optional[str] = None,
    
    # Multiomics Preprocessing parameters
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
    multiomics_random_state: int = 42,
    multiomics_metadata_sep: str = ",",
    multiomics_rna_sample_column: str = "sample",
    multiomics_atac_sample_column: str = "sample",
    
    # Multiomics Training parameters
    multiomics_consistency_threshold: float = 0.05,
    multiomics_save_prefix: str = "glue",
    
    # Multiomics Gene activity computation parameters
    multiomics_k_neighbors: int = 10,
    multiomics_use_rep: str = "X_glue",
    multiomics_metric: str = "cosine",
    multiomics_existing_cell_types: bool = False,
    multiomics_n_target_clusters: int = 10,
    multiomics_cluster_resolution: float = 0.8,
    multiomics_use_rep_celltype: str = "X_glue",
    multiomics_markers: Optional[List] = None,
    multiomics_method: str = 'average',
    multiomics_metric_celltype: str = 'euclidean',
    multiomics_distance_mode: str = 'centroid',
    multiomics_generate_umap_celltype: bool = True,
    multiomics_plot_columns: Optional[List[str]] = None,
    
    # Multiomics Integration preprocessing parameters
    multiomics_integrate_output_dir: Optional[str] = None,
    multiomics_h5ad_path: Optional[str] = None,
    multiomics_sample_column: str = 'sample',
    multiomics_min_cells_sample: int = 1,
    multiomics_min_cell_gene: int = 10,
    multiomics_min_features: int = 500,
    multiomics_pct_mito_cutoff: int = 20,
    multiomics_exclude_genes: Optional[List] = None,
    multiomics_doublet: bool = True,
    
    # Multiomics Pseudobulk parameters
    multiomics_batch_col: str = 'batch',
    multiomics_sample_col: str = 'sample',
    multiomics_celltype_col: str = 'cell_type',
    multiomics_pseudobulk_output_dir: Optional[str] = None,
    multiomics_save: bool = True,
    multiomics_n_features: int = 2000,
    multiomics_normalize: bool = True,
    multiomics_target_sum: float = 1e4,
    multiomics_atac: bool = False,
    
    # Multiomics PCA parameters
    multiomics_pca_sample_col: str = 'sample',
    multiomics_n_expression_pcs: int = 10,
    multiomics_n_proportion_pcs: int = 10,
    multiomics_pca_output_dir: Optional[str] = None,
    multiomics_integrated_data: bool = False,
    multiomics_not_save: bool = False,
    multiomics_pca_atac: bool = False,
    multiomics_use_snapatac2_dimred: bool = False,
    
    # Multiomics Visualization parameters
    multiomics_modality_col: str = 'modality',
    multiomics_color_col: str = 'color',
    multiomics_target_modality: str = 'ATAC',
    multiomics_expression_key: str = 'X_DR_expression',
    multiomics_proportion_key: str = 'X_DR_proportion',
    multiomics_figsize: Tuple[int, int] = (20, 8),
    multiomics_point_size: int = 60,
    multiomics_alpha: float = 0.8,
    multiomics_colormap: str = 'viridis',
    multiomics_viz_output_dir: Optional[str] = None,
    multiomics_show_sample_names: bool = False,
    multiomics_force_data_type: Optional[str] = None,
    
    # Multiomics Optimal resolution parameters
    multiomics_optimization_target: str = "rna",
    multiomics_dr_type: str = "expression",
    multiomics_resolution_n_features: int = 40000,
    multiomics_sev_col: str = "sev.level",
    multiomics_resolution_batch_col: Optional[str] = None,
    multiomics_resolution_sample_col: str = "sample",
    multiomics_resolution_modality_col: str = "modality",
    multiomics_resolution_use_rep: str = 'X_glue',
    multiomics_num_dr_components: int = 30,
    multiomics_num_pcs: int = 20,
    multiomics_num_pvalue_simulations: int = 1000,
    multiomics_n_pcs: int = 2,
    multiomics_compute_pvalues: bool = True,
    multiomics_visualize_embeddings: bool = True,
    multiomics_resolution_output_dir: Optional[str] = None,
    
    # Multiomics Global parameters
    multiomics_verbose: bool = True,
    multiomics_integrated_h5ad_path: str = None,
    multiomics_pseudobulk_h5ad_path: str = None,  
) -> Dict[str, Any]:
    """
    Comprehensive main wrapper for single-cell multi-modal analysis pipelines.
    
    This wrapper integrates RNA-seq, ATAC-seq, and multiomics analysis pipelines
    with proper status tracking, Linux/GPU support, and flexible parameter control.
    
    Parameters
    ----------
    output_dir : str
        Main output directory for all results
    run_rna_pipeline : bool
        Whether to run RNA-seq analysis pipeline
    run_atac_pipeline : bool
        Whether to run ATAC-seq analysis pipeline
    run_multiomics_pipeline : bool
        Whether to run integrated multiomics pipeline
    use_gpu : bool
        Whether to use GPU acceleration (Linux only)
    initialization : bool
        Whether to initialize from scratch or resume from previous run
    
    [All other parameters are passed to respective pipeline wrappers]
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'status_flags': Overall pipeline status tracking
        - 'rna_results': Results from RNA pipeline (if run)
        - 'atac_results': Results from ATAC pipeline (if run)
        - 'multiomics_results': Results from multiomics pipeline (if run)
        - 'system_info': System configuration information
    """
    if run_multiomics_pipeline:
        from .multiomics_wrapper import multiomics_wrapper
    # Create main output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize system information
    system_info = {
        'platform': platform.system(),
        'python_version': sys.version,
        'use_gpu': use_gpu,
        'linux_system': False
    }
    
    # Check for Linux system and GPU setup
    linux_system = False
    if platform.system() == "Linux" and use_gpu:
        print("Linux system detected with GPU enabled.")
        linux_system = True
        system_info['linux_system'] = True
        
        if initialization:
            try:
                # Install dependencies for rapids-singlecell
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
                use_gpu = False
                system_info['use_gpu'] = False
    
    # Initialize or load status flags
    status_file_path = os.path.join(output_dir, "sys_log", "main_process_status.json")
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
        },
        "atac": {
            "preprocessing": False,
            "cell_type_cluster": False,
            "dimensionality_reduction": False,
            "sample_distance_calculation": False,
            "trajectory_analysis": False,
            "trajectory_dge": False,
            "cluster_dge": False,
            "visualization": False
        },
        "multiomics": {
            "glue_integration": False,
            "integration_preprocessing": False,
            "pseudobulk_computation": False,
            "pca_processing": False,
            "embedding_visualization": False,
            "optimal_resolution": False
        },
        "system_info": system_info
    }
    
    # Ensure sys_log directory exists
    os.makedirs(os.path.dirname(status_file_path), exist_ok=True)
    
    # Load existing status if not initializing
    if os.path.exists(status_file_path) and not initialization:
        try:
            with open(status_file_path, 'r') as f:
                saved_status = json.load(f)
                # Update only the pipeline-specific flags, keep system info
                for key in ['rna', 'atac', 'multiomics']:
                    if key in saved_status:
                        status_flags[key].update(saved_status[key])
                print("Resuming process from previous progress:")
                print(json.dumps(status_flags, indent=4))
        except Exception as e:
            print(f"Error reading status file: {e}")
            print("Reinitializing status from scratch.")
    else:
        if initialization:
            # Clean up existing results if initializing
            for subdir in ['rna', 'atac', 'multiomics']:
                check_dir = os.path.join(output_dir, subdir, "result")
                if os.path.exists(check_dir):
                    try:
                        shutil.rmtree(check_dir)
                        print(f"Removed existing result directory: {check_dir}")
                    except Exception as e:
                        print(f"Failed to remove existing result directory: {e}")
        
        print("Initializing main process status file.")
        with open(status_file_path, 'w') as f:
            json.dump(status_flags, f, indent=4)
    
    # Initialize results dictionary
    results = {
        'status_flags': status_flags,
        'system_info': system_info
    }

    # Set default values for list parameters
    if atac_grouping_columns is None:
        atac_grouping_columns = ['sev.level']
    if atac_visualization_grouping_columns is None:
        atac_visualization_grouping_columns = ['current_severity']
    if atac_tree_building_method is None:
        atac_tree_building_method = ['HRA_VEC', 'HRC_VEC', 'NN', 'UPGMA']
    if atac_sample_distance_methods is None:
        atac_sample_distance_methods = ['cosine', 'correlation']
    if rna_grouping_columns is None:
        rna_grouping_columns = ['sev.level']
    if rna_batch_col is None:
        rna_batch_col = []
    if rna_trajectory_visualization_label is None:
        rna_trajectory_visualization_label = ['sev.level']
    if rna_tree_building_method is None:
        rna_tree_building_method = ['HRA_VEC', 'HRC_VEC', 'NN', 'UPGMA']
    if rna_sample_distance_methods is None:
        rna_sample_distance_methods = ['cosine', 'correlation']
    if atac_trajectory_visualization_label is None:
        atac_trajectory_visualization_label = ['sev.level']

    # ========================================
    # RNA PIPELINE
    # ========================================
    if run_rna_pipeline:
        print("\n" + "="*60)
        print("RUNNING RNA PIPELINE")
        print("="*60)
        
        # Set default RNA output directory if not specified
        if rna_output_dir is None:
            rna_output_dir = os.path.join(output_dir, 'rna')
        
        # Validate required RNA parameters
        if rna_count_data_path is None or rna_sample_meta_path is None:
            raise ValueError("RNA pipeline requires rna_count_data_path and rna_sample_meta_path")
        
        try:
            rna_results = rna_wrapper(
                # Required parameters
                rna_count_data_path=rna_count_data_path,
                rna_sample_meta_path=rna_sample_meta_path,
                rna_output_dir=rna_output_dir,
                
                # Process control flags
                preprocessing=rna_preprocessing,
                cell_type_cluster=rna_cell_type_cluster,
                DimensionalityReduction=rna_dimensionality_reduction,
                sample_distance_calculation=rna_sample_distance_calculation,
                trajectory_analysis=rna_trajectory_analysis,
                trajectory_DGE=rna_trajectory_dge,
                sample_cluster=rna_sample_cluster,
                cluster_DGE=rna_cluster_dge,
                visualize_data=rna_visualize_data,
                
                # Basic parameters
                sample_col=rna_sample_col,
                grouping_columns=rna_grouping_columns,
                cell_type_column=rna_cell_type_column,
                cell_meta_path=rna_cell_meta_path,
                batch_col=rna_batch_col,
                markers=rna_markers,
                cluster_resolution=rna_cluster_resolution,
                num_PCs=rna_num_pcs,
                num_harmony=rna_num_harmony,
                num_features=rna_num_features,
                min_cells=rna_min_cells,
                min_features=rna_min_features,
                pct_mito_cutoff=rna_pct_mito_cutoff,
                exclude_genes=rna_exclude_genes,
                doublet=rna_doublet,
                combat=rna_combat,
                method=rna_method,
                metric=rna_metric,
                distance_mode=rna_distance_mode,
                vars_to_regress=rna_vars_to_regress,
                verbose=verbose,
                
                # Cell type clustering parameters
                existing_cell_types=rna_existing_cell_types,
                n_target_cell_clusters=rna_n_target_cell_clusters,
                umap=rna_umap,
                cell_type_save=rna_cell_type_save,
                assign_save=rna_assign_save,
                
                # Cell type annotation parameters
                cell_type_annotation=rna_cell_type_annotation,
                rna_cell_type_annotation_model_name=rna_cell_type_annotation_model_name,
                rna_cell_type_annotation_custom_model_path=rna_cell_type_annotation_custom_model_path,
                
                # Pseudobulk parameters
                celltype_col=rna_celltype_col,
                pseudobulk_output_dir=rna_pseudobulk_output_dir,
                n_features=rna_pseudobulk_n_features,
                frac=rna_frac,
                pseudobulk_verbose=rna_pseudobulk_verbose,
                
                # PCA parameters
                n_expression_components=rna_n_expression_components,
                n_proportion_components=rna_n_proportion_components,
                dr_output_dir=rna_dr_output_dir,
                AnnData_sample_path=rna_anndata_sample_path,
                dr_verbose=rna_dr_verbose,
                
                # Trajectory analysis parameters
                trajectory_supervised=rna_trajectory_supervised,
                n_components_for_cca_rna = n_components_for_cca_rna,
                cca_output_dir=rna_cca_output_dir,
                sev_col_cca=rna_sev_col_cca,
                cca_optimal_cell_resolution=rna_cca_optimal_cell_resolution,
                cca_pvalue=rna_cca_pvalue,
                trajectory_verbose=rna_trajectory_verbose,
                TSCAN_origin=rna_tscan_origin,
                
                # Trajectory differential gene parameters
                fdr_threshold=rna_fdr_threshold,
                effect_size_threshold=rna_effect_size_threshold,
                top_n_genes=rna_top_n_genes,
                trajectory_diff_gene_covariate=rna_trajectory_diff_gene_covariate,
                num_splines=rna_num_splines,
                spline_order=rna_spline_order,
                trajectory_diff_gene_output_dir=rna_trajectory_diff_gene_output_dir,
                visualization_gene_list=rna_visualization_gene_list,
                visualize_all_deg=rna_visualize_all_deg,
                top_n_heatmap=rna_top_n_heatmap,
                trajectory_diff_gene_verbose=rna_trajectory_diff_gene_verbose,
                top_gene_number=rna_top_gene_number,
                n_pcs_for_null=rna_n_pcs_for_null,
                
                # Paths for skipping preprocessing
                AnnData_cell_path=rna_anndata_cell_path,
                
                # Distance methods
                summary_sample_csv_path=rna_summary_sample_csv_path,
                sample_distance_methods=rna_sample_distance_methods,
                
                # Visualization parameters
                verbose_Visualization=rna_verbose_visualization,
                trajectory_visualization_label=rna_trajectory_visualization_label,
                age_bin_size=rna_age_bin_size,
                dot_size=rna_dot_size,
                plot_dendrogram_flag=rna_plot_dendrogram_flag,
                plot_cell_umap_by_plot_group_flag=rna_plot_cell_umap_by_plot_group_flag,
                plot_umap_by_cell_type_flag=rna_plot_umap_by_cell_type_flag,
                plot_pca_2d_flag=rna_plot_pca_2d_flag,
                plot_pca_3d_flag=rna_plot_pca_3d_flag,
                plot_3d_cells_flag=rna_plot_3d_cells_flag,
                plot_cell_type_proportions_pca_flag=rna_plot_cell_type_proportions_pca_flag,
                plot_cell_type_expression_pca_flag=rna_plot_cell_type_expression_pca_flag,
                plot_pseudobulk_batch_test_expression_flag=rna_plot_pseudobulk_batch_test_expression_flag,
                plot_pseudobulk_batch_test_proportion_flag=rna_plot_pseudobulk_batch_test_proportion_flag,
                
                # Cluster based DEG
                Kmeans_based_cluster_flag=rna_kmeans_based_cluster_flag,
                Tree_building_method=rna_tree_building_method,
                proportion_test=rna_proportion_test,
                RAISIN_analysis=rna_raisin_analysis,
                cluster_distance_method=rna_cluster_distance_method,
                cluster_number=rna_cluster_number,
                user_provided_sample_to_clade=rna_user_provided_sample_to_clade,
                
                # System parameters
                linux_system=linux_system,
                use_gpu=use_gpu,
                status_flags=status_flags
            )
            
            results['rna_results'] = rna_results
            
            # Update status flags from RNA results
            if 'status_flags' in rna_results:
                status_flags = rna_results['status_flags']
                
            # Save updated status
            with open(status_file_path, 'w') as f:
                json.dump(status_flags, f, indent=4)
            
            print("\n✓ RNA pipeline completed successfully!")
            
        except Exception as e:
            print(f"\n✗ RNA pipeline failed with error: {e}")
            results['rna_error'] = str(e)
            if verbose:
                import traceback
                traceback.print_exc()
    
    # ========================================
    # ATAC PIPELINE
    # ========================================
    if run_atac_pipeline:
        print("\n" + "="*60)
        print("RUNNING ATAC PIPELINE")
        print("="*60)
        
        # Set default ATAC output directory if not specified
        if atac_output_dir is None:
            atac_output_dir = os.path.join(output_dir, 'atac')
        
        # Validate required ATAC parameters
        if atac_count_data_path is None or atac_metadata_path is None:
            raise ValueError("ATAC pipeline requires atac_count_data_path and atac_metadata_path")
        
        try:
            atac_results = atac_wrapper(
                # File paths and directories
                atac_count_data_path=atac_count_data_path,
                atac_output_dir=atac_output_dir,
                atac_metadata_path=atac_metadata_path,
                atac_pseudobulk_output_dir=atac_pseudobulk_output_dir,
                atac_dr_output_dir=atac_dr_output_dir,
                atac_cca_output_dir=atac_cca_output_dir,
                
                # Column specifications
                atac_sample_col=atac_sample_col,
                atac_batch_col=atac_batch_col,
                atac_cell_type_column=atac_cell_type_column,
                
                # Pipeline configuration
                atac_pipeline_verbose=atac_pipeline_verbose,
                use_snapatac2_dimred=use_snapatac2_dimred,
                use_gpu=use_gpu,
                
                # Process control flags
                atac_preprocessing=atac_preprocessing,
                atac_cell_type_cluster=atac_cell_type_cluster,
                atac_pseudobulk_dimensionality_reduction=atac_pseudobulk_dimensionality_reduction,
                atac_visualization_processing=atac_visualization_processing,
                trajectory_analysis_atac=atac_trajectory_analysis,
                sample_distance_calculation=atac_sample_distance_calculation,
                atac_sample_cluster = atac_sample_cluster,
                cluster_DGE=atac_cluster_dge,
                trajectory_DGE=atac_trajectory_dge,
                visualize_data=atac_visualize_data,
                
                # QC and filtering parameters
                atac_min_cells=atac_min_cells,
                atac_min_genes=atac_min_genes,
                atac_max_genes=atac_max_genes,
                atac_min_cells_per_sample=atac_min_cells_per_sample,
                
                # Processing parameters
                atac_doublet=atac_doublet,
                atac_tfidf_scale_factor=atac_tfidf_scale_factor,
                atac_num_features=atac_num_features,
                atac_n_lsi_components=atac_n_lsi_components,
                atac_drop_first_lsi=atac_drop_first_lsi,
                atac_harmony_max_iter=atac_harmony_max_iter,
                atac_n_neighbors=atac_n_neighbors,
                atac_n_pcs=atac_n_pcs,
                
                # Leiden clustering parameters
                atac_leiden_resolution=atac_leiden_resolution,
                atac_existing_cell_types=atac_existing_cell_types,
                atac_n_target_cell_clusters=atac_n_target_cell_clusters,
                
                # Cell type annotation parameters
                cell_type_annotation=atac_cell_type_annotation,
                atac_cell_type_annotation_model_name=atac_cell_type_annotation_model_name,
                atac_cell_type_annotation_custom_model_path=atac_cell_type_annotation_custom_model_path,
                
                # UMAP parameters
                atac_umap_min_dist=atac_umap_min_dist,
                atac_umap_spread=atac_umap_spread,
                atac_umap_random_state=atac_umap_random_state,
                atac_plot_dpi=atac_plot_dpi,
                
                # Paths for skipping preprocessing
                atac_cell_path=atac_cell_path,
                atac_sample_path=atac_sample_path,
                
                # Pseudobulk parameters
                atac_pseudobulk_n_features=atac_pseudobulk_n_features,
                atac_pseudobulk_verbose=atac_pseudobulk_verbose,
                
                # PCA parameters
                atac_dr_n_expression_components=atac_dr_n_expression_components,
                atac_dr_n_proportion_components=atac_dr_n_proportion_components,
                atac_dr_verbose=atac_dr_verbose,
                
                # Trajectory analysis parameters
                trajectory_supervised_atac=atac_trajectory_supervised,
                n_components_for_cca_atac = n_components_for_cca_atac,
                sev_col_cca=atac_sev_col_cca,
                trajectory_verbose=atac_trajectory_verbose,
                cca_pvalue=atac_cca_pvalue,
                atac_cca_optimal_cell_resolution=atac_cca_optimal_cell_resolution,
                n_pcs_for_null_atac=atac_n_pcs_for_null,
                TSCAN_origin=atac_tscan_origin,
                trajectory_visualization_label=atac_trajectory_visualization_label,
                
                # Trajectory differential analysis parameters
                fdr_threshold=atac_fdr_threshold,
                effect_size_threshold=atac_effect_size_threshold,
                top_n_genes=atac_top_n_genes,
                trajectory_diff_gene_covariate=atac_trajectory_diff_gene_covariate,
                num_splines=atac_num_splines,
                spline_order=atac_spline_order,
                visualization_gene_list=atac_visualization_gene_list,
                visualize_all_deg=atac_visualize_all_deg,
                top_n_heatmap=atac_top_n_heatmap,
                trajectory_diff_gene_verbose=atac_trajectory_diff_gene_verbose,
                top_gene_number=atac_top_gene_number,
                atac_trajectory_diff_gene_output_dir=atac_trajectory_diff_gene_output_dir,
                
                # Sample distance parameters
                sample_distance_methods=atac_sample_distance_methods,
                summary_sample_csv_path=atac_summary_sample_csv_path,
                grouping_columns=atac_grouping_columns,
                
                # Clustering parameters
                Kmeans_based_cluster_flag=atac_kmeans_based_cluster_flag,
                Tree_building_method=atac_tree_building_method,
                proportion_test=atac_proportion_test,
                RAISIN_analysis=atac_raisin_analysis,
                cluster_distance_method=atac_cluster_distance_method,
                cluster_number=atac_cluster_number,
                user_provided_sample_to_clade=atac_user_provided_sample_to_clade,
                
                # Visualization parameters
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
                
                # Additional parameters
                sample_col='sample',
                batch_col='batch',
                verbose=verbose,
                status_flags=status_flags,
            )
            
            results['atac_results'] = atac_results
            
            # Update status flags
            if isinstance(atac_results, dict) and 'status_flags' in atac_results:
                status_flags = atac_results['status_flags']
            elif isinstance(atac_results, tuple) and len(atac_results) > 3:
                # Handle tuple return format (atac_sample, atac_cell, pseudobulk_anndata)
                results['atac_results'] = {
                    'atac_sample': atac_results[0],
                    'atac_cell': atac_results[1],
                    'pseudobulk_anndata': atac_results[2]
                }
            
            # Save updated status
            with open(status_file_path, 'w') as f:
                json.dump(status_flags, f, indent=4)
            
            print("\n✓ ATAC pipeline completed successfully!")
            
        except Exception as e:
            print(f"\n✗ ATAC pipeline failed with error: {e}")
            results['atac_error'] = str(e)
            if verbose:
                import traceback
                traceback.print_exc()
    
    # ========================================
    # MULTIOMICS PIPELINE
    # ========================================
    if run_multiomics_pipeline:
        print("\n" + "="*60)
        print("RUNNING MULTIOMICS PIPELINE")
        print("="*60)
        
        # Set default multiomics output directory if not specified
        if multiomics_output_dir is None:
            multiomics_output_dir = os.path.join(output_dir, 'multiomics')
        
        # Validate required multiomics parameters
        if (multiomics_run_glue and 
            (multiomics_rna_file is None or multiomics_atac_file is None or
             multiomics_rna_sample_meta_file is None or multiomics_atac_sample_meta_file is None)):
            raise ValueError("Multiomics pipeline with GLUE requires all data files and metadata")
        
        try:
            multiomics_results = multiomics_wrapper(
                # Pipeline control flags
                run_glue=multiomics_run_glue,
                run_integrate_preprocess=multiomics_run_integrate_preprocess,
                run_compute_pseudobulk=multiomics_run_compute_pseudobulk,
                run_process_pca=multiomics_run_process_pca,
                run_visualize_embedding=multiomics_run_visualize_embedding,
                run_find_optimal_resolution=multiomics_run_find_optimal_resolution,
                
                # GLUE parameters
                rna_file=multiomics_rna_file,
                atac_file=multiomics_atac_file,
                rna_sample_meta_file=multiomics_rna_sample_meta_file,
                atac_sample_meta_file=multiomics_atac_sample_meta_file,
                multiomics_output_dir=multiomics_output_dir,
                
                # Preprocessing parameters
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
                random_state=multiomics_random_state,
                metadata_sep=multiomics_metadata_sep,
                rna_sample_column=multiomics_rna_sample_column,
                atac_sample_column=multiomics_atac_sample_column,
                
                # Training parameters
                consistency_threshold=multiomics_consistency_threshold,
                save_prefix=multiomics_save_prefix,
                
                # Gene activity computation parameters
                k_neighbors=multiomics_k_neighbors,
                use_rep=multiomics_use_rep,
                metric=multiomics_metric,
                use_gpu=use_gpu,
                existing_cell_types=multiomics_existing_cell_types,
                n_target_clusters=multiomics_n_target_clusters,
                cluster_resolution=multiomics_cluster_resolution,
                use_rep_celltype=multiomics_use_rep_celltype,
                markers=multiomics_markers,
                method=multiomics_method,
                metric_celltype=multiomics_metric_celltype,
                distance_mode=multiomics_distance_mode,
                generate_umap_celltype=multiomics_generate_umap_celltype,
                plot_columns=multiomics_plot_columns,
                
                # Integration preprocessing parameters
                integrate_output_dir=multiomics_integrate_output_dir,
                h5ad_path=multiomics_h5ad_path,
                sample_column=multiomics_sample_column,
                min_cells_sample=multiomics_min_cells_sample,
                min_cell_gene=multiomics_min_cell_gene,
                min_features=multiomics_min_features,
                pct_mito_cutoff=multiomics_pct_mito_cutoff,
                exclude_genes=multiomics_exclude_genes,
                doublet=multiomics_doublet,
                
                # Pseudobulk parameters
                batch_col=multiomics_batch_col,
                sample_col=multiomics_sample_col,
                celltype_col=multiomics_celltype_col,
                pseudobulk_output_dir=multiomics_pseudobulk_output_dir,
                Save=multiomics_save,
                n_features=multiomics_n_features,
                normalize=multiomics_normalize,
                target_sum=multiomics_target_sum,
                atac=multiomics_atac,
                
                # PCA parameters
                pca_sample_col=multiomics_pca_sample_col,
                n_expression_pcs=multiomics_n_expression_pcs,
                n_proportion_pcs=multiomics_n_proportion_pcs,
                pca_output_dir=multiomics_pca_output_dir,
                integrated_data=multiomics_integrated_data,
                not_save=multiomics_not_save,
                pca_atac=multiomics_pca_atac,
                use_snapatac2_dimred=multiomics_use_snapatac2_dimred,
                
                # Visualization parameters
                modality_col=multiomics_modality_col,
                color_col=multiomics_color_col,
                target_modality=multiomics_target_modality,
                expression_key=multiomics_expression_key,
                proportion_key=multiomics_proportion_key,
                figsize=multiomics_figsize,
                point_size=multiomics_point_size,
                alpha=multiomics_alpha,
                colormap=multiomics_colormap,
                viz_output_dir=multiomics_viz_output_dir,
                show_sample_names=multiomics_show_sample_names,
                force_data_type=multiomics_force_data_type,
                
                # Optimal resolution parameters
                optimization_target=multiomics_optimization_target,
                dr_type=multiomics_dr_type,
                resolution_n_features=multiomics_resolution_n_features,
                sev_col=multiomics_sev_col,
                resolution_batch_col=multiomics_resolution_batch_col,
                resolution_sample_col=multiomics_resolution_sample_col,
                resolution_modality_col=multiomics_resolution_modality_col,
                resolution_use_rep=multiomics_resolution_use_rep,
                num_DR_components=multiomics_num_dr_components,
                num_PCs=multiomics_num_pcs,
                num_pvalue_simulations=multiomics_num_pvalue_simulations,
                n_pcs=multiomics_n_pcs,
                compute_pvalues=multiomics_compute_pvalues,
                visualize_embeddings=multiomics_visualize_embeddings,
                resolution_output_dir=multiomics_resolution_output_dir,
                
                # Global parameters
                multiomics_verbose=multiomics_verbose,
                save_intermediate=save_intermediate,
                integrated_h5ad_path=multiomics_integrated_h5ad_path,
                pseudobulk_h5ad_path=multiomics_pseudobulk_h5ad_path,
                
                # Status flags
                status_flags=status_flags,
            )
            
            results['multiomics_results'] = multiomics_results
            
            # Update status flags from multiomics results
            if 'status_flags' in multiomics_results:
                status_flags = multiomics_results['status_flags']
                
            # Save updated status
            with open(status_file_path, 'w') as f:
                json.dump(status_flags, f, indent=4)
            
            print("\n✓ Multiomics pipeline completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Multiomics pipeline failed with error: {e}")
            results['multiomics_error'] = str(e)
            if verbose:
                import traceback
                traceback.print_exc()
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*60)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*60)
    
    # Print completion status for each pipeline
    if run_rna_pipeline:
        rna_complete = all(status_flags['rna'].values())
        print(f"RNA Pipeline: {'✓ COMPLETE' if rna_complete else '⚠ PARTIAL'}")
        if verbose:
            for step, completed in status_flags['rna'].items():
                print(f"  - {step}: {'✓' if completed else '✗'}")
    
    if run_atac_pipeline:
        atac_complete = all(status_flags['atac'].values())
        print(f"ATAC Pipeline: {'✓ COMPLETE' if atac_complete else '⚠ PARTIAL'}")
        if verbose:
            for step, completed in status_flags['atac'].items():
                print(f"  - {step}: {'✓' if completed else '✗'}")
    
    if run_multiomics_pipeline:
        multiomics_complete = all(status_flags['multiomics'].values())
        print(f"Multiomics Pipeline: {'✓ COMPLETE' if multiomics_complete else '⚠ PARTIAL'}")
        if verbose:
            for step, completed in status_flags['multiomics'].items():
                print(f"  - {step}: {'✓' if completed else '✗'}")
    
    print("\nResults saved to:", output_dir)
    print("Status file:", status_file_path)
    
    # Save final status
    with open(status_file_path, 'w') as f:
        json.dump(status_flags, f, indent=4)
    
    # Return comprehensive results
    return results