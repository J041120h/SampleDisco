import os
import scanpy as sc
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preparation.atac_preprocess_gpu import preprocess_linux
from preparation.atac_preprocess_cpu import preprocess
from preparation.ATAC_cell_type import cell_types_atac
from visualization.ATAC_visualization import DR_visualization_all
from sample_trajectory.ATAC_CCA_test import find_optimal_cell_resolution_atac
from sample_embedding.calculate_sample_embedding import calculate_sample_embedding
from sample_trajectory.CCA import CCA_Call
from sample_trajectory.CCA_test import cca_pvalue_test
from sample_trajectory.TSCAN import TSCAN
from sample_distance.sample_distance import sample_distance
from cluster import cluster
from sample_trajectory.trajectory_diff_gene import run_trajectory_gam_differential_gene_analysis
from sample_clustering.RAISIN import raisinfit
from sample_clustering.RAISIN_TEST import run_pairwise_tests
from sample_clustering.proportion_test import proportion_test as run_proportion_test


def atac_wrapper(
    # Required paths
    atac_count_data_path: str = None,
    atac_output_dir: str = None,
    
    # Pipeline step flags (aligned with RNA wrapper naming)
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
    
    # Runtime options
    use_gpu: bool = False,
    verbose: bool = True,
    status_flags: dict = None,
    
    # Data paths (aligned with RNA wrapper)
    adata_cell_path: str = None,
    adata_sample_path: str = None,
    atac_sample_meta_path: str = None,
    cell_meta_path: str = None,
    pseudo_adata_path: str = None,
    
    # Common parameters (aligned with RNA wrapper naming)
    sample_col: str = 'sample',
    batch_col: str = None,
    celltype_col: str = 'cell_type',
    cell_embedding_column: str = "X_lsi_harmony",
    
    # Preprocessing parameters (ATAC-specific)
    min_cells: int = 1,
    min_features: int = 2000,
    max_features: int = 15000,
    min_cells_per_sample: int = 1,
    exclude_features: list = None,
    vars_to_regress: list = None,
    doublet_detection: bool = True,
    num_cell_hvfs: int = 50000,
    cell_embedding_num_pcs: int = 50,
    num_harmony_iterations: int = 30,
    tfidf_scale_factor: float = 1e4,
    log_transform: bool = True,
    drop_first_lsi: bool = True,
    
    # Cell type clustering parameters
    leiden_cluster_resolution: float = 0.8,
    existing_cell_types: bool = False,
    n_target_cell_clusters: int = None,
    umap: bool = False,
    
    # Sample embedding parameters
    sample_hvg_number: int = 50000,
    sample_embedding_dimension: int = 30,
    harmony_for_proportion: bool = True,
    preserve_cols_in_sample_embedding: list = None,
    
    # Trajectory parameters
    n_cca_pcs: int = 2,
    trajectory_col: str = "sev.level",
    trajectory_supervised: bool = True,
    trajectory_visualization_label: list = None,
    cca_pvalue: bool = False,
    tscan_origin: str = None,
    
    # Trajectory DGE parameters
    fdr_threshold: float = 0.05,
    effect_size_threshold: float = 1.0,
    top_n_genes: int = 100,
    trajectory_diff_gene_covariate: list = None,
    num_splines: int = 5,
    spline_order: int = 3,
    visualization_gene_list: list = None,
    
    # Sample distance parameters
    sample_distance_methods: list = None,
    grouping_columns: list = None,
    summary_sample_csv_path: str = None,
    
    # Clustering parameters
    cluster_number: int = 4,
    cluster_differential_gene_group_col: str = None,
    
    # Visualization parameters
    age_bin_size: int = None,
    age_column: str = 'age',
    plot_dendrogram_flag: bool = True,
    plot_cell_type_proportions_pca_flag: bool = False,
    plot_cell_type_expression_umap_flag: bool = False,
    atac_figsize: tuple = (10, 8),
    atac_point_size: int = 50,
    atac_visualization_grouping_columns: list = None,
    atac_show_sample_names: bool = True,
    
) -> dict:
    """
    ATAC-seq analysis wrapper function.
    
    This function orchestrates the complete ATAC-seq analysis pipeline including:
    preprocessing, cell type clustering, sample embedding, trajectory analysis,
    differential gene expression, sample clustering, and visualization.
    
    Parameters
    ----------
    atac_count_data_path : str
        Path to the input h5ad file containing ATAC count data.
    atac_output_dir : str
        Directory to save all output files.
    preprocessing : bool, default=True
        Whether to run preprocessing step.
    cell_type_cluster : bool, default=True
        Whether to run cell type clustering.
    derive_sample_embedding : bool, default=True
        Whether to derive sample embeddings.
    cca_based_cell_resolution_selection : bool, default=False
        Whether to find optimal cell resolution using CCA.
    sample_distance_calculation : bool, default=True
        Whether to calculate sample distances.
    trajectory_analysis : bool, default=True
        Whether to run trajectory analysis.
    trajectory_DGE : bool, default=True
        Whether to run trajectory differential gene expression.
    sample_cluster : bool, default=True
        Whether to run sample clustering.
    proportion_test : bool, default=False
        Whether to run proportion tests.
    cluster_DGE : bool, default=False
        Whether to run cluster differential gene expression (RAISIN).
    visualize_data : bool, default=True
        Whether to generate visualizations.
    use_gpu : bool, default=False
        Whether to use GPU acceleration.
    verbose : bool, default=True
        Whether to print progress messages.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'adata_cell': Cell-level AnnData object
        - 'adata_sample': Sample-level AnnData object
        - 'pseudobulk_df': Pseudobulk DataFrame
        - 'pseudo_adata': Pseudobulk AnnData object
        - 'status_flags': Status flags for each pipeline step
    """
    print("Starting ATAC wrapper function...")
    
    if atac_count_data_path is None or atac_output_dir is None:
        raise ValueError("Required parameters atac_count_data_path and atac_output_dir must be provided.")
    
    # Set default values for list parameters
    if grouping_columns is None:
        grouping_columns = ['sev.level']
    if vars_to_regress is None:
        vars_to_regress = []
    if sample_distance_methods is None:
        sample_distance_methods = ['cosine', 'correlation']
    if trajectory_visualization_label is None:
        trajectory_visualization_label = ['sev.level']
    if atac_visualization_grouping_columns is None:
        atac_visualization_grouping_columns = ['sev.level']
    if summary_sample_csv_path is None:
        summary_sample_csv_path = os.path.join(atac_output_dir, 'summary_sample.csv')
    
    trajectory_diff_gene_output_dir = os.path.join(atac_output_dir, 'trajectoryDEG')
    
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
        status_flags = {"atac": default_status.copy()}
    elif "atac" not in status_flags:
        status_flags["atac"] = default_status.copy()

    adata_cell = None
    adata_sample = None
    pseudobulk_df = None
    pseudo_adata = None

    # =========================================================================
    # STEP 1: PREPROCESSING
    # =========================================================================
    if preprocessing:
        print("Starting preprocessing...")
        preprocess_func = preprocess_linux if use_gpu else preprocess
        
        adata_cell, adata_sample = preprocess_func(
            h5ad_path=atac_count_data_path,
            sample_meta_path=atac_sample_meta_path,
            output_dir=atac_output_dir,
            sample_column=sample_col,
            cell_meta_path=cell_meta_path,
            batch_key=batch_col,
            cell_embedding_num_PCs=cell_embedding_num_pcs,
            num_harmony_iterations=num_harmony_iterations,
            num_cell_hvfs=num_cell_hvfs,
            min_cells=min_cells,
            min_features=min_features,
            max_features=max_features,
            min_cells_per_sample=min_cells_per_sample,
            exclude_features=exclude_features,
            vars_to_regress=vars_to_regress,
            doublet_detection=doublet_detection,
            tfidf_scale_factor=tfidf_scale_factor,
            log_transform=log_transform,
            drop_first_lsi=drop_first_lsi,
            verbose=verbose
        )
        status_flags["atac"]["preprocessing"] = True
    else:
        cell_path = adata_cell_path or os.path.join(atac_output_dir, "preprocess", "adata_cell.h5ad")
        sample_path = adata_sample_path or os.path.join(atac_output_dir, "preprocess", "adata_sample.h5ad")
        
        if not os.path.exists(cell_path) or not os.path.exists(sample_path):
            raise ValueError("Preprocessed data paths not provided and default files do not exist.")
        
        status_flags["atac"]["preprocessing"] = True
        status_flags["atac"]["cell_type_cluster"] = True
        
        needs_data = any([
            cell_type_cluster, derive_sample_embedding, trajectory_analysis,
            trajectory_DGE, sample_cluster, cluster_DGE, proportion_test,
            cca_based_cell_resolution_selection, visualize_data
        ])
        
        if needs_data:
            adata_cell = sc.read(cell_path)
            adata_sample = sc.read(sample_path)
    
    if not status_flags["atac"]["preprocessing"]:
        raise ValueError("ATAC preprocessing skipped but no preprocessed data found.")

    # =========================================================================
    # STEP 2: CELL TYPE CLUSTERING
    # =========================================================================
    if cell_type_cluster:
        print(f"Starting cell type clustering at resolution: {leiden_cluster_resolution}")
        
        adata_sample = cell_types_atac(
            adata=adata_sample,
            cell_column=celltype_col,
            Save=True,
            existing_cell_types=existing_cell_types,
            n_target_clusters=n_target_cell_clusters,
            cluster_resolution=leiden_cluster_resolution,
            use_rep=cell_embedding_column,
            method='average',
            metric='euclidean',
            distance_mode='centroid',
            num_DMs=cell_embedding_num_pcs,
            output_dir=atac_output_dir,
            verbose=verbose
        )
        status_flags["atac"]["cell_type_cluster"] = True

    # =========================================================================
    # STEP 3: SAMPLE EMBEDDING DERIVATION
    # =========================================================================
    if derive_sample_embedding:
        print("Starting sample embedding derivation...")
        
        if not status_flags["atac"]["cell_type_cluster"]:
            raise ValueError("Cell type clustering required before sample embedding derivation.")
        
        pseudobulk_df, pseudo_adata = calculate_sample_embedding(
            adata=adata_sample,
            sample_col=sample_col,
            celltype_col=celltype_col,
            batch_col=batch_col,
            output_dir=atac_output_dir,
            sample_hvg_number=sample_hvg_number,
            n_expression_components=sample_embedding_dimension,
            n_proportion_components=sample_embedding_dimension,
            harmony_for_proportion=harmony_for_proportion,
            preserve_cols_in_sample_embedding=preserve_cols_in_sample_embedding,
            use_gpu=use_gpu,
            atac=True,
            save=True,
            verbose=verbose,
        )
        status_flags["atac"]["derive_sample_embedding"] = True
    else:
        pseudobulk_path = pseudo_adata_path or os.path.join(atac_output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
        
        needs_pseudobulk = any([trajectory_analysis, trajectory_DGE, sample_cluster, cluster_DGE, visualize_data])
        
        if needs_pseudobulk and not os.path.exists(pseudobulk_path):
            raise ValueError("Sample embedding skipped but no sample embedding data found.")
        
        if os.path.exists(pseudobulk_path):
            print(f"Loading pseudobulk from: {pseudobulk_path}")
            pseudo_adata = sc.read(pseudobulk_path)
        
        status_flags["atac"]["derive_sample_embedding"] = True

    # =========================================================================
    # STEP 4: CCA-BASED CELL RESOLUTION SELECTION (OPTIONAL)
    # =========================================================================
    if cca_based_cell_resolution_selection:
        print("Finding optimal cell resolution...")
        
        for column in ["X_DR_expression", "X_DR_proportion"]:
            find_optimal_cell_resolution_atac(
                AnnData_cell=adata_cell,
                AnnData_sample=adata_sample,
                output_dir=atac_output_dir,
                column=column,
                n_features=sample_hvg_number,
                sample_col=sample_col,
                batch_col=batch_col,
                num_DR_components=sample_embedding_dimension,
                num_DMs=cell_embedding_num_pcs,
                n_pcs=n_cca_pcs,
                preserve_cols=preserve_cols_in_sample_embedding,
            )
        
        from utils.unify_optimal import replace_optimal_dimension_reduction
        pseudo_adata = replace_optimal_dimension_reduction(atac_output_dir)

    # =========================================================================
    # STEP 5: SAMPLE DISTANCE CALCULATION
    # =========================================================================
    if sample_distance_calculation:
        print("Starting sample distance calculation...")
        
        for distance_method in sample_distance_methods:
            print(f"Running sample distance: {distance_method}")
            sample_distance(
                adata=pseudo_adata,
                output_dir=os.path.join(atac_output_dir, 'Sample_distance'),
                method=distance_method,
                grouping_columns=grouping_columns,
                summary_csv_path=summary_sample_csv_path,
                cell_adata=adata_sample,
                cell_type_column=celltype_col,
                sample_column=sample_col,
                pseudobulk_adata=pseudo_adata
            )
        
        status_flags["atac"]["sample_distance_calculation"] = True
        if verbose:
            print(f"Sample distance calculation completed: {os.path.join(atac_output_dir, 'Sample_distance')}")

    # =========================================================================
    # STEP 6: TRAJECTORY ANALYSIS
    # =========================================================================
    ptime_expression = None
    ptime_proportion = None
    
    if trajectory_analysis:
        print("Starting trajectory analysis...")
        
        if not status_flags["atac"]["derive_sample_embedding"]:
            raise ValueError("Sample embedding derivation required before trajectory analysis.")
        
        if trajectory_supervised:
            if trajectory_col not in pseudo_adata.obs.columns:
                raise ValueError(f"Trajectory column '{trajectory_col}' not found in pseudo_adata.obs.")
            
            cca_score_proportion, cca_score_expression, ptime_proportion, ptime_expression = CCA_Call(
                adata=pseudo_adata,
                n_components=n_cca_pcs,
                output_dir=atac_output_dir,
                trajectory_col=trajectory_col,
                verbose=verbose
            )
            
            if cca_pvalue:
                cca_pvalue_test(
                    pseudo_adata=pseudo_adata,
                    column="X_DR_proportion",
                    input_correlation=cca_score_proportion,
                    output_directory=atac_output_dir,
                    trajectory_col=trajectory_col,
                    verbose=verbose
                )
                cca_pvalue_test(
                    pseudo_adata=pseudo_adata,
                    column="X_DR_expression",
                    input_correlation=cca_score_expression,
                    output_directory=atac_output_dir,
                    trajectory_col=trajectory_col,
                    verbose=verbose
                )
        else:
            tscan_result_expression = TSCAN(
                AnnData_sample=pseudo_adata,
                column="X_DR_expression",
                output_dir=atac_output_dir,
                grouping_columns=trajectory_visualization_label,
                verbose=verbose,
                origin=tscan_origin
            )
            
            tscan_result_proportion = TSCAN(
                AnnData_sample=pseudo_adata,
                column="X_DR_proportion",
                output_dir=atac_output_dir,
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
        
        status_flags["atac"]["trajectory_analysis"] = True

        # Trajectory DGE (nested under trajectory_analysis like RNA wrapper)
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
            
            status_flags["atac"]["trajectory_dge"] = True
            print("Trajectory differential gene analysis completed!")

    # Clean up summary CSV
    if os.path.exists(summary_sample_csv_path):
        os.remove(summary_sample_csv_path)

    # =========================================================================
    # STEP 7: SAMPLE CLUSTERING
    # =========================================================================
    expr_results, prop_results = {}, {}
    
    if sample_cluster:
        print("Starting sample clustering...")
        
        expr_results, prop_results = cluster(
            pseudobulk_adata=pseudo_adata,
            output_dir=atac_output_dir,
            number_of_clusters=cluster_number,
            use_expression=True,
            use_proportion=True,
            random_state=0,
        )

    # =========================================================================
    # STEP 8: PROPORTION TEST
    # =========================================================================
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
                    output_dir=os.path.join(atac_output_dir, "sample_cluster", "expression", "proportion_test"),
                    verbose=True
                )
            
            if cluster_differential_gene_group_col is not None or prop_results:
                run_proportion_test(
                    adata=adata_sample,
                    sample_col=sample_col,
                    sample_to_clade=prop_results,
                    group_col=cluster_differential_gene_group_col,
                    celltype_col=celltype_col,
                    output_dir=os.path.join(atac_output_dir, "sample_cluster", "proportion", "proportion_test"),
                    verbose=True
                )
            
            print("Proportion tests completed.")
        except Exception as e:
            print(f"Error in proportion test: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # STEP 9: CLUSTER DGE (RAISIN)
    # =========================================================================
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
                    output_dir=os.path.join(atac_output_dir, 'raisin_results_expression'),
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

    status_flags["atac"]["cluster_dge"] = True

    # =========================================================================
    # STEP 10: VISUALIZATION
    # =========================================================================
    if visualize_data:
        print("Starting visualization...")
        
        if plot_dendrogram_flag and not status_flags["atac"]["cell_type_cluster"]:
            raise ValueError("Cell type clustering required for dendrogram visualization.")
        
        if (plot_cell_type_proportions_pca_flag or plot_cell_type_expression_umap_flag) and not status_flags["atac"]["derive_sample_embedding"]:
            raise ValueError("Sample embedding derivation required for requested visualization.")
        
        visualization_output_dir = os.path.join(atac_output_dir, "visualization")
        
        DR_visualization_all(
            adata_sample,
            figsize=atac_figsize,
            point_size=atac_point_size,
            alpha=0.7,
            output_dir=visualization_output_dir,
            grouping_columns=atac_visualization_grouping_columns,
            age_bin_size=age_bin_size,
            show_sample_names=atac_show_sample_names,
            sample_col=sample_col
        )
        
        status_flags["atac"]["visualization"] = True

    print("ATAC analysis completed successfully!")
    
    return {
        'adata_cell': adata_cell,
        'adata_sample': adata_sample,
        'pseudobulk_df': pseudobulk_df,
        'pseudo_adata': pseudo_adata,
        'status_flags': status_flags
    }