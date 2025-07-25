import anndata as ad
import networkx as nx
import scanpy as sc
import sys
import scglue
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import pyensembl
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from linux.CellType_linux import *
from CellType import *
from integration.integrate_glue import *
from integration.integrate_preprocess import *
from integration.integration_CCA_test import *
from integration.integration_optimal_resolution import *
from integration.integration_validation import *
from integration.integration_visualization import *

def multiomics_pipeline(
    # ========================================
    # PIPELINE CONTROL FLAGS
    # ========================================
    run_glue: bool = True,
    run_integrate_preprocess: bool = True,
    run_compute_pseudobulk: bool = True,
    run_process_pca: bool = True,
    run_visualize_embedding: bool = True,
    run_find_optimal_resolution: bool = False,
    
    # ========================================
    # GLUE FUNCTION PARAMETERS
    # ========================================
    # Data files
    rna_file: Optional[str] = None,
    atac_file: Optional[str] = None,
    rna_sample_meta_file: Optional[str] = None,
    atac_sample_meta_file: Optional[str] = None,
    
    # Preprocessing parameters
    ensembl_release: int = 98,
    species: str = "homo_sapiens",
    use_highly_variable: bool = True,
    n_top_genes: int = 2000,
    n_pca_comps: int = 50,
    n_lsi_comps: int = 50,
    lsi_n_iter: int = 15,
    gtf_by: str = "gene_name",
    flavor: str = "seurat_v3",
    generate_umap: bool = False,
    compression: str = "gzip",
    random_state: int = 42,
    metadata_sep: str = ",",
    rna_sample_column: str = "sample",
    atac_sample_column: str = "sample",
    
    # Training parameters
    consistency_threshold: float = 0.05,
    save_prefix: str = "glue",
    
    # Gene activity computation parameters
    k_neighbors: int = 10,
    use_rep: str = "X_glue",
    metric: str = "cosine",
    use_gpu: bool = True,
    existing_cell_types: bool = False,
    n_target_clusters: int = 10,
    cluster_resolution: float = 0.8,
    use_rep_celltype: str = "X_glue",
    markers: Optional[List] = None,
    method: str = 'average',
    metric_celltype: str = 'euclidean',
    distance_mode: str = 'centroid',
    generate_umap_celltype: bool = True,
    
    # Visualization parameters
    plot_columns: Optional[List[str]] = None,
    
    # ========================================
    # INTEGRATE_PREPROCESS FUNCTION PARAMETERS
    # ========================================
    integrate_output_dir: Optional[str] = None,
    h5ad_path: Optional[str] = None,
    sample_column: str = 'sample',
    min_cells_sample: int = 1,
    min_cell_gene: int = 10,
    min_features: int = 500,
    pct_mito_cutoff: int = 20,
    exclude_genes: Optional[List] = None,
    doublet: bool = True,
    
    # ========================================
    # COMPUTE_PSEUDOBULK_ADATA FUNCTION PARAMETERS
    # ========================================
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    pseudobulk_output_dir: Optional[str] = None,
    Save: bool = True,
    n_features: int = 2000,
    normalize: bool = True,
    target_sum: float = 1e4,
    atac: bool = False,
    
    # ========================================
    # PROCESS_ANNDATA_WITH_PCA FUNCTION PARAMETERS
    # ========================================
    pca_sample_col: str = 'sample',
    n_expression_pcs: int = 10,
    n_proportion_pcs: int = 10,
    pca_output_dir: Optional[str] = None,
    integrated_data: bool = False,
    not_save: bool = False,
    pca_atac: bool = False,
    use_snapatac2_dimred: bool = False,
    
    # ========================================
    # VISUALIZE_MULTIMODAL_EMBEDDING FUNCTION PARAMETERS
    # ========================================
    modality_col: str = 'modality',
    color_col: str = 'color',
    target_modality: str = 'ATAC',
    expression_key: str = 'X_DR_expression',
    proportion_key: str = 'X_DR_proportion',
    figsize: Tuple[int, int] = (20, 8),
    point_size: int = 60,
    alpha: float = 0.8,
    colormap: str = 'viridis',
    viz_output_dir: Optional[str] = None,
    show_sample_names: bool = False,
    force_data_type: Optional[str] = None,
    
    # ========================================
    # FIND_OPTIMAL_CELL_RESOLUTION_INTEGRATION FUNCTION PARAMETERS
    # ========================================
    optimization_target: str = "rna",  # "rna" or "atac"
    dr_type: str = "expression",  # "expression" or "proportion"
    resolution_n_features: int = 40000,
    sev_col: str = "sev.level",
    resolution_batch_col: Optional[str] = None,
    resolution_sample_col: str = "sample",
    resolution_modality_col: str = "modality",
    resolution_use_rep: str = 'X_glue',
    num_DR_components: int = 30,
    num_PCs: int = 20,
    num_pvalue_simulations: int = 1000,
    n_pcs: int = 2,
    compute_pvalues: bool = True,
    visualize_embeddings: bool = True,
    resolution_output_dir: Optional[str] = None,
    
    # ========================================
    # GLOBAL PARAMETERS
    # ========================================
    output_dir: str = "./pipeline_results",
    verbose: bool = True,
    save_intermediate: bool = True,
    
    # Alternative input paths (for skipping early steps)
    integrated_h5ad_path: Optional[str] = None,
    pseudobulk_h5ad_path: Optional[str] = None,
    
) -> Dict[str, Any]:
    """
    Comprehensive wrapper for multi-modal single-cell analysis pipeline with all parameters explicitly defined.
    
    Parameters:
    -----------
    
    PIPELINE CONTROL:
    run_glue : bool, default True
        Whether to run the glue function
    run_integrate_preprocess : bool, default True
        Whether to run the integrate_preprocess function
    run_compute_pseudobulk : bool, default True
        Whether to run the compute_pseudobulk_adata function
    run_process_pca : bool, default True
        Whether to run the process_anndata_with_pca function
    run_visualize_embedding : bool, default True
        Whether to run the visualize_multimodal_embedding function
    run_find_optimal_resolution : bool, default False
        Whether to run the find_optimal_cell_resolution_integration function
    
    GLUE FUNCTION PARAMETERS:
    rna_file : str, optional
        Path to RNA h5ad file
    atac_file : str, optional
        Path to ATAC h5ad file
    rna_sample_meta_file : str, optional
        Path to RNA sample metadata CSV file
    atac_sample_meta_file : str, optional
        Path to ATAC sample metadata CSV file
    ensembl_release : int, default 98
        Ensembl database release version
    species : str, default "homo_sapiens"
        Species name for Ensembl
    use_highly_variable : bool, default True
        Whether to use highly variable genes
    n_top_genes : int, default 2000
        Number of top genes to select
    n_pca_comps : int, default 50
        Number of PCA components
    n_lsi_comps : int, default 50
        Number of LSI components
    lsi_n_iter : int, default 15
        Number of LSI iterations
    gtf_by : str, default "gene_name"
        GTF annotation method
    flavor : str, default "seurat_v3"
        Flavor for highly variable gene selection
    generate_umap : bool, default False
        Whether to generate UMAP
    compression : str, default "gzip"
        Compression method for output files
    random_state : int, default 42
        Random seed
    metadata_sep : str, default ","
        Separator for metadata files
    rna_sample_column : str, default "sample"
        Sample column name in RNA metadata
    atac_sample_column : str, default "sample"
        Sample column name in ATAC metadata
    consistency_threshold : float, default 0.05
        Consistency threshold for training
    save_prefix : str, default "glue"
        Prefix for saved files
    k_neighbors : int, default 10
        Number of neighbors for gene activity computation
    use_rep : str, default "X_glue"
        Representation to use
    metric : str, default "cosine"
        Distance metric
    use_gpu : bool, default True
        Whether to use GPU
    existing_cell_types : bool, default False
        Whether cell types already exist
    n_target_clusters : int, default 10
        Number of target clusters
    cluster_resolution : float, default 0.8
        Clustering resolution
    use_rep_celltype : str, default "X_glue"
        Representation for cell type analysis
    markers : List, optional
        Marker genes list
    method : str, default 'average'
        Method for computation
    metric_celltype : str, default 'euclidean'
        Metric for cell type analysis
    distance_mode : str, default 'centroid'
        Distance computation mode
    generate_umap_celltype : bool, default True
        Whether to generate UMAP for cell types
    plot_columns : List[str], optional
        Columns to plot
    
    INTEGRATE_PREPROCESS PARAMETERS:
    integrate_output_dir : str, optional
        Output directory for integration preprocessing (defaults to output_dir/preprocess)
    h5ad_path : str, optional
        Path to input h5ad file
    sample_column : str, default 'sample'
        Sample column name
    min_cells_sample : int, default 1
        Minimum cells per sample
    min_cell_gene : int, default 10
        Minimum genes per cell
    min_features : int, default 500
        Minimum features
    pct_mito_cutoff : int, default 20
        Mitochondrial percentage cutoff
    exclude_genes : List, optional
        Genes to exclude
    doublet : bool, default True
        Whether to detect doublets
    
    COMPUTE_PSEUDOBULK_ADATA PARAMETERS:
    batch_col : str, default 'batch'
        Batch column name
    sample_col : str, default 'sample'
        Sample column name
    celltype_col : str, default 'cell_type'
        Cell type column name
    pseudobulk_output_dir : str, optional
        Output directory for pseudobulk (defaults to output_dir/pseudobulk)
    Save : bool, default True
        Whether to save results
    n_features : int, default 2000
        Number of features
    normalize : bool, default True
        Whether to normalize
    target_sum : float, default 1e4
        Target sum for normalization
    atac : bool, default False
        Whether data is ATAC
    
    PROCESS_ANNDATA_WITH_PCA PARAMETERS:
    pca_sample_col : str, default 'sample'
        Sample column for PCA
    n_expression_pcs : int, default 10
        Number of expression PCs
    n_proportion_pcs : int, default 10
        Number of proportion PCs
    pca_output_dir : str, optional
        Output directory for PCA (defaults to output_dir/pca)
    integrated_data : bool, default False
        Whether data is integrated
    not_save : bool, default False
        Whether to not save results
    pca_atac : bool, default False
        Whether data is ATAC for PCA
    use_snapatac2_dimred : bool, default False
        Whether to use snapATAC2 dimensionality reduction
    
    VISUALIZE_MULTIMODAL_EMBEDDING PARAMETERS:
    modality_col : str, default 'modality'
        Modality column name
    color_col : str, default 'color'
        Color column name
    target_modality : str, default 'ATAC'
        Target modality for visualization
    expression_key : str, default 'X_DR_expression'
        Key for expression data
    proportion_key : str, default 'X_DR_proportion'
        Key for proportion data
    figsize : Tuple[int, int], default (20, 8)
        Figure size
    point_size : int, default 60
        Point size in plots
    alpha : float, default 0.8
        Point transparency
    colormap : str, default 'viridis'
        Colormap for plots
    viz_output_dir : str, optional
        Output directory for visualization (defaults to output_dir/visualization)
    show_sample_names : bool, default False
        Whether to show sample names
    force_data_type : str, optional
        Force specific data type
    
    FIND_OPTIMAL_CELL_RESOLUTION_INTEGRATION PARAMETERS:
    optimization_target : str, default "rna"
        Optimization target ("rna" or "atac")
    dr_type : str, default "expression"
        Dimensionality reduction type ("expression" or "proportion")
    resolution_n_features : int, default 40000
        Number of features for resolution optimization
    sev_col : str, default "sev.level"
        Severity level column
    resolution_batch_col : str, optional
        Batch column for resolution optimization
    resolution_sample_col : str, default "sample"
        Sample column for resolution optimization
    resolution_modality_col : str, default "modality"
        Modality column for resolution optimization
    resolution_use_rep : str, default 'X_glue'
        Representation for resolution optimization
    num_DR_components : int, default 30
        Number of DR components
    num_PCs : int, default 20
        Number of principal components
    num_pvalue_simulations : int, default 1000
        Number of p-value simulations
    n_pcs : int, default 2
        Number of PCs for analysis
    compute_pvalues : bool, default True
        Whether to compute p-values
    visualize_embeddings : bool, default True
        Whether to visualize embeddings
    resolution_output_dir : str, optional
        Output directory for resolution optimization (defaults to output_dir/resolution)
    
    GLOBAL PARAMETERS:
    output_dir : str, default "./pipeline_results"
        Main output directory
    verbose : bool, default True
        Whether to print verbose output
    save_intermediate : bool, default True
        Whether to save intermediate results
    integrated_h5ad_path : str, optional
        Path to existing integrated h5ad file (skips glue step)
    pseudobulk_h5ad_path : str, optional
        Path to existing pseudobulk h5ad file (skips earlier steps)
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results from each executed step
    """
    
    results = {}
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Starting multi-modal pipeline with output directory: {output_dir}")
    
    # Set default subdirectories if not specified
    if integrate_output_dir is None:
        integrate_output_dir = f"{output_dir}/preprocess"
    if pseudobulk_output_dir is None:
        pseudobulk_output_dir = f"{output_dir}/pseudobulk"
    if pca_output_dir is None:
        pca_output_dir = f"{output_dir}/pca"
    if viz_output_dir is None:
        viz_output_dir = f"{output_dir}/visualization"
    if resolution_output_dir is None:
        resolution_output_dir = f"{output_dir}/resolution_optimization"
    
    # Step 1: GLUE integration
    if run_glue:
        if verbose:
            print("Step 1: Running GLUE integration...")
        
        if not rna_file or not atac_file:
            raise ValueError("rna_file and atac_file must be provided when run_glue=True")
        
        glue_result = glue(
            # Data files
            rna_file=rna_file,
            atac_file=atac_file,
            rna_sample_meta_file=rna_sample_meta_file,
            atac_sample_meta_file=atac_sample_meta_file,
            # Preprocessing parameters
            ensembl_release=ensembl_release,
            species=species,
            use_highly_variable=use_highly_variable,
            n_top_genes=n_top_genes,
            n_pca_comps=n_pca_comps,
            n_lsi_comps=n_lsi_comps,
            lsi_n_iter=lsi_n_iter,
            gtf_by=gtf_by,
            flavor=flavor,
            generate_umap=generate_umap,
            compression=compression,
            random_state=random_state,
            metadata_sep=metadata_sep,
            rna_sample_column=rna_sample_column,
            atac_sample_column=atac_sample_column,
            # Training parameters
            consistency_threshold=consistency_threshold,
            save_prefix=save_prefix,
            # Gene activity computation parameters
            k_neighbors=k_neighbors,
            use_rep=use_rep,
            metric=metric,
            use_gpu=use_gpu,
            verbose=verbose,
            existing_cell_types=existing_cell_types,
            n_target_clusters=n_target_clusters,
            cluster_resolution=cluster_resolution,
            use_rep_celltype=use_rep_celltype,
            markers=markers,
            method=method,
            metric_celltype=metric_celltype,
            distance_mode=distance_mode,
            generate_umap_celltype=generate_umap_celltype,
            # Visualization parameters
            plot_columns=plot_columns,
            # Output directory
            output_dir=output_dir
        )
        
        results['glue'] = glue_result
        
        # Set integrated h5ad path for next steps
        if h5ad_path is None:
            h5ad_path = f"{output_dir}/glue/atac_rna_integrated.h5ad"
    
    # Step 2: Integration preprocessing
    if run_integrate_preprocess:
        if verbose:
            print("Step 2: Running integration preprocessing...")
        
        if h5ad_path is None and integrated_h5ad_path:
            h5ad_path = integrated_h5ad_path
        
        if h5ad_path is None:
            raise ValueError("h5ad_path must be provided when run_integrate_preprocess=True")
        
        adata = integrate_preprocess(
            output_dir=integrate_output_dir,
            h5ad_path=h5ad_path,
            sample_column=sample_column,
            min_cells_sample=min_cells_sample,
            min_cell_gene=min_cell_gene,
            min_features=min_features,
            pct_mito_cutoff=pct_mito_cutoff,
            exclude_genes=exclude_genes,
            doublet=doublet,
            verbose=verbose
        )
        
        results['adata'] = adata
    
    # Step 3: Compute pseudobulk
    if run_compute_pseudobulk:
        if verbose:
            print("Step 3: Computing pseudobulk...")
        
        adata_for_pseudobulk = results.get('adata')
        if adata_for_pseudobulk is None:
            raise ValueError("adata must be available from previous step when run_compute_pseudobulk=True")
        
        atac_pseudobulk_df, pseudobulk_adata = compute_pseudobulk_adata(
            adata=adata_for_pseudobulk,
            batch_col=batch_col,
            sample_col=sample_col,
            celltype_col=celltype_col,
            output_dir=pseudobulk_output_dir,
            Save=Save,
            n_features=n_features,
            normalize=normalize,
            target_sum=target_sum,
            atac=atac,
            verbose=verbose
        )
        
        results['atac_pseudobulk_df'] = atac_pseudobulk_df
        results['pseudobulk_adata'] = pseudobulk_adata
    
    # Step 4: Process with PCA
    if run_process_pca:
        if verbose:
            print("Step 4: Processing with PCA...")
        
        adata_for_pca = results.get('adata')
        pseudobulk_for_pca = results.get('atac_pseudobulk_df')
        pseudobulk_adata_for_pca = results.get('pseudobulk_adata')
        
        if not all([adata_for_pca, pseudobulk_for_pca, pseudobulk_adata_for_pca]):
            raise ValueError("adata, atac_pseudobulk_df, and pseudobulk_adata must be available from previous steps when run_process_pca=True")
        
        pseudobulk_anndata_processed = process_anndata_with_pca(
            adata=adata_for_pca,
            pseudobulk=pseudobulk_for_pca,
            pseudobulk_anndata=pseudobulk_adata_for_pca,
            sample_col=pca_sample_col,
            n_expression_pcs=n_expression_pcs,
            n_proportion_pcs=n_proportion_pcs,
            output_dir=pca_output_dir,
            integrated_data=integrated_data,
            not_save=not_save or not save_intermediate,
            atac=pca_atac,
            use_snapatac2_dimred=use_snapatac2_dimred,
            verbose=verbose
        )
        
        results['pseudobulk_anndata_processed'] = pseudobulk_anndata_processed
    
    # Alternative: Load pseudobulk from file
    if pseudobulk_h5ad_path and not run_process_pca:
        if verbose:
            print(f"Loading pseudobulk data from: {pseudobulk_h5ad_path}")
        results['pseudobulk_anndata_processed'] = sc.read_h5ad(pseudobulk_h5ad_path)
    
    # Step 5: Visualize multimodal embedding
    if run_visualize_embedding:
        if verbose:
            print("Step 5: Visualizing multimodal embedding...")
        
        adata_for_viz = results.get('pseudobulk_anndata_processed')
        if adata_for_viz is None:
            raise ValueError("pseudobulk_anndata_processed must be available from previous step when run_visualize_embedding=True")
        
        fig, axes = visualize_multimodal_embedding(
            adata=adata_for_viz,
            modality_col=modality_col,
            color_col=color_col,
            target_modality=target_modality,
            expression_key=expression_key,
            proportion_key=proportion_key,
            figsize=figsize,
            point_size=point_size,
            alpha=alpha,
            colormap=colormap,
            output_dir=viz_output_dir,
            show_sample_names=show_sample_names,
            force_data_type=force_data_type,
            verbose=verbose
        )
        
        results['visualization'] = {'fig': fig, 'axes': axes}
    
    # Step 6: Find optimal resolution (optional)
    if run_find_optimal_resolution:
        if verbose:
            print("Step 6: Finding optimal cell resolution...")
        
        # Setup GPU if available
        try:
            import rmm
            from rmm.allocators.cupy import rmm_cupy_allocator
            import cupy as cp
            rmm.reinitialize(managed_memory=True, pool_allocator=False)
            cp.cuda.set_allocator(rmm_cupy_allocator)
        except:
            if verbose:
                print("GPU setup failed, continuing with CPU")
        
        import torch
        if torch.cuda.is_available() and verbose:
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"GPU Memory Available: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        # Get integrated data for resolution optimization
        integrated_adata_for_resolution = results.get('adata')
        if integrated_adata_for_resolution is None and integrated_h5ad_path:
            integrated_adata_for_resolution = sc.read_h5ad(integrated_h5ad_path)
        
        if integrated_adata_for_resolution is None:
            raise ValueError("Integrated AnnData must be available when run_find_optimal_resolution=True")
        
        # Suppress warnings
        suppress_warnings()
        
        optimal_res, results_df = find_optimal_cell_resolution_integration(
            AnnData_integrated=integrated_adata_for_resolution,
            output_dir=resolution_output_dir,
            optimization_target=optimization_target,
            dr_type=dr_type,
            n_features=resolution_n_features,
            sev_col=sev_col,
            batch_col=resolution_batch_col,
            sample_col=resolution_sample_col,
            modality_col=resolution_modality_col,
            use_rep=resolution_use_rep,
            num_DR_components=num_DR_components,
            num_PCs=num_PCs,
            num_pvalue_simulations=num_pvalue_simulations,
            n_pcs=n_pcs,
            compute_pvalues=compute_pvalues,
            visualize_embeddings=visualize_embeddings,
            verbose=verbose
        )
        
        results['optimal_resolution'] = optimal_res
        results['resolution_results_df'] = results_df
    
    if verbose:
        print("Pipeline completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Available results: {list(results.keys())}")
    
    return results