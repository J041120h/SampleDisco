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
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from linux.CellType_linux import *
from Cell_type import *
from integration.integrate_glue import *
from integration.integrate_preprocess import *
from integration.integration_CCA_test import *
from integration.integration_optimal_resolution import *
from integration.integration_validation import *
from integration.integration_visualization import *

def multiomics_wrapper(
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
    rna_file: str = None,
    atac_file: str = None,
    rna_sample_meta_file: str = None,
    atac_sample_meta_file: str = None,
    multiomics_output_dir: str = None,
    
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
    multiomics_verbose: bool = True,
    save_intermediate: bool = True,
    
    # Alternative input paths (for skipping early steps)
    integrated_h5ad_path: Optional[str] = None,
    pseudobulk_h5ad_path: Optional[str] = None,
    
    # ========================================
    # STATUS FLAGS (NEW ADDITION)
    # ========================================
    status_flags: Optional[Dict[str, Any]] = None,
    
) -> Dict[str, Any]:
    """
    Comprehensive wrapper for multi-modal single-cell analysis pipeline with all parameters explicitly defined
    and status flag tracking similar to the RNA wrapper.
    
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
    
    STATUS FLAGS:
    status_flags : Dict[str, Any], optional
        Dictionary to track completion status of each pipeline step.
        If None, will be initialized with default values.
        Structure: {
            "multiomics": {
                "glue_integration": False,
                "integration_preprocessing": False,
                "pseudobulk_computation": False,
                "pca_processing": False,
                "embedding_visualization": False,
                "optimal_resolution": False
            }
        }
    
    [All other parameters remain the same as documented in original function]
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - Results from each executed step
        - Updated status_flags tracking completion
        - All intermediate data objects
    """

    if any(var is None for var in [rna_file, atac_file, rna_sample_meta_file, atac_sample_meta_file, multiomics_output_dir]):
        raise ValueError("All parameters must be provided (none can be None)")
    
    # Initialize status flags if not provided
    if status_flags is None:
        status_flags = {
            "multiomics": {
                "glue_integration": False,
                "integration_preprocessing": False,
                "pseudobulk_computation": False,
                "pca_processing": False,
                "embedding_visualization": False,
                "optimal_resolution": False
            }
        }
    
    # Ensure multiomics section exists in status_flags
    if "multiomics" not in status_flags:
        status_flags["multiomics"] = {
            "glue_integration": False,
            "integration_preprocessing": False,
            "pseudobulk_computation": False,
            "pca_processing": False,
            "embedding_visualization": False,
            "optimal_resolution": False
        }
    
    results = {}
    
    # Create output directory
    Path(multiomics_output_dir).mkdir(parents=True, exist_ok=True)
    
    if multiomics_verbose:
        print(f"Starting multi-modal pipeline with output directory: {multiomics_output_dir}")
        print(f"Initial status flags: {status_flags['multiomics']}")
    
    # Set default subdirectories if not specified
    if integrate_output_dir is None:
        integrate_output_dir = f"{multiomics_output_dir}/preprocess"
    if pseudobulk_output_dir is None:
        pseudobulk_output_dir = f"{multiomics_output_dir}/pseudobulk"
    if pca_output_dir is None:
        pca_output_dir = f"{multiomics_output_dir}/pca"
    if viz_output_dir is None:
        viz_output_dir = f"{multiomics_output_dir}/visualization"
    if resolution_output_dir is None:
        resolution_output_dir = f"{multiomics_output_dir}/resolution_optimization"
    
    # Step 1: GLUE integration
    if run_glue:
        if multiomics_verbose:
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
            verbose=multiomics_verbose,
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
            output_dir=multiomics_output_dir
        )
        
        results['glue'] = glue_result
        status_flags["multiomics"]["glue_integration"] = True
        
        # Set integrated h5ad path for next steps
        if h5ad_path is None:
            h5ad_path = f"{multiomics_output_dir}/glue/atac_rna_integrated.h5ad"
        
        if multiomics_verbose:
            print("✓ GLUE integration completed successfully")
    else:
        # Load existing integrated data if available
        if not status_flags["multiomics"]["glue_integration"]:
            if integrated_h5ad_path and os.path.exists(integrated_h5ad_path):
                h5ad_path = integrated_h5ad_path
                if multiomics_verbose:
                    print(f"Skipping GLUE integration, using existing data: {h5ad_path}")
            else:
                temp_integrated_path = f"{multiomics_output_dir}/glue/atac_rna_integrated.h5ad"
                if os.path.exists(temp_integrated_path):
                    h5ad_path = temp_integrated_path
                    if multiomics_verbose:
                        print(f"Using previously generated integrated data: {h5ad_path}")
                else:
                    raise ValueError("GLUE integration is skipped, but no integrated data found. "
                                   "Either set run_glue=True or provide integrated_h5ad_path.")
    
    # Step 2: Integration preprocessing
    if run_integrate_preprocess:
        if multiomics_verbose:
            print("Step 2: Running integration preprocessing...")
        
        if not status_flags["multiomics"]["glue_integration"] and not h5ad_path:
            raise ValueError("GLUE integration is required before integration preprocessing.")
        
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
            verbose=multiomics_verbose
        )
        
        results['adata'] = adata
        status_flags["multiomics"]["integration_preprocessing"] = True
        
        if multiomics_verbose:
            print("✓ Integration preprocessing completed successfully")
    else:
        # Load preprocessed data if needed for subsequent steps
        if (run_compute_pseudobulk or run_process_pca) and not status_flags["multiomics"]["integration_preprocessing"]:
            temp_preprocessed_path = f"{integrate_output_dir}/adata_preprocessed.h5ad"
            if os.path.exists(temp_preprocessed_path):
                results['adata'] = sc.read(temp_preprocessed_path)
                status_flags["multiomics"]["integration_preprocessing"] = True
                if multiomics_verbose:
                    print(f"Loaded preprocessed data from: {temp_preprocessed_path}")
            else:
                raise ValueError("Integration preprocessing is required for subsequent steps. "
                               "Either set run_integrate_preprocess=True or ensure preprocessed data exists.")
    
    # Step 3: Compute pseudobulk
    if run_compute_pseudobulk:
        if multiomics_verbose:
            print("Step 3: Computing pseudobulk...")
        
        if not status_flags["multiomics"]["integration_preprocessing"]:
            raise ValueError("Integration preprocessing is required before pseudobulk computation.")
        
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
            verbose=multiomics_verbose
        )
        
        results['atac_pseudobulk_df'] = atac_pseudobulk_df
        results['pseudobulk_adata'] = pseudobulk_adata
        status_flags["multiomics"]["pseudobulk_computation"] = True
        
        if multiomics_verbose:
            print("✓ Pseudobulk computation completed successfully")
    else:
        # Load pseudobulk data if needed for subsequent steps
        if run_process_pca and not status_flags["multiomics"]["pseudobulk_computation"]:
            temp_pseudobulk_df_path = f"{pseudobulk_output_dir}/pseudobulk_df.csv"
            temp_pseudobulk_adata_path = f"{pseudobulk_output_dir}/pseudobulk_adata.h5ad"
            
            if os.path.exists(temp_pseudobulk_df_path) and os.path.exists(temp_pseudobulk_adata_path):
                import pandas as pd
                results['atac_pseudobulk_df'] = pd.read_csv(temp_pseudobulk_df_path, index_col=0)
                results['pseudobulk_adata'] = sc.read(temp_pseudobulk_adata_path)
                status_flags["multiomics"]["pseudobulk_computation"] = True
                if multiomics_verbose:
                    print("Loaded pseudobulk data from existing files")
            else:
                raise ValueError("Pseudobulk computation is required for PCA processing. "
                               "Either set run_compute_pseudobulk=True or ensure pseudobulk data exists.")
    
    # Step 4: Process with PCA
    if run_process_pca:
        if multiomics_verbose:
            print("Step 4: Processing with PCA...")
        
        if not status_flags["multiomics"]["pseudobulk_computation"]:
            raise ValueError("Pseudobulk computation is required before PCA processing.")
        
        adata_for_pca = results.get('adata')
        pseudobulk_for_pca = results.get('atac_pseudobulk_df')
        pseudobulk_adata_for_pca = results.get('pseudobulk_adata')
        
        if not all([adata_for_pca is not None, pseudobulk_for_pca is not None, pseudobulk_adata_for_pca is not None]):
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
            verbose=multiomics_verbose
        )
        
        results['pseudobulk_anndata_processed'] = pseudobulk_anndata_processed
        status_flags["multiomics"]["pca_processing"] = True
        
        if multiomics_verbose:
            print("✓ PCA processing completed successfully")
    
    # Alternative: Load pseudobulk from file
    if pseudobulk_h5ad_path and not run_process_pca:
        if multiomics_verbose:
            print(f"Loading pseudobulk data from: {pseudobulk_h5ad_path}")
        results['pseudobulk_anndata_processed'] = sc.read_h5ad(pseudobulk_h5ad_path)
        status_flags["multiomics"]["pca_processing"] = True
    elif (run_visualize_embedding or run_find_optimal_resolution) and not status_flags["multiomics"]["pca_processing"]:
        # Try to load from default location
        temp_pca_path = f"{pca_output_dir}/pseudobulk_sample.h5ad"
        if os.path.exists(temp_pca_path):
            results['pseudobulk_anndata_processed'] = sc.read(temp_pca_path)
            status_flags["multiomics"]["pca_processing"] = True
            if multiomics_verbose:
                print(f"Loaded PCA-processed data from: {temp_pca_path}")
        else:
            raise ValueError("PCA processing is required for subsequent steps. "
                           "Either set run_process_pca=True or provide pseudobulk_h5ad_path.")
    
    # Step 5: Visualize multimodal embedding
    if run_visualize_embedding:
        if multiomics_verbose:
            print("Step 5: Visualizing multimodal embedding...")
        
        if not status_flags["multiomics"]["pca_processing"]:
            raise ValueError("PCA processing is required before embedding visualization.")
        
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
            verbose=multiomics_verbose
        )
        
        results['visualization'] = {'fig': fig, 'axes': axes}
        status_flags["multiomics"]["embedding_visualization"] = True
        
        if multiomics_verbose:
            print("✓ Embedding visualization completed successfully")
    
    # Step 6: Find optimal resolution (optional)
    if run_find_optimal_resolution:
        if multiomics_verbose:
            print("Step 6: Finding optimal cell resolution...")
        
        # Setup GPU if available
        try:
            import rmm
            from rmm.allocators.cupy import rmm_cupy_allocator
            import cupy as cp
            rmm.reinitialize(managed_memory=True, pool_allocator=False)
            cp.cuda.set_allocator(rmm_cupy_allocator)
        except:
            if multiomics_verbose:
                print("GPU setup failed, continuing with CPU")
        
        import torch
        if torch.cuda.is_available() and multiomics_verbose:
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"GPU Memory Available: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        # Get integrated data for resolution optimization
        integrated_adata_for_resolution = results.get('adata')
        if integrated_adata_for_resolution is None and integrated_h5ad_path:
            integrated_adata_for_resolution = sc.read_h5ad(integrated_h5ad_path)
        elif integrated_adata_for_resolution is None:
            # Try to load from default location
            temp_integrated_path = f"{integrate_output_dir}/adata_preprocessed.h5ad"
            if os.path.exists(temp_integrated_path):
                integrated_adata_for_resolution = sc.read(temp_integrated_path)
            
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
            verbose=multiomics_verbose
        )
        
        results['optimal_resolution'] = optimal_res
        results['resolution_results_df'] = results_df
        status_flags["multiomics"]["optimal_resolution"] = True
        
        if multiomics_verbose:
            print("✓ Optimal resolution finding completed successfully")
    
    # Add status_flags to results
    results['status_flags'] = status_flags
    
    if multiomics_verbose:
        print("\nPipeline completed successfully!")
        print(f"Results saved to: {multiomics_output_dir}")
        print(f"Available results: {list(results.keys())}")
        print(f"Final status flags: {status_flags['multiomics']}")
        print(f"Completed steps: {sum(status_flags['multiomics'].values())}/{len(status_flags['multiomics'])}")
    
    return results