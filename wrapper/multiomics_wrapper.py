import anndata as ad
import networkx as nx
import scanpy as sc
import sys
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_embedding.calculate_multiomics_sample_embedding import calculate_multiomics_sample_embedding
from preparation.multi_omics_glue import multiomics_preparation
from preparation.multi_omics_preprocess import integrate_preprocess
from preparation.multi_omics_cell_type_cpu import cell_types_multiomics
from parameter_selection.multi_omics_unify_optimal import replace_optimal_dimension_reduction


def multiomics_wrapper(
    # ===== Required Parameters =====
    rna_file=None,
    atac_file=None,
    multiomics_output_dir=None,
    
    # ===== Process Control Flags =====
    integration=True,
    integration_preprocessing=True,
    dimensionality_reduction=True,
    find_optimal_resolution=False,

    # ===== Basic Parameters =====
    rna_sample_meta_file=None,
    atac_sample_meta_file=None,
    additional_hvg_file=None,
    rna_sample_column="sample",
    atac_sample_column="sample",
    sample_col='sample',
    batch_col=None,
    celltype_col='cell_type',
    modality_col='modality',
    multiomics_verbose=True,
    save_intermediate=True,
    use_gpu=True,
    random_state=42,
    
    # ===== GLUE Integration Parameters =====
    run_glue_preprocessing=True,
    run_glue_training=True,
    run_glue_gene_activity=True,
    cell_type_cluster=True,
    run_glue_visualization=True,
    
    # GLUE preprocessing parameters
    ensembl_release=98,
    species="homo_sapiens",
    use_highly_variable=True,
    n_top_genes=2000,
    n_pca_comps=50,
    n_lsi_comps=50,
    lsi_n_iter=15,
    gtf_by="gene_name",
    flavor="seurat_v3",
    generate_umap=False,
    compression="gzip",
    
    # GLUE training parameters
    consistency_threshold=0.05,
    treat_sample_as_batch=True,
    save_prefix="glue",
    
    # GLUE gene activity parameters
    k_neighbors=10,
    use_rep="X_glue",
    metric="cosine",
    
    # GLUE cell type parameters
    existing_cell_types=False,
    n_target_clusters=10,
    cluster_resolution=0.8,
    use_rep_celltype="X_glue",
    markers=None,
    generate_umap_celltype=True,
    
    # GLUE visualization parameters
    plot_columns=None,
    
    # ===== Integration Preprocessing Parameters =====
    min_cells_sample=1,
    min_cell_gene=10,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
    
    # ===== Dimensionality Reduction Parameters =====
    sample_hvg_number=2000,
    preserve_cols_for_sample_embedding=None,
    n_expression_components=10,
    n_proportion_components=10,
    multiomics_harmony_for_proportion=True,
    
    # ===== Optimal Resolution Parameters =====
    optimization_target="rna",
    sev_col="sev.level",
    resolution_use_rep='X_glue',
    num_PCs=20,
    visualize_cell_types=True,
    coarse_start=0.1,
    coarse_end=1.0,
    coarse_step=0.1,
    fine_range=0.05,
    fine_step=0.01,
    analyze_modality_alignment=True,
    compute_corrected_pvalues=False,
    
    # ===== Paths for Skipping Steps =====
    integrated_h5ad_path=None,
    pseudobulk_h5ad_path=None,
    
    # ===== System Parameters =====
    status_flags=None,
) -> Dict[str, Any]:
    """
    Multi-omics wrapper: GLUE integration, preprocessing, cell typing,
    optional CCA-based resolution selection, and sample embedding.
    
    Note: find_optimal_resolution runs BEFORE dimensionality_reduction.
    
    Returns dict with adata, adata_sample, pseudo_adata, status_flags.
    Downstream analysis (trajectory, clustering, etc.) is handled by the shared
    downstream_analysis() function in wrapper.py.
    """
    
    if any(var is None for var in [rna_file, atac_file, multiomics_output_dir]):
        raise ValueError("rna_file, atac_file, and multiomics_output_dir must all be provided")
    
    # Initialize status flags
    default_status = {
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
    }
    
    if status_flags is None:
        status_flags = {"multiomics": default_status.copy()}
    elif "multiomics" not in status_flags:
        status_flags["multiomics"] = default_status.copy()
    
    results = {}
    Path(multiomics_output_dir).mkdir(parents=True, exist_ok=True)
    
    if multiomics_verbose:
        print(f"Starting multi-modal pipeline with output directory: {multiomics_output_dir}")
        print(f"GPU mode: {'Enabled' if use_gpu else 'Disabled'}")
    
    h5ad_path = integrated_h5ad_path if integrated_h5ad_path and os.path.exists(integrated_h5ad_path) else f"{multiomics_output_dir}/preprocess/adata_sample.h5ad"
    
    current_adata = None

    # ==================== STEP 1: GLUE INTEGRATION ====================
    if integration:
        if multiomics_verbose:
            print("Step 1: Running GLUE integration...")
        
        glue_result = multiomics_preparation(
            rna_file=rna_file,
            atac_file=atac_file,
            rna_sample_meta_file=rna_sample_meta_file,
            atac_sample_meta_file=atac_sample_meta_file,
            additional_hvg_file=additional_hvg_file,
            run_preprocessing=run_glue_preprocessing,
            run_training=run_glue_training,
            run_gene_activity=run_glue_gene_activity,
            run_visualization=run_glue_visualization,
            ensembl_release=ensembl_release,
            species=species,
            use_highly_variable=use_highly_variable,
            n_top_genes=n_top_genes,
            n_pca_comps=n_pca_comps,
            n_lsi_comps=n_lsi_comps,
            gtf_by=gtf_by,
            flavor=flavor,
            generate_umap=generate_umap,
            rna_sample_column=rna_sample_column,
            atac_sample_column=atac_sample_column,
            consistency_threshold=consistency_threshold,
            treat_sample_as_batch=treat_sample_as_batch,
            save_prefix=save_prefix,
            k_neighbors=k_neighbors,
            use_rep=use_rep,
            metric=metric,
            use_gpu=use_gpu,
            verbose=multiomics_verbose,
            plot_columns=plot_columns,
            output_dir=multiomics_output_dir
        )
            
        results['glue'] = glue_result
        status_flags["multiomics"]["glue_integration"] = True
        
        if run_glue_preprocessing:
            status_flags["multiomics"]["glue_preprocessing"] = True
        if run_glue_training:
            status_flags["multiomics"]["glue_training"] = True
        if run_glue_gene_activity:
            status_flags["multiomics"]["glue_gene_activity"] = True
        if run_glue_visualization:
            status_flags["multiomics"]["glue_visualization"] = True
            
        if multiomics_verbose:
            print("GLUE integration completed successfully")
    
    # ==================== STEP 2: INTEGRATION PREPROCESSING ====================
    if integration_preprocessing:
        if multiomics_verbose:
            print("Step 2: Running integration preprocessing...")
        
        if not status_flags["multiomics"]["glue_integration"] and not os.path.exists(h5ad_path):
            raise ValueError("GLUE integration is required before integration preprocessing.")
        
        current_adata = integrate_preprocess(
            output_dir=multiomics_output_dir,
            h5ad_path=h5ad_path,
            sample_column=sample_col,
            modality_col=modality_col,
            min_cells_sample=min_cells_sample,
            min_cell_gene=min_cell_gene,
            min_features=min_features,
            pct_mito_cutoff=pct_mito_cutoff,
            exclude_genes=exclude_genes,
            doublet=doublet,
            verbose=multiomics_verbose,
            rna_sample_meta_file=rna_sample_meta_file,
            atac_sample_meta_file=atac_sample_meta_file,
        )

        results['adata'] = current_adata
        status_flags["multiomics"]["integration_preprocessing"] = True

        if multiomics_verbose:
            print("Integration preprocessing completed successfully")
    else:
        preprocessed_path = f"{multiomics_output_dir}/preprocess/adata_sample.h5ad"
        if os.path.exists(preprocessed_path):
            current_adata = sc.read(preprocessed_path)
            results['adata'] = current_adata
            status_flags["multiomics"]["integration_preprocessing"] = True
            if multiomics_verbose:
                print(f"Loaded preprocessed data from: {preprocessed_path}")
        else:
            raise ValueError("Integration preprocessing is required. Set integration_preprocessing=True or ensure preprocessed data exists.")
    
    # ==================== STEP 2b: CELL TYPE CLUSTERING ====================
    if cell_type_cluster:
        if multiomics_verbose:
            print("Step 2b: Running cell type assignment...")
        
        if current_adata is None:
            current_adata = ad.read_h5ad(h5ad_path)
        
        cell_types_func = cell_types_multiomics
        if use_gpu:
            from preparation.multi_omics_cell_type_gpu import cell_types_multiomics_linux
            cell_types_func = cell_types_multiomics_linux
        
        current_adata = cell_types_func(
            adata=current_adata,
            modality_column=modality_col,
            rna_modality_value="RNA",
            atac_modality_value="ATAC",
            cell_type_column=celltype_col,
            cluster_resolution=cluster_resolution,
            use_rep=use_rep_celltype,
            k_neighbors=3,
            transfer_metric=metric,
            compute_umap=generate_umap_celltype,
            save=True,
            output_dir=multiomics_output_dir,
            defined_output_path=h5ad_path,
            verbose=multiomics_verbose,
            generate_plots=run_glue_visualization,
        )
        
        results['adata'] = current_adata
        status_flags["multiomics"]["glue_cell_types"] = True
        
        if multiomics_verbose:
            print("Cell type assignment completed successfully")

    # ==================== STEP 3: FIND OPTIMAL RESOLUTION (before sample embedding) ====================
    if find_optimal_resolution:
        if multiomics_verbose:
            print("Step 3: Finding optimal cell resolution...")
        
        if current_adata is None:
            current_adata = ad.read_h5ad(h5ad_path)
            results['adata'] = current_adata

        adata_for_resolution = current_adata.copy()
        
        if use_gpu:
            from parameter_selection.multi_omics_optimal_resolution_gpu import (
                find_optimal_cell_resolution_multiomics_linux as find_resolution_func,
                suppress_warnings
            )
        else:
            from parameter_selection.multi_omics_optimal_resolution_cpu import (
                find_optimal_cell_resolution_multiomics as find_resolution_func,
                suppress_warnings
            )
        
        suppress_warnings()
        
        for dr_type in ["expression", "proportion"]:
            if multiomics_verbose:
                print(f"\n  Running optimization for {dr_type.upper()}...")
            
            resolution_output_dir = f"{multiomics_output_dir}/resolution_optimization_{dr_type}"
            
            find_resolution_func(
                AnnData_integrated=adata_for_resolution.copy(),
                output_dir=resolution_output_dir,
                optimization_target=optimization_target,
                dr_type=dr_type,
                sev_col=sev_col,
                batch_col=batch_col,
                sample_col=sample_col,
                celltype_col=celltype_col,
                modality_col=modality_col,
                use_rep=resolution_use_rep,
                sample_hvg_number=sample_hvg_number,
                n_expression_components=n_expression_components,
                n_proportion_components=n_proportion_components,
                harmony_for_proportion=multiomics_harmony_for_proportion,
                preserve_cols=preserve_cols_for_sample_embedding,
                hvg_modality="RNA",
                visualize_cell_types=visualize_cell_types,
                verbose=multiomics_verbose
            )
        
        status_flags["multiomics"]["optimal_resolution"] = True
        
        if multiomics_verbose:
            print("Optimal resolution finding completed successfully")

    # ==================== STEP 4: DIMENSIONALITY REDUCTION ====================
    pseudobulk_adata = None
    
    if dimensionality_reduction:
        if multiomics_verbose:
            print("Step 4: Running dimensionality reduction...")
        
        if not status_flags["multiomics"]["integration_preprocessing"]:
            raise ValueError("Integration preprocessing is required before dimensionality reduction.")
        
        if current_adata is None:
            current_adata = ad.read_h5ad(h5ad_path)
            results['adata'] = current_adata
        
        batch_cols = [batch_col] if isinstance(batch_col, str) else list(batch_col or [])
        if modality_col not in batch_cols:
            batch_cols.append(modality_col)

        _, pseudobulk_adata = calculate_multiomics_sample_embedding(
            adata=current_adata,
            sample_col=sample_col,
            celltype_col=celltype_col,
            batch_col=batch_cols,
            output_dir=multiomics_output_dir,
            sample_hvg_number=sample_hvg_number,
            n_expression_components=n_expression_components,
            n_proportion_components=n_proportion_components,
            harmony_for_proportion=multiomics_harmony_for_proportion,
            preserve_cols_in_sample_embedding=preserve_cols_for_sample_embedding,
            use_gpu=use_gpu,
            atac=False,
            save=True,
            verbose=multiomics_verbose,
            hvg_modality="RNA",
            modality_col=modality_col,
        )

        status_flags["multiomics"]["dimensionality_reduction"] = True
        
        if multiomics_verbose:
            print("Dimensionality reduction completed successfully")
    else:
        if pseudobulk_h5ad_path and os.path.exists(pseudobulk_h5ad_path):
            pseudobulk_adata = sc.read_h5ad(pseudobulk_h5ad_path)
            status_flags["multiomics"]["dimensionality_reduction"] = True
            if multiomics_verbose:
                print(f"Loaded dimensionality reduction data from: {pseudobulk_h5ad_path}")
        else:
            pseudobulk_adata_path = os.path.join(multiomics_output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
            
            if not os.path.exists(pseudobulk_adata_path):
                raise FileNotFoundError(
                    f"Dimensionality reduction is required. Missing file: {pseudobulk_adata_path}"
                )
            pseudobulk_adata = sc.read(pseudobulk_adata_path)
            status_flags["multiomics"]["dimensionality_reduction"] = True
            if multiomics_verbose:
                print("Loaded dimensionality reduction data from existing files")

    # If optimal resolution was run, replace embeddings with optimal ones
    if find_optimal_resolution:
        pseudobulk_adata = replace_optimal_dimension_reduction(
            base_path=multiomics_output_dir,
            optimization_target=optimization_target,
            verbose=multiomics_verbose
        )

    print("Multiomics preprocessing pipeline completed successfully!")
    
    return {
        'adata': current_adata,
        'adata_sample': current_adata,  # alias for downstream_analysis compatibility
        'pseudo_adata': pseudobulk_adata,
        'status_flags': status_flags
    }