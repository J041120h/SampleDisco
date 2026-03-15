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
from sample_embedding.DR import dimension_reduction
from sample_embedding.multi_omics_pseudobulk_cpu import compute_pseudobulk_adata_cpu
from preparation.multi_omics_glue import multiomics_preparation
from preparation.multi_omics_preprocess import integrate_preprocess
from sample_trajectory.multi_omics_CCA_test import integration_CCA_test
from parameter_selection.multi_omics_optimal_resolution import find_optimal_cell_resolution_integration, suppress_warnings
from visualization.multi_omics_visualization import visualize_multimodal_embedding
from preparation.multi_omics_cell_type_cpu import cell_types_multiomics
from utils.multi_omics_unify_optimal import replace_optimal_dimension_reduction

def multiomics_wrapper(
    # ===== Required Parameters =====
    rna_file=None,
    atac_file=None,
    multiomics_output_dir=None,
    
    # ===== Process Control Flags =====
    integration=True,
    integration_preprocessing=True,
    dimensionality_reduction=True,
    visualize_embedding=True,
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
    
    # ===== Visualization Parameters =====
    color_col=None,
    visualization_grouping_column=None,
    target_modality='ATAC',
    expression_key='X_DR_expression',
    proportion_key='X_DR_proportion',
    figsize=(20, 8),
    point_size=60,
    alpha=0.8,
    colormap='viridis',
    show_sample_names=False,
    force_data_type=None,
    
    # ===== Optimal Resolution Parameters =====
    optimization_target="rna",
    sev_col="sev.level",
    resolution_use_rep='X_glue',
    num_PCs=20,
    visualize_cell_types=True,
    
    # ===== Paths for Skipping Steps =====
    integrated_h5ad_path=None,
    pseudobulk_h5ad_path=None,
    
    # ===== System Parameters =====
    status_flags=None,
) -> Dict[str, Any]:
    """
    Comprehensive wrapper for multi-modal single-cell analysis pipeline.
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
        "dimensionality_reduction": False,
        "embedding_visualization": False,
        "optimal_resolution": False
    }
    
    if status_flags is None:
        status_flags = {"multiomics": default_status.copy()}
    elif "multiomics" not in status_flags:
        status_flags["multiomics"] = default_status.copy()
    
    results = {}
    Path(multiomics_output_dir).mkdir(parents=True, exist_ok=True)
    
    if multiomics_verbose:
        print(f"Starting multi-modal pipeline with output directory: {multiomics_output_dir}")
    
    # Determine integrated h5ad path
    h5ad_path = integrated_h5ad_path if integrated_h5ad_path and os.path.exists(integrated_h5ad_path) else f"{multiomics_output_dir}/preprocess/atac_rna_integrated.h5ad"
    
    # Step 1: GLUE integration
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

    # Step 1b: Cell type assignment
    if cell_type_cluster:
        if multiomics_verbose:
            print("Step 1b: Running cell type assignment...")
        cell_types_func = cell_types_multiomics
        if use_gpu:
            from preparation.multi_omics_cell_type_gpu import cell_types_multiomics_linux
            cell_types_func = cell_types_multiomics_linux
        
        merged_adata = results.get('glue')
        if merged_adata is None:
            if os.path.exists(h5ad_path):
                merged_adata = ad.read_h5ad(h5ad_path)
            else:
                raise ValueError(f"Integrated data not found at {h5ad_path}. Run GLUE gene activity first.")
        
        merged_adata = cell_types_func(
            adata=merged_adata,
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
        
        results['glue'] = merged_adata
        status_flags["multiomics"]["glue_cell_types"] = True
        
        if multiomics_verbose:
            print("Cell type assignment completed successfully")
    
    # Step 2: Integration preprocessing
    if integration_preprocessing:
        if multiomics_verbose:
            print("Step 2: Running integration preprocessing...")
        
        if not status_flags["multiomics"]["glue_integration"] and not os.path.exists(h5ad_path):
            raise ValueError("GLUE integration is required before integration preprocessing.")
        
        adata = integrate_preprocess(
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

        results['adata'] = adata
        status_flags["multiomics"]["integration_preprocessing"] = True

        if multiomics_verbose:
            print("Integration preprocessing completed successfully")
    else:
        preprocessed_path = f"{multiomics_output_dir}/preprocess/adata_sample.h5ad"
        if os.path.exists(preprocessed_path):
            results['adata'] = sc.read(preprocessed_path)
            status_flags["multiomics"]["integration_preprocessing"] = True
            if multiomics_verbose:
                print(f"Loaded preprocessed data from: {preprocessed_path}")
        else:
            raise ValueError("Integration preprocessing is required. Set integration_preprocessing=True or ensure preprocessed data exists.")

    # Step 3: Dimensionality Reduction
    if dimensionality_reduction:
        if multiomics_verbose:
            print("Step 3: Running dimensionality reduction...")
        
        if not status_flags["multiomics"]["integration_preprocessing"]:
            raise ValueError("Integration preprocessing is required before dimensionality reduction.")
        
        adata_for_dr = results.get('adata')
        if adata_for_dr is None:
            raise ValueError("adata must be available from integration preprocessing")
        
        # Ensure modality is included in batch columns
        batch_cols = [batch_col] if isinstance(batch_col, str) else list(batch_col or [])
        if modality_col not in batch_cols:
            batch_cols.append(modality_col)

        compute_pseudobulk_func = compute_pseudobulk_adata_cpu
        if use_gpu:
            from sample_embedding.multi_omics_pseudobulk_gpu import compute_pseudobulk_adata_linux
            compute_pseudobulk_func = compute_pseudobulk_adata_linux

        pseudobulk_df, pseudobulk_adata = compute_pseudobulk_func(
            adata=adata_for_dr,
            batch_col=batch_cols,
            sample_col=sample_col,
            celltype_col=celltype_col,
            output_dir=multiomics_output_dir,
            save=True,
            sample_hvg_number=sample_hvg_number,
            atac=False,
            verbose=multiomics_verbose,
            preserve_cols=preserve_cols_for_sample_embedding,
        )

        pseudobulk_adata = dimension_reduction(
            adata=adata_for_dr,
            pseudobulk=pseudobulk_df,
            pseudobulk_anndata=pseudobulk_adata,
            sample_col=sample_col,
            n_expression_components=n_expression_components,
            n_proportion_components=n_proportion_components,
            batch_col=batch_cols,
            harmony_for_proportion=multiomics_harmony_for_proportion,
            preserve_cols_in_sample_embedding=preserve_cols_for_sample_embedding,
            output_dir=multiomics_output_dir,
            not_save=False,
            atac=False,
            verbose=multiomics_verbose,
        )

        results['pseudobulk_df'] = pseudobulk_df
        results['pseudobulk_adata'] = pseudobulk_adata
        status_flags["multiomics"]["dimensionality_reduction"] = True
        
        if multiomics_verbose:
            print("Dimensionality reduction completed successfully")
    else:
        if pseudobulk_h5ad_path and os.path.exists(pseudobulk_h5ad_path):
            results['pseudobulk_adata'] = sc.read_h5ad(pseudobulk_h5ad_path)
            status_flags["multiomics"]["dimensionality_reduction"] = True
            if multiomics_verbose:
                print(f"Loaded dimensionality reduction data from: {pseudobulk_h5ad_path}")
        else:
            pseudobulk_df_path = os.path.join(multiomics_output_dir, "pseudobulk", "expression_hvg.csv")
            pseudobulk_adata_path = os.path.join(multiomics_output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
            
            missing_files = [p for p in [pseudobulk_df_path, pseudobulk_adata_path] if not os.path.exists(p)]
            
            if missing_files:
                raise FileNotFoundError(
                    "Dimensionality reduction is required. Missing file(s):\n" + 
                    "\n".join(f" - {path}" for path in missing_files)
                )
            
            results['pseudobulk_df'] = pd.read_csv(pseudobulk_df_path, index_col=0)
            results['pseudobulk_adata'] = sc.read(pseudobulk_adata_path)
            status_flags["multiomics"]["dimensionality_reduction"] = True
            if multiomics_verbose:
                print("Loaded dimensionality reduction data from existing files")
                
    # Step 4: Find optimal resolution
    if find_optimal_resolution:
        if multiomics_verbose:
            print("Step 4: Finding optimal cell resolution...")
        
        integrated_adata = results.get('adata')
        if integrated_adata is None:
            preprocessed_path = f"{multiomics_output_dir}/preprocess/adata_sample.h5ad"
            if os.path.exists(preprocessed_path):
                integrated_adata = sc.read_h5ad(preprocessed_path)
            else:
                raise ValueError("Integrated AnnData must be available for optimal resolution finding")
        
        suppress_warnings()
        resolution_output_dir = f"{multiomics_output_dir}/resolution_optimization"
        
        for dr_type, num_components in [("expression", n_expression_components), ("proportion", n_proportion_components)]:
            if multiomics_verbose:
                print(f"\n  Running optimization for {dr_type.upper()}...")
            
            find_optimal_cell_resolution_integration(
                AnnData_integrated=integrated_adata,
                output_dir=f"{resolution_output_dir}_{dr_type}",
                optimization_target=optimization_target,
                dr_type=dr_type,
                n_features=sample_hvg_number,
                sev_col=sev_col,
                batch_col=batch_col,
                sample_col=sample_col,
                modality_col=modality_col,
                use_rep=resolution_use_rep,
                num_PCs=num_PCs,
                visualize_cell_types=visualize_cell_types,
                verbose=multiomics_verbose
            )
        
        status_flags["multiomics"]["optimal_resolution"] = True
        
        if multiomics_verbose:
            print("\n  Updating pseudobulk with optimal embeddings...")
        
        results['pseudobulk_adata'] = replace_optimal_dimension_reduction(
            base_path=multiomics_output_dir,
            expression_resolution_dir=f"{resolution_output_dir}_expression",
            proportion_resolution_dir=f"{resolution_output_dir}_proportion",
            verbose=multiomics_verbose
        )
        
        if multiomics_verbose:
            print("Optimal resolution finding completed successfully")
            
    # Step 5: Visualize embedding
    if visualize_embedding:
        if multiomics_verbose:
            print("Step 5: Visualizing multimodal embedding...")
        
        if not status_flags["multiomics"]["dimensionality_reduction"]:
            raise ValueError("Dimensionality reduction is required before embedding visualization.")
        
        fig, axes = visualize_multimodal_embedding(
            adata=results.get('pseudobulk_adata'),
            modality_col=modality_col,
            color_col=color_col,
            visualization_grouping_column=visualization_grouping_column,
            target_modality=target_modality,
            expression_key=expression_key,
            proportion_key=proportion_key,
            figsize=figsize,
            point_size=point_size,
            alpha=alpha,
            colormap=colormap,
            output_dir=multiomics_output_dir,
            show_sample_names=show_sample_names,
            force_data_type=force_data_type,
            verbose=multiomics_verbose,
        )
        
        results['visualization'] = {'fig': fig, 'axes': axes}
        status_flags["multiomics"]["embedding_visualization"] = True
        
        if multiomics_verbose:
            print("Embedding visualization completed successfully")
    
    results['status_flags'] = status_flags
    
    if multiomics_verbose:
        print("\nPipeline completed successfully!")
        print(f"Results saved to: {multiomics_output_dir}")
        completed_steps = sum(status_flags['multiomics'].values())
        print(f"Completed steps: {completed_steps}/{len(status_flags['multiomics'])}")
        
    return results