import anndata as ad
import networkx as nx
import scanpy as sc
import sys
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DR import dimension_reduction
from integration.integration_pseudobulk import compute_pseudobulk_adata_linux
from integration.integration_glue import *
from integration.integration_preprocess import *
from integration.integration_CCA_test import *
from integration.integration_optimal_resolution import *
from integration.integration_validation import *
from integration.integration_visualization import *
from integration.integration_cell_type import cell_types_multiomics
from utils.multi_omics_unify_optimal import replace_optimal_dimension_reduction

def multiomics_wrapper(
    # ===== Required Parameters =====
    rna_file=None,
    atac_file=None,
    multiomics_output_dir=None,
    
    # ===== Process Control Flags =====
    run_glue=True,
    run_integrate_preprocess=True,
    run_dimensionality_reduction=True,
    run_visualize_embedding=True,
    run_find_optimal_resolution=False,

    # ===== Basic Parameters =====
    rna_sample_meta_file=None,
    atac_sample_meta_file=None,
    additional_hvg_file = None,
    rna_sample_column="sample",
    atac_sample_column="sample",
    sample_column='sample',
    sample_col='sample',
    batch_col= None,
    celltype_col='cell_type',
    multiomics_verbose=True,
    save_intermediate=True,
    large_data_need_extra_memory=False,
    use_gpu=True,
    random_state=42,
    
    # ===== GLUE Integration Parameters =====
    # Process control flags for GLUE sub-steps
    run_glue_preprocessing=True,
    run_glue_training=True,
    run_glue_gene_activity=True,
    run_glue_cell_types=True,
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
    metadata_sep=",",
    
    # GLUE training parameters
    consistency_threshold=0.05,
    treat_sample_as_batch= True,
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
    method='average',
    metric_celltype='euclidean',
    distance_mode='centroid',
    generate_umap_celltype=True,
    
    # GLUE visualization parameters
    plot_columns=None,
    
    # ===== Integration Preprocessing Parameters =====
    integrate_output_dir=None,
    min_cells_sample=1,
    min_cell_gene=10,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
    
    # ===== Combined Dimensionality Reduction Parameters =====
    # Pseudobulk Parameters
    pseudobulk_output_dir=None,
    Save=True,
    n_features=2000,
    normalize=True,
    target_sum=1e4,
    atac=False,
    preserve_cols_for_sample_embedding = None,
    
    # PCA Parameters
    pca_sample_col='sample',
    n_expression_pcs=10,
    n_proportion_pcs=10,
    multiomics_harmony_for_proportion = True,
    pca_output_dir=None,
    integrated_data=False,
    not_save=False,
    pca_atac=False,
    use_snapatac2_dimred=False,
    
    # ===== Visualization Parameters =====
    modality_col='modality',
    color_col= None,
    visualization_grouping_column = None,
    target_modality='ATAC',
    expression_key='X_DR_expression',
    proportion_key='X_DR_proportion',
    figsize=(20, 8),
    point_size=60,
    alpha=0.8,
    colormap='viridis',
    viz_output_dir=None,
    show_sample_names=False,
    force_data_type=None,
    
    # ===== Optimal Resolution Parameters =====
    optimization_target="rna",
    resolution_n_features=40000,
    sev_col="sev.level",
    resolution_batch_col=None,
    resolution_sample_col="sample",
    resolution_modality_col="modality",
    resolution_use_rep='X_glue',
    num_DR_components=30,
    num_PCs=20,
    num_pvalue_simulations=1000,
    n_pcs=10,
    compute_pvalues=True,
    visualize_embeddings=True,
    resolution_output_dir=None,
    
    # ===== Paths for Skipping Steps =====
    integrated_h5ad_path=None,
    pseudobulk_h5ad_path=None,
    
    # ===== System Parameters =====
    status_flags=None,
) -> Dict[str, Any]:
    """
    Comprehensive wrapper for multi-modal single-cell analysis pipeline with combined
    dimensionality reduction step (pseudobulk + PCA) to match RNA wrapper structure.
    
    UPDATED: When run_find_optimal_resolution=True, automatically runs optimization for
    both expression and proportion, then updates the pseudobulk embeddings with optimal results.
    
    Parameters:
    -----------
    
    PIPELINE CONTROL:
    run_glue : bool, default True
        Whether to run the glue function
    run_integrate_preprocess : bool, default True
        Whether to run the integrate_preprocess function
    run_dimensionality_reduction : bool, default True
        Whether to run the combined pseudobulk computation and PCA processing
    run_visualize_embedding : bool, default True
        Whether to run the visualize_multimodal_embedding function
    run_find_optimal_resolution : bool, default False
        Whether to run the find_optimal_cell_resolution_integration function
        NOTE: Now automatically runs both expression AND proportion optimizations
    
    GLUE SUB-STEP CONTROL:
    run_glue_preprocessing : bool, default True
        Whether to run GLUE preprocessing step
    run_glue_training : bool, default True
        Whether to run GLUE training step
    run_glue_gene_activity : bool, default True
        Whether to run GLUE gene activity computation step
    run_glue_cell_types : bool, default True
        Whether to run GLUE cell type assignment step
    run_glue_visualization : bool, default True
        Whether to run GLUE visualization step
    
    STATUS FLAGS:
    status_flags : Dict[str, Any], optional
        Dictionary to track completion status of each pipeline step.
        If None, will be initialized with default values.
        Structure: {
            "multiomics": {
                "glue_integration": False,
                "glue_preprocessing": False,
                "glue_training": False,
                "glue_gene_activity": False,
                "glue_cell_types": False,
                "glue_visualization": False,
                "integration_preprocessing": False,
                "dimensionality_reduction": False,  # Combined pseudobulk + PCA
                "embedding_visualization": False,
                "optimal_resolution": False
            }
        }
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - Results from each executed step
        - Updated status_flags tracking completion
        - All intermediate data objects
        - When optimal resolution is run: both expression and proportion results
    """
    
    if any(var is None for var in [rna_file, atac_file, multiomics_output_dir]):
        raise ValueError("All parameters must be provided (none can be None)")
    
    # Initialize status flags if not provided
    if status_flags is None:
        status_flags = {
            "multiomics": {
                "glue_integration": False,
                "glue_preprocessing": False,
                "glue_training": False,
                "glue_gene_activity": False,
                "glue_cell_types": False,
                "glue_visualization": False,
                "integration_preprocessing": False,
                "dimensionality_reduction": False,  # Combined step
                "embedding_visualization": False,
                "optimal_resolution": False
            }
        }
    
    # Ensure multiomics section exists in status_flags
    if "multiomics" not in status_flags:
        status_flags["multiomics"] = {
            "glue_integration": False,
            "glue_preprocessing": False,
            "glue_training": False,
            "glue_gene_activity": False,
            "glue_cell_types": False,
            "glue_visualization": False,
            "integration_preprocessing": False,
            "dimensionality_reduction": False,  # Combined step
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
        integrate_output_dir = multiomics_output_dir
    if pseudobulk_output_dir is None:
        pseudobulk_output_dir = multiomics_output_dir
    if pca_output_dir is None:
        pca_output_dir = multiomics_output_dir
    if viz_output_dir is None:
        viz_output_dir = multiomics_output_dir
    if resolution_output_dir is None:
        resolution_output_dir = f"{multiomics_output_dir}/resolution_optimization"
    
    # Initialize h5ad_path for downstream use
    h5ad_path = None
    
    # Step 1: GLUE integration with sub-step control
    if run_glue:
        if multiomics_verbose:
            print("Step 1: Running GLUE integration...")
            print(f"  Sub-steps: Preprocessing={run_glue_preprocessing}, Training={run_glue_training}, "
                f"Gene Activity={run_glue_gene_activity}, "
                f"Visualization={run_glue_visualization}")
        if not rna_file or not atac_file:
            raise ValueError("rna_file and atac_file must be provided when run_glue=True")
        glue_result = glue(
            # Data files
            rna_file=rna_file,
            atac_file=atac_file,
            rna_sample_meta_file=rna_sample_meta_file,
            atac_sample_meta_file=atac_sample_meta_file,
            additional_hvg_file=additional_hvg_file,
            # Process control flags for GLUE sub-steps
            run_preprocessing=run_glue_preprocessing,
            run_training=run_glue_training,
            run_gene_activity=run_glue_gene_activity,
            run_visualization=run_glue_visualization,
            # Preprocessing parameters
            ensembl_release=ensembl_release,
            species=species,
            use_highly_variable=use_highly_variable,
            n_top_genes=n_top_genes,
            n_pca_comps=n_pca_comps,
            n_lsi_comps=n_lsi_comps,
            gtf_by=gtf_by,
            flavor=flavor,
            generate_umap=generate_umap,
            compression=compression,
            random_state=random_state,
            rna_sample_column=rna_sample_column,
            atac_sample_column=atac_sample_column,
            # Training parameters
            consistency_threshold=consistency_threshold,
            treat_sample_as_batch=treat_sample_as_batch,
            save_prefix=save_prefix,
            # Gene activity computation parameters
            k_neighbors=k_neighbors,
            use_rep=use_rep,
            metric=metric,
            use_gpu=use_gpu,
            verbose=multiomics_verbose,
            # Visualization parameters
            plot_columns=plot_columns,
            # Output directory
            output_dir=multiomics_output_dir
        )
            
        results['glue'] = glue_result
        
        # Update status flags for completed GLUE sub-steps
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
            print("✓ GLUE integration completed successfully")
            completed_substeps = sum([
                run_glue_preprocessing, run_glue_training, run_glue_gene_activity,
                run_glue_visualization
            ])
            print(f"  Completed {completed_substeps}/4 GLUE sub-steps")

    if integrated_h5ad_path and os.path.exists(integrated_h5ad_path):
        h5ad_path = integrated_h5ad_path
        if multiomics_verbose:
            print(f"Skipping GLUE integration, using existing data: {h5ad_path}")
    else:
        h5ad_path = f"{multiomics_output_dir}/preprocess/atac_rna_integrated.h5ad"
    
    # Step 1b: Cell type assignment (separate from GLUE)
    if run_glue_cell_types:
        if multiomics_verbose:
            print("Step 1b: Running cell type assignment...")
        
        # Load integrated data if not already in memory
        if results.get('glue') is not None:
            merged_adata = results['glue']
        elif os.path.exists(h5ad_path):
            merged_adata = ad.read_h5ad(h5ad_path)
        else:
            raise ValueError(f"Integrated data not found at {h5ad_path}. Run GLUE gene activity first.")
        
        # Run cell type assignment
        merged_adata = cell_types_multiomics(
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
            print("✓ Cell type assignment completed successfully")
    
    # Step 2: Integration preprocessing
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
            modality_col=modality_col,
            min_cells_sample=min_cells_sample,
            min_cell_gene=min_cell_gene,
            min_features=min_features,
            pct_mito_cutoff=pct_mito_cutoff,
            exclude_genes=exclude_genes,
            doublet=doublet,
            verbose=multiomics_verbose,
            # NEW: re-merge sample metadata for RNA & ATAC
            rna_sample_meta_file=rna_sample_meta_file,
            atac_sample_meta_file=atac_sample_meta_file,
        )

        results['adata'] = adata
        status_flags["multiomics"]["integration_preprocessing"] = True

        if multiomics_verbose:
            print("✓ Integration preprocessing completed successfully")
    else:
        temp_preprocessed_path = f"{integrate_output_dir}/preprocess/adata_sample.h5ad"
        if os.path.exists(temp_preprocessed_path):
            results['adata'] = sc.read(temp_preprocessed_path)
            status_flags["multiomics"]["integration_preprocessing"] = True
            if multiomics_verbose:
                print(f"Loaded preprocessed data from: {temp_preprocessed_path}")
        else:
            raise ValueError("Integration preprocessing is required for subsequent steps. "
                            "Either set run_integrate_preprocess=True or ensure preprocessed data exists.")

    
    # Step 3: Combined Dimensionality Reduction (Pseudobulk + PCA)
    if run_dimensionality_reduction:
        if multiomics_verbose:
            print("Step 3: Running dimensionality reduction (pseudobulk + PCA)...")
        
        if not status_flags["multiomics"]["integration_preprocessing"]:
            raise ValueError("Integration preprocessing is required before dimensionality reduction.")
        
        adata_for_dr = results.get('adata')
        if adata_for_dr is None:
            raise ValueError("adata must be available from integration preprocessing when run_dimensionality_reduction=True")
        
        # Sub-step 3a: Compute pseudobulk
        if multiomics_verbose:
            print("  Sub-step 3a: Computing pseudobulk...")
            
        # Normalize batch_col into a list and ensure modality is included
        if batch_col is None:
            batch_cols = [resolution_modality_col]
        elif isinstance(batch_col, str):
            batch_cols = [batch_col]
        else:
            batch_cols = list(batch_col)

        if resolution_modality_col not in batch_cols:
            batch_cols.append(resolution_modality_col)

        atac_pseudobulk_df, pseudobulk_adata = compute_pseudobulk_adata_linux(
            adata=adata_for_dr,
            batch_col=batch_cols,
            sample_col=sample_col,
            celltype_col=celltype_col,
            output_dir=pseudobulk_output_dir,
            save=Save,
            n_features=n_features,
            normalize=normalize,
            target_sum=target_sum,
            atac=atac,
            verbose=multiomics_verbose,
            preserve_cols=preserve_cols_for_sample_embedding,
        )

        
        # Sub-step 3b: Process with PCA
        if multiomics_verbose:
            print("  Sub-step 3b: Processing with PCA...")
        
        pseudobulk_anndata_processed = dimension_reduction(
            adata=adata_for_dr,
            pseudobulk=atac_pseudobulk_df,
            pseudobulk_anndata=pseudobulk_adata,
            sample_col=pca_sample_col,
            n_expression_components=n_expression_pcs,
            n_proportion_components=n_proportion_pcs,
            batch_col=batch_cols,
            harmony_for_proportion=multiomics_harmony_for_proportion,
            output_dir=pca_output_dir,
            verbose=multiomics_verbose,
            preserve_cols=preserve_cols_for_sample_embedding,
        )

        # Store results
        results['atac_pseudobulk_df'] = atac_pseudobulk_df
        results['pseudobulk_adata'] = pseudobulk_adata
        results['pseudobulk_anndata_processed'] = pseudobulk_anndata_processed
        status_flags["multiomics"]["dimensionality_reduction"] = True
        
        if multiomics_verbose:
            print("✓ Dimensionality reduction completed successfully")
    else:
        if pseudobulk_h5ad_path and os.path.exists(pseudobulk_h5ad_path):
            results['pseudobulk_anndata_processed'] = sc.read_h5ad(pseudobulk_h5ad_path)
            status_flags["multiomics"]["dimensionality_reduction"] = True
            if multiomics_verbose:
                print(f"Loaded dimensionality reduction data from: {pseudobulk_h5ad_path}")
        else:
            # Try to load from default locations
            temp_pseudobulk_df_path = os.path.join(pseudobulk_output_dir, "pseudobulk", "expression_hvg.csv")
            temp_pseudobulk_adata_path = os.path.join(pseudobulk_output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
            
            missing_files = []
            if not os.path.exists(temp_pseudobulk_df_path):
                missing_files.append(temp_pseudobulk_df_path)
            if not os.path.exists(temp_pseudobulk_adata_path):
                missing_files.append(temp_pseudobulk_adata_path)
            
            if not missing_files:
                status_flags["multiomics"]["dimensionality_reduction"] = True
                results['atac_pseudobulk_df'] = pd.read_csv(temp_pseudobulk_df_path, index_col=0)
                results['pseudobulk_adata'] = sc.read(temp_pseudobulk_adata_path)
                status_flags["multiomics"]["dimensionality_reduction"] = True
                if multiomics_verbose:
                    print("Loaded dimensionality reduction data from existing files")
            else:
                raise FileNotFoundError(
                    "Dimensionality reduction is required for subsequent steps. "
                    "Missing file(s):\n" + "\n".join(f" - {path}" for path in missing_files) +
                    "\nEither set run_dimensionality_reduction=True or ensure required data exists."
                )
    
    # Step 4: Visualize multimodal embedding
    if run_visualize_embedding:
        if multiomics_verbose:
            print("Step 4: Visualizing multimodal embedding...")
        
        if not status_flags["multiomics"]["dimensionality_reduction"]:
            raise ValueError("Dimensionality reduction is required before embedding visualization.")
        
        adata_for_viz = results.get('pseudobulk_adata')
        
        fig, axes = visualize_multimodal_embedding(
            adata=adata_for_viz,
            modality_col=modality_col,
            color_col=color_col,
            visualization_grouping_column = visualization_grouping_column,
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
            verbose=multiomics_verbose,
        )
        
        results['visualization'] = {'fig': fig, 'axes': axes}
        status_flags["multiomics"]["embedding_visualization"] = True
        
        if multiomics_verbose:
            print("✓ Embedding visualization completed successfully")
    
    # Step 5: Find optimal resolution (optional) - runs both expression and proportion
    if run_find_optimal_resolution:
        if multiomics_verbose:
            print("Step 5: Finding optimal cell resolution for both expression and proportion...")
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
        
        # Run optimization for expression
        if multiomics_verbose:
            print("\n  Running optimization for EXPRESSION...")
        expression_resolution_dir = f"{resolution_output_dir}_expression"
        optimal_res_expression, results_df_expression = find_optimal_cell_resolution_integration(
            AnnData_integrated=integrated_adata_for_resolution,
            output_dir=expression_resolution_dir,
            optimization_target=optimization_target,
            dr_type="expression",
            n_features=resolution_n_features,
            sev_col=sev_col,
            batch_col=batch_col,
            sample_col=resolution_sample_col,
            modality_col=resolution_modality_col,
            use_rep=resolution_use_rep,
            num_DR_components=num_DR_components,
            num_PCs=num_PCs,
            n_pcs=n_pcs,
            visualize_cell_types=True,
            verbose=multiomics_verbose
        )
        
        if multiomics_verbose:
            print("  ✓ Expression optimization completed")
        
        # Run optimization for proportion
        if multiomics_verbose:
            print("\n  Running optimization for PROPORTION...")
        proportion_resolution_dir = f"{resolution_output_dir}_proportion"
        optimal_res_proportion, results_df_proportion = find_optimal_cell_resolution_integration(
            AnnData_integrated=integrated_adata_for_resolution,
            output_dir=proportion_resolution_dir,
            optimization_target=optimization_target,
            dr_type="proportion",
            n_features=resolution_n_features,
            sev_col=sev_col,
            batch_col=batch_col,
            sample_col=resolution_sample_col,
            modality_col=resolution_modality_col,
            use_rep=resolution_use_rep,
            num_DR_components=num_DR_components,
            num_PCs=num_PCs,
            n_pcs=n_pcs,
            visualize_cell_types=True,
            verbose=multiomics_verbose
        )
        
        if multiomics_verbose:
            print("  ✓ Proportion optimization completed")
        status_flags["multiomics"]["optimal_resolution"] = True
        
        # Update pseudobulk with optimal embeddings
        if multiomics_verbose:
            print("\n  Updating pseudobulk with optimal embeddings...")
        
        pseudobulk_sample_updated = replace_optimal_dimension_reduction(
            base_path=multiomics_output_dir,
            expression_resolution_dir=expression_resolution_dir,
            proportion_resolution_dir=proportion_resolution_dir,
            verbose=multiomics_verbose
        )
        
        results['pseudobulk_adata'] = pseudobulk_sample_updated
        
        if multiomics_verbose:
            print("✓ Optimal resolution finding and embedding update completed successfully")
    
    # Add status_flags to results
    results['status_flags'] = status_flags
    
    if multiomics_verbose:
        print("\nPipeline completed successfully!")
        print(f"Results saved to: {multiomics_output_dir}")
        print(f"Available results: {list(results.keys())}")
        print(f"Final status flags: {status_flags['multiomics']}")
        completed_steps = sum(status_flags['multiomics'].values())
        total_steps = len(status_flags['multiomics'])
        print(f"Completed steps: {completed_steps}/{total_steps}")
        
        # Show GLUE sub-step completion summary
        glue_substeps = ['glue_preprocessing', 'glue_training', 'glue_gene_activity', 'glue_cell_types', 'glue_visualization']
        completed_glue_substeps = sum(status_flags['multiomics'][step] for step in glue_substeps if step in status_flags['multiomics'])
        print(f"GLUE sub-steps completed: {completed_glue_substeps}/5")
        
    return results