#!/usr/bin/env python3

import os
import sys
from typing import List, Optional

import scanpy as sc
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sample_embedding.calculate_sample_embedding import calculate_sample_embedding
from preparation.rna_preprocess_cpu import preprocess
from preparation.cell_type_cpu import cell_types
from parameter_selection.cpu_optimal_resolution import find_optimal_cell_resolution


def rna_wrapper(
    rna_count_data_path: str = None,
    rna_output_dir: str = None,
    
    # Pipeline control flags
    preprocessing: bool = True,
    cell_type_cluster: bool = True,
    derive_sample_embedding: bool = True,
    cca_based_cell_resolution_selection: bool = False,
    
    # General settings
    use_gpu: bool = False,
    verbose: bool = True,
    status_flags: dict = None,
    
    # Input data paths (for resuming)
    adata_cell_path: str = None,
    adata_sample_path: str = None,
    rna_sample_meta_path: str = None,
    cell_meta_path: str = None,
    pseudo_adata_path: str = None,
    
    # Common column names
    sample_col: str = 'sample',
    sample_level_batch_col: Optional[List[str]] = None,
    celltype_col: str = 'cell_type',
    
    # Preprocessing parameters
    min_cells: int = 500,
    min_genes: int = 500,
    pct_mito_cutoff: float = 20,
    exclude_genes: list = None,
    num_cell_hvgs: int = 2000,
    cell_embedding_num_pcs: int = 20,
    num_harmony_iterations: int = 30,
    cell_level_batch_key: list = None,
    
    # Cell type clustering parameters
    leiden_cluster_resolution: float = 0.8,
    cell_embedding_column: str = None,
    existing_cell_types: bool = False,
    n_target_cell_clusters: int = None,
    umap: bool = False,
    
    # Sample embedding parameters
    sample_hvg_number: int = 2000,
    sample_embedding_dimension: int = 10,
    harmony_for_proportion: bool = True,
    preserve_cols_in_sample_embedding: list = None,
    
    # CCA-based resolution selection parameters
    trajectory_col: str = "sev.level",
    n_cca_pcs: int = 2,
    cca_compute_corrected_pvalues: bool = True,
    cca_coarse_start: float = 0.1,
    cca_coarse_end: float = 1.0,
    cca_coarse_step: float = 0.1,
    cca_fine_range: float = 0.02,
    cca_fine_step: float = 0.01,
    
) -> dict:
    """
    RNA-seq analysis wrapper: preprocessing, cell type clustering, sample embedding,
    and optional CCA-based resolution selection.
    
    Returns dict with adata_cell, adata_sample, pseudo_adata, status_flags.
    Downstream analysis (trajectory, clustering, etc.) is handled by the shared
    downstream_analysis() function in wrapper.py.
    """
    print("Starting RNA wrapper function...")
    
    if rna_count_data_path is None or rna_output_dir is None:
        raise ValueError("Required parameters rna_count_data_path and rna_output_dir must be provided.")
    
    if use_gpu:
        from preparation.rna_preprocess_gpu import preprocess_linux
        from preparation.cell_type_gpu import cell_types_linux
        from parameter_selection.gpu_optimal_resolution import find_optimal_cell_resolution_linux
    
    if cell_level_batch_key is None:
        cell_level_batch_key = []
    if sample_level_batch_col is None:
        sample_level_batch_col = []
    
    # Initialize status flags
    default_status = {
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
    
    if status_flags is None:
        status_flags = {"rna": default_status.copy()}
    elif "rna" not in status_flags:
        status_flags["rna"] = default_status.copy()

    adata_cell = None
    adata_sample = None
    pseudo_adata = None

    # ==================== PREPROCESSING ====================
    if preprocessing:
        print("Starting preprocessing...")
        preprocess_func = preprocess_linux if use_gpu else preprocess
        
        adata_cell, adata_sample = preprocess_func(
            h5ad_path=rna_count_data_path,
            sample_meta_path=rna_sample_meta_path,
            output_dir=rna_output_dir,
            sample_column=sample_col,
            cell_meta_path=cell_meta_path,
            sample_level_batch_key=sample_level_batch_col,
            cell_embedding_num_PCs=cell_embedding_num_pcs,
            num_harmony_iterations=num_harmony_iterations,
            num_cell_hvgs=num_cell_hvgs,
            min_cells=min_cells,
            min_genes=min_genes,
            pct_mito_cutoff=pct_mito_cutoff,
            exclude_genes=exclude_genes,
            cell_level_batch_key=cell_level_batch_key,
            verbose=verbose
        )
        status_flags["rna"]["preprocessing"] = True
    else:
        cell_path = adata_cell_path or os.path.join(rna_output_dir, "preprocess", "adata_cell.h5ad")
        sample_path = adata_sample_path or os.path.join(rna_output_dir, "preprocess", "adata_sample.h5ad")
        
        if not os.path.exists(cell_path) or not os.path.exists(sample_path):
            raise ValueError("Preprocessed data paths not provided and default files do not exist.")
        
        status_flags["rna"]["preprocessing"] = True
        status_flags["rna"]["cell_type_cluster"] = True
        
        # Always load when skipping preprocessing — downstream may need these
        adata_cell = sc.read(cell_path)
        adata_sample = sc.read(sample_path)
    
    if not status_flags["rna"]["preprocessing"]:
        raise ValueError("RNA preprocessing skipped but no preprocessed data found.")

    # ==================== CELL TYPE CLUSTERING ====================
    if cell_type_cluster:
        print(f"Starting cell type clustering at resolution: {leiden_cluster_resolution}")
        cell_types_func = cell_types_linux if use_gpu else cell_types
        
        adata_cell, adata_sample = cell_types_func(
            anndata_cell=adata_cell,
            anndata_sample=adata_sample,
            cell_type_column=celltype_col,
            existing_cell_types=existing_cell_types,
            n_target_clusters=n_target_cell_clusters,
            umap=umap,
            save=True,
            output_dir=rna_output_dir,
            leiden_cluster_resolution=leiden_cluster_resolution,
            cell_embedding_column=cell_embedding_column,
            cell_embedding_num_PCs=cell_embedding_num_pcs,
            verbose=verbose,
            umap_plots=True,
        )
        status_flags["rna"]["cell_type_cluster"] = True

    # ==================== SAMPLE EMBEDDING ====================
    if derive_sample_embedding:
        print("Starting sample embedding derivation...")

        if celltype_col not in adata_sample.obs.columns:
            raise ValueError(
                f"Cell type column '{celltype_col}' not found in adata_sample.obs. "
                "Provide a preprocessed input with the cell type column present, "
                "or enable rna_cell_type_cluster to generate it."
            )

        _, pseudo_adata = calculate_sample_embedding(
            adata=adata_sample,
            sample_col=sample_col,
            celltype_col=celltype_col,
            batch_col=sample_level_batch_col or None,
            output_dir=rna_output_dir,
            sample_hvg_number=sample_hvg_number,
            n_expression_components=sample_embedding_dimension,
            n_proportion_components=sample_embedding_dimension,
            harmony_for_proportion=harmony_for_proportion,
            preserve_cols_in_sample_embedding=preserve_cols_in_sample_embedding,
            use_gpu=use_gpu,
            atac=False,
            save=True,
            verbose=verbose,
        )
        status_flags["rna"]["derive_sample_embedding"] = True
    else:
        pseudobulk_path = pseudo_adata_path or os.path.join(rna_output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
        
        if os.path.exists(pseudobulk_path):
            print(f"Loading pseudobulk from: {pseudobulk_path}")
            pseudo_adata = sc.read(pseudobulk_path)
        
        status_flags["rna"]["derive_sample_embedding"] = True

    # ==================== CCA-BASED RESOLUTION SELECTION ====================
    if cca_based_cell_resolution_selection:
        print("Finding optimal cell resolution using CCA optimization...")
        
        find_resolution_func = find_optimal_cell_resolution_linux if use_gpu else find_optimal_cell_resolution
        
        for column in ["X_DR_expression", "X_DR_proportion"]:
            print(f"\nOptimizing resolution for {column}...")
            
            optimal_resolution, results_df = find_resolution_func(
                adata_cell=adata_cell,
                adata_sample=adata_sample,
                output_dir=rna_output_dir,
                column=column,
                modality="rna",
                trajectory_col=trajectory_col,
                batch_col=sample_level_batch_col or None,
                sample_col=sample_col,
                celltype_col=celltype_col,
                cell_embedding_column=cell_embedding_column,
                cell_embedding_num_pcs=cell_embedding_num_pcs,
                n_hvg_features=sample_hvg_number,
                sample_embedding_dimension=sample_embedding_dimension,
                harmony_for_proportion=harmony_for_proportion,
                preserve_cols_in_sample_embedding=preserve_cols_in_sample_embedding,
                n_cca_pcs=n_cca_pcs,
                compute_corrected_pvalues=cca_compute_corrected_pvalues,
                coarse_start=cca_coarse_start,
                coarse_end=cca_coarse_end,
                coarse_step=cca_coarse_step,
                fine_range=cca_fine_range,
                fine_step=cca_fine_step,
                verbose=verbose,
            )
            print(f"Optimal resolution for {column}: {optimal_resolution:.3f}")
        
        from utils.unify_optimal import replace_optimal_dimension_reduction
        pseudo_adata = replace_optimal_dimension_reduction(rna_output_dir, modality="RNA")
        
        status_flags["rna"]["cca_based_cell_resolution_selection"] = True
        print("CCA-based resolution selection completed!")

    print("RNA preprocessing pipeline completed successfully!")
    
    return {
        'adata_cell': adata_cell,
        'adata_sample': adata_sample,
        'pseudo_adata': pseudo_adata,
        'status_flags': status_flags
    }