#!/usr/bin/env python3

import os
import sys
from typing import List, Optional

import scanpy as sc


from sampledisco.preparation.atac_preprocess_cpu import preprocess
from sampledisco.preparation.cell_type_cpu import cell_types
from sampledisco.sample_embedding import compute_sample_embedding


def atac_wrapper(
    atac_count_data_path: str = None,
    atac_output_dir: str = None,

    # Pipeline control flags
    preprocessing: bool = True,
    cell_type_cluster: bool = True,
    derive_sample_embedding: bool = True,
    autotune_enable: bool = False,

    # General settings
    use_gpu: bool = False,
    verbose: bool = True,
    status_flags: dict = None,

    # Input data paths (for resuming)
    adata_path: str = None,
    atac_sample_meta_path: str = None,
    cell_meta_path: str = None,
    sample_adata_path: str = None,

    # Common column names
    sample_col: str = 'sample',
    sample_level_batch_col: Optional[List[str]] = None,
    celltype_col: str = 'cell_type',
    cell_embedding_column: str = None,

    # ATAC-specific preprocessing parameters
    min_cells: int = 1,
    min_features: int = 2000,
    max_features: int = 15000,
    min_cells_per_sample: int = 1,
    exclude_features: list = None,
    cell_level_batch_key: list = None,
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

    # Sample embedding parameters (new method)
    sample_embedding_medium_K: int = 120,
    sample_embedding_fine_K: int = 300,
    sample_embedding_rmd_dim: int = 8,
    sample_embedding_use_clr: bool = False,
    sample_embedding_use_rmd: bool = True,
    sample_embedding_block_weights: Optional[List[float]] = None,
    sample_embedding_rmd_weight: float = 0.60,
    sample_embedding_pca_components: int = 10,
    sample_embedding_batch_method: str = "harmony",

    # Autotune parameters
    autotune_search: str = "bayesian",
    autotune_scoring: str = "auto",
    autotune_scope: str = "alpha_only",
    autotune_alpha_bounds=(0.1, 10.0),
    autotune_grouping_col: Optional[str] = None,

    seed: int = 42,
) -> dict:
    """ATAC-seq wrapper: preprocessing, cell typing, single-key sample embedding,
    optional autotune. Returns dict with adata, sample_adata, status_flags."""
    print("Starting ATAC wrapper function...")
    if atac_count_data_path is None or atac_output_dir is None:
        raise ValueError("Required: atac_count_data_path and atac_output_dir")

    if use_gpu:
        try:
            from sampledisco.preparation.atac_preprocess_gpu import preprocess_gpu
            from sampledisco.preparation.cell_type_gpu import cell_types_gpu
        except ImportError as e:
            print(f"Warning: GPU modules unavailable ({e}). Falling back to CPU.")
            use_gpu = False

    cell_level_batch_key = cell_level_batch_key or []
    sample_level_batch_col = sample_level_batch_col or []

    default_status = {
        "preprocessing": False,
        "cell_type_cluster": False,
        "derive_sample_embedding": False,
        "autotune": False,
        "sample_distance_calculation": False,
        "trajectory_analysis": False,
        "trajectory_dge": False,
        "sample_cluster": False,
        "proportion_test": False,
        "cluster_dge": False,
        "visualization": False,
    }
    if status_flags is None:
        status_flags = {"atac": default_status.copy()}
    elif "atac" not in status_flags:
        status_flags["atac"] = default_status.copy()

    adata = None

    # ============================ PREPROCESSING ============================
    if preprocessing:
        print("Starting preprocessing...")
        preprocess_func = preprocess_gpu if use_gpu else preprocess
        adata = preprocess_func(
            h5ad_path=atac_count_data_path,
            sample_meta_path=atac_sample_meta_path,
            output_dir=atac_output_dir,
            sample_column=sample_col,
            cell_meta_path=cell_meta_path,
            sample_level_batch_key=sample_level_batch_col,
            cell_embedding_num_PCs=cell_embedding_num_pcs,
            num_harmony_iterations=num_harmony_iterations,
            num_cell_hvfs=num_cell_hvfs,
            min_cells=min_cells,
            min_features=min_features,
            max_features=max_features,
            min_cells_per_sample=min_cells_per_sample,
            exclude_features=exclude_features,
            cell_level_batch_key=cell_level_batch_key,
            doublet_detection=doublet_detection,
            tfidf_scale_factor=tfidf_scale_factor,
            log_transform=log_transform,
            drop_first_lsi=drop_first_lsi,
            verbose=verbose,
        )
        status_flags["atac"]["preprocessing"] = True
    else:
        cell_path = adata_path or os.path.join(atac_output_dir, "preprocess", "adata_preprocessed.h5ad")
        if not os.path.exists(cell_path):
            raise ValueError(f"Preprocessed data not found at {cell_path}.")
        adata = sc.read(cell_path)
        status_flags["atac"]["preprocessing"] = True

    # ============================ CELL TYPE CLUSTERING =====================
    if cell_type_cluster:
        print(f"Starting cell type clustering at resolution={leiden_cluster_resolution}")
        cell_types_func = cell_types_gpu if use_gpu else cell_types
        adata = cell_types_func(
            anndata_cell=adata,
            cell_type_column=celltype_col,
            existing_cell_types=existing_cell_types,
            n_target_clusters=n_target_cell_clusters,
            umap=umap,
            save=True,
            output_dir=atac_output_dir,
            leiden_cluster_resolution=leiden_cluster_resolution,
            cell_embedding_column=cell_embedding_column,
            cell_embedding_num_PCs=cell_embedding_num_pcs,
            verbose=verbose,
            umap_plots=True,
        )
        status_flags["atac"]["cell_type_cluster"] = True

    # ============================ SAMPLE EMBEDDING =========================
    if derive_sample_embedding:
        print("Starting sample embedding derivation (composition + RMD)...")
        if celltype_col not in adata.obs.columns:
            raise ValueError(
                f"Cell type column '{celltype_col}' not found in adata.obs.")

        cluster_emb_key = cell_embedding_column or "Z_clust"
        rmd_emb_key = "Z_rmd" if "Z_rmd" in adata.obsm else cluster_emb_key

        if autotune_enable:
            from sampledisco.parameter_selection.autotune import run_autotune
            run_autotune(
                adata, atac_output_dir,
                sample_col=sample_col,
                celltype_col=celltype_col,
                cluster_emb_key=cluster_emb_key,
                rmd_emb_key=rmd_emb_key,
                modality_col=None,
                batch_col=sample_level_batch_col or None,
                grouping_col=autotune_grouping_col,
                medium_K=sample_embedding_medium_K,
                fine_K=sample_embedding_fine_K,
                rmd_dim=sample_embedding_rmd_dim,
                pca_components=sample_embedding_pca_components,
                batch_method=sample_embedding_batch_method,
                scoring=autotune_scoring,
                search=autotune_search,
                scope=autotune_scope,
                alpha_bounds=autotune_alpha_bounds,
                save=True, verbose=verbose,
            )
            status_flags["atac"]["autotune"] = True
        else:
            compute_sample_embedding(
                adata, atac_output_dir,
                use_gpu=use_gpu,
                sample_col=sample_col,
                celltype_col=celltype_col,
                cluster_emb_key=cluster_emb_key,
                rmd_emb_key=rmd_emb_key,
                modality_col=None,
                batch_col=sample_level_batch_col or None,
                medium_K=sample_embedding_medium_K,
                fine_K=sample_embedding_fine_K,
                rmd_dim_per_cluster=sample_embedding_rmd_dim,
                use_clr=sample_embedding_use_clr,
                use_rmd=sample_embedding_use_rmd,
                block_weights=sample_embedding_block_weights,
                rmd_weight=sample_embedding_rmd_weight,
                pca_components=sample_embedding_pca_components,
                batch_method=sample_embedding_batch_method,
                save=True, verbose=verbose,
                seed=seed,
            )
        status_flags["atac"]["derive_sample_embedding"] = True
    else:
        if adata is not None and "X_DR_sample" in adata.uns:
            status_flags["atac"]["derive_sample_embedding"] = True

    print("ATAC preprocessing pipeline completed successfully!")
    return {
        'adata': adata,
        'status_flags': status_flags,
    }
