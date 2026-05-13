#!/usr/bin/env python3

import os
import sys
from typing import List, Optional

import scanpy as sc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sample_embedding import compute_sample_embedding
from preparation.rna_preprocess_cpu import preprocess
from preparation.cell_type_cpu import cell_types


def rna_wrapper(
    rna_count_data_path: str = None,
    rna_output_dir: str = None,

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
    rna_sample_meta_path: str = None,
    cell_meta_path: str = None,
    sample_adata_path: str = None,

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

    # Sample embedding parameters (new method)
    sample_embedding_medium_K: int = 120,
    sample_embedding_fine_K: int = 300,
    sample_embedding_cmd_dim: int = 8,
    sample_embedding_use_clr: bool = False,
    sample_embedding_use_cmd: bool = True,
    sample_embedding_block_weights: Optional[List[float]] = None,
    sample_embedding_cmd_weight: float = 0.60,
    sample_embedding_pca_components: int = 10,
    sample_embedding_batch_method: str = "harmony",

    # Autotune parameters
    autotune_search: str = "bayesian",
    autotune_scoring: str = "auto",
    autotune_scope: str = "alpha_only",
    autotune_alpha_bounds=(0.1, 10.0),
    autotune_grouping_col: Optional[str] = None,
) -> dict:
    """RNA-seq wrapper: preprocessing, cell-typing, single-key sample embedding,
    and optional autotune. Returns dict with adata, sample_adata, status_flags.

    Downstream analysis (trajectory, clustering, etc.) is handled by the
    shared ``downstream_analysis()`` function in wrapper.py.
    """
    print("Starting RNA wrapper function...")
    if rna_count_data_path is None or rna_output_dir is None:
        raise ValueError("Required: rna_count_data_path and rna_output_dir")

    if use_gpu:
        from preparation.rna_preprocess_gpu import preprocess_linux
        from preparation.cell_type_gpu import cell_types_linux

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
        status_flags = {"rna": default_status.copy()}
    elif "rna" not in status_flags:
        status_flags["rna"] = default_status.copy()

    adata = None

    # ============================ PREPROCESSING ============================
    if preprocessing:
        print("Starting preprocessing...")
        preprocess_func = preprocess_linux if use_gpu else preprocess
        adata = preprocess_func(
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
            verbose=verbose,
        )
        status_flags["rna"]["preprocessing"] = True
    else:
        cell_path = adata_path or os.path.join(rna_output_dir, "preprocess", "adata_preprocessed.h5ad")
        if not os.path.exists(cell_path):
            raise ValueError(f"Preprocessed data not found at {cell_path}.")
        adata = sc.read(cell_path)
        status_flags["rna"]["preprocessing"] = True

    # ============================ CELL TYPE CLUSTERING =====================
    if cell_type_cluster:
        print(f"Starting cell type clustering at resolution={leiden_cluster_resolution}")
        cell_types_func = cell_types_linux if use_gpu else cell_types
        adata = cell_types_func(
            anndata_cell=adata,
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

    # ============================ SAMPLE EMBEDDING =========================
    if derive_sample_embedding:
        print("Starting sample embedding derivation (composition + CMD)...")
        if celltype_col not in adata.obs.columns:
            raise ValueError(
                f"Cell type column '{celltype_col}' not found in adata.obs. "
                "Run cell-type clustering or provide an input with celltype labels."
            )

        cluster_emb_key = cell_embedding_column or "X_pca_harmony"
        cmd_emb_key = f"{cluster_emb_key}_nosamp" if f"{cluster_emb_key}_nosamp" in adata.obsm else cluster_emb_key

        if autotune_enable:
            from parameter_selection.autotune import run_autotune
            run_autotune(
                adata, rna_output_dir,
                sample_col=sample_col,
                celltype_col=celltype_col,
                cluster_emb_key=cluster_emb_key,
                cmd_emb_key=cmd_emb_key,
                modality_col=None,
                batch_col=sample_level_batch_col or None,
                grouping_col=autotune_grouping_col,
                medium_K=sample_embedding_medium_K,
                fine_K=sample_embedding_fine_K,
                cmd_dim=sample_embedding_cmd_dim,
                pca_components=sample_embedding_pca_components,
                batch_method=sample_embedding_batch_method,
                scoring=autotune_scoring,
                search=autotune_search,
                scope=autotune_scope,
                alpha_bounds=autotune_alpha_bounds,
                save=True, verbose=verbose,
            )
            status_flags["rna"]["autotune"] = True
        else:
            compute_sample_embedding(
                adata, rna_output_dir,
                use_gpu=use_gpu,
                sample_col=sample_col,
                celltype_col=celltype_col,
                cluster_emb_key=cluster_emb_key,
                cmd_emb_key=cmd_emb_key,
                modality_col=None,
                batch_col=sample_level_batch_col or None,
                medium_K=sample_embedding_medium_K,
                fine_K=sample_embedding_fine_K,
                cmd_dim_per_cluster=sample_embedding_cmd_dim,
                use_clr=sample_embedding_use_clr,
                use_cmd=sample_embedding_use_cmd,
                block_weights=sample_embedding_block_weights,
                cmd_weight=sample_embedding_cmd_weight,
                pca_components=sample_embedding_pca_components,
                batch_method=sample_embedding_batch_method,
                save=True, verbose=verbose,
            )
        status_flags["rna"]["derive_sample_embedding"] = True
    else:
        # If derive_sample_embedding=False, the cell-level adata loaded above
        # is expected to already carry .uns['X_DR_sample'] (re-saved on a
        # previous run). No separate sample-level h5ad is involved.
        if adata is not None and "X_DR_sample" in adata.uns:
            status_flags["rna"]["derive_sample_embedding"] = True

    print("RNA preprocessing pipeline completed successfully!")
    return {
        'adata': adata,
        'status_flags': status_flags,
    }
