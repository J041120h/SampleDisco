import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata as ad
import scanpy as sc

from sampledisco.sample_embedding import compute_sample_embedding
from sampledisco.preparation.multi_omics_glue import multiomics_preparation
from sampledisco.preparation.multi_omics_merge import propagate_cell_type
from sampledisco.preparation.multi_omics_cell_type_cpu import cell_types_multiomics
from sampledisco.preparation.multi_omics_batch_correction import (
    harmonize_xglue,
    Z_RMD_KEY,      # paper "Z_rmd"   — sample-preserved, RMD displacement role
    Z_CLUST_KEY,    # paper "Z_clust" — sample-removed,   cluster / composition role
    XGLUE_KEY,      # internal — scGLUE's native obsm key (fallback only)
)


def _resolve_embedding_keys(adata, cluster_override=None, rmd_override=None):
    """Return (cluster_emb_key, rmd_emb_key) for the SE / cell typing stack.

    Paper-aligned (Fig. 1, Stage 2): the cluster role reads ``Z_clust`` and
    the RMD role reads ``Z_rmd``. Both keys are written into the integrated
    h5ad by either the Harmony post-pass (Mode A) or the 2-run scGLUE merge
    (Mode B). Falls back to ``X_glue`` only when neither has run (un-
    supported in the current pipeline, but keeps the failure mode honest).
    """
    # Cluster role MUST use the sample-removed Z_clust; falling back to
    # X_glue (sample-preserved) here would silently leak per-sample variance
    # into the composition blocks.
    if cluster_override:
        cluster = cluster_override
    elif Z_CLUST_KEY in adata.obsm:
        cluster = Z_CLUST_KEY
    else:
        raise KeyError(
            f"Cluster embedding key {Z_CLUST_KEY!r} not found in adata.obsm "
            f"(available: {list(adata.obsm.keys())}). "
            "The sample-removed cluster embedding is required for the composition "
            "blocks. Run harmonize_xglue or set run_glue_twice_for_sample_removal=True."
        )
    rmd = rmd_override or (
        Z_RMD_KEY if Z_RMD_KEY in adata.obsm else XGLUE_KEY
    )
    for role, key in (("cluster", cluster), ("rmd", rmd)):
        if key not in adata.obsm:
            raise KeyError(
                f"{role}_emb_key={key!r} not in adata.obsm "
                f"(available: {list(adata.obsm.keys())})")
    return cluster, rmd


def multiomics_wrapper(
    # ===== Required Parameters =====
    rna_file=None,
    atac_file=None,
    multiomics_output_dir=None,

    # ===== Process Control Flags =====
    integration=True,
    derive_sample_embedding=True,
    autotune_enable=False,
    # scGLUE removes batch but PRESERVES per-sample variance — its output
    # X_glue is the RMD displacement embedding. For the cluster /
    # composition role we need a SAMPLE-removed variant; the pipeline
    # always provides one via ONE of two paths:
    #
    #   (default) Harmony post-pass on X_glue with the sample column
    #             (and batch, if present) → X_glue_harmony.
    #   (opt-in)  Train scGLUE TWICE in STEP 1; the second run (with
    #             treat_sample_as_batch=True) yields X_glue_harmony
    #             end-to-end. When this is enabled the Harmony post-pass
    #             auto-skips because X_glue_harmony already exists.
    #
    # No "legacy" path exists — the cluster embedding is always derived.
    harmonize_xglue_max_iter=50,
    run_glue_twice_for_sample_removal=False,

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
    run_glue_merge=True,                  # build embedding-only union AnnData
    run_glue_preprocess_per_modality=True, # per-modality QC + normalize for DGE
    cell_type_cluster=True,
    run_glue_visualization=True,

    # GLUE preprocessing parameters
    ensembl_release=98,
    species="homo_sapiens",
    use_highly_variable=True,
    n_top_genes=2000,
    n_top_peaks=50000,
    atac_min_cells_floor=10,
    n_pca_comps=50,
    n_lsi_comps=50,
    lsi_n_iter=15,
    gtf_by="gene_name",
    flavor="seurat_v3",
    generate_umap=False,
    compression="gzip",

    # GLUE training parameters
    consistency_threshold=0.05,
    # V2 default: scGLUE removes the technical batch column (``batch_col``)
    # during training but PRESERVES per-sample variance, so its output
    # X_glue is suitable as the RMD displacement embedding. Set
    # treat_sample_as_batch=True only to force sample removal inside GLUE
    # itself (legacy V1 behavior, or the 2-run secondary pass for the
    # cluster embedding).
    treat_sample_as_batch=False,
    save_prefix="glue",
    # scGLUE training throughput knobs.
    #   data_batch_size       : cells per minibatch. Library default 128;
    #                           bigger saturates modern GPUs better.
    #   max_epochs            : None → scglue's "AUTO" (adaptive cap).
    #   dataloader_*          : torch DataLoader workers/prefetch — bigger
    #                           overlaps I/O with GPU compute.
    #   *_shuffle_num_workers : background workers for array/graph shuffle.
    # See preparation/multi_omics_glue.py:glue_train docstring for details.
    glue_data_batch_size: int = 1024,
    glue_max_epochs: Optional[int] = None,
    glue_dataloader_num_workers: int = 4,
    glue_dataloader_fetches_per_worker: int = 8,
    glue_array_shuffle_num_workers: int = 4,
    glue_graph_shuffle_num_workers: int = 4,

    # Cell-typing SNN label-transfer metric (RNA→ATAC neighbor distance).
    metric="cosine",

    # ===== Per-modality preprocess QC parameters =====
    rna_min_cells: int = 500,
    rna_min_genes: int = 500,
    rna_pct_mito_cutoff: float = 20.0,
    rna_exclude_genes: Optional[List[str]] = None,
    atac_min_cells: int = 1,
    atac_min_features: int = 2000,
    atac_max_features: int = 15000,
    atac_min_cells_per_sample: int = 1,
    atac_exclude_features: Optional[List[str]] = None,
    atac_doublet_detection: bool = True,
    atac_tfidf_scale_factor: float = 1e4,
    atac_log_transform: bool = True,

    # GLUE cell type parameters
    existing_cell_types=False,
    n_target_clusters=10,
    cluster_resolution=0.8,
    # If None: auto-resolved at runtime to X_glue_harmony (sample-removed,
    # cluster role) when present, else X_glue. Explicit override accepted.
    use_rep_celltype=None,
    markers=None,
    generate_umap_celltype=True,

    # GLUE visualization parameters
    plot_columns=None,

    # ===== Sample Embedding Parameters (new method) =====
    sample_embedding_medium_K: int = 120,
    sample_embedding_fine_K: int = 300,
    sample_embedding_rmd_dim: int = 8,
    sample_embedding_use_clr: bool = False,
    sample_embedding_use_rmd: bool = True,
    sample_embedding_block_weights: Optional[List[float]] = None,
    sample_embedding_rmd_weight: float = 0.60,
    sample_embedding_pca_components: int = 10,
    sample_embedding_batch_method: str = "harmony",

    # ===== Autotune Parameters =====
    autotune_search: str = "bayesian",
    autotune_scoring: str = "auto",
    autotune_scope: str = "alpha_only",
    autotune_alpha_bounds=(0.1, 10.0),
    autotune_grouping_col: Optional[str] = None,
    autotune_tune_on_modality: Optional[str] = None,

    # ===== Paths for Skipping Steps =====
    integrated_h5ad_path=None,
    sample_adata_path=None,

    # ===== System Parameters =====
    status_flags=None,
) -> Dict[str, Any]:
    """Multi-omics wrapper: GLUE integration, preprocessing, cell typing, and the
    new single-key sample embedding (composition + RMD). The cluster / composition
    role uses ``Z_clust`` (sample-removed, from Harmony post-pass or 2-run GLUE);
    the RMD displacement role uses ``Z_rmd`` (= ``X_glue``, sample-preserved).
    Multi-omics groups RMD blocks by ``modality_col``.

    Returns dict with adata, sample_adata, status_flags.
    """
    if any(var is None for var in [rna_file, atac_file, multiomics_output_dir]):
        raise ValueError("rna_file, atac_file, and multiomics_output_dir must all be provided")

    default_status = {
        "glue_integration": False,
        "glue_preprocessing": False,
        "glue_training": False,
        "glue_merge": False,
        "glue_preprocess_per_modality": False,
        "glue_cell_types": False,
        "glue_visualization": False,
        "harmonize_xglue": False,
        "derive_sample_embedding": False,
        "autotune": False,
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

    results: Dict[str, Any] = {}
    Path(multiomics_output_dir).mkdir(parents=True, exist_ok=True)

    if multiomics_verbose:
        print(f"Starting multi-modal pipeline with output directory: {multiomics_output_dir}")
        print(f"GPU mode: {'Enabled' if use_gpu else 'Disabled'}")

    # `adata_sample.h5ad` is now the embedding-only union written by
    # build_embedding_union (preparation/multi_omics_merge.py). It carries
    # obs (sample, modality, batch, sev.level …) + obsm (X_glue, Z_clust,
    # Z_rmd) but no expression X — DGE/RAISIN reads the per-modality
    # preprocessed h5ads instead.
    h5ad_path = (integrated_h5ad_path
                 if integrated_h5ad_path and os.path.exists(integrated_h5ad_path)
                 else f"{multiomics_output_dir}/preprocess/adata_sample.h5ad")
    rna_pre_path  = f"{multiomics_output_dir}/preprocess/adata_rna_preprocessed.h5ad"
    atac_pre_path = f"{multiomics_output_dir}/preprocess/adata_atac_preprocessed.h5ad"

    current_adata = None

    # ==================== STEP 1: GLUE INTEGRATION ====================
    if integration:
        if multiomics_verbose:
            print("Step 1: Running GLUE integration...")
        glue_result = multiomics_preparation(
            rna_file=rna_file, atac_file=atac_file,
            rna_sample_meta_file=rna_sample_meta_file,
            atac_sample_meta_file=atac_sample_meta_file,
            additional_hvg_file=additional_hvg_file,
            run_preprocessing=run_glue_preprocessing,
            run_training=run_glue_training,
            run_merge=run_glue_merge,
            run_preprocess_per_modality=run_glue_preprocess_per_modality,
            run_visualization=run_glue_visualization,
            ensembl_release=ensembl_release, species=species,
            use_highly_variable=use_highly_variable, n_top_genes=n_top_genes,
            n_top_peaks=n_top_peaks, atac_min_cells_floor=atac_min_cells_floor,
            n_pca_comps=n_pca_comps, n_lsi_comps=n_lsi_comps, gtf_by=gtf_by,
            flavor=flavor, generate_umap=generate_umap,
            rna_sample_column=rna_sample_column, atac_sample_column=atac_sample_column,
            consistency_threshold=consistency_threshold,
            treat_sample_as_batch=treat_sample_as_batch, save_prefix=save_prefix,
            batch_key=batch_col, sample_key=sample_col,
            data_batch_size=glue_data_batch_size, max_epochs=glue_max_epochs,
            dataloader_num_workers=glue_dataloader_num_workers,
            dataloader_fetches_per_worker=glue_dataloader_fetches_per_worker,
            array_shuffle_num_workers=glue_array_shuffle_num_workers,
            graph_shuffle_num_workers=glue_graph_shuffle_num_workers,
            run_second_glue_for_sample_removal=run_glue_twice_for_sample_removal,
            rna_min_cells=rna_min_cells, rna_min_genes=rna_min_genes,
            rna_pct_mito_cutoff=rna_pct_mito_cutoff,
            rna_exclude_genes=rna_exclude_genes,
            atac_min_cells=atac_min_cells,
            atac_min_features=atac_min_features, atac_max_features=atac_max_features,
            atac_min_cells_per_sample=atac_min_cells_per_sample,
            atac_exclude_features=atac_exclude_features,
            atac_doublet_detection=atac_doublet_detection,
            atac_tfidf_scale_factor=atac_tfidf_scale_factor,
            atac_log_transform=atac_log_transform,
            verbose=multiomics_verbose,
            plot_columns=plot_columns, output_dir=multiomics_output_dir,
        )
        results['glue'] = glue_result
        status_flags["multiomics"]["glue_integration"] = True
        if run_glue_preprocessing:
            status_flags["multiomics"]["glue_preprocessing"] = True
        if run_glue_training:
            status_flags["multiomics"]["glue_training"] = True
        if run_glue_merge:
            status_flags["multiomics"]["glue_merge"] = True
        if run_glue_preprocess_per_modality:
            status_flags["multiomics"]["glue_preprocess_per_modality"] = True
        if run_glue_visualization:
            status_flags["multiomics"]["glue_visualization"] = True
        if multiomics_verbose:
            print("GLUE integration completed successfully")

    # Load the embedding-only union (built in STEP 1 by build_embedding_union).
    if os.path.exists(h5ad_path):
        current_adata = sc.read(h5ad_path)
        if multiomics_verbose:
            print(f"Loaded embedding union from: {h5ad_path}")
    else:
        raise ValueError(
            f"Embedding union not found at {h5ad_path}. Set "
            "run_glue_merge=True or provide integrated_h5ad_path.")
    results['adata'] = current_adata

    # ==================== STEP 2b: PROVIDE Z_clust (paper-aligned cluster view) ===
    # Z_clust (sample-removed cluster / composition embedding) is required
    # downstream. Two equivalent providers — STEP 1 may have set it
    # already via the 2-run scGLUE merge; otherwise this step runs a
    # Harmony pass on Z_rmd (= X_glue) with sample as batch_key.
    if current_adata is None:
        current_adata = ad.read_h5ad(h5ad_path)
    if Z_CLUST_KEY in current_adata.obsm:
        if multiomics_verbose:
            print(f"Step 2b: obsm['{Z_CLUST_KEY}'] already present "
                  f"(end-to-end from 2-run scGLUE) — no Harmony pass needed.")
        status_flags["multiomics"]["harmonize_xglue"] = True
    else:
        if multiomics_verbose:
            print(f"Step 2b: Harmony post-pass on obsm['{XGLUE_KEY}'] "
                  f"→ obsm['{Z_CLUST_KEY}'] (sample-removed cluster view)...")
        current_adata = harmonize_xglue(
            current_adata,
            sample_col=sample_col,
            batch_col=batch_col,
            use_gpu=use_gpu,
            max_iter=harmonize_xglue_max_iter,
            random_state=random_state,
            verbose=multiomics_verbose,
        )
        if Z_CLUST_KEY in current_adata.obsm:
            status_flags["multiomics"]["harmonize_xglue"] = True
            if save_intermediate:
                sc.write(h5ad_path, current_adata)
                if multiomics_verbose:
                    print(f"[xglue-harmony] re-saved {h5ad_path}")

    # ==================== STEP 2c: CELL TYPE CLUSTERING ====================
    # Uses the paper's Z_clust (sample-removed cluster embedding), produced
    # in STEP 2b by either the Harmony post-pass or the 2-run scGLUE merge.
    if cell_type_cluster:
        if current_adata is None:
            current_adata = ad.read_h5ad(h5ad_path)

        cluster_key, _ = _resolve_embedding_keys(
            current_adata, cluster_override=use_rep_celltype)
        if multiomics_verbose:
            print(f"Step 2c: Cell type assignment (use_rep={cluster_key})...")

        cell_types_func = cell_types_multiomics
        if use_gpu:
            try:
                from sampledisco.preparation.multi_omics_cell_type_gpu import cell_types_multiomics_gpu
                cell_types_func = cell_types_multiomics_gpu
            except ImportError as e:
                print(f"Warning: GPU cell-typing unavailable ({e}). Falling back to CPU.")
                use_gpu = False

        current_adata = cell_types_func(
            adata=current_adata,
            modality_column=modality_col,
            rna_modality_value="RNA",
            atac_modality_value="ATAC",
            cell_type_column=celltype_col,
            cluster_resolution=cluster_resolution,
            use_rep=cluster_key,
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

        # Propagate cell_type labels from the union back onto the per-modality
        # h5ads so downstream DGE / RAISIN can read them without re-running.
        if os.path.exists(rna_pre_path) or os.path.exists(atac_pre_path):
            propagate_cell_type(
                union_path=h5ad_path,
                per_modality_paths=[rna_pre_path, atac_pre_path],
                celltype_col=celltype_col,
                verbose=multiomics_verbose,
            )

    # ==================== STEP 3: SAMPLE EMBEDDING ====================
    if derive_sample_embedding:
        if multiomics_verbose:
            print("Step 3: Sample embedding (composition + RMD)...")

        if current_adata is None:
            current_adata = ad.read_h5ad(h5ad_path)
            results['adata'] = current_adata

        if celltype_col not in current_adata.obs.columns:
            raise ValueError(
                f"Cell type column '{celltype_col}' not in adata.obs. Run "
                "cell_type_cluster=True or provide pre-typed input.")

        cluster_emb_key, rmd_emb_key = _resolve_embedding_keys(current_adata)

        if autotune_enable:
            from sampledisco.parameter_selection.autotune import run_autotune
            run_autotune(
                current_adata, multiomics_output_dir,
                sample_col=sample_col,
                celltype_col=celltype_col,
                cluster_emb_key=cluster_emb_key,
                rmd_emb_key=rmd_emb_key,
                modality_col=modality_col,
                batch_col=batch_col,
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
                tune_on_modality=autotune_tune_on_modality,
                save=True, verbose=multiomics_verbose,
            )
            status_flags["multiomics"]["autotune"] = True
        else:
            compute_sample_embedding(
                current_adata, multiomics_output_dir,
                use_gpu=use_gpu,
                sample_col=sample_col,
                celltype_col=celltype_col,
                cluster_emb_key=cluster_emb_key,
                rmd_emb_key=rmd_emb_key,
                modality_col=modality_col,
                batch_col=batch_col,
                medium_K=sample_embedding_medium_K,
                fine_K=sample_embedding_fine_K,
                rmd_dim_per_cluster=sample_embedding_rmd_dim,
                use_clr=sample_embedding_use_clr,
                use_rmd=sample_embedding_use_rmd,
                block_weights=sample_embedding_block_weights,
                rmd_weight=sample_embedding_rmd_weight,
                pca_components=sample_embedding_pca_components,
                batch_method=sample_embedding_batch_method,
                save=True, verbose=multiomics_verbose,
            )
        status_flags["multiomics"]["derive_sample_embedding"] = True
        # Persist X_DR_sample (set in-memory by the SE step) into the union so
        # a resume-from-disk downstream run can read it. The SE step's own
        # re-save targets the single-omics adata_preprocessed.h5ad, which
        # does not exist in the multiomics flow.
        if save_intermediate and "X_DR_sample" in current_adata.uns:
            sc.write(h5ad_path, current_adata)
    else:
        if current_adata is not None and "X_DR_sample" in current_adata.uns:
            status_flags["multiomics"]["derive_sample_embedding"] = True

    print("Multiomics preprocessing pipeline completed successfully!")

    return {
        'adata': current_adata,
        'status_flags': status_flags,
    }
