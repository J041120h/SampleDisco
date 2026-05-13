import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata as ad
import scanpy as sc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_embedding import compute_sample_embedding
from preparation.multi_omics_glue import multiomics_preparation
from preparation.multi_omics_preprocess import integrate_preprocess
from preparation.multi_omics_cell_type_cpu import cell_types_multiomics


def multiomics_wrapper(
    # ===== Required Parameters =====
    rna_file=None,
    atac_file=None,
    multiomics_output_dir=None,

    # ===== Process Control Flags =====
    integration=True,
    integration_preprocessing=True,
    derive_sample_embedding=True,
    autotune_enable=False,

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

    # ===== Sample Embedding Parameters (new method) =====
    sample_embedding_medium_K: int = 120,
    sample_embedding_fine_K: int = 300,
    sample_embedding_cmd_dim: int = 8,
    sample_embedding_use_clr: bool = False,
    sample_embedding_use_cmd: bool = True,
    sample_embedding_block_weights: Optional[List[float]] = None,
    sample_embedding_cmd_weight: float = 0.60,
    sample_embedding_pca_components: int = 10,
    sample_embedding_batch_method: str = "harmony",

    # ===== Autotune Parameters =====
    autotune_search: str = "bayesian",
    autotune_scoring: str = "auto",
    autotune_scope: str = "alpha_only",
    autotune_alpha_bounds=(0.1, 10.0),
    autotune_grouping_col: Optional[str] = None,

    # ===== Paths for Skipping Steps =====
    integrated_h5ad_path=None,
    sample_adata_path=None,

    # ===== System Parameters =====
    status_flags=None,
) -> Dict[str, Any]:
    """Multi-omics wrapper: GLUE integration, preprocessing, cell typing, and the
    new single-key sample embedding (composition + CMD). Multi-omics groups CMD
    by ``modality_col`` and uses ``X_glue`` as both the cluster and CMD
    cell-level embedding.

    Returns dict with adata, sample_adata, status_flags.
    """
    if any(var is None for var in [rna_file, atac_file, multiomics_output_dir]):
        raise ValueError("rna_file, atac_file, and multiomics_output_dir must all be provided")

    default_status = {
        "glue_integration": False,
        "glue_preprocessing": False,
        "glue_training": False,
        "glue_gene_activity": False,
        "glue_cell_types": False,
        "glue_visualization": False,
        "integration_preprocessing": False,
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

    # `h5ad_path` is the GLUE-merged intermediate that integrate_preprocess reads;
    # integrate_preprocess writes `preprocess/adata_preprocessed.h5ad`.
    h5ad_path = (integrated_h5ad_path
                 if integrated_h5ad_path and os.path.exists(integrated_h5ad_path)
                 else f"{multiomics_output_dir}/preprocess/adata_sample.h5ad")

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
            run_gene_activity=run_glue_gene_activity,
            run_visualization=run_glue_visualization,
            ensembl_release=ensembl_release, species=species,
            use_highly_variable=use_highly_variable, n_top_genes=n_top_genes,
            n_pca_comps=n_pca_comps, n_lsi_comps=n_lsi_comps, gtf_by=gtf_by,
            flavor=flavor, generate_umap=generate_umap,
            rna_sample_column=rna_sample_column, atac_sample_column=atac_sample_column,
            consistency_threshold=consistency_threshold,
            treat_sample_as_batch=treat_sample_as_batch, save_prefix=save_prefix,
            k_neighbors=k_neighbors, use_rep=use_rep, metric=metric,
            use_gpu=use_gpu, verbose=multiomics_verbose,
            plot_columns=plot_columns, output_dir=multiomics_output_dir,
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
            raise ValueError("GLUE integration required before integration preprocessing.")
        current_adata = integrate_preprocess(
            output_dir=multiomics_output_dir, h5ad_path=h5ad_path,
            sample_column=sample_col, modality_col=modality_col,
            min_cells_sample=min_cells_sample, min_cell_gene=min_cell_gene,
            min_features=min_features, pct_mito_cutoff=pct_mito_cutoff,
            exclude_genes=exclude_genes, doublet=doublet,
            verbose=multiomics_verbose,
            rna_sample_meta_file=rna_sample_meta_file,
            atac_sample_meta_file=atac_sample_meta_file,
        )
        results['adata'] = current_adata
        status_flags["multiomics"]["integration_preprocessing"] = True
        if multiomics_verbose:
            print("Integration preprocessing completed successfully")
    else:
        preprocessed_path = f"{multiomics_output_dir}/preprocess/adata_preprocessed.h5ad"
        if os.path.exists(preprocessed_path):
            current_adata = sc.read(preprocessed_path)
            results['adata'] = current_adata
            status_flags["multiomics"]["integration_preprocessing"] = True
            if multiomics_verbose:
                print(f"Loaded preprocessed data from: {preprocessed_path}")
        else:
            raise ValueError(
                "Integration preprocessing required. Set integration_preprocessing=True "
                "or ensure preprocessed data exists.")

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

    # ==================== STEP 3: SAMPLE EMBEDDING ====================
    if derive_sample_embedding:
        if multiomics_verbose:
            print("Step 3: Sample embedding (composition + CMD)...")

        if not status_flags["multiomics"]["integration_preprocessing"]:
            raise ValueError("Integration preprocessing required before sample embedding.")
        if current_adata is None:
            current_adata = ad.read_h5ad(h5ad_path)
            results['adata'] = current_adata

        if celltype_col not in current_adata.obs.columns:
            raise ValueError(
                f"Cell type column '{celltype_col}' not in adata.obs. Run "
                "cell_type_cluster=True or provide pre-typed input.")

        # Multi-omics: GLUE's X_glue serves as both cluster and CMD embedding
        cluster_emb_key = "X_glue" if "X_glue" in current_adata.obsm else "X_pca_harmony"
        cmd_emb_key = cluster_emb_key  # already sample-preserved by GLUE

        if autotune_enable:
            from parameter_selection.autotune import run_autotune
            run_autotune(
                current_adata, multiomics_output_dir,
                sample_col=sample_col,
                celltype_col=celltype_col,
                cluster_emb_key=cluster_emb_key,
                cmd_emb_key=cmd_emb_key,
                modality_col=modality_col,
                batch_col=batch_col,
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
                cmd_emb_key=cmd_emb_key,
                modality_col=modality_col,
                batch_col=batch_col,
                medium_K=sample_embedding_medium_K,
                fine_K=sample_embedding_fine_K,
                cmd_dim_per_cluster=sample_embedding_cmd_dim,
                use_clr=sample_embedding_use_clr,
                use_cmd=sample_embedding_use_cmd,
                block_weights=sample_embedding_block_weights,
                cmd_weight=sample_embedding_cmd_weight,
                pca_components=sample_embedding_pca_components,
                batch_method=sample_embedding_batch_method,
                save=True, verbose=multiomics_verbose,
            )
        status_flags["multiomics"]["derive_sample_embedding"] = True
    else:
        if current_adata is not None and "X_DR_sample" in current_adata.uns:
            status_flags["multiomics"]["derive_sample_embedding"] = True

    print("Multiomics preprocessing pipeline completed successfully!")

    return {
        'adata': current_adata,
        'status_flags': status_flags,
    }
