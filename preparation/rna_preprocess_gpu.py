import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
import rapids_singlecell as rsc
from harmony import harmonize
from scipy.sparse import issparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.safe_save import safe_h5ad_write
from utils.random_seed import set_global_seed
from utils.merge_sample_meta import merge_sample_metadata


def anndata_cluster(
    adata,
    output_dir,
    sample_column="sample",
    num_cell_hvgs=2000,
    cell_embedding_num_PCs=20,
    num_harmony_iterations=30,
    cell_level_batch_key_for_harmony=None,
    cell_level_batch_key_no_sample=None,
    verbose=True,
):
    """GPU dual-Harmony preprocessing — single saved file.

    Produces TWO cell-level Harmony embeddings (one preprocessing pass):
      - obsm['Z_clust']        — Harmony with `cell_level_batch_key_for_harmony`
                                         (typically batch + sample → sample-removed)
      - obsm['Z_rmd'] — Harmony with `cell_level_batch_key_no_sample`
                                         (no sample → sample-preserved, used by RMD)

    Keeps all genes in `.X` (post normalize+log1p); HVG selection is recorded as
    `.var['highly_variable']` but does not subset `.X`. Raw counts are preserved
    in `.layers['counts']`.

    Writes a single file: `<output_dir>/adata_preprocessed.h5ad`.
    """
    if verbose:
        print("=== [GPU] Processing data for clustering (dual Harmony) ===")

    rsc.get.anndata_to_CPU(adata)

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # HVG1: sample-aware — same span-retry loop as before.
    if verbose:
        print("Running HVG1 (sample-aware) selection on CPU (Seurat v3)...")
    hvg_spans = [0.3, 0.5, 0.8, 1.0]
    for attempt, span in enumerate(hvg_spans):
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=num_cell_hvgs, flavor="seurat_v3",
                batch_key=sample_column if sample_column in adata.obs.columns else None,
                span=span,
            )
            if verbose and attempt > 0:
                print(f"HVG1 selection succeeded with span={span}")
            break
        except ValueError as e:
            arg = e.args[0] if e.args else ""
            msg = arg.decode("utf-8", "ignore") if isinstance(arg, bytes) else str(arg)
            if "reciprocal condition number" not in msg:
                raise
            if attempt == len(hvg_spans) - 1:
                raise
            if verbose:
                print(f"HVG1 LOESS failed with span={span} ({msg}); retrying span={hvg_spans[attempt + 1]}")
    n_hvg = int(adata.var["highly_variable"].sum())
    if verbose:
        print(f"After HVG1 (sample-aware) selection: {n_hvg} flagged / {adata.shape[1]} total genes")

    if verbose:
        print("Normalization, log1p, PCA on CPU (scanpy ARPACK); Harmony on GPU...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # PCA on HVG1 (sample-aware) — identical to pre-Batch-2 recipe.
    sc.tl.pca(adata, n_comps=cell_embedding_num_PCs,
              svd_solver="arpack", use_highly_variable=True)

    # --- Pass 1: sample-removed (byte-identical to pre-Batch-2: nothing new ran before this) ---
    if verbose:
        print("=== [GPU] Harmony pass 1: WITH sample (sample-removed) ===")
        print("  batch keys:", ", ".join(cell_level_batch_key_for_harmony or []))
    adata.obsm["Z_clust"] = harmonize(
        adata.obsm["X_pca"], adata.obs,
        batch_key=cell_level_batch_key_for_harmony,
        max_iter_harmony=num_harmony_iterations,
        use_gpu=True,
    )

    # RMD basis block — moved here (after Z_clust) so the RNG state consumed by
    # Z_clust Harmony is byte-identical to pre-Batch-2.  HVG2 runs on the counts
    # layer because .X is now normalized.
    if verbose:
        print("Running HVG2 (sample-naive) selection on CPU (Seurat v3, layer='counts')...")
    adata.var["highly_variable_clust"] = adata.var["highly_variable"].to_numpy().copy()
    hvg_spans2 = [0.3, 0.5, 0.8, 1.0]
    last_err2 = None
    for attempt, span in enumerate(hvg_spans2):
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=num_cell_hvgs, flavor="seurat_v3",
                layer="counts", batch_key=None, span=span,
            )
            if verbose and attempt > 0:
                print(f"HVG2 selection succeeded with span={span}")
            break
        except ValueError as e:
            arg = e.args[0] if e.args else ""
            msg = arg.decode("utf-8", "ignore") if isinstance(arg, bytes) else str(arg)
            if "reciprocal condition number" not in msg:
                raise
            last_err2 = e
            if attempt == len(hvg_spans2) - 1:
                raise last_err2
            if verbose:
                print(f"HVG2 LOESS failed with span={span} ({msg}); retrying span={hvg_spans2[attempt + 1]}")
    adata.var["highly_variable_rmd"] = adata.var["highly_variable"].to_numpy().copy()
    adata.var["highly_variable"] = adata.var["highly_variable_clust"]

    sub = adata[:, adata.var["highly_variable_rmd"].to_numpy()].copy()
    sc.tl.pca(sub, n_comps=cell_embedding_num_PCs, svd_solver="arpack",
              use_highly_variable=False)
    adata.obsm["X_pca_rmd"] = sub.obsm["X_pca"]

    # --- Pass 2: sample-preserved ---
    if cell_level_batch_key_no_sample:
        if verbose:
            print("=== [GPU] Harmony pass 2: NO sample (sample-preserved) ===")
            print("  batch keys:", ", ".join(cell_level_batch_key_no_sample))
        adata.obsm["Z_rmd"] = harmonize(
            adata.obsm["X_pca_rmd"], adata.obs,
            batch_key=cell_level_batch_key_no_sample,
            max_iter_harmony=num_harmony_iterations,
            use_gpu=True,
        )
    else:
        if verbose:
            print("=== [GPU] Harmony pass 2: no extra batch covariate → using raw X_pca_rmd ===")
        adata.obsm["Z_rmd"] = np.asarray(
            adata.obsm["X_pca_rmd"], dtype=np.float32)

    if verbose:
        print(f"  Z_clust   shape: {adata.obsm['Z_clust'].shape}")
        print(f"  X_pca_rmd shape: {adata.obsm['X_pca_rmd'].shape}")
        print(f"  Z_rmd     shape: {adata.obsm['Z_rmd'].shape}")

    rsc.get.anndata_to_CPU(adata)
    save_path = os.path.join(output_dir, "adata_preprocessed.h5ad")
    safe_h5ad_write(adata, save_path)
    if verbose:
        print(f"Wrote {save_path}")

    return adata


def preprocess_gpu(
    h5ad_path,
    sample_meta_path,
    output_dir,
    cell_meta_path=None,
    sample_column="sample",
    sample_level_batch_key="batch",
    cell_embedding_num_PCs=20,
    num_harmony_iterations=30,
    num_cell_hvgs=2000,
    min_cells=500,
    min_genes=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    cell_level_batch_key=None,
    verbose=True,
):
    """GPU-accelerated end-to-end RNA preprocessing.

    Produces a single `adata_preprocessed.h5ad` with:
      - `.X`               normalized + log1p expression (all genes)
      - `.layers['counts']` raw counts
      - `.var['highly_variable']` HVG flag
      - `.obsm['X_pca']`         PCA on HVG subset
      - `.obsm['Z_clust']`        sample-removed Harmony
      - `.obsm['Z_rmd']` sample-preserved Harmony (used by RMD)
    """
    set_global_seed(seed=42)
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=== Reading input dataset ===")
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f"Raw shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if cell_meta_path is None:
        if sample_column not in adata.obs.columns:
            if verbose:
                print(f"No '{sample_column}' column in adata.obs; inferring from obs_names")
            adata.obs[sample_column] = adata.obs_names.str.split(":").str[0]
    else:
        if verbose:
            print(f"   Merging cell-level metadata from: {cell_meta_path}")
        cell_metadata = pd.read_csv(cell_meta_path).set_index("barcode")
        adata.obs = adata.obs.join(cell_metadata, how="left")
        if sample_column not in adata.obs.columns:
            adata.obs[sample_column] = adata.obs_names.str.split(":").str[0]

    if sample_meta_path is not None:
        if verbose:
            print("=== Merging sample-level metadata into adata.obs ===")
        adata = merge_sample_metadata(
            adata=adata, metadata_path=sample_meta_path,
            sample_column=sample_column, verbose=verbose,
        )

    cell_level_batch_key = cell_level_batch_key or []
    flattened_cell_level_batch_key = []
    for var in cell_level_batch_key:
        if isinstance(var, (list, tuple, np.ndarray, pd.Index)):
            flattened_cell_level_batch_key.extend(map(str, list(var)))
        else:
            flattened_cell_level_batch_key.append(str(var))

    # Pass 1 (sample-removed): include sample.
    cell_level_batch_key_for_harmony = flattened_cell_level_batch_key.copy()
    if sample_column not in cell_level_batch_key_for_harmony:
        cell_level_batch_key_for_harmony.append(sample_column)
    # Pass 2 (sample-preserved): exclude sample.
    cell_level_batch_key_no_sample = [
        k for k in flattened_cell_level_batch_key if k != sample_column
    ]

    flattened_sample_level_batch_keys = []
    if sample_level_batch_key:
        if isinstance(sample_level_batch_key, (list, tuple, np.ndarray, pd.Index)):
            flattened_sample_level_batch_keys.extend(map(str, list(sample_level_batch_key)))
        else:
            flattened_sample_level_batch_keys.append(str(sample_level_batch_key))

    required_columns = list(
        dict.fromkeys(flattened_cell_level_batch_key + flattened_sample_level_batch_keys)
    )
    missing_columns = sorted(set(required_columns) - set(map(str, adata.obs.columns)))
    if missing_columns:
        raise KeyError(f"The following variables are missing from adata.obs: {missing_columns}")
    if verbose:
        print("All required columns are present in adata.obs.")

    if adata.X.dtype != np.float32:
        adata.X = (
            adata.X.astype(np.float32)
            if issparse(adata.X)
            else np.asarray(adata.X, dtype=np.float32)
        )

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if verbose:
        print(f"After initial filtering: {adata.shape[0]} cells × {adata.shape[1]} genes")

    mito_gene_mask = adata.var_names.str.startswith(("MT-", "mt-"))
    adata.var["mt"] = mito_gene_mask
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], log1p=False, inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < pct_mito_cutoff].copy()
    if verbose:
        print(f"After mitochondrial filtering: {adata.shape[0]} cells × {adata.shape[1]} genes")

    mito_genes = adata.var_names[adata.var_names.str.startswith("MT-")]
    genes_to_exclude = set(mito_genes) | set(exclude_genes or [])
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()
    if verbose:
        print(f"After gene exclusion: {adata.shape[0]} cells × {adata.shape[1]} genes")

    adata = anndata_cluster(
        adata=adata, output_dir=output_dir,
        sample_column=sample_column, num_cell_hvgs=num_cell_hvgs,
        cell_embedding_num_PCs=cell_embedding_num_PCs,
        num_harmony_iterations=num_harmony_iterations,
        cell_level_batch_key_for_harmony=cell_level_batch_key_for_harmony,
        cell_level_batch_key_no_sample=cell_level_batch_key_no_sample,
        verbose=verbose,
    )

    if verbose:
        print(f"Total runtime: {time.time() - start_time:.2f} seconds")

    return adata
