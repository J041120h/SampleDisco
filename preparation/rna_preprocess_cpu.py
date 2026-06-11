import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
from harmony import harmonize
from scipy.sparse import issparse

from utils.safe_save import safe_h5ad_write
from utils.random_seed import set_global_seed
from utils.merge_sample_meta import merge_sample_metadata


def _hvg_with_retries(adata, n_top_genes, sample_column):
    """Seurat-v3 HVG with span retries (LOESS sometimes fails on small data)."""
    hvg_spans = [0.3, 0.5, 0.8, 1.0]
    last_err = None
    for attempt, span in enumerate(hvg_spans):
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_top_genes, flavor="seurat_v3",
                batch_key=sample_column if sample_column in adata.obs.columns else None,
                span=span,
            )
            return
        except ValueError as exc:
            arg = exc.args[0] if exc.args else ""
            msg = arg.decode("utf-8", "ignore") if isinstance(arg, bytes) else str(arg)
            if "reciprocal condition number" not in msg:
                raise
            last_err = exc
    raise last_err if last_err else RuntimeError("HVG selection failed")


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
    """Process AnnData for clustering — dual-Harmony, single saved file.

    Produces TWO cell-level Harmony embeddings on `adata`:
      - obsm['Z_clust']        — Harmony with `cell_level_batch_key_for_harmony`
                                         (typically batch + sample → sample-removed)
      - obsm['Z_rmd'] — Harmony with `cell_level_batch_key_no_sample`
                                         (no sample → sample-preserved, used by RMD)

    Keeps all genes in `.X` (post normalize+log1p); HVG selection is recorded as a
    flag in `.var['highly_variable']` but does not subset `.X`. Raw counts are
    preserved in `.layers['counts']`.

    Writes a single file: `<output_dir>/adata_preprocessed.h5ad`.
    """
    if verbose:
        print("=== [CPU] Processing data for clustering (dual Harmony) ===")

    # Preserve raw counts as a layer (before normalization).
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # HVG selection on raw counts (Seurat v3 wants counts) — flag only, no subset.
    _hvg_with_retries(adata, n_top_genes=num_cell_hvgs, sample_column=sample_column)
    n_hvg = int(adata.var["highly_variable"].sum())
    if verbose:
        print(f"After HVG selection: {n_hvg} flagged / {adata.shape[1]} total genes")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # PCA on HVG subset only, then write embedding back to the full-gene adata.
    sc.tl.pca(adata, n_comps=cell_embedding_num_PCs, svd_solver="arpack",
              use_highly_variable=True)

    # --- Pass 1: sample-removed (matches original recipe) ---
    if verbose:
        print("=== [CPU] Harmony pass 1: WITH sample (sample-removed) ===")
        print("  batch keys:", ", ".join(cell_level_batch_key_for_harmony or []))
    adata.obsm["Z_clust"] = harmonize(
        adata.obsm["X_pca"], adata.obs,
        batch_key=cell_level_batch_key_for_harmony,
        max_iter_harmony=num_harmony_iterations,
        use_gpu=False,
    )

    # --- Pass 2: sample-preserved (used by RMD displacement) ---
    if cell_level_batch_key_no_sample:
        if verbose:
            print("=== [CPU] Harmony pass 2: NO sample (sample-preserved) ===")
            print("  batch keys:", ", ".join(cell_level_batch_key_no_sample))
        adata.obsm["Z_rmd"] = harmonize(
            adata.obsm["X_pca"], adata.obs,
            batch_key=cell_level_batch_key_no_sample,
            max_iter_harmony=num_harmony_iterations,
            use_gpu=False,
        )
    else:
        if verbose:
            print("=== [CPU] Harmony pass 2: no extra batch covariate → using raw X_pca ===")
        adata.obsm["Z_rmd"] = np.asarray(
            adata.obsm["X_pca"], dtype=np.float32)

    if verbose:
        print(f"  Z_clust        shape: {adata.obsm['Z_clust'].shape}")
        print(f"  Z_rmd shape: {adata.obsm['Z_rmd'].shape}")

    save_path = os.path.join(output_dir, "adata_preprocessed.h5ad")
    safe_h5ad_write(adata, save_path)
    if verbose:
        print(f"Wrote {save_path}")

    return adata


def _flatten_to_strings(values):
    """Flatten nested iterables to a list of strings."""
    flattened = []
    for value in values:
        if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
            flattened.extend(str(x) for x in value)
        else:
            flattened.append(str(value))
    return flattened


def _ensure_sample_column(adata, sample_column, verbose=True):
    """Infer sample column from obs_names if not present."""
    if sample_column not in adata.obs.columns:
        if verbose:
            print(f"   No '{sample_column}' column in adata.obs; inferring from obs_names")
        adata.obs[sample_column] = adata.obs_names.str.split(":").str[0]


def preprocess(
    h5ad_path,
    sample_meta_path,
    output_dir,
    sample_column="sample",
    cell_meta_path=None,
    sample_level_batch_key=None,
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
    """End-to-end CPU preprocessing for single-cell RNA-seq.

    Produces a single `adata_preprocessed.h5ad` with:
      - `.X`               normalized + log1p expression (all genes)
      - `.layers['counts']` original raw counts
      - `.var['highly_variable']` HVG flag (no subsetting)
      - `.obsm['X_pca']`         PCA on HVG subset
      - `.obsm['Z_clust']`        Harmony pass 1 (sample-removed)
      - `.obsm['Z_rmd']` Harmony pass 2 (sample-preserved; used by RMD)

    Returns the AnnData (no separate `adata_sample_diff` is produced).
    """
    start_time = time.time()
    set_global_seed(seed=42)

    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=== Reading input dataset ===")
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f"Raw shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if cell_meta_path is None:
        _ensure_sample_column(adata, sample_column, verbose)
    else:
        if verbose:
            print(f"   Merging cell-level metadata from: {cell_meta_path}")
        cell_metadata = pd.read_csv(cell_meta_path).set_index("barcode")
        adata.obs = adata.obs.join(cell_metadata, how="left")
        _ensure_sample_column(adata, sample_column, verbose)

    if sample_meta_path is not None:
        if verbose:
            print("=== Merging sample-level metadata into adata.obs ===")
        adata = merge_sample_metadata(
            adata=adata, metadata_path=sample_meta_path,
            sample_column=sample_column, verbose=verbose,
        )

    flattened_cell_level_batch_key = _flatten_to_strings(cell_level_batch_key or [])
    # Pass 1 (sample-removed): include sample.
    cell_level_batch_key_for_harmony = flattened_cell_level_batch_key.copy()
    if sample_column not in cell_level_batch_key_for_harmony:
        cell_level_batch_key_for_harmony.append(sample_column)
    # Pass 2 (sample-preserved): exclude sample.
    cell_level_batch_key_no_sample = [
        k for k in flattened_cell_level_batch_key if k != sample_column
    ]

    flattened_sample_level_batch_keys = _flatten_to_strings(
        [sample_level_batch_key] if sample_level_batch_key else []
    )
    required_columns = list(
        dict.fromkeys(flattened_cell_level_batch_key + flattened_sample_level_batch_keys)
    )
    if required_columns:
        missing = sorted(set(required_columns) - set(adata.obs.columns.astype(str)))
        if missing:
            raise KeyError(f"The following variables are missing from adata.obs: {missing}")
    if verbose:
        print("All required columns are present in adata.obs.")

    if adata.X.dtype != np.float32:
        if verbose:
            print(f"Converting adata.X from {adata.X.dtype} to float32")
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
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                 log1p=False, inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < pct_mito_cutoff].copy()
    if verbose:
        print(f"After mitochondrial filtering: {adata.shape[0]} cells × {adata.shape[1]} genes")

    mito_genes = adata.var_names[adata.var_names.str.startswith("MT-")]
    genes_to_exclude = set(mito_genes) | set(exclude_genes or [])
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()
    if verbose:
        print(f"After gene exclusion: {adata.shape[0]} cells × {adata.shape[1]} genes")

    cells_per_sample = adata.obs.groupby(sample_column).size()
    samples_to_keep = cells_per_sample[cells_per_sample >= min_cells].index
    adata = adata[adata.obs[sample_column].isin(samples_to_keep)].copy()
    if verbose:
        print(f"After sample filtering: {adata.shape[0]} cells × {adata.shape[1]} genes")
        print(f"Samples remaining: {len(samples_to_keep)}")

    min_cells_for_gene = int(0.001 * adata.n_obs)
    sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)
    if verbose:
        print(f"Final shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
        print("Preprocessing complete!")

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
