import os
import time
import contextlib
import io
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
from muon import atac as ac
from harmony import harmonize
from scipy.sparse import issparse

from utils.safe_save import safe_h5ad_write
from utils.random_seed import set_global_seed
from utils.merge_sample_meta import merge_sample_metadata


def _run_lsi_on_hvg(adata, n_comps, drop_first, hvg_col="highly_variable", out_key="X_lsi"):
    """Run LSI on the HVF subset (hvg_col) and write the embedding to adata.obsm[out_key].

    muon's `ac.tl.lsi` doesn't accept `use_highly_variable`, so we run on a
    temporary HVF-only copy. varm['LSI'] and uns['lsi'] are only written for the
    canonical out_key='X_lsi' to avoid clobbering the Z_clust basis.
    """
    hvg_mask = adata.var[hvg_col].values
    if hvg_mask.sum() < n_comps + 1:
        hvg_mask = np.ones(adata.n_vars, dtype=bool)
    sub = adata[:, hvg_mask].copy()
    ac.tl.lsi(sub, n_comps=n_comps)
    if drop_first:
        sub.obsm["X_lsi"] = sub.obsm["X_lsi"][:, 1:]
        sub.varm["LSI"] = sub.varm["LSI"][:, 1:]
        sub.uns["lsi"]["stdev"] = sub.uns["lsi"]["stdev"][1:]
    adata.obsm[out_key] = sub.obsm["X_lsi"]
    if out_key == "X_lsi":
        # Expand varm['LSI'] back to full-feature shape with zeros for non-HVF rows.
        full_varm = np.zeros((adata.n_vars, sub.varm["LSI"].shape[1]),
                              dtype=sub.varm["LSI"].dtype)
        full_varm[np.where(hvg_mask)[0]] = sub.varm["LSI"]
        adata.varm["LSI"] = full_varm
        adata.uns["lsi"] = {"stdev": sub.uns["lsi"]["stdev"]}


def anndata_cluster(
    adata,
    output_dir,
    sample_column="sample",
    num_cell_hvfs=50000,
    cell_embedding_num_PCs=50,
    num_harmony_iterations=30,
    cell_level_batch_key_for_harmony=None,
    cell_level_batch_key_no_sample=None,
    tfidf_scale_factor=1e4,
    log_transform=True,
    drop_first_lsi=True,
    verbose=True,
):
    """ATAC dual LSI-Harmony preprocessing — single saved file.

    Produces:
      - `.layers['counts']`              raw counts
      - `.X`                             TF-IDF + log1p normalized
      - `.var['highly_variable']`        HVF flag (no subsetting)
      - `.obsm['X_lsi']`                 LSI on HVF subset (drop_first applied if set)
      - `.obsm['Z_clust']`         sample-removed Harmony
      - `.obsm['Z_rmd']`  sample-preserved Harmony (RMD)

    Writes a single file: `<output_dir>/adata_preprocessed.h5ad`.
    """
    if verbose:
        print("=== [CPU] Processing ATAC for clustering (dual Harmony) ===")

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    if verbose:
        print("Running TF-IDF normalization...")
    ac.pp.tfidf(adata, scale_factor=tfidf_scale_factor)
    if log_transform:
        if verbose:
            print("Applying log1p transformation...")
        sc.pp.log1p(adata)

    # HVF1: sample-aware — for Z_clust (byte-identical to pre-Batch-2).
    if verbose:
        print("Running HVF1 (sample-aware) selection (flag only, no subset)...")
    sc.pp.highly_variable_genes(
        adata, n_top_genes=num_cell_hvfs, flavor="seurat_v3",
        batch_key=sample_column if sample_column in adata.obs.columns else None,
    )
    n_hvf = int(adata.var["highly_variable"].sum())
    if verbose:
        print(f"After HVF1 (sample-aware) selection: {n_hvf} flagged / {adata.shape[1]} total features")

    if verbose:
        print(f"Running LSI (HVF1) with {cell_embedding_num_PCs} components (drop_first={drop_first_lsi})...")
    _run_lsi_on_hvg(adata, n_comps=cell_embedding_num_PCs, drop_first=drop_first_lsi,
                    hvg_col="highly_variable", out_key="X_lsi")

    # --- Pass 1: sample-removed (byte-identical to pre-Batch-2: nothing new ran before this) ---
    if verbose:
        print("=== [CPU] Harmony pass 1: WITH sample (sample-removed) ===")
        print("  batch keys:", ", ".join(cell_level_batch_key_for_harmony or []))
    if cell_level_batch_key_for_harmony:
        adata.obsm["Z_clust"] = harmonize(
            adata.obsm["X_lsi"], adata.obs,
            batch_key=cell_level_batch_key_for_harmony,
            max_iter_harmony=num_harmony_iterations,
            use_gpu=False,
        )
    else:
        adata.obsm["Z_clust"] = adata.obsm["X_lsi"].copy()

    # RMD basis block — moved here (after Z_clust) so the RNG state consumed by
    # Z_clust Harmony is byte-identical to pre-Batch-2.  HVF2 runs on the same
    # TF-IDF+log1p .X as HVF1 (no counts layer for ATAC).
    adata.var["highly_variable_clust"] = adata.var["highly_variable"].to_numpy().copy()
    if verbose:
        print("Running HVF2 (sample-naive) selection...")
    sc.pp.highly_variable_genes(
        adata, n_top_genes=num_cell_hvfs, flavor="seurat_v3", batch_key=None,
    )
    adata.var["highly_variable_rmd"] = adata.var["highly_variable"].to_numpy().copy()
    adata.var["highly_variable"] = adata.var["highly_variable_clust"]

    if verbose:
        print(f"Running LSI (HVF2/rmd) with {cell_embedding_num_PCs} components...")
    _run_lsi_on_hvg(adata, n_comps=cell_embedding_num_PCs, drop_first=drop_first_lsi,
                    hvg_col="highly_variable_rmd", out_key="X_lsi_rmd")

    # --- Pass 2: sample-preserved (Z_rmd from X_lsi_rmd) ---
    if cell_level_batch_key_no_sample:
        if verbose:
            print("=== [CPU] Harmony pass 2: NO sample (sample-preserved) ===")
            print("  batch keys:", ", ".join(cell_level_batch_key_no_sample))
        adata.obsm["Z_rmd"] = harmonize(
            adata.obsm["X_lsi_rmd"], adata.obs,
            batch_key=cell_level_batch_key_no_sample,
            max_iter_harmony=num_harmony_iterations,
            use_gpu=False,
        )
    else:
        if verbose:
            print("=== [CPU] Harmony pass 2: no extra batch covariate → using raw X_lsi_rmd ===")
        adata.obsm["Z_rmd"] = np.asarray(
            adata.obsm["X_lsi_rmd"], dtype=np.float32)

    if verbose:
        print(f"  Z_clust   shape: {adata.obsm['Z_clust'].shape}")
        print(f"  X_lsi_rmd shape: {adata.obsm['X_lsi_rmd'].shape}")
        print(f"  Z_rmd     shape: {adata.obsm['Z_rmd'].shape}")

    save_path = os.path.join(output_dir, "adata_preprocessed.h5ad")
    safe_h5ad_write(adata, save_path)
    if verbose:
        print(f"Wrote {save_path}")
    return adata


def _flatten_to_strings(values):
    flattened = []
    for value in values:
        if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
            flattened.extend(str(x) for x in value)
        else:
            flattened.append(str(value))
    return flattened


def _ensure_sample_column(adata, sample_column, verbose=True):
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
    cell_embedding_num_PCs=50,
    num_harmony_iterations=30,
    num_cell_hvfs=50000,
    min_cells=1,
    min_features=2000,
    max_features=15000,
    min_cells_per_sample=1,
    exclude_features=None,
    cell_level_batch_key=None,
    doublet_detection=True,
    tfidf_scale_factor=1e4,
    log_transform=True,
    drop_first_lsi=True,
    verbose=True,
):
    """End-to-end CPU ATAC preprocessing — single `adata_preprocessed.h5ad` output."""
    start_time = time.time()
    set_global_seed(seed=42)

    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=== Reading input dataset ===")
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f"Raw shape: {adata.shape[0]} cells × {adata.shape[1]} features")

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
    cell_level_batch_key_for_harmony = flattened_cell_level_batch_key.copy()
    if sample_column not in cell_level_batch_key_for_harmony:
        cell_level_batch_key_for_harmony.append(sample_column)
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
        adata.X = (
            adata.X.astype(np.float32)
            if issparse(adata.X)
            else np.asarray(adata.X, dtype=np.float32)
        )

    if verbose:
        print("=== QC filtering ===")
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(adata, "n_cells_by_counts", lambda x: x >= min_cells)
    mu.pp.filter_obs(adata, "n_genes_by_counts",
                     lambda x: (x >= min_features) & (x <= max_features))
    if verbose:
        print(f"After initial filtering: {adata.shape[0]} cells × {adata.shape[1]} features")

    if doublet_detection and adata.n_vars >= 50:
        try:
            if verbose:
                print("Running doublet detection...")
            with contextlib.redirect_stdout(io.StringIO()):
                n_prin = min(30, adata.n_vars - 1, adata.n_obs - 1)
                sc.pp.scrublet(adata, batch_key=sample_column, n_prin_comps=n_prin)
                n_doublets = adata.obs["predicted_doublet"].sum()
                adata = adata[~adata.obs["predicted_doublet"]].copy()
            if verbose:
                print(f"After doublet removal: {adata.shape[0]} cells (removed {n_doublets} doublets)")
        except (ValueError, RuntimeError) as e:
            if verbose:
                print(f"Scrublet failed ({e}) — continuing without doublet removal.")

    if exclude_features:
        adata = adata[:, ~adata.var_names.isin(exclude_features)].copy()
        if verbose:
            print(f"After feature exclusion: {adata.shape[0]} cells × {adata.shape[1]} features")

    cells_per_sample = adata.obs.groupby(sample_column).size()
    samples_to_keep = cells_per_sample[cells_per_sample >= min_cells_per_sample].index
    adata = adata[adata.obs[sample_column].isin(samples_to_keep)].copy()
    if verbose:
        print(f"After sample filtering: {adata.shape[0]} cells × {adata.shape[1]} features")
        print(f"Samples remaining: {len(samples_to_keep)}")

    min_cells_for_feature = int(0.001 * adata.n_obs)
    sc.pp.filter_genes(adata, min_cells=min_cells_for_feature)
    if verbose:
        print(f"Final shape: {adata.shape[0]} cells × {adata.shape[1]} features")
        print("QC filtering complete!")

    adata = anndata_cluster(
        adata=adata, output_dir=output_dir,
        sample_column=sample_column, num_cell_hvfs=num_cell_hvfs,
        cell_embedding_num_PCs=cell_embedding_num_PCs,
        num_harmony_iterations=num_harmony_iterations,
        cell_level_batch_key_for_harmony=cell_level_batch_key_for_harmony,
        cell_level_batch_key_no_sample=cell_level_batch_key_no_sample,
        tfidf_scale_factor=tfidf_scale_factor,
        log_transform=log_transform,
        drop_first_lsi=drop_first_lsi,
        verbose=verbose,
    )

    if verbose:
        print(f"Total runtime: {time.time() - start_time:.2f} seconds")

    return adata
