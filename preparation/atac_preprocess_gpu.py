import os
import sys
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.safe_save import safe_h5ad_write
from utils.random_seed import set_global_seed
from utils.merge_sample_meta import merge_sample_metadata


def _run_lsi_on_hvg(adata, n_comps, drop_first):
    """Run LSI on the HVG subset and write the embedding back to the full adata."""
    hvg_mask = adata.var["highly_variable"].values
    if hvg_mask.sum() < n_comps + 1:
        hvg_mask = np.ones(adata.n_vars, dtype=bool)
    sub = adata[:, hvg_mask].copy()
    ac.tl.lsi(sub, n_comps=n_comps)
    if drop_first:
        sub.obsm["X_lsi"] = sub.obsm["X_lsi"][:, 1:]
        sub.varm["LSI"] = sub.varm["LSI"][:, 1:]
        sub.uns["lsi"]["stdev"] = sub.uns["lsi"]["stdev"][1:]
    adata.obsm["X_lsi"] = sub.obsm["X_lsi"]
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
    """ATAC GPU dual LSI-Harmony preprocessing — single saved file.

    Produces:
      - `.layers['counts']`              raw counts
      - `.X`                             TF-IDF + log1p normalized
      - `.var['highly_variable']`        HVF flag (no subsetting)
      - `.obsm['X_lsi']`                 LSI on HVF subset (drop_first applied if set)
      - `.obsm['Z_clust']`         sample-removed Harmony (GPU)
      - `.obsm['Z_cmd']`  sample-preserved Harmony (CMD)

    Writes a single file: `<output_dir>/adata_preprocessed.h5ad`.
    """
    if verbose:
        print("=== [GPU] Processing ATAC for clustering (dual Harmony) ===")

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    if verbose:
        print("Running TF-IDF normalization...")
    ac.pp.tfidf(adata, scale_factor=tfidf_scale_factor)
    if log_transform:
        if verbose:
            print("Applying log1p transformation...")
        sc.pp.log1p(adata)

    if verbose:
        print("Running HVF selection (flag only, no subset)...")
    sc.pp.highly_variable_genes(
        adata, n_top_genes=num_cell_hvfs, flavor="seurat_v3",
        batch_key=sample_column if sample_column in adata.obs.columns else None,
    )
    n_hvf = int(adata.var["highly_variable"].sum())
    if verbose:
        print(f"After HVF selection: {n_hvf} flagged / {adata.shape[1]} total features")

    if verbose:
        print(f"Running LSI with {cell_embedding_num_PCs} components (drop_first={drop_first_lsi})...")
    _run_lsi_on_hvg(adata, n_comps=cell_embedding_num_PCs, drop_first=drop_first_lsi)

    # --- Pass 1: sample-removed ---
    if verbose:
        print("=== [GPU] Harmony pass 1: WITH sample (sample-removed) ===")
        print("  batch keys:", ", ".join(cell_level_batch_key_for_harmony or []))
    if cell_level_batch_key_for_harmony:
        adata.obsm["Z_clust"] = harmonize(
            adata.obsm["X_lsi"], adata.obs,
            batch_key=cell_level_batch_key_for_harmony,
            max_iter_harmony=num_harmony_iterations,
            use_gpu=True,
        )
    else:
        adata.obsm["Z_clust"] = adata.obsm["X_lsi"].copy()

    # --- Pass 2: sample-preserved ---
    if cell_level_batch_key_no_sample:
        if verbose:
            print("=== [GPU] Harmony pass 2: NO sample (sample-preserved) ===")
            print("  batch keys:", ", ".join(cell_level_batch_key_no_sample))
        adata.obsm["Z_cmd"] = harmonize(
            adata.obsm["X_lsi"], adata.obs,
            batch_key=cell_level_batch_key_no_sample,
            max_iter_harmony=num_harmony_iterations,
            use_gpu=True,
        )
    else:
        if verbose:
            print("=== [GPU] Harmony pass 2: no extra batch covariate → using raw X_lsi ===")
        adata.obsm["Z_cmd"] = np.asarray(
            adata.obsm["X_lsi"], dtype=np.float32)

    if verbose:
        print(f"  Z_clust        shape: {adata.obsm['Z_clust'].shape}")
        print(f"  Z_cmd shape: {adata.obsm['Z_cmd'].shape}")

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
    """End-to-end GPU ATAC preprocessing — single `adata_preprocessed.h5ad` output."""
    set_global_seed(seed=42)
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=== Reading input dataset ===")
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f"Raw shape: {adata.shape[0]} cells × {adata.shape[1]} features")

    if cell_meta_path is None:
        if sample_column not in adata.obs.columns:
            if verbose:
                print(f"   No '{sample_column}' column in adata.obs; inferring from obs_names")
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

    cell_level_batch_key_for_harmony = flattened_cell_level_batch_key.copy()
    if sample_column not in cell_level_batch_key_for_harmony:
        cell_level_batch_key_for_harmony.append(sample_column)
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

    if verbose:
        print("=== QC filtering ===")
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(adata, "n_cells_by_counts", lambda x: x >= min_cells)
    mu.pp.filter_obs(adata, "n_genes_by_counts",
                     lambda x: (x >= min_features) & (x <= max_features))
    if verbose:
        print(f"After QC filtering: {adata.shape[0]} cells × {adata.shape[1]} features")

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
                print(f"Removed {n_doublets} doublets")
        except (ValueError, RuntimeError) as e:
            if verbose:
                print(f"Scrublet failed ({e}) — continuing without doublet removal.")

    counts = adata.obs[sample_column].value_counts()
    adata = adata[counts.loc[adata.obs[sample_column]].values >= min_cells_per_sample].copy()
    if verbose:
        print(f"After sample filtering: {adata.shape[0]} cells")

    if exclude_features:
        adata = adata[:, ~adata.var_names.isin(exclude_features)].copy()
        if verbose:
            print(f"After excluding features: {adata.shape[1]} features remaining")

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
