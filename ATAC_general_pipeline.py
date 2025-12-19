import os, time, warnings
from   datetime import datetime
import contextlib, io
import numpy  as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from harmony import harmonize
import muon as mu
from muon import atac as ac
from ATAC_cell_type import *
warnings.filterwarnings("ignore")


def log(msg, level="INFO", verbose=True):
    """Timestamped logger."""
    if verbose:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {level}: {msg}")


def merge_sample_metadata(
    adata, metadata_path, sample_column="sample", sep=",", verbose=True
):
    meta = pd.read_csv(metadata_path, sep=sep).set_index(sample_column)
    adata.obs = adata.obs.join(meta, on=sample_column)
    if verbose:
        print(f"Merged {meta.shape[1]} sample-level columns")
    return adata


def run_scatac_pipeline(
    filepath,
    output_dir,
    metadata_path=None,
    sample_column="sample",
    batch_key=None,  # Can be a string or list of strings
    verbose=True,
    # QC and filtering parameters
    min_cells=1,
    min_genes=2000,
    max_genes=15000,
    min_cells_per_sample=1,
    # Doublet detection
    doublet=True,
    # TF-IDF parameters
    tfidf_scale_factor=1e4,
    # Log transformation
    log_transform=True,
    # HVF parameters
    num_features=50000,
    # LSI / dimensionality reduction
    n_lsi_components=50,
    drop_first_lsi=True,
    # Harmony
    harmony_max_iter=30,
    harmony_use_gpu=True,
    # Neighbours
    n_neighbors=10,
    n_pcs=30,
    # UMAP
    umap_min_dist=0.3,
    umap_spread=1.0,
    umap_random_state=0,
):
    t0 = time.time()
    from utils.random_seed import set_global_seed
    set_global_seed(seed=42, verbose=verbose)
    log("="*60 + "\nStarting scATAC-seq pipeline\n" + "="*60, verbose)

    # Create sub-folder
    output_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(output_dir, exist_ok=True)

    # Normalize batch_key to list format
    if batch_key is not None:
        batch_keys = batch_key if isinstance(batch_key, list) else [batch_key]
    else:
        batch_keys = None

    # 1. Load data
    atac = sc.read_h5ad(filepath)
    log(f"Loaded {atac.n_obs} cells × {atac.n_vars} features", verbose)

    # 2. Sample metadata
    if metadata_path:
        atac = merge_sample_metadata(atac, metadata_path, sample_column, verbose=verbose)

    # 3. QC filtering
    log("QC filtering", verbose)
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(atac, "n_cells_by_counts", lambda x: x >= min_cells)
    mu.pp.filter_obs(atac, "n_genes_by_counts",
                     lambda x: (x >= min_genes) & (x <= max_genes))

    # 3b. Doublet detection (scanpy Scrublet)
    if doublet:
        if atac.n_vars >= 50:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    n_prin = min(30, atac.n_vars - 1, atac.n_obs - 1)
                    sc.pp.scrublet(atac, batch_key=sample_column, n_prin_comps=n_prin)
                    atac = atac[~atac.obs["predicted_doublet"]].copy()
                log("Doublets removed", verbose)
            except (ValueError, RuntimeError) as e:
                log(f"Scrublet failed ({e}) – continuing.", verbose)

    # 3c. Sample-level filtering
    log(f"Filtering samples with <{min_cells_per_sample} cells", verbose)
    counts = atac.obs[sample_column].value_counts()
    atac = atac[counts.loc[atac.obs[sample_column]].values >= min_cells_per_sample].copy()
    log(f"Remaining cells: {atac.n_obs}", verbose)

    atac_sample = atac.copy()

    # 4. TF-IDF
    log("TF-IDF normalisation", verbose)
    ac.pp.tfidf(atac, scale_factor=tfidf_scale_factor)

    if log_transform:
        log("Log1p transform", verbose)
        sc.pp.log1p(atac)

    # 5. HVF + LSI
    log("Selecting HVFs", verbose)
    # For HVG selection, use the first batch key (scanpy HVG doesn't support multiple batches)
    hvg_batch_key = batch_keys[0] if batch_keys else None
    sc.pp.highly_variable_genes(
        atac, n_top_genes=num_features, flavor="seurat_v3", batch_key=hvg_batch_key
    )
    atac.var["HVF"] = atac.var["highly_variable"]

    log("Running LSI", verbose)
    ac.tl.lsi(atac, n_comps=n_lsi_components)
    if drop_first_lsi:
        atac.obsm["X_lsi"] = atac.obsm["X_lsi"][:, 1:]
        atac.varm["LSI"]   = atac.varm["LSI"][:, 1:]
        atac.uns["lsi"]["stdev"] = atac.uns["lsi"]["stdev"][1:]
    dimred_key = "X_lsi"

    # 6. Harmony (supports multiple batch keys)
    if batch_keys:
        log(f"Harmony batch correction with batch keys: {batch_keys}", verbose)
        Z = harmonize(
            atac.obsm[dimred_key],
            atac.obs,
            batch_key=batch_keys,
            max_iter_harmony=harmony_max_iter,
            use_gpu=harmony_use_gpu,
        )
        atac.obsm["X_DM_harmony"] = Z
    else:
        atac.obsm["X_DM_harmony"] = atac.obsm[dimred_key].copy()
    use_rep = "X_DM_harmony"

    # 7. Neighbours / UMAP
    n_pcs = min(n_pcs, atac.obsm[use_rep].shape[1])
    sc.pp.neighbors(atac, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
    sc.tl.umap(
        atac,
        min_dist=umap_min_dist,
        spread=umap_spread,
        random_state=umap_random_state,
    )

    atac = clean_obs_for_saving(atac, verbose=verbose)
    sc.write(os.path.join(output_dir, "adata_cell.h5ad"), atac)
    log("Saved adata_cell.h5ad", verbose)

    # 10. Save sample-level object with copied embeddings
    log("Writing H5AD …", verbose)
    atac_sample.obsm["X_DM_harmony"] = atac.obsm["X_DM_harmony"].copy()
    atac_sample.obsm["X_umap"] = atac.obsm["X_umap"].copy()
    atac_sample = clean_obs_for_saving(atac_sample, verbose=verbose)
    sc.write(os.path.join(output_dir, "adata_sample.h5ad"), atac_sample)

    # 11. Summary
    log("="*60, verbose)
    log(f"Finished in {(time.time() - t0) / 60:.1f} min", verbose)
    return atac_sample, atac