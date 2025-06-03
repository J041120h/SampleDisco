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
import scipy.sparse as sp           # used for re-sparsifying
from sklearn.neighbors import NearestNeighbors
from ATAC_cell_type import *
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#                               Utility helpers                               #
# --------------------------------------------------------------------------- #

def log(msg, level="INFO", verbose=True):
    """Timestamped logger."""
    if verbose:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {level}: {msg}")

# --------------------------------------------------------------------------- #
#                          Metadata / I/O convenience                         #
# --------------------------------------------------------------------------- #

def merge_sample_metadata(
    adata, metadata_path, sample_column="sample", sep=",", verbose=True
):
    meta = pd.read_csv(metadata_path, sep=sep).set_index(sample_column)
    adata.obs = adata.obs.join(meta, on=sample_column)
    if verbose:
        print(f"Merged {meta.shape[1]} sample-level columns")
    return adata


# --------------------------------------------------------------------------- #
#                       snapATAC2 dimensionality reduction                    #
# --------------------------------------------------------------------------- #
def snapatac2_dimensionality_reduction(
    adata,
    n_components=50,
    num_features=50000,
    doublet=True,
    verbose=True
):
    """
    Use snapATAC2 only for dimensionality reduction (SVD / spectral).
    Assumes data has already been processed with TF-IDF and feature selection.
    """
    try:
        import snapatac2 as snap
    except ImportError:
        raise ImportError("snapATAC2 is required. Install with: pip install snapatac2")

    log(f"Running snapATAC2 SVD with {n_components} components", verbose=verbose)

    # Convert to float32 but keep sparse structure if present
    if adata.X.dtype != np.float32:
        log("Converting data matrix to float32", verbose=verbose)
        if sp.issparse(adata.X):
            adata.X = adata.X.astype(np.float32)
        else:
            adata.X = sp.csr_matrix(adata.X.astype(np.float32))

    # Feature selection
    snap.pp.select_features(adata, n_features=num_features)

    # Doublet filtering
    if doublet:
        log("Filtering doublets (snapATAC2-Scrublet)", verbose=verbose)
        try:
            snap.pp.scrublet(adata)
            snap.pp.filter_doublets(adata)
        except (AttributeError, TypeError) as e:
            if "nonzero" in str(e):
                log("snapATAC2-Scrublet compatibility issue – skipping.", verbose=verbose)
            else:
                raise e

    # Save HVF
    if 'selected' in adata.var.columns:
        adata.var['HVF'] = adata.var['selected']

    # Spectral decomposition
    try:
        snap.tl.spectral(adata, n_comps=n_components)
    except RuntimeError as e:
        if "Cannot convert CsrMatrix" in str(e):
            if sp.issparse(adata.X):
                adata.X = adata.X.astype(np.float32)
            else:
                adata.X = sp.csr_matrix(adata.X.astype(np.float32))
            snap.tl.spectral(adata, n_comps=n_components)
        else:
            raise e

    log("snapATAC2 dimensionality reduction complete", verbose=verbose)

    # ---------- NEW: convert back to sparse if snapATAC2 returned dense -------- #
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
        log("Re-sparsified adata.X after snapATAC2", verbose=verbose)
    # -------------------------------------------------------------------------- #

    return adata


# --------------------------------------------------------------------------- #
#                             Main analysis pipeline                          #
# --------------------------------------------------------------------------- #
def run_scatac_pipeline(
    filepath,
    output_dir,
    metadata_path=None,
    sample_column="sample",
    batch_key=None,
    verbose=True,
    use_snapatac2_dimred=False,
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
    #Cell type clustering
    existing_cell_types = False,
    n_target_clusters = 3,
    cluster_resolution= 0.8,
    # UMAP
    umap_min_dist=0.3,
    umap_spread=1.0,
    umap_random_state=0,
    # Output
    output_subdirectory='harmony',
    plot_dpi=300,
    # Additional
    cell_type_column='cell_type'
):
    t0 = time.time()
    log("="*60 + "\nStarting scATAC-seq pipeline\n" + "="*60, verbose)

    # Create sub-folder
    output_dir = os.path.join(output_dir, output_subdirectory)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    atac = sc.read_h5ad(filepath)
    log(f"Loaded {atac.n_obs} cells × {atac.n_vars} features", verbose)

    # 2. Sample metadata
    if metadata_path:
        atac = merge_sample_metadata(atac, metadata_path, sample_column, verbose=verbose)

    # 3. QC filtering
    log("QC filtering", verbose)
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(atac, 'n_cells_by_counts', lambda x: x >= min_cells)
    mu.pp.filter_obs(atac, 'n_genes_by_counts',
                     lambda x: (x >= min_genes) & (x <= max_genes))

    # 3b. Doublet detection (scanpy path only)
    if not use_snapatac2_dimred and doublet:
        if atac.n_vars >= 50:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    n_prin = min(30, atac.n_vars-1, atac.n_obs-1)
                    sc.pp.scrublet(atac, batch_key=sample_column, n_prin_comps=n_prin)
                    atac = atac[~atac.obs['predicted_doublet']].copy()
                log("Doublets removed", verbose)
            except (ValueError, RuntimeError) as e:
                log(f"Scrublet failed ({e}) – continuing.", verbose)

    # 3c. Sample-level filtering
    log(f"Filtering samples with <{min_cells_per_sample} cells", verbose)
    counts = atac.obs[sample_column].value_counts()
    atac = atac[counts.loc[atac.obs[sample_column]].values >= min_cells_per_sample].copy()
    log(f"Remaining cells: {atac.n_obs}", verbose)

    # 4. TF-IDF
    log("TF-IDF normalisation", verbose)
    ac.pp.tfidf(atac, scale_factor=tfidf_scale_factor)

    if log_transform:
        log("Log1p transform", verbose)
        sc.pp.log1p(atac)

    atac_sample = atac.copy()     # a lightweight copy for obs-level results

    # 5. Dimensionality reduction
    if use_snapatac2_dimred:
        atac = snapatac2_dimensionality_reduction(
            atac,
            n_components=n_lsi_components,
            num_features=num_features,
            doublet=doublet,
            verbose=verbose
        )
        dimred_key = 'X_spectral'
    else:
        log("Selecting HVFs", verbose)
        sc.pp.highly_variable_genes(
            atac, n_top_genes=num_features, flavor='seurat_v3', batch_key=batch_key
        )
        atac.var['HVF'] = atac.var['highly_variable']

        # atac.raw = atac.copy()            # ← removed to save RAM / IO

        log("Running LSI", verbose)
        ac.tl.lsi(atac, n_comps=n_lsi_components)
        if drop_first_lsi:
            atac.obsm['X_lsi'] = atac.obsm['X_lsi'][:, 1:]
            atac.varm["LSI"]   = atac.varm["LSI"][:, 1:]
            atac.uns["lsi"]["stdev"] = atac.uns["lsi"]["stdev"][1:]
        dimred_key = 'X_lsi'

    # 6. Harmony
    if batch_key:
        log("Harmony batch correction", verbose)
        batches = batch_key if isinstance(batch_key, list) else [batch_key]
        Z = harmonize(atac.obsm[dimred_key], atac.obs,
                      batch_key=batches, max_iter_harmony=harmony_max_iter,
                      use_gpu=harmony_use_gpu)
        atac.obsm['X_DM_harmony'] = Z
    else:
        atac.obsm['X_DM_harmony'] = atac.obsm[dimred_key].copy()
    use_rep = 'X_DM_harmony'

    # 7. Neighbours / UMAP
    n_pcs = min(n_pcs, atac.obsm[use_rep].shape[1])
    sc.pp.neighbors(atac, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
    sc.tl.umap(atac, min_dist=umap_min_dist, spread=umap_spread,
               random_state=umap_random_state)

    # 8. Leiden clustering
    atac = cell_types_atac(
        atac,
        cell_column=cell_type_column, 
        existing_cell_types=existing_cell_types,
        n_target_clusters=n_target_clusters,
        cluster_resolution=cluster_resolution,
        use_rep='X_DM_harmony',
        method='average', 
        metric='cosine', 
        distance_mode='centroid',
        num_DMs=n_lsi_components, 
        verbose=verbose
    )
    cell_type_key = 'cell_type'

    # 9. Plots
    sc.pl.umap(atac, color=cell_type_key, legend_loc="on data",
               show=False)
    plt.savefig(os.path.join(output_dir, f"umap_{cell_type_key}.png"),
                dpi=plot_dpi); plt.close()

    sc.pl.umap(atac, color=[cell_type_key, "n_genes_by_counts"],
               legend_loc="on data", show=False)
    plt.savefig(os.path.join(output_dir, "umap_n_genes_by_counts.png"),
                dpi=plot_dpi); plt.close()

    if batch_key:
        for key in (batch_key if isinstance(batch_key, list) else [batch_key]):
            sc.pl.umap(atac, color=key, legend_loc="on data", show=False)
            plt.savefig(os.path.join(output_dir, f"umap_{key}.png"),
                        dpi=plot_dpi); plt.close()

    # 10. Save
    log("Writing H5AD …", verbose)
    atac_sample.obs[cell_type_column] = atac.obs[cell_type_column].copy()

    sc.write(os.path.join(output_dir, "ATAC_sample.h5ad"), atac_sample)

    # 11. Summary
    log("="*60, verbose)
    log(f"Finished in {(time.time()-t0)/60:.1f} min", verbose)
    return atac_sample, atac
