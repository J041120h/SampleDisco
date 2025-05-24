#!/usr/bin/env python3
"""
Minimal scATAC-seq processing pipeline (single-resolution clustering)

▪ QC  ▪ TF-IDF  ▪ LSI  ▪ optional Harmony  ▪ kNN/UMAP  ▪ Leiden clustering
"""

import os, time, warnings
from   datetime import datetime

import numpy  as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from harmony import harmonize
from   sklearn.decomposition import TruncatedSVD
from   scipy.sparse import issparse, csr_matrix
import muon as mu
from muon import atac as ac

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#                               Utility helpers                               #
# --------------------------------------------------------------------------- #

def log(msg, level="INFO", verbose=True):
    """Timestamped logger."""
    if verbose:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {level}: {msg}")

# --------------------------------------------------------------------------- #
#                          Core processing functions                          #
# --------------------------------------------------------------------------- #

def load_atac_data(path, verbose=True):
    log(f"Loading h5ad: {path}", verbose=verbose)
    adata = sc.read_h5ad(path)
    log(f"Loaded {adata.n_obs:,} cells × {adata.n_vars:,} peaks", verbose=verbose)
    return adata

def calculate_qc_metrics(adata, verbose=True):
    """Adds n_features_by_counts / total_counts to .obs (Scanpy ≥1.9)."""
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    if verbose:
        qc = adata.obs[['n_features_by_counts', 'total_counts']].describe().loc[
            ['mean','50%','min','max']]
        print("\nQC summary (peaks per cell / total counts):")
        print(qc.rename(index={'50%':'median'}))
    return adata

def plot_qc_metrics(adata):
    sc.pl.violin(
        adata, ["n_features_by_counts", "total_counts"],
        jitter=0.4, multi_panel=True, rotation=45
    )

def filter_peaks(adata, min_cells_pct=0.01, verbose=True):
    min_cells = int(min_cells_pct * adata.n_obs)
    log(f"Filtering peaks present in < {min_cells} cells "
        f"({min_cells_pct*100:.1f} %)", verbose)
    before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)  # peaks are treated as genes
    log(f"Removed {before-adata.n_vars:,} peaks; kept {adata.n_vars:,}", verbose)
    return adata

def tfidf_normalization(adata, verbose=True):
    log("Running TF-IDF normalisation", verbose)
    # Binarise
    X = adata.X
    if not issparse(X):        # ensure sparse for memory/speed
        X = csr_matrix(X)
    X_bin = (X > 0).astype(np.float32)

    # term frequency
    tf = X_bin.multiply(1.0 / X_bin.sum(axis=1))

    # inverse document frequency
    idf = np.log1p(adata.n_obs / (X_bin.sum(axis=0).A1 + 1))
    tfidf = tf.multiply(idf)

    adata.layers["tfidf"] = tfidf
    adata.X = tfidf
    if verbose:
        nz = tfidf.nnz / (tfidf.shape[0] * tfidf.shape[1]) * 100
        print(f"TF-IDF matrix density: {nz:.2f} %")
    return adata

def perform_lsi(adata, n_components=50, verbose=True):
    log(f"Computing LSI with {n_components} components", verbose)
    svd  = TruncatedSVD(n_components=n_components, random_state=0)
    X_lsi = svd.fit_transform(adata.X)

    adata.obsm["X_lsi"]           = X_lsi
    adata.obsm["X_lsi_corrected"] = X_lsi[:, 1:]   # drop depth component
    adata.uns["lsi"] = dict(
        explained_variance_ratio = svd.explained_variance_ratio_,
        components               = svd.components_
    )
    if verbose:
        var = svd.explained_variance_ratio_
        print(f"Variance explained (first 10 comps): {var[:10].sum()*100:.1f} %")
    return adata

def run_harmony(adata, batch_key, verbose=True):
    """Optional Harmony batch correction on LSI components."""
    import harmonypy as hm
    log(f"Running Harmony on '{batch_key}'", verbose)
    ho = hm.run_harmony(adata.obsm["X_lsi_corrected"], adata.obs, batch_key)
    adata.obsm["X_lsi_harmony"] = ho.Z_corr.T
    return adata

def build_neighbors(adata, n_neighbors=30, use_rep="auto", verbose=True):
    if use_rep == "auto":
        use_rep = "X_lsi_harmony" if "X_lsi_harmony" in adata.obsm else "X_lsi_corrected"
    log(f"Building kNN graph (k={n_neighbors}, rep={use_rep})", verbose)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    return adata

def compute_umap(adata, min_dist=0.3, spread=1.0, verbose=True):
    log("Computing UMAP", verbose)
    sc.tl.umap(adata, min_dist=min_dist, spread=spread, random_state=0)
    return adata

def perform_leiden(adata, resolution=0.5, random_state=0, verbose=True):
    log(f"Leiden clustering (resolution={resolution})", verbose)
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden", random_state=random_state)
    n = adata.obs["leiden"].nunique()
    log(f"Identified {n} clusters", verbose)
    return adata

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

def save_processed(adata, path, verbose=True):
    log(f"Saving processed data → {path}", verbose)
    adata.write(path)
    if verbose:
        print(f"File size: {os.path.getsize(path)/1e6:.1f} MB")

# --------------------------------------------------------------------------- #
#                             Orchestrating runner                            #
# --------------------------------------------------------------------------- #
def run_scatac_pipeline(
    filepath,
    output_dir,
    metadata_path=None,
    batch_key=None,
    verbose=True,
    min_cells=1,
    min_genes=2000,
    max_genes=15000,
    n_components=50,
    n_neighbors=30,
    umap_min_dist=0.3,
    leiden_resolution=0.5
):
    t0 = time.time()
    log("=" * 60 + "\nStarting scATAC-seq pipeline\n" + "=" * 60, verbose)
    output_dir = os.path.join(output_dir, 'atac_harmony')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating harmony subdirectory")

    # 1. Load data  -----------------------------------------------------------
    atac = sc.read_h5ad(filepath)

    # 2. Optional sample-level metadata merge  -------------------------------
    if metadata_path:
        atac = merge_sample_metadata(atac, metadata_path, verbose=verbose)

    # 3. QC + filtering  ------------------------------------------------------
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(atac, 'n_cells_by_counts', lambda x: x >= min_cells)
    mu.pp.filter_obs(atac, 'n_genes_by_counts', lambda x: (x >= min_genes) & (x <= max_genes))

    # 4. TF-IDF  --------------------------------------------------------------
    ac.pp.tfidf(atac, scale_factor=1e4)
    sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5)
    atac.raw = atac.copy()

    # 5. LSI  -----------------------------------------------------------------
    ac.tl.lsi(atac)
    atac.obsm['X_lsi'] = atac.obsm['X_lsi'][:,1:]
    atac.varm["LSI"] = atac.varm["LSI"][:,1:]
    atac.uns["lsi"]["stdev"] = atac.uns["lsi"]["stdev"][1:]

    # 6. Harmony  ----------------------------------------------------
    vars_to_harmonize = batch_key if isinstance(batch_key, list) else [batch_key]
    if vars_to_harmonize is None:
        vars_to_harmonize.append("sample")  # default to sample if no batch key provided
    Z = harmonize(
        atac.obsm['X_lsi'],
        atac.obs,
        batch_key = vars_to_harmonize,
        max_iter_harmony=30,
        use_gpu = True
    )
    atac.obsm['X_lsi_harmony'] = Z

    # 7. Neighbours → UMAP → Leiden  ------------------------------------------
    sc.pp.neighbors(atac, n_neighbors=10, n_pcs=30)
    sc.tl.leiden(atac, resolution=.5)
    sc.tl.umap(atac)
    sc.pl.umap(
        atac,
        legend_loc="on data",
        show=False
    )
    plt.savefig(os.path.join(output_dir,"umap_leiden.png"), dpi=300)
    plt.close()

    sc.pl.umap(
        atac,
        color=["leiden", "n_genes_by_counts"],
        legend_loc="on data",
        show=False
    )
    plt.savefig(os.path.join(output_dir,"umap_n_genes_by_counts.png"), dpi=300)
    plt.close()

    # 9. Save  ---------------------------------------------------------------
    sc.write(os.path.join(output_dir,"ATAC.h5ad"), atac)
    log(f"Pipeline finished in {(time.time() - t0) / 60:.1f} min", verbose)
    return atac


# --------------------------------------------------------------------------- #
#                                   CLI demo                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    run_scatac_pipeline(
        filepath     = "/Users/harry/Desktop/GenoDistance/Data/test_ATAC.h5ad",
        output_dir  = "/Users/harry/Desktop/GenoDistance/result",
        metadata_path= "/Users/harry/Desktop/GenoDistance/Data/ATAC_Metadata.csv",
        batch_key    = 'Donor',          # e.g. "batch" if you have one
        leiden_resolution = 0.6       # **single resolution only**

    )
