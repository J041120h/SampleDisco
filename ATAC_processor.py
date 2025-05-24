#!/usr/bin/env python3
"""
Minimal scATAC-seq processing pipeline (single-resolution clustering)

▪ QC  ▪ TF-IDF  ▪ LSI/snapATAC2  ▪ optional Harmony  ▪ kNN/UMAP  ▪ Leiden clustering
▪ Option to use snapATAC2 only for dimensionality reduction

Improvements:
- Final dimensionality reduction always saved in 'X_DM_harmony'
- Highly variable features always saved in 'HVF' column
- Output always saved in 'output_dir/harmony' subdirectory
"""

import os, time, warnings
from   datetime import datetime
import contextlib
import io
import numpy  as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from harmony import harmonize
import muon as mu
from muon import atac as ac
import scipy.sparse as sp

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
#                          snapATAC2 Dimensionality Reduction                 #
# --------------------------------------------------------------------------- #
def snapatac2_dimensionality_reduction(
    adata,
    n_components=50,
    num_features=50000,
    doublet=True,
    verbose=True
):
    """
    Use snapATAC2 only for dimensionality reduction (SVD/spectral decomposition).
    Assumes data has already been processed with TF-IDF and feature selection.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object that has been processed with TF-IDF
    n_components : int
        Number of components for dimensionality reduction
    verbose : bool
        Whether to print progress messages
    """
    try:
        import snapatac2 as snap
    except ImportError:
        raise ImportError("snapATAC2 is required for this processing method. "
                         "Install it with: pip install snapatac2")
    log(f"Running snapATAC2 SVD with {n_components} components", verbose=verbose)
    
    # Convert data to float32 to avoid dtype issues
    if adata.X.dtype != np.float32:
        log("Converting data matrix to float32 for snapATAC2 compatibility", verbose=verbose)
        adata.X = adata.X.astype(np.float32)
    
    # Run feature selection
    snap.pp.select_features(adata, n_features=num_features)

    if doublet:
        log("Filtering doublets using snapATAC2", verbose=verbose)
        try:
            snap.pp.scrublet(adata)
            snap.pp.filter_doublets(adata)
        except (AttributeError, TypeError) as e:
            if "'Series' object has no attribute 'nonzero'" in str(e) or "nonzero" in str(e):
                log("snapATAC2 scrublet compatibility issue detected. Skipping doublet detection.", verbose=verbose)
                log("Consider using LSI mode with doublet=True for doublet detection via scanpy.", verbose=verbose)
            else:
                raise e
    
    # Save highly variable features in consistent column name
    if 'selected' in adata.var.columns:
        adata.var['HVF'] = adata.var['selected']
        log("Saved highly variable features in 'HVF' column", verbose=verbose)
    
    # Run spectral decomposition with proper error handling
    try:
        snap.tl.spectral(adata, n_comps=n_components)
    except RuntimeError as e:
        if "Cannot convert CsrMatrix" in str(e):
            log("Data type conversion issue detected. Trying alternative approach...", verbose=verbose)
            # Alternative: ensure the matrix is in the right format
            if sp.issparse(adata.X):
                adata.X = adata.X.astype(np.float32)
            else:
                adata.X = sp.csr_matrix(adata.X.astype(np.float32))
            snap.tl.spectral(adata, n_comps=n_components)
        else:
            raise e
    
    log("snapATAC2 dimensionality reduction complete", verbose=verbose)
    return adata

# --------------------------------------------------------------------------- #
#                             Orchestrating runner                            #
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
    # Doublet detection
    doublet=True,
    # TF-IDF parameters
    tfidf_scale_factor=1e4,
    # Highly variable genes parameters
    num_features=50000,
    # LSI/dimensionality reduction parameters
    n_lsi_components=50,
    drop_first_lsi=True,
    # Harmony parameters
    harmony_max_iter=30,
    harmony_use_gpu=True,
    # Neighbors parameters
    n_neighbors=10,
    n_pcs=30,
    # Leiden clustering parameters
    leiden_resolution=0.5,
    leiden_random_state=0,
    # UMAP parameters
    umap_min_dist=0.3,
    umap_spread=1.0,
    umap_random_state=0,
    # Output parameters
    output_subdirectory='harmony',  # Always use 'harmony' subdirectory
    plot_dpi=300
):
    t0 = time.time()
    log("=" * 60 + "\nStarting scATAC-seq pipeline\n" + "=" * 60, verbose)
    
    if use_snapatac2_dimred:
        log("Using snapATAC2 for dimensionality reduction", verbose=verbose)
    else:
        log("Using LSI for dimensionality reduction", verbose=verbose)
    
    # Always use 'harmony' subdirectory regardless of method
    output_dir = os.path.join(output_dir, "ATAC", output_subdirectory)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Automatically generating '{output_subdirectory}' subdirectory")

    # 1. Load data  -----------------------------------------------------------
    atac = sc.read_h5ad(filepath)
    log(f"Loaded data with {atac.n_obs} cells and {atac.n_vars} features", verbose=verbose)

    # 2. Optional sample-level metadata merge  -------------------------------
    if metadata_path:
        atac = merge_sample_metadata(atac, metadata_path, sample_column=sample_column, verbose=verbose)

    # 3. QC + filtering  ------------------------------------------------------
    log("Performing QC and filtering", verbose=verbose)
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(atac, 'n_cells_by_counts', lambda x: x >= min_cells)
    mu.pp.filter_obs(atac, 'n_genes_by_counts', lambda x: (x >= min_genes) & (x <= max_genes))
    if not use_snapatac2_dimred and doublet:
        # Check if we have enough features for scrublet PCA
        min_features_for_scrublet = 50  # Minimum features needed for reliable scrublet
        if atac.n_vars < min_features_for_scrublet:
            log(f"Skipping doublet detection: only {atac.n_vars} features available, need at least {min_features_for_scrublet} for reliable scrublet analysis", verbose=verbose)
        else:
            try:
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    # Calculate appropriate n_prin_comps based on available features
                    n_prin_comps = min(30, atac.n_vars - 1, atac.n_obs - 1)
                    sc.pp.scrublet(atac, batch_key=sample_column, n_prin_comps=n_prin_comps)
                    atac = atac[~atac.obs['predicted_doublet']].copy()
                log("Doublet detection completed successfully", verbose=verbose)
            except (ValueError, RuntimeError) as e:
                log(f"Doublet detection failed: {str(e)}. Continuing without doublet removal.", verbose=verbose)

    log(f"After filtering: {atac.n_obs} cells and {atac.n_vars} features", verbose=verbose)

    # 4. TF-IDF normalization  -----------------------------------------------
    log("Performing TF-IDF normalization", verbose=verbose)
    ac.pp.tfidf(atac, scale_factor=tfidf_scale_factor)

    log("Saving cluster results", verbose=verbose)
    sc.write(os.path.join(output_dir,"ATAC_sample.h5ad"), atac)
    
    # 6. Dimensionality reduction: LSI or snapATAC2  -------------------------
    if use_snapatac2_dimred:
        # Use snapATAC2 for dimensionality reduction only
        atac = snapatac2_dimensionality_reduction(
            atac,
            n_components=n_lsi_components,
            num_features=num_features,
            doublet=doublet,
            verbose=verbose
        )
        dimred_key = 'X_spectral'
    else:
        # Standard LSI with scanpy
        # 5. Feature selection (highly variable genes)  --------------------------
        log("Selecting highly variable features", verbose=verbose)
        sc.pp.highly_variable_genes(
            atac,
            n_top_genes=num_features,
            flavor='seurat_v3',
            batch_key=batch_key
        )
        
        # Save highly variable features in consistent column name
        atac.var['HVF'] = atac.var['highly_variable']
        log("Saved highly variable features in 'HVF' column", verbose=verbose)
        
        atac.raw = atac.copy()
        log("Performing LSI dimensionality reduction", verbose=verbose)
        ac.tl.lsi(atac, n_comps=n_lsi_components)
        if drop_first_lsi:
            atac.obsm['X_lsi'] = atac.obsm['X_lsi'][:,1:]
            atac.varm["LSI"] = atac.varm["LSI"][:,1:]
            atac.uns["lsi"]["stdev"] = atac.uns["lsi"]["stdev"][1:]
        dimred_key = 'X_lsi'

    # 7. Harmony batch correction (optional) and final naming  ----------------
    if batch_key is not None:
        log(f"Applying Harmony batch correction on {dimred_key}", verbose=verbose)
        vars_to_harmonize = batch_key if isinstance(batch_key, list) else [batch_key]
        Z = harmonize(
            atac.obsm[dimred_key],
            atac.obs,
            batch_key = vars_to_harmonize,
            max_iter_harmony=harmony_max_iter,
            use_gpu = harmony_use_gpu
        )
        # Store harmonized results with consistent key name
        atac.obsm['X_DM_harmony'] = Z
        use_rep = 'X_DM_harmony'
    else:
        # Even without harmony, save the final dimensionality reduction with consistent name
        atac.obsm['X_DM_harmony'] = atac.obsm[dimred_key].copy()
        use_rep = 'X_DM_harmony'
    
    log(f"Final dimensionality reduction saved in 'X_DM_harmony'", verbose=verbose)

    # 8. Neighbours → UMAP → Leiden  ------------------------------------------
    log("Computing neighbors", verbose=verbose)
    sc.pp.neighbors(atac, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
    
    log("Running Leiden clustering", verbose=verbose)
    sc.tl.leiden(atac, resolution=leiden_resolution, random_state=leiden_random_state)
    
    log("Computing UMAP embedding", verbose=verbose)
    sc.tl.umap(atac, min_dist=umap_min_dist, spread=umap_spread, random_state=umap_random_state)
    
    # 9. Plotting  ------------------------------------------------------------
    log("Generating plots", verbose=verbose)
    
    sc.pl.umap(
        atac,
        color='leiden',
        legend_loc="on data",
        show=False
    )
    plt.savefig(os.path.join(output_dir,"umap_leiden.png"), dpi=plot_dpi)
    plt.close()

    sc.pl.umap(
        atac,
        color=["leiden", "n_genes_by_counts"],
        legend_loc="on data",
        show=False
    )
    plt.savefig(os.path.join(output_dir,"umap_n_genes_by_counts.png"), dpi=plot_dpi)
    plt.close()
    
    # Additional plot for batch effect if batch_key is provided
    if batch_key:
        batch_keys = batch_key if isinstance(batch_key, list) else [batch_key]
        for key in batch_keys:
            sc.pl.umap(
                atac,
                color=key,
                legend_loc="on data",
                show=False
            )
            plt.savefig(os.path.join(output_dir,f"umap_{key}.png"), dpi=plot_dpi)
            plt.close()

    # 10. Save  ---------------------------------------------------------------
    log("Saving cluster results", verbose=verbose)
    sc.write(os.path.join(output_dir,"ATAC_cluster.h5ad"), atac)
    
    # Summary information
    log("=" * 60, verbose)
    log(f"Pipeline finished in {(time.time() - t0) / 60:.1f} min", verbose)
    log(f"Dimensionality reduction method: {'snapATAC2' if use_snapatac2_dimred else 'LSI'}", verbose)
    log(f"Final representation saved in: X_DM_harmony", verbose)
    log(f"Highly variable features saved in: var['HVF']", verbose)
    log(f"Batch correction applied: {'Yes' if batch_key else 'No'}", verbose)
    log("=" * 60, verbose)
    
    return atac


# --------------------------------------------------------------------------- #
#                                   CLI demo                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Example 1: Standard usage (scanpy LSI)
    run_scatac_pipeline(
        filepath     = "/Users/harry/Desktop/GenoDistance/Data/test_ATAC.h5ad",
        output_dir  = "/Users/harry/Desktop/GenoDistance/result",
        metadata_path= "/Users/harry/Desktop/GenoDistance/Data/ATAC_Metadata.csv",
        sample_column= "sample",
        batch_key    = 'Donor',
        leiden_resolution = 0.8,
        use_snapatac2_dimred = True  # Use snapATAC2 for dim reduction
    )