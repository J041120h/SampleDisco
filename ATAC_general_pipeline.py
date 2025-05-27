#!/usr/bin/env python3
"""
Minimal scATAC-seq processing pipeline (single-resolution clustering)

▪ QC  ▪ TF-IDF  ▪ LSI/snapATAC2  ▪ optional Harmony  ▪ kNN/UMAP  ▪ Leiden clustering OR cell type transfer
▪ Option to use snapATAC2 only for dimensionality reduction
▪ Option to transfer cell types from RNA-seq reference

Improvements:
- Final dimensionality reduction always saved in 'X_DM_harmony'
- Highly variable features always saved in 'HVF' column
- Output always saved in 'output_dir/harmony' subdirectory
- Cell type transfer from RNA reference using nearest neighbors
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
from sklearn.neighbors import NearestNeighbors

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
#                          Cell Type Transfer Function                         #
# --------------------------------------------------------------------------- #

def transfer_cell_types_from_rna(
    atac_adata,
    rna_adata_path,
    rna_cell_type_column='cell_type',
    atac_use_rep='X_DM_harmony',
    rna_use_rep='X_pca_harmony',
    n_neighbors=5,
    metric='cosine',
    verbose=True
):
    """
    Transfer cell types from RNA-seq reference to ATAC-seq data using nearest neighbors.
    
    Parameters:
    -----------
    atac_adata : AnnData
        ATAC-seq data with computed embeddings
    rna_adata_path : str
        Path to RNA-seq reference AnnData file
    rna_cell_type_column : str
        Column name in RNA data containing cell type annotations
    atac_use_rep : str
        Representation to use from ATAC data (default: 'X_DM_harmony')
    rna_use_rep : str
        Representation to use from RNA data (default: 'X_pca')
    n_neighbors : int
        Number of nearest neighbors to consider for transfer
    metric : str
        Distance metric for nearest neighbor search
    verbose : bool
        Whether to print progress messages
    
    Returns:
    --------
    atac_adata : AnnData
        ATAC data with transferred cell types in obs['transferred_cell_type']
    """
    
    log("Starting cell type transfer from RNA reference", verbose=verbose)
    
    # Load RNA reference data
    log(f"Loading RNA reference data from {rna_adata_path}", verbose=verbose)
    rna_adata = sc.read_h5ad(rna_adata_path)
    log(f"RNA reference: {rna_adata.n_obs} cells, {rna_adata.n_vars} genes", verbose=verbose)
    
    # Check if required representations exist
    if atac_use_rep not in atac_adata.obsm:
        raise ValueError(f"Representation '{atac_use_rep}' not found in ATAC data. "
                        f"Available: {list(atac_adata.obsm.keys())}")
    
    if rna_use_rep not in rna_adata.obsm:
        raise ValueError(f"Representation '{rna_use_rep}' not found in RNA data. "
                        f"Available: {list(rna_adata.obsm.keys())}")
    
    if rna_cell_type_column not in rna_adata.obs.columns:
        raise ValueError(f"Cell type column '{rna_cell_type_column}' not found in RNA data. "
                        f"Available: {list(rna_adata.obs.columns)}")
    
    # Get embeddings
    atac_embedding = atac_adata.obsm[atac_use_rep]
    rna_embedding = rna_adata.obsm[rna_use_rep]
    
    log(f"ATAC embedding shape: {atac_embedding.shape}", verbose=verbose)
    log(f"RNA embedding shape: {rna_embedding.shape}", verbose=verbose)
    
    # Match dimensions if needed
    min_dims = min(atac_embedding.shape[1], rna_embedding.shape[1])
    if atac_embedding.shape[1] != rna_embedding.shape[1]:
        log(f"Dimension mismatch detected. Using first {min_dims} dimensions from both datasets", 
            verbose=verbose)
        atac_embedding = atac_embedding[:, :min_dims]
        rna_embedding = rna_embedding[:, :min_dims]
    
    # Fit nearest neighbors on RNA data
    log(f"Fitting nearest neighbors with {n_neighbors} neighbors using {metric} metric", 
        verbose=verbose)
    nn_model = NearestNeighbors(
        n_neighbors=n_neighbors, 
        metric=metric, 
        n_jobs=-1
    ).fit(rna_embedding)
    
    # Find nearest neighbors for each ATAC cell
    log("Finding nearest neighbors for ATAC cells", verbose=verbose)
    distances, indices = nn_model.kneighbors(atac_embedding)
    
    # Transfer cell types using majority voting
    log("Transferring cell types using majority voting", verbose=verbose)
    rna_cell_types = rna_adata.obs[rna_cell_type_column].values
    transferred_cell_types = []
    transfer_confidence = []
    
    for i in range(len(indices)):
        # Get cell types of nearest neighbors
        neighbor_cell_types = rna_cell_types[indices[i]]
        
        # Count occurrences of each cell type
        unique_types, counts = np.unique(neighbor_cell_types, return_counts=True)
        
        # Assign most frequent cell type
        most_frequent_idx = np.argmax(counts)
        assigned_cell_type = unique_types[most_frequent_idx]
        confidence = counts[most_frequent_idx] / n_neighbors
        
        transferred_cell_types.append(assigned_cell_type)
        transfer_confidence.append(confidence)
    
    # Add results to ATAC data
    atac_adata.obs['transferred_cell_type'] = transferred_cell_types
    atac_adata.obs['transfer_confidence'] = transfer_confidence
    
    # Summary statistics
    unique_transferred = np.unique(transferred_cell_types)
    log(f"Transferred {len(unique_transferred)} unique cell types", verbose=verbose)
    log(f"Cell type distribution:", verbose=verbose)
    if verbose:
        transfer_counts = pd.Series(transferred_cell_types).value_counts()
        for cell_type, count in transfer_counts.items():
            print(f"  {cell_type}: {count} cells")
    
    avg_confidence = np.mean(transfer_confidence)
    log(f"Average transfer confidence: {avg_confidence:.3f}", verbose=verbose)
    
    return atac_adata

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

def run_scatac_pipeline(
    filepath,
    output_dir,
    metadata_path=None,
    sample_column="sample",
    batch_key=None,
    verbose=True,
    use_snapatac2_dimred=False,
    # Cell type transfer parameters
    transfer_cell_type=False,
    rna_adata_path=None,
    rna_cell_type_column='cell_type',
    transfer_n_neighbors=5,
    transfer_metric='cosine',
    # QC and filtering parameters
    min_cells=1,
    min_genes=2000,
    max_genes=15000,
    min_cells_per_sample=1,  # <<< NEW PARAMETER
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
    plot_dpi=300,
    # Additional parameters
    cell_type_column='cell_type'
):
    t0 = time.time()
    log("=" * 60 + "\nStarting scATAC-seq pipeline\n" + "=" * 60, verbose)
    
    if use_snapatac2_dimred:
        log("Using snapATAC2 for dimensionality reduction", verbose=verbose)
    else:
        log("Using LSI for dimensionality reduction", verbose=verbose)
    
    if transfer_cell_type:
        log("Cell type transfer mode enabled", verbose=verbose)
        if not rna_adata_path:
            raise ValueError("rna_adata_path must be provided when transfer_cell_type=True")
    else:
        log("Standard Leiden clustering mode", verbose=verbose)
    
    output_dir = os.path.join(output_dir, output_subdirectory)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Automatically generating '{output_subdirectory}' subdirectory")

    # 1. Load data
    atac = sc.read_h5ad(filepath)
    log(f"Loaded data with {atac.n_obs} cells and {atac.n_vars} features", verbose=verbose)

    # 2. Merge metadata
    if metadata_path:
        atac = merge_sample_metadata(atac, metadata_path, sample_column=sample_column, verbose=verbose)

    # 3. QC filtering
    log("Performing QC and filtering", verbose=verbose)
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(atac, 'n_cells_by_counts', lambda x: x >= min_cells)
    mu.pp.filter_obs(atac, 'n_genes_by_counts', lambda x: (x >= min_genes) & (x <= max_genes))

    # 3b. Doublet detection
    if not use_snapatac2_dimred and doublet:
        min_features_for_scrublet = 50
        if atac.n_vars < min_features_for_scrublet:
            log(f"Skipping doublet detection: only {atac.n_vars} features available", verbose=verbose)
        else:
            try:
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    n_prin_comps = min(30, atac.n_vars - 1, atac.n_obs - 1)
                    sc.pp.scrublet(atac, batch_key=sample_column, n_prin_comps=n_prin_comps)
                    atac = atac[~atac.obs['predicted_doublet']].copy()
                log("Doublet detection completed successfully", verbose=verbose)
            except (ValueError, RuntimeError) as e:
                log(f"Doublet detection failed: {str(e)}. Continuing without doublet removal.", verbose=verbose)

    log(f"After filtering: {atac.n_obs} cells and {atac.n_vars} features", verbose=verbose)

    # 3c. Sample-level filtering
    log(f"Filtering samples with fewer than {min_cells_per_sample} cells", verbose=verbose)
    sample_counts = atac.obs[sample_column].value_counts()
    valid_samples = sample_counts[sample_counts >= min_cells_per_sample].index
    initial_cells = atac.n_obs
    atac = atac[atac.obs[sample_column].isin(valid_samples)].copy()
    log(f"Removed {initial_cells - atac.n_obs} cells from low-count samples", verbose=verbose)
    log(f"Remaining samples: {len(valid_samples)}, cells: {atac.n_obs}", verbose=verbose)

    # 4. TF-IDF normalization
    log("Performing TF-IDF normalization", verbose=verbose)
    ac.pp.tfidf(atac, scale_factor=tfidf_scale_factor)
    atac_sample = atac.copy()

    # 5. Feature selection and dimensionality reduction
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
        log("Selecting highly variable features", verbose=verbose)
        sc.pp.highly_variable_genes(
            atac,
            n_top_genes=num_features,
            flavor='seurat_v3',
            batch_key=batch_key
        )
        atac.var['HVF'] = atac.var['highly_variable']
        log("Saved highly variable features in 'HVF' column", verbose=verbose)
        atac.raw = atac.copy()
        log("Performing LSI dimensionality reduction", verbose=verbose)
        ac.tl.lsi(atac, n_comps=n_lsi_components)
        if drop_first_lsi:
            atac.obsm['X_lsi'] = atac.obsm['X_lsi'][:, 1:]
            atac.varm["LSI"] = atac.varm["LSI"][:, 1:]
            atac.uns["lsi"]["stdev"] = atac.uns["lsi"]["stdev"][1:]
        dimred_key = 'X_lsi'

    # 6. Harmony integration
    if batch_key is not None:
        log(f"Applying Harmony batch correction on {dimred_key}", verbose=verbose)
        vars_to_harmonize = batch_key if isinstance(batch_key, list) else [batch_key]
        Z = harmonize(
            atac.obsm[dimred_key],
            atac.obs,
            batch_key=vars_to_harmonize,
            max_iter_harmony=harmony_max_iter,
            use_gpu=harmony_use_gpu
        )
        atac.obsm['X_DM_harmony'] = Z
        use_rep = 'X_DM_harmony'
    else:
        atac.obsm['X_DM_harmony'] = atac.obsm[dimred_key].copy()
        use_rep = 'X_DM_harmony'

    log("Final dimensionality reduction saved in 'X_DM_harmony'", verbose=verbose)

    # ------------------------------------------------------------------- #
    # 6½. Guarantee n_pcs ≤ available components in the chosen representation
    # ------------------------------------------------------------------- #
    max_pcs = atac.obsm[use_rep].shape[1]
    if n_pcs > max_pcs:
        log(f"n_pcs ({n_pcs}) > available dims in {use_rep} ({max_pcs}); "
            f"setting n_pcs = {max_pcs}", verbose=verbose)
        n_pcs = max_pcs
    # ------------------------------------------------------------------- #

    # 7. Neighbors and UMAP (always needed for visualization)
    log("Computing neighbors", verbose=verbose)
    sc.pp.neighbors(atac, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
    log("Computing UMAP embedding", verbose=verbose)
    sc.tl.umap(atac, min_dist=umap_min_dist, spread=umap_spread, random_state=umap_random_state)

    # 8. Cell type assignment
    if transfer_cell_type:
        # Transfer cell types from RNA reference
        atac = transfer_cell_types_from_rna(
            atac,
            rna_adata_path=rna_adata_path,
            rna_cell_type_column=rna_cell_type_column,
            atac_use_rep=use_rep,
            n_neighbors=transfer_n_neighbors,
            metric=transfer_metric,
            verbose=verbose
        )
        # Use transferred cell types
        atac.obs[cell_type_column] = atac.obs['transferred_cell_type'].copy()
        cell_type_key = 'transferred_cell_type'
    else:
        # Standard Leiden clustering
        log("Running Leiden clustering", verbose=verbose)
        sc.tl.leiden(atac, resolution=leiden_resolution, random_state=leiden_random_state)
        atac.obs[cell_type_column] = atac.obs['leiden'].copy()
        cell_type_key = 'leiden'

    # 9. Plotting
    log("Generating plots", verbose=verbose)
    sc.pl.umap(atac, color=cell_type_key, legend_loc="on data", show=False)
    plt.savefig(os.path.join(output_dir, f"umap_{cell_type_key}.png"), dpi=plot_dpi)
    plt.close()

    sc.pl.umap(atac, color=[cell_type_key, "n_genes_by_counts"], legend_loc="on data", show=False)
    plt.savefig(os.path.join(output_dir, "umap_n_genes_by_counts.png"), dpi=plot_dpi)
    plt.close()

    # Additional plot for transferred cell types showing confidence
    if transfer_cell_type:
        sc.pl.umap(atac, color='transfer_confidence', show=False)
        plt.savefig(os.path.join(output_dir, "umap_transfer_confidence.png"), dpi=plot_dpi)
        plt.close()

    if batch_key:
        batch_keys = batch_key if isinstance(batch_key, list) else [batch_key]
        for key in batch_keys:
            sc.pl.umap(atac, color=key, legend_loc="on data", show=False)
            plt.savefig(os.path.join(output_dir, f"umap_{key}.png"), dpi=plot_dpi)
            plt.close()

    # 10. Save results
    atac_sample.obs[cell_type_column] = atac.obs[cell_type_column].copy()

    log("Saving results", verbose=verbose)
    sc.write(os.path.join(output_dir, "ATAC_cluster.h5ad"), atac)
    sc.write(os.path.join(output_dir, "ATAC_sample.h5ad"), atac_sample)

    # 11. Summary
    log("=" * 60, verbose)
    log(f"Pipeline finished in {(time.time() - t0) / 60:.1f} min", verbose)
    log(f"Dimensionality reduction method: {'snapATAC2' if use_snapatac2_dimred else 'LSI'}", verbose)
    log(f"Cell type assignment: {'Transfer from RNA' if transfer_cell_type else 'Leiden clustering'}", verbose)
    if transfer_cell_type:
        avg_conf = np.mean(atac.obs['transfer_confidence'])
        log(f"Average transfer confidence: {avg_conf:.3f}", verbose)
    log(f"Final representation saved in: X_DM_harmony", verbose)
    log(f"Highly variable features saved in: var['HVF']", verbose)
    log(f"Batch correction applied: {'Yes' if batch_key else 'No'}", verbose)
    log("=" * 60, verbose)

    return atac_sample, atac