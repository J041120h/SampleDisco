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
- Marker genes/peaks saved locally for each cluster
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
#                          Marker Gene/Peak Saving Functions                  #
# --------------------------------------------------------------------------- #

def save_marker_genes_all_formats(adata, output_dir, groupby='leiden', 
                                  top_n_markers=50, verbose=True):
    """
    Save marker genes/peaks in multiple formats:
    1. Combined Excel file with all clusters
    2. Individual CSV files per cluster
    3. Summary statistics file
    """
    log("Saving marker genes/peaks to local files", verbose=verbose)
    
    # Create markers subdirectory
    markers_dir = os.path.join(output_dir, "markers")
    os.makedirs(markers_dir, exist_ok=True)
    
    # Extract results from rank_genes_groups
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    # Create comprehensive dataframe
    all_markers = []
    cluster_summaries = []
    
    for group in groups:
        # Extract data for this cluster
        cluster_data = pd.DataFrame({
            'peak_name': result['names'][group][:top_n_markers],
            'gene_symbol': result['genes'][group][:top_n_markers] if 'genes' in result else [''] * top_n_markers,
            'pval': result['pvals'][group][:top_n_markers],
            'pval_adj': result['pvals_adj'][group][:top_n_markers] if 'pvals_adj' in result else [1.0] * top_n_markers,
            'logfoldchange': result['logfoldchanges'][group][:top_n_markers] if 'logfoldchanges' in result else [0.0] * top_n_markers,
            'score': result['scores'][group][:top_n_markers] if 'scores' in result else [0.0] * top_n_markers,
            'cluster': group,
            'rank': range(1, top_n_markers + 1)
        })
        
        all_markers.append(cluster_data)
        
        # Calculate summary statistics for this cluster
        n_cells = sum(adata.obs[groupby] == group)
        n_significant = sum(cluster_data['pval'] < 0.05)
        avg_logfc = cluster_data['logfoldchange'].mean()
        
        cluster_summaries.append({
            'cluster': group,
            'n_cells': n_cells,
            'n_significant_markers': n_significant,
            'avg_logfoldchange': avg_logfc,
            'top_marker': cluster_data.iloc[0]['peak_name'],
            'top_marker_pval': cluster_data.iloc[0]['pval']
        })
        
        # Save individual cluster file
        cluster_file = os.path.join(markers_dir, f"cluster_{group}_markers.csv")
        cluster_data.to_csv(cluster_file, index=False)
        log(f"Saved markers for cluster {group} to {cluster_file}", verbose=verbose)
    
    # Combine all markers
    all_markers_df = pd.concat(all_markers, ignore_index=True)
    
    # Save combined Excel file with multiple sheets
    excel_file = os.path.join(markers_dir, "all_clusters_markers.xlsx")
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # All markers in one sheet
        all_markers_df.to_excel(writer, sheet_name='All_Markers', index=False)
        
        # Individual sheets per cluster
        for group in groups:
            cluster_data = all_markers_df[all_markers_df['cluster'] == group]
            cluster_data.to_excel(writer, sheet_name=f'Cluster_{group}', index=False)
        
        # Summary sheet
        summary_df = pd.DataFrame(cluster_summaries)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    log(f"Saved comprehensive Excel file to {excel_file}", verbose=verbose)
    
    # Save summary CSV
    summary_file = os.path.join(markers_dir, "cluster_summary.csv")
    pd.DataFrame(cluster_summaries).to_csv(summary_file, index=False)
    
    # Save all markers as single CSV
    all_markers_file = os.path.join(markers_dir, "all_markers.csv")
    all_markers_df.to_csv(all_markers_file, index=False)
    
    log(f"Marker gene analysis complete. Files saved in: {markers_dir}", verbose=verbose)
    log(f"- Individual cluster files: cluster_X_markers.csv", verbose=verbose)
    log(f"- Combined Excel file: all_clusters_markers.xlsx", verbose=verbose)
    log(f"- Summary statistics: cluster_summary.csv", verbose=verbose)
    log(f"- All markers CSV: all_markers.csv", verbose=verbose)
    
    return all_markers_df, pd.DataFrame(cluster_summaries)

def save_top_markers_summary(adata, output_dir, groupby='leiden', 
                           top_n=10, verbose=True):
    """
    Save a quick summary of top N markers per cluster in a readable format
    """
    log(f"Creating top {top_n} markers summary", verbose=verbose)
    
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    # Create a readable summary
    summary_lines = []
    summary_lines.append(f"Top {top_n} Marker Genes/Peaks per Cluster")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Total clusters: {len(groups)}")
    summary_lines.append("")
    
    for group in groups:
        n_cells = sum(adata.obs[groupby] == group)
        summary_lines.append(f"CLUSTER {group} ({n_cells} cells)")
        summary_lines.append("-" * 30)
        
        for i in range(min(top_n, len(result['names'][group]))):
            peak = result['names'][group][i]
            pval = result['pvals'][group][i]
            gene = result['genes'][group][i] if 'genes' in result else ''
            
            if gene and gene != '':
                summary_lines.append(f"{i+1:2d}. {peak} ({gene}) - p={pval:.2e}")
            else:
                summary_lines.append(f"{i+1:2d}. {peak} - p={pval:.2e}")
        
        summary_lines.append("")
    
    # Save summary file
    summary_file = os.path.join(output_dir, "markers", "top_markers_summary.txt")
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    log(f"Top markers summary saved to: {summary_file}", verbose=verbose)

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
    # Log transformation parameters
    log_transform=True,
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
    cell_type_column='cell_type',
    # NEW: Marker gene parameters
    save_markers=True,
    top_n_markers=50,
    top_n_summary=10
):
    t0 = time.time()
    log("=" * 60 + "\nStarting scATAC-seq pipeline\n" + "=" * 60, verbose)
    
    if use_snapatac2_dimred:
        log("Using snapATAC2 for dimensionality reduction", verbose=verbose)
    else:
        log("Using LSI for dimensionality reduction", verbose=verbose)
    
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
    
    # 4b. Log transformation (log1p)
    if log_transform:
        log("Applying log1p transformation", verbose=verbose)
        sc.pp.log1p(atac)
    
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

    # 8. Standard Leiden clustering
    log("Running Leiden clustering", verbose=verbose)
    sc.tl.leiden(atac, resolution=leiden_resolution, random_state=leiden_random_state)
    atac.obs[cell_type_column] = atac.obs['leiden'].copy()
    cell_type_key = 'leiden'
    
    # 8b. Differential peak analysis
    log("Finding marker genes/peaks for each cluster", verbose=verbose)
    ac.tl.rank_peaks_groups(atac, 'leiden', method='t-test')
    
    # Display quick preview of results
    result = atac.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    pd.set_option("max_columns", 50)
    preview_df = pd.DataFrame(
        {group + '_' + key[:1]: result[key][group]
        for group in groups for key in ['names', 'genes', 'pvals']}).head(10)
    log("Preview of top marker peaks:", verbose=verbose)
    if verbose:
        print(preview_df)

    # 8c. Save marker genes/peaks locally
    if save_markers:
        all_markers_df, summary_df = save_marker_genes_all_formats(
            atac, output_dir, groupby='leiden', 
            top_n_markers=top_n_markers, verbose=verbose
        )
        save_top_markers_summary(
            atac, output_dir, groupby='leiden', 
            top_n=top_n_summary, verbose=verbose
        )

    # 9. Plotting
    log("Generating plots", verbose=verbose)
    sc.pl.umap(atac, color=cell_type_key, legend_loc="on data", show=False)
    plt.savefig(os.path.join(output_dir, f"umap_{cell_type_key}.png"), dpi=plot_dpi)
    plt.close()

    sc.pl.umap(atac, color=[cell_type_key, "n_genes_by_counts"], legend_loc="on data", show=False)
    plt.savefig(os.path.join(output_dir, "umap_n_genes_by_counts.png"), dpi=plot_dpi)
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
    log(f"Cell type assignment: Leiden clustering", verbose)
    log(f"TF-IDF normalization applied: Yes", verbose)
    log(f"Log transformation applied: {'Yes' if log_transform else 'No'}", verbose)
    log(f"Final representation saved in: X_DM_harmony", verbose)
    log(f"Highly variable features saved in: var['HVF']", verbose)
    log(f"Batch correction applied: {'Yes' if batch_key else 'No'}", verbose)
    if save_markers:
        log(f"Marker genes/peaks saved: Yes (top {top_n_markers} per cluster)", verbose)
        log(f"Marker files location: {output_dir}/markers/", verbose)
    log("=" * 60, verbose)

    return atac_sample, atac