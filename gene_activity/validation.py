#!/usr/bin/env python3

# ----------------- Imports (headless plotting) -----------------
import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import matplotlib
matplotlib.use("Agg")  # ensure no X server is required
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------- Function -----------------
def analyze_rna_atac_with_markers(
    rna_h5ad_path,
    atac_gene_activity_path,
    output_dir='./results',
    min_genes=200,
    min_cells=3,
    max_mt_percent=20.0,
    n_top_genes=3000,
    n_neighbors=15,
    n_pcs=40,
    resolution=0.8,
    max_celltypes=6,          # NEW: show at most this many clusters (by size)
    markers_per_type=10       # NEW: exactly this many markers per selected cluster
):
    """
    Complete function for RNA preprocessing, clustering, marker gene identification,
    and visualization of marker genes in both RNA and ATAC modalities.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save-only plotting (no GUI)
    sc.settings.autoshow = False
    sc.settings.figdir = output_dir  # harmless; we save manually anyway

    # =========================================================================
    # RNA PREPROCESSING AND CLUSTERING
    # =========================================================================
    print("1. Loading and preprocessing RNA data...")
    adata_rna = sc.read_h5ad(rna_h5ad_path)
    print(f"   Loaded: {adata_rna.shape[0]} cells x {adata_rna.shape[1]} genes")

    # QC metrics
    adata_rna.var['mt'] = adata_rna.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_rna, qc_vars=['mt'], inplace=True)

    # Filter cells and genes
    sc.pp.filter_cells(adata_rna, min_genes=min_genes)
    sc.pp.filter_genes(adata_rna, min_cells=min_cells)
    adata_rna = adata_rna[adata_rna.obs['pct_counts_mt'] < max_mt_percent, :].copy()
    print(f"   After filtering: {adata_rna.shape[0]} cells x {adata_rna.shape[1]} genes")

    # Normalize and log transform
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)

    # IMPORTANT: set .raw after log1p so rank_genes_groups uses log data
    adata_rna.raw = adata_rna.copy()

    # HVGs
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=n_top_genes)
    adata_rna = adata_rna[:, adata_rna.var.highly_variable].copy()

    # Scale / PCA / neighbors / UMAP / Leiden
    sc.pp.scale(adata_rna, max_value=10)
    sc.tl.pca(adata_rna, svd_solver='arpack', n_comps=n_pcs)
    sc.pp.neighbors(adata_rna, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata_rna)
    sc.tl.leiden(adata_rna, resolution=resolution)
    print(f"   Found {len(adata_rna.obs['leiden'].unique())} clusters")

    # =========================================================================
    # MARKER GENE IDENTIFICATION
    # =========================================================================
    print("\n2. Finding marker genes for each cluster...")
    adata_rna_raw = adata_rna.raw.to_adata()
    adata_rna_raw.obs['leiden'] = adata_rna.obs['leiden']

    sc.tl.rank_genes_groups(adata_rna_raw, 'leiden', method='wilcoxon')

    # Determine which clusters to show: top N by size
    cluster_sizes = (
        adata_rna.obs['leiden']
        .value_counts()
        .sort_values(ascending=False)
    )
    selected_clusters = list(cluster_sizes.index.astype(str))[:max_celltypes]
    print(f"   Selected clusters (top by size): {selected_clusters}")

    # Get exactly markers_per_type marker genes per selected cluster
    # Build an ordered gene list; keep uniqueness while preserving order
    ordered_marker_genes = []
    for cl in selected_clusters:
        genes_for_cl = list(adata_rna_raw.uns['rank_genes_groups']['names'][cl][:markers_per_type])
        for g in genes_for_cl:
            if g not in ordered_marker_genes:
                ordered_marker_genes.append(g)
    print(f"   Total unique selected markers: {len(ordered_marker_genes)} "
          f"(<= {markers_per_type} per cluster)")

    # Subset RNA (for visualization) to selected clusters only
    adata_rna_raw_sel = adata_rna_raw[adata_rna_raw.obs['leiden'].isin(selected_clusters), :].copy()

    # =========================================================================
    # VISUALIZE MARKER GENE EXPRESSION IN RNA (SAVE ONLY)
    # =========================================================================
    print("\n3. Visualizing marker gene expression across selected clusters...")

    # 3.1 UMAP with clusters (full adata_rna; still useful to see all clusters)
    umap_fig = sc.pl.umap(
        adata_rna,
        color='leiden',
        legend_loc='on data',
        title='Cell Clusters (All)',
        show=False,
        return_fig=True,
    )
    umap_fig.savefig(os.path.join(output_dir, "rna_marker_umap.png"), dpi=150, bbox_inches="tight")
    plt.close(umap_fig)

    # 3.2 Dotplot of selected markers (selected clusters only)
    sc.pl.dotplot(
        adata_rna_raw_sel,
        var_names=ordered_marker_genes,
        groupby='leiden',
        show=False
    )
    fig = plt.gcf()
    fig.savefig(os.path.join(output_dir, "rna_markers_dotplot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3.3 Heatmap of selected markers (selected clusters only)
    sc.pl.heatmap(
        adata_rna_raw_sel,
        var_names=ordered_marker_genes,
        groupby='leiden',
        cmap='RdBu_r',
        show=False
    )
    fig = plt.gcf()
    fig.savefig(os.path.join(output_dir, "rna_markers_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3.4 Violin plot: top 3 from the first selected cluster (kept small)
    if selected_clusters:
        first_cluster = selected_clusters[0]
        first3 = list(adata_rna_raw.uns['rank_genes_groups']['names'][first_cluster][:3])
        sc.pl.violin(
            adata_rna_raw_sel,
            keys=first3,
            groupby='leiden',
            rotation=0,
            show=False
        )
        fig = plt.gcf()
        fig.savefig(os.path.join(output_dir, "rna_top3_violin.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # =========================================================================
    # ATAC DATA PROCESSING
    # =========================================================================
    print("\n4. Loading and processing ATAC gene activity data...")
    adata_atac = sc.read_h5ad(atac_gene_activity_path)
    print(f"   Loaded: {adata_atac.shape[0]} cells x {adata_atac.shape[1]} genes")

    sc.pp.normalize_total(adata_atac, target_sum=1e4)
    sc.pp.log1p(adata_atac)

    # Find common cells (paired)
    common_cells = list(set(adata_rna.obs_names) & set(adata_atac.obs_names))
    print(f"   Found {len(common_cells)} paired cells")

    markers_in_atac = []  # default if no pairing/overlap
    if len(common_cells) > 0:
        adata_rna_paired = adata_rna_raw[common_cells, :].copy()
        adata_atac_paired = adata_atac[common_cells, :].copy()
        adata_atac_paired.obs['cluster'] = adata_rna[common_cells, :].obs['leiden'].values

        # Subset both paired objects to the selected clusters
        adata_rna_paired_sel = adata_rna_paired[adata_rna_paired.obs['leiden'].isin(selected_clusters), :].copy()
        adata_atac_paired_sel = adata_atac_paired[adata_atac_paired.obs['cluster'].isin(selected_clusters), :].copy()

        # Restrict markers to genes present in ATAC data
        markers_in_atac = [g for g in ordered_marker_genes if g in adata_atac_paired_sel.var_names]
        print(f"   For ATAC: {len(markers_in_atac)}/{len(ordered_marker_genes)} selected markers present")

        if len(markers_in_atac) > 0:
            print("\n5. Comparing marker gene activity between RNA and ATAC (violins + ATAC dotplot)...")

            # --- Paired violins (single figure) using a small subset for readability ---
            markers_to_plot = markers_in_atac[:min(6, len(markers_in_atac))]
            fig, axes = plt.subplots(len(markers_to_plot), 2, figsize=(10, 4 * len(markers_to_plot)))
            if len(markers_to_plot) == 1:
                axes = np.array([axes])

            for i, gene in enumerate(markers_to_plot):
                sc.pl.violin(
                    adata_rna_paired_sel,
                    keys=[gene],
                    groupby='leiden',
                    show=False,
                    ax=axes[i, 0]
                )
                axes[i, 0].set_title(f'RNA: {gene}')
                axes[i, 0].set_xlabel('Cluster')

                sc.pl.violin(
                    adata_atac_paired_sel,
                    keys=[gene],
                    groupby='cluster',
                    show=False,
                    ax=axes[i, 1]
                )
                axes[i, 1].set_title(f'ATAC: {gene}')
                axes[i, 1].set_xlabel('Cluster')

            plt.suptitle('Marker Gene Expression: RNA vs ATAC Gene Activity (selected clusters)', fontsize=14)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, "rna_atac_marker_comparison.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)

            # --- NEW: ATAC dot plot over selected clusters, using the same markers set ---
            sc.pl.dotplot(
                adata_atac_paired_sel,
                var_names=markers_in_atac,
                groupby='cluster',
                show=False
            )
            fig = plt.gcf()
            fig.savefig(os.path.join(output_dir, "atac_markers_dotplot.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

            # --- Cluster-wise correlations heatmap (optional, still useful) ---
            print("\n6. Computing cluster-wise RNA-ATAC correlations (selected clusters)...")
            correlations = []
            for cluster in selected_clusters:
                cluster_mask_rna = adata_rna_paired_sel.obs['leiden'] == cluster
                cluster_mask_atac = adata_atac_paired_sel.obs['cluster'] == cluster
                for gene in markers_in_atac[:min(20, len(markers_in_atac))]:
                    if gene in adata_rna_paired_sel.var_names:
                        rna_expr = adata_rna_paired_sel[cluster_mask_rna, gene].X
                        atac_expr = adata_atac_paired_sel[cluster_mask_atac, gene].X
                        if sparse.issparse(rna_expr):
                            rna_expr = rna_expr.toarray().flatten()
                        else:
                            rna_expr = np.asarray(rna_expr).flatten()
                        if sparse.issparse(atac_expr):
                            atac_expr = atac_expr.toarray().flatten()
                        else:
                            atac_expr = np.asarray(atac_expr).flatten()
                        if rna_expr.size > 1 and atac_expr.size > 1:
                            corr = np.corrcoef(rna_expr, atac_expr)[0, 1]
                        else:
                            corr = np.nan
                        correlations.append({'cluster': cluster, 'gene': gene, 'correlation': corr})

            corr_df = pd.DataFrame(correlations)
            if not corr_df.empty:
                corr_matrix = corr_df.pivot(index='gene', columns='cluster', values='correlation')
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    corr_matrix,
                    cmap='RdBu_r',
                    center=0,
                    vmin=-1,
                    vmax=1,
                    cbar_kws={'label': 'RNA-ATAC Correlation'}
                )
                plt.title('RNA-ATAC Correlation for Selected Marker Genes by Cluster')
                plt.xlabel('Cluster')
                plt.ylabel('Marker Gene')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "rna_atac_correlation_heatmap.png"), dpi=150, bbox_inches='tight')
                plt.close("all")

                corr_df.to_csv(os.path.join(output_dir, "rna_atac_marker_correlations.csv"), index=False)
                print(f"   Saved correlations to {os.path.join(output_dir, 'rna_atac_marker_correlations.csv')}")

    # Save marker genes list (selected clusters only, exactly markers_per_type each)
    marker_rows = []
    for cl in selected_clusters:
        genes = list(adata_rna_raw.uns['rank_genes_groups']['names'][cl][:markers_per_type])
        scores = list(adata_rna_raw.uns['rank_genes_groups']['scores'][cl][:markers_per_type])
        marker_rows.append(pd.DataFrame({'cluster': cl, 'gene': genes, 'score': scores}))
    marker_df = pd.concat(marker_rows, ignore_index=True) if marker_rows else pd.DataFrame(columns=['cluster', 'gene', 'score'])
    marker_df.to_csv(os.path.join(output_dir, "marker_genes_selected.csv"), index=False)
    print(f"\n7. Saved selected marker genes to {os.path.join(output_dir, 'marker_genes_selected.csv')}")
    print("\nAnalysis complete!")

    return {
        'rna_data': adata_rna_raw,
        'atac_data': adata_atac,  # or paired_sel if you need the subset
        'selected_clusters': selected_clusters,
        'marker_genes_selected': marker_df,
        'markers_in_atac_selected': markers_in_atac
    }


def compute_rna_atac_cell_gene_correlations(
    adata_rna,
    adata_atac,
    output_dir: str,
    min_cells_for_gene_corr: int = 3,
    sample_genes: int | None = 1000,   # set None to use all shared genes
    verbose: bool = True
) -> dict:
    """
    Compute per-cell and per-gene RNA–ATAC correlations for paired cells
    (assumes overlapping adata_rna.obs_names & adata_atac.obs_names denote pairs).

    Per-cell correlation:
        For each paired cell, correlate its RNA vs ATAC vector across shared genes,
        using a union non-zero mask per-cell: (rna != 0) | (atac != 0).

    Per-gene correlation:
        For each shared gene, correlate RNA vs ATAC across paired cells,
        but only on cells where BOTH are non-zero; requires >= min_cells_for_gene_corr.

    Saves:
        - per_cell_correlations.csv
        - per_gene_correlations.csv  (with n_coexpressing_cells)
        - correlation_summary.json
        - correlation_plots.png

    Returns a dict with paths and summary stats.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from scipy import sparse
    import json
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    def _to_array(X):
        if sparse.issparse(X):
            return X.toarray()
        return np.asarray(X)

    # -----------------------------
    # 1) Pair cells by name
    # -----------------------------
    rna_cells = set(map(str, adata_rna.obs_names))
    atac_cells = set(map(str, adata_atac.obs_names))
    common_cells = sorted(rna_cells & atac_cells)
    if verbose:
        print(f"[correlation] Paired cells: {len(common_cells)}")

    if len(common_cells) == 0:
        raise ValueError("No paired cells found (no overlap in obs_names).")

    # Subset and align rows consistently
    rna_sub = adata_rna[common_cells, :].copy()
    atac_sub = adata_atac[common_cells, :].copy()

    # -----------------------------
    # 2) Align genes by name
    # -----------------------------
    rna_genes = set(map(str, rna_sub.var_names))
    atac_genes = set(map(str, atac_sub.var_names))
    shared_genes = sorted(rna_genes & atac_genes)

    if len(shared_genes) == 0:
        raise ValueError("No shared genes between RNA and ATAC.")

    if sample_genes is not None and sample_genes < len(shared_genes):
        rng = np.random.default_rng(42)
        shared_genes = sorted(rng.choice(shared_genes, size=sample_genes, replace=False).tolist())

    # Subset and column-align to shared genes in the same order
    rna_sub = rna_sub[:, shared_genes].copy()
    atac_sub = atac_sub[:, shared_genes].copy()

    # Extract dense arrays
    rna_X = _to_array(rna_sub.X)
    atac_X = _to_array(atac_sub.X)

    n_cells, n_genes = rna_X.shape
    if verbose:
        print(f"[correlation] Using {n_cells} paired cells × {n_genes} shared genes")

    # -----------------------------
    # 3) Per-cell correlations
    # -----------------------------
    per_cell_corr = np.full(n_cells, np.nan, dtype=float)
    for i in range(n_cells):
        r = rna_X[i, :]
        a = atac_X[i, :]
        mask = (r != 0) | (a != 0)                # union non-zero per cell
        if mask.sum() >= 3 and np.std(r[mask]) > 0 and np.std(a[mask]) > 0:
            per_cell_corr[i] = np.corrcoef(r[mask], a[mask])[0, 1]

    per_cell_df = pd.DataFrame({
        "cell": common_cells,
        "pearson_corr": per_cell_corr
    })

    # -----------------------------
    # 4) Per-gene correlations
    # -----------------------------
    per_gene_corr = np.full(n_genes, np.nan, dtype=float)
    per_gene_nco = np.zeros(n_genes, dtype=int)

    # transpose once for cache-friendliness
    rna_T = rna_X.T   # shape (n_genes, n_cells)
    atac_T = atac_X.T

    for j in tqdm(range(n_genes), desc="[correlation] per-gene", disable=not verbose):
        r = rna_T[j, :]
        a = atac_T[j, :]
        # Only cells where BOTH modalities are non-zero
        co_mask = (r != 0) & (a != 0)
        nco = int(co_mask.sum())
        per_gene_nco[j] = nco
        if nco >= min_cells_for_gene_corr and np.std(r[co_mask]) > 0 and np.std(a[co_mask]) > 0:
            per_gene_corr[j] = np.corrcoef(r[co_mask], a[co_mask])[0, 1]

    per_gene_df = pd.DataFrame({
        "gene": shared_genes,
        "pearson_corr": per_gene_corr,
        "n_coexpressing_cells": per_gene_nco
    })

    # -----------------------------
    # 5) Save CSVs
    # -----------------------------
    per_cell_csv = os.path.join(output_dir, "per_cell_correlations.csv")
    per_gene_csv = os.path.join(output_dir, "per_gene_correlations.csv")
    per_cell_df.to_csv(per_cell_csv, index=False)
    per_gene_df.to_csv(per_gene_csv, index=False)

    # -----------------------------
    # 6) Plots (PNG)
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Per-cell histogram
    valid_cc = np.isfinite(per_cell_corr)
    if valid_cc.any():
        axes[0, 0].hist(per_cell_corr[valid_cc], bins=50, edgecolor='black', alpha=0.8)
        axes[0, 0].axvline(np.nanmean(per_cell_corr), ls='--', color='r',
                           label=f"Mean={np.nanmean(per_cell_corr):.3f}")
        axes[0, 0].set_title("Per-cell Pearson correlation")
        axes[0, 0].set_xlabel("Correlation")
        axes[0, 0].set_ylabel("Cells")
        axes[0, 0].legend()

    # Per-gene histogram
    valid_gc = np.isfinite(per_gene_corr)
    if valid_gc.any():
        axes[0, 1].hist(per_gene_corr[valid_gc], bins=50, edgecolor='black', alpha=0.8, label="per-gene corr")
        axes[0, 1].axvline(np.nanmean(per_gene_corr), ls='--', color='r',
                           label=f"Mean={np.nanmean(per_gene_corr):.3f}")
        axes[0, 1].set_title(f"Per-gene Pearson correlation (n≥{min_cells_for_gene_corr} co-expressing cells)")
        axes[0, 1].set_xlabel("Correlation")
        axes[0, 1].set_ylabel("Genes")
        axes[0, 1].legend()

    # Co-expressing cells distribution (log scale on x)
    nz_co = per_gene_nco[per_gene_nco > 0]
    if nz_co.size > 0:
        axes[1, 0].hist(nz_co, bins=50, edgecolor='black', alpha=0.8)
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_title("Co-expressing cell counts per gene")
        axes[1, 0].set_xlabel("Number of co-expressing cells (log)")
        axes[1, 0].set_ylabel("Genes")

    # Corr vs co-expressing cells
    if valid_gc.any():
        axes[1, 1].scatter(per_gene_nco[valid_gc], per_gene_corr[valid_gc], s=8, alpha=0.5)
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_xlabel("Co-expressing cells (log)")
        axes[1, 1].set_ylabel("Gene correlation")
        axes[1, 1].set_title("Per-gene corr vs co-expressing cells")
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "correlation_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------
    # 7) Summary JSON
    # -----------------------------
    summary = {
        "n_paired_cells": int(n_cells),
        "n_shared_genes": int(n_genes),
        "per_cell_mean_corr": float(np.nanmean(per_cell_corr)) if valid_cc.any() else float("nan"),
        "per_cell_median_corr": float(np.nanmedian(per_cell_corr)) if valid_cc.any() else float("nan"),
        "per_gene_mean_corr": float(np.nanmean(per_gene_corr)) if valid_gc.any() else float("nan"),
        "per_gene_median_corr": float(np.nanmedian(per_gene_corr)) if valid_gc.any() else float("nan"),
        "min_cells_for_gene_corr": int(min_cells_for_gene_corr),
        "sample_genes": (int(sample_genes) if sample_genes is not None else None),
        "paths": {
            "per_cell_csv": per_cell_csv,
            "per_gene_csv": per_gene_csv,
            "plots_png": plot_path,
        },
    }
    with open(os.path.join(output_dir, "correlation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"[correlation] Saved:\n  {per_cell_csv}\n  {per_gene_csv}\n  {plot_path}")

    return {
        "per_cell": per_cell_df,
        "per_gene": per_gene_df,
        "summary": summary,
    }

# ----------------- Script entrypoint -----------------
if __name__ == "__main__":
    # results = analyze_rna_atac_with_markers(
    #     rna_h5ad_path='/dcs07/hongkai/data/harry/result/gene_activity/rna_corrected.h5ad',
    #     atac_gene_activity_path='/dcs07/hongkai/data/harry/result/gene_activity/ATAC_ArchR/gene_activity_matrix.h5ad',
    #     output_dir='/dcs07/hongkai/data/harry/result/gene_activity/results',
    #     max_celltypes=6,          # change if you want fewer/more clusters
    #     markers_per_type=10       # exactly 10 per selected cluster
    # )

    adata_rna = sc.read_h5ad('/dcs07/hongkai/data/harry/result/gene_activity/rna_corrected.h5ad')
    adata_atac = sc.read_h5ad('/dcs07/hongkai/data/harry/result/gene_activity/ATAC_ArchR/gene_activity_matrix.h5ad')

    compute_rna_atac_cell_gene_correlations(
        adata_rna,
        adata_atac,
        output_dir="/dcs07/hongkai/data/harry/result/gene_activity/results_corr"
    )
