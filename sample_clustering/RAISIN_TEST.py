#!/usr/bin/env python3
"""
RAISIN Testing - Python port of R RAISIN package

This module performs statistical testing for the RAISIN model.
Tests if a coefficient or combination of coefficients in the fixed effects
design matrix is 0, generating corresponding p-values and FDRs.

Author: Original R code by Zhicheng Ji, Wenpin Hou, Hongkai Ji
Python port maintains compatibility with R implementation.

Enhanced with pseudobulk/cluster-averaged visualizations (Seurat-style).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import zscore
import warnings
from statsmodels.stats.multitest import multipletests
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import matplotlib.patches as mpatches
from adjustText import adjust_text

# ============================================================================
# PLOTTING AESTHETICS
# ============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

# White background and no grid globally
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False

sns.set_context("paper")
sns.set_style("white")  # clean, no grid


# ============================================================================
# NEW VISUALIZATION FUNCTIONS
# ============================================================================

def plot_journal_volcano(
    result,
    title,
    output_path,
    fdr_threshold=0.05,
    label_genes=False,
    top_n_genes=5
):
    """
    Creates a minimalist, publication-ready volcano plot.

    If label_genes is True, the top_n_genes (by smallest FDR among significant)
    will be labeled with numbers 1..N on the plot, and a legend on the right 
    will map number -> gene name.
    
    Parameters
    ----------
    result : pd.DataFrame
        DE results with 'Foldchange' and 'FDR' columns
    title : str
        Plot title
    output_path : str
        Path to save the figure
    fdr_threshold : float
        FDR threshold for significance
    label_genes : bool
        Whether to label top differential genes
    top_n_genes : int
        Total number of genes to label with numbers
    """
    df = result.copy()
    df['nlog10'] = -np.log10(df['FDR'] + 1e-300)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(False)

    # significance masks
    is_sig = df['FDR'] < fdr_threshold
    is_up = df['Foldchange'] > 0
    
    # non-significant
    ax.scatter(
        df.loc[~is_sig, 'Foldchange'],
        df.loc[~is_sig, 'nlog10'],
        c='#dddddd',
        s=5,
        alpha=0.5,
        linewidth=0,
        rasterized=True,
    )
    
    # significant up
    ax.scatter(
        df.loc[is_sig & is_up, 'Foldchange'],
        df.loc[is_sig & is_up, 'nlog10'],
        c='#B31B1B',
        s=15,
        alpha=0.8,
        linewidth=0,
        rasterized=True,
    )
    
    # significant down
    ax.scatter(
        df.loc[is_sig & ~is_up, 'Foldchange'],
        df.loc[is_sig & ~is_up, 'nlog10'],
        c='#0047AB',
        s=15,
        alpha=0.8,
        linewidth=0,
        rasterized=True,
    )
    
    # Guidelines
    ax.axhline(
        -np.log10(fdr_threshold),
        linestyle='--',
        color='black',
        linewidth=0.8,
        alpha=0.5,
    )
    ax.axvline(
        0,
        linestyle='-',
        color='black',
        linewidth=0.5,
        alpha=0.5,
    )
    
    # Label top genes with NUMBERS + legend if requested
    if label_genes and top_n_genes > 0:
        # pick top genes among significant ones by smallest FDR
        sig_df = df[is_sig].sort_values('FDR')
        sig_df = sig_df.head(top_n_genes)
        
        texts = []
        labeled_genes = []
        
        for idx, (gene, row) in enumerate(sig_df.iterrows(), start=1):
            # numeric label on the plot
            t = ax.text(
                row['Foldchange'],
                row['nlog10'],
                str(idx),
                fontsize=7,
                fontweight='bold',
                ha='center',
                va='bottom'
            )
            texts.append(t)
            labeled_genes.append((idx, gene))
        
        # avoid overlaps between numeric labels
        if texts:
            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
            )
        
        # legend on the right: number -> gene name
        if labeled_genes:
            legend_handles = [
                mpatches.Patch(color='none', label=f"{i}: {g}")
                for i, g in labeled_genes
            ]
            ax.legend(
                handles=legend_handles,
                title="Labeled genes",
                loc='upper left',
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
                fontsize=7,
                title_fontsize=8,
                borderaxespad=0.0
            )
    
    ax.set_title(title, fontsize=10, weight='bold')
    ax.set_xlabel("Log Fold Change", fontsize=9)
    ax.set_ylabel("-log10(FDR)", fontsize=9)
    
    # Style axes - black spines, smaller tick labels, no grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis='both', which='major', labelsize=7, colors='black')
    ax.tick_params(axis='both', which='minor', labelsize=6, colors='black')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pseudobulk_heatmap(
    fit,
    all_results,
    output_path,
    fdr_threshold=0.05,
    top_n_per_cluster=20,
    figsize=(8, 10)
):
    """
    Generates a standard "DoHeatmap" style visualization.

    1. Aggregates expression to pseudobulk (Cluster Means).
    2. Identifies marker genes for each cluster from the DE results.
    3. Plots Z-scored expression with genes grouped by cluster.
    """

    # --- 1. Compute Pseudobulk (Average Expression per Group) ---
    # fit['mean'] is typically Gene x Sample. We need Gene x Group.
    means = fit['mean']
    if isinstance(means, pd.DataFrame):
        means_df = means.copy()
    else:
        means_df = pd.DataFrame(means)

    groups = np.array(fit['group']).astype(str)

    # sanity check: groups length should match number of samples (columns)
    if means_df.shape[1] != len(groups):
        raise ValueError(
            f"Mismatch between means columns ({means_df.shape[1]}) "
            f"and group labels ({len(groups)})."
        )
    
    # Group samples by cluster and take the mean
    # Result: Columns = Clusters, Rows = Genes
    pseudobulk = means_df.groupby(groups, axis=1).mean()

    # --- 2. Identify and Group Marker Genes ---
    cluster_markers = {g: [] for g in pseudobulk.columns}

    for comp_name, res in all_results.items():
        # comp_name is like "Cluster1_vs_Cluster2"
        parts = comp_name.split('_vs_')
        if len(parts) != 2:
            continue
        pos_grp, neg_grp = parts[0], parts[1]
        
        # Get significant UP genes (markers for pos_grp)
        sig_up = res[(res['FDR'] < fdr_threshold) & (res['Foldchange'] > 0)]

        if sig_up.empty:
            continue
        
        # We only care about the top N strongest markers
        top_genes = sig_up.nlargest(top_n_per_cluster, 'Foldchange').index.tolist()
        
        if pos_grp in cluster_markers:
            cluster_markers[pos_grp].extend(top_genes)

    # Dedup and assign genes to their "primary" cluster
    final_gene_list = []
    gene_cluster_labels = []
    seen_genes = set()
    
    # Sort clusters naturally
    sorted_clusters = sorted(map(str, pseudobulk.columns))

    # ensure columns are in sorted order as strings
    pseudobulk = pseudobulk.loc[:, sorted_clusters]

    for cluster in sorted_clusters:
        candidates = cluster_markers.get(cluster, [])
        for gene in candidates:
            # Only keep genes that actually exist in pseudobulk
            if gene not in seen_genes and gene in pseudobulk.index:
                final_gene_list.append(gene)
                gene_cluster_labels.append(cluster)
                seen_genes.add(gene)

    if not final_gene_list:
        print("Warning: No significant markers found to plot. Skipping pseudobulk heatmap.")
        return

    # --- 3. Prepare Heatmap Matrix ---
    # Subset pseudobulk matrix to marker genes
    plot_matrix = pseudobulk.loc[final_gene_list, sorted_clusters]

    # Convert to float64 NumPy array explicitly to avoid object dtypes
    plot_values = np.asarray(plot_matrix, dtype=float)

    # Z-score row-wise (standardize gene expression across clusters)
    # axis=1 → per gene
    z_matrix_values = zscore(plot_values, axis=1, nan_policy='omit')

    # If only one gene, zscore may return 1D array; ensure 2D
    if z_matrix_values.ndim == 1:
        z_matrix_values = z_matrix_values[np.newaxis, :]

    # Handle NaNs/inf if variance is 0, etc.
    z_matrix_values = np.nan_to_num(
        z_matrix_values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
    
    # Clip outliers for visualization (standard practice is -2.5 to 2.5)
    z_matrix_values = np.clip(z_matrix_values, -2.5, 2.5)

    # Convert back to DataFrame for plotting
    z_matrix = pd.DataFrame(
        z_matrix_values,
        index=plot_matrix.index,
        columns=plot_matrix.columns
    )

    # --- 4. Plotting ---
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(False)
    
    sns.heatmap(
        z_matrix,
        cmap="RdBu_r",
        center=0,
        xticklabels=True,
        yticklabels=False,  # hide gene names
        cbar_kws={"label": "Z-score Expr", "shrink": 0.5},
        ax=ax
    )
    
    # Add grouping lines for genes by cluster
    current_cluster = gene_cluster_labels[0]
    for i, cluster in enumerate(gene_cluster_labels):
        if cluster != current_cluster:
            ax.hlines(i, *ax.get_xlim(), colors='white', linewidth=1)
            current_cluster = cluster
    
    # Style axes - black spines, smaller tick labels
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis='both', which='major', labelsize=7, colors='black')
            
    ax.set_xlabel("Clusters", fontsize=12, weight='bold')
    ax.set_ylabel(f"Marker Genes (n={len(final_gene_list)})", fontsize=12)
    ax.set_title("Pseudobulk Expression of Markers", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Pseudobulk Heatmap: {output_path}")


def plot_journal_dotplot(
    all_results,
    output_path,
    fdr_threshold=0.05,
    top_n=10,
    figsize=None
):
    """
    Creates a Seurat-style Dot Plot.
    """
    plot_data = []
    
    target_genes = set()
    for comp, df in all_results.items():
        sig = df[df['FDR'] < fdr_threshold]
        top = sig.nlargest(top_n, 'Foldchange').index.tolist()
        target_genes.update(top)
    
    target_genes = sorted(list(target_genes))
    if not target_genes:
        print("Warning: No genes to plot in DotPlot.")
        return

    for comp, df in all_results.items():
        available = df.index.intersection(target_genes)
        if len(available) == 0:
            continue
        subset = df.loc[available].copy()
        subset['Gene'] = subset.index
        # Clean comparison name for display
        clean_name = comp.replace('_vs_Rest', '')
        subset['Comparison'] = clean_name
        
        subset['LogFDR'] = -np.log10(subset['FDR'] + 1e-300)
        subset['LogFDR_clipped'] = subset['LogFDR'].clip(upper=50)
        plot_data.append(subset[['Gene', 'Comparison', 'Foldchange', 'LogFDR_clipped']])

    if not plot_data:
        print("Warning: No data to plot in DotPlot after filtering.")
        return

    final_df = pd.concat(plot_data)

    if figsize is None:
        figsize = (
            len(final_df['Comparison'].unique()) * 0.8 + 2,
            len(target_genes) * 0.25 + 2
        )
        
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(False)
    
    # Mapping
    comparisons = sorted(final_df['Comparison'].unique())
    genes = sorted(final_df['Gene'].unique())
    x_map = {c: i for i, c in enumerate(comparisons)}
    y_map = {g: i for i, g in enumerate(genes)}
    
    sc = ax.scatter(
        x=final_df['Comparison'].map(x_map),
        y=final_df['Gene'].map(y_map),
        s=final_df['LogFDR_clipped'] * 5,
        c=final_df['Foldchange'],
        cmap='RdBu_r',
        edgecolors='black',
        linewidth=0.5,
        vmin=-2.5, vmax=2.5
    )
    
    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels(comparisons, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=6)
    
    # Style axes - black spines, no grid
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis='both', which='major', labelsize=7, colors='black')
    
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label("Log Fold Change", fontsize=9)
    cbar.ax.tick_params(labelsize=6)
    
    sizes = [10, 30, 50]
    legend_elements = [
        plt.scatter([], [], s=s*5, c='gray', label=str(s))
        for s in sizes
    ]
    ax.legend(
        handles=legend_elements,
        title="-log10(FDR)",
        bbox_to_anchor=(1.2, 1),
        loc='upper left',
        fontsize=7,
        title_fontsize=8
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# CORE TESTING FUNCTIONS
# ============================================================================

def raisintest(
    fit,
    coef: int = 2,
    contrast=None,
    fdrmethod: str = 'fdr_bh',
    n_permutations: int = 100,
    output_dir: str = None,
    min_samples: int = None,
    fdr_threshold: float = 0.05,
    make_volcano: bool = True,
    make_ma_plot: bool = False,
    top_n_label: int = 10,
    verbose: bool = True,
):
    """
    Statistical testing for RAISIN model.
    """
    try:
        # Extract Data
        X = fit['X']
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float64)
            x_cols = list(fit['X'].columns)
        else:
            X = np.array(X, dtype=np.float64)
            x_cols = [f"coef_{i}" for i in range(X.shape[1])]
        
        means = fit['mean']
        if isinstance(means, pd.DataFrame):
            gene_names = np.array(means.index)
            means = means.values.astype(np.float64)
        else:
            gene_names = np.arange(means.shape[0])
            means = np.array(means, dtype=np.float64)
            
        group = np.array(fit['group'], dtype=str)
        Z = np.array(fit['Z'], dtype=np.float64)
        sigma2 = fit['sigma2']
        if isinstance(sigma2, pd.DataFrame):
            sigma2_values = sigma2.values.astype(np.float64)
            sigma2_columns = list(sigma2.columns)
        else:
            sigma2_values = np.array(sigma2, dtype=np.float64)
            sigma2_columns = list(range(sigma2_values.shape[1]))
        omega2 = np.array(fit['omega2'], dtype=np.float64)

        # Filtering
        if min_samples is not None:
            sample_counts = np.sum(means > 0, axis=1)
            mask = sample_counts >= min_samples
            means = means[mask, :]
            gene_names = np.array(gene_names)[mask]
            sigma2_values = sigma2_values[mask, :]
            omega2 = omega2[mask, :]
        G = means.shape[0]

        # Contrast
        if contrast is None:
            contrast = np.zeros(X.shape[1])
            if coef < 1 or coef > X.shape[1]:
                raise ValueError(f"coef must be between 1 and {X.shape[1]}")
            contrast[coef - 1] = 1
            contrast_label = x_cols[coef - 1]
        else:
            contrast = np.array(contrast, dtype=np.float64)
            contrast_label = "custom_contrast"

        # Statistics
        XTX = X.T @ X
        XTX_inv = np.linalg.pinv(XTX)
        k = contrast @ XTX_inv @ X.T
        b = means @ k
        
        kZ = k @ Z
        sigma2_by_group = np.zeros((G, len(group)))
        for i, g in enumerate(group):
            if g in sigma2_columns:
                col_idx = sigma2_columns.index(g)
                sigma2_by_group[:, i] = sigma2_values[:, col_idx]
        
        random_var = np.sum((kZ ** 2) * sigma2_by_group, axis=1)
        fixed_var = np.sum((k ** 2) * omega2, axis=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            stat = b / np.sqrt(random_var + fixed_var)
        stat = np.where(np.isfinite(stat), stat, 0)

        # Permutations
        if verbose:
            print(f"Running {n_permutations} permutations...")
        simu_stat = []
        for _ in range(n_permutations):
            perm_idx = np.random.permutation(X.shape[0])
            perX = X[perm_idx, :]
            perm_k = contrast @ np.linalg.pinv(perX.T @ perX) @ perX.T
            perm_var = (
                np.sum(((perm_k @ Z) ** 2) * sigma2_by_group, axis=1) +
                np.sum((perm_k ** 2) * omega2, axis=1)
            )
            with np.errstate(divide='ignore', invalid='ignore'):
                p_s = (means @ perm_k) / np.sqrt(perm_var)
            p_s = p_s[np.isfinite(p_s)]
            if len(p_s) > 1000:
                p_s = np.random.choice(p_s, 1000, replace=False)
            simu_stat.extend(p_s)

        # P-values
        simu_stat = np.array(simu_stat)
        if len(simu_stat) == 0:
            pval = 2 * stats.norm.sf(np.abs(stat))
        else:
            pnorm_ll = np.sum(stats.norm.logpdf(simu_stat))
            best_ll = -np.inf
            best_df = 100
            for d in np.arange(1, 100.1, 0.5):
                ll = np.sum(stats.t.logpdf(simu_stat, df=d))
                if ll > best_ll:
                    best_ll = ll
                    best_df = d
            
            if best_ll > pnorm_ll:
                pval = 2 * stats.t.sf(np.abs(stat), df=best_df)
            else:
                pval = 2 * stats.norm.sf(np.abs(stat))

        _, fdr, _, _ = multipletests(pval, method=fdrmethod)
        
        res = pd.DataFrame({
            'Foldchange': b,
            'stat': stat,
            'pvalue': pval,
            'FDR': fdr
        }, index=gene_names)
        res = res.sort_values('FDR')

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            res.to_csv(os.path.join(output_dir, "raisin_results.csv"))
            if make_volcano:
                # Version 1: Without gene labels
                plot_journal_volcano(
                    res, 
                    title=f"Contrast: {contrast_label}", 
                    output_path=os.path.join(output_dir, "volcano.png"),
                    fdr_threshold=fdr_threshold,
                    label_genes=False
                )
                # Version 2: With numeric gene labels + legend (5 genes)
                plot_journal_volcano(
                    res, 
                    title=f"Contrast: {contrast_label}", 
                    output_path=os.path.join(output_dir, "volcano_labeled.png"),
                    fdr_threshold=fdr_threshold,
                    label_genes=True,
                    top_n_genes=5
                )

        return res

    except Exception as e:
        print(f"ERROR in raisintest: {e}")
        traceback.print_exc()
        raise


def run_pairwise_tests(
    fit,
    output_dir,
    groups_to_compare=None,
    control_group=None,
    fdrmethod='fdr_bh',
    n_permutations=100,
    fdr_threshold=0.05,
    top_n_genes: int = 50,
    make_summary_plots: bool = True,
    verbose=True
):
    """
    Runs pairwise comparisons and generates comprehensive visualizations.
    """
    
    X = fit['X']
    if isinstance(X, pd.DataFrame):
        x_cols = list(X.columns)
    else:
        unique_grps = np.unique(fit['group'])
        x_cols = ["(Intercept)"] + list(unique_grps)

    available_groups = np.unique(fit['group'])
    if groups_to_compare is None:
        groups_to_compare = available_groups
    
    pairs = []
    if control_group is not None:
        control_group_str = str(control_group)
        for g in groups_to_compare:
            if str(g) != control_group_str:
                pairs.append((g, control_group))
    else:
        pairs = list(combinations(groups_to_compare, 2))

    if verbose:
        print(f"Starting pairwise comparisons. Found {len(pairs)} pairs.")

    results_summary = {}
    all_results = {}

    for g1, g2 in pairs:
        comp_name = f"{g1}_vs_{g2}"
        if verbose:
            print(f"Testing: {comp_name}")
        
        idx_g1 = -1
        idx_g2 = -1
        for i, col in enumerate(x_cols):
            if str(g1) in str(col):
                idx_g1 = i
            if str(g2) in str(col):
                idx_g2 = i
            
        if idx_g1 == -1 or idx_g2 == -1:
            print(f"Skipping {comp_name}: Columns not found in design matrix.")
            continue

        contrast = np.zeros(len(x_cols))
        contrast[idx_g1] = 1
        contrast[idx_g2] = -1
        
        sub_dir = os.path.join(output_dir, comp_name)
        try:
            res = raisintest(
                fit,
                contrast=contrast,
                fdrmethod=fdrmethod,
                n_permutations=n_permutations,
                output_dir=sub_dir,
                fdr_threshold=fdr_threshold,
                make_volcano=True,
                verbose=False
            )
            results_summary[comp_name] = np.sum(res['FDR'] < fdr_threshold)
            all_results[comp_name] = res
        except Exception as e:
            print(f"Failed {comp_name}: {e}")
    
    # -------------------------------------------------------------------------
    # Generate Pseudobulk Visualizations
    # -------------------------------------------------------------------------
    if make_summary_plots and all_results and output_dir:
        summary_dir = os.path.join(output_dir, "summary_plots")
        os.makedirs(summary_dir, exist_ok=True)
        
        print("\nGenerating Visualizations...")
        
        # 1. Pseudobulk Heatmap (Standard "DoHeatmap" style)
        heatmap_path = os.path.join(summary_dir, "pseudobulk_heatmap.png")
        try:
            plot_pseudobulk_heatmap(
                fit,
                all_results, 
                heatmap_path, 
                fdr_threshold=fdr_threshold,
                top_n_per_cluster=top_n_genes
            )
        except Exception as e:
            print(f"Warning: failed to generate pseudobulk heatmap: {e}")
            traceback.print_exc()
        
        # 2. Dot Plot
        dotplot_path = os.path.join(summary_dir, "summary_dotplot.png")
        try:
            plot_journal_dotplot(
                all_results,
                dotplot_path,
                fdr_threshold=fdr_threshold,
                top_n=10
            )
        except Exception as e:
            print(f"Warning: failed to generate dotplot: {e}")
            traceback.print_exc()
        
        # 3. CSV of all results
        combined_results = []
        for comp, res in all_results.items():
            res_copy = res.copy()
            res_copy['comparison'] = comp
            combined_results.append(res_copy)
        if combined_results:
            pd.concat(combined_results).to_csv(
                os.path.join(summary_dir, "all_results_combined.csv")
            )
            
    return results_summary, all_results