#!/usr/bin/env python3
"""
RAISIN statistical testing — Python port of R RAISIN package (Ji, Hou, Ji).

Tests whether a fixed-effect contrast is zero, using permutation-calibrated
p-values. Enhanced with pseudobulk/cluster-averaged Seurat-style visualizations.
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

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

# White background and no grid globally
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False

sns.set_context("paper")
sns.set_style("white")


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

    is_sig = df['FDR'] < fdr_threshold
    is_up = df['Foldchange'] > 0

    ax.scatter(
        df.loc[~is_sig, 'Foldchange'],
        df.loc[~is_sig, 'nlog10'],
        c='#dddddd',
        s=5,
        alpha=0.5,
        linewidth=0,
        rasterized=True,
    )

    ax.scatter(
        df.loc[is_sig & is_up, 'Foldchange'],
        df.loc[is_sig & is_up, 'nlog10'],
        c='#B31B1B',
        s=15,
        alpha=0.8,
        linewidth=0,
        rasterized=True,
    )

    ax.scatter(
        df.loc[is_sig & ~is_up, 'Foldchange'],
        df.loc[is_sig & ~is_up, 'nlog10'],
        c='#0047AB',
        s=15,
        alpha=0.8,
        linewidth=0,
        rasterized=True,
    )

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
    
    if label_genes and top_n_genes > 0:
        sig_df = df[is_sig].sort_values('FDR')
        sig_df = sig_df.head(top_n_genes)
        
        texts = []
        labeled_genes = []

        for idx, (gene, row) in enumerate(sig_df.iterrows(), start=1):
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

        if texts:
            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
            )

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
    "DoHeatmap"-style pseudobulk heatmap.

    Averages fit['mean'] (gene × sample) to gene × group, selects the top
    upregulated marker genes per cluster from DE results, and plots row-wise
    z-scored expression.
    """
    means = fit['mean']
    if isinstance(means, pd.DataFrame):
        means_df = means.copy()
    else:
        means_df = pd.DataFrame(means)

    groups = np.array(fit['group']).astype(str)

    if means_df.shape[1] != len(groups):
        raise ValueError(
            f"Mismatch between means columns ({means_df.shape[1]}) "
            f"and group labels ({len(groups)})."
        )
    
    pseudobulk = means_df.T.groupby(groups).mean().T  # genes × clusters

    cluster_markers = {g: [] for g in pseudobulk.columns}

    for comp_name, res in all_results.items():
        parts = comp_name.split('_vs_')
        if len(parts) != 2:
            continue
        pos_grp, neg_grp = parts[0], parts[1]
        
        # Get significant UP genes (markers for pos_grp)
        sig_up = res[(res['FDR'] < fdr_threshold) & (res['Foldchange'] > 0)]

        if sig_up.empty:
            continue

        top_genes = sig_up.nlargest(top_n_per_cluster, 'Foldchange').index.tolist()
        
        if pos_grp in cluster_markers:
            cluster_markers[pos_grp].extend(top_genes)

    final_gene_list = []
    gene_cluster_labels = []
    seen_genes = set()

    sorted_clusters = sorted(map(str, pseudobulk.columns))
    pseudobulk = pseudobulk.loc[:, sorted_clusters]

    for cluster in sorted_clusters:
        candidates = cluster_markers.get(cluster, [])
        for gene in candidates:
            if gene not in seen_genes and gene in pseudobulk.index:
                final_gene_list.append(gene)
                gene_cluster_labels.append(cluster)
                seen_genes.add(gene)

    if not final_gene_list:
        print("Warning: No significant markers found to plot. Skipping pseudobulk heatmap.")
        return

    plot_matrix = pseudobulk.loc[final_gene_list, sorted_clusters]

    plot_values = np.asarray(plot_matrix, dtype=float)

    # Row-wise z-score (per gene across clusters)
    z_matrix_values = zscore(plot_values, axis=1, nan_policy='omit')

    if z_matrix_values.ndim == 1:  # single-gene edge case
        z_matrix_values = z_matrix_values[np.newaxis, :]

    z_matrix_values = np.nan_to_num(
        z_matrix_values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    z_matrix_values = np.clip(z_matrix_values, -2.5, 2.5)

    z_matrix = pd.DataFrame(
        z_matrix_values,
        index=plot_matrix.index,
        columns=plot_matrix.columns
    )

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(False)
    
    sns.heatmap(
        z_matrix,
        cmap="RdBu_r",
        center=0,
        xticklabels=True,
        yticklabels=False,
        cbar_kws={"label": "Z-score Expr", "shrink": 0.5},
        ax=ax
    )
    
    current_cluster = gene_cluster_labels[0]
    for i, cluster in enumerate(gene_cluster_labels):
        if cluster != current_cluster:
            ax.hlines(i, *ax.get_xlim(), colors='white', linewidth=1)
            current_cluster = cluster
    
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
    """Seurat-style dot plot: dot size = -log10(FDR), color = log fold change."""
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
    random_state: int = 42,
):
    """
    Compute gene-level test statistics and permutation-calibrated p-values for a
    contrast of fixed-effect coefficients, then write optional volcano and DE outputs.
    """
    try:
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

        if min_samples is not None:
            sample_counts = np.sum(means > 0, axis=1)
            mask = sample_counts >= min_samples
            means = means[mask, :]
            gene_names = np.array(gene_names)[mask]
            sigma2_values = sigma2_values[mask, :]
            omega2 = omega2[mask, :]
        G = means.shape[0]

        if contrast is None:
            contrast = np.zeros(X.shape[1])
            if coef < 1 or coef > X.shape[1]:
                raise ValueError(f"coef must be between 1 and {X.shape[1]}")
            contrast[coef - 1] = 1
            contrast_label = x_cols[coef - 1]
        else:
            contrast = np.array(contrast, dtype=np.float64)
            contrast_label = "custom_contrast"

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

        if verbose:
            print(f"Running {n_permutations} permutations...")
        rng = np.random.default_rng(random_state)
        simu_stat = []
        for _ in range(n_permutations):
            perm_idx = rng.permutation(X.shape[0])
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
                p_s = rng.choice(p_s, 1000, replace=False)
            simu_stat.extend(p_s)

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
                plot_journal_volcano(
                    res,
                    title=f"Contrast: {contrast_label}",
                    output_path=os.path.join(output_dir, "volcano.png"),
                    fdr_threshold=fdr_threshold,
                    label_genes=False
                )
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


def plot_cluster_gene_zscore(group_gene_df, output_path, title=None, max_genes=60):
    """Cluster x gene z-score heatmap from a (genes x clusters) group-mean matrix.

    Rows = genes (names shown), columns = clusters/groups; values are per-gene
    z-scores of the group-mean expression (standardised across clusters). Renders
    as long as >= 1 gene is provided (no significance gating here -- the caller
    chooses the genes, so it is produced for every cell type).
    """
    if group_gene_df is None or group_gene_df.shape[0] == 0:
        print("plot_cluster_gene_zscore: no genes to plot; skipping.")
        return
    df = group_gene_df.iloc[:max_genes]
    cols = sorted(df.columns, key=str)
    vals = np.asarray(df.loc[:, cols], dtype=float)
    z = zscore(vals, axis=1, nan_policy="omit")
    if z.ndim == 1:
        z = z[np.newaxis, :]
    z = np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -2.5, 2.5)
    zdf = pd.DataFrame(z, index=df.index, columns=cols)

    fig, ax = plt.subplots(figsize=(max(4, 0.7 * len(cols) + 2),
                                    max(4, 0.22 * len(zdf) + 1)))
    fig.patch.set_facecolor("white")
    sns.heatmap(zdf, cmap="RdBu_r", center=0, xticklabels=True, yticklabels=True,
                cbar_kws={"label": "z-score (per gene across clusters)", "shrink": 0.5},
                ax=ax)
    ax.set_xlabel("Cluster", fontsize=12, weight="bold")
    ax.set_ylabel(f"Gene (n={len(zdf)})", fontsize=12)
    ax.set_title(title or "Cluster x gene z-score (group-mean expression)", fontsize=13)
    ax.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved cluster x gene z-score heatmap: {output_path}")


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
    verbose=True,
    random_state: int = 42,
):
    """
    Run all pairwise (or vs-control) RAISIN tests and generate summary visualizations.

    If control_group is given, tests every other group against it. Otherwise
    all pairs are tested. Writes per-comparison volcano plots, a CSV/TXT summary,
    a pseudobulk heatmap, a dot plot, and a cluster×gene z-score heatmap.
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
        
        # Exact match group -> design column (substring matching is ambiguous,
        # e.g. group "1" spuriously matches "10"/"(Intercept)", and fails when
        # the columns are unnamed integers).
        col_lookup = {str(c): i for i, c in enumerate(x_cols)}
        idx_g1 = col_lookup.get(str(g1), -1)
        idx_g2 = col_lookup.get(str(g2), -1)

        if idx_g1 == -1 or idx_g2 == -1:
            print(f"Skipping {comp_name}: groups {g1!r}/{g2!r} not found in "
                  f"design columns {[str(c) for c in x_cols]}.")
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
                verbose=False,
                random_state=random_state,
            )
            results_summary[comp_name] = np.sum(res['FDR'] < fdr_threshold)
            all_results[comp_name] = res
        except Exception as e:
            print(f"Failed {comp_name}: {e}")
    
    if all_results and output_dir:
        summary_dir = os.path.join(output_dir, "summary_plots")
        os.makedirs(summary_dir, exist_ok=True)
        rows = []
        for comp, res in all_results.items():
            sig = res["FDR"] < fdr_threshold
            fc = res["Foldchange"]
            rows.append({"comparison": comp, "n_genes": int(len(res)),
                         "n_sig": int(sig.sum()),
                         "n_sig_up": int((sig & (fc > 0)).sum()),
                         "n_sig_down": int((sig & (fc < 0)).sum())})
        summ = pd.DataFrame(rows)
        summ.to_csv(os.path.join(summary_dir, "raisin_summary.csv"), index=False)
        with open(os.path.join(summary_dir, "raisin_summary.txt"), "w") as fh:
            fh.write(f"RAISIN pairwise summary (FDR < {fdr_threshold})\n")
            fh.write("=" * 60 + "\n\n" + summ.to_string(index=False) + "\n\n")
            for comp, res in all_results.items():
                sig = res[res["FDR"] < fdr_threshold].sort_values("FDR")
                fh.write(f"--- {comp}: top {min(top_n_genes, len(sig))} of "
                         f"{len(sig)} significant (FDR<{fdr_threshold}) ---\n")
                if len(sig) == 0:
                    fh.write("  (none)\n\n")
                    continue
                for gene, r in sig.head(top_n_genes).iterrows():
                    fh.write(f"  {str(gene):20s} logFC={r['Foldchange']:+.3f}  "
                             f"FDR={r['FDR']:.2e}\n")
                fh.write("\n")
        if verbose:
            print(f"Wrote RAISIN summary: {os.path.join(summary_dir, 'raisin_summary.txt')}")

    if make_summary_plots and all_results and output_dir:
        summary_dir = os.path.join(output_dir, "summary_plots")
        os.makedirs(summary_dir, exist_ok=True)
        
        print("\nGenerating Visualizations...")

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

        # Cluster×gene z-score heatmap; falls back to top-ranked genes when none are significant.
        try:
            means = fit['mean']
            means_df = means.copy() if isinstance(means, pd.DataFrame) else pd.DataFrame(means)
            grp = np.array(fit['group']).astype(str)
            if means_df.shape[1] == len(grp):
                pseudobulk = means_df.T.groupby(grp).mean().T  # genes × clusters
                pooled = pd.concat(list(all_results.values()))
                sig = pooled[pooled['FDR'] < fdr_threshold].sort_values('FDR')
                genes = list(dict.fromkeys(sig.index.tolist()))
                if not genes:
                    genes = list(dict.fromkeys(pooled.sort_values('FDR').index.tolist()))
                genes = [g for g in genes if g in pseudobulk.index][:top_n_genes]
                if genes:
                    plot_cluster_gene_zscore(
                        pseudobulk.loc[genes],
                        os.path.join(summary_dir, "cluster_gene_zscore.png"),
                    )
        except Exception as e:
            print(f"Warning: failed to generate cluster_gene_zscore heatmap: {e}")
            traceback.print_exc()

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