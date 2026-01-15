"""
RAISIN Testing - Python port of R RAISIN package

This module performs statistical testing for the RAISIN model.
Tests if a coefficient or combination of coefficients in the fixed effects
design matrix is 0, generating corresponding p-values and FDRs.

Author: Original R code by Zhicheng Ji, Wenpin Hou, Hongkai Ji
Python port maintains compatibility with R implementation.

Enhanced with journal-style visualizations (Seaborn) for multi-group DE analysis.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
from statsmodels.stats.multitest import multipletests
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# ============================================================================
# PLOTTING AESTHETICS (Journal Style)
# ============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'  # Text as text, not paths
sns.set_context("paper")  # Scale elements for publication

# ============================================================================
# VISUALIZATION FUNCTIONS (New)
# ============================================================================

def plot_journal_volcano(result, title, output_path, fdr_threshold=0.05):
    """
    Creates a minimalist, publication-ready volcano plot.
    """
    df = result.copy()
    # Handle zero p-values for log transformation
    df['nlog10'] = -np.log10(df['FDR'] + 1e-300)
    
    plt.figure(figsize=(5, 5))
    
    # Assign colors
    colors = []
    sizes = []
    alphas = []
    
    for _, row in df.iterrows():
        if row['FDR'] > fdr_threshold:
            colors.append('#dddddd')  # Grey (NS)
            sizes.append(10)
            alphas.append(0.5)
        elif row['Foldchange'] > 0:
            colors.append('#B31B1B')  # Deep Red (Up)
            sizes.append(25)
            alphas.append(0.8)
        else:
            colors.append('#0047AB')  # Cobalt Blue (Down)
            sizes.append(25)
            alphas.append(0.8)
            
    plt.scatter(df['Foldchange'], df['nlog10'], c=colors, s=sizes, alpha=0.7, 
                linewidth=0, rasterized=True)
    
    # Guidelines
    plt.axhline(-np.log10(fdr_threshold), linestyle='--', color='black', linewidth=0.8, alpha=0.5)
    plt.axvline(0, linestyle='-', color='black', linewidth=0.5, alpha=0.5)
    
    # Labels
    plt.title(title, fontsize=10, weight='bold')
    plt.xlabel("Log Fold Change", fontsize=9)
    plt.ylabel("-log10(FDR)", fontsize=9)
    
    # Label top 5 significant genes
    top_genes = df.nsmallest(5, 'FDR')
    texts = []
    for gene, row in top_genes.iterrows():
        if row['FDR'] < fdr_threshold:
            texts.append(plt.text(row['Foldchange'], row['nlog10'], gene, fontsize=7))
            
    try:
        from adjustText import adjust_text
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
    except ImportError:
        pass # Graceful degradation if adjustText not installed

    sns.despine()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_journal_heatmap(all_results, output_path, fdr_threshold=0.05, top_n_per_cluster=20, figsize=(10, 12)):
    """
    Generates a Clustermap (Heatmap clustered by gene).
    """
    # 1. Identify Top Markers per comparison
    marker_genes = set()
    for comp_name, df in all_results.items():
        sig_df = df[df['FDR'] < fdr_threshold]
        # Prioritize genes that are UP in the cluster (Positive FC)
        top_genes = sig_df[sig_df['Foldchange'] > 0].nlargest(top_n_per_cluster, 'Foldchange').index.tolist()
        marker_genes.update(top_genes)
        
    marker_genes = list(marker_genes)
    if len(marker_genes) < 2:
        print("Warning: Not enough significant marker genes for heatmap.")
        return

    # 2. Build Data Matrix (Genes x Comparisons)
    comps = list(all_results.keys())
    matrix = pd.DataFrame(0.0, index=marker_genes, columns=comps)
    
    for comp_name, df in all_results.items():
        common = df.index.intersection(marker_genes)
        matrix.loc[common, comp_name] = df.loc[common, 'Foldchange']

    # 3. Clean labels
    matrix.columns = [c.replace('_vs_Rest', '').replace('Cluster', 'C') for c in matrix.columns]

    # 4. Plot with Seaborn Clustermap
    # row_cluster=True performs the requested clustering by gene
    g = sns.clustermap(
        matrix,
        cmap="RdBu_r",
        center=0,
        z_score=None,  # Plot raw LogFC. Set to 0 to Z-score normalize rows.
        col_cluster=True,
        row_cluster=True,
        figsize=figsize,
        dendrogram_ratio=(0.15, 0.05),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        yticklabels=True,
        xticklabels=True
    )
    
    g.ax_heatmap.set_ylabel("Genes", fontsize=10)
    g.ax_heatmap.tick_params(axis='y', labelsize=6)
    g.ax_heatmap.tick_params(axis='x', labelsize=9, rotation=45)
    g.ax_cbar.set_title("LogFC", fontsize=8)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Clustered Heatmap: {output_path}")


def plot_journal_dotplot(all_results, output_path, fdr_threshold=0.05, top_n=10, figsize=None):
    """
    Creates a Seurat-style Dot Plot.
    Color = Log Fold Change, Size = Significance.
    """
    plot_data = []
    
    # Define target genes (Top N up-regulated per group)
    target_genes = set()
    for comp, df in all_results.items():
        sig = df[df['FDR'] < fdr_threshold]
        top = sig.nlargest(top_n, 'Foldchange').index.tolist()
        target_genes.update(top)
    
    target_genes = sorted(list(target_genes))
    if not target_genes:
        print("Warning: No genes to plot in DotPlot.")
        return

    # Build Long Format DataFrame
    for comp, df in all_results.items():
        available = df.index.intersection(target_genes)
        subset = df.loc[available].copy()
        subset['Gene'] = subset.index
        subset['Comparison'] = comp.replace('_vs_Rest', '')
        # Cap significance for visualization
        subset['LogFDR'] = -np.log10(subset['FDR'] + 1e-300)
        subset['LogFDR_clipped'] = subset['LogFDR'].clip(upper=50)
        plot_data.append(subset[['Gene', 'Comparison', 'Foldchange', 'LogFDR_clipped']])

    if not plot_data:
        return

    final_df = pd.concat(plot_data)

    if figsize is None:
        figsize = (len(all_results) * 0.8 + 2, len(target_genes) * 0.25 + 2)
        
    fig, ax = plt.subplots(figsize=figsize)
    
    # Mapping
    comparisons = final_df['Comparison'].unique()
    genes = final_df['Gene'].unique()
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
    ax.set_xticklabels(comparisons, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label("Log Fold Change", fontsize=9)
    
    # Size Legend
    sizes = [10, 30, 50]
    legend_elements = [plt.scatter([], [], s=s*5, c='gray', label=str(s)) for s in sizes]
    ax.legend(handles=legend_elements, title="-log10(FDR)", bbox_to_anchor=(1.2, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Dotplot: {output_path}")


# ============================================================================
# CORE TESTING FUNCTIONS (Logic Preserved)
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
    make_ma_plot: bool = False, # Deprecated in new style
    top_n_label: int = 10,
    verbose: bool = True,
):
    """
    Statistical testing for RAISIN model.
    """
    try:
        # -----------------------------------------------------------------
        # 1. Extract Data
        # -----------------------------------------------------------------
        X = fit['X']
        if isinstance(X, pd.DataFrame):
            X_df = X
            X = X.values.astype(np.float64)
            x_colnames = list(X_df.columns)
        else:
            X = np.array(X, dtype=np.float64)
            x_colnames = [f"coef_{i}" for i in range(X.shape[1])]
        
        means = fit['mean']
        if isinstance(means, pd.DataFrame):
            gene_names = np.array(means.index)
            means = means.values.astype(np.float64)
        else:
            gene_names = np.arange(means.shape[0])
            means = np.array(means, dtype=np.float64)
        
        G = means.shape[0]
        
        group = fit['group']
        if isinstance(group, pd.Series):
            group = group.values
        group = np.array(group, dtype=str)
        
        Z = fit['Z']
        if isinstance(Z, pd.DataFrame):
            Z = Z.values.astype(np.float64)
        else:
            Z = np.array(Z, dtype=np.float64)
        
        sigma2 = fit['sigma2']
        if isinstance(sigma2, pd.DataFrame):
            sigma2_values = sigma2.values.astype(np.float64)
            sigma2_columns = list(sigma2.columns)
        else:
            sigma2_values = np.array(sigma2, dtype=np.float64)
            sigma2_columns = list(range(sigma2_values.shape[1]))
        
        omega2 = fit['omega2']
        if isinstance(omega2, pd.DataFrame):
            omega2 = omega2.values.astype(np.float64)
        else:
            omega2 = np.array(omega2, dtype=np.float64)
            
        failgroup = fit.get('failgroup', [])

        # -----------------------------------------------------------------
        # 2. Filter Genes (Optional)
        # -----------------------------------------------------------------
        if min_samples is not None:
            sample_counts = np.sum(means > 0, axis=1)
            gene_mask = sample_counts >= min_samples
            
            if verbose:
                print(f"Filtering: {np.sum(~gene_mask)} genes removed (< {min_samples} samples)")
            
            means = means[gene_mask, :]
            gene_names = np.array(gene_names)[gene_mask]
            sigma2_values = sigma2_values[gene_mask, :]
            omega2 = omega2[gene_mask, :]
            G = means.shape[0]

        # -----------------------------------------------------------------
        # 3. Handle Contrast
        # -----------------------------------------------------------------
        if contrast is None:
            contrast = np.zeros(X.shape[1])
            if coef < 1 or coef > X.shape[1]:
                raise ValueError(f"coef must be between 1 and {X.shape[1]}")
            contrast[coef - 1] = 1
            contrast_label = x_colnames[coef - 1]
            
            if "(Intercept)" in x_colnames and np.sum(contrast) != 0:
                if verbose:
                    print(f"Warning: Testing single coefficient '{contrast_label}' in model with Intercept.")
        else:
            contrast = np.array(contrast, dtype=np.float64)
            contrast_label = "custom_contrast"

        # -----------------------------------------------------------------
        # 4. Calculate Statistics
        # -----------------------------------------------------------------
        XTX = X.T @ X
        XTX_inv = np.linalg.pinv(XTX)
        
        k = contrast @ XTX_inv @ X.T
        b = means @ k  # Fold change
        
        unique_groups = np.unique(group)
        if failgroup and set(unique_groups).issubset(set(failgroup)):
            warnings.warn('All groups failed variance estimation. Returning FDR=1.')
            return pd.DataFrame({'Foldchange': b, 'FDR': 1.0}, index=gene_names)
        
        kZ = k @ Z
        sigma2_by_group = np.zeros((G, len(group)))
        for i, g in enumerate(group):
            if g in sigma2_columns:
                col_idx = sigma2_columns.index(g)
                sigma2_by_group[:, i] = sigma2_values[:, col_idx]
        
        random_var = np.sum((kZ ** 2) * sigma2_by_group, axis=1)
        fixed_var = np.sum((k ** 2) * omega2, axis=1)
        a = random_var + fixed_var
        
        with np.errstate(divide='ignore', invalid='ignore'):
            stat = b / np.sqrt(a)
        stat = np.where(np.isfinite(stat), stat, 0)

        # -----------------------------------------------------------------
        # 5. Permutations (Degrees of Freedom Estimation)
        # -----------------------------------------------------------------
        if verbose:
            print(f"Running {n_permutations} permutations...")
            
        simu_stat = []
        for _ in range(n_permutations):
            perm_idx = np.random.permutation(X.shape[0])
            perX = X[perm_idx, :]
            
            perXTX = perX.T @ perX
            perXTX_inv = np.linalg.pinv(perXTX)
            perm_k = contrast @ perXTX_inv @ perX.T
            
            perm_kZ = perm_k @ Z
            perm_random_var = np.sum((perm_kZ ** 2) * sigma2_by_group, axis=1)
            perm_fixed_var = np.sum((perm_k ** 2) * omega2, axis=1)
            perm_a = perm_random_var + perm_fixed_var
            
            with np.errstate(divide='ignore', invalid='ignore'):
                perm_stat = (means @ perm_k) / np.sqrt(perm_a)
            
            perm_stat = perm_stat[np.isfinite(perm_stat)]
            
            if len(perm_stat) > 2000:
                 perm_stat = np.random.choice(perm_stat, 2000, replace=False)
            
            simu_stat.extend(perm_stat.tolist())
            
        simu_stat = np.array(simu_stat)
        
        # -----------------------------------------------------------------
        # 6. Fit Distribution (Normal vs t)
        # -----------------------------------------------------------------
        if len(simu_stat) == 0:
            pval = 2 * stats.norm.sf(np.abs(stat))
        else:
            pnorm_ll = np.sum(stats.norm.logpdf(simu_stat))
            
            df_range = np.arange(1, 100.1, 0.5)
            best_ll = -np.inf
            best_df = 100
            
            for df_val in df_range:
                ll = np.sum(stats.t.logpdf(simu_stat, df=df_val))
                if ll > best_ll:
                    best_ll = ll
                    best_df = df_val
            
            if best_ll > pnorm_ll:
                if verbose:
                    print(f"Estimated distribution: t-distribution (df={best_df:.1f})")
                pval = 2 * stats.t.sf(np.abs(stat), df=best_df)
            else:
                if verbose:
                    print("Estimated distribution: Normal")
                pval = 2 * stats.norm.sf(np.abs(stat))

        # -----------------------------------------------------------------
        # 7. FDR and Output
        # -----------------------------------------------------------------
        _, fdr, _, _ = multipletests(pval, method=fdrmethod)
        
        res = pd.DataFrame({
            'Foldchange': b,
            'stat': stat,
            'pvalue': pval,
            'FDR': fdr
        }, index=gene_names)
        
        res = res.iloc[np.lexsort((-np.abs(res['stat'].values), res['FDR'].values))]
        
        if verbose:
            n_sig = np.sum(res['FDR'] < fdr_threshold)
            print(f"Found {n_sig} significant genes (FDR < {fdr_threshold})")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            res.to_csv(os.path.join(output_dir, "raisin_results.csv"))
            
            if make_volcano:
                # REPLACED: Use new journal style volcano
                plot_journal_volcano(
                    res, 
                    title=f"Contrast: {contrast_label}", 
                    output_path=os.path.join(output_dir, "volcano_plot.png"),
                    fdr_threshold=fdr_threshold
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
    Automatically runs pairwise comparisons between groups and generates
    comprehensive journal-style visualizations.
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
    if control_group:
        control_group_str = str(control_group)
        avail_str = [str(g) for g in available_groups]
        
        if control_group_str not in avail_str:
            raise ValueError(f"Control group '{control_group}' not found in data.")
            
        for g in groups_to_compare:
            if str(g) != control_group_str:
                pairs.append((g, control_group))
    else:
        pairs = list(combinations(groups_to_compare, 2))

    if verbose:
        print(f"Starting pairwise comparisons. Found {len(pairs)} pairs to test.")

    results_summary = {}
    all_results = {}

    for g1, g2 in pairs:
        comp_name = f"{g1}_vs_{g2}"
        if verbose:
            print(f"\n--- Testing: {g1} (Positive) vs {g2} (Negative) ---")
        
        idx_g1 = -1
        idx_g2 = -1
        
        g1_str = str(g1)
        g2_str = str(g2)
        
        for i, col in enumerate(x_cols):
            col_str = str(col)
            if col_str == g1_str or col_str.endswith(g1_str):
                idx_g1 = i
            if col_str == g2_str or col_str.endswith(g2_str):
                idx_g2 = i
        
        if idx_g1 == -1 or idx_g2 == -1:
            print(f"Skipping {comp_name}: Could not find corresponding columns in Design Matrix.")
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
                verbose=verbose
            )
            results_summary[comp_name] = np.sum(res['FDR'] < fdr_threshold)
            all_results[comp_name] = res
            
        except Exception as e:
            print(f"Failed comparison {comp_name}: {e}")
            traceback.print_exc()

    if verbose:
        print("\n" + "=" * 60)
        print("Pairwise Comparison Summary")
        print("=" * 60)
        for pair, count in results_summary.items():
            print(f"  {pair}: {count} significant genes")
    
    # -------------------------------------------------------------------------
    # Generate Journal Style Visualizations
    # -------------------------------------------------------------------------
    if make_summary_plots and all_results and output_dir:
        summary_dir = os.path.join(output_dir, "summary_plots")
        os.makedirs(summary_dir, exist_ok=True)
        
        print("\nGenerating Journal-Style Visualizations...")
        
        # 1. Clustered Heatmap (Clustered by Gene as requested)
        heatmap_path = os.path.join(summary_dir, "heatmap_clustered.png")
        plot_journal_heatmap(
            all_results, 
            heatmap_path, 
            fdr_threshold=fdr_threshold,
            top_n_per_cluster=top_n_genes
        )
        
        # 2. Dot Plot
        dotplot_path = os.path.join(summary_dir, "summary_dotplot.png")
        plot_journal_dotplot(
            all_results,
            dotplot_path,
            fdr_threshold=fdr_threshold,
            top_n=10
        )
        
        # 3. Combined CSV
        combined_results = []
        for comp, res in all_results.items():
            res_copy = res.copy()
            res_copy['comparison'] = comp
            combined_results.append(res_copy)
        
        if combined_results:
            combined_df = pd.concat(combined_results)
            combined_df.to_csv(os.path.join(summary_dir, "all_results_combined.csv"))
            
    return results_summary, all_results