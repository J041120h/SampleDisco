"""
Cell Type Proportion Testing - Python port of R RAISIN package

This module performs statistical testing for cell type proportions.
Tests if the proportions of cell types are differential across groups.

The key difference from simple OLS is the use of empirical Bayes moderation
similar to limma's eBayes, which provides more robust variance estimates
for small sample sizes.

Author: Original R code by Zhicheng Ji, Wenpin Hou, Hongkai Ji
Python port maintains compatibility with R implementation.
"""

import os
import itertools
import pandas as pd
import numpy as np
import scanpy as sc
import statsmodels.api as sm
from scipy.special import logit, digamma, polygamma
from scipy.optimize import brentq
from statsmodels.stats.multitest import multipletests
import warnings
import traceback


# ---------------------------------------------------------------------
#  limma-like eBayes functions
# ---------------------------------------------------------------------

def trigamma(x):
    """Trigamma function (second derivative of log-gamma)."""
    return polygamma(1, x)


def fit_f_dist(s2, df):
    """
    Fit an F-distribution to sample variances.
    
    This matches limma's fitFDist function which estimates:
    - scale (s0^2): prior variance
    - df2 (d0): prior degrees of freedom
    
    Uses the method from Smyth (2004) Stat. Appl. Genet. Mol. Biol.
    """
    s2 = np.array(s2)
    if np.isscalar(df):
        df = np.full_like(s2, df)
    
    ok = (s2 > 0) & np.isfinite(s2)
    if ok.sum() < 2:
        return np.median(s2[ok]) if ok.any() else 1.0, np.inf
    
    s2_ok = s2[ok]
    df_ok = df[ok]
    
    log_s2 = np.log(s2_ok)
    mean_log_s2 = np.mean(log_s2)
    var_log_s2 = np.var(log_s2, ddof=1)
    
    target_var = var_log_s2 - np.mean([trigamma(d/2) for d in df_ok])
    
    if target_var <= 0:
        df2 = np.inf
        scale = np.exp(mean_log_s2 - np.mean([digamma(d/2) - np.log(d/2) for d in df_ok]))
    else:
        try:
            def obj(df2):
                return trigamma(df2/2) - target_var
            
            if obj(0.01) * obj(1e6) < 0:
                df2 = brentq(obj, 0.01, 1e6)
            else:
                df2 = np.inf
        except Exception:
            df2 = np.inf
        
        if np.isfinite(df2):
            scale = np.exp(
                mean_log_s2
                - np.mean([digamma(d/2) for d in df_ok])
                + digamma(df2/2)
                - np.log(df2)
                + np.mean(np.log(df_ok))
            )
        else:
            scale = np.exp(mean_log_s2 - np.mean([digamma(d/2) - np.log(d/2) for d in df_ok]))
    
    return scale, df2


def squeeze_var(var, df, robust=False):
    """
    Squeeze sample variances toward a common value using empirical Bayes.
    Matches limma's squeezeVar function.
    """
    var = np.array(var)
    var_prior, df_prior = fit_f_dist(var, df)
    
    if np.isscalar(df):
        df = np.full_like(var, df)
    
    if np.isfinite(df_prior):
        var_post = (df * var + df_prior * var_prior) / (df + df_prior)
        df_post = df + df_prior
    else:
        var_post = np.full_like(var, var_prior)
        df_post = np.full_like(var, np.inf)
    
    return {
        'var_post': var_post,
        'var_prior': var_prior,
        'df_prior': df_prior,
        'df_post': df_post
    }


def ebayes_test(Y, X, coef=1):
    """
    Perform limma-style empirical Bayes moderated t-test.
    Mimics limma's lmFit + eBayes + topTable workflow.
    """
    Y = np.array(Y)
    X = np.array(X)
    
    n_features = Y.shape[0]
    n_samples = Y.shape[1] if Y.ndim > 1 else len(Y)
    
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    
    XTX_inv = np.linalg.pinv(X.T @ X)
    beta = XTX_inv @ X.T @ Y.T
    
    fitted = X @ beta
    residuals = Y.T - fitted
    
    df_residual = n_samples - X.shape[1]
    
    if df_residual > 0:
        sigma2 = np.sum(residuals ** 2, axis=0) / df_residual
    else:
        sigma2 = np.ones(n_features)
    
    var_coef = XTX_inv[coef, coef]
    
    squeeze_result = squeeze_var(sigma2, df_residual)
    sigma2_post = squeeze_result['var_post']
    df_post = squeeze_result['df_post']
    
    se_mod = np.sqrt(sigma2_post * var_coef)
    logFC = beta[coef, :]
    t_stat = logFC / se_mod
    
    from scipy import stats
    if np.isscalar(df_post):
        pval = 2 * stats.t.sf(np.abs(t_stat), df=df_post)
    else:
        pval = np.array([2 * stats.t.sf(np.abs(t), df=d) 
                        for t, d in zip(t_stat, df_post)])
    
    _, adj_pval, _, _ = multipletests(pval, method='fdr_bh')
    
    return pd.DataFrame({
        'logFC': logFC,
        't': t_stat,
        'P.Value': pval,
        'adj.P.Val': adj_pval
    })


# ---------------------------------------------------------------------
#  Main proportion test function (matching raisinfit interface)
# ---------------------------------------------------------------------

def proportion_test(
    adata,
    sample_col,
    group_col=None,
    sample_to_clade=None,
    celltype_col='celltype',
    output_dir=None,
    verbose=True
):
    """
    Perform proportion test on cell type proportions.
    
    Tests if proportions of cell types differ across groups using
    limma-style eBayes moderation on logit-transformed proportions.
    
    Parameters
    ----------
    adata : AnnData
        Cell-level AnnData with per-cell metadata in adata.obs.
    sample_col : str
        Column in adata.obs indicating sample ID.
    group_col : str, optional
        Column in adata.obs indicating sample group / cluster.
        If provided, this is used for grouping and `sample_to_clade`
        is ignored.
    sample_to_clade : dict, optional
        Mapping {sample_id -> group_label}. Only used if `group_col`
        is None.
    celltype_col : str
        Column in adata.obs indicating cell type.
    output_dir : str, optional
        Directory to save results and plots.
    verbose : bool
        Currently unused (kept for API compatibility).
    """
    
    # Significance level used for visualization and summary
    significance_level = 0.01

    # Validate that at least one source of grouping is provided
    if group_col is None and sample_to_clade is None:
        raise ValueError("Either group_col or sample_to_clade must be provided")
    
    # If both are provided, prefer group_col and ignore sample_to_clade
    if group_col is not None and sample_to_clade is not None:
        warnings.warn(
            "Both sample_to_clade and group_col provided. "
            "Using group_col and ignoring sample_to_clade."
        )
    
    # Validate columns
    if sample_col not in adata.obs.columns:
        raise KeyError(f"sample_col '{sample_col}' not found in adata.obs")
    
    if celltype_col not in adata.obs.columns:
        raise KeyError(f"celltype_col '{celltype_col}' not found in adata.obs")
    
    # If group_col is provided, it must exist in adata.obs
    if group_col is not None and group_col not in adata.obs.columns:
        raise KeyError(f"group_col '{group_col}' not found in adata.obs")
    
    # Get sample and celltype info
    samples = np.array(adata.obs[sample_col].values)
    celltypes = np.array(adata.obs[celltype_col].values)
    unique_samples = np.unique(samples)
    
    # -----------------------------------------------------------------
    # Build sample → group mapping
    # -----------------------------------------------------------------
    if group_col is not None:
        # Use group_col from adata.obs (preferred when available)
        sample_groups = {}
        for s in unique_samples:
            mask = samples == s
            vals = adata.obs.loc[mask, group_col].values
            # For safety, take the most common value among cells of the same sample
            most_common = pd.Series(vals).value_counts().idxmax()
            sample_groups[s] = most_common
    else:
        # Fall back to sample_to_clade mapping
        common_samples = [s for s in unique_samples if s in sample_to_clade]
        if len(common_samples) == 0:
            raise ValueError("No samples in data match keys in sample_to_clade")
        
        # Restrict samples and celltypes to those present in mapping
        sample_mask = np.isin(samples, common_samples)
        samples = samples[sample_mask]
        celltypes = celltypes[sample_mask]
        unique_samples = np.array(common_samples)
        
        sample_groups = {s: sample_to_clade[s] for s in unique_samples}
    
    # -----------------------------------------------------------------
    # Calculate cell type proportions per sample
    # -----------------------------------------------------------------
    ct_sample_counts = pd.crosstab(celltypes, samples)
    ct_sample_counts = ct_sample_counts.reindex(columns=unique_samples, fill_value=0)
    
    prop = ct_sample_counts.values.astype(float)
    prop = prop / prop.sum(axis=0, keepdims=True)
    
    # Handle boundary values
    min_nonzero = prop[prop > 0].min() if (prop > 0).any() else 1e-10
    prop = np.clip(prop, min_nonzero, 1 - min_nonzero)
    
    # Logit transform
    prop_logit = np.log(prop / (1 - prop))
    
    prop_logit_df = pd.DataFrame(
        prop_logit,
        index=ct_sample_counts.index,
        columns=unique_samples
    )
    
    # Get unique groups
    unique_groups = sorted(set(sample_groups.values()))
    
    if len(unique_groups) < 2:
        raise ValueError("Need at least 2 groups for comparison")
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # -----------------------------------------------------------------
    # Perform pairwise comparisons
    # -----------------------------------------------------------------
    all_results = {}
    
    for group1, group2 in itertools.combinations(unique_groups, 2):
        samples_g1 = [s for s in unique_samples if sample_groups[s] == group1]
        samples_g2 = [s for s in unique_samples if sample_groups[s] == group2]
        
        selected_samples = samples_g1 + samples_g2
        selected_prop_logit = prop_logit_df[selected_samples]
        
        group_labels = [1 if s in samples_g1 else 0 for s in selected_samples]
        X = np.column_stack([np.ones(len(group_labels)), group_labels])
        
        Y = selected_prop_logit.values
        result_df = ebayes_test(Y, X, coef=1)
        result_df.index = selected_prop_logit.index
        
        df_result = pd.DataFrame({
            'celltype': result_df.index,
            'logFC': result_df['logFC'].values,
            'p_value': result_df['P.Value'].values,
            'FDR': result_df['adj.P.Val'].values
        })
        
        df_result = df_result.sort_values('FDR')
        
        comparison_name = f"{group1}_vs_{group2}"
        all_results[comparison_name] = df_result
        
        if output_dir is not None:
            output_path = os.path.join(output_dir, f"proportion_test_{comparison_name}.csv")
            df_result.to_csv(output_path, index=False)
    
    # -----------------------------------------------------------------
    # Generate visualizations
    # -----------------------------------------------------------------
    if output_dir is not None:
        _proportion_test_visualization(
            prop_df=pd.DataFrame(prop, index=ct_sample_counts.index, columns=unique_samples),
            output_dir=output_dir,
            sample_groups=sample_groups,
            results=all_results,
            significance_level=significance_level
        )

        # -----------------------------------------------------------------
        # Write summary TXT of significant findings
        # -----------------------------------------------------------------
        summary_path = os.path.join(output_dir, "proportion_test_significant_summary.txt")
        lines = []
        lines.append(
            f"Significant cell type proportion differences (FDR < {significance_level})"
        )
        lines.append("")

        for comp_name in sorted(all_results.keys()):
            df = all_results[comp_name]
            sig_df = df[df["FDR"] < significance_level]

            lines.append(f"Comparison: {comp_name}")
            if sig_df.empty:
                lines.append("  No significant cell types.")
            else:
                for _, row in sig_df.iterrows():
                    lines.append(
                        f"  {row['celltype']}: "
                        f"logFC={row['logFC']:.4f}, "
                        f"p_value={row['p_value']:.4e}, "
                        f"FDR={row['FDR']:.4e}"
                    )
            lines.append("")

        with open(summary_path, "w") as f:
            f.write("\n".join(lines))
    
    return all_results


def _proportion_test_visualization(prop_df, output_dir, sample_groups, results,
                                   significance_level=0.05, verbose=False):
    """
    Internal function to generate visualizations.

    - Heatmap: group-averaged cell type proportions (Groups × Cell Types)
    - Boxplots: per-sample proportions for top significant cell types per comparison
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set a clean, journal-like style
    sns.set(style="whitegrid", context="talk")
    
    # -------------------------------------------------------------
    # Group-averaged heatmap: Groups (rows) × Cell Types (columns)
    # -------------------------------------------------------------
    group_labels = sorted(set(sample_groups.values()))
    group_prop = pd.DataFrame(index=prop_df.index, columns=group_labels, dtype=float)
    
    for g in group_labels:
        samples_g = [s for s, gg in sample_groups.items() if gg == g and s in prop_df.columns]
        if len(samples_g) == 0:
            group_prop[g] = np.nan
        else:
            group_prop[g] = prop_df[samples_g].mean(axis=1)
    
    # Drop groups with all NaNs just in case
    group_prop = group_prop.dropna(axis=1, how='all')
    
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        group_prop.T,
        cmap="viridis",
        annot=False,
        cbar=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean proportion"}
    )
    ax.set_title("Cell Type Proportions (Group-averaged)", pad=16)
    ax.set_xlabel("Cell Types")
    ax.set_ylabel("Groups")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "proportion_heatmap_group_by_celltype.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Boxplots for significant cell types (per comparison)
    # -------------------------------------------------------------
    top_n_per_comp = 6  # limit to top N cell types per comparison for clarity
    
    for comp_name, result_df in results.items():
        sig_df = result_df.loc[result_df['FDR'] < significance_level]
        if sig_df.empty:
            continue
        
        sig_celltypes = sig_df['celltype'].tolist()
        sig_celltypes = sig_celltypes[:top_n_per_comp]
        
        parts = comp_name.split("_vs_")
        if len(parts) != 2:
            continue
        group1, group2 = parts
        
        samples_g1 = [s for s, g in sample_groups.items() if g == group1 and s in prop_df.columns]
        samples_g2 = [s for s, g in sample_groups.items() if g == group2 and s in prop_df.columns]
        
        if len(samples_g1) == 0 or len(samples_g2) == 0:
            continue
        
        long_records = []
        for cell_type in sig_celltypes:
            if cell_type not in prop_df.index:
                continue
            # group1 samples
            for s in samples_g1:
                long_records.append({
                    "celltype": str(cell_type),
                    "Proportion": prop_df.loc[cell_type, s],
                    "Group": group1
                })
            # group2 samples
            for s in samples_g2:
                long_records.append({
                    "celltype": str(cell_type),
                    "Proportion": prop_df.loc[cell_type, s],
                    "Group": group2
                })
        
        if not long_records:
            continue
        
        plot_df = pd.DataFrame(long_records)
        
        plt.figure(figsize=(max(6, 1.8 * len(sig_celltypes) + 2), 6))
        ax = sns.boxplot(
            data=plot_df,
            x="celltype",
            y="Proportion",
            hue="Group"
        )
        ax.set_title(f"Cell Type Proportions: {comp_name}", pad=16)
        ax.set_xlabel("Cell Types")
        ax.set_ylabel("Proportion")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
        plt.tight_layout()
        
        clean_comp = comp_name.replace(" ", "_")
        boxplot_path = os.path.join(output_dir, f"proportion_boxplot_{clean_comp}.png")
        plt.savefig(boxplot_path, dpi=300)
        plt.close()
