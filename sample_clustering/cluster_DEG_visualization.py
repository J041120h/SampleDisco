import os
import itertools
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, kruskal, zscore
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from matplotlib.colors import LinearSegmentedColormap

def cluster_dge_visualization(sample_to_clade: dict, folder_path: str, output_dir: str):
    """
    Perform pairwise differential expression analysis across all valid clade pairs.
    
    Parameters:
        sample_to_clade (dict): Mapping from sample name to clade ID.
        folder_path (str): Path to folder containing expression.csv.
        output_dir (str): Directory to save results and plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    expr_path = os.path.join(folder_path, "expression.csv")

    # Load expression and add clade column
    expr = pd.read_csv(expr_path, index_col=0)
    expr = expr.loc[expr.index.intersection(sample_to_clade)]
    expr["clade"] = expr.index.map(sample_to_clade)

    # Determine valid clades with >=2 samples
    clade_counts = expr["clade"].value_counts()
    valid_clades = clade_counts[clade_counts >= 2].index.tolist()

    pair_summary = []

    for clade1, clade2 in itertools.combinations(valid_clades, 2):
        group1 = expr[expr["clade"] == clade1].drop(columns="clade")
        group2 = expr[expr["clade"] == clade2].drop(columns="clade")

        # Filter genes
        min_expr_threshold = 0.1
        min_samples_expressed = 3
        filtered_genes = [
            g for g in group1.columns
            if ((group1[g] > min_expr_threshold).sum() >= min_samples_expressed or
                (group2[g] > min_expr_threshold).sum() >= min_samples_expressed)
        ]

        results = []
        for gene in filtered_genes:
            v1, v2 = group1[gene], group2[gene]
            _, t_pval = ttest_ind(v1, v2, equal_var=False)
            _, w_pval = mannwhitneyu(v1, v2, alternative='two-sided')
            m1, m2 = np.mean(v1), np.mean(v2)
            log2fc = np.log2((m1 + 1e-6) / (m2 + 1e-6)) if (m1 > 0 and m2 > 0) else np.nan
            results.append({'gene': gene, 't_pval': t_pval, 'w_pval': w_pval, 'log2fc': log2fc})

        dge_df = pd.DataFrame(results)
        dge_df["t_qval"] = multipletests(dge_df["t_pval"], method="fdr_bh")[1]
        dge_df["w_qval"] = multipletests(dge_df["w_pval"], method="fdr_bh")[1]
        dge_df["sig_t"] = dge_df["t_qval"] < 0.05
        dge_df["sig_w"] = dge_df["w_qval"] < 0.05
        dge_df["sig_both"] = dge_df["sig_t"] & dge_df["sig_w"]
        dge_df = dge_df.sort_values(["sig_both", "t_qval", "w_qval"], ascending=[False, True, True])

        # Save to file
        tag = f"clade{clade1}_vs_clade{clade2}"
        dge_csv_path = os.path.join(output_dir, f"DEG_{tag}.csv")
        dge_df.to_csv(dge_csv_path, index=False)

        # Summary for report
        n_sig = dge_df["sig_both"].sum()
        pair_summary.append({"clade1": clade1, "clade2": clade2, "num_DEG": n_sig})

        # Volcano plot
        dge_df["-log10_t_qval"] = -np.log10(dge_df["t_qval"])
        dge_df["color_t"] = np.where(dge_df["sig_t"], "red", "gray")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=dge_df, x="log2fc", y="-log10_t_qval", hue="color_t",
                        palette={"gray": "gray", "red": "red"}, legend=False)
        top_t = dge_df[dge_df["sig_t"]].nsmallest(10, "t_qval")
        texts = [
            plt.text(row["log2fc"], row["-log10_t_qval"], row["gene"],
                     fontsize=8, ha="right" if row["log2fc"] < 0 else "left")
            for _, row in top_t.iterrows()
        ]
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
        plt.axhline(-np.log10(0.05), color="blue", linestyle="--")
        plt.axvline(0, color="black", linestyle="-")
        plt.xlabel("log2 Fold Change")
        plt.ylabel("-log10 FDR-adjusted p-value (t-test)")
        plt.title(f"Volcano Plot ({tag})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"volcano_{tag}.png"), dpi=300)
        plt.close()

    # Save overall summary
    summary_df = pd.DataFrame(pair_summary)
    summary_df.to_csv(os.path.join(output_dir, "pairwise_DEG_summary.csv"), index=False)

    print(f"\nCompleted DGE for {len(pair_summary)} clade pairs.")
    print(summary_df)

def create_expression_heatmap(
        sample_to_clade: dict,
        folder_path: str,
        output_dir: str,
        gene_list=None,
        figsize=(12, 14),
        dpi=300,
        heatmap_tag="all"):                   # ← new
    """
    Draw a clade-annotated heat-map.

    Parameters
    ----------
    heatmap_tag : str
        Suffix inserted into the PNG filename:
        'all' (default)  →  expression_heatmap_all.png
        'dge'            →  expression_heatmap_dge.png
        or any custom tag you like.
    """
    # … [identical debug logic omitted for brevity — nothing changed] …

    out_png = os.path.join(output_dir, f"expression_heatmap_{heatmap_tag}.png")
    g.fig.savefig(out_png, dpi=dpi)
    # rest unchanged …
    print(f"✅  heat-map saved → {out_png}")
    return True



# ────────────────────────────────────────────────────────────────────────────
#  multi-clade DGE pipeline  (now calls two heat-maps)
# ────────────────────────────────────────────────────────────────────────────
def multi_clade_dge_analysis(sample_to_clade: dict,
                             folder_path: str,
                             output_dir: str):
    """
    1. Performs one-way ANOVA  &  Kruskal-Wallis across all clades.
    2. Saves summary CSV  +  per-gene histograms.
    3. Builds TWO heat-maps:
         • expression_heatmap_all.png   (unchanged)
         • expression_heatmap_dge.png   (only sig. DEGs)
    """
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from scipy.stats import f_oneway, kruskal
    from statsmodels.stats.multitest import multipletests

    os.makedirs(output_dir, exist_ok=True)
    expr_path = os.path.join(folder_path, "expression.csv")

    # ── load & filter samples ───────────────────────────────────────────────
    expr = pd.read_csv(expr_path, index_col=0)
    expr = expr.loc[expr.index.intersection(sample_to_clade)]
    expr["clade"] = expr.index.map(sample_to_clade)

    valid_clades = expr["clade"].value_counts()[lambda s: s >= 2].index
    expr = expr[expr["clade"].isin(valid_clades)]

    grouped     = expr.groupby("clade")
    clade_exprs = {c: g.drop(columns="clade") for c, g in grouped}

    # ── pre-filter genes (≥3 expressed samples in any clade) ───────────────
    candidates = []
    for gene in expr.columns.difference(["clade"]):
        if any((df[gene] > 0.1).sum() >= 3 for df in clade_exprs.values()):
            candidates.append(gene)

    # ── ANOVA & Kruskal-Wallis ──────────────────────────────────────────────
    res = []
    for gene in candidates:
        groups = [df[gene].values for df in clade_exprs.values()]
        try:
            _, p_anova = f_oneway(*groups)
        except ValueError:
            p_anova = 1.0
        _, p_kw = kruskal(*groups)

        means   = [g.mean() for g in groups]
        log2fc  = (np.log2((max(means)+1e-6)/(min(means)+1e-6))
                   if min(means) > 0 else np.nan)
        res.append({"gene": gene,
                    "anova_pval": p_anova,
                    "kw_pval":    p_kw,
                    "log2fc":     log2fc})

    dge = pd.DataFrame(res)
    dge["anova_q"] = multipletests(dge.anova_pval, method="fdr_bh")[1]
    dge["kw_q"]    = multipletests(dge.kw_pval,   method="fdr_bh")[1]
    dge["sig"]     = (dge.anova_q < 0.05) & (dge.kw_q < 0.05)

    sig_df = dge[dge.sig].copy()
    sig_df.to_csv(os.path.join(output_dir,
                               "significant_dge_summary.csv"), index=False)

    # ── per-gene histograms ────────────────────────────────────────────────
    plot_dir = os.path.join(output_dir, "pairwise_visualization")
    os.makedirs(plot_dir, exist_ok=True)
    for gene in sig_df.gene:
        plt.figure(figsize=(8, 4))
        for clade, df in clade_exprs.items():
            plt.hist(df[gene], bins=20, alpha=0.5,
                     label=f"Clade {clade}", edgecolor="black")
        plt.title(f"Expression Histogram : {gene}")
        plt.xlabel("Expression");  plt.ylabel("Frequency")
        plt.legend();  plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{gene}_histogram.png"), dpi=300)
        plt.close()

    # ── 1st heat-map : all / top-variance genes  ──────────────────────────
    create_expression_heatmap(sample_to_clade,
                              folder_path,
                              output_dir,
                              gene_list=None,          # default behaviour
                              heatmap_tag="all")

    # ── 2nd heat-map : only significant DEGs  ─────────────────────────────
    if not sig_df.empty:
        create_expression_heatmap(sample_to_clade,
                                  folder_path,
                                  output_dir,
                                  gene_list=sig_df.gene.tolist(),
                                  heatmap_tag="dge")
    else:
        print("⚠️  No significant DEGs – DGE-only heat-map skipped.")

    # ── summary ───────────────────────────────────────────────────────────
    print("\n✅  Multi-clade DGE complete.")
    print(f"🧬  Significant DEGs : {len(sig_df)}  →  "
          f"{os.path.join(output_dir, 'significant_dge_summary.csv')}")
    print(f"📊  Histograms saved →  {plot_dir}")
    print(f"🗺️   Heat-maps saved  →  "
          f"{os.path.join(output_dir, 'expression_heatmap_all.png')}  &  "
          f"{os.path.join(output_dir, 'expression_heatmap_dge.png')}")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, kruskal, zscore
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import LinearSegmentedColormap
import traceback

# ─── helper: safe context that turns off Figure.tight_layout() ────────────────
from contextlib import contextmanager
from contextlib import contextmanager
@contextmanager
def _no_tight_layout():
    import matplotlib.figure as _mf
    _orig = _mf.Figure.tight_layout
    _mf.Figure.tight_layout = lambda *a, **k: None
    try:
        yield
    finally:
        _mf.Figure.tight_layout = _orig


# ────────────────────────────────────────────────────────────────────────────
#  main heat-map builder  (adds `heatmap_tag` kwarg)
# ────────────────────────────────────────────────────────────────────────────
def create_expression_heatmap(
        sample_to_clade: dict,
        folder_path: str,
        output_dir: str,
        gene_list=None,
        figsize=(12, 14),
        dpi=300,
        heatmap_tag="all"): 
    """
    Build a clade-annotated expression heat-map.  Prints debug breadcrumbs.
    Returns True on success, False if any fatal error occurs.
    """

    try:
        # ── 0. I/O & matrix ------------------------------------------------------------------
        os.makedirs(output_dir, exist_ok=True)
        expr_path = os.path.join(folder_path, "expression.csv")
        expr      = pd.read_csv(expr_path, index_col=0)
        expr      = expr.loc[expr.index.intersection(sample_to_clade)]
        if expr.empty:
            print("❌  [DEBUG-I/O] no overlapping samples");  return False
        print(f"✔️  [DEBUG-I/O] expression matrix shape after filter: {expr.shape}")

        # ── 1. gene list ---------------------------------------------------------------------
        if gene_list is None:
            gene_list = expr.var().nlargest(min(100, expr.shape[1])).index.tolist()
            print(f"✔️  [DEBUG-GENE] using top {len(gene_list)} variable genes")
        else:
            print(f"✔️  [DEBUG-GENE] using provided gene list ({len(gene_list)})")
        gene_list = [g for g in gene_list if g in expr.columns]
        if not gene_list:
            print("❌  [DEBUG-GENE] none of the requested genes in matrix");  return False
        expr = expr[gene_list]

        # ── 2. clade metadata & colours ------------------------------------------------------
        sample_clades  = expr.index.to_series().map(sample_to_clade)
        unique_clades  = sorted(sample_clades.unique())
        if not unique_clades:
            print("❌  [DEBUG-CLADE] no clades found");  return False
        clade_palette  = sns.color_palette("husl", len(unique_clades))
        lut            = dict(zip(unique_clades, clade_palette))

        ordered_samples = [s for c in unique_clades
                           for s in sample_clades[sample_clades == c].index]
        sample_colors   = pd.Series(sample_clades.loc[ordered_samples]).map(lut)

        print("✔️  [DEBUG-CLADE] clades:", unique_clades)
        print("✔️  [DEBUG-CLADE] ordered_samples:", len(ordered_samples))
        print("✔️  [DEBUG-CLADE] sample_colors:", sample_colors.notna().sum())

        # ── 3. scale expression --------------------------------------------------------------
        expr_z = pd.DataFrame(
            zscore(expr.loc[ordered_samples].T, axis=1, nan_policy="omit"),
            index=expr.columns, columns=ordered_samples).fillna(0)
        print("✔️  [DEBUG-SCALE] Z-score matrix shape:", expr_z.shape)

        cmap = LinearSegmentedColormap.from_list(
               "bwr", ["#0000FF", "#FFFFFF", "#FF0000"])
        extra_w  = max(0, (len(unique_clades) - 5) * 1.5)
        fig_size = (figsize[0] + extra_w, figsize[1])

        # ── 4. clustermap inside “no tight_layout” context -----------------------------------
        with _no_tight_layout():
            g = sns.clustermap(expr_z,
                               cmap=cmap,
                               col_colors=sample_colors,
                               col_cluster=False, row_cluster=False,
                               xticklabels=False,
                               yticklabels=len(gene_list) <= 100,
                               cbar_kws={"label": "Z-score"},
                               figsize=fig_size,
                               dendrogram_ratio=(0, 0),
                               colors_ratio=0.02)

        # legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=lut[c])
                   for c in unique_clades]
        g.ax_col_dendrogram.legend(
            handles, [f"Clade {c}" for c in unique_clades],
            title="Clades",
            loc="center left", bbox_to_anchor=(1.05, .5), borderaxespad=0)

        g.fig.suptitle("Gene Expression Heatmap by Clade", y=1.02)
        out_png = os.path.join(output_dir, f"expression_heatmap_{heatmap_tag}.png")
        g.fig.savefig(out_png, dpi=dpi)
        # rest unchanged …
        print(f"✅  heat-map saved → {out_png}")
        return True

    # ── fallback: plain heatmap --------------------------------------------------------------
    except Exception as err:
        print("❌  [DEBUG-EXCEPT] clustermap failed:", err)
        print(traceback.format_exc())
        print("⚠️  [DEBUG-EXCEPT] falling back to simple heatmap")
        try:
            plt.figure(figsize=figsize)
            sns.heatmap(expr_z if 'expr_z' in locals() else expr.T,
                        cmap=cmap,
                        xticklabels=False,
                        yticklabels=len(gene_list) <= 100,
                        cbar_kws={"label": "Z-score"})
            for c in unique_clades:
                plt.plot([], [], color=lut[c], label=f"Clade {c}")
            plt.legend(title="Clades", loc="upper right",
                       bbox_to_anchor=(1.25, 1))
            plt.title("Gene Expression Heatmap by Clade (fallback)")
            out_png = os.path.join(output_dir,
                                   "expression_heatmap_fallback.png")
            plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
            plt.close()
            print(f"✅  fallback heat-map saved → {out_png}")
            return True
        except Exception as err2:
            print("❌  [DEBUG-FALLBACK] even fallback failed:", err2)
            print(traceback.format_exc())
            return False