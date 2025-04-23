import os
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

def multi_clade_dge_analysis(sample_to_clade: dict, folder_path: str, output_dir: str):
    """
    Perform DGE across all clades at once using ANOVA and Kruskal-Wallis H-test.
    Save summary CSV and histogram plots for significant genes.

    Parameters:
        sample_to_clade (dict): Mapping from sample name to clade ID.
        folder_path (str): Path to folder containing expression.csv.
        output_dir (str): Directory to save results and plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    expr_path = os.path.join(folder_path, "expression.csv")

    # Load expression data
    expr = pd.read_csv(expr_path, index_col=0)
    expr = expr.loc[expr.index.intersection(sample_to_clade)]
    expr["clade"] = expr.index.map(sample_to_clade)

    # Keep only clades with ≥2 samples
    clade_counts = expr["clade"].value_counts()
    valid_clades = clade_counts[clade_counts >= 2].index.tolist()
    expr = expr[expr["clade"].isin(valid_clades)]

    # Group expression by clade
    grouped = expr.groupby("clade")
    clade_exprs = {clade: group.drop(columns="clade") for clade, group in grouped}

    # Filter genes: keep genes expressed in ≥3 samples in at least one clade
    genes = expr.drop(columns="clade").columns
    filtered_genes = []
    for gene in genes:
        if any((clade_df[gene] > 0.1).sum() >= 3 for clade_df in clade_exprs.values()):
            filtered_genes.append(gene)

    # Perform ANOVA and Kruskal-Wallis
    results = []
    for gene in filtered_genes:
        groups = [df[gene].values for df in clade_exprs.values()]
        try:
            _, p_anova = f_oneway(*groups)
        except ValueError:
            p_anova = 1.0
        _, p_kw = kruskal(*groups)

        group_means = [np.mean(g) for g in groups]
        log2fc = np.log2((max(group_means) + 1e-6) / (min(group_means) + 1e-6)) if min(group_means) > 0 else np.nan

        results.append({
            "gene": gene,
            "anova_pval": p_anova,
            "kw_pval": p_kw,
            "log2fc_max_min": log2fc
        })

    dge_df = pd.DataFrame(results)
    dge_df["anova_qval"] = multipletests(dge_df["anova_pval"], method="fdr_bh")[1]
    dge_df["kw_qval"] = multipletests(dge_df["kw_pval"], method="fdr_bh")[1]
    dge_df["sig_both"] = (dge_df["anova_qval"] < 0.05) & (dge_df["kw_qval"] < 0.05)

    # Save only significant results
    sig_df = dge_df[dge_df["sig_both"]].copy()
    sig_df.to_csv(os.path.join(output_dir, "significant_dge_summary.csv"), index=False)

    # Histogram plots
    plot_dir = os.path.join(output_dir, "histogram_plots")
    os.makedirs(plot_dir, exist_ok=True)

    for gene in sig_df["gene"]:
        plt.figure(figsize=(8, 4))
        for clade, clade_df in clade_exprs.items():
            plt.hist(clade_df[gene], bins=20, alpha=0.5, label=f"Clade {clade}", edgecolor='black')
        plt.title(f"Expression Histogram for {gene}")
        plt.xlabel("Expression")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{gene}_histogram.png"), dpi=300)
        plt.close()

    print(f"\n✅ Multi-clade DGE complete.")
    print(f"🧬 Significant DEGs: {len(sig_df)} saved to: {os.path.join(output_dir, 'significant_dge_summary.csv')}")
    print(f"📊 Histogram plots saved to: {plot_dir}")
