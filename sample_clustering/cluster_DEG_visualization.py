import os
import itertools
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

def cluster_dge_visualization(sample_to_clade: dict, folder_path: str, output_dir: str):
    """
    Perform pairwise differential expression analysis across all valid clade pairs.
    
    Parameters:
        sample_to_clade (dict): Mapping from sample name to clade ID.
        folder_path (str): Path to folder containing expression.csv. Outputs will go to folder_path/cluster_DEG.
    """
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
