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
    plot_dir = os.path.join(output_dir, "pairwise_visualization")
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
    
    # Create expression heatmap for significant genes
    if len(sig_df) > 0:
        create_expression_heatmap(sample_to_clade, folder_path, output_dir, gene_list=sig_df["gene"].tolist())
    else:
        print("⚠️ No significant DEGs found")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, kruskal, zscore
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import LinearSegmentedColormap
import traceback

def create_expression_heatmap(sample_to_clade: dict, folder_path: str, output_dir: str, 
                             gene_list=None, figsize=(12, 14), dpi=300):
    """
    Create an improved gene expression heatmap with samples grouped by clade.
    
    Parameters:
        sample_to_clade (dict): Mapping from sample name to clade ID
        folder_path (str): Path to folder containing expression.csv
        output_dir (str): Directory to save the heatmap
        gene_list (list, optional): List of genes to include in the heatmap
                                   If None, uses top variable genes
        figsize (tuple): Figure size (width, height)
        dpi (int): Resolution for saved figure
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load expression data
        expr_path = os.path.join(folder_path, "expression.csv")
        if not os.path.exists(expr_path):
            print(f"❌ Error: Expression file not found at {expr_path}")
            return False
            
        expr = pd.read_csv(expr_path, index_col=0)
        
        # Filter samples to those in sample_to_clade
        common_samples = expr.index.intersection(sample_to_clade.keys())
        if len(common_samples) == 0:
            print("❌ Error: No matching samples found between expression data and clade mapping")
            return False
            
        expr = expr.loc[common_samples]
        print(f"📊 Using {len(expr)} samples for heatmap")
        
        # Verify we have data after filtering
        if expr.shape[0] == 0 or expr.shape[1] == 0:
            print("❌ Error: Expression matrix is empty after filtering")
            return False
        
        # If no gene list provided, use top variable genes
        if gene_list is None:
            gene_var = expr.var(axis=0)
            if gene_var.empty:
                print("❌ Error: Unable to calculate gene variance - expression data may be invalid")
                return False
                
            # Take top 100 or fewer if not enough genes
            n_genes = min(100, len(gene_var))
            gene_list = gene_var.nlargest(n_genes).index.tolist()
            print(f"📊 Using top {len(gene_list)} variable genes for heatmap")
        
        # Verify gene list
        if not gene_list or len(gene_list) == 0:
            print("❌ Error: Empty gene list for heatmap")
            return False
            
        # Filter to selected genes and check if any remain
        available_genes = [g for g in gene_list if g in expr.columns]
        if not available_genes:
            print("❌ Error: None of the specified genes found in expression data")
            return False
            
        # Use only available genes
        if len(available_genes) != len(gene_list):
            print(f"⚠️ Warning: Only {len(available_genes)}/{len(gene_list)} genes found in expression data")
            gene_list = available_genes
            
        expr = expr[gene_list]
        
        # Add clade information
        sample_clades = pd.DataFrame({'clade': [sample_to_clade[s] for s in expr.index]}, index=expr.index)
        
        # Get unique clades for colormap
        unique_clades = sorted(set(sample_to_clade[s] for s in expr.index))
        
        # Verify we have clades
        if not unique_clades:
            print("❌ Error: No clades found for samples")
            return False
            
        num_clades = len(unique_clades)
        print(f"📊 Found {num_clades} unique clades")
        
        # Create a mapping from clade to color index
        clade_to_color = {clade: i for i, clade in enumerate(unique_clades)}
        
        # Create a color palette for the clades
        clade_palette = sns.color_palette("husl", num_clades)
        
        # Create a custom colormap for the heatmap (blue to white to red)
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#0000FF", "#FFFFFF", "#FF0000"])
        
        # Transpose the dataframe for clustering (genes as rows, samples as columns)
        expr_transpose = expr.T
        
        # Z-score normalize each gene across samples
        # Handle potential NaN values or constant genes
        try:
            expr_zscore = pd.DataFrame(
                data=zscore(expr_transpose.values, axis=1, nan_policy='omit'),
                index=expr_transpose.index,
                columns=expr_transpose.columns
            )
            
            # Replace any remaining NaNs with 0
            expr_zscore.fillna(0, inplace=True)
        except Exception as e:
            print(f"❌ Error during Z-score normalization: {str(e)}")
            # Fall back to simple normalization if zscore fails
            print("⚠️ Falling back to simple min-max normalization")
            expr_norm = expr_transpose.copy()
            for idx in expr_norm.index:
                row_min = expr_norm.loc[idx].min()
                row_max = expr_norm.loc[idx].max()
                if row_max > row_min:  # Avoid division by zero
                    expr_norm.loc[idx] = (expr_norm.loc[idx] - row_min) / (row_max - row_min) * 2 - 1
            expr_zscore = expr_norm
        
        # Sort samples by clade
        lut = dict(zip(unique_clades, clade_palette))
        
        # Organize samples by clade
        ordered_samples = []
        for clade in unique_clades:
            clade_samples = sample_clades[sample_clades['clade'] == clade].index.tolist()
            ordered_samples.extend(clade_samples)
        
        if not ordered_samples:
            print("❌ Error: No samples left after ordering by clade")
            return False
        
        # Update the z-scored data with ordered samples
        expr_zscore = expr_zscore[ordered_samples]
        sample_colors = pd.Series(sample_clades.loc[ordered_samples, 'clade']).map(lut)
        
        extra_width = max(0, (num_clades - 5) * 1.5)
        adjusted_figsize = (figsize[0] + extra_width, figsize[1])

        # Handle potential empty data gracefully
        if expr_zscore.empty:
            print("❌ Error: Expression matrix is empty after processing")
            return False
            
        # Create figure first then the clustermap
        plt.figure(figsize=(1, 1))  # Create a dummy figure to avoid potential bbox issues
        plt.close()

        # Create heatmap with explicit figure initialization
        plt.figure(figsize=adjusted_figsize)
        
        try:
            g = sns.clustermap(
                expr_zscore,
                cmap=cmap,
                col_colors=sample_colors,
                col_cluster=False,     # Don't cluster samples
                row_cluster=False,     # Don't cluster genes
                z_score=None,          # Already z-scored
                xticklabels=False,
                yticklabels=True if len(gene_list) <= 100 else False,
                cbar_kws={"label": "Z-score"},
                figsize=adjusted_figsize,
                dendrogram_ratio=(0, 0),  # Remove dendrogram spaces
                colors_ratio=0.02
            )
            
            # Check if ax_col_dendrogram exists before trying to add legend
            if hasattr(g, 'ax_col_dendrogram') and g.ax_col_dendrogram is not None:
                # Create legend handles
                handles = [plt.Rectangle((0,0), 1, 1, color=lut[clade]) for clade in unique_clades]
                
                # Add legend if handles exist
                if handles:
                    g.ax_col_dendrogram.legend(
                        handles,
                        [f"Clade {clade}" for clade in unique_clades],
                        title="Clades",
                        loc="center left",
                        bbox_to_anchor=(1.05, 0.5),
                        borderaxespad=0
                    )
            else:
                # Alternative legend placement if ax_col_dendrogram is not available
                plt.legend(
                    [plt.Rectangle((0,0), 1, 1, color=lut[clade]) for clade in unique_clades],
                    [f"Clade {clade}" for clade in unique_clades],
                    title="Clades",
                    loc="upper right",
                    bbox_to_anchor=(1.2, 1)
                )

            plt.suptitle("Gene Expression Heatmap by Clade", fontsize=16, y=1.02)
            
            heatmap_path = os.path.join(output_dir, "expression_heatmap.png")
            plt.savefig(heatmap_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Heatmap saved to: {heatmap_path}")
            
            # Optional: create a labeled version if few samples
            if len(ordered_samples) <= 50:
                plt.figure(figsize=adjusted_figsize)  # Create new figure for labeled version
                
                g = sns.clustermap(
                    expr_zscore,
                    cmap=cmap,
                    col_colors=sample_colors,
                    col_cluster=False,
                    row_cluster=False,   # Keep consistent
                    z_score=None,
                    xticklabels=True,
                    yticklabels=True if len(gene_list) <= 100 else False,
                    cbar_kws={"label": "Z-score"},
                    figsize=adjusted_figsize,
                    dendrogram_ratio=(0, 0),
                    colors_ratio=0.02
                )
                
                # Same legend handling as before
                if hasattr(g, 'ax_col_dendrogram') and g.ax_col_dendrogram is not None:
                    handles = [plt.Rectangle((0,0), 1, 1, color=lut[clade]) for clade in unique_clades]
                    if handles:
                        g.ax_col_dendrogram.legend(
                            handles,
                            [f"Clade {clade}" for clade in unique_clades],
                            title="Clades",
                            loc="center left",
                            bbox_to_anchor=(1.05, 0.5),
                            borderaxespad=0
                        )
                else:
                    plt.legend(
                        [plt.Rectangle((0,0), 1, 1, color=lut[clade]) for clade in unique_clades],
                        [f"Clade {clade}" for clade in unique_clades],
                        title="Clades",
                        loc="upper right",
                        bbox_to_anchor=(1.2, 1)
                    )
                    
                plt.suptitle("Gene Expression Heatmap by Clade (with sample labels)", fontsize=16, y=1.02)
                labeled_heatmap_path = os.path.join(output_dir, "expression_heatmap_with_labels.png")
                plt.savefig(labeled_heatmap_path, dpi=dpi, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Labeled heatmap saved to: {labeled_heatmap_path}")
                
            return True
            
        except Exception as sns_error:
            print(f"❌ Error during clustermap creation: {str(sns_error)}")
            
            # Fall back to basic heatmap if clustermap fails
            print("⚠️ Falling back to basic heatmap")
            plt.figure(figsize=adjusted_figsize)
            
            # Create a simple heatmap instead
            ax = sns.heatmap(
                expr_zscore,
                cmap=cmap,
                xticklabels=False,
                yticklabels=True if len(gene_list) <= 100 else False,
                cbar_kws={"label": "Z-score"}
            )
            
            # Add color bar on top for clade indication
            clade_colorbar = pd.DataFrame(
                [sample_clades.loc[s, 'clade'] for s in ordered_samples],
                index=ordered_samples,
                columns=['clade']
            )
            
            # Manual legend for clades
            for i, clade in enumerate(unique_clades):
                plt.plot([], [], color=lut[clade], label=f"Clade {clade}")
            
            plt.legend(title="Clades", loc="upper right", bbox_to_anchor=(1.2, 1))
            plt.title("Gene Expression Heatmap by Clade (fallback version)")
            
            fallback_path = os.path.join(output_dir, "expression_heatmap_fallback.png")
            plt.savefig(fallback_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Fallback heatmap saved to: {fallback_path}")
            return True
            
    except Exception as e:
        print(f"❌ Error in heatmap generation: {str(e)}")
        print(traceback.format_exc())
        return False