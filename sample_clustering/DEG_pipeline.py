# Part 4: Differential Gene Expression Analysis
# Input file required:
# Expression matrix (.csv)
# sample distance matrix (.csv)
# tree file (.nex)
# Key parameters to adjust: input file name, resolution factor, and selected clade to compare
import pandas as pd
from Bio import Phylo
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import numpy as np
from Bio import Phylo

## TREE-PREPROCESSING
## used to define branch lengths so that resolution-based cutting works
## this step is SKIPPED in expression-based tree, as branch lengths are well defined
## however, this step is performed in proportion-based tree, to at least give nominal branch lengths for analysis
# load tree and distance matrix
tree = Phylo.read("Consensus_Majority_Rule_Tree_cds.nex", "nexus")
dist = pd.read_csv("distance_matrix_proportion.csv", index_col=0)

#define nominal branch length for every non‑root branch
for cl in tree.find_clades(order="preorder"):
    if cl is tree.root:
        cl.branch_length = 0.0
    elif cl.branch_length in (None, 0):
        cl.branch_length = 0.01

# for every binary internal node, overwrite with
# half the mean pairwise distance between its two sub‑clades
for cl in tree.get_nonterminals(order="postorder"):
    if len(cl.clades) != 2:
        continue
    L, R = cl.clades
    leaves_L = [t.name for t in L.get_terminals()]
    leaves_R = [t.name for t in R.get_terminals()]

    vals = [
        dist.loc[a, b]
        for a in leaves_L for b in leaves_R
        if a in dist.index and b in dist.columns
    ]
    if vals:
        cl.branch_length = np.mean(vals) / 2.0

# write to a .nwk file
Phylo.write(tree, "Consensus_Majority_Rule_Tree_cds_fixed.nwk", "newick")

## ACTUAL PIPELINE
# key variables to be defined - could directly adjust here
resolution = 0.5
clade1 = 2 # clade number used in DGE
clade2 = 3

## RESOLUTION-FACTOR BASED TREE CUTTING
# load expression and distance matrix
expr = pd.read_csv('expression.csv', index_col=0)
dist = pd.read_csv('distance_matrix_proportion.csv', index_col=0)

# read tree in .nwk format
# processed using helper functions if required
# tree = Phylo.read('Consensus_Expression.nex', 'nexus') # uncomment this for expression tree if needed
tree = Phylo.read('Consensus_Majority_Rule_Tree_cds_fixed.nwk', 'newick')

# gets the maximum depth of the tree
def get_max_depth(clade, current=0):
    depths = [get_max_depth(c, current + c.branch_length) for c in clade.clades]
    return max(depths) if depths else current

# cut tree at threshold = resolution * max_height
# larger resolution = smaller clades
# then assign samples to clades
def assign_clades_by_resolution(tree, resolution):
    max_height = get_max_depth(tree.root)
    threshold  = resolution * max_height

    clade_id = 1
    sample_to_clade = {}

    def collect_leaves(clade, cid):
        for t in clade.get_terminals():
            sample_to_clade[t.name] = cid

    def traverse(node, depth=0):
        nonlocal clade_id
        for child in node.clades:
            child_depth = depth + child.branch_length
            if depth < threshold <= child_depth:  # if edge crosses the cut
                collect_leaves(child, clade_id)
                clade_id += 1
            else:
                traverse(child, child_depth)

    traverse(tree.root)

    # any leaf still unlabelled gets its own clade
    for leaf in tree.get_terminals():
        if leaf.name not in sample_to_clade:
            sample_to_clade[leaf.name] = clade_id
            clade_id += 1
    return sample_to_clade

# set resolution factor (between 0.0 and 1.0), cut tree
# 0.0 = 1 clade, 1.0 = each sample its own clade
sample_to_clade = assign_clades_by_resolution(tree, resolution)

# assign clades to expression data
expr = expr.loc[expr.index.isin(sample_to_clade)]
expr['clade'] = expr.index.map(sample_to_clade)  # add new col to expression matrix
print(expr['clade'].value_counts()) # print to visually inspect each clade


## CLADE SELECTION and DATA PROCESSING
# select clades to compare, to be adjusted
group1 = expr[expr['clade'] == clade1].drop(columns='clade')
group2 = expr[expr['clade'] == clade2].drop(columns='clade')

# filter genes by minimum expression
min_expr_threshold = 0.1
min_samples_expressed = 3  # at least expressed in 3 samples in either group

filtered_genes = [
    g for g in group1.columns
    if ((group1[g] > min_expr_threshold).sum() >= min_samples_expressed or
        (group2[g] > min_expr_threshold).sum() >= min_samples_expressed)
]


# DIFFERENTIAL EXPRESSION ANALYSIS
# t-test, Wilcoxon rank-sum test, log2FC
results = []
for gene in filtered_genes:
    v1, v2 = group1[gene], group2[gene]

    _, t_pval = ttest_ind(v1, v2, equal_var=False)
    _, w_pval = mannwhitneyu(v1, v2, alternative='two-sided')

    m1, m2 = np.mean(v1), np.mean(v2)
    log2fc = np.log2((m1 + 1e-6) / (m2 + 1e-6)) if (m1 > 0 and m2 > 0) else np.nan # avoid division by 0
    results.append({'gene': gene, 't_pval': t_pval, 'w_pval': w_pval, 'log2fc': log2fc})

dge_df = pd.DataFrame(results)

# FDR multiple testing correction to control false positive rate
dge_df['t_qval'] = multipletests(dge_df['t_pval'], method='fdr_bh')[1]
dge_df['w_qval'] = multipletests(dge_df['w_pval'], method='fdr_bh')[1]

# flag significant genes based on adjusted q value (threshold=0.05)
dge_df['sig_t']   = dge_df['t_qval'] < 0.05
dge_df['sig_w']   = dge_df['w_qval'] < 0.05
dge_df['sig_both'] = dge_df['sig_t'] & dge_df['sig_w']

# output results to .csv
dge_df = dge_df.sort_values(['sig_both', 't_qval', 'w_qval'],
                            ascending=[False, True, True])
dge_df.to_csv('dge_results_multimethod.csv', index=False)

# print top DEGs that pass both tests
# visually inspect log2FC to be sure
print("\nTop genes significant in both tests:")
print(dge_df[dge_df['sig_both']].head(20)[['gene', 'log2fc', 't_qval', 'w_qval']])

# show what samples in each clade for visual inspection
for cid, g in expr.groupby('clade'):
    print(f"\nClade {cid} ({len(g)} samples):")
    print(g.index.tolist())


## VISUALIZATION OF DEG - VOLCANO PLOTS
from adjustText import adjust_text # used for text labels
import matplotlib.pyplot as plt
import seaborn as sns

# t-test result visualization
dge_df['-log10_t_qval'] = -np.log10(dge_df['t_qval'])
dge_df['color_t'] = 'gray'
dge_df.loc[dge_df['sig_t'], 'color_t'] = 'red'

plt.figure(figsize=(10, 6))
sns.scatterplot(data=dge_df, x='log2fc', y='-log10_t_qval', hue='color_t',
                palette={'gray': 'gray', 'red': 'red'}, legend=False)

top_t = dge_df[dge_df['sig_t']].nsmallest(10, 't_qval') # pick 10 smallest
texts = [
    plt.text(row['log2fc'], row['-log10_t_qval'], row['gene'],
             fontsize=8, ha='right' if row['log2fc'] < 0 else 'left')
    for _, row in top_t.iterrows()
]
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

plt.axhline(-np.log10(0.05), color='blue', linestyle='--')
plt.axvline(0, color='black', linestyle='-')
plt.xlabel('log2 Fold Change')
plt.ylabel('-log10 FDR-adjusted p-value (t-test)')
plt.title('Volcano Plot (t-test)')
plt.tight_layout()
plt.savefig('volcano_ttest_labeled.png', dpi=300)
plt.show()

# wilcoxon/u-test result visualization
dge_df['-log10_w_qval'] = -np.log10(dge_df['w_qval'])
dge_df['color_w'] = 'gray'
dge_df.loc[dge_df['sig_w'], 'color_w'] = 'red'

plt.figure(figsize=(10, 6))
sns.scatterplot(data=dge_df, x='log2fc', y='-log10_w_qval', hue='color_w',
                palette={'gray': 'gray', 'red': 'red'}, legend=False)

top_w = dge_df[dge_df['sig_w']].nsmallest(10, 'w_qval')
texts = [
    plt.text(row['log2fc'], row['-log10_w_qval'], row['gene'],
             fontsize=8, ha='right' if row['log2fc'] < 0 else 'left')
    for _, row in top_w.iterrows()
]
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

plt.axhline(-np.log10(0.05), color='blue', linestyle='--')
plt.axvline(0, color='black', linestyle='-')
plt.xlabel('log2 Fold Change')
plt.ylabel('-log10 FDR-adjusted p-value (Wilcoxon)')
plt.title('Volcano Plot (Wilcoxon Rank-Sum Test)')
plt.tight_layout()
plt.savefig('volcano_wilcoxon_labeled.png', dpi=300)
plt.show()

# VISUALIZATION OF CLADE ON CONSENSUS TREE
from Bio import Phylo
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_hex
import seaborn as sns

# Assign colors to clades
unique_clades = sorted(set(sample_to_clade.values()))
palette = sns.color_palette("hls", len(unique_clades))
clade_colors = {cid: to_hex(color) for cid, color in zip(unique_clades, palette)}

# Create a mapping from leaf name to color
label_colors = {
    leaf.name: clade_colors[sample_to_clade[leaf.name]]
    for leaf in tree.get_terminals() if leaf.name in sample_to_clade
}

# Build custom legend handles
legend_handles = [
    Patch(color=clade_colors[cid], label=f"Clade {cid}")
    for cid in unique_clades
]

# Draw tree with colored tip labels
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1)
Phylo.draw(tree, label_colors=label_colors, do_show=False, axes=ax)

# Add legend to the side
plt.legend(
    handles=legend_handles,
    loc='center left',
    bbox_to_anchor=(1.05, 0.5),
    title="Clade Assignment"
)

plt.title(f"Phylogenetic Tree with Clade Coloring (Resolution = {resolution})")
plt.tight_layout()
plt.savefig("tree_colored_by_clade_with_legend.png", dpi=300, bbox_inches='tight')
plt.show()