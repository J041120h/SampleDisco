"""Generate a cluster(severity) x top-gene z-score heatmap for each RAISIN cell
type from existing results -- NO RAISIN re-run.

Group-mean expression is built from adata_rna_preprocessed (X = log1p): per cell
type, cells -> per-sample pseudobulk (mean over cells) -> per-group mean over
samples (grouped by reconciled_severity), matching RAISIN's sample-level design.
Top genes are the most significant across the pairwise comparisons (union by
FDR; falls back to top-by-FDR when none pass). Uses the same plotting helper the
package now calls in run_pairwise_tests (plot_cluster_gene_zscore).

Also moves each cell type's raisin_summary.{txt,csv} into its summary_plots/
subfolder so the summary text, csv and figure live together.
"""
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, "/users/hjiang/GenoDistance/code")
from sampledisco.sample_clustering.RAISIN_TEST import plot_cluster_gene_zscore

BASE = ("/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics/"
        "sample_embedding_tune-on-RNA/cluster_severity_deg")
RR = f"{BASE}/raisin_results"
RNA = ("/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics/"
       "preprocess/adata_rna_preprocessed.h5ad")
CT, GROUP, SAMPLE = "cell_type", "reconciled_severity", "sample"
FDR, TOP_N = 0.05, 50


def safe(name):
    return name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")


def top_genes_for(ct_dir):
    """Significant genes (FDR<thr, up+down) across comparisons, ranked by FDR;
    fall back to top-by-FDR when none are significant."""
    pooled = []
    for f in glob.glob(os.path.join(ct_dir, "*", "raisin_results.csv")):
        pooled.append(pd.read_csv(f, index_col=0))
    if not pooled:
        return []
    pooled = pd.concat(pooled)
    sig = pooled[pooled["FDR"] < FDR].sort_values("FDR")
    genes = list(dict.fromkeys(sig.index.astype(str).tolist()))
    if not genes:
        genes = list(dict.fromkeys(pooled.sort_values("FDR").index.astype(str).tolist()))
    return genes[:TOP_N]


print(f"[load] {RNA}", flush=True)
adata = sc.read_h5ad(RNA)
adata.obs[CT] = adata.obs[CT].astype(str)

for ct in sorted(adata.obs[CT].unique()):
    ct_dir = os.path.join(RR, safe(ct))
    if not os.path.isdir(ct_dir):
        print(f"  [skip] no results dir for {ct}", flush=True)
        continue
    sp = os.path.join(ct_dir, "summary_plots")
    os.makedirs(sp, exist_ok=True)

    genes = [g for g in top_genes_for(ct_dir) if g in adata.var_names]
    if not genes:
        print(f"  [skip] {ct}: no genes", flush=True)
        continue

    sub = adata[adata.obs[CT] == ct, genes]
    X = sub.to_df()
    X[SAMPLE] = sub.obs[SAMPLE].astype(str).values
    samp_mean = X.groupby(SAMPLE)[genes].mean()                       # sample x gene
    s2g = (sub.obs[[SAMPLE, GROUP]].astype(str).drop_duplicates()
           .set_index(SAMPLE)[GROUP])
    samp_mean[GROUP] = s2g.reindex(samp_mean.index).values
    grp_gene = samp_mean.groupby(GROUP)[genes].mean().T               # gene x group

    plot_cluster_gene_zscore(
        grp_gene,
        os.path.join(sp, "cluster_gene_zscore.png"),
        title=f"{ct}: cluster (severity) x gene z-score  (n={len(genes)} genes)",
    )

    # Move the per-cell-type summary text/csv into the subfolder.
    for fn in ("raisin_summary.txt", "raisin_summary.csv"):
        src = os.path.join(ct_dir, fn)
        if os.path.exists(src):
            shutil.move(src, os.path.join(sp, fn))
    print(f"  [{ct}] done -> {sp}", flush=True)

print("ZSCORE_HEATMAPS_DONE", flush=True)
