"""
Balanced 69-vs-69 extreme-contrast DGE on round6 PC8.

Instead of KMeans (which gave 69 vs 247 imbalanced + 28-sample impurity),
take the 69 lowest-PC8 samples (Female-enriched tail) and the 69 highest-PC8
samples (Male tail). The middle 178 "ambiguous" samples are excluded.

This addresses the three reasons the prior run failed FDR<0.05:
  1. Cluster impurity (extremes are purer than the KMeans boundary).
  2. Class imbalance (now 69 vs 69 instead of 69 vs 247).
  3. (Multiple-testing burden unchanged — still 16,679 genes.)

Output dir: round7_kmeans_sex/pc8_balanced_extremes/
"""

import json
import os
import sys
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import anndata as ad
import numpy as np
import pandas as pd

from sample_clustering.RAISIN import raisinfit
from sample_clustering.RAISIN_TEST import run_pairwise_tests

EMB_CSV = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round6_multicov_batch_age/sample_embedding/sample_embedding/sample_embedding.csv"
CELL_H5AD = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/preprocess/adata_preprocessed.h5ad"
OUT_DIR = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round7_kmeans_sex/pc8_balanced_extremes"
SAMPLE_COL = "Tube_id"
N_PER_GROUP = 69  # matches the small-cluster size of the prior KMeans

SEX_PANEL = {
    "Y_linked":       ["RPS4Y1","RPS4Y2","DDX3Y","EIF1AY","KDM5D","UTY","USP9Y","NLGN4Y","TMSB4Y","ZFY","PRKY","BCORP1"],
    "X_inactivation": ["XIST","TSIX"],
    "X_escape":       ["KDM6A","DDX3X","EIF1AX","KDM5C"],
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Build balanced extreme labels ----
    emb = pd.read_csv(EMB_CSV, index_col=0)
    emb.index = emb.index.astype(str)
    emb_sorted = emb.sort_values("PC8")
    low = emb_sorted.head(N_PER_GROUP)  # most negative PC8 (Female-rich tail)
    high = emb_sorted.tail(N_PER_GROUP)  # most positive PC8 (Male tail)
    print(f"low PC8 (cluster_low):  n={len(low)}, range [{low['PC8'].min():.3f}, {low['PC8'].max():.3f}]")
    print(f"high PC8 (cluster_high): n={len(high)}, range [{high['PC8'].min():.3f}, {high['PC8'].max():.3f}]")
    print(f"middle (excluded):       n={len(emb)-2*N_PER_GROUP}")

    # Use simple "0"/"1" labels — RAISIN's run_pairwise_tests does substring
    # matching against design-matrix column names and fails on richer labels.
    sample_to_clade = {}
    for s in low.index: sample_to_clade[s] = "0"   # low PC8 = Female-rich tail
    for s in high.index: sample_to_clade[s] = "1"  # high PC8 = Male tail

    # ---- Sanity-check vs true Sex (purity per group) ----
    a_full = ad.read_h5ad(CELL_H5AD, backed="r")
    meta = a_full.obs[[SAMPLE_COL, "Sex"]].drop_duplicates(SAMPLE_COL).set_index(SAMPLE_COL)
    meta.index = meta.index.astype(str)
    low_sex = meta.loc[meta.index.intersection(low.index), "Sex"].value_counts().to_dict()
    high_sex = meta.loc[meta.index.intersection(high.index), "Sex"].value_counts().to_dict()
    print(f"low-PC8 group Sex composition:  {low_sex}")
    print(f"high-PC8 group Sex composition: {high_sex}")

    # save labels CSV
    labels_df = pd.DataFrame({
        "sample": list(sample_to_clade.keys()),
        "cluster": list(sample_to_clade.values()),
        "PC8": [emb.loc[s, "PC8"] for s in sample_to_clade],
        "true_sex": [meta.loc[s, "Sex"] if s in meta.index else "?" for s in sample_to_clade],
    })
    labels_path = os.path.join(OUT_DIR, "balanced_extremes_labels.csv")
    labels_df.to_csv(labels_path, index=False)
    print(f"wrote {labels_path}")

    # ---- Load full cell adata (heavy) ----
    print("\nLoading full cell adata (~11 GB) ...", flush=True)
    a = ad.read_h5ad(CELL_H5AD)
    print(f"  shape: {a.shape}")

    # ---- RAISIN fit (skip ComBat; sample_to_clade contains only the 138 selected samples) ----
    print("\nRunning raisinfit (batch_col=None, 138 samples / 178 dropped) ...", flush=True)
    fit = raisinfit(
        adata=a,
        sample_col=SAMPLE_COL,
        testtype="unpaired",
        batch_col=None,
        sample_to_clade=sample_to_clade,
        intercept=True,
        n_jobs=-1,
        verbose=True,
    )

    raisin_out = os.path.join(OUT_DIR, "raisin_results")
    print(f"\nRunning pairwise tests; outputs -> {raisin_out}", flush=True)
    run_pairwise_tests(
        fit=fit,
        output_dir=raisin_out,
        fdrmethod="fdr_bh",
        fdr_threshold=0.05,
        verbose=True,
    )

    # ---- Sex-panel summary ----
    print("\n=== Sex panel cross-check ===")
    out_csv = os.path.join(raisin_out, "0_vs_1", "raisin_results.csv")
    if not os.path.exists(out_csv):
        # fall back to first found
        for d in (os.listdir(raisin_out) if os.path.isdir(raisin_out) else []):
            sub = os.path.join(raisin_out, d, "raisin_results.csv")
            if os.path.exists(sub):
                out_csv = sub
                break
    print(f"reading {out_csv}")
    df = pd.read_csv(out_csv, index_col=0)
    df.index.name = "gene"
    print(f"shape: {df.shape}; min FDR: {df.FDR.min():.3f}; n FDR<0.05: {(df.FDR<0.05).sum()}; n raw p<0.05: {(df.pvalue<0.05).sum()}")

    panel = [g for L in SEX_PANEL.values() for g in L]
    hits = df.loc[df.index.intersection(panel)].sort_values("pvalue")
    print("\n=== Sex-panel hits (ranked by raw p) ===")
    print(hits.to_string())

    # save summary JSON
    summary = {
        "n_per_group": N_PER_GROUP,
        "low_sex_composition": low_sex,
        "high_sex_composition": high_sex,
        "n_genes_tested": int(df.shape[0]),
        "n_FDR_lt_05": int((df.FDR < 0.05).sum()),
        "n_raw_p_lt_05": int((df.pvalue < 0.05).sum()),
        "min_FDR": float(df.FDR.min()),
        "min_pvalue": float(df.pvalue.min()),
        "sex_panel_genes_in_universe": int(len(hits)),
        "sex_panel_FDR_lt_05": int((hits.FDR < 0.05).sum()),
        "sex_panel_raw_p_lt_05": int((hits.pvalue < 0.05).sum()),
        "sex_panel_top": hits.head(10).reset_index().to_dict(orient="records"),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'summary.json')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
