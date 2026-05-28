"""
DGE between PC8-KMeans clusters using SampleDisco's raisinfit + run_pairwise_tests.

Skips ComBat (batch_col=None) since multi-cov Harmony already corrected at
sample level in round6 and ComBat's dense .toarray() blew through 100 GB.
Sex genes have huge fold-changes (10-100x), so batch correction at cell
level shouldn't be required to find them.

Outputs (under <result_dir>/dge/):
  raisin_results/...        pairwise test outputs
  sex_gene_overlap.json     hits among canonical sex panel
  sex_gene_summary.csv      per-gene FDR / fold-change for the panel
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

CELL_H5AD = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/preprocess/adata_preprocessed.h5ad"
PC8_CLUSTERS = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round7_kmeans_sex/pc8_only/kmeans_pc8_clusters.csv"
OUT_DIR = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round7_kmeans_sex/pc8_only/dge"
SAMPLE_COL = "Tube_id"

# Canonical sex-linked panel
SEX_PANEL = {
    "Y_linked": ["RPS4Y1","RPS4Y2","DDX3Y","EIF1AY","KDM5D","UTY","USP9Y","NLGN4Y","TMSB4Y","ZFY","PRKY","BCORP1"],
    "X_inactivation": ["XIST","TSIX"],
    "X_escape": ["KDM6A","DDX3X","EIF1AX","KDM5C"],
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Load cluster labels ----
    labels = pd.read_csv(PC8_CLUSTERS)
    labels["sample"] = labels["sample"].astype(str)
    sample_to_clade = dict(zip(labels["sample"], labels["cluster_pc8"].astype(str)))
    print(f"PC8 cluster labels: {len(sample_to_clade)} samples")
    print(f"  cluster 0 (Female-rich, majority Female): n={sum(v=='0' for v in sample_to_clade.values())}")
    print(f"  cluster 1 (Male-rich, majority Male):     n={sum(v=='1' for v in sample_to_clade.values())}")

    # ---- Load full cell adata ----
    print("\nLoading full cell adata (~11 GB) ...", flush=True)
    a = ad.read_h5ad(CELL_H5AD)
    print(f"  shape: {a.shape}; obs cols: {len(a.obs.columns)}")

    # ---- raisinfit (skip ComBat) ----
    print("\nRunning raisinfit (batch_col=None to skip ComBat) ...", flush=True)
    fit = raisinfit(
        adata=a,
        sample_col=SAMPLE_COL,
        testtype="unpaired",
        batch_col=None,                # SKIP ComBat - main change from default
        sample_to_clade=sample_to_clade,
        intercept=True,
        n_jobs=-1,
        verbose=True,
    )

    # ---- pairwise tests ----
    raisin_out = os.path.join(OUT_DIR, "raisin_results")
    print(f"\nRunning pairwise tests; outputs -> {raisin_out}", flush=True)
    run_pairwise_tests(
        fit=fit,
        output_dir=raisin_out,
        fdrmethod="fdr_bh",
        fdr_threshold=0.05,
        verbose=True,
    )

    # ---- Sex panel cross-check ----
    print("\n=== Sex panel cross-check ===")
    # Find any per-pair DE output file (typically TSV)
    candidates = []
    for root, _, files in os.walk(raisin_out):
        for f in files:
            if f.endswith((".tsv", ".csv")):
                candidates.append(os.path.join(root, f))
    print(f"  found {len(candidates)} test output files")
    for c in candidates[:10]:
        print(f"    {c}")

    panel_flat = [g for L in SEX_PANEL.values() for g in L]
    summary_rows = []
    for c in candidates:
        try:
            df = pd.read_csv(c, sep="\t" if c.endswith(".tsv") else ",")
        except Exception:
            continue
        # try to find a gene column
        gene_col = None
        for cand in ("gene", "Gene", "feature", "Feature", "gene_name"):
            if cand in df.columns:
                gene_col = cand; break
        if gene_col is None:
            continue
        # find effect / FDR cols
        eff_col = next((x for x in ["log2FC","logFC","log2_fold_change","effect_size","coef","beta"] if x in df.columns), None)
        fdr_col = next((x for x in ["fdr","FDR","padj","adj_pval","q_value"] if x in df.columns), None)
        pval_col = next((x for x in ["pval","p_value","pvalue","p"] if x in df.columns), None)
        for g in panel_flat:
            hits = df[df[gene_col].astype(str) == g]
            for _, row in hits.iterrows():
                summary_rows.append({
                    "file": os.path.basename(c),
                    "gene": g,
                    "category": next(k for k,v in SEX_PANEL.items() if g in v),
                    "effect": float(row[eff_col]) if eff_col and pd.notna(row.get(eff_col)) else None,
                    "fdr":    float(row[fdr_col]) if fdr_col and pd.notna(row.get(fdr_col)) else None,
                    "pval":   float(row[pval_col]) if pval_col and pd.notna(row.get(pval_col)) else None,
                })
    summ = pd.DataFrame(summary_rows)
    if not summ.empty:
        out_csv = os.path.join(OUT_DIR, "sex_gene_summary.csv")
        summ.to_csv(out_csv, index=False)
        print(f"\nSex-panel hits (showing top 30):")
        if "fdr" in summ.columns and summ["fdr"].notna().any():
            summ_sorted = summ.sort_values("fdr")
        else:
            summ_sorted = summ
        print(summ_sorted.head(30).to_string(index=False))
        print(f"\nwrote {out_csv}")

        # JSON: per-category recovery
        recovery = {}
        for cat, genes in SEX_PANEL.items():
            hit = set(summ.loc[summ.category==cat, "gene"].unique())
            recovery[cat] = {
                "panel_size": len(genes),
                "hit_in_DE_output": len(hit),
                "hit_genes": sorted(hit),
                "missing":  sorted(set(genes) - hit),
            }
        out_json = os.path.join(OUT_DIR, "sex_gene_overlap.json")
        with open(out_json, "w") as f:
            json.dump(recovery, f, indent=2)
        print(f"wrote {out_json}")
    else:
        print("\nNo sex panel gene rows found in DE output. Inspect raisin_results/ manually.")

    print("\nDone.")


if __name__ == "__main__":
    main()
