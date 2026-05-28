"""
Per-cell-type top age-DGE table from round1_batch/trajectory_age/.

The default pseudoDEG selection (top 100 by effect-size across all celltypes)
was dominated by CD8 because CD8 has the strongest per-gene aging effects.
This script breaks results out per cell type so every compartment's signature
is visible, then cross-checks each against canonical paper markers.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GAM_TSV = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/trajectory_age/trajectoryDEG/gam_all_genes_20260524_182717.tsv"
OUT_DIR = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/trajectory_age/per_celltype_DGE"
TOP_N = 25

# Canonical paper / aging-literature markers per cell type (Terekhova 2023 + Mogilenko 2021 + Hashimoto 2019)
PAPER_MARKERS = {
    "B cells": {
        "UP_with_age": ["TBX21","ZEB2","ITGAX","FCRL5","IL10RB","CXCR3","JUN","FOS","NR4A1","CR2","TCL1A"],  # atypical memory + activated
        "DOWN_with_age": ["TCL1A","FCER2"],  # naive markers (some shared)
    },
    "CD4+ T cells": {
        "UP_with_age": ["GATA3","CCR4","IL4","IL2RA","HLA-DRA","HLA-DRB1","HLA-DPA1","CIITA","IL2RB","FOXP3","CTLA4","TIGIT","ANXA1","KLRB1","RORC"],
        "DOWN_with_age": ["CCR7","LEF1","TCF7","SELL","IL7R","NELL2","CAMK4","SATB1"],
    },
    "DN T cells": {
        "UP_with_age": ["GZMK","TBX21","EOMES","NKG7","CCL5"],
        "DOWN_with_age": ["CCR7","LEF1","TCF7"],
    },
    "MAIT cells": {
        # Paper Fig 1D: MAIT abundance decreases with age. Less transcriptional detail per cluster.
        "UP_with_age": ["GZMA","GZMK","NKG7","CCL5"],
        "DOWN_with_age": ["IL7R","TCF7","KLRB1"],
    },
    "Myeloid cells": {
        "UP_with_age": ["GZMK","HLA-DRA","CXCL10","IFI6","ISG15","MX1","IL10","FCGR3A","C1QA","C1QB"],  # GZMK+ macro / IFN signature
        "DOWN_with_age": ["LYZ","S100A8","S100A9","CD14"],
    },
    "NK cells": {
        "UP_with_age": ["TBX21","EOMES","ZNF683","IKZF2","PRF1","GZMB","GZMH","CX3CR1","B3GAT1","FCGR3A","KLRG1","FOSB","JUND","ZBTB38","ZEB2"],
        "DOWN_with_age": ["KLRC1","XCL1","XCL2","CCR7","IL7R","TCF7","SELL"],
    },
    "Progenitor cells": {
        "UP_with_age": [],   # not specifically discussed in paper
        "DOWN_with_age": ["CD34","SOX4","MEIS1","SPINK2","KIT"],  # canonical progenitor markers may drop as progenitor frequency declines
    },
    "TRAV1-2- CD8+ T cells": {
        "UP_with_age": ["GZMK","GZMB","NKG7","CCL5","GZMA","CST7","ZEB2","BHLHE40","KLRG1","CX3CR1","HLA-DRA","HLA-DPA1","CIITA","TBX21","EOMES","IKZF2","ZNF683","CCR4"],
        "DOWN_with_age": ["CCR7","LEF1","TCF7","SELL","IL7R","TMIGD2","LRRN3","NELL2","MYC","SATB1","CAMK4","NT5E","TGFBR2","TXK","ITGA6","MAL"],
    },
    "gd T cells": {
        "UP_with_age": ["GZMB","TBX21","EOMES","ZEB2","CX3CR1"],   # Vd1 GZMB+ expansion
        "DOWN_with_age": ["TCF7","LEF1","CCR7","CD27"],            # naive gd decline
    },
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(GAM_TSV, sep="\t")
    df[["celltype","gene_sym"]] = df["gene"].str.rsplit(" - ", n=1, expand=True)
    df_sig = df[df["significant"] == True].copy()
    print(f"loaded {len(df):,} fits; {len(df_sig):,} significant at FDR<0.05")

    per_celltype = {}
    paper_hits_summary = []

    for ct in sorted(df_sig["celltype"].unique()):
        sub = df_sig[df_sig["celltype"] == ct].copy()
        up = sub[sub["regulation"] == "UP"].sort_values("effect_size", ascending=False).head(TOP_N)
        dn = sub[sub["regulation"] == "DOWN"].sort_values("effect_size", ascending=False).head(TOP_N)

        # Save per-CT CSV
        out_ct = up.assign(direction="UP")._append(dn.assign(direction="DOWN"))
        out_ct = out_ct[["gene_sym","direction","effect_size","dev_exp","pval","fdr"]]
        safe_name = ct.replace(" ", "_").replace("/", "_").replace("+","plus").replace("-","minus")
        out_ct_path = os.path.join(OUT_DIR, f"top_DGE_{safe_name}.csv")
        out_ct.to_csv(out_ct_path, index=False)

        # Paper marker overlap (case-insensitive simple match)
        paper = PAPER_MARKERS.get(ct, {"UP_with_age": [], "DOWN_with_age": []})
        up_genes = set(up["gene_sym"].astype(str).str.upper())
        dn_genes = set(dn["gene_sym"].astype(str).str.upper())
        paper_up = set(g.upper() for g in paper["UP_with_age"])
        paper_dn = set(g.upper() for g in paper["DOWN_with_age"])

        up_recovered = sorted(up_genes & paper_up)
        dn_recovered = sorted(dn_genes & paper_dn)
        # Concordance: a "miss" = paper says UP but we have DOWN, or vice versa
        up_in_dn = sorted(dn_genes & paper_up)
        dn_in_up = sorted(up_genes & paper_dn)

        per_celltype[ct] = {
            "n_significant": int(len(sub)),
            "n_up": int((sub["regulation"]=="UP").sum()),
            "n_down": int((sub["regulation"]=="DOWN").sum()),
            "top_up": list(up["gene_sym"]),
            "top_down": list(dn["gene_sym"]),
            "paper_up_recovered_in_topN_UP": up_recovered,
            "paper_down_recovered_in_topN_DOWN": dn_recovered,
            "paper_up_appeared_in_topN_DOWN": up_in_dn,
            "paper_down_appeared_in_topN_UP": dn_in_up,
            "paper_up_total": len(paper_up),
            "paper_down_total": len(paper_dn),
            "n_paper_up_recovered": len(up_recovered),
            "n_paper_down_recovered": len(dn_recovered),
        }
        # also a flat summary row
        paper_hits_summary.append({
            "celltype": ct,
            "n_paper_up": len(paper_up),
            "n_paper_up_recovered_in_top25_UP": len(up_recovered),
            "n_paper_down": len(paper_dn),
            "n_paper_down_recovered_in_top25_DOWN": len(dn_recovered),
            "discordant_paper_up_in_DOWN": len(up_in_dn),
            "discordant_paper_down_in_UP": len(dn_in_up),
        })

    # Save summary JSON
    out_json = os.path.join(OUT_DIR, "paper_marker_recovery.json")
    with open(out_json, "w") as f:
        json.dump(per_celltype, f, indent=2)

    # Save concordance table
    summ_df = pd.DataFrame(paper_hits_summary)
    summ_csv = os.path.join(OUT_DIR, "paper_marker_recovery_summary.csv")
    summ_df.to_csv(summ_csv, index=False)

    # Print human report
    print("\n" + "="*78)
    print("PER-CELL-TYPE TOP AGE DGE  (FDR<0.05, ranked by GAM effect size)")
    print("="*78)
    for ct, d in per_celltype.items():
        print(f"\n### {ct}  (n_sig={d['n_significant']}, {d['n_up']} UP / {d['n_down']} DOWN)")
        print(f"  Top {len(d['top_up'])} UP:  ", ", ".join(d["top_up"][:15]))
        print(f"  Top {len(d['top_down'])} DOWN:", ", ".join(d["top_down"][:15]))
        paper_n_up = d["paper_up_total"]; paper_n_dn = d["paper_down_total"]
        ru = d["paper_up_recovered_in_topN_UP"]; rd = d["paper_down_recovered_in_topN_DOWN"]
        print(f"  Paper UP markers recovered in top-{TOP_N} UP: {len(ru)}/{paper_n_up}  -> {ru}")
        print(f"  Paper DOWN markers recovered in top-{TOP_N} DOWN: {len(rd)}/{paper_n_dn} -> {rd}")
        if d["paper_up_appeared_in_topN_DOWN"]:
            print(f"  [discordant] paper says UP but we found DOWN: {d['paper_up_appeared_in_topN_DOWN']}")
        if d["paper_down_appeared_in_topN_UP"]:
            print(f"  [discordant] paper says DOWN but we found UP: {d['paper_down_appeared_in_topN_UP']}")

    print(f"\nwrote: {OUT_DIR}/")
    print(f"  paper_marker_recovery_summary.csv")
    print(f"  paper_marker_recovery.json")
    for f_ in sorted(os.listdir(OUT_DIR)):
        if f_.startswith("top_DGE_"):
            print(f"  {f_}")


if __name__ == "__main__":
    main()
