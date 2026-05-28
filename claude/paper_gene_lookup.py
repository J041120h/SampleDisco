"""
Targeted paper-gene lookup against full GAM table.

The per-celltype top-25 covers high-effect-size genes. This script
complements it by asking: for each gene the paper names as a cluster marker
defining an age-changing cluster, is that gene significant in our GAM, in
which direction, and at what rank within its cell type?

Output: paper_gene_lookup.csv + paper_gene_lookup.md
"""

import os
from pathlib import Path

import pandas as pd

GAM_TSV = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/trajectory_age/trajectoryDEG/gam_all_genes_20260524_182717.tsv"
OUT_CSV = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/trajectory_age/per_celltype_DGE/paper_gene_lookup.csv"
OUT_MD  = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/trajectory_age/per_celltype_DGE/paper_gene_lookup.md"

# (celltype, gene, paper_implied_direction, paper_basis)
# paper_implied_direction is the direction the cluster-proportion change implies
# for the bulk-celltype expression. "?" = uncertain or not predicted.
PAPER_HITS = [
    # CD4+ T cells — Fig 2D age findings: Th2 ↑***, HLA-DR+ memory ↑***, Treg memory ↑*, Treg naive ↓**
    ("CD4+ T cells", "GATA3",   "UP",   "Th2 marker (Fig 2B); Th2 cluster ↑*** with age"),
    ("CD4+ T cells", "CCR4",    "UP",   "Th2 marker (Fig 2B); Th2 cluster ↑*** with age"),
    ("CD4+ T cells", "IL4",     "UP",   "Th2 cytokine (Fig 5L); Th2 cluster ↑*** with age"),
    ("CD4+ T cells", "HLA-DRA", "UP",   "HLA-DR+ memory marker (Fig 2B); cluster ↑*** with age"),
    ("CD4+ T cells", "HLA-DRB1","UP",   "HLA-DR+ memory marker (Fig 2B); cluster ↑*** with age"),
    ("CD4+ T cells", "HLA-DPA1","UP",   "HLA-DR+ memory marker (Fig 2B); cluster ↑*** with age"),
    ("CD4+ T cells", "CIITA",   "UP",   "HLA-DR+ memory regulator; cluster ↑*** with age"),
    ("CD4+ T cells", "FOXP3",   "UP",   "Treg marker (Fig 2B); Treg memory ↑* with age"),
    ("CD4+ T cells", "IL2RA",   "UP",   "Treg marker (Fig 2B / Fig 3J); Treg memory ↑* with age"),
    ("CD4+ T cells", "CTLA4",   "UP",   "Treg activation marker (Fig 2B)"),
    ("CD4+ T cells", "CCR7",    "DOWN", "Naive Treg marker; Treg naive ↓** with age"),
    ("CD4+ T cells", "LEF1",    "DOWN", "Naive Treg marker; Treg naive ↓** with age"),

    # CD8+ T cells (TRAV1-2-) — Fig 4E: Naive ↓***, Tcm CCR4- ↑***, Tcm CCR4+ ↑**, Tmem KLRC2+ ↓***, Tem GZMK+ ↑***
    ("TRAV1-2- CD8+ T cells", "SELL",  "DOWN", "Naive marker (Fig 4B); Naive cluster ↓*** with age"),
    ("TRAV1-2- CD8+ T cells", "CCR7",  "DOWN", "Naive marker (Fig 4B); Naive cluster ↓*** with age"),
    ("TRAV1-2- CD8+ T cells", "LEF1",  "DOWN", "Naive/Tcm marker (Fig 4B); Naive cluster ↓*** with age"),
    ("TRAV1-2- CD8+ T cells", "TCF7",  "DOWN", "Naive/Tcm marker (Fig 4B); Naive cluster ↓*** with age"),
    ("TRAV1-2- CD8+ T cells", "IL7R",  "DOWN", "Naive/memory homeostasis marker"),
    ("TRAV1-2- CD8+ T cells", "GZMK",  "UP",   "Tem GZMK+ marker (Fig 4D); cluster ↑*** with age"),
    ("TRAV1-2- CD8+ T cells", "KLRC2", "DOWN", "Tmem KLRC2+ marker (Fig 5B); cluster ↓*** with age (paper's novel cluster)"),
    ("TRAV1-2- CD8+ T cells", "CCR4",  "UP",   "Tcm CCR4+ marker (Fig 5E); cluster ↑** with age"),
    ("TRAV1-2- CD8+ T cells", "CCL5",  "UP",   "effector cytokine"),
    ("TRAV1-2- CD8+ T cells", "NKG7",  "UP",   "effector granule marker"),

    # γδ T cells — Fig 6H: γδ naive ↓***, Vδ1 GZMB+ ↑**
    ("gd T cells", "TCF7",  "DOWN", "γδ naive marker (Fig 6G); cluster ↓*** with age"),
    ("gd T cells", "LEF1",  "DOWN", "γδ naive marker (Fig 6G); cluster ↓*** with age"),
    ("gd T cells", "CD27",  "DOWN", "γδ naive marker (Fig 6G); cluster ↓*** with age"),
    ("gd T cells", "GZMB",  "UP",   "Vδ1 GZMB+ marker (Fig 6G); cluster ↑** with age"),
    ("gd T cells", "PRF1",  "UP",   "Vδ1 GZMB+ marker (Fig 6G)"),
    ("gd T cells", "ZEB2",  "UP",   "terminal effector TF; Vδ1 GZMB+ marker"),

    # NK cells — paper explicitly: NO age-associated changes (p.2848). Listed for negative control.
    ("NK cells", "IKZF2",  "UP?", "CD56dim CD57+ marker (Fig 6B) — paper says NK is age-invariant; ANY direction is uncertain"),
    ("NK cells", "TBX21",  "UP?", "CD56dim CD57+ marker (Fig 6B) — paper says NK is age-invariant"),
    ("NK cells", "GZMB",   "UP?", "CD56dim CD57+ marker — paper says NK is age-invariant"),
    ("NK cells", "PRF1",   "UP?", "CD56dim CD57+ marker — paper says NK is age-invariant"),
    ("NK cells", "TCF7",   "DOWN?","CD56bright marker (Fig 6B); CD56bright marginally trends ↓ but paper marks NK as age-invariant overall"),

    # B cells — paper: limited age changes (Fig 7C). Listed for negative control on transcriptional signature.
    ("B cells", "TBX21",  "UP?", "atypical memory marker (Fig 7B) — paper reports BCR clonal change in CD5+ but no transcriptional aging signature"),
    ("B cells", "FCRL5",  "UP?", "atypical memory marker (Fig 7B) — same caveat"),
    ("B cells", "MX1",    "UP?", "naive-IFN marker (Fig 7B)"),

    # MAIT cells — paper: abundance ↓ after 55 yr (Fig 1D), no per-cluster transcriptional signature
    ("MAIT cells", "KLRB1", "DOWN?", "MAIT marker; paper reports abundance decline but no within-MAIT signature"),
    ("MAIT cells", "IL7R",  "DOWN?", "MAIT marker; abundance decline only"),

    # Myeloid — paper: no age-associated remodeling (p.2838)
    ("Myeloid cells", "HLA-DRA", "?", "paper reports no age signal in myeloid"),
    ("Myeloid cells", "CD14",    "?", "paper reports no age signal in myeloid"),
]


def main():
    df = pd.read_csv(GAM_TSV, sep="\t")
    df[["celltype", "gene_sym"]] = df["gene"].str.rsplit(" - ", n=1, expand=True)

    rows = []
    for ct, gene, expected, basis in PAPER_HITS:
        sub = df[df["celltype"] == ct]
        # rank within CT by effect size, in the direction the paper implies (or by absolute effect for "?")
        if expected.rstrip("?") == "UP":
            sub_dir = sub[sub.get("regulation", "") == "UP"]
        elif expected.rstrip("?") == "DOWN":
            sub_dir = sub[sub.get("regulation", "") == "DOWN"]
        else:
            sub_dir = sub
        sub_dir = sub_dir.sort_values("effect_size", ascending=False).reset_index(drop=True)
        sub_dir["rank_in_direction"] = sub_dir.index + 1

        hit = sub_dir[sub_dir["gene_sym"] == gene]
        if hit.empty:
            # try in any direction
            in_any = sub[sub["gene_sym"] == gene]
            if in_any.empty:
                rows.append({
                    "celltype": ct, "gene": gene, "paper_direction": expected, "paper_basis": basis,
                    "found": False, "our_direction": None, "effect_size": None, "fdr": None,
                    "rank_in_expected_dir_within_CT": None, "comment": "NOT in GAM table for this CT (not HVG)",
                })
            else:
                r = in_any.iloc[0]
                rows.append({
                    "celltype": ct, "gene": gene, "paper_direction": expected, "paper_basis": basis,
                    "found": True,
                    "our_direction": r["regulation"],
                    "effect_size": float(r["effect_size"]),
                    "fdr": float(r["fdr"]),
                    "rank_in_expected_dir_within_CT": None,
                    "comment": f"Significant in opposite direction" if r["significant"] else "Not significant",
                })
        else:
            r = hit.iloc[0]
            rows.append({
                "celltype": ct, "gene": gene, "paper_direction": expected, "paper_basis": basis,
                "found": True,
                "our_direction": r["regulation"],
                "effect_size": float(r["effect_size"]),
                "fdr": float(r["fdr"]),
                "rank_in_expected_dir_within_CT": int(r["rank_in_direction"]),
                "comment": "concordant with paper" if (
                    (expected.rstrip("?") == r["regulation"]) and r["significant"]
                ) else ("ns" if not r["significant"] else "discordant"),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")

    # markdown table per cell type
    md = ["# Paper-implied gene lookup vs. our GAM age DGE",
          "",
          "For each gene named in the paper as a marker of a cluster that **the paper itself reports as age-changing**, we look it up by name in our per-cell-type GAM table (full 16,324-fit table, not just top-25). Cells with `?` direction are paper-negative controls where the paper says no age change is expected — we report what we see for completeness.",
          "",
          "**Columns:**",
          "- `paper_direction`: direction implied by the paper's cluster-proportion test for that compartment.",
          "- `our_direction`: direction our GAM assigned this gene (UP / DOWN within the cell type, by GAM coefficient).",
          "- `effect_size`, `fdr`: from `gam_all_genes_20260524_182717.tsv`.",
          "- `rank_in_expected_dir_within_CT`: rank of this gene among all genes moving in the paper-expected direction within this cell type (1 = strongest).",
          "- `comment`: concordant / discordant / ns / not-in-table.",
          "",
          "## Results",
          ""]
    for ct in sorted(out["celltype"].unique()):
        sub = out[out["celltype"] == ct]
        md.append(f"### {ct}")
        md.append("")
        md.append("| Gene | Paper dir | Our dir | Effect | FDR | Rank in expected dir | Concordance | Paper basis |")
        md.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            if not r["found"]:
                md.append(f"| {r['gene']} | {r['paper_direction']} | — | — | — | — | not in GAM table for this CT | {r['paper_basis']} |")
                continue
            eff = "—" if pd.isna(r['effect_size']) else f"{r['effect_size']:.3f}"
            fdr = "—" if pd.isna(r['fdr']) else f"{r['fdr']:.2e}"
            rk = "—" if pd.isna(r['rank_in_expected_dir_within_CT']) else f"{int(r['rank_in_expected_dir_within_CT'])}"
            our_dir = r['our_direction'] if isinstance(r['our_direction'], str) else "—"
            md.append(f"| {r['gene']} | {r['paper_direction']} | {our_dir} | {eff} | {fdr} | {rk} | {r['comment']} | {r['paper_basis']} |")
        md.append("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))
    print(f"wrote {OUT_MD}")

    # console summary
    print("\n=== Concordance summary ===")
    print(out.groupby("comment").size().to_string())


if __name__ == "__main__":
    main()
