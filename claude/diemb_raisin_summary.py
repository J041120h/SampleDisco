"""Consolidated RAISIN summary across cell types.

Reads raisin_results/<cell_type>/<comparison>/raisin_results.csv (written by
RAISIN_TEST.run_pairwise_tests) and produces a single cross-cell-type summary:
  RAISIN_SUMMARY.csv  — one row per (cell_type, comparison): n_sig / up / down
  RAISIN_SUMMARY.txt  — a pivot table of significant-DEG counts + per-cell-type
                        top genes.
No re-fitting; purely aggregates existing result CSVs.

Usage: python diemb_raisin_summary.py <raisin_results_dir> [fdr_threshold] [top_n]
"""
import glob
import os
import sys

import pandas as pd

RR = (sys.argv[1] if len(sys.argv) > 1 else
      "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics/"
      "sample_embedding_tune-on-RNA/cluster_severity_deg/raisin_results")
FDR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
TOP_N = int(sys.argv[3]) if len(sys.argv) > 3 else 25

rows, top_genes = [], {}
for ct in sorted(d for d in os.listdir(RR) if os.path.isdir(os.path.join(RR, d))):
    for f in sorted(glob.glob(os.path.join(RR, ct, "*", "raisin_results.csv"))):
        comp = os.path.basename(os.path.dirname(f))
        df = pd.read_csv(f, index_col=0)
        sig = df[df["FDR"] < FDR].sort_values("FDR")
        rows.append({"cell_type": ct, "comparison": comp,
                     "n_genes": int(len(df)), "n_sig": int(len(sig)),
                     "n_up": int((sig["Foldchange"] > 0).sum()),
                     "n_down": int((sig["Foldchange"] < 0).sum())})
        top_genes[(ct, comp)] = sig.head(TOP_N)

summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(RR, "RAISIN_SUMMARY.csv"), index=False)

pivot = (summ.pivot(index="cell_type", columns="comparison", values="n_sig")
         .fillna(0).astype(int)) if len(summ) else pd.DataFrame()

with open(os.path.join(RR, "RAISIN_SUMMARY.txt"), "w") as fh:
    fh.write("RAISIN cluster-DEG summary across cell types "
             f"(significant = FDR < {FDR})\n")
    fh.write("=" * 70 + "\n\n")
    fh.write("Significant DEG counts (rows=cell type, cols=comparison):\n")
    fh.write(pivot.to_string() + "\n\n")
    fh.write(f"Totals: {int(summ['n_sig'].sum())} significant calls across "
             f"{summ['cell_type'].nunique()} cell types / "
             f"{summ['comparison'].nunique()} comparisons.\n\n")
    fh.write("=" * 70 + "\nPer cell type / comparison — top genes\n" + "=" * 70 + "\n\n")
    for (ct, comp), sig in top_genes.items():
        if len(sig) == 0:
            continue
        fh.write(f"--- {ct}  |  {comp}  ({len(sig)} sig, "
                 f"showing top {min(TOP_N, len(sig))}) ---\n")
        for gene, r in sig.iterrows():
            fh.write(f"  {str(gene):20s} logFC={r['Foldchange']:+.3f}  "
                     f"FDR={r['FDR']:.2e}\n")
        fh.write("\n")

print(f"Wrote {os.path.join(RR, 'RAISIN_SUMMARY.csv')}")
print(f"Wrote {os.path.join(RR, 'RAISIN_SUMMARY.txt')}")
print("\nSignificant DEG counts (FDR<%.2g):" % FDR)
print(pivot.to_string())
