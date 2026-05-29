"""(1) Random-shuffle (permutation) baseline for the ATAC-RNA concordance, and
(2) extraction of strongly-concordant up/down genes as candidates.

Shuffle baseline: the observed concordance is the fraction of gene-peak pairs
with matching directions. We permute peak directions across pairs (breaking the
gene-peak link, preserving marginals) N times to build a null distribution, then
report observed vs null (z-score + empirical p) and a histogram.

Candidates: per gene, fraction of its peaks concordant with the gene direction;
keep significant genes with >=5 peaks and >=80% concordant, split up/down.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/users/hjiang/GenoDistance/figure/figure5"
TSV = f"{BASE}/concordance_results.tsv"
N_PERM = 2000
RNG = np.random.default_rng(0)

df = pd.read_csv(TSV, sep="\t")
gd = (df["gene_direction"] == "up").to_numpy()      # bool: gene up?
pdir = (df["peak_direction"] == "up").to_numpy()     # bool: peak up?
obs = float((gd == pdir).mean())

null = np.empty(N_PERM)
for i in range(N_PERM):
    null[i] = (gd == RNG.permutation(pdir)).mean()
mu, sd = null.mean(), null.std()
z = (obs - mu) / sd
emp_p = (np.sum(null >= obs) + 1) / (N_PERM + 1)

print("=== Shuffle baseline (overall gene-peak concordance) ===", flush=True)
print(f"  n pairs: {len(df):,}", flush=True)
print(f"  observed concordance: {obs*100:.2f}%", flush=True)
print(f"  shuffled null: mean={mu*100:.2f}%  sd={sd*100:.3f}%  (range {null.min()*100:.2f}-{null.max()*100:.2f}%)", flush=True)
print(f"  z-score: {z:.1f}   empirical p: {emp_p:.2e}  ({N_PERM} permutations)", flush=True)

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(null * 100, bins=40, color="#9e9ac8", edgecolor="white", label=f"shuffled null (n={N_PERM})")
ax.axvline(obs * 100, color="#d7301f", lw=3, label=f"observed = {obs*100:.1f}%")
ax.axvline(mu * 100, color="black", ls="--", lw=1.5, label=f"null mean = {mu*100:.1f}%")
ax.set_xlabel("Gene-peak concordance (%)")
ax.set_ylabel("permutations")
ax.set_title(f"ATAC-RNA concordance vs random-shuffle baseline\n"
             f"z = {z:.1f},  empirical p < {max(emp_p,1/(N_PERM+1)):.0e}")
ax.legend()
fig.tight_layout()
fig.savefig(f"{BASE}/visualizations/concordance_shuffle_baseline.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  wrote {BASE}/visualizations/concordance_shuffle_baseline.png", flush=True)

with open(f"{BASE}/concordance_shuffle_baseline.txt", "w") as fh:
    fh.write("ATAC-RNA concordance — random-shuffle (permutation) baseline\n")
    fh.write("=" * 64 + "\n\n")
    fh.write(f"n gene-peak pairs        : {len(df):,}\n")
    fh.write(f"observed concordance     : {obs*100:.2f}%\n")
    fh.write(f"shuffled null mean +- sd : {mu*100:.2f}% +- {sd*100:.3f}%\n")
    fh.write(f"z-score                  : {z:.2f}\n")
    fh.write(f"empirical p              : {emp_p:.2e}  ({N_PERM} permutations)\n")

# ── concordant gene candidates ────────────────────────────────────────────── #
g = df.groupby("gene_full").agg(
    gene_symbol=("gene_symbol", "first"), cell_type=("cell_type", "first"),
    gene_direction=("gene_direction", "first"), gene_fdr=("gene_fdr", "first"),
    effect=("gene_effect_size", "first"), n_peaks=("is_concordant", "size"),
    n_conc=("is_concordant", "sum")).reset_index()
g["conc_frac"] = g["n_conc"] / g["n_peaks"]
strong = g[(g["gene_fdr"] < 0.05) & (g["n_peaks"] >= 5) & (g["conc_frac"] >= 0.8)]
for d in ["up", "down"]:
    s = strong[strong["gene_direction"] == d].sort_values(["conc_frac", "effect"], ascending=False)
    print(f"\n=== strongly-concordant {d.upper()} genes (fdr<0.05, >=5 peaks, >=80% concordant): {len(s)} ===", flush=True)
    for _, r in s.head(40).iterrows():
        print(f"  {r['gene_symbol']:12s} [{r['cell_type']}] eff={r['effect']:.2f} "
              f"conc={int(r['n_conc'])}/{int(r['n_peaks'])} ({r['conc_frac']*100:.0f}%) fdr={r['gene_fdr']:.0e}", flush=True)
strong.to_csv(f"{BASE}/concordant_gene_candidates.csv", index=False)
print(f"\nwrote {BASE}/concordant_gene_candidates.csv", flush=True)
print("SHUFFLE_DONE", flush=True)
