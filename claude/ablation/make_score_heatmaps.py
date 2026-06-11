#!/usr/bin/env python
"""Paper-style score heatmaps for the ablation.

(A) Dot heatmap: rows = (dataset, metric), columns = the 5 variants (sorted by
    overall mean rank); dot SIZE = rank within that metric (bigger = better),
    dot COLOR = within-metric min-max-normalized value (viridis). Mirrors
    figure3/figure4 `plot_dots`.
(B) Grand mean-rank heatmap: variant x dataset, cell = mean rank (viridis_r,
    annotated; brighter = better).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ABL = "/dcs07/hongkai/data/harry/result/ablation"
VARIANTS = ["proportion_only", "rmd_only", "no_batch_removal", "linear_regression", "original"]
N = len(VARIANTS)

# metric -> (display label, direction)  direction: 'up' higher=better, 'inv' 1/x, 'abs' |x|
COVID_M = {
    "batch_partial_eta_sq": ("batch eta² (inv)", "inv"), "iLISI_norm": ("iLISI", "up"),
    "ASW_batch": ("ASW batch", "up"), "severity_partial_eta_sq": ("severity eta²", "up"),
    "Spearman_Correlation": ("|Spearman|", "abs"), "Custom_ANOVA_eta_sq": ("Su ANOVA eta²", "up"),
    "ARI": ("ARI", "up"), "NMI": ("NMI", "up"), "Avg_Purity": ("purity", "up"),
    "Mean_NN_Severity_Gap": ("NN sev gap (inv)", "inv"),
}
MO_M = {"paired_partner_rank": ("1/partner_rank", "inv"), "ASW_modality": ("ASW modality", "up")}
MO_PRES = {"ENCODE": "tissue_preservation_score", "heart": "disease_state_preservation_score",
           "retina": "cca_score", "lutea": "cca_score"}
HA_M = {
    "age_CCA": ("age CCA", "up"), "age_best_PC_spearman": ("age |spear|", "abs"),
    "cd4cd8_CCA": ("CD4/CD8 CCA", "up"), "cd4cd8_best_PC_spearman": ("CD4/CD8 |spear|", "abs"),
    "mait_CCA": ("MAIT CCA", "up"), "mait_best_PC_spearman": ("MAIT |spear|", "abs"),
    "ASW_batch": ("ASW batch", "up"), "iLISI_batch_norm": ("iLISI batch", "up"),
    "ASW_file": ("ASW file", "up"), "iLISI_file_norm": ("iLISI file", "up"),
}


def tr(v, d):
    if pd.isna(v): return np.nan
    if d == "inv": return 1.0/v if v not in (0, 0.0) else np.nan
    if d == "abs": return abs(v)
    return v


def covid_vals(metric, d):
    df = pd.read_csv(f"{ABL}/covid/ablation_summary_covid_ALL.csv", index_col=0)
    sizes = sorted({c.split("-")[1] for c in df.columns}, key=int)
    out = {}
    for v in VARIANTS:
        cols = [f"{v}-{s}" for s in sizes if f"{v}-{s}" in df.columns]
        out[v] = np.nanmean([tr(x, d) for x in pd.to_numeric(df.loc[metric, cols], errors="coerce")])
    return out


def mo_vals(ds, metric, d):
    df = pd.read_csv(f"{ABL}/multiomics/{ds}/ablation_summary_mo_{ds}.csv", index_col=0)
    return {v: tr(float(df.loc[metric, v]), d) for v in VARIANTS}


def ha_vals(metric, d):
    df = pd.read_csv(f"{ABL}/health_aging/ablation_summary_health_aging.csv", index_col=0)
    return {v: tr(float(df.loc[v, metric]), d) for v in VARIANTS}


def build_rows(exclude=()):
    rows = []   # (dataset, label, {variant: value})
    if "COVID" not in exclude:
        for m, (lab, d) in COVID_M.items():
            rows.append(("COVID", lab, covid_vals(m, d)))
    for ds in ["ENCODE", "heart", "retina", "lutea"]:
        if ds in exclude:
            continue
        for m, (lab, d) in MO_M.items():
            rows.append((ds, lab, mo_vals(ds, m, d)))
        rows.append((ds, "preservation", mo_vals(ds, MO_PRES[ds], "up")))
    if "health_aging" not in exclude:
        for m, (lab, d) in HA_M.items():
            rows.append(("health_aging", lab, ha_vals(m, d)))
    return rows


def main(exclude=(), suffix=""):
    rows = build_rows(exclude)
    # per-row rank + normalized value
    rank_rows, norm_rows = [], []
    for _, _, vals in rows:
        s = pd.Series(vals).reindex(VARIANTS)
        rank_rows.append(s.rank(ascending=False, method="min", na_option="bottom"))
        lo, hi = s.min(), s.max()
        norm_rows.append((s - lo) / (hi - lo) if hi > lo else s * 0 + 0.5)
    rank_df = pd.DataFrame(rank_rows)          # row x variant
    norm_df = pd.DataFrame(norm_rows)
    order = rank_df.mean(axis=0).sort_values().index.tolist()   # variants by overall mean rank

    # ---------- (A) dot heatmap ----------
    nrow = len(rows)
    cmap = plt.cm.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7.0, 0.34*nrow + 2.2))
    smin, smax = 40, 360
    for yi, (ds, lab, _) in enumerate(rows):
        y = nrow - 1 - yi
        for xi, v in enumerate(order):
            r = rank_df.iloc[yi][v]; nv = norm_df.iloc[yi][v]
            if pd.isna(r): continue
            size = smin + (N - r) / (N - 1) * (smax - smin)
            ax.scatter(xi, y, s=size, c=[cmap(nv if not pd.isna(nv) else 0.5)],
                       edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_xticks(range(N)); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([f"{lab}" for ds, lab, _ in rows][::-1], fontsize=7.5)
    # dataset separators + headers
    boundaries, prev = [], None
    for yi, (ds, _, _) in enumerate(rows):
        if ds != prev:
            boundaries.append((nrow - yi, ds)); prev = ds
    for b, _ in boundaries[1:]:
        ax.axhline(b - 0.5, color="#cccccc", lw=0.8)
    for b, ds in boundaries:
        ax.text(N - 0.3, b - 1, ds, fontsize=8, fontweight="bold", color="#2C3E50", va="top")
    ax.set_xlim(-0.6, N - 0.2); ax.set_ylim(-0.6, nrow - 0.4)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title("Ablation scores — dot heatmap\n(size = rank, 1=best; color = normalized value)",
                 fontsize=10, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.12)
    cb.set_label("within-metric normalized value (1=best)", fontsize=7)
    size_leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                       markersize=np.sqrt(smin + (N - r)/(N-1)*(smax-smin))/2.2,
                       label=f"rank {int(r)}") for r in [1, 3, 5]]
    ax.legend(handles=size_leg, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=7, title="dot size", title_fontsize=7, labelspacing=1.0)
    outA = f"{ABL}/ablation_score_dotheatmap{suffix}.png"
    fig.savefig(outA, dpi=300, bbox_inches="tight"); plt.close(fig); print("->", outA)

    # ---------- (B) grand mean-rank heatmap (variant x dataset) ----------
    g = pd.read_csv(f"{ABL}/ablation_GRAND_overview.csv", index_col=0)
    dss = [c for c in ["COVID", "ENCODE", "heart", "retina", "lutea", "health_aging"]
           if c in g.columns and c not in exclude]
    g_order = g[dss].mean(axis=1).sort_values().index.tolist()   # re-rank over remaining datasets
    order = g_order
    M = g.loc[order, dss]
    fig, ax = plt.subplots(figsize=(1.0*len(dss) + 2.5, 0.6*N + 1.8))
    im = ax.imshow(M.values, cmap="viridis_r", aspect="auto", vmin=1, vmax=N)
    ax.set_xticks(range(len(dss))); ax.set_xticklabels(dss, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(N)); ax.set_yticklabels(order, fontsize=9)
    for i in range(N):
        for j in range(len(dss)):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if M.values[i, j] > 3 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("mean rank (1=best)", fontsize=8)
    ax.set_title("Ablation mean rank by dataset\n(variants sorted by overall rank)", fontsize=10, fontweight="bold")
    outB = f"{ABL}/ablation_grand_rank_heatmap{suffix}.png"
    fig.savefig(outB, dpi=300, bbox_inches="tight"); plt.close(fig); print("->", outB)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="", help="comma-separated datasets to drop")
    ap.add_argument("--suffix", default="", help="filename suffix")
    a = ap.parse_args()
    exclude = tuple(x for x in a.exclude.split(",") if x)
    main(exclude=exclude, suffix=a.suffix)
