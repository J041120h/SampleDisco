#!/usr/bin/env python
"""Paper-style ablation embedding figures.

For each dataset, draw a 5-variant x 2-coloring grid of the sample embedding
(PC1 vs PC2): left column colored by the BATCH component (study batch, or
modality for multi-omics), right column colored by the BIOLOGICAL ground truth
(severity / tissue / disease_state / developmental age / age group). Styling
follows figure/embedding/*/embedding.py: white-edged dots, equal axes, clean
spines, paper palettes, paired-sample connector lines for multi-omics.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ABL = "/dcs07/hongkai/data/harry/result/ablation"
VARIANTS = ["proportion_only", "rmd_only", "no_batch_removal",
            "linear_regression", "original"]
TISSUE_PAL = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
              "#e5c494", "#b3b3b3", "#1f78b4", "#fb9a99", "#cab2d6", "#fdbf6f"]
MOD_COLORS = {"RNA": "#6CC6D8", "ATAC": "#EE7564"}


def _strip_mod(idx):
    return re.sub(r"_(RNA|ATAC|rna|atac)$", "", str(idx))


def _modality(idx):
    m = re.search(r"_(RNA|ATAC|rna|atac)$", str(idx))
    return m.group(1).upper() if m else "NA"


def build_labels(emb_index, cfg):
    """Return (batch_series, bio_series) aligned to emb_index."""
    meta = pd.read_csv(cfg["meta"], index_col=0)
    meta.index = meta.index.astype(str)
    idx = pd.Index([str(i) for i in emb_index])
    if cfg["multiomics"]:
        batch = pd.Series([_modality(i) for i in idx], index=idx)        # batch = modality
        bio_lookup = meta[cfg["bio_col"]]
        if cfg.get("meta_per_modality"):
            bio = pd.Series([bio_lookup.get(i, np.nan) for i in idx], index=idx)
        else:
            bio = pd.Series([bio_lookup.get(_strip_mod(i), np.nan) for i in idx], index=idx)
    else:
        batch = pd.Series([meta[cfg["batch_col"]].get(i, np.nan) for i in idx], index=idx)
        bio = pd.Series([meta[cfg["bio_col"]].get(i, np.nan) for i in idx], index=idx)
    return batch, bio


def _style(ax):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xm, ym = 0.5*(x0+x1), 0.5*(y0+y1); h = 0.5*max(x1-x0, y1-y0)
    ax.set_xlim(xm-h, xm+h); ax.set_ylim(ym-h, ym+h)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_linewidth(1.2)
    ax.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    ax.set_aspect("equal", adjustable="box")


def panel(ax, df, color, mode, title, draw_lines=False, continuous=False):
    if draw_lines:  # connect paired sample x modality units
        df = df.copy(); df["__b"] = [_strip_mod(i) for i in df.index]
        for _, p in df.groupby("__b"):
            if len(p) == 2:
                ax.plot(p["PC1"], p["PC2"], color="gray", alpha=0.3, lw=1.5, zorder=1)
    if continuous:
        c = pd.to_numeric(df[color], errors="coerce")
        sc = ax.scatter(df["PC1"], df["PC2"], c=c, cmap="viridis", s=42, alpha=0.85,
                        edgecolors="white", linewidths=0.4, zorder=2)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    else:
        groups = [g for g in pd.unique(df[color].dropna())]
        groups = sorted(groups, key=lambda x: str(x))
        if mode == "modality":
            cmap = {g: MOD_COLORS.get(str(g), "#333") for g in groups}
        elif len(groups) <= len(TISSUE_PAL):
            cmap = {g: TISSUE_PAL[i] for i, g in enumerate(groups)}
        else:
            tab = plt.cm.tab20(np.linspace(0, 1, len(groups)))
            cmap = {g: tab[i] for i, g in enumerate(groups)}
        for g in groups:
            d = df[df[color] == g]
            ax.scatter(d["PC1"], d["PC2"], s=42, alpha=0.8, color=cmap[g],
                       edgecolors="white", linewidths=0.4, zorder=2, label=str(g))
        if len(groups) <= 11:
            ax.legend(loc="best", frameon=False, fontsize=6, handletextpad=0.2,
                      borderpad=0.2, labelspacing=0.2)
    ax.set_title(title, fontsize=9, fontweight="bold")
    _style(ax)


def make_figure(name, cfg):
    fig, axes = plt.subplots(len(VARIANTS), 2, figsize=(8.5, 3.6*len(VARIANTS)))
    for r, v in enumerate(VARIANTS):
        emb_csv = os.path.join(cfg["root"], v, "embedding.csv")
        if not os.path.exists(emb_csv):
            for c in (0, 1): axes[r, c].axis("off")
            continue
        emb = pd.read_csv(emb_csv, index_col=0)
        batch, bio = build_labels(emb.index, cfg)
        df = emb[["PC1", "PC2"]].copy()
        df["__batch"] = batch.values; df["__bio"] = bio.values
        nb = df["__batch"].nunique()
        panel(axes[r, 0], df, "__batch",
              "modality" if cfg["multiomics"] else "categorical",
              f"{v}\nby {'modality' if cfg['multiomics'] else 'batch'} ({nb})",
              draw_lines=cfg["multiomics"])
        panel(axes[r, 1], df, "__bio", "categorical",
              f"{v}\nby {cfg['bio_name']}", draw_lines=False,
              continuous=cfg.get("bio_continuous", False))
    fig.suptitle(f"{name} — ablation embeddings (PC1 vs PC2)", fontsize=12, fontweight="bold")
    out = os.path.join(ABL, f"{name}_embeddings_batch_vs_biology.png")
    fig.tight_layout(rect=[0, 0, 1, 0.99]); fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[{name}] -> {out}")


DATASETS = {
    "COVID": dict(root=f"{ABL}/covid/covid_400", multiomics=False,
                  meta="/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv",
                  batch_col="batch", bio_col="sev.level", bio_name="severity"),
    "ENCODE": dict(root=f"{ABL}/multiomics/ENCODE", multiomics=True, meta_per_modality=True,
                   meta="/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
                   bio_col="tissue", bio_name="tissue"),
    "heart": dict(root=f"{ABL}/multiomics/heart", multiomics=True,
                  meta="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
                  bio_col="disease_state", bio_name="disease_state"),
    "retina": dict(root=f"{ABL}/multiomics/retina", multiomics=True,
                   meta=f"{ABL}/multiomics/retina/meta_built.csv",
                   bio_col="age", bio_name="developmental age", bio_continuous=True),
    "lutea": dict(root=f"{ABL}/multiomics/lutea", multiomics=True,
                  meta=f"{ABL}/multiomics/lutea/meta_built.csv",
                  bio_col="age", bio_name="developmental age", bio_continuous=True),
    "health_aging": dict(root=f"{ABL}/health_aging", multiomics=False,
                         meta="/dcs07/hongkai/data/harry/result/health_aging_PBMC/benchmark/meta_pbmc.csv",
                         batch_col="Batch", bio_col="Age", bio_name="age", bio_continuous=True),
}


def main():
    for name, cfg in DATASETS.items():
        try:
            make_figure(name, cfg)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
