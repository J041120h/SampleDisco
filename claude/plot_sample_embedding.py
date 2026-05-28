"""
Sample-embedding pair plots — round1 (Batch only) vs round5 (Batch+AgeGroup).

Mirrors style of round1_batch/autotune_alpha_age/sample_embedding/figures/pairs_*.png:
6-panel grid (5 PC pairs + 1 legend) colored by a metadata variable.

Outputs
-------
round5_batch_age/sample_embedding/figures/pairs_{sex,age,age_group,batch}.png
/users/hjiang/GenoDistance/figure/health_aging_PBMC/sample_embedding/round1_vs_round5_sex.png
"""

import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROUND1_EMB = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/autotune_alpha_age/sample_embedding/sample_embedding.csv"
ROUND5_EMB = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round5_batch_age/sample_embedding/sample_embedding/sample_embedding.csv"
ROUND6_EMB = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round6_multicov_batch_age/sample_embedding/sample_embedding/sample_embedding.csv"
PREPROC_H5AD = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/preprocess/adata_preprocessed.h5ad"

ROUND5_FIG_DIR = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round5_batch_age/sample_embedding/figures"
ROUND6_FIG_DIR = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round6_multicov_batch_age/sample_embedding/figures"
COMPARE_DIR    = "/users/hjiang/GenoDistance/figure/health_aging_PBMC/sample_embedding"

SEX_COLORS = {"Female": "#D7462F", "Male": "#F28E2D"}
AGEGRP_COLORS = {"A": "#440154", "B": "#3B528B", "C": "#21908C",
                 "D": "#5DC863", "E": "#FDE725"}


def load_sample_meta(h5_path):
    a = ad.read_h5ad(h5_path, backed="r")
    obs = a.obs[["Tube_id", "Sex", "Age", "Age_group", "Batch"]].copy()
    obs["Tube_id"] = obs["Tube_id"].astype(str)
    return obs.drop_duplicates(subset="Tube_id").set_index("Tube_id")


def load_emb(p):
    d = pd.read_csv(p, index_col=0)
    d.index = d.index.astype(str)
    return d


def _align(emb, meta):
    common = sorted(set(emb.index) & set(meta.index))
    return emb.loc[common].values, meta.loc[common].copy()


def pairs_plot(arr, meta, color_col, title, out_path,
               palette=None, cmap=None, dpi=200):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor="white")
    axes = axes.flatten()
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    if palette is not None:
        cats_raw = meta[color_col].astype(str).values
        uniq = sorted(set(cats_raw))
        col_map = {c: palette.get(c, plt.cm.tab20((i % 20) / 20)) for i, c in enumerate(uniq)}
        cols = np.array([col_map[c] for c in cats_raw])
        for ax, (i, j) in zip(axes[:5], pairs):
            ax.scatter(arr[:, i], arr[:, j], c=cols, s=20, alpha=0.85, edgecolors="none")
            ax.set_xlabel(f"PC{i+1}", fontsize=10)
            ax.set_ylabel(f"PC{j+1}", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.axhline(0, color="#CCCCCC", lw=0.5, zorder=0)
            ax.axvline(0, color="#CCCCCC", lw=0.5, zorder=0)
        handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=col_map[c], markersize=8, label=c)
                   for c in uniq]
        axes[5].axis("off")
        axes[5].legend(handles=handles, loc="center", title=color_col,
                       fontsize=9, title_fontsize=11, frameon=False, ncol=2 if len(uniq) > 6 else 1)
    else:
        vals = pd.to_numeric(meta[color_col], errors="coerce").values
        for ax, (i, j) in zip(axes[:5], pairs):
            sc = ax.scatter(arr[:, i], arr[:, j], c=vals, cmap=cmap or "viridis",
                            s=20, alpha=0.85, edgecolors="none")
            ax.set_xlabel(f"PC{i+1}", fontsize=10)
            ax.set_ylabel(f"PC{j+1}", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.axhline(0, color="#CCCCCC", lw=0.5, zorder=0)
            ax.axvline(0, color="#CCCCCC", lw=0.5, zorder=0)
        axes[5].axis("off")
        cax = fig.add_axes([0.79, 0.15, 0.018, 0.3])
        fig.colorbar(sc, cax=cax, label=color_col)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path}")


def compare_sex(emb1, emb5, emb6, meta, out_path, dpi=200):
    """Side-by-side: round1 (Batch only) | round5 (joint) | round6 (multi-cov)."""
    arr1, m1 = _align(emb1, meta)
    arr5, m5 = _align(emb5, meta)
    arr6, m6 = _align(emb6, meta)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")
    panels = [
        (axes[0], arr1, m1, 8, 9, "round1\n(Batch only)",                                 0.54),
        (axes[1], arr5, m5, 6, 7, "round5\n(joint Batch:AgeGroup workaround)",            0.34),
        (axes[2], arr6, m6, 6, 7, "round6\n(multi-cov: Batch + Age_group regressed)",     0.53),
    ]
    for ax, arr, meta_sub, dim_x, dim_y, label, r2 in panels:
        cats = meta_sub["Sex"].astype(str).values
        cols = np.array([SEX_COLORS.get(c, "#7F7F7F") for c in cats])
        ax.scatter(arr[:, dim_x], arr[:, dim_y], c=cols, s=26,
                   alpha=0.85, edgecolors="white", linewidths=0.4)
        ax.set_xlabel(f"PC{dim_x+1}", fontsize=11)
        ax.set_ylabel(f"PC{dim_y+1}", fontsize=11)
        ax.set_title(f"{label}\nSex max R²={r2:.2f} at PC{dim_y+1}",
                     fontsize=11, fontweight="bold")
        ax.axhline(0, color="#CCCCCC", lw=0.5)
        ax.axvline(0, color="#CCCCCC", lw=0.5)

    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=SEX_COLORS[c], markersize=10, label=c)
               for c in ["Female", "Male"]]
    fig.legend(handles=handles, loc="center right", title="Sex", fontsize=10,
               title_fontsize=11, frameon=False, bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Sex separation across rounds (Batch + Age_group regression strategies)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.95, 0.92])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    os.makedirs(ROUND5_FIG_DIR, exist_ok=True)
    os.makedirs(ROUND6_FIG_DIR, exist_ok=True)
    os.makedirs(COMPARE_DIR, exist_ok=True)

    print("Loading metadata ...")
    meta = load_sample_meta(PREPROC_H5AD)
    print(f"  meta samples: {len(meta)}")

    print("Loading embeddings ...")
    e1 = load_emb(ROUND1_EMB)
    e5 = load_emb(ROUND5_EMB)
    e6 = load_emb(ROUND6_EMB)
    print(f"  round1: {e1.shape}  round5: {e5.shape}  round6: {e6.shape}")

    print("\nRound5 pair plots:")
    arr5, m5 = _align(e5, meta)
    pairs_plot(arr5, m5, "Sex", "Sex — round5 (joint Batch:AgeGroup)",
               os.path.join(ROUND5_FIG_DIR, "pairs_sex.png"), palette=SEX_COLORS)
    pairs_plot(arr5, m5, "Age", "Age — round5",
               os.path.join(ROUND5_FIG_DIR, "pairs_age.png"), cmap="viridis")
    pairs_plot(arr5, m5, "Age_group", "Age_group — round5",
               os.path.join(ROUND5_FIG_DIR, "pairs_age_group.png"), palette=AGEGRP_COLORS)
    pairs_plot(arr5, m5, "Batch", "Batch — round5 (14 sequencing batches)",
               os.path.join(ROUND5_FIG_DIR, "pairs_batch.png"), palette={})

    print("\nRound6 pair plots:")
    arr6, m6 = _align(e6, meta)
    pairs_plot(arr6, m6, "Sex", "Sex — round6 (multi-cov Batch + Age_group)",
               os.path.join(ROUND6_FIG_DIR, "pairs_sex.png"), palette=SEX_COLORS)
    pairs_plot(arr6, m6, "Age", "Age — round6",
               os.path.join(ROUND6_FIG_DIR, "pairs_age.png"), cmap="viridis")
    pairs_plot(arr6, m6, "Age_group", "Age_group — round6",
               os.path.join(ROUND6_FIG_DIR, "pairs_age_group.png"), palette=AGEGRP_COLORS)
    pairs_plot(arr6, m6, "Batch", "Batch — round6 (14 sequencing batches)",
               os.path.join(ROUND6_FIG_DIR, "pairs_batch.png"), palette={})

    print("\n3-way Sex comparison:")
    compare_sex(e1, e5, e6, meta,
                os.path.join(COMPARE_DIR, "round1_vs_round5_vs_round6_sex.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
