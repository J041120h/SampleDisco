"""
Visualize both autotuned sample embeddings.

Inputs:
  /dcs07/.../run1_autotune_batch/rna/sample_embedding/sample_embedding.csv
  /dcs07/.../run3_autotune_filename/rna/sample_embedding/sample_embedding.csv

Outputs (per run, under rna/sample_embedding/figures/):
  pairs_age.png            PC1-2 / 3-4 / 5-6 / 7-8 / 9-10, Age viridis
  pairs_age_group.png      same, Age_group A-E
  pairs_sex.png            same, Sex
  pairs_batch.png          same, Batch (14)
  pairs_file_name.png      same, File_name (lane)
  pc{j}_vs_age.png         best 3 Age PCs (highest |Pearson r|) scatter vs Age
  variance_heatmap.png     R^2 of every obs col x PC1..10
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/dcs07/hongkai/data/harry/result/health_aging_PBMC")
META_CSV = ROOT / "data/all_pbmcs/all_pbmcs_metadata.csv"

RUNS = [
    ("run1_autotune_batch", ROOT / "run1_autotune_batch"),
    ("run3_autotune_filename", ROOT / "run3_autotune_filename"),
]


def scatter_matrix(values, color, palette_name, title, out, is_continuous):
    pcs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    point_kw = dict(s=22, alpha=0.85, linewidths=0)
    if is_continuous:
        cmap = plt.get_cmap(palette_name)
        norm = plt.Normalize(vmin=np.nanmin(color), vmax=np.nanmax(color))
        for ax, (a, b) in zip(axes, pcs):
            sc = ax.scatter(values[:, a], values[:, b], c=color,
                            cmap=cmap, norm=norm, **point_kw)
            ax.set_xlabel(f"PC{a+1}"); ax.set_ylabel(f"PC{b+1}")
            ax.axhline(0, lw=0.4, color="0.7"); ax.axvline(0, lw=0.4, color="0.7")
        axes[-1].axis("off")
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        plt.colorbar(sc, cax=cax, label=title)
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=[0, 0, 0.91, 0.95])
    else:
        cats = sorted(pd.Series(color).astype(str).unique())
        cmap = plt.get_cmap(palette_name, max(3, len(cats)))
        color_lookup = {c: cmap(i) for i, c in enumerate(cats)}
        for ax, (a, b) in zip(axes, pcs):
            for c in cats:
                m = np.array([str(v) == c for v in color])
                ax.scatter(values[m, a], values[m, b], color=color_lookup[c],
                           label=c, **point_kw)
            ax.set_xlabel(f"PC{a+1}"); ax.set_ylabel(f"PC{b+1}")
            ax.axhline(0, lw=0.4, color="0.7"); ax.axvline(0, lw=0.4, color="0.7")
        axes[-1].axis("off")
        handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                              color=color_lookup[c], label=c, markersize=7)
                   for c in cats]
        ncol = 1 if len(cats) <= 14 else 4
        axes[-1].legend(handles=handles, title=title, loc="center left",
                        frameon=False, fontsize=7, ncol=ncol)
        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def single_age(x, y, out, title):
    from scipy.stats import pearsonr, spearmanr
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sc = ax.scatter(x, y, c=x, cmap="viridis", s=28, alpha=0.85, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Age")
    r_p, _ = pearsonr(x, y); r_s, _ = spearmanr(x, y)
    ax.set_title(f"{title}\nPearson r={r_p:+.3f}  Spearman r={r_s:+.3f}", fontsize=11)
    ax.set_xlabel("Age (years)")
    ax.axhline(0, lw=0.4, color="0.7"); ax.axvline(0, lw=0.4, color="0.7")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def variance_heatmap(var_df, out):
    pivot = (var_df.pivot_table(index="variable", columns="component",
                                values="r2", aggfunc="first")
                    .reindex(columns=[f"PC{i+1}" for i in range(10)]))
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(pivot) + 1))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma", vmin=0,
                   vmax=min(1.0, np.nanmax(pivot.values)))
    ax.set_xticks(range(10)); ax.set_xticklabels([f"PC{i+1}" for i in range(10)])
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v) and v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=("white" if v < 0.5 else "black"), fontsize=7)
    plt.colorbar(im, ax=ax, label="R^2")
    ax.set_title(f"Variance explained per PC")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


print("loading per-cell metadata once and reducing to per-sample table...")
cell_meta = pd.read_csv(META_CSV, low_memory=False,
                        usecols=["Tube_id", "Donor_id", "Age", "Age_group",
                                 "Sex", "Batch", "File_name"])
sample_meta_full = (cell_meta.drop_duplicates("Tube_id")
                             .set_index("Tube_id"))

for label, run_dir in RUNS:
    print(f"\n=== {label} ===")
    emb = pd.read_csv(run_dir / "rna/sample_embedding/sample_embedding.csv",
                      index_col=0)
    print(f"  embedding shape = {emb.shape}")
    sm = sample_meta_full.reindex(emb.index)
    print(f"  per-sample meta aligned: missing rows = {sm.isna().any(axis=1).sum()}")
    var_df = pd.read_csv(run_dir / "rna/sample_association/variance_explained_sample.csv")

    fig_dir = run_dir / "rna/sample_embedding/figures"
    fig_dir.mkdir(exist_ok=True)

    V = emb.values
    age = sm["Age"].astype(float).values

    print("  PC scatter matrices...")
    scatter_matrix(V, age, "viridis", "Age (years)",
                   fig_dir / "pairs_age.png", is_continuous=True)
    scatter_matrix(V, sm["Age_group"].astype(str).values, "tab10",
                   "Age_group", fig_dir / "pairs_age_group.png", False)
    scatter_matrix(V, sm["Sex"].astype(str).values, "Set1",
                   "Sex", fig_dir / "pairs_sex.png", False)
    scatter_matrix(V, sm["Batch"].astype(str).values, "tab20",
                   f"Batch ({sm['Batch'].nunique()})",
                   fig_dir / "pairs_batch.png", False)
    scatter_matrix(V, sm["File_name"].astype(str).values, "hsv",
                   f"File_name ({sm['File_name'].nunique()})",
                   fig_dir / "pairs_file_name.png", False)

    age_rows = var_df[(var_df["variable"] == "Age") &
                      (var_df["component"].str.startswith("PC"))].copy()
    age_rows = age_rows.assign(absR=age_rows["pearson_r"].abs())
    top_pcs = (age_rows.sort_values("absR", ascending=False)
                       .head(3)["component"].tolist())
    print(f"  top 3 Age PCs by |pearson r| = {top_pcs}")
    for pc_name in top_pcs:
        j = int(pc_name.replace("PC", "")) - 1
        single_age(age, V[:, j], fig_dir / f"{pc_name.lower()}_vs_age.png",
                   f"{label}: {pc_name} vs Age")

    print("  variance heatmap...")
    variance_heatmap(var_df, fig_dir / "variance_heatmap.png")
    print(f"  -> figures in {fig_dir}")

print("\nAll figures done.")
