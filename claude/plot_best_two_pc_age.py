"""
For each autotuned run, pick the 2 PCs with the strongest |Pearson r| against
Age (which equals their CCA score with a 1-D grouping variable), and make a
2-D scatter of those PCs colored by Age.

Outputs:
  round1_batch/autotune_alpha_age/sample_embedding/figures/best2_pcs_vs_age.png
  round3_filename/autotune_alpha_age/sample_embedding/figures/best2_pcs_vs_age.png
  combined_best2_pcs_vs_age.png  (top-level, side-by-side)
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
    ("R1 (cell-level=Batch, α=1.15)",   ROOT / "round1_batch/autotune_alpha_age"),
    ("R3 (cell-level=File_name, α=1.23)", ROOT / "round3_filename/autotune_alpha_age"),
]


def best_two_pcs(var_df: pd.DataFrame) -> list[tuple[str, float, float]]:
    """Return list [(PC_name, pearson_r, r2), (PC_name, pearson_r, r2)] for the
    two Age-vs-PC rows with the highest |Pearson r|."""
    age = var_df[(var_df["variable"] == "Age") &
                 (var_df["component"].str.startswith("PC"))].copy()
    age["absR"] = age["pearson_r"].abs()
    top = age.sort_values("absR", ascending=False).head(2)
    return [(r["component"], float(r["pearson_r"]), float(r["r2"]))
            for _, r in top.iterrows()]


def scatter_two_pc(V, pc_a, pc_b, age, run_label, info_a, info_b, out_path):
    """V: (n_samples, n_pcs); pc_a, pc_b: 1-based PC numbers. info_a, info_b: (r, r2)."""
    ja, jb = pc_a - 1, pc_b - 1
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(V[:, ja], V[:, jb], c=age, cmap="viridis",
                    s=42, alpha=0.9, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Age (years)")
    ra, r2a = info_a; rb, r2b = info_b
    ax.set_xlabel(f"PC{pc_a}   (vs Age: r={ra:+.3f}  R²={r2a:.3f})")
    ax.set_ylabel(f"PC{pc_b}   (vs Age: r={rb:+.3f}  R²={r2b:.3f})")
    ax.set_title(f"{run_label}\nTwo PCs with highest individual Age correlation",
                 fontsize=11)
    ax.axhline(0, lw=0.4, color="0.7")
    ax.axvline(0, lw=0.4, color="0.7")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# load metadata once
cell_meta = pd.read_csv(META_CSV, low_memory=False,
                        usecols=["Tube_id", "Age"])
sm_full = cell_meta.drop_duplicates("Tube_id").set_index("Tube_id")


# combined side-by-side figure
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))

for ax, (label, run_dir) in zip(axes, RUNS):
    emb = pd.read_csv(run_dir / "sample_embedding/sample_embedding.csv", index_col=0)
    var_df = pd.read_csv(run_dir / "sample_association/variance_explained_sample.csv")
    age = sm_full["Age"].reindex(emb.index).astype(float).values

    top2 = best_two_pcs(var_df)
    print(f"{label}: best 2 PCs = {[t[0] for t in top2]}")

    pc_a, r_a, r2_a = top2[0]
    pc_b, r_b, r2_b = top2[1]
    ja = int(pc_a.replace("PC", "")) - 1
    jb = int(pc_b.replace("PC", "")) - 1

    # per-run standalone plot
    scatter_two_pc(emb.values, ja + 1, jb + 1, age, label,
                   (r_a, r2_a), (r_b, r2_b),
                   run_dir / "sample_embedding/figures/best2_pcs_vs_age.png")

    # populate side-by-side
    sc = ax.scatter(emb.values[:, ja], emb.values[:, jb], c=age,
                    cmap="viridis", s=42, alpha=0.9, linewidths=0)
    ax.set_xlabel(f"{pc_a}  (r={r_a:+.3f}, R²={r2_a:.3f})")
    ax.set_ylabel(f"{pc_b}  (r={r_b:+.3f}, R²={r2_b:.3f})")
    ax.set_title(label, fontsize=11)
    ax.axhline(0, lw=0.4, color="0.7")
    ax.axvline(0, lw=0.4, color="0.7")

cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, label="Age (years)")
fig.suptitle("Top 2 Age-correlated PCs per run (colored by Age)",
             fontsize=13)
out_combined = ROOT / "combined_best2_pcs_vs_age.png"
fig.savefig(out_combined, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nCombined: {out_combined}")
