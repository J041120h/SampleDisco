"""
KMeans (K=2) on PC8 ONLY of the round6 multi-covariate sample embedding.

Hypothesis: round6's PC8 carries ~all of the Sex variance (R²=0.53). Clustering
on PC8 alone should recover Sex with near-perfect concordance (vs the failed
attempt on the full 10-d embedding, which split on PC1 = batch/composition).

Outputs (under <result_dir>/):
  - kmeans_pc8_clusters.csv      sample, cluster, true_sex
  - sex_concordance.json         ARI / NMI / purity / accuracy
  - pc8_distribution.png         PC8 strip-plot colored by Sex + cluster boundary
  - pc7_vs_pc8.png               2-D scatter (PC7,PC8) colored by cluster vs true Sex side-by-side
"""

import json
import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

EMB_CSV = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round6_multicov_batch_age/sample_embedding/sample_embedding/sample_embedding.csv"
PREPROC_H5AD = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/preprocess/adata_preprocessed.h5ad"
OUT_DIR = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round7_kmeans_sex/pc8_only"

PC_FOR_GROUPING = "PC8"
SEX_COLORS = {"Female": "#D7462F", "Male": "#F28E2D"}
CLUSTER_COLORS = {"0": "#1F77B4", "1": "#FF7F0E"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Load ----
    emb = pd.read_csv(EMB_CSV, index_col=0)
    emb.index = emb.index.astype(str)
    print(f"embedding: {emb.shape}")

    a = ad.read_h5ad(PREPROC_H5AD, backed="r")
    meta = a.obs[["Tube_id", "Sex", "Age", "Age_group", "Batch"]].copy()
    meta["Tube_id"] = meta["Tube_id"].astype(str)
    meta = meta.drop_duplicates("Tube_id").set_index("Tube_id")
    common = sorted(set(emb.index) & set(meta.index))
    emb = emb.loc[common]
    meta = meta.loc[common]
    n = len(common)
    print(f"aligned: {n} samples")

    # ---- KMeans on PC8 only ----
    x = emb[[PC_FOR_GROUPING]].values  # (n, 1)
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(x)
    labels = km.labels_.astype(int)
    centers = km.cluster_centers_.flatten()
    print(f"\nKMeans on {PC_FOR_GROUPING}: centers = {centers.tolist()}")

    # ---- Concordance vs true Sex ----
    sex = meta["Sex"].astype(str).values
    ari = adjusted_rand_score(sex, labels)
    nmi = normalized_mutual_info_score(sex, labels)
    crosstab = pd.crosstab(pd.Series(labels, name="cluster"),
                           pd.Series(sex, name="Sex"))
    print("\nCross-tab cluster vs Sex:")
    print(crosstab.to_string())

    # Best mapping
    cluster_to_sex = crosstab.idxmax(axis=1).to_dict()
    pred_sex = np.array([cluster_to_sex[c] for c in labels])
    acc = float((pred_sex == sex).mean())
    purity = float(crosstab.max(axis=1).sum() / n)

    # Per-cluster sex composition
    per_cluster = {}
    for c in [0, 1]:
        m = labels == c
        row = crosstab.loc[c]
        per_cluster[f"cluster_{c}"] = {
            "n": int(m.sum()),
            "n_Female": int(row.get("Female", 0)),
            "n_Male": int(row.get("Male", 0)),
            "fraction_Female": float(row.get("Female", 0) / max(m.sum(), 1)),
            "majority_sex": cluster_to_sex[c],
            "center_PC8": float(centers[c]),
        }

    summary = {
        "n_samples": n,
        "PC_used": PC_FOR_GROUPING,
        "k": 2,
        "ARI": float(ari),
        "NMI": float(nmi),
        "purity": purity,
        "accuracy_majority_map": acc,
        "cluster_to_sex": {str(k): v for k, v in cluster_to_sex.items()},
        "per_cluster": per_cluster,
        "sex_baseline_male_frac": float((sex == "Male").mean()),
    }
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    # ---- Save labels CSV ----
    out_csv = os.path.join(OUT_DIR, "kmeans_pc8_clusters.csv")
    pd.DataFrame({
        "sample": common,
        "cluster_pc8": labels,
        "true_sex": sex,
        "PC8": emb[PC_FOR_GROUPING].values,
    }).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # ---- Save JSON summary ----
    out_json = os.path.join(OUT_DIR, "sex_concordance.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")

    # ---- Plot 1: PC8 strip plot colored by Sex ----
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    y_jitter = np.random.RandomState(0).uniform(-0.15, 0.15, n)
    for sex_label in ["Male", "Female"]:
        m = sex == sex_label
        ax.scatter(emb.loc[m, PC_FOR_GROUPING], y_jitter[m],
                   c=SEX_COLORS[sex_label], s=22, alpha=0.85,
                   edgecolors="white", linewidths=0.4, label=sex_label)
    # cluster boundary = midpoint of centers
    mid = float((centers[0] + centers[1]) / 2.0)
    ax.axvline(mid, color="#333333", ls="--", lw=1.5, label=f"KMeans split @ {mid:.3f}")
    for c, ctr in enumerate(centers):
        ax.axvline(ctr, color=CLUSTER_COLORS[str(c)], ls=":", lw=1.2, alpha=0.6)
    ax.set_xlabel(f"{PC_FOR_GROUPING}", fontsize=12)
    ax.set_yticks([])
    ax.set_title(f"{PC_FOR_GROUPING} distribution colored by true Sex (n={n})\n"
                 f"KMeans (K=2) on {PC_FOR_GROUPING} | accuracy={acc:.3f} | ARI={ari:.3f}",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "pc8_distribution.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    # ---- Plot 2: PC7 vs PC8 — left=predicted cluster, right=true Sex ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    # left: clusters
    for c in [0, 1]:
        m = labels == c
        axes[0].scatter(emb.loc[m, "PC7"], emb.loc[m, "PC8"],
                        c=CLUSTER_COLORS[str(c)], s=24, alpha=0.85,
                        edgecolors="white", linewidths=0.4,
                        label=f"cluster {c} ({per_cluster[f'cluster_{c}']['majority_sex']})")
    axes[0].set_xlabel("PC7", fontsize=11)
    axes[0].set_ylabel("PC8", fontsize=11)
    axes[0].set_title("KMeans cluster (on PC8)", fontsize=11, fontweight="bold")
    axes[0].axhline(0, color="#CCCCCC", lw=0.5); axes[0].axvline(0, color="#CCCCCC", lw=0.5)
    axes[0].legend(loc="best", fontsize=9, frameon=False)

    # right: true Sex
    for sex_label in ["Male", "Female"]:
        m = sex == sex_label
        axes[1].scatter(emb.loc[m, "PC7"], emb.loc[m, "PC8"],
                        c=SEX_COLORS[sex_label], s=24, alpha=0.85,
                        edgecolors="white", linewidths=0.4, label=sex_label)
    axes[1].set_xlabel("PC7", fontsize=11)
    axes[1].set_ylabel("PC8", fontsize=11)
    axes[1].set_title("True Sex", fontsize=11, fontweight="bold")
    axes[1].axhline(0, color="#CCCCCC", lw=0.5); axes[1].axvline(0, color="#CCCCCC", lw=0.5)
    axes[1].legend(loc="best", fontsize=9, frameon=False)

    fig.suptitle(f"round6 PC7 vs PC8: KMeans (K=2 on PC8) vs true Sex  | accuracy={acc:.3f}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png2 = os.path.join(OUT_DIR, "pc7_vs_pc8_cluster_vs_sex.png")
    fig.savefig(out_png2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
