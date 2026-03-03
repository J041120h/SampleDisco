#!/usr/bin/env python3
"""
UMAP plot colored by cell_type, with text labels, using precomputed X_umap.

Inputs:
  - adata.obsm["X_umap"] must exist
  - adata.obs["cell_type"] must exist

Outputs:
  /dcs07/hongkai/data/harry/result/long_covid/analysis/clustering/resolution_0.25/
    - umap_cell_type.png
    - umap_cell_type.pdf

User can directly edit ADATA_PATH / OUT_DIR.
"""

import os
import sys
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

# =============================================================================
# USER-EDITABLE CONFIG
# =============================================================================
ADATA_PATH = "/dcs07/hongkai/data/harry/result/long_covid/analysis/preprocess/adata_sample.h5ad"
OUT_DIR = "/dcs07/hongkai/data/harry/result/long_covid/analysis/clustering/resolution_0.25"

COLOR_KEY = "cell_type"
DPI = 200
FIGSIZE = (10, 8)
LABEL_FONTSIZE = 8


def main() -> None:
    if not os.path.exists(ADATA_PATH):
        raise FileNotFoundError(f"Input not found: {ADATA_PATH}")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[INFO] Loading: {ADATA_PATH}")
    adata = sc.read_h5ad(ADATA_PATH)

    if "X_umap" not in adata.obsm_keys():
        raise KeyError(f"Missing adata.obsm['X_umap']. Available: {list(adata.obsm_keys())}")
    if COLOR_KEY not in adata.obs.columns:
        raise KeyError(f"Missing adata.obs['{COLOR_KEY}']. Available: {list(adata.obs.columns)}")

    # make sure categorical for stable legend
    if not hasattr(adata.obs[COLOR_KEY].dtype, "categories"):
        adata.obs[COLOR_KEY] = adata.obs[COLOR_KEY].astype("category")

    # Plot with scanpy, then add labels via matplotlib
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sc.pl.embedding(
        adata,
        basis="umap",
        color=COLOR_KEY,
        ax=ax,
        show=False,
        title=f"UMAP colored by {COLOR_KEY}",
        legend_loc="right margin",
        size=6,
    )

    xy = adata.obsm["X_umap"]
    groups = np.asarray(adata.obs[COLOR_KEY].astype(str).values)

    for g in np.unique(groups):
        m = groups == g
        if m.sum() == 0:
            continue
        x, y = float(xy[m, 0].mean()), float(xy[m, 1].mean())
        ax.text(
            x,
            y,
            g,
            fontsize=LABEL_FONTSIZE,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
        )

    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, "umap_cell_type.png")
    out_pdf = os.path.join(OUT_DIR, "umap_cell_type.pdf")
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"[DONE] Saved:\n  - {out_png}\n  - {out_pdf}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)