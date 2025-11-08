import scanpy as sc

def summarize_h5ad(h5ad_path: str, n_examples: int = 10):
    """
    Summarize an AnnData .h5ad file by printing examples of cell names, obs, and var entries.
    
    Parameters
    ----------
    h5ad_path : str
        Path to the .h5ad file.
    n_examples : int
        Number of examples to display for obs/var (default: 10).
    """
    try:
        print(f"🔍 Loading AnnData from: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path, backed=None)
        
        print("\n📦 Basic Info")
        print(f"  - Shape (cells × genes): {adata.n_obs} × {adata.n_vars}")
        print(f"  - Layers: {list(adata.layers.keys()) if hasattr(adata, 'layers') else 'None'}")
        print(f"  - obs columns: {list(adata.obs.columns)}")
        print(f"  - var columns: {list(adata.var.columns)}")

        # Example cell names
        print("\n🧫 Example cell names:")
        for name in adata.obs_names[:n_examples]:
            print("  -", name)

        # Example obs (metadata)
        print("\n📋 Example obs rows:")
        print(adata.obs.head(n_examples))

        # Example var (gene information)
        print("\n🧬 Example var rows:")
        print(adata.var.head(n_examples))

    except Exception as e:
        print(f"❌ Error reading {h5ad_path}: {e}")

#!/usr/bin/env python3
"""
transfer_obs_columns_simple.py

Directly edit PATH_A and PATH_B below to transfer missing .obs columns
between two .h5ad files. The script will:
  • Add any missing obs columns from A → B and B → A
  • Match by shared cell IDs (obs_names)
  • Preserve categorical dtypes
  • Write output to "<A>__obs_from_B.h5ad" and "<B>__obs_from_A.h5ad"
"""

import anndata as ad
import pandas as pd
import numpy as np
import sys

# ────────────────────────────────
# 🧩 EDIT THESE PATHS
PATH_A = '/dcl01/hongkai/data/data/hjiang/Data/paired/atac/all.h5ad'
PATH_B = '/dcl01/hongkai/data/data/hjiang/Data/paired/rna/all.h5ad'
# ────────────────────────────────


def transfer_columns(ad_src, ad_dst, overwrite=False):
    """Transfer missing .obs columns from ad_src → ad_dst for shared cells."""
    common_cells = ad_src.obs.index.intersection(ad_dst.obs.index)
    if len(common_cells) == 0:
        print("[WARN] No overlapping cells; skipping.")
        return ad_dst, []

    src_cols = set(ad_src.obs.columns)
    dst_cols = set(ad_dst.obs.columns)
    missing_cols = src_cols - dst_cols
    updated_cols = []

    for col in sorted(missing_cols):
        src_series = ad_src.obs[col]
        if pd.api.types.is_categorical_dtype(src_series):
            cat = src_series.cat
            empty = pd.Categorical([np.nan] * ad_dst.n_obs, categories=cat.categories)
            ad_dst.obs[col] = empty
            ad_dst.obs.loc[common_cells, col] = src_series.loc[common_cells].astype("category")
            ad_dst.obs[col] = ad_dst.obs[col].astype(pd.CategoricalDtype(categories=cat.categories))
        else:
            ad_dst.obs[col] = np.nan
            ad_dst.obs.loc[common_cells, col] = src_series.loc[common_cells].values
        updated_cols.append(col)

    return ad_dst, updated_cols


def main():
    print(f"📂 Loading A: {PATH_A}")
    A = ad.read_h5ad(PATH_A)
    print(f"📂 Loading B: {PATH_B}")
    B = ad.read_h5ad(PATH_B)

    print("🔍 Checking missing columns...")
    _, missing_in_A = set(B.obs.columns), set(A.obs.columns)
    _, missing_in_B = set(A.obs.columns), set(B.obs.columns)

    # # B → A
    # print("⬅️  Transferring from B → A")
    # A_updated, cols_A = transfer_columns(B, A)
    # if cols_A:
    #     out_a = f"{PATH_A}__obs_from_B.h5ad"
    #     A_updated.write_h5ad(out_a)
    #     print(f"💾 Saved updated A → {out_a}")
    # else:
    #     print("✅ No new columns added to A")

    # A → B
    print("➡️  Transferring from A → B")
    B_updated, cols_B = transfer_columns(A, B)
    if cols_B:
        out_b = PATH_B
        B_updated.write_h5ad(out_b)
        print(f"💾 Saved updated B → {out_b}")
    else:
        print("✅ No new columns added to B")

    print("🎉 Done.")

if __name__ == "__main__":
    # main()
    summarize_h5ad(h5ad_path = '/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_400_sample/rna/pseudobulk/pseudobulk_sample.h5ad')