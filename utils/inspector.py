import scanpy as sc
import numpy as np
from scipy.sparse import issparse

def summarize_h5ad(h5ad_path: str, n_examples: int = 10):
    """
    Summarize an AnnData .h5ad file by printing examples of cell names, obs, var,
    and inspecting .X (dtype, NaN/Inf, integer-like vs fractional, min/max, and example values).
    """
    try:
        print(f"🔍 Loading AnnData from: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path, backed=None)
        
        print("\n📦 Basic Info")
        print(f"  - Shape (cells × genes): {adata.n_obs} × {adata.n_vars}")
        print(f"  - Layers: {list(adata.layers.keys()) if hasattr(adata, 'layers') else 'None'}")
        print(f"  - obs columns: {list(adata.obs.columns)}")
        print(f"  - var columns: {list(adata.var.columns)}")

        # 🔎 Inspect .X
        print("\n🔎 Inspecting .X matrix:")
        X = adata.X

        if issparse(X):
            print(f"  - storage type: sparse ({type(X).__name__})")
            dtype = X.dtype
            data_array_for_checks = X.data
        else:
            print(f"  - storage type: dense ({type(X).__name__})")
            dtype = X.dtype
            data_array_for_checks = np.asarray(X)

        print(f"  - dtype: {dtype}")

        # Classify dtype using actual data content
        is_numeric = np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)
        if is_numeric:
            flat = data_array_for_checks.ravel()
            if flat.size == 0:
                print("  - value type: numeric (empty matrix, cannot inspect values)")
                is_integer_like = True
            else:
                max_check = min(100000, flat.size)
                sample = flat[:max_check]
                is_integer_like = np.allclose(sample, np.round(sample), atol=1e-8)

            if np.issubdtype(dtype, np.integer):
                msg = "integer"
            elif np.issubdtype(dtype, np.floating):
                msg = "float"
            elif np.issubdtype(dtype, np.bool_):
                msg = "bool"
            else:
                msg = "numeric (custom type)"

            if is_integer_like:
                msg += " — values appear integer-like (no fractional parts)"
            else:
                msg += " — fractional values detected"
            print(f"  - value type: {msg}")

        else:
            print("  - ⚠️ unsupported / non-numeric dtype (e.g. object/string)")

        # NaN / Inf checks
        if is_numeric:
            try:
                has_nan = bool(np.isnan(data_array_for_checks).any())
            except TypeError:
                has_nan = False
                print("  - ⚠️ np.isnan failed on this dtype; skipping NaN check")

            has_inf = False
            if np.issubdtype(dtype, np.floating):
                has_inf = bool(np.isinf(data_array_for_checks).any())

            print(f"  - contains NaN: {has_nan}")
            if np.issubdtype(dtype, np.floating):
                print(f"  - contains Inf: {has_inf}")
        else:
            print("  - Skipping NaN/Inf check due to non-numeric dtype.")

        # Min / Max values
        if is_numeric and data_array_for_checks.size > 0:
            try:
                min_val = float(np.nanmin(data_array_for_checks))
                max_val = float(np.nanmax(data_array_for_checks))
                print(f"  - min value: {min_val:.6g}")
                print(f"  - max value: {max_val:.6g}")
            except Exception as e:
                print(f"  - ⚠️ Could not compute min/max: {e}")

        # Example .X values
        n_rows = min(n_examples, adata.n_obs)
        n_cols = min(10, adata.n_vars)
        print(f"\n🧮 Example .X values (first {n_rows} cells × {n_cols} genes):")
        if issparse(X):
            X_sub = X[:n_rows, :n_cols].toarray()
        else:
            X_sub = np.asarray(X[:n_rows, :n_cols])
        print(X_sub)

        # Example cell names
        print("\n🧫 Example cell names:")
        for name in adata.obs_names[:n_examples]:
            print("  -", name)

        # Example obs
        print("\n📋 Example obs rows:")
        print(adata.obs.head(n_examples))

        # Example var
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
    summarize_h5ad(
        h5ad_path = "/dcl01/hongkai/data/data/hjiang/Data/long_covid/long_covid.h5ad"
    )
