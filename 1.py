#!/usr/bin/env python3
import argparse
import anndata as ad
import pandas as pd
import numpy as np

def main():
    adata = ad.read_h5ad('/dcs07/hongkai/data/harry/result/Benchmark/covid_25_sample/rna/pseudobulk/pseudobulk_sample.h5ad')
    print(f"✅ Loaded AnnData: shape={adata.n_obs} cells × {adata.n_vars} features\n")

    # ---- OBS ----
    print("=== .obs (observations) ===")
    print(f"shape: {adata.obs.shape}")
    if not adata.obs.empty:
        print("columns:", list(adata.obs.columns))
        print("\n.obs head:")
        print(adata.obs.head())
    else:
        print("(empty)")
    print()

    # ---- UNS ----
    print("=== .uns (unstructured) ===")
    if len(adata.uns) == 0:
        print("(empty)\n")
    else:
        print("keys:", list(adata.uns.keys()))
        # Print lightweight summaries per key
        for k, v in adata.uns.items():
            # Keep it short to avoid dumping huge data
            if isinstance(v, (pd.DataFrame, pd.Series, np.ndarray, list, dict)):
                try:
                    if isinstance(v, pd.DataFrame):
                        print(f"  - {k}: DataFrame {v.shape} (columns={list(v.columns)[:10]}...)")
                    elif isinstance(v, pd.Series):
                        print(f"  - {k}: Series len={len(v)} (name={v.name})")
                    elif isinstance(v, np.ndarray):
                        print(f"  - {k}: ndarray shape={v.shape}, dtype={v.dtype}")
                    elif isinstance(v, list):
                        print(f"  - {k}: list len={len(v)} (first={v[:1]})")
                    elif isinstance(v, dict):
                        print(f"  - {k}: dict keys={list(v.keys())[:10]}...")
                except Exception:
                    print(f"  - {k}: (summary unavailable)")
            else:
                print(f"  - {k}: type={type(v).__name__}")
        print()

    # ---- VAR ----
    print("=== .var (variables/features) ===")
    print(f"shape: {adata.var.shape}")
    if not adata.var.empty:
        print("columns:", list(adata.var.columns))
        print("\n.var head:")
        print(adata.var.head())
    else:
        print("(empty)")
    print()

if __name__ == "__main__":
    main()
