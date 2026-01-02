#!/usr/bin/env python3
import os
import scanpy as sc
import pandas as pd
from pathlib import Path

def export_dr_embeddings(h5ad_path, output_dir):
    """
    Read a .h5ad file and export X_DR_expression and X_DR_proportion to CSV if found.

    Parameters
    ----------
    h5ad_path : str or Path
        Path to the AnnData file (.h5ad)
    output_dir : str or Path
        Directory to save the CSV files
    """
    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    # ---- Helper to save cleanly ----
    def save_df(name, data):
        df = pd.DataFrame(data, index=adata.obs_names)
        out = output_dir / f"{name}.csv"
        df.to_csv(out)
        print(f"   ✓ Saved {name} → {out} (shape={df.shape})")

    # ---- X_DR_expression ----
    if "X_DR_expression" in adata.uns:
        save_df("X_DR_expression", adata.uns["X_DR_expression"])
    elif "X_DR_expression" in adata.obsm:
        save_df("X_DR_expression", adata.obsm["X_DR_expression"])
    else:
        print("   ⚠️ X_DR_expression not found")

    # ---- X_DR_proportion ----
    if "X_DR_proportion" in adata.uns:
        save_df("X_DR_proportion", adata.uns["X_DR_proportion"])
    elif "X_DR_proportion" in adata.obsm:
        save_df("X_DR_proportion", adata.obsm["X_DR_proportion"])
    else:
        print("   ⚠️ X_DR_proportion not found")


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    export_dr_embeddings(
        h5ad_path = "/dcl01/hongkai/data/data/hjiang/result/integration/pseudobulk/pseudobulk_sample.h5ad",
        output_dir = "/dcl01/hongkai/data/data/hjiang/result/integration/"
    )
