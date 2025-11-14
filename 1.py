import scanpy as sc
import pandas as pd
from pathlib import Path
import re

def fix_sample_ids(input_h5ad: str, sample_col: str = "sample"):
    """
    Load a .h5ad file, clean the sample column, and save the corrected file.

    Rules implemented:
    - If 'source_file' exists, extract sample ID from filename like:
        '1--104-M1_sample_filtered_feature_bc_matrix.h5'
        → '104-M1'
    - Otherwise, fall back to the existing sample column, which currently looks like:
        '1--104-M1' → extract '104-M1'

    Parameters
    ----------
    input_h5ad : str
        Path to input .h5ad
    output_h5ad : str
        Path to write modified .h5ad
    sample_col : str
        Name of the sample column to fix (default: 'sample')
    """
    output_h5ad = input_h5ad
    print(f"🔍 Loading AnnData: {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)

    if "source_file" in adata.obs.columns:
        print("📦 Fixing sample IDs using `source_file` column...")

        def extract_from_filename(x):
            # Example filename:
            # '1--104-M1_sample_filtered_feature_bc_matrix.h5'
            # Extract middle part before first underscore
            base = Path(x).stem
            main = base.split("_")[0]        # '1--104-M1'
            parts = main.split("-")
            # Expected format: ['1', '', '104', 'M1']
            # Keep last two parts: '104-M1'
            return "-".join(parts[-2:])

        adata.obs[sample_col] = adata.obs["source_file"].apply(extract_from_filename)

    else:
        print("📦 No `source_file` found. Fixing sample IDs using existing sample column...")

        def extract_from_sample(x):
            # Example: '1--104-M1'
            parts = x.split("-")
            return "-".join(parts[-2:])      # '104-M1'

        adata.obs[sample_col] = adata.obs[sample_col].astype(str).apply(extract_from_sample)

    print("✅ Example corrected sample IDs:")
    print(adata.obs[sample_col].head())

    print(f"💾 Saving corrected file to: {output_h5ad}")
    adata.write_h5ad(output_h5ad)

    print("🎉 Done.")


# Example usage:
fix_sample_ids("/dcl01/hongkai/data/data/hjiang/Data/long_covid/long_covid.h5ad")
