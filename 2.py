#!/usr/bin/env python3

import scanpy as sc
import sys


def print_unique_samples(anndata_path: str) -> None:
    """
    Print all unique values in adata.obs['sample'].
    """
    adata = sc.read_h5ad(anndata_path)

    if "sample" not in adata.obs.columns:
        raise ValueError("Column 'sample' not found in adata.obs")

    unique_samples = adata.obs["sample"].unique()

    print(f"Number of unique samples: {len(unique_samples)}")
    print("Unique sample values:")
    for s in sorted(unique_samples):
        print(s)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python print_unique_samples.py <anndata_path.h5ad>")

    print_unique_samples(sys.argv[1])
