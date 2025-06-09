#!/usr/bin/env python3

import anndata as ad
import argparse

def main(adata_path):
    # Load AnnData
    adata = ad.read_h5ad(adata_path)

    # Print .var and .obs column names
    print("Columns in .var:")
    print(adata.var.columns.tolist())
    print("\nColumns in .obs:")
    print(adata.obs.columns.tolist())

if __name__ == "__main__":
    
    main()
