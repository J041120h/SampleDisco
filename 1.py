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

import pandas as pd
import anndata as ad

def update_anndata_with_cell_meta(cell_meta, adata_path, index_col='cell_id', overwrite=True):
    """
    Update `.obs` in an AnnData object with external metadata.
    Drops or overwrites overlapping columns. Removes '_index' column if present.

    Parameters:
    - cell_meta: str or pd.DataFrame
    - adata_path: str
    - index_col: str — column in metadata that matches AnnData obs_names
    - overwrite: bool — if True, overwrite overlapping columns

    Returns:
    - None
    """
    # Load metadata
    if isinstance(cell_meta, str):
        if cell_meta.endswith(".csv"):
            meta_df = pd.read_csv(cell_meta, index_col=index_col)
        elif cell_meta.endswith(".tsv"):
            meta_df = pd.read_csv(cell_meta, sep='\t', index_col=index_col)
        else:
            raise ValueError("Unsupported file format.")
    elif isinstance(cell_meta, pd.DataFrame):
        meta_df = cell_meta.set_index(index_col) if index_col in cell_meta.columns else cell_meta
    else:
        raise TypeError("cell_meta must be a path or DataFrame")

    # Load AnnData
    adata = ad.read_h5ad(adata_path)

    # Subset metadata
    matching_meta = meta_df.loc[meta_df.index.intersection(adata.obs_names)]

    # Drop overlapping columns if overwriting
    overlapping_cols = adata.obs.columns.intersection(matching_meta.columns)
    if overwrite:
        adata.obs = adata.obs.drop(columns=overlapping_cols)

    # Join metadata
    adata.obs = adata.obs.join(matching_meta, how='left')

    # Drop invalid column if present
    if '_index' in adata.obs.columns:
        adata.obs.drop(columns=['_index'], inplace=True)

    # Convert object-type columns to string
    for col in adata.obs.select_dtypes(include='object').columns:
        adata.obs[col] = adata.obs[col].astype(str)

    # Save back to disk
    adata.write(adata_path, compression='gzip')



if __name__ == "__main__":
    # main("/users/hjiang/GenoDistance/result/harmony/pseudobulk_sample.h5ad")
    main("/users/hjiang/GenoDistance/result/integration/pseudobulk/pseudobulk_adata.h5ad")
