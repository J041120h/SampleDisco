#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
import scanpy as sc


def add_raw_counts_layer_by_obs_names(
    preprocessed_h5ad_path,
    raw_h5ad_path,
    layer_name: str = "raw_counts",
    verbose: bool = True,
):
    """
    Add a raw-counts layer to a preprocessed AnnData by aligning cells
    and genes using obs_names / var_names.

    Assumes:
      - adata_pre.obs_names are a subset of (or equal to) adata_raw.obs_names
      - adata_pre.var_names are a subset of (or equal to) adata_raw.var_names
    """

    if verbose:
        print(f"[add_raw_counts_layer_by_obs_names] Reading preprocessed: {preprocessed_h5ad_path}")
        print(f"[add_raw_counts_layer_by_obs_names] Reading raw:          {raw_h5ad_path}")

    adata_pre = sc.read_h5ad(preprocessed_h5ad_path)
    adata_raw = sc.read_h5ad(raw_h5ad_path)

    # ------------------------------------------------------------------
    # 1) Check obs_names alignment (cells)
    # ------------------------------------------------------------------
    if not adata_pre.obs_names.is_unique:
        raise ValueError("preprocessed AnnData has non-unique obs_names; cannot align safely.")

    if not adata_raw.obs_names.is_unique:
        raise ValueError("raw AnnData has non-unique obs_names; cannot align safely.")

    # Cells present in pre that are missing in raw
    missing_cells = adata_pre.obs_names.difference(adata_raw.obs_names)
    if len(missing_cells) > 0:
        # Hard fail so you don't silently misalign anything
        n_show = min(10, len(missing_cells))
        raise ValueError(
            f"{len(missing_cells)} cells in preprocessed AnnData are not found in raw AnnData.\n"
            f"Examples: {list(missing_cells[:n_show])}"
        )

    if verbose:
        print(f"  Cells in preprocessed: {adata_pre.n_obs}")
        print(f"  Cells in raw:         {adata_raw.n_obs}")
        print("  All preprocessed cells are present in raw. Aligning in that order...")

    # Reorder raw AnnData rows to match preprocessed cells exactly
    adata_raw = adata_raw[adata_pre.obs_names, :]

    # ------------------------------------------------------------------
    # 2) Check var_names alignment (genes)
    # ------------------------------------------------------------------
    genes_pre = adata_pre.var_names
    genes_raw = adata_raw.var_names

    missing_genes = genes_pre.difference(genes_raw)
    if len(missing_genes) > 0:
        n_show = min(10, len(missing_genes))
        raise ValueError(
            f"{len(missing_genes)} genes in preprocessed AnnData are not found in raw AnnData.\n"
            f"Examples: {list(missing_genes[:n_show])}"
        )

    if verbose:
        print(f"  Genes in preprocessed: {len(genes_pre)}")
        print(f"  Genes in raw:          {len(genes_raw)}")
        print("  All preprocessed genes are present in raw. Aligning columns...")

    # Map gene order: pre → raw indices
    gene_to_raw = pd.Series(np.arange(len(genes_raw)), index=genes_raw)
    gene_idx = gene_to_raw.loc[genes_pre].values

    # ------------------------------------------------------------------
    # 3) Slice raw.X to [cells in pre order, genes in pre order]
    # ------------------------------------------------------------------
    raw_X = adata_raw.X
    if isinstance(raw_X, spmatrix):
        # For sparse matrices, rely on AnnData's own slicing (keeps sparsity)
        # Rows already aligned by obs; just align genes by label
        aligned_raw = adata_raw[:, genes_pre].X
    else:
        # Dense NumPy array – index by integer positions
        aligned_raw = raw_X[:, gene_idx]

    # ------------------------------------------------------------------
    # 4) Add as new layer and write back
    # ------------------------------------------------------------------
    adata_pre.layers[layer_name] = aligned_raw

    if verbose:
        print(f"  Added layer '{layer_name}' with shape {aligned_raw.shape} to adata_pre.")
        print(f"  Writing updated AnnData back to: {preprocessed_h5ad_path}")

    adata_pre.write(preprocessed_h5ad_path)

    if verbose:
        print("Done: added raw counts layer by obs_names / var_names alignment.")


def print_batch_sample_counts(h5ad_path, batch_col="batch"):
    """
    Load a pseudobulk AnnData (samples × genes) and print how many samples
    exist in each batch.

    Parameters
    ----------
    h5ad_path : str
        Path to pseudobulk_sample.h5ad (samples are rows).
    batch_col : str
        Column name in .obs containing batch labels.
    """
    print(f"Loading pseudobulk AnnData from: {h5ad_path}")
    adata = sc.read(h5ad_path)

    if batch_col not in adata.obs.columns:
        print(f"[ERROR] '{batch_col}' not found in adata.obs!")
        print(f"Available columns: {list(adata.obs.columns)}")
        return

    batch_counts = adata.obs[batch_col].value_counts()

    print("\n=== Batch → Sample Counts ===")
    for batch, count in batch_counts.items():
        print(f"{batch:15s} : {count} samples")

    print("\nTotal samples:", adata.n_obs)
    print("Total batches:", len(batch_counts))


if __name__ == "__main__":
    add_raw_counts_layer_by_obs_names(
        preprocessed_h5ad_path="/dcs07/hongkai/data/harry/result/processed_data/100_adata_cell.h5ad",
        raw_h5ad_path="/dcl01/hongkai/data/data/hjiang/Data/covid_data/Benchmark/count_data_subsample_100samples.h5ad",
        layer_name="raw_counts",
        verbose=True,
    )
    # print_batch_sample_counts("/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_50_sample/rna/pseudobulk/pseudobulk_sample.h5ad")
