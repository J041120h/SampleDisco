

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
import scanpy as sc
from utils.merge_sample_meta import merge_sample_metadata

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

import pandas as pd

from pathlib import Path
from typing import Union, Tuple

import anndata as ad
from anndata import AnnData

# assuming merge_sample_metadata is already defined / imported above


def split_h5ad_by_organ_part(
    h5ad_path: Union[str, Path],
    metadata_path: Union[str, Path],
    sample_column: str = "sample",
    organ_part_col: str = "organ_part",
    verbose: bool = True,
) -> Tuple[AnnData, AnnData]:
    """
    1) Load an h5ad file.
    2) Merge sample-level metadata using `merge_sample_metadata`.
    3) Split cells into two AnnData objects based on `organ_part`:
          - "macula lutea"      -> lutea.h5ad
          - "peripheral region of retina" -> retina.h5ad
    4) Save the two new h5ad files in the SAME directory as the input.

    Returns
    -------
    lutea_adata, retina_adata
    """
    h5ad_path = Path(h5ad_path)
    out_dir = h5ad_path.parent

    if verbose:
        print(f"📥 Loading AnnData from: {h5ad_path}")

    adata = ad.read_h5ad(h5ad_path)

    # --- merge sample metadata ---
    if verbose:
        print("🔗 Merging sample-level metadata...")
    adata = merge_sample_metadata(
        adata=adata,
        metadata_path=metadata_path,
        sample_column=sample_column,
        verbose=verbose,
    )

    # --- check organ_part column ---
    if organ_part_col not in adata.obs.columns:
        raise KeyError(
            f"Column '{organ_part_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    organ_series = (
        adata.obs[organ_part_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Define label mapping (case-insensitive)
    lutea_label = "macula lutea"
    retina_label = "peripheral region of retina"

    mask_lutea = organ_series == lutea_label
    mask_retina = organ_series == retina_label

    if verbose:
        print(f"🔍 Cells with '{lutea_label}': {mask_lutea.sum()}")
        print(f"🔍 Cells with '{retina_label}': {mask_retina.sum()}")

    if mask_lutea.sum() == 0 or mask_retina.sum() == 0:
        print("⚠️ Warning: one of the organ parts has zero cells after filtering.")

    # --- subset and copy ---
    lutea_adata = adata[mask_lutea].copy()
    retina_adata = adata[mask_retina].copy()

    # --- save to same directory, names 'lutea.h5ad' and 'retina.h5ad' ---
    lutea_path = out_dir / "lutea.h5ad"
    retina_path = out_dir / "retina.h5ad"

    if verbose:
        print(f"💾 Saving macula lutea subset to: {lutea_path}")
    lutea_adata.write_h5ad(lutea_path)

    if verbose:
        print(f"💾 Saving peripheral retina subset to: {retina_path}")
    retina_adata.write_h5ad(retina_path)

    return lutea_adata, retina_adata

if __name__ == "__main__":
    add_raw_counts_layer_by_obs_names(
        preprocessed_h5ad_path= '/dcs07/hongkai/data/harry/result/long_covid/rna/preprocess/adata_cell.h5ad',
        raw_h5ad_path= '/dcl01/hongkai/data/data/hjiang/Data/long_covid/long_covid.h5ad',
        layer_name="raw_counts",
        verbose=True,
    )
    # print_batch_sample_counts("/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_50_sample/rna/pseudobulk/pseudobulk_sample.h5ad")
    # split_h5ad_by_organ_part(
    # "/dcs07/hongkai/data/harry/result/multi_omics_eye/data/atac_raw.h5ad",
    # "//dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",  # or .csv / .xlsx etc.
    # sample_column="sample",          # or whatever your sample column is
    # )


