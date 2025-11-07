from __future__ import annotations
import os
from typing import Optional, Tuple, Sequence
import pandas as pd
import anndata as ad


def subset_h5ad_by_batch_samples(
    csv_path: str,
    h5ad_path: str,
    out_path: str,
    *,
    batch_name: str = "Su",
    csv_sample_col: str = "sample",
    csv_batch_col: str = "batch",
    ad_sample_col: str = "sample",
) -> Tuple[str, int, int]:
    """
    Subset an .h5ad dataset to only cells originating from samples whose `csv_batch_col == batch_name`
    (default "Su"), preserving all other information. Saves the subset to `out_path`.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing at least the columns [csv_sample_col, csv_batch_col].
    h5ad_path : str
        Path to the input .h5ad file to be subsetted.
    out_path : str
        Path to save the subset .h5ad file.
    batch_name : str, optional
        The batch name to filter samples by (default: "Su").
    csv_sample_col : str, optional
        Column in the CSV containing sample IDs (default: "sample").
    csv_batch_col : str, optional
        Column in the CSV containing batch labels (default: "batch").
    ad_sample_col : str, optional
        Column in `adata.obs` that contains the sample ID for each cell (default: "sample").

    Returns
    -------
    (out_path, n_cells, n_vars) : Tuple[str, int, int]
        The path written, number of cells retained, and number of variables.

    Raises
    ------
    FileNotFoundError
        If csv_path or h5ad_path do not exist.
    ValueError
        If required columns are missing, or no matching samples/cells are found.
    """
    # --- Basic checks ---
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isfile(h5ad_path):
        raise FileNotFoundError(f"H5AD not found: {h5ad_path}")

    # --- Load CSV and validate columns ---
    meta = pd.read_csv(csv_path)
    for col in (csv_sample_col, csv_batch_col):
        if col not in meta.columns:
            raise ValueError(
                f"CSV missing required column '{col}'. "
                f"Available columns: {list(meta.columns)}"
            )

    # --- Collect sample IDs from the desired batch ---
    su_samples: Sequence[str] = (
        meta.loc[meta[csv_batch_col].astype(str) == str(batch_name), csv_sample_col]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )
    if len(su_samples) == 0:
        raise ValueError(
            f"No samples found in CSV where {csv_batch_col} == '{batch_name}'."
        )

    # --- Read the AnnData object ---
    adata = ad.read_h5ad(h5ad_path)

    # --- Ensure the AnnData has the required sample column ---
    if ad_sample_col not in adata.obs.columns:
        # Provide a helpful hint if there are similarly named columns
        similar = [c for c in adata.obs.columns if "sample" in c.lower()]
        hint = f" Similar columns in adata.obs: {similar}" if similar else ""
        raise ValueError(
            f"AnnData obs missing required column '{ad_sample_col}'.{hint}"
        )

    # --- Build mask for cells whose sample is in the desired set ---
    cell_samples = adata.obs[ad_sample_col].astype(str)
    mask = cell_samples.isin(su_samples)

    n_keep = int(mask.sum())
    if n_keep == 0:
        # Give a concise diagnostic about overlap
        n_unique_adata_samples = int(cell_samples.nunique())
        raise ValueError(
            "No cells matched. "
            f"CSV had {len(su_samples)} '{batch_name}' samples; "
            f"AnnData has {n_unique_adata_samples} unique samples "
            f"in obs['{ad_sample_col}']. Check that sample IDs align."
        )

    # --- Subset and save ---
    adata_sub = adata[mask].copy()
    # (Optional) annotate provenance
    adata_sub.uns = dict(adata_sub.uns)  # ensure it's a plain dict
    adata_sub.uns["subset_by_batch"] = {
        "csv_path": csv_path,
        "h5ad_path": h5ad_path,
        "batch_name": batch_name,
        "csv_sample_col": csv_sample_col,
        "csv_batch_col": csv_batch_col,
        "ad_sample_col": ad_sample_col,
        "n_source_cells": int(adata.n_obs),
        "n_retained_cells": int(adata_sub.n_obs),
        "n_vars": int(adata_sub.n_vars),
        "n_su_samples_in_csv": len(su_samples),
    }

    # Ensure output folder exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    adata_sub.write(out_path)

    return out_path, int(adata_sub.n_obs), int(adata_sub.n_vars)

if __name__ == "__main__":
    out, n_cells, n_vars = subset_h5ad_by_batch_samples(
        csv_path="/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv",
        h5ad_path="/dcl01/hongkai/data/data/hjiang/Data/covid_data/count_data.h5ad",
        out_path="/dcl01/hongkai/data/data/hjiang/Data/covid_data/count_data_Su_subset.h5ad",
        batch_name="Su",
        csv_sample_col="sample",
        csv_batch_col="batch",
        ad_sample_col="sample",
    )
    print(f"Subset written to: {out} (n_cells={n_cells}, n_vars={n_vars})")