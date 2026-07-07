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
    """Subset an h5ad to cells from samples where ``csv_batch_col == batch_name``.

    Parameters
    ----------
    csv_path : str
        CSV with at least [csv_sample_col, csv_batch_col].
    h5ad_path : str
        Input h5ad.
    out_path : str
        Destination for the subset h5ad.
    batch_name : str
        Batch label to keep (default "Su").
    csv_sample_col, csv_batch_col : str
        Column names in the CSV.
    ad_sample_col : str
        Column in ``adata.obs`` carrying the sample ID.

    Returns
    -------
    (out_path, n_cells_retained, n_vars)
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isfile(h5ad_path):
        raise FileNotFoundError(f"H5AD not found: {h5ad_path}")

    meta = pd.read_csv(csv_path)
    for col in (csv_sample_col, csv_batch_col):
        if col not in meta.columns:
            raise ValueError(
                f"CSV missing required column '{col}'. "
                f"Available columns: {list(meta.columns)}"
            )

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

    adata = ad.read_h5ad(h5ad_path)

    if ad_sample_col not in adata.obs.columns:
        similar = [c for c in adata.obs.columns if "sample" in c.lower()]
        hint = f" Similar columns in adata.obs: {similar}" if similar else ""
        raise ValueError(
            f"AnnData obs missing required column '{ad_sample_col}'.{hint}"
        )

    cell_samples = adata.obs[ad_sample_col].astype(str)
    mask = cell_samples.isin(su_samples)

    n_keep = int(mask.sum())
    if n_keep == 0:
        n_unique_adata_samples = int(cell_samples.nunique())
        raise ValueError(
            "No cells matched. "
            f"CSV had {len(su_samples)} '{batch_name}' samples; "
            f"AnnData has {n_unique_adata_samples} unique samples "
            f"in obs['{ad_sample_col}']. Check that sample IDs align."
        )

    adata_sub = adata[mask].copy()
    adata_sub.uns = dict(adata_sub.uns)  # ensure it's a plain dict before adding keys
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
