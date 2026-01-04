import anndata as ad
import pandas as pd
from pathlib import Path

def append_sample_metadata_and_overwrite(
    h5ad_path: str | Path,
    sample_meta_csv: str | Path,
    sample_col_csv: str = "sample",
    modality_suffixes: tuple = ("_RNA", "_ATAC"),
    drop_unnamed: bool = True,
    fail_if_missing: bool = False,
    verbose: bool = True,
):
    """
    Debug version:
    - Shows overlapping columns that would break pd.join
    - Drops Unnamed:* columns from CSV (optional)
    - Joins only non-overlapping columns
    - Then overwrites overlapping columns explicitly with CSV values
    - Overwrites original h5ad in-place
    """

    h5ad_path = Path(h5ad_path)
    sample_meta_csv = Path(sample_meta_csv)

    if verbose:
        print(f"[1] Loading AnnData: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)

    if verbose:
        print(f"[2] Loading metadata CSV: {sample_meta_csv}")
    meta = pd.read_csv(sample_meta_csv)

    if drop_unnamed:
        unnamed = [c for c in meta.columns if str(c).startswith("Unnamed:")]
        if unnamed and verbose:
            print(f"[2a] Dropping {len(unnamed)} columns from CSV: {unnamed}")
        meta = meta.drop(columns=unnamed, errors="ignore")

    if sample_col_csv not in meta.columns:
        raise ValueError(
            f"[ERROR] '{sample_col_csv}' not found in metadata CSV columns. "
            f"Available columns: {meta.columns.tolist()}"
        )

    # Index by base sample
    meta = meta.set_index(sample_col_csv)

    # --------------------------------------------------
    # 1) Build base sample id from obs_names
    # --------------------------------------------------
    obs = adata.obs.copy()
    obs["__base_sample__"] = obs.index.astype(str)

    for suf in modality_suffixes:
        obs["__base_sample__"] = obs["__base_sample__"].str.replace(
            f"{suf}$", "", regex=True
        )

    # --------------------------------------------------
    # 2) Debug: overlap + coverage
    # --------------------------------------------------
    obs_cols = set(obs.columns)
    meta_cols = set(meta.columns)

    overlap = sorted((obs_cols & meta_cols) - {"__base_sample__"})
    if verbose:
        print(f"[3] adata.obs columns: {len(obs_cols)}")
        print(f"[3] meta columns (excluding key): {len(meta_cols)}")
        print(f"[3] Overlapping columns: {len(overlap)}")
        if overlap:
            print("    -> overlap:", overlap)

    base_samples = pd.Index(obs["__base_sample__"].unique())
    meta_samples = pd.Index(meta.index.unique())

    n_total = len(base_samples)
    n_found = base_samples.isin(meta_samples).sum()
    n_missing = n_total - n_found

    if verbose:
        print(f"[4] Base samples in h5ad: {n_total}")
        print(f"[4] Matched in CSV:       {n_found}")
        print(f"[4] Missing in CSV:       {n_missing}")

    if n_missing > 0 and verbose:
        missing_list = base_samples[~base_samples.isin(meta_samples)].tolist()
        print(f"[4a] Example missing (up to 20): {missing_list[:20]}")

    if fail_if_missing and n_missing > 0:
        raise ValueError(
            f"[ERROR] {n_missing}/{n_total} base samples missing in CSV. "
            f"Set fail_if_missing=False to allow NaNs."
        )

    # --------------------------------------------------
    # 3) Join only non-overlapping columns to avoid pandas error
    # --------------------------------------------------
    non_overlap_cols = [c for c in meta.columns if c not in obs.columns]
    if verbose:
        print(f"[5] Non-overlapping meta columns to add via join: {len(non_overlap_cols)}")
        if non_overlap_cols:
            print("    -> add:", non_overlap_cols)

    merged = obs.join(meta[non_overlap_cols], on="__base_sample__", how="left")

    # Add non-overlapping columns (these are new)
    for col in non_overlap_cols:
        adata.obs[col] = merged[col]

    # --------------------------------------------------
    # 4) Overwrite overlapping columns explicitly (CSV wins)
    #    Do this via map for clarity and safety.
    # --------------------------------------------------
    if verbose and overlap:
        print(f"[6] Overwriting {len(overlap)} overlapping columns from CSV...")

    for col in overlap:
        # Map base sample -> value for this column
        mapping = meta[col]
        adata.obs[col] = obs["__base_sample__"].map(mapping)

    # Cleanup temp column if it didn't exist before
    if "__base_sample__" in adata.obs.columns:
        adata.obs.drop(columns="__base_sample__", inplace=True)

    # --------------------------------------------------
    # 5) Write back (overwrite)
    # --------------------------------------------------
    if verbose:
        print(f"[7] Overwriting AnnData at: {h5ad_path}")
    adata.write_h5ad(h5ad_path)

    if verbose:
        print("[DONE] ✅ Metadata appended/updated and h5ad overwritten.")



if __name__ == "__main__":
    append_sample_metadata_and_overwrite(
        h5ad_path="/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/pseudobulk/pseudobulk_sample.h5ad",
        sample_meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        sample_col_csv="sample",  # change if needed
    )
