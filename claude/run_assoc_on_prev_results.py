#!/usr/bin/env python3
"""
Run run_dimension_association_analysis() on four previously-computed
result directories. Outputs go to <result_root>/sample_association/.

Per-dataset prep:
  - lutea / retina: obs already has age/sex/disease_state/organ_part. Coerce
    `age` from category-string to numeric so it is tested as continuous.
    Drop dataset/batch/source_file (redundant with original_sample) and the
    Unnamed / all-constant columns. Add a `sample` column for exclusion.
  - heart_SD: same cleanup; disease_state has 3 biologically meaningful
    levels; `age` is constant so it stays skipped.
  - ENCODE: merge /dcl01/.../sample_metadata.csv on obs_names to pull in
    `tissue`, then run association.

Row-alignment sanity check: for each embedding, if uns[X_DR_*] is a
DataFrame we verify its values row-match obsm[X_DR_*] (guaranteed ordered
by obs_names). If they diverge, we drop the uns version so the association
module falls through to obsm.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, "/users/hjiang/GenoDistance/code")
from sample_association.association import run_dimension_association_analysis


DATASETS = [
    dict(
        name="eye_lutea",
        pseudo="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/pseudobulk/pseudobulk_sample.h5ad",
        out_dir="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/sample_association",
        meta_csv=None,  # already merged into obs
        keep_cols=["modality", "original_sample", "age", "sex", "disease_state", "organ_part"],
        numeric_cols=["age"],
    ),
    dict(
        name="eye_retina",
        pseudo="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/pseudobulk/pseudobulk_sample.h5ad",
        out_dir="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/sample_association",
        meta_csv=None,
        keep_cols=["modality", "original_sample", "age", "sex", "disease_state", "organ_part"],
        numeric_cols=["age"],
    ),
    dict(
        name="heart_SD",
        pseudo="/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/pseudobulk/pseudobulk_sample.h5ad",
        out_dir="/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/sample_association",
        meta_csv=None,
        keep_cols=["modality", "original_sample", "age", "sex", "disease_state", "organ_part"],
        numeric_cols=[],  # age here is "post-embryonic stage" for all; leave as-is
    ),
    dict(
        name="ENCODE",
        pseudo="/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/pseudobulk/pseudobulk_sample.h5ad",
        out_dir="/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/sample_association",
        meta_csv="/dcl01/hongkai/data/data/hjiang/Data/paired/sample_metadata.csv",
        meta_sample_col="sample",  # matches adata.obs_names (with _ATAC/_RNA suffix)
        keep_cols=["modality", "original_sample", "tissue", "predicted_doublet"],
        numeric_cols=[],
    ),
]

# Common kwargs
N_PERMUTATIONS = 999
RANDOM_STATE = 42


def check_dr_alignment(adata: ad.AnnData, verbose: bool = True) -> None:
    """Verify uns['X_DR_*'] DataFrames are row-aligned to obsm (and thus obs_names).
    If a uns DataFrame disagrees with its obsm counterpart, drop the uns entry so
    the association module uses the obs-aligned obsm array instead.
    """
    for key in ("X_DR_expression", "X_DR_proportion"):
        has_uns = (
            key in adata.uns and isinstance(adata.uns[key], pd.DataFrame)
        )
        has_obsm = key in adata.obsm
        if not has_uns or not has_obsm:
            continue
        uns_vals = np.asarray(adata.uns[key].values, dtype=float)
        obsm_vals = np.asarray(adata.obsm[key], dtype=float)
        if uns_vals.shape != obsm_vals.shape:
            if verbose:
                print(f"  [align] {key}: shape mismatch uns={uns_vals.shape} obsm={obsm_vals.shape} — dropping uns")
            del adata.uns[key]
            continue
        if not np.allclose(uns_vals, obsm_vals, equal_nan=True, atol=1e-6):
            if verbose:
                print(f"  [align] {key}: uns values differ from obsm — dropping uns (will use obsm)")
            del adata.uns[key]
        else:
            if verbose:
                print(f"  [align] {key}: uns ≡ obsm ✓ (using uns for PC column names)")


def prepare_adata(ds: dict) -> ad.AnnData:
    print(f"\n--- loading {ds['name']} ---")
    print(f"  pseudo: {ds['pseudo']}")
    adata = ad.read_h5ad(ds["pseudo"])
    print(f"  shape = {adata.shape}")

    # 1. Merge external metadata (ENCODE only).
    if ds["meta_csv"]:
        print(f"  merging metadata from: {ds['meta_csv']}")
        meta = pd.read_csv(ds["meta_csv"], sep=None, engine="python", encoding="utf-8-sig")
        meta.columns = meta.columns.astype(str).str.strip()
        sc = ds["meta_sample_col"]
        if sc not in meta.columns:
            raise ValueError(f"{sc!r} not in metadata columns: {list(meta.columns)}")
        meta = meta.set_index(sc)
        # Join on obs_names
        join_key = adata.obs_names.astype(str)
        # Drop overlapping cols from obs (keep metadata version)
        overlap = [c for c in meta.columns if c in adata.obs.columns]
        if overlap:
            print(f"    dropping overlapping obs cols (keeping metadata versions): {overlap}")
            adata.obs = adata.obs.drop(columns=overlap)
        merged = adata.obs.copy()
        merged["__join__"] = join_key.values
        merged = merged.merge(meta, left_on="__join__", right_index=True, how="left")
        merged = merged.drop(columns="__join__")
        merged.index = adata.obs.index
        n_matched = merged[meta.columns[0]].notna().sum() if len(meta.columns) else 0
        print(f"    matched {n_matched}/{len(merged)} rows")
        adata.obs = merged

    # 2. Coerce numeric columns (e.g. age stored as category-string).
    for c in ds["numeric_cols"]:
        if c in adata.obs.columns:
            before_unique = adata.obs[c].astype(str).nunique()
            adata.obs[c] = pd.to_numeric(adata.obs[c].astype(str), errors="coerce")
            after_unique = adata.obs[c].dropna().nunique()
            print(f"  coerced {c!r} to numeric: {before_unique} string uniques → {after_unique} numeric uniques, "
                  f"{adata.obs[c].isna().sum()} NaN")

    # 3. Filter obs down to the columns we want tested, plus a sample column.
    keep = [c for c in ds["keep_cols"] if c in adata.obs.columns]
    missing = [c for c in ds["keep_cols"] if c not in adata.obs.columns]
    if missing:
        print(f"  WARN: requested keep_cols missing from obs: {missing}")
    new_obs = adata.obs[keep].copy()
    new_obs["sample"] = adata.obs_names.astype(str)
    adata.obs = new_obs
    print(f"  obs after cleanup: {list(adata.obs.columns)}")

    # 4. Verify DR row alignment.
    check_dr_alignment(adata, verbose=True)

    return adata


def run_one(ds: dict) -> None:
    print("=" * 88)
    print(f"RUNNING: {ds['name']}  →  {ds['out_dir']}")
    print("=" * 88)

    adata = prepare_adata(ds)

    os.makedirs(ds["out_dir"], exist_ok=True)
    result = run_dimension_association_analysis(
        pseudo_adata=adata,
        output_dir=ds["out_dir"],
        continuous_cols=None,    # auto-classify from cleaned obs
        categorical_cols=None,
        n_permutations=N_PERMUTATIONS,
        sample_col="sample",
        random_state=RANDOM_STATE,
        verbose=True,
    )

    # Short summary per embedding
    for emb, df in result["results"].items():
        if df.empty:
            print(f"  [{ds['name']}/{emb}] no rows produced")
            continue
        top = df.sort_values(["variable", "r2"], ascending=[True, False]).groupby("variable").head(1)
        print(f"\n  [{ds['name']}/{emb}] top R² per variable:")
        for _, r in top.iterrows():
            print(f"    {r['variable']:>20s}  {r['component']:<8s}  "
                  f"R²={r['r2']:.3f}  perm_p={r['perm_p']:.3g}  FDR={r['fdr']:.3g}")


def main():
    for ds in DATASETS:
        try:
            run_one(ds)
        except Exception as e:
            print(f"!! {ds['name']} FAILED: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
