#!/usr/bin/env python3
"""
Pre-flight inspection for running run_dimension_association_analysis() against
four previously-computed result directories.

For each dataset we report:
  * pseudo_adata.h5ad shape, obs dtypes, uns/obsm keys
  * presence + shape of X_DR_expression / X_DR_proportion
  * sample-id values present in pseudo_adata.obs
  * metadata CSV columns + dtypes + n_unique + sample ID
  * overlap between pseudo_adata sample ids and metadata sample ids
  * which columns (beyond sample_col) would be picked up by the
    association module's auto-classification

Read-only — no files are written anywhere.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import anndata as ad


# -------------------------------------------------------------------------
# Copy of the classifier used inside sample_association.association, so this
# script can tell us exactly what the real analysis would test.
# -------------------------------------------------------------------------
_INTERNAL_COL_PATTERNS = (
    r"^pseudotime(_.*)?$",
    r"^cluster_.*_kmeans$",
    r"^X_DR_.*$",
    r"^_",
)


def _is_internal_col(col: str) -> bool:
    return any(re.match(p, col) for p in _INTERNAL_COL_PATTERNS)


def classify_variables(
    obs: pd.DataFrame,
    sample_col: str = "sample",
    min_unique: int = 2,
    categorical_max_levels: int = 10,
):
    n = len(obs)
    continuous, categorical, skipped = [], [], []
    for col in obs.columns:
        if col == sample_col or _is_internal_col(col):
            skipped.append((col, "sample-col/internal"))
            continue
        s = obs[col].dropna()
        n_unique = s.nunique()
        if n_unique < min_unique:
            skipped.append((col, f"n_unique={n_unique}"))
            continue
        if pd.api.types.is_bool_dtype(s):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            is_float = pd.api.types.is_float_dtype(s)
            if n_unique > categorical_max_levels:
                continuous.append(col)
            elif is_float and n_unique > max(5, int(0.3 * n)):
                continuous.append(col)
            else:
                categorical.append(col)
        else:
            if n_unique <= max(categorical_max_levels, int(0.5 * len(s)) + 1):
                categorical.append(col)
            else:
                skipped.append((col, f"string high-cardinality ({n_unique}/{n})"))
    return continuous, categorical, skipped


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def read_csv_robust(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"^﻿", "", regex=True)
        .str.strip()
    )
    # drop all-empty trailing rows/columns
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    return df


def describe_series(s: pd.Series, head: int = 6) -> str:
    s_nn = s.dropna()
    uniq = s_nn.unique()
    sample = list(uniq[:head])
    more = "" if len(uniq) <= head else f" ... ({len(uniq) - head} more)"
    return (
        f"dtype={s.dtype}, n={len(s)}, n_nonnull={len(s_nn)}, "
        f"n_unique={s_nn.nunique()}, examples={sample}{more}"
    )


def describe_obs(obs: pd.DataFrame) -> None:
    print(f"  obs.shape = {obs.shape}")
    print(f"  obs columns:")
    for c in obs.columns:
        print(f"    - {c!r}: {describe_series(obs[c])}")


def describe_uns_obsm(adata: ad.AnnData) -> None:
    print(f"  uns keys: {list(adata.uns.keys())}")
    for k in ("X_DR_expression", "X_DR_proportion"):
        if k in adata.uns:
            v = adata.uns[k]
            if isinstance(v, pd.DataFrame):
                print(f"    uns[{k!r}]: DataFrame shape={v.shape}, "
                      f"cols={list(v.columns)[:8]}, "
                      f"index_head={list(v.index[:5])}")
            else:
                print(f"    uns[{k!r}]: type={type(v).__name__}, "
                      f"shape={getattr(v, 'shape', None)}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")
    for k in adata.obsm.keys():
        arr = adata.obsm[k]
        print(f"    obsm[{k!r}]: shape={getattr(arr, 'shape', None)}")


def compare_samples(
    adata_ids: list[str],
    meta_ids: list[str],
    meta_id_after_strip_suffix: Optional[list[str]] = None,
) -> None:
    s_ad = set(adata_ids)
    s_meta = set(meta_ids)
    inter = s_ad & s_meta
    only_ad = s_ad - s_meta
    only_meta = s_meta - s_ad
    print(f"  sample overlap: {len(inter)} / adata={len(s_ad)} / meta={len(s_meta)}")
    if only_ad:
        print(f"    in adata, NOT in meta ({len(only_ad)}): "
              f"{sorted(only_ad)[:10]}{' ...' if len(only_ad) > 10 else ''}")
    if only_meta:
        print(f"    in meta, NOT in adata ({len(only_meta)}): "
              f"{sorted(only_meta)[:10]}{' ...' if len(only_meta) > 10 else ''}")
    if meta_id_after_strip_suffix is not None:
        s_meta2 = set(meta_id_after_strip_suffix)
        inter2 = s_ad & s_meta2
        print(f"  [after stripping _ATAC/_RNA suffix from metadata] "
              f"overlap: {len(inter2)} / adata={len(s_ad)} / meta(stripped-unique)={len(s_meta2)}")


# -------------------------------------------------------------------------
# Per-dataset inspection
# -------------------------------------------------------------------------

DATASETS = [
    dict(
        name="eye_lutea",
        pseudo="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/pseudobulk/pseudobulk_sample.h5ad",
        meta="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        note="meta CSV has a trailing empty column + 2 blank rows; BOM on header",
    ),
    dict(
        name="eye_retina",
        pseudo="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/pseudobulk/pseudobulk_sample.h5ad",
        meta="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        note="same metadata file as lutea",
    ),
    dict(
        name="heart_SD",
        pseudo="/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/pseudobulk/pseudobulk_sample.h5ad",
        meta="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/heart_updated_sample_meta.csv",
        note="meta has 'organ part' (space, not underscore); lots of trailing empty cols",
    ),
    dict(
        name="ENCODE",
        pseudo="/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/pseudobulk/pseudobulk_sample.h5ad",
        meta="/dcl01/hongkai/data/data/hjiang/Data/paired/sample_metadata.csv",
        meta_fixed="/dcl01/hongkai/data/data/hjiang/Data/paired/sample_metadata_fixed.csv",
        note="metadata duplicates each biosample as _ATAC and _RNA rows; "
             "sample_metadata_fixed.csv has one row per biosample without suffix",
    ),
]


def inspect_one(ds: dict) -> None:
    name = ds["name"]
    print("=" * 88)
    print(f"# {name}")
    print("=" * 88)
    print(f"pseudobulk: {ds['pseudo']}")
    print(f"metadata:   {ds['meta']}")
    if ds.get("note"):
        print(f"note:       {ds['note']}")
    print()

    # -- pseudo_adata --
    if not os.path.exists(ds["pseudo"]):
        print(f"!! MISSING: {ds['pseudo']}")
        return
    print("[1] pseudo_adata inspection")
    try:
        adata = ad.read_h5ad(ds["pseudo"])
    except Exception as e:
        print(f"  !! failed to load: {e}")
        return
    print(f"  adata = {adata}")
    describe_obs(adata.obs)
    describe_uns_obsm(adata)

    # -- metadata --
    print()
    print("[2] metadata inspection")
    if not os.path.exists(ds["meta"]):
        print(f"  !! MISSING: {ds['meta']}")
        return
    try:
        meta = read_csv_robust(ds["meta"])
    except Exception as e:
        print(f"  !! failed to read metadata: {e}")
        return
    print(f"  meta.shape = {meta.shape}")
    print(f"  meta columns:")
    for c in meta.columns:
        print(f"    - {c!r}: {describe_series(meta[c])}")
    meta_sample_col = None
    for cand in ("sample", "Sample", "sample_id"):
        if cand in meta.columns:
            meta_sample_col = cand
            break
    print(f"  meta sample column detected: {meta_sample_col!r}")

    # -- alignment --
    print()
    print("[3] sample-id alignment")
    adata_ids = adata.obs_names.astype(str).tolist()
    if "sample" in adata.obs.columns:
        obs_sample_ids = adata.obs["sample"].astype(str).tolist()
        # compare obs_names vs obs['sample'] — sometimes they differ
        if set(obs_sample_ids) != set(adata_ids):
            print("  NOTE: obs_names and obs['sample'] differ")
            print(f"    obs_names head: {adata_ids[:5]}")
            print(f"    obs['sample'] head: {obs_sample_ids[:5]}")
        else:
            print("  obs_names == obs['sample'] (set-equal)")
    print(f"  adata sample ids ({len(adata_ids)}): "
          f"{adata_ids[:10]}{' ...' if len(adata_ids) > 10 else ''}")
    if meta_sample_col is not None:
        meta_ids = meta[meta_sample_col].astype(str).tolist()
        print(f"  meta sample ids ({len(meta_ids)}): "
              f"{meta_ids[:10]}{' ...' if len(meta_ids) > 10 else ''}")
        stripped = None
        # For ENCODE: strip _RNA / _ATAC suffix so we can judge whether the
        # non-suffixed form matches pseudo_adata sample ids.
        if name == "ENCODE":
            stripped_series = (
                meta[meta_sample_col]
                .astype(str)
                .str.replace(r"_(ATAC|RNA)$", "", regex=True)
            )
            stripped = list(pd.unique(stripped_series))
        compare_samples(adata_ids, meta_ids, meta_id_after_strip_suffix=stripped)

    # ENCODE-specific: also show the fixed metadata
    if name == "ENCODE" and ds.get("meta_fixed"):
        mf_path = ds["meta_fixed"]
        print()
        print(f"[3b] ENCODE fixed metadata inspection: {mf_path}")
        if os.path.exists(mf_path):
            mf = read_csv_robust(mf_path)
            print(f"  meta_fixed.shape = {mf.shape}")
            for c in mf.columns:
                print(f"    - {c!r}: {describe_series(mf[c])}")
            if "sample" in mf.columns:
                compare_samples(adata_ids, mf["sample"].astype(str).tolist())

    # -- auto classification on merged obs --
    print()
    print("[4] auto-classification on (current obs) — BEFORE any merge")
    cont_a, cat_a, skip_a = classify_variables(adata.obs)
    print(f"  continuous: {cont_a}")
    print(f"  categorical: {cat_a}")
    if skip_a:
        print(f"  skipped: {skip_a[:20]}{' ...' if len(skip_a) > 20 else ''}")

    print()
    print("[5] auto-classification on obs AFTER a trial merge with metadata")
    # Build a merged copy (same logic the pipeline uses) and re-classify so we
    # know what the association module would actually test after we attach the
    # metadata to the existing pseudo_adata.
    trial_obs = adata.obs.copy()
    if meta_sample_col is not None:
        meta_for_merge = meta.copy()
        if name == "ENCODE":
            # Keep both raw and stripped sample ids; prefer stripped when it
            # matches the pseudo_adata obs_names.
            meta_for_merge[meta_sample_col] = (
                meta_for_merge[meta_sample_col]
                .astype(str)
                .str.replace(r"_(ATAC|RNA)$", "", regex=True)
            )
            meta_for_merge = meta_for_merge.drop_duplicates(subset=[meta_sample_col])
        meta_for_merge = meta_for_merge.rename(columns={meta_sample_col: "sample"})
        if "sample" not in trial_obs.columns:
            trial_obs["sample"] = trial_obs.index.astype(str)
        overlap_cols = [c for c in meta_for_merge.columns
                        if c != "sample" and c in trial_obs.columns]
        if overlap_cols:
            trial_obs = trial_obs.drop(columns=overlap_cols)
        merged = trial_obs.merge(meta_for_merge, on="sample", how="left")
        merged.index = trial_obs.index
        n_matched = merged.drop(columns=["sample"]).notna().any(axis=1).sum()
        print(f"  trial merge: matched {n_matched}/{len(merged)} rows with any metadata")
        cont_b, cat_b, skip_b = classify_variables(merged)
        print(f"  continuous: {cont_b}")
        print(f"  categorical: {cat_b}")
        if skip_b:
            print(f"  skipped: {skip_b[:20]}{' ...' if len(skip_b) > 20 else ''}")
    else:
        print("  (no metadata sample column detected — can't trial-merge)")


def main():
    for ds in DATASETS:
        try:
            inspect_one(ds)
        except Exception as e:
            print(f"!! inspection failed for {ds['name']}: {e}")
            import traceback; traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
