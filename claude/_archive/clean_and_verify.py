"""Clean up the 1M-scBloodNL AnnData and verify metadata provenance.

Steps:
  1. Drop the single cell whose `timepoint` is <NA> (the original 'nan' label).
  2. Check whether `assignment` alone uniquely identifies a sample, or whether
     a sample is really (assignment, stimulation_conditions, timepoint, chem).
  3. Verify the AnnData's obs/var match the source files
     `sample_meta.tsv` and `unfiltered_features.tsv.gz`.
  4. Overwrite the .h5ad (gzip compressed) with the cleaned object.

Run with the same env used to build the file:
  /users/hjiang/.conda/envs/hongkai/bin/python -u clean_and_verify.py
"""

import gzip
import re

import anndata as ad
import numpy as np
import pandas as pd

DATA_DIR = "/dcs07/hongkai/data/harry/result/1M-scBloodNL/data"
H5AD_PATH = f"{DATA_DIR}/1M-scBloodNL.h5ad"
SAMPLE_META = f"{DATA_DIR}/sample_meta.tsv"
FEATURES = f"{DATA_DIR}/unfiltered_features.tsv.gz"

ORIG_LABEL_PATTERN = re.compile(r"^(\d+)h(CA|MTB|PA)$")


def reconstruct_original_timepoint(stim, time):
    """Inverse of split_timepoint.split_label."""
    if pd.isna(stim) or pd.isna(time):
        return pd.NA
    if stim == "UT":
        return "UT"
    return f"{int(time)}h{stim}"


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    section(f"Loading {H5AD_PATH}")
    adata = ad.read_h5ad(H5AD_PATH)
    print(f"  shape: {adata.shape}")
    print(f"  obs columns: {list(adata.obs.columns)}")

    # ------------------------------------------------------------------
    # 1. Drop the cell with NA timepoint
    # ------------------------------------------------------------------
    section("Step 1: drop cells with NA timepoint")
    na_mask = adata.obs["timepoint"].isna()
    n_na = int(na_mask.sum())
    print(f"  cells with NA timepoint: {n_na}")
    if n_na > 0:
        print("  dropping the following barcodes:")
        for b in adata.obs.index[na_mask]:
            print(f"    {b}")
        adata = adata[~na_mask].copy()
        # Cast timepoint back to a non-nullable int now that NAs are gone.
        adata.obs["timepoint"] = adata.obs["timepoint"].astype(np.int32)
        # Drop now-unused categories from stimulation_conditions.
        adata.obs["stimulation_conditions"] = (
            adata.obs["stimulation_conditions"].cat.remove_unused_categories()
        )
        print(f"  new shape: {adata.shape}")
        print(f"  timepoint dtype: {adata.obs['timepoint'].dtype}")

    # ------------------------------------------------------------------
    # 2. Does `assignment` represent a sample?
    # ------------------------------------------------------------------
    section("Step 2: is `assignment` a sample identifier?")
    obs = adata.obs
    n_assignments = obs["assignment"].nunique()
    print(f"  unique assignments: {n_assignments}")

    grp = obs.groupby("assignment", observed=True).agg(
        n_cells=("chem", "size"),
        n_chem=("chem", "nunique"),
        n_timepoints=("timepoint", "nunique"),
        n_stim=("stimulation_conditions", "nunique"),
        n_combos=("chem", lambda s: len(set(zip(
            s,
            obs.loc[s.index, "stimulation_conditions"],
            obs.loc[s.index, "timepoint"],
        )))),
    )
    print(f"  per-assignment combo counts (chem x stim x timepoint):")
    print(f"    n_chem        : min={grp['n_chem'].min()}, max={grp['n_chem'].max()}")
    print(f"    n_timepoints  : min={grp['n_timepoints'].min()}, max={grp['n_timepoints'].max()}")
    print(f"    n_stim        : min={grp['n_stim'].min()}, max={grp['n_stim'].max()}")
    print(f"    n_combos      : min={grp['n_combos'].min()}, max={grp['n_combos'].max()}")

    if (grp["n_combos"] == 1).all():
        verdict = "YES — `assignment` already uniquely identifies a sample."
    else:
        n_multi = int((grp["n_combos"] > 1).sum())
        verdict = (
            f"NO — {n_multi}/{n_assignments} assignments span multiple "
            "(stim, timepoint, chem) combos. A sample is "
            "(assignment, stimulation_conditions, timepoint, chem)."
        )
    print(f"  verdict: {verdict}")

    # Sample = (assignment, stimulation_conditions, timepoint, chem) — show count.
    sample_keys = obs[
        ["assignment", "stimulation_conditions", "timepoint", "chem"]
    ].drop_duplicates()
    print(f"  total unique (assignment, stim, timepoint, chem) tuples: {len(sample_keys)}")

    # ------------------------------------------------------------------
    # 3a. Verify sample_meta.tsv matches obs
    # ------------------------------------------------------------------
    section("Step 3a: verify sample_meta.tsv vs adata.obs")
    sm = pd.read_csv(SAMPLE_META, sep="\t", dtype=str)
    print(f"  sample_meta.tsv rows: {len(sm)}, columns: {list(sm.columns)}")
    sm = sm.set_index("barcode")

    missing_in_meta = adata.obs.index.difference(sm.index)
    extra_in_meta = sm.index.difference(adata.obs.index)
    print(f"  barcodes in obs but missing from sample_meta: {len(missing_in_meta)}")
    print(f"  barcodes in sample_meta but missing from obs: {len(extra_in_meta)}")

    sm_aligned = sm.loc[adata.obs.index]

    # assignment: obs is stored as e.g. "1.0", source is "1". Compare as floats.
    obs_assignment = pd.to_numeric(adata.obs["assignment"].astype(str), errors="coerce")
    src_assignment = pd.to_numeric(sm_aligned["assignment"], errors="coerce")
    n_diff_assignment = int(((obs_assignment != src_assignment) & ~(
        obs_assignment.isna() & src_assignment.isna()
    )).sum())
    print(f"  assignment mismatches: {n_diff_assignment}")

    # chem: direct string compare.
    n_diff_chem = int((adata.obs["chem"].astype(str).values != sm_aligned["chem"].values).sum())
    print(f"  chem mismatches: {n_diff_chem}")

    # timepoint: reconstruct original label from (stim, time) and compare.
    reconstructed = [
        reconstruct_original_timepoint(s, t)
        for s, t in zip(
            adata.obs["stimulation_conditions"].astype(object),
            adata.obs["timepoint"].astype(object),
        )
    ]
    src_tp = sm_aligned["timepoint"].values
    n_diff_tp = sum(
        1 for r, s in zip(reconstructed, src_tp)
        if not (pd.isna(r) and (pd.isna(s) or str(s).lower() == "nan")) and r != s
    )
    print(f"  timepoint mismatches (reconstructed vs source): {n_diff_tp}")

    # ------------------------------------------------------------------
    # 3b. Verify features file matches adata.var_names
    # ------------------------------------------------------------------
    section("Step 3b: verify unfiltered_features.tsv.gz vs adata.var_names")
    with gzip.open(FEATURES, "rt") as f:
        features = [line.strip() for line in f if line.strip()]
    print(f"  features file rows: {len(features)}")
    print(f"  adata.var_names   : {adata.n_vars}")
    if len(features) != adata.n_vars:
        print("  LENGTH MISMATCH")
    else:
        feat_arr = np.asarray(features)
        var_arr = np.asarray(adata.var_names)
        n_diff_feat = int((feat_arr != var_arr).sum())
        print(f"  feature mismatches (positional): {n_diff_feat}")
        first_diff = np.where(feat_arr != var_arr)[0]
        if len(first_diff):
            for i in first_diff[:5]:
                print(f"    pos {i}: file={feat_arr[i]!r}, var={var_arr[i]!r}")

    # ------------------------------------------------------------------
    # 4. Save cleaned AnnData
    # ------------------------------------------------------------------
    section(f"Step 4: writing cleaned AnnData (gzip) -> {H5AD_PATH}")
    print(f"  final shape: {adata.shape}")
    print("  final timepoint counts:")
    print(adata.obs["timepoint"].value_counts(dropna=False))
    print("  final stimulation_conditions counts:")
    print(adata.obs["stimulation_conditions"].value_counts(dropna=False))
    adata.write_h5ad(H5AD_PATH, compression="gzip")
    print("Done.")


if __name__ == "__main__":
    main()
