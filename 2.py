#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc


# ============================================================
# USER CONFIG
# ============================================================
H5AD_PATH = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/count_data.h5ad"
SAMPLE_META_PATH = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv"
OUTPUT_H5AD_PATH = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/test_RNA.h5ad"
OUTPUT_SAMPLE_CSV_PATH = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/count_data_subsample_8samples_selected_samples.csv"

# Column in adata.obs that identifies sample
SAMPLE_COLUMN = "sample"

# Column names in sample metadata
# Change these if your sample_data.csv uses different names
BATCH_CANDIDATE_COLUMNS = ["batch", "study", "dataset", "cohort", "source"]
SEVERITY_CANDIDATE_COLUMNS = ["sev.level", "sev_level", "severity", "sev"]

# Desired design
N_TOTAL_SAMPLES = 8
N_SEV1 = 4
N_SEV4 = 4
SEV1_VALUE = 1
SEV4_VALUE = 4

# Random seed for reproducibility
RANDOM_SEED = 42

# Whether to require balanced selection within each batch:
#   True  -> try to pick 2 sev1 + 2 sev4 from each of 2 batches
#   False -> only enforce total 4 sev1 + 4 sev4 across 2 batches
REQUIRE_PER_BATCH_BALANCE = True
# ============================================================


def detect_column(df, candidates, required=True, label="column"):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"Could not find {label}. Tried columns: {candidates}. "
            f"Available columns are: {df.columns.tolist()}"
        )
    return None


def standardize_sample_meta(sample_meta, sample_col, batch_col, sev_col):
    meta = sample_meta.copy()

    if sample_col not in meta.columns:
        raise KeyError(
            f"Sample metadata must contain sample column '{sample_col}'. "
            f"Available columns: {meta.columns.tolist()}"
        )

    meta[sample_col] = meta[sample_col].astype(str).str.strip()
    meta[batch_col] = meta[batch_col].astype(str).str.strip()

    # Convert severity to numeric if possible
    meta[sev_col] = pd.to_numeric(meta[sev_col], errors="coerce")

    # Drop rows missing critical info
    meta = meta.dropna(subset=[sample_col, batch_col, sev_col]).copy()

    # Keep only one row per sample
    # If duplicates exist, keep the first one
    meta = meta.drop_duplicates(subset=[sample_col]).copy()

    return meta


def choose_samples_balanced_per_batch(meta, sample_col, batch_col, sev_col, rng):
    """
    Try to find two batches such that each batch has at least:
      - 2 samples with sev.level == 1
      - 2 samples with sev.level == 4
    Then pick 2+2 from each batch, total 8 samples.
    """
    batch_summary = []
    for b, dfb in meta.groupby(batch_col):
        sev1_samples = sorted(dfb.loc[dfb[sev_col] == SEV1_VALUE, sample_col].unique().tolist())
        sev4_samples = sorted(dfb.loc[dfb[sev_col] == SEV4_VALUE, sample_col].unique().tolist())
        batch_summary.append({
            "batch": b,
            "n_sev1": len(sev1_samples),
            "n_sev4": len(sev4_samples),
            "sev1_samples": sev1_samples,
            "sev4_samples": sev4_samples,
        })

    batch_summary = pd.DataFrame(batch_summary)

    eligible = batch_summary[
        (batch_summary["n_sev1"] >= 2) &
        (batch_summary["n_sev4"] >= 2)
    ].copy()

    if eligible.shape[0] < 2:
        return None, batch_summary

    chosen_batches = sorted(
        rng.choice(eligible["batch"].values, size=2, replace=False).tolist()
    )

    selected_rows = []
    for b in chosen_batches:
        row = eligible.loc[eligible["batch"] == b].iloc[0]
        sev1_pick = sorted(rng.choice(row["sev1_samples"], size=2, replace=False).tolist())
        sev4_pick = sorted(rng.choice(row["sev4_samples"], size=2, replace=False).tolist())

        for s in sev1_pick:
            selected_rows.append({
                sample_col: s,
                batch_col: b,
                sev_col: SEV1_VALUE
            })
        for s in sev4_pick:
            selected_rows.append({
                sample_col: s,
                batch_col: b,
                sev_col: SEV4_VALUE
            })

    selected_df = pd.DataFrame(selected_rows)
    return selected_df, batch_summary


def choose_samples_total_only(meta, sample_col, batch_col, sev_col, rng):
    """
    Relaxed fallback:
      - choose 2 batches
      - total 4 samples with sev==1 and 4 samples with sev==4 across those 2 batches
    """
    batches = sorted(meta[batch_col].unique().tolist())
    if len(batches) < 2:
        raise ValueError(f"Need at least 2 batches, found only {len(batches)}.")

    # Evaluate all pairs
    best_pair = None
    for i in range(len(batches)):
        for j in range(i + 1, len(batches)):
            b1, b2 = batches[i], batches[j]
            sub = meta[meta[batch_col].isin([b1, b2])].copy()

            sev1_samples = sorted(sub.loc[sub[sev_col] == SEV1_VALUE, sample_col].unique().tolist())
            sev4_samples = sorted(sub.loc[sub[sev_col] == SEV4_VALUE, sample_col].unique().tolist())

            if len(sev1_samples) >= N_SEV1 and len(sev4_samples) >= N_SEV4:
                best_pair = (b1, b2)
                break
        if best_pair is not None:
            break

    if best_pair is None:
        raise ValueError(
            "Could not find any pair of batches with enough samples to satisfy "
            f"{N_SEV1} samples of severity {SEV1_VALUE} and "
            f"{N_SEV4} samples of severity {SEV4_VALUE}."
        )

    sub = meta[meta[batch_col].isin(best_pair)].copy()
    sev1_pool = sorted(sub.loc[sub[sev_col] == SEV1_VALUE, sample_col].unique().tolist())
    sev4_pool = sorted(sub.loc[sub[sev_col] == SEV4_VALUE, sample_col].unique().tolist())

    sev1_pick = sorted(rng.choice(sev1_pool, size=N_SEV1, replace=False).tolist())
    sev4_pick = sorted(rng.choice(sev4_pool, size=N_SEV4, replace=False).tolist())

    selected_samples = sev1_pick + sev4_pick
    selected_df = sub[sub[sample_col].isin(selected_samples)][[sample_col, batch_col, sev_col]].drop_duplicates().copy()

    return selected_df


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print("=== Reading AnnData ===")
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"AnnData shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if SAMPLE_COLUMN not in adata.obs.columns:
        raise KeyError(
            f"'{SAMPLE_COLUMN}' not found in adata.obs. "
            f"Available obs columns: {adata.obs.columns.tolist()}"
        )

    print("=== Reading sample metadata ===")
    sample_meta = pd.read_csv(SAMPLE_META_PATH)
    print(f"Sample metadata shape: {sample_meta.shape}")

    batch_col = detect_column(
        sample_meta,
        BATCH_CANDIDATE_COLUMNS,
        required=True,
        label="batch/study column"
    )
    sev_col = detect_column(
        sample_meta,
        SEVERITY_CANDIDATE_COLUMNS,
        required=True,
        label="severity column"
    )

    print(f"Detected batch/study column: {batch_col}")
    print(f"Detected severity column: {sev_col}")

    sample_meta = standardize_sample_meta(
        sample_meta=sample_meta,
        sample_col=SAMPLE_COLUMN,
        batch_col=batch_col,
        sev_col=sev_col,
    )

    # Keep only samples that actually exist in the AnnData
    adata_samples = pd.Index(adata.obs[SAMPLE_COLUMN].astype(str).unique())
    sample_meta = sample_meta[sample_meta[SAMPLE_COLUMN].isin(adata_samples)].copy()

    print("=== Metadata after filtering to samples present in AnnData ===")
    print(f"Number of metadata rows kept: {sample_meta.shape[0]}")
    print(f"Unique batches/studies: {sample_meta[batch_col].nunique()}")
    print("Severity counts by sample:")
    print(sample_meta[sev_col].value_counts(dropna=False).sort_index())

    print("\n=== Trying to choose samples ===")
    selected_df = None

    if REQUIRE_PER_BATCH_BALANCE:
        selected_df, batch_summary = choose_samples_balanced_per_batch(
            meta=sample_meta,
            sample_col=SAMPLE_COLUMN,
            batch_col=batch_col,
            sev_col=sev_col,
            rng=rng,
        )

        print("Per-batch summary:")
        print(batch_summary[[ "batch", "n_sev1", "n_sev4" ]].sort_values("batch").to_string(index=False))

        if selected_df is None:
            print("\nCould not satisfy strict per-batch balance (2 sev1 + 2 sev4 in each batch).")
            print("Falling back to relaxed selection across 2 batches.")
            selected_df = choose_samples_total_only(
                meta=sample_meta,
                sample_col=SAMPLE_COLUMN,
                batch_col=batch_col,
                sev_col=sev_col,
                rng=rng,
            )
    else:
        selected_df = choose_samples_total_only(
            meta=sample_meta,
            sample_col=SAMPLE_COLUMN,
            batch_col=batch_col,
            sev_col=sev_col,
            rng=rng,
        )

    selected_df = selected_df.drop_duplicates(subset=[SAMPLE_COLUMN]).copy()

    # Final checks
    if selected_df.shape[0] != N_TOTAL_SAMPLES:
        raise ValueError(
            f"Expected {N_TOTAL_SAMPLES} selected samples, but got {selected_df.shape[0]}."
        )

    batch_n = selected_df[batch_col].nunique()
    if batch_n != 2:
        raise ValueError(f"Expected 2 batches/studies, but got {batch_n}.")

    sev_counts = selected_df[sev_col].value_counts().to_dict()
    if sev_counts.get(SEV1_VALUE, 0) != N_SEV1:
        raise ValueError(
            f"Expected {N_SEV1} samples with severity {SEV1_VALUE}, "
            f"but got {sev_counts.get(SEV1_VALUE, 0)}."
        )
    if sev_counts.get(SEV4_VALUE, 0) != N_SEV4:
        raise ValueError(
            f"Expected {N_SEV4} samples with severity {SEV4_VALUE}, "
            f"but got {sev_counts.get(SEV4_VALUE, 0)}."
        )

    selected_samples = sorted(selected_df[SAMPLE_COLUMN].astype(str).tolist())

    print("\n=== Selected samples ===")
    print(selected_df.sort_values([batch_col, sev_col, SAMPLE_COLUMN]).to_string(index=False))

    # Subset AnnData by sample
    print("\n=== Subsetting AnnData ===")
    mask = adata.obs[SAMPLE_COLUMN].astype(str).isin(selected_samples)
    adata_sub = adata[mask].copy()

    print(f"Subset shape: {adata_sub.shape[0]} cells × {adata_sub.shape[1]} genes")
    print(f"Unique samples in subset: {adata_sub.obs[SAMPLE_COLUMN].nunique()}")

    # Optional: add selected sample-level metadata into obs
    selected_meta = selected_df.set_index(SAMPLE_COLUMN)
    adata_sub.obs = adata_sub.obs.join(selected_meta, on=SAMPLE_COLUMN, rsuffix="_samplemeta")

    print("\n=== Saving outputs ===")
    os.makedirs(os.path.dirname(OUTPUT_H5AD_PATH), exist_ok=True)
    adata_sub.write_h5ad(OUTPUT_H5AD_PATH)

    selected_df.sort_values([batch_col, sev_col, SAMPLE_COLUMN]).to_csv(
        OUTPUT_SAMPLE_CSV_PATH,
        index=False
    )

    print(f"Saved subset h5ad: {OUTPUT_H5AD_PATH}")
    print(f"Saved selected sample table: {OUTPUT_SAMPLE_CSV_PATH}")

    print("\n=== Final verification ===")
    print("Cells per selected sample:")
    print(adata_sub.obs[SAMPLE_COLUMN].value_counts().sort_index().to_string())

    print("\nSeverity distribution among selected samples:")
    print(selected_df[sev_col].value_counts().sort_index().to_string())

    print("\nBatch/study distribution among selected samples:")
    print(selected_df[batch_col].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()