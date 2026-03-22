#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from datetime import datetime


# ============================================================
# USER CONFIG
# ============================================================
H5AD_PATH = "/dcl01/hongkai/data/data/hjiang/Data/ATAC.h5ad"
SAMPLE_META_PATH = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/ATAC_Metadata.csv"
OUTPUT_H5AD_PATH = "/dcl01/hongkai/data/data/hjiang/Data/test_ATAC.h5ad"
OUTPUT_SUMMARY_PATH = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/subsample_summary.txt"

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

# Minimum cell count per sample (samples with fewer cells will be excluded)
MIN_CELLS_PER_SAMPLE = 200

# Random seed for reproducibility
# Set to None for truly random selection each run
# Set to an integer (e.g., 42) for reproducible selection
RANDOM_SEED = None

# Whether to require balanced selection within each batch:
#   True  -> try to pick 2 sev1 + 2 sev4 from each of 2 batches
#   False -> only enforce total 4 sev1 + 4 sev4 across 2 batches
#   (Only applies when batch information is available)
REQUIRE_PER_BATCH_BALANCE = False

# Whether to require batch information:
#   True  -> will raise error if no batch column found
#   False -> will proceed without batch constraints if no batch column found
REQUIRE_BATCH = False

# Number of batches to use (only applies when batch information is available)
N_BATCHES = 2
# ============================================================


def get_random_seed():
    """Generate or return random seed for this run."""
    if RANDOM_SEED is None:
        # Generate a random seed based on current time
        seed = int(datetime.now().timestamp() * 1000000) % (2**32)
        return seed
    return RANDOM_SEED


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


def compute_cells_per_sample(adata, sample_col):
    """Compute the number of cells per sample in the AnnData object."""
    cell_counts = adata.obs[sample_col].astype(str).value_counts()
    return cell_counts.to_dict()


def filter_samples_by_cell_count(sample_meta, sample_col, cells_per_sample, min_cells):
    """
    Filter sample metadata to only include samples with at least min_cells.
    
    Returns:
        filtered_meta: DataFrame with only samples meeting the threshold
        excluded_samples: DataFrame with samples that were excluded and their cell counts
    """
    # Add cell count to metadata
    meta = sample_meta.copy()
    meta['_cell_count'] = meta[sample_col].map(cells_per_sample).fillna(0).astype(int)
    
    # Split into included and excluded
    included_mask = meta['_cell_count'] >= min_cells
    filtered_meta = meta[included_mask].copy()
    excluded_meta = meta[~included_mask].copy()
    
    # Remove temporary column from filtered
    filtered_meta = filtered_meta.drop(columns=['_cell_count'])
    
    # Keep cell count info for excluded samples reporting
    excluded_samples = excluded_meta[[sample_col, '_cell_count']].copy()
    excluded_samples.columns = [sample_col, 'cell_count']
    
    return filtered_meta, excluded_samples


def standardize_sample_meta(sample_meta, sample_col, batch_col, sev_col):
    meta = sample_meta.copy()

    if sample_col not in meta.columns:
        raise KeyError(
            f"Sample metadata must contain sample column '{sample_col}'. "
            f"Available columns: {meta.columns.tolist()}"
        )

    meta[sample_col] = meta[sample_col].astype(str).str.strip()
    
    if batch_col is not None:
        meta[batch_col] = meta[batch_col].astype(str).str.strip()

    # Convert severity to numeric if possible
    meta[sev_col] = pd.to_numeric(meta[sev_col], errors="coerce")

    # Drop rows missing critical info
    required_cols = [sample_col, sev_col]
    if batch_col is not None:
        required_cols.append(batch_col)
    meta = meta.dropna(subset=required_cols).copy()

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

    if eligible.shape[0] < N_BATCHES:
        return None, batch_summary

    # Shuffle eligible batches and pick
    eligible_batches = eligible["batch"].values.tolist()
    rng.shuffle(eligible_batches)
    chosen_batches = sorted(eligible_batches[:N_BATCHES])

    selected_rows = []
    for b in chosen_batches:
        row = eligible.loc[eligible["batch"] == b].iloc[0]
        
        # Shuffle and pick samples
        sev1_list = row["sev1_samples"].copy()
        sev4_list = row["sev4_samples"].copy()
        rng.shuffle(sev1_list)
        rng.shuffle(sev4_list)
        sev1_pick = sorted(sev1_list[:2])
        sev4_pick = sorted(sev4_list[:2])

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


def choose_samples_with_batch(meta, sample_col, batch_col, sev_col, rng):
    """
    Relaxed selection with batch constraints:
      - choose N_BATCHES batches
      - total N_SEV1 samples with sev==SEV1_VALUE and N_SEV4 samples with sev==SEV4_VALUE 
        across those batches
    """
    batches = sorted(meta[batch_col].unique().tolist())
    if len(batches) < N_BATCHES:
        raise ValueError(f"Need at least {N_BATCHES} batches, found only {len(batches)}.")

    # Evaluate all pairs (or combinations for N_BATCHES > 2)
    from itertools import combinations
    
    valid_combos = []
    for combo in combinations(batches, N_BATCHES):
        sub = meta[meta[batch_col].isin(combo)].copy()

        sev1_samples = sub.loc[sub[sev_col] == SEV1_VALUE, sample_col].unique().tolist()
        sev4_samples = sub.loc[sub[sev_col] == SEV4_VALUE, sample_col].unique().tolist()

        if len(sev1_samples) >= N_SEV1 and len(sev4_samples) >= N_SEV4:
            valid_combos.append(combo)

    if not valid_combos:
        raise ValueError(
            f"Could not find any combination of {N_BATCHES} batches with enough samples to satisfy "
            f"{N_SEV1} samples of severity {SEV1_VALUE} and "
            f"{N_SEV4} samples of severity {SEV4_VALUE}."
        )

    # Randomly select from valid combinations
    rng.shuffle(valid_combos)
    best_combo = valid_combos[0]

    sub = meta[meta[batch_col].isin(best_combo)].copy()
    sev1_pool = sub.loc[sub[sev_col] == SEV1_VALUE, sample_col].unique().tolist()
    sev4_pool = sub.loc[sub[sev_col] == SEV4_VALUE, sample_col].unique().tolist()

    # Shuffle and pick
    rng.shuffle(sev1_pool)
    rng.shuffle(sev4_pool)
    sev1_pick = sorted(sev1_pool[:N_SEV1])
    sev4_pick = sorted(sev4_pool[:N_SEV4])

    selected_samples = sev1_pick + sev4_pick
    selected_df = sub[sub[sample_col].isin(selected_samples)][[sample_col, batch_col, sev_col]].drop_duplicates().copy()

    return selected_df


def choose_samples_without_batch(meta, sample_col, sev_col, rng):
    """
    Selection without batch constraints:
      - Select N_SEV1 samples with sev==SEV1_VALUE 
      - Select N_SEV4 samples with sev==SEV4_VALUE
      - No batch balancing required
    """
    sev1_pool = meta.loc[meta[sev_col] == SEV1_VALUE, sample_col].unique().tolist()
    sev4_pool = meta.loc[meta[sev_col] == SEV4_VALUE, sample_col].unique().tolist()

    print(f"Available samples with severity {SEV1_VALUE}: {len(sev1_pool)}")
    print(f"Available samples with severity {SEV4_VALUE}: {len(sev4_pool)}")

    if len(sev1_pool) < N_SEV1:
        raise ValueError(
            f"Not enough samples with severity {SEV1_VALUE}. "
            f"Need {N_SEV1}, but only {len(sev1_pool)} available."
        )
    if len(sev4_pool) < N_SEV4:
        raise ValueError(
            f"Not enough samples with severity {SEV4_VALUE}. "
            f"Need {N_SEV4}, but only {len(sev4_pool)} available."
        )

    # Shuffle and pick
    rng.shuffle(sev1_pool)
    rng.shuffle(sev4_pool)
    sev1_pick = sorted(sev1_pool[:N_SEV1])
    sev4_pick = sorted(sev4_pool[:N_SEV4])

    # Build selected_df with only sample and severity columns
    selected_rows = []
    for s in sev1_pick:
        selected_rows.append({
            sample_col: s,
            sev_col: SEV1_VALUE
        })
    for s in sev4_pick:
        selected_rows.append({
            sample_col: s,
            sev_col: SEV4_VALUE
        })
    
    selected_df = pd.DataFrame(selected_rows)
    return selected_df


def generate_summary_report(
    output_path,
    seed_used,
    h5ad_input_path,
    h5ad_output_path,
    meta_input_path,
    original_adata_shape,
    subset_adata_shape,
    sample_col,
    batch_col,
    sev_col,
    use_batch,
    require_per_batch_balance,
    selected_df,
    cells_per_sample,
    original_meta_shape,
    filtered_meta_shape,
    available_sev1_count,
    available_sev4_count,
    min_cells_threshold,
    excluded_samples_df,
    samples_before_cell_filter,
    samples_after_cell_filter,
    available_batches=None,
):
    """Generate a detailed summary report of the subsampling."""
    
    lines = []
    lines.append("=" * 70)
    lines.append("SUBSAMPLING SUMMARY REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Timestamp and seed
    lines.append("RUN INFORMATION")
    lines.append("-" * 40)
    lines.append(f"Timestamp:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Random Seed Used:    {seed_used}")
    lines.append(f"Seed Configured:     {RANDOM_SEED if RANDOM_SEED is not None else 'None (auto-generated)'}")
    lines.append("")
    
    # Input/Output files
    lines.append("FILE PATHS")
    lines.append("-" * 40)
    lines.append(f"Input H5AD:          {h5ad_input_path}")
    lines.append(f"Input Metadata:      {meta_input_path}")
    lines.append(f"Output H5AD:         {h5ad_output_path}")
    lines.append(f"Output Summary:      {output_path}")
    lines.append("")
    
    # Configuration
    lines.append("CONFIGURATION")
    lines.append("-" * 40)
    lines.append(f"Sample Column:       {sample_col}")
    lines.append(f"Batch Column:        {batch_col if batch_col else 'N/A (not used)'}")
    lines.append(f"Severity Column:     {sev_col}")
    lines.append(f"Use Batch:           {use_batch}")
    if use_batch:
        lines.append(f"Require Per-Batch Balance: {require_per_batch_balance}")
        lines.append(f"Number of Batches:   {N_BATCHES}")
    lines.append(f"Target Total Samples: {N_TOTAL_SAMPLES}")
    lines.append(f"Target Severity {SEV1_VALUE} Samples: {N_SEV1}")
    lines.append(f"Target Severity {SEV4_VALUE} Samples: {N_SEV4}")
    lines.append(f"Min Cells/Sample:    {min_cells_threshold}")
    lines.append("")
    
    # Data overview
    lines.append("DATA OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Original AnnData:    {original_adata_shape[0]:,} cells × {original_adata_shape[1]:,} features")
    lines.append(f"Subset AnnData:      {subset_adata_shape[0]:,} cells × {subset_adata_shape[1]:,} features")
    lines.append(f"Cells Retained:      {subset_adata_shape[0]:,} / {original_adata_shape[0]:,} ({100*subset_adata_shape[0]/original_adata_shape[0]:.2f}%)")
    lines.append("")
    lines.append(f"Original Metadata Rows:  {original_meta_shape}")
    lines.append(f"Filtered Metadata Rows:  {filtered_meta_shape} (after matching to AnnData)")
    lines.append("")
    
    # Cell count filtering summary
    lines.append("CELL COUNT FILTERING")
    lines.append("-" * 40)
    lines.append(f"Minimum Cells Required:  {min_cells_threshold}")
    lines.append(f"Samples Before Filter:   {samples_before_cell_filter}")
    lines.append(f"Samples After Filter:    {samples_after_cell_filter}")
    lines.append(f"Samples Excluded:        {len(excluded_samples_df)}")
    
    if len(excluded_samples_df) > 0:
        lines.append("")
        lines.append("Excluded Samples (below cell threshold):")
        lines.append(f"  {'Sample':<40} {'Cells':<10}")
        lines.append(f"  {'-'*40} {'-'*10}")
        for _, row in excluded_samples_df.sort_values('cell_count').iterrows():
            lines.append(f"  {str(row[sample_col]):<40} {row['cell_count']:<10}")
    lines.append("")
    
    # Available pool (after filtering)
    lines.append("AVAILABLE SAMPLE POOL (after cell count filtering)")
    lines.append("-" * 40)
    lines.append(f"Samples with Severity {SEV1_VALUE}: {available_sev1_count}")
    lines.append(f"Samples with Severity {SEV4_VALUE}: {available_sev4_count}")
    if use_batch and available_batches is not None:
        lines.append(f"Available Batches:   {len(available_batches)}")
        for b in sorted(available_batches):
            lines.append(f"  - {b}")
    lines.append("")
    
    # Selected samples
    lines.append("SELECTED SAMPLES")
    lines.append("-" * 40)
    lines.append(f"Total Selected:      {len(selected_df)}")
    
    sev_counts = selected_df[sev_col].value_counts().sort_index()
    for sev_val, count in sev_counts.items():
        lines.append(f"  Severity {int(sev_val)}:       {count} samples")
    
    if use_batch and batch_col in selected_df.columns:
        lines.append("")
        batch_counts = selected_df[batch_col].value_counts().sort_index()
        lines.append(f"Batches Used:        {len(batch_counts)}")
        for batch_name, count in batch_counts.items():
            lines.append(f"  - {batch_name}: {count} samples")
    lines.append("")
    
    # Detailed sample list
    lines.append("DETAILED SAMPLE LIST")
    lines.append("-" * 40)
    
    if use_batch and batch_col in selected_df.columns:
        sorted_df = selected_df.sort_values([batch_col, sev_col, sample_col])
        lines.append(f"{'Sample':<30} {'Batch':<20} {'Severity':<10} {'Cells':<10}")
        lines.append("-" * 70)
        for _, row in sorted_df.iterrows():
            sample_name = str(row[sample_col])
            batch_name = str(row[batch_col])
            severity = int(row[sev_col])
            n_cells = cells_per_sample.get(sample_name, 0)
            lines.append(f"{sample_name:<30} {batch_name:<20} {severity:<10} {n_cells:<10}")
    else:
        sorted_df = selected_df.sort_values([sev_col, sample_col])
        lines.append(f"{'Sample':<40} {'Severity':<10} {'Cells':<10}")
        lines.append("-" * 60)
        for _, row in sorted_df.iterrows():
            sample_name = str(row[sample_col])
            severity = int(row[sev_col])
            n_cells = cells_per_sample.get(sample_name, 0)
            lines.append(f"{sample_name:<40} {severity:<10} {n_cells:<10}")
    
    lines.append("")
    
    # Cell count statistics for selected samples
    lines.append("CELL COUNT STATISTICS (selected samples)")
    lines.append("-" * 40)
    selected_sample_names = selected_df[sample_col].astype(str).tolist()
    cell_counts = [cells_per_sample.get(s, 0) for s in selected_sample_names]
    lines.append(f"Total Cells:         {sum(cell_counts):,}")
    lines.append(f"Mean Cells/Sample:   {np.mean(cell_counts):,.1f}")
    lines.append(f"Median Cells/Sample: {np.median(cell_counts):,.1f}")
    lines.append(f"Min Cells/Sample:    {min(cell_counts):,}")
    lines.append(f"Max Cells/Sample:    {max(cell_counts):,}")
    lines.append(f"Std Dev:             {np.std(cell_counts):,.1f}")
    lines.append("")
    
    # Cells by severity
    lines.append("CELLS BY SEVERITY")
    lines.append("-" * 40)
    for sev_val in sorted(selected_df[sev_col].unique()):
        sev_samples = selected_df.loc[selected_df[sev_col] == sev_val, sample_col].astype(str).tolist()
        sev_cells = sum(cells_per_sample.get(s, 0) for s in sev_samples)
        lines.append(f"Severity {int(sev_val)}:         {sev_cells:,} cells ({len(sev_samples)} samples)")
    lines.append("")
    
    # Cells by batch (if applicable)
    if use_batch and batch_col in selected_df.columns:
        lines.append("CELLS BY BATCH")
        lines.append("-" * 40)
        for batch_name in sorted(selected_df[batch_col].unique()):
            batch_samples = selected_df.loc[selected_df[batch_col] == batch_name, sample_col].astype(str).tolist()
            batch_cells = sum(cells_per_sample.get(s, 0) for s in batch_samples)
            lines.append(f"{batch_name}:  {batch_cells:,} cells ({len(batch_samples)} samples)")
        lines.append("")
    
    # Reproducibility note
    lines.append("REPRODUCIBILITY")
    lines.append("-" * 40)
    if RANDOM_SEED is not None:
        lines.append(f"To reproduce this exact selection, use RANDOM_SEED = {RANDOM_SEED}")
    else:
        lines.append(f"To reproduce this exact selection, set RANDOM_SEED = {seed_used}")
    lines.append("")
    
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    # Also print to console
    print('\n'.join(lines))


def main():
    # Get random seed (either configured or auto-generated)
    seed_used = get_random_seed()
    rng = np.random.default_rng(seed_used)
    
    print(f"Using random seed: {seed_used}")
    if RANDOM_SEED is None:
        print("(Auto-generated seed - set RANDOM_SEED in config for reproducibility)")

    print("\n=== Reading AnnData ===")
    adata = sc.read_h5ad(H5AD_PATH)
    original_adata_shape = adata.shape
    print(f"AnnData shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if SAMPLE_COLUMN not in adata.obs.columns:
        raise KeyError(
            f"'{SAMPLE_COLUMN}' not found in adata.obs. "
            f"Available obs columns: {adata.obs.columns.tolist()}"
        )

    # Compute cells per sample from AnnData
    print("\n=== Computing cells per sample ===")
    cells_per_sample = compute_cells_per_sample(adata, SAMPLE_COLUMN)
    print(f"Total unique samples in AnnData: {len(cells_per_sample)}")
    print(f"Cell count range: {min(cells_per_sample.values())} - {max(cells_per_sample.values())}")

    print("\n=== Reading sample metadata ===")
    sample_meta = pd.read_csv(SAMPLE_META_PATH)
    original_meta_shape = sample_meta.shape[0]
    print(f"Sample metadata shape: {sample_meta.shape}")

    # Try to detect batch column (may be None if not found and not required)
    batch_col = detect_column(
        sample_meta,
        BATCH_CANDIDATE_COLUMNS,
        required=REQUIRE_BATCH,
        label="batch/study column"
    )
    
    sev_col = detect_column(
        sample_meta,
        SEVERITY_CANDIDATE_COLUMNS,
        required=True,
        label="severity column"
    )

    # Determine if we're using batch-based selection
    use_batch = batch_col is not None
    
    if use_batch:
        print(f"Detected batch/study column: {batch_col}")
    else:
        print("No batch/study column detected. Will proceed without batch constraints.")
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
    filtered_meta_shape = sample_meta.shape[0]

    print("\n=== Metadata after filtering to samples present in AnnData ===")
    print(f"Number of metadata rows kept: {sample_meta.shape[0]}")

    # Apply cell count filter
    print(f"\n=== Filtering samples with < {MIN_CELLS_PER_SAMPLE} cells ===")
    samples_before_cell_filter = sample_meta.shape[0]
    
    sample_meta, excluded_samples_df = filter_samples_by_cell_count(
        sample_meta=sample_meta,
        sample_col=SAMPLE_COLUMN,
        cells_per_sample=cells_per_sample,
        min_cells=MIN_CELLS_PER_SAMPLE
    )
    
    samples_after_cell_filter = sample_meta.shape[0]
    
    print(f"Samples before cell filter: {samples_before_cell_filter}")
    print(f"Samples after cell filter:  {samples_after_cell_filter}")
    print(f"Samples excluded:           {len(excluded_samples_df)}")
    
    if len(excluded_samples_df) > 0:
        print("\nExcluded samples (below cell threshold):")
        for _, row in excluded_samples_df.sort_values('cell_count').head(10).iterrows():
            print(f"  {row[SAMPLE_COLUMN]}: {row['cell_count']} cells")
        if len(excluded_samples_df) > 10:
            print(f"  ... and {len(excluded_samples_df) - 10} more")

    # Track available pool for summary (after cell count filtering)
    available_sev1_count = len(sample_meta[sample_meta[sev_col] == SEV1_VALUE])
    available_sev4_count = len(sample_meta[sample_meta[sev_col] == SEV4_VALUE])
    available_batches = sample_meta[batch_col].unique().tolist() if use_batch else None

    print("\n=== Sample pool after all filtering ===")
    if use_batch:
        print(f"Unique batches/studies: {sample_meta[batch_col].nunique()}")
    print("Severity counts by sample:")
    print(sample_meta[sev_col].value_counts(dropna=False).sort_index())

    # Check if we have enough samples after filtering
    if available_sev1_count < N_SEV1:
        raise ValueError(
            f"After filtering, only {available_sev1_count} samples with severity {SEV1_VALUE} remain. "
            f"Need at least {N_SEV1}. Consider lowering MIN_CELLS_PER_SAMPLE."
        )
    if available_sev4_count < N_SEV4:
        raise ValueError(
            f"After filtering, only {available_sev4_count} samples with severity {SEV4_VALUE} remain. "
            f"Need at least {N_SEV4}. Consider lowering MIN_CELLS_PER_SAMPLE."
        )

    print("\n=== Trying to choose samples ===")
    selected_df = None

    if use_batch:
        # Batch-based selection
        if REQUIRE_PER_BATCH_BALANCE:
            selected_df, batch_summary = choose_samples_balanced_per_batch(
                meta=sample_meta,
                sample_col=SAMPLE_COLUMN,
                batch_col=batch_col,
                sev_col=sev_col,
                rng=rng,
            )

            print("Per-batch summary:")
            print(batch_summary[["batch", "n_sev1", "n_sev4"]].sort_values("batch").to_string(index=False))

            if selected_df is None:
                print("\nCould not satisfy strict per-batch balance (2 sev1 + 2 sev4 in each batch).")
                print("Falling back to relaxed selection across batches.")
                selected_df = choose_samples_with_batch(
                    meta=sample_meta,
                    sample_col=SAMPLE_COLUMN,
                    batch_col=batch_col,
                    sev_col=sev_col,
                    rng=rng,
                )
        else:
            selected_df = choose_samples_with_batch(
                meta=sample_meta,
                sample_col=SAMPLE_COLUMN,
                batch_col=batch_col,
                sev_col=sev_col,
                rng=rng,
            )
    else:
        # No batch - simple severity-based selection
        print("Selecting samples based on severity only (no batch constraints).")
        selected_df = choose_samples_without_batch(
            meta=sample_meta,
            sample_col=SAMPLE_COLUMN,
            sev_col=sev_col,
            rng=rng,
        )

    selected_df = selected_df.drop_duplicates(subset=[SAMPLE_COLUMN]).copy()

    # Final checks
    if selected_df.shape[0] != N_TOTAL_SAMPLES:
        raise ValueError(
            f"Expected {N_TOTAL_SAMPLES} selected samples, but got {selected_df.shape[0]}."
        )

    if use_batch:
        batch_n = selected_df[batch_col].nunique()
        if batch_n != N_BATCHES:
            raise ValueError(f"Expected {N_BATCHES} batches/studies, but got {batch_n}.")

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
    os.makedirs(os.path.dirname(OUTPUT_SUMMARY_PATH), exist_ok=True)
    
    adata_sub.write_h5ad(OUTPUT_H5AD_PATH)
    print(f"Saved subset h5ad: {OUTPUT_H5AD_PATH}")

    # Generate detailed summary report
    generate_summary_report(
        output_path=OUTPUT_SUMMARY_PATH,
        seed_used=seed_used,
        h5ad_input_path=H5AD_PATH,
        h5ad_output_path=OUTPUT_H5AD_PATH,
        meta_input_path=SAMPLE_META_PATH,
        original_adata_shape=original_adata_shape,
        subset_adata_shape=adata_sub.shape,
        sample_col=SAMPLE_COLUMN,
        batch_col=batch_col,
        sev_col=sev_col,
        use_batch=use_batch,
        require_per_batch_balance=REQUIRE_PER_BATCH_BALANCE,
        selected_df=selected_df,
        cells_per_sample=cells_per_sample,
        original_meta_shape=original_meta_shape,
        filtered_meta_shape=filtered_meta_shape,
        available_sev1_count=available_sev1_count,
        available_sev4_count=available_sev4_count,
        min_cells_threshold=MIN_CELLS_PER_SAMPLE,
        excluded_samples_df=excluded_samples_df,
        samples_before_cell_filter=samples_before_cell_filter,
        samples_after_cell_filter=samples_after_cell_filter,
        available_batches=available_batches,
    )
    
    print(f"\nSaved summary report: {OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()