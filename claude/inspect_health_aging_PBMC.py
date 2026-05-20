"""
Sample-embedding benchmark pre-checklist for the Terekhova 2023 healthy-aging
PBMC dataset (Synapse syn49637038).

Answers the 5/7 questions from the user's checklist:
  1. Sample unit?      2. Biology variable?   3. Batch variable?
  4. Biology x batch confounding?  5. Cells per sample distribution?
  6. Are raw counts present?       7. What biology to preserve?
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

DATA_DIR = Path("/dcs07/hongkai/data/harry/result/health_aging_PBMC")
RAW_H5AD = DATA_DIR / "raw_counts_h5ad" / "pbmc_gex_raw_with_var_obs.h5ad"
ALL_H5AD = DATA_DIR / "all_pbmcs" / "all_pbmcs_rna.h5ad"
ALL_HARMONY_H5AD = DATA_DIR / "all_pbmcs" / "all_pbmcs_rna_harmony.h5ad"
ALL_META = DATA_DIR / "all_pbmcs" / "all_pbmcs_metadata.csv"
ALL_UMAP = DATA_DIR / "all_pbmcs" / "all_pbmcs_umap.csv"


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def peek_h5ad(path: Path):
    """Open backed; print shape, obs/var columns, X dtype + raw-ness."""
    if not path.exists():
        print(f"[missing] {path}")
        return None
    print(f"[opening] {path}  ({path.stat().st_size / 1e9:.2f} GB)")
    a = ad.read_h5ad(path, backed="r")
    print(f"  shape (cells, genes) = {a.shape}")
    print(f"  X dtype = {a.X.dtype}  layers = {list(a.layers.keys())}")
    print(f"  obs columns ({len(a.obs.columns)}):")
    for c in a.obs.columns:
        s = a.obs[c]
        nun = s.nunique(dropna=True)
        sample_vals = s.dropna().unique()[:5]
        print(f"    - {c:<32s}  nunique={nun:<7d}  dtype={str(s.dtype):<12s} "
              f"e.g. {list(sample_vals)}")
    print(f"  var columns ({len(a.var.columns)}): {list(a.var.columns)}")
    print(f"  uns keys: {list(a.uns.keys())}")
    print(f"  obsm keys: {list(a.obsm.keys())}")
    return a


def raw_count_check(a):
    """Check first few X values to confirm they look like integer counts."""
    try:
        chunk = a.X[:200].toarray() if hasattr(a.X[:200], "toarray") else np.array(a.X[:200])
    except Exception as e:
        print(f"  [raw-check error] {e}")
        return
    nz = chunk[chunk > 0]
    if nz.size == 0:
        print("  X first 200 rows are all zero — cannot verify counts vs normalized")
        return
    frac_int = (nz == np.floor(nz)).mean()
    print(f"  first-200-rows nonzero stats: min={nz.min():.4g}  max={nz.max():.4g}  "
          f"mean={nz.mean():.4g}  fraction-integer={frac_int:.3f}")
    print(f"  -> looks like {'COUNTS' if frac_int > 0.99 else 'NORMALIZED'}")


def benchmark_checklist(obs: pd.DataFrame):
    banner("BENCHMARK CHECKLIST")

    # heuristic column detection
    def find_col(cands):
        for c in cands:
            for col in obs.columns:
                if col.lower() == c.lower():
                    return col
        for c in cands:
            for col in obs.columns:
                if c.lower() in col.lower():
                    return col
        return None

    sample_col = find_col(["sample_id", "sample", "library", "library_id"])
    donor_col = find_col(["donor_id", "donor", "subject", "patient", "patient_id"])
    batch_col = find_col(["batch", "batch_id", "pool", "library_batch", "seq_batch"])
    age_col = find_col(["age", "age_years", "age_at_visit"])
    age_group_col = find_col(["age_group", "agegroup", "age_bin"])
    sex_col = find_col(["sex", "gender"])
    visit_col = find_col(["visit", "timepoint"])
    celltype_col = find_col(["cell_type", "celltype", "annotation", "cluster", "leiden"])

    detected = dict(sample=sample_col, donor=donor_col, batch=batch_col,
                    age=age_col, age_group=age_group_col, sex=sex_col,
                    visit=visit_col, celltype=celltype_col)
    print("Detected columns:")
    for k, v in detected.items():
        print(f"  {k:<10s} -> {v}")

    # 1. sample unit
    banner("Q1. Sample unit")
    if sample_col is None:
        print("  !! No sample_id column detected -- REJECT")
    else:
        n_samp = obs[sample_col].nunique()
        print(f"  n_samples (unique {sample_col}) = {n_samp}")
    if donor_col is not None:
        print(f"  n_donors (unique {donor_col})  = {obs[donor_col].nunique()}")
        if sample_col is not None and donor_col is not None:
            visits_per_donor = obs[[donor_col, sample_col]].drop_duplicates().groupby(donor_col).size()
            print(f"  visits per donor: min={visits_per_donor.min()} median={visits_per_donor.median():.0f} "
                  f"max={visits_per_donor.max()} mean={visits_per_donor.mean():.2f}")

    # 2. biology
    banner("Q2. Biology variable")
    if age_col is not None:
        ages = pd.to_numeric(obs[age_col], errors="coerce").dropna()
        print(f"  {age_col}: n={len(ages)} min={ages.min()} median={ages.median()} "
              f"max={ages.max()} mean={ages.mean():.1f}")
    if age_group_col is not None:
        print(f"  {age_group_col} value counts:")
        print(obs[age_group_col].value_counts(dropna=False).to_string())

    # 3. batch
    banner("Q3. Batch variable")
    if batch_col is None:
        print("  !! No batch column detected -- looking for likely candidates")
        for col in obs.columns:
            if obs[col].nunique() < 50 and obs[col].nunique() > 1:
                print(f"    candidate: {col} (nunique={obs[col].nunique()})")
    else:
        print(f"  {batch_col} nunique = {obs[batch_col].nunique()}")
        print(obs[batch_col].value_counts(dropna=False).head(20).to_string())

    # 4. biology x batch confounding
    banner("Q4. Biology x batch confounding")
    if sample_col is not None and batch_col is not None:
        sb = obs[[sample_col, batch_col]].drop_duplicates()
        samps_per_batch = sb.groupby(batch_col).size()
        print(f"  samples per batch: min={samps_per_batch.min()} median={samps_per_batch.median():.0f} "
              f"max={samps_per_batch.max()}")

        # is batch == sample?
        if sb[sample_col].nunique() == sb[batch_col].nunique():
            print(f"  !! batch_id appears 1-to-1 with sample_id -- REJECT")

    if batch_col is not None and age_col is not None and sample_col is not None:
        sample_meta = obs[[sample_col, batch_col, age_col]].drop_duplicates(subset=[sample_col])
        sample_meta[age_col] = pd.to_numeric(sample_meta[age_col], errors="coerce")
        print("\n  age distribution per batch (sample-level):")
        print(sample_meta.groupby(batch_col)[age_col].describe()[
            ["count", "mean", "std", "min", "max"]].round(1).to_string())

    if batch_col is not None and age_group_col is not None and sample_col is not None:
        sample_meta = obs[[sample_col, batch_col, age_group_col]].drop_duplicates(subset=[sample_col])
        ct = pd.crosstab(sample_meta[age_group_col], sample_meta[batch_col])
        print("\n  age_group x batch crosstab (sample-level):")
        print(ct.to_string())

    # 5. cells per sample
    banner("Q5. Cells per sample")
    if sample_col is not None:
        cps = obs.groupby(sample_col).size()
        print(f"  n_samples = {cps.size}")
        print(f"  cells/sample: min={cps.min()} q05={int(cps.quantile(0.05))} "
              f"median={int(cps.median())} mean={cps.mean():.0f} "
              f"q95={int(cps.quantile(0.95))} max={cps.max()}")
        print(f"  samples with <300 cells:  {(cps < 300).sum()}")
        print(f"  samples with <500 cells:  {(cps < 500).sum()}")
        print(f"  samples with <1000 cells: {(cps < 1000).sum()}")


def main():
    banner("FILES ON DISK")
    for p in [RAW_H5AD, ALL_H5AD, ALL_HARMONY_H5AD, ALL_META, ALL_UMAP]:
        if p.exists():
            print(f"  {p.stat().st_size/1e9:7.2f} GB  {p}")
        else:
            print(f"  [missing]       {p}")

    banner("RAW COUNTS  pbmc_gex_raw_with_var_obs.h5ad")
    a_raw = peek_h5ad(RAW_H5AD)
    if a_raw is not None:
        raw_count_check(a_raw)
        obs = a_raw.obs.copy()
        benchmark_checklist(obs)
        a_raw.file.close()

    banner("ALL_PBMCS  all_pbmcs_rna.h5ad")
    a_all = peek_h5ad(ALL_H5AD)
    if a_all is not None:
        raw_count_check(a_all)
        a_all.file.close()

    banner("ALL_PBMCS  all_pbmcs_rna_harmony.h5ad")
    a_h = peek_h5ad(ALL_HARMONY_H5AD)
    if a_h is not None:
        a_h.file.close()

    banner("ALL_PBMCS  metadata CSV")
    if ALL_META.exists():
        meta = pd.read_csv(ALL_META, low_memory=False)
        print(f"  shape = {meta.shape}")
        print(f"  columns ({len(meta.columns)}): {list(meta.columns)}")
        print(meta.head(3).to_string())


if __name__ == "__main__":
    main()
