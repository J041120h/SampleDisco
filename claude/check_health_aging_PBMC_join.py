"""
Follow-up to inspect_health_aging_PBMC.py:

1. Confirm cell-barcode alignment between the raw h5ad and the metadata CSV.
2. Spot-check whether X in the raw h5ad is actually integer counts
   (using h5py directly to avoid a known anndata-backed slicing bug).
3. Full sample-embedding benchmark checklist on the metadata CSV.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DATA = Path("/dcs07/hongkai/data/harry/result/health_aging_PBMC")
RAW = DATA / "raw_counts_h5ad" / "pbmc_gex_raw_with_var_obs.h5ad"
META_CSV = DATA / "all_pbmcs" / "all_pbmcs_metadata.csv"


def banner(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ----------------------------------------------------------------------------
# 1. barcode alignment + count-vs-normalized check via raw h5py
# ----------------------------------------------------------------------------
banner("1. Inspect raw h5ad via h5py (sidesteps anndata backed slicing bug)")
with h5py.File(RAW, "r") as h:
    print(f"  top-level keys: {list(h.keys())}")
    if "X" in h:
        Xg = h["X"]
        print(f"  X keys: {list(Xg.keys()) if isinstance(Xg, h5py.Group) else 'dataset'}")
        if isinstance(Xg, h5py.Group):
            for k in Xg.keys():
                ds = Xg[k]
                print(f"    X/{k}: shape={ds.shape} dtype={ds.dtype}")
            data = Xg["data"][:2000]
            print(f"\n  first 2000 nonzero entries of X/data:")
            print(f"    min={data.min():.4g}  max={data.max():.4g}  mean={data.mean():.3g}")
            frac_int = float(np.mean(data == np.floor(data)))
            print(f"    fraction integer = {frac_int:.4f}")
            print(f"    -> {'COUNTS (integer)' if frac_int > 0.999 else 'NOT pure integer counts'}")
            print(f"    sample values: {data[:20].tolist()}")
        else:
            print(f"  X is a dense dataset; first 100 nonzero values:")
            arr = Xg[0:5, :].ravel()
            arr = arr[arr != 0]
            print(f"    min={arr.min()} max={arr.max()} dtype={arr.dtype}")
    # obs barcode list
    if "obs" in h:
        obsg = h["obs"]
        print(f"\n  obs keys: {list(obsg.keys())}")
        idx_name = obsg.attrs.get("_index", "_index")
        print(f"  obs index field = {idx_name!r}")
        idx = obsg[idx_name][:].astype(str)
        print(f"  obs n={idx.size}  first 3 = {idx[:3].tolist()}")
    if "var" in h:
        varg = h["var"]
        print(f"\n  var keys: {list(varg.keys())}")
        v_idx = varg.attrs.get("_index", "_index")
        vidx = varg[v_idx][:].astype(str)
        print(f"  var n={vidx.size}  first 3 = {vidx[:3].tolist()}")

# ----------------------------------------------------------------------------
# 2. CSV  +  alignment with raw obs_names
# ----------------------------------------------------------------------------
banner("2. Barcode alignment: raw h5ad obs vs metadata CSV")
meta = pd.read_csv(META_CSV, low_memory=False)
meta.rename(columns={meta.columns[0]: "cell_barcode"}, inplace=True)
print(f"  CSV n_rows = {len(meta):,}")
print(f"  CSV first barcodes: {meta['cell_barcode'].iloc[:3].tolist()}")

raw_idx = pd.Index(idx)
csv_idx = pd.Index(meta["cell_barcode"].astype(str))
inter = raw_idx.intersection(csv_idx)
print(f"  intersection size = {len(inter):,}")
print(f"  same order? {(raw_idx.values == csv_idx.values).all() if len(raw_idx) == len(csv_idx) else 'lengths differ'}")
order_match = (raw_idx.values == csv_idx.values).all() if len(raw_idx) == len(csv_idx) else False
if not order_match:
    # measure how scrambled they are
    pos = pd.Series(np.arange(len(csv_idx)), index=csv_idx)
    csv_pos_for_raw = pos.reindex(raw_idx).values
    print(f"  ordering differs; CSV positions for raw[:5] = {csv_pos_for_raw[:5]}")

# ----------------------------------------------------------------------------
# 3. Sample-level benchmark checklist from the CSV
# ----------------------------------------------------------------------------
banner("3. Sample-embedding benchmark checklist (from CSV)")
print(f"  n_cells = {len(meta):,}")
print(f"  n_donors  (Donor_id)   = {meta['Donor_id'].nunique()}")
print(f"  n_samples (Tube_id)    = {meta['Tube_id'].nunique()}")
print(f"  n_batches (Batch)      = {meta['Batch'].nunique()}")
print(f"  n_libraries (File_name)= {meta['File_name'].nunique()}")

sample = (meta.drop_duplicates("Tube_id")
              .set_index("Tube_id")[["Donor_id", "Age", "Age_group", "Sex", "Batch", "File_name"]])
print(f"\n  per-sample table shape = {sample.shape}")
print(f"\n  visits per donor (count distribution):")
print(sample.groupby("Donor_id").size().value_counts().sort_index().to_string())

print("\n  Age (sample-level) summary:")
print(sample["Age"].describe().round(1).to_string())

print("\n  Age_group (sample-level) value_counts:")
print(sample["Age_group"].value_counts().sort_index().to_string())

print("\n  Sex (sample-level) value_counts:")
print(sample["Sex"].value_counts().to_string())

print("\n  Batch (sample-level) value_counts:")
print(sample["Batch"].value_counts().sort_index().to_string())

banner("4. Biology x batch confounding")
print("  Age distribution per batch (sample-level):")
print(sample.groupby("Batch")["Age"].describe()[["count", "mean", "std", "min", "max"]].round(1).to_string())

print("\n  Age_group x Batch crosstab (sample counts):")
ct = pd.crosstab(sample["Age_group"], sample["Batch"])
print(ct.to_string())
print(f"\n  age groups present per batch (out of 5):")
print((ct > 0).sum(axis=0).to_string())
print(f"\n  batches present per age group (out of 14):")
print((ct > 0).sum(axis=1).to_string())

print("\n  Sex x Batch crosstab:")
print(pd.crosstab(sample["Sex"], sample["Batch"]).to_string())

# Confounding metric: how well does batch predict age?
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
X = OneHotEncoder(sparse_output=False).fit_transform(sample[["Batch"]])
y = sample["Age"].astype(float).values
r2 = LinearRegression().fit(X, y).score(X, y)
print(f"\n  R^2(age ~ batch_one_hot) at sample level = {r2:.3f}  "
      f"(near 0 = batch tells you nothing about age; near 1 = perfectly confounded)")

banner("5. Cells per sample")
cps = meta.groupby("Tube_id").size()
print(f"  n_samples = {cps.size}")
print(f"  cells/sample: min={cps.min()} q05={int(cps.quantile(0.05))} "
      f"q25={int(cps.quantile(0.25))} median={int(cps.median())} mean={cps.mean():.0f} "
      f"q75={int(cps.quantile(0.75))} q95={int(cps.quantile(0.95))} max={cps.max()}")
print(f"  samples with <300 cells:  {(cps < 300).sum()}")
print(f"  samples with <500 cells:  {(cps < 500).sum()}")
print(f"  samples with <1000 cells: {(cps < 1000).sum()}")

# How is sample×donor structured?  (multiple Tubes per donor = repeated visits)
banner("6. Sample x Donor structure (visits)")
visits = sample.groupby("Donor_id").size()
print(f"  donors with 1 visit:  {(visits == 1).sum()}")
print(f"  donors with 2 visits: {(visits == 2).sum()}")
print(f"  donors with 3 visits: {(visits == 3).sum()}")
print(f"  donors with 4+:       {(visits >= 4).sum()}")

print("\nDone.")
