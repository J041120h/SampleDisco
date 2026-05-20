"""
Build pbmc_benchmark_ready.h5ad by joining

  raw_counts_h5ad/pbmc_gex_raw_with_var_obs.h5ad   (raw integer CSR counts, 3 obs cols)
  all_pbmcs/all_pbmcs_metadata.csv                  (full 20-col cell metadata)

Strategy
  - Copy the raw .h5ad with shutil.copyfile so the heavy /X group is left
    untouched (~72 GB I/O once, no X decode).
  - Open the copy with h5py r+, delete /obs, and rewrite /obs with the
    metadata reindexed to the file's existing obs_names order.
  - Never load X into memory.
"""

import shutil
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

# anndata moved write_elem around between versions
try:
    from anndata._io.specs import write_elem  # anndata >= 0.10
except Exception:
    from anndata.experimental import write_elem  # anndata 0.9

DATA = Path("/dcs07/hongkai/data/harry/result/health_aging_PBMC")
SRC = DATA / "raw_counts_h5ad" / "pbmc_gex_raw_with_var_obs.h5ad"
META_CSV = DATA / "all_pbmcs" / "all_pbmcs_metadata.csv"

OUT_DIR = DATA / "benchmark_ready"
OUT_DIR.mkdir(exist_ok=True)
DST = OUT_DIR / "pbmc_benchmark_ready.h5ad"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"src  = {SRC}  ({SRC.stat().st_size / 1e9:.2f} GB)")
    log(f"meta = {META_CSV}")
    log(f"dst  = {DST}")

    # ------------------------------------------------------------------
    # 1. read the cell-order from the raw h5ad without touching X
    # ------------------------------------------------------------------
    log("reading obs_names from source h5ad via h5py ...")
    with h5py.File(SRC, "r") as h:
        idx_field = h["obs"].attrs.get("_index", "_index")
        obs_names = h["obs"][idx_field][:].astype(str)
        var_names_field = h["var"].attrs.get("_index", "_index")
        var_names = h["var"][var_names_field][:].astype(str)
        x_nnz = h["X"]["data"].shape[0]
    log(f"  source has n_obs={len(obs_names):,}  n_var={len(var_names):,}  X.nnz={x_nnz:,}")

    # ------------------------------------------------------------------
    # 2. read the metadata CSV and reindex to match the source order
    # ------------------------------------------------------------------
    log("reading metadata CSV ...")
    meta = pd.read_csv(META_CSV, low_memory=False)
    meta.rename(columns={meta.columns[0]: "cell_barcode"}, inplace=True)
    log(f"  metadata: shape={meta.shape}  cols={list(meta.columns)}")

    # drop useless 'orig.ident' column (single value) -- keep everything else
    if (meta["orig.ident"].nunique() if "orig.ident" in meta.columns else 99) == 1:
        meta = meta.drop(columns=["orig.ident"])
        log("  dropped 'orig.ident' (single value)")

    meta = meta.set_index("cell_barcode")
    if not meta.index.is_unique:
        raise RuntimeError("cell_barcode in metadata CSV is not unique")

    missing = set(obs_names) - set(meta.index)
    extra = set(meta.index) - set(obs_names)
    log(f"  obs in source but missing from metadata: {len(missing)}")
    log(f"  obs in metadata but missing from source: {len(extra)}")
    if missing:
        raise RuntimeError(f"{len(missing)} source cells have no metadata row")

    # reindex to source order
    log("reindexing metadata to source obs_names order ...")
    obs_df = meta.reindex(obs_names)
    obs_df.index.name = None  # anndata convention

    # nice categorical dtypes for low-cardinality columns
    for col in ["Donor_id", "Tube_id", "Batch", "File_name",
                "Age_group", "Sex", "Cluster_names"]:
        if col in obs_df.columns:
            obs_df[col] = obs_df[col].astype("category")
    log(f"  obs ready: shape={obs_df.shape}  dtypes:")
    for c, d in obs_df.dtypes.items():
        nun = obs_df[c].nunique()
        print(f"    {c:<20s}  {str(d):<12s}  nunique={nun}")

    # ------------------------------------------------------------------
    # 3. copy the source file, then patch /obs
    # ------------------------------------------------------------------
    if DST.exists():
        log(f"removing existing {DST}")
        DST.unlink()

    log(f"copying source -> dst ({SRC.stat().st_size / 1e9:.1f} GB) ...")
    t0 = time.time()
    shutil.copyfile(SRC, DST)
    log(f"  copy done in {time.time() - t0:.1f}s")

    log("patching /obs in dst with merged metadata ...")
    with h5py.File(DST, "r+") as h:
        if "obs" in h:
            del h["obs"]
        write_elem(h, "obs", obs_df)
        # also embed a short provenance note
        if "uns" not in h:
            h.create_group("uns")
        h["uns"].attrs["benchmark_ready_provenance"] = (
            "X from raw_counts_h5ad/pbmc_gex_raw_with_var_obs.h5ad (integer CSR counts); "
            "obs merged from all_pbmcs/all_pbmcs_metadata.csv; "
            "built by build_benchmark_ready_h5ad.py"
        )
    log("patch done")

    # ------------------------------------------------------------------
    # 4. verify
    # ------------------------------------------------------------------
    log("verifying ...")
    a = ad.read_h5ad(DST, backed="r")
    print(f"  dst shape = {a.shape}")
    print(f"  dst obs columns ({len(a.obs.columns)}): {list(a.obs.columns)}")
    print(f"  dst obs head:")
    print(a.obs.head(3).to_string())
    print(f"  dst uns: {dict(a.uns) if hasattr(a, 'uns') else 'n/a'}")
    a.file.close()

    log(f"DONE. wrote {DST}  ({DST.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
