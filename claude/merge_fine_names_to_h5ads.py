"""
Append fine-resolution cluster annotations (from per-cell-type Synapse CSVs)
into three h5ad files via in-place h5py writes (no X rewrite).

Source: /dcs07/.../data/celltype_annotations/*/[*_metadata.csv]
Targets:
  - data/benchmark_ready/pbmc_benchmark_ready.h5ad      (pipeline input)
  - round1_batch/preprocess/adata_preprocessed.h5ad     (canonical preprocessed)
  - round1_batch/preprocess/adata_preprocessed_hvg.h5ad (HVG copy for R)

Adds four obs columns:
  - Cluster_fine_numbers       (Int32 nullable; -1 -> NA)
  - Cluster_fine_names         (Categorical; broad name for cells absent from any CSV)
  - Cluster_helper_memory_numbers  (Int32 nullable; only for CD4+ helper memory)
  - Cluster_helper_memory_names    (Categorical; NA for non-helper-memory)
"""

import glob
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

CSV_DIR = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/data/celltype_annotations"

DIR_TO_BROAD = {
    "b_cells": "B cells",
    "cd4_t_cells": "CD4+ T cells",
    "conventional_cd8_t_cells": "TRAV1-2- CD8+ T cells",
    "gd_t_cells": "gd T cells",
    "mait_cells": "MAIT cells",
    "myeloid_cells": "Myeloid cells",
    "nk_cells": "NK cells",
    "progenitor_cells": "Progenitor cells",
}

TARGETS = [
    "/dcs07/hongkai/data/harry/result/health_aging_PBMC/data/benchmark_ready/pbmc_benchmark_ready.h5ad",
    "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/preprocess/adata_preprocessed.h5ad",
    "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/preprocess/adata_preprocessed_hvg.h5ad",
]


def build_per_cell_lookup():
    """Return (fine_dict, helper_dict). Keys = barcode, values = (num, name)."""
    fine = {}
    helper = {}

    for d, broad in DIR_TO_BROAD.items():
        csv = glob.glob(f"{CSV_DIR}/{d}/*_metadata.csv")[0]
        print(f"  reading {os.path.basename(csv)} ...", flush=True)
        df = pd.read_csv(csv)
        bc_col = df.columns[0]  # first column is unnamed = barcode
        has_names = "Cluster_names" in df.columns
        nums = df["Cluster_numbers"].to_numpy()
        bcs = df[bc_col].to_numpy()
        if has_names:
            names = df["Cluster_names"].to_numpy()
        else:
            names = np.array([f"{broad}_{int(n)}" for n in nums], dtype=object)
        for bc, n, nm in zip(bcs, nums, names):
            fine[bc] = (int(n), str(nm))

    # CD4 helper memory level-2 (overlay on top of CD4+ T cells)
    csv = glob.glob(f"{CSV_DIR}/cd4_t_helper_memory_cells/*_metadata.csv")[0]
    print(f"  reading {os.path.basename(csv)} (level-2) ...", flush=True)
    df = pd.read_csv(csv)
    bc_col = df.columns[0]
    for bc, n, nm in zip(df[bc_col], df["Cluster_numbers"], df["Cluster_names"]):
        helper[bc] = (int(n), str(nm))

    return fine, helper


def _delete_if_exists(obs, name):
    if name in obs:
        del obs[name]


def _write_nullable_int(obs, name, values, mask):
    """Write Int32 nullable per anndata 0.8+ format."""
    _delete_if_exists(obs, name)
    g = obs.create_group(name)
    g.attrs["encoding-type"] = "nullable-integer"
    g.attrs["encoding-version"] = "0.1.0"
    g.create_dataset("values", data=values.astype(np.int32))
    g.create_dataset("mask", data=mask.astype(bool))


def _write_categorical(obs, name, categories, codes, ordered=False):
    """Write Categorical per anndata 0.8+ format. codes=-1 means NA."""
    _delete_if_exists(obs, name)
    g = obs.create_group(name)
    g.attrs["encoding-type"] = "categorical"
    g.attrs["encoding-version"] = "0.2.0"
    g.attrs["ordered"] = ordered

    str_dt = h5py.string_dtype("utf-8")
    cats = g.create_dataset(
        "categories", data=np.array(categories, dtype=object), dtype=str_dt
    )
    cats.attrs["encoding-type"] = "string-array"
    cats.attrs["encoding-version"] = "0.2.0"

    code_dt = np.int8 if len(categories) < 127 else np.int16
    cs = g.create_dataset("codes", data=codes.astype(code_dt))
    cs.attrs["encoding-type"] = "array"
    cs.attrs["encoding-version"] = "0.2.0"


def _update_column_order(obs, new_cols):
    order = list(obs.attrs.get("column-order", []))
    for c in new_cols:
        if c not in order:
            order.append(c)
    obs.attrs["column-order"] = np.array(order, dtype=object)


def annotate_file(path, fine_lookup, helper_lookup):
    print(f"\n=== {os.path.basename(path)} ===", flush=True)
    t0 = time.time()
    with h5py.File(path, "r+") as f:
        obs = f["obs"]
        idx_name = obs.attrs.get("_index", "_index")
        barcodes = obs[idx_name][:]
        barcodes = np.array([b.decode() if isinstance(b, bytes) else b for b in barcodes])
        n = len(barcodes)
        print(f"  {n:,} cells", flush=True)

        # also need broad name per cell to fall back when no CSV match
        broad_grp = obs["Cluster_names"]
        broad_cats = np.array([c.decode() if isinstance(c, bytes) else c for c in broad_grp["categories"][:]])
        broad_codes = broad_grp["codes"][:]
        broad_per_cell = np.array(
            [broad_cats[c] if c >= 0 else "" for c in broad_codes], dtype=object
        )

        # Build columns
        fine_nums = np.full(n, -1, dtype=np.int32)
        fine_mask = np.ones(n, dtype=bool)  # True = NA
        fine_names = np.empty(n, dtype=object)
        helper_nums = np.full(n, -1, dtype=np.int32)
        helper_mask = np.ones(n, dtype=bool)
        helper_names = np.empty(n, dtype=object)

        for i, bc in enumerate(barcodes):
            fh = fine_lookup.get(bc)
            if fh is None:
                fine_names[i] = broad_per_cell[i]
            else:
                fine_nums[i] = fh[0]
                fine_mask[i] = False
                fine_names[i] = fh[1]

            hh = helper_lookup.get(bc)
            if hh is None:
                helper_names[i] = None
            else:
                helper_nums[i] = hh[0]
                helper_mask[i] = False
                helper_names[i] = hh[1]

        cov_fine = (~fine_mask).sum() / n * 100
        cov_helper = (~helper_mask).sum() / n * 100
        print(f"  fine coverage: {cov_fine:.2f}%  helper memory coverage: {cov_helper:.2f}%", flush=True)

        # Categorical encoding for fine names
        codes_fine, cats_fine = pd.factorize(pd.Series(fine_names), sort=True)
        codes_fine = codes_fine.astype(np.int16 if len(cats_fine) >= 127 else np.int8)

        # For helper names: NA cells get code -1
        helper_name_series = pd.Series(helper_names)
        codes_helper, cats_helper = pd.factorize(helper_name_series, sort=True)
        codes_helper = codes_helper.astype(np.int16 if len(cats_helper) >= 127 else np.int8)
        # pandas codes NaN as -1 already

        # Write
        _write_nullable_int(obs, "Cluster_fine_numbers", fine_nums, fine_mask)
        _write_categorical(obs, "Cluster_fine_names", list(cats_fine), codes_fine)
        _write_nullable_int(obs, "Cluster_helper_memory_numbers", helper_nums, helper_mask)
        _write_categorical(obs, "Cluster_helper_memory_names", list(cats_helper), codes_helper)

        # Drop legacy placeholder "Cluster_fine" if present
        if "Cluster_fine" in obs:
            del obs["Cluster_fine"]

        # Update column-order
        cols_added = [
            "Cluster_fine_numbers",
            "Cluster_fine_names",
            "Cluster_helper_memory_numbers",
            "Cluster_helper_memory_names",
        ]
        # also remove "Cluster_fine" from column order if present
        order = list(obs.attrs.get("column-order", []))
        order = [c for c in order if c != "Cluster_fine"]
        for c in cols_added:
            if c not in order:
                order.append(c)
        obs.attrs["column-order"] = np.array(order, dtype=object)

    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # Verify with anndata
    import anndata as ad
    a = ad.read_h5ad(path, backed="r")
    vc = a.obs["Cluster_fine_names"].value_counts()
    print(f"  fine_names unique={len(vc)}  top:")
    print("  " + vc.head(8).to_string().replace("\n", "\n  "))


def main():
    print("Building per-cell lookups from CSVs ...", flush=True)
    fine, helper = build_per_cell_lookup()
    print(f"  total fine entries: {len(fine):,}")
    print(f"  total helper entries: {len(helper):,}")

    for p in TARGETS:
        if not os.path.exists(p):
            print(f"WARN: missing {p}", file=sys.stderr)
            continue
        annotate_file(p, fine, helper)

    print("\nALL DONE.")


if __name__ == "__main__":
    main()
