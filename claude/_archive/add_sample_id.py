"""Add a per-sample `id` column to the 1M-scBloodNL AnnData.

A sample = (donor, stimulation_conditions, timepoint). Chemistry is
treated as a batch effect and is *not* part of the id, so cells from the
same donor/condition/timepoint that happened to be processed on both
v2 and v3 chips collapse into the same sample id.

id format: "D{assignment}_{stim}_{timepoint}h"
  e.g. "D1_CA_3h", "D12_UT_0h", "D43_MTB_24h"

Overwrites the .h5ad in place (gzip).
"""

import anndata as ad
import pandas as pd

H5AD_PATH = "/dcs07/hongkai/data/harry/result/1M-scBloodNL/data/1M-scBloodNL.h5ad"


def main():
    print(f"Loading: {H5AD_PATH}")
    adata = ad.read_h5ad(H5AD_PATH)
    print(f"  shape: {adata.shape}")

    # `assignment` is stored as a categorical of strings like "1.0", "43.0".
    # Strip the trailing ".0" so the id reads cleanly.
    donor = (
        pd.to_numeric(adata.obs["assignment"].astype(str), errors="raise")
        .astype(int)
        .astype(str)
    )
    stim = adata.obs["stimulation_conditions"].astype(str)
    time = adata.obs["timepoint"].astype(int).astype(str)

    sample_id = "D" + donor.values + "_" + stim.values + "_" + time.values + "h"
    adata.obs["id"] = pd.Categorical(sample_id)

    n_samples = adata.obs["id"].nunique()
    print(f"  unique sample ids: {n_samples}")

    # Sanity: each id should map to exactly one (donor, stim, timepoint).
    grp = adata.obs.groupby("id", observed=True).agg(
        n_donor=("assignment", "nunique"),
        n_stim=("stimulation_conditions", "nunique"),
        n_time=("timepoint", "nunique"),
        n_chem=("chem", "nunique"),
        n_cells=("chem", "size"),
    )
    bad = grp[(grp["n_donor"] > 1) | (grp["n_stim"] > 1) | (grp["n_time"] > 1)]
    if len(bad):
        raise RuntimeError(f"id is not unique per (donor, stim, time):\n{bad.head()}")
    print(f"  cells per sample: min={grp['n_cells'].min()}, "
          f"median={int(grp['n_cells'].median())}, max={grp['n_cells'].max()}")

    # ------------------------------------------------------------------
    # Chemistry-mixing audit: does any sample id contain both v2 and v3?
    # ------------------------------------------------------------------
    print("\n  ── chemistry mixing audit ──")
    mixed_ids = grp.index[grp["n_chem"] > 1]
    print(f"  samples spanning >1 chemistry: {len(mixed_ids)} / {n_samples} "
          f"({100 * len(mixed_ids) / n_samples:.2f}%)")

    if len(mixed_ids) > 0:
        # Per-chem cell counts for each mixed id.
        mixed_obs = adata.obs[adata.obs["id"].isin(mixed_ids)]
        per_chem = (
            mixed_obs.groupby(["id", "chem"], observed=True)
            .size()
            .unstack(fill_value=0)
            .sort_values(by=list(mixed_obs["chem"].cat.categories), ascending=False)
        )
        per_chem["total"] = per_chem.sum(axis=1)
        per_chem["minor_frac"] = (
            per_chem.drop(columns="total").min(axis=1) / per_chem["total"]
        )

        print(f"  cells from minor chemistry within a mixed id:")
        print(f"    min  fraction: {per_chem['minor_frac'].min():.3f}")
        print(f"    mean fraction: {per_chem['minor_frac'].mean():.3f}")
        print(f"    max  fraction: {per_chem['minor_frac'].max():.3f}")

        n_show = min(20, len(per_chem))
        print(f"\n  top {n_show} mixed ids (per-chem cell counts):")
        print(per_chem.head(n_show).to_string())

        # Recommendation heuristic.
        if per_chem["minor_frac"].max() < 0.05:
            print("\n  → minor-chemistry contamination is <5% in every mixed id.")
            print("    Safe to keep `id` as (donor, stim, timepoint); treat chem")
            print("    as a covariate in modeling.")
        else:
            print("\n  → some mixed ids have substantial cells from both chemistries.")
            print("    Consider either (a) splitting those ids by chem, or")
            print("    (b) being explicit about chem as a within-sample batch.")
    else:
        print("  → no sample id mixes chemistries. `id` is fully clean;")
        print("    chem is a strict between-sample batch variable.")

    print("\n  example ids:")
    print(adata.obs["id"].value_counts().head(10))

    print(f"\nWriting (gzip) -> {H5AD_PATH}")
    adata.write_h5ad(H5AD_PATH, compression="gzip")
    print("Done.")


if __name__ == "__main__":
    main()
