"""Dry run for add_sample_id.py — computes the proposed sample `id`
column and runs every check, but does NOT modify or overwrite the h5ad.

Use this to decide whether (donor, stim, timepoint) is a clean key
before committing to it.

Checks performed:
  1. `assignment` parses cleanly to integers.
  2. Constructed `id` is unique per (donor, stim, timepoint).
  3. No id is shared across multiple donors / stims / timepoints.
  4. Cell-count distribution per sample.
  5. Chemistry mixing: how many ids contain both v2 and v3, and the
     per-chem cell counts / minor-chemistry fraction within those ids.
  6. Total sample count vs paper expectation (~840 designed, ~782 after QC).
"""

import anndata as ad
import pandas as pd

H5AD_PATH = "/dcs07/hongkai/data/harry/result/1M-scBloodNL/data/1M-scBloodNL.h5ad"


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    section(f"Loading (read-only): {H5AD_PATH}")
    adata = ad.read_h5ad(H5AD_PATH)
    print(f"  shape: {adata.shape}")
    print(f"  obs columns: {list(adata.obs.columns)}")

    obs = adata.obs

    # ------------------------------------------------------------------
    # 1. Parse assignment -> int
    # ------------------------------------------------------------------
    section("Check 1: assignment parses cleanly to integers")
    raw_assignment = obs["assignment"].astype(str)
    print(f"  example raw values: {list(raw_assignment.unique()[:5])}")
    try:
        donor_int = pd.to_numeric(raw_assignment, errors="raise").astype(int)
        print(f"  ✓ all {len(raw_assignment)} values parsed to int")
        print(f"  donor range: [{donor_int.min()}, {donor_int.max()}], "
              f"unique donors: {donor_int.nunique()}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return

    # ------------------------------------------------------------------
    # 2. Construct proposed id (in memory only)
    # ------------------------------------------------------------------
    section("Check 2: construct proposed `id` in memory (NOT saved)")
    donor_str = donor_int.astype(str)
    stim = obs["stimulation_conditions"].astype(str)
    time = obs["timepoint"].astype(int).astype(str)
    proposed_id = pd.Series(
        "D" + donor_str.values + "_" + stim.values + "_" + time.values + "h",
        index=obs.index,
        name="id",
    )
    n_samples = proposed_id.nunique()
    print(f"  unique sample ids: {n_samples}")
    print(f"  paper expectation: 840 designed, ~782 after QC (per Fig.1 + page 5)")
    if 750 <= n_samples <= 850:
        print(f"  ✓ within expected range")
    else:
        print(f"  ⚠ outside expected range — investigate")

    # ------------------------------------------------------------------
    # 3. Each id maps to one (donor, stim, timepoint)
    # ------------------------------------------------------------------
    section("Check 3: id is unique per (donor, stim, timepoint)")
    tmp = obs.assign(_id=proposed_id.values)
    grp = tmp.groupby("_id", observed=True).agg(
        n_donor=("assignment", "nunique"),
        n_stim=("stimulation_conditions", "nunique"),
        n_time=("timepoint", "nunique"),
        n_chem=("chem", "nunique"),
        n_cells=("chem", "size"),
    )
    bad = grp[(grp["n_donor"] > 1) | (grp["n_stim"] > 1) | (grp["n_time"] > 1)]
    if len(bad) == 0:
        print(f"  ✓ every id maps to exactly one (donor, stim, timepoint)")
    else:
        print(f"  ✗ {len(bad)} ids violate uniqueness:")
        print(bad.head().to_string())
        return

    # ------------------------------------------------------------------
    # 4. Cell-count distribution per sample
    # ------------------------------------------------------------------
    section("Check 4: cells per sample")
    nc = grp["n_cells"]
    print(f"  min   : {nc.min()}")
    print(f"  q25   : {int(nc.quantile(0.25))}")
    print(f"  median: {int(nc.median())}")
    print(f"  q75   : {int(nc.quantile(0.75))}")
    print(f"  max   : {nc.max()}")
    n_tiny = int((nc < 50).sum())
    if n_tiny:
        print(f"  ⚠ {n_tiny} samples have <50 cells — may be too small for pseudobulk")
        print(grp.nsmallest(min(10, n_tiny), "n_cells")[["n_cells", "n_chem"]].to_string())
    else:
        print(f"  ✓ no sample has <50 cells")

    # ------------------------------------------------------------------
    # 5. Chemistry mixing audit
    # ------------------------------------------------------------------
    section("Check 5: chemistry mixing within id")
    mixed_ids = grp.index[grp["n_chem"] > 1]
    print(f"  samples spanning >1 chemistry: {len(mixed_ids)} / {n_samples} "
          f"({100 * len(mixed_ids) / n_samples:.2f}%)")

    if len(mixed_ids) > 0:
        mixed_obs = tmp[tmp["_id"].isin(mixed_ids)]
        per_chem = (
            mixed_obs.groupby(["_id", "chem"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        per_chem["total"] = per_chem.sum(axis=1)
        per_chem["minor_frac"] = (
            per_chem.drop(columns="total").min(axis=1) / per_chem["total"]
        )
        per_chem = per_chem.sort_values("total", ascending=False)

        print(f"  minor-chemistry fraction within mixed ids:")
        print(f"    min  = {per_chem['minor_frac'].min():.3f}")
        print(f"    mean = {per_chem['minor_frac'].mean():.3f}")
        print(f"    max  = {per_chem['minor_frac'].max():.3f}")

        n_show = min(20, len(per_chem))
        print(f"\n  top {n_show} mixed ids (per-chem cell counts):")
        print(per_chem.head(n_show).to_string())

        if per_chem["minor_frac"].max() < 0.05:
            print("\n  → recommendation: KEEP id as (donor, stim, timepoint).")
            print("    Minor-chem contamination <5% everywhere; treat chem as covariate.")
        else:
            print("\n  → recommendation: review these ids before committing.")
            print("    Either split id by chem for these samples, or accept the mixing.")
    else:
        print("  ✓ no id mixes chemistries — chem is a strict between-sample batch.")

    # ------------------------------------------------------------------
    # 6. Sample preview
    # ------------------------------------------------------------------
    section("Preview: 10 most/least populated proposed ids")
    print("Most populated:")
    print(grp.nlargest(10, "n_cells")[["n_cells", "n_chem"]].to_string())
    print("\nLeast populated:")
    print(grp.nsmallest(10, "n_cells")[["n_cells", "n_chem"]].to_string())

    print("\n" + "=" * 72)
    print("DRY RUN COMPLETE — no changes were written.")
    print("If everything looks good, run add_sample_id.py to commit the column.")
    print("=" * 72)


if __name__ == "__main__":
    main()
