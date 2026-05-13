"""Explore latent batch variables in the 1M-scBloodNL AnnData.

Most obvious batch is `chem` (V2 / V3), which is in obs.
A finer batch is encoded in the cell barcode suffix:
  AAACCTGAGAAACCAT_180920_lane1
                    ^^^^^^_^^^^^
                    date   lane     -> 10x chip / lane / sequencing batch

Per the paper: each lane pooled 8 donors x 2 stim-time combos. So lane is
the true experimental batch (finer than chem, coarser than donor).

This script is read-only: it parses the suffix, counts unique batches at
several granularities, and shows how each batch variable cross-tabs with
chem / stim / timepoint / donor. Use the output to decide what to add as
a batch column.
"""

import re
import anndata as ad
import pandas as pd

H5AD_PATH = "/dcs07/hongkai/data/harry/result/1M-scBloodNL/data/1M-scBloodNL.h5ad"

SUFFIX_RE = re.compile(r"^[ACGT]+_(?P<date>\d+)_(?P<lane>lane\d+)$")


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    section(f"Loading (read-only): {H5AD_PATH}")
    adata = ad.read_h5ad(H5AD_PATH)
    print(f"  shape: {adata.shape}")

    # ------------------------------------------------------------------
    # Parse barcode suffix
    # ------------------------------------------------------------------
    section("Parse barcode suffix -> (date, lane)")
    parsed = adata.obs.index.to_series().str.extract(SUFFIX_RE)
    n_unparsed = parsed["date"].isna().sum()
    print(f"  cells with unparseable barcode: {n_unparsed}")
    if n_unparsed:
        bad = adata.obs.index[parsed["date"].isna()][:5].tolist()
        print(f"  examples: {bad}")

    obs = adata.obs.copy()
    obs["seq_date"] = parsed["date"].values
    obs["seq_lane"] = parsed["lane"].values
    obs["seq_batch"] = obs["seq_date"].astype(str) + "_" + obs["seq_lane"].astype(str)

    print(f"  unique seq_date : {obs['seq_date'].nunique()}")
    print(f"  unique seq_lane : {obs['seq_lane'].nunique()}")
    print(f"  unique seq_batch (date_lane): {obs['seq_batch'].nunique()}")
    print(f"\n  date value counts:")
    print(obs["seq_date"].value_counts().head(20).to_string())
    print(f"\n  lane value counts:")
    print(obs["seq_lane"].value_counts().head(20).to_string())

    # ------------------------------------------------------------------
    # Granularity check: is each batch within one chem? one stim-time? etc.
    # ------------------------------------------------------------------
    section("Per seq_batch: how many chem / stim / timepoint / donor?")
    grp = obs.groupby("seq_batch", observed=True).agg(
        n_cells=("chem", "size"),
        n_chem=("chem", "nunique"),
        n_stim=("stimulation_conditions", "nunique"),
        n_time=("timepoint", "nunique"),
        n_donors=("assignment", "nunique"),
        n_combos=("id", "nunique"),
    )
    print(f"  total batches: {len(grp)}")
    print(f"  cells per batch  : min={grp['n_cells'].min()}, "
          f"median={int(grp['n_cells'].median())}, max={grp['n_cells'].max()}")
    for col in ["n_chem", "n_stim", "n_time", "n_donors", "n_combos"]:
        vc = grp[col].value_counts().sort_index()
        print(f"  {col:10s} distribution: {dict(vc)}")

    print("\n  paper claim: each lane = 8 donors x 2 stim-time combos, 1 chem.")
    print("  → expect n_chem=1 always, n_donors≈8, n_combos≈16, n_stim/n_time small.")

    # ------------------------------------------------------------------
    # Confounding: how is each batch variable confounded with biology?
    # ------------------------------------------------------------------
    section("Confounding audit: batch vs biological variables")

    # Donor confounding with chem/seq_batch is critical (each donor was
    # likely processed on one chem, possibly across few lanes).
    donor_grp = obs.groupby("assignment", observed=True).agg(
        n_chem=("chem", "nunique"),
        n_batch=("seq_batch", "nunique"),
        n_date=("seq_date", "nunique"),
    )
    print("  per-donor variable spread (how many distinct batches a donor spans):")
    for col in ["n_chem", "n_batch", "n_date"]:
        vc = donor_grp[col].value_counts().sort_index()
        print(f"    {col:8s}: {dict(vc)}")

    # ------------------------------------------------------------------
    # Top batches preview
    # ------------------------------------------------------------------
    section("Preview: 10 largest seq_batches")
    print(grp.nlargest(10, "n_cells").to_string())

    print("\n" + "=" * 72)
    print("INTERPRETATION GUIDE")
    print("=" * 72)
    print("""
- `chem` (V2/V3): coarsest batch, largest technical effect (capture rate).
- `seq_batch` (date_lane): true experimental batch — each lane is one 10x
  chip run, mixing 8 donors × 2 stim-time combos. Use this for batch
  correction (Harmony, scVI, Seurat CCA) and as a random effect in models.
- `seq_date` alone: less granular (multiple lanes per day), but useful if
  lanes within a day shared reagent lots.

Rule of thumb:
  Within-chem analysis: use `seq_batch` as the batch variable.
  Across-chem: stratify by chem first (paper standard), THEN seq_batch
  inside each stratum.

If a single (donor, stim, timepoint) sample is split across multiple
seq_batches, that means cells from one well were captured on different
lanes/days — those should be treated as one biological sample but
multiple technical batches.
""")


if __name__ == "__main__":
    main()
