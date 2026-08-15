#!/usr/bin/env python
"""Re-render panels from an existing _cache (no block rebuild). Layout iteration."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_similarity_panel import DATASETS, block_shares, similarity_panel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outroot", default="/dcs07/hongkai/data/claude/fig2d_similarity")
    args = ap.parse_args()

    outdir = os.path.join(args.outroot, args.dataset)
    cache = os.path.join(outdir, "_cache")
    d = {k: np.load(os.path.join(cache, f"{k}.npy"))
         for k in ("A1", "A2", "A3", "RMD", "Zc")}
    # keep_default_na=False: "NA" is a real sex category in the unpaired
    # datasets and must not be parsed into NaN.
    units = pd.read_csv(os.path.join(cache, "units.csv"),
                        keep_default_na=False, dtype=str)
    with open(os.path.join(cache, "block_meta.json")) as fh:
        meta = json.load(fh)
    d["meta"] = meta
    weights = [meta["weights"][b] for b in ("A1", "A2", "A3", "RMD")]
    shares = block_shares(d, weights)
    print("shares:", {k: round(v * 100, 1) for k, v in shares.items()})

    for name, _col, order in DATASETS[args.dataset]["labels"]:
        if name not in units.columns or units[name].nunique() < 2:
            continue
        similarity_panel(d, units, name, order, args.dataset,
                         os.path.join(outdir, f"fig2d_similarity_{args.dataset}_{name}.png"),
                         shares=shares)
    return 0


if __name__ == "__main__":
    sys.exit(main())
