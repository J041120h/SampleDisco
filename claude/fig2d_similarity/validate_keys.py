#!/usr/bin/env python
"""Empirically verify the comp / RMD key assignment on every dataset.

The paper defines the two cell-level views by what variance they carry:
  z^comp — sample-associated variation REMOVED   -> eta^2(sample) should be LOW
  z^RMD  — sample-associated variation RETAINED  -> eta^2(sample) should be HIGH

eta^2 = between-sample scatter / total scatter, on the cell-level embedding.
Reported alongside the same statistic for the batch/study variable, which shows
whether the technical covariate was removed as the paper requires.

Names can lie; this measures it. Read-only.
"""
import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, "/users/hjiang/GenoDistance/code/claude/fig2d_similarity")
from build_similarity_panel import DATASETS  # noqa: E402


def eta2(Z, labels):
    """Fraction of total scatter that lies between groups."""
    Z = np.asarray(Z, dtype=np.float64)
    lab = pd.Series(np.asarray(labels).astype(str))
    grand = Z.mean(axis=0)
    tot = float(((Z - grand) ** 2).sum())
    if tot <= 0:
        return float("nan")
    between = 0.0
    for _, idx in lab.groupby(lab).groups.items():
        sub = Z[np.asarray(idx, dtype=int)]
        between += len(sub) * float(((sub.mean(axis=0) - grand) ** 2).sum())
    return between / tot


def main():
    names = sys.argv[1:]
    if not names:
        # one representative per distinct (adata, comp, rmd); skip derived comps
        seen, names = set(), []
        for n, c in DATASETS.items():
            if c.get("harmonize_comp") or c.get("subset"):
                continue
            k = (c["adata"], c["comp_key"], c["rmd_key"])
            if k in seen:
                continue
            seen.add(k)
            names.append(n)

    print(f"{'dataset':<20}{'comp_key':<22}{'eta2_samp':>10}{'eta2_batch':>11}"
          f"   {'rmd_key':<24}{'eta2_samp':>10}{'eta2_batch':>11}   verdict")
    print("-" * 122)
    for n in names:
        cfg = DATASETS[n]
        try:
            ab = ad.read_h5ad(cfg["adata"], backed="r")
            samp = ab.obs[cfg["sample_col"]].astype(str).values
            bcol = cfg.get("batch_col")
            batch = ab.obs[bcol].astype(str).values if bcol and bcol in ab.obs.columns else None
            out = {}
            for role in ("comp_key", "rmd_key"):
                k = cfg[role]
                if k not in ab.obsm:
                    out[role] = (float("nan"), float("nan"))
                    continue
                Z = np.asarray(ab.obsm[k], dtype=np.float32)
                out[role] = (eta2(Z, samp),
                             eta2(Z, batch) if batch is not None else float("nan"))
            ab.file.close()
        except Exception as e:
            print(f"{n:<20} FAILED {type(e).__name__}: {e}")
            continue

        cs, cb = out["comp_key"]
        rs, rb = out["rmd_key"]
        ok = (not np.isnan(cs)) and (not np.isnan(rs)) and rs > cs
        verdict = "OK  (rmd > comp)" if ok else "*** SUSPECT ***"
        print(f"{n:<20}{cfg['comp_key']:<22}{cs:>10.4f}{cb:>11.4f}   "
              f"{cfg['rmd_key']:<24}{rs:>10.4f}{rb:>11.4f}   {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
