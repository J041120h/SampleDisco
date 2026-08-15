#!/usr/bin/env python
"""Autotune the RMD weight and render the similarity panels at several alphas.

Produces, under <outroot>/<dataset>/ :
    alpha_default/            w_RMD = 0.60 (the shipped default, no search)
    alpha_tuned_bounded/      autotuned inside --bounded-range (shipped [0.1, 10])
    alpha_tuned_unbounded/    autotuned inside --unbounded-range on a LOG grid
    autotune_bounded/         autotune_record.txt from the bounded search
    autotune_unbounded/       autotune_record.txt from the wide search
    ALPHA.json                every alpha, its bounds, and whether it hit a bound

Only the fusion step depends on alpha, so the four blocks are built once and
re-fused per alpha; the panels differ by alpha and nothing else.

The search itself is ``sampledisco.parameter_selection.autotune.run_autotune``.
For the wide run the package's ``search_bayesian`` is swapped for a log-scale
wrapper around the SAME function: the shipped search seeds with
``np.linspace(lo, hi, 5)``, so over [0.01, 1000] it would place its first five
probes at 0.01/250/500/750/1000 and never look below 250. Searching log10(alpha)
covers the orders of magnitude evenly. The objective is untouched.

CIRCULARITY WARNING: autotune scores alpha against ``grouping_col``, so a panel
grouped by that same column is not independent evidence. The dataset's other
labels are the honest check.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/users/hjiang/GenoDistance/code/src")

from build_similarity_panel import (  # noqa: E402
    DATASETS, MEDIUM_K, FINE_K, RMD_DIM, PCA_N, SEED, RMD_WEIGHT,
    _log, block_shares, build_blocks, load_adata, similarity_panel, unit_labels,
)
from sampledisco.sample_embedding.blocks import (  # noqa: E402
    build_emb_from_blocks, derive_weights,
)
from sampledisco.parameter_selection import autotune as _at  # noqa: E402
from sampledisco.parameter_selection.autotune import run_autotune  # noqa: E402

# Which obs column autotune scores alpha against, per dataset.
GROUPING = {
    "ENCODE": "tissue",
    "scBloodNL_V1": "timepoint",
}


def _log_scale_bayesian(obj, bounds):
    """search_bayesian over log10(alpha) instead of alpha."""
    lo, hi = bounds
    t_lo, t_hi = math.log10(max(lo, 1e-6)), math.log10(hi)
    best_t, best_s, trace = _at.search_bayesian(
        lambda t: obj(10.0 ** t), bounds=(t_lo, t_hi), n_init=5, n_iter=10, seed=SEED)
    return 10.0 ** best_t, best_s, [(10.0 ** t, s) for t, s in trace]


def ensure_grouping_col(adata, cfg, col):
    """ENCODE's tissue lives in the metadata CSV, not in obs — inject it."""
    if col in adata.obs.columns:
        return adata
    if not cfg.get("meta_csv"):
        raise SystemExit(f"grouping col {col!r} absent and no meta_csv to source it")
    meta = pd.read_csv(cfg["meta_csv"]).set_index("sample")
    samp = adata.obs[cfg["sample_col"]].astype(str)
    adata.obs[col] = samp.map(meta[col]).astype(str).values
    _log(f"injected obs[{col!r}] from meta_csv "
         f"({int((adata.obs[col] == 'nan').sum())} unmapped cells)")
    return adata


def tune(adata, cfg, gcol, bounds, outdir, log_scale):
    """One autotune search. Returns (best_alpha, best_score)."""
    # log_scale is a no-op since sampledisco 0.3.0: search_bayesian itself
    # operates on log10(alpha). The shim is kept only to reproduce pre-0.3.0 runs.
    saved = _at.SEARCH_FUNCS.get("bayesian")
    if log_scale and os.environ.get("F2D_FORCE_LOG_SHIM"):
        _at.SEARCH_FUNCS["bayesian"] = _log_scale_bayesian
    try:
        res = run_autotune(
            adata, output_dir=outdir,
            sample_col=cfg["sample_col"], celltype_col=cfg["celltype_col"],
            comp_emb_key=cfg["comp_key"], rmd_emb_key=cfg["rmd_key"],
            modality_col=cfg["modality_col"], batch_col=cfg["batch_col"],
            grouping_col=gcol, medium_K=MEDIUM_K, fine_K=FINE_K, rmd_dim=RMD_DIM,
            pca_components=PCA_N, batch_method="harmony", scoring="auto",
            search="bayesian", scope="alpha_only",
            alpha_bounds=tuple(bounds), seed=SEED, save=True, verbose=True,
        )
    finally:
        if log_scale and saved is not None and os.environ.get("F2D_FORCE_LOG_SHIM"):
            _at.SEARCH_FUNCS["bayesian"] = saved
    return (float(res["best_params"]["rmd_weight"]),
            float(res.get("best_score", float("nan"))))


def render_at(d, units, cfg, dataset, alpha, outdir, note):
    os.makedirs(outdir, exist_ok=True)
    cache = os.path.join(outdir, "_cache")
    os.makedirs(cache, exist_ok=True)

    m = d["meta"]
    weights = derive_weights(m["K_c"], m["K_med"], m["K_fine"],
                             rmd_weight=alpha, n_blocks=4)
    Zc = build_emb_from_blocks(
        [d["A1"], d["A2"], d["A3"], d["RMD"]], weights,
        unit_ids=d["unit_ids"], unit_groups=d["unit_groups"],
        unit_batches=d["unit_batches"], pca_components=PCA_N,
        batch_method="harmony", seed=SEED, verbose=False,
    ).values.astype(np.float32)

    dd = dict(d)
    dd["Zc"] = Zc
    shares = block_shares(dd, weights)
    _log(f"alpha={alpha:.4g} [{note}]  "
         + ", ".join(f"{k}={v * 100:.1f}%" for k, v in shares.items()))

    meta_out = dict(m)
    meta_out["weights"] = dict(zip(("A1", "A2", "A3", "RMD"), map(float, weights)))
    meta_out["rmd_weight"] = float(alpha)
    meta_out["alpha_source"] = note
    meta_out["block_share_top10PC"] = {k: float(v) for k, v in shares.items()}
    for k in ("A1", "A2", "A3", "RMD", "Zc"):
        np.save(os.path.join(cache, f"{k}.npy"), dd[k])
    units.to_csv(os.path.join(cache, "units.csv"), index=False)
    with open(os.path.join(cache, "block_meta.json"), "w") as fh:
        json.dump(meta_out, fh, indent=2)

    tag = f"a{alpha:.4g}".replace(".", "p")
    for name, _col, order in cfg["labels"]:
        if units[name].nunique() < 2:
            continue
        similarity_panel(dd, units, name, order,
                         f"{dataset}  ({note}, w_RMD={alpha:.4g})",
                         os.path.join(outdir,
                                      f"fig2d_similarity_{dataset}_{tag}_{name}.png"),
                         shares=shares, alpha=alpha)
    return shares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(GROUPING))
    ap.add_argument("--outroot", default="/users/hjiang/GenoDistance/figure/figure2")
    ap.add_argument("--default-alpha", type=float, default=RMD_WEIGHT,
                    help="alpha for the no-search panel (default: the shipped 0.60)")
    ap.add_argument("--bounded-range", nargs=2, type=float, default=(0.1, 10.0),
                    metavar=("LO", "HI"),
                    help="alpha bounds for the 'current bound' search "
                         "(default: the shipped 0.1 10)")
    ap.add_argument("--unbounded-range", nargs=2, type=float, default=(0.01, 1000.0),
                    metavar=("LO", "HI"),
                    help="alpha bounds for the wide log-scale search")
    ap.add_argument("--skip-unbounded", action="store_true")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    gcol = GROUPING[args.dataset]
    outdir = os.path.join(args.outroot, args.dataset)
    os.makedirs(outdir, exist_ok=True)

    adata = load_adata(cfg)
    adata = ensure_grouping_col(adata, cfg, gcol)

    _log(f"=== bounded search: grouping_col={gcol!r} bounds={tuple(args.bounded_range)}")
    a_bounded, s_bounded = tune(adata, cfg, gcol, args.bounded_range,
                                os.path.join(outdir, "autotune_bounded"), False)
    hit_b = (abs(a_bounded - args.bounded_range[0]) < 1e-6 or
             abs(a_bounded - args.bounded_range[1]) < 1e-6)
    _log(f"=== bounded  w_RMD = {a_bounded:.4g}" + ("   *** AT BOUND ***" if hit_b else ""))

    a_wide = s_wide = None
    hit_w = False
    if not args.skip_unbounded:
        _log(f"=== wide log-scale search: bounds={tuple(args.unbounded_range)}")
        a_wide, s_wide = tune(adata, cfg, gcol, args.unbounded_range,
                              os.path.join(outdir, "autotune_unbounded"), True)
        hit_w = (abs(a_wide - args.unbounded_range[0]) / args.unbounded_range[0] < 1e-3 or
                 abs(a_wide - args.unbounded_range[1]) / args.unbounded_range[1] < 1e-3)
        _log(f"=== unbounded w_RMD = {a_wide:.4g}" + ("   *** AT BOUND ***" if hit_w else ""))

    # blocks are alpha-independent: build once, re-fuse per alpha
    d = build_blocks(adata, cfg)
    units = unit_labels(adata, cfg, d["unit_ids"], d["unit_groups"])

    out = {"dataset": args.dataset, "grouping_col": gcol, "variants": []}
    sh = render_at(d, units, cfg, args.dataset, args.default_alpha,
                   os.path.join(outdir, "alpha_default"), "default")
    out["variants"].append({"name": "default", "alpha": args.default_alpha,
                            "bounds": None, "at_bound": False, "score": None,
                            "share_RMD": sh["RMD"]})

    sh = render_at(d, units, cfg, args.dataset, a_bounded,
                   os.path.join(outdir, "alpha_tuned_bounded"), "tuned, bounded")
    out["variants"].append({"name": "tuned_bounded", "alpha": a_bounded,
                            "bounds": list(args.bounded_range), "at_bound": hit_b,
                            "score": s_bounded, "share_RMD": sh["RMD"]})

    if a_wide is not None:
        sh = render_at(d, units, cfg, args.dataset, a_wide,
                       os.path.join(outdir, "alpha_tuned_unbounded"),
                       "tuned, wide log-scale")
        out["variants"].append({"name": "tuned_unbounded", "alpha": a_wide,
                                "bounds": list(args.unbounded_range), "at_bound": hit_w,
                                "score": s_wide, "share_RMD": sh["RMD"]})

    with open(os.path.join(outdir, "ALPHA.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
