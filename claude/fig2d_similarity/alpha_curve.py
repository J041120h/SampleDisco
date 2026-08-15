#!/usr/bin/env python
"""Dense sweep of the autotune objective over alpha — is the surface flat?

Evaluates the SAME objective ``run_autotune`` optimises (autotune.build_blocks +
autotune.make_scorer, so no re-implementation) on a log-spaced grid of alpha,
and writes the curve plus a plot marking every alpha the searches returned.

This settles two questions at once:
  * whether GP-EI Bayesian search is worth its 10 extra evaluations, by showing
    what a plain grid of the same size would have found;
  * how sharply the "optimal" alpha is identified — a flat curve means the
    single reported alpha is over-precise and a range should be quoted instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/users/hjiang/GenoDistance/code/src")

from build_similarity_panel import (  # noqa: E402
    DATASETS, MEDIUM_K, FINE_K, RMD_DIM, PCA_N, SEED, _log, load_adata,
)
from tune_and_render import GROUPING, ensure_grouping_col  # noqa: E402
from sampledisco.parameter_selection.autotune import (  # noqa: E402
    build_blocks as at_build_blocks, make_scorer,
)
from sampledisco.sample_embedding.blocks import (  # noqa: E402
    build_emb_from_blocks, derive_weights,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(GROUPING))
    ap.add_argument("--outroot", default="/users/hjiang/GenoDistance/figure/figure2")
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--range", nargs=2, type=float, default=(0.01, 1000.0))
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    gcol = GROUPING[args.dataset]
    outdir = os.path.join(args.outroot, args.dataset)
    os.makedirs(outdir, exist_ok=True)

    adata = load_adata(cfg)
    adata = ensure_grouping_col(adata, cfg, gcol)

    blocks = at_build_blocks(
        adata, sample_col=cfg["sample_col"], celltype_col=cfg["celltype_col"],
        comp_emb_key=cfg["comp_key"], rmd_emb_key=cfg["rmd_key"],
        modality_col=cfg["modality_col"], batch_col=cfg["batch_col"],
        grouping_col=gcol, medium_K=MEDIUM_K, fine_K=FINE_K, rmd_dim=RMD_DIM,
        seed=SEED, verbose=True,
    )
    score_fn = make_scorer("auto", blocks)

    lo, hi = args.range
    alphas = np.logspace(np.log10(lo), np.log10(hi), args.grid)
    rows = []
    for i, a in enumerate(alphas):
        w = derive_weights(blocks["K_c"], blocks["K_med"], blocks["K_fine"],
                           rmd_weight=float(a), n_blocks=4)
        emb = build_emb_from_blocks(
            [blocks["A1"], blocks["A2"], blocks["A3"], blocks["RMD"]], w,
            unit_ids=blocks["unit_ids"], unit_groups=blocks["unit_groups"],
            unit_batches=blocks["unit_batches"], pca_components=PCA_N,
            batch_method="harmony", seed=SEED, verbose=False,
        )
        s = float(score_fn(np.asarray(emb.values, dtype=np.float32)))
        share = a * a / (300.0 / blocks["K_c"] + 300.0 / MEDIUM_K + 1.0 + a * a)
        rows.append({"alpha": float(a), "score": s, "share_RMD": float(share)})
        _log(f"[{i + 1:>3}/{len(alphas)}] alpha={a:9.4g}  score={s:.6f}  "
             f"RMD={share * 100:5.1f}%")

    best = max(rows, key=lambda r: r["score"])
    smax, smin = best["score"], min(r["score"] for r in rows)
    # every alpha within 1% of the peak — the honest "optimal region"
    tol = smax - 0.01 * abs(smax)
    plateau = [r["alpha"] for r in rows if r["score"] >= tol]

    out = {"dataset": args.dataset, "grouping_col": gcol,
           "grid": rows, "best": best,
           "score_min": smin, "score_max": smax,
           "relative_span_pct": (smax - smin) / abs(smax) * 100,
           "within_1pct_of_peak": [min(plateau), max(plateau)],
           "n_within_1pct": len(plateau)}
    with open(os.path.join(outdir, "ALPHA_CURVE.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx([r["alpha"] for r in rows], [r["score"] for r in rows],
                "-o", ms=3, lw=1.4, color="#2166ac", label="autotune objective")
    ax.axhspan(tol, smax, color="#d6604d", alpha=0.15,
               label="within 1% of peak")
    ax.axvline(best["alpha"], color="#b2182b", lw=1.4,
               label=f"grid best α={best['alpha']:.3g}")
    try:
        with open(os.path.join(outdir, "ALPHA.json")) as fh:
            for v in json.load(fh)["variants"]:
                ax.axvline(v["alpha"], ls="--", lw=1.1, alpha=0.8,
                           color={"default": "#666666",
                                  "tuned_bounded": "#1b7837",
                                  "tuned_unbounded": "#762a83"}.get(v["name"], "#999"),
                           label=f"{v['name']} α={v['alpha']:.3g}")
    except Exception:
        pass
    ax.set_xlabel("w_RMD  (α)")
    ax.set_ylabel("autotune objective (higher = better)")
    ax.set_title(f"{args.dataset} — objective vs α  "
                 f"(grouping_col={gcol}; total span {out['relative_span_pct']:.1f}%)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ALPHA_CURVE.png"), dpi=200)
    _log(f"peak α={best['alpha']:.4g}  score span {out['relative_span_pct']:.2f}%  "
         f"within-1% region = [{min(plateau):.3g}, {max(plateau):.3g}] "
         f"({len(plateau)}/{len(rows)} grid points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
