#!/usr/bin/env python
"""Which block separates the samples best — composition, or RMD?

Weight-independent: each block is scored on its own, so the answer does not
depend on w_RMD at all. Two scores per (dataset, label, block):

  sil    cosine silhouette of the label on that block's unit x feature matrix
  delta  mean within-group cosine similarity - mean between-group similarity,
         computed on exactly the centred cosine matrix the heat map displays

A dataset "needs RMD" when RMD outscores every composition block (A1/A2/A3).
That is the evidence for keeping the displacement term in the method, and it is
a different question from how much weight the fusion gives it.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

ROOT = "/users/hjiang/GenoDistance/figure/figure2"
BLOCKS = ["A1", "A2", "A3", "RMD"]
SKIP_LABELS = {"unit", "group"}


def cos_sim(M):
    Mc = M - M.mean(axis=0, keepdims=True)
    n = np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-8
    Mn = Mc / n
    return Mn @ Mn.T


def delta_score(S, labels):
    """mean within-group similarity minus mean between-group similarity."""
    lab = np.asarray(labels).astype(str)
    n = len(lab)
    same = lab[:, None] == lab[None, :]
    off = ~np.eye(n, dtype=bool)
    w = same & off
    b = (~same) & off
    if w.sum() == 0 or b.sum() == 0:
        return float("nan")
    return float(S[w].mean() - S[b].mean())


def analyse(cache, name):
    units = pd.read_csv(f"{cache}/units.csv", keep_default_na=False, dtype=str)
    blocks = {b: np.load(f"{cache}/{b}.npy") for b in BLOCKS}
    blocks["Combined"] = np.load(f"{cache}/Zc.npy")
    rows = []
    for label in units.columns:
        if label in SKIP_LABELS:
            continue
        y = units[label].values
        k = pd.Series(y).nunique()
        if k < 2 or k >= len(y):
            continue
        rec = {"dataset": name, "label": label, "n_units": len(y), "n_groups": k}
        for b, M in blocks.items():
            S = cos_sim(M)
            rec[f"delta_{b}"] = delta_score(S, y)
            try:
                rec[f"sil_{b}"] = float(silhouette_score(M, y, metric="cosine"))
            except Exception:
                rec[f"sil_{b}"] = float("nan")
        rows.append(rec)
    return rows


def main():
    caches = sorted(glob.glob(f"{ROOT}/*/_cache/units.csv") +
                    glob.glob(f"{ROOT}/*/*/_cache/units.csv") +
                    glob.glob(f"{ROOT}/*/*/*/_cache/units.csv"))
    rows = []
    for c in caches:
        cache = os.path.dirname(c)
        name = os.path.relpath(os.path.dirname(cache), ROOT)
        try:
            rows.extend(analyse(cache, name))
        except Exception as e:
            print(f"[warn] {name}: {type(e).__name__}: {e}", file=sys.stderr)
    df = pd.DataFrame(rows)
    comp_d = df[[f"delta_{b}" for b in ("A1", "A2", "A3")]].max(axis=1)
    comp_s = df[[f"sil_{b}" for b in ("A1", "A2", "A3")]].max(axis=1)
    df["best_comp_delta"] = comp_d
    df["best_comp_sil"] = comp_s
    df["RMD_wins_delta"] = df["delta_RMD"] > comp_d
    df["RMD_wins_sil"] = df["sil_RMD"] > comp_s
    df["delta_gap"] = df["delta_RMD"] - comp_d

    out = f"{ROOT}/RMD_vs_composition_separation.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    show = ["dataset", "label", "n_units", "n_groups",
            "delta_A1", "delta_A2", "delta_A3", "delta_RMD", "delta_Combined",
            "sil_RMD", "best_comp_sil"]
    print("\n===== RMD BEATS EVERY COMPOSITION BLOCK (delta) =====")
    win = df[df["RMD_wins_delta"]].sort_values("delta_gap", ascending=False)
    print(win[show].to_string(index=False, float_format=lambda v: f"{v:7.3f}")
          if len(win) else "  (none)")

    print("\n===== RMD BEATS EVERY COMPOSITION BLOCK (silhouette) =====")
    wins = df[df["RMD_wins_sil"]].sort_values("sil_RMD", ascending=False)
    print(wins[show].to_string(index=False, float_format=lambda v: f"{v:7.3f}")
          if len(wins) else "  (none)")

    print("\n===== FULL TABLE (sorted by RMD - best composition, delta) =====")
    print(df.sort_values("delta_gap", ascending=False)[show]
          .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
