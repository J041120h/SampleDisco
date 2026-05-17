"""Compute per-dataset per-metric rankings of all benchmarked methods.

Reads /dcs07/.../test_run_multiomics/<dataset>/Benchmark_result/summary.csv
for each dataset, computes ranks (with correct direction per metric), and
writes a consolidated ranking table.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd

ROOT = "/dcs07/hongkai/data/harry/result/test/test_run_multiomics"
DATASETS = ["ENCODE", "eye_retina", "eye_lutea", "heart"]

# True if HIGHER is better for that metric
HIGHER_IS_BETTER = {
    "paired_partner_rank":         False,  # median normalised rank, lower=better
    "tissue_preservation_score":   True,   # between/within
    "disease_state_preservation_score": True,
    "cca_score":                   True,
    "ASW_modality":                True,
}


def rank_one(df_metric: pd.Series, higher_better: bool) -> pd.Series:
    """Returns rank 1..N (1 = best). NaN → NaN."""
    return df_metric.rank(method='min', ascending=not higher_better)


def main():
    all_blocks = []
    for ds in DATASETS:
        p = f"{ROOT}/{ds}/Benchmark_result/summary.csv"
        if not os.path.exists(p):
            print(f"[skip] {ds}: {p} missing"); continue
        df = pd.read_csv(p, index_col=0)            # rows=metric, cols=method
        df = df.apply(pd.to_numeric, errors='coerce')
        print(f"\n=== {ds} === ({df.shape[0]} metrics × {df.shape[1]} methods)")
        for metric in df.index:
            if metric not in HIGHER_IS_BETTER:
                print(f"  ?? unknown metric: {metric} (skipping rank)"); continue
            hb = HIGHER_IS_BETTER[metric]
            vals = df.loc[metric].dropna()
            ranks = rank_one(vals, hb).astype(int)
            order = vals.sort_values(ascending=not hb)
            row = []
            for rk, (m, v) in enumerate(zip(order.index, order.values), 1):
                tag = "★" if "SD_v2" in m else " "
                row.append(f"{rk}. {tag}{m}={v:.4f}")
            print(f"  {metric}  ({'higher' if hb else 'lower'} better)")
            print("    " + "   ".join(row))
            # Also store rank rows
            for m in vals.index:
                all_blocks.append({"dataset": ds, "metric": metric,
                                   "method": m, "value": float(vals[m]),
                                   "rank": int(ranks[m])})
    out_df = pd.DataFrame(all_blocks)
    out_csv = f"{ROOT}/rankings.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nrankings table → {out_csv}")

    print("\n" + "=" * 60)
    print("SUMMARY: SD_v2_tuned & SD_v2_default rank per metric per dataset")
    print("=" * 60)
    pivot_tuned = (out_df[out_df['method'] == 'SD_v2_tuned']
                   .pivot(index='dataset', columns='metric', values='rank'))
    pivot_default = (out_df[out_df['method'] == 'SD_v2_default']
                     .pivot(index='dataset', columns='metric', values='rank'))
    print("\nSD_v2_tuned ranks (lower = better; 1 = #1):")
    print(pivot_tuned.to_string())
    print("\nSD_v2_default ranks:")
    print(pivot_default.to_string())


if __name__ == "__main__":
    main()
