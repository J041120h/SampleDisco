#!/usr/bin/env python
"""Aggregate the COVID ablation per-size summaries into one wide table + ranking.

Merges ablation_summary_covid_{size}.csv (columns {variant}-{nsamples}) into a
single Metric x {variant}-{size} table, then ranks the 5 variants with the
SAME logic as Benchmark_covid/rank.py, using the *fair* conventions documented
in figure/error_check/ (inverse for smaller-is-better, ASW_batch ascending,
sign-invariant |Spearman|). Per (metric,size) the 5 variants are ranked
(1=best); ranks are averaged over the 6 sizes, then over the 10 metrics.
Final score = 1 / mean_rank (higher = better).
"""
import glob, os, re
import numpy as np
import pandas as pd

ROOT = "/dcs07/hongkai/data/harry/result/ablation/covid"
VARIANTS = ["proportion_only", "rmd_only", "no_batch_removal",
            "linear_regression", "original"]
COVID_METRICS = ["batch_partial_eta_sq", "iLISI_norm", "ASW_batch",
                 "severity_partial_eta_sq", "Spearman_Correlation",
                 "Custom_ANOVA_eta_sq", "ARI", "NMI", "Avg_Purity",
                 "Mean_NN_Severity_Gap"]
DESCENDING = {"batch_partial_eta_sq", "Mean_NN_Severity_Gap"}   # smaller better -> 1/x
SIGN_INVARIANT = {"Spearman_Correlation"}                        # fair: |x|


def transform(val, metric):
    if pd.isna(val):
        return np.nan
    if metric in SIGN_INVARIANT:
        return abs(val)
    if metric in DESCENDING:
        return 1.0 / val if val not in (0, 0.0) else np.nan
    return val


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "ablation_summary_covid_*.csv")))
    if not files:
        raise SystemExit(f"no per-size summaries under {ROOT}")
    wide = None
    for f in files:
        df = pd.read_csv(f, index_col=0)
        wide = df if wide is None else wide.join(df, how="outer")
    wide.to_csv(os.path.join(ROOT, "ablation_summary_covid_ALL.csv"), index_label="Metric")
    print(f"merged {len(files)} size files -> {wide.shape[1]} columns")

    # parse columns "{variant}-{size}"
    cols = {}
    for c in wide.columns:
        m = re.match(r"^(.*)-(\d+)$", c)
        if m and m.group(1) in VARIANTS:
            cols.setdefault(m.group(2), {})[m.group(1)] = c
    sizes = sorted(cols, key=int)

    # per (metric,size): rank the 5 variants; collect rank per variant
    per_metric_ranks = {}   # metric -> DataFrame(size x variant) of ranks
    for metric in COVID_METRICS:
        if metric not in wide.index:
            print(f"  [warn] metric {metric} absent; skipping"); continue
        rank_rows = {}
        for size in sizes:
            vals = {}
            for v in VARIANTS:
                col = cols[size].get(v)
                vals[v] = transform(wide.loc[metric, col], metric) if col else np.nan
            s = pd.Series(vals)
            rank_rows[size] = s.rank(ascending=False, method="min", na_option="bottom")
        per_metric_ranks[metric] = pd.DataFrame(rank_rows).T  # size x variant

    # average rank over sizes -> per variant per metric ; then over metrics
    metric_mean = pd.DataFrame(
        {m: df.mean(axis=0) for m, df in per_metric_ranks.items()})  # variant x metric
    overall = metric_mean.mean(axis=1).sort_values()                 # variant -> mean rank
    out = pd.DataFrame({
        "mean_rank": overall,
        "score_1_over_rank": 1.0 / overall,
    })
    metric_mean.to_csv(os.path.join(ROOT, "ablation_ranks_by_metric_covid.csv"),
                       index_label="variant")
    out.to_csv(os.path.join(ROOT, "ablation_ranking_covid.csv"), index_label="variant")

    pd.set_option("display.width", 160)
    print("\n=== per-metric mean rank (over 6 sizes; 1=best) ===")
    print(metric_mean.round(2))
    print("\n=== OVERALL RANKING (lower mean_rank = better) ===")
    print(out.round(3))


if __name__ == "__main__":
    main()
