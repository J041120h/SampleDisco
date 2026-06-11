#!/usr/bin/env python
"""Rank the 5 ablation variants on each paired multi-omics dataset.

Per dataset, metrics = paired_partner_rank (smaller better -> 1/x),
ASW_modality, iLISI_norm_mean, and the condition-preservation score
(tissue / disease_state / cca). Higher transformed value = better; variants
ranked 1..5 per metric, then averaged. Also a combined MO ranking
(mean rank across the 4 datasets).
"""
import os
import numpy as np
import pandas as pd

ROOT = "/dcs07/hongkai/data/harry/result/ablation/multiomics"
VARIANTS = ["proportion_only", "rmd_only", "no_batch_removal",
            "linear_regression", "original"]
DATASETS = ["ENCODE", "heart", "retina", "lutea"]
DESCENDING = {"paired_partner_rank"}
PRESERVATION = ["tissue_preservation_score", "disease_state_preservation_score",
                "cca_score"]
BASE_METRICS = ["paired_partner_rank", "ASW_modality", "iLISI_norm_mean"]


def transform(v, m):
    if pd.isna(v):
        return np.nan
    return (1.0 / v) if (m in DESCENDING and v != 0) else v


def rank_dataset(summary_csv):
    df = pd.read_csv(summary_csv, index_col=0)
    metrics = [m for m in BASE_METRICS if m in df.index]
    metrics += [m for m in PRESERVATION if m in df.index]
    rk = {}
    for m in metrics:
        vals = pd.Series({v: transform(df.loc[m, v], m) for v in VARIANTS if v in df.columns})
        rk[m] = vals.rank(ascending=False, method="min", na_option="bottom")
    R = pd.DataFrame(rk)              # variant x metric
    return R, R.mean(axis=1)         # per-metric ranks, mean rank per variant


def main():
    per_ds_mean = {}
    for ds in DATASETS:
        f = os.path.join(ROOT, ds, f"ablation_summary_mo_{ds}.csv")
        if not os.path.exists(f):
            print(f"[skip] {ds}: no summary yet ({f})"); continue
        R, mean = rank_dataset(f)
        per_ds_mean[ds] = mean
        R.to_csv(os.path.join(ROOT, ds, f"ablation_ranks_{ds}.csv"), index_label="variant")
        print(f"\n=== {ds}: per-metric rank (1=best) ===")
        print(R.round(2))
        print(f"--- {ds} mean rank ---")
        print(mean.sort_values().round(3))

    if per_ds_mean:
        combined = pd.DataFrame(per_ds_mean)             # variant x dataset
        combined["MO_mean_rank"] = combined.mean(axis=1)
        combined["score_1_over_rank"] = 1.0 / combined["MO_mean_rank"]
        combined = combined.sort_values("MO_mean_rank")
        combined.to_csv(os.path.join(ROOT, "ablation_ranking_mo_combined.csv"),
                        index_label="variant")
        print("\n=== COMBINED multi-omics ranking (mean rank over datasets) ===")
        print(combined.round(3))


if __name__ == "__main__":
    main()
