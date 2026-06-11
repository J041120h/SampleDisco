#!/usr/bin/env python
"""Rank the 5 ablation variants on the health-aging PBMC benchmark.

Reads <root>/<variant>/{P1,P2,P3,G1}.json, builds a variant x metric table,
and ranks on the 4-test metric suite (all higher = better; |spearman|).
"""
import json, os
import numpy as np
import pandas as pd

ROOT = "/dcs07/hongkai/data/harry/result/ablation/health_aging"
VARIANTS = ["proportion_only", "rmd_only", "no_batch_removal",
            "linear_regression", "original"]
TESTS = ["P1", "P2", "P3", "G1"]
RANK_METRICS = ["age_CCA", "age_best_PC_spearman",
                "cd4cd8_CCA", "cd4cd8_best_PC_spearman",
                "mait_CCA", "mait_best_PC_spearman",
                "ASW_batch", "iLISI_batch_norm", "ASW_file", "iLISI_file_norm"]
ABS = {"age_best_PC_spearman", "cd4cd8_best_PC_spearman", "mait_best_PC_spearman"}


def load_variant(v):
    row = {}
    for t in TESTS:
        p = os.path.join(ROOT, v, f"{t}.json")
        if os.path.exists(p):
            row.update({k: val for k, val in json.load(open(p)).items()})
    return row


def main():
    tab = pd.DataFrame({v: load_variant(v) for v in VARIANTS}).T  # variant x metric
    tab.to_csv(os.path.join(ROOT, "ablation_summary_health_aging.csv"), index_label="variant")
    metrics = [m for m in RANK_METRICS if m in tab.columns]
    rk = {}
    for m in metrics:
        s = tab[m].astype(float)
        if m in ABS:
            s = s.abs()
        rk[m] = s.rank(ascending=False, method="min", na_option="bottom")
    R = pd.DataFrame(rk)
    mean = R.mean(axis=1).sort_values()
    out = pd.DataFrame({"mean_rank": mean, "score_1_over_rank": 1.0 / mean})
    R.to_csv(os.path.join(ROOT, "ablation_ranks_by_metric_health_aging.csv"), index_label="variant")
    out.to_csv(os.path.join(ROOT, "ablation_ranking_health_aging.csv"), index_label="variant")
    pd.set_option("display.width", 200)
    print("=== per-metric rank (1=best) ===\n", R.round(2))
    print("\n=== health-aging OVERALL ranking ===\n", out.round(3))


if __name__ == "__main__":
    main()
