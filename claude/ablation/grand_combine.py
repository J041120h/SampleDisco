#!/usr/bin/env python
"""Combine the per-dataset variant mean-ranks into one cross-dataset overview.

Reads the per-dataset rankings (COVID, the 4 paired multi-omics datasets, and
health-aging) and builds a variant x dataset table of mean ranks plus an
overall cross-dataset mean rank (each dataset weighted equally).
"""
import os
import numpy as np
import pandas as pd

ABL = "/dcs07/hongkai/data/harry/result/ablation"
VARIANTS = ["proportion_only", "rmd_only", "no_batch_removal",
            "linear_regression", "original"]


def main():
    cols = {}
    # COVID
    f = f"{ABL}/covid/ablation_ranking_covid.csv"
    if os.path.exists(f):
        cols["COVID"] = pd.read_csv(f, index_col=0)["mean_rank"]
    # MO datasets (per-dataset columns are in the combined file)
    f = f"{ABL}/multiomics/ablation_ranking_mo_combined.csv"
    if os.path.exists(f):
        mo = pd.read_csv(f, index_col=0)
        for ds in ["ENCODE", "heart", "retina", "lutea"]:
            if ds in mo.columns:
                cols[ds] = mo[ds]
    # health-aging
    f = f"{ABL}/health_aging/ablation_ranking_health_aging.csv"
    if os.path.exists(f):
        cols["health_aging"] = pd.read_csv(f, index_col=0)["mean_rank"]

    tab = pd.DataFrame(cols).reindex(VARIANTS)
    tab["overall_mean_rank"] = tab.mean(axis=1)
    tab["score_1_over_rank"] = 1.0 / tab["overall_mean_rank"]
    tab = tab.sort_values("overall_mean_rank")
    tab.to_csv(f"{ABL}/ablation_GRAND_overview.csv", index_label="variant")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print("=== GRAND cross-dataset ablation overview (per-dataset mean rank; 1=best of 5) ===\n")
    print(tab.round(3))


if __name__ == "__main__":
    main()
