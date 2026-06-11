#!/usr/bin/env python
"""Does Harmony remove BIOLOGICAL signal along with batch?

Clean single-knob contrast (identical blocks A1/A2/A3 + RMD; only the
sample-level batch-removal step differs): no_batch_removal vs linear_regression
vs original(Harmony). For each dataset we report RAW values of a batch-removal
axis and a biological-preservation axis, and the delta of Harmony vs no-batch.
If Harmony improves the batch axis but LOWERS the biological axis vs no_batch,
the over-correction concern is supported for that dataset.
"""
import os
import numpy as np
import pandas as pd

ABL = "/dcs07/hongkai/data/harry/result/ablation"
V3 = ["no_batch_removal", "linear_regression", "original"]
rows = []   # dataset, axis(batch/bio), metric, direction, no_batch, linear, original


def add(ds, axis, metric, direction, vals):
    rows.append(dict(dataset=ds, axis=axis, metric=metric, direction=direction,
                     no_batch_removal=vals.get("no_batch_removal"),
                     linear_regression=vals.get("linear_regression"),
                     original=vals.get("original")))


# ---- COVID: average each metric across the 6 sizes ----
f = f"{ABL}/covid/ablation_summary_covid_ALL.csv"
if os.path.exists(f):
    df = pd.read_csv(f, index_col=0)
    sizes = sorted({c.split("-")[1] for c in df.columns}, key=int)
    def covid_mean(metric, absval=False):
        out = {}
        for v in V3:
            cols = [f"{v}-{s}" for s in sizes if f"{v}-{s}" in df.columns]
            s = pd.to_numeric(df.loc[metric, cols], errors="coerce")
            if absval: s = s.abs()
            out[v] = float(s.mean())
        return out
    add("COVID", "batch", "batch_partial_eta_sq", "lower=better", covid_mean("batch_partial_eta_sq"))
    add("COVID", "batch", "iLISI_norm", "higher=better", covid_mean("iLISI_norm"))
    add("COVID", "batch", "ASW_batch", "higher=better", covid_mean("ASW_batch"))
    add("COVID", "bio", "severity_partial_eta_sq", "higher=better", covid_mean("severity_partial_eta_sq"))
    add("COVID", "bio", "|Spearman|", "higher=better", covid_mean("Spearman_Correlation", absval=True))
    add("COVID", "bio", "ARI", "higher=better", covid_mean("ARI"))
    add("COVID", "bio", "Custom_ANOVA_eta_sq", "higher=better", covid_mean("Custom_ANOVA_eta_sq"))

# ---- multi-omics ----
PRES = {"ENCODE": "tissue_preservation_score", "heart": "disease_state_preservation_score",
        "retina": "cca_score", "lutea": "cca_score"}
for ds, pres in PRES.items():
    f = f"{ABL}/multiomics/{ds}/ablation_summary_mo_{ds}.csv"
    if not os.path.exists(f): continue
    df = pd.read_csv(f, index_col=0)
    def mo(metric): return {v: float(df.loc[metric, v]) for v in V3 if v in df.columns}
    add(ds, "batch", "ASW_modality", "higher=better", mo("ASW_modality"))
    add(ds, "pairing", "paired_partner_rank", "lower=better", mo("paired_partner_rank"))
    add(ds, "bio", pres, "higher=better", mo(pres))

# ---- health-aging ----
f = f"{ABL}/health_aging/ablation_summary_health_aging.csv"
if os.path.exists(f):
    df = pd.read_csv(f, index_col=0)   # variant x metric
    def ha(metric, absval=False):
        out = {}
        for v in V3:
            x = float(df.loc[v, metric])
            out[v] = abs(x) if absval else x
        return out
    add("health-aging", "batch", "ASW_batch", "higher=better", ha("ASW_batch"))
    add("health-aging", "batch", "iLISI_batch_norm", "higher=better", ha("iLISI_batch_norm"))
    add("health-aging", "bio", "age_CCA", "higher=better", ha("age_CCA"))
    add("health-aging", "bio", "cd4cd8_CCA", "higher=better", ha("cd4cd8_CCA"))
    add("health-aging", "bio", "mait_CCA", "higher=better", ha("mait_CCA"))

tab = pd.DataFrame(rows)
tab.to_csv(f"{ABL}/harmony_tradeoff.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print(tab.round(4).to_string(index=False))

# ---- verdict per dataset: Harmony better on batch? worse on bio (vs no_batch)? ----
print("\n=== VERDICT (Harmony=original vs no_batch_removal) ===")
def better(direction, a, b):  # is a better than b?
    return (a < b) if "lower" in direction else (a > b)
for ds in tab["dataset"].unique():
    sub = tab[tab.dataset == ds]
    batch = sub[sub.axis == "batch"]; bio = sub[sub.axis == "bio"]
    b_better = sum(better(r.direction, r.original, r.no_batch_removal) for r in batch.itertuples())
    bio_worse = sum((not better(r.direction, r.original, r.no_batch_removal)) for r in bio.itertuples())
    print(f"{ds:14s}: Harmony better-on-batch {b_better}/{len(batch)} | "
          f"Harmony worse-on-bio-vs-noBatch {bio_worse}/{len(bio)}")
