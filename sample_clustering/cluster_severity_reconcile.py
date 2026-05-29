"""Reconcile data-driven KMeans sample clusters with a known sample label
(e.g. severity) via an optimal 1:1 majority-vote assignment.

KMeans on the sample embedding yields arbitrary cluster ids (0..k-1). To make
those clusters interpretable — and usable as a grouping for differential
analysis — we match each cluster to one level of a reference label so that
each reference level is represented by exactly one cluster. A greedy
per-cluster majority vote can assign two clusters to the same level (leaving
another level unrepresented); the Hungarian algorithm on the
cluster × level count matrix instead maximizes total agreement under a strict
bijection, guaranteeing "each level has one cluster" when k == n_levels.

Reports the contingency table, the chosen mapping, overall accuracy and the
confusion matrix so the user can see how well the embedding clusters recover
the reference label.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def reconcile_clusters_to_label(
    sample_to_cluster: Dict[str, object],
    sample_to_label: Dict[str, object],
    *,
    label_name: str = "severity",
    output_txt: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, object], Dict[object, object], dict]:
    """Match KMeans cluster ids to reference-label levels (1:1, Hungarian).

    Parameters
    ----------
    sample_to_cluster
        {sample_id -> kmeans cluster id}.
    sample_to_label
        {sample_id -> reference level} (e.g. severity). Samples missing a
        label are dropped from the matching.
    label_name
        Name of the reference label, used only in the report.
    output_txt
        If given, write the human-readable reconciliation report here.

    Returns
    -------
    sample_to_pred
        {sample_id -> matched reference level} for every sample in
        ``sample_to_cluster`` (cluster id mapped through the bijection).
    cluster_to_label
        {cluster id -> reference level} mapping chosen by the assignment.
    stats
        dict with keys ``accuracy``, ``contingency`` (DataFrame),
        ``confusion`` (DataFrame), ``n_matched``.
    """
    common = [s for s in sample_to_cluster if s in sample_to_label
              and pd.notna(sample_to_label[s])]
    if len(common) == 0:
        raise ValueError("No samples have both a cluster and a reference label.")

    clusters = pd.Series({s: sample_to_cluster[s] for s in common}, name="cluster")
    labels = pd.Series({s: sample_to_label[s] for s in common}, name=label_name)

    cluster_levels = sorted(clusters.unique(), key=lambda x: str(x))
    label_levels = sorted(labels.unique(), key=lambda x: str(x))

    # Contingency: rows = cluster, cols = reference level.
    contingency = pd.crosstab(clusters, labels).reindex(
        index=cluster_levels, columns=label_levels, fill_value=0)

    # Hungarian on -counts → maximize agreement under a 1:1 matching.
    cost = -contingency.values.astype(float)
    row_idx, col_idx = linear_sum_assignment(cost)
    cluster_to_label = {cluster_levels[r]: label_levels[c]
                        for r, c in zip(row_idx, col_idx)}

    # Any cluster not matched (when k > n_levels) → assign its row-majority level.
    for cl in cluster_levels:
        if cl not in cluster_to_label:
            cluster_to_label[cl] = contingency.loc[cl].idxmax()

    sample_to_pred = {s: cluster_to_label[c] for s, c in sample_to_cluster.items()}

    pred = pd.Series({s: cluster_to_label[clusters[s]] for s in common})
    accuracy = float((pred.values == labels.values).mean())
    confusion = pd.crosstab(labels, pred,
                            rownames=[f"true_{label_name}"],
                            colnames=[f"pred_{label_name}"]).reindex(
        index=label_levels, columns=label_levels, fill_value=0)

    stats = {"accuracy": accuracy, "contingency": contingency,
             "confusion": confusion, "n_matched": len(common)}

    report = _format_report(label_name, contingency, cluster_to_label,
                            accuracy, confusion, len(common))
    if verbose:
        print(report)
    if output_txt:
        with open(output_txt, "w") as fh:
            fh.write(report)
        if verbose:
            print(f"[reconcile] wrote {output_txt}")

    return sample_to_pred, cluster_to_label, stats


def _format_report(label_name, contingency, cluster_to_label,
                   accuracy, confusion, n_matched) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"KMeans cluster ↔ {label_name} reconciliation (Hungarian 1:1)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Samples matched: {n_matched}")
    lines.append(f"Overall agreement (accuracy): {accuracy:.4f}")
    lines.append("")
    lines.append(f"Contingency (rows=KMeans cluster, cols={label_name}):")
    lines.append(contingency.to_string())
    lines.append("")
    lines.append(f"Chosen mapping (cluster → {label_name}):")
    for cl, lab in sorted(cluster_to_label.items(), key=lambda kv: str(kv[1])):
        lines.append(f"  cluster {cl}  →  {label_name} {lab}")
    lines.append("")
    lines.append(f"Confusion (rows=true {label_name}, cols=predicted {label_name}):")
    lines.append(confusion.to_string())
    lines.append("")
    return "\n".join(lines)
