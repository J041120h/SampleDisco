#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified Batch-Removal Evaluation using iLISI and ASW-batch.

Usage: Edit the function call in main() with your file paths and run the script.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples


def read_metadata(meta_csv: str) -> pd.DataFrame:
    md = pd.read_csv(meta_csv)
    md.columns = [c.lower() for c in md.columns]
    required = {"sample", "batch"}
    missing = required - set(md.columns)
    if missing:
        raise ValueError(f"Metadata must contain columns {sorted(required)}; missing: {sorted(missing)}")
    if md["sample"].duplicated().any():
        dups = md.loc[md["sample"].duplicated(), "sample"].unique()
        raise ValueError(f"'sample' IDs must be unique. Duplicates: {dups[:10]}")
    return md


def read_embedding_or_distance(data_csv: str, mode: str) -> Tuple[pd.DataFrame, bool]:
    df = pd.read_csv(data_csv, index_col=0)
    if mode == "distance":
        if df.shape[0] != df.shape[1]:
            raise ValueError("Distance matrix must be square (samples × samples).")
        # Ensure zero diagonal and symmetry
        vals = df.values.astype(float)
        np.fill_diagonal(vals, 0.0)
        if not np.allclose(vals, vals.T, equal_nan=True, rtol=1e-5, atol=1e-8):
            print("Warning: distance matrix not exactly symmetric; proceeding.", file=sys.stderr)
        # Update the dataframe with corrected values
        df.iloc[:, :] = vals
        return df, True
    elif mode == "embedding":
        if df.shape[1] < 1:
            raise ValueError("Embedding file must have ≥1 dimension columns.")
        return df, False
    else:
        raise ValueError("mode must be 'embedding' or 'distance'.")


def align_by_samples(md: pd.DataFrame, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if md.index.name != "sample":
        md = md.set_index("sample", drop=False)
    common = md.index.intersection(data.index)
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between metadata and data.")
    if len(common) < len(md):
        print(f"Note: dropping {len(md) - len(common)} metadata rows without data match.", file=sys.stderr)
    if len(common) < len(data):
        print(f"Note: dropping {len(data) - len(common)} data rows without metadata.", file=sys.stderr)
    md2 = md.loc[sorted(common)].copy()
    data2 = data.loc[sorted(common)].copy()
    return md2, data2


def _compute_knn_from_embedding(emb: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k, emb.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine", n_jobs=-1)
    nn.fit(emb)
    _, idx = nn.kneighbors(emb, return_distance=True)
    return idx


def _compute_knn_from_distance(dist: np.ndarray, k: int) -> np.ndarray:
    n = dist.shape[0]
    k_eff = min(k, n)
    return np.argsort(dist, axis=1)[:, :k_eff]


def _inverse_simpson(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    denom = np.sum(p * p)
    return 0.0 if denom <= 0 else 1.0 / denom


def compute_ilisi(labels_int: np.ndarray, knn_idx: np.ndarray, include_self: bool) -> np.ndarray:
    n = labels_int.shape[0]
    L = int(labels_int.max()) + 1
    out = np.zeros(n, dtype=float)
    for i in range(n):
        neigh = knn_idx[i]
        if not include_self:
            neigh = neigh[neigh != i]
        counts = np.bincount(labels_int[neigh], minlength=L)
        out[i] = _inverse_simpson(counts)
    return out


def compute_asw_batch(labels_int: np.ndarray, emb: np.ndarray | None, dist: np.ndarray | None) -> Tuple[float, np.ndarray]:
    if (emb is None) == (dist is None):
        raise ValueError("Provide exactly one of emb or dist.")
    if emb is not None:
        s_overall = silhouette_score(emb, labels_int, metric="euclidean")
        s_per = silhouette_samples(emb, labels_int, metric="euclidean")
    else:
        s_overall = silhouette_score(dist, labels_int, metric="precomputed")
        s_per = silhouette_samples(dist, labels_int, metric="precomputed")
    asw_overall = float(np.clip((1.0 - s_overall) / 2.0, 0.0, 1.0))
    asw_per = np.clip((1.0 - s_per) / 2.0, 0.0, 1.0)
    return asw_overall, asw_per


def evaluate_batch_removal(
    meta_csv: str,
    data_csv: str,
    mode: str,
    outdir: str,
    k: int = 30,
    include_self: bool = False,
) -> Dict[str, object]:
    """
    Compute iLISI and ASW-batch metrics for batch effect evaluation.

    Parameters
    ----------
    meta_csv : str
        Path to metadata CSV with 'sample' and 'batch' columns
    data_csv : str  
        Path to data CSV (embedding: samples×dims, distance: samples×samples)
    mode : str
        'embedding' or 'distance'
    outdir : str
        Output directory for results
    k : int, default=30
        Number of neighbors for iLISI KNN
    include_self : bool, default=False
        Include sample itself in KNN neighborhood
    """
    import os
    os.makedirs(output_dir_path, exist_ok=True)
    output_dir_path = os.path.join(output_dir_path, 'Batch_Removal_Evaluation')
    os.makedirs(output_dir_path, exist_ok=True)

    # Read & align data
    md = read_metadata(meta_csv)
    data_df, is_distance = read_embedding_or_distance(data_csv, mode)
    md_aln, data_aln = align_by_samples(md, data_df)
    
    # Prepare arrays
    batches_str = md_aln["batch"].astype(str).values
    uniq_batches, labels_int = np.unique(batches_str, return_inverse=True)
    n_batches = len(uniq_batches)
    n_samples = data_aln.shape[0]
    k_eff = min(max(int(k), 1), n_samples)

    # Compute metrics
    if mode == "embedding":
        emb = data_aln.values.astype(float)
        knn_idx = _compute_knn_from_embedding(emb, k=k_eff)
        dist = None
    else:
        dist = data_aln.values.astype(float)
        # Ensure diagonal is zero for silhouette calculation
        np.fill_diagonal(dist, 0.0)
        knn_idx = _compute_knn_from_distance(dist, k=k_eff)
        emb = None

    ilisi_per = compute_ilisi(labels_int, knn_idx, include_self=include_self)
    ilisi_mean = float(np.mean(ilisi_per))
    ilisi_std = float(np.std(ilisi_per, ddof=1)) if n_samples > 1 else 0.0
    ilisi_norm_mean = float(ilisi_mean / max(1, n_batches))

    asw_overall, asw_per = compute_asw_batch(labels_int, emb=emb, dist=dist)

    # Save results
    per_sample_df = pd.DataFrame({
        "sample": md_aln.index.astype(str),
        "batch": batches_str,
        "iLISI": ilisi_per,
        "ASW_batch": asw_per,
    }).set_index("sample")
    
    per_sample_path = outdir_p / "per_sample_metrics.csv"
    per_sample_df.to_csv(per_sample_path)

    # Summary
    summary_lines = [
        "Batch-removal Evaluation Summary",
        "================================",
        f"Samples: {n_samples}",
        f"Batches: {n_batches} -> {list(uniq_batches)}",
        "",
        f"iLISI (mean ± sd): {ilisi_mean:.4f} ± {ilisi_std:.4f}",
        f"iLISI_norm: {ilisi_norm_mean:.4f}",
        f"ASW-batch: {asw_overall:.4f}",
        "",
        f"Results saved to: {outdir_p}",
    ]
    summary_path = outdir_p / "batch_info_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))

    return {
        "n_samples": n_samples,
        "batches": list(uniq_batches),
        "iLISI_mean": ilisi_mean,
        "iLISI_std": ilisi_std,
        "iLISI_norm_mean": ilisi_norm_mean,
        "ASW_batch_overall": asw_overall,
        "per_sample_path": str(per_sample_path),
        "summary_path": str(summary_path),
    }


if __name__ == "__main__":
    # EDIT THESE PATHS:
    evaluate_batch_removal(
        meta_csv="/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv",
        data_csv="/dcs07/hongkai/data/harry/result/Benchmark/covid_25_sample/rna/Sample_distance/cosine/expression_DR_distance/expression_DR_coordinates.csv",
        mode="embedding",  # or "embedding"
        outdir="/dcs07/hongkai/data/harry/result/Benchmark/covid_25_sample",
        k=15,  # reduced from 30 since only 25 samples
        include_self=False
    )
