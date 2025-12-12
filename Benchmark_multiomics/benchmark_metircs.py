#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multimodal Integration Benchmark

Evaluates multimodal embeddings based on three criteria:
1. Paired sample matching: samples with same sample_id but different modality should be close
2. Modality mixing: modalities should be well-mixed (iLISI_norm, ASW_batch on modality)
3. Tissue preservation: within-tissue distances should be smaller than between-tissue distances

Usage:
    results = evaluate_multimodal_integration(
        meta_csv="sample_metadata.csv",
        embedding_csv="embeddings.csv",
        outdir="results/",
    )
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples


# =============================================================================
# I/O and Alignment
# =============================================================================

def read_metadata(meta_csv: str) -> pd.DataFrame:
    """Read metadata CSV with required columns."""
    md = pd.read_csv(meta_csv, index_col=0)
    md.columns = [c.lower() for c in md.columns]
    
    required = {"modality", "tissue"}
    missing = required - set(md.columns)
    if missing:
        raise ValueError(f"Metadata must contain columns {sorted(required)}; missing: {sorted(missing)}")
    
    # Derive sample_id by stripping _ATAC or _RNA suffix from index
    md["sample_id"] = md.index.str.replace(r"_(ATAC|RNA)$", "", regex=True)
    
    return md


def read_embedding(embedding_csv: str) -> pd.DataFrame:
    """Read embedding CSV (samples × dimensions)."""
    df = pd.read_csv(embedding_csv, index_col=0)
    if df.shape[1] < 1:
        raise ValueError("Embedding file must have ≥1 dimension columns.")
    return df


def align_data(md: pd.DataFrame, emb: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align metadata and embedding by sample index."""
    common = md.index.intersection(emb.index)
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between metadata and embedding.")
    
    if len(common) < len(md):
        print(f"Note: dropping {len(md) - len(common)} metadata rows without embedding.", file=sys.stderr)
    if len(common) < len(emb):
        print(f"Note: dropping {len(emb) - len(common)} embedding rows without metadata.", file=sys.stderr)
    
    # Sort to ensure consistent ordering
    common_sorted = sorted(common)
    return md.loc[common_sorted].copy(), emb.loc[common_sorted].copy()


# =============================================================================
# Metric 1: Paired Sample Distance
# =============================================================================

def compute_paired_distance(
    md: pd.DataFrame,
    emb: np.ndarray,
    sample_id_col: str = "sample_id",
    modality_col: str = "modality",
    metric: str = "euclidean",
) -> Dict[str, Any]:
    """
    Compute average distance between paired samples (same sample_id, different modality).
    
    Lower distance = better pairing/alignment of modalities.
    """
    # Build sample_id -> {modality: row_idx} mapping
    sample_id_to_idx: Dict[str, Dict[str, int]] = {}
    
    for i, (idx, row) in enumerate(md.iterrows()):
        sid = str(row[sample_id_col])
        mod = str(row[modality_col])
        if sid not in sample_id_to_idx:
            sample_id_to_idx[sid] = {}
        sample_id_to_idx[sid][mod] = i
    
    # Find all paired samples (those with exactly 2 modalities)
    paired_distances = []
    paired_info = []
    
    for sid, mod_dict in sample_id_to_idx.items():
        modalities = list(mod_dict.keys())
        if len(modalities) == 2:
            idx1 = mod_dict[modalities[0]]
            idx2 = mod_dict[modalities[1]]
            
            # Compute distance between paired samples
            vec1 = emb[idx1].reshape(1, -1)
            vec2 = emb[idx2].reshape(1, -1)
            dist = cdist(vec1, vec2, metric=metric)[0, 0]
            
            paired_distances.append(dist)
            paired_info.append({
                "sample_id": sid,
                "modality_1": modalities[0],
                "modality_2": modalities[1],
                "distance": dist,
            })
    
    if len(paired_distances) == 0:
        return {
            "n_pairs": 0,
            "mean_paired_distance": np.nan,
            "std_paired_distance": np.nan,
            "median_paired_distance": np.nan,
            "paired_details": [],
        }
    
    paired_distances = np.array(paired_distances)
    
    return {
        "n_pairs": len(paired_distances),
        "mean_paired_distance": float(np.mean(paired_distances)),
        "std_paired_distance": float(np.std(paired_distances, ddof=1)) if len(paired_distances) > 1 else 0.0,
        "median_paired_distance": float(np.median(paired_distances)),
        "min_paired_distance": float(np.min(paired_distances)),
        "max_paired_distance": float(np.max(paired_distances)),
        "paired_details": paired_info,
    }


# =============================================================================
# Metric 2: Modality Mixing (iLISI, ASW-batch)
# =============================================================================

def _inverse_simpson(counts: np.ndarray) -> float:
    """Compute inverse Simpson index from counts."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    denom = np.sum(p * p)
    return 0.0 if denom <= 0 else 1.0 / denom


def compute_ilisi(
    labels_int: np.ndarray,
    knn_idx: np.ndarray,
    include_self: bool = False,
) -> np.ndarray:
    """Compute per-sample iLISI (integration Local Inverse Simpson Index)."""
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


def compute_modality_mixing(
    md: pd.DataFrame,
    emb: np.ndarray,
    modality_col: str = "modality",
    k: int = 15,
    include_self: bool = False,
) -> Dict[str, Any]:
    """
    Compute modality mixing metrics: iLISI and ASW-batch.
    
    Higher iLISI_norm and ASW_batch = better modality mixing.
    """
    # Encode modality as integers
    modalities_str = md[modality_col].astype(str).values
    unique_modalities, labels_int = np.unique(modalities_str, return_inverse=True)
    n_modalities = len(unique_modalities)
    n_samples = emb.shape[0]
    
    # KNN for iLISI
    k_eff = min(max(int(k), 1), n_samples)
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=-1)
    nn.fit(emb)
    _, knn_idx = nn.kneighbors(emb)
    
    # iLISI
    ilisi_per = compute_ilisi(labels_int, knn_idx, include_self=include_self)
    ilisi_mean = float(np.mean(ilisi_per))
    ilisi_std = float(np.std(ilisi_per, ddof=1)) if n_samples > 1 else 0.0
    ilisi_norm_mean = float(ilisi_mean / max(1, n_modalities))
    
    # ASW-batch (higher = better mixing, using the (1-silhouette)/2 transformation)
    if n_modalities > 1 and n_samples > n_modalities:
        s_overall = silhouette_score(emb, labels_int, metric="euclidean")
        s_per = silhouette_samples(emb, labels_int, metric="euclidean")
        asw_overall = float(np.clip((1.0 - s_overall) / 2.0, 0.0, 1.0))
        asw_per = np.clip((1.0 - s_per) / 2.0, 0.0, 1.0)
    else:
        asw_overall = np.nan
        asw_per = np.full(n_samples, np.nan)
    
    return {
        "n_samples": n_samples,
        "n_modalities": n_modalities,
        "modalities": list(unique_modalities),
        "k_neighbors": k_eff,
        "iLISI_mean": ilisi_mean,
        "iLISI_std": ilisi_std,
        "iLISI_norm_mean": ilisi_norm_mean,
        "ASW_modality_overall": asw_overall,
        "iLISI_per_sample": ilisi_per,
        "ASW_per_sample": asw_per,
    }


# =============================================================================
# Metric 3: Tissue Preservation
# =============================================================================

def compute_tissue_preservation(
    md: pd.DataFrame,
    emb: np.ndarray,
    tissue_col: str = "tissue",
    metric: str = "euclidean",
) -> Dict[str, Any]:
    """
    Compute tissue preservation: ratio of between-tissue to within-tissue distance.
    
    Higher ratio = better tissue separation (biological signal preserved).
    
    Formula: tissue_preservation_score = mean_between_tissue_dist / mean_within_tissue_dist
    """
    tissues_str = md[tissue_col].astype(str).values
    unique_tissues, tissue_labels = np.unique(tissues_str, return_inverse=True)
    n_tissues = len(unique_tissues)
    
    if n_tissues < 2:
        return {
            "n_tissues": n_tissues,
            "tissues": list(unique_tissues),
            "mean_within_tissue_distance": np.nan,
            "mean_between_tissue_distance": np.nan,
            "tissue_preservation_score": np.nan,
            "tissue_details": {},
        }
    
    # Compute pairwise distance matrix
    dist_matrix = squareform(pdist(emb, metric=metric))
    
    # Compute within-tissue and between-tissue distances
    within_distances = []
    between_distances = []
    tissue_details = {}
    
    for t_idx, tissue in enumerate(unique_tissues):
        tissue_mask = tissue_labels == t_idx
        tissue_indices = np.where(tissue_mask)[0]
        other_indices = np.where(~tissue_mask)[0]
        
        # Within-tissue distances (upper triangle only to avoid double counting)
        if len(tissue_indices) > 1:
            within_dists = []
            for i in range(len(tissue_indices)):
                for j in range(i + 1, len(tissue_indices)):
                    within_dists.append(dist_matrix[tissue_indices[i], tissue_indices[j]])
            within_distances.extend(within_dists)
            tissue_details[tissue] = {
                "n_samples": len(tissue_indices),
                "mean_within_distance": float(np.mean(within_dists)) if within_dists else np.nan,
            }
        else:
            tissue_details[tissue] = {
                "n_samples": len(tissue_indices),
                "mean_within_distance": np.nan,
            }
        
        # Between-tissue distances
        if len(tissue_indices) > 0 and len(other_indices) > 0:
            between_dists = dist_matrix[np.ix_(tissue_indices, other_indices)].flatten()
            between_distances.extend(between_dists.tolist())
    
    mean_within = float(np.mean(within_distances)) if within_distances else np.nan
    mean_between = float(np.mean(between_distances)) if between_distances else np.nan
    
    # Tissue preservation score: higher = better separation
    if mean_within > 0 and not np.isnan(mean_within):
        preservation_score = mean_between / mean_within
    else:
        preservation_score = np.nan
    
    return {
        "n_tissues": n_tissues,
        "tissues": list(unique_tissues),
        "mean_within_tissue_distance": mean_within,
        "std_within_tissue_distance": float(np.std(within_distances, ddof=1)) if len(within_distances) > 1 else 0.0,
        "mean_between_tissue_distance": mean_between,
        "std_between_tissue_distance": float(np.std(between_distances, ddof=1)) if len(between_distances) > 1 else 0.0,
        "tissue_preservation_score": float(preservation_score) if not np.isnan(preservation_score) else np.nan,
        "tissue_details": tissue_details,
    }


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_multimodal_integration(
    meta_csv: str,
    embedding_csv: str,
    outdir: str,
    sample_id_col: str = "sample_id",
    modality_col: str = "modality",
    tissue_col: str = "tissue",
    k_neighbors: int = 15,
    distance_metric: str = "euclidean",
    include_self: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate multimodal integration quality.
    
    Parameters
    ----------
    meta_csv : str
        Path to metadata CSV with columns: sample_id, modality, tissue
    embedding_csv : str
        Path to embedding CSV (samples × dimensions), indexed by sample name
    outdir : str
        Output directory for results
    sample_id_col : str
        Column name for sample ID (pairs ATAC/RNA)
    modality_col : str
        Column name for modality (ATAC/RNA)
    tissue_col : str
        Column name for tissue type
    k_neighbors : int
        Number of neighbors for iLISI computation
    distance_metric : str
        Distance metric for pairwise distances
    include_self : bool
        Include self in KNN neighborhood
        
    Returns
    -------
    Dict with all metrics and paths to saved files
    """
    # Setup output directory
    os.makedirs(outdir, exist_ok=True)
    output_path = Path(outdir) / "Multimodal_Integration_Evaluation"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and align data
    print(f"Loading metadata from: {meta_csv}")
    md = read_metadata(meta_csv)
    
    print(f"Loading embedding from: {embedding_csv}")
    emb_df = read_embedding(embedding_csv)
    
    md_aligned, emb_aligned = align_data(md, emb_df)
    emb_array = emb_aligned.values.astype(float)
    
    print(f"Aligned data: {len(md_aligned)} samples, {emb_array.shape[1]} dimensions")
    
    # Compute all metrics
    print("\n1. Computing paired sample distances...")
    paired_results = compute_paired_distance(
        md_aligned, emb_array,
        sample_id_col=sample_id_col,
        modality_col=modality_col,
        metric=distance_metric,
    )
    
    print("\n2. Computing modality mixing (iLISI, ASW)...")
    mixing_results = compute_modality_mixing(
        md_aligned, emb_array,
        modality_col=modality_col,
        k=k_neighbors,
        include_self=include_self,
    )
    
    print("\n3. Computing tissue preservation...")
    tissue_results = compute_tissue_preservation(
        md_aligned, emb_array,
        tissue_col=tissue_col,
        metric=distance_metric,
    )
    
    # Save per-sample metrics
    per_sample_df = pd.DataFrame({
        "sample": md_aligned.index,
        "sample_id": md_aligned[sample_id_col].values,
        "modality": md_aligned[modality_col].values,
        "tissue": md_aligned[tissue_col].values,
        "iLISI": mixing_results["iLISI_per_sample"],
        "ASW_modality": mixing_results["ASW_per_sample"],
    }).set_index("sample")
    
    per_sample_path = output_path / "per_sample_metrics.csv"
    per_sample_df.to_csv(per_sample_path)
    
    # Save paired sample details
    if paired_results["paired_details"]:
        paired_df = pd.DataFrame(paired_results["paired_details"])
        paired_path = output_path / "paired_sample_distances.csv"
        paired_df.to_csv(paired_path, index=False)
    else:
        paired_path = None
    
    # Save tissue details
    tissue_details_df = pd.DataFrame(tissue_results["tissue_details"]).T
    tissue_details_df.index.name = "tissue"
    tissue_details_path = output_path / "tissue_details.csv"
    tissue_details_df.to_csv(tissue_details_path)
    
    # Generate summary
    summary_lines = [
        "=" * 60,
        "Multimodal Integration Evaluation Summary",
        "=" * 60,
        "",
        f"Total samples: {len(md_aligned)}",
        f"Embedding dimensions: {emb_array.shape[1]}",
        "",
        "--- 1. Paired Sample Distance (lower = better) ---",
        f"  Number of pairs: {paired_results['n_pairs']}",
        f"  Mean paired distance: {paired_results['mean_paired_distance']:.4f}",
        f"  Std paired distance:  {paired_results['std_paired_distance']:.4f}",
        f"  Median paired distance: {paired_results['median_paired_distance']:.4f}",
        "",
        "--- 2. Modality Mixing (higher = better) ---",
        f"  Modalities: {mixing_results['modalities']}",
        f"  iLISI mean:       {mixing_results['iLISI_mean']:.4f}",
        f"  iLISI normalized: {mixing_results['iLISI_norm_mean']:.4f}",
        f"  ASW modality:     {mixing_results['ASW_modality_overall']:.4f}",
        "",
        "--- 3. Tissue Preservation (higher = better) ---",
        f"  Number of tissues: {tissue_results['n_tissues']}",
        f"  Tissues: {tissue_results['tissues']}",
        f"  Mean within-tissue distance:  {tissue_results['mean_within_tissue_distance']:.4f}",
        f"  Mean between-tissue distance: {tissue_results['mean_between_tissue_distance']:.4f}",
        f"  Tissue preservation score:    {tissue_results['tissue_preservation_score']:.4f}",
        "",
        "=" * 60,
        f"Results saved to: {output_path}",
        "=" * 60,
    ]
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    summary_path = output_path / "integration_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    
    # Aggregate results for return
    return {
        # Core metrics for summary CSV aggregation
        "n_samples": len(md_aligned),
        "n_pairs": paired_results["n_pairs"],
        "mean_paired_distance": paired_results["mean_paired_distance"],
        "std_paired_distance": paired_results["std_paired_distance"],
        "median_paired_distance": paired_results["median_paired_distance"],
        "iLISI_mean": mixing_results["iLISI_mean"],
        "iLISI_norm_mean": mixing_results["iLISI_norm_mean"],
        "ASW_modality_overall": mixing_results["ASW_modality_overall"],
        "tissue_preservation_score": tissue_results["tissue_preservation_score"],
        "mean_within_tissue_distance": tissue_results["mean_within_tissue_distance"],
        "mean_between_tissue_distance": tissue_results["mean_between_tissue_distance"],
        # Metadata
        "n_modalities": mixing_results["n_modalities"],
        "modalities": mixing_results["modalities"],
        "n_tissues": tissue_results["n_tissues"],
        "tissues": tissue_results["tissues"],
        # File paths
        "per_sample_path": str(per_sample_path),
        "paired_path": str(paired_path) if paired_path else None,
        "tissue_details_path": str(tissue_details_path),
        "summary_path": str(summary_path),
    }


# =============================================================================
# Summary CSV Aggregation (following BenchmarkWrapper pattern)
# =============================================================================

DEFAULT_SUMMARY_CSV = "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/summary.csv"


def save_to_summary_csv(
    results: Dict[str, Any],
    method_name: str,
    summary_csv_path: str = DEFAULT_SUMMARY_CSV,
) -> None:
    """
    Save results to a summary CSV file, appending as a new column.
    
    Structure:
    - Rows: metric names
    - Columns: method_name (e.g., GLUE, scGLUE)
    """
    summary_path = Path(summary_csv_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Metrics to save (excluding lists and paths)
    metrics_to_save = {
        "n_samples": results.get("n_samples"),
        "n_pairs": results.get("n_pairs"),
        "mean_paired_distance": results.get("mean_paired_distance"),
        "median_paired_distance": results.get("median_paired_distance"),
        "iLISI_mean": results.get("iLISI_mean"),
        "iLISI_norm_mean": results.get("iLISI_norm_mean"),
        "ASW_modality": results.get("ASW_modality_overall"),
        "tissue_preservation_score": results.get("tissue_preservation_score"),
        "mean_within_tissue_distance": results.get("mean_within_tissue_distance"),
        "mean_between_tissue_distance": results.get("mean_between_tissue_distance"),
    }
    
    # Column name is just method_name
    col_name = method_name
    
    # Load existing or create new
    if summary_path.exists() and summary_path.stat().st_size > 0:
        try:
            summary_df = pd.read_csv(summary_path, index_col=0)
        except pd.errors.EmptyDataError:
            summary_df = pd.DataFrame()
    else:
        summary_df = pd.DataFrame()
    
    # Add/update column
    for metric, value in metrics_to_save.items():
        summary_df.loc[metric, col_name] = value
    
    # Save
    summary_df.to_csv(summary_path, index_label="Metric")
    print(f"\nUpdated summary CSV: {summary_path} with column '{col_name}'")


if __name__ == "__main__":
    # Example usage with sample data
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_SD/multiomics/rna/Sample_distance/correlation/expression_DR_distance/expression_DR_coordinates.csv",
        outdir="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/SD_expression",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="SD_expression")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_SD/multiomics/rna/Sample_distance/correlation/proportion_DR_distance/proportion_DR_coordinates.csv",
        outdir="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/SD_proportion",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="SD_proportion")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/pilot/pilot_native_embedding.csv",
        outdir="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/pilot",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="pilot")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/pseudobulk/pseudobulk/pca_embeddings.csv",
        outdir="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/pseudobulk",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="pseudobulk")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/QOT/88_qot_distance_matrix_mds_10d.csv",
        outdir="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/QOT",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="QOT")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/GEDI/gedi_sample_embedding.csv",
        outdir="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/GEDI",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="GEDI")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/Gloscope/knn_divergence_mds_10d.csv",
        outdir="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/Gloscope",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="Gloscope")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/MFA/sample_embeddings.csv",
        outdir="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/MFA",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="MFA")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/mustard/sample_embedding.csv",
        outdir="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/mustard",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="mustard")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/scPoli/sample_embeddings_full.csv",
        outdir="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/scPoli",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="scPoli")
    
    
    print("Multimodal Integration Benchmark module loaded.")
    print("Use evaluate_multimodal_integration() to run evaluation.")