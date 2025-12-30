#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multimodal Integration Benchmark - FLEXIBLE VERSION

Evaluates multimodal embeddings based on three criteria:
1. Paired sample matching: samples with same sample_id but different modality should be close
2. Modality mixing: modalities should be well-mixed (iLISI_norm, ASW_batch on modality)
3. Biological grouping preservation: within-group distances should be smaller than between-group distances

Usage:
    # Use organ_part for preservation (if multiple organs)
    results = evaluate_multimodal_integration(
        meta_csv="sample_metadata.csv",
        embedding_csv="embeddings.csv",
        outdir="results/",
        preservation_col="organ_part",
    )
    
    # Or use disease_state for preservation
    results = evaluate_multimodal_integration(
        meta_csv="sample_metadata.csv",
        embedding_csv="embeddings.csv",
        outdir="results/",
        preservation_col="disease_state",
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

def read_metadata(meta_csv: str, modalities: List[str] = ["RNA", "ATAC"]) -> pd.DataFrame:
    """
    Read metadata CSV and expand to include both modalities.
    """
    md = pd.read_csv(meta_csv, index_col=0)
    md.columns = [c.lower() for c in md.columns]
    
    # Remove rows with NaN index
    md = md[md.index.notna()]
    
    # Store original sample IDs
    md["sample_id"] = md.index
    
    # Expand metadata for each modality
    expanded_rows = []
    for sample_id, row in md.iterrows():
        for modality in modalities:
            new_row = row.copy()
            new_row["modality"] = modality
            new_row.name = f"{sample_id}_{modality}"
            expanded_rows.append(new_row)
    
    md_expanded = pd.DataFrame(expanded_rows)
    
    print(f"Expanded metadata from {len(md)} samples to {len(md_expanded)} samples ({len(modalities)} modalities)")
    
    return md_expanded


def read_embedding(embedding_csv: str) -> pd.DataFrame:
    """Read embedding CSV (samples × dimensions)."""
    df = pd.read_csv(embedding_csv, index_col=0)
    if df.shape[1] < 1:
        raise ValueError("Embedding file must have ≥1 dimension columns.")
    return df


def align_data(md: pd.DataFrame, emb: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align metadata and embedding by sample index.
    Handles case-insensitive matching.
    """
    # Create case-normalized mapping for embedding (convert to string first to handle numeric indices)
    emb_lower_to_original = {str(idx).lower(): idx for idx in emb.index}
    md_lower_to_original = {str(idx).lower(): idx for idx in md.index}
    
    # Find common samples (case-insensitive)
    common_lower = set(emb_lower_to_original.keys()).intersection(set(md_lower_to_original.keys()))
    
    if len(common_lower) == 0:
        print("\n" + "!"*60)
        print("ERROR: No overlapping sample IDs found!")
        print("!"*60)
        print("\nMetadata sample names (first 10):")
        for i, name in enumerate(list(md.index)[:10]):
            print(f"  {i+1}. {name}")
        
        print("\nEmbedding sample names (first 10):")
        for i, name in enumerate(list(emb.index)[:10]):
            print(f"  {i+1}. {name}")
        
        print("\nPossible issues:")
        print("  1. Embedding file has different sample naming format")
        print("  2. Sample IDs don't include modality suffix (_RNA, _ATAC)")
        print("  3. Sample IDs have different prefixes or formats")
        print("  4. Embedding file has numeric indices instead of sample names")
        print("\nPlease check the embedding file format and ensure sample names match the pattern:")
        print("  Expected: <sample_id>_<modality> (e.g., MA10_heart_RNA, MA10_heart_ATAC)")
        
        raise ValueError("No overlapping sample IDs between metadata and embedding.")
    
    # Map back to original case from embedding (use embedding's case as canonical)
    common_emb_original = [emb_lower_to_original[k] for k in sorted(common_lower)]
    common_md_original = [md_lower_to_original[k] for k in sorted(common_lower)]
    
    if len(common_lower) < len(md):
        print(f"Note: dropping {len(md) - len(common_lower)} metadata rows without embedding.", file=sys.stderr)
    if len(common_lower) < len(emb):
        print(f"Note: dropping {len(emb) - len(common_lower)} embedding rows without metadata.", file=sys.stderr)
    
    # Align using original indices
    md_aligned = md.loc[common_md_original].copy()
    emb_aligned = emb.loc[common_emb_original].copy()
    
    # Update metadata index to match embedding case
    md_aligned.index = emb_aligned.index
    
    return md_aligned, emb_aligned


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
# Metric 3: Biological Group Preservation (Flexible)
# =============================================================================

def compute_group_preservation(
    md: pd.DataFrame,
    emb: np.ndarray,
    group_col: str,
    metric: str = "euclidean",
) -> Dict[str, Any]:
    """
    Compute biological group preservation: ratio of between-group to within-group distance.
    
    Higher ratio = better group separation (biological signal preserved).
    
    Formula: preservation_score = mean_between_group_dist / mean_within_group_dist
    
    Parameters
    ----------
    group_col : str
        Column name for grouping (e.g., 'organ_part', 'disease_state', 'cell_type')
    
    Returns
    -------
    Dict with preservation metrics (returns NaN if only 1 group)
    """
    groups_str = md[group_col].astype(str).values
    unique_groups, group_labels = np.unique(groups_str, return_inverse=True)
    n_groups = len(unique_groups)
    
    if n_groups < 2:
        print(f"\nWARNING: Only {n_groups} unique {group_col} value(s) found: {unique_groups.tolist()}")
        print(f"Preservation score requires ≥2 groups to compute within vs. between distances.")
        print(f"Returning NaN for {group_col} preservation metrics.")
        return {
            "n_groups": n_groups,
            "groups": list(unique_groups),
            "mean_within_group_distance": np.nan,
            "mean_between_group_distance": np.nan,
            "preservation_score": np.nan,
            "group_details": {},
            "group_col": group_col,
        }
    
    # Compute pairwise distance matrix
    dist_matrix = squareform(pdist(emb, metric=metric))
    
    # Compute within-group and between-group distances
    within_distances = []
    between_distances = []
    group_details = {}
    
    for g_idx, group in enumerate(unique_groups):
        group_mask = group_labels == g_idx
        group_indices = np.where(group_mask)[0]
        other_indices = np.where(~group_mask)[0]
        
        # Within-group distances (upper triangle only to avoid double counting)
        if len(group_indices) > 1:
            within_dists = []
            for i in range(len(group_indices)):
                for j in range(i + 1, len(group_indices)):
                    within_dists.append(dist_matrix[group_indices[i], group_indices[j]])
            within_distances.extend(within_dists)
            group_details[group] = {
                "n_samples": len(group_indices),
                "mean_within_distance": float(np.mean(within_dists)) if within_dists else np.nan,
            }
        else:
            group_details[group] = {
                "n_samples": len(group_indices),
                "mean_within_distance": np.nan,
            }
        
        # Between-group distances
        if len(group_indices) > 0 and len(other_indices) > 0:
            between_dists = dist_matrix[np.ix_(group_indices, other_indices)].flatten()
            between_distances.extend(between_dists.tolist())
    
    mean_within = float(np.mean(within_distances)) if within_distances else np.nan
    mean_between = float(np.mean(between_distances)) if between_distances else np.nan
    
    # Preservation score: higher = better separation
    if mean_within > 0 and not np.isnan(mean_within):
        preservation_score = mean_between / mean_within
    else:
        preservation_score = np.nan
    
    return {
        "n_groups": n_groups,
        "groups": list(unique_groups),
        "mean_within_group_distance": mean_within,
        "std_within_group_distance": float(np.std(within_distances, ddof=1)) if len(within_distances) > 1 else 0.0,
        "mean_between_group_distance": mean_between,
        "std_between_group_distance": float(np.std(between_distances, ddof=1)) if len(between_distances) > 1 else 0.0,
        "preservation_score": float(preservation_score) if not np.isnan(preservation_score) else np.nan,
        "group_details": group_details,
        "group_col": group_col,
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
    preservation_col: str = "disease_state",  # Changed default to disease_state
    modalities: List[str] = ["RNA", "ATAC"],
    k_neighbors: int = 15,
    distance_metric: str = "euclidean",
    include_self: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate multimodal integration quality.
    
    Parameters
    ----------
    meta_csv : str
        Path to metadata CSV
        Index contains original sample names (e.g., sample1, sample2)
    embedding_csv : str
        Path to embedding CSV (samples × dimensions), indexed by sample name
        Sample names must be in format: <sample_id>_<modality> (e.g., sample1_RNA, sample1_ATAC)
    outdir : str
        Output directory for results
    sample_id_col : str
        Column name for sample ID (pairs ATAC/RNA) - automatically derived
    modality_col : str
        Column name for modality (ATAC/RNA) - automatically derived
    preservation_col : str
        Column name for biological grouping preservation metric
        Options: 'organ_part', 'disease_state', 'cell_type', etc.
        Default: 'disease_state' (for heart dataset with different disease states)
    modalities : List[str]
        List of modalities (default: ["RNA", "ATAC"])
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
    md = read_metadata(meta_csv, modalities=modalities)
    
    print(f"Loading embedding from: {embedding_csv}")
    emb_df = read_embedding(embedding_csv)
    
    md_aligned, emb_aligned = align_data(md, emb_df)
    emb_array = emb_aligned.values.astype(float)
    
    print(f"Aligned data: {len(md_aligned)} samples, {emb_array.shape[1]} dimensions")
    print(f"Detected modalities: {sorted(md_aligned[modality_col].unique())}")
    print(f"Number of unique biological samples: {md_aligned[sample_id_col].nunique()}")
    print(f"Preservation grouping by '{preservation_col}': {sorted(md_aligned[preservation_col].dropna().unique())}")
    
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
    
    print(f"\n3. Computing {preservation_col} preservation...")
    group_results = compute_group_preservation(
        md_aligned, emb_array,
        group_col=preservation_col,
        metric=distance_metric,
    )
    
    # Save per-sample metrics
    per_sample_df = pd.DataFrame({
        "sample": md_aligned.index,
        "sample_id": md_aligned[sample_id_col].values,
        "modality": md_aligned[modality_col].values,
        preservation_col: md_aligned[preservation_col].values,
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
    
    # Save group details
    group_details_df = pd.DataFrame(group_results["group_details"]).T
    group_details_df.index.name = preservation_col
    group_details_path = output_path / f"{preservation_col}_details.csv"
    group_details_df.to_csv(group_details_path)
    
    # Generate summary
    summary_lines = [
        "=" * 60,
        "Multimodal Integration Evaluation Summary",
        "=" * 60,
        "",
        f"Total samples: {len(md_aligned)}",
        f"Unique biological samples: {md_aligned[sample_id_col].nunique()}",
        f"Modalities: {sorted(md_aligned[modality_col].unique())}",
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
        f"--- 3. {preservation_col.replace('_', ' ').title()} Preservation (higher = better) ---",
        f"  Number of {preservation_col} groups: {group_results['n_groups']}",
        f"  Groups: {group_results['groups']}",
        f"  Mean within-group distance:  {group_results['mean_within_group_distance']:.4f}",
        f"  Mean between-group distance: {group_results['mean_between_group_distance']:.4f}",
        f"  Preservation score:          {group_results['preservation_score']:.4f}",
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
        "preservation_score": group_results["preservation_score"],
        "mean_within_group_distance": group_results["mean_within_group_distance"],
        "mean_between_group_distance": group_results["mean_between_group_distance"],
        # Metadata
        "n_modalities": mixing_results["n_modalities"],
        "modalities": mixing_results["modalities"],
        "n_groups": group_results["n_groups"],
        "groups": group_results["groups"],
        "preservation_col": preservation_col,
        # File paths
        "per_sample_path": str(per_sample_path),
        "paired_path": str(paired_path) if paired_path else None,
        "group_details_path": str(group_details_path),
        "summary_path": str(summary_path),
    }


# =============================================================================
# Summary CSV Aggregation (following BenchmarkWrapper pattern)
# =============================================================================

DEFAULT_SUMMARY_CSV = "/dcs07/hongkai/data/harry/result/multi_omics_heart/summary.csv"


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
        f"{results.get('preservation_col', 'preservation')}_score": results.get("preservation_score"),
        "mean_within_group_distance": results.get("mean_within_group_distance"),
        "mean_between_group_distance": results.get("mean_between_group_distance"),
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
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/embeddings/sample_expression_embedding.csv",
        outdir="/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/SD_expression",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="SD_expression")
    
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/embeddings/sample_proportion_embedding.csv",
        outdir="/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/SD_proportion",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="SD_proportion")

    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/pilot/wasserstein_distance_mds_10d.csv",
        outdir="/dcs07/hongkai/data/harry/result/multi_omics_heart/pilot",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="pilot")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/pseudobulk/pseudobulk/pca_embeddings.csv",
        outdir="/dcs07/hongkai/data/harry/result/multi_omics_heart/pseudobulk",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="pseudobulk")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/QOT/44_qot_distance_matrix_mds_10d.csv",
        outdir="/dcs07/hongkai/data/harry/result/multi_omics_heart/QOT",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="QOT")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/GEDI/gedi_sample_embedding.csv",
        outdir="/dcs07/hongkai/data/harry/result/multi_omics_heart/GEDI",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="GEDI")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/Gloscope/knn_divergence_mds_10d.csv",
        outdir="/dcs07/hongkai/data/harry/result/multi_omics_heart/Gloscope",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="Gloscope")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/MFA/sample_embeddings.csv",
        outdir="/dcs07/hongkai/data/harry/result/multi_omics_heart/MFA",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="MFA")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/mustard/sample_embedding.csv",
        outdir="/dcs07/hongkai/data/harry/result/multi_omics_heart/mustard",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="mustard")
    
    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_heart/scPoli/sample_embeddings_full.csv",
        outdir="/dcs07/hongkai/data/harry/result/multi_omics_heart/scPoli",
        k_neighbors=15,
    )
    save_to_summary_csv(results, method_name="scPoli")
    
    
    print("Multimodal Integration Benchmark module loaded.")
    print("Use evaluate_multimodal_integration() to run evaluation.")