#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multimodal Integration Benchmark - WITH TRAJECTORY ANALYSIS

Evaluates multimodal embeddings based on three criteria:
1. Paired sample matching: samples with same sample_id but different modality should be close
2. Modality mixing: modalities should be well-mixed (iLISI_norm, ASW_batch on modality)
3. Trajectory alignment: embedding should preserve age-based trajectory (CCA correlation)

Usage:
    results = evaluate_multimodal_integration(
        meta_csv="sample_metadata.csv",
        embedding_csv="embeddings.csv",
        summary_csv="results/summary.csv",  # Visualizations saved to same directory
        age_col="age",
    )
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# =============================================================================
# I/O and Alignment
# =============================================================================

def read_metadata(meta_csv: str, modalities: List[str] = ["RNA", "ATAC"]) -> pd.DataFrame:
    """Read metadata CSV and expand to include both modalities."""
    md = pd.read_csv(meta_csv, index_col=0)
    md.columns = [c.lower() for c in md.columns]
    md = md[md.index.notna()]
    md["sample_id"] = md.index
    
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
    """Align metadata and embedding by sample index (case-insensitive)."""
    emb_lower_to_original = {str(idx).lower(): idx for idx in emb.index}
    md_lower_to_original = {str(idx).lower(): idx for idx in md.index}
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
        raise ValueError("No overlapping sample IDs between metadata and embedding.")
    
    common_emb_original = [emb_lower_to_original[k] for k in sorted(common_lower)]
    common_md_original = [md_lower_to_original[k] for k in sorted(common_lower)]
    
    if len(common_lower) < len(md):
        print(f"Note: dropping {len(md) - len(common_lower)} metadata rows without embedding.", file=sys.stderr)
    if len(common_lower) < len(emb):
        print(f"Note: dropping {len(emb) - len(common_lower)} embedding rows without metadata.", file=sys.stderr)
    
    md_aligned = md.loc[common_md_original].copy()
    emb_aligned = emb.loc[common_emb_original].copy()
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
    sample_id_to_idx: Dict[str, Dict[str, int]] = {}
    
    for i, (idx, row) in enumerate(md.iterrows()):
        sid = str(row[sample_id_col])
        mod = str(row[modality_col])
        if sid not in sample_id_to_idx:
            sample_id_to_idx[sid] = {}
        sample_id_to_idx[sid][mod] = i
    
    paired_distances = []
    paired_info = []
    
    for sid, mod_dict in sample_id_to_idx.items():
        modalities = list(mod_dict.keys())
        if len(modalities) == 2:
            idx1 = mod_dict[modalities[0]]
            idx2 = mod_dict[modalities[1]]
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
    modalities_str = md[modality_col].astype(str).values
    unique_modalities, labels_int = np.unique(modalities_str, return_inverse=True)
    n_modalities = len(unique_modalities)
    n_samples = emb.shape[0]
    
    k_eff = min(max(int(k), 1), n_samples)
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=-1)
    nn.fit(emb)
    _, knn_idx = nn.kneighbors(emb)
    
    ilisi_per = compute_ilisi(labels_int, knn_idx, include_self=include_self)
    ilisi_mean = float(np.mean(ilisi_per))
    ilisi_std = float(np.std(ilisi_per, ddof=1)) if n_samples > 1 else 0.0
    ilisi_norm_mean = float(ilisi_mean / max(1, n_modalities))
    
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
# Metric 3: Trajectory Analysis (CCA with Age) - Using First 2 PCs Only
# =============================================================================

def _assign_pseudotime_from_cca(coords_2d: np.ndarray, cca: CCA, scale_to_unit: bool = True) -> np.ndarray:
    """Project onto CCA x-weights to get pseudotime."""
    direction = cca.x_weights_[:, 0]
    proj = coords_2d @ direction
    
    if not scale_to_unit:
        return proj
    
    lo, hi = float(np.min(proj)), float(np.max(proj))
    denom = max(hi - lo, 1e-16)
    return (proj - lo) / denom


def compute_trajectory_analysis(
    md: pd.DataFrame,
    emb: np.ndarray,
    age_col: str = "age",
    sample_id_col: str = "sample_id",
    save_plot: bool = True,
    plot_dir: Optional[str] = None,
    method_name: str = "embedding",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute trajectory analysis using CCA alignment with age.
    
    Uses the first 2 dimensions (PC1 and PC2) for CCA and visualization.
    Higher CCA score = better trajectory/age alignment preserved in embedding.
    
    Parameters
    ----------
    md : pd.DataFrame
        Metadata with age column
    emb : np.ndarray
        Embedding matrix (n_samples x n_dims)
    age_col : str
        Column name for age (continuous variable)
    sample_id_col : str
        Column name for sample ID
    save_plot : bool
        Whether to save trajectory visualization
    plot_dir : str, optional
        Directory for saving plots
    method_name : str
        Name of method for plot filename
    verbose : bool
        Print progress
        
    Returns
    -------
    Dict with trajectory metrics including CCA score and pseudotime
    """
    # Check embedding dimensions
    if emb.shape[1] < 2:
        raise ValueError("Need at least 2 dimensions for trajectory CCA.")
    
    # Get age values (use unique sample_id to avoid duplicate modalities)
    # Average age per sample_id to get one value per biological sample
    sample_ages = md.groupby(sample_id_col)[age_col].first()
    
    # Map age back to all rows
    age_values = md[sample_id_col].map(sample_ages).values
    age_values = pd.to_numeric(age_values, errors='coerce').astype(float)
    
    # Handle missing values
    if np.isnan(age_values).any():
        n_missing = int(np.isnan(age_values).sum())
        if verbose:
            print(f"  Warning: {n_missing} samples have missing {age_col} values; imputing with mean.")
        mean_age = np.nanmean(age_values)
        age_values = np.where(np.isnan(age_values), mean_age, age_values)
    
    # Use only the first 2 dimensions (PC1 and PC2)
    dim1, dim2 = 0, 1
    coords_2d = emb[:, [dim1, dim2]]
    
    if verbose:
        print(f"  Using first 2 dimensions (PC1 and PC2) for CCA...")
    
    # Fit CCA
    age_2d = age_values.reshape(-1, 1)
    cca = CCA(n_components=1)
    cca.fit(coords_2d, age_2d)
    U, V = cca.transform(coords_2d, age_2d)
    cca_score = float(abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1]))
    
    if verbose:
        print(f"  CCA score (PC1 & PC2): {cca_score:.4f}")
    
    # Compute pseudotime
    pseudotime = _assign_pseudotime_from_cca(coords_2d, cca, scale_to_unit=True)
    
    # Compute correlation between pseudotime and age
    pseudotime_age_corr = float(np.corrcoef(pseudotime, age_values)[0, 1])
    
    # Save trajectory plot
    if save_plot and plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        fig_path = os.path.join(plot_dir, f"trajectory_{method_name}.png")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel 1: Embedding colored by age
        age_norm = (age_values - age_values.min()) / (age_values.max() - age_values.min() + 1e-16)
        sc1 = axes[0].scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=age_norm, cmap='viridis',
            edgecolors='black', alpha=0.8, s=80, linewidths=0.3
        )
        cbar1 = plt.colorbar(sc1, ax=axes[0], label=f'Normalized {age_col}')
        
        # Add CCA direction arrow
        dx, dy = cca.x_weights_[:, 0]
        scale = 0.35 * max(np.ptp(coords_2d[:, 0]), np.ptp(coords_2d[:, 1]))
        center_x, center_y = np.mean(coords_2d[:, 0]), np.mean(coords_2d[:, 1])
        axes[0].arrow(
            center_x, center_y, scale*dx, scale*dy,
            head_width=0.04*scale, head_length=0.06*scale,
            fc='red', ec='red', linewidth=2.5, alpha=0.8
        )
        
        axes[0].set_xlabel(f"PC1 (Dimension {dim1+1})", fontsize=11)
        axes[0].set_ylabel(f"PC2 (Dimension {dim2+1})", fontsize=11)
        axes[0].set_title(f"Embedding Space (CCA score: {cca_score:.3f})", fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Panel 2: Pseudotime vs Age scatter
        axes[1].scatter(age_values, pseudotime, c='steelblue', alpha=0.7, s=60, edgecolors='black', linewidths=0.3)
        
        # Add regression line
        z = np.polyfit(age_values, pseudotime, 1)
        p = np.poly1d(z)
        age_sorted = np.sort(age_values)
        axes[1].plot(age_sorted, p(age_sorted), 'r--', linewidth=2, label=f'r = {pseudotime_age_corr:.3f}')
        
        axes[1].set_xlabel(f"{age_col}", fontsize=11)
        axes[1].set_ylabel("Pseudotime", fontsize=11)
        axes[1].set_title(f"Pseudotime vs {age_col}", fontsize=12, fontweight='bold')
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle(f"Trajectory Analysis: {method_name}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"  Saved trajectory plot: {fig_path}")
    
    return {
        "cca_score": cca_score,
        "pseudotime_age_correlation": pseudotime_age_corr,
        "best_dim_pair": (dim1 + 1, dim2 + 1),  # Always (1, 2) now
        "pseudotime": pseudotime,
        "age_values": age_values,
        "age_col": age_col,
        "n_samples": len(age_values),
        "age_range": (float(np.min(age_values)), float(np.max(age_values))),
    }

def evaluate_multimodal_integration(
    meta_csv: str,
    embedding_csv: str,
    summary_csv: str,
    sample_id_col: str = "sample_id",
    modality_col: str = "modality",
    age_col: str = "age",
    modalities: List[str] = ["RNA", "ATAC"],
    k_neighbors: int = 15,
    distance_metric: str = "euclidean",
    include_self: bool = False,
    method_name: str = "method",
    save_plots: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate multimodal integration quality.
    
    Parameters
    ----------
    meta_csv : str
        Path to metadata CSV (index: sample names)
    embedding_csv : str
        Path to embedding CSV (samples × dimensions)
        Sample names format: <sample_id>_<modality>
    summary_csv : str
        Path to summary CSV for aggregating results.
        Trajectory visualizations will be saved to 'trajectory_plots' folder
        at the same level as the summary CSV.
    sample_id_col : str
        Column name for sample ID (pairs ATAC/RNA)
    modality_col : str
        Column name for modality
    age_col : str
        Column name for age (continuous variable for trajectory)
    modalities : List[str]
        List of modalities (default: ["RNA", "ATAC"])
    k_neighbors : int
        Number of neighbors for iLISI computation
    distance_metric : str
        Distance metric for pairwise distances
    include_self : bool
        Include self in KNN neighborhood
    method_name : str
        Name of the method being evaluated
    save_plots : bool
        Whether to save visualization plots
        
    Returns
    -------
    Dict with all metrics
    """
    # Setup output directory (same as summary CSV directory)
    summary_path = Path(summary_csv)
    output_dir = summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for detailed results
    details_dir = output_dir / "evaluation_details" / method_name
    details_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dedicated directory for trajectory/CCA visualizations
    trajectory_plots_dir = output_dir / "trajectory_plots"
    trajectory_plots_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Check age column
    if age_col not in md_aligned.columns:
        raise KeyError(f"'{age_col}' not found in metadata. Available columns: {list(md_aligned.columns)}")
    
    age_range = md_aligned[age_col].dropna()
    print(f"Age range: {age_range.min():.1f} - {age_range.max():.1f}")
    
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
    
    print(f"\n3. Computing trajectory analysis ({age_col}) using PC1 & PC2...")
    trajectory_results = compute_trajectory_analysis(
        md_aligned, emb_array,
        age_col=age_col,
        sample_id_col=sample_id_col,
        save_plot=save_plots,
        plot_dir=str(trajectory_plots_dir),  # Use dedicated trajectory plots directory
        method_name=method_name,
        verbose=True,
    )
    
    # Save per-sample metrics
    per_sample_df = pd.DataFrame({
        "sample": md_aligned.index,
        "sample_id": md_aligned[sample_id_col].values,
        "modality": md_aligned[modality_col].values,
        age_col: md_aligned[age_col].values,
        "pseudotime": trajectory_results["pseudotime"],
        "iLISI": mixing_results["iLISI_per_sample"],
        "ASW_modality": mixing_results["ASW_per_sample"],
    }).set_index("sample")
    
    per_sample_path = details_dir / "per_sample_metrics.csv"
    per_sample_df.to_csv(per_sample_path)
    
    # Save paired sample details
    if paired_results["paired_details"]:
        paired_df = pd.DataFrame(paired_results["paired_details"])
        paired_path = details_dir / "paired_sample_distances.csv"
        paired_df.to_csv(paired_path, index=False)
    
    # Generate summary
    summary_lines = [
        "=" * 60,
        f"Multimodal Integration Evaluation: {method_name}",
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
        f"--- 3. Trajectory Analysis ({age_col}) using PC1 & PC2 (higher = better) ---",
        f"  Dimension pair: {trajectory_results['best_dim_pair']} (PC1 & PC2)",
        f"  CCA score:           {trajectory_results['cca_score']:.4f}",
        f"  Pseudotime-age corr: {trajectory_results['pseudotime_age_correlation']:.4f}",
        f"  Age range:           {trajectory_results['age_range'][0]:.1f} - {trajectory_results['age_range'][1]:.1f}",
        "",
        "=" * 60,
        f"Results saved to: {details_dir}",
        f"Trajectory plots saved to: {trajectory_plots_dir}",
        "=" * 60,
    ]
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    summary_txt_path = details_dir / "integration_summary.txt"
    summary_txt_path.write_text(summary_text, encoding="utf-8")
    
    # Aggregate results
    results = {
        # Core metrics for summary CSV
        "n_samples": len(md_aligned),
        "n_pairs": paired_results["n_pairs"],
        "mean_paired_distance": paired_results["mean_paired_distance"],
        "std_paired_distance": paired_results["std_paired_distance"],
        "median_paired_distance": paired_results["median_paired_distance"],
        "iLISI_mean": mixing_results["iLISI_mean"],
        "iLISI_norm_mean": mixing_results["iLISI_norm_mean"],
        "ASW_modality_overall": mixing_results["ASW_modality_overall"],
        "cca_score": trajectory_results["cca_score"],
        "pseudotime_age_correlation": trajectory_results["pseudotime_age_correlation"],
        # Metadata
        "n_modalities": mixing_results["n_modalities"],
        "modalities": mixing_results["modalities"],
        "age_col": age_col,
        "best_dim_pair": trajectory_results["best_dim_pair"],
        # File paths
        "per_sample_path": str(per_sample_path),
        "summary_txt_path": str(summary_txt_path),
        "trajectory_plots_dir": str(trajectory_plots_dir),
    }
    
    # Auto-save to summary CSV
    save_to_summary_csv(results, method_name, str(summary_csv))
    
    return results


# =============================================================================
# Summary CSV Aggregation
# =============================================================================

def save_to_summary_csv(
    results: Dict[str, Any],
    method_name: str,
    summary_csv_path: str,
) -> None:
    """
    Save results to a summary CSV file, appending as a new column.
    
    Structure:
    - Rows: metric names
    - Columns: method_name
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
        "cca_score": results.get("cca_score"),
        "pseudotime_age_correlation": results.get("pseudotime_age_correlation"),
    }
    
    col_name = method_name
    
    if summary_path.exists() and summary_path.stat().st_size > 0:
        try:
            summary_df = pd.read_csv(summary_path, index_col=0)
        except pd.errors.EmptyDataError:
            summary_df = pd.DataFrame()
    else:
        summary_df = pd.DataFrame()
    
    for metric, value in metrics_to_save.items():
        summary_df.loc[metric, col_name] = value
    
    summary_df.to_csv(summary_path, index_label="Metric")
    print(f"\nUpdated summary CSV: {summary_path} with column '{col_name}'")

if __name__ == '__main__':
    summary_csv = "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/summary.csv"

    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/embeddings/sample_expression_embedding.csv",
        summary_csv=summary_csv,
        k_neighbors=3,
        method_name="SD_expression",
    )

    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/embeddings/sample_proportion_embedding.csv",
        summary_csv=summary_csv,
        k_neighbors=3,
        method_name="SD_proportion",
    )

    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/pilot/wasserstein_distance_mds_10d.csv",
        summary_csv=summary_csv,
        k_neighbors=3,
        method_name="pilot",
    )

    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/pseudobulk/pseudobulk/pca_embeddings.csv",
        summary_csv=summary_csv,
        k_neighbors=3,
        method_name="pseudobulk",
    )

    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/QOT/24_qot_distance_matrix_mds_10d.csv",
        summary_csv=summary_csv,
        k_neighbors=3,
        method_name="QOT",
    )

    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/GEDI/gedi_sample_embedding.csv",
        summary_csv=summary_csv,
        k_neighbors=3,
        method_name="GEDI",
    )

    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/Gloscope/knn_divergence_mds_10d.csv",
        summary_csv=summary_csv,
        k_neighbors=3,
        method_name="Gloscope",
    )

    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/MFA/sample_embeddings.csv",
        summary_csv=summary_csv,
        k_neighbors=3,
        method_name="MFA",
    )

    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/mustard/sample_embedding.csv",
        summary_csv=summary_csv,
        k_neighbors=3,
        method_name="mustard",
    )

    results = evaluate_multimodal_integration(
        meta_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/scPoli/sample_embeddings_full.csv",
        summary_csv=summary_csv,
        k_neighbors=3,
        method_name="scPoli",
    )

    print("Multimodal Integration Benchmark module loaded.")
    print("Use evaluate_multimodal_integration() to run evaluation.")
