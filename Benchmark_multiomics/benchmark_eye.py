#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved Multimodal Integration Benchmark with Trajectory Analysis

Evaluates multimodal embeddings based on three criteria:
1. Paired sample matching: samples with same sample_id but different modality should be close
2. Modality mixing: modalities should be well-mixed (iLISI_norm, ASW_batch on modality)
3. Trajectory alignment: embedding should preserve age-based trajectory (CCA correlation)

Improvements over original:
- Professional visualizations with enhanced styling
- Permutation testing for statistical significance
- Organized output directory structure with method subfolders
- Better error handling and debugging
- Comprehensive result aggregation

Usage:
    results = evaluate_multimodal_integration(
        meta_csv="sample_metadata.csv",
        embedding_csv="embeddings.csv",
        method_name="method_name",
        general_outdir="results/",
        age_col="age",
    )
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set high-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for multimodal integration benchmark."""
    # KNN settings
    k_neighbors: int = 15
    distance_metric: str = "euclidean"
    include_self: bool = False
    
    # Permutation testing
    n_permutations: int = 1000
    random_seed: int = 42
    
    # Visualization
    figsize: Tuple[int, int] = (12, 10)
    point_size: int = 120
    point_alpha: float = 0.75
    line_alpha: float = 0.25
    line_width: float = 0.8
    dpi: int = 300
    
    # Colors - using a professional palette
    modality_colors: Dict[str, str] = field(default_factory=lambda: {
        'RNA': '#3498db',    # Bright blue
        'ATAC': '#e74c3c',   # Coral red
    })
    connection_color: str = '#7f8c8d'
    
    # Trajectory colors
    age_cmap: str = 'viridis'
    pseudotime_cmap: str = 'plasma'


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
    print("\n" + "="*60)
    print("DEBUG: align_data() function")
    print("="*60)
    
    emb_lower_to_original = {str(idx).lower(): idx for idx in emb.index}
    md_lower_to_original = {str(idx).lower(): idx for idx in md.index}
    common_lower = set(emb_lower_to_original.keys()).intersection(set(md_lower_to_original.keys()))
    
    print(f"\n1. Input sizes:")
    print(f"   - Metadata: {len(md)} samples")
    print(f"   - Embedding: {len(emb)} samples")
    
    print(f"\n2. Sample name examples (first 5):")
    print(f"   - Metadata index: {list(md.index[:5])}")
    print(f"   - Embedding index: {list(emb.index[:5])}")
    
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
    
    print(f"\n3. Alignment results:")
    print(f"   - Common samples: {len(common_lower)}")
    if len(common_lower) < len(md):
        print(f"   - Dropping {len(md) - len(common_lower)} metadata rows without embedding")
    if len(common_lower) < len(emb):
        print(f"   - Dropping {len(emb) - len(common_lower)} embedding rows without metadata")
    
    md_aligned = md.loc[common_md_original].copy()
    emb_aligned = emb.loc[common_emb_original].copy()
    md_aligned.index = emb_aligned.index
    
    print(f"\n4. Aligned datasets:")
    print(f"   - Total aligned samples: {len(md_aligned)}")
    print(f"   - First 5 aligned sample names: {list(md_aligned.index[:5])}")
    print("="*60 + "\n")
    
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
    paired_indices = []
    
    for sid, mod_dict in sample_id_to_idx.items():
        modalities = list(mod_dict.keys())
        if len(modalities) == 2:
            idx1 = mod_dict[modalities[0]]
            idx2 = mod_dict[modalities[1]]
            vec1 = emb[idx1].reshape(1, -1)
            vec2 = emb[idx2].reshape(1, -1)
            dist = cdist(vec1, vec2, metric=metric)[0, 0]
            paired_distances.append(dist)
            paired_indices.append((idx1, idx2))
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
            "paired_indices": [],
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
        "paired_indices": paired_indices,
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
        "modality_labels": labels_int,
        "knn_idx": knn_idx,
    }


# =============================================================================
# Metric 3: Trajectory Analysis (CCA with Age)
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
) -> Dict[str, Any]:
    """
    Compute trajectory analysis using CCA alignment with age.
    
    Uses the first 2 dimensions (PC1 and PC2) for CCA.
    Higher CCA score = better trajectory/age alignment preserved in embedding.
    """
    # Check embedding dimensions
    if emb.shape[1] < 2:
        raise ValueError("Need at least 2 dimensions for trajectory CCA.")
    
    # Get age values (use unique sample_id to avoid duplicate modalities)
    sample_ages = md.groupby(sample_id_col)[age_col].first()
    age_values = md[sample_id_col].map(sample_ages).values
    age_values = pd.to_numeric(age_values, errors='coerce').astype(float)
    
    # Handle missing values
    if np.isnan(age_values).any():
        n_missing = int(np.isnan(age_values).sum())
        print(f"  Warning: {n_missing} samples have missing {age_col} values; imputing with mean.")
        mean_age = np.nanmean(age_values)
        age_values = np.where(np.isnan(age_values), mean_age, age_values)
    
    # Use only the first 2 dimensions (PC1 and PC2)
    coords_2d = emb[:, [0, 1]]
    
    # Fit CCA
    age_2d = age_values.reshape(-1, 1)
    cca = CCA(n_components=1)
    cca.fit(coords_2d, age_2d)
    U, V = cca.transform(coords_2d, age_2d)
    cca_score = float(abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1]))
    
    # Compute pseudotime
    pseudotime = _assign_pseudotime_from_cca(coords_2d, cca, scale_to_unit=True)
    
    # Compute correlation between pseudotime and age
    pseudotime_age_corr = float(np.corrcoef(pseudotime, age_values)[0, 1])
    
    return {
        "cca_score": cca_score,
        "pseudotime_age_correlation": pseudotime_age_corr,
        "pseudotime": pseudotime,
        "age_values": age_values,
        "age_col": age_col,
        "n_samples": len(age_values),
        "age_range": (float(np.min(age_values)), float(np.max(age_values))),
        "cca_x_weights": cca.x_weights_[:, 0],
    }


# =============================================================================
# Permutation Testing
# =============================================================================

def permutation_test_paired_distance(
    md: pd.DataFrame,
    emb: np.ndarray,
    observed_mean: float,
    paired_indices: List[Tuple[int, int]],
    n_permutations: int = 1000,
    metric: str = "euclidean",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Permutation test for paired sample distance."""
    if len(paired_indices) == 0 or np.isnan(observed_mean):
        return {"p_value": np.nan, "null_distribution": []}
    
    rng = np.random.default_rng(random_seed)
    null_means = []
    n_samples = emb.shape[0]
    n_pairs = len(paired_indices)
    
    for _ in range(n_permutations):
        shuffled_idx = rng.permutation(n_samples)
        perm_distances = []
        for i in range(0, min(2 * n_pairs, n_samples - 1), 2):
            if i + 1 < n_samples:
                vec1 = emb[shuffled_idx[i]].reshape(1, -1)
                vec2 = emb[shuffled_idx[i + 1]].reshape(1, -1)
                dist = cdist(vec1, vec2, metric=metric)[0, 0]
                perm_distances.append(dist)
        
        if perm_distances:
            null_means.append(np.mean(perm_distances))
    
    null_means = np.array(null_means)
    p_value = float(np.mean(null_means <= observed_mean))
    
    return {
        "p_value": p_value,
        "null_mean": float(np.mean(null_means)),
        "null_std": float(np.std(null_means)),
        "null_distribution": null_means.tolist(),
    }


def permutation_test_modality_mixing(
    md: pd.DataFrame,
    emb: np.ndarray,
    observed_ilisi: float,
    observed_asw: float,
    modality_col: str = "modality",
    k: int = 15,
    n_permutations: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Permutation test for modality mixing metrics."""
    rng = np.random.default_rng(random_seed)
    modalities = md[modality_col].values.copy()
    unique_mods, labels_int = np.unique(modalities, return_inverse=True)
    n_modalities = len(unique_mods)
    n_samples = emb.shape[0]
    
    if n_modalities < 2:
        return {
            "iLISI_p_value": np.nan,
            "ASW_p_value": np.nan,
            "iLISI_null_distribution": [],
            "ASW_null_distribution": [],
        }
    
    # Precompute KNN
    k_eff = min(max(int(k), 1), n_samples)
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=-1)
    nn.fit(emb)
    _, knn_idx = nn.kneighbors(emb)
    
    null_ilisi = []
    null_asw = []
    
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels_int)
        
        ilisi_per = compute_ilisi(perm_labels, knn_idx, include_self=False)
        ilisi_norm = float(np.mean(ilisi_per) / max(1, n_modalities))
        null_ilisi.append(ilisi_norm)
        
        if n_samples > n_modalities:
            s_overall = silhouette_score(emb, perm_labels, metric="euclidean")
            asw = float(np.clip((1.0 - s_overall) / 2.0, 0.0, 1.0))
            null_asw.append(asw)
    
    null_ilisi = np.array(null_ilisi)
    null_asw = np.array(null_asw) if null_asw else np.array([np.nan])
    
    ilisi_p = float(np.mean(null_ilisi >= observed_ilisi)) if not np.isnan(observed_ilisi) else np.nan
    asw_p = float(np.mean(null_asw >= observed_asw)) if not np.isnan(observed_asw) else np.nan
    
    return {
        "iLISI_p_value": ilisi_p,
        "ASW_p_value": asw_p,
        "iLISI_null_mean": float(np.nanmean(null_ilisi)),
        "ASW_null_mean": float(np.nanmean(null_asw)),
        "iLISI_null_distribution": null_ilisi.tolist(),
        "ASW_null_distribution": null_asw.tolist(),
    }


def permutation_test_trajectory(
    md: pd.DataFrame,
    emb: np.ndarray,
    observed_cca: float,
    age_col: str = "age",
    sample_id_col: str = "sample_id",
    n_permutations: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Permutation test for trajectory CCA score."""
    if np.isnan(observed_cca):
        return {"p_value": np.nan, "null_distribution": []}
    
    rng = np.random.default_rng(random_seed)
    
    # Get age values
    sample_ages = md.groupby(sample_id_col)[age_col].first()
    age_values = md[sample_id_col].map(sample_ages).values
    age_values = pd.to_numeric(age_values, errors='coerce').astype(float)
    
    if np.isnan(age_values).any():
        mean_age = np.nanmean(age_values)
        age_values = np.where(np.isnan(age_values), mean_age, age_values)
    
    coords_2d = emb[:, [0, 1]]
    null_ccas = []
    
    for _ in range(n_permutations):
        # Shuffle age labels
        perm_age = rng.permutation(age_values)
        
        age_2d = perm_age.reshape(-1, 1)
        cca = CCA(n_components=1)
        cca.fit(coords_2d, age_2d)
        U, V = cca.transform(coords_2d, age_2d)
        cca_score = float(abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1]))
        null_ccas.append(cca_score)
    
    null_ccas = np.array(null_ccas)
    p_value = float(np.mean(null_ccas >= observed_cca))
    
    return {
        "p_value": p_value,
        "null_mean": float(np.mean(null_ccas)),
        "null_std": float(np.std(null_ccas)),
        "null_distribution": null_ccas.tolist(),
    }


# =============================================================================
# Enhanced Visualization Functions
# =============================================================================

def reduce_to_2d(emb: np.ndarray) -> np.ndarray:
    """Reduce embedding to 2D using PCA if necessary."""
    if emb.shape[1] <= 2:
        if emb.shape[1] == 1:
            return np.column_stack([emb, np.zeros(emb.shape[0])])
        return emb[:, :2]
    
    pca = PCA(n_components=2)
    return pca.fit_transform(emb)


def plot_embedding_by_modality(
    emb_2d: np.ndarray,
    md: pd.DataFrame,
    output_path: str,
    config: BenchmarkConfig,
    modality_col: str = "modality",
    method_name: str = "",
) -> None:
    """Plot 2D embedding colored by modality with enhanced styling."""
    fig, ax = plt.subplots(figsize=config.figsize)
    
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    modalities = md[modality_col].values
    unique_mods = sorted(np.unique(modalities))
    
    for mod in unique_mods:
        mask = modalities == mod
        color = config.modality_colors.get(mod, '#333333')
        
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            s=config.point_size,
            c=color,
            alpha=config.point_alpha,
            label=mod,
            edgecolors='white',
            linewidths=0.8,
            zorder=3,
        )
    
    ax.set_xlabel('PC1', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Embedding by Modality', fontsize=16, fontweight='bold', pad=20)
    
    legend = ax.legend(
        title='Modality', 
        loc='upper right',
        frameon=True,
        fancybox=True,
        fontsize=13,
        title_fontsize=14,
        edgecolor='#cccccc',
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=11, length=5, width=1.2)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)


def plot_paired_connections(
    emb_2d: np.ndarray,
    md: pd.DataFrame,
    output_path: str,
    config: BenchmarkConfig,
    sample_id_col: str = "sample_id",
    modality_col: str = "modality",
    method_name: str = "",
) -> None:
    """Plot 2D embedding with connections between paired samples."""
    fig, ax = plt.subplots(figsize=config.figsize)
    
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    # Build sample_id -> {modality: row_idx} mapping
    sample_id_to_idx: Dict[str, Dict[str, int]] = {}
    
    for i, (idx, row) in enumerate(md.iterrows()):
        sid = str(row[sample_id_col])
        mod = str(row[modality_col])
        if sid not in sample_id_to_idx:
            sample_id_to_idx[sid] = {}
        sample_id_to_idx[sid][mod] = i
    
    # Draw connections
    for sid, mod_dict in sample_id_to_idx.items():
        modalities = list(mod_dict.keys())
        if len(modalities) == 2:
            idx1 = mod_dict[modalities[0]]
            idx2 = mod_dict[modalities[1]]
            
            ax.plot(
                [emb_2d[idx1, 0], emb_2d[idx2, 0]],
                [emb_2d[idx1, 1], emb_2d[idx2, 1]],
                color=config.connection_color,
                alpha=config.line_alpha + 0.2,
                linewidth=config.line_width,
                zorder=1,
            )
    
    # Draw points
    modalities = md[modality_col].values
    unique_mods = sorted(np.unique(modalities))
    markers = {'RNA': 'o', 'ATAC': 's'}
    
    for mod in unique_mods:
        mask = modalities == mod
        color = config.modality_colors.get(mod, '#333333')
        marker = markers.get(mod, 'o')
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            s=config.point_size,
            c=color,
            alpha=config.point_alpha,
            label=mod,
            marker=marker,
            edgecolors='white',
            linewidths=1.0,
            zorder=3,
        )
    
    ax.set_xlabel('PC1', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Paired Sample Connections', fontsize=16, fontweight='bold', pad=20)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=config.modality_colors.get('RNA', '#3498db'),
               markersize=12, label='RNA', markeredgecolor='white', markeredgewidth=1.5),
        Line2D([0], [0], marker='s', color='w', 
               markerfacecolor=config.modality_colors.get('ATAC', '#e74c3c'),
               markersize=12, label='ATAC', markeredgecolor='white', markeredgewidth=1.5),
        Line2D([0], [0], color=config.connection_color, linewidth=3, alpha=0.7, label='Paired connection'),
    ]
    
    legend = ax.legend(
        handles=legend_elements, 
        loc='upper right',
        frameon=True,
        fancybox=True,
        fontsize=13,
        title_fontsize=14,
        edgecolor='#cccccc',
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=11, length=5, width=1.2)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)


def plot_trajectory_analysis(
    emb_2d: np.ndarray,
    trajectory_results: Dict[str, Any],
    output_path: str,
    config: BenchmarkConfig,
    method_name: str = "",
) -> None:
    """
    Plot trajectory analysis with age and pseudotime visualizations.
    Creates two separate figure files.
    """
    age_values = trajectory_results['age_values']
    pseudotime = trajectory_results['pseudotime']
    cca_score = trajectory_results['cca_score']
    pseudotime_age_corr = trajectory_results['pseudotime_age_correlation']
    age_col = trajectory_results['age_col']
    
    # Determine output paths for two separate figures
    output_path_obj = Path(output_path)
    embedding_path = output_path_obj.parent / f"{output_path_obj.stem}_embedding.png"
    scatter_path = output_path_obj.parent / f"{output_path_obj.stem}_scatter.png"
    
    # =========================================================================
    # Figure 1: Embedding colored by age with proper colorbar height
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    ax1.set_facecolor('#fafafa')
    fig1.patch.set_facecolor('white')
    
    # Normalize age for colormap
    age_norm = (age_values - age_values.min()) / (age_values.max() - age_values.min() + 1e-16)
    
    sc1 = ax1.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=age_norm,
        cmap=config.age_cmap,
        s=config.point_size,
        alpha=config.point_alpha,
        edgecolors='white',
        linewidths=0.8,
        zorder=3,
    )
    
    # Create colorbar with same height as plot
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(sc1, cax=cax)
    cbar1.set_label(f'{age_col.capitalize()} (normalized)', fontsize=12, fontweight='bold')
    cbar1.ax.tick_params(labelsize=10)
    
    ax1.set_xlabel('PC1', fontsize=14, fontweight='bold', labelpad=10)
    ax1.set_ylabel('PC2', fontsize=14, fontweight='bold', labelpad=10)
    ax1.set_title(f'Embedding (CCA score: {cca_score:.3f})', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#cccccc')
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='both', which='major', labelsize=11, length=5, width=1.2)
    
    plt.tight_layout()
    fig1.savefig(embedding_path, dpi=config.dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig1)
    
    # =========================================================================
    # Figure 2: Pseudotime vs Age scatter
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.set_facecolor('#fafafa')
    fig2.patch.set_facecolor('white')
    
    ax2.scatter(
        age_values, pseudotime,
        c='steelblue',
        s=80,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5,
        zorder=3,
    )
    
    # Add regression line
    z = np.polyfit(age_values, pseudotime, 1)
    p = np.poly1d(z)
    age_sorted = np.sort(age_values)
    ax2.plot(age_sorted, p(age_sorted), 'r--', linewidth=2.5, 
            label=f'r = {pseudotime_age_corr:.3f}', zorder=2)
    
    ax2.set_xlabel(f'{age_col.capitalize()}', fontsize=14, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Pseudotime', fontsize=14, fontweight='bold', labelpad=10)
    ax2.set_title(f'Pseudotime vs {age_col.capitalize()}', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=12, frameon=True, fancybox=True, 
              edgecolor='#cccccc', facecolor='white', framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#cccccc')
    ax2.set_axisbelow(True)
    ax2.tick_params(axis='both', which='major', labelsize=11, length=5, width=1.2)
    
    plt.tight_layout()
    fig2.savefig(scatter_path, dpi=config.dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig2)



def create_all_visualizations(
    emb: np.ndarray,
    md: pd.DataFrame,
    trajectory_results: Dict[str, Any],
    method_outdir: Path,
    method_name: str,
    config: BenchmarkConfig,
    sample_id_col: str = "sample_id",
    modality_col: str = "modality",
) -> Dict[str, str]:
    """Create all visualization plots for a method."""
    emb_2d = reduce_to_2d(emb)
    
    viz_dir = method_outdir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: By modality
    modality_path = viz_dir / f"{method_name}_by_modality.png"
    plot_embedding_by_modality(
        emb_2d, md, str(modality_path), config,
        modality_col=modality_col,
        method_name=method_name,
    )
    
    # Plot 2: Paired connections
    paired_path = viz_dir / f"{method_name}_paired_connections.png"
    plot_paired_connections(
        emb_2d, md, str(paired_path), config,
        sample_id_col=sample_id_col,
        modality_col=modality_col,
        method_name=method_name,
    )
    
    # Plot 3: Trajectory analysis (creates 2 files)
    trajectory_path = viz_dir / f"{method_name}_trajectory.png"
    plot_trajectory_analysis(
        emb_2d, trajectory_results, str(trajectory_path), config,
        method_name=method_name,
    )
    
    # The trajectory function creates two files
    trajectory_embedding_path = viz_dir / f"{method_name}_trajectory_embedding.png"
    trajectory_scatter_path = viz_dir / f"{method_name}_trajectory_scatter.png"
    
    return {
        "modality_plot": str(modality_path),
        "paired_plot": str(paired_path),
        "trajectory_embedding_plot": str(trajectory_embedding_path),
        "trajectory_scatter_plot": str(trajectory_scatter_path),
    }


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_multimodal_integration(
    meta_csv: str,
    embedding_csv: str,
    method_name: str,
    general_outdir: str,
    sample_id_col: str = "sample_id",
    modality_col: str = "modality",
    age_col: str = "age",
    modalities: List[str] = ["RNA", "ATAC"],
    k_neighbors: int = 15,
    distance_metric: str = "euclidean",
    include_self: bool = False,
    n_permutations: int = 1000,
    random_seed: int = 42,
    create_visualizations: bool = True,
    config: Optional[BenchmarkConfig] = None,
) -> Dict[str, Any]:
    """
    Evaluate multimodal integration quality with trajectory analysis.
    
    Parameters
    ----------
    meta_csv : str
        Path to metadata CSV (index: sample names without modality suffix)
    embedding_csv : str
        Path to embedding CSV (samples × dimensions)
        Sample names format: <sample_id>_<modality>
    method_name : str
        Name of the integration method being evaluated
    general_outdir : str
        General output directory (method will have its own subfolder)
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
    n_permutations : int
        Number of permutations for p-value computation
    random_seed : int
        Random seed for reproducibility
    create_visualizations : bool
        Whether to create visualization plots
    config : BenchmarkConfig, optional
        Configuration object (created with defaults if not provided)
        
    Returns
    -------
    Dict with all metrics and paths to saved files
    """
    if config is None:
        config = BenchmarkConfig(
            k_neighbors=k_neighbors,
            distance_metric=distance_metric,
            include_self=include_self,
            n_permutations=n_permutations,
            random_seed=random_seed,
        )
    
    # Setup output directories
    general_path = Path(general_outdir)
    benchmark_results_path = general_path / "Benchmark_result"
    benchmark_results_path.mkdir(parents=True, exist_ok=True)
    
    method_outdir = benchmark_results_path / method_name
    method_outdir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Evaluating method: {method_name}")
    print(f"{'='*60}")
    
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
    )
    print(f"  CCA score (PC1 & PC2): {trajectory_results['cca_score']:.4f}")
    print(f"  Pseudotime-age correlation: {trajectory_results['pseudotime_age_correlation']:.4f}")
    
    # Permutation testing
    print("\n4. Running permutation tests...")
    
    print("   - Paired distance permutation test...")
    paired_perm = permutation_test_paired_distance(
        md_aligned, emb_array,
        paired_results['mean_paired_distance'],
        paired_results['paired_indices'],
        n_permutations=n_permutations,
        metric=distance_metric,
        random_seed=random_seed,
    )
    
    print("   - Modality mixing permutation test...")
    mixing_perm = permutation_test_modality_mixing(
        md_aligned, emb_array,
        mixing_results['iLISI_norm_mean'],
        mixing_results['ASW_modality_overall'],
        modality_col=modality_col,
        k=k_neighbors,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    
    print("   - Trajectory CCA permutation test...")
    trajectory_perm = permutation_test_trajectory(
        md_aligned, emb_array,
        trajectory_results['cca_score'],
        age_col=age_col,
        sample_id_col=sample_id_col,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    
    # Create visualizations
    viz_paths = {}
    if create_visualizations:
        print("\n5. Creating visualizations...")
        viz_paths = create_all_visualizations(
            emb_array, md_aligned, trajectory_results,
            method_outdir, method_name, config,
            sample_id_col=sample_id_col,
            modality_col=modality_col,
        )
        print(f"   - Modality plot: {viz_paths['modality_plot']}")
        print(f"   - Paired connections plot: {viz_paths['paired_plot']}")
        print(f"   - Trajectory embedding plot: {viz_paths['trajectory_embedding_plot']}")
        print(f"   - Trajectory scatter plot: {viz_paths['trajectory_scatter_plot']}")
    
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
    
    per_sample_path = method_outdir / "per_sample_metrics.csv"
    per_sample_df.to_csv(per_sample_path)
    
    # Save paired sample details
    paired_path = None
    if paired_results["paired_details"]:
        paired_df = pd.DataFrame(paired_results["paired_details"])
        paired_path = method_outdir / "paired_sample_distances.csv"
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
        f"Permutations: {n_permutations}",
        "",
        "--- KEY METRICS ---",
        "",
        "1. Mean Paired Distance (lower = better)",
        f"   Value: {paired_results['mean_paired_distance']:.4f}",
        f"   P-value: {paired_perm['p_value']:.2e}",
        "",
        "2. ASW Modality (higher = better)",
        f"   Value: {mixing_results['ASW_modality_overall']:.4f}",
        f"   P-value: {mixing_perm['ASW_p_value']:.2e}",
        "",
        "3. Trajectory CCA Score (higher = better)",
        f"   Value: {trajectory_results['cca_score']:.4f}",
        f"   P-value: {trajectory_perm['p_value']:.2e}",
        "",
        "--- ADDITIONAL DETAILS ---",
        "",
        "Paired Sample Distance:",
        f"  Number of pairs: {paired_results['n_pairs']}",
        f"  Std paired distance: {paired_results['std_paired_distance']:.4f}",
        f"  Median paired distance: {paired_results['median_paired_distance']:.4f}",
        "",
        "Modality Mixing:",
        f"  Modalities: {mixing_results['modalities']}",
        f"  iLISI mean: {mixing_results['iLISI_mean']:.4f}",
        f"  iLISI normalized: {mixing_results['iLISI_norm_mean']:.4f}",
        f"  iLISI P-value: {mixing_perm['iLISI_p_value']:.2e}",
        "",
        "Trajectory Analysis:",
        f"  Age column: {age_col}",
        f"  Age range: {trajectory_results['age_range'][0]:.1f} - {trajectory_results['age_range'][1]:.1f}",
        f"  CCA score: {trajectory_results['cca_score']:.4f}",
        f"  Pseudotime-age correlation: {trajectory_results['pseudotime_age_correlation']:.4f}",
        "",
        "=" * 60,
        f"Results saved to: {method_outdir}",
        "=" * 60,
    ]
    
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    
    summary_path = method_outdir / "integration_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    
    # Aggregate results for return
    return {
        # Method identifier
        "method_name": method_name,
        # Core metrics for summary CSV aggregation
        "n_samples": len(md_aligned),
        "n_pairs": paired_results["n_pairs"],
        "mean_paired_distance": paired_results["mean_paired_distance"],
        "std_paired_distance": paired_results["std_paired_distance"],
        "median_paired_distance": paired_results["median_paired_distance"],
        "paired_distance_pvalue": paired_perm["p_value"],
        "iLISI_mean": mixing_results["iLISI_mean"],
        "iLISI_norm_mean": mixing_results["iLISI_norm_mean"],
        "iLISI_pvalue": mixing_perm["iLISI_p_value"],
        "ASW_modality_overall": mixing_results["ASW_modality_overall"],
        "ASW_pvalue": mixing_perm["ASW_p_value"],
        "cca_score": trajectory_results["cca_score"],
        "cca_pvalue": trajectory_perm["p_value"],
        "pseudotime_age_correlation": trajectory_results["pseudotime_age_correlation"],
        # Metadata
        "n_modalities": mixing_results["n_modalities"],
        "modalities": mixing_results["modalities"],
        "age_col": age_col,
        "age_range": trajectory_results["age_range"],
        # File paths
        "method_outdir": str(method_outdir),
        "per_sample_path": str(per_sample_path),
        "paired_path": str(paired_path) if paired_path else None,
        "summary_path": str(summary_path),
        "visualization_paths": viz_paths,
    }


# =============================================================================
# Summary CSV Aggregation
# =============================================================================

def save_to_summary_csv(
    results: Dict[str, Any],
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
    
    method_name = results.get("method_name", "unknown")
    
    # Metrics to save
    metrics_to_save = {
        "n_samples": results.get("n_samples"),
        "n_pairs": results.get("n_pairs"),
        "mean_paired_distance": results.get("mean_paired_distance"),
        "median_paired_distance": results.get("median_paired_distance"),
        "paired_distance_pvalue": results.get("paired_distance_pvalue"),
        "iLISI_mean": results.get("iLISI_mean"),
        "iLISI_norm_mean": results.get("iLISI_norm_mean"),
        "iLISI_pvalue": results.get("iLISI_pvalue"),
        "ASW_modality": results.get("ASW_modality_overall"),
        "ASW_pvalue": results.get("ASW_pvalue"),
        "cca_score": results.get("cca_score"),
        "cca_pvalue": results.get("cca_pvalue"),
        "pseudotime_age_correlation": results.get("pseudotime_age_correlation"),
    }
    
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
        summary_df.loc[metric, method_name] = value
    
    # Save
    summary_df.to_csv(summary_path, index_label="Metric")
    print(f"\nUpdated summary CSV: {summary_path} with column '{method_name}'")


# =============================================================================
# Main execution
# =============================================================================
if __name__ == '__main__':
    # Define paths
    meta_csv = "/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv"
    general_outdir = "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea"
    summary_csv_path = f"{general_outdir}/Benchmark_result/summary.csv"
    
    # SD_expression
    results = evaluate_multimodal_integration(
        meta_csv=meta_csv,
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/embeddings/sample_expression_embedding.csv",
        method_name="SD_expression",
        general_outdir=general_outdir,
        k_neighbors=3,
    )
    save_to_summary_csv(results, summary_csv_path)
    
    # SD_proportion
    results = evaluate_multimodal_integration(
        meta_csv=meta_csv,
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/embeddings/sample_proportion_embedding.csv",
        method_name="SD_proportion",
        general_outdir=general_outdir,
        k_neighbors=3,
    )
    save_to_summary_csv(results, summary_csv_path)
    
    # pilot
    results = evaluate_multimodal_integration(
        meta_csv=meta_csv,
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/pilot/wasserstein_distance_mds_10d.csv",
        method_name="pilot",
        general_outdir=general_outdir,
        k_neighbors=3,
    )
    save_to_summary_csv(results, summary_csv_path)
    
    # pseudobulk
    results = evaluate_multimodal_integration(
        meta_csv=meta_csv,
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/pseudobulk/pseudobulk/pca_embeddings.csv",
        method_name="pseudobulk",
        general_outdir=general_outdir,
        k_neighbors=3,
    )
    save_to_summary_csv(results, summary_csv_path)
    
    # QOT
    results = evaluate_multimodal_integration(
        meta_csv=meta_csv,
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/QOT/24_qot_distance_matrix_mds_10d.csv",
        method_name="QOT",
        general_outdir=general_outdir,
        k_neighbors=3,
    )
    save_to_summary_csv(results, summary_csv_path)
    
    # GEDI
    results = evaluate_multimodal_integration(
        meta_csv=meta_csv,
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/GEDI/gedi_sample_embedding.csv",
        method_name="GEDI",
        general_outdir=general_outdir,
        k_neighbors=3,
    )
    save_to_summary_csv(results, summary_csv_path)
    
    # Gloscope
    results = evaluate_multimodal_integration(
        meta_csv=meta_csv,
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/Gloscope/knn_divergence_mds_10d.csv",
        method_name="Gloscope",
        general_outdir=general_outdir,
        k_neighbors=3,
    )
    save_to_summary_csv(results, summary_csv_path)
    
    # MFA
    results = evaluate_multimodal_integration(
        meta_csv=meta_csv,
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/MFA/sample_embeddings.csv",
        method_name="MFA",
        general_outdir=general_outdir,
        k_neighbors=3,
    )
    save_to_summary_csv(results, summary_csv_path)
    
    # mustard
    results = evaluate_multimodal_integration(
        meta_csv=meta_csv,
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/mustard/sample_embedding.csv",
        method_name="mustard",
        general_outdir=general_outdir,
        k_neighbors=3,
    )
    save_to_summary_csv(results, summary_csv_path)
    
    # scPoli
    results = evaluate_multimodal_integration(
        meta_csv=meta_csv,
        embedding_csv="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/scPoli/sample_embeddings_full.csv",
        method_name="scPoli",
        general_outdir=general_outdir,
        k_neighbors=3,
    )
    save_to_summary_csv(results, summary_csv_path)
    
    print("\n" + "="*60)
    print("Benchmark suite completed!")
    print(f"Summary saved to: {summary_csv_path}")
    print("="*60)
