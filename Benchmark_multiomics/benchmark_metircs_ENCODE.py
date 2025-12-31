#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved Multimodal Integration Benchmark v2

Evaluates multimodal embeddings based on three criteria:
1. Paired sample matching: samples with same sample_id but different modality should be close
2. Modality mixing: modalities should be well-mixed (iLISI_norm, ASW_batch on modality)
3. Tissue preservation: within-tissue distances should be smaller than between-tissue distances

Improvements:
- Infers modality from sample name suffix (_RNA, _ATAC)
- Generates three visualization graphs per method
- Supports permutation testing for p-values
- Organized output directory structure with method subfolders
- Enhanced visualizations with professional styling

Usage:
    results = evaluate_multimodal_integration(
        meta_csv="sample_metadata.csv",
        embedding_csv="embeddings.csv",
        method_name="method_name",
        general_outdir="results/",
    )
"""

from __future__ import annotations
import os
import sys
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import seaborn as sns

# Set high-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


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
    
    # Tissue color palette (extended for many tissues)
    tissue_palette: str = 'husl'


# =============================================================================
# Professional Color Palettes
# =============================================================================

# Custom tissue color palette - visually distinct and colorblind-friendly
TISSUE_COLORS = [
    '#e41a1c',  # Red
    '#377eb8',  # Blue
    '#4daf4a',  # Green
    '#984ea3',  # Purple
    '#ff7f00',  # Orange
    '#ffff33',  # Yellow
    '#a65628',  # Brown
    '#f781bf',  # Pink
    '#999999',  # Gray
    '#66c2a5',  # Teal
    '#fc8d62',  # Salmon
    '#8da0cb',  # Periwinkle
    '#e78ac3',  # Magenta
    '#a6d854',  # Lime
    '#ffd92f',  # Gold
    '#e5c494',  # Tan
    '#b3b3b3',  # Light gray
    '#1b9e77',  # Dark teal
    '#d95f02',  # Dark orange
    '#7570b3',  # Violet
]


def get_tissue_colors(n_tissues: int) -> List[str]:
    """Get a list of distinct colors for tissues."""
    if n_tissues <= len(TISSUE_COLORS):
        return TISSUE_COLORS[:n_tissues]
    else:
        # Generate additional colors using HSL
        colors = TISSUE_COLORS.copy()
        for i in range(len(TISSUE_COLORS), n_tissues):
            hue = (i * 0.618033988749895) % 1  # Golden ratio for distribution
            colors.append(plt.cm.hsv(hue))
        return colors


# =============================================================================
# Modality Inference from Sample Names
# =============================================================================

def infer_modality_from_name(sample_name: str) -> Tuple[str, str]:
    """
    Infer sample_id and modality from sample name suffix.
    
    Supports formats:
    - suffix: SAMPLEID_RNA, SAMPLEID_ATAC
    - prefix: RNA_SAMPLEID, ATAC_SAMPLEID
    
    Returns:
    --------
    Tuple[str, str]: (sample_id, modality)
    """
    sample_name = str(sample_name)
    
    # Check suffix format first (more common)
    if sample_name.endswith('_RNA'):
        return sample_name[:-4], 'RNA'
    elif sample_name.endswith('_ATAC'):
        return sample_name[:-5], 'ATAC'
    # Check prefix format
    elif sample_name.startswith('RNA_'):
        return sample_name[4:], 'RNA'
    elif sample_name.startswith('ATAC_'):
        return sample_name[5:], 'ATAC'
    else:
        # Return original name with unknown modality
        return sample_name, 'unknown'


def parse_sample_names(sample_names: np.ndarray) -> pd.DataFrame:
    """
    Parse sample names to extract sample_id and modality.
    
    Parameters:
    -----------
    sample_names : np.ndarray
        Array of sample names
        
    Returns:
    --------
    pd.DataFrame with columns: sample, sample_id, modality
    """
    records = []
    for name in sample_names:
        sample_id, modality = infer_modality_from_name(name)
        records.append({
            'sample': name,
            'sample_id': sample_id,
            'modality': modality,
        })
    
    df = pd.DataFrame(records)
    df = df.set_index('sample')
    return df


# =============================================================================
# I/O and Alignment
# =============================================================================

def read_metadata(meta_csv: str) -> pd.DataFrame:
    """Read metadata CSV with required columns."""
    md = pd.read_csv(meta_csv, index_col=0)
    md.columns = [c.lower() for c in md.columns]
    
    # Only require tissue column - modality will be inferred
    if 'tissue' not in md.columns:
        raise ValueError("Metadata must contain 'tissue' column")
    
    return md


def read_embedding(embedding_csv: str) -> pd.DataFrame:
    """Read embedding CSV (samples × dimensions)."""
    df = pd.read_csv(embedding_csv, index_col=0)
    if df.shape[1] < 1:
        raise ValueError("Embedding file must have ≥1 dimension columns.")
    return df


def align_data(
    md: pd.DataFrame, 
    emb: pd.DataFrame, 
    sample_info: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align metadata, embedding, and sample info by sample index.
    
    Returns:
    --------
    Tuple of aligned (metadata, embedding, sample_info) DataFrames
    """
    # Find common samples across all three
    common = md.index.intersection(emb.index).intersection(sample_info.index)
    
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between metadata, embedding, and sample info.")
    
    if len(common) < len(md):
        print(f"Note: dropping {len(md) - len(common)} metadata rows without embedding.", file=sys.stderr)
    if len(common) < len(emb):
        print(f"Note: dropping {len(emb) - len(common)} embedding rows without metadata.", file=sys.stderr)
    
    # Sort to ensure consistent ordering
    common_sorted = sorted(common)
    return (
        md.loc[common_sorted].copy(),
        emb.loc[common_sorted].copy(),
        sample_info.loc[common_sorted].copy()
    )


# =============================================================================
# Metric 1: Paired Sample Distance
# =============================================================================

def compute_paired_distance(
    sample_info: pd.DataFrame,
    emb: np.ndarray,
    metric: str = "euclidean",
) -> Dict[str, Any]:
    """
    Compute average distance between paired samples (same sample_id, different modality).
    
    Lower distance = better pairing/alignment of modalities.
    """
    # Build sample_id -> {modality: row_idx} mapping
    sample_id_to_idx: Dict[str, Dict[str, int]] = {}
    
    for i, (idx, row) in enumerate(sample_info.iterrows()):
        sid = str(row['sample_id'])
        mod = str(row['modality'])
        if sid not in sample_id_to_idx:
            sample_id_to_idx[sid] = {}
        sample_id_to_idx[sid][mod] = i
    
    # Find all paired samples (those with exactly 2 modalities)
    paired_distances = []
    paired_info = []
    paired_indices = []  # Store (idx1, idx2) for permutation testing
    
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
    sample_info: pd.DataFrame,
    emb: np.ndarray,
    k: int = 3,
    include_self: bool = False,
) -> Dict[str, Any]:
    """
    Compute modality mixing metrics: iLISI and ASW-batch.
    
    Higher iLISI_norm and ASW_batch = better modality mixing.
    """
    # Encode modality as integers
    modalities_str = sample_info['modality'].astype(str).values
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
        "modality_labels": labels_int,
        "knn_idx": knn_idx,
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
            "tissue_labels": tissue_labels,
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
        "tissue_labels": tissue_labels,
    }


# =============================================================================
# Permutation Testing
# =============================================================================

def permutation_test_paired_distance(
    sample_info: pd.DataFrame,
    emb: np.ndarray,
    observed_mean: float,
    paired_indices: List[Tuple[int, int]],
    n_permutations: int = 1000,
    metric: str = "euclidean",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Permutation test for paired sample distance.
    
    Null hypothesis: The mean paired distance is not different from random pairing.
    Shuffle the embedding indices to break the true pairing.
    """
    if len(paired_indices) == 0 or np.isnan(observed_mean):
        return {"p_value": np.nan, "null_distribution": []}
    
    rng = np.random.default_rng(random_seed)
    null_means = []
    n_samples = emb.shape[0]
    n_pairs = len(paired_indices)
    
    for _ in range(n_permutations):
        # Shuffle indices to create random pairings
        shuffled_idx = rng.permutation(n_samples)
        
        # Compute distances for random pairs (same number as true pairs)
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
    
    # One-sided p-value: proportion of null values <= observed (lower is better)
    p_value = float(np.mean(null_means <= observed_mean))
    
    return {
        "p_value": p_value,
        "null_mean": float(np.mean(null_means)),
        "null_std": float(np.std(null_means)),
        "null_distribution": null_means.tolist(),
    }


def permutation_test_modality_mixing(
    sample_info: pd.DataFrame,
    emb: np.ndarray,
    observed_ilisi: float,
    observed_asw: float,
    k: int = 3,
    n_permutations: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Permutation test for modality mixing metrics.
    
    Null hypothesis: Modality labels are random (no association with embedding).
    Shuffle modality labels.
    """
    rng = np.random.default_rng(random_seed)
    modalities = sample_info['modality'].values.copy()
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
        # Shuffle modality labels
        perm_labels = rng.permutation(labels_int)
        
        # Compute iLISI
        ilisi_per = compute_ilisi(perm_labels, knn_idx, include_self=False)
        ilisi_norm = float(np.mean(ilisi_per) / max(1, n_modalities))
        null_ilisi.append(ilisi_norm)
        
        # Compute ASW
        if n_samples > n_modalities:
            s_overall = silhouette_score(emb, perm_labels, metric="euclidean")
            asw = float(np.clip((1.0 - s_overall) / 2.0, 0.0, 1.0))
            null_asw.append(asw)
    
    null_ilisi = np.array(null_ilisi)
    null_asw = np.array(null_asw) if null_asw else np.array([np.nan])
    
    # Higher iLISI and ASW = better mixing
    # P-value: proportion of null values >= observed
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


def permutation_test_tissue_preservation(
    md: pd.DataFrame,
    emb: np.ndarray,
    observed_score: float,
    tissue_col: str = "tissue",
    n_permutations: int = 1000,
    metric: str = "euclidean",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Permutation test for tissue preservation score.
    
    Null hypothesis: Tissue labels are random (no association with embedding).
    Shuffle tissue labels.
    """
    rng = np.random.default_rng(random_seed)
    tissues = md[tissue_col].values.copy()
    unique_tissues, tissue_labels = np.unique(tissues, return_inverse=True)
    n_tissues = len(unique_tissues)
    
    if n_tissues < 2 or np.isnan(observed_score):
        return {
            "p_value": np.nan,
            "null_distribution": [],
        }
    
    # Precompute distance matrix
    dist_matrix = squareform(pdist(emb, metric=metric))
    
    null_scores = []
    
    for _ in range(n_permutations):
        # Shuffle tissue labels
        perm_labels = rng.permutation(tissue_labels)
        
        # Compute within and between distances
        within_dists = []
        between_dists = []
        
        for t_idx in range(n_tissues):
            tissue_mask = perm_labels == t_idx
            tissue_indices = np.where(tissue_mask)[0]
            other_indices = np.where(~tissue_mask)[0]
            
            if len(tissue_indices) > 1:
                for i in range(len(tissue_indices)):
                    for j in range(i + 1, len(tissue_indices)):
                        within_dists.append(dist_matrix[tissue_indices[i], tissue_indices[j]])
            
            if len(tissue_indices) > 0 and len(other_indices) > 0:
                between_dists.extend(dist_matrix[np.ix_(tissue_indices, other_indices)].flatten())
        
        mean_within = np.mean(within_dists) if within_dists else np.nan
        mean_between = np.mean(between_dists) if between_dists else np.nan
        
        if mean_within > 0 and not np.isnan(mean_within):
            null_scores.append(mean_between / mean_within)
    
    null_scores = np.array(null_scores)
    
    # Higher preservation score = better
    # P-value: proportion of null values >= observed
    p_value = float(np.mean(null_scores >= observed_score))
    
    return {
        "p_value": p_value,
        "null_mean": float(np.mean(null_scores)),
        "null_std": float(np.std(null_scores)),
        "null_distribution": null_scores.tolist(),
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


def add_text_with_outline(ax, x, y, text, fontsize=10, color='black', outline_color='white', outline_width=2):
    """Add text with outline for better visibility."""
    txt = ax.text(x, y, text, fontsize=fontsize, fontweight='bold', color=color,
                  ha='center', va='center', transform=ax.transAxes)
    txt.set_path_effects([
        path_effects.Stroke(linewidth=outline_width, foreground=outline_color),
        path_effects.Normal()
    ])
    return txt


def plot_embedding_by_modality(
    emb_2d: np.ndarray,
    sample_info: pd.DataFrame,
    output_path: str,
    config: BenchmarkConfig,
    method_name: str = "",
    mixing_results: Optional[Dict] = None,
) -> plt.Figure:
    """Plot 2D embedding colored by modality with enhanced styling."""
    fig, ax = plt.subplots(figsize=config.figsize)
    
    # Set background
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    modalities = sample_info['modality'].values
    unique_mods = sorted(np.unique(modalities))
    
    # Plot each modality
    for mod in unique_mods:
        mask = modalities == mod
        color = config.modality_colors.get(mod, '#333333')
        
        scatter = ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            s=config.point_size,
            c=color,
            alpha=config.point_alpha,
            label=mod,
            edgecolors='white',
            linewidths=0.8,
            zorder=3,
        )
    
    # Styling
    ax.set_xlabel('PC1', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Embedding by Modality', fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    legend = ax.legend(
        title='Modality', 
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=13,
        title_fontsize=14,
        edgecolor='#cccccc',
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=11, length=5, width=1.2)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)


def plot_embedding_by_tissue(
    emb_2d: np.ndarray,
    md: pd.DataFrame,
    output_path: str,
    config: BenchmarkConfig,
    tissue_col: str = "tissue",
    method_name: str = "",
    tissue_results: Optional[Dict] = None,
) -> plt.Figure:
    """Plot 2D embedding colored by tissue with enhanced styling."""
    fig, ax = plt.subplots(figsize=config.figsize)
    
    # Set background
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    tissues = md[tissue_col].values
    unique_tissues = sorted(np.unique(tissues))
    n_tissues = len(unique_tissues)
    
    # Get colors
    colors = get_tissue_colors(n_tissues)
    color_map = {t: colors[i] for i, t in enumerate(unique_tissues)}
    
    # Plot each tissue
    for tissue in unique_tissues:
        mask = tissues == tissue
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            s=config.point_size,
            c=[color_map[tissue]],
            alpha=config.point_alpha,
            label=f'{tissue}',
            edgecolors='white',
            linewidths=0.8,
            zorder=3,
        )
    
    # Styling
    ax.set_xlabel('PC1', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Embedding by Tissue', fontsize=16, fontweight='bold', pad=20)
    
    # Legend - always inside the plot
    ncol = 1 if n_tissues <= 10 else 2
    legend = ax.legend(
        title='Tissue', 
        loc='best',
        frameon=True,
        fancybox=True,
        fontsize=10,
        title_fontsize=11,
        edgecolor='#cccccc',
        ncol=ncol,
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=11, length=5, width=1.2)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)


def plot_paired_connections(
    emb_2d: np.ndarray,
    sample_info: pd.DataFrame,
    output_path: str,
    config: BenchmarkConfig,
    method_name: str = "",
    paired_results: Optional[Dict] = None,
) -> plt.Figure:
    """Plot 2D embedding with connections between paired samples - enhanced styling."""
    fig, ax = plt.subplots(figsize=config.figsize)
    
    # Set background
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    # Build sample_id -> {modality: row_idx} mapping
    sample_id_to_idx: Dict[str, Dict[str, int]] = {}
    
    for i, (idx, row) in enumerate(sample_info.iterrows()):
        sid = str(row['sample_id'])
        mod = str(row['modality'])
        if sid not in sample_id_to_idx:
            sample_id_to_idx[sid] = {}
        sample_id_to_idx[sid][mod] = i
    
    # Draw connections with single color
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
    
    # Draw points with different markers for each modality
    modalities = sample_info['modality'].values
    unique_mods = sorted(np.unique(modalities))
    markers = {'RNA': 'o', 'ATAC': 's'}  # circle for RNA, square for ATAC
    
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
    
    # Styling
    ax.set_xlabel('PC1', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Paired Sample Connections', fontsize=16, fontweight='bold', pad=20)
    
    # Custom legend
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
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=11, length=5, width=1.2)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)


def create_all_visualizations(
    emb: np.ndarray,
    md: pd.DataFrame,
    sample_info: pd.DataFrame,
    method_outdir: Path,
    method_name: str,
    config: BenchmarkConfig,
    paired_results: Optional[Dict] = None,
    mixing_results: Optional[Dict] = None,
    tissue_results: Optional[Dict] = None,
) -> Dict[str, str]:
    """Create all three visualization plots for a method."""
    # Reduce to 2D
    emb_2d = reduce_to_2d(emb)
    
    # Create visualization directory
    viz_dir = method_outdir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: By modality
    modality_path = viz_dir / f"{method_name}_by_modality.png"
    plot_embedding_by_modality(
        emb_2d, sample_info, str(modality_path), config,
        method_name=method_name,
        mixing_results=mixing_results,
    )
    
    # Plot 2: By tissue
    tissue_path = viz_dir / f"{method_name}_by_tissue.png"
    plot_embedding_by_tissue(
        emb_2d, md, str(tissue_path), config,
        method_name=method_name,
        tissue_results=tissue_results,
    )
    
    # Plot 3: Paired connections
    paired_path = viz_dir / f"{method_name}_paired_connections.png"
    plot_paired_connections(
        emb_2d, sample_info, str(paired_path), config,
        method_name=method_name,
        paired_results=paired_results,
    )
    
    return {
        "modality_plot": str(modality_path),
        "tissue_plot": str(tissue_path),
        "paired_plot": str(paired_path),
    }


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_multimodal_integration(
    meta_csv: str,
    embedding_csv: str,
    method_name: str,
    general_outdir: str,
    tissue_col: str = "tissue",
    k_neighbors: int = 15,
    distance_metric: str = "euclidean",
    include_self: bool = False,
    n_permutations: int = 1000,
    random_seed: int = 42,
    create_visualizations: bool = True,
    config: Optional[BenchmarkConfig] = None,
) -> Dict[str, Any]:
    """
    Evaluate multimodal integration quality.
    
    Parameters
    ----------
    meta_csv : str
        Path to metadata CSV with columns: tissue (modality inferred from sample names)
    embedding_csv : str
        Path to embedding CSV (samples × dimensions), indexed by sample name
    method_name : str
        Name of the integration method being evaluated
    general_outdir : str
        General output directory (method will have its own subfolder)
    tissue_col : str
        Column name for tissue type
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
    md = read_metadata(meta_csv)
    
    print(f"Loading embedding from: {embedding_csv}")
    emb_df = read_embedding(embedding_csv)
    
    # Parse sample names to extract modality
    print("Inferring modality from sample names...")
    sample_info = parse_sample_names(emb_df.index.values)
    
    # Check modality distribution
    mod_counts = sample_info['modality'].value_counts()
    print(f"Modality distribution: {dict(mod_counts)}")
    
    if 'unknown' in mod_counts.index and mod_counts['unknown'] > 0:
        warnings.warn(f"Found {mod_counts['unknown']} samples with unknown modality")
    
    # Align data
    md_aligned, emb_aligned, sample_info_aligned = align_data(md, emb_df, sample_info)
    emb_array = emb_aligned.values.astype(float)
    
    print(f"Aligned data: {len(md_aligned)} samples, {emb_array.shape[1]} dimensions")
    
    # Compute all metrics
    print("\n1. Computing paired sample distances...")
    paired_results = compute_paired_distance(
        sample_info_aligned, emb_array,
        metric=distance_metric,
    )
    
    print("\n2. Computing modality mixing (iLISI, ASW)...")
    mixing_results = compute_modality_mixing(
        sample_info_aligned, emb_array,
        k=k_neighbors,
        include_self=include_self,
    )
    
    print("\n3. Computing tissue preservation...")
    tissue_results = compute_tissue_preservation(
        md_aligned, emb_array,
        tissue_col=tissue_col,
        metric=distance_metric,
    )
    
    # Permutation testing
    print("\n4. Running permutation tests...")
    
    print("   - Paired distance permutation test...")
    paired_perm = permutation_test_paired_distance(
        sample_info_aligned, emb_array,
        paired_results['mean_paired_distance'],
        paired_results['paired_indices'],
        n_permutations=n_permutations,
        metric=distance_metric,
        random_seed=random_seed,
    )
    
    print("   - Modality mixing permutation test...")
    mixing_perm = permutation_test_modality_mixing(
        sample_info_aligned, emb_array,
        mixing_results['iLISI_norm_mean'],
        mixing_results['ASW_modality_overall'],
        k=k_neighbors,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    
    print("   - Tissue preservation permutation test...")
    tissue_perm = permutation_test_tissue_preservation(
        md_aligned, emb_array,
        tissue_results['tissue_preservation_score'],
        tissue_col=tissue_col,
        n_permutations=n_permutations,
        metric=distance_metric,
        random_seed=random_seed,
    )
    
    # Create visualizations
    viz_paths = {}
    if create_visualizations:
        print("\n5. Creating visualizations...")
        viz_paths = create_all_visualizations(
            emb_array, md_aligned, sample_info_aligned,
            method_outdir, method_name, config,
            paired_results=paired_results,
            mixing_results=mixing_results,
            tissue_results=tissue_results,
        )
        print(f"   - Modality plot: {viz_paths['modality_plot']}")
        print(f"   - Tissue plot: {viz_paths['tissue_plot']}")
        print(f"   - Paired connections plot: {viz_paths['paired_plot']}")
    
    # Save per-sample metrics
    per_sample_df = pd.DataFrame({
        "sample": md_aligned.index,
        "sample_id": sample_info_aligned['sample_id'].values,
        "modality": sample_info_aligned['modality'].values,
        "tissue": md_aligned[tissue_col].values,
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
    
    # Save tissue details
    tissue_details_df = pd.DataFrame(tissue_results["tissue_details"]).T
    tissue_details_df.index.name = "tissue"
    tissue_details_path = method_outdir / "tissue_details.csv"
    tissue_details_df.to_csv(tissue_details_path)
    
    # Generate summary
    summary_lines = [
        "=" * 60,
        f"Multimodal Integration Evaluation: {method_name}",
        "=" * 60,
        "",
        f"Total samples: {len(md_aligned)}",
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
        "3. Tissue Preservation Score (higher = better)",
        f"   Value: {tissue_results['tissue_preservation_score']:.4f}",
        f"   P-value: {tissue_perm['p_value']:.2e}",
        "",
        "--- ADDITIONAL DETAILS ---",
        "",
        "Paired Sample Distance:",
        f"  Number of pairs: {paired_results['n_pairs']}",
        f"  Std paired distance:  {paired_results['std_paired_distance']:.4f}",
        f"  Median paired distance: {paired_results['median_paired_distance']:.4f}",
        "",
        "Modality Mixing:",
        f"  Modalities: {mixing_results['modalities']}",
        f"  iLISI mean:       {mixing_results['iLISI_mean']:.4f}",
        f"  iLISI normalized: {mixing_results['iLISI_norm_mean']:.4f}",
        f"  iLISI P-value:    {mixing_perm['iLISI_p_value']:.2e}",
        "",
        "Tissue Preservation:",
        f"  Number of tissues: {tissue_results['n_tissues']}",
        f"  Tissues: {tissue_results['tissues']}",
        f"  Mean within-tissue distance:  {tissue_results['mean_within_tissue_distance']:.4f}",
        f"  Mean between-tissue distance: {tissue_results['mean_between_tissue_distance']:.4f}",
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
        "tissue_preservation_score": tissue_results["tissue_preservation_score"],
        "tissue_preservation_pvalue": tissue_perm["p_value"],
        "mean_within_tissue_distance": tissue_results["mean_within_tissue_distance"],
        "mean_between_tissue_distance": tissue_results["mean_between_tissue_distance"],
        # Metadata
        "n_modalities": mixing_results["n_modalities"],
        "modalities": mixing_results["modalities"],
        "n_tissues": tissue_results["n_tissues"],
        "tissues": tissue_results["tissues"],
        # File paths
        "method_outdir": str(method_outdir),
        "per_sample_path": str(per_sample_path),
        "paired_path": str(paired_path) if paired_path else None,
        "tissue_details_path": str(tissue_details_path),
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
    
    # Metrics to save (excluding lists and paths)
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
        "tissue_preservation_score": results.get("tissue_preservation_score"),
        "tissue_preservation_pvalue": results.get("tissue_preservation_pvalue"),
        "mean_within_tissue_distance": results.get("mean_within_tissue_distance"),
        "mean_between_tissue_distance": results.get("mean_between_tissue_distance"),
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


def run_benchmark_suite(
    meta_csv: str,
    embedding_configs: List[Dict[str, str]],
    general_outdir: str,
    **kwargs
) -> pd.DataFrame:
    """
    Run benchmark for multiple methods and generate combined summary.
    
    Parameters
    ----------
    meta_csv : str
        Path to metadata CSV
    embedding_configs : List[Dict[str, str]]
        List of dicts with 'method_name' and 'embedding_csv' keys
    general_outdir : str
        General output directory
    **kwargs
        Additional arguments passed to evaluate_multimodal_integration
        
    Returns
    -------
    pd.DataFrame : Summary of all methods
    """
    summary_csv_path = Path(general_outdir) / "summary.csv"
    
    all_results = []
    
    for config in embedding_configs:
        method_name = config['method_name']
        embedding_csv = config['embedding_csv']
        
        try:
            results = evaluate_multimodal_integration(
                meta_csv=meta_csv,
                embedding_csv=embedding_csv,
                method_name=method_name,
                general_outdir=general_outdir,
                **kwargs
            )
            save_to_summary_csv(results, str(summary_csv_path))
            all_results.append(results)
        except Exception as e:
            print(f"Error evaluating {method_name}: {e}")
            continue
    
    # Load and return summary
    if summary_csv_path.exists():
        return pd.read_csv(summary_csv_path, index_col=0)
    return pd.DataFrame()


# =============================================================================
# Original Function Calls - Updated for New API
# =============================================================================

if __name__ == "__main__":
    
    # Configuration
    META_CSV = "/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv"
    GENERAL_OUTDIR = "/dcs07/hongkai/data/harry/result/Benchmark_multiomics"
    SUMMARY_CSV = f"{GENERAL_OUTDIR}/summary.csv"
    N_PERMUTATIONS = 1000
    K_NEIGHBORS = 5
    
    # Define all methods and their embedding paths
    embedding_configs = [
        {
            "method_name": "SD_expression",
            "embedding_csv": "/dcs07/hongkai/data/harry/result/multi_omics_SD/multiomics/rna/Sample_distance/correlation/expression_DR_distance/expression_DR_coordinates.csv",
        },
        {
            "method_name": "SD_proportion",
            "embedding_csv": "/dcs07/hongkai/data/harry/result/multi_omics_SD/multiomics/rna/Sample_distance/correlation/proportion_DR_distance/proportion_DR_coordinates.csv",
        },
        {
            "method_name": "pilot",
            "embedding_csv": "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/pilot/pilot_native_embedding.csv",
        },
        {
            "method_name": "pseudobulk",
            "embedding_csv": "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/pseudobulk/pseudobulk/pca_embeddings.csv",
        },
        {
            "method_name": "QOT",
            "embedding_csv": "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/QOT/88_qot_distance_matrix_mds_10d.csv",
        },
        {
            "method_name": "GEDI",
            "embedding_csv": "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/GEDI/gedi_sample_embedding.csv",
        },
        {
            "method_name": "Gloscope",
            "embedding_csv": "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/Gloscope/knn_divergence_mds_10d.csv",
        },
        {
            "method_name": "MFA",
            "embedding_csv": "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/MFA/sample_embeddings.csv",
        },
        {
            "method_name": "mustard",
            "embedding_csv": "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/mustard/sample_embedding.csv",
        },
        {
            "method_name": "scPoli",
            "embedding_csv": "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/scPoli/sample_embeddings_full.csv",
        },
    ]
    
    # Option 1: Run all methods using run_benchmark_suite (recommended)
    print("=" * 80)
    print("MULTIMODAL INTEGRATION BENCHMARK")
    print("=" * 80)
    print(f"\nMetadata: {META_CSV}")
    print(f"Output directory: {GENERAL_OUTDIR}")
    print(f"Number of methods: {len(embedding_configs)}")
    print(f"Permutations: {N_PERMUTATIONS}")
    print("=" * 80)
    
    summary_df = run_benchmark_suite(
        meta_csv=META_CSV,
        embedding_configs=embedding_configs,
        general_outdir=GENERAL_OUTDIR,
        k_neighbors=K_NEIGHBORS,
        n_permutations=N_PERMUTATIONS,
    )
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nSummary saved to: {SUMMARY_CSV}")
    print("\nSummary:")
    print(summary_df.to_string())