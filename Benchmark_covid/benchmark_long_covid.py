"""
Comprehensive Benchmark Wrapper for Eye Dataset

This wrapper provides both:
1. Trajectory-focused benchmarking (age as continuous variable)
2. Multimodal integration benchmarking (month as categorical variable)

Features:
- Age-based trajectory analysis (ANOVA, Spearman correlation)
- Month-based multimodal integration (paired distance, modality mixing, tissue preservation)
- Case-insensitive sample ID matching
- Automatic numerical label detection
- Robust error handling and detailed logging
- Comprehensive visualization suite
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import logging
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import pdist, squareform, cdist

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from spearman_test import run_trajectory_analysis

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set high-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# =============================================================================
# Multimodal Integration Configuration
# =============================================================================

@dataclass
class MultimodalConfig:
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
    
    # Colors
    modality_colors: Dict[str, str] = field(default_factory=lambda: {
        'RNA': '#3498db',
        'ATAC': '#e74c3c',
    })
    connection_color: str = '#7f8c8d'
    tissue_palette: str = 'husl'


# Custom tissue color palette
TISSUE_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
    '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb',
    '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3', '#1b9e77',
    '#d95f02', '#7570b3',
]


def get_tissue_colors(n_tissues: int) -> List[str]:
    """Get a list of distinct colors for tissues."""
    if n_tissues <= len(TISSUE_COLORS):
        return TISSUE_COLORS[:n_tissues]
    else:
        colors = TISSUE_COLORS.copy()
        for i in range(len(TISSUE_COLORS), n_tissues):
            hue = (i * 0.618033988749895) % 1
            colors.append(plt.cm.hsv(hue))
        return colors


# =============================================================================
# Modality Inference
# =============================================================================

def infer_modality_from_name(sample_name: str) -> Tuple[str, str]:
    """Infer sample_id and modality from sample name suffix."""
    sample_name = str(sample_name)
    
    if sample_name.endswith('_RNA'):
        return sample_name[:-4], 'RNA'
    elif sample_name.endswith('_ATAC'):
        return sample_name[:-5], 'ATAC'
    elif sample_name.startswith('RNA_'):
        return sample_name[4:], 'RNA'
    elif sample_name.startswith('ATAC_'):
        return sample_name[5:], 'ATAC'
    else:
        return sample_name, 'unknown'


def parse_sample_names(sample_names: np.ndarray) -> pd.DataFrame:
    """Parse sample names to extract sample_id and modality."""
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
# Multimodal Integration Metrics
# =============================================================================

def compute_paired_distance(
    sample_info: pd.DataFrame,
    emb: np.ndarray,
    metric: str = "euclidean",
) -> Dict[str, Any]:
    """Compute average distance between paired samples."""
    sample_id_to_idx: Dict[str, Dict[str, int]] = {}
    
    for i, (idx, row) in enumerate(sample_info.iterrows()):
        sid = str(row['sample_id'])
        mod = str(row['modality'])
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
    """Compute per-sample iLISI."""
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
    k: int = 15,
    include_self: bool = False,
) -> Dict[str, Any]:
    """Compute modality mixing metrics: iLISI and ASW-batch."""
    modalities_str = sample_info['modality'].astype(str).values
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


def compute_month_preservation(
    md: pd.DataFrame,
    emb: np.ndarray,
    month_col: str = "month",
    metric: str = "euclidean",
) -> Dict[str, Any]:
    """Compute month preservation: ratio of between-month to within-month distance."""
    months_str = md[month_col].astype(str).values
    unique_months, month_labels = np.unique(months_str, return_inverse=True)
    n_months = len(unique_months)
    
    if n_months < 2:
        return {
            "n_months": n_months,
            "months": list(unique_months),
            "mean_within_month_distance": np.nan,
            "mean_between_month_distance": np.nan,
            "month_preservation_score": np.nan,
            "month_details": {},
            "month_labels": month_labels,
        }
    
    dist_matrix = squareform(pdist(emb, metric=metric))
    
    within_distances = []
    between_distances = []
    month_details = {}
    
    for m_idx, month in enumerate(unique_months):
        month_mask = month_labels == m_idx
        month_indices = np.where(month_mask)[0]
        other_indices = np.where(~month_mask)[0]
        
        if len(month_indices) > 1:
            within_dists = []
            for i in range(len(month_indices)):
                for j in range(i + 1, len(month_indices)):
                    within_dists.append(dist_matrix[month_indices[i], month_indices[j]])
            within_distances.extend(within_dists)
            month_details[month] = {
                "n_samples": len(month_indices),
                "mean_within_distance": float(np.mean(within_dists)) if within_dists else np.nan,
            }
        else:
            month_details[month] = {
                "n_samples": len(month_indices),
                "mean_within_distance": np.nan,
            }
        
        if len(month_indices) > 0 and len(other_indices) > 0:
            between_dists = dist_matrix[np.ix_(month_indices, other_indices)].flatten()
            between_distances.extend(between_dists.tolist())
    
    mean_within = float(np.mean(within_distances)) if within_distances else np.nan
    mean_between = float(np.mean(between_distances)) if between_distances else np.nan
    
    if mean_within > 0 and not np.isnan(mean_within):
        preservation_score = mean_between / mean_within
    else:
        preservation_score = np.nan
    
    return {
        "n_months": n_months,
        "months": list(unique_months),
        "mean_within_month_distance": mean_within,
        "std_within_month_distance": float(np.std(within_distances, ddof=1)) if len(within_distances) > 1 else 0.0,
        "mean_between_month_distance": mean_between,
        "std_between_month_distance": float(np.std(between_distances, ddof=1)) if len(between_distances) > 1 else 0.0,
        "month_preservation_score": float(preservation_score) if not np.isnan(preservation_score) else np.nan,
        "month_details": month_details,
        "month_labels": month_labels,
    }


# =============================================================================
# Permutation Testing
# =============================================================================

def permutation_test_paired_distance(
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
    }


def permutation_test_modality_mixing(
    sample_info: pd.DataFrame,
    emb: np.ndarray,
    observed_ilisi: float,
    observed_asw: float,
    k: int = 15,
    n_permutations: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Permutation test for modality mixing metrics."""
    rng = np.random.default_rng(random_seed)
    modalities = sample_info['modality'].values.copy()
    unique_mods, labels_int = np.unique(modalities, return_inverse=True)
    n_modalities = len(unique_mods)
    n_samples = emb.shape[0]
    
    if n_modalities < 2:
        return {"iLISI_p_value": np.nan, "ASW_p_value": np.nan}
    
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
    }


def permutation_test_month_preservation(
    md: pd.DataFrame,
    emb: np.ndarray,
    observed_score: float,
    month_col: str = "month",
    n_permutations: int = 1000,
    metric: str = "euclidean",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Permutation test for month preservation score."""
    rng = np.random.default_rng(random_seed)
    months = md[month_col].values.copy()
    unique_months, month_labels = np.unique(months, return_inverse=True)
    n_months = len(unique_months)
    
    if n_months < 2 or np.isnan(observed_score):
        return {"p_value": np.nan}
    
    dist_matrix = squareform(pdist(emb, metric=metric))
    null_scores = []
    
    for _ in range(n_permutations):
        perm_labels = rng.permutation(month_labels)
        within_dists = []
        between_dists = []
        
        for m_idx in range(n_months):
            month_mask = perm_labels == m_idx
            month_indices = np.where(month_mask)[0]
            other_indices = np.where(~month_mask)[0]
            
            if len(month_indices) > 1:
                for i in range(len(month_indices)):
                    for j in range(i + 1, len(month_indices)):
                        within_dists.append(dist_matrix[month_indices[i], month_indices[j]])
            
            if len(month_indices) > 0 and len(other_indices) > 0:
                between_dists.extend(dist_matrix[np.ix_(month_indices, other_indices)].flatten())
        
        mean_within = np.mean(within_dists) if within_dists else np.nan
        mean_between = np.mean(between_dists) if between_dists else np.nan
        
        if mean_within > 0 and not np.isnan(mean_within):
            null_scores.append(mean_between / mean_within)
    
    null_scores = np.array(null_scores)
    p_value = float(np.mean(null_scores >= observed_score))
    
    return {
        "p_value": p_value,
        "null_mean": float(np.mean(null_scores)),
        "null_std": float(np.std(null_scores)),
    }


# =============================================================================
# Visualization Functions
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
    sample_info: pd.DataFrame,
    output_path: str,
    config: MultimodalConfig,
) -> None:
    """Plot 2D embedding colored by modality."""
    fig, ax = plt.subplots(figsize=config.figsize)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    modalities = sample_info['modality'].values
    unique_mods = sorted(np.unique(modalities))
    
    for mod in unique_mods:
        mask = modalities == mod
        color = config.modality_colors.get(mod, '#333333')
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            s=config.point_size, c=color, alpha=config.point_alpha,
            label=mod, edgecolors='white', linewidths=0.8, zorder=3,
        )
    
    ax.set_xlabel('PC1', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Embedding by Modality', fontsize=16, fontweight='bold', pad=20)
    ax.legend(title='Modality', loc='upper right', frameon=True, fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#cccccc')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)


def plot_embedding_by_month(
    emb_2d: np.ndarray,
    md: pd.DataFrame,
    output_path: str,
    config: MultimodalConfig,
    month_col: str = "month",
) -> None:
    """Plot 2D embedding colored by month."""
    fig, ax = plt.subplots(figsize=config.figsize)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    months = md[month_col].values
    unique_months = sorted(np.unique(months))
    n_months = len(unique_months)
    
    colors = get_tissue_colors(n_months)
    color_map = {m: colors[i] for i, m in enumerate(unique_months)}
    
    for month in unique_months:
        mask = months == month
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            s=config.point_size, c=[color_map[month]], alpha=config.point_alpha,
            label=f'{month}', edgecolors='white', linewidths=0.8, zorder=3,
        )
    
    ax.set_xlabel('PC1', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Embedding by Month', fontsize=16, fontweight='bold', pad=20)
    ax.legend(title='Month', loc='best', frameon=True, fontsize=10, ncol=1 if n_months <= 10 else 2)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#cccccc')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)


def plot_paired_connections(
    emb_2d: np.ndarray,
    sample_info: pd.DataFrame,
    output_path: str,
    config: MultimodalConfig,
) -> None:
    """Plot 2D embedding with connections between paired samples."""
    fig, ax = plt.subplots(figsize=config.figsize)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    sample_id_to_idx: Dict[str, Dict[str, int]] = {}
    for i, (idx, row) in enumerate(sample_info.iterrows()):
        sid = str(row['sample_id'])
        mod = str(row['modality'])
        if sid not in sample_id_to_idx:
            sample_id_to_idx[sid] = {}
        sample_id_to_idx[sid][mod] = i
    
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
    
    modalities = sample_info['modality'].values
    unique_mods = sorted(np.unique(modalities))
    markers = {'RNA': 'o', 'ATAC': 's'}
    
    for mod in unique_mods:
        mask = modalities == mod
        color = config.modality_colors.get(mod, '#333333')
        marker = markers.get(mod, 'o')
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            s=config.point_size, c=color, alpha=config.point_alpha,
            label=mod, marker=marker, edgecolors='white', linewidths=1.0, zorder=3,
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
    
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#cccccc')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Main Benchmark Wrapper Class
# =============================================================================

class BenchmarkWrapper:
    """
    Comprehensive benchmark wrapper combining trajectory and multimodal integration analysis.
    """

    def __init__(
        self,
        meta_csv_path: str,
        embedding_csv_path: str,
        method_name: str = "method",
        label_col: str = "age",
        month_col: str = "month",
        pseudotime_csv_path: Optional[str] = None,
        output_base_dir: Optional[str] = None,
        summary_csv_path: Optional[str] = None,
        multimodal_config: Optional[MultimodalConfig] = None,
    ):
        # Store and validate core inputs
        self.meta_csv_path = Path(meta_csv_path).resolve()
        self.embedding_csv_path = Path(embedding_csv_path).resolve()
        self.pseudotime_csv_path = Path(pseudotime_csv_path).resolve() if pseudotime_csv_path else None
        self.method_name = method_name
        self.label_col = label_col
        self.month_col = month_col

        if not self.meta_csv_path.exists() or not self.meta_csv_path.is_file():
            raise FileNotFoundError(f"Metadata CSV does not exist: {self.meta_csv_path}")
        
        if not self.embedding_csv_path.exists() or not self.embedding_csv_path.is_file():
            raise FileNotFoundError(f"Embedding CSV does not exist: {self.embedding_csv_path}")

        # Output directory strategy
        if output_base_dir is None:
            self.output_base_dir = self.meta_csv_path.parent
        else:
            self.output_base_dir = Path(output_base_dir).resolve()

        # Summary CSV path
        if summary_csv_path is not None:
            self.summary_csv_path = Path(summary_csv_path).resolve()
        else:
            self.summary_csv_path = self.output_base_dir / "benchmark_summary.csv"

        # Output directory for this run
        self.run_output_dir = self.output_base_dir / f"benchmark_results_{self.method_name}"
        self.run_output_dir.mkdir(parents=True, exist_ok=True)

        # Multimodal config
        self.multimodal_config = multimodal_config or MultimodalConfig()

        logger.info("=" * 70)
        logger.info("Initialized Comprehensive BenchmarkWrapper")
        logger.info("=" * 70)
        logger.info(f"  Meta CSV:          {self.meta_csv_path}")
        logger.info(f"  Embedding CSV:     {self.embedding_csv_path}")
        logger.info(f"  Pseudotime CSV:    {self.pseudotime_csv_path if self.pseudotime_csv_path else '(not provided)'}")
        logger.info(f"  Method name:       {self.method_name}")
        logger.info(f"  Label column:      {self.label_col}")
        logger.info(f"  Month column:      {self.month_col}")
        logger.info(f"  Output base dir:   {self.output_base_dir}")
        logger.info(f"  Run output dir:    {self.run_output_dir}")
        logger.info(f"  Summary CSV:       {self.summary_csv_path}")
        logger.info("=" * 70)

    # ------------------------- Helper Methods -------------------------

    def _create_output_dir(self, benchmark_name: str) -> Path:
        """Create and return output directory for a specific benchmark."""
        out = self.run_output_dir / benchmark_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _check_file_exists(self, file_path: Optional[Path], file_description: str) -> bool:
        """Check if a file exists and log diagnostics if not."""
        if file_path is None:
            logger.error(f"ERROR: {file_description} was not provided.")
            return False

        if not file_path.exists():
            logger.error(f"ERROR: {file_description} not found at: {file_path}")
            return False

        if not file_path.is_file():
            logger.error(f"ERROR: {file_description} path is not a file: {file_path}")
            return False

        return True

    def _normalize_sample_ids(self, series_or_index) -> pd.Index:
        """Normalize sample IDs for case-insensitive matching."""
        if isinstance(series_or_index, pd.Series):
            return pd.Index(series_or_index.astype(str).str.lower().str.strip())
        else:
            return pd.Index(series_or_index.astype(str).str.lower().str.strip())

    def _save_summary_csv(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Save benchmark results to summary CSV."""
        summary_csv_path = self.summary_csv_path
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

        all_metrics = {}
        
        for benchmark_name, bench_result in results.items():
            if bench_result.get("status") != "success":
                logger.warning(f"Skipping {benchmark_name} in summary - status was not 'success'")
                continue
            
            result = bench_result.get("result", {})
            if result is None:
                result = {}
            
            # Trajectory benchmarks
            if benchmark_name == "embedding_visualization":
                if "n_samples" in result:
                    all_metrics["n_samples"] = result["n_samples"]

            elif benchmark_name == "trajectory_anova":
                anova_table = result.get("anova_table")
                if anova_table is not None and hasattr(anova_table, 'loc'):
                    try:
                        target_row = self.label_col
                        if target_row in anova_table.index and 'partial_eta_sq' in anova_table.columns:
                            val = anova_table.loc[target_row, 'partial_eta_sq']
                            all_metrics[f"One_way_ANOVA_eta_sq"] = float(val)
                    except Exception as e:
                        logger.warning(f"Could not extract ANOVA metrics: {e}")
                    
            elif benchmark_name == "trajectory_analysis":
                if "spearman_corr" in result:
                    all_metrics["Spearman_Correlation"] = abs(result["spearman_corr"])
                if "spearman_p" in result:
                    all_metrics["Spearman_pval"] = result["spearman_p"]
            
            # Multimodal benchmarks
            elif benchmark_name == "multimodal_integration":
                if "mean_paired_distance" in result:
                    all_metrics["Mean_Paired_Distance"] = result["mean_paired_distance"]
                if "paired_distance_pvalue" in result:
                    all_metrics["Paired_Distance_pval"] = result["paired_distance_pvalue"]
                if "ASW_modality_overall" in result:
                    all_metrics["ASW_Modality"] = result["ASW_modality_overall"]
                if "ASW_pvalue" in result:
                    all_metrics["ASW_Modality_pval"] = result["ASW_pvalue"]
                if "month_preservation_score" in result:
                    all_metrics["Month_Preservation_Score"] = result["month_preservation_score"]
                if "month_preservation_pvalue" in result:
                    all_metrics["Month_Preservation_pval"] = result["month_preservation_pvalue"]
        
        if not all_metrics:
            logger.warning("No metrics collected - nothing to save to summary CSV")
            return
        
        col_name = self.method_name
        
        if summary_csv_path.exists():
            summary_df = pd.read_csv(summary_csv_path, index_col=0)
        else:
            summary_df = pd.DataFrame()
        
        for metric, value in all_metrics.items():
            summary_df.loc[metric, col_name] = value
        
        summary_df.to_csv(summary_csv_path, index_label="Metric")
        logger.info(f"✓ Updated summary CSV at: {summary_csv_path} with column '{col_name}'")

    # ------------------------- Trajectory Benchmark Methods -------------------------

    def run_embedding_visualization(
        self,
        n_components: int = 2,
        figsize: tuple = (8, 6),
        dpi: int = 300,
        **kwargs
    ) -> Dict[str, Any]:
        """Visualize embeddings colored by label_col."""
        logger.info("Running Embedding Visualization...")
        output_dir = self._create_output_dir("embedding_visualization")

        if not self._check_file_exists(self.embedding_csv_path, "Embedding CSV file"):
            return {"status": "error", "message": "Missing or invalid embedding CSV path."}

        try:
            meta_df = pd.read_csv(self.meta_csv_path)
            if 'sample' in meta_df.columns:
                meta_df['sample'] = self._normalize_sample_ids(meta_df['sample'])
            
            embedding_df = pd.read_csv(self.embedding_csv_path, index_col=0)
            embedding_df.index = self._normalize_sample_ids(embedding_df.index)

            required_cols = [self.label_col]
            missing_cols = [c for c in required_cols if c not in meta_df.columns]
            if missing_cols:
                return {"status": "error", "message": f"Missing columns: {missing_cols}"}

            if 'sample' in meta_df.columns:
                meta_df = meta_df.set_index('sample')

            meta_df.index = self._normalize_sample_ids(meta_df.index)
            embedding_df.index = self._normalize_sample_ids(embedding_df.index)
            
            common_ids = meta_df.index.intersection(embedding_df.index)
            if len(common_ids) == 0:
                raise ValueError("No overlapping sample IDs!")

            embedding_df = embedding_df.loc[common_ids]
            meta_df = meta_df.loc[embedding_df.index]

            pca = PCA(n_components=n_components)
            embedding_2d = pca.fit_transform(embedding_df)
            variance_explained = pca.explained_variance_ratio_

            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
            
            label_values_raw = meta_df[self.label_col]
            label_numeric = pd.to_numeric(label_values_raw, errors='coerce')
            is_numerical = label_numeric.notna().sum() / len(label_numeric) > 0.5
            
            if is_numerical:
                scatter = ax.scatter(
                    embedding_2d[:, 0], embedding_2d[:, 1],
                    c=label_numeric, cmap='viridis',
                    edgecolors='black', alpha=0.8, s=100, linewidths=0.5
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(f'{self.label_col}', fontsize=10)
            else:
                unique_labels = sorted(label_values_raw.astype(str).unique().tolist())
                n_unique = len(unique_labels)
                label_to_num = {lbl: i for i, lbl in enumerate(unique_labels)}
                label_colors = [label_to_num[str(lbl)] for lbl in label_values_raw]
                
                scatter = ax.scatter(
                    embedding_2d[:, 0], embedding_2d[:, 1],
                    c=label_colors, cmap='tab20',
                    edgecolors='black', alpha=0.8, s=100, linewidths=0.5
                )
                cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_unique))
                cbar.set_label(f'{self.label_col}', fontsize=10)

            ax.set_xlabel(f'PC1 ({variance_explained[0]:.1%})', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'PC2 ({variance_explained[1]:.1%})', fontsize=12, fontweight='bold')
            ax.set_title(f'Embeddings colored by {self.label_col}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')

            plt.tight_layout()
            output_path = output_dir / 'embedding_overview.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
            plt.close()

            result = {
                "variance_explained": variance_explained.tolist(),
                "n_samples": int(embedding_df.shape[0]),
                "output_plot": str(output_path),
            }

            logger.info(f"✓ Embedding visualization completed.")
            return {"status": "success", "output_dir": str(output_dir), "result": result}

        except Exception as e:
            logger.error(f"✗ Error in embedding visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def run_trajectory_anova(self, pseudotime_col: str = "pseudotime", **kwargs) -> Dict[str, Any]:
        """Run one-way regression ANOVA for pseudotime ~ age."""
        logger.info("Running Trajectory ANOVA Analysis...")
        output_dir = self._create_output_dir("trajectory_anova")

        if not self._check_file_exists(self.pseudotime_csv_path, "Pseudotime CSV file"):
            return {"status": "error", "message": "Missing pseudotime CSV path."}

        try:
            meta_df = pd.read_csv(self.meta_csv_path)
            pseudotime_df = pd.read_csv(self.pseudotime_csv_path)

            if "sample" not in meta_df.columns or "sample" not in pseudotime_df.columns:
                raise ValueError("'sample' column missing")

            if self.label_col not in meta_df.columns:
                raise ValueError(f"Label column '{self.label_col}' not found")

            if pseudotime_col not in pseudotime_df.columns:
                raise ValueError(f"Pseudotime column '{pseudotime_col}' not found")

            meta_df["sample"] = self._normalize_sample_ids(meta_df["sample"])
            pseudotime_df["sample"] = self._normalize_sample_ids(pseudotime_df["sample"])
            
            merged_df = pd.merge(meta_df, pseudotime_df, on="sample", how="inner")
            clean_df = merged_df[["sample", self.label_col, pseudotime_col]].copy()
            clean_df[self.label_col] = pd.to_numeric(clean_df[self.label_col], errors="coerce")
            clean_df = clean_df.dropna(subset=[self.label_col, pseudotime_col])
            
            n_samples = clean_df.shape[0]
            if n_samples < 3:
                raise ValueError(f"Not enough samples for ANOVA (n={n_samples})")

            formula = f"{pseudotime_col} ~ {self.label_col}"
            model = ols(formula, data=clean_df).fit()
            anova_table = anova_lm(model, typ=1)

            effect_row = self.label_col
            if effect_row not in anova_table.index:
                logger.warning(f"Expected row '{effect_row}' not found.")
                possible_rows = [i for i in anova_table.index if self.label_col in i]
                if possible_rows:
                    effect_row = possible_rows[0]

            ss_effect = float(anova_table.loc[effect_row, "sum_sq"])
            ss_resid = float(anova_table.loc["Residual", "sum_sq"])
            partial_eta_sq = ss_effect / (ss_effect + ss_resid) if (ss_effect + ss_resid) > 0 else np.nan

            anova_table["partial_eta_sq"] = np.nan
            anova_table.loc[effect_row, "partial_eta_sq"] = partial_eta_sq

            anova_csv_path = output_dir / "trajectory_anova_table.csv"
            anova_table.to_csv(anova_csv_path)

            result = {
                "anova_table": anova_table,
                "n_samples": int(n_samples),
                "label_col": self.label_col
            }

            logger.info(f"✓ Trajectory ANOVA completed. Eta-sq: {partial_eta_sq:.4f}")
            return {"status": "success", "output_dir": str(output_dir), "result": result}

        except Exception as e:
            logger.error(f"✗ Error in trajectory ANOVA: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def run_trajectory_analysis(self, pseudotime_col: str = "pseudotime", **kwargs) -> Dict[str, Any]:
        """Run trajectory analysis (Spearman correlation)."""
        logger.info("Running Trajectory Analysis...")
        output_dir = self._create_output_dir("trajectory_analysis")

        if not self._check_file_exists(self.pseudotime_csv_path, "Pseudotime CSV file"):
            return {"status": "error", "message": "Missing pseudotime CSV path."}

        try:
            raw_result = run_trajectory_analysis(
                meta_csv_path=str(self.meta_csv_path),
                pseudotime_csv_path=str(self.pseudotime_csv_path),
                output_dir_path=str(output_dir),
                severity_col=self.label_col,
                pseudotime_col=pseudotime_col,
                **kwargs,
            )

            if isinstance(raw_result, dict) and "spearman_corr" in raw_result:
                raw_corr = raw_result["spearman_corr"]
                abs_corr = abs(raw_corr) if raw_corr is not None else None
                raw_result["spearman_corr_raw"] = raw_corr
                raw_result["spearman_corr"] = abs_corr

            logger.info(f"✓ Trajectory analysis completed.")
            return {"status": "success", "output_dir": str(output_dir), "result": raw_result}
        except Exception as e:
            logger.error(f"✗ Error in trajectory analysis: {e}")
            return {"status": "error", "message": str(e)}

    # ------------------------- Multimodal Integration Benchmark -------------------------

    def run_multimodal_integration(self, **kwargs) -> Dict[str, Any]:
        """Run multimodal integration benchmark treating month as categorical."""
        logger.info("Running Multimodal Integration Benchmark...")
        output_dir = self._create_output_dir("multimodal_integration")

        if not self._check_file_exists(self.embedding_csv_path, "Embedding CSV file"):
            return {"status": "error", "message": "Missing embedding CSV path."}

        try:
            # Load data
            meta_df = pd.read_csv(self.meta_csv_path)
            if 'sample' in meta_df.columns:
                meta_df = meta_df.set_index('sample')
            
            embedding_df = pd.read_csv(self.embedding_csv_path, index_col=0)
            
            # Parse sample names
            sample_info = parse_sample_names(embedding_df.index.values)
            
            # Align
            meta_df.index = self._normalize_sample_ids(meta_df.index)
            embedding_df.index = self._normalize_sample_ids(embedding_df.index)
            sample_info.index = self._normalize_sample_ids(sample_info.index)
            
            common = meta_df.index.intersection(embedding_df.index).intersection(sample_info.index)
            if len(common) == 0:
                raise ValueError("No overlapping sample IDs!")
            
            common_sorted = sorted(common)
            meta_aligned = meta_df.loc[common_sorted].copy()
            emb_aligned = embedding_df.loc[common_sorted].copy()
            sample_info_aligned = sample_info.loc[common_sorted].copy()
            
            emb_array = emb_aligned.values.astype(float)
            
            logger.info(f"Aligned data: {len(meta_aligned)} samples, {emb_array.shape[1]} dimensions")
            
            # Compute metrics
            logger.info("Computing paired sample distances...")
            paired_results = compute_paired_distance(
                sample_info_aligned, emb_array,
                metric=self.multimodal_config.distance_metric,
            )
            
            logger.info("Computing modality mixing...")
            mixing_results = compute_modality_mixing(
                sample_info_aligned, emb_array,
                k=self.multimodal_config.k_neighbors,
                include_self=self.multimodal_config.include_self,
            )
            
            logger.info("Computing month preservation...")
            month_results = compute_month_preservation(
                meta_aligned, emb_array,
                month_col=self.month_col,
                metric=self.multimodal_config.distance_metric,
            )
            
            # Permutation tests
            logger.info("Running permutation tests...")
            paired_perm = permutation_test_paired_distance(
                emb_array,
                paired_results['mean_paired_distance'],
                paired_results['paired_indices'],
                n_permutations=self.multimodal_config.n_permutations,
                metric=self.multimodal_config.distance_metric,
                random_seed=self.multimodal_config.random_seed,
            )
            
            mixing_perm = permutation_test_modality_mixing(
                sample_info_aligned, emb_array,
                mixing_results['iLISI_norm_mean'],
                mixing_results['ASW_modality_overall'],
                k=self.multimodal_config.k_neighbors,
                n_permutations=self.multimodal_config.n_permutations,
                random_seed=self.multimodal_config.random_seed,
            )
            
            month_perm = permutation_test_month_preservation(
                meta_aligned, emb_array,
                month_results['month_preservation_score'],
                month_col=self.month_col,
                n_permutations=self.multimodal_config.n_permutations,
                metric=self.multimodal_config.distance_metric,
                random_seed=self.multimodal_config.random_seed,
            )
            
            # Visualizations
            logger.info("Creating visualizations...")
            emb_2d = reduce_to_2d(emb_array)
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            modality_path = viz_dir / f"{self.method_name}_by_modality.png"
            plot_embedding_by_modality(
                emb_2d, sample_info_aligned, str(modality_path), self.multimodal_config
            )
            
            month_path = viz_dir / f"{self.method_name}_by_month.png"
            plot_embedding_by_month(
                emb_2d, meta_aligned, str(month_path), self.multimodal_config, month_col=self.month_col
            )
            
            paired_path = viz_dir / f"{self.method_name}_paired_connections.png"
            plot_paired_connections(
                emb_2d, sample_info_aligned, str(paired_path), self.multimodal_config
            )
            
            # Save per-sample metrics
            per_sample_df = pd.DataFrame({
                "sample": meta_aligned.index,
                "sample_id": sample_info_aligned['sample_id'].values,
                "modality": sample_info_aligned['modality'].values,
                "month": meta_aligned[self.month_col].values,
                "iLISI": mixing_results["iLISI_per_sample"],
                "ASW_modality": mixing_results["ASW_per_sample"],
            }).set_index("sample")
            
            per_sample_path = output_dir / "per_sample_metrics.csv"
            per_sample_df.to_csv(per_sample_path)
            
            # Save paired details
            if paired_results["paired_details"]:
                paired_df = pd.DataFrame(paired_results["paired_details"])
                paired_details_path = output_dir / "paired_sample_distances.csv"
                paired_df.to_csv(paired_details_path, index=False)
            
            # Save month details
            month_details_df = pd.DataFrame(month_results["month_details"]).T
            month_details_df.index.name = "month"
            month_details_path = output_dir / "month_details.csv"
            month_details_df.to_csv(month_details_path)
            
            # Summary
            summary_lines = [
                "=" * 60,
                f"Multimodal Integration: {self.method_name}",
                "=" * 60,
                "",
                f"Total samples: {len(meta_aligned)}",
                f"Embedding dimensions: {emb_array.shape[1]}",
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
                "3. Month Preservation Score (higher = better)",
                f"   Value: {month_results['month_preservation_score']:.4f}",
                f"   P-value: {month_perm['p_value']:.2e}",
                "",
                "=" * 60,
            ]
            
            summary_text = "\n".join(summary_lines)
            logger.info("\n" + summary_text)
            
            summary_path = output_dir / "integration_summary.txt"
            summary_path.write_text(summary_text, encoding="utf-8")
            
            result = {
                "n_samples": len(meta_aligned),
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
                "month_preservation_score": month_results["month_preservation_score"],
                "month_preservation_pvalue": month_perm["p_value"],
            }
            
            logger.info(f"✓ Multimodal integration benchmark completed.")
            return {"status": "success", "output_dir": str(output_dir), "result": result}

        except Exception as e:
            logger.error(f"✗ Error in multimodal integration: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    # ------------------------- Orchestration -------------------------

    def run_all_benchmarks(
        self,
        skip_benchmarks: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """Run all benchmark analyses."""
        skip_benchmarks = skip_benchmarks or []
        results: Dict[str, Dict[str, Any]] = {}

        benchmark_methods = {
            "embedding_visualization": self.run_embedding_visualization,
            "trajectory_anova": self.run_trajectory_anova,
            "trajectory_analysis": self.run_trajectory_analysis,
            "multimodal_integration": self.run_multimodal_integration,
        }

        for name, method in benchmark_methods.items():
            if name in skip_benchmarks:
                continue

            logger.info(f"Running: {name}")
            method_kwargs = kwargs.get(name, {})
            results[name] = method(**method_kwargs)
            
        self._save_summary_csv(results)
        return results


# ------------------------- Convenience Function -------------------------

def run_benchmarks(
    meta_csv_path: str,
    embedding_csv_path: str,
    method_name: str = "method",
    label_col: str = "age",
    month_col: str = "month",
    pseudotime_csv_path: Optional[str] = None,
    benchmarks_to_run: Optional[List[str]] = None,
    output_base_dir: Optional[str] = None,
    summary_csv_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """Convenience function to run benchmarks."""
    try:
        wrapper = BenchmarkWrapper(
            meta_csv_path=meta_csv_path,
            embedding_csv_path=embedding_csv_path,
            method_name=method_name,
            label_col=label_col,
            month_col=month_col,
            pseudotime_csv_path=pseudotime_csv_path,
            output_base_dir=output_base_dir,
            summary_csv_path=summary_csv_path,
        )

        if benchmarks_to_run:
            all_benchmarks = ["embedding_visualization", "trajectory_anova", 
                            "trajectory_analysis", "multimodal_integration"]
            skip_benchmarks = [b for b in all_benchmarks if b not in benchmarks_to_run]
            return wrapper.run_all_benchmarks(skip_benchmarks=skip_benchmarks, **kwargs)
        else:
            return wrapper.run_all_benchmarks(**kwargs)

    except Exception as e:
        logger.error(f"✗ Failed to initialize BenchmarkWrapper: {e}")
        return {"initialization_error": {"status": "error", "message": str(e)}}


# ------------------------- Usage Examples -------------------------

if __name__ == "__main__":
    
    # Base paths
    base_dir = '/dcs07/hongkai/data/harry/result/Benchmark_long_covid'
    meta_csv_path = '/dcl01/hongkai/data/data/hjiang/Data/long_covid/sample_meta.csv'
    summary_csv_path = f'{base_dir}/benchmark_summary_long_covid.csv'
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK SUITE - Long COVID Dataset")
    print("=" * 80)
    
    # Common parameters for all methods
    common_params = {
        "meta_csv_path": meta_csv_path,
        "summary_csv_path": summary_csv_path,
        "label_col": "month",
        "month_col": "month",
        "embedding_visualization": {"dpi": 300, "figsize": (8, 6)},
    }
    
    # List of method configurations
    methods = [
        ("SD_expression", f'{base_dir}/../long_covid/rna/embeddings/sample_expression_embedding.csv', f'{base_dir}/../long_covid/rna/CCA/pseudotime_expression.csv', f'{base_dir}/../long_covid/rna'),
        ("SD_proportion", f'{base_dir}/../long_covid/rna/embeddings/sample_proportion_embedding.csv', f'{base_dir}/../long_covid/rna/CCA/pseudotime_proportion.csv', f'{base_dir}/../long_covid/rna'),
        ("GEDI", f'{base_dir}/GEDI/gedi_sample_embedding.csv', f'{base_dir}/GEDI/trajectory/pseudotime_results.csv', f'{base_dir}/GEDI'),
        ("Gloscope", f'{base_dir}/Gloscope/knn_divergence_mds_10d.csv', f'{base_dir}/Gloscope/trajectory/pseudotime_results.csv', f'{base_dir}/Gloscope'),
        ("MFA", f'{base_dir}/MFA/sample_embeddings.csv', f'{base_dir}/MFA/trajectory/pseudotime_results.csv', f'{base_dir}/MFA'),
        ("pseudobulk", f'{base_dir}/pseudobulk/pseudobulk/pca_embeddings.csv', f'{base_dir}/pseudobulk/pseudobulk/trajectory/pseudotime_results.csv', f'{base_dir}/pseudobulk'),
        ("pilot", f'{base_dir}/pilot/wasserstein_distance_mds_10d.csv', f'{base_dir}/pilot/trajectory/pseudotime_results.csv', f'{base_dir}/pilot'),
        ("QOT", f'{base_dir}/QOT/35_qot_distance_matrix_mds_10d.csv', f'{base_dir}/QOT/trajectory/pseudotime_results.csv', f'{base_dir}/QOT'),
        ("scPoli", f'{base_dir}/scPoli/sample_embeddings_full.csv', f'{base_dir}/scPoli/trajectory/pseudotime_results.csv', f'{base_dir}/scPoli'),
        ("MUSTARD", f'{base_dir}/mustard/sample_embedding.csv', f'{base_dir}/mustard/trajectory/pseudotime_results.csv', f'{base_dir}/mustard')
    ]

    for i, (m_name, m_emb, m_pseudo, m_out) in enumerate(methods):
        print(f"\n[{i+1}/{len(methods)}] Running {m_name}...")
        run_benchmarks(
            embedding_csv_path=m_emb,
            pseudotime_csv_path=m_pseudo,
            method_name=m_name,
            output_base_dir=m_out,
            **common_params
        )
    
    print("\n" + "=" * 80)
    print("ALL BENCHMARKS COMPLETED!")
    print(f"Summary CSV saved to: {summary_csv_path}")
    print("=" * 80 + "\n")