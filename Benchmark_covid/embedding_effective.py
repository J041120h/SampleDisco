#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARI-based Evaluation of Embedding/Distance Quality for Disease Severity Clustering.

This script evaluates how well KNN-based clustering aligns with known disease severity levels
using Adjusted Rand Index (ARI) as the primary metric.

Usage: Edit the function call in main() with your file paths and run the script.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns


def read_metadata(meta_csv: str) -> pd.DataFrame:
    """Read metadata CSV containing sample IDs and severity levels."""
    md = pd.read_csv(meta_csv)
    md.columns = [c.lower().strip() for c in md.columns]
    
    # Check for required columns
    required = {"sample", "sev.level"}
    missing = required - set(md.columns)
    if missing:
        # Try alternative column names
        if "sev_level" in md.columns:
            md["sev.level"] = md["sev_level"]
        elif "severity" in md.columns:
            md["sev.level"] = md["severity"]
        elif "severity.level" in md.columns:
            md["sev.level"] = md["severity.level"]
        else:
            raise ValueError(f"Metadata must contain columns {sorted(required)}; missing: {sorted(missing)}")
    
    # Validate sample uniqueness
    if md["sample"].duplicated().any():
        dups = md.loc[md["sample"].duplicated(), "sample"].unique()
        raise ValueError(f"'sample' IDs must be unique. Duplicates: {dups[:10]}")
    
    # Validate severity levels
    sev_levels = md["sev.level"].dropna().unique()
    print(f"Found severity levels: {sorted(sev_levels)}", file=sys.stderr)
    
    return md


def read_embedding_or_distance(data_csv: str, mode: str) -> Tuple[pd.DataFrame, bool]:
    """Read embedding matrix or distance matrix from CSV."""
    df = pd.read_csv(data_csv, index_col=0)
    
    if mode == "distance":
        # Validate distance matrix
        if df.shape[0] != df.shape[1]:
            raise ValueError(f"Distance matrix must be square, got shape {df.shape}")
        
        # Ensure zero diagonal and symmetry
        vals = df.values.astype(float)
        np.fill_diagonal(vals, 0.0)
        
        if not np.allclose(vals, vals.T, equal_nan=True, rtol=1e-5, atol=1e-8):
            print("Warning: distance matrix not exactly symmetric; symmetrizing.", file=sys.stderr)
            vals = (vals + vals.T) / 2.0
            np.fill_diagonal(vals, 0.0)
        
        df.iloc[:, :] = vals
        return df, True
        
    elif mode == "embedding":
        if df.shape[1] < 1:
            raise ValueError(f"Embedding must have ≥1 dimension, got {df.shape[1]}")
        print(f"Loaded embedding with shape {df.shape} (samples × dimensions)", file=sys.stderr)
        return df, False
    else:
        raise ValueError(f"mode must be 'embedding' or 'distance', got '{mode}'")


def align_by_samples(md: pd.DataFrame, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align metadata and data by common sample IDs."""
    if md.index.name != "sample":
        md = md.set_index("sample", drop=False)
    
    common = md.index.intersection(data.index)
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between metadata and data.")
    
    if len(common) < len(md):
        print(f"Note: dropping {len(md) - len(common)} metadata rows without data match.", file=sys.stderr)
    if len(common) < len(data):
        print(f"Note: dropping {len(data) - len(common)} data rows without metadata.", file=sys.stderr)
    
    # Keep same order
    common_sorted = sorted(common)
    md2 = md.loc[common_sorted].copy()
    data2 = data.loc[common_sorted].copy()
    
    return md2, data2


def knn_clustering(data: np.ndarray, n_clusters: int, k: int = 10, is_distance: bool = False) -> np.ndarray:
    """
    Perform KNN-based clustering using connectivity graph.
    
    Parameters
    ----------
    data : np.ndarray
        Either embedding matrix (n_samples × n_features) or distance matrix (n_samples × n_samples)
    n_clusters : int
        Number of clusters to form
    k : int
        Number of neighbors for KNN graph
    is_distance : bool
        Whether data is a distance matrix
        
    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample
    """
    n_samples = data.shape[0]
    k_eff = min(k, n_samples - 1)
    
    # Build KNN connectivity matrix
    if is_distance:
        # Use distance matrix to find k nearest neighbors
        knn_idx = np.argsort(data, axis=1)[:, 1:k_eff+1]  # exclude self
        connectivity = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            connectivity[i, knn_idx[i]] = 1
        connectivity = (connectivity + connectivity.T) / 2.0
    else:
        # Use embedding to find k nearest neighbors
        nn = NearestNeighbors(n_neighbors=k_eff + 1, metric='euclidean')
        nn.fit(data)
        connectivity = nn.kneighbors_graph(data, mode='connectivity')
        connectivity = connectivity.toarray()
    
    # Perform clustering using the connectivity constraint
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        connectivity=connectivity,
        linkage='average'
    )
    
    if is_distance:
        # For distance matrix, use precomputed distance in clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(data)
    else:
        labels = clustering.fit_predict(data)
    
    return labels


def compute_ari_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute ARI and other clustering metrics.
    
    Parameters
    ----------
    true_labels : np.ndarray
        Ground truth labels (severity levels)
    pred_labels : np.ndarray
        Predicted cluster labels
        
    Returns
    -------
    metrics : dict
        Dictionary containing ARI, NMI, and other metrics
    """
    # Filter out any NaN values
    mask = ~(pd.isna(true_labels) | pd.isna(pred_labels))
    true_filtered = true_labels[mask]
    pred_filtered = pred_labels[mask]
    
    if len(true_filtered) < 2:
        raise ValueError("Need at least 2 samples with valid labels for ARI calculation")
    
    # Calculate metrics
    ari = adjusted_rand_score(true_filtered, pred_filtered)
    nmi = normalized_mutual_info_score(true_filtered, pred_filtered)
    
    # Calculate purity for each cluster
    n_clusters = len(np.unique(pred_filtered))
    n_true_classes = len(np.unique(true_filtered))
    
    purity_scores = []
    for cluster_id in np.unique(pred_filtered):
        cluster_mask = pred_filtered == cluster_id
        cluster_true_labels = true_filtered[cluster_mask]
        if len(cluster_true_labels) > 0:
            most_common = np.bincount(cluster_true_labels.astype(int)).argmax()
            purity = np.sum(cluster_true_labels == most_common) / len(cluster_true_labels)
            purity_scores.append(purity)
    
    avg_purity = np.mean(purity_scores) if purity_scores else 0.0
    
    return {
        'ari': ari,
        'nmi': nmi,
        'avg_purity': avg_purity,
        'n_clusters': n_clusters,
        'n_true_classes': n_true_classes,
        'n_samples': len(true_filtered)
    }


def plot_clustering_results(
    data: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    metrics: Dict[str, float],
    output_path: Path,
    is_distance: bool = False
) -> None:
    """Generate visualization of clustering results."""
    
    # Use PCA for visualization if high-dimensional
    if not is_distance and data.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        explained_var = pca.explained_variance_ratio_
    elif not is_distance and data.shape[1] == 2:
        data_2d = data
        explained_var = [1.0, 0.0]
    else:
        # Use MDS for distance matrix
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        data_2d = mds.fit_transform(data)
        explained_var = [0.5, 0.5]  # MDS doesn't provide explained variance
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot true labels (severity levels)
    scatter1 = axes[0].scatter(data_2d[:, 0], data_2d[:, 1], 
                               c=true_labels, cmap='viridis', 
                               alpha=0.7, s=50)
    axes[0].set_title(f'True Severity Levels\n(1=healthy, 4=most severe)')
    axes[0].set_xlabel(f'Dim 1 ({explained_var[0]:.1%} var)' if not is_distance else 'MDS 1')
    axes[0].set_ylabel(f'Dim 2 ({explained_var[1]:.1%} var)' if not is_distance else 'MDS 2')
    plt.colorbar(scatter1, ax=axes[0], label='Severity')
    
    # Plot predicted clusters
    scatter2 = axes[1].scatter(data_2d[:, 0], data_2d[:, 1], 
                               c=pred_labels, cmap='tab10', 
                               alpha=0.7, s=50)
    axes[1].set_title(f'KNN Clusters\n(ARI={metrics["ari"]:.3f})')
    axes[1].set_xlabel(f'Dim 1 ({explained_var[0]:.1%} var)' if not is_distance else 'MDS 1')
    axes[1].set_ylabel(f'Dim 2 ({explained_var[1]:.1%} var)' if not is_distance else 'MDS 2')
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')
    
    plt.tight_layout()
    plt.savefig(output_path / 'clustering_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Cluster')
    ax.set_ylabel('True Severity Level')
    ax.set_title(f'Confusion Matrix\nARI={metrics["ari"]:.3f}, NMI={metrics["nmi"]:.3f}')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_ari_clustering(
    meta_csv: str,
    data_csv: str,
    mode: str,
    outdir: str,
    k_neighbors: int = 15,
    n_clusters: Optional[int] = None,
    create_plots: bool = True
) -> Dict[str, object]:
    """
    Evaluate embedding/distance quality using ARI between KNN clusters and severity levels.
    
    Parameters
    ----------
    meta_csv : str
        Path to metadata CSV with 'sample' and 'sev.level' columns
    data_csv : str
        Path to data CSV (embedding: samples×dims, distance: samples×samples)
    mode : str
        'embedding' or 'distance'
    outdir : str
        Output directory for results
    k_neighbors : int, default=15
        Number of neighbors for KNN clustering
    n_clusters : int, optional
        Number of clusters (defaults to number of unique severity levels)
    create_plots : bool, default=True
        Whether to generate visualization plots
    
    Returns
    -------
    results : dict
        Dictionary containing evaluation metrics and output paths
    """
    # Setup output directory
    import os
    os.makedirs(output_dir_path, exist_ok=True)
    output_dir_path = os.path.join(output_dir_path, 'ARI_Clustering_Evaluation')
    os.makedirs(output_dir_path, exist_ok=True)
    
    # Read and align data
    print("Reading metadata and data files...", file=sys.stderr)
    md = read_metadata(meta_csv)
    data_df, is_distance = read_embedding_or_distance(data_csv, mode)
    md_aln, data_aln = align_by_samples(md, data_df)
    
    # Extract severity labels
    severity_labels = md_aln["sev.level"].values
    unique_severities = np.unique(severity_labels[~pd.isna(severity_labels)])
    n_severity_levels = len(unique_severities)
    
    if n_clusters is None:
        n_clusters = n_severity_levels
    
    print(f"Found {n_severity_levels} unique severity levels: {unique_severities}", file=sys.stderr)
    print(f"Using {n_clusters} clusters for KNN clustering", file=sys.stderr)
    
    # Prepare data array
    if is_distance:
        data_array = data_aln.values.astype(float)
    else:
        data_array = data_aln.values.astype(float)
    
    n_samples = data_array.shape[0]
    k_eff = min(k_neighbors, n_samples - 1)
    
    # Perform KNN-based clustering
    print(f"Performing KNN clustering with k={k_eff}...", file=sys.stderr)
    cluster_labels = knn_clustering(
        data_array, 
        n_clusters=n_clusters, 
        k=k_eff, 
        is_distance=is_distance
    )
    
    # Map severity levels to integers (1=healthy, 4=most severe)
    severity_mapping = {v: i for i, v in enumerate(sorted(unique_severities))}
    severity_labels_int = np.array([severity_mapping[v] if not pd.isna(v) else -1 
                                    for v in severity_labels])
    
    # Compute metrics
    print("Computing ARI and other metrics...", file=sys.stderr)
    valid_mask = severity_labels_int >= 0
    metrics = compute_ari_metrics(
        severity_labels_int[valid_mask],
        cluster_labels[valid_mask]
    )
    
    # Create per-sample results
    per_sample_df = pd.DataFrame({
        'sample': md_aln.index.astype(str),
        'severity_level': severity_labels,
        'knn_cluster': cluster_labels,
        'severity_int': severity_labels_int
    }).set_index('sample')
    
    per_sample_path = outdir_p / 'per_sample_clusters.csv'
    per_sample_df.to_csv(per_sample_path)
    
    # Generate plots
    if create_plots:
        print("Generating visualizations...", file=sys.stderr)
        plot_clustering_results(
            data_array[valid_mask],
            severity_labels_int[valid_mask],
            cluster_labels[valid_mask],
            metrics,
            outdir_p,
            is_distance=is_distance
        )
    
    # Create summary
    summary_lines = [
        "=" * 60,
        "ARI-based Clustering Evaluation Summary",
        "=" * 60,
        f"Data mode: {mode}",
        f"Total samples: {n_samples}",
        f"Samples with valid severity: {metrics['n_samples']}",
        f"Unique severity levels: {n_severity_levels} -> {list(unique_severities)}",
        f"Number of clusters used: {n_clusters}",
        f"K neighbors for KNN: {k_eff}",
        "",
        "METRICS:",
        "-" * 30,
        f"Adjusted Rand Index (ARI): {metrics['ari']:.4f}",
        f"Normalized Mutual Info (NMI): {metrics['nmi']:.4f}",
        f"Average Cluster Purity: {metrics['avg_purity']:.4f}",
        "",
        "INTERPRETATION:",
        "-" * 30,
    ]
    
    # Add interpretation
    if metrics['ari'] > 0.7:
        interpretation = "EXCELLENT: Strong agreement between clusters and severity levels"
    elif metrics['ari'] > 0.5:
        interpretation = "GOOD: Moderate agreement between clusters and severity levels"
    elif metrics['ari'] > 0.3:
        interpretation = "FAIR: Weak agreement between clusters and severity levels"
    elif metrics['ari'] > 0.1:
        interpretation = "POOR: Very weak agreement between clusters and severity levels"
    else:
        interpretation = "VERY POOR: Almost no agreement between clusters and severity levels"
    
    summary_lines.append(interpretation)
    summary_lines.append("")
    summary_lines.append(f"Results saved to: {outdir_p}")
    summary_lines.append("=" * 60)
    
    # Save summary
    summary_path = outdir_p / 'ari_evaluation_summary.txt'
    summary_path.write_text("\n".join(summary_lines), encoding='utf-8')
    print("\n".join(summary_lines))
    
    # Save detailed metrics as JSON for programmatic access
    import json
    metrics_json = {
        **metrics,
        'k_neighbors': k_eff,
        'unique_severities': unique_severities.tolist(),
        'interpretation': interpretation
    }
    metrics_path = outdir_p / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    return {
        'metrics': metrics,
        'interpretation': interpretation,
        'n_samples': n_samples,
        'severity_levels': list(unique_severities),
        'per_sample_path': str(per_sample_path),
        'summary_path': str(summary_path),
        'metrics_json_path': str(metrics_path)
    }


if __name__ == "__main__":

    evaluate_ari_clustering(
        meta_csv="/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv",
        data_csv="/dcs07/hongkai/data/harry/result/Benchmark/covid_25_sample/rna/Sample_distance/cosine/expression_DR_distance/expression_DR_coordinates.csv",
        mode="embedding",  # Use "embedding" or "distance"
        outdir="/dcs07/hongkai/data/harry/result/Benchmark/covid_25_sample",
        k_neighbors=15,  # Number of neighbors for KNN
        n_clusters=None,  # None = auto-detect from severity levels, or specify number
        create_plots=True  # Generate visualizations
    )