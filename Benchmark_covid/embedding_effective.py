#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KMeans-based Evaluation of Embedding Quality for Disease Severity Clustering.

This script evaluates how well KMeans clustering aligns with known disease severity levels
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


def _load_csv_robust(filepath: str, index_col: Optional[int] = None) -> pd.DataFrame:
    """
    Load CSV with robust handling of encoding issues (BOM, etc.) and NaN sample IDs.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    index_col : int, optional
        Column to use as index
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe with NaN sample IDs removed (if 'sample' column exists)
    """
    # Try reading with UTF-8-sig first (handles BOM)
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig', index_col=index_col)
    except Exception:
        # Fallback to default encoding
        df = pd.read_csv(filepath, index_col=index_col)
    
    # If this is metadata (has 'sample' column), remove rows where sample is NaN or empty
    if 'sample' in df.columns:
        n_before = len(df)
        # Remove rows with NaN sample IDs
        df = df[df['sample'].notna()].copy()
        # Also remove rows with empty string sample IDs
        df = df[df['sample'].astype(str).str.strip() != ''].copy()
        n_after = len(df)
        if n_before != n_after:
            print(f"[INFO] Removed {n_before - n_after} rows with missing/empty sample IDs from metadata", file=sys.stderr)
    
    return df


def read_metadata(meta_csv: str, label_col: str = "tissue") -> pd.DataFrame:
    """
    Read metadata CSV containing sample IDs and labels for clustering.
    
    Parameters
    ----------
    meta_csv : str
        Path to metadata CSV file
    label_col : str
        Name of the label column to use (e.g., 'tissue', 'sev.level', 'cell_type')
    
    Returns
    -------
    pd.DataFrame
        Metadata with 'sample' and specified label column
    """
    # Use robust CSV loading
    md = _load_csv_robust(meta_csv)
    md.columns = [c.lower().strip() for c in md.columns]
    
    # Check for required columns
    required = {"sample", label_col.lower()}
    missing = required - set(md.columns)
    if missing:
        raise ValueError(f"Metadata must contain columns {sorted(required)}; missing: {sorted(missing)}")
    
    # Normalize sample IDs to lowercase for case-insensitive matching
    md["sample"] = md["sample"].astype(str).str.lower().str.strip()
    
    # Validate sample uniqueness (after removing NaN/empty)
    if md["sample"].duplicated().any():
        dups = md.loc[md["sample"].duplicated(), "sample"].unique()
        raise ValueError(f"'sample' IDs must be unique. Duplicates: {list(dups[:10])}")
    
    # Validate labels
    label_values = md[label_col].dropna().unique()
    print(f"Found {len(label_values)} unique values in '{label_col}': {sorted(label_values)}", file=sys.stderr)
    
    return md


def read_embedding_or_distance(data_csv: str, mode: str) -> Tuple[pd.DataFrame, bool]:
    """
    Read embedding matrix (only) from CSV.

    Parameters
    ----------
    data_csv : str
        Path to data CSV
    mode : str
        Must be 'embedding'. Distance mode is not supported in this version.

    Returns
    -------
    df : pd.DataFrame
        Embedding matrix (samples × dimensions)
    is_distance : bool
        Always False in this version
    """
    if mode != "embedding":
        raise ValueError(f"This script now only supports mode='embedding', got mode='{mode}'")
    
    # Use robust CSV loading
    df = _load_csv_robust(data_csv, index_col=0)
    if df.shape[1] < 1:
        raise ValueError(f"Embedding must have ≥1 dimension, got {df.shape[1]}")
    print(f"Loaded embedding with shape {df.shape} (samples × dimensions)", file=sys.stderr)
    return df, False


def align_by_samples(md: pd.DataFrame, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align metadata and data by common sample IDs (case-insensitive)."""
    if md.index.name != "sample":
        md = md.set_index("sample", drop=False)
    
    # NORMALIZE CASE: Convert both indices to lowercase for case-insensitive matching
    print("Normalizing sample IDs to lowercase for case-insensitive matching...", file=sys.stderr)
    md.index = md.index.astype(str).str.lower().str.strip()
    data.index = data.index.astype(str).str.lower().str.strip()
    
    common = md.index.intersection(data.index)
    if len(common) == 0:
        # Provide helpful diagnostics
        print("\n" + "="*60, file=sys.stderr)
        print("ERROR: No overlapping sample IDs!", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(f"Metadata samples (first 10): {list(md.index[:10])}", file=sys.stderr)
        print(f"Data samples (first 10): {list(data.index[:10])}", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)
        raise ValueError("No overlapping sample IDs between metadata and data.")
    
    if len(common) < len(md):
        print(f"Note: dropping {len(md) - len(common)} metadata rows without data match.", file=sys.stderr)
    if len(common) < len(data):
        print(f"Note: dropping {len(data) - len(common)} data rows without metadata.", file=sys.stderr)
    
    # Keep same order
    common_sorted = sorted(common)
    md2 = md.loc[common_sorted].copy()
    data2 = data.loc[common_sorted].copy()
    
    print(f"Successfully aligned {len(common)} samples.", file=sys.stderr)
    
    return md2, data2


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
    is_distance: bool = False,
    label_name: str = "Label"
) -> None:
    """
    Generate visualization of clustering results.
    
    Parameters
    ----------
    data : np.ndarray
        Data array for visualization
    true_labels : np.ndarray
        True labels (ground truth)
    pred_labels : np.ndarray
        Predicted cluster labels
    metrics : dict
        Metrics dictionary
    output_path : Path
        Output directory path
    is_distance : bool
        Whether data is a distance matrix (unused here; assumed False)
    label_name : str
        Name of the label column for plot titles
    """
    
    # Use PCA for visualization if high-dimensional
    if data.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        explained_var = pca.explained_variance_ratio_
    elif data.shape[1] == 2:
        data_2d = data
        explained_var = [1.0, 0.0]
    else:
        # 1D embedding: pad with zeros
        data_2d = np.column_stack([data[:, 0], np.zeros_like(data[:, 0])])
        explained_var = [1.0, 0.0]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot true labels
    scatter1 = axes[0].scatter(
        data_2d[:, 0], data_2d[:, 1],
        c=true_labels, cmap='viridis',
        alpha=0.7, s=50
    )
    axes[0].set_title(f'True {label_name}')
    axes[0].set_xlabel(f'Dim 1 ({explained_var[0]:.1%} var)')
    axes[0].set_ylabel(f'Dim 2 ({explained_var[1]:.1%} var)')
    plt.colorbar(scatter1, ax=axes[0], label=label_name)
    
    # Plot predicted clusters
    scatter2 = axes[1].scatter(
        data_2d[:, 0], data_2d[:, 1],
        c=pred_labels, cmap='tab10',
        alpha=0.7, s=50
    )
    axes[1].set_title(f'KMeans Clusters\n(ARI={metrics["ari"]:.3f})')
    axes[1].set_xlabel(f'Dim 1 ({explained_var[0]:.1%} var)')
    axes[1].set_ylabel(f'Dim 2 ({explained_var[1]:.1%} var)')
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
    ax.set_ylabel(f'True {label_name}')
    ax.set_title(f'Confusion Matrix\nARI={metrics["ari"]:.3f}, NMI={metrics["nmi"]:.3f}')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_ari_clustering(
    meta_csv: str,
    data_csv: str,
    mode: str,
    outdir: str,
    label_col: str = "tissue",
    k_neighbors: int = 15,  # kept for API compatibility; unused in KMeans
    n_clusters: Optional[int] = None,
    create_plots: bool = True
) -> Dict[str, object]:
    """
    Evaluate embedding quality using ARI between KMeans clusters and true labels.
    
    Parameters
    ----------
    meta_csv : str
        Path to metadata CSV with 'sample' and label column
    data_csv : str
        Path to data CSV (embedding: samples×dims)
    mode : str
        Must be 'embedding' in this version
    outdir : str
        Output directory for results
    label_col : str, default='tissue'
        Name of the label column in metadata (e.g., 'tissue', 'sev.level', 'cell_type')
    k_neighbors : int, default=15
        Kept for backward compatibility; not used in KMeans version
    n_clusters : int, optional
        Number of clusters (defaults to number of unique label values)
    create_plots : bool, default=True
        Whether to generate visualization plots
    
    Returns
    -------
    results : dict
        Dictionary containing evaluation metrics and output paths
    """
    import os
    from pathlib import Path
    
    # Use 'outdir' parameter correctly
    os.makedirs(outdir, exist_ok=True)
    output_dir_path = os.path.join(outdir, 'ARI_Clustering_Evaluation')
    os.makedirs(output_dir_path, exist_ok=True)
    
    # Define outdir_p as Path object
    outdir_p = Path(output_dir_path)
    
    # Read and align data
    print("Reading metadata and data files...", file=sys.stderr)
    md = read_metadata(meta_csv, label_col=label_col)
    data_df, is_distance = read_embedding_or_distance(data_csv, mode)
    md_aln, data_aln = align_by_samples(md, data_df)
    
    # Extract labels
    true_labels = md_aln[label_col].values
    unique_labels = np.unique(true_labels[~pd.isna(true_labels)])
    n_unique_labels = len(unique_labels)
    
    if n_clusters is None:
        n_clusters = n_unique_labels
    
    print(f"Found {n_unique_labels} unique {label_col} values: {unique_labels}", file=sys.stderr)
    print(f"Using {n_clusters} clusters for KMeans clustering", file=sys.stderr)
    
    # Prepare data array (embedding)
    data_array = data_aln.values.astype(float)
    n_samples = data_array.shape[0]
    
    # Perform KMeans clustering
    print(f"Performing KMeans clustering with k={n_clusters}...", file=sys.stderr)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(data_array)
    
    # Map label values to integers
    label_mapping = {v: i for i, v in enumerate(sorted(unique_labels))}
    true_labels_int = np.array([
        label_mapping[v] if not pd.isna(v) else -1
        for v in true_labels
    ])
    
    # Compute metrics
    print("Computing ARI and other metrics...", file=sys.stderr)
    valid_mask = true_labels_int >= 0
    metrics = compute_ari_metrics(
        true_labels_int[valid_mask],
        cluster_labels[valid_mask]
    )
    
    # Create per-sample results
    per_sample_df = pd.DataFrame({
        'sample': md_aln.index.astype(str),
        f'{label_col}': true_labels,
        'kmeans_cluster': cluster_labels,
        f'{label_col}_int': true_labels_int
    }).set_index('sample')
    
    per_sample_path = outdir_p / 'per_sample_clusters.csv'
    per_sample_df.to_csv(per_sample_path)
    
    # Generate plots
    if create_plots:
        print("Generating visualizations...", file=sys.stderr)
        plot_clustering_results(
            data_array[valid_mask],
            true_labels_int[valid_mask],
            cluster_labels[valid_mask],
            metrics,
            outdir_p,
            is_distance=False,
            label_name=label_col
        )
    
    # Create summary
    summary_lines = [
        "=" * 60,
        "ARI-based Clustering Evaluation Summary (KMeans)",
        "=" * 60,
        f"Label column: {label_col}",
        f"Data mode: {mode}",
        f"Total samples: {n_samples}",
        f"Samples with valid labels: {metrics['n_samples']}",
        f"Unique {label_col} values: {n_unique_labels} -> {list(unique_labels)}",
        f"Number of clusters used: {n_clusters}",
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
        interpretation = f"EXCELLENT: Strong agreement between clusters and {label_col}"
    elif metrics['ari'] > 0.5:
        interpretation = f"GOOD: Moderate agreement between clusters and {label_col}"
    elif metrics['ari'] > 0.3:
        interpretation = f"FAIR: Weak agreement between clusters and {label_col}"
    elif metrics['ari'] > 0.1:
        interpretation = f"POOR: Very weak agreement between clusters and {label_col}"
    else:
        interpretation = f"VERY POOR: Almost no agreement between clusters and {label_col}"
    
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
        'k_neighbors': k_neighbors,  # kept for compatibility
        'label_column': label_col,
        f'unique_{label_col}': unique_labels.tolist(),
        'interpretation': interpretation
    }
    metrics_path = outdir_p / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    return {
        'metrics': metrics,
        'interpretation': interpretation,
        'n_samples': n_samples,
        'label_column': label_col,
        f'{label_col}_values': list(unique_labels),
        'per_sample_path': str(per_sample_path),
        'summary_path': str(summary_path),
        'metrics_json_path': str(metrics_path)
    }


if __name__ == "__main__":
    # Example CLI-style usage (edit or wrap with argparse as needed)
    if len(sys.argv) < 5:
        print(
            "Usage: python embedding_effective.py META_CSV DATA_CSV MODE OUTDIR [LABEL_COL]",
            file=sys.stderr
        )
        sys.exit(1)
    
    meta_csv = sys.argv[1]
    data_csv = sys.argv[2]
    mode = sys.argv[3]  # must be 'embedding'
    outdir = sys.argv[4]
    label_col = sys.argv[5] if len(sys.argv) > 5 else "tissue"
    
    evaluate_ari_clustering(
        meta_csv=meta_csv,
        data_csv=data_csv,
        mode=mode,
        outdir=outdir,
        label_col=label_col
    )