import os
from typing import Dict, Optional, Tuple

import anndata as ad
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


def cluster(
    pseudobulk_adata: ad.AnnData,
    output_dir: str,
    number_of_clusters: int = 5,
    use_expression: bool = True,   # legacy arg; ignored
    use_proportion: bool = True,   # legacy arg; ignored
    random_state: int = 0,
) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    """K-means on the precomputed sample-level DR embedding (``obsm['X_DR_sample']``).

    Returns the legacy ``(expr_results, prop_results)`` tuple for backward
    compatibility; both slots contain the same {sample_id -> cluster_label} mapping.
    ``use_expression`` and ``use_proportion`` are accepted but ignored.

    Parameters
    ----------
    pseudobulk_adata : ad.AnnData
        Samples × features AnnData with ``obsm['X_DR_sample']``.
    output_dir : str
        Directory for clustering results and plots.
    number_of_clusters : int
        K for K-means.
    random_state : int
        Seed for K-means reproducibility.
    """
    if not isinstance(pseudobulk_adata, ad.AnnData):
        raise TypeError("pseudobulk_adata must be an AnnData object.")

    sample_cluster_dir = os.path.join(output_dir, "sample_cluster")
    os.makedirs(sample_cluster_dir, exist_ok=True)
    print(f"[INFO] K-means output directory: {sample_cluster_dir}")

    sample_ids = np.array(pseudobulk_adata.obs_names).astype(str)
    expr_results: Optional[Dict[str, int]] = None
    prop_results: Optional[Dict[str, int]] = None

    def _plot_embedding(
        X: np.ndarray,
        labels: np.ndarray,
        sample_ids: np.ndarray,
        title: str,
        save_path: str,
    ):
        """2D scatter of the first two embedding dimensions, colored by cluster."""
        if X.shape[1] < 2:
            raise ValueError(
                f"Embedding for {title} has shape {X.shape}, "
                "need at least 2 dimensions to plot."
            )

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        n_clusters = len(np.unique(labels))
        colors = sns.color_palette("husl", n_clusters)

        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[cluster_id]],
                       s=80, alpha=0.7, edgecolors='white', linewidth=1.5,
                       label=f'Cluster {cluster_id}')

        ax.set_xlabel("Dimension 1", fontsize=12, fontweight='bold')
        ax.set_ylabel("Dimension 2", fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[INFO] Saved plot: {save_path}")

    if "X_DR_sample" not in pseudobulk_adata.obsm_keys():
        raise KeyError("'X_DR_sample' not found in pseudobulk_adata.obsm.")
    dr_key = "X_DR_sample"
    X = pseudobulk_adata.obsm[dr_key]
    print(f"[INFO] Running K-means on '{dr_key}', shape={X.shape}")
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=random_state,
                     n_init="auto")
    labels = kmeans.fit_predict(X)
    label_map = {sid: int(lbl) for sid, lbl in zip(sample_ids, labels)}
    pseudobulk_adata.obs["cluster_sample_kmeans"] = labels.astype(str)

    csv_path = os.path.join(sample_cluster_dir, "kmeans_clusters_sample.csv")
    import pandas as pd
    pd.DataFrame({"sample": sample_ids,
                   "cluster_sample_kmeans": labels.astype(int)}).to_csv(csv_path, index=False)
    print(f"[INFO] Saved sample clustering CSV: {csv_path}")

    plot_path = os.path.join(sample_cluster_dir, "kmeans_sample_embedding.png")
    _plot_embedding(
        X=X, labels=labels, sample_ids=sample_ids,
        title=f"K-means Clustering on Sample Embedding (k={number_of_clusters})",
        save_path=plot_path,
    )

    # Return the same shape for backward compatibility — both slots point to the
    # same per-sample cluster mapping.
    expr_results = label_map
    prop_results = label_map

    print("[INFO] K-means clustering completed.")
    return expr_results, prop_results