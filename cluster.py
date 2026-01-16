import os
from typing import Dict, Optional, Tuple

import anndata as ad
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def cluster(
    pseudobulk_adata: ad.AnnData,
    output_dir: str,
    number_of_clusters: int = 5,
    use_expression: bool = True,
    use_proportion: bool = True,
    random_state: int = 0,
) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    """
    Cluster samples using K-means on precomputed DR embeddings stored in AnnData.obsm,
    save clustering results to an output directory, and generate embedding plots.

    Expected embeddings in pseudobulk_adata.obsm:
        - 'X_DR_expression'
        - 'X_DR_proportion'

    Parameters
    ----------
    pseudobulk_adata : ad.AnnData
        Pseudobulk AnnData object (samples x genes) with DR embeddings in `.obsm`.
    output_dir : str
        Directory where clustering results and plots will be saved.
    number_of_clusters : int, default=5
        Number of clusters for K-means.
    use_expression : bool, default=True
        If True, run K-means on 'X_DR_expression'.
    use_proportion : bool, default=True
        If True, run K-means on 'X_DR_proportion'.
    random_state : int, default=0
        Random seed for K-means reproducibility.

    Returns
    -------
    expr_results : dict or None
        Mapping {sample_id -> cluster_label} for expression embedding, or None if not run.
    prop_results : dict or None
        Mapping {sample_id -> cluster_label} for proportion embedding, or None if not run.
    """
    if not isinstance(pseudobulk_adata, ad.AnnData):
        raise TypeError("pseudobulk_adata must be an AnnData object.")

    # -------------------------------------------------
    # Prepare output directory
    # -------------------------------------------------
    sample_cluster_dir = os.path.join(output_dir, "sample_cluster")
    os.makedirs(sample_cluster_dir, exist_ok=True)
    print(f"[INFO] K-means output directory: {sample_cluster_dir}")

    sample_ids = np.array(pseudobulk_adata.obs_names).astype(str)
    expr_results: Optional[Dict[str, int]] = None
    prop_results: Optional[Dict[str, int]] = None

    # -------------------------------------------------
    # Helper for plotting
    # -------------------------------------------------
    def _plot_embedding(
        X: np.ndarray,
        labels: np.ndarray,
        sample_ids: np.ndarray,
        title: str,
        save_path: str,
    ):
        """
        Simple 2D scatter of the first two dimensions of X, colored by cluster.
        """
        if X.shape[1] < 2:
            raise ValueError(
                f"Embedding for {title} has shape {X.shape}, "
                "need at least 2 dimensions to plot."
            )

        plt.figure(figsize=(6, 5))
        plt.scatter(
            X[:, 0],
            X[:, 1],
            c=labels,
            s=40,
            alpha=0.8,
        )
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title(title)
        # Optionally annotate points lightly (comment out if cluttered)
        # for sid, x, y in zip(sample_ids, X[:, 0], X[:, 1]):
        #     plt.text(x, y, str(sid), fontsize=6, alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved plot: {save_path}")

    # -------------------------------------------------
    # Expression embedding
    # -------------------------------------------------
    if use_expression:
        if "X_DR_expression" not in pseudobulk_adata.obsm_keys():
            raise KeyError(
                "Embedding 'X_DR_expression' not found in pseudobulk_adata.obsm. "
                "Available keys: " + ", ".join(pseudobulk_adata.obsm_keys())
            )

        X_expr = pseudobulk_adata.obsm["X_DR_expression"]
        print(f"[INFO] Running K-means on expression embedding 'X_DR_expression', shape={X_expr.shape}")

        kmeans_expr = KMeans(
            n_clusters=number_of_clusters,
            random_state=random_state,
            n_init="auto",
        )
        labels_expr = kmeans_expr.fit_predict(X_expr)

        # Map sample -> cluster label
        expr_results = {sid: int(lbl) for sid, lbl in zip(sample_ids, labels_expr)}

        # Store cluster labels in obs
        pseudobulk_adata.obs["cluster_expression_kmeans"] = labels_expr.astype(str)

        # Save clustering to CSV
        expr_csv_path = os.path.join(sample_cluster_dir, "kmeans_clusters_expression.csv")
        expr_df = (
            # two columns: sample, cluster
            # ensure consistent ordering
            __import__("pandas").DataFrame(
                {
                    "sample": sample_ids,
                    "cluster_expression_kmeans": labels_expr.astype(int),
                }
            )
        )
        expr_df.to_csv(expr_csv_path, index=False)
        print(f"[INFO] Saved expression clustering CSV: {expr_csv_path}")

        # Plot embedding
        expr_plot_path = os.path.join(sample_cluster_dir, "kmeans_expression_embedding.png")
        _plot_embedding(
            X=X_expr,
            labels=labels_expr,
            sample_ids=sample_ids,
            title=f"K-means (expression) - k={number_of_clusters}",
            save_path=expr_plot_path,
        )

    # -------------------------------------------------
    # Proportion embedding
    # -------------------------------------------------
    if use_proportion:
        if "X_DR_proportion" not in pseudobulk_adata.obsm_keys():
            raise KeyError(
                "Embedding 'X_DR_proportion' not found in pseudobulk_adata.obsm. "
                "Available keys: " + ", ".join(pseudobulk_adata.obsm_keys())
            )

        X_prop = pseudobulk_adata.obsm["X_DR_proportion"]
        print(f"[INFO] Running K-means on proportion embedding 'X_DR_proportion', shape={X_prop.shape}")

        kmeans_prop = KMeans(
            n_clusters=number_of_clusters,
            random_state=random_state,
            n_init="auto",
        )
        labels_prop = kmeans_prop.fit_predict(X_prop)

        # Map sample -> cluster label
        prop_results = {sid: int(lbl) for sid, lbl in zip(sample_ids, labels_prop)}

        # Store cluster labels in obs
        pseudobulk_adata.obs["cluster_proportion_kmeans"] = labels_prop.astype(str)

        # Save clustering to CSV
        prop_csv_path = os.path.join(sample_cluster_dir, "kmeans_clusters_proportion.csv")
        prop_df = (
            __import__("pandas").DataFrame(
                {
                    "sample": sample_ids,
                    "cluster_proportion_kmeans": labels_prop.astype(int),
                }
            )
        )
        prop_df.to_csv(prop_csv_path, index=False)
        print(f"[INFO] Saved proportion clustering CSV: {prop_csv_path}")

        # Plot embedding
        prop_plot_path = os.path.join(sample_cluster_dir, "kmeans_proportion_embedding.png")
        _plot_embedding(
            X=X_prop,
            labels=labels_prop,
            sample_ids=sample_ids,
            title=f"K-means (proportion) - k={number_of_clusters}",
            save_path=prop_plot_path,
        )

    print("[INFO] K-means clustering completed.")
    return expr_results, prop_results
