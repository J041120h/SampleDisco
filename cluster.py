import anndata as ad
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Optional, Tuple


def cluster(
    pseudobulk_adata: ad.AnnData,
    number_of_clusters: int = 5,
    use_expression: bool = True,
    use_proportion: bool = True,
    random_state: int = 0,
) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    """
    Cluster samples using K-means on precomputed DR embeddings stored in AnnData.obsm.

    Expected embeddings:
        - pseudobulk_adata.obsm['X_DR_expression']
        - pseudobulk_adata.obsm['X_DR_proportion']

    Parameters
    ----------
    pseudobulk_adata : ad.AnnData
        Pseudobulk AnnData object (samples x genes) with DR embeddings in `.obsm`.
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

    sample_ids = np.array(pseudobulk_adata.obs_names).astype(str)
    expr_results: Optional[Dict[str, int]] = None
    prop_results: Optional[Dict[str, int]] = None

    # -------------------------
    # Expression embedding
    # -------------------------
    if use_expression:
        if "X_DR_expression" not in pseudobulk_adata.obsm_keys():
            raise KeyError(
                "Embedding 'X_DR_expression' not found in pseudobulk_adata.obsm. "
                "Available keys: " + ", ".join(pseudobulk_adata.obsm_keys())
            )

        X_expr = pseudobulk_adata.obsm["X_DR_expression"]
        print(f"[INFO] Running K-means on expression embedding: X_DR_expression, shape={X_expr.shape}")

        kmeans_expr = KMeans(
            n_clusters=number_of_clusters,
            random_state=random_state,
            n_init="auto",
        )
        labels_expr = kmeans_expr.fit_predict(X_expr)

        # Map sample -> cluster label
        expr_results = {sid: int(lbl) for sid, lbl in zip(sample_ids, labels_expr)}

        # Optionally store in obs (comment out if you don't want this)
        pseudobulk_adata.obs["cluster_expression_kmeans"] = labels_expr.astype(str)

    # -------------------------
    # Proportion embedding
    # -------------------------
    if use_proportion:
        if "X_DR_proportion" not in pseudobulk_adata.obsm_keys():
            raise KeyError(
                "Embedding 'X_DR_proportion' not found in pseudobulk_adata.obsm. "
                "Available keys: " + ", ".join(pseudobulk_adata.obsm_keys())
            )

        X_prop = pseudobulk_adata.obsm["X_DR_proportion"]
        print(f"[INFO] Running K-means on proportion embedding: X_DR_proportion, shape={X_prop.shape}")

        kmeans_prop = KMeans(
            n_clusters=number_of_clusters,
            random_state=random_state,
            n_init="auto",
        )
        labels_prop = kmeans_prop.fit_predict(X_prop)

        # Map sample -> cluster label
        prop_results = {sid: int(lbl) for sid, lbl in zip(sample_ids, labels_prop)}

        # Optionally store in obs (comment out if you don't want this)
        pseudobulk_adata.obs["cluster_proportion_kmeans"] = labels_prop.astype(str)

    print("[INFO] K-means clustering completed.")
    return expr_results, prop_results
