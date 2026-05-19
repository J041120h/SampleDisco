import numpy as np
import pandas as pd
import scanpy as sc
from typing import Tuple, Optional


def filter_modality_imbalanced_clusters(
    adata: sc.AnnData,
    modality_column: str = "modality",
    cluster_column: str = "cell_type",
    modality_values: Tuple[str, str] = ("RNA", "ATAC"),
    min_proportion_of_expected: float = 0.05,
    verbose: bool = True,
) -> sc.AnnData:
    """
    Filter out clusters dominated by a single modality.
    
    Parameters
    ----------
    adata : sc.AnnData
        Integrated AnnData object with both modalities
    modality_column : str
        Column in adata.obs containing modality labels
    cluster_column : str
        Column in adata.obs containing cluster/cell type labels
    modality_values : Tuple[str, str]
        The two modality labels to compare
    min_proportion_of_expected : float, default 0.05
        Minimum proportion of expected representation required (5% = 95% imbalance threshold)
    verbose : bool
        Whether to print diagnostic information
        
    Returns
    -------
    sc.AnnData
        Filtered AnnData with imbalanced clusters removed
    """
    
    modality_1, modality_2 = modality_values
    
    # Calculate expected proportions from overall data
    total_mod1 = (adata.obs[modality_column] == modality_1).sum()
    total_mod2 = (adata.obs[modality_column] == modality_2).sum()
    total_cells = total_mod1 + total_mod2
    
    expected_prop_mod1 = total_mod1 / total_cells
    expected_prop_mod2 = total_mod2 / total_cells
    
    # Find imbalanced clusters
    imbalanced_clusters = []
    
    for cluster in adata.obs[cluster_column].unique():
        cluster_mask = adata.obs[cluster_column] == cluster
        cluster_cells = cluster_mask.sum()
        
        cluster_mod1 = ((adata.obs[modality_column] == modality_1) & cluster_mask).sum()
        cluster_mod2 = ((adata.obs[modality_column] == modality_2) & cluster_mask).sum()
        
        obs_prop_mod1 = cluster_mod1 / cluster_cells
        obs_prop_mod2 = cluster_mod2 / cluster_cells
        
        ratio_mod1 = obs_prop_mod1 / expected_prop_mod1 if expected_prop_mod1 > 0 else 0
        ratio_mod2 = obs_prop_mod2 / expected_prop_mod2 if expected_prop_mod2 > 0 else 0
        
        if min(ratio_mod1, ratio_mod2) < min_proportion_of_expected:
            imbalanced_clusters.append(cluster)
    
    # Filter
    if imbalanced_clusters:
        cells_before = adata.n_obs
        adata_filtered = adata[~adata.obs[cluster_column].isin(imbalanced_clusters)].copy()
        cells_removed = cells_before - adata_filtered.n_obs
        
        if verbose:
            print(f"[Modality Balance] Removed {len(imbalanced_clusters)} imbalanced cluster(s): {imbalanced_clusters}")
            print(f"[Modality Balance] Cells removed: {cells_removed:,} ({cells_removed/cells_before:.1%})")
    else:
        adata_filtered = adata.copy()
        if verbose:
            print(f"[Modality Balance] All clusters balanced, no filtering needed")
    
    return adata_filtered