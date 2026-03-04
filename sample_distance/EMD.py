"""
EMD between samples using cell type proportions and centroid distances.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
import ot


def compute_emd_distances(
    adata,
    sample_column='sample',
    cell_type_column='cell_type',
    embedding_key='X_pca_harmony',
    n_pcs=20,
    proportions=None,
    centroids=None,
    normalize=True
):
    """
    Compute EMD distances between samples.
    
    Each sample is represented as a distribution over cell types (proportions),
    with ground distance defined by cell type centroid distances in embedding space.
    
    Parameters
    ----------
    adata : AnnData
        Single-cell dataset with cell type annotations and embeddings.
    sample_column : str
        Column name for sample information.
    cell_type_column : str
        Column name for cell type annotations.
    embedding_key : str
        Key in adata.obsm for cell embeddings.
        Common options: 'X_pca_harmony', 'X_pca', 'X_umap'
    n_pcs : int
        Number of dimensions to use from embedding.
    proportions : pd.DataFrame, optional
        Pre-computed cell type proportions (samples × cell_types).
        If None, computed from adata.
        Can also be found in adata.uns['cell_proportions'] from pseudobulk.
    centroids : pd.DataFrame or np.ndarray, optional
        Pre-computed cell type centroids (cell_types × dimensions).
        If None, computed from adata.
    normalize : bool
        Normalize distances to [0, 1].
        
    Returns
    -------
    pd.DataFrame
        Symmetric EMD distance matrix (samples × samples).
    """
    # Get samples and cell types
    samples = adata.obs[sample_column].unique()
    cell_types = adata.obs[cell_type_column].unique()
    n_samples = len(samples)
    n_cell_types = len(cell_types)
    
    # 1. Get or compute proportions
    if proportions is not None:
        # Use provided proportions
        prop_matrix = proportions.reindex(index=samples, columns=cell_types, fill_value=0).values
    elif 'cell_proportions' in adata.uns:
        # Use pre-computed proportions from pseudobulk
        prop_df = adata.uns['cell_proportions']
        # Handle both (cell_types × samples) and (samples × cell_types) orientations
        if set(prop_df.index).issubset(set(cell_types)):
            prop_df = prop_df.T  # Transpose to (samples × cell_types)
        prop_matrix = prop_df.reindex(index=samples, columns=cell_types, fill_value=0).values
    else:
        # Compute from adata
        prop_matrix = np.zeros((n_samples, n_cell_types))
        for i, sample in enumerate(samples):
            mask = adata.obs[sample_column] == sample
            sample_ct = adata.obs.loc[mask, cell_type_column]
            counts = sample_ct.value_counts()
            for j, ct in enumerate(cell_types):
                if ct in counts.index:
                    prop_matrix[i, j] = counts[ct]
            if prop_matrix[i].sum() > 0:
                prop_matrix[i] /= prop_matrix[i].sum()
    
    # Ensure proportions sum to 1
    row_sums = prop_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    prop_matrix = prop_matrix / row_sums
    
    # 2. Get or compute centroids
    if centroids is not None:
        # Use provided centroids
        if isinstance(centroids, pd.DataFrame):
            centroid_matrix = centroids.reindex(index=cell_types).values
        else:
            centroid_matrix = centroids
    else:
        # Compute from adata
        embeddings = adata.obsm[embedding_key]
        if issparse(embeddings):
            embeddings = embeddings.toarray()
        if hasattr(embeddings, 'get'):  # GPU array
            embeddings = embeddings.get()
        
        if n_pcs is not None and embeddings.shape[1] > n_pcs:
            embeddings = embeddings[:, :n_pcs]
        
        centroid_matrix = np.zeros((n_cell_types, embeddings.shape[1]))
        for j, ct in enumerate(cell_types):
            mask = adata.obs[cell_type_column] == ct
            if mask.sum() > 0:
                centroid_matrix[j] = embeddings[mask].mean(axis=0)
    
    # 3. Compute ground distance matrix (cell type to cell type)
    ground_dist = cdist(centroid_matrix, centroid_matrix, metric='euclidean')
    ground_dist = ground_dist.astype(np.float64)
    
    if ground_dist.max() > 0:
        ground_dist /= ground_dist.max()
    
    # 4. Compute pairwise EMD
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = ot.emd2(
                prop_matrix[i].astype(np.float64),
                prop_matrix[j].astype(np.float64),
                ground_dist
            )
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    if normalize and dist_matrix.max() > 0:
        dist_matrix /= dist_matrix.max()
    
    return pd.DataFrame(dist_matrix, index=samples, columns=samples)