"""
Sample distance computation module.

Computes distances between samples using various methods:
- Standard metrics (euclidean, cosine, etc.) on dimension reduction results
- EMD (Earth Mover's Distance) using cell type proportions and centroids
- Chi-square and Jensen-Shannon distances on proportions
"""

import os
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse import issparse
from typing import Optional, List, Dict, Tuple, Union

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visualization.visualization_helper import visualizeDistanceMatrix
from sample_distance.distance_test import distanceCheck
from .ChiSquare import chi_square_distance
from .jensenshannon import jensen_shannon_distance


# =============================================================================
# EMD Distance Computation
# =============================================================================

def compute_emd_distances(
    adata: AnnData,
    sample_column: str = 'sample',
    cell_type_column: str = 'cell_type',
    embedding_key: str = 'X_pca_harmony',
    n_pcs: int = 20,
    proportions: Optional[pd.DataFrame] = None,
    centroids: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    normalize: bool = True
) -> pd.DataFrame:
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
    try:
        import ot
    except ImportError:
        raise ImportError("POT library required for EMD. Install with: pip install POT")
    
    # Get samples and cell types
    samples = adata.obs[sample_column].unique()
    cell_types = adata.obs[cell_type_column].unique()
    n_samples = len(samples)
    n_cell_types = len(cell_types)
    
    # 1. Get or compute proportions
    prop_matrix = _get_proportions(
        adata, samples, cell_types, sample_column, 
        cell_type_column, proportions
    )
    
    # 2. Get or compute centroids
    centroid_matrix = _get_centroids(
        adata, cell_types, cell_type_column, 
        embedding_key, n_pcs, centroids
    )
    
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


def _get_proportions(
    adata: AnnData,
    samples: np.ndarray,
    cell_types: np.ndarray,
    sample_column: str,
    cell_type_column: str,
    proportions: Optional[pd.DataFrame]
) -> np.ndarray:
    """Get or compute cell type proportions matrix."""
    n_samples = len(samples)
    n_cell_types = len(cell_types)
    
    if proportions is not None:
        # Use provided proportions
        prop_matrix = proportions.reindex(
            index=samples, columns=cell_types, fill_value=0
        ).values
    elif 'cell_proportions' in adata.uns:
        # Use pre-computed proportions from pseudobulk
        prop_df = adata.uns['cell_proportions']
        # Handle both orientations: (cell_types × samples) or (samples × cell_types)
        if set(prop_df.index).intersection(set(cell_types)):
            if len(set(prop_df.index).intersection(set(cell_types))) > len(set(prop_df.index).intersection(set(samples))):
                prop_df = prop_df.T  # Transpose to (samples × cell_types)
        prop_matrix = prop_df.reindex(
            index=samples, columns=cell_types, fill_value=0
        ).values
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
    
    return prop_matrix


def _get_centroids(
    adata: AnnData,
    cell_types: np.ndarray,
    cell_type_column: str,
    embedding_key: str,
    n_pcs: int,
    centroids: Optional[Union[pd.DataFrame, np.ndarray]]
) -> np.ndarray:
    """Get or compute cell type centroids matrix."""
    n_cell_types = len(cell_types)
    
    if centroids is not None:
        # Use provided centroids
        if isinstance(centroids, pd.DataFrame):
            return centroids.reindex(index=cell_types).values
        return centroids
    
    # Compute from adata
    if embedding_key not in adata.obsm:
        raise ValueError(
            f"Embedding key '{embedding_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    
    embeddings = adata.obsm[embedding_key]
    if issparse(embeddings):
        embeddings = embeddings.toarray()
    if hasattr(embeddings, 'get'):  # GPU array (cupy)
        embeddings = embeddings.get()
    
    if n_pcs is not None and embeddings.shape[1] > n_pcs:
        embeddings = embeddings[:, :n_pcs]
    
    centroid_matrix = np.zeros((n_cell_types, embeddings.shape[1]))
    for j, ct in enumerate(cell_types):
        mask = adata.obs[cell_type_column] == ct
        if mask.sum() > 0:
            centroid_matrix[j] = embeddings[mask].mean(axis=0)
    
    return centroid_matrix


def emd_distance(
    adata: AnnData,
    output_dir: str,
    sample_column: str = 'sample',
    cell_type_column: str = 'cell_type',
    embedding_key: str = 'X_pca_harmony',
    n_pcs: int = 20,
    proportions: Optional[pd.DataFrame] = None,
    centroids: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    summary_csv_path: Optional[str] = None,
    grouping_columns: Optional[List[str]] = None,
    pseudobulk_adata: Optional[AnnData] = None
) -> pd.DataFrame:
    """
    Compute and save EMD distance matrix between samples.
    
    Parameters
    ----------
    adata : AnnData
        Single-cell dataset with cell type annotations and embeddings.
    output_dir : str
        Directory to save output files.
    sample_column : str
        Column name for sample information.
    cell_type_column : str
        Column name for cell type annotations.
    embedding_key : str
        Key in adata.obsm for cell embeddings.
    n_pcs : int
        Number of dimensions to use from embedding.
    proportions : pd.DataFrame, optional
        Pre-computed cell type proportions.
    centroids : pd.DataFrame or np.ndarray, optional
        Pre-computed cell type centroids.
    summary_csv_path : str, optional
        Path to summary CSV for logging results.
    grouping_columns : list, optional
        Columns for grouping analysis.
    pseudobulk_adata : AnnData, optional
        Pseudobulk AnnData for metadata in distance check.
        
    Returns
    -------
    pd.DataFrame
        EMD distance matrix.
    """
    # Create output directory
    emd_output_dir = os.path.join(output_dir, 'EMD_distance')
    os.makedirs(emd_output_dir, exist_ok=True)
    
    print(f"Computing EMD distances...")
    print(f"  Sample column: {sample_column}")
    print(f"  Cell type column: {cell_type_column}")
    print(f"  Embedding: {embedding_key} (first {n_pcs} dims)")
    
    # Compute EMD distances
    distance_df = compute_emd_distances(
        adata=adata,
        sample_column=sample_column,
        cell_type_column=cell_type_column,
        embedding_key=embedding_key,
        n_pcs=n_pcs,
        proportions=proportions,
        centroids=centroids,
        normalize=True
    )
    
    # Save distance matrix
    distance_path = os.path.join(emd_output_dir, 'distance_matrix_EMD.csv')
    distance_df.to_csv(distance_path)
    print(f"  Distance matrix saved to: {distance_path}")
    
    # Save proportions for reference
    samples = adata.obs[sample_column].unique()
    cell_types = adata.obs[cell_type_column].unique()
    prop_matrix = _get_proportions(
        adata, samples, cell_types, sample_column, 
        cell_type_column, proportions
    )
    prop_df = pd.DataFrame(prop_matrix, index=samples, columns=cell_types)
    prop_df.to_csv(os.path.join(emd_output_dir, 'cell_type_proportions.csv'))
    
    # Save centroids for reference
    centroid_matrix = _get_centroids(
        adata, cell_types, cell_type_column, 
        embedding_key, n_pcs, centroids
    )
    centroid_df = pd.DataFrame(
        centroid_matrix, 
        index=cell_types,
        columns=[f'PC{i+1}' for i in range(centroid_matrix.shape[1])]
    )
    centroid_df.to_csv(os.path.join(emd_output_dir, 'cell_type_centroids.csv'))
    
    # Perform distance check
    check_adata = pseudobulk_adata if pseudobulk_adata is not None else adata
    try:
        score = distanceCheck(
            distance_df=distance_df,
            row='EMD',
            method='EMD',
            output_dir=emd_output_dir,
            adata=check_adata,
            grouping_columns=grouping_columns,
            summary_csv_path=summary_csv_path
        )
        print(f"  Distance check score: {score:.6f}")
    except Exception as e:
        print(f"  Warning: Distance check failed: {e}")
    
    # Generate visualization
    try:
        visualizeDistanceMatrix(
            distance_df,
            os.path.join(emd_output_dir, 'distance_matrix_EMD_heatmap.pdf')
        )
    except Exception as e:
        print(f"  Warning: Failed to create heatmap: {e}")
    
    return distance_df


# =============================================================================
# DR-based Distance Computation
# =============================================================================

def calculate_sample_distances_DR(
    adata: AnnData,
    DR_key: str,
    output_dir: str,
    method: str = 'euclidean',
    grouping_columns: Optional[List[str]] = None,
    dr_name: str = 'DR',
    summary_csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute sample distance matrix using dimensionality reduction results.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with DR results in .uns[DR_key].
    DR_key : str
        Key in adata.uns where the DR DataFrame is stored.
    output_dir : str
        Directory for output files.
    method : str
        Distance metric for scipy.spatial.distance.pdist.
    grouping_columns : list, optional
        Columns for grouping analysis.
    dr_name : str
        Name for file naming.
    summary_csv_path : str, optional
        Path to summary CSV for logging results.
        
    Returns
    -------
    pd.DataFrame
        Sample distance matrix.
    """
    # Validate inputs
    if DR_key not in adata.uns:
        raise KeyError(
            f"DR key '{DR_key}' not found in adata.uns. "
            f"Available keys: {list(adata.uns.keys())}"
        )
    
    DR = adata.uns[DR_key]
    if DR is None or DR.empty:
        raise ValueError(f"DR DataFrame for key '{DR_key}' is empty.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process DR data
    dr_data = DR.fillna(0)
    dr_data = _match_samples(dr_data, adata)
    
    # Save DR coordinates
    dr_data.to_csv(os.path.join(output_dir, f'{dr_name}_coordinates.csv'))
    
    # Compute distances
    distance_matrix = pdist(dr_data.values, metric=method)
    distance_df = pd.DataFrame(
        squareform(distance_matrix),
        index=dr_data.index,
        columns=dr_data.index
    )
    
    # Log transform and normalize
    distance_df = np.log1p(np.maximum(distance_df, 0))
    if distance_df.max().max() > 0:
        distance_df = distance_df / distance_df.max().max()
    
    # Save distance matrix
    distance_path = os.path.join(output_dir, f'distance_matrix_{dr_name}.csv')
    distance_df.to_csv(distance_path)
    
    # Distance check
    try:
        score = distanceCheck(
            distance_df=distance_df,
            row=dr_name,
            method=method,
            output_dir=output_dir,
            adata=adata,
            grouping_columns=grouping_columns,
            summary_csv_path=summary_csv_path
        )
        print(f"  Distance check for {dr_name}: score = {score:.6f}")
    except Exception as e:
        print(f"  Warning: Distance check failed for {dr_name}: {e}")
    
    # Visualization
    try:
        visualizeDistanceMatrix(
            distance_df,
            os.path.join(output_dir, f'sample_distance_{dr_name}_heatmap.pdf')
        )
    except Exception as e:
        print(f"  Warning: Failed to create heatmap for {dr_name}: {e}")
    
    print(f"  {dr_name} distance matrix saved to: {output_dir}")
    return distance_df


def _match_samples(dr_data: pd.DataFrame, adata: AnnData) -> pd.DataFrame:
    """Match DR sample names with AnnData sample names."""
    dr_samples = set(dr_data.index)
    adata_samples = set(adata.obs.index)
    
    # Check exact matches first
    exact_matches = dr_samples.intersection(adata_samples)
    
    if len(exact_matches) > 0:
        return dr_data.loc[list(exact_matches)].copy()
    
    # Try case-insensitive matching
    dr_lower = {name.lower(): name for name in dr_samples}
    adata_lower = {name.lower(): name for name in adata_samples}
    lowercase_matches = set(dr_lower.keys()).intersection(set(adata_lower.keys()))
    
    if len(lowercase_matches) > 0:
        sample_mapping = {
            dr_lower[low]: adata_lower[low] 
            for low in lowercase_matches
        }
        matching_samples = list(sample_mapping.keys())
        dr_filtered = dr_data.loc[matching_samples].copy()
        dr_filtered.index = [sample_mapping[name] for name in dr_filtered.index]
        return dr_filtered
    
    raise ValueError(
        f"No matching samples found. "
        f"DR samples: {sorted(list(dr_samples))[:5]}, "
        f"AnnData samples: {sorted(list(adata_samples))[:5]}"
    )


# =============================================================================
# Helper Functions for DR Key Selection
# =============================================================================

def get_best_expression_dr_key(adata: AnnData, data_type: str = 'ATAC') -> Optional[str]:
    """Get best available expression DR key based on data type."""
    if 'X_DR_expression' in adata.uns:
        return 'X_DR_expression'
    
    if data_type.upper() == 'ATAC':
        priority = ['X_lsi_expression_method', 'X_pca_expression_method', 'X_spectral_expression_method']
    else:
        priority = ['X_pca_expression_method', 'X_lsi_expression_method', 'X_spectral_expression_method']
    
    for key in priority:
        if key in adata.uns:
            return key
    return None


def get_best_proportion_dr_key(adata: AnnData) -> Optional[str]:
    """Get best available proportion DR key."""
    if 'X_DR_proportion' in adata.uns:
        return 'X_DR_proportion'
    
    for key in ['X_pca_proportion_method', 'X_lsi_proportion_method', 'X_spectral_proportion_method']:
        if key in adata.uns:
            return key
    return None


# =============================================================================
# Vector Distance Computation
# =============================================================================

def sample_distance_vector(
    adata: AnnData,
    output_dir: str,
    method: str,
    data_type: str = 'ATAC',
    grouping_columns: Optional[List[str]] = None,
    summary_csv_path: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Compute sample distances using dimension reduction results.
    
    Creates:
    - expression_DR_distance: Using best available expression DR
    - proportion_DR_distance: Using best available proportion DR
    
    Parameters
    ----------
    adata : AnnData
        AnnData with DR results in .uns.
    output_dir : str
        Output directory.
    method : str
        Distance metric.
    data_type : str
        'ATAC' or 'RNA' (affects DR prioritization).
    grouping_columns : list, optional
        Columns for grouping analysis.
    summary_csv_path : str, optional
        Path to summary CSV.
        
    Returns
    -------
    dict
        Dictionary of computed distance matrices.
    """
    method_output_dir = os.path.join(output_dir, method)
    os.makedirs(method_output_dir, exist_ok=True)
    
    # Validate grouping columns
    if grouping_columns:
        valid_cols = [c for c in grouping_columns if c in adata.obs.columns]
        if len(valid_cols) < len(grouping_columns):
            missing = set(grouping_columns) - set(valid_cols)
            print(f"  Warning: Grouping columns not found: {missing}")
        grouping_columns = valid_cols if valid_cols else None
    
    distance_results = {}
    
    # Expression DR distance
    expr_key = get_best_expression_dr_key(adata, data_type)
    if expr_key:
        try:
            print(f"Computing expression DR distances ({expr_key})...")
            distance_results['expression_DR'] = calculate_sample_distances_DR(
                adata=adata,
                DR_key=expr_key,
                output_dir=os.path.join(method_output_dir, 'expression_DR_distance'),
                method=method,
                grouping_columns=grouping_columns,
                dr_name='expression_DR',
                summary_csv_path=summary_csv_path
            )
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print("  Warning: No expression DR results found")
    
    # Proportion DR distance
    prop_key = get_best_proportion_dr_key(adata)
    if prop_key:
        try:
            print(f"Computing proportion DR distances ({prop_key})...")
            distance_results['proportion_DR'] = calculate_sample_distances_DR(
                adata=adata,
                DR_key=prop_key,
                output_dir=os.path.join(method_output_dir, 'proportion_DR_distance'),
                method=method,
                grouping_columns=grouping_columns,
                dr_name='proportion_DR',
                summary_csv_path=summary_csv_path
            )
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print("  Warning: No proportion DR results found")
    
    if not distance_results:
        raise ValueError("No dimension reduction results found in adata.uns")
    
    # Save summary statistics
    _save_distance_statistics(distance_results, method_output_dir, method)
    
    return distance_results


def _save_distance_statistics(
    distance_results: Dict[str, pd.DataFrame],
    output_dir: str,
    method: str
) -> None:
    """Save summary statistics for distance matrices."""
    try:
        stats = {}
        for name, dist_df in distance_results.items():
            vals = dist_df.values[np.triu_indices_from(dist_df.values, k=1)]
            stats.update({
                f"{name}_mean": np.mean(vals),
                f"{name}_std": np.std(vals),
                f"{name}_min": np.min(vals),
                f"{name}_max": np.max(vals),
                f"{name}_median": np.median(vals)
            })
        
        stats_df = pd.DataFrame([stats])
        stats_path = os.path.join(output_dir, f'distance_statistics_summary_{method}.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"Statistics saved to: {stats_path}")
    except Exception as e:
        print(f"Warning: Failed to save statistics: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

# Valid scipy pdist metrics
VALID_PDIST_METRICS = {
    "euclidean", "sqeuclidean", "minkowski", "cityblock", "chebyshev",
    "cosine", "correlation", "hamming", "jaccard", "canberra",
    "braycurtis", "matching"
}

# Specialized methods requiring cell-level data
SPECIALIZED_METHODS = {"EMD", "chi_square", "jensen_shannon"}


def sample_distance(
    adata: AnnData,
    output_dir: str,
    method: str,
    data_type: str = 'ATAC',
    grouping_columns: Optional[List[str]] = None,
    summary_csv_path: Optional[str] = None,
    # EMD-specific parameters
    cell_adata: Optional[AnnData] = None,
    cell_type_column: str = 'cell_type',
    sample_column: str = 'sample',
    embedding_key: str = 'X_pca_harmony',
    n_pcs: int = 20,
    proportions: Optional[pd.DataFrame] = None,
    centroids: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    pseudobulk_adata: Optional[AnnData] = None
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Unified function to compute sample distance matrices.
    
    Handles:
    - Standard metrics (euclidean, cosine, etc.) on DR results
    - EMD using cell type proportions and centroids
    - Chi-square and Jensen-Shannon on proportions
    
    Parameters
    ----------
    adata : AnnData
        AnnData with DR results (for standard metrics) or sample metadata.
    output_dir : str
        Output directory.
    method : str
        Distance method: standard metric name, 'EMD', 'chi_square', or 'jensen_shannon'.
    data_type : str
        'ATAC' or 'RNA' (affects DR prioritization).
    grouping_columns : list, optional
        Columns for grouping analysis.
    summary_csv_path : str, optional
        Path to summary CSV.
    cell_adata : AnnData, optional
        Cell-level AnnData (required for EMD, chi_square, jensen_shannon).
    cell_type_column : str
        Column name for cell types.
    sample_column : str
        Column name for samples.
    embedding_key : str
        Key in obsm for embeddings (EMD only).
    n_pcs : int
        Number of PCs to use (EMD only).
    proportions : pd.DataFrame, optional
        Pre-computed proportions (EMD only).
    centroids : pd.DataFrame or np.ndarray, optional
        Pre-computed centroids (EMD only).
    pseudobulk_adata : AnnData, optional
        Pseudobulk AnnData for metadata.
        
    Returns
    -------
    dict or None
        Dictionary of distance matrices, or None if method unknown.
    """
    from utils.random_seed import set_global_seed
    set_global_seed(seed=42)
    
    print(f"Computing {method} distance...")
    
    if method in VALID_PDIST_METRICS:
        # Standard distance metrics on DR results
        return sample_distance_vector(
            adata=adata,
            output_dir=output_dir,
            method=method,
            data_type=data_type,
            grouping_columns=grouping_columns,
            summary_csv_path=summary_csv_path
        )
    
    elif method == "EMD":
        if cell_adata is None:
            raise ValueError("cell_adata required for EMD distance")
        
        distance_df = emd_distance(
            adata=cell_adata,
            output_dir=output_dir,
            sample_column=sample_column,
            cell_type_column=cell_type_column,
            embedding_key=embedding_key,
            n_pcs=n_pcs,
            proportions=proportions,
            centroids=centroids,
            summary_csv_path=summary_csv_path,
            grouping_columns=grouping_columns,
            pseudobulk_adata=pseudobulk_adata
        )
        return {'EMD': distance_df}
    
    elif method == "chi_square":
        if cell_adata is None:
            raise ValueError("cell_adata required for chi_square distance")
        
        print("Computing Chi-square distance...")
        chi_output_dir = os.path.join(output_dir, 'chi_square')
        chi_square_distance(
            adata=cell_adata,
            output_dir=chi_output_dir,
            summary_csv_path=summary_csv_path,
            cell_type_column=cell_type_column,
            sample_column=sample_column,
            pseudobulk_adata=pseudobulk_adata
        )
        return None  # chi_square_distance saves internally
    
    elif method == "jensen_shannon":
        if cell_adata is None:
            raise ValueError("cell_adata required for jensen_shannon distance")
        
        print("Computing Jensen-Shannon distance...")
        js_output_dir = os.path.join(output_dir, 'jensen_shannon')
        jensen_shannon_distance(
            adata=cell_adata,
            output_dir=js_output_dir,
            summary_csv_path=summary_csv_path,
            cell_type_column=cell_type_column,
            sample_column=sample_column,
            pseudobulk_adata=pseudobulk_adata
        )
        return None  # jensen_shannon_distance saves internally
    
    else:
        print(f"Warning: Unknown distance method '{method}'. Skipping...")
        return None