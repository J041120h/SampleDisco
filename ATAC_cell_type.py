import numpy as np
import pandas as pd
import os
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import scanpy as sc
from sklearn.neighbors import KNeighborsTransformer
import time

def cell_types_atac(
    adata, 
    cell_column='cell_type', 
    existing_cell_types=False,
    n_target_clusters=None,
    umap=False,
    Save=False,
    output_dir=None,
    cluster_resolution=0.8,
    use_rep='X_DM_harmony',
    peaks=None, 
    method='average', 
    metric='euclidean', 
    distance_mode='centroid',
    num_DMs=20, 
    max_resolution=5.0,
    resolution_step=0.5,
    _recursion_depth=0,  # Internal parameter to track recursion
    verbose=True
):
    """
    Assigns cell types based on existing annotations or performs Leiden clustering if no annotation exists.
    Uses recursive strategy to adaptively find optimal clustering resolution when target clusters specified.
    
    ATAC VERSION: Uses dimension reduction (X_DM_harmony) for dendrogram construction and differential peaks.

    Parameters:
    - adata: AnnData object
    - cell_column: Column name containing cell type annotations
    - existing_cell_types: Boolean, whether to use existing cell type annotations
    - n_target_clusters: int, optional. Target number of clusters.
    - umap: Boolean, whether to compute UMAP
    - Save: Boolean, whether to save the output
    - output_dir: Directory to save the output if Save=True
    - cluster_resolution: Starting resolution for Leiden clustering
    - use_rep: Representation to use for neighborhood graph (default: 'X_DM_harmony')
    - peaks: List of peak names for mapping numeric IDs to names
    - method, metric, distance_mode: Parameters for hierarchical clustering
    - num_DMs: Number of diffusion map components for neighborhood graph
    - max_resolution: Maximum resolution to try before giving up
    - resolution_step: Step size for increasing resolution
    - _recursion_depth: Internal parameter (do not set manually)
    - verbose: Whether to print progress messages

    Returns:
    - Updated AnnData object with assigned cell types
    """
    start_time = time.time() if verbose else None
    
    # Track recursion depth for debugging and preventing infinite loops
    if _recursion_depth > 10:
        raise RuntimeError(f"Maximum recursion depth exceeded. Could not achieve {n_target_clusters} clusters.")

    # ============================================================================
    # EXISTING CELL TYPE ANNOTATION PROCESSING
    # ============================================================================
    if cell_column in adata.obs.columns and existing_cell_types:
        if verbose and _recursion_depth == 0:
            print("[cell_types_atac] Found existing cell type annotation.")
        
        # Standardize cell type column
        adata.obs['cell_type'] = adata.obs[cell_column].astype(str)

        # Count current number of unique cell types
        current_n_types = adata.obs['cell_type'].nunique()
        if verbose:
            prefix = "  " * _recursion_depth  # Indent for recursion levels
            print(f"{prefix}[cell_types_atac] Current number of cell types: {current_n_types}")

        # ========================================================================
        # CONDITIONAL DENDROGRAM CONSTRUCTION AND AGGREGATION
        # ========================================================================
        apply_dendrogram = (
            n_target_clusters is not None and 
            current_n_types > n_target_clusters
        )
        
        if apply_dendrogram:
            if verbose:
                prefix = "  " * _recursion_depth
                print(f"{prefix}[cell_types_atac] Aggregating {current_n_types} cell types into {n_target_clusters} clusters using dendrogram.")
                print(f"{prefix}[cell_types_atac] Using dimension reduction ({use_rep}) for dendrogram construction...")
            
            # Apply dendrogram clustering using dimension reduction
            adata = cell_type_dendrogram_atac(
                adata=adata,
                n_clusters=n_target_clusters,
                groupby='cell_type',
                method=method,
                metric=metric,
                distance_mode=distance_mode,
                use_rep=use_rep,
                num_DMs=num_DMs,
                verbose=verbose
            )
            
            final_n_types = adata.obs['cell_type'].nunique()
            if verbose:
                print(f"{prefix}[cell_types_atac] Successfully aggregated to {final_n_types} cell types.")
        
        else:
            if n_target_clusters is not None and current_n_types <= n_target_clusters:
                if verbose:
                    prefix = "  " * _recursion_depth
                    print(f"{prefix}[cell_types_atac] Current cell types ({current_n_types}) <= target clusters ({n_target_clusters}). Using as-is.")

        # Build neighborhood graph for existing annotations (only on first call)
        if _recursion_depth == 0:
            if verbose:
                print("[cell_types_atac] Building neighborhood graph...")
            # For ATAC data, we don't use n_pcs parameter since we're using diffusion maps
            sc.pp.neighbors(adata, use_rep=use_rep, metric='cosine')

    # ============================================================================
    # DE NOVO CLUSTERING (NO EXISTING ANNOTATIONS) - RECURSIVE STRATEGY
    # ============================================================================
    else:
        if verbose and _recursion_depth == 0:
            print("[cell_types_atac] No cell type annotation found. Performing clustering.")

        # Build neighborhood graph (only on first call)
        if _recursion_depth == 0:
            if verbose:
                print("[cell_types_atac] Building neighborhood graph...")
            # For ATAC data, we don't use n_pcs parameter since we're using diffusion maps
            sc.pp.neighbors(adata, use_rep=use_rep)

        # ========================================================================
        # ADAPTIVE CLUSTERING WITH RECURSION
        # ========================================================================
        if n_target_clusters is not None:
            if verbose:
                prefix = "  " * _recursion_depth
                print(f"{prefix}[cell_types_atac] Target: {n_target_clusters} clusters. Trying resolution: {cluster_resolution:.1f}")
            
            # Perform Leiden clustering with current resolution
            sc.tl.leiden(
                adata,
                resolution=cluster_resolution,
                flavor='igraph',
                n_iterations=2,
                directed=False,
                key_added='cell_type'
            )
            
            # Convert cluster labels to 1-based indexing as strings
            adata.obs['cell_type'] = (adata.obs['cell_type'].astype(int) + 1).astype(str).astype('category')
            num_clusters = adata.obs['cell_type'].nunique()
            
            if verbose:
                prefix = "  " * _recursion_depth
                print(f"{prefix}[cell_types_atac] Leiden clustering produced {num_clusters} clusters.")
            
            # Decision logic: recurse or aggregate
            if num_clusters >= n_target_clusters:
                if num_clusters == n_target_clusters:
                    if verbose:
                        print(f"{prefix}[cell_types_atac] Perfect! Got exactly {n_target_clusters} clusters.")
                else:
                    if verbose:
                        print(f"{prefix}[cell_types_atac] Got {num_clusters} clusters (>= target). Recursing with existing_cell_types=True...")
                    
                    # RECURSIVE CALL: Now treat current clustering as "existing" and aggregate
                    adata = cell_types_atac(
                        adata=adata,
                        cell_column='cell_type',
                        existing_cell_types=True,
                        n_target_clusters=n_target_clusters,
                        umap=False,  # Don't compute UMAP in recursion
                        Save=False,  # Don't save in recursion
                        output_dir=None,
                        method=method,
                        metric=metric,
                        distance_mode=distance_mode,
                        use_rep=use_rep,
                        num_DMs=num_DMs,
                        _recursion_depth=_recursion_depth + 1,
                        verbose=verbose
                    )
            
            else:  # num_clusters < n_target_clusters
                # Need more clusters - increase resolution and try again
                new_resolution = cluster_resolution + resolution_step
                
                if new_resolution > max_resolution:
                    if verbose:
                        print(f"{prefix}[cell_types_atac] Warning: Reached max resolution ({max_resolution}). Got {num_clusters} clusters instead of {n_target_clusters}.")
                else:
                    if verbose:
                        print(f"{prefix}[cell_types_atac] Need more clusters. Increasing resolution to {new_resolution:.1f}...")
                    
                    # RECURSIVE CALL: Try higher resolution
                    return cell_types_atac(
                        adata=adata,
                        cell_column=cell_column,
                        existing_cell_types=False,
                        n_target_clusters=n_target_clusters,
                        umap=False,  # Don't compute UMAP in recursion
                        Save=False,  # Don't save in recursion
                        output_dir=None,
                        cluster_resolution=new_resolution,
                        use_rep=use_rep,
                        peaks=peaks,
                        method=method,
                        metric=metric,
                        distance_mode=distance_mode,
                        num_DMs=num_DMs,
                        max_resolution=max_resolution,
                        resolution_step=resolution_step,
                        _recursion_depth=_recursion_depth + 1,
                        verbose=verbose
                    )
        
        else:
            # No target specified - standard clustering
            if verbose:
                prefix = "  " * _recursion_depth
                print(f"{prefix}[cell_types_atac] No target clusters specified. Using standard Leiden clustering (resolution={cluster_resolution})...")
            
            sc.tl.leiden(
                adata,
                resolution=cluster_resolution,
                flavor='igraph',
                n_iterations=2,
                directed=False,
                key_added='cell_type'
            )

            # Convert cluster labels to 1-based indexing as strings
            adata.obs['cell_type'] = (adata.obs['cell_type'].astype(int) + 1).astype(str).astype('category')
            num_clusters = adata.obs['cell_type'].nunique()
            
            if verbose:
                print(f"[cell_types_atac] Found {num_clusters} clusters after Leiden clustering.")

    # ============================================================================
    # FINAL PROCESSING (ONLY ON TOP-LEVEL CALL)
    # ============================================================================
    if _recursion_depth == 0:
        # Apply peak mapping if provided and appropriate
        final_cluster_count = adata.obs['cell_type'].nunique()
        if peaks is not None and len(peaks) == final_cluster_count:
            if verbose:
                print(f"[cell_types_atac] Applying custom peak names to {final_cluster_count} clusters...")
            # Create mapping for string cluster labels
            peak_dict = {str(i): peaks[i - 1] for i in range(1, len(peaks) + 1)}
            adata.obs['cell_type'] = adata.obs['cell_type'].map(peak_dict)
        elif peaks is not None:
            if verbose:
                print(f"[cell_types_atac] Warning: Peak list length ({len(peaks)}) doesn't match cluster count ({final_cluster_count}). Skipping peak mapping.")

        # Find differential peaks if requested
        if verbose:
            print("[cell_types_atac] Finding differential peaks between cell types...")
        try:
            sc.tl.rank_genes_groups(adata, 'cell_type', method='wilcoxon')
            if verbose:
                print("[cell_types_atac] Successfully computed differential peaks.")
        except Exception as e:
            if verbose:
                print(f"[cell_types_atac] Warning: Could not compute differential peaks. Error: {e}")

        if verbose:
            print("[cell_types_atac] Finished assigning cell types.")
        
        # Compute UMAP if requested
        if umap:
            if verbose:
                print("[cell_types_atac] Computing UMAP...")
            sc.tl.umap(adata, min_dist=0.5)
        
        # Save results if requested
        if Save and output_dir:
            output_dir = os.path.join(output_dir, 'harmony')
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'adata_cell_atac.h5ad')
            adata.write(save_path)
            if verbose:
                print(f"[cell_types_atac] Saved AnnData object to {save_path}")
        
        # Report total execution time
        if verbose:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[cell_types_atac] Total runtime: {elapsed_time:.2f} seconds")

    return adata


def cell_type_dendrogram_atac(
    adata,
    n_clusters,
    groupby='cell_type',
    method='average',
    metric='euclidean',
    distance_mode='centroid',
    use_rep='X_DM_harmony',
    num_DMs=20,
    verbose=True
):
    """
    Constructs a dendrogram of cell types based on dimension reduction results (e.g., X_DM_harmony)
    and aggregates them into a specified number of clusters.
    
    ATAC VERSION: Uses diffusion map space for clustering.
    """
    start_time = time.time()
    
    # ============================================================================
    # PARAMETER VALIDATION AND SETUP
    # ============================================================================
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    
    if verbose:
        print(f'=== Preparing data for dendrogram (using {use_rep}) ===')

    if groupby not in adata.obs.columns:
        raise ValueError(f"The groupby key '{groupby}' is not present in adata.obs.")

    if use_rep not in adata.obsm:
        raise ValueError(f"The representation '{use_rep}' is not present in adata.obsm.")

    # ============================================================================
    # DATA PREPARATION USING DIMENSION REDUCTION
    # ============================================================================
    # Get the dimension reduction data
    if num_DMs is not None and 'DM' in use_rep:
        # Use only the first num_DMs components if specified
        dim_data = adata.obsm[use_rep][:, :num_DMs]
        if verbose:
            print(f'Using first {num_DMs} components from {use_rep}')
    else:
        dim_data = adata.obsm[use_rep]
        if verbose:
            print(f'Using all {dim_data.shape[1]} components from {use_rep}')
    
    # Create DataFrame with dimension reduction data and cell types
    df_dims = pd.DataFrame(
        dim_data,
        index=adata.obs_names,
        columns=[f'DM{i+1}' for i in range(dim_data.shape[1])]
    )
    df_dims[groupby] = adata.obs[groupby].values

    # ============================================================================
    # CENTROID COMPUTATION AND DISTANCE MATRIX CALCULATION
    # ============================================================================
    if distance_mode == 'centroid':
        if verbose:
            print(f'=== Computing centroids of cell types in {use_rep} space ===')
        centroids = df_dims.groupby(groupby).mean()
        original_n_types = centroids.shape[0]
        
        if verbose:
            print(f'Calculated centroids for {original_n_types} cell types.')
            print(f'Centroid shape: {centroids.shape}')
            print(f'=== Computing distance matrix between centroids using {metric} distance ===')
        
        dist_matrix = pdist(centroids.values, metric=metric)
        labels = centroids.index.tolist()
    else:
        raise ValueError(f"Unsupported distance_mode '{distance_mode}' for dimension reduction approach.")

    # ============================================================================
    # HIERARCHICAL CLUSTERING
    # ============================================================================
    if verbose:
        print(f'=== Performing hierarchical clustering on {use_rep} centroids ===')
        print(f'Linkage method: {method}, Distance metric: {metric}')
    
    Z = sch.linkage(dist_matrix, method=method)
    adata.uns['cell_type_linkage'] = Z

    # ============================================================================
    # CLUSTER CUTTING AND VALIDATION
    # ============================================================================
    if n_clusters > original_n_types:
        if verbose:
            print(f'Warning: Requested {n_clusters} clusters, but only {original_n_types} original cell types exist.')
            print(f'Setting n_clusters to {original_n_types}')
        n_clusters = original_n_types
    
    if verbose:
        print(f'=== Aggregating cell types into {n_clusters} clusters ===')
    
    cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    actual_n_clusters = len(np.unique(cluster_labels))
    
    if verbose:
        print(f'Successfully created {actual_n_clusters} clusters')

    # ============================================================================
    # CELL TYPE ANNOTATION UPDATE AND MAPPING CREATION
    # ============================================================================
    # Create mapping from original cell types to new clusters (as strings)
    celltype_to_cluster = dict(zip(centroids.index, [str(label) for label in cluster_labels]))
    
    # Apply the mapping to update cell type annotations
    adata.obs[f'{groupby}_original'] = adata.obs[groupby].copy()  # Backup original
    adata.obs[groupby] = adata.obs[groupby].map(celltype_to_cluster).astype('category')
    
    # Create a mapping dictionary for interpretation
    cluster_mapping = {}
    for original_type, new_cluster in celltype_to_cluster.items():
        if new_cluster not in cluster_mapping:
            cluster_mapping[new_cluster] = []
        cluster_mapping[new_cluster].append(original_type)
    
    adata.uns['cluster_mapping'] = cluster_mapping
    
    # ============================================================================
    # RESULTS REPORTING AND CLEANUP
    # ============================================================================
    if verbose:
        print('\n=== Cluster Composition ===')
        for cluster_id, original_types in sorted(cluster_mapping.items()):
            print(f'Cluster {cluster_id}: {", ".join(map(str, sorted(original_types)))}')
        
        # Compute and report average within-cluster distances
        print('\n=== Cluster Quality Metrics ===')
        for cluster_id in sorted(cluster_mapping.keys()):
            cluster_types = cluster_mapping[cluster_id]
            if len(cluster_types) > 1:
                # Get centroids of types in this cluster
                cluster_centroids = centroids.loc[cluster_types]
                # Compute pairwise distances
                if cluster_centroids.shape[0] > 1:
                    within_cluster_dist = pdist(cluster_centroids.values, metric=metric)
                    avg_dist = np.mean(within_cluster_dist)
                    print(f'Cluster {cluster_id}: Average within-cluster distance = {avg_dist:.4f}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose:
        print(f"\nFunction execution time: {elapsed_time:.2f} seconds")

    return adata


def cell_type_assign_atac(adata_cluster, adata, Save=False, output_dir=None, verbose=True):
    """
    Assign cell type labels from one AnnData object to another and optionally save the result.
    ATAC version with appropriate file naming.

    Parameters
    ----------
    adata_cluster : AnnData
        AnnData object containing a 'cell_type' column in `.obs` to be used for assignment.
    adata : AnnData
        Target AnnData object to receive the 'cell_type' labels.
    Save : bool, optional
        If True, saves the modified `adata` object to disk.
    output_dir : str, optional
        Directory to save the `adata` object if `Save` is True.
    verbose : bool, optional
        If True and saving is enabled, prints the save location.
    """
    if 'cell_type' not in adata_cluster.obs.columns or adata_cluster.obs['cell_type'].nunique() == 0:
        adata_cluster.obs['cell_type'] = '1'

    adata.obs['cell_type'] = adata_cluster.obs['cell_type']
    
    if Save and output_dir:
        output_dir = os.path.join(output_dir, 'harmony')
        save_path = os.path.join(output_dir, 'adata_sample_atac.h5ad')
        sc.write(save_path, adata)
        if verbose:
            print(f"[cell_types_atac] Saved AnnData object to {save_path}")