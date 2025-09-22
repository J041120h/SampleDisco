import numpy as np
import pandas as pd
import os
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import rapids_singlecell as rsc 
import scanpy as sc
from sklearn.neighbors import KNeighborsTransformer
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scanpy as sc
from visualization.visualization_helper import generate_umap_visualizations


def clean_obs_for_saving(adata, verbose=True):
    """
    Clean adata.obs to prevent string conversion errors during H5AD saving.
    Simplified version that handles all common data types efficiently.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to clean
    verbose : bool
        Whether to print cleaning operations
        
    Returns
    -------
    adata : AnnData
        Cleaned AnnData object
    """
    if verbose:
        print("[clean_obs_for_saving] Cleaning observation metadata for H5AD compatibility...")
    
    # Create a copy of obs to avoid modifying during iteration
    obs_copy = adata.obs.copy()
    
    for col in obs_copy.columns:
        col_data = obs_copy[col].copy()
        
        # Handle categorical columns
        if pd.api.types.is_categorical_dtype(col_data):
            # Convert categories to strings
            new_categories = []
            for cat in col_data.cat.categories:
                if pd.isna(cat):
                    new_cat = 'Unknown'
                elif isinstance(cat, bool) or isinstance(cat, np.bool_):
                    new_cat = 'True' if cat else 'False'
                elif isinstance(cat, (int, np.integer, float, np.floating)):
                    new_cat = str(cat).replace('.0', '') if float(cat).is_integer() else str(cat)
                elif isinstance(cat, str):
                    new_cat = cat if cat.strip() else 'Unknown'
                else:
                    new_cat = str(cat)
                new_categories.append(new_cat)
            
            # Handle duplicates if any
            if len(new_categories) != len(set(new_categories)):
                seen = {}
                final_categories = []
                for cat in new_categories:
                    if cat in seen:
                        seen[cat] += 1
                        final_categories.append(f"{cat}_{seen[cat]}")
                    else:
                        seen[cat] = 0
                        final_categories.append(cat)
                new_categories = final_categories
            
            # Map values to new categories
            mapping = dict(zip(col_data.cat.categories, new_categories))
            col_values = col_data.to_numpy()
            new_values = [mapping.get(val, 'Unknown') if not pd.isna(val) else 'Unknown' 
                         for val in col_values]
            
            col_data = pd.Categorical(new_values, categories=new_categories)
        
        # Handle object columns
        elif col_data.dtype == 'object':
            new_values = []
            for val in col_data:
                if pd.isna(val):
                    new_val = 'Unknown'
                elif isinstance(val, bool):
                    new_val = 'True' if val else 'False'
                elif isinstance(val, (int, float)):
                    new_val = str(val).replace('.0', '') if isinstance(val, float) and val.is_integer() else str(val)
                elif isinstance(val, str):
                    new_val = val if val.strip() else 'Unknown'
                else:
                    new_val = str(val)
                new_values.append(new_val)
            
            col_data = pd.Series(new_values, index=col_data.index)
            col_data = col_data.replace(['None', 'nan', 'NaN', 'NULL', '', '<NA>'], 'Unknown')
            col_data = pd.Categorical(col_data)
        
        # Handle boolean columns
        elif col_data.dtype in ['bool', np.bool_]:
            new_values = ['True' if val else 'False' if not pd.isna(val) else 'Unknown' 
                         for val in col_data]
            col_data = pd.Categorical(new_values)
        
        # Handle numeric columns that might be categorical
        elif pd.api.types.is_numeric_dtype(col_data):
            n_unique = col_data.nunique()
            if n_unique < 20 and n_unique > 0:  # Likely categorical
                if col_data.isna().any():
                    col_data = col_data.fillna(-999)
                col_data = col_data.astype(str).replace(['-999', '-999.0'], 'Unknown')
                col_data = pd.Categorical(col_data)
            else:
                # Keep as numeric but handle NaN
                if col_data.isna().any():
                    col_data = col_data.fillna(-1)
        
        else:
            # Any other dtype - convert to string categorical
            col_data = col_data.astype(str).fillna('Unknown')
            col_data = col_data.replace(['None', 'nan', 'NaN', 'NULL', '', '<NA>'], 'Unknown')
            col_data = pd.Categorical(col_data)
        
        # Update the column
        obs_copy[col] = col_data
    
    # Replace the entire obs dataframe
    adata.obs = obs_copy
    
    if verbose:
        print("[clean_obs_for_saving] Cleaning completed successfully")
        # Quick summary of categorical columns
        cat_cols = [col for col in adata.obs.columns if pd.api.types.is_categorical_dtype(adata.obs[col])]
        print(f"[clean_obs_for_saving] {len(cat_cols)} categorical columns processed")
    
    return adata


def ensure_cpu_arrays(adata):
    """
    Ensure all arrays in AnnData object are on CPU (not GPU).
    This prevents "Implicit conversion to NumPy array" errors.
    """
    # Convert main matrix
    if hasattr(adata.X, 'get'):
        adata.X = adata.X.get()
    
    # Convert layers
    if hasattr(adata, 'layers'):
        for key in list(adata.layers.keys()):
            if hasattr(adata.layers[key], 'get'):
                adata.layers[key] = adata.layers[key].get()
    
    # Convert obsm (embeddings)
    if hasattr(adata, 'obsm'):
        for key in list(adata.obsm.keys()):
            if hasattr(adata.obsm[key], 'get'):
                adata.obsm[key] = adata.obsm[key].get()
    
    # Convert varm
    if hasattr(adata, 'varm'):
        for key in list(adata.varm.keys()):
            if hasattr(adata.varm[key], 'get'):
                adata.varm[key] = adata.varm[key].get()
    
    # Convert obsp (pairwise arrays)
    if hasattr(adata, 'obsp'):
        for key in list(adata.obsp.keys()):
            if hasattr(adata.obsp[key], 'get'):
                adata.obsp[key] = adata.obsp[key].get()
    
    # Convert varp
    if hasattr(adata, 'varp'):
        for key in list(adata.varp.keys()):
            if hasattr(adata.varp[key], 'get'):
                adata.varp[key] = adata.varp[key].get()
    
    return adata


def safe_h5ad_write(adata, filepath, verbose=True):
    """
    Safely write AnnData object to H5AD format with proper error handling.
    Simplified version without extensive debugging.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to save
    filepath : str
        Path to save the file
    verbose : bool
        Whether to print progress messages
    """
    try:
        if verbose:
            print(f"[safe_h5ad_write] Preparing to save to: {filepath}")
        
        # Create a copy to avoid modifying the original
        adata_copy = adata.copy()
        
        # Ensure CPU arrays
        adata_copy = ensure_cpu_arrays(adata_copy)
        
        # Clean observation metadata
        adata_copy = clean_obs_for_saving(adata_copy, verbose=verbose)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write the file
        if verbose:
            print(f"[safe_h5ad_write] Writing H5AD file...")
        
        sc.write(filepath, adata_copy)
        
        if verbose:
            print(f"[safe_h5ad_write] Successfully saved to: {filepath}")
            
    except Exception as e:
        if verbose:
            print(f"[safe_h5ad_write] Error saving H5AD file: {str(e)}")
            
            # Provide basic diagnostic information
            print("\n=== DIAGNOSTIC INFORMATION ===")
            print(f"Error type: {type(e).__name__}")
            print(f"adata.obs shape: {adata.obs.shape}")
            
            # Check for non-string categories
            print("\nChecking for non-string categories:")
            for col in adata.obs.columns:
                if pd.api.types.is_categorical_dtype(adata.obs[col]):
                    cats = adata.obs[col].cat.categories
                    non_string = [c for c in cats if not isinstance(c, str)]
                    if non_string:
                        print(f"  - {col}: Found {len(non_string)} non-string categories")
                        print(f"    Examples: {non_string[:3]}")
        
        raise e
    
def cell_types_linux(
    adata, 
    cell_type_column='cell_type', 
    existing_cell_types=False,
    n_target_clusters=None,
    umap=True,
    Save=False,
    output_dir=None,
    defined_output_path=None,
    cluster_resolution=0.8,
    use_rep='X_pca_harmony',
    markers=None, 
    method='average', 
    metric='euclidean', 
    distance_mode='centroid',
    num_PCs=20, 
    max_resolution=5.0,
    resolution_step=0.5,
    _recursion_depth=0,  # Internal parameter to track recursion
    verbose=True,
    # New parameters for UMAP visualization
    generate_plots=True,
    figsize=(12, 8),
    point_size=20,
    dpi=300,
    plot_palette="tab20"
):
    """
    Linux/GPU version of cell_types function using rapids_singlecell.
    Assigns cell types based on existing annotations or performs Leiden clustering if no annotation exists.
    Uses recursive strategy to adaptively find optimal clustering resolution when target clusters specified.
    
    IMPROVED: Now uses dimension reduction (X_pca_harmony) for dendrogram construction instead of marker genes.
    Also includes proper GPU to CPU array conversion to prevent errors.
    UPDATED: Integrated UMAP visualization generation with existing neighborhood graph.

    Parameters:
    - adata: AnnData object
    - cell_type_column: Column name containing cell type annotations
    - existing_cell_types: Boolean, whether to use existing cell type annotations
    - n_target_clusters: int, optional. Target number of clusters.
    - umap: Boolean, whether to compute UMAP
    - Save: Boolean, whether to save the output
    - output_dir: Directory to save the output if Save=True
    - defined_output_path: Alternative output path for saving
    - cluster_resolution: Starting resolution for Leiden clustering
    - use_rep: Representation to use for neighborhood graph (default: 'X_pca_harmony')
    - markers: List of markers for mapping numeric IDs to names
    - method, metric, distance_mode: Parameters for hierarchical clustering
    - num_PCs: Number of principal components for neighborhood graph
    - max_resolution: Maximum resolution to try before giving up
    - resolution_step: Step size for increasing resolution
    - _recursion_depth: Internal parameter (do not set manually)
    - verbose: Whether to print progress messages
    - generate_plots: Boolean, whether to generate UMAP plots
    - figsize, point_size, dpi: Plot appearance parameters
    - plot_palette: Color palette for UMAP plots

    Returns:
    - Updated AnnData object with assigned cell types
    """
    start_time = time.time() if verbose else None
    
    # Track recursion depth for debugging and preventing infinite loops
    if _recursion_depth > 10:
        raise RuntimeError(f"Maximum recursion depth exceeded. Could not achieve {n_target_clusters} clusters.")

    # Move data to GPU on first call
    if _recursion_depth == 0:
        rsc.get.anndata_to_GPU(adata)

    # ============================================================================
    # EXISTING CELL TYPE ANNOTATION PROCESSING
    # ============================================================================
    if cell_type_column in adata.obs.columns and existing_cell_types:
        if verbose and _recursion_depth == 0:
            print("[cell_types] Found existing cell type annotation.")
        
        # Standardize cell type column
        adata.obs['cell_type'] = adata.obs[cell_type_column].astype(str)

        # Count current number of unique cell types
        current_n_types = adata.obs['cell_type'].nunique()
        if verbose:
            prefix = "  " * _recursion_depth  # Indent for recursion levels
            print(f"{prefix}[cell_types] Current number of cell types: {current_n_types}")

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
                print(f"{prefix}[cell_types] Aggregating {current_n_types} cell types into {n_target_clusters} clusters using dendrogram.")
                print(f"{prefix}[cell_types] Using dimension reduction ({use_rep}) for dendrogram construction...")
            
            # Apply dendrogram clustering using dimension reduction
            adata = cell_type_dendrogram_linux(
                adata=adata,
                n_clusters=n_target_clusters,
                groupby='cell_type',
                method=method,
                metric=metric,
                distance_mode=distance_mode,
                use_rep=use_rep,
                num_PCs=num_PCs,
                verbose=verbose
            )
            
            final_n_types = adata.obs['cell_type'].nunique()
            if verbose:
                print(f"{prefix}[cell_types] Successfully aggregated to {final_n_types} cell types.")
        
        else:
            if n_target_clusters is not None and current_n_types <= n_target_clusters:
                if verbose:
                    prefix = "  " * _recursion_depth
                    print(f"{prefix}[cell_types] Current cell types ({current_n_types}) <= target clusters ({n_target_clusters}). Using as-is.")

        # Build neighborhood graph for existing annotations (only on first call)
        if _recursion_depth == 0:
            if verbose:
                print("[cell_types] Building neighborhood graph...")
            rsc.pp.neighbors(adata, use_rep=use_rep, n_pcs=num_PCs)

    # ============================================================================
    # DE NOVO CLUSTERING (NO EXISTING ANNOTATIONS) - RECURSIVE STRATEGY
    # ============================================================================
    else:
        if verbose and _recursion_depth == 0:
            print("[cell_types] No cell type annotation found. Performing clustering.")

        # Build neighborhood graph (only on first call)
        if _recursion_depth == 0:
            if verbose:
                print("[cell_types] Building neighborhood graph...")
            rsc.pp.neighbors(adata, use_rep=use_rep, n_pcs=num_PCs)

        # ========================================================================
        # ADAPTIVE CLUSTERING WITH RECURSION
        # ========================================================================
        if n_target_clusters is not None:
            if verbose:
                prefix = "  " * _recursion_depth
                print(f"{prefix}[cell_types] Target: {n_target_clusters} clusters. Trying resolution: {cluster_resolution:.1f}")
            
            # Perform Leiden clustering with current resolution
            rsc.tl.leiden(
                adata,
                resolution=cluster_resolution,
                key_added='cell_type'
            )
            
            # Convert cluster labels to 1-based indexing as strings
            adata.obs['cell_type'] = (adata.obs['cell_type'].astype(int) + 1).astype(str).astype('category')
            num_clusters = adata.obs['cell_type'].nunique()
            
            if verbose:
                prefix = "  " * _recursion_depth
                print(f"{prefix}[cell_types] Leiden clustering produced {num_clusters} clusters.")
            
            # Decision logic: recurse or aggregate
            if num_clusters >= n_target_clusters:
                if num_clusters == n_target_clusters:
                    if verbose:
                        print(f"{prefix}[cell_types] Perfect! Got exactly {n_target_clusters} clusters.")
                else:
                    if verbose:
                        print(f"{prefix}[cell_types] Got {num_clusters} clusters (>= target). Recursing with existing_cell_types=True...")
                    
                    # RECURSIVE CALL: Now treat current clustering as "existing" and aggregate
                    adata = cell_types_linux(
                        adata=adata,
                        cell_type_column='cell_type',
                        existing_cell_types=True,
                        n_target_clusters=n_target_clusters,
                        umap=False,  # Don't compute UMAP in recursion
                        Save=False,  # Don't save in recursion
                        output_dir=None,
                        method=method,
                        metric=metric,
                        distance_mode=distance_mode,
                        use_rep=use_rep,
                        num_PCs=num_PCs,
                        _recursion_depth=_recursion_depth + 1,
                        verbose=verbose,
                        generate_plots=False  # Don't generate plots in recursion
                    )
            
            else:  # num_clusters < n_target_clusters
                # Need more clusters - increase resolution and try again
                new_resolution = cluster_resolution + resolution_step
                
                if new_resolution > max_resolution:
                    if verbose:
                        print(f"{prefix}[cell_types] Warning: Reached max resolution ({max_resolution}). Got {num_clusters} clusters instead of {n_target_clusters}.")
                else:
                    if verbose:
                        print(f"{prefix}[cell_types] Need more clusters. Increasing resolution to {new_resolution:.1f}...")
                    
                    # RECURSIVE CALL: Try higher resolution
                    return cell_types_linux(
                        adata=adata,
                        cell_type_column=cell_type_column,
                        existing_cell_types=False,
                        n_target_clusters=n_target_clusters,
                        umap=False,  # Don't compute UMAP in recursion
                        Save=False,  # Don't save in recursion
                        output_dir=None,
                        cluster_resolution=new_resolution,
                        use_rep=use_rep,
                        markers=markers,
                        method=method,
                        metric=metric,
                        distance_mode=distance_mode,
                        num_PCs=num_PCs,
                        max_resolution=max_resolution,
                        resolution_step=resolution_step,
                        _recursion_depth=_recursion_depth + 1,
                        verbose=verbose,
                        generate_plots=False  # Don't generate plots in recursion
                    )
        
        else:
            # No target specified - standard clustering
            if verbose:
                prefix = "  " * _recursion_depth
                print(f"{prefix}[cell_types] No target clusters specified. Using standard Leiden clustering (resolution={cluster_resolution})...")
            
            rsc.tl.leiden(
                adata,
                resolution=cluster_resolution,
                key_added='cell_type'
            )

            # Convert cluster labels to 1-based indexing as strings
            adata.obs['cell_type'] = (adata.obs['cell_type'].astype(int) + 1).astype(str).astype('category')
            num_clusters = adata.obs['cell_type'].nunique()
            
            if verbose:
                print(f"[cell_types] Found {num_clusters} clusters after Leiden clustering.")

    # ============================================================================
    # FINAL PROCESSING (ONLY ON TOP-LEVEL CALL)
    # ============================================================================
    if _recursion_depth == 0:
        # Apply marker mapping if provided and appropriate
        final_cluster_count = adata.obs['cell_type'].nunique()
        if markers is not None and len(markers) == final_cluster_count:
            if verbose:
                print(f"[cell_types] Applying custom marker names to {final_cluster_count} clusters...")
            # Create mapping for string cluster labels
            marker_dict = {str(i): markers[i - 1] for i in range(1, len(markers) + 1)}
            adata.obs['cell_type'] = adata.obs['cell_type'].map(marker_dict)
        elif markers is not None:
            if verbose:
                print(f"[cell_types] Warning: Marker list length ({len(markers)}) doesn't match cluster count ({final_cluster_count}). Skipping marker mapping.")

        if verbose:
            print("[cell_types] Finished assigning cell types.")
        
        # Compute UMAP if requested
        if umap:
            if verbose:
                print("[cell_types] Computing UMAP...")
            rsc.tl.umap(adata, min_dist=0.5)
        
        # Move back to CPU before saving and visualization - IMPROVED conversion
        if verbose:
            print("[cell_types] Converting GPU arrays to CPU...")
        rsc.get.anndata_to_CPU(adata)
        
        # Ensure complete CPU conversion
        adata = ensure_cpu_arrays(adata)
        
        # Generate UMAP visualization plots if requested
        if generate_plots and umap and output_dir:
            if verbose:
                print("[cell_types] Generating UMAP visualizations...")
            adata = generate_umap_visualizations(
                adata=adata,
                output_dir=output_dir,
                groupby='cell_type',
                figsize=figsize,
                point_size=point_size,
                dpi=dpi,
                palette=plot_palette,
                verbose=verbose
            )
        
        # Save results if requested - USING SAFE SAVING METHOD
        if Save and output_dir and not defined_output_path:
            output_dir = os.path.join(output_dir, 'preprocess')
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'adata_cell.h5ad')

            # Use safe saving method
            safe_h5ad_write(adata, save_path, verbose=verbose)
        
        if defined_output_path:
            safe_h5ad_write(adata, defined_output_path, verbose=verbose)
        
        # Report total execution time
        if verbose:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[cell_types] Total runtime: {elapsed_time:.2f} seconds")

    return adata


def cell_type_dendrogram_linux(
    adata,
    n_clusters,
    groupby='cell_type',
    method='average',
    metric='euclidean',
    distance_mode='centroid',
    use_rep='X_pca_harmony',
    num_PCs=20,
    verbose=True
):
    """
    Linux/GPU version of cell_type_dendrogram using rapids_singlecell.
    Constructs a dendrogram of cell types based on dimension reduction results (e.g., X_pca_harmony)
    and aggregates them into a specified number of clusters.
    
    IMPROVED: Now uses dimension reduction space instead of marker genes for more stable clustering.
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
    # Ensure data is on CPU for this operation
    temp_adata = adata.copy()
    if hasattr(temp_adata.obsm[use_rep], 'get'):
        obsm_data = temp_adata.obsm[use_rep].get()
    else:
        obsm_data = temp_adata.obsm[use_rep]
    
    # Get the dimension reduction data
    if num_PCs is not None and use_rep.startswith('X_pca'):
        # Use only the first num_PCs components if specified
        dim_data = obsm_data[:, :num_PCs]
        if verbose:
            print(f'Using first {num_PCs} components from {use_rep}')
    else:
        dim_data = obsm_data
        if verbose:
            print(f'Using all {dim_data.shape[1]} components from {use_rep}')
    
    # Create DataFrame with dimension reduction data and cell types
    df_dims = pd.DataFrame(
        dim_data,
        index=adata.obs_names,
        columns=[f'PC{i+1}' for i in range(dim_data.shape[1])]
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


def cell_type_assign_linux(adata_cluster, adata, Save=False, output_dir=None, verbose=True):
    """
    Linux version of cell_type_assign function.
    Assign cell type labels from one AnnData object to another and optionally save the result.

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
    
    # Ensure CPU arrays before saving
    adata = ensure_cpu_arrays(adata)
    
    if Save and output_dir:
        output_dir = os.path.join(output_dir, 'preprocess')
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'adata_sample.h5ad')
        
        # Use safe saving method
        safe_h5ad_write(adata, save_path, verbose=verbose)

    return adata