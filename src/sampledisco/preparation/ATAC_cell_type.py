import numpy as np
import pandas as pd
import os
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import scanpy as sc
import time
import matplotlib.pyplot as plt
from sampledisco.utils.safe_save import safe_h5ad_write  # Importing the new safe save method

def clean_obs_for_saving(adata, verbose=True):
    """
    Clean adata.obs to prevent string conversion errors during H5AD saving.

    Parameters
    ----------
    adata : AnnData
    verbose : bool

    Returns
    -------
    adata : AnnData
    """
    if verbose:
        print("[clean_obs_for_saving] Cleaning observation metadata for H5AD compatibility...")

    obs_copy = adata.obs.copy()
    
    for col in obs_copy.columns:
        col_data = obs_copy[col].copy()
        
        if pd.api.types.is_categorical_dtype(col_data):
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
            
            mapping = dict(zip(col_data.cat.categories, new_categories))
            col_values = col_data.to_numpy()
            new_values = [mapping.get(val, 'Unknown') if not pd.isna(val) else 'Unknown' 
                         for val in col_values]
            
            col_data = pd.Categorical(new_values, categories=new_categories)
        
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
        
        elif col_data.dtype in ['bool', np.bool_]:
            new_values = ['True' if val else 'False' if not pd.isna(val) else 'Unknown' 
                         for val in col_data]
            col_data = pd.Categorical(new_values)
        
        elif pd.api.types.is_numeric_dtype(col_data):
            n_unique = col_data.nunique()
            if n_unique < 20 and n_unique > 0:  # heuristic: treat low-cardinality numerics as categorical
                if col_data.isna().any():
                    col_data = col_data.fillna(-999)
                col_data = col_data.astype(str).replace(['-999', '-999.0'], 'Unknown')
                col_data = pd.Categorical(col_data)
            else:
                if col_data.isna().any():
                    col_data = col_data.fillna(-1)
        
        else:
            col_data = col_data.astype(str).fillna('Unknown')
            col_data = col_data.replace(['None', 'nan', 'NaN', 'NULL', '', '<NA>'], 'Unknown')
            col_data = pd.Categorical(col_data)
        
        obs_copy[col] = col_data

    adata.obs = obs_copy
    
    if verbose:
        print("[clean_obs_for_saving] Cleaning completed successfully")
        cat_cols = [col for col in adata.obs.columns if pd.api.types.is_categorical_dtype(adata.obs[col])]
        print(f"[clean_obs_for_saving] {len(cat_cols)} categorical columns processed")
    
    return adata

def safe_h5ad_write(adata, filepath, verbose=True):
    """
    Write AnnData to H5AD, cleaning obs metadata first.

    Parameters
    ----------
    adata : AnnData
    filepath : str
    verbose : bool
    """
    try:
        if verbose:
            print(f"[safe_h5ad_write] Preparing to save to: {filepath}")
        
        adata_copy = adata.copy()
        adata_copy = clean_obs_for_saving(adata_copy, verbose=verbose)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if verbose:
            print(f"[safe_h5ad_write] Writing H5AD file...")
        
        sc.write(filepath, adata_copy)
        
        if verbose:
            print(f"[safe_h5ad_write] Successfully saved to: {filepath}")
            
    except Exception as e:
        if verbose:
            print(f"[safe_h5ad_write] Error saving H5AD file: {str(e)}")
            
            print("\n=== DIAGNOSTIC INFORMATION ===")
            print(f"Error type: {type(e).__name__}")
            print(f"adata.obs shape: {adata.obs.shape}")
            print("\nChecking for non-string categories:")
            for col in adata.obs.columns:
                if pd.api.types.is_categorical_dtype(adata.obs[col]):
                    cats = adata.obs[col].cat.categories
                    non_string = [c for c in cats if not isinstance(c, str)]
                    if non_string:
                        print(f"  - {col}: Found {len(non_string)} non-string categories")
                        print(f"    Examples: {non_string[:3]}")
        
        raise e


def standardize_cell_type_column(adata, verbose=True):
    """
    Standardize cell_type column to string categorical format.
    Only performs necessary conversions without redundancy.
    """
    if 'cell_type' not in adata.obs.columns:
        adata.obs['cell_type'] = '1'
        if verbose:
            print("[standardize] Created missing cell_type column with default value '1'")
    
    current_dtype = adata.obs['cell_type'].dtype

    # Leiden outputs 0-based integers; convert to 1-based strings.
    if pd.api.types.is_integer_dtype(current_dtype):
        adata.obs['cell_type'] = (adata.obs['cell_type'].astype(int) + 1).astype(str)
        if verbose:
            print("[standardize] Converted integer clusters to 1-based string format")
    
    elif current_dtype == 'object' or pd.api.types.is_string_dtype(current_dtype):
        adata.obs['cell_type'] = adata.obs['cell_type'].astype(str)
        if verbose:
            print("[standardize] Ensured consistent string format")
    
    elif pd.api.types.is_categorical_dtype(current_dtype):
        if pd.api.types.is_integer_dtype(adata.obs['cell_type'].cat.categories.dtype):
            # cat.codes are 0-based; shift to 1-based
            adata.obs['cell_type'] = (adata.obs['cell_type'].cat.codes + 1).astype(str)
            if verbose:
                print("[standardize] Converted categorical integer to 1-based string format")
        else:
            adata.obs['cell_type'] = adata.obs['cell_type'].astype(str)
            if verbose:
                print("[standardize] Converted categorical to string format")
    
    else:
        adata.obs['cell_type'] = adata.obs['cell_type'].astype(str)
        if verbose:
            print(f"[standardize] Converted {current_dtype} to string format")
    
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
    
    if verbose:
        unique_types = adata.obs['cell_type'].nunique()
        print(f"[standardize] Final cell types (n={unique_types}): {sorted(adata.obs['cell_type'].cat.categories.tolist())}")


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
    # New plotting parameters
    generate_plots=True,
    cell_type_key='cell_type',
    batch_key=None,
    plot_dpi=300,
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
    - generate_plots: Boolean, whether to generate UMAP plots
    - cell_type_key: Column name for cell types in plots (default: 'cell_type')
    - batch_key: Column name(s) for batch information (str or list)
    - plot_dpi: DPI for saved plots
    - _recursion_depth: Internal parameter (do not set manually)
    - verbose: Whether to print progress messages

    Returns:
    - Updated AnnData object with assigned cell types
    """
    start_time = time.time() if verbose else None
    from sampledisco.utils.random_seed import set_global_seed
    set_global_seed(seed = 42)
    
    if _recursion_depth > 10:
        raise RuntimeError(f"Maximum recursion depth exceeded. Could not achieve {n_target_clusters} clusters.")

    if cell_column in adata.obs.columns and existing_cell_types:
        if verbose and _recursion_depth == 0:
            print("[cell_types_atac] Found existing cell type annotation.")
        
        adata.obs['cell_type'] = adata.obs[cell_column].astype(str)

        current_n_types = adata.obs['cell_type'].nunique()
        if verbose:
            prefix = "  " * _recursion_depth
            print(f"{prefix}[cell_types_atac] Current number of cell types: {current_n_types}")

        apply_dendrogram = (
            n_target_clusters is not None and 
            current_n_types > n_target_clusters
        )
        
        if apply_dendrogram:
            if verbose:
                prefix = "  " * _recursion_depth
                print(f"{prefix}[cell_types_atac] Aggregating {current_n_types} cell types into {n_target_clusters} clusters using dendrogram.")
                print(f"{prefix}[cell_types_atac] Using dimension reduction ({use_rep}) for dendrogram construction...")
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

        # Build neighborhood graph (only on first call; ATAC uses diffusion maps, not PCA)
        if _recursion_depth == 0:
            if verbose:
                print("[cell_types_atac] Building neighborhood graph...")
            sc.pp.neighbors(adata, use_rep=use_rep, metric='cosine')

    else:
        if verbose and _recursion_depth == 0:
            print(f"[cell_types_atac] No cell type annotation found. Performing clustering at resolution {cluster_resolution}.")

        # Build neighborhood graph (only on first call; ATAC uses diffusion maps, not PCA)
        if _recursion_depth == 0:
            if verbose:
                print("[cell_types_atac] Building neighborhood graph...")
            sc.pp.neighbors(adata, use_rep=use_rep)

        if n_target_clusters is not None:
            if verbose:
                prefix = "  " * _recursion_depth
                print(f"{prefix}[cell_types_atac] Target: {n_target_clusters} clusters. Trying resolution: {cluster_resolution:.1f}")
            
            sc.tl.leiden(
                adata,
                resolution=cluster_resolution,
                flavor='igraph',
                n_iterations=2,
                directed=False,
                key_added='cell_type',
                
            )
            
            adata.obs['cell_type'] = (adata.obs['cell_type'].astype(int) + 1).astype(str).astype('category')
            num_clusters = adata.obs['cell_type'].nunique()
            if verbose:
                prefix = "  " * _recursion_depth
                print(f"{prefix}[cell_types_atac] Leiden clustering produced {num_clusters} clusters.")
            if num_clusters >= n_target_clusters:
                if num_clusters == n_target_clusters:
                    if verbose:
                        print(f"{prefix}[cell_types_atac] Perfect! Got exactly {n_target_clusters} clusters.")
                else:
                    if verbose:
                        print(f"{prefix}[cell_types_atac] Got {num_clusters} clusters (>= target). Recursing with existing_cell_types=True...")
                    adata = cell_types_atac(
                        adata=adata,
                        cell_column='cell_type',
                        existing_cell_types=True,
                        n_target_clusters=n_target_clusters,
                        umap=False,
                        Save=False,
                        output_dir=None,
                        method=method,
                        metric=metric,
                        distance_mode=distance_mode,
                        use_rep=use_rep,
                        num_DMs=num_DMs,
                        generate_plots=False,
                        _recursion_depth=_recursion_depth + 1,
                        verbose=verbose
                    )

            else:  # num_clusters < n_target_clusters
                new_resolution = cluster_resolution + resolution_step
                
                if new_resolution > max_resolution:
                    if verbose:
                        print(f"{prefix}[cell_types_atac] Warning: Reached max resolution ({max_resolution}). Got {num_clusters} clusters instead of {n_target_clusters}.")
                else:
                    if verbose:
                        print(f"{prefix}[cell_types_atac] Need more clusters. Increasing resolution to {new_resolution:.1f}...")
                    return cell_types_atac(
                        adata=adata,
                        cell_column=cell_column,
                        existing_cell_types=False,
                        n_target_clusters=n_target_clusters,
                        umap=False,
                        Save=False,
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
                        generate_plots=False,
                        _recursion_depth=_recursion_depth + 1,
                        verbose=verbose
                    )
        
        else:
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

            adata.obs['cell_type'] = (adata.obs['cell_type'].astype(int) + 1).astype(str).astype('category')
            num_clusters = adata.obs['cell_type'].nunique()
            if verbose:
                print(f"[cell_types_atac] Found {num_clusters} clusters after Leiden clustering.")

    if _recursion_depth == 0:
        final_cluster_count = adata.obs['cell_type'].nunique()
        if peaks is not None and len(peaks) == final_cluster_count:
            if verbose:
                print(f"[cell_types_atac] Applying custom peak names to {final_cluster_count} clusters...")
            peak_dict = {str(i): peaks[i - 1] for i in range(1, len(peaks) + 1)}
            adata.obs['cell_type'] = adata.obs['cell_type'].map(peak_dict)
        elif peaks is not None:
            if verbose:
                print(f"[cell_types_atac] Warning: Peak list length ({len(peaks)}) doesn't match cluster count ({final_cluster_count}). Skipping peak mapping.")

        if verbose:
            print("[cell_types_atac] Finished assigning cell types.")
        
        if umap:
            if verbose:
                print("[cell_types_atac] Computing UMAP...")
            sc.tl.umap(adata, min_dist=0.5)
        
        if generate_plots and output_dir and 'X_umap' in adata.obsm:
            if verbose:
                print("[cell_types_atac] Generating UMAP plots...")
            output_dir = os.path.join(output_dir, 'preprocess')
            os.makedirs(output_dir, exist_ok=True)
            
            sc.pl.umap(adata, color=cell_type_key, legend_loc="on data", show=False)
            plt.savefig(os.path.join(output_dir, f"umap_{cell_type_key}.png"), dpi=plot_dpi)
            plt.close()
            if verbose:
                print(f"[cell_types_atac] Saved UMAP plot colored by {cell_type_key}")
            
            if 'n_genes_by_counts' in adata.obs.columns:
                sc.pl.umap(adata, color=[cell_type_key, "n_genes_by_counts"], 
                          legend_loc="on data", show=False)
                plt.savefig(os.path.join(output_dir, "umap_n_genes_by_counts.png"), dpi=plot_dpi)
                plt.close()
                if verbose:
                    print("[cell_types_atac] Saved UMAP plot with n_genes_by_counts")
            
            if batch_key:
                batch_keys = batch_key if isinstance(batch_key, list) else [batch_key]
                for key in batch_keys:
                    if key in adata.obs.columns:
                        sc.pl.umap(adata, color=key, legend_loc="on data", show=False)
                        plt.savefig(os.path.join(output_dir, f"umap_{key}.png"), dpi=plot_dpi)
                        plt.close()
                        if verbose:
                            print(f"[cell_types_atac] Saved UMAP plot colored by {key}")
                    else:
                        if verbose:
                            print(f"[cell_types_atac] Warning: Batch key '{key}' not found in adata.obs")
        
        elif generate_plots and 'X_umap' not in adata.obsm:
            if verbose:
                print("[cell_types_atac] Warning: Cannot generate plots - UMAP coordinates not found. Set umap=True to compute UMAP first.")
        
        standardize_cell_type_column(adata, verbose=verbose)

        if Save and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'adata_sample.h5ad')
            safe_h5ad_write(adata, save_path, verbose=verbose)
        
        if verbose:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[cell_types_atac] Total runtime: {elapsed_time:.2f} seconds")

    return adata
