import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from typing import List, Union, Tuple, Optional
import re

def _extract_numeric_value(value: str) -> Optional[float]:
    """
    Extract numeric value from a string if possible.
    Returns None if no numeric value can be extracted.
    """
    if pd.isna(value):
        return None
    
    # Convert to string if not already
    value_str = str(value)
    
    # Try to extract numeric pattern (handles decimals and negative numbers)
    match = re.search(r'-?\d+\.?\d*', value_str)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def _determine_color_mapping(group_series: pd.Series, verbose: bool = False) -> Tuple[np.ndarray, str, bool]:
    """
    Determine appropriate color mapping based on data type.
    
    Returns:
    - color_values: Array of values for color mapping
    - color_label: Label for colorbar
    - is_continuous: Whether the mapping is continuous (True) or categorical (False)
    """
    # First, check if the series is already numeric
    if pd.api.types.is_numeric_dtype(group_series):
        if verbose:
            print("Detected numeric grouping column - using continuous color mapping")
        # Normalize numeric values
        min_val, max_val = group_series.min(), group_series.max()
        if min_val != max_val:
            color_values = (group_series - min_val) / (max_val - min_val)
        else:
            color_values = np.ones(len(group_series)) * 0.5
        return color_values.values, 'Value', True
    
    # Try to extract numeric values from strings
    numeric_values = group_series.apply(_extract_numeric_value)
    
    # Check if we successfully extracted numeric values for most entries
    if numeric_values.notna().sum() >= 0.8 * len(group_series):  # 80% threshold
        if verbose:
            print("Extracted numeric values from grouping column - using continuous color mapping")
        # Fill NaN with median for color mapping
        median_val = numeric_values.median()
        numeric_values = numeric_values.fillna(median_val)
        
        # Normalize
        min_val, max_val = numeric_values.min(), numeric_values.max()
        if min_val != max_val:
            color_values = (numeric_values - min_val) / (max_val - min_val)
        else:
            color_values = np.ones(len(numeric_values)) * 0.5
        return color_values.values, 'Extracted Value', True
    
    # Fall back to categorical mapping
    if verbose:
        print("Using categorical color mapping for grouping column")
    categories = group_series.astype('category')
    color_values = categories.cat.codes
    return color_values.values, 'Category', False


def plot_sample_cell_proportions_embedding(
    adata: AnnData,
    output_dir: str,
    grouping_columns: List[str] = ['sev.level'],
    verbose: bool = False
) -> None:
    """
    Visualizes PCA results for cell type proportions from adata.uns["X_DR_proportion"].
    Automatically determines whether to use continuous or categorical coloring based on grouping data.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing pseudobulk data (samples x genes) with PCA results
    output_dir : str
        Directory to save the PCA plot
    grouping_columns : List[str]
        Columns used for sample grouping. Default is ['sev.level']
    verbose : bool
        Whether to print detailed information. Default is False
    """
    
    # Create output directory
    output_dir = os.path.join(output_dir, 'harmony')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for PCA results with updated key name
    if "X_DR_proportion" not in adata.uns:
        raise KeyError("Missing 'X_DR_proportion' in adata.uns. Ensure PCA was run on cell proportions.")
    
    pca_data = adata.uns["X_DR_proportion"]
    
    # Handle different PCA data formats
    if hasattr(pca_data, 'values'):  # DataFrame
        pca_coords = pca_data.values
    else:  # numpy array
        pca_coords = pca_data
    
    # Validate PCA dimensions
    if pca_coords.shape[1] < 2:
        raise ValueError(f"PCA data has {pca_coords.shape[1]} components, but 2 required for plotting")
    
    # Get sample information
    if 'sample' in adata.obs.columns:
        samples = adata.obs['sample'].unique()
    else:
        samples = adata.obs.index.unique()
        if verbose:
            print("No 'sample' column found, using observation index as samples")
    
    # Validate sample count consistency
    if len(samples) != pca_coords.shape[0]:
        raise ValueError(f"Number of samples ({len(samples)}) doesn't match PCA data rows ({pca_coords.shape[0]})")
    
    # Create PCA DataFrame
    pca_df = pd.DataFrame(pca_coords[:, :2], index=samples, columns=['PC1', 'PC2'])
    
    # Get grouping information
    try:
        diff_groups = find_sample_grouping(adata, samples, grouping_columns)
        
        if isinstance(diff_groups, dict):
            diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])
        
        # Standardize index format and merge
        diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
        pca_df.index = pca_df.index.astype(str).str.strip().str.lower()
        
        diff_groups = diff_groups.reset_index().rename(columns={'index': 'sample'})
        pca_df = pca_df.reset_index().rename(columns={'index': 'sample'})
        pca_df = pca_df.merge(diff_groups, on='sample', how='left')
        
        if verbose:
            print(f"Successfully merged grouping data. Missing values: {pca_df['plot_group'].isna().sum()}")
        
        # Determine color mapping based on grouping data
        color_values, color_label, is_continuous = _determine_color_mapping(
            pca_df['plot_group'], verbose=verbose
        )
        
    except Exception as e:
        if verbose:
            print(f"Could not retrieve grouping information: {e}. Using sample index for coloring.")
        pca_df = pca_df.reset_index().rename(columns={'index': 'sample'})
        color_values = np.arange(len(pca_df))
        color_label = 'Sample Index'
        is_continuous = False
    
    # Plot PCA
    plt.figure(figsize=(10, 8))
    
    if is_continuous:
        # Continuous color mapping
        sc = plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                        c=color_values, 
                        cmap='viridis', 
                        s=100, 
                        alpha=0.8, 
                        edgecolors='k',
                        linewidth=0.5)
        plt.colorbar(sc, label=color_label)
    else:
        # Categorical color mapping
        unique_vals = np.unique(color_values[~np.isnan(color_values)])
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_vals)))
        
        for i, val in enumerate(unique_vals):
            mask = color_values == val
            if 'plot_group' in pca_df.columns:
                label = pca_df.loc[mask, 'plot_group'].iloc[0] if mask.any() else f'Group {int(val)}'
            else:
                label = f'Group {int(val)}'
            
            plt.scatter(pca_df.loc[mask, 'PC1'], 
                       pca_df.loc[mask, 'PC2'],
                       c=[colors[i]], 
                       s=100, 
                       alpha=0.8, 
                       edgecolors='k',
                       linewidth=0.5,
                       label=label)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.title('2D PCA of Cell Type Proportions', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'sample_relationship_pca_2D_sample_proportion.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"PCA plot saved to: {plot_path}")
    plt.show()


def plot_sample_cell_expression_embedding(
    adata: AnnData, 
    output_dir: str,
    grouping_columns: List[str] = ['sev.level'],
    verbose: bool = False
) -> AnnData:
    """
    Computes and plots UMAP for pseudobulk expression data (samples x genes).
    Automatically determines whether to use continuous or categorical coloring based on grouping data.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with samples as observations and genes as variables
    output_dir : str
        Directory to save the UMAP plot
    grouping_columns : List[str]
        Columns used for sample grouping. Default is ['sev.level']
    verbose : bool
        Whether to print detailed information. Default is False
        
    Returns:
    --------
    AnnData : Copy of input data with UMAP coordinates added
    """
    import scanpy as sc
    
    # Create output directory
    output_dir = os.path.join(output_dir, 'harmony')
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a copy to avoid modifying the original
    adata_copy = adata.copy()
    
    # Compute PCA if not already done
    if 'X_pca' not in adata_copy.obsm:
        if verbose:
            print("Computing PCA...")
        sc.pp.pca(adata_copy, n_comps=50)
    
    # Compute neighborhood graph
    if verbose:
        print("Computing neighborhood graph...")
    sc.pp.neighbors(adata_copy, n_neighbors=15, n_pcs=40)
    
    # Compute UMAP
    if verbose:
        print("Computing UMAP...")
    sc.tl.umap(adata_copy)
    
    # Retrieve grouping info
    samples = adata_copy.obs.index.tolist()
    
    try:
        diff_groups = find_sample_grouping(adata_copy, samples, grouping_columns)
        
        if isinstance(diff_groups, dict):
            diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])
        
        # Normalize sample names for matching
        diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
        adata_copy.obs.index = adata_copy.obs.index.astype(str).str.strip().str.lower()
        
        # Add grouping info to adata.obs
        adata_copy.obs['plot_group'] = adata_copy.obs.index.map(diff_groups['plot_group'])
        
        # Determine color mapping
        color_values, color_label, is_continuous = _determine_color_mapping(
            adata_copy.obs['plot_group'], verbose=verbose
        )
        
        # Add color values to adata
        adata_copy.obs['color_values'] = color_values
        adata_copy.obs['color_type'] = 'continuous' if is_continuous else 'categorical'
        
    except Exception as e:
        if verbose:
            print(f"Could not retrieve grouping information: {e}. Using default UMAP coloring.")
        is_continuous = False
        color_label = None
    
    # Plot UMAP
    plt.figure(figsize=(10, 8))
    
    if 'color_values' in adata_copy.obs.columns:
        if is_continuous:
            # Manual plotting for continuous values with better control
            umap_coords = adata_copy.obsm['X_umap']
            sc = plt.scatter(umap_coords[:, 0], umap_coords[:, 1],
                           c=adata_copy.obs['color_values'],
                           cmap='viridis',
                           s=100,
                           alpha=0.8,
                           edgecolors='k',
                           linewidth=0.5)
            plt.colorbar(sc, label=color_label)
            plt.xlabel('UMAP1', fontsize=12)
            plt.ylabel('UMAP2', fontsize=12)
            plt.title(f'UMAP of Pseudobulk Expression (Colored by {color_label})', fontsize=14)
        else:
            # Use scanpy for categorical plotting
            sc.pl.umap(
                adata_copy, 
                color='plot_group',
                size=200,
                show=False,
                frameon=True,
                legend_loc='right margin',
                legend_fontsize=10
            )
            plt.title('UMAP of Pseudobulk Expression (Colored by Category)', fontsize=14)
    else:
        # Default UMAP without coloring
        sc.pl.umap(
            adata_copy,
            show=False,
            frameon=True,
            size=200
        )
        plt.title('UMAP of Pseudobulk Expression', fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'sample_relationship_umap_2D_sample.pdf')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    if verbose:
        print(f"UMAP plot saved to: {plot_path}")
    plt.show()
    
    return adata_copy