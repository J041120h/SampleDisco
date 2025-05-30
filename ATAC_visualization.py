from Visualization import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from Grouping import find_sample_grouping
from pandas.api.types import is_numeric_dtype
from matplotlib.colors import ListedColormap

def create_quantitative_colormap(values, colormap='viridis'):
    """
    Create a color mapping for quantitative/ordinal values using proportional spacing.
    This ensures consistent colors regardless of the actual range (0-3, 0-7, 0-10, etc.)
    
    Parameters:
    -----------
    values : array-like
        Quantitative values to map
    colormap : str, default 'viridis'
        Matplotlib colormap name
        
    Returns:
    --------
    dict : Mapping from value to color
    """
    unique_values = sorted(set(values))
    
    if len(unique_values) == 1:
        # Single value case
        base_cmap = plt.cm.get_cmap(colormap)
        return {unique_values[0]: base_cmap(0.5)}
    
    # Map values proportionally to 0-1 range
    min_val, max_val = min(unique_values), max(unique_values)
    base_cmap = plt.cm.get_cmap(colormap)
    
    color_map = {}
    for val in unique_values:
        # Normalize to 0-1
        normalized = (val - min_val) / (max_val - min_val)
        color_map[val] = base_cmap(normalized)
    
    return color_map

def is_quantitative_column(values, threshold_unique_ratio=0.5):
    """
    Determine if a column should be treated as quantitative.
    
    Parameters:
    -----------
    values : array-like
        Column values to check
    threshold_unique_ratio : float, default 0.5
        If unique_values/total_values > threshold, treat as quantitative
        
    Returns:
    --------
    bool : True if should be treated as quantitative
    """
    # Remove NaN values for analysis
    clean_values = [v for v in values if pd.notna(v)]
    
    if not clean_values:
        return False
    
    # Check if all values are numeric
    try:
        numeric_values = [float(v) for v in clean_values]
        
        # Check if values look like discrete stages/categories
        # (e.g., 0,1,2,3,4 or 0,2,4,6,8)
        unique_vals = sorted(set(numeric_values))
        
        # If few unique values and they're integers, likely ordinal/categorical
        if len(unique_vals) <= 10 and all(v == int(v) for v in unique_vals):
            return True
            
        # If many unique values relative to total, likely continuous
        unique_ratio = len(unique_vals) / len(clean_values)
        return unique_ratio > threshold_unique_ratio
        
    except (ValueError, TypeError):
        # Non-numeric values
        return False

def ATAC_visualization(adata, analysis_type='pca', pca_type='expression', figsize=(10, 8), 
                      point_size=50, alpha=0.7, save_path=None, 
                      title=None, grouping_columns=None, age_bin_size=None,
                      sample_col='sample', output_dir=None, show_sample_names=False, 
                      verbose=True):
    """
    Visualize samples using first and second principal components from PCA, LSI, or Spectral analysis.
    Now supports separate visualization of each analysis type when available.
    Always generates a plot labeled by sample names, plus additional plots colored by grouping columns.
    
    Handles both categorical and quantitative grouping columns:
    - Categorical: Uses discrete colors with legend
    - Quantitative: Uses proportional color mapping with colorbar
    
    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object containing analysis results
    analysis_type : str, default 'pca'
        Type of analysis to visualize ('pca', 'lsi', or 'spectral')
    pca_type : str, default 'expression'
        Type of analysis to visualize ('expression' or 'proportion')
    figsize : tuple, default (10, 8)
        Figure size for the plot
    point_size : int, default 50
        Size of scatter plot points
    alpha : float, default 0.7
        Transparency of points
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Custom title for the plot
    grouping_columns : list, optional
        List of columns to use for grouping. If provided, generates separate plots for each column.
    age_bin_size : int, optional
        Bin size for age grouping if 'age' is in grouping_columns
    sample_col : str, default 'sample'
        Column name for sample identification
    output_dir : str, optional
        Directory to save the figure (overrides save_path if provided)
    show_sample_names : bool, default False
        Whether to show sample names as annotations on the plots
    verbose : bool, default True
        Whether to print detailed information about data type detection and visualization choices
    verbose : bool, default True
        Whether to print detailed information about data type detection and visualization choices
    """
    
    # Determine which analysis results to use based on analysis_type
    if pca_type == 'expression':
        if analysis_type == 'spectral':
            if 'X_spectral_expression' in adata.uns:
                analysis_key = 'X_spectral_expression'
                method_name = 'Spectral'
                default_title = 'Spectral Analysis - Expression Data'
                component_names = ('Spectral1', 'Spectral2')
                legend_prefix = 'Spectral Groups'
            else:
                raise KeyError("Spectral analysis results not found in adata.uns['X_spectral_expression']. "
                              "Please run spectral analysis first.")
        elif analysis_type == 'lsi':
            if 'X_DR_expression' in adata.uns:
                analysis_key = 'X_DR_expression'
                method_name = 'LSI'
                default_title = 'LSI - Expression Data'
                component_names = ('LSI1', 'LSI2')
                legend_prefix = 'LSI Groups'
            else:
                raise KeyError("LSI analysis results not found in adata.uns['X_DR_expression']. "
                              "Please run LSI analysis first.")
        elif analysis_type == 'pca':
            if 'X_pca_expression' in adata.uns:
                analysis_key = 'X_pca_expression'
                method_name = 'PCA'
                default_title = 'PCA - Expression Data'
                component_names = ('PC1', 'PC2')
                legend_prefix = 'PCA Groups'
            else:
                raise KeyError("PCA results not found in adata.uns['X_pca_expression']. "
                              "Please run PCA analysis first.")
        else:
            raise ValueError("analysis_type must be 'pca', 'lsi', or 'spectral'")
        
        group_key = 'X_pca_expression_groups'  # Groups are still stored under PCA key
        
    elif pca_type == 'proportion':
        # For proportion, only PCA is typically used
        if analysis_type != 'pca':
            raise ValueError("For proportion data, only 'pca' analysis_type is supported")
        
        if 'X_pca_proportion' in adata.uns:
            analysis_key = 'X_pca_proportion'
            method_name = 'PCA'
            default_title = 'PCA - Cell Proportion Data'
            component_names = ('PC1', 'PC2')
            legend_prefix = 'PCA Groups'
        else:
            raise KeyError("PCA results not found in adata.uns['X_pca_proportion']. "
                          "Please run PCA analysis first.")
        
        group_key = None  # proportion PCA doesn't store groups
    else:
        raise ValueError("pca_type must be 'expression' or 'proportion'")
    
    # Get analysis data
    analysis_df = adata.uns[analysis_key].copy()
    
    # Check if we have at least 2 components
    if analysis_df.shape[1] < 2:
        raise ValueError(f"Need at least 2 components for visualization (found {analysis_df.shape[1]})")
    
    # Get first and second components
    comp1 = analysis_df.iloc[:, 0]  # First component
    comp2 = analysis_df.iloc[:, 1]  # Second component
    sample_names = analysis_df.index
    
    # Generate plots
    plots_generated = []
    
    # 1. Always generate a plot labeled by sample names (no coloring by groups)
    plt.figure(figsize=figsize)
    plt.scatter(comp1, comp2, s=point_size, alpha=alpha, 
               c='skyblue', edgecolors='black', linewidth=0.5)
    
    # Add sample labels only if requested
    if show_sample_names:
        for i, sample in enumerate(sample_names):
            plt.annotate(sample, (comp1.iloc[i], comp2.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    # Set labels and title using component names
    plt.xlabel(f'{component_names[0]} ({analysis_df.columns[0]})')
    plt.ylabel(f'{component_names[1]} ({analysis_df.columns[1]})')
    
    sample_title = title if title else f'{default_title} - Sample Labels'
    plt.title(sample_title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save sample-labeled plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        sample_filename = f"{method_name.lower()}_{pca_type}_samples.png"
        sample_save_path = os.path.join(output_dir, sample_filename)
        plt.savefig(sample_save_path, dpi=300, bbox_inches='tight')
        print(f"Sample-labeled plot saved to: {sample_save_path}")
        plots_generated.append(sample_save_path)
    elif save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample-labeled plot saved to: {save_path}")
        plots_generated.append(save_path)
    
    plt.close()  # Close the figure instead of showing it
    
    # 2. Generate additional plots for each grouping column
    if grouping_columns:
        available_samples = list(sample_names)
        for grouping_col in grouping_columns:
            try:
                # Get grouping for this specific column
                groups = find_sample_grouping(
                    adata, 
                    available_samples, 
                    grouping_columns=[grouping_col],  # Single column at a time
                    age_bin_size=age_bin_size,
                    sample_column=sample_col
                )
                
                # Align groups with sample names
                sample_groups = [groups.get(sample, np.nan) for sample in sample_names]
                
                # Remove NaN values for analysis
                valid_groups = [g for g in sample_groups if pd.notna(g)]
                
                if not valid_groups:
                    print(f"Warning: No valid values found for grouping column '{grouping_col}'")
                    continue
                
                # Determine if this is a quantitative column
                is_quantitative = is_quantitative_column(valid_groups)
                
                # Create new figure for this grouping
                plt.figure(figsize=figsize)
                
                if is_quantitative:
                    # Handle quantitative data with proportional color mapping
                    if verbose:
                        print(f"✓ Detected quantitative data for '{grouping_col}'")
                        print(f"  → Using graduated color scale (viridis) for visualization")
                        print(f"  → Values range: {min(valid_groups)} to {max(valid_groups)}")
                    
                    # Convert to numeric, handling NaN
                    numeric_groups = []
                    for g in sample_groups:
                        try:
                            numeric_groups.append(float(g) if pd.notna(g) else np.nan)
                        except (ValueError, TypeError):
                            numeric_groups.append(np.nan)
                    
                    # Create color mapping for non-NaN values
                    valid_numeric = [g for g in numeric_groups if pd.notna(g)]
                    color_map = create_quantitative_colormap(valid_numeric, 'viridis')
                    
                    # Create colors array
                    colors = []
                    for g in numeric_groups:
                        if pd.notna(g):
                            colors.append(color_map[g])
                        else:
                            colors.append('lightgray')  # Color for missing values
                    
                    # Create scatter plot
                    scatter = plt.scatter(comp1, comp2, c=colors, s=point_size, alpha=alpha, 
                                        edgecolors='black', linewidth=0.5)
                    
                    # Create custom colorbar for quantitative data
                    unique_values = sorted([g for g in set(valid_numeric)])
                    if len(unique_values) > 1:
                        # Create a mappable for colorbar using the actual scatter plot
                        from matplotlib.cm import ScalarMappable
                        from matplotlib.colors import Normalize
                        
                        norm = Normalize(vmin=min(unique_values), vmax=max(unique_values))
                        sm = ScalarMappable(norm=norm, cmap='viridis')
                        sm.set_array([])  # Required for ScalarMappable
                        
                        # Create colorbar with discrete ticks, linked to current axes
                        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
                        cbar.set_ticks(unique_values)
                        cbar.set_ticklabels([f'{v:.0f}' if v == int(v) else f'{v:.2f}' for v in unique_values])
                        cbar.set_label(f'{grouping_col} (Quantitative)', rotation=270, labelpad=20)
                    
                else:
                    # Handle categorical data with discrete colors
                    if verbose:
                        print(f"✓ Detected categorical data for '{grouping_col}'")
                        print(f"  → Using discrete color mapping for visualization")
                        print(f"  → Categories: {sorted(list(set([g for g in sample_groups if pd.notna(g)])))}")
                    
                    unique_groups = list(set([g for g in sample_groups if pd.notna(g)]))
                    
                    # Create color map for categorical data
                    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
                    group_colors = {group: colors[i] for i, group in enumerate(unique_groups)}
                    group_colors[np.nan] = 'lightgray'  # Color for missing values
                    
                    # Plot each group
                    for group in unique_groups:
                        mask = [g == group for g in sample_groups]
                        plt.scatter(comp1[mask], comp2[mask], 
                                   c=[group_colors[group]], 
                                   s=point_size, alpha=alpha, 
                                   label=str(group), edgecolors='black', linewidth=0.5)
                    
                    # Handle missing values if any
                    nan_mask = [pd.isna(g) for g in sample_groups]
                    if any(nan_mask):
                        plt.scatter(comp1[nan_mask], comp2[nan_mask], 
                                   c='lightgray', s=point_size, alpha=alpha,
                                   label='Missing', edgecolors='black', linewidth=0.5)
                    
                    # Add legend for categorical data
                    legend_title = f'{legend_prefix} ({grouping_col})'
                    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Add sample labels only if requested
                if show_sample_names:
                    for i, sample in enumerate(sample_names):
                        plt.annotate(sample, (comp1.iloc[i], comp2.iloc[i]), 
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=8, alpha=0.8)
                
                # Set labels and title using component names
                plt.xlabel(f'{component_names[0]} ({analysis_df.columns[0]})')
                plt.ylabel(f'{component_names[1]} ({analysis_df.columns[1]})')
                
                group_title = title if title else f'{default_title} - Grouped by {grouping_col}'
                plt.title(group_title)
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save group-colored plot
                if output_dir:
                    group_filename = f"{method_name.lower()}_{pca_type}_grouped_by_{grouping_col}.png"
                    group_save_path = os.path.join(output_dir, group_filename)
                    plt.savefig(group_save_path, dpi=300, bbox_inches='tight')
                    print(f"Group-colored plot ({grouping_col}) saved to: {group_save_path}")
                    plots_generated.append(group_save_path)
                
                plt.close()  # Close the figure instead of showing it
                
                if is_quantitative:
                    if verbose:
                        print(f"Generated quantitative plot for '{grouping_col}' with graduated color scale")
                        print(f"  → Color range: {min(unique_values):.2f} (dark) to {max(unique_values):.2f} (bright)")
                else:
                    if verbose:
                        print(f"Generated categorical plot for '{grouping_col}' with discrete colors")
                        print(f"  → {len(unique_groups)} distinct categories: {unique_groups}")
                
            except Exception as e:
                print(f"Warning: Could not generate plot for grouping column '{grouping_col}': {str(e)}")
    
    # Print summary
    if verbose:
        print(f"Used {method_name} for visualization")
        print(f"Total plots generated: {len(plots_generated)}")
        if plots_generated:
            print("Saved plots:")
            for plot_path in plots_generated:
                print(f"  - {plot_path}")

def get_available_analyses(adata, pca_type='expression'):
    """
    Detect which analysis types are available for the given data type.
    
    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object containing analysis results
    pca_type : str, default 'expression'
        Type of analysis to check ('expression' or 'proportion')
    
    Returns:
    --------
    list : Available analysis types
    """
    available = []
    
    if pca_type == 'expression':
        if 'X_spectral_expression' in adata.uns:
            available.append('spectral')
        if 'X_DR_expression' in adata.uns:
            available.append('lsi')
        if 'X_pca_expression' in adata.uns:
            available.append('pca')
    elif pca_type == 'proportion':
        if 'X_pca_proportion' in adata.uns:
            available.append('pca')
    
    return available

def ATAC_visualization_all(adata, figsize=(10, 8), point_size=50, 
                          alpha=0.7, output_dir=None, grouping_columns=None, 
                          age_bin_size=None, sample_col='sample', show_sample_names=False, 
                          verbose=True):
    """
    Generate separate plots for all available analysis types (Spectral, LSI, PCA) 
    for both expression and proportion data when available.
    
    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object containing analysis results
    figsize : tuple, default (10, 8)
        Figure size for the plots
    point_size : int, default 50
        Size of scatter plot points
    alpha : float, default 0.7
        Transparency of points
    output_dir : str, optional
        Directory to save the figures
    grouping_columns : list, optional
        List of columns to use for grouping
    age_bin_size : int, optional
        Bin size for age grouping if 'age' is in grouping_columns
    sample_col : str, default 'sample'
        Column name for sample identification
    show_sample_names : bool, default False
        Whether to show sample names as annotations on the plots
    """
    print("=== ATAC Visualization - All Available Analyses ===")
    
    # Check expression analyses
    expression_analyses = get_available_analyses(adata, 'expression')
    print(f"Available expression analyses: {expression_analyses}")
    
    for analysis_type in expression_analyses:
        print(f"\n--- Generating {analysis_type.upper()} Expression plots ---")
        try:
            ATAC_visualization(adata, analysis_type=analysis_type, pca_type='expression', 
                             figsize=figsize, point_size=point_size, alpha=alpha, 
                             grouping_columns=grouping_columns, age_bin_size=age_bin_size, 
                             sample_col=sample_col, output_dir=output_dir, 
                             show_sample_names=show_sample_names, verbose=verbose)
        except Exception as e:
            print(f"Error generating {analysis_type} expression plots: {str(e)}")
    
    # Check proportion analyses
    proportion_analyses = get_available_analyses(adata, 'proportion')
    print(f"\nAvailable proportion analyses: {proportion_analyses}")
    
    for analysis_type in proportion_analyses:
        print(f"\n--- Generating {analysis_type.upper()} Proportion plots ---")
        try:
            ATAC_visualization(adata, analysis_type=analysis_type, pca_type='proportion', 
                             figsize=figsize, point_size=point_size, alpha=alpha, 
                             grouping_columns=grouping_columns, age_bin_size=age_bin_size, 
                             sample_col=sample_col, output_dir=output_dir, 
                             show_sample_names=show_sample_names, verbose=verbose)
        except Exception as e:
            print(f"Error generating {analysis_type} proportion plots: {str(e)}")
    
    print("\n=== Visualization Complete ===")

# Keep the old function name for backward compatibility
def ATAC_visualization_both(adata, figsize=(10, 8), point_size=50, 
                          alpha=0.7, output_dir=None, grouping_columns=None, 
                          age_bin_size=None, sample_col='sample', show_sample_names=False, 
                          verbose=True):
    """
    Backward compatibility wrapper - now calls ATAC_visualization_all.
    """
    print("Note: ATAC_visualization_both is deprecated. Use ATAC_visualization_all for all analysis types.")
    ATAC_visualization_all(adata, figsize=figsize, point_size=point_size, 
                          alpha=alpha, output_dir=output_dir, grouping_columns=grouping_columns, 
                          age_bin_size=age_bin_size, sample_col=sample_col, 
                          show_sample_names=show_sample_names, verbose=verbose)