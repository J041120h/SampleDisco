from .visualization_helper import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from utils.Grouping import find_sample_grouping
from pandas.api.types import is_numeric_dtype
from matplotlib.colors import ListedColormap

def create_quantitative_colormap(values, colormap='viridis'):
    """Map unique values proportionally to [0, 1] on the given colormap; returns {value: rgba}."""
    unique_values = sorted(set(values))
    
    if len(unique_values) == 1:
        base_cmap = plt.get_cmap(colormap)
        return {unique_values[0]: base_cmap(0.5)}

    min_val, max_val = min(unique_values), max(unique_values)
    base_cmap = plt.get_cmap(colormap)
    
    color_map = {}
    for val in unique_values:
        normalized = (val - min_val) / (max_val - min_val)
        color_map[val] = base_cmap(normalized)
    
    return color_map

def is_quantitative_column(values, threshold_unique_ratio=0.5):
    """
    Return True if values should be treated as quantitative.

    Integers with <= 10 unique values are treated as ordinal (quantitative).
    Otherwise falls back to unique-ratio heuristic.
    """
    clean_values = [v for v in values if pd.notna(v)]
    
    if not clean_values:
        return False

    try:
        numeric_values = [float(v) for v in clean_values]
        unique_vals = sorted(set(numeric_values))

        if len(unique_vals) <= 10 and all(v == int(v) for v in unique_vals):
            return True

        unique_ratio = len(unique_vals) / len(clean_values)
        return unique_ratio > threshold_unique_ratio

    except (ValueError, TypeError):
        return False

def detect_analysis_method(adata, data_type='expression'):
    """
    Infer (method_name, component_prefix, data_key) from adata.uns keys.

    Checks method-specific sentinel keys first; falls back to inspecting
    column names of the unified result DataFrame.
    """
    if data_type == 'expression':
        unified_key = 'X_DR_expression'
        
        if unified_key not in adata.uns:
            raise KeyError(f"No dimension reduction results found in adata.uns['{unified_key}']. "
                          "Please run dimension reduction analysis first.")
        
        if 'X_spectral_expression_method' in adata.uns:
            return 'Spectral', 'Spectral', unified_key
        elif 'X_lsi_expression_method' in adata.uns:
            return 'LSI', 'LSI', unified_key
        elif 'X_pca_expression_method' in adata.uns:
            return 'PCA', 'PC', unified_key
        else:
            df = adata.uns[unified_key]
            first_col = df.columns[0]
            if 'Spectral' in first_col:
                return 'Spectral', 'Spectral', unified_key
            elif 'LSI' in first_col:
                return 'LSI', 'LSI', unified_key
            elif 'PC' in first_col:
                return 'PCA', 'PC', unified_key
            else:
                return 'Dimension Reduction', 'Component', unified_key
                
    elif data_type == 'proportion':
        unified_key = 'X_DR_proportion'
        
        if unified_key not in adata.uns:
            raise KeyError(f"No dimension reduction results found in adata.uns['{unified_key}']. "
                          "Please run dimension reduction analysis first.")
        
        return 'PCA', 'PC', unified_key
    
    else:
        raise ValueError("data_type must be 'expression' or 'proportion'")

def DR_visualization(adata, data_type='expression', figsize=(10, 8), 
                     point_size=50, alpha=0.7, save_path=None, 
                     title=None, grouping_columns=None, age_bin_size=None,
                     sample_col='sample', output_dir=None, show_sample_names=False, 
                     verbose=True):
    """
    Scatter plot of samples on the first two DR components.

    Always produces a plain sample-labels plot; if grouping_columns is given,
    also produces one plot per column (categorical → discrete legend,
    quantitative → viridis colorbar).
    """
    
    method_name, component_prefix, data_key = detect_analysis_method(adata, data_type)

    if data_type == 'expression':
        default_title = f'{method_name} - Expression Data'
        legend_prefix = f'{method_name} Groups'
    else:  # proportion
        default_title = f'{method_name} - Cell Proportion Data'
        legend_prefix = f'{method_name} Groups'
    
    analysis_df = adata.uns[data_key].copy()

    if analysis_df.shape[1] < 2:
        raise ValueError(f"Need at least 2 components for visualization (found {analysis_df.shape[1]})")
    
    comp1 = analysis_df.iloc[:, 0]
    comp2 = analysis_df.iloc[:, 1]
    sample_names = analysis_df.index
    comp1_name = analysis_df.columns[0]
    comp2_name = analysis_df.columns[1]
    
    if verbose:
        print(f"✓ Detected {method_name} analysis for {data_type} data")
        print(f"  → Using components: {comp1_name} vs {comp2_name}")
        print(f"  → Data shape: {analysis_df.shape}")
    
    plots_generated = []

    plt.figure(figsize=figsize)
    plt.scatter(comp1, comp2, s=point_size, alpha=alpha, 
               c='skyblue', edgecolors='black', linewidth=0.5)
    
    if show_sample_names:
        for i, sample in enumerate(sample_names):
            plt.annotate(sample, (comp1.iloc[i], comp2.iloc[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)

    plt.xlabel(f'{comp1_name}')
    plt.ylabel(f'{comp2_name}')

    sample_title = title if title else f'{default_title} - Sample Labels'
    plt.title(sample_title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        sample_filename = f"{method_name.lower()}_{data_type}_samples.png"
        sample_save_path = os.path.join(output_dir, sample_filename)
        plt.savefig(sample_save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Sample-labeled plot saved to: {sample_save_path}")
        plots_generated.append(sample_save_path)
    elif save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Sample-labeled plot saved to: {save_path}")
        plots_generated.append(save_path)
    
    plt.close()

    if grouping_columns:
        available_samples = list(sample_names)
        for grouping_col in grouping_columns:
            try:
                groups = find_sample_grouping(
                    adata, 
                    available_samples, 
                    grouping_columns=[grouping_col],  # Single column at a time
                    age_bin_size=age_bin_size,
                    sample_column=sample_col
                )
                
                sample_groups = [groups.get(sample, np.nan) for sample in sample_names]
                valid_groups = [g for g in sample_groups if pd.notna(g)]
                
                if not valid_groups:
                    if verbose:
                        print(f"Warning: No valid values found for grouping column '{grouping_col}'")
                    continue
                
                is_quantitative = is_quantitative_column(valid_groups)

                plt.figure(figsize=figsize)
                
                if is_quantitative:
                    # Handle quantitative data with proportional color mapping
                    if verbose:
                        print(f"✓ Detected quantitative data for '{grouping_col}'")
                        print(f"  → Using graduated color scale (viridis) for visualization")
                        print(f"  → Values range: {min(valid_groups)} to {max(valid_groups)}")
                    
                    numeric_groups = []
                    for g in sample_groups:
                        try:
                            numeric_groups.append(float(g) if pd.notna(g) else np.nan)
                        except (ValueError, TypeError):
                            numeric_groups.append(np.nan)
                    
                    valid_numeric = [g for g in numeric_groups if pd.notna(g)]
                    color_map = create_quantitative_colormap(valid_numeric, 'viridis')

                    colors = []
                    for g in numeric_groups:
                        if pd.notna(g):
                            colors.append(color_map[g])
                        else:
                            colors.append('lightgray')

                    scatter = plt.scatter(comp1, comp2, c=colors, s=point_size, alpha=alpha, 
                                        edgecolors='black', linewidth=0.5)
                    
                    unique_values = sorted([g for g in set(valid_numeric)])
                    if len(unique_values) > 1:
                        from matplotlib.cm import ScalarMappable
                        from matplotlib.colors import Normalize

                        norm = Normalize(vmin=min(unique_values), vmax=max(unique_values))
                        sm = ScalarMappable(norm=norm, cmap='viridis')
                        sm.set_array([])

                        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
                        cbar.set_ticks(unique_values)
                        cbar.set_ticklabels([f'{v:.0f}' if v == int(v) else f'{v:.2f}' for v in unique_values])
                        cbar.set_label(f'{grouping_col} (Quantitative)', rotation=270, labelpad=20)
                    
                else:
                    if verbose:
                        print(f"✓ Detected categorical data for '{grouping_col}'")
                        print(f"  → Using discrete color mapping for visualization")
                        print(f"  → Categories: {sorted(list(set([g for g in sample_groups if pd.notna(g)])))}")
                    
                    unique_groups = list(set([g for g in sample_groups if pd.notna(g)]))

                    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
                    group_colors = {group: colors[i] for i, group in enumerate(unique_groups)}
                    group_colors[np.nan] = 'lightgray'

                    for group in unique_groups:
                        mask = [g == group for g in sample_groups]
                        plt.scatter(comp1[mask], comp2[mask], 
                                   c=[group_colors[group]], 
                                   s=point_size, alpha=alpha, 
                                   label=str(group), edgecolors='black', linewidth=0.5)
                    
                    nan_mask = [pd.isna(g) for g in sample_groups]
                    if any(nan_mask):
                        plt.scatter(comp1[nan_mask], comp2[nan_mask], 
                                   c='lightgray', s=point_size, alpha=alpha,
                                   label='Missing', edgecolors='black', linewidth=0.5)
                    
                    legend_title = f'{legend_prefix} ({grouping_col})'
                    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
                
                if show_sample_names:
                    for i, sample in enumerate(sample_names):
                        plt.annotate(sample, (comp1.iloc[i], comp2.iloc[i]),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=8, alpha=0.8)

                plt.xlabel(f'{comp1_name}')
                plt.ylabel(f'{comp2_name}')
                
                group_title = title if title else f'{default_title} - Grouped by {grouping_col}'
                plt.title(group_title)
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                if output_dir:
                    group_filename = f"{method_name.lower()}_{data_type}_grouped_by_{grouping_col}.png"
                    group_save_path = os.path.join(output_dir, group_filename)
                    plt.savefig(group_save_path, dpi=300, bbox_inches='tight')
                    if verbose:
                        print(f"Group-colored plot ({grouping_col}) saved to: {group_save_path}")
                    plots_generated.append(group_save_path)
                
                plt.close()
                
                if verbose:
                    if is_quantitative:
                        print(f"Generated quantitative plot for '{grouping_col}' with graduated color scale")
                        unique_values = sorted([g for g in set(valid_groups)])
                        print(f"  → Color range: {min(unique_values):.2f} (dark) to {max(unique_values):.2f} (bright)")
                    else:
                        unique_groups = list(set([g for g in sample_groups if pd.notna(g)]))
                        print(f"Generated categorical plot for '{grouping_col}' with discrete colors")
                        print(f"  → {len(unique_groups)} distinct categories: {unique_groups}")
                
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not generate plot for grouping column '{grouping_col}': {str(e)}")
    
    if verbose:
        print(f"Used {method_name} for {data_type} data visualization")
        print(f"Total plots generated: {len(plots_generated)}")
        if plots_generated:
            print("Saved plots:")
            for plot_path in plots_generated:
                print(f"  - {plot_path}")

def get_available_data_types(adata):
    """Return list of (data_type, method_name) for DR results present in adata.uns."""
    available = []

    if 'X_DR_expression' in adata.uns:
        method_name, _, _ = detect_analysis_method(adata, 'expression')
        available.append(('expression', method_name))
    
    if 'X_DR_proportion' in adata.uns:
        method_name, _, _ = detect_analysis_method(adata, 'proportion')
        available.append(('proportion', method_name))
    
    return available

def DR_visualization_all(adata, figsize=(10, 8), point_size=50,
                        alpha=0.7, output_dir=None, grouping_columns=None,
                        age_bin_size=None, sample_col='sample', show_sample_names=False,
                        verbose=True):
    """Call DR_visualization for every data type that has results in adata.uns."""
    if verbose:
        print("=== Dimension Reduction Visualization - All Available Data ===")

    available_data = get_available_data_types(adata)
    
    if not available_data:
        print("No dimension reduction results found. Please run dimension reduction analysis first.")
        return
    
    if verbose:
        print(f"Available data types:")
        for data_type, method in available_data:
            print(f"  - {data_type}: {method}")
    
    for data_type, method in available_data:
        if verbose:
            print(f"\n--- Generating {method} {data_type.title()} plots ---")
        try:
            DR_visualization(adata, data_type=data_type, 
                           figsize=figsize, point_size=point_size, alpha=alpha, 
                           grouping_columns=grouping_columns, age_bin_size=age_bin_size, 
                           sample_col=sample_col, output_dir=output_dir, 
                           show_sample_names=show_sample_names, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"Error generating {method} {data_type} plots: {str(e)}")
    
    if verbose:
        print("\n=== Visualization Complete ===")