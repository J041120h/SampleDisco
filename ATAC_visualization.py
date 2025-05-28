from Visualization import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from Grouping import find_sample_grouping
from pandas.api.types import is_numeric_dtype

def ATAC_visualization(adata, pca_type='expression', figsize=(10, 8), 
                      point_size=50, alpha=0.7, save_path=None, 
                      title=None, grouping_columns=None, age_bin_size=None,
                      sample_col='sample', output_dir=None):
    """
    Visualize samples using first and second principal components from PCA or LSI analysis.
    Prioritizes LSI (X_DR_expression) over PCA when available.
    Always generates a plot labeled by sample names, plus additional plots colored by grouping columns.
    
    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object containing PCA/LSI results
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
    """
    
    # Determine which analysis results to use (prioritize LSI over PCA)
    if pca_type == 'expression':
        # Check for LSI first, then PCA
        if 'X_DR_expression' in adata.uns:
            print("Using LSI analysis for expression data")
            analysis_key = 'X_DR_expression'
            analysis_type = 'LSI'
            default_title = 'LSI - Expression Data'
            component_names = ('LSI1', 'LSI2')
            legend_prefix = 'LSI Groups'
        elif 'X_pca_expression' in adata.uns:
            analysis_key = 'X_pca_expression'
            analysis_type = 'PCA'
            default_title = 'PCA - Expression Data'
            component_names = ('PC1', 'PC2')
            legend_prefix = 'PCA Groups'
        else:
            raise KeyError("Neither LSI (X_DR_expression) nor PCA (X_pca_expression) results found. "
                          "Please run dimensionality reduction analysis first.")
        
        group_key = 'X_pca_expression_groups'  # Groups are still stored under PCA key
        
    elif pca_type == 'proportion':
        # For proportion, only check PCA (LSI not typically used for proportions)
        if 'X_pca_proportion' in adata.uns:
            analysis_key = 'X_pca_proportion'
            analysis_type = 'PCA'
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
    
    # Add sample labels
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
        sample_filename = f"{analysis_type.lower()}_{pca_type}_samples.png"
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
                
                # Create new figure for this grouping
                plt.figure(figsize=figsize)
                
                # Align groups with sample names
                sample_groups = [groups.get(sample, 'Unknown') for sample in sample_names]
                unique_groups = list(set(sample_groups))
                
                # Create color map
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
                group_colors = {group: colors[i] for i, group in enumerate(unique_groups)}
                
                # Plot each group
                for group in unique_groups:
                    mask = [g == group for g in sample_groups]
                    plt.scatter(comp1[mask], comp2[mask], 
                               c=[group_colors[group]], 
                               s=point_size, alpha=alpha, 
                               label=group, edgecolors='black', linewidth=0.5)
                
                # Add sample labels
                for i, sample in enumerate(sample_names):
                    plt.annotate(sample, (comp1.iloc[i], comp2.iloc[i]), 
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, alpha=0.8)
                
                # Set labels and title using component names
                plt.xlabel(f'{component_names[0]} ({analysis_df.columns[0]})')
                plt.ylabel(f'{component_names[1]} ({analysis_df.columns[1]})')
                
                group_title = title if title else f'{default_title} - Grouped by {grouping_col}'
                plt.title(group_title)
                
                # Use analysis-specific legend title
                legend_title = f'{legend_prefix} ({grouping_col})'
                plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save group-colored plot
                if output_dir:
                    group_filename = f"{analysis_type.lower()}_{pca_type}_grouped_by_{grouping_col}.png"
                    group_save_path = os.path.join(output_dir, group_filename)
                    plt.savefig(group_save_path, dpi=300, bbox_inches='tight')
                    print(f"Group-colored plot ({grouping_col}) saved to: {group_save_path}")
                    plots_generated.append(group_save_path)
                
                plt.close()  # Close the figure instead of showing it
                
                print(f"Generated plot for grouping column '{grouping_col}' with groups: {unique_groups}")
                
            except Exception as e:
                print(f"Warning: Could not generate plot for grouping column '{grouping_col}': {str(e)}")
    
    # Print summary
    print(f"Used {analysis_type} for visualization")
    print(f"Total plots generated: {len(plots_generated)}")
    if plots_generated:
        print("Saved plots:")
        for plot_path in plots_generated:
            print(f"  - {plot_path}")

def ATAC_visualization_both(adata, figsize=(10, 8), point_size=50, 
                          alpha=0.7, output_dir=None, grouping_columns=None, 
                          age_bin_size=None, sample_col='sample'):
    """
    Generate separate plots for both expression and proportion analysis.
    For expression: prioritizes LSI (X_DR_expression) over PCA (X_pca_expression)
    For proportion: uses PCA (X_pca_proportion) only
    
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
    """
    # Plot expression analysis (LSI preferred, fallback to PCA)
    has_lsi = 'X_DR_expression' in adata.uns
    has_pca_expr = 'X_pca_expression' in adata.uns
    if has_lsi or has_pca_expr:
        analysis_type = "LSI" if has_lsi else "PCA"
        print(f"Generating Expression {analysis_type} plot...")
        ATAC_visualization(adata, pca_type='expression', figsize=figsize,
                         point_size=point_size, alpha=alpha, grouping_columns=grouping_columns,
                         age_bin_size=age_bin_size, sample_col=sample_col,
                         output_dir=output_dir)
    else:
        print("Neither LSI (X_DR_expression) nor PCA (X_pca_expression) results found")
    
    # Plot proportion PCA
    if 'X_pca_proportion' in adata.uns:
        print("Generating Proportion PCA plot...")
        ATAC_visualization(adata, pca_type='proportion', figsize=figsize,
                         point_size=point_size, alpha=alpha, grouping_columns=grouping_columns,
                         age_bin_size=age_bin_size, sample_col=sample_col,
                         output_dir=output_dir)
    else:
        print("Proportion PCA results not found in adata.uns")