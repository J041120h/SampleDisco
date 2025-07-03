import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import make_interp_spline
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def visualize_cell_proportion_pseudotime(
    pseudotime: Dict[str, float],
    output_dir: str,
    verbose: bool = False,
    top_n_celltypes: int = 10,
    smooth_curve: bool = True,
    correlation_method: str = 'spearman'
) -> pd.DataFrame:
    """
    Visualize how cell type proportions change with pseudotime.
    
    Parameters
    ----------
    final_adata : AnnData
        Final AnnData object containing cell proportion information
    pseudotime : Dict[str, float]
        Dictionary mapping sample names to pseudotime values
    output_dir : str
        Output directory for saving plots
    verbose : bool
        Whether to print progress information
    top_n_celltypes : int
        Number of top variable cell types to highlight
    smooth_curve : bool
        Whether to add smoothed trend lines
    correlation_method : str
        Method for correlation calculation ('pearson' or 'spearman')
    
    Returns
    -------
    correlation_df : pd.DataFrame
        DataFrame containing correlation statistics for each cell type
    """
    
    if verbose:
        print("Starting cell proportion vs pseudotime visualization...")
    
    # Create output directory
    viz_dir = os.path.join(output_dir, 'proportion_pseudotime_viz')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Extract cell proportions from the stored pseudobulk data
    # Based on the code, proportions should be in the pseudobulk directory
    pseudobulk_dir = os.path.join(output_dir, 'pseudobulk')
    proportion_file = os.path.join(pseudobulk_dir, 'proportion.csv')
    
    if os.path.exists(proportion_file):
        if verbose:
            print(f"Loading cell proportions from {proportion_file}")
        cell_proportion_df = pd.read_csv(proportion_file, index_col=0)
    else:
        # Try to reconstruct from final_adata if file doesn't exist
        if verbose:
            print("Proportion file not found, attempting to reconstruct from adata...")
        # This would need the original cell type information which might not be in final_adata
        raise FileNotFoundError(f"Cell proportion file not found at {proportion_file}")
    
    # Align samples between pseudotime and proportions
    common_samples = list(set(pseudotime.keys()) & set(cell_proportion_df.columns))
    if len(common_samples) == 0:
        raise ValueError("No common samples found between pseudotime and cell proportions")
    
    if verbose:
        print(f"Found {len(common_samples)} common samples")
    
    # Sort samples by pseudotime
    sorted_samples = sorted(common_samples, key=lambda x: pseudotime[x])
    pseudotime_values = [pseudotime[s] for s in sorted_samples]
    
    # Prepare proportion data
    proportion_matrix = cell_proportion_df[sorted_samples]
    
    # Calculate correlation between each cell type and pseudotime
    correlations = []
    for cell_type in proportion_matrix.index:
        proportions = proportion_matrix.loc[cell_type].values
        
        if correlation_method == 'pearson':
            corr, p_value = pearsonr(pseudotime_values, proportions)
        else:
            corr, p_value = spearmanr(pseudotime_values, proportions)
        
        correlations.append({
            'cell_type': cell_type,
            'correlation': corr,
            'p_value': p_value,
            'abs_correlation': abs(corr)
        })
    
    correlation_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    
    if verbose:
        print("\nTop cell types by correlation with pseudotime:")
        print(correlation_df.head(10))
    
    # Save correlation results
    correlation_df.to_csv(os.path.join(viz_dir, 'celltype_pseudotime_correlations.csv'), index=False)
    
    # 1. Overview heatmap
    if verbose:
        print("\nCreating overview heatmap...")
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with samples ordered by pseudotime
    sns.heatmap(
        proportion_matrix,
        cmap='viridis',
        cbar_kws={'label': 'Cell Type Proportion'},
        xticklabels=False,  # Too many samples to show labels
        yticklabels=True
    )
    
    plt.title('Cell Type Proportions Across Pseudotime')
    plt.xlabel('Samples (ordered by pseudotime →)')
    plt.ylabel('Cell Types')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'proportion_heatmap_pseudotime.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top variable cell types line plot
    if verbose:
        print("Creating line plots for top variable cell types...")
    
    top_celltypes = correlation_df.head(top_n_celltypes)['cell_type'].tolist()
    
    # Create figure with subplots
    n_cols = 2
    n_rows = (len(top_celltypes) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, cell_type in enumerate(top_celltypes):
        ax = axes[idx]
        proportions = proportion_matrix.loc[cell_type].values
        
        # Scatter plot
        ax.scatter(pseudotime_values, proportions, alpha=0.6, s=50)
        
        # Add smooth curve if requested
        if smooth_curve and len(pseudotime_values) > 3:
            try:
                # Sort for interpolation
                sorted_indices = np.argsort(pseudotime_values)
                x_sorted = np.array(pseudotime_values)[sorted_indices]
                y_sorted = proportions[sorted_indices]
                
                # Create smooth curve
                x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
                spl = make_interp_spline(x_sorted, y_sorted, k=3)
                y_smooth = spl(x_smooth)
                
                ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, alpha=0.8)
            except:
                # If smoothing fails, just add a linear fit
                z = np.polyfit(pseudotime_values, proportions, 1)
                p = np.poly1d(z)
                ax.plot(sorted(pseudotime_values), p(sorted(pseudotime_values)), 
                       "r--", linewidth=2, alpha=0.8)
        
        # Get correlation info
        corr_info = correlation_df[correlation_df['cell_type'] == cell_type].iloc[0]
        
        ax.set_xlabel('Pseudotime')
        ax.set_ylabel('Proportion')
        ax.set_title(f'{cell_type}\n(r={corr_info["correlation"]:.3f}, p={corr_info["p_value"]:.3e})')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(top_celltypes), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'Top {len(top_celltypes)} Cell Types by Correlation with Pseudotime', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'top_celltypes_pseudotime_trends.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Combined plot for all cell types
    if verbose:
        print("Creating combined plot for all cell types...")
    
    plt.figure(figsize=(10, 8))
    
    # Plot each cell type with transparency
    for cell_type in proportion_matrix.index:
        proportions = proportion_matrix.loc[cell_type].values
        plt.plot(pseudotime_values, proportions, alpha=0.5, linewidth=1)
    
    # Highlight top cell types
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_celltypes)))
    for idx, cell_type in enumerate(top_celltypes):
        proportions = proportion_matrix.loc[cell_type].values
        plt.plot(pseudotime_values, proportions, 
                color=colors[idx], linewidth=2.5, label=cell_type, alpha=0.9)
    
    plt.xlabel('Pseudotime')
    plt.ylabel('Cell Type Proportion')
    plt.title('Cell Type Proportions Along Pseudotime Trajectory')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'all_celltypes_pseudotime_overlay.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Stacked area plot
    if verbose:
        print("Creating stacked area plot...")
    
    plt.figure(figsize=(12, 6))
    
    # Prepare data for stacked area plot
    # Use only top cell types for clarity
    top_proportion_matrix = proportion_matrix.loc[top_celltypes]
    
    # Create stacked area plot
    ax = plt.gca()
    ax.stackplot(pseudotime_values, top_proportion_matrix.values,
                labels=top_celltypes, alpha=0.8)
    
    plt.xlabel('Pseudotime')
    plt.ylabel('Cell Type Proportion')
    plt.title(f'Stacked Cell Type Proportions Along Pseudotime (Top {len(top_celltypes)} Cell Types)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(min(pseudotime_values), max(pseudotime_values))
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'stacked_proportions_pseudotime.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Correlation barplot
    if verbose:
        print("Creating correlation barplot...")
    
    plt.figure(figsize=(10, 8))
    
    # Plot correlations
    y_pos = np.arange(len(correlation_df))
    colors = ['red' if x < 0 else 'blue' for x in correlation_df['correlation']]
    
    plt.barh(y_pos, correlation_df['correlation'], color=colors, alpha=0.7)
    plt.yticks(y_pos, correlation_df['cell_type'])
    plt.xlabel(f'{correlation_method.capitalize()} Correlation with Pseudotime')
    plt.title('Cell Type Correlations with Pseudotime')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'correlation_barplot.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"\nVisualization complete! Results saved to {viz_dir}")
        print(f"Generated files:")
        print(f"  - proportion_heatmap_pseudotime.pdf")
        print(f"  - top_celltypes_pseudotime_trends.pdf")
        print(f"  - all_celltypes_pseudotime_overlay.pdf")
        print(f"  - stacked_proportions_pseudotime.pdf")
        print(f"  - correlation_barplot.pdf")
        print(f"  - celltype_pseudotime_correlations.csv")
    
    return correlation_df