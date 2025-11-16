import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from scipy import stats
import os
from typing import Optional, Tuple, Dict, Any, List, Union
import warnings

# Set style for better aesthetics
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def detect_data_type(values: np.ndarray) -> Tuple[str, List]:
    """
    Enhanced data type detection with better handling of edge cases.
    """
    # Remove NaN values for analysis
    valid_values = values[pd.notna(values)]
    
    if len(valid_values) == 0:
        return 'categorical', []
    
    unique_values = np.unique(valid_values)
    n_unique = len(unique_values)
    
    # Try to convert to numbers
    try:
        numeric_values = pd.to_numeric(valid_values, errors='coerce')
        n_numeric = np.sum(~pd.isna(numeric_values))
        
        # If most values are numeric
        if n_numeric / len(valid_values) > 0.8:
            # Check if it's essentially categorical (few unique values)
            if n_unique <= 10 and n_unique / len(valid_values) < 0.1:
                # Check for binary 0/1
                if n_unique == 2 and set(numeric_values[~pd.isna(numeric_values)]) == {0, 1}:
                    return 'categorical', unique_values.tolist()
                # For other small sets, check if they're ordinal
                if all(x.is_integer() for x in numeric_values[~pd.isna(numeric_values)]):
                    return 'ordinal', unique_values.tolist()
            return 'numerical', unique_values.tolist()
    except:
        pass
    
    return 'categorical', unique_values.tolist()


def create_gradient_colormap(name: str = 'viridis', n_colors: int = 256) -> LinearSegmentedColormap:
    """
    Create a smooth gradient colormap with customization options.
    """
    base_cmaps = {
        'viridis': plt.cm.viridis,
        'plasma': plt.cm.plasma,
        'coolwarm': plt.cm.coolwarm,
        'RdYlBu': plt.cm.RdYlBu_r,
        'spectral': plt.cm.Spectral_r,
        'turbo': plt.cm.turbo
    }
    
    if name in base_cmaps:
        return base_cmaps[name]
    
    # Create custom gradient if name not found
    colors = ['#440154', '#31688e', '#35b779', '#fde724']  # Viridis-like
    return LinearSegmentedColormap.from_list('custom', colors, N=n_colors)


def add_density_contours(ax, x: np.ndarray, y: np.ndarray, levels: int = 5, 
                         alpha: float = 0.3, colors: str = 'gray'):
    """
    Add density contours to show data concentration.
    """
    try:
        from scipy.stats import gaussian_kde
        
        # Calculate density
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        
        # Create grid
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        positions = np.vstack([xx.ravel(), yy.ravel()])
        
        # Evaluate KDE
        density = kde(positions).reshape(xx.shape)
        
        # Add contours
        ax.contour(xx, yy, density, levels=levels, colors=colors, 
                  alpha=alpha, linewidths=1)
    except Exception as e:
        warnings.warn(f"Could not add density contours: {e}")


def add_confidence_ellipses(ax, x: np.ndarray, y: np.ndarray, labels: np.ndarray,
                           confidence: float = 0.95, alpha: float = 0.2):
    """
    Add confidence ellipses for each group.
    """
    unique_labels = np.unique(labels[pd.notna(labels)])
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if np.sum(mask) < 3:  # Need at least 3 points
            continue
            
        points = np.column_stack([x[mask], y[mask]])
        
        # Calculate covariance and mean
        cov = np.cov(points.T)
        mean = np.mean(points, axis=0)
        
        # Calculate ellipse parameters
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # Chi-square value for confidence level
        chi2_val = stats.chi2.ppf(confidence, df=2)
        width = 2 * np.sqrt(chi2_val * eigenvalues[0])
        height = 2 * np.sqrt(chi2_val * eigenvalues[1])
        
        # Create and add ellipse
        ellipse = Ellipse(mean, width, height, angle=angle,
                         facecolor=colors[i], alpha=alpha,
                         edgecolor=colors[i], linewidth=2)
        ax.add_patch(ellipse)


def visualize_single_omics_embedding(
    adata,
    color_col: str,
    embedding_key: str = 'X_umap',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    point_size: Union[int, np.ndarray] = 60,
    alpha: float = 0.8,
    colormap: str = 'viridis',
    show_density: bool = False,
    show_ellipses: bool = False,
    show_legend: bool = True,
    show_colorbar: bool = True,
    highlight_samples: Optional[List[str]] = None,
    annotate_samples: Optional[List[str]] = None,
    style: str = 'modern',  # 'modern', 'classic', 'minimal'
    output_path: Optional[str] = None,
    dpi: int = 300,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Enhanced single-omics embedding visualization with multiple style options.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    color_col : str
        Column name for coloring points
    embedding_key : str
        Key for embedding coordinates (default: 'X_umap')
    title : str, optional
        Plot title (auto-generated if None)
    figsize : tuple
        Figure size (width, height)
    point_size : int or array
        Size of scatter points (can be array for variable sizes)
    alpha : float
        Transparency of points
    colormap : str
        Colormap name for numerical data
    show_density : bool
        Add density contours
    show_ellipses : bool
        Add confidence ellipses for groups
    show_legend : bool
        Show legend for categorical data
    show_colorbar : bool
        Show colorbar for numerical data
    highlight_samples : list, optional
        List of sample names to highlight
    annotate_samples : list, optional
        List of sample names to annotate
    style : str
        Visual style ('modern', 'classic', 'minimal')
    output_path : str, optional
        Path to save the figure
    dpi : int
        Resolution for saved figure
    **kwargs : additional arguments
        edge_color, edge_width, grid_alpha, etc.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    
    # Style presets
    style_presets = {
        'modern': {
            'edge_color': 'white',
            'edge_width': 0.5,
            'grid': True,
            'grid_alpha': 0.15,
            'spine_width': 1.5,
            'tick_size': 8
        },
        'classic': {
            'edge_color': 'black',
            'edge_width': 0.5,
            'grid': True,
            'grid_alpha': 0.3,
            'spine_width': 1.0,
            'tick_size': 10
        },
        'minimal': {
            'edge_color': 'none',
            'edge_width': 0,
            'grid': False,
            'grid_alpha': 0,
            'spine_width': 0.5,
            'tick_size': 8
        }
    }
    
    # Apply style preset
    style_config = style_presets.get(style, style_presets['modern'])
    for key, value in style_config.items():
        if key not in kwargs:
            kwargs[key] = value
    
    # Extract embedding coordinates
    if embedding_key in adata.obsm:
        embedding = adata.obsm[embedding_key]
    elif embedding_key in adata.uns:
        embedding = adata.uns[embedding_key]
        if isinstance(embedding, pd.DataFrame):
            embedding = embedding.values
    else:
        raise KeyError(f"Embedding '{embedding_key}' not found")
    
    x_coords = embedding[:, 0]
    y_coords = embedding[:, 1]
    
    # Get color values
    if color_col in adata.obs.columns:
        color_values = adata.obs[color_col].values
    else:
        raise KeyError(f"Column '{color_col}' not found in adata.obs")
    
    # Detect data type
    data_type, unique_values = detect_data_type(color_values)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle different data types
    if data_type == 'numerical' or data_type == 'ordinal':
        # Numerical/ordinal coloring
        valid_mask = pd.notna(color_values)
        valid_values = pd.to_numeric(color_values[valid_mask], errors='coerce')
        
        # Create colormap
        cmap = create_gradient_colormap(colormap)
        
        # Main scatter plot
        scatter = ax.scatter(
            x_coords[valid_mask], 
            y_coords[valid_mask],
            c=valid_values,
            s=point_size if isinstance(point_size, (int, float)) else point_size[valid_mask],
            alpha=alpha,
            cmap=cmap,
            edgecolors=kwargs.get('edge_color', 'white'),
            linewidths=kwargs.get('edge_width', 0.5),
            zorder=2
        )
        
        # Add colorbar
        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label(color_col, rotation=270, labelpad=20, fontsize=11)
            
            # Format colorbar ticks for ordinal data
            if data_type == 'ordinal' and len(unique_values) <= 10:
                cbar.set_ticks(sorted(unique_values))
                cbar.set_ticklabels([str(int(v)) if float(v).is_integer() else f'{v:.1f}' 
                                    for v in sorted(unique_values)])
        
        # Plot missing values
        missing_mask = ~valid_mask
        if np.any(missing_mask):
            ax.scatter(
                x_coords[missing_mask],
                y_coords[missing_mask],
                c='lightgray',
                s=point_size if isinstance(point_size, (int, float)) else point_size[missing_mask],
                alpha=alpha * 0.5,
                edgecolors='gray',
                linewidths=kwargs.get('edge_width', 0.5),
                label='Missing',
                zorder=1
            )
    
    else:  # Categorical
        # Create color palette
        n_categories = len(unique_values)
        if n_categories <= 10:
            colors = sns.color_palette("Set3", n_categories)
        elif n_categories <= 20:
            colors = sns.color_palette("tab20", n_categories)
        else:
            colors = sns.color_palette("husl", n_categories)
        
        color_map = {val: colors[i] for i, val in enumerate(unique_values)}
        
        # Plot each category
        for category in unique_values:
            mask = color_values == category
            if np.any(mask):
                ax.scatter(
                    x_coords[mask],
                    y_coords[mask],
                    c=[color_map[category]],
                    s=point_size if isinstance(point_size, (int, float)) else point_size[mask],
                    alpha=alpha,
                    edgecolors=kwargs.get('edge_color', 'white'),
                    linewidths=kwargs.get('edge_width', 0.5),
                    label=str(category),
                    zorder=2
                )
        
        # Add confidence ellipses if requested
        if show_ellipses:
            add_confidence_ellipses(ax, x_coords, y_coords, color_values)
    
    # Add density contours if requested
    if show_density:
        add_density_contours(ax, x_coords, y_coords)
    
    # Highlight specific samples
    if highlight_samples:
        sample_names = adata.obs.index
        for sample in highlight_samples:
            if sample in sample_names:
                idx = np.where(sample_names == sample)[0][0]
                ax.scatter(x_coords[idx], y_coords[idx], 
                          s=200, facecolors='none', 
                          edgecolors='red', linewidths=3,
                          zorder=10)
    
    # Annotate specific samples
    if annotate_samples:
        sample_names = adata.obs.index
        for sample in annotate_samples:
            if sample in sample_names:
                idx = np.where(sample_names == sample)[0][0]
                ax.annotate(sample, (x_coords[idx], y_coords[idx]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=9, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', 
                                  facecolor='yellow', alpha=0.7),
                          zorder=11)
    
    # Styling
    ax.set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
    
    # Title
    if title is None:
        title = f'{embedding_key.replace("_", " ").title()} colored by {color_col}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    if kwargs.get('grid', True):
        ax.grid(True, alpha=kwargs.get('grid_alpha', 0.15), linestyle='--')
    
    # Spines
    for spine in ax.spines.values():
        spine.set_linewidth(kwargs.get('spine_width', 1.5))
    
    # Legend for categorical data
    if data_type == 'categorical' and show_legend:
        if n_categories <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     frameon=True, fancybox=True, shadow=True)
        else:
            # For many categories, use a more compact legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     ncol=2, frameon=True, fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {output_path}")
    
    return fig, ax


def create_multi_panel_embedding(
    adata,
    color_cols: List[str],
    embedding_key: str = 'X_umap',
    n_cols: int = 2,
    figsize_per_panel: Tuple[int, int] = (6, 5),
    shared_axes: bool = True,
    main_title: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create multi-panel visualization for comparing different metadata overlays.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    color_cols : List[str]
        List of column names for coloring different panels
    embedding_key : str
        Key for embedding coordinates
    n_cols : int
        Number of columns in subplot grid
    figsize_per_panel : tuple
        Size of each panel
    shared_axes : bool
        Whether to share axis limits
    main_title : str, optional
        Overall figure title
    output_path : str, optional
        Path to save the figure
    **kwargs : additional arguments passed to visualize_single_omics_embedding
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    
    n_panels = len(color_cols)
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    figsize = (figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                            sharex=shared_axes, sharey=shared_axes)
    
    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Get embedding coordinates once
    if embedding_key in adata.obsm:
        embedding = adata.obsm[embedding_key]
    else:
        embedding = adata.uns[embedding_key]
        if isinstance(embedding, pd.DataFrame):
            embedding = embedding.values
    
    x_coords = embedding[:, 0]
    y_coords = embedding[:, 1]
    
    # Create each panel
    for idx, color_col in enumerate(color_cols):
        ax = axes[idx]
        
        # Plot on existing axis
        plt.sca(ax)
        
        # Get color values
        color_values = adata.obs[color_col].values
        data_type, unique_values = detect_data_type(color_values)
        
        # Plot based on data type
        if data_type in ['numerical', 'ordinal']:
            valid_mask = pd.notna(color_values)
            valid_values = pd.to_numeric(color_values[valid_mask], errors='coerce')
            
            scatter = ax.scatter(
                x_coords[valid_mask],
                y_coords[valid_mask],
                c=valid_values,
                s=kwargs.get('point_size', 30),
                alpha=kwargs.get('alpha', 0.8),
                cmap=kwargs.get('colormap', 'viridis'),
                edgecolors=kwargs.get('edge_color', 'white'),
                linewidths=kwargs.get('edge_width', 0.5)
            )
            
            if kwargs.get('show_colorbar', True):
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                cbar.set_label(color_col, rotation=270, labelpad=15, fontsize=10)
        
        else:  # Categorical
            n_categories = len(unique_values)
            colors = sns.color_palette("Set3", n_categories) if n_categories <= 10 else \
                    sns.color_palette("husl", n_categories)
            
            for i, category in enumerate(unique_values):
                mask = color_values == category
                if np.any(mask):
                    ax.scatter(
                        x_coords[mask],
                        y_coords[mask],
                        c=[colors[i]],
                        s=kwargs.get('point_size', 30),
                        alpha=kwargs.get('alpha', 0.8),
                        edgecolors=kwargs.get('edge_color', 'white'),
                        linewidths=kwargs.get('edge_width', 0.5),
                        label=str(category)
                    )
            
            if kwargs.get('show_legend', True) and n_categories <= 10:
                ax.legend(loc='best', fontsize=8, frameon=True)
        
        # Styling
        ax.set_title(f'{color_col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dimension 1' if idx >= len(color_cols) - n_cols else '', fontsize=10)
        ax.set_ylabel('Dimension 2' if idx % n_cols == 0 else '', fontsize=10)
        ax.grid(True, alpha=0.15, linestyle='--')
    
    # Remove empty subplots
    for idx in range(n_panels, len(axes)):
        fig.delaxes(axes[idx])
    
    # Main title
    if main_title:
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Multi-panel figure saved to: {output_path}")
    
    return fig, axes


# Example usage functions
def plot_expression_embedding(
    adata,
    color_col: str,
    output_dir: str,
    embedding_key: str = 'X_DR_expression',
    filename_prefix: str = 'expression_embedding',
    file_format: str = 'png',
    dpi: int = 300,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Convenience function for plotting expression-based embeddings with automatic saving.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    color_col : str
        Column name for coloring points
    output_dir : str
        Directory to save the plot
    embedding_key : str
        Key for embedding coordinates (default: 'X_DR_expression')
    filename_prefix : str
        Prefix for the output filename (default: 'expression_embedding')
    file_format : str
        File format for saving ('png', 'pdf', 'svg', etc.)
    dpi : int
        Resolution for saved figure
    **kwargs : additional arguments passed to visualize_single_omics_embedding
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    safe_color_col = color_col.replace(' ', '_').replace('/', '_')
    filename = f"{filename_prefix}_{safe_color_col}.{file_format}"
    output_path = os.path.join(output_dir, filename)
    
    # Set default parameters
    kwargs.setdefault('title', f'Expression Embedding: {color_col}')
    kwargs.setdefault('colormap', 'viridis')
    kwargs['output_path'] = output_path
    kwargs['dpi'] = dpi
    
    # Create plot
    fig, ax = visualize_single_omics_embedding(adata, color_col, embedding_key, **kwargs)
    
    print(f"Expression embedding plot saved to: {output_path}")
    
    return fig, ax


def plot_proportion_embedding(
    adata,
    color_col: str,
    output_dir: str,
    embedding_key: str = 'X_DR_proportion',
    filename_prefix: str = 'proportion_embedding',
    file_format: str = 'png',
    dpi: int = 300,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Convenience function for plotting proportion-based embeddings with automatic saving.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    color_col : str
        Column name for coloring points
    output_dir : str
        Directory to save the plot
    embedding_key : str
        Key for embedding coordinates (default: 'X_DR_proportion')
    filename_prefix : str
        Prefix for the output filename (default: 'proportion_embedding')
    file_format : str
        File format for saving ('png', 'pdf', 'svg', etc.)
    dpi : int
        Resolution for saved figure
    **kwargs : additional arguments passed to visualize_single_omics_embedding
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    safe_color_col = color_col.replace(' ', '_').replace('/', '_')
    filename = f"{filename_prefix}_{safe_color_col}.{file_format}"
    output_path = os.path.join(output_dir, filename)
    
    # Set default parameters
    kwargs.setdefault('title', f'Cell Proportion Embedding: {color_col}')
    kwargs.setdefault('colormap', 'RdYlBu')
    kwargs['output_path'] = output_path
    kwargs['dpi'] = dpi
    
    # Create plot
    fig, ax = visualize_single_omics_embedding(adata, color_col, embedding_key, **kwargs)
    
    print(f"Proportion embedding plot saved to: {output_path}")
    
    return fig, ax