import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os

def detect_data_type(values):
    """
    Detect if the data is numerical/gradual or categorical.
    
    Returns:
        tuple: (data_type, unique_values)
        where data_type is 'numerical' or 'categorical'
    """
    # Remove NaN values for analysis
    valid_values = [v for v in values if pd.notna(v)]
    
    if not valid_values:
        return 'categorical', []
    
    unique_values = list(set(valid_values))
    
    # Check if all values can be converted to numbers
    try:
        numeric_values = [float(v) for v in valid_values]
        
        # If we can convert to numbers, default to numerical
        n_unique = len(unique_values)
        
        # Special case: binary data (only 2 values) that are 0/1 or similar
        if n_unique == 2:
            sorted_vals = sorted(numeric_values)
            # Check if it's 0/1 (common for binary categories)
            if sorted_vals[0] == 0 and sorted_vals[1] == 1:
                return 'categorical', unique_values
        
        # For sequential numeric data (like 1,2,3,4 or 1.0,2.0,3.0,4.0), 
        # always treat as numerical for gradient visualization
        sorted_unique = sorted(unique_values)
        
        # Check if values form a sequence (with gaps allowed)
        # This handles severity levels, stages, scores, etc.
        if all(isinstance(v, (int, float)) for v in sorted_unique):
            # Any ordered numeric sequence should be numerical
            return 'numerical', unique_values
        
        # Default: treat numeric data as numerical
        return 'numerical', unique_values
        
    except (ValueError, TypeError):
        # If conversion to float fails, it's definitely categorical
        return 'categorical', unique_values

def create_categorical_colormap(unique_values, colormap='tab20'):
    """
    Create a color mapping for categorical data.
    """
    n_categories = len(unique_values)
    
    # Choose appropriate colormap based on number of categories
    if n_categories <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_categories))
    elif n_categories <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_categories))
    else:
        # For many categories, use a continuous colormap
        base_cmap = plt.cm.get_cmap(colormap)
        colors = base_cmap(np.linspace(0, 1, n_categories))
    
    # Create mapping from value to color
    color_map = {val: colors[i] for i, val in enumerate(sorted(unique_values))}
    
    return color_map

def create_quantitative_colormap(values, colormap='viridis'):
    """
    Create a color mapping for numerical/gradual data.
    """
    unique_values = sorted(set(values))
    
    if len(unique_values) == 1:
        base_cmap = plt.cm.get_cmap(colormap)
        return {unique_values[0]: base_cmap(0.5)}
    
    min_val, max_val = min(unique_values), max(unique_values)
    base_cmap = plt.cm.get_cmap(colormap)
    
    color_map = {}
    for val in unique_values:
        normalized = (val - min_val) / (max_val - min_val)
        color_map[val] = base_cmap(normalized)
    
    return color_map

def get_embedding_data(adata, embedding_key, verbose=True):
    """
    Extract embedding coordinates from AnnData object.
    """
    if embedding_key in adata.obsm:
        embedding = adata.obsm[embedding_key]
        coord_source = "obsm"
        if verbose:
            print(f"Found embedding '{embedding_key}' in adata.obsm")
    elif embedding_key in adata.uns:
        embedding = adata.uns[embedding_key]
        coord_source = "uns"
        if verbose:
            print(f"Found embedding '{embedding_key}' in adata.uns")
    else:
        available_obsm = list(adata.obsm.keys()) if hasattr(adata, 'obsm') else []
        available_uns = list(adata.uns.keys()) if hasattr(adata, 'uns') else []
        raise KeyError(f"Embedding '{embedding_key}' not found in adata.obsm {available_obsm} or adata.uns {available_uns}")
    
    if verbose:
        print(f"Embedding shape: {embedding.shape}")
    
    if embedding.shape[1] < 2:
        raise ValueError(f"Need at least 2 dimensions for visualization (found {embedding.shape[1]})")
    
    if coord_source == "obsm":
        x_coords = embedding[:, 0]
        y_coords = embedding[:, 1]
        sample_names = adata.obs.index
    else:
        if isinstance(embedding, pd.DataFrame):
            x_coords = embedding.iloc[:, 0]
            y_coords = embedding.iloc[:, 1]
            sample_names = embedding.index
        else:
            x_coords = embedding[:, 0]
            y_coords = embedding[:, 1]
            sample_names = adata.obs.index
    
    return x_coords, y_coords, sample_names, coord_source

def plot_multimodal_embedding(adata, modality_col, color_col, target_modality,
                             embedding_key, ax, point_size=60, alpha=0.8, 
                             colormap='viridis', show_sample_names=False, 
                             non_target_color='lightgray', non_target_alpha=0.4,
                             data_type=None, unique_values=None):
    """
    Plot embedding with points colored by specified column values.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    modality_col : str
        Column name for modality information
    color_col : str
        Column name for coloring (can be numerical or categorical)
    target_modality : str
        Which modality to highlight
    embedding_key : str
        Key for embedding coordinates
    ax : matplotlib axis
        Axis to plot on
    show_sample_names : bool
        Whether to show sample names (only for target modality)
    """
    
    x_coords, y_coords, sample_names, coord_source = get_embedding_data(adata, embedding_key, verbose=False)
    
    modality_values = adata.obs[modality_col].values
    color_values = adata.obs[color_col].values
    
    target_mask = modality_values == target_modality
    non_target_mask = ~target_mask
    
    # Auto-detect data type if not provided
    if data_type is None:
        target_color_values = color_values[target_mask]
        data_type, unique_values = detect_data_type(target_color_values)
    
    # Plot non-target modality samples first (background)
    if np.any(non_target_mask):
        ax.scatter(x_coords[non_target_mask], y_coords[non_target_mask], 
                  c=non_target_color, s=point_size, alpha=non_target_alpha,
                  edgecolors='black', linewidth=0.5, 
                  label=f'Other modalities', zorder=1)
    
    # Plot target modality samples with appropriate coloring
    if np.any(target_mask):
        target_x = x_coords[target_mask]
        target_y = y_coords[target_mask]
        target_color_values = color_values[target_mask]
        target_sample_names = sample_names[target_mask]  # Get target modality sample names
        
        # Handle valid color values
        valid_mask = pd.notna(target_color_values)
        
        if np.any(valid_mask):
            valid_values = target_color_values[valid_mask]
            valid_x = target_x[valid_mask]
            valid_y = target_y[valid_mask]
            valid_names = target_sample_names[valid_mask]  # Names for valid samples
            
            # Create color mapping based on data type
            if data_type == 'numerical':
                color_map = create_quantitative_colormap(valid_values, colormap)
                colors = [color_map[val] for val in valid_values]
                
                # Single scatter for numerical data
                scatter = ax.scatter(valid_x, valid_y, c=colors, s=point_size, alpha=alpha,
                                   edgecolors='black', linewidth=0.5, 
                                   label=f'{target_modality} (by {color_col})', zorder=2)
            else:  # categorical
                color_map = create_categorical_colormap(unique_values, colormap)
                
                # Plot each category separately for legend
                for category in sorted(unique_values):
                    cat_mask = valid_values == category
                    if np.any(cat_mask):
                        ax.scatter(valid_x[cat_mask], valid_y[cat_mask], 
                                 c=[color_map[category]], s=point_size, alpha=alpha,
                                 edgecolors='black', linewidth=0.5, 
                                 label=f'{target_modality}: {category}', zorder=2)
            
            # Add sample labels for valid target modality samples only
            if show_sample_names:
                for i, sample in enumerate(valid_names):
                    ax.annotate(sample, (valid_x[i], valid_y[i]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
        
        # Plot samples with missing values
        missing_mask = ~valid_mask
        if np.any(missing_mask):
            missing_x = target_x[missing_mask]
            missing_y = target_y[missing_mask]
            missing_names = target_sample_names[missing_mask]  # Names for missing samples
            
            ax.scatter(missing_x, missing_y, c='red', s=point_size, alpha=alpha,
                      edgecolors='black', linewidth=0.5, 
                      label=f'{target_modality} (missing {color_col})', zorder=2)
            
            # Add sample labels for missing target modality samples only
            if show_sample_names:
                for i, sample in enumerate(missing_names):
                    ax.annotate(sample, (missing_x[i], missing_y[i]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8, color='red')
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(True, alpha=0.3)
    
    # Adjust legend position based on data type
    if data_type == 'categorical' and len(unique_values) > 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1 if len(unique_values) <= 15 else 2)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return ax, data_type, unique_values

def create_single_embedding_plot(adata, modality_col, color_col, target_modality,
                                embedding_key, embedding_type, figsize=(10, 8), 
                                point_size=60, alpha=0.8, colormap='viridis', 
                                show_sample_names=False, verbose=True):
    """
    Create a single embedding plot with colorbar (for numerical) or legend (for categorical).
    """
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Detect data type
    target_mask = adata.obs[modality_col].values == target_modality
    target_values = adata.obs[color_col].values[target_mask]
    data_type, unique_values = detect_data_type(target_values)
    
    if verbose:
        print(f"Detected data type for {color_col}: {data_type}")
        if data_type == 'categorical':
            print(f"Categories: {sorted(unique_values)}")
    
    # Create the plot
    ax, data_type, unique_values = plot_multimodal_embedding(
        adata, modality_col, color_col, target_modality, embedding_key, ax,
        point_size, alpha, colormap, show_sample_names,
        data_type=data_type, unique_values=unique_values
    )
    
    # Get valid values
    valid_values = target_values[pd.notna(target_values)]
    
    # Add colorbar for numerical data
    if data_type == 'numerical' and len(valid_values) > 1:
        norm = Normalize(vmin=min(valid_values), vmax=max(valid_values))
        sm = ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        
        # Add colorbar
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        
        # Set colorbar ticks and labels
        unique_vals = sorted(set(valid_values))
        if len(unique_vals) <= 10:
            cbar.set_ticks(unique_vals)
            cbar.set_ticklabels([f'{v:.1f}' if v != int(v) else f'{int(v)}' for v in unique_vals])
        else:
            # If too many unique values, use fewer ticks
            n_ticks = 5
            tick_values = np.linspace(min(valid_values), max(valid_values), n_ticks)
            cbar.set_ticks(tick_values)
            cbar.set_ticklabels([f'{v:.1f}' for v in tick_values])
        
        cbar.set_label(f'{color_col} ({target_modality})', rotation=270, labelpad=20)
    
    # Set title
    title = f'{embedding_type} Embedding: {target_modality} colored by {color_col}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig, ax

def plot_default_embedding(adata, embedding_key, ax, point_size=60, alpha=0.8, 
                          show_sample_names=False, sample_color='steelblue'):
    """
    Plot embedding with all samples shown equally without modality separation or coloring.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    embedding_key : str
        Key for embedding coordinates
    ax : matplotlib axis
        Axis to plot on
    point_size : int
        Size of scatter points
    alpha : float
        Transparency of points
    show_sample_names : bool
        Whether to show sample names
    sample_color : str
        Color for all samples
    """
    
    x_coords, y_coords, sample_names, _ = get_embedding_data(adata, embedding_key, verbose=False)
    
    # Plot all samples with the same color
    ax.scatter(x_coords, y_coords, 
              c=sample_color, s=point_size, alpha=alpha,
              edgecolors='black', linewidth=0.5, 
              label='All samples', zorder=2)
    
    # Add sample labels if requested
    if show_sample_names:
        for i, sample in enumerate(sample_names):
            ax.annotate(sample, (x_coords[i], y_coords[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    return ax


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def detect_data_type(values):
    """
    Detect whether values are numerical or categorical.
    
    Returns:
    --------
    data_type : str
        'numerical' or 'categorical'
    unique_values : list
        List of unique values
    """
    # Remove NaN values
    valid_values = [v for v in values if pd.notna(v)]
    
    if len(valid_values) == 0:
        return 'categorical', []
    
    # Check if values are numeric
    try:
        numeric_values = [float(v) for v in valid_values]
        unique_values = sorted(set(numeric_values))
        
        # If few unique values and they look like integers, might be categorical
        if len(unique_values) <= 10 and all(v == int(v) for v in unique_values):
            # Could be either - default to numerical but could be overridden
            return 'numerical', unique_values
        return 'numerical', unique_values
    except (ValueError, TypeError):
        # Not numeric, treat as categorical
        unique_values = list(set(valid_values))
        return 'categorical', unique_values


def get_embedding_data(adata, embedding_key):
    """
    Get embedding data from adata.obsm or adata.uns.
    
    Returns:
    --------
    embedding : np.ndarray
        2D embedding array
    """
    if embedding_key in adata.obsm:
        embedding = adata.obsm[embedding_key]
    elif embedding_key in adata.uns:
        embedding = adata.uns[embedding_key]
    else:
        raise ValueError(f"Embedding key '{embedding_key}' not found in adata.obsm or adata.uns")
    
    # Ensure 2D
    if embedding.ndim == 1:
        raise ValueError(f"Embedding '{embedding_key}' is 1D, expected 2D")
    
    return embedding


def plot_default_embedding(adata, embedding_key, ax, point_size=60, alpha=0.8, 
                           show_sample_names=False):
    """
    Plot default embedding without any coloring (all samples in same color).
    """
    embedding = get_embedding_data(adata, embedding_key)
    
    ax.scatter(embedding[:, 0], embedding[:, 1], 
               s=point_size, alpha=alpha, c='steelblue', edgecolors='white', linewidths=0.5)
    
    if show_sample_names:
        for i, sample_name in enumerate(adata.obs_names):
            ax.annotate(sample_name, (embedding[i, 0], embedding[i, 1]),
                       fontsize=6, alpha=0.7, ha='center', va='bottom')
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    return ax


def plot_embedding_colored_by_column(adata, embedding_key, color_col, ax, 
                                     point_size=60, alpha=0.8, 
                                     colormap='viridis', categorical_cmap='tab10',
                                     show_sample_names=False, force_data_type=None,
                                     verbose=True):
    """
    Plot embedding colored by a specific column.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    embedding_key : str
        Key for the embedding in adata.obsm or adata.uns
    color_col : str
        Column name in adata.obs to color by
    ax : matplotlib axis
        Axis to plot on
    point_size : int
        Size of scatter points
    alpha : float
        Transparency
    colormap : str
        Colormap for numerical data
    categorical_cmap : str
        Colormap for categorical data
    show_sample_names : bool
        Whether to show sample names
    force_data_type : str or None
        Force 'numerical' or 'categorical'
    verbose : bool
        Print messages
    
    Returns:
    --------
    ax : matplotlib axis
    data_type : str
    unique_values : list
    """
    embedding = get_embedding_data(adata, embedding_key)
    color_values = adata.obs[color_col].values
    
    # Detect or force data type
    if force_data_type is not None:
        if force_data_type not in ['numerical', 'categorical']:
            raise ValueError("force_data_type must be 'numerical' or 'categorical'")
        data_type = force_data_type
        if data_type == 'categorical':
            unique_values = list(set([v for v in color_values if pd.notna(v)]))
        else:
            unique_values = sorted(set([float(v) for v in color_values if pd.notna(v)]))
    else:
        data_type, unique_values = detect_data_type(color_values)
    
    if verbose:
        print(f"  Column '{color_col}': {data_type} with {len(unique_values)} unique values")
    
    if data_type == 'numerical':
        # Numerical coloring
        valid_mask = pd.notna(color_values)
        numeric_values = np.array([float(v) if pd.notna(v) else np.nan for v in color_values])
        
        # Plot NaN values first (in gray)
        if (~valid_mask).any():
            ax.scatter(embedding[~valid_mask, 0], embedding[~valid_mask, 1],
                      s=point_size, alpha=alpha*0.5, c='lightgray', 
                      edgecolors='white', linewidths=0.5, label='NA')
        
        # Plot valid values with colormap
        if valid_mask.any():
            scatter = ax.scatter(embedding[valid_mask, 0], embedding[valid_mask, 1],
                                s=point_size, alpha=alpha, c=numeric_values[valid_mask],
                                cmap=colormap, edgecolors='white', linewidths=0.5)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label(color_col, rotation=270, labelpad=15)
    
    else:
        # Categorical coloring
        # Get colormap
        if isinstance(categorical_cmap, str):
            cmap = plt.cm.get_cmap(categorical_cmap)
        else:
            cmap = categorical_cmap
        
        # Sort unique values for consistent coloring
        try:
            sorted_values = sorted(unique_values)
        except TypeError:
            sorted_values = list(unique_values)
        
        # Create color mapping
        n_colors = len(sorted_values)
        colors = {val: cmap(i / max(n_colors - 1, 1)) for i, val in enumerate(sorted_values)}
        colors['NA'] = (0.8, 0.8, 0.8, 1.0)  # Gray for NA
        
        # Plot each category
        for val in sorted_values:
            mask = color_values == val
            if mask.any():
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                          s=point_size, alpha=alpha, c=[colors[val]],
                          edgecolors='white', linewidths=0.5, label=str(val))
        
        # Plot NA values
        na_mask = pd.isna(color_values)
        if na_mask.any():
            ax.scatter(embedding[na_mask, 0], embedding[na_mask, 1],
                      s=point_size, alpha=alpha*0.5, c=[colors['NA']],
                      edgecolors='white', linewidths=0.5, label='NA')
        
        # Add legend
        legend = ax.legend(title=color_col, bbox_to_anchor=(1.02, 1), loc='upper left',
                          fontsize=8, title_fontsize=9, framealpha=0.9)
    
    # Add sample names if requested
    if show_sample_names:
        for i, sample_name in enumerate(adata.obs_names):
            ax.annotate(sample_name, (embedding[i, 0], embedding[i, 1]),
                       fontsize=6, alpha=0.7, ha='center', va='bottom')
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    return ax, data_type, unique_values


def plot_multimodal_embedding(adata, modality_col, color_col, target_modality, embedding_key, ax,
                              point_size=60, alpha=0.8, colormap='viridis', show_sample_names=False,
                              data_type=None, unique_values=None):
    """
    Plot embedding with modality-specific coloring (original function logic).
    """
    embedding = get_embedding_data(adata, embedding_key)
    
    # Get masks
    target_mask = adata.obs[modality_col].values == target_modality
    other_mask = ~target_mask
    
    # Plot other modality samples in gray
    if other_mask.any():
        ax.scatter(embedding[other_mask, 0], embedding[other_mask, 1],
                  s=point_size * 0.5, alpha=alpha * 0.3, c='lightgray',
                  edgecolors='none', label='Other')
    
    # Get color values for target modality
    color_values = adata.obs[color_col].values
    target_colors = color_values[target_mask]
    target_embedding = embedding[target_mask]
    
    # Detect data type if not provided
    if data_type is None:
        data_type, unique_values = detect_data_type(target_colors)
    
    if data_type == 'numerical':
        valid_mask = pd.notna(target_colors)
        numeric_values = np.array([float(v) if pd.notna(v) else np.nan for v in target_colors])
        
        # Plot valid values
        if valid_mask.any():
            scatter = ax.scatter(target_embedding[valid_mask, 0], target_embedding[valid_mask, 1],
                                s=point_size, alpha=alpha, c=numeric_values[valid_mask],
                                cmap=colormap, edgecolors='white', linewidths=0.5)
    else:
        # Categorical
        cmap = plt.cm.get_cmap('tab10')
        try:
            sorted_values = sorted(unique_values)
        except TypeError:
            sorted_values = list(unique_values)
        
        n_colors = len(sorted_values)
        colors = {val: cmap(i / max(n_colors - 1, 1)) for i, val in enumerate(sorted_values)}
        
        for val in sorted_values:
            mask = target_colors == val
            if mask.any():
                ax.scatter(target_embedding[mask, 0], target_embedding[mask, 1],
                          s=point_size, alpha=alpha, c=[colors[val]],
                          edgecolors='white', linewidths=0.5, label=str(val))
        
        ax.legend(title=color_col, bbox_to_anchor=(1.02, 1), loc='upper left',
                 fontsize=8, title_fontsize=9)
    
    # Add sample names for target modality only
    if show_sample_names:
        target_names = adata.obs_names[target_mask]
        for i, sample_name in enumerate(target_names):
            ax.annotate(sample_name, (target_embedding[i, 0], target_embedding[i, 1]),
                       fontsize=6, alpha=0.7, ha='center', va='bottom')
    
    return ax, data_type, unique_values


def create_single_embedding_plot(adata, modality_col, color_col, target_modality,
                                 embedding_key, embedding_type, figsize=(10, 8),
                                 point_size=60, alpha=0.8, colormap='viridis',
                                 show_sample_names=False, verbose=True):
    """
    Create a single embedding plot for modality-specific visualization.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax, data_type, unique_values = plot_multimodal_embedding(
        adata, modality_col, color_col, target_modality, embedding_key, ax,
        point_size, alpha, colormap, show_sample_names
    )
    
    title = f'{embedding_type} Embedding: {target_modality} by {color_col}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig, ax


def visualize_multimodal_embedding(adata, modality_col=None, color_col=None, target_modality=None,
                                  expression_key='X_DR_expression', proportion_key='X_DR_proportion',
                                  figsize=(20, 8), point_size=60, alpha=0.8, 
                                  colormap='viridis', categorical_cmap='tab10',
                                  output_dir=None, 
                                  show_sample_names=False, force_data_type=None, 
                                  show_default=True, verbose=True,
                                  visualization_grouping_column=None):
    """
    Visualize multimodal embeddings with flexible coloring by any column.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing embeddings and metadata
    modality_col : str or None
        Column name in adata.obs containing modality information (None for default plot)
    color_col : str or None
        Column name in adata.obs to use for coloring points (None for default plot)
    target_modality : str or None
        Which modality to highlight in the visualization (None for default plot)
    expression_key : str
        Key for expression-based embedding (default: 'X_DR_expression')
    proportion_key : str
        Key for proportion-based embedding (default: 'X_DR_proportion')
    figsize : tuple
        Figure size for combined plot (default: (20, 8))
    point_size : int
        Size of scatter points (default: 60)
    alpha : float
        Transparency of points (default: 0.8)
    colormap : str
        Colormap to use for numerical data (default: 'viridis')
    categorical_cmap : str
        Colormap to use for categorical data (default: 'tab10')
    output_dir : str
        Directory or file path to save plots
    show_sample_names : bool
        Whether to show sample names on plot (default: False)
    force_data_type : str or None
        Force data type to 'numerical' or 'categorical' instead of auto-detection (default: None)
    show_default : bool
        If True, show default embedding without modality separation or coloring (default: True)
    verbose : bool
        Print progress messages (default: True)
    visualization_grouping_column : list of str or None
        List of column names in adata.obs to create separate colored visualizations for.
        Each column will generate its own set of plots. If provided in default mode,
        will create plots colored by each of these columns. (default: None)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
        The created visualization (None if saved separately)
    """
    
    # Determine if we should show the default plot
    if show_default or (modality_col is None and color_col is None and target_modality is None):
        show_default = True
        if verbose:
            print("Creating default embedding visualization (all samples)")
    else:
        # Validate that all required parameters are provided for modality-specific plot
        if modality_col is None or color_col is None or target_modality is None:
            raise ValueError("For modality-specific plots, modality_col, color_col, and target_modality must all be provided. "
                           "Set show_default=True or provide no parameters for default plot.")
    
    if verbose:
        if not show_default:
            print(f"Creating multimodal embedding visualization for {target_modality}")
            print(f"Coloring by: {color_col}")
        print(f"Expression key: {expression_key}")
        print(f"Proportion key: {proportion_key}")
        if visualization_grouping_column:
            print(f"Additional color columns: {visualization_grouping_column}")
        if show_sample_names:
            if show_default:
                print("Sample names will be shown for all samples")
            else:
                print(f"Sample names will be shown only for {target_modality} modality")
    
    # IMPROVED: Better output directory handling
    if output_dir:
        # Ensure the base output directory exists
        os.makedirs(os.path.dirname(output_dir) if os.path.dirname(output_dir) else '.', exist_ok=True)
        
        # If it's a directory path, create the visualization subdirectory
        if os.path.isdir(output_dir) or (not os.path.splitext(output_dir)[1]):
            output_dir = os.path.join(output_dir, 'visualization')
            os.makedirs(output_dir, exist_ok=True)
            if verbose:
                print(f"Created/using visualization directory: {output_dir}")
        else:
            # If it's a file path, ensure the parent directory exists
            parent_dir = os.path.dirname(output_dir)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
                if verbose:
                    print(f"Created/using parent directory: {parent_dir}")
    
    # Check which embeddings are available
    available_embeddings = []
    
    if expression_key in adata.obsm or expression_key in adata.uns:
        available_embeddings.append(('Expression', expression_key))
    else:
        if verbose:
            print(f"Warning: Expression embedding '{expression_key}' not found")
    
    if proportion_key in adata.obsm or proportion_key in adata.uns:
        available_embeddings.append(('Proportion', proportion_key))
    else:
        if verbose:
            print(f"Warning: Proportion embedding '{proportion_key}' not found")
    
    if not available_embeddings:
        available_obsm = list(adata.obsm.keys()) if hasattr(adata, 'obsm') else []
        available_uns = list(adata.uns.keys()) if hasattr(adata, 'uns') else []
        raise ValueError(f"No embeddings found. Available in obsm: {available_obsm}, uns: {available_uns}")
    
    # Validate visualization_grouping_column
    if visualization_grouping_column is not None:
        if isinstance(visualization_grouping_column, str):
            visualization_grouping_column = [visualization_grouping_column]
        
        # Check all columns exist
        missing_cols = [col for col in visualization_grouping_column if col not in adata.obs.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in adata.obs: {missing_cols}. "
                           f"Available columns: {list(adata.obs.columns)}")
    
    # Handle default plot case
    if show_default:
        # =====================================================================
        # NEW: Handle visualization_grouping_column in default mode
        # =====================================================================
        if visualization_grouping_column is not None and len(visualization_grouping_column) > 0:
            if verbose:
                print(f"\nCreating colored visualizations for {len(visualization_grouping_column)} column(s)...")
            
            all_figures = {}
            
            for color_col_item in visualization_grouping_column:
                if verbose:
                    print(f"\n--- Processing color column: {color_col_item} ---")
                
                # If output_dir is provided, save separate plots for each embedding and color column
                if output_dir:
                    for embedding_type, embedding_key in available_embeddings:
                        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                        
                        ax, data_type, unique_values = plot_embedding_colored_by_column(
                            adata, embedding_key, color_col_item, ax,
                            point_size=point_size, alpha=alpha,
                            colormap=colormap, categorical_cmap=categorical_cmap,
                            show_sample_names=show_sample_names,
                            force_data_type=force_data_type, verbose=verbose
                        )
                        
                        title = f'{embedding_type} Embedding: All Samples colored by {color_col_item}'
                        ax.set_title(title, fontsize=14, fontweight='bold')
                        
                        plt.tight_layout()
                        
                        # Create safe filename (replace problematic characters)
                        safe_col_name = color_col_item.replace('/', '_').replace('\\', '_').replace(' ', '_')
                        filename = f"all_samples_{embedding_type.lower()}_by_{safe_col_name}.png"
                        save_path = os.path.join(output_dir, filename)
                        
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        
                        if verbose:
                            print(f"  Saved: {save_path}")
                        
                        plt.close(fig)
                    
                    # Also create combined plot for this color column
                    n_plots = len(available_embeddings)
                    fig_combined, axes_combined = plt.subplots(1, n_plots, figsize=figsize)
                    
                    if n_plots == 1:
                        axes_combined = [axes_combined]
                    
                    for i, (embedding_type, embedding_key) in enumerate(available_embeddings):
                        ax = axes_combined[i]
                        
                        ax, data_type, unique_values = plot_embedding_colored_by_column(
                            adata, embedding_key, color_col_item, ax,
                            point_size=point_size, alpha=alpha,
                            colormap=colormap, categorical_cmap=categorical_cmap,
                            show_sample_names=show_sample_names,
                            force_data_type=force_data_type, verbose=False
                        )
                        
                        if embedding_type == 'Expression':
                            title = 'Expression Embedding'
                        else:
                            title = 'Cell Proportion Embedding'
                        
                        ax.set_title(title, fontsize=14, fontweight='bold')
                        if i == 0:
                            ax.set_ylabel('Dimension 2')
                    
                    fig_combined.suptitle(f'Multi-modal Embedding: All Samples colored by {color_col_item}', 
                                         fontsize=16, fontweight='bold', y=0.98)
                    plt.tight_layout()
                    
                    safe_col_name = color_col_item.replace('/', '_').replace('\\', '_').replace(' ', '_')
                    combined_filename = f"all_samples_combined_by_{safe_col_name}.png"
                    combined_save_path = os.path.join(output_dir, combined_filename)
                    plt.savefig(combined_save_path, dpi=300, bbox_inches='tight')
                    
                    if verbose:
                        print(f"  Combined saved: {combined_save_path}")
                    
                    plt.close(fig_combined)
                
                else:
                    # No output_dir, create and store figures
                    n_plots = len(available_embeddings)
                    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
                    
                    if n_plots == 1:
                        axes = [axes]
                    
                    for i, (embedding_type, embedding_key) in enumerate(available_embeddings):
                        ax = axes[i]
                        
                        ax, data_type, unique_values = plot_embedding_colored_by_column(
                            adata, embedding_key, color_col_item, ax,
                            point_size=point_size, alpha=alpha,
                            colormap=colormap, categorical_cmap=categorical_cmap,
                            show_sample_names=show_sample_names,
                            force_data_type=force_data_type, verbose=verbose
                        )
                        
                        if embedding_type == 'Expression':
                            title = 'Expression Embedding'
                        else:
                            title = 'Cell Proportion Embedding'
                        
                        ax.set_title(title, fontsize=14, fontweight='bold')
                        if i == 0:
                            ax.set_ylabel('Dimension 2')
                    
                    fig.suptitle(f'Multi-modal Embedding: All Samples colored by {color_col_item}', 
                                fontsize=16, fontweight='bold', y=0.98)
                    plt.tight_layout()
                    
                    all_figures[color_col_item] = (fig, axes)
            
            # Return based on what was created
            if output_dir:
                return None, None
            elif len(all_figures) == 1:
                return list(all_figures.values())[0]
            else:
                return all_figures
        
        # =====================================================================
        # Original default plot logic (no visualization_grouping_column)
        # =====================================================================
        
        # If both embeddings are available and output_dir is provided, save separately
        if len(available_embeddings) == 2 and output_dir:
            saved_files = []
            
            for embedding_type, embedding_key in available_embeddings:
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                
                # Create default plot
                ax = plot_default_embedding(
                    adata, embedding_key, ax,
                    point_size=point_size, alpha=alpha,
                    show_sample_names=show_sample_names
                )
                
                # Set title
                title = f'{embedding_type} Embedding: All Samples'
                ax.set_title(title, fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                
                # IMPROVED: Save plot with proper directory handling
                filename = f"all_samples_{embedding_type.lower()}.png"
                save_path = os.path.join(output_dir, filename)
                
                # Ensure directory exists before saving
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                saved_files.append(save_path)
                
                if verbose:
                    print(f"{embedding_type} plot saved to: {save_path}")
                
                plt.close(fig)
            
            return None, None
        
        # Create combined plot for default view
        n_plots = len(available_embeddings)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=True)
        
        if n_plots == 1:
            axes = [axes]
        
        for i, (embedding_type, embedding_key) in enumerate(available_embeddings):
            ax = axes[i]
            
            ax = plot_default_embedding(
                adata, embedding_key, ax,
                point_size=point_size, alpha=alpha,
                show_sample_names=show_sample_names
            )
            
            # Set title based on embedding type
            if embedding_type == 'Expression':
                title = 'Expression Embedding'
            else:
                title = 'Cell Proportion Embedding'
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Dimension 1')
            if i == 0:
                ax.set_ylabel('Dimension 2')
        
        # Add main title
        fig.suptitle('Multi-modal Embedding: All Samples', fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        # IMPROVED: Save combined plot if requested
        if output_dir:
            save_path = os.path.join(output_dir, "all_samples_combined.png")
            
            # Ensure directory exists before saving
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Combined plot saved to: {save_path}")
        
        return fig, axes
    
    # =========================================================================
    # If not default plot, continue with the original modality-specific logic
    # =========================================================================
    
    # Detect data type early
    target_mask = adata.obs[modality_col].values == target_modality
    target_values = adata.obs[color_col].values[target_mask]
    
    # Use forced data type if provided, otherwise auto-detect
    if force_data_type is not None:
        if force_data_type not in ['numerical', 'categorical']:
            raise ValueError("force_data_type must be 'numerical' or 'categorical'")
        data_type = force_data_type
        if data_type == 'categorical':
            unique_values = list(set([v for v in target_values if pd.notna(v)]))
        else:
            unique_values = sorted(set([v for v in target_values if pd.notna(v)]))
    else:
        data_type, unique_values = detect_data_type(target_values)
    
    if verbose:
        print(f"\nDetected data type for {color_col}: {data_type}")
        if data_type == 'categorical':
            print(f"Categories found: {sorted(unique_values)}")
    
    # IMPROVED: If both embeddings are available and output_dir is provided, save separately
    if len(available_embeddings) == 2 and output_dir:
        # Check if output_dir is a directory or a file path
        if os.path.isdir(output_dir) or (not os.path.splitext(output_dir)[1]):
            # It's a directory path
            save_dir = output_dir
            # Create filename based on modality and color column
            base_name = f"{target_modality}_{color_col}"
            extension = '.png'
        else:
            # It's a file path
            save_dir = os.path.dirname(output_dir)
            base_name = os.path.splitext(os.path.basename(output_dir))[0]
            extension = os.path.splitext(output_dir)[1] or '.png'
        
        # IMPROVED: Create output directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = []
        
        # Save each embedding as a separate plot
        for embedding_type, embedding_key in available_embeddings:
            fig, ax = create_single_embedding_plot(
                adata, modality_col, color_col, target_modality,
                embedding_key, embedding_type, figsize=(10, 8),
                point_size=point_size, alpha=alpha, colormap=colormap, 
                show_sample_names=show_sample_names, verbose=verbose
            )
            
            # Create filename for this embedding type
            filename = f"{base_name}_{embedding_type.lower()}{extension}"
            separate_save_path = os.path.join(save_dir, filename)
            
            # IMPROVED: Ensure directory exists before saving
            os.makedirs(os.path.dirname(separate_save_path), exist_ok=True)
            
            plt.savefig(separate_save_path, dpi=300, bbox_inches='tight')
            saved_files.append(separate_save_path)
            
            if verbose:
                print(f"{embedding_type} plot saved to: {separate_save_path}")
            
            plt.close(fig)  # Close to free memory
        
        if verbose:
            print(f"Separate plots saved: {saved_files}")
        
        # Return early since we've saved the separate plots
        return None, None
    
    # Create subplots for combined view
    n_plots = len(available_embeddings)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=True)
    
    if n_plots == 1:
        axes = [axes]
    
    # Get valid values
    valid_values = target_values[pd.notna(target_values)]
    
    # Create plots for each available embedding
    for i, (embedding_type, embedding_key) in enumerate(available_embeddings):
        ax = axes[i]
        
        ax, _, _ = plot_multimodal_embedding(
            adata, modality_col, color_col, target_modality, embedding_key, ax,
            point_size, alpha, colormap, show_sample_names,
            data_type=data_type, unique_values=unique_values
        )
        
        # Set title based on embedding type
        if embedding_type == 'Expression':
            title = f'Expression Embedding'
        else:
            title = f'Cell Proportion Embedding'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Dimension 1')
        if i == 0:
            ax.set_ylabel('Dimension 2')
    
    # Add shared colorbar for numerical data only
    if data_type == 'numerical' and len(valid_values) > 1:
        # Create colorbar on the right side
        norm = Normalize(vmin=min(valid_values), vmax=max(valid_values))
        sm = ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        
        # Add colorbar to the figure
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        
        # Set colorbar ticks and labels
        unique_vals = sorted(set(valid_values))
        if len(unique_vals) <= 10:
            cbar.set_ticks(unique_vals)
            cbar.set_ticklabels([f'{v:.1f}' if v != int(v) else f'{int(v)}' for v in unique_vals])
        else:
            # If too many unique values, use fewer ticks
            n_ticks = 5
            tick_values = np.linspace(min(valid_values), max(valid_values), n_ticks)
            cbar.set_ticks(tick_values)
            cbar.set_ticklabels([f'{v:.1f}' for v in tick_values])
        
        cbar.set_label(f'{color_col} ({target_modality})', rotation=270, labelpad=20)
    
    # Add main title
    main_title = f'Multi-modal Embedding: {target_modality} colored by {color_col}'
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout to accommodate colorbar or legend
    if data_type == 'numerical':
        plt.subplots_adjust(right=0.9)
    else:
        # For categorical data with many categories, might need more space
        if len(unique_values) > 10:
            plt.subplots_adjust(right=0.85)
        else:
            plt.subplots_adjust(right=0.9)
    
    # IMPROVED: Save combined plot if requested and no separate plots were saved
    if output_dir and not (len(available_embeddings) == 2):
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(output_dir) if os.path.dirname(output_dir) else '.', exist_ok=True)
        
        plt.savefig(output_dir, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Combined plot saved to: {output_dir}")
    
    return fig, axes


# ----------------- New Functions for CCA Visualization -----------------
def visualize_multimodal_embedding_with_cca(
    adata, 
    modality_col, 
    color_col, 
    target_modality,
    cca_results_df=None,
    expression_key='X_DR_expression',
    proportion_key='X_DR_proportion',
    figsize=(10, 8),
    point_size=60,
    alpha=0.8,
    colormap='viridis',
    output_dir=None,
    show_sample_names=False,
    show_cca_vectors=True,
    vector_scale=0.3,
    verbose=True
):
    """
    Create 3 plots for multimodal embedding visualization with CCA direction vectors.
    """
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    embeddings = [
        ('expression', expression_key),
        ('proportion', proportion_key)
    ]
    
    # Create plots 1 & 2: Expression and Proportion embeddings with CCA
    for emb_type, emb_key in embeddings:
        if emb_key not in adata.obsm and emb_key not in adata.uns:
            if verbose:
                print(f"Warning: {emb_type} embedding '{emb_key}' not found")
            continue
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get data using helper function
        x_coords, y_coords, sample_names, _ = get_embedding_data(adata, emb_key, verbose=False)
        
        # Get modality and color values
        modalities = adata.obs[modality_col].values
        colors = adata.obs[color_col].values
        
        # Plot non-target modality as gray background
        non_target_mask = modalities != target_modality
        if non_target_mask.any():
            ax.scatter(x_coords[non_target_mask], y_coords[non_target_mask],
                      c='lightgray', s=point_size, alpha=0.4,
                      edgecolors='black', linewidth=0.5,
                      label='Other modalities')
        
        # Plot target modality with proper gradient coloring
        target_mask = modalities == target_modality
        if target_mask.any():
            target_colors = colors[target_mask]
            target_x = x_coords[target_mask]
            target_y = y_coords[target_mask]
            
            # Detect data type using helper
            data_type, unique_values = detect_data_type(target_colors)
            
            # Handle numerical data with continuous gradient
            if data_type == 'numerical':
                numeric_colors = pd.to_numeric(target_colors, errors='coerce')
                # Use pandas isna for proper handling
                valid_mask = ~pd.isna(numeric_colors)
                
                if valid_mask.any():
                    # Use continuous gradient colormap
                    scatter = ax.scatter(target_x[valid_mask], target_y[valid_mask],
                                       c=numeric_colors[valid_mask], 
                                       s=point_size, alpha=alpha,
                                       cmap=colormap, edgecolors='black', 
                                       linewidth=0.5)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                    cbar.set_label(color_col, rotation=270, labelpad=20)
                
                # Plot missing values
                if (~valid_mask).any():
                    ax.scatter(target_x[~valid_mask], target_y[~valid_mask],
                             c='red', s=point_size, alpha=alpha,
                             edgecolors='black', linewidth=0.5,
                             label=f'{target_modality} (missing)')
            else:
                # Categorical coloring using helper
                color_map = create_categorical_colormap(unique_values, colormap)
                for category in sorted(unique_values):
                    cat_mask = target_colors == category
                    if cat_mask.any():
                        ax.scatter(target_x[cat_mask], target_y[cat_mask],
                                 c=[color_map[category]], s=point_size, alpha=alpha,
                                 edgecolors='black', linewidth=0.5,
                                 label=f'{target_modality}: {category}')
        
        # Add CCA vectors if requested
        cca_score_text = ""
        if cca_results_df is not None and show_cca_vectors:
            cca_score_text = add_cca_vectors(ax, cca_results_df, emb_key, 
                                           target_modality, vector_scale, verbose)
        
        # Add sample names if requested
        if show_sample_names and target_mask.any():
            for i, name in enumerate(sample_names[target_mask]):
                ax.annotate(name, (target_x[i], target_y[i]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(f'{emb_type.capitalize()} Embedding: {target_modality} by {color_col}{cca_score_text}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        
        plt.tight_layout()
        
        # Save plot
        if output_dir:
            filename = f'{target_modality}_{color_col}_{emb_type}_embedding.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            if verbose:
                print(f"Saved {emb_type} plot to: {filepath}")
        
        plt.close()
    
    # Create plot 3: Modality comparison
    create_modality_comparison_plot(adata, modality_col, embeddings, 
                                   point_size, alpha, output_dir, 
                                   saved_files, verbose)
    
    if verbose:
        print(f"\nVisualization complete. Created {len(saved_files)} plots.")
    
    return saved_files


def create_modality_comparison_plot(adata, modality_col, embeddings, 
                                   point_size, alpha, output_dir, 
                                   saved_files, verbose):
    """Create modality comparison plot showing both embeddings side by side."""
    
    # Check which embeddings are available
    available_embeddings = []
    for emb_name, emb_key in embeddings:
        if emb_key in adata.obsm or emb_key in adata.uns:
            available_embeddings.append((emb_name, emb_key))
    
    if len(available_embeddings) == 0:
        if verbose:
            print("No embeddings found for modality comparison plot")
        return
    
    # Create subplots based on available embeddings
    if len(available_embeddings) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        axes = axes.flatten()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        axes = [ax]
    
    # Define modality colors
    modality_colors = {'RNA': '#2E86AB', 'ATAC': '#E63946'}
    
    # Plot each available embedding
    for idx, (emb_name, emb_key) in enumerate(available_embeddings):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Get embedding data
        x_coords, y_coords, _, _ = get_embedding_data(adata, emb_key, verbose=False)
        modalities = adata.obs[modality_col].values
        
        # Plot each modality with different colors
        for mod in pd.unique(modalities):
            mod_mask = modalities == mod
            color = modality_colors.get(mod, '#A8DADC')
            ax.scatter(x_coords[mod_mask], y_coords[mod_mask],
                      c=color, s=point_size, alpha=alpha,
                      edgecolors='black', linewidth=0.5,
                      label=mod)
        
        # Formatting
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(f'{emb_name.capitalize()} Embedding: Modality Comparison', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        filepath = os.path.join(output_dir, 'modality_comparison_embedding.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        saved_files.append(filepath)
        if verbose:
            print(f"Saved modality comparison plot to: {filepath}")
    
    plt.close()


def add_cca_vectors(ax, cca_results_df, embedding_key, modality, 
                    scale_factor=0.3, verbose=True):
    """Add CCA direction vectors to the plot."""
    
    # Find matching CCA result
    mask = (cca_results_df['column'] == embedding_key) & \
           (cca_results_df['modality'] == modality)
    
    if not mask.any():
        return ""
    
    row = cca_results_df[mask].iloc[0]
    
    # Get CCA score
    cca_score = row['cca_score'] if 'cca_score' in row else np.nan
    cca_score_text = f" (CCA: {cca_score:.3f})" if not np.isnan(cca_score) else ""
    
    # Get weights
    x_weights = row['X_weights'] if 'X_weights' in row else None
    
    if x_weights is None or len(x_weights) < 2:
        return cca_score_text
    
    # Normalize weights
    x_weights = np.array(x_weights)[:2]
    weight_norm = np.linalg.norm(x_weights)
    if weight_norm > 0:
        x_weights_norm = x_weights / weight_norm
    else:
        return cca_score_text
    
    # Get plot center and scale
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    center_x = (xlim[0] + xlim[1]) / 2
    center_y = (ylim[0] + ylim[1]) / 2
    plot_scale = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) * scale_factor
    
    # Draw CCA vectors
    dx = x_weights_norm[0] * plot_scale
    dy = x_weights_norm[1] * plot_scale
    
    # Main arrow
    ax.arrow(center_x, center_y, dx, dy,
            head_width=plot_scale*0.1, head_length=plot_scale*0.1,
            fc='darkred', ec='darkred', linewidth=4, alpha=0.9,
            label='CCA Direction', zorder=15)
    
    # Negative direction
    ax.arrow(center_x, center_y, -dx, -dy,
            head_width=plot_scale*0.08, head_length=plot_scale*0.08,
            fc='darkred', ec='darkred', alpha=0.4, linewidth=2.5,
            zorder=14)
    
    # Origin marker
    circle = plt.Circle((center_x, center_y), plot_scale*0.03, 
                       color='white', ec='darkred', linewidth=2, zorder=16)
    ax.add_patch(circle)
    
    if verbose:
        print(f"Added CCA vector for {modality} {embedding_key}")
    
    return cca_score_text


import os
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent figure warnings

def glue_visualize(integrated_path, output_dir=None, plot_columns=None):
    """
    Load integrated RNA-ATAC data and create joint UMAP visualization
    
    Parameters:
    -----------
    integrated_path : str
        Path to integrated h5ad file (e.g., atac_rna_integrated.h5ad)
    output_dir : str, optional
        Output directory. If None, uses directory of integrated file
    plot_columns : list, optional
        List of column names to plot. If None, defaults to ['modality', 'cell_type']
    """
    
    # Load the integrated data
    print("Loading integrated RNA-ATAC data...")
    combined = ad.read_h5ad(integrated_path)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(integrated_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if scGLUE embeddings exist
    if "X_glue" not in combined.obsm:
        print("Error: X_glue embeddings not found in integrated data. Run scGLUE integration first.")
        return
    
    print("Computing UMAP from scGLUE embeddings...")
    # Compute neighbors and UMAP using the scGLUE embeddings
    sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")
    sc.tl.umap(combined)
    
    # Set up plotting parameters
    sc.settings.set_figure_params(dpi=80, facecolor='white', figsize=(8, 6))
    plt.rcParams['figure.max_open_warning'] = 50  # Increase the warning threshold
    
    # Set default columns if none specified
    if plot_columns is None:
        plot_columns = ['modality', 'cell_type']
    
    print(f"Generating visualizations for columns: {plot_columns}")
    
    # Generate plots for specified columns
    for col in plot_columns:
        if col not in combined.obs.columns:
            print(f"Warning: Column '{col}' not found in data. Skipping...")
            continue
        
        # Skip columns that also exist in var_names to avoid ambiguity
        if col in combined.var_names:
            print(f"Warning: Column '{col}' exists in both obs.columns and var_names. Skipping...")
            continue
        
        try:
            plt.figure(figsize=(12, 8))
            sc.pl.umap(combined, color=col, 
                       title=f"scGLUE Integration: {col}",
                       save=False, show=False, wspace=0.65)
            plt.tight_layout()
            col_plot_path = os.path.join(output_dir, f"scglue_umap_{col}.png")
            plt.savefig(col_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {col} plot: {col_plot_path}")
        except Exception as e:
            print(f"Error plotting {col}: {str(e)}")
            plt.close()  # Close the figure even if there's an error
    
    # Create modality-cell type distribution heatmap if both columns exist and are in plot_columns
    if ("modality" in plot_columns and "modality" in combined.obs.columns and 
        "cell_type" in plot_columns and "cell_type" in combined.obs.columns):
        print("\nCreating modality-cell type distribution heatmap...")
        
        # Create a cross-tabulation of modality and cell type
        modality_celltype_counts = pd.crosstab(
            combined.obs['cell_type'], 
            combined.obs['modality']
        )
        
        # Calculate proportions (percentage of each modality within each cell type)
        modality_celltype_props = modality_celltype_counts.div(
            modality_celltype_counts.sum(axis=1), 
            axis=0
        ) * 100
        
        # Create two heatmaps: one for counts and one for proportions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Heatmap 1: Raw counts
        sns.heatmap(
            modality_celltype_counts,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar_kws={'label': 'Number of cells'},
            ax=ax1
        )
        ax1.set_title('Cell Count Distribution: Modality vs Cell Type', fontsize=14, pad=20)
        ax1.set_xlabel('Modality', fontsize=12)
        ax1.set_ylabel('Cell Type', fontsize=12)
        
        # Heatmap 2: Proportions
        sns.heatmap(
            modality_celltype_props,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Percentage (%)'},
            ax=ax2
        )
        ax2.set_title('Modality Distribution within Each Cell Type (%)', fontsize=14, pad=20)
        ax2.set_xlabel('Modality', fontsize=12)
        ax2.set_ylabel('Cell Type', fontsize=12)
        
        plt.tight_layout()
        heatmap_plot_path = os.path.join(output_dir, "scglue_modality_celltype_heatmap.png")
        plt.savefig(heatmap_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved modality-cell type heatmap: {heatmap_plot_path}")
        
        # Also create a stacked bar plot for better visualization
        plt.figure(figsize=(14, 8))
        
        # Calculate percentages for stacked bar
        modality_celltype_props_T = modality_celltype_props.T
        
        # Create stacked bar plot
        modality_celltype_props_T.plot(
            kind='bar',
            stacked=True,
            colormap='tab20',
            width=0.8
        )
        
        plt.title('Modality Composition by Cell Type', fontsize=16, pad=20)
        plt.xlabel('Modality', fontsize=12)
        plt.ylabel('Percentage of Cells (%)', fontsize=12)
        plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        stacked_plot_path = os.path.join(output_dir, "scglue_modality_celltype_stacked.png")
        plt.savefig(stacked_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved modality-cell type stacked plot: {stacked_plot_path}")
        
        # Print summary statistics
        print("\n=== Modality-Cell Type Distribution Summary ===")
        print(f"Total cell types: {len(modality_celltype_counts.index)}")
        print(f"Total modalities: {len(modality_celltype_counts.columns)}")
        
        # Print cells per modality per cell type
        print("\nCell counts by modality and cell type:")
        print(modality_celltype_counts.to_string())
        
        # Save the distribution table as CSV
        csv_path = os.path.join(output_dir, "scglue_modality_celltype_distribution.csv")
        modality_celltype_counts.to_csv(csv_path)
        print(f"\nSaved distribution table to: {csv_path}")
        
    # Generate summary statistics
    print("\n=== Integration Summary ===")
    print(f"Total cells: {combined.n_obs}")
    print(f"Total features: {combined.n_vars}")
    print(f"Available metadata columns: {list(combined.obs.columns)}")
    
    # Show breakdown by modality if available
    if "modality" in combined.obs.columns:
        modality_counts = combined.obs['modality'].value_counts()
        print(f"\nModality breakdown:")
        for modality, count in modality_counts.items():
            print(f"  {modality}: {count} cells")
    
    # Check feature usage
    hvg_used = combined.var['highly_variable'].sum() if 'highly_variable' in combined.var else combined.n_vars
    print(f"\nFeatures used: {hvg_used}/{combined.n_vars}")
    
    print("\nVisualization complete!")
    