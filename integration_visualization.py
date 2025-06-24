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

def visualize_multimodal_embedding(adata, modality_col, color_col, target_modality,
                                  expression_key='X_DR_expression', proportion_key='X_DR_proportion',
                                  figsize=(20, 8), point_size=60, alpha=0.8, 
                                  colormap='viridis', output_dir=None, 
                                  show_sample_names=False, force_data_type=None, verbose=True):
    """
    Visualize multimodal embeddings with flexible coloring by any column.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing embeddings and metadata
    modality_col : str
        Column name in adata.obs containing modality information
    color_col : str
        Column name in adata.obs to use for coloring points (numerical or categorical)
    target_modality : str
        Which modality to highlight in the visualization
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
    output_dir : str
        Directory or file path to save plots
    show_sample_names : bool
        Whether to show sample names on plot (only for target modality, default: False)
    force_data_type : str or None
        Force data type to 'numerical' or 'categorical' instead of auto-detection (default: None)
    verbose : bool
        Print progress messages (default: True)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
        The created visualization (None if saved separately)
    """
    
    if verbose:
        print(f"Creating multimodal embedding visualization for {target_modality}")
        print(f"Coloring by: {color_col}")
        print(f"Expression key: {expression_key}")
        print(f"Proportion key: {proportion_key}")
        if show_sample_names:
            print(f"Sample names will be shown only for {target_modality} modality")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating output_dir")
    output_dir = os.path.join(output_dir, 'visualization')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating visualization output_dir")
    
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
    
    # If both embeddings are available and output_dir is provided, save separately
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
        
        # Create output directory if it doesn't exist
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
    
    # Save combined plot if requested and no separate plots were saved
    if output_dir and not (len(available_embeddings) == 2):
        save_dir = os.path.dirname(output_dir)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(output_dir, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Combined plot saved to: {output_dir}")
    
    return fig, axes

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

def extract_cca_vectors(cca_results_df, embedding_type, modality, verbose=True):
    """
    Extract CCA direction vectors from CCA results dataframe.
    
    Parameters:
    -----------
    cca_results_df : pandas.DataFrame
        DataFrame containing CCA results with columns like 'column', 'modality', 'X_weights', 'Y_weights'
    embedding_type : str
        Either 'expression' or 'proportion'
    modality : str
        Target modality ('RNA' or 'ATAC')
    
    Returns:
    --------
    tuple: (x_weights, y_weights) or (None, None) if not found
    """
    if cca_results_df is None or cca_results_df.empty:
        if verbose:
            print("No CCA results provided")
        return None, None
    
    # Look for the specific embedding type and modality
    target_column = f'X_DR_{embedding_type}'
    mask = (cca_results_df['column'] == target_column) & (cca_results_df['modality'] == modality)
    
    if not mask.any():
        if verbose:
            print(f"No CCA results found for {modality} {embedding_type}")
        return None, None
    
    row = cca_results_df[mask].iloc[0]
    
    # Extract the first canonical direction (first 2 components for visualization)
    x_weights = row['X_weights']  # Should be array-like
    y_weights = row['Y_weights']  # Should be array-like
    
    if x_weights is not None and len(x_weights) >= 2:
        x_direction = x_weights[:2]  # First 2 components
    else:
        x_direction = None
    
    if y_weights is not None and len(y_weights) >= 2:
        y_direction = y_weights[:2]  # First 2 components
    else:
        y_direction = None
    
    if verbose and x_direction is not None:
        print(f"Found CCA vectors for {modality} {embedding_type}")
    
    return x_direction, y_direction

def plot_cca_vectors(ax, x_direction, y_direction, center_point=None, scale_factor=2.0, 
                     vector_color='red', vector_width=3, alpha=0.8):
    """
    Plot CCA direction vectors as arrows on the embedding plot.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    x_direction : array-like
        X canonical weights (first 2 components)
    y_direction : array-like  
        Y canonical weights (first 2 components)
    center_point : tuple
        (x, y) coordinates for vector origin. If None, use plot center
    scale_factor : float
        Scaling factor for vector length
    """
    if x_direction is None or y_direction is None:
        return
    
    # Get plot limits to determine center and scale
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    if center_point is None:
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
    else:
        center_x, center_y = center_point
    
    # Scale vectors based on plot dimensions
    plot_width = xlim[1] - xlim[0]
    plot_height = ylim[1] - ylim[0]
    base_scale = min(plot_width, plot_height) * 0.2 * scale_factor
    
    # Plot X canonical direction (severity -> embedding space)
    if len(x_direction) >= 2:
        dx = x_direction[0] * base_scale
        dy = x_direction[1] * base_scale
        ax.arrow(center_x, center_y, dx, dy, 
                head_width=base_scale*0.1, head_length=base_scale*0.1,
                fc=vector_color, ec=vector_color, alpha=alpha, linewidth=vector_width,
                label='CCA Direction (Severity→Embedding)', zorder=10)
    
    # Optionally plot Y canonical direction (embedding -> severity space)  
    # Usually we only show the X direction for clarity, but this can be enabled
    # if len(y_direction) >= 2:
    #     dx2 = y_direction[0] * base_scale * 0.5  # Smaller scale
    #     dy2 = y_direction[1] * base_scale * 0.5
    #     ax.arrow(center_x, center_y, dx2, dy2,
    #             head_width=base_scale*0.05, head_length=base_scale*0.05,
    #             fc='blue', ec='blue', alpha=alpha*0.7, linewidth=vector_width-1,
    #             label='CCA Y Direction', zorder=9)

def plot_embedding_with_cca(adata, modality_col, color_col, target_modality,
                           embedding_key, embedding_type, cca_results_df,
                           ax, point_size=60, alpha=0.8, colormap='viridis', 
                           show_sample_names=False, non_target_color='lightgray', 
                           non_target_alpha=0.4, data_type=None, unique_values=None,
                           show_cca_vectors=True, verbose=True):
    """
    Plot embedding with CCA direction vectors overlaid.
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
        target_sample_names = sample_names[target_mask]
        
        # Handle valid color values
        valid_mask = pd.notna(target_color_values)
        
        if np.any(valid_mask):
            valid_values = target_color_values[valid_mask]
            valid_x = target_x[valid_mask]
            valid_y = target_y[valid_mask]
            valid_names = target_sample_names[valid_mask]
            
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
            missing_names = target_sample_names[missing_mask]
            
            ax.scatter(missing_x, missing_y, c='red', s=point_size, alpha=alpha,
                      edgecolors='black', linewidth=0.5, 
                      label=f'{target_modality} (missing {color_col})', zorder=2)
            
            # Add sample labels for missing target modality samples only
            if show_sample_names:
                for i, sample in enumerate(missing_names):
                    ax.annotate(sample, (missing_x[i], missing_y[i]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8, color='red')
    
    # Add CCA direction vectors
    if show_cca_vectors and cca_results_df is not None:
        x_direction, y_direction = extract_cca_vectors(
            cca_results_df, embedding_type, target_modality, verbose=verbose
        )
        
        if x_direction is not None:
            plot_cca_vectors(ax, x_direction, y_direction, 
                           scale_factor=2.0, vector_color='red', 
                           vector_width=3, alpha=0.8)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(True, alpha=0.3)
    
    # Adjust legend position based on data type
    if data_type == 'categorical' and len(unique_values) > 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1 if len(unique_values) <= 15 else 2)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return ax, data_type, unique_values

def plot_modality_comparison(adata, modality_col, embedding_key, ax, point_size=60, alpha=0.8):
    """
    Plot embedding with samples colored by their modality (RNA vs ATAC).
    """
    
    x_coords, y_coords, sample_names, coord_source = get_embedding_data(adata, embedding_key, verbose=False)
    
    modality_values = adata.obs[modality_col].values
    unique_modalities = sorted(set(modality_values))
    
    # Create color mapping for modalities
    modality_colors = {'RNA': '#1f77b4', 'ATAC': '#ff7f0e'}  # Blue for RNA, Orange for ATAC
    
    # If there are other modalities, assign additional colors
    if len(unique_modalities) > 2:
        additional_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_modalities)))
        for i, mod in enumerate(unique_modalities):
            if mod not in modality_colors:
                modality_colors[mod] = additional_colors[i]
    
    # Plot each modality separately for clear legend
    for modality in unique_modalities:
        mask = modality_values == modality
        if np.any(mask):
            color = modality_colors.get(modality, '#gray')
            ax.scatter(x_coords[mask], y_coords[mask], 
                     c=color, s=point_size, alpha=alpha,
                     edgecolors='black', linewidth=0.5, 
                     label=f'{modality}', zorder=2)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return ax

def visualize_multimodal_embedding_with_cca(adata, modality_col, color_col, target_modality,
                                           cca_results_df=None, 
                                           expression_key='X_DR_expression', 
                                           proportion_key='X_DR_proportion',
                                           figsize=(24, 8), point_size=60, alpha=0.8, 
                                           colormap='viridis', output_dir=None, 
                                           show_sample_names=False, force_data_type=None,
                                           show_cca_vectors=True, verbose=True):
    """
    Enhanced multimodal embedding visualization with CCA direction vectors.
    
    Creates 3 plots:
    1. Expression embedding with target modality colored by color_col + CCA vectors
    2. Proportion embedding with target modality colored by color_col + CCA vectors  
    3. Combined modality plot showing all samples colored by modality
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing embeddings and metadata
    modality_col : str
        Column name in adata.obs containing modality information
    color_col : str
        Column name in adata.obs to use for coloring points (numerical or categorical)
    target_modality : str
        Which modality to highlight in the first two plots
    cca_results_df : pandas.DataFrame
        DataFrame containing CCA results with columns like 'column', 'modality', 'X_weights', 'Y_weights'
    expression_key : str
        Key for expression-based embedding (default: 'X_DR_expression')
    proportion_key : str
        Key for proportion-based embedding (default: 'X_DR_proportion')
    figsize : tuple
        Figure size for combined plot (default: (24, 8))
    point_size : int
        Size of scatter points (default: 60)
    alpha : float
        Transparency of points (default: 0.8)
    colormap : str
        Colormap to use for numerical data (default: 'viridis')
    output_dir : str
        Directory to save plots
    show_sample_names : bool
        Whether to show sample names on plot (default: False)
    force_data_type : str or None
        Force data type to 'numerical' or 'categorical' (default: None)
    show_cca_vectors : bool
        Whether to show CCA direction vectors (default: True)
    verbose : bool
        Print progress messages (default: True)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
        The created visualization
    """
    
    if verbose:
        print(f"Creating enhanced multimodal embedding visualization for {target_modality}")
        print(f"Coloring by: {color_col}")
        print(f"Expression key: {expression_key}")
        print(f"Proportion key: {proportion_key}")
        if show_cca_vectors:
            print("CCA direction vectors will be overlaid")
        if show_sample_names:
            print(f"Sample names will be shown only for {target_modality} modality")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Output directory: {output_dir}")
    
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
    
    # Create 3 subplots
    n_plots = len(available_embeddings) + 1  # +1 for modality comparison plot
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=True)
    
    if n_plots == 1:
        axes = [axes]
    
    # Get valid values for colorbar
    valid_values = target_values[pd.notna(target_values)]
    
    # Plot 1 & 2: Create plots for each available embedding with CCA vectors
    for i, (embedding_type, embedding_key) in enumerate(available_embeddings):
        ax = axes[i]
        
        ax, _, _ = plot_embedding_with_cca(
            adata, modality_col, color_col, target_modality, embedding_key, 
            embedding_type.lower(), cca_results_df, ax,
            point_size, alpha, colormap, show_sample_names,
            data_type=data_type, unique_values=unique_values,
            show_cca_vectors=show_cca_vectors, verbose=verbose
        )
        
        # Set title based on embedding type
        title = f'{embedding_type} Embedding\n{target_modality} by {color_col}'
        if show_cca_vectors:
            title += ' + CCA'
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Dimension 1')
        if i == 0:
            ax.set_ylabel('Dimension 2')
    
    # Plot 3: Modality comparison plot (use the first available embedding)
    if available_embeddings:
        comparison_ax = axes[len(available_embeddings)]
        comparison_embedding_key = available_embeddings[0][1]  # Use first available embedding
        
        comparison_ax = plot_modality_comparison(
            adata, modality_col, comparison_embedding_key, comparison_ax,
            point_size, alpha
        )
        
        comparison_ax.set_title(f'Modality Comparison\n(All samples by modality)', fontsize=12, fontweight='bold')
        comparison_ax.set_xlabel('Dimension 1')
    
    # Add shared colorbar for numerical data only (for first two plots)
    if data_type == 'numerical' and len(valid_values) > 1:
        # Create colorbar on the right side
        norm = Normalize(vmin=min(valid_values), vmax=max(valid_values))
        sm = ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        
        # Add colorbar to the figure
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
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
    main_title = f'Enhanced Multi-modal Embedding Analysis: {target_modality}'
    if show_cca_vectors:
        main_title += ' with CCA Directions'
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout to accommodate colorbar or legend
    if data_type == 'numerical':
        plt.subplots_adjust(right=0.91)
    else:
        # For categorical data with many categories, might need more space
        if len(unique_values) > 10:
            plt.subplots_adjust(right=0.85)
        else:
            plt.subplots_adjust(right=0.88)
    
    # Save plot if requested
    if output_dir:
        save_path = os.path.join(output_dir, f'enhanced_embedding_{target_modality}_{color_col}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Enhanced plot saved to: {save_path}")
    
    return fig, axes