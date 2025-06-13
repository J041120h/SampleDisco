import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os

def create_quantitative_colormap(values, colormap='viridis'):
    """
    Create a color mapping for quantitative/ordinal values using proportional spacing.
    This ensures consistent colors regardless of the actual range.
    
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

def get_embedding_data(adata, embedding_key, verbose=True):
    """
    Extract embedding data from either obsm or uns with proper handling.
    
    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object containing embedding results
    embedding_key : str
        Key for embedding coordinates
    verbose : bool
        Whether to print information
        
    Returns:
    --------
    tuple : (x_coords, y_coords, sample_names, coord_source)
    """
    # Get embedding coordinates
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
    
    # Check if we have at least 2 dimensions
    if embedding.shape[1] < 2:
        raise ValueError(f"Need at least 2 dimensions for visualization (found {embedding.shape[1]})")
    
    # Get coordinates
    if coord_source == "obsm":
        x_coords = embedding[:, 0]
        y_coords = embedding[:, 1]
        sample_names = adata.obs.index
    else:  # uns - assume it's a DataFrame
        if isinstance(embedding, pd.DataFrame):
            x_coords = embedding.iloc[:, 0]
            y_coords = embedding.iloc[:, 1]
            sample_names = embedding.index
        else:
            x_coords = embedding[:, 0]
            y_coords = embedding[:, 1]
            sample_names = adata.obs.index
    
    return x_coords, y_coords, sample_names, coord_source

def multimodal_embedding_visualization(adata, modality_col, severity_col, target_modality,
                                     embedding_key='X_umap', figsize=(12, 8), 
                                     point_size=60, alpha=0.8, colormap='viridis',
                                     save_path=None, title=None, show_sample_names=False,
                                     non_target_color='lightgray', non_target_alpha=0.4,
                                     verbose=True):
    """
    Visualize multi-modal embeddings with severity-based coloring for target modality only.
    
    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object containing embedding results and metadata
    modality_col : str
        Column name indicating the modality (e.g., 'modality', 'data_type')
    severity_col : str
        Column name for numerical severity values
    target_modality : str
        The modality to color by severity (other modalities will be gray)
    embedding_key : str, default 'X_umap'
        Key for embedding coordinates in adata.obsm or adata.uns
    figsize : tuple, default (12, 8)
        Figure size for the plot
    point_size : int, default 60
        Size of scatter plot points
    alpha : float, default 0.8
        Transparency of colored points
    colormap : str, default 'viridis'
        Matplotlib colormap for severity coloring
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Custom title for the plot
    show_sample_names : bool, default False
        Whether to show sample names as annotations
    non_target_color : str, default 'lightgray'
        Color for non-target modality samples
    non_target_alpha : float, default 0.4
        Transparency for non-target modality samples
    verbose : bool, default True
        Whether to print detailed information
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Get embedding data
    x_coords, y_coords, sample_names, coord_source = get_embedding_data(adata, embedding_key, verbose)
    
    # Get metadata
    modality_values = adata.obs[modality_col].values
    severity_values = adata.obs[severity_col].values
    
    if verbose:
        unique_modalities = list(set(modality_values))
        print(f"Found modalities: {unique_modalities}")
        print(f"Target modality for severity coloring: {target_modality}")
        print(f"Severity column: {severity_col}")
        
        # Check severity values for target modality
        target_mask = modality_values == target_modality
        target_severity = severity_values[target_mask]
        target_severity_clean = [s for s in target_severity if pd.notna(s)]
        
        if target_severity_clean:
            print(f"Severity range for {target_modality}: {min(target_severity_clean):.2f} to {max(target_severity_clean):.2f}")
            print(f"Number of {target_modality} samples: {len(target_severity_clean)}")
        else:
            print(f"Warning: No valid severity values found for {target_modality}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate samples by modality
    target_mask = modality_values == target_modality
    non_target_mask = ~target_mask
    
    # Plot non-target modality samples first (so they appear behind)
    if np.any(non_target_mask):
        ax.scatter(x_coords[non_target_mask], y_coords[non_target_mask], 
                  c=non_target_color, s=point_size, alpha=non_target_alpha,
                  edgecolors='black', linewidth=0.5, 
                  label=f'Other modalities', zorder=1)
    
    # Plot target modality samples with severity coloring
    if np.any(target_mask):
        target_x = x_coords[target_mask]
        target_y = y_coords[target_mask]
        target_severity = severity_values[target_mask]
        
        # Handle missing severity values
        valid_severity_mask = pd.notna(target_severity)
        
        if np.any(valid_severity_mask):
            # Get valid severity values for color mapping
            valid_severity = target_severity[valid_severity_mask]
            valid_x = target_x[valid_severity_mask]
            valid_y = target_y[valid_severity_mask]
            
            # Create color mapping
            color_map = create_quantitative_colormap(valid_severity, colormap)
            colors = [color_map[sev] for sev in valid_severity]
            
            # Plot samples with valid severity
            scatter = ax.scatter(valid_x, valid_y, c=colors, s=point_size, alpha=alpha,
                               edgecolors='black', linewidth=0.5, 
                               label=f'{target_modality} (by severity)', zorder=2)
            
            # Create colorbar
            unique_severity = sorted(set(valid_severity))
            if len(unique_severity) > 1:
                norm = Normalize(vmin=min(unique_severity), vmax=max(unique_severity))
                sm = ScalarMappable(norm=norm, cmap=colormap)
                sm.set_array([])
                
                cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
                cbar.set_ticks(unique_severity)
                cbar.set_ticklabels([f'{v:.1f}' if v != int(v) else f'{int(v)}' for v in unique_severity])
                cbar.set_label(f'{severity_col} ({target_modality})', rotation=270, labelpad=20)
        
        # Plot samples with missing severity values
        missing_severity_mask = ~valid_severity_mask
        if np.any(missing_severity_mask):
            missing_x = target_x[missing_severity_mask]
            missing_y = target_y[missing_severity_mask]
            ax.scatter(missing_x, missing_y, c='red', s=point_size, alpha=alpha,
                      edgecolors='black', linewidth=0.5, 
                      label=f'{target_modality} (missing severity)', zorder=2)
    
    # Add sample labels if requested
    if show_sample_names:
        for i, sample in enumerate(sample_names):
            ax.annotate(sample, (x_coords[i], y_coords[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel(f'Dimension 1')
    ax.set_ylabel(f'Dimension 2')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Multi-modal Embedding: {target_modality} colored by {severity_col}')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Plot saved to: {save_path}")
    
    return fig, ax

def multimodal_dual_embedding_visualization(adata, modality_col, severity_col,
                                          expression_key=None, proportion_key=None,
                                          atac=False, figsize=(20, 8), 
                                          point_size=60, alpha=0.8, colormap='viridis',
                                          save_path=None, show_sample_names=False,
                                          verbose=True):
    """
    Create dual visualization plots for both expression and proportion embeddings.
    Automatically detects available embedding keys based on the improved PCA code structure.
    
    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object containing embedding results and metadata
    modality_col : str
        Column name indicating the modality
    severity_col : str
        Column name for numerical severity values
    expression_key : str, optional
        Key for expression embedding. If None, auto-detects based on atac parameter
    proportion_key : str, optional
        Key for proportion embedding. Defaults to 'X_pca_proportion'
    atac : bool, default False
        If True, uses ATAC-specific keys (X_DR_expression) instead of X_pca_expression
    figsize : tuple, default (20, 8)
        Figure size for side-by-side plots
    point_size : int, default 60
        Size of scatter plot points
    alpha : float, default 0.8
        Transparency of points
    colormap : str, default 'viridis'
        Matplotlib colormap for severity coloring
    save_path : str, optional
        Path to save the figure
    show_sample_names : bool, default False
        Whether to show sample names as annotations
    verbose : bool, default True
        Whether to print detailed information
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Auto-detect embedding keys based on the improved PCA code structure
    if expression_key is None:
        if atac:
            expression_key = 'X_DR_expression'  # LSI for ATAC
        else:
            expression_key = 'X_pca_expression'  # PCA for RNA
    
    if proportion_key is None:
        proportion_key = 'X_pca_proportion'
    
    if verbose:
        print(f"Creating dual embedding visualization:")
        print(f"Expression embedding key: {expression_key}")
        print(f"Proportion embedding key: {proportion_key}")
        print(f"ATAC mode: {atac}")
    
    # Check which embeddings are available
    available_embeddings = []
    embedding_info = []
    
    # Check expression embedding
    expression_available = False
    if expression_key in adata.obsm or expression_key in adata.uns:
        available_embeddings.append(('Expression', expression_key))
        embedding_info.append(f"Expression: {expression_key}")
        expression_available = True
    else:
        if verbose:
            print(f"Warning: Expression embedding '{expression_key}' not found")
    
    # Check proportion embedding
    proportion_available = False
    if proportion_key in adata.obsm or proportion_key in adata.uns:
        available_embeddings.append(('Proportion', proportion_key))
        embedding_info.append(f"Proportion: {proportion_key}")
        proportion_available = True
    else:
        if verbose:
            print(f"Warning: Proportion embedding '{proportion_key}' not found")
    
    if not available_embeddings:
        available_obsm = list(adata.obsm.keys()) if hasattr(adata, 'obsm') else []
        available_uns = list(adata.uns.keys()) if hasattr(adata, 'uns') else []
        raise ValueError(f"No embeddings found. Available in obsm: {available_obsm}, uns: {available_uns}")
    
    if verbose:
        print(f"Available embeddings: {embedding_info}")
    
    # Create subplots
    n_plots = len(available_embeddings)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=True)
    
    # Handle single plot case
    if n_plots == 1:
        axes = [axes]
    
    # Get unique modalities for consistent coloring
    unique_modalities = sorted(list(set(adata.obs[modality_col].values)))
    
    if verbose:
        print(f"Found modalities: {unique_modalities}")
    
    # Create plots for each available embedding
    for i, (embedding_type, embedding_key) in enumerate(available_embeddings):
        ax = axes[i]
        
        # Plot each modality separately for better visualization
        _plot_multimodal_on_axis(
            ax, adata, modality_col, severity_col, embedding_key,
            point_size, alpha, colormap, show_sample_names, verbose=False
        )
        
        # Set title based on embedding type and ATAC mode
        if embedding_type == 'Expression':
            if atac:
                title = 'Expression Embedding (LSI)'
            else:
                title = 'Expression Embedding (PCA)'
        else:
            title = 'Cell Proportion Embedding (PCA)'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Dimension 1')
        if i == 0:  # Only set ylabel for first plot
            ax.set_ylabel('Dimension 2')
    
    # Add overall title
    main_title = f'Multi-modal Embedding Analysis'
    if atac:
        main_title += ' (ATAC-seq)'
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Dual embedding plot saved to: {save_path}")
    
    return fig, axes

def _plot_multimodal_on_axis(ax, adata, modality_col, severity_col, embedding_key,
                            point_size, alpha, colormap, show_sample_names, verbose=True):
    """
    Helper function to plot multi-modal data on a specific axis with all modalities colored by severity.
    """
    # Get embedding data
    x_coords, y_coords, sample_names, coord_source = get_embedding_data(adata, embedding_key, verbose)
    
    # Get metadata
    modality_values = adata.obs[modality_col].values
    severity_values = adata.obs[severity_col].values
    
    # Get unique modalities
    unique_modalities = sorted(list(set(modality_values)))
    
    # Define colors for different modalities
    modality_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_modalities)))
    modality_color_map = dict(zip(unique_modalities, modality_colors))
    
    # Plot each modality separately with severity-based coloring
    for modality in unique_modalities:
        modality_mask = modality_values == modality
        
        if np.any(modality_mask):
            mod_x = x_coords[modality_mask]
            mod_y = y_coords[modality_mask]
            mod_severity = severity_values[modality_mask]
            
            # Handle valid severity values
            valid_mask = pd.notna(mod_severity)
            
            if np.any(valid_mask):
                valid_severity = mod_severity[valid_mask]
                valid_x = mod_x[valid_mask]
                valid_y = mod_y[valid_mask]
                
                if len(set(valid_severity)) > 1:
                    # Multiple severity values - use colormap
                    color_map = create_quantitative_colormap(valid_severity, colormap)
                    colors = [color_map[sev] for sev in valid_severity]
                else:
                    # Single severity value - use modality color
                    colors = [modality_color_map[modality]] * len(valid_severity)
                
                scatter = ax.scatter(valid_x, valid_y, c=colors, s=point_size, alpha=alpha,
                                   edgecolors='black', linewidth=0.5, 
                                   label=f'{modality}', zorder=2)
            
            # Handle missing severity values
            missing_mask = ~valid_mask
            if np.any(missing_mask):
                missing_x = mod_x[missing_mask]
                missing_y = mod_y[missing_mask]
                ax.scatter(missing_x, missing_y, c='red', s=point_size, alpha=alpha,
                          edgecolors='black', linewidth=0.5, 
                          label=f'{modality} (missing severity)', zorder=2)
    
    # Add sample labels if requested
    if show_sample_names:
        for i, sample in enumerate(sample_names):
            ax.annotate(sample, (x_coords[i], y_coords[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
    
    # Add colorbar for severity if we have multiple severity values
    all_severity = severity_values[pd.notna(severity_values)]
    if len(set(all_severity)) > 1:
        norm = Normalize(vmin=min(all_severity), vmax=max(all_severity))
        sm = ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        
        # Only add colorbar to the rightmost plot to avoid clutter
        # This should be handled by the calling function
    
    # Add legend and styling
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

def multimodal_embedding_comparison(adata, modality_col, severity_col, 
                                  embedding_key='X_umap', figsize=(16, 6),
                                  point_size=60, alpha=0.8, colormap='viridis',
                                  save_path=None, show_sample_names=False,
                                  verbose=True):
    """
    Create side-by-side comparison plots for each modality colored by severity.
    
    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object containing embedding results and metadata
    modality_col : str
        Column name indicating the modality
    severity_col : str
        Column name for numerical severity values
    embedding_key : str, default 'X_umap'
        Key for embedding coordinates
    figsize : tuple, default (16, 6)
        Figure size for the subplot layout
    point_size : int, default 60
        Size of scatter plot points
    alpha : float, default 0.8
        Transparency of points
    colormap : str, default 'viridis'
        Matplotlib colormap for severity coloring
    save_path : str, optional
        Path to save the figure
    show_sample_names : bool, default False
        Whether to show sample names as annotations
    verbose : bool, default True
        Whether to print detailed information
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Get unique modalities
    unique_modalities = sorted(list(set(adata.obs[modality_col].values)))
    
    if verbose:
        print(f"Creating comparison plots for modalities: {unique_modalities}")
    
    # Create subplots
    n_modalities = len(unique_modalities)
    fig, axes = plt.subplots(1, n_modalities, figsize=figsize, sharey=True)
    
    # Handle single modality case
    if n_modalities == 1:
        axes = [axes]
    
    for i, modality in enumerate(unique_modalities):
        # Plot directly on the subplot
        _plot_on_axis(axes[i], adata, modality_col, severity_col, modality, 
                     embedding_key, point_size, alpha, colormap, show_sample_names)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Comparison plot saved to: {save_path}")
    
    return fig, axes

def _plot_on_axis(ax, adata, modality_col, severity_col, target_modality, 
                  embedding_key, point_size, alpha, colormap, show_sample_names):
    """
    Helper function to plot on a specific axis (for subplot functionality).
    """
    # Get embedding data
    x_coords, y_coords, sample_names, coord_source = get_embedding_data(adata, embedding_key, verbose=False)
    
    # Get metadata
    modality_values = adata.obs[modality_col].values
    severity_values = adata.obs[severity_col].values
    
    # Separate samples by modality
    target_mask = modality_values == target_modality
    non_target_mask = ~target_mask
    
    # Plot non-target modality samples
    if np.any(non_target_mask):
        ax.scatter(x_coords[non_target_mask], y_coords[non_target_mask], 
                  c='lightgray', s=point_size, alpha=0.4,
                  edgecolors='black', linewidth=0.5, zorder=1)
    
    # Plot target modality samples with severity coloring
    if np.any(target_mask):
        target_x = x_coords[target_mask]
        target_y = y_coords[target_mask]
        target_severity = severity_values[target_mask]
        
        # Handle valid severity values
        valid_mask = pd.notna(target_severity)
        if np.any(valid_mask):
            valid_severity = target_severity[valid_mask]
            valid_x = target_x[valid_mask]
            valid_y = target_y[valid_mask]
            
            # Create color mapping and plot
            color_map = create_quantitative_colormap(valid_severity, colormap)
            colors = [color_map[sev] for sev in valid_severity]
            
            ax.scatter(valid_x, valid_y, c=colors, s=point_size, alpha=alpha,
                      edgecolors='black', linewidth=0.5, zorder=2)
    
    # Add sample labels if requested
    if show_sample_names:
        target_samples = sample_names[target_mask]
        for i, sample in enumerate(target_samples):
            ax.annotate(sample, (target_x[i], target_y[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'{target_modality} by {severity_col}')
    ax.grid(True, alpha=0.3)

# Convenience function for automatic dual visualization
def auto_multimodal_visualization(adata, modality_col, severity_col,
                                atac=False, output_dir=None, 
                                figsize=(20, 8), verbose=True, **kwargs):
    """
    Automatically create dual embedding visualization plots with intelligent key detection.
    This is the main function users should call for standard visualization.
    
    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object containing embedding results and metadata
    modality_col : str
        Column name indicating the modality
    severity_col : str
        Column name for numerical severity values
    atac : bool, default False
        If True, uses ATAC-specific embedding keys
    output_dir : str, optional
        Directory to save plots. If None, plots are shown but not saved
    figsize : tuple, default (20, 8)
        Figure size for the plots
    verbose : bool, default True
        Whether to print detailed information
    **kwargs : additional keyword arguments
        Passed to multimodal_dual_embedding_visualization
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    if verbose:
        print("=== Auto Multi-modal Visualization ===")
        print(f"ATAC mode: {atac}")
        print(f"Modality column: {modality_col}")
        print(f"Severity column: {severity_col}")
    
    # Determine save path
    save_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        data_type = "ATAC" if atac else "RNA"
        filename = f"multimodal_embedding_{data_type.lower()}.png"
        save_path = os.path.join(output_dir, filename)
    
    # Create the dual visualization
    fig, axes = multimodal_dual_embedding_visualization(
        adata=adata,
        modality_col=modality_col,
        severity_col=severity_col,
        atac=atac,
        figsize=figsize,
        save_path=save_path,
        verbose=verbose,
        **kwargs
    )
    
    if verbose:
        print("=== Visualization Complete ===")
    
    return fig, axes