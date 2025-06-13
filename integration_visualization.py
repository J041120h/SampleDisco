import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os

def create_quantitative_colormap(values, colormap='viridis'):
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

def plot_multimodal_embedding(adata, modality_col, severity_col, target_modality,
                             embedding_key, ax, point_size=60, alpha=0.8, 
                             colormap='viridis', show_sample_names=False, 
                             non_target_color='lightgray', non_target_alpha=0.4):
    
    x_coords, y_coords, sample_names, coord_source = get_embedding_data(adata, embedding_key, verbose=False)
    
    modality_values = adata.obs[modality_col].values
    severity_values = adata.obs[severity_col].values
    
    target_mask = modality_values == target_modality
    non_target_mask = ~target_mask
    
    # Plot non-target modality samples first (background)
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
        
        # Handle valid severity values
        valid_severity_mask = pd.notna(target_severity)
        
        if np.any(valid_severity_mask):
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
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return ax

def create_single_embedding_plot(adata, modality_col, severity_col, target_modality,
                                embedding_key, embedding_type, figsize=(10, 8), 
                                point_size=60, alpha=0.8, colormap='viridis', 
                                show_sample_names=False, verbose=True):
    """Create a single embedding plot with colorbar"""
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create the plot
    plot_multimodal_embedding(
        adata, modality_col, severity_col, target_modality, embedding_key, ax,
        point_size, alpha, colormap, show_sample_names
    )
    
    # Get severity values for colorbar
    target_mask = adata.obs[modality_col].values == target_modality
    target_severity = adata.obs[severity_col].values[target_mask]
    valid_severity = target_severity[pd.notna(target_severity)]
    
    # Add colorbar for severity
    if len(valid_severity) > 1:
        norm = Normalize(vmin=min(valid_severity), vmax=max(valid_severity))
        sm = ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        
        # Add colorbar
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        
        # Set colorbar ticks and labels
        unique_severity = sorted(set(valid_severity))
        if len(unique_severity) <= 10:
            cbar.set_ticks(unique_severity)
            cbar.set_ticklabels([f'{v:.1f}' if v != int(v) else f'{int(v)}' for v in unique_severity])
        else:
            # If too many unique values, use fewer ticks
            n_ticks = 5
            tick_values = np.linspace(min(valid_severity), max(valid_severity), n_ticks)
            cbar.set_ticks(tick_values)
            cbar.set_ticklabels([f'{v:.1f}' for v in tick_values])
        
        cbar.set_label(f'{severity_col} ({target_modality})', rotation=270, labelpad=20)
    
    # Set title
    title = f'{embedding_type} Embedding: {target_modality} colored by {severity_col}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig, ax

def visualize_severity_trend(adata, modality_col, severity_col, target_modality,
                            expression_key='X_DR_expression', proportion_key='X_DR_proportion',
                            figsize=(20, 8), point_size=60, alpha=0.8, 
                            colormap='viridis', output_dir=None, 
                            show_sample_names=False, verbose=True):
    
    if verbose:
        print(f"Creating dual embedding visualization for {target_modality}")
        print(f"Expression key: {expression_key}")
        print(f"Proportion key: {proportion_key}")
    
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
            # Create filename based on modality and severity column
            base_name = f"{target_modality}_{severity_col}"
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
                adata, modality_col, severity_col, target_modality,
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
    
    # Get severity values for colorbar
    target_mask = adata.obs[modality_col].values == target_modality
    target_severity = adata.obs[severity_col].values[target_mask]
    valid_severity = target_severity[pd.notna(target_severity)]
    
    # Create plots for each available embedding
    for i, (embedding_type, embedding_key) in enumerate(available_embeddings):
        ax = axes[i]
        
        plot_multimodal_embedding(
            adata, modality_col, severity_col, target_modality, embedding_key, ax,
            point_size, alpha, colormap, show_sample_names
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
    
    # Add shared colorbar for severity
    if len(valid_severity) > 1:
        # Create colorbar on the right side
        norm = Normalize(vmin=min(valid_severity), vmax=max(valid_severity))
        sm = ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        
        # Add colorbar to the figure
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        
        # Set colorbar ticks and labels
        unique_severity = sorted(set(valid_severity))
        if len(unique_severity) <= 10:
            cbar.set_ticks(unique_severity)
            cbar.set_ticklabels([f'{v:.1f}' if v != int(v) else f'{int(v)}' for v in unique_severity])
        else:
            # If too many unique values, use fewer ticks
            n_ticks = 5
            tick_values = np.linspace(min(valid_severity), max(valid_severity), n_ticks)
            cbar.set_ticks(tick_values)
            cbar.set_ticklabels([f'{v:.1f}' for v in tick_values])
        
        cbar.set_label(f'{severity_col} ({target_modality})', rotation=270, labelpad=20)
    
    # Add main title
    main_title = f'Multi-modal Embedding: {target_modality} colored by {severity_col}'
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout to accommodate colorbar
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