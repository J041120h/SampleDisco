import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import to_rgba
import os
from typing import Optional, Tuple, Dict, Any, List, Union
import warnings

# Set style for better aesthetics
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def extract_sample_id(sample_name: str) -> str:
    """
    Extract core sample ID by removing modality prefix (RNA_ or ATAC_).
    
    Parameters:
    -----------
    sample_name : str
        Full sample name with modality prefix
    
    Returns:
    --------
    str : Core sample ID without prefix
    """
    if sample_name.startswith('RNA_'):
        return sample_name[4:]
    elif sample_name.startswith('ATAC_'):
        return sample_name[5:]
    else:
        return sample_name

def visualize_cross_modal_connections(
    adata,
    expression_embedding_key: str = 'X_DR_expression',
    proportion_embedding_key: str = 'X_DR_proportion',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    point_size: Union[int, np.ndarray] = 100,
    point_alpha: float = 0.7,
    line_alpha: float = 0.4,
    line_width: float = 1.5,
    expression_color: str = '#1f77b4',  # Blue
    proportion_color: str = '#ff7f0e',  # Orange
    line_color: str = 'gray',
    show_sample_ids: bool = False,
    highlight_samples: Optional[List[str]] = None,
    highlight_color: str = 'red',
    highlight_line_width: float = 2.5,
    style: str = 'modern',
    output_dir: Optional[str] = None,
    filename_prefix: str = 'cross_modal',
    dpi: int = 300,
    **kwargs
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Visualize cross-modal sample connections between RNA expression and ATAC proportion embeddings.
    (Improved: robust ID parsing supports both prefix 'RNA_/ATAC_' and suffix '_RNA/_ATAC' formats, plus extra diagnostics.)
    """
    print("Starting cross-modal visualization...")

    # -----------------------
    # Style presets (unchanged)
    # -----------------------
    style_presets = {
        'modern': {'edge_color': 'white', 'edge_width': 1.0, 'grid': True, 'grid_alpha': 0.15, 'spine_width': 1.5},
        'classic': {'edge_color': 'black', 'edge_width': 0.5, 'grid': True, 'grid_alpha': 0.3, 'spine_width': 1.0},
        'minimal': {'edge_color': 'none', 'edge_width': 0, 'grid': False, 'grid_alpha': 0, 'spine_width': 0.5},
    }
    style_config = style_presets.get(style, style_presets['modern'])
    for key, value in style_config.items():
        if key not in kwargs:
            kwargs[key] = value

    print("Extracting embeddings...")

    # -----------------------
    # Embedding presence & shape checks
    # -----------------------
    if expression_embedding_key not in adata.obsm:
        print(f"[ERROR] Expression embedding key '{expression_embedding_key}' not found in adata.obsm. "
              f"Available keys (first 10): {list(adata.obsm.keys())[:10]}")
        raise KeyError(f"Expression embedding '{expression_embedding_key}' not found in adata.obsm")
    if proportion_embedding_key not in adata.obsm:
        print(f"[ERROR] Proportion embedding key '{proportion_embedding_key}' not found in adata.obsm. "
              f"Available keys (first 10): {list(adata.obsm.keys())[:10]}")
        raise KeyError(f"Proportion embedding '{proportion_embedding_key}' not found in adata.obsm")

    expr_embedding = adata.obsm[expression_embedding_key]
    prop_embedding = adata.obsm[proportion_embedding_key]

    if getattr(expr_embedding, "ndim", 2) != 2 or expr_embedding.shape[1] < 2:
        print(f"[ERROR] Expression embedding must be 2D with ≥2 columns. Got {getattr(expr_embedding, 'shape', None)}.")
        raise ValueError("Expression embedding must have at least 2 columns.")
    if getattr(prop_embedding, "ndim", 2) != 2 or prop_embedding.shape[1] < 2:
        print(f"[ERROR] Proportion embedding must be 2D with ≥2 columns. Got {getattr(prop_embedding, 'shape', None)}.")
        raise ValueError("Proportion embedding must have at least 2 columns.")

    expr_embedding = expr_embedding[:, :2]
    prop_embedding = prop_embedding[:, :2]

    for name, arr in [("expression", expr_embedding), ("proportion", prop_embedding)]:
        if not np.isfinite(arr).all():
            n_bad = np.size(arr) - np.isfinite(arr).sum()
            print(f"[WARNING] {name} embedding contains {n_bad} non-finite values (NaN/Inf). Plotting may be distorted.")

    # -----------------------
    # Sample names & diagnostics
    # -----------------------
    sample_names = np.asarray(adata.obs.index.values, dtype=str)
    n_samples = len(sample_names)
    print(f"Processing {n_samples} samples...")

    if isinstance(point_size, (np.ndarray, list)) and len(point_size) != n_samples:
        print(f"[WARNING] 'point_size' array length {len(point_size)} ≠ number of samples {n_samples}. Indexing may fail.")

    # Count naming patterns
    starts_rna = np.sum([s.startswith("RNA_") for s in sample_names])
    starts_atac = np.sum([s.startswith("ATAC_") for s in sample_names])
    ends_rna = np.sum([s.endswith("_RNA") for s in sample_names])
    ends_atac = np.sum([s.endswith("_ATAC") for s in sample_names])
    print(f"[Naming] prefix counts -> RNA_: {starts_rna}, ATAC_: {starts_atac}; "
          f"suffix counts -> _RNA: {ends_rna}, _ATAC: {ends_atac}")

    # Decide parsing mode
    if (starts_rna + starts_atac) >= (ends_rna + ends_atac) and (starts_rna + starts_atac) > 0:
        naming_mode = "prefix"
    elif (ends_rna + ends_atac) > 0:
        naming_mode = "suffix"
    else:
        naming_mode = "auto-none"  # nothing matches known patterns
    print(f"[Parsing] Using naming mode: {naming_mode}")

    # Robust parser that supports both formats
    def _parse_id_and_modality(name: str) -> Optional[Tuple[str, str]]:
        if naming_mode in ("prefix", "auto-none"):
            if name.startswith("RNA_"):
                return name[4:], "RNA"
            if name.startswith("ATAC_"):
                return name[5:], "ATAC"
        if naming_mode in ("suffix", "auto-none"):
            if name.endswith("_RNA"):
                return name[:-4], "RNA"
            if name.endswith("_ATAC"):
                return name[:-5], "ATAC"
        return None  # unknown format

    # -----------------------
    # Build modality maps
    # -----------------------
    rna_samples: Dict[str, Tuple[int, str]] = {}
    atac_samples: Dict[str, Tuple[int, str]] = {}
    dup_rna, dup_atac = [], []

    for idx, sample in enumerate(sample_names):
        parsed = _parse_id_and_modality(sample)
        if parsed is None:
            continue
        core_id, modality = parsed
        if modality == "RNA":
            if core_id in rna_samples:
                dup_rna.append(core_id)
            else:
                rna_samples[core_id] = (idx, sample)
        elif modality == "ATAC":
            if core_id in atac_samples:
                dup_atac.append(core_id)
            else:
                atac_samples[core_id] = (idx, sample)

    if dup_rna:
        print(f"[WARNING] Duplicate RNA core IDs detected (keeping first occurrence): {dup_rna[:10]}"
              f"{' ...' if len(dup_rna) > 10 else ''}")
    if dup_atac:
        print(f"[WARNING] Duplicate ATAC core IDs detected (keeping first occurrence): {dup_atac[:10]}"
              f"{' ...' if len(dup_atac) > 10 else ''}")

    matching_ids = set(rna_samples.keys()) & set(atac_samples.keys())
    print(f"Found {len(matching_ids)} matching sample pairs between RNA and ATAC")
    print(f"RNA samples detected: {len(rna_samples)}, ATAC samples detected: {len(atac_samples)}")

    if len(matching_ids) == 0:
        preview = list(sample_names[:10])
        print("[WARNING] No matching RNA/ATAC pairs found after robust parsing.")
        print("          First few sample names for inspection:", preview)
        print("          Expect names like 'RNA_<id>'/'ATAC_<id>' OR '<id>_RNA'/'<id>_ATAC'.")

    # -----------------------
    # Highlights
    # -----------------------
    highlight_mask: Dict[str, bool] = {}
    if highlight_samples:
        missing_highlights = []
        for core_id in highlight_samples:
            if core_id in matching_ids:
                highlight_mask[core_id] = True
            else:
                missing_highlights.append(core_id)
        print(f"Highlighting {len(highlight_mask)} sample pairs")
        if missing_highlights:
            print(f"[WARNING] {len(missing_highlights)} requested highlight IDs are not matched pairs: "
                  f"{missing_highlights[:10]}{' ...' if len(missing_highlights)>10 else ''}")

    # -----------------------
    # Output dir
    # -----------------------
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                print(f"[WARNING] Output directory '{output_dir}' is not writable. Saving may fail.")
        except Exception as e:
            print(f"[WARNING] Could not create/access output directory '{output_dir}': {e}")
        print(f"Output directory: {output_dir}")

    # ====================
    # PLOT 1: Expression Embedding
    # ====================
    print("\nCreating expression embedding plot...")
    fig1, ax1 = plt.subplots(figsize=figsize)

    for core_id, (idx, sample_name) in rna_samples.items():
        is_highlighted = core_id in highlight_mask
        color = highlight_color if is_highlighted else expression_color
        size = point_size * 1.5 if is_highlighted else point_size
        alpha = point_alpha * 1.2 if is_highlighted else point_alpha
        zorder = 5 if is_highlighted else 3
        try:
            ax1.scatter(
                expr_embedding[idx, 0], expr_embedding[idx, 1],
                s=size if isinstance(size, (int, float)) else size[idx],
                c=color, alpha=alpha,
                edgecolors=kwargs.get('edge_color', 'white'),
                linewidths=kwargs.get('edge_width', 1.0),
                zorder=zorder, marker='o'
            )
        except Exception as e:
            print(f"[WARNING] Failed plotting RNA sample '{sample_name}' (idx {idx}) in expression space: {e}")

    for core_id, (idx, sample_name) in atac_samples.items():
        is_highlighted = core_id in highlight_mask
        color = highlight_color if is_highlighted else proportion_color
        size = point_size * 1.5 if is_highlighted else point_size
        alpha = point_alpha * 1.2 if is_highlighted else point_alpha
        zorder = 5 if is_highlighted else 3
        try:
            ax1.scatter(
                expr_embedding[idx, 0], expr_embedding[idx, 1],
                s=size if isinstance(size, (int, float)) else size[idx],
                c=color, alpha=alpha,
                edgecolors=kwargs.get('edge_color', 'white'),
                linewidths=kwargs.get('edge_width', 1.0),
                zorder=zorder, marker='^'
            )
        except Exception as e:
            print(f"[WARNING] Failed plotting ATAC sample '{sample_name}' (idx {idx}) in expression space: {e}")

    print(f"Drawing {len(matching_ids)} connection lines in expression plot...")
    for core_id in matching_ids:
        rna_idx, _ = rna_samples[core_id]
        atac_idx, _ = atac_samples[core_id]
        is_highlighted = core_id in highlight_mask
        lw = highlight_line_width if is_highlighted else line_width
        lc = highlight_color if is_highlighted else line_color
        la = line_alpha * 1.5 if is_highlighted else line_alpha
        zorder = 2 if is_highlighted else 1
        try:
            ax1.plot(
                [expr_embedding[rna_idx, 0], expr_embedding[atac_idx, 0]],
                [expr_embedding[rna_idx, 1], expr_embedding[atac_idx, 1]],
                color=lc, alpha=la, linewidth=lw, zorder=zorder, linestyle='-'
            )
        except Exception as e:
            print(f"[WARNING] Failed drawing expression-space line for pair '{core_id}': {e}")

    ax1.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax1.set_title('Expression Embedding Space', fontsize=14, fontweight='bold', pad=15)
    if kwargs.get('grid', True):
        ax1.grid(True, alpha=kwargs.get('grid_alpha', 0.15), linestyle='--')
    for spine in ax1.spines.values():
        spine.set_linewidth(kwargs.get('spine_width', 1.5))

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=expression_color,
               markersize=10, label='RNA samples', markeredgecolor=kwargs.get('edge_color', 'white')),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=proportion_color,
               markersize=10, label='ATAC samples', markeredgecolor=kwargs.get('edge_color', 'white'))
    ]
    if highlight_samples:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=highlight_color,
                   markersize=10, label='Highlighted', markeredgecolor=kwargs.get('edge_color', 'white'))
        )
    try:
        ax1.legend(handles=legend_elements, loc='best', frameon=True, fancybox=True, shadow=True)
    except Exception as e:
        print(f"[WARNING] Failed to add legend to expression plot: {e}")

    plt.tight_layout()
    if output_dir:
        expr_path = os.path.join(output_dir, f"{filename_prefix}_expression.png")
        try:
            fig1.savefig(expr_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Expression plot saved to: {expr_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save expression plot to '{expr_path}': {e}")

    # ====================
    # PLOT 2: Proportion Embedding
    # ====================
    print("\nCreating proportion embedding plot...")
    fig2, ax2 = plt.subplots(figsize=figsize)

    for core_id, (idx, sample_name) in rna_samples.items():
        is_highlighted = core_id in highlight_mask
        color = highlight_color if is_highlighted else expression_color
        size = point_size * 1.5 if is_highlighted else point_size
        alpha = point_alpha * 1.2 if is_highlighted else point_alpha
        zorder = 5 if is_highlighted else 3
        try:
            ax2.scatter(
                prop_embedding[idx, 0], prop_embedding[idx, 1],
                s=size if isinstance(size, (int, float)) else size[idx],
                c=color, alpha=alpha,
                edgecolors=kwargs.get('edge_color', 'white'),
                linewidths=kwargs.get('edge_width', 1.0),
                zorder=zorder, marker='o'
            )
        except Exception as e:
            print(f"[WARNING] Failed plotting RNA sample '{sample_name}' (idx {idx}) in proportion space: {e}")

    for core_id, (idx, sample_name) in atac_samples.items():
        is_highlighted = core_id in highlight_mask
        color = highlight_color if is_highlighted else proportion_color
        size = point_size * 1.5 if is_highlighted else point_size
        alpha = point_alpha * 1.2 if is_highlighted else point_alpha
        zorder = 5 if is_highlighted else 3
        try:
            ax2.scatter(
                prop_embedding[idx, 0], prop_embedding[idx, 1],
                s=size if isinstance(size, (int, float)) else size[idx],
                c=color, alpha=alpha,
                edgecolors=kwargs.get('edge_color', 'white'),
                linewidths=kwargs.get('edge_width', 1.0),
                zorder=zorder, marker='^'
            )
        except Exception as e:
            print(f"[WARNING] Failed plotting ATAC sample '{sample_name}' (idx {idx}) in proportion space: {e}")

    print(f"Drawing {len(matching_ids)} connection lines in proportion plot...")
    for core_id in matching_ids:
        rna_idx, _ = rna_samples[core_id]
        atac_idx, _ = atac_samples[core_id]
        is_highlighted = core_id in highlight_mask
        lw = highlight_line_width if is_highlighted else line_width
        lc = highlight_color if is_highlighted else line_color
        la = line_alpha * 1.5 if is_highlighted else line_alpha
        zorder = 2 if is_highlighted else 1
        try:
            ax2.plot(
                [prop_embedding[rna_idx, 0], prop_embedding[atac_idx, 0]],
                [prop_embedding[rna_idx, 1], prop_embedding[atac_idx, 1]],
                color=lc, alpha=la, linewidth=lw, zorder=zorder, linestyle='-'
            )
        except Exception as e:
            print(f"[WARNING] Failed drawing proportion-space line for pair '{core_id}': {e}")

    if show_sample_ids and len(matching_ids) > 0:
        n_to_show = min(5, len(matching_ids))
        for core_id in list(matching_ids)[:n_to_show]:
            atac_idx, _ = atac_samples[core_id]
            try:
                ax2.annotate(
                    core_id, (prop_embedding[atac_idx, 0], prop_embedding[atac_idx, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )
            except Exception as e:
                print(f"[WARNING] Failed annotating '{core_id}' on proportion plot: {e}")

    ax2.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax2.set_title('Proportion Embedding Space', fontsize=14, fontweight='bold', pad=15)
    if kwargs.get('grid', True):
        ax2.grid(True, alpha=kwargs.get('grid_alpha', 0.15), linestyle='--')
    for spine in ax2.spines.values():
        spine.set_linewidth(kwargs.get('spine_width', 1.5))
    try:
        ax2.legend(handles=legend_elements, loc='best', frameon=True, fancybox=True, shadow=True)
    except Exception as e:
        print(f"[WARNING] Failed to add legend to proportion plot: {e}")

    plt.tight_layout()
    if output_dir:
        prop_path = os.path.join(output_dir, f"{filename_prefix}_proportion.png")
        try:
            fig2.savefig(prop_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Proportion plot saved to: {prop_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save proportion plot to '{prop_path}': {e}")

    print("\nVisualization complete!")
    print(f"Total samples processed: {n_samples}")
    print(f"Matching pairs connected: {len(matching_ids)}")

    return fig1, fig2



def plot_cross_modal_connections(
    adata,
    output_dir: str,
    expression_embedding_key: str = 'X_DR_expression',
    proportion_embedding_key: str = 'X_DR_proportion',
    filename_prefix: str = 'cross_modal_connections',
    dpi: int = 300,
    **kwargs
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Convenience function for plotting cross-modal connections with automatic saving.
    
    This function creates two separate plots saved as different PNG files:
    - {filename_prefix}_expression.png: Expression embedding with cross-modal connections
    - {filename_prefix}_proportion.png: Proportion embedding with cross-modal connections
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    output_dir : str
        Directory to save the plots
    expression_embedding_key : str
        Key for expression embedding coordinates
    proportion_embedding_key : str
        Key for proportion embedding coordinates
    filename_prefix : str
        Prefix for the output filenames
    dpi : int
        Resolution for saved figures
    **kwargs : additional arguments passed to visualize_cross_modal_connections
    
    Returns:
    --------
    fig1, fig2 : matplotlib figures (expression and proportion plots)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")
    
    # Set output directory and filename prefix
    kwargs['output_dir'] = output_dir
    kwargs['filename_prefix'] = filename_prefix
    kwargs['dpi'] = dpi
    
    # Create plots
    fig1, fig2 = visualize_cross_modal_connections(
        adata,
        expression_embedding_key=expression_embedding_key,
        proportion_embedding_key=proportion_embedding_key,
        **kwargs
    )
    
    return fig1, fig2


# Example usage function
def example_cross_modal_plot():
    """
    Example function demonstrating how to use the cross-modal visualization.
    
    Assumes sample names are in format: [sample_id]_RNA and [sample_id]_ATAC
    For example: "sample001_RNA" and "sample001_ATAC" will be connected
    """
    print("Running example cross-modal visualization...")
    print("Sample naming format: [sample_id]_RNA and [sample_id]_ATAC")
    
    import scanpy as sc
    
    # Load data
    print("Loading data...")
    adata = sc.read_h5ad('/dcs07/hongkai/data/harry/result/all/multiomics/pseudobulk/pseudobulk_sample.h5ad')
    
    # Basic usage - creates two separate PNG files
    fig1, fig2 = plot_cross_modal_connections(
        adata=adata,
        output_dir='/dcs07/hongkai/data/harry/result/all/multiomics/visualization',
        filename_prefix='sample_connections',
        dpi=300
    )
    
    # Advanced usage with customization
    # fig1, fig2 = visualize_cross_modal_connections(
    #     adata=adata,
    #     expression_embedding_key='X_DR_expression',
    #     proportion_embedding_key='X_DR_proportion',
    #     figsize=(14, 12),
    #     point_size=120,
    #     point_alpha=0.8,
    #     line_alpha=0.4,
    #     line_width=1.5,
    #     expression_color='#2E86AB',  # Custom blue
    #     proportion_color='#A23B72',   # Custom purple
    #     line_color='#555555',
    #     highlight_samples=['sample001', 'sample045', 'sample123'],  # Core IDs without prefix
    #     highlight_color='#FF0000',
    #     highlight_line_width=3.0,
    #     show_sample_ids=True,
    #     style='modern',
    #     output_dir='/dcs07/hongkai/data/harry/result/heart/multiomics/visualization',
    #     filename_prefix='cross_modal_highlighted',
    #     dpi=300
    # )
    
    plt.show()
    print("Example completed!")


if __name__ == "__main__":
    example_cross_modal_plot()