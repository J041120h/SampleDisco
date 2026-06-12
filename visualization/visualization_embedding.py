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

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def detect_data_type(values: np.ndarray) -> Tuple[str, List]:
    """
    Classify values as 'numerical', 'ordinal', or 'categorical'.

    Rules (in order):
    - If < 80 % of non-NaN values are numeric → categorical.
    - Binary 0/1 → categorical (avoids misleading gradients).
    - Small integer sets (≤ 10 unique, < 10 % unique ratio) → ordinal.
    - All other numeric → numerical.
    - Non-convertible → categorical.

    Returns
    -------
    data_type : {'numerical', 'ordinal', 'categorical'}
    unique_values : list
        Numeric values for numerical/ordinal; raw labels for categorical.
    """
    valid_values = values[pd.notna(values)]

    if len(valid_values) == 0:
        return 'categorical', []

    unique_values = np.unique(valid_values)
    n_unique = len(unique_values)

    try:
        numeric_values = pd.to_numeric(valid_values, errors='coerce')
        numeric_mask = ~pd.isna(numeric_values)
        n_numeric = np.sum(numeric_mask)

        if n_numeric / len(valid_values) > 0.8:
            numeric_valid = numeric_values[numeric_mask]

            if n_unique <= 10 and n_unique / len(valid_values) < 0.1:
                unique_numeric = np.unique(pd.to_numeric(unique_values, errors='coerce'))
                unique_numeric = unique_numeric[~pd.isna(unique_numeric)]

                if len(unique_numeric) == 2 and set(unique_numeric) == {0, 1}:
                    return 'categorical', unique_values.tolist()

                # Small integer set → ordinal; return numeric unique values for proper tick placement.
                if np.all(np.mod(numeric_valid, 1) == 0):
                    ordinal_unique = np.unique(numeric_valid.astype(float))
                    return 'ordinal', ordinal_unique.tolist()

            numeric_unique = np.unique(numeric_valid.astype(float))
            return 'numerical', numeric_unique.tolist()
    except Exception:
        pass

    return 'categorical', unique_values.tolist()



def create_gradient_colormap(name: str = 'viridis', n_colors: int = 256) -> LinearSegmentedColormap:
    """
    Return a matplotlib colormap by name, or a viridis-like custom gradient as fallback.
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

    colors = ['#440154', '#31688e', '#35b779', '#fde724']
    return LinearSegmentedColormap.from_list('custom', colors, N=n_colors)


def add_density_contours(ax, x: np.ndarray, y: np.ndarray, levels: int = 5,
                         alpha: float = 0.3, colors: str = 'gray'):
    """
    Overlay KDE-estimated density contours on `ax`. Silently warns on failure.
    """
    try:
        from scipy.stats import gaussian_kde

        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        positions = np.vstack([xx.ravel(), yy.ravel()])

        density = kde(positions).reshape(xx.shape)

        ax.contour(xx, yy, density, levels=levels, colors=colors,
                  alpha=alpha, linewidths=1)
    except Exception as e:
        warnings.warn(f"Could not add density contours: {e}")


def add_confidence_ellipses(ax, x: np.ndarray, y: np.ndarray, labels: np.ndarray,
                           confidence: float = 0.95, alpha: float = 0.2):
    """
    Draw chi-square confidence ellipses (one per unique label) on `ax`.

    Uses `scipy.stats.chi2.ppf` to scale axes by the requested confidence level.
    Requires ≥ 3 points per group.
    """
    unique_labels = np.unique(labels[pd.notna(labels)])
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if np.sum(mask) < 3:
            continue

        points = np.column_stack([x[mask], y[mask]])

        cov = np.cov(points.T)
        mean = np.mean(points, axis=0)

        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        chi2_val = stats.chi2.ppf(confidence, df=2)
        width = 2 * np.sqrt(chi2_val * eigenvalues[0])
        height = 2 * np.sqrt(chi2_val * eigenvalues[1])

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
    style: str = 'modern',
    output_path: Optional[str] = None,
    dpi: int = 300,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of samples on a 2D embedding coloured by `color_col`.

    Parameters
    ----------
    adata : AnnData
    color_col : str
    embedding_key : str
    title : str, optional
    figsize : tuple
    point_size : int or array
        Can be per-sample array for variable sizes.
    alpha : float
    colormap : str
    show_density : bool
    show_ellipses : bool
    show_legend : bool
        For categorical data.
    show_colorbar : bool
        For numerical data.
    highlight_samples : list, optional
        Sample names to outline with a large red circle.
    annotate_samples : list, optional
        Sample names to annotate with a text label.
    style : {'modern', 'classic', 'minimal'}
    output_path : str, optional
    dpi : int
    **kwargs
        Accepted: edge_color, edge_width, grid_alpha, grid, spine_width, tick_size.

    Returns
    -------
    fig, ax
    """
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
    
    style_config = style_presets.get(style, style_presets['modern'])
    for key, value in style_config.items():
        if key not in kwargs:
            kwargs[key] = value

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

    if color_col in adata.obs.columns:
        color_values = adata.obs[color_col].values
    else:
        raise KeyError(f"Column '{color_col}' not found in adata.obs")

    data_type, unique_values = detect_data_type(color_values)

    fig, ax = plt.subplots(figsize=figsize)

    if data_type == 'numerical' or data_type == 'ordinal':
        valid_mask = pd.notna(color_values)
        valid_values = pd.to_numeric(color_values[valid_mask], errors='coerce')

        cmap = create_gradient_colormap(colormap)

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

        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label(color_col, rotation=270, labelpad=20, fontsize=11)

            # For ordinal data, tick at each discrete value.
            if data_type == 'ordinal' and len(unique_values) <= 10:
                cbar.set_ticks(sorted(unique_values))
                cbar.set_ticklabels([str(int(v)) if float(v).is_integer() else f'{v:.1f}'
                                    for v in sorted(unique_values)])

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

    else:
        n_categories = len(unique_values)
        if n_categories <= 10:
            colors = sns.color_palette("Set3", n_categories)
        elif n_categories <= 20:
            colors = sns.color_palette("tab20", n_categories)
        else:
            colors = sns.color_palette("husl", n_categories)

        color_map = {val: colors[i] for i, val in enumerate(unique_values)}

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

        if show_ellipses:
            add_confidence_ellipses(ax, x_coords, y_coords, color_values)

    if show_density:
        add_density_contours(ax, x_coords, y_coords)

    if highlight_samples:
        sample_names = adata.obs.index
        for sample in highlight_samples:
            if sample in sample_names:
                idx = np.where(sample_names == sample)[0][0]
                ax.scatter(x_coords[idx], y_coords[idx],
                          s=200, facecolors='none',
                          edgecolors='red', linewidths=3,
                          zorder=10)

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

    ax.set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimension 2', fontsize=12, fontweight='bold')

    if title is None:
        title = f'{embedding_key.replace("_", " ").title()} colored by {color_col}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    if kwargs.get('grid', True):
        ax.grid(True, alpha=kwargs.get('grid_alpha', 0.15), linestyle='--')

    for spine in ax.spines.values():
        spine.set_linewidth(kwargs.get('spine_width', 1.5))

    if data_type == 'categorical' and show_legend:
        if n_categories <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     frameon=True, fancybox=True, shadow=True)
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     ncol=2, frameon=True, fontsize=8)

    plt.tight_layout()

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
    Grid of scatter plots, one panel per column in `color_cols`.

    Parameters
    ----------
    adata : AnnData
    color_cols : list of str
    embedding_key : str
    n_cols : int
        Columns in the subplot grid.
    figsize_per_panel : tuple
    shared_axes : bool
    main_title : str, optional
    output_path : str, optional
    **kwargs
        point_size, alpha, colormap, edge_color, edge_width, show_colorbar, show_legend.

    Returns
    -------
    fig, axes
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

    if embedding_key in adata.obsm:
        embedding = adata.obsm[embedding_key]
    else:
        embedding = adata.uns[embedding_key]
        if isinstance(embedding, pd.DataFrame):
            embedding = embedding.values

    x_coords = embedding[:, 0]
    y_coords = embedding[:, 1]

    for idx, color_col in enumerate(color_cols):
        ax = axes[idx]

        plt.sca(ax)

        color_values = adata.obs[color_col].values
        data_type, unique_values = detect_data_type(color_values)

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

        else:
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

        ax.set_title(f'{color_col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dimension 1' if idx >= len(color_cols) - n_cols else '', fontsize=10)
        ax.set_ylabel('Dimension 2' if idx % n_cols == 0 else '', fontsize=10)
        ax.grid(True, alpha=0.15, linestyle='--')

    for idx in range(n_panels, len(axes)):
        fig.delaxes(axes[idx])

    if main_title:
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Multi-panel figure saved to: {output_path}")

    return fig, axes


def plot_expression_embedding(
    adata,
    color_col: str,
    output_dir: str,
    embedding_key: str = 'X_DR_sample',
    filename_prefix: str = 'expression_embedding',
    file_format: str = 'png',
    dpi: int = 300,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Save an expression-embedding scatter plot coloured by `color_col`.

    Parameters
    ----------
    adata : AnnData
    color_col : str
    output_dir : str
    embedding_key : str
    filename_prefix : str
    file_format : str
    dpi : int
    **kwargs
        Passed to `visualize_single_omics_embedding`.

    Returns
    -------
    fig, ax
    """
    os.makedirs(output_dir, exist_ok=True)

    safe_color_col = color_col.replace(' ', '_').replace('/', '_')
    filename = f"{filename_prefix}_{safe_color_col}.{file_format}"
    output_path = os.path.join(output_dir, filename)

    kwargs.setdefault('title', f'Expression Embedding: {color_col}')
    kwargs.setdefault('colormap', 'viridis')
    kwargs['output_path'] = output_path
    kwargs['dpi'] = dpi

    fig, ax = visualize_single_omics_embedding(adata, color_col, embedding_key, **kwargs)

    print(f"Expression embedding plot saved to: {output_path}")

    return fig, ax


def plot_proportion_embedding(
    adata,
    color_col: str,
    output_dir: str,
    embedding_key: str = 'X_DR_sample',
    filename_prefix: str = 'proportion_embedding',
    file_format: str = 'png',
    dpi: int = 300,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Save a cell-proportion-embedding scatter plot coloured by `color_col`.

    Uses RdYlBu colormap by default (diverging, suited for proportions).

    Parameters
    ----------
    adata : AnnData
    color_col : str
    output_dir : str
    embedding_key : str
    filename_prefix : str
    file_format : str
    dpi : int
    **kwargs
        Passed to `visualize_single_omics_embedding`.

    Returns
    -------
    fig, ax
    """
    os.makedirs(output_dir, exist_ok=True)

    safe_color_col = color_col.replace(' ', '_').replace('/', '_')
    filename = f"{filename_prefix}_{safe_color_col}.{file_format}"
    output_path = os.path.join(output_dir, filename)

    kwargs.setdefault('title', f'Cell Proportion Embedding: {color_col}')
    kwargs.setdefault('colormap', 'RdYlBu')
    kwargs['output_path'] = output_path
    kwargs['dpi'] = dpi

    fig, ax = visualize_single_omics_embedding(adata, color_col, embedding_key, **kwargs)

    print(f"Proportion embedding plot saved to: {output_path}")

    return fig, ax