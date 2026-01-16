#!/usr/bin/env python3
"""
Trajectory Differential Gene Visualization Module
=================================================

Comprehensive visualization functions for GAM-based trajectory differential gene analysis,
following the visualization patterns from the Lamian paper (Hou et al., Nature Communications 2023).

Key visualization types:
1. TDE/XDE Heatmaps - Gene expression patterns along pseudotime
2. Gene Expression Curves - Sample-level and group-level fitted curves
3. Cell Density Plots - Pseudotemporal density patterns
4. Cluster Pattern Summaries - Averaged patterns by gene cluster
5. Volcano and MA Plots - Statistical summary visualizations
6. Multi-panel Gene Profiles - Detailed per-gene visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# COLOR SCHEMES (following Lamian paper style)
# =============================================================================

def get_lamian_colormap(name: str = 'expression') -> LinearSegmentedColormap:
    """
    Get Lamian-style colormaps.
    
    Parameters
    ----------
    name : str
        'expression' - blue-white-red for expression data
        'pseudotime' - viridis-like for pseudotime
        'difference' - diverging for trend differences
        'cluster' - categorical for clusters
    """
    if name == 'expression':
        # Blue-white-red diverging colormap
        colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', 
                  '#F7F7F7', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
        return LinearSegmentedColormap.from_list('lamian_expr', colors)
    elif name == 'difference':
        # Purple-white-orange for differences
        colors = ['#5E3C99', '#B2ABD2', '#F7F7F7', '#FDB863', '#E66101']
        return LinearSegmentedColormap.from_list('lamian_diff', colors)
    elif name == 'pseudotime':
        # Custom gradient similar to Lamian
        colors = ['#440154', '#3B528B', '#21918C', '#5DC863', '#FDE725']
        return LinearSegmentedColormap.from_list('lamian_ptime', colors)
    elif name == 'mean_shift':
        # Green-white-purple for mean shifts
        colors = ['#1B7837', '#7FBF7B', '#F7F7F7', '#AF8DC3', '#762A83']
        return LinearSegmentedColormap.from_list('lamian_mean', colors)
    else:
        return plt.cm.viridis


def get_cluster_colors(n_clusters: int) -> List[str]:
    """Get distinct colors for clusters."""
    if n_clusters <= 10:
        return plt.cm.tab10.colors[:n_clusters]
    elif n_clusters <= 20:
        return plt.cm.tab20.colors[:n_clusters]
    else:
        return [plt.cm.viridis(i / n_clusters) for i in range(n_clusters)]


# =============================================================================
# HEATMAP VISUALIZATIONS (Lamian Fig 3a-d style)
# =============================================================================
def plot_tde_heatmap(
    Y: pd.DataFrame,
    results: pd.DataFrame,
    pseudotime: np.ndarray,
    gam_models: Optional[Dict] = None,
    n_clusters: int = 3,  # Default to 3 as discussed
    cluster_method: str = 'kmeans',
    figsize: Tuple[int, int] = (14, 10),
    show_gene_labels: bool = False,
    top_n_genes: Optional[int] = None,
    title: str = "TDE Genes Along Pseudotime",
    output_path: Optional[str] = None,
    verbose: bool = False
) -> Tuple[plt.Figure, Dict]:
    """
    Plot TDE Heatmap with WATERFALL ordering (Cluster -> Peak Time).
    """
    if verbose:
        print("[VIZ] Creating TDE heatmap (using GAM fitted values + Waterfall sort)...")
    
    # --- 1. Select Genes ---
    if 'fdr' in results.columns:
        sig_genes = results[results['fdr'] < 0.05]['gene'].tolist()
    else:
        sig_genes = results['gene'].tolist()

    if top_n_genes and len(sig_genes) > top_n_genes:
        if 'effect_size' in results.columns:
            sig_genes = results[results['fdr'] < 0.05].nlargest(top_n_genes, 'effect_size')['gene'].tolist()
        else:
            sig_genes = results[results['fdr'] < 0.05].nsmallest(top_n_genes, 'fdr')['gene'].tolist()
    
    available_genes = [g for g in sig_genes if g in Y.columns]
    if gam_models:
        available_genes = [g for g in available_genes if g in gam_models]

    if len(available_genes) == 0:
        if verbose: print("[VIZ] No significant genes found for heatmap")
        return None, {}
    
    if verbose: print(f"[VIZ] Plotting {len(available_genes)} genes")
    
    # --- 2. Prepare Data (Fitted) ---
    sort_idx = np.argsort(pseudotime)
    ptime_sorted = pseudotime[sort_idx]
    X_pred = ptime_sorted.reshape(-1, 1)
    
    fitted_data = []
    for gene in available_genes:
        try:
            model = gam_models[gene]
            y_pred = model.predict(X_pred)
            fitted_data.append(y_pred)
        except Exception:
            raw_vals = Y[gene].iloc[sort_idx].values
            fitted_data.append(raw_vals)

    expr_matrix = np.array(fitted_data) # (n_genes, n_samples)
    
    # Z-score normalize
    means = expr_matrix.mean(axis=1, keepdims=True)
    stds = expr_matrix.std(axis=1, keepdims=True) + 1e-10
    expr_zscore = (expr_matrix - means) / stds
    
    # --- 3. Cluster Genes ---
    if cluster_method == 'kmeans':
        n_clusters = min(n_clusters, len(available_genes))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(expr_zscore)
    elif cluster_method == 'hierarchical':
        if len(available_genes) > 1:
            linkage_matrix = linkage(expr_zscore, method='ward')
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        else:
            cluster_labels = np.array([0])
    else:
        cluster_labels = np.zeros(len(available_genes), dtype=int)
    
    # --- 4. WATERFALL SORTING (Cluster -> Peak Time) ---
    # Calculate the index of peak expression for every gene
    peak_indices = np.argmax(expr_zscore, axis=1)
    
    # lexsort sorts by the last key passed first.
    # We want: Primary = Cluster, Secondary = Peak Time
    sort_keys = np.lexsort((peak_indices, cluster_labels))
    
    genes_sorted = [available_genes[i] for i in sort_keys]
    cluster_labels_sorted = cluster_labels[sort_keys]
    expr_for_plot = expr_zscore[sort_keys]
    
    # --- 5. Plotting ---
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.03, 1, 0.05], wspace=0.02)
    
    # Cluster Bar
    ax_cluster = fig.add_subplot(gs[0])
    cluster_colors = get_cluster_colors(n_clusters)
    cluster_cmap = LinearSegmentedColormap.from_list('clusters', cluster_colors)
    
    ax_cluster.imshow(cluster_labels_sorted.reshape(-1, 1), aspect='auto', cmap=cluster_cmap, interpolation='nearest')
    ax_cluster.set_xticks([])
    ax_cluster.set_yticks([])
    ax_cluster.set_ylabel('Genes (Clustered & Peak-Sorted)', fontsize=10)
    
    # Heatmap
    ax_heat = fig.add_subplot(gs[1])
    cmap = get_lamian_colormap('expression')
    
    im = ax_heat.imshow(expr_for_plot, aspect='auto', cmap=cmap, vmin=-2.5, vmax=2.5, interpolation='nearest')
    ax_heat.set_xlabel('Pseudotime', fontsize=11)
    ax_heat.set_yticks([])
    
    # Pseudotime Bar
    divider_top = ax_heat.inset_axes([0, 1.01, 1, 0.03])
    ptime_norm = (ptime_sorted - ptime_sorted.min()) / (ptime_sorted.max() - ptime_sorted.min() + 1e-10)
    divider_top.imshow(ptime_norm.reshape(1, -1), aspect='auto', cmap=get_lamian_colormap('pseudotime'))
    divider_top.set_xticks([])
    divider_top.set_yticks([])
    
    # Colorbar
    ax_cbar = fig.add_subplot(gs[2])
    plt.colorbar(im, cax=ax_cbar, label='Z-score (Fitted)')
    
    # Add Cluster Labels
    unique_clusters = np.unique(cluster_labels_sorted)
    for c in unique_clusters:
        mask = cluster_labels_sorted == c
        indices = np.where(mask)[0]
        if len(indices) > 0:
            mid_idx = indices[len(indices) // 2]
            ax_cluster.text(-0.5, mid_idx, str(c + 1), ha='right', va='center', fontsize=9, fontweight='bold', color=cluster_colors[c])

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        if verbose: print(f"[VIZ] Saved TDE heatmap to {output_path}")
    
    info = {
        'genes': genes_sorted,
        'cluster_labels': cluster_labels_sorted,
        'n_clusters': n_clusters,
        'n_genes': len(genes_sorted)
    }
    return fig, info

def plot_tde_heatmap(
    Y: pd.DataFrame,
    results: pd.DataFrame,
    pseudotime: np.ndarray,
    gam_models: Optional[Dict] = None,
    n_clusters: int = 3,
    cluster_method: str = 'kmeans',
    figsize: Tuple[int, int] = (14, 10),
    show_gene_labels: bool = False,
    top_n_genes: Optional[int] = None,
    title: str = "TDE Genes Along Pseudotime",
    output_path: Optional[str] = None,
    verbose: bool = False
) -> Tuple[plt.Figure, Dict]:
    """
    Plot TDE Heatmap with:
    1. Fitted Values (Smooth trends)
    2. Temporal Cluster Ordering (Cluster 1=Early -> Cluster N=Late)
    3. Waterfall Gene Sorting (Sorted by peak time within cluster)
    """
    if verbose:
        print("[VIZ] Creating TDE heatmap (Fitted + Temporal Cluster Sort + Waterfall)...")
    
    # --- 1. Select Genes ---
    if 'fdr' in results.columns:
        sig_genes = results[results['fdr'] < 0.05]['gene'].tolist()
    else:
        sig_genes = results['gene'].tolist()

    if top_n_genes and len(sig_genes) > top_n_genes:
        # Prioritize by Effect Size
        if 'effect_size' in results.columns:
            sig_genes = results[results['fdr'] < 0.05].nlargest(top_n_genes, 'effect_size')['gene'].tolist()
        else:
            sig_genes = results[results['fdr'] < 0.05].nsmallest(top_n_genes, 'fdr')['gene'].tolist()
    
    available_genes = [g for g in sig_genes if g in Y.columns]
    if gam_models:
        available_genes = [g for g in available_genes if g in gam_models]

    if len(available_genes) == 0:
        if verbose: print("[VIZ] No significant genes found for heatmap")
        return None, {}
    
    # --- 2. Prepare Data (Fitted) ---
    sort_idx = np.argsort(pseudotime)
    ptime_sorted = pseudotime[sort_idx]
    X_pred = ptime_sorted.reshape(-1, 1)
    
    fitted_data = []
    for gene in available_genes:
        try:
            model = gam_models[gene]
            y_pred = model.predict(X_pred)
            fitted_data.append(y_pred)
        except Exception:
            raw_vals = Y[gene].iloc[sort_idx].values
            fitted_data.append(raw_vals)

    expr_matrix = np.array(fitted_data) # (n_genes, n_samples)
    
    # Z-score normalize
    means = expr_matrix.mean(axis=1, keepdims=True)
    stds = expr_matrix.std(axis=1, keepdims=True) + 1e-10
    expr_zscore = (expr_matrix - means) / stds
    
    # --- 3. Initial Clustering ---
    if cluster_method == 'kmeans':
        n_clusters = min(n_clusters, len(available_genes))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        initial_labels = kmeans.fit_predict(expr_zscore)
    elif cluster_method == 'hierarchical':
        if len(available_genes) > 1:
            linkage_matrix = linkage(expr_zscore, method='ward')
            initial_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        else:
            initial_labels = np.array([0])
    else:
        initial_labels = np.zeros(len(available_genes), dtype=int)
    
    # --- 4. REORDER CLUSTERS BY TIME (The Fix) ---
    # We want Cluster 0 to be the "Earliest Peaking" and Cluster N to be "Latest Peaking"
    cluster_peak_times = []
    unique_labels = np.unique(initial_labels)
    
    for label in unique_labels:
        # Get all genes in this cluster
        mask = initial_labels == label
        if mask.sum() == 0:
            cluster_peak_times.append(99999) # Should not happen
            continue
            
        # Calculate the average expression profile for this cluster
        avg_profile = expr_zscore[mask].mean(axis=0)
        # Find index of max expression
        peak_idx = np.argmax(avg_profile)
        cluster_peak_times.append(peak_idx)
        
    # Determine new mapping: sorted by peak time
    # e.g. if peaks are at [100, 10, 50], argsort gives [1, 2, 0]
    # So old label 1 becomes new label 0, old 2 becomes 1, old 0 becomes 2.
    sorted_order = np.argsort(cluster_peak_times)
    map_old_to_new = {old: new for new, old in enumerate(sorted_order)}
    
    # Apply mapping
    cluster_labels = np.array([map_old_to_new[l] for l in initial_labels])
    
    # --- 5. WATERFALL SORT (Within Cluster) ---
    # Determine peak for every single gene
    gene_peak_indices = np.argmax(expr_zscore, axis=1)
    
    # Sort primarily by Cluster Label (0->1->2), secondarily by Peak Time
    sort_keys = np.lexsort((gene_peak_indices, cluster_labels))
    
    genes_sorted = [available_genes[i] for i in sort_keys]
    cluster_labels_sorted = cluster_labels[sort_keys]
    expr_for_plot = expr_zscore[sort_keys]
    
    # --- 6. Plotting ---
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.03, 1, 0.05], wspace=0.02)
    
    # Cluster Bar
    ax_cluster = fig.add_subplot(gs[0])
    cluster_colors = get_cluster_colors(n_clusters)
    cluster_cmap = LinearSegmentedColormap.from_list('clusters', cluster_colors)
    
    ax_cluster.imshow(cluster_labels_sorted.reshape(-1, 1), aspect='auto', cmap=cluster_cmap, interpolation='nearest')
    ax_cluster.set_xticks([])
    ax_cluster.set_yticks([])
    ax_cluster.set_ylabel('Genes (Sorted by Pseudotime)', fontsize=10)
    
    # Heatmap
    ax_heat = fig.add_subplot(gs[1])
    cmap = get_lamian_colormap('expression')
    
    im = ax_heat.imshow(expr_for_plot, aspect='auto', cmap=cmap, vmin=-2.5, vmax=2.5, interpolation='nearest')
    ax_heat.set_xlabel('Pseudotime (Early $\\to$ Late)', fontsize=11)
    ax_heat.set_yticks([])
    
    # Pseudotime Bar
    divider_top = ax_heat.inset_axes([0, 1.01, 1, 0.03])
    ptime_norm = (ptime_sorted - ptime_sorted.min()) / (ptime_sorted.max() - ptime_sorted.min() + 1e-10)
    divider_top.imshow(ptime_norm.reshape(1, -1), aspect='auto', cmap=get_lamian_colormap('pseudotime'))
    divider_top.set_xticks([])
    divider_top.set_yticks([])
    
    # Colorbar
    ax_cbar = fig.add_subplot(gs[2])
    plt.colorbar(im, cax=ax_cbar, label='Z-score (Fitted)')
    
    # Add Cluster Labels
    # Since we reordered, Cluster 0 is now definitely the "Top" (Earliest) block
    for c in range(n_clusters):
        mask = cluster_labels_sorted == c
        indices = np.where(mask)[0]
        if len(indices) > 0:
            mid_idx = indices[len(indices) // 2]
            ax_cluster.text(-0.5, mid_idx, str(c + 1), ha='right', va='center', fontsize=9, fontweight='bold', color=cluster_colors[c])

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        if verbose: print(f"[VIZ] Saved TDE heatmap to {output_path}")
    
    info = {
        'genes': genes_sorted,
        'cluster_labels': cluster_labels_sorted,
        'n_clusters': n_clusters,
        'n_genes': len(genes_sorted)
    }
    return fig, info


"""
Improved Gene Expression Curve Visualization with GAM Smoothing
================================================================

This module contains the improved plot_gene_expression_curves function
that properly uses fitted GAM models to generate smooth expression curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import uniform_filter1d
import warnings

warnings.filterwarnings('ignore')

"""
Fixed plot_gene_expression_curves function

The key fix: The GAM models were fitted with only the columns present in X during fitting
(typically just 'pseudotime'), but the visualization was trying to predict with X_viz 
which has additional columns like 'sample' and 'group_col'. 

Solution: Only use the pseudotime column for prediction, matching what was used during fitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import uniform_filter1d
import warnings

warnings.filterwarnings('ignore')


def _plot_fallback_smooth(ax, x, y, color, linewidth=2, alpha=0.9):
    """Plot a simple moving average smooth as fallback when GAM is not available."""
    if len(x) < 5:
        return
    
    sort_idx = np.argsort(x)
    x_sorted = np.asarray(x)[sort_idx]
    y_sorted = np.asarray(y)[sort_idx]
    
    # Use a rolling window smooth
    window = max(3, len(y_sorted) // 10)
    y_smooth = uniform_filter1d(y_sorted.astype(float), size=window, mode='nearest')
    
    ax.plot(x_sorted, y_smooth, c=color, linewidth=linewidth, alpha=alpha, zorder=5)


def plot_gene_expression_curves(
    genes: List[str],
    Y: pd.DataFrame,
    X: pd.DataFrame,
    gam_models: Optional[Dict] = None,
    results: Optional[pd.DataFrame] = None,
    group_col: Optional[str] = None,
    n_cols: int = 4,
    figsize_per_gene: Tuple[float, float] = (3.5, 3),
    show_samples: bool = True,
    show_gam_fit: bool = True,
    n_curve_points: int = 100,
    title: str = "Gene Expression Along Pseudotime",
    output_path: Optional[str] = None,
    verbose: bool = False
) -> plt.Figure:
    """
    Plot gene expression curves along pseudotime using GAM fitted models.
    
    FIXED: Now correctly determines the number of features the GAM was trained with
    and only uses pseudotime for prediction (since GAMs were fit with pseudotime only).
    
    Parameters
    ----------
    genes : list
        List of gene names to plot
    Y : pd.DataFrame
        Expression matrix (samples x genes)
    X : pd.DataFrame
        Design matrix with 'pseudotime' column (may have additional columns for viz)
    gam_models : dict, optional
        Dictionary of fitted GAM models per gene {gene_name: LinearGAM}
    results : pd.DataFrame, optional
        Results with statistics to annotate
    group_col : str, optional
        Column for grouping samples
    n_cols : int
        Number of columns in subplot grid
    figsize_per_gene : tuple
        Size per subplot (width, height)
    show_samples : bool
        Whether to show individual sample points
    show_gam_fit : bool
        Whether to show GAM fitted curves
    n_curve_points : int
        Number of points for smooth curve rendering
    title : str
        Overall figure title
    output_path : str, optional
        Path to save figure
    verbose : bool
        Print progress messages
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    available_genes = [g for g in genes if g in Y.columns]
    if len(available_genes) == 0:
        if verbose:
            print("[VIZ] No genes found in expression matrix")
        return None
    
    n_genes = len(available_genes)
    n_rows = int(np.ceil(n_genes / n_cols))
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_gene[0] * n_cols, figsize_per_gene[1] * n_rows),
        squeeze=False
    )
    
    if "pseudotime" not in X.columns:
        raise ValueError("'pseudotime' column not found in X for plotting.")
    
    ptime_full = X["pseudotime"].values
    
    # Determine how many features the GAM was trained with
    # by checking the first available GAM model
    gam_n_features = None
    if gam_models:
        for gene in available_genes:
            if gene in gam_models:
                try:
                    # Get the number of features from the GAM model
                    gam = gam_models[gene]
                    # Try to infer from model structure
                    if hasattr(gam, 'terms'):
                        # Count unique feature indices in terms
                        feature_indices = set()
                        for term in gam.terms:
                            if hasattr(term, 'feature'):
                                feature_indices.add(term.feature)
                        gam_n_features = len(feature_indices) if feature_indices else 1
                    else:
                        gam_n_features = 1  # Default assumption
                    break
                except:
                    gam_n_features = 1
    
    if verbose and gam_models:
        print(f"[VIZ] GAM models provided: {len(gam_models)} models")
        print(f"[VIZ] Inferred GAM feature count: {gam_n_features}")
    
    # Group color setup
    if group_col and group_col in X.columns:
        groups = sorted(X[group_col].unique())
        if len(groups) >= 2:
            group_colors = {groups[0]: "#E66101", groups[1]: "#5E3C99"}
            if len(groups) > 2:
                extra_colors = plt.cm.tab10.colors
                for i, g in enumerate(groups[2:]):
                    group_colors[g] = extra_colors[i % len(extra_colors)]
        else:
            group_colors = {groups[0]: "#E66101"}
    else:
        groups = None
        group_colors = {}
    
    for idx, gene in enumerate(available_genes):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        
        expr_full = Y[gene].values
        
        # Mask for samples where gene is present (matching GAM fitting logic)
        nonzero_mask = (expr_full != 0.0) & np.isfinite(expr_full) & np.isfinite(ptime_full)
        n_nonzero = nonzero_mask.sum()
        
        if n_nonzero < 3:
            ax.text(
                0.5, 0.5,
                "Insufficient data\nfor this gene",
                ha="center", va="center", fontsize=8,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            ax.set_title(gene, fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        ptime = ptime_full[nonzero_mask]
        expr = expr_full[nonzero_mask]
        
        # Check if we have a GAM model for this gene
        has_gam = gam_models is not None and gene in gam_models
        
        if groups and group_col:
            # Plot by group
            group_vals_full = X[group_col].values
            group_vals = group_vals_full[nonzero_mask]
            
            for g in groups:
                mask_g = group_vals == g
                if mask_g.sum() == 0:
                    continue
                
                xg = ptime[mask_g]
                yg = expr[mask_g]
                color = group_colors.get(g, "gray")
                
                # Show sample points
                if show_samples:
                    ax.scatter(
                        xg, yg,
                        c=color, alpha=0.4, s=15, 
                        edgecolors='none',
                        label=str(g)
                    )
                
                # Show GAM fit if available
                if show_gam_fit and has_gam and mask_g.sum() > 3:
                    try:
                        gam = gam_models[gene]
                        
                        # Create prediction points along pseudotime range for this group
                        ptime_min, ptime_max = xg.min(), xg.max()
                        ptime_pred = np.linspace(ptime_min, ptime_max, n_curve_points)
                        
                        # KEY FIX: Only use pseudotime for prediction
                        # The GAM was fitted with pseudotime only (1 feature)
                        X_pred_array = ptime_pred.reshape(-1, 1)
                        
                        # Generate predictions
                        y_pred = gam.predict(X_pred_array)
                        
                        # Plot smooth curve
                        ax.plot(
                            ptime_pred, y_pred,
                            c=color, linewidth=2.5, alpha=0.95,
                            zorder=10
                        )
                            
                    except Exception as e:
                        if verbose and idx < 5:
                            print(f"[VIZ] Could not plot GAM fit for {gene}, group {g}: {e}")
                        # Fallback to simple smooth if GAM fails
                        _plot_fallback_smooth(ax, xg, yg, color)
                
                elif show_gam_fit and not has_gam and mask_g.sum() > 10:
                    _plot_fallback_smooth(ax, xg, yg, color)
            
            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", fontsize=7, framealpha=0.8)
        
        else:
            # Single-group case (no group column)
            if show_samples:
                ax.scatter(
                    ptime, expr,
                    c="steelblue", alpha=0.4, s=15,
                    edgecolors='none'
                )
            
            # Show GAM fit if available
            if show_gam_fit and has_gam and len(expr) > 3:
                try:
                    gam = gam_models[gene]
                    
                    # Create prediction points
                    ptime_min, ptime_max = ptime.min(), ptime.max()
                    ptime_pred = np.linspace(ptime_min, ptime_max, n_curve_points)
                    
                    # KEY FIX: Only use pseudotime for prediction
                    X_pred_array = ptime_pred.reshape(-1, 1)
                    
                    # Generate predictions
                    y_pred = gam.predict(X_pred_array)
                    
                    # Plot smooth curve
                    ax.plot(
                        ptime_pred, y_pred,
                        c="darkblue", linewidth=2.5, alpha=0.95,
                        zorder=10
                    )
                        
                except Exception as e:
                    if verbose and idx < 5:
                        print(f"[VIZ] Could not plot GAM fit for {gene}: {e}")
                    _plot_fallback_smooth(ax, ptime, expr, "darkblue")
            
            elif show_gam_fit and not has_gam and len(expr) > 10:
                _plot_fallback_smooth(ax, ptime, expr, "darkblue")
        
        # Add statistics annotation
        if results is not None and "gene" in results.columns and gene in results["gene"].values:
            gene_stats = results.loc[results["gene"] == gene].iloc[0]
            fdr = gene_stats.get("fdr", np.nan)
            effect_size = gene_stats.get("effect_size", np.nan)
            
            if pd.notna(fdr):
                if pd.notna(effect_size):
                    stat_text = f"FDR={fdr:.2e}\nES={effect_size:.2f}"
                else:
                    stat_text = f"FDR={fdr:.2e}"
                
                ax.text(
                    0.95, 0.95,
                    stat_text,
                    transform=ax.transAxes,
                    fontsize=7,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor='gray', linewidth=0.5),
                )
        
        ax.set_xlabel("Pseudotime", fontsize=9)
        ax.set_ylabel("Expression", fontsize=9)
        ax.set_title(gene, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide unused subplots
    for idx in range(n_genes, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        if verbose:
            print(f"[VIZ] Saved gene expression curves to {output_path}")
    
    return fig


def plot_sample_level_curves(
    gene: str,
    Y: pd.DataFrame,
    X: pd.DataFrame,
    gam_model=None,
    sample_col: str = 'sample',
    group_col: Optional[str] = None,
    n_bins: int = 30,
    n_curve_points: int = 100,
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = False
) -> plt.Figure:
    """
    Plot sample-level expression curves with GAM fits.
    
    FIXED: Only uses pseudotime for GAM prediction to match fitting.
    """
    if gene not in Y.columns:
        if verbose:
            print(f"[VIZ] Gene '{gene}' not found in expression matrix")
        return None
    
    if title is None:
        title = f"Sample-Level Expression: {gene}"
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ptime = X['pseudotime'].values
    expr = Y[gene].values
    samples = X[sample_col].unique() if sample_col in X.columns else ['all']
    
    if group_col and group_col in X.columns:
        groups = sorted(X[group_col].unique())
        group_colors = {groups[0]: '#E66101', groups[1]: '#5E3C99'} if len(groups) >= 2 else {}
        if len(groups) > 2:
            extra_colors = plt.cm.tab10.colors
            for i, g in enumerate(groups[2:]):
                group_colors[g] = extra_colors[i % len(extra_colors)]
    else:
        groups = None
        group_colors = {}
    
    # Left panel: Sample-level patterns (binned/smoothed)
    ax1 = axes[0]
    for sample in samples:
        if sample_col in X.columns:
            mask = X[sample_col] == sample
        else:
            mask = np.ones(len(X), dtype=bool)
        
        sample_ptime = ptime[mask]
        sample_expr = expr[mask]
        
        if len(sample_ptime) < 2:
            continue
        
        # Sort and smooth
        sort_idx = np.argsort(sample_ptime)
        x_sorted = sample_ptime[sort_idx]
        y_sorted = sample_expr[sort_idx]
        
        # Bin and smooth
        if len(x_sorted) > 3:
            bins = np.linspace(x_sorted.min(), x_sorted.max(), min(n_bins + 1, len(x_sorted)))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            binned = np.zeros(len(bin_centers))
            
            for i in range(len(bin_centers)):
                bin_mask = (x_sorted >= bins[i]) & (x_sorted < bins[i + 1])
                if bin_mask.sum() > 0:
                    binned[i] = y_sorted[bin_mask].mean()
                else:
                    binned[i] = np.nan
            
            binned = pd.Series(binned).interpolate(limit_direction='both').values
            plot_x, plot_y = bin_centers, binned
        else:
            plot_x, plot_y = x_sorted, y_sorted
        
        # Get color
        if groups and group_col in X.columns:
            sample_group = X.loc[X[sample_col] == sample, group_col].iloc[0] if sample_col in X.columns else None
            color = group_colors.get(sample_group, 'gray')
        else:
            color = 'steelblue'
        
        ax1.plot(plot_x, plot_y, color=color, alpha=0.5, linewidth=1.5)
    
    ax1.set_xlabel('Pseudotime', fontsize=11)
    ax1.set_ylabel('Expression', fontsize=11)
    ax1.set_title('Sample-Level Patterns', fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    if groups:
        for g in groups:
            ax1.plot([], [], color=group_colors.get(g, 'gray'), linewidth=2, label=str(g))
        ax1.legend(loc='best', fontsize=9)
    
    # Right panel: Group-level GAM fits
    ax2 = axes[1]
    
    # Mask for non-zero expression (matching GAM fitting)
    nonzero_mask = (expr != 0.0) & np.isfinite(expr) & np.isfinite(ptime)
    
    if groups and gam_model is not None:
        # Use GAM model for smooth predictions by group
        for g in groups:
            g_mask_full = X[group_col] == g
            g_mask = g_mask_full & nonzero_mask
            
            if g_mask.sum() == 0:
                continue
            
            g_ptime = ptime[g_mask]
            g_expr = expr[g_mask]
            
            # Show scatter points
            ax2.scatter(g_ptime, g_expr, c=group_colors.get(g, 'gray'), 
                       alpha=0.3, s=15, edgecolors='none')
            
            try:
                # Create prediction grid
                ptime_min, ptime_max = g_ptime.min(), g_ptime.max()
                ptime_pred = np.linspace(ptime_min, ptime_max, n_curve_points)
                
                # KEY FIX: Only use pseudotime for prediction (1D array reshaped to 2D)
                X_pred_array = ptime_pred.reshape(-1, 1)
                
                # Generate predictions
                y_pred = gam_model.predict(X_pred_array)
                
                # Plot GAM curve
                ax2.plot(ptime_pred, y_pred, color=group_colors.get(g, 'gray'), 
                        linewidth=3, label=str(g), alpha=0.95, zorder=10)
                
                # Add confidence band (approximate using residual std)
                # Predict on the actual data points for this group
                X_actual = g_ptime.reshape(-1, 1)
                y_fitted = gam_model.predict(X_actual)
                residuals = g_expr - y_fitted
                std_approx = np.std(residuals)
                
                ax2.fill_between(ptime_pred, y_pred - std_approx, y_pred + std_approx,
                               color=group_colors.get(g, 'gray'), alpha=0.15)
                
            except Exception as e:
                if verbose:
                    print(f"[VIZ] Could not plot GAM fit for group {g}: {e}")
                _plot_fallback_smooth(ax2, g_ptime, g_expr, group_colors.get(g, 'gray'), linewidth=3)
        
        ax2.legend(loc='best', fontsize=10)
        
    elif gam_model is not None:
        # Single group with GAM
        ptime_nz = ptime[nonzero_mask]
        expr_nz = expr[nonzero_mask]
        
        ax2.scatter(ptime_nz, expr_nz, c='steelblue', alpha=0.3, s=15, edgecolors='none')
        
        try:
            ptime_pred = np.linspace(ptime_nz.min(), ptime_nz.max(), n_curve_points)
            
            # KEY FIX: Only use pseudotime for prediction
            X_pred_array = ptime_pred.reshape(-1, 1)
            
            y_pred = gam_model.predict(X_pred_array)
            ax2.plot(ptime_pred, y_pred, c='darkblue', linewidth=3, alpha=0.95)
            
            # Confidence band
            X_actual = ptime_nz.reshape(-1, 1)
            y_fitted = gam_model.predict(X_actual)
            residuals = expr_nz - y_fitted
            std_approx = np.std(residuals)
            ax2.fill_between(ptime_pred, y_pred - std_approx, y_pred + std_approx,
                           color='darkblue', alpha=0.15)
            
        except Exception as e:
            if verbose:
                print(f"[VIZ] Could not plot GAM fit: {e}")
            _plot_fallback_smooth(ax2, ptime_nz, expr_nz, 'darkblue', linewidth=3)
    
    else:
        # No GAM model - use simple binning
        if groups:
            for g in groups:
                g_mask = (X[group_col] == g) & nonzero_mask
                if g_mask.sum() == 0:
                    continue
                
                g_ptime = ptime[g_mask]
                g_expr = expr[g_mask]
                
                ax2.scatter(g_ptime, g_expr, c=group_colors.get(g, 'gray'), 
                           alpha=0.3, s=15, edgecolors='none')
                _plot_fallback_smooth(ax2, g_ptime, g_expr, group_colors.get(g, 'gray'), linewidth=3)
            
            for g in groups:
                ax2.plot([], [], color=group_colors.get(g, 'gray'), linewidth=3, label=str(g))
            ax2.legend(loc='best', fontsize=10)
        else:
            ptime_nz = ptime[nonzero_mask]
            expr_nz = expr[nonzero_mask]
            ax2.scatter(ptime_nz, expr_nz, c='steelblue', alpha=0.3, s=15, edgecolors='none')
            _plot_fallback_smooth(ax2, ptime_nz, expr_nz, 'darkblue', linewidth=3)
    
    ax2.set_xlabel('Pseudotime', fontsize=11)
    ax2.set_ylabel('Expression', fontsize=11)
    ax2.set_title('Group-Level GAM Fitted' if gam_model else 'Group-Level Smoothed', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        if verbose:
            print(f"[VIZ] Saved sample-level curves to {output_path}")
    
    return fig


# =============================================================================
# CELL DENSITY PLOTS (Lamian Fig 3f-g style)
# =============================================================================

def plot_cell_density(
    X: pd.DataFrame,
    sample_col: str = 'sample',
    group_col: Optional[str] = None,
    n_bins: int = 50,
    figsize: Tuple[int, int] = (10, 5),
    title: str = "Cell Density Along Pseudotime",
    output_path: Optional[str] = None,
    verbose: bool = False
) -> plt.Figure:
    """
    Plot cell density along pseudotime (Lamian Fig 3f-g style).
    
    Shows one density curve per sample, optionally colored by group.
    
    Parameters
    ----------
    X : pd.DataFrame
        Design matrix with 'pseudotime' column
    sample_col : str
        Column identifying samples
    group_col : str, optional
        Column for grouping samples by condition
    n_bins : int
        Number of bins for density estimation
    figsize : tuple
        Figure size
    title : str
        Plot title
    output_path : str, optional
        Path to save figure
    verbose : bool
        Print progress messages
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    if verbose:
        print("[VIZ] Creating cell density plot...")
    
    ptime = X['pseudotime'].values
    bins = np.linspace(ptime.min(), ptime.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    fig, ax = plt.subplots(figsize=figsize)
    
    samples = X[sample_col].unique() if sample_col in X.columns else ['all']
    
    if group_col and group_col in X.columns:
        groups = sorted(X[group_col].unique())
        group_colors = {groups[0]: '#E66101', groups[1]: '#5E3C99'} if len(groups) >= 2 else {}
    else:
        groups = None
        group_colors = {}
    
    for sample in samples:
        if sample_col in X.columns:
            mask = X[sample_col] == sample
            sample_ptime = ptime[mask]
        else:
            sample_ptime = ptime
        
        # Calculate density
        counts, _ = np.histogram(sample_ptime, bins=bins)
        density = counts / (counts.sum() + 1e-10)
        
        # Get color
        if groups and group_col in X.columns:
            sample_group = X.loc[X[sample_col] == sample, group_col].iloc[0] if sample_col in X.columns else None
            color = group_colors.get(sample_group, 'gray')
            alpha = 0.6
        else:
            color = 'steelblue'
            alpha = 0.4
        
        ax.plot(bin_centers, density, color=color, alpha=alpha, linewidth=1.5)
    
    # Add legend for groups
    if groups:
        for g in groups:
            ax.plot([], [], color=group_colors.get(g, 'gray'), linewidth=2, label=str(g))
        ax.legend(loc='upper right', fontsize=10)
    
    ax.set_xlabel('Pseudotime', fontsize=12)
    ax.set_ylabel('Cell Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        if verbose:
            print(f"[VIZ] Saved cell density plot to {output_path}")
    
    return fig


# =============================================================================
# CLUSTER PATTERN SUMMARIES (Lamian Fig 4d, 5f style)
# =============================================================================

def plot_cluster_patterns(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    cluster_info: Dict,
    group_col: Optional[str] = None,
    n_example_genes: int = 2,
    figsize: Tuple[int, int] = (16, 12),
    title: str = "Gene Cluster Patterns",
    output_path: Optional[str] = None,
    verbose: bool = False
) -> plt.Figure:
    """
    Plot cluster pattern summaries (Lamian Fig 4d, 5f style).
    
    Shows averaged expression pattern for each cluster and example genes.
    
    Parameters
    ----------
    Y : pd.DataFrame
        Expression matrix (samples x genes)
    X : pd.DataFrame
        Design matrix with 'pseudotime' column
    cluster_info : dict
        Dictionary with 'genes' and 'cluster_labels' from heatmap functions
    group_col : str, optional
        Column for group comparison
    n_example_genes : int
        Number of example genes per cluster
    figsize : tuple
        Figure size
    title : str
        Plot title
    output_path : str, optional
        Path to save figure
    verbose : bool
        Print progress messages
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    if 'genes' not in cluster_info or 'cluster_labels' not in cluster_info:
        if verbose:
            print("[VIZ] cluster_info must contain 'genes' and 'cluster_labels'")
        return None
    
    genes = cluster_info['genes']
    cluster_labels = cluster_info['cluster_labels']
    n_clusters = len(np.unique(cluster_labels))
    
    ptime = X['pseudotime'].values
    sort_idx = np.argsort(ptime)
    ptime_sorted = ptime[sort_idx]
    
    # Bin pseudotime
    n_bins = 50
    bins = np.linspace(ptime.min(), ptime.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Setup figure
    n_cols = 1 + n_example_genes
    fig, axes = plt.subplots(n_clusters, n_cols, figsize=figsize, squeeze=False)
    
    cluster_colors = get_cluster_colors(n_clusters)
    
    if group_col and group_col in X.columns:
        groups = sorted(X[group_col].unique())
        group_colors = {groups[0]: '#E66101', groups[1]: '#5E3C99'} if len(groups) >= 2 else {}
    else:
        groups = None
        group_colors = {}
    
    for c_idx in range(n_clusters):
        cluster_mask = np.array(cluster_labels) == c_idx
        cluster_genes = [genes[i] for i in range(len(genes)) if cluster_mask[i]]
        available_cluster_genes = [g for g in cluster_genes if g in Y.columns]
        
        if len(available_cluster_genes) == 0:
            continue
        
        # Cluster average pattern
        ax_avg = axes[c_idx, 0]
        
        cluster_expr = Y[available_cluster_genes].iloc[sort_idx]
        
        if groups:
            for g in groups:
                g_mask = X[group_col].values[sort_idx] == g
                g_expr = cluster_expr.iloc[g_mask].mean(axis=1)
                
                # Bin and smooth
                binned = np.zeros(n_bins)
                g_ptime = ptime_sorted[g_mask]
                for i in range(n_bins):
                    bin_mask = (g_ptime >= bins[i]) & (g_ptime < bins[i + 1])
                    if bin_mask.sum() > 0:
                        binned[i] = g_expr.iloc[bin_mask].mean()
                    else:
                        binned[i] = np.nan
                
                # Interpolate NaNs
                binned = pd.Series(binned).interpolate(limit_direction='both').values
                
                ax_avg.plot(bin_centers, binned, color=group_colors.get(g, 'gray'), 
                           linewidth=2, label=str(g))
            ax_avg.legend(fontsize=8)
        else:
            avg_expr = cluster_expr.mean(axis=1)
            binned = np.zeros(n_bins)
            for i in range(n_bins):
                bin_mask = (ptime_sorted >= bins[i]) & (ptime_sorted < bins[i + 1])
                if bin_mask.sum() > 0:
                    binned[i] = avg_expr.iloc[bin_mask].mean()
            binned = pd.Series(binned).interpolate(limit_direction='both').values
            ax_avg.plot(bin_centers, binned, color=cluster_colors[c_idx], linewidth=2)
        
        ax_avg.set_ylabel(f'Cluster {c_idx + 1}\n({len(available_cluster_genes)} genes)', fontsize=9)
        ax_avg.set_xlabel('Pseudotime', fontsize=9)
        ax_avg.tick_params(labelsize=8)
        
        # Example genes
        example_genes = available_cluster_genes[:n_example_genes]
        for g_idx, gene in enumerate(example_genes):
            ax_gene = axes[c_idx, 1 + g_idx]
            gene_expr = Y[gene].values
            
            if groups:
                for g in groups:
                    g_mask = X[group_col].values == g
                    ax_gene.scatter(ptime[g_mask], gene_expr[g_mask], 
                                   c=group_colors.get(g, 'gray'), alpha=0.3, s=10)
            else:
                ax_gene.scatter(ptime, gene_expr, c=cluster_colors[c_idx], alpha=0.3, s=10)
            
            ax_gene.set_title(gene, fontsize=9, fontweight='bold')
            ax_gene.set_xlabel('Pseudotime', fontsize=8)
            ax_gene.tick_params(labelsize=7)
        
        # Hide empty gene subplots
        for g_idx in range(len(example_genes), n_example_genes):
            axes[c_idx, 1 + g_idx].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        if verbose:
            print(f"[VIZ] Saved cluster patterns to {output_path}")
    
    return fig


# =============================================================================
# STATISTICAL SUMMARY PLOTS
# =============================================================================

def plot_volcano(
    results: pd.DataFrame,
    effect_col: str = 'effect_size',
    pval_col: str = 'fdr',
    gene_col: str = 'gene',
    fdr_threshold: float = 0.05,
    effect_threshold: float = 1.0,
    highlight_genes: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Volcano Plot: Differential Gene Expression",
    output_path: Optional[str] = None,
    verbose: bool = False
) -> plt.Figure:
    """
    Create volcano plot of differential expression results.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results with effect size and p-value columns
    effect_col : str
        Column name for effect size
    pval_col : str
        Column name for p-value or FDR
    gene_col : str
        Column name for gene identifiers
    fdr_threshold : float
        Significance threshold
    effect_threshold : float
        Effect size threshold for highlighting
    highlight_genes : list, optional
        Specific genes to label
    figsize : tuple
        Figure size
    title : str
        Plot title
    output_path : str, optional
        Path to save figure
    verbose : bool
        Print progress messages
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    if effect_col not in results.columns:
        if verbose:
            print(f"[VIZ] Effect column '{effect_col}' not found")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate -log10(FDR)
    log_pval = -np.log10(results[pval_col].clip(lower=1e-300))
    effect = results[effect_col].values
    
    # Categorize points
    sig_mask = results[pval_col] < fdr_threshold
    effect_mask = np.abs(effect) > effect_threshold
    
    # Non-significant
    ns_mask = ~sig_mask
    ax.scatter(effect[ns_mask], log_pval[ns_mask], c='lightgray', alpha=0.5, s=20, label='NS')
    
    # Significant but low effect
    sig_low_mask = sig_mask & ~effect_mask
    ax.scatter(effect[sig_low_mask], log_pval[sig_low_mask], c='steelblue', alpha=0.6, s=30, label='Sig (low effect)')
    
    # Significant and high effect
    sig_high_mask = sig_mask & effect_mask
    ax.scatter(effect[sig_high_mask], log_pval[sig_high_mask], c='crimson', alpha=0.8, s=50, label='Sig (high effect)')
    
    # Add threshold lines
    ax.axhline(-np.log10(fdr_threshold), color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(effect_threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(-effect_threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Label highlighted genes
    if highlight_genes:
        for gene in highlight_genes:
            if gene in results[gene_col].values:
                gene_row = results[results[gene_col] == gene].iloc[0]
                x = gene_row[effect_col]
                y = -np.log10(gene_row[pval_col])
                ax.annotate(gene, (x, y), fontsize=8, ha='center', va='bottom',
                           xytext=(0, 5), textcoords='offset points')
    
    # Label top genes
    top_genes = results[sig_high_mask].nsmallest(10, pval_col)
    for _, row in top_genes.iterrows():
        x = row[effect_col]
        y = -np.log10(row[pval_col])
        ax.annotate(row[gene_col], (x, y), fontsize=7, ha='center', va='bottom',
                   xytext=(0, 3), textcoords='offset points', alpha=0.8)
    
    ax.set_xlabel('Effect Size', fontsize=12)
    ax.set_ylabel('-log10(FDR)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        if verbose:
            print(f"[VIZ] Saved volcano plot to {output_path}")
    
    return fig


def plot_results_summary(
    results: pd.DataFrame,
    fdr_threshold: float = 0.05,
    figsize: Tuple[int, int] = (14, 10),
    title: str = "Differential Gene Analysis Summary",
    output_path: Optional[str] = None,
    verbose: bool = False
) -> plt.Figure:
    """
    Create multi-panel summary of differential analysis results.
    
    Includes:
    - FDR distribution histogram
    - Effect size distribution
    - P-value vs effect size scatter
    - Top genes bar plot
    
    Parameters
    ----------
    results : pd.DataFrame
        Results dataframe
    fdr_threshold : float
        Significance threshold
    figsize : tuple
        Figure size
    title : str
        Overall title
    output_path : str, optional
        Path to save figure
    verbose : bool
        Print progress messages
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # FDR distribution
    ax1 = fig.add_subplot(gs[0, 0])
    fdr_vals = results['fdr'].clip(upper=1).values
    ax1.hist(fdr_vals, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax1.axvline(fdr_threshold, color='red', linestyle='--', linewidth=2, label=f'FDR={fdr_threshold}')
    ax1.set_xlabel('FDR', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('FDR Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    
    n_sig = (results['fdr'] < fdr_threshold).sum()
    ax1.text(0.95, 0.95, f'Significant: {n_sig}\nTotal: {len(results)}',
            transform=ax1.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Effect size distribution (if available)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'effect_size' in results.columns:
        effect_vals = results['effect_size'].dropna().values
        ax2.hist(effect_vals, bins=50, color='coral', alpha=0.7, edgecolor='white')
        ax2.set_xlabel('Effect Size', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Effect Size Distribution', fontsize=12, fontweight='bold')
    else:
        # Deviance explained
        if 'dev_exp' in results.columns:
            dev_vals = results['dev_exp'].dropna().values
            ax2.hist(dev_vals, bins=50, color='coral', alpha=0.7, edgecolor='white')
            ax2.set_xlabel('Deviance Explained', fontsize=11)
            ax2.set_ylabel('Count', fontsize=11)
            ax2.set_title('Model Fit (Deviance Explained)', fontsize=12, fontweight='bold')
    
    # P-value vs effect/deviance scatter
    ax3 = fig.add_subplot(gs[1, 0])
    log_fdr = -np.log10(results['fdr'].clip(lower=1e-300))
    
    if 'effect_size' in results.columns:
        x_vals = results['effect_size'].fillna(0).values
        x_label = 'Effect Size'
    elif 'dev_exp' in results.columns:
        x_vals = results['dev_exp'].fillna(0).values
        x_label = 'Deviance Explained'
    else:
        x_vals = np.zeros(len(results))
        x_label = 'Index'
    
    colors = ['crimson' if fdr < fdr_threshold else 'gray' for fdr in results['fdr']]
    ax3.scatter(x_vals, log_fdr, c=colors, alpha=0.5, s=20)
    ax3.axhline(-np.log10(fdr_threshold), color='red', linestyle='--', linewidth=1)
    ax3.set_xlabel(x_label, fontsize=11)
    ax3.set_ylabel('-log10(FDR)', fontsize=11)
    ax3.set_title(f'{x_label} vs Significance', fontsize=12, fontweight='bold')
    
    # Top genes bar plot
    ax4 = fig.add_subplot(gs[1, 1])
    top_genes = results.nsmallest(15, 'fdr')
    
    y_pos = np.arange(len(top_genes))
    colors = ['crimson' if row['fdr'] < fdr_threshold else 'steelblue' 
              for _, row in top_genes.iterrows()]
    
    ax4.barh(y_pos, -np.log10(top_genes['fdr'].clip(lower=1e-300)), color=colors, alpha=0.8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_genes['gene'].values, fontsize=9)
    ax4.set_xlabel('-log10(FDR)', fontsize=11)
    ax4.set_title('Top 15 Genes by Significance', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        if verbose:
            print(f"[VIZ] Saved results summary to {output_path}")
    
    return fig


def _get_top_genes_by_effect_size(
    results: pd.DataFrame,
    n_genes: int,
    fdr_threshold: float = 0.05
) -> List[str]:
    """
    Get top genes by effect size, filtered by FDR threshold.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results dataframe with 'gene', 'fdr', and 'effect_size' columns
    n_genes : int
        Number of top genes to return
    fdr_threshold : float
        FDR threshold for filtering significant genes
        
    Returns
    -------
    list
        List of top gene names sorted by effect size (descending)
    """
    # Filter to significant genes
    sig_results = results[results['fdr'] < fdr_threshold].copy()
    
    if len(sig_results) == 0:
        # Fallback to smallest FDR if no significant genes
        return results.nsmallest(min(n_genes, len(results)), 'fdr')['gene'].tolist()
    
    # Check if effect_size column exists and has valid values
    if 'effect_size' not in sig_results.columns:
        # Fallback to FDR-based selection
        return sig_results.nsmallest(min(n_genes, len(sig_results)), 'fdr')['gene'].tolist()
    
    # Filter out NaN effect sizes
    sig_with_effect = sig_results[sig_results['effect_size'].notna()].copy()
    
    if len(sig_with_effect) == 0:
        # Fallback to FDR-based selection
        return sig_results.nsmallest(min(n_genes, len(sig_results)), 'fdr')['gene'].tolist()
    
    # Sort by effect size (descending) and get top n genes
    top_genes = sig_with_effect.nlargest(min(n_genes, len(sig_with_effect)), 'effect_size')['gene'].tolist()
    
    return top_genes


def generate_all_visualizations(
    X: 'pd.DataFrame',
    Y: 'pd.DataFrame',
    results: 'pd.DataFrame',
    gam_models: 'Dict[str, LinearGAM]',
    pseudobulk_adata: 'ad.AnnData',
    output_dir: str,
    group_col: str = None,
    n_clusters: int = 5,
    top_n_genes_for_curves: int = 20,
    fdr_threshold: float = 0.05,
    verbose: bool = True
):
    """
    Generate Lamian-style visualizations with proper GAM model passing.
    
    UPDATED: Now selects top genes by effect size (among those meeting FDR threshold)
    instead of by smallest FDR.
    
    This is the improved version that correctly passes gam_models to the
    plot_gene_expression_curves function.
    """
    import os
    import importlib.util
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if verbose:
        print("[DEBUG] Step 5/5: Generating Lamian-style visualizations...")
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Prepare X with additional columns for visualization
    X_viz = X.copy()
    X_viz['sample'] = X_viz.index
    
    # Add group column if specified and available in adata
    if group_col and group_col in pseudobulk_adata.obs.columns:
        group_info = pseudobulk_adata.obs.loc[X_viz.index, group_col]
        X_viz[group_col] = group_info.values
        if verbose:
            groups = X_viz[group_col].unique()
            print(f"[DEBUG] Added group column '{group_col}' with groups: {list(groups)}")
    
    pseudotime = X['pseudotime'].values
    
    # 5.1 Results Summary
    if verbose:
        print("[VIZ] Creating results summary...")
    try:
        fig = plot_results_summary(
            results,
            fdr_threshold=fdr_threshold,
            title="GAM Differential Gene Analysis Summary",
            output_path=os.path.join(viz_dir, "01_results_summary.png"),
            verbose=False
        )
        if fig:
            plt.close(fig)
    except Exception as e:
        if verbose:
            print(f"[VIZ] Warning: Could not create results summary: {e}")
    
    # 5.2 TDE Heatmap
    if verbose:
        print("[VIZ] Creating TDE heatmap...")
    cluster_info = {}
    try:
        fig, cluster_info = plot_tde_heatmap(
            Y, results, pseudotime,
            n_clusters=n_clusters,
            gam_models=gam_models,
            title="Differential Genes Along Pseudotime",
            output_path=os.path.join(viz_dir, "02_tde_heatmap.png"),
            verbose=False
        )
        if fig:
            plt.close(fig)
    except Exception as e:
        if verbose:
            print(f"[VIZ] Warning: Could not create TDE heatmap: {e}")
    
    # 5.3 XDE Heatmap (if group column provided)
    if group_col and group_col in X_viz.columns:
        if verbose:
            print("[VIZ] Creating XDE heatmap...")
        try:
            fig, _ = plot_xde_heatmap(
                Y, X_viz, results,
                group_col=group_col,
                n_clusters=n_clusters,
                title=f"Differential Expression by {group_col}",
                output_path=os.path.join(viz_dir, "03_xde_heatmap.png"),
                verbose=False
            )
            if fig:
                plt.close(fig)
        except Exception as e:
            if verbose:
                print(f"[VIZ] Warning: Could not create XDE heatmap: {e}")
    
    # 5.4 Volcano Plot (if effect size available)
    if 'effect_size' in results.columns:
        if verbose:
            print("[VIZ] Creating volcano plot...")
        try:
            fig = plot_volcano(
                results,
                fdr_threshold=fdr_threshold,
                title="Volcano Plot: Differential Genes",
                output_path=os.path.join(viz_dir, "04_volcano_plot.png"),
                verbose=False
            )
            if fig:
                plt.close(fig)
        except Exception as e:
            if verbose:
                print(f"[VIZ] Warning: Could not create volcano plot: {e}")
    
    # 5.5 Gene Expression Curves - UPDATED: Use effect size for top gene selection
    if verbose:
        print("[VIZ] Creating gene expression curves with GAM fits...")
    try:
        # CHANGED: Use effect size to select top genes instead of FDR
        top_genes_for_plot = _get_top_genes_by_effect_size(
            results, 
            n_genes=top_n_genes_for_curves, 
            fdr_threshold=fdr_threshold
        )
        
        if len(top_genes_for_plot) > 0:
            if verbose:
                print(f"[VIZ] Plotting {len(top_genes_for_plot)} genes (selected by effect size, FDR < {fdr_threshold})")
                print(f"[VIZ] GAM models available: {len(gam_models)}")
            
            # THIS IS THE KEY FIX - pass gam_models parameter
            fig = plot_gene_expression_curves(
                genes=top_genes_for_plot,
                Y=Y,
                X=X_viz,
                gam_models=gam_models,  # <-- KEY: Pass the GAM models!
                results=results,
                group_col=group_col,
                title="Top Differential Genes Along Pseudotime (by Effect Size)",
                output_path=os.path.join(viz_dir, "05_gene_curves.png"),
                verbose=verbose  # Enable verbose to debug
            )
            if fig:
                plt.close(fig)
    except Exception as e:
        if verbose:
            print(f"[VIZ] Warning: Could not create gene curves: {e}")
            import traceback
            traceback.print_exc()
    
    # 5.6 Cell/Sample Density Plot
    if verbose:
        print("[VIZ] Creating sample density plot...")
    try:
        fig = plot_cell_density(
            X_viz, sample_col='sample', group_col=group_col,
            title="Sample Density Along Pseudotime",
            output_path=os.path.join(viz_dir, "06_sample_density.png"),
            verbose=False
        )
        if fig:
            plt.close(fig)
    except Exception as e:
        if verbose:
            print(f"[VIZ] Warning: Could not create density plot: {e}")
    
    # 5.7 Cluster Patterns (if cluster info available)
    if cluster_info and len(cluster_info.get('genes', [])) > 0:
        if verbose:
            print("[VIZ] Creating cluster pattern summary...")
        try:
            fig = plot_cluster_patterns(
                Y, X_viz, cluster_info,
                group_col=group_col,
                title="Gene Cluster Expression Patterns",
                output_path=os.path.join(viz_dir, "07_cluster_patterns.png"),
                verbose=False
            )
            if fig:
                plt.close(fig)
        except Exception as e:
            if verbose:
                print(f"[VIZ] Warning: Could not create cluster patterns: {e}")
    
    # 5.8 Sample-level curves for top gene - UPDATED: Use effect size for top gene selection
    if verbose:
        print("[VIZ] Creating sample-level expression curves...")
    try:
        # CHANGED: Get top gene by effect size instead of smallest FDR
        top_gene_list = _get_top_genes_by_effect_size(results, n_genes=1, fdr_threshold=fdr_threshold)
        top_gene = top_gene_list[0] if top_gene_list else None
        
        if top_gene and top_gene in gam_models:
            if verbose:
                print(f"[VIZ] Top gene by effect size: {top_gene}")
            fig = plot_sample_level_curves(
                gene=top_gene,
                Y=Y,
                X=X_viz,
                gam_model=gam_models[top_gene],  # <-- KEY: Pass the specific GAM model!
                sample_col='sample',
                group_col=group_col,
                title=f"Sample-Level Expression: {top_gene} (Top by Effect Size)",
                output_path=os.path.join(viz_dir, "08_sample_level_curves.png"),
                verbose=verbose
            )
            if fig:
                plt.close(fig)
        elif top_gene:
            if verbose:
                print(f"[VIZ] Warning: Top gene '{top_gene}' not in gam_models, using fallback smoothing")
            fig = plot_sample_level_curves(
                gene=top_gene,
                Y=Y,
                X=X_viz,
                gam_model=None,
                sample_col='sample',
                group_col=group_col,
                title=f"Sample-Level Expression: {top_gene} (Top by Effect Size)",
                output_path=os.path.join(viz_dir, "08_sample_level_curves.png"),
                verbose=verbose
            )
            if fig:
                plt.close(fig)
    except Exception as e:
        if verbose:
            print(f"[VIZ] Warning: Could not create sample-level curves: {e}")
    
    if verbose:
        print(f"[VIZ] Visualizations saved to: {viz_dir}")