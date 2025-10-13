"""
Pseudobulk Preprocessing Pipeline

This module performs single-cell data preprocessing and pseudobulk analysis with PCA,
using enhanced visualization methods for comprehensive analysis.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import time
import contextlib
import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visualization.visualization_emebedding import visualize_single_omics_embedding, create_multi_panel_embedding


def compute_sample_distance_cosine(
    pca_result: dict,
    n_components: int | None = None,
    output_dir: str | None = None,
    save_plot: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute pairwise cosine distances between samples using their PCA coordinates.

    Parameters
    ----------
    pca_result : dict
        The output of perform_pca(...). Must contain key 'coords' (DataFrame, samples x PCs).
    n_components : int or None
        Number of leading PCs to use. If None, use all available PCs.
    output_dir : str or None
        If provided, saves CSV to <output_dir>/sample_distance_cosine_pca.csv
        and (optionally) a heatmap to <output_dir>/figures/sample_distance_cosine_pca.png
    save_plot : bool
        Whether to render and save a heatmap PNG.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dist_df : DataFrame
        Square DataFrame of cosine distances (samples x samples).
    """
    coords_df: pd.DataFrame = pca_result['coords']
    if n_components is None:
        X = coords_df.values
        used_cols = list(coords_df.columns)
    else:
        use_k = max(1, min(n_components, coords_df.shape[1]))
        used_cols = [f'PC{i+1}' for i in range(use_k)]
        X = coords_df[used_cols].values

    from sklearn.metrics.pairwise import cosine_distances
    D = cosine_distances(X)
    dist_df = pd.DataFrame(D, index=coords_df.index, columns=coords_df.index)

    if output_dir is not None:
        # Save CSV
        csv_path = os.path.join(output_dir, 'sample_distance_cosine_pca.csv')
        dist_df.to_csv(csv_path)
        if verbose:
            print(f"Saved cosine distance (PCA) matrix to: {csv_path}")

        # Optional heatmap
        if save_plot:
            fig_dir = os.path.join(output_dir, 'figures')
            os.makedirs(fig_dir, exist_ok=True)
            plt.figure(figsize=(14, 12))
            ax = sns.heatmap(
                dist_df,
                cmap='viridis',
                square=True,
                linewidths=0.3,
                cbar_kws={'label': 'Cosine Distance', 'shrink': 0.8}
            )
            ax.set_xlabel('Samples', fontsize=12, fontweight='bold')
            ax.set_ylabel('Samples', fontsize=12, fontweight='bold')
            title_extra = f" (using {len(used_cols)} PCs)" if n_components is not None else ""
            plt.title(f'Sample–Sample Cosine Distance on PCA{title_extra}', fontsize=14, fontweight='bold', pad=12)
            out_png = os.path.join(fig_dir, 'sample_distance_cosine_pca.png')
            plt.tight_layout()
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close()
            if verbose:
                print(f"Saved cosine distance heatmap to: {out_png}")

    return dist_df


from sklearn.cross_decomposition import CCA
from itertools import combinations

def _find_best_2pc_for_cca(pca_coords: np.ndarray, sev_levels: np.ndarray, verbose: bool=False):
    """Search all 2-PC pairs, return (pc_idx_pair, score, fitted_cca, coords_2d)."""
    ncomp = pca_coords.shape[1]
    if ncomp < 2:
        raise ValueError("Need at least 2 PCs for trajectory CCA.")
    sev2d = sev_levels.reshape(-1, 1)

    # If exactly 2 PCs, just use them
    if ncomp == 2:
        cca = CCA(n_components=1)
        cca.fit(pca_coords, sev2d)
        U, V = cca.transform(pca_coords, sev2d)
        score = float(abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1]))
        return (0, 1), score, cca, pca_coords

    best = (-1.0, None, None, None)
    for pc1, pc2 in combinations(range(ncomp), 2):
        sub = pca_coords[:, [pc1, pc2]]
        try:
            cca = CCA(n_components=1)
            cca.fit(sub, sev2d)
            U, V = cca.transform(sub, sev2d)
            score = float(abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1]))
            if score > best[0]:
                best = (score, (pc1, pc2), cca, sub)
            if verbose:
                print(f"PC{pc1+1}+PC{pc2+1} CCA score={score:.4f}")
        except Exception as e:
            if verbose:
                print(f"Skip PC{pc1+1}+PC{pc2+1}: {e}")
            continue

    if best[1] is None:
        raise RuntimeError("Failed to find a valid 2-PC CCA combination.")
    return best[1], best[0], best[2], best[3]


def _assign_pseudotime_from_cca(pca_coords_2d: np.ndarray, cca: CCA, scale_to_unit: bool=True) -> np.ndarray:
    """Project onto CCA x-weights and optionally scale to [0,1]."""
    direction = cca.x_weights_[:, 0]  # shape (2,)
    proj = pca_coords_2d @ direction   # shape (n_samples,)
    if not scale_to_unit:
        return proj
    lo, hi = float(np.min(proj)), float(np.max(proj))
    denom = max(hi - lo, 1e-16)
    return (proj - lo) / denom


def compute_pseudotime_from_pca(
    pca_result: dict,
    sample_metadata: pd.DataFrame,
    sev_col: str = "sev.level",
    output_dir: str | None = None,
    auto_select_best_2pc: bool = True,
    save_plot: bool = True,
    verbose: bool = False,
    plot_filename: str = "trajectory_pc_space.png"
) -> pd.DataFrame:
    """
    Compute pseudotime on PCA space via CCA against severity and save CSV.
    Returns a DataFrame with columns: sample, pseudotime, pc1, pc2, cca_score.
    """
    # Align samples (use the same order as PCA coords / normalized_df index)
    samples = pca_result['coords'].index.to_numpy()
    pca_coords_full = pca_result['coords'].values

    if sev_col not in sample_metadata.columns:
        raise KeyError(f"'{sev_col}' not found in sample_metadata.")

    sev_levels = pd.to_numeric(
        sample_metadata.set_index('sample').loc[samples, sev_col],
        errors='coerce'
    ).to_numpy()
    # Impute missing with mean if needed
    if np.isnan(sev_levels).any():
        if verbose:
            nmiss = int(np.isnan(sev_levels).sum())
            print(f"Warning: {nmiss} sev.level missing; imputing with mean.")
        sev_levels = np.where(np.isnan(sev_levels), np.nanmean(sev_levels), sev_levels)

    # Choose PC pair + fit CCA
    if auto_select_best_2pc and pca_coords_full.shape[1] > 2:
        (pc1, pc2), score, cca, coords_2d = _find_best_2pc_for_cca(pca_coords_full, sev_levels, verbose)
    else:
        pc1, pc2 = 0, 1
        coords_2d = pca_coords_full[:, [pc1, pc2]]
        cca = CCA(n_components=1)
        cca.fit(coords_2d, sev_levels.reshape(-1, 1))
        U, V = cca.transform(coords_2d, sev_levels.reshape(-1, 1))
        score = float(abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1]))

    # Pseudotime = projection along CCA x-weights, scaled to [0,1]
    pseudotime = _assign_pseudotime_from_cca(coords_2d, cca, scale_to_unit=True)

    # Optional diagnostic plot (PNG as per your preference)
    if save_plot and output_dir:
        figpath = os.path.join(output_dir, "trajectory", plot_filename)
        os.makedirs(os.path.dirname(figpath), exist_ok=True)

        plt.figure(figsize=(9, 7))
        # color by severity (normalized for colormap)
        sev_norm = (sev_levels - sev_levels.min()) / (sev_levels.max() - sev_levels.min() + 1e-16)
        sc = plt.scatter(coords_2d[:, 0], coords_2d[:, 1], c=sev_norm, cmap='viridis_r',
                         edgecolors='k', alpha=0.85, s=60)
        cbar = plt.colorbar(sc, label=f'Normalized {sev_col}')

        # CCA direction overlay
        dx, dy = cca.x_weights_[:, 0]
        scale = 0.5 * max(np.ptp(coords_2d[:, 0]), np.ptp(coords_2d[:, 1]))
        plt.plot([-scale*dx, scale*dx], [-scale*dy, scale*dy],
                 '--', linewidth=3, label=f'CCA dir (score={score:.3f})', color='red')

        plt.xlabel(f"PC{pc1+1}")
        plt.ylabel(f"PC{pc2+1}")
        plt.title("Trajectory on PCA space (CCA vs severity)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
        plt.close()
        if verbose:
            print(f"Saved trajectory plot: {figpath}")

    # Assemble output DF
    out = pd.DataFrame({
        "sample": samples,
        "pseudotime": pseudotime,
        "pc1": pc1 + 1,
        "pc2": pc2 + 1,
        "cca_score": score
    })

    # Save CSV
    if output_dir:
        csv_path = os.path.join(output_dir, "trajectory", "pseudotime_expression.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        out.to_csv(csv_path, index=False)
        if verbose:
            print(f"Saved pseudotime CSV: {csv_path}")

    return out


def pseudobulk_samples(adata, sample_column='sample', min_cells_per_sample=10):
    """
    Pseudobulk the data by summing counts per sample.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with cells x genes
    sample_column : str
        Column in adata.obs containing sample identifiers
    min_cells_per_sample : int
        Minimum number of cells required per sample
        
    Returns:
    --------
    pseudobulk_df : DataFrame
        Pseudobulked expression matrix (samples x genes)
    sample_metadata : DataFrame
        Metadata for each sample
    """
    # Get unique samples
    samples = adata.obs[sample_column].unique()
    
    # Initialize pseudobulk matrix
    pseudobulk_data = []
    sample_info = []
    
    for sample in samples:
        # Get cells for this sample
        sample_mask = adata.obs[sample_column] == sample
        sample_cells = adata[sample_mask]
        
        # Skip if too few cells
        if sample_cells.n_obs < min_cells_per_sample:
            print(f"Skipping sample {sample}: only {sample_cells.n_obs} cells")
            continue
            
        # Sum expression across cells (pseudobulk)
        if hasattr(sample_cells.X, 'toarray'):
            pseudobulk_expr = sample_cells.X.toarray().sum(axis=0)
        else:
            pseudobulk_expr = sample_cells.X.sum(axis=0)
        
        # Ensure it's a 1D array
        if pseudobulk_expr.ndim > 1:
            pseudobulk_expr = np.asarray(pseudobulk_expr).flatten()
            
        pseudobulk_data.append(pseudobulk_expr)
        
        # Collect sample metadata (take first row as representative)
        sample_meta = sample_cells.obs.iloc[0].to_dict()
        sample_meta['n_cells'] = sample_cells.n_obs
        sample_info.append(sample_meta)
    
    # Create DataFrame
    pseudobulk_df = pd.DataFrame(
        pseudobulk_data,
        index=[s[sample_column] for s in sample_info],
        columns=adata.var_names
    )
    
    sample_metadata = pd.DataFrame(sample_info)
    
    return pseudobulk_df, sample_metadata

def normalize_pseudobulk(pseudobulk_df, method='CPM'):
    """
    Normalize pseudobulk expression data.
    
    Parameters:
    -----------
    pseudobulk_df : DataFrame
        Raw pseudobulk counts (samples x genes)
    method : str
        Normalization method ('CPM', 'TPM', 'log1p_CPM')
        
    Returns:
    --------
    normalized_df : DataFrame
        Normalized expression matrix
    """
    if method == 'CPM':
        # Counts per million
        total_counts = pseudobulk_df.sum(axis=1)
        normalized_df = pseudobulk_df.div(total_counts, axis=0) * 1e6
    elif method == 'log1p_CPM':
        # Log-transformed CPM
        total_counts = pseudobulk_df.sum(axis=1)
        cpm = pseudobulk_df.div(total_counts, axis=0) * 1e6
        normalized_df = np.log1p(cpm)
    else:
        normalized_df = pseudobulk_df.copy()
    
    return normalized_df

def perform_pca(normalized_df, n_components=20, scale=True):
    """
    Perform PCA on normalized pseudobulk data.
    
    Parameters:
    -----------
    normalized_df : DataFrame
        Normalized expression matrix (samples x genes)
    n_components : int
        Number of principal components
    scale : bool
        Whether to scale features before PCA
        
    Returns:
    --------
    pca_result : dict
        Dictionary containing PCA results and model
    """
    # Prepare data
    if scale:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(normalized_df)
    else:
        scaled_data = normalized_df.values
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, min(scaled_data.shape)-1))
    pca_coords = pca.fit_transform(scaled_data)
    
    # Create DataFrame with PCA coordinates
    pca_df = pd.DataFrame(
        pca_coords,
        index=normalized_df.index,
        columns=[f'PC{i+1}' for i in range(pca_coords.shape[1])]
    )
    
    # Get loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=normalized_df.columns,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])]
    )
    
    return {
        'coords': pca_df,
        'loadings': loadings,
        'explained_variance': pca.explained_variance_ratio_,
        'pca_model': pca,
        'scaler': scaler if scale else None
    }

def create_pseudobulk_anndata(pca_result, sample_metadata, normalized_df):
    """
    Create an AnnData object from pseudobulk results for visualization.
    
    Parameters:
    -----------
    pca_result : dict
        PCA results from perform_pca()
    sample_metadata : DataFrame
        Sample metadata
    normalized_df : DataFrame
        Normalized expression matrix
        
    Returns:
    --------
    adata_vis : AnnData
        AnnData object for visualization
    """
    # Create AnnData object with normalized expression as main matrix
    adata_vis = ad.AnnData(X=normalized_df.values)
    
    # Add sample metadata to obs
    adata_vis.obs = sample_metadata.set_index('sample').loc[normalized_df.index].copy()
    adata_vis.obs_names = normalized_df.index
    
    # Add gene names to var
    adata_vis.var_names = normalized_df.columns
    
    # Store PCA coordinates as embedding
    pca_coords = pca_result['coords'].values
    adata_vis.obsm['X_pca'] = pca_coords
    
    # Store first 2 PCs as default embedding for visualization
    adata_vis.obsm['X_embedding'] = pca_coords[:, :2]
    
    # Store additional PC pairs for multi-dimensional visualization
    if pca_coords.shape[1] >= 3:
        adata_vis.obsm['X_pca_2_3'] = pca_coords[:, 1:3]
    if pca_coords.shape[1] >= 4:
        adata_vis.obsm['X_pca_1_3'] = pca_coords[:, [0, 2]]
        adata_vis.obsm['X_pca_1_4'] = pca_coords[:, [0, 3]]
        adata_vis.obsm['X_pca_3_4'] = pca_coords[:, [2, 3]]
    
    # Store variance explained
    adata_vis.uns['pca'] = {
        'variance_ratio': pca_result['explained_variance'],
        'variance': pca_result['explained_variance'] * 100
    }
    
    return adata_vis

def visualize_pseudobulk_results(
    pseudobulk_df,
    normalized_df,
    pca_result,
    sample_metadata,
    output_dir,
    color_columns=None,
    highlight_samples=None,
    annotate_samples=None,
    visualization_style='modern',
    show_density=False,
    show_ellipses=False,
    top_n_genes=50,
    additional_pc_pairs=True  # kept for API compatibility, ignored (we now only use PC1 vs PC2)
):
    """
    Generate visualizations for pseudobulk analysis using ONLY the first two PCs (PC1 vs PC2),
    and use 'sev.level' as the default label unless the user explicitly supplies `color_columns`.

    Parameters
    ----------
    pseudobulk_df : DataFrame
        Raw pseudobulk counts (samples x genes)
    normalized_df : DataFrame
        Normalized pseudobulk expression (samples x genes)
    pca_result : dict
        PCA results from perform_pca()
    sample_metadata : DataFrame
        Metadata for each sample
    output_dir : str
        Directory to save plots
    color_columns : list or None
        If provided and non-empty, use these columns for coloring.
        If None or empty, default to ['sev.level'] when present; otherwise no per-column PCA plots.
    highlight_samples : list or None
        Samples to highlight in plots
    annotate_samples : list or None
        Samples to annotate in plots
    visualization_style : str
        Style for plots ('modern', 'classic', 'minimal')
    show_density : bool
        Add density contours to PCA plots
    show_ellipses : bool
        Add confidence ellipses for groups
    top_n_genes : int
        Number of top variable genes to show
    additional_pc_pairs : bool
        Ignored (kept for compatibility). Visualization is restricted to PC1 vs PC2.

    Notes
    -----
    - Only PC1 vs PC2 is used for scatter visualizations.
    - Default label is 'sev.level' if present in sample metadata (unless user provides `color_columns`).
    """

    # Create figure directory
    fig_dir = os.path.join(output_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Create pseudo-AnnData object for visualization (X_embedding is PC1 vs PC2)
    adata_vis = create_pseudobulk_anndata(pca_result, sample_metadata, normalized_df)

    # Determine color columns:
    # - If user provided a non-empty list, use it as-is (filter to those that actually exist).
    # - Else, use ['sev.level'] when present; otherwise, use [] (no per-column PCA plots).
    if color_columns is not None and len(color_columns) > 0:
        # keep only columns present in metadata
        color_columns = [c for c in color_columns if c in adata_vis.obs.columns]
        if len(color_columns) == 0:
            print("Warning: Provided color_columns not found in metadata; no per-column PCA plots will be created.")
    else:
        if 'sev.level' in adata_vis.obs.columns:
            color_columns = ['sev.level']
        else:
            color_columns = []
            print("Note: 'sev.level' not found in metadata; skipping per-column PCA plots by default.")

    # 1) Individual PCA plots (ONLY PC1 vs PC2) per selected metadata column
    for color_col in color_columns:
        fig, ax = visualize_single_omics_embedding(
            adata_vis,
            color_col=color_col,
            embedding_key='X_embedding',               # PC1 vs PC2 only
            title=f'PCA: {color_col} (PC1 vs PC2)',
            figsize=(10, 8),
            point_size=120,
            alpha=0.8,
            show_density=show_density,
            show_ellipses=show_ellipses,
            highlight_samples=highlight_samples,
            annotate_samples=annotate_samples,
            style=visualization_style,
            output_path=os.path.join(fig_dir, f'pca_pc1_pc2_{color_col}.png'),
            show_legend=True,
            show_colorbar=True
        )
        plt.close()

    # 2) Multi-panel PCA comparisons (ONLY PC1 vs PC2)
    if len(color_columns) > 1:
        fig, axes = create_multi_panel_embedding(
            adata_vis,
            color_cols=color_columns[:min(6, len(color_columns))],
            embedding_key='X_embedding',               # PC1 vs PC2 only
            n_cols=3 if len(color_columns) > 2 else 2,
            figsize_per_panel=(6, 5),
            main_title='PCA of Pseudobulk Samples: Multiple Metadata Views (PC1 vs PC2)',
            output_path=os.path.join(fig_dir, 'pca_multi_panel_pc1_pc2.png'),
            point_size=80,
            alpha=0.8,
            show_legend=True,
            show_colorbar=True
        )
        plt.close()

    # 3) Enhanced PCA Scree plot (unchanged)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    variance_explained = pca_result['explained_variance'] * 100

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(variance_explained)))
    bars = ax.bar(range(1, len(variance_explained) + 1), variance_explained,
                  color=colors, edgecolor='navy', linewidth=1.5)

    cumulative_var = np.cumsum(variance_explained)
    ax2 = ax.twinx()
    line = ax2.plot(range(1, len(variance_explained) + 1), cumulative_var,
                    'ro-', linewidth=2, markersize=8, label='Cumulative',
                    markerfacecolor='red', markeredgecolor='darkred')
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.grid(False)

    ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='80% threshold')
    ax2.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% threshold')

    ax.set_xlabel('Principal Component', fontsize=13, fontweight='bold')
    ax.set_ylabel('Variance Explained (%)', fontsize=13, fontweight='bold')
    ax.set_title('PCA Scree Plot with Cumulative Variance', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(1, min(21, len(variance_explained) + 1)))
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    for i in range(min(10, len(bars))):
        height = bars[i].get_height()
        ax.text(bars[i].get_x() + bars[i].get_width() / 2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.legend(loc='center right')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'pca_scree_plot_enhanced.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4) Sample correlation heatmap with annotations (unchanged)
    fig, ax = plt.subplots(figsize=(14, 12))
    sample_corr = normalized_df.T.corr()
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    g = sns.clustermap(
        sample_corr,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8, 'orientation': 'horizontal'},
        figsize=(14, 12),
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.02, 0.83, 0.05, 0.015),
        annot=True if len(sample_corr) <= 15 else False,
        fmt='.2f' if len(sample_corr) <= 15 else '',
        annot_kws={'size': 8} if len(sample_corr) <= 15 else {}
    )
    g.ax_heatmap.set_xlabel('Samples', fontsize=13, fontweight='bold')
    g.ax_heatmap.set_ylabel('Samples', fontsize=13, fontweight='bold')
    plt.suptitle('Sample-Sample Correlation with Hierarchical Clustering',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(fig_dir, 'sample_correlation_clustered.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 5) Top variable genes heatmap (unchanged)
    gene_var = normalized_df.var(axis=0)
    top_var_genes = gene_var.nlargest(top_n_genes).index

    top_genes_data = normalized_df[top_var_genes].T
    z_scores = (top_genes_data - top_genes_data.mean(axis=1, keepdims=True)) / \
               (top_genes_data.std(axis=1, keepdims=True) + 1e-8)

    g = sns.clustermap(
        z_scores,
        cmap='RdBu_r',
        center=0,
        row_cluster=True,
        col_cluster=True,
        linewidths=0,
        cbar_kws={'label': 'Z-score', 'shrink': 0.5, 'orientation': 'vertical'},
        figsize=(16, 12),
        vmin=-3,
        vmax=3,
        dendrogram_ratio=(0.1, 0.15),
        yticklabels=True if top_n_genes <= 50 else False,
        xticklabels=True
    )
    g.ax_heatmap.set_xlabel('Samples', fontsize=13, fontweight='bold')
    g.ax_heatmap.set_ylabel('Genes', fontsize=13, fontweight='bold')
    plt.suptitle(f'Top {top_n_genes} Variable Genes (Z-score normalized)',
                 fontsize=15, fontweight='bold', y=1.0)
    plt.savefig(os.path.join(fig_dir, 'top_variable_genes_clustered.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 6) Sample statistics dashboard (unchanged)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    bar_color_palette = sns.color_palette("husl", len(sample_metadata))

    # Total counts per sample
    ax1 = fig.add_subplot(gs[0, :2])
    total_counts = pseudobulk_df.sum(axis=1)
    bars = ax1.bar(range(len(total_counts)), total_counts.values,
                   color=bar_color_palette, edgecolor='navy', linewidth=1.5)
    ax1.set_xticks(range(len(total_counts)))
    ax1.set_xticklabels(total_counts.index, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Total Counts (millions)', fontsize=11, fontweight='bold')
    ax1.set_title('Total Counts per Sample', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

    # Number of detected genes
    ax2 = fig.add_subplot(gs[1, :2])
    detected_genes = (pseudobulk_df > 0).sum(axis=1)
    bars = ax2.bar(range(len(detected_genes)), detected_genes.values,
                   color=bar_color_palette, edgecolor='darkgreen', linewidth=1.5)
    ax2.set_xticks(range(len(detected_genes)))
    ax2.set_xticklabels(detected_genes.index, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Number of Genes', fontsize=11, fontweight='bold')
    ax2.set_title('Detected Genes per Sample', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=detected_genes.mean(), color='red', linestyle='--',
                alpha=0.5, label=f'Mean: {detected_genes.mean():.0f}')
    ax2.legend()

    # Number of cells per sample
    if 'n_cells' in sample_metadata.columns:
        ax3 = fig.add_subplot(gs[2, :2])
        n_cells = sample_metadata.set_index('sample')['n_cells']
        bars = ax3.bar(range(len(n_cells)), n_cells.values,
                       color=bar_color_palette, edgecolor='darkred', linewidth=1.5)
        ax3.set_xticks(range(len(n_cells)))
        ax3.set_xticklabels(n_cells.index, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('Number of Cells', fontsize=11, fontweight='bold')
        ax3.set_title('Cells per Sample', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=n_cells.mean(), color='red', linestyle='--',
                    alpha=0.5, label=f'Mean: {n_cells.mean():.0f}')
        ax3.legend()

    # Distributions
    ax4 = fig.add_subplot(gs[0, 2])
    log_counts = np.log10(total_counts.values + 1)
    ax4.hist(log_counts, bins=20, color='skyblue', edgecolor='black',
             alpha=0.7, density=True)
    from scipy import stats
    kde = stats.gaussian_kde(log_counts)
    x_range = np.linspace(log_counts.min(), log_counts.max(), 100)
    ax4.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    ax4.set_xlabel('Log10(Total Counts + 1)', fontsize=10)
    ax4.set_ylabel('Density', fontsize=10)
    ax4.set_title('Library Size Distribution', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(detected_genes.values, bins=20, color='lightgreen',
             edgecolor='black', alpha=0.7, density=True)
    kde = stats.gaussian_kde(detected_genes.values)
    x_range = np.linspace(detected_genes.min(), detected_genes.max(), 100)
    ax5.plot(x_range, kde(x_range), 'darkgreen', linewidth=2, label='KDE')
    ax5.set_xlabel('Number of Genes', fontsize=10)
    ax5.set_ylabel('Density', fontsize=10)
    ax5.set_title('Gene Detection Distribution', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    if 'n_cells' in sample_metadata.columns:
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.hist(n_cells.values, bins=20, color='salmon',
                 edgecolor='black', alpha=0.7, density=True)
        kde = stats.gaussian_kde(n_cells.values)
        x_range = np.linspace(n_cells.min(), n_cells.max(), 100)
        ax6.plot(x_range, kde(x_range), 'darkred', linewidth=2, label='KDE')
        ax6.set_xlabel('Number of Cells', fontsize=10)
        ax6.set_ylabel('Density', fontsize=10)
        ax6.set_title('Cell Count Distribution', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()

    plt.suptitle('Sample Statistics Dashboard', fontsize=18, fontweight='bold')
    plt.savefig(os.path.join(fig_dir, 'sample_statistics_dashboard_enhanced.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"All visualizations successfully saved to {fig_dir}")
    if len(color_columns) > 0:
        print(f"Generated {len(color_columns)} PC1–PC2 PCA plots and comparative visualizations")
    else:
        print("Generated comparative visualizations without per-column PCA plots")

def preprocess_pseudobulk(
    h5ad_path,
    sample_meta_path,
    output_dir,
    sample_column='sample',
    cell_meta_path=None,
    batch_key='batch',
    min_cells=500,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
    normalization_method='log1p_CPM',
    n_pcs=20,
    color_columns=None,
    highlight_samples=None,
    annotate_samples=None,
    visualization_style='modern',
    show_density=False,
    show_ellipses=False,
    verbose=True
):
    """
    Preprocess single-cell data and perform pseudobulk analysis with PCA.
    
    This function:
      1. Reads and preprocesses the data (filter genes/cells, remove MT genes, etc.)
      2. Performs pseudobulking by sample
      3. Normalizes pseudobulk data
      4. Performs PCA
      5. Generates enhanced visualizations
      
    Parameters:
    -----------
    h5ad_path : str
        Path to input H5AD file
    sample_meta_path : str
        Path to sample metadata CSV
    output_dir : str
        Output directory for results
    sample_column : str
        Column name for sample identifiers
    cell_meta_path : str, optional
        Path to cell metadata CSV
    batch_key : str
        Batch key for metadata
    min_cells : int
        Minimum cells per sample
    min_features : int
        Minimum features per cell
    pct_mito_cutoff : float
        Maximum mitochondrial percentage
    exclude_genes : list
        Genes to exclude
    doublet : bool
        Perform doublet detection
    normalization_method : str
        Method for normalizing pseudobulk ('CPM', 'log1p_CPM')
    n_pcs : int
        Number of principal components
    color_columns : list
        Metadata columns to visualize in PCA
    highlight_samples : list
        Samples to highlight in plots
    annotate_samples : list
        Samples to annotate in plots
    visualization_style : str
        Style for visualizations ('modern', 'classic', 'minimal')
    show_density : bool
        Show density contours in PCA
    show_ellipses : bool
        Show confidence ellipses in PCA
    verbose : bool
        Print progress messages
      
    Returns:
    --------
      - adata: Preprocessed single-cell data
      - pseudobulk_df: Raw pseudobulk counts
      - normalized_df: Normalized pseudobulk expression
      - pca_result: PCA results
      - adata_vis: AnnData object for visualization
    """
    # Start timing
    start_time = time.time()

    # 0. Create output directories if not present
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating output directory")

    # 1. Read the raw count data from an existing H5AD
    if verbose:
        print('=== Read input dataset ===')
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f'Dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # Attach sample info
    if cell_meta_path is None:
        if sample_column not in adata.obs.columns: 
            adata.obs[sample_column] = adata.obs_names.str.split(':').str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path)
        cell_meta.set_index('barcode', inplace=True)
        adata.obs = adata.obs.join(cell_meta, how='left')

    # Merge sample metadata if provided
    if sample_meta_path is not None:
        sample_meta = pd.read_csv(sample_meta_path)
        adata.obs = adata.obs.merge(sample_meta, on=sample_column, how='left')
    
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=min_features)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if verbose:
        print(f"After basic filtering -- Cells remaining: {adata.n_obs}, Genes remaining: {adata.n_vars}")

    # Mito QC
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()

    # Exclude genes if needed
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()
    if verbose:
        print(f"After removing MT genes and excluded genes -- Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # Filter samples with too few cells
    cell_counts_per_patient = adata.obs.groupby(sample_column).size()
    if verbose:
        print("\nSample counts BEFORE filtering:")
        print(cell_counts_per_patient.sort_values(ascending=False))
    
    patients_to_keep = cell_counts_per_patient[cell_counts_per_patient >= min_cells].index
    if verbose:
        print(f"\nSamples retained (>= {min_cells} cells): {list(patients_to_keep)}")
    
    adata = adata[adata.obs[sample_column].isin(patients_to_keep)].copy()
    
    cell_counts_after = adata.obs[sample_column].value_counts()
    if verbose:
        print("\nSample counts AFTER filtering:")
        print(cell_counts_after.sort_values(ascending=False))

    # Drop genes that are too rare
    min_cells_for_gene = int(0.01 * adata.n_obs)
    sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)
    if verbose:
        print(f"Final filtering -- Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # Optional doublet detection
    if doublet:
        if verbose:
            print("Performing doublet detection...")
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            sc.pp.scrublet(adata, batch_key=sample_column)
            adata = adata[~adata.obs['predicted_doublet']].copy()
        if verbose:
            print(f"After doublet removal -- Cells: {adata.n_obs}")
    
    # Save raw data
    adata.raw = adata.copy()
    
    # ========== PSEUDOBULKING ==========
    if verbose:
        print("\n=== Performing Pseudobulking ===")
    
    # Pseudobulk by sample
    pseudobulk_df, sample_metadata = pseudobulk_samples(adata, sample_column=sample_column)
    if verbose:
        print(f"Pseudobulk matrix shape: {pseudobulk_df.shape}")
    
    # Normalize pseudobulk data
    normalized_df = normalize_pseudobulk(pseudobulk_df, method=normalization_method)
    if verbose:
        print(f"Normalization method: {normalization_method}")
    
    # Perform PCA
    if verbose:
        print("\n=== Performing PCA ===")
    pca_result = perform_pca(normalized_df, n_components=n_pcs)
    
    # Print variance explained
    if verbose:
        print("\nVariance explained by top PCs:")
        cumulative = 0
        for i in range(min(10, len(pca_result['explained_variance']))):
            var_exp = pca_result['explained_variance'][i]*100
            cumulative += var_exp
            print(f"  PC{i+1}: {var_exp:.2f}% (Cumulative: {cumulative:.2f}%)")
        print(f"\n  Total variance explained by first {n_pcs} PCs: {sum(pca_result['explained_variance'])*100:.2f}%")
    
    # Generate visualizations
    if verbose:
        print("\n=== Generating Enhanced Visualizations ===")
    
    try:
        visualize_pseudobulk_results(
            pseudobulk_df=pseudobulk_df,
            normalized_df=normalized_df,
            pca_result=pca_result,
            sample_metadata=sample_metadata,
            output_dir=output_dir,
            color_columns=color_columns,
            highlight_samples=highlight_samples,
            annotate_samples=annotate_samples,
            visualization_style=visualization_style,
            show_density=show_density,
            show_ellipses=show_ellipses
        )
    except Exception as e:
        print(f"Warning: Some visualizations may have failed: {e}")
        print("Continuing with data saving...")
    
    # Create AnnData object for visualization (for potential reuse)
    adata_vis = create_pseudobulk_anndata(pca_result, sample_metadata, normalized_df)
    
    # Save results
    if verbose:
        print("\n=== Saving Results ===")
    
    # Save processed single-cell data
    sc.write(os.path.join(output_dir, 'adata_processed.h5ad'), adata)
    
    # Save pseudobulk data
    pseudobulk_df.to_csv(os.path.join(output_dir, 'pseudobulk_counts.csv'))
    normalized_df.to_csv(os.path.join(output_dir, 'pseudobulk_normalized.csv'))
    
    # Save PCA results
    pca_result['coords'].to_csv(os.path.join(output_dir, 'pca_coordinates.csv'))
    pca_result['loadings'].to_csv(os.path.join(output_dir, 'pca_loadings.csv'))
    
    # Save sample metadata
    sample_metadata.to_csv(os.path.join(output_dir, 'sample_metadata.csv'), index=False)
    
    # Save variance explained
    var_explained_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(pca_result['explained_variance']))],
        'Variance_Explained': pca_result['explained_variance'],
        'Cumulative_Variance': np.cumsum(pca_result['explained_variance'])
    })
    var_explained_df.to_csv(os.path.join(output_dir, 'pca_variance_explained.csv'), index=False)
    
    # Save visualization AnnData
    adata_vis.write(os.path.join(output_dir, 'adata_visualization.h5ad'))
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if verbose:
        print(f"\n=== Function execution time: {elapsed_time:.2f} seconds ===")
        print(f"Results saved to: {output_dir}")
        print("\nOutput files:")
        print(f"  - adata_processed.h5ad: Preprocessed single-cell data")
        print(f"  - adata_visualization.h5ad: Visualization-ready AnnData")
        print(f"  - pseudobulk_counts.csv: Raw pseudobulk counts")
        print(f"  - pseudobulk_normalized.csv: Normalized pseudobulk expression")
        print(f"  - pca_coordinates.csv: PCA coordinates")
        print(f"  - pca_loadings.csv: PCA gene loadings")
        print(f"  - sample_metadata.csv: Sample metadata")
        print(f"  - pca_variance_explained.csv: Variance explained by PCs")
        print(f"  - figures/: Visualization outputs")
    
    return adata, pseudobulk_df, normalized_df, pca_result, adata_vis

if __name__ == "__main__":
    # Example usage with enhanced parameters
    h5ad_path = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/count_data.h5ad"
    sample_meta_path = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv"
    output_dir = "/dcs07/hongkai/data/harry/result/naive_pseudobulk/covid_400_sample"
    
    # Run with enhanced visualizations
    adata, pseudobulk_df, normalized_df, pca_result, adata_vis = preprocess_pseudobulk(
        h5ad_path=h5ad_path,
        sample_meta_path=sample_meta_path,
        output_dir=output_dir,
        normalization_method="log1p_CPM",
        n_pcs=20,
        visualization_style="modern",
        show_density=False,
        show_ellipses=True,
        verbose=True
    )

    # === NEW: sample–sample cosine distance on PCA ===
    try:
        # Use the first k PCs (or set n_components=None to use all PCs)
        k = min(10, pca_result['coords'].shape[1])
        dist_df = compute_sample_distance_cosine(
            pca_result=pca_result,
            n_components=k,
            output_dir=output_dir,
            save_plot=True,
            verbose=True
        )
    except Exception as e:
        print(f"Warning: cosine distance computation failed: {e}")

    # === Trajectory on PCA space (CCA vs sev.level) ===
    try:
        # Build sample_metadata from adata_vis.obs (index is sample)
        sample_metadata_df = adata_vis.obs.copy().reset_index().rename(columns={"index": "sample"})

        traj_df = compute_pseudotime_from_pca(
            pca_result=pca_result,
            sample_metadata=sample_metadata_df,
            sev_col="sev.level",
            output_dir=output_dir,       # saves CSV to <output_dir>/trajectory/pseudotime_expression.csv
            auto_select_best_2pc=True,
            save_plot=True,              # saves diagnostic PNG to <output_dir>/trajectory/trajectory_pc_space.png
            verbose=True
        )

        # Attach pseudotime to adata_vis and save updated file
        adata_vis.obs = adata_vis.obs.join(traj_df.set_index("sample")[["pseudotime"]], how="left")
        adata_vis.write(os.path.join(output_dir, "adata_visualization.h5ad"))
        print("Pseudotime added to adata_vis.obs and file updated.")

    except Exception as e:
        print(f"Warning: trajectory pseudotime computation failed in __main__: {e}")
