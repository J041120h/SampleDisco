#!/usr/bin/env python3
"""
Multi-omics Optimal Resolution Finding Module - GPU/Linux Version

This module finds the optimal clustering resolution for integrated RNA+ATAC single-cell data
by maximizing CCA (Canonical Correlation Analysis) correlation between dimension reduction
embeddings and severity levels.

GPU-accelerated version using RAPIDS for faster computation.

Key Features:
- Two-pass search strategy: coarse (0.1 steps) → fine (0.01 steps around best)
- Supports optimization on either RNA or ATAC modality
- Supports either expression or proportion DR type
- Uses cell_types_multiomics_linux for GPU-accelerated clustering
- Uses calculate_multiomics_sample_embedding with use_gpu=True for GPU-accelerated sample embedding
- CCA visualization plots with automatic best PC selection
- Optional modality correlation analysis
- Comprehensive visualization and summary outputs
- Optional corrected p-value computation

Author: Adapted for GPU/Linux usage
"""

import os
import sys
import time
import shutil
import warnings
import pickle
from typing import Optional, Union, List, Tuple, Dict, Any, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preparation.multi_omics_cell_type_gpu import cell_types_multiomics_linux
from sample_embedding.calculate_multiomics_sample_embedding import calculate_multiomics_sample_embedding
from sample_trajectory.CCA import plot_cca_on_2d_pca
from sample_trajectory.CCA_test import (
    generate_null_distribution,
    generate_corrected_null_distribution,
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def suppress_warnings():
    """
    Suppress specific warnings that are expected during CCA analysis.
    """
    warnings.filterwarnings('ignore', category=UserWarning, 
                          message='.*y residual is constant at iteration.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, 
                          message='.*invalid value encountered in divide.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, 
                          message='.*All-NaN slice encountered.*')


def ensure_non_categorical_columns(adata: AnnData, columns: List) -> AnnData:
    """
    Convert specified columns from categorical to string type.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to modify
    columns : list
        List of column names to check and convert
        
    Returns
    -------
    AnnData
        Modified AnnData object with non-categorical columns
    """
    flat_columns = []
    for col in columns:
        if isinstance(col, list):
            flat_columns.extend(col)
        elif col is not None:
            flat_columns.append(col)
    
    for col in flat_columns:
        if col in adata.obs.columns:
            if pd.api.types.is_categorical_dtype(adata.obs[col]):
                adata.obs[col] = adata.obs[col].astype(str)
    
    return adata


# =============================================================================
# CCA ANALYSIS FUNCTIONS
# =============================================================================

def cca_analysis_multiomics(
    pseudobulk_adata: AnnData,
    modality: str,
    column: str,
    sev_col: str,
    n_components: int = 2,
    num_PCs: Optional[int] = None
) -> dict:
    """
    Perform CCA analysis between DR embeddings and severity levels for a specific modality.
    
    Parameters
    ----------
    pseudobulk_adata : AnnData
        Pseudobulk AnnData object containing DR results in .uns
    modality : str
        Modality to analyze ('RNA' or 'ATAC')
    column : str
        Column name in .uns containing DR coordinates
    sev_col : str
        Column name in .obs for severity levels
    n_components : int
        Number of CCA components to compute
    num_PCs : int, optional
        Number of PCs to use for CCA
        
    Returns
    -------
    dict
        Results dictionary with CCA score and metadata
    """
    result = {
        'cca_score': np.nan,
        'n_samples': 0,
        'n_features': 0,
        'modality': modality,
        'column': column,
        'valid': False,
        'error_message': None,
        'X_weights': None,
        'Y_weights': None,
        'X_loadings': None,
        'Y_loadings': None,
        'num_PCs_used': 0
    }
    
    try:
        # Input validation
        if column not in pseudobulk_adata.uns:
            result['error_message'] = f"Column '{column}' not found in uns"
            return result
        if 'modality' not in pseudobulk_adata.obs.columns:
            result['error_message'] = "Column 'modality' not found in obs"
            return result
        if sev_col not in pseudobulk_adata.obs.columns:
            result['error_message'] = f"Column '{sev_col}' not found in obs"
            return result
        
        # Standardize indices to lowercase
        obs_standardized = pseudobulk_adata.obs.copy()
        obs_standardized.index = obs_standardized.index.str.lower()
        
        uns_data_standardized = pseudobulk_adata.uns[column].copy()
        uns_data_standardized.index = uns_data_standardized.index.str.lower()
        
        # Extract modality-specific samples
        modality_mask = obs_standardized['modality'] == modality
        
        if not modality_mask.any():
            result['error_message'] = f"No samples found for modality: {modality}"
            return result
        
        dr_coords_full = uns_data_standardized.loc[modality_mask].copy()
        sev_levels = obs_standardized.loc[modality_mask, sev_col].values
        
        # Determine number of PCs to use
        max_pcs = dr_coords_full.shape[1]
        num_PCs_to_use = min(num_PCs, max_pcs) if num_PCs else max_pcs
        
        dr_coords = dr_coords_full.iloc[:, :num_PCs_to_use]
        
        # Basic validation
        if len(dr_coords) < 3:
            result['error_message'] = f"Insufficient samples: {len(dr_coords)}"
            return result
        if len(np.unique(sev_levels)) < 2:
            result['error_message'] = "Insufficient severity level variance"
            return result
        
        # Prepare data for CCA
        X = dr_coords.values
        y = sev_levels.reshape(-1, 1)
        
        result['n_samples'] = len(X)
        result['n_features'] = X.shape[1]
        result['num_PCs_used'] = num_PCs_to_use
        
        # Limit components based on data dimensions
        max_components = min(X.shape[1], y.shape[1], X.shape[0] - 1)
        n_components_actual = min(n_components, max_components)
        
        if n_components_actual < 1:
            result['error_message'] = "Cannot compute CCA components"
            return result
        
        # Standardize data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        # Fit CCA
        cca = CCA(n_components=n_components_actual, max_iter=1000, tol=1e-6)
        cca.fit(X_scaled, y_scaled)
        
        X_c, y_c = cca.transform(X_scaled, y_scaled)
        correlation = np.corrcoef(X_c[:, 0], y_c[:, 0])[0, 1]
        cca_score = abs(correlation)
        
        result['cca_score'] = cca_score
        result['valid'] = True
        result['X_weights'] = cca.x_weights_.flatten()
        result['Y_weights'] = cca.y_weights_.flatten()
        result['X_loadings'] = cca.x_loadings_ if hasattr(cca, 'x_loadings_') else None
        result['Y_loadings'] = cca.y_loadings_ if hasattr(cca, 'y_loadings_') else None
        
    except Exception as e:
        result['error_message'] = f"CCA failed: {str(e)}"
    
    return result


def batch_cca_analysis_multiomics(
    pseudobulk_adata: AnnData,
    dr_column: str,
    sev_col: str,
    dr_type: str,
    modalities: Optional[List[str]] = None,
    n_components: int = 2,
    num_PCs: Optional[int] = None,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Run CCA analysis across modalities for a single DR column.
    
    Parameters
    ----------
    pseudobulk_adata : AnnData
        Pseudobulk AnnData with DR results
    dr_column : str
        DR column name to analyze
    sev_col : str
        Severity column name
    dr_type : str
        DR type for file naming
    modalities : list, optional
        Modalities to analyze
    n_components : int
        Number of CCA components
    num_PCs : int, optional
        Number of PCs to use
    output_dir : str, optional
        Directory to save results
        
    Returns
    -------
    pd.DataFrame
        Results with CCA scores for all modalities
    """
    if modalities is None:
        modalities = pseudobulk_adata.obs['modality'].unique()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for modality in modalities:
        if dr_column in pseudobulk_adata.uns:
            result = cca_analysis_multiomics(
                pseudobulk_adata=pseudobulk_adata,
                modality=modality,
                column=dr_column,
                sev_col=sev_col,
                n_components=n_components,
                num_PCs=num_PCs
            )
            results.append(result)
        else:
            results.append({
                'cca_score': np.nan,
                'n_samples': 0,
                'n_features': 0,
                'modality': modality,
                'column': dr_column,
                'valid': False,
                'error_message': f"Column '{dr_column}' not found",
                'X_weights': None,
                'Y_weights': None,
                'X_loadings': None,
                'Y_loadings': None,
                'num_PCs_used': 0
            })
    
    results_df = pd.DataFrame(results)
    
    if output_dir:
        summary_df = results_df[['modality', 'column', 'cca_score', 'n_samples', 
                                'n_features', 'num_PCs_used', 'valid', 'error_message']]
        summary_path = os.path.join(output_dir, f'cca_results_{dr_type}_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        complete_path = os.path.join(output_dir, f'cca_results_{dr_type}_complete.pkl')
        with open(complete_path, 'wb') as f:
            pickle.dump(results_df, f)
    
    return results_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_cca_vs_resolution_multiomics(
    df_results: pd.DataFrame,
    optimization_target: str,
    dr_type: str,
    best_resolution: float,
    output_dir: str
) -> None:
    """
    Create visualization of CCA scores across resolutions for multi-omics data.
    """
    df_sorted = df_results.sort_values('resolution').copy()
    resolutions = df_sorted['resolution'].values
    
    colors = {'rna': '#2E86AB', 'atac': '#A23B72'}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    best_score = None
    for modality in ['rna', 'atac']:
        cca_col = f'{modality}_cca_{dr_type}'
        if cca_col not in df_sorted.columns:
            continue
        cca_values = df_sorted[cca_col].values
        
        linewidth = 2.5 if modality == optimization_target else 1.5
        alpha = 1.0 if modality == optimization_target else 0.7
        
        ax.plot(resolutions, cca_values, 
               color=colors[modality], 
               linewidth=linewidth, 
               alpha=alpha,
               marker='o', 
               markersize=6 if modality == optimization_target else 4,
               label=f'{modality.upper()}')
        
        if modality == optimization_target:
            best_score = df_sorted.loc[
                df_sorted['resolution'] == best_resolution, cca_col
            ].iloc[0]
    
    ax.axvline(x=best_resolution, color='#E74C3C', linestyle='--', 
              linewidth=2, alpha=0.8, label=f'Best: {best_resolution:.3f}')
    
    if best_score is not None:
        ax.scatter([best_resolution], [best_score], 
                  color='#E74C3C', s=150, zorder=5, edgecolors='white', linewidth=2)
    
    ax.set_xlabel('Resolution', fontsize=12, fontweight='medium')
    ax.set_ylabel('CCA Score', fontsize=12, fontweight='medium')
    ax.set_title(f'CCA Score vs Resolution\nTarget: {optimization_target.upper()} {dr_type.capitalize()}', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#FAFAFA')
    ax.set_xlim(resolutions.min() - 0.02, resolutions.max() + 0.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cca_vs_resolution_{optimization_target}_{dr_type}.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def analyze_modality_correlation_multiomics(
    df_results: pd.DataFrame,
    dr_type: str,
    optimization_target: str,
    output_dir: str,
    verbose: bool = True
) -> dict:
    """
    Analyze correlation between RNA and ATAC CCA scores across resolutions.
    """
    corr_dir = os.path.join(output_dir, "modality_correlation")
    os.makedirs(corr_dir, exist_ok=True)
    
    rna_col = f'rna_cca_{dr_type}'
    atac_col = f'atac_cca_{dr_type}'
    
    if rna_col not in df_results.columns or atac_col not in df_results.columns:
        if verbose:
            print("Warning: Required columns not found for correlation analysis")
        return {'pearson_r': np.nan, 'interpretation': 'Columns not found'}
    
    valid_mask = ~df_results[rna_col].isna() & ~df_results[atac_col].isna()
    valid_df = df_results[valid_mask].copy()
    
    if len(valid_df) < 3:
        if verbose:
            print("Warning: Insufficient data for correlation analysis")
        return {'pearson_r': np.nan, 'interpretation': 'Insufficient data'}
    
    rna_scores = valid_df[rna_col].values
    atac_scores = valid_df[atac_col].values
    resolutions = valid_df['resolution'].values
    
    pearson_r, pearson_p = stats.pearsonr(rna_scores, atac_scores)
    spearman_r, spearman_p = stats.spearmanr(rna_scores, atac_scores)
    
    if pearson_r > 0.7:
        interpretation = "Strong positive alignment"
    elif pearson_r > 0.3:
        interpretation = "Moderate positive alignment"
    elif pearson_r > -0.3:
        interpretation = "Weak/no alignment"
    elif pearson_r > -0.7:
        interpretation = "Moderate trade-off"
    else:
        interpretation = "Strong trade-off"
    
    results = {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'interpretation': interpretation, 'n_points': len(valid_df)
    }
    
    if verbose:
        print(f"\n--- Modality Correlation ({dr_type}) ---")
        print(f"Pearson r = {pearson_r:.4f} (p = {pearson_p:.4f})")
        print(f"Spearman r = {spearman_r:.4f} (p = {spearman_p:.4f})")
        print(f"Interpretation: {interpretation}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    scatter = ax1.scatter(rna_scores, atac_scores, c=resolutions, 
                          cmap='viridis', s=80, alpha=0.8, edgecolors='white', linewidth=1)
    z = np.polyfit(rna_scores, atac_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(rna_scores.min(), rna_scores.max(), 100)
    ax1.plot(x_line, p(x_line), color='#E74C3C', linestyle='--', linewidth=2)
    ax1.set_xlabel('RNA CCA Score', fontsize=11)
    ax1.set_ylabel('ATAC CCA Score', fontsize=11)
    ax1.set_title(f'RNA vs ATAC (r = {pearson_r:.3f})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Resolution', fontsize=10)
    
    ax2 = axes[1]
    ax2.plot(resolutions, rna_scores, 'o-', color='#2E86AB', linewidth=2, markersize=6, label='RNA')
    ax2.plot(resolutions, atac_scores, 's-', color='#A23B72', linewidth=2, markersize=6, label='ATAC')
    ax2.set_xlabel('Resolution', fontsize=11)
    ax2.set_ylabel('CCA Score', fontsize=11)
    ax2.set_title(f'{dr_type.capitalize()} DR Scores', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(corr_dir, f'modality_correlation_{optimization_target}_{dr_type}.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    with open(os.path.join(corr_dir, f'correlation_results_{optimization_target}_{dr_type}.txt'), 'w') as f:
        f.write(f"Modality Correlation Analysis - {optimization_target.upper()} {dr_type.upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Pearson r:  {pearson_r:.4f} (p = {pearson_p:.4f})\n")
        f.write(f"Spearman r: {spearman_r:.4f} (p = {spearman_p:.4f})\n\n")
        f.write(f"Interpretation: {interpretation}\n")
    
    return results


def create_resolution_optimization_summary_multiomics(
    df_results: pd.DataFrame,
    final_best_resolution: float,
    optimization_target: str,
    dr_type: str,
    summary_dir: str
) -> None:
    """
    Create summary text files for the optimization results.
    """
    modalities = ['rna', 'atac']
    df_sorted = df_results.sort_values('resolution').copy()
    
    for modality in modalities:
        cca_col = f'{modality}_cca_{dr_type}'
        
        if cca_col not in df_sorted.columns:
            continue
        
        txt_path = os.path.join(summary_dir, f'{modality}_{dr_type}_results.txt')
        
        with open(txt_path, 'w') as f:
            f.write(f"{modality.upper()} - {dr_type.capitalize()} DR Results\n")
            f.write("=" * 60 + "\n\n")
            
            if modality == optimization_target:
                f.write("*** OPTIMIZATION TARGET ***\n")
                f.write(f"Best Resolution: {final_best_resolution:.3f}\n")
                best_cca = df_sorted.loc[
                    df_sorted['resolution'] == final_best_resolution, cca_col
                ].iloc[0]
                f.write(f"Best CCA Score: {best_cca:.4f}\n\n")
            
            f.write(f"{'Resolution':<12} {'CCA Score':<12}\n")
            f.write("-" * 30 + "\n")
            
            for _, row in df_sorted.iterrows():
                cca_val = row.get(cca_col, np.nan)
                cca_str = f"{cca_val:.4f}" if not np.isnan(cca_val) else "N/A"
                f.write(f"{row['resolution']:<12.3f} {cca_str:<12}\n")
            
            valid_mask = ~df_sorted[cca_col].isna()
            if valid_mask.any():
                f.write("\n" + "-" * 30 + "\n")
                f.write(f"Min: {df_sorted.loc[valid_mask, cca_col].min():.4f}  ")
                f.write(f"Max: {df_sorted.loc[valid_mask, cca_col].max():.4f}  ")
                f.write(f"Mean: {df_sorted.loc[valid_mask, cca_col].mean():.4f}\n")


# =============================================================================
# MAIN RESOLUTION OPTIMIZATION FUNCTION - GPU VERSION
# =============================================================================

def find_optimal_cell_resolution_multiomics_linux(
    AnnData_integrated: AnnData,
    output_dir: str,
    optimization_target: Literal["rna", "atac"] = "rna",
    dr_type: Literal["expression", "proportion"] = "expression",
    sev_col: str = "sev.level",
    batch_col: Optional[Union[str, List[str]]] = None,
    sample_col: str = "sample",
    celltype_col: str = "cell_type",
    modality_col: str = "modality",
    use_rep: str = 'X_glue',
    # Sample embedding parameters
    sample_hvg_number: int = 2000,
    n_expression_components: int = 10,
    n_proportion_components: int = 10,
    harmony_for_proportion: bool = True,
    preserve_cols: Optional[Union[str, List[str]]] = None,
    hvg_modality: str = "RNA",
    # Search parameters
    coarse_start: float = 0.1,
    coarse_end: float = 1.0,
    coarse_step: float = 0.1,
    fine_range: float = 0.05,
    fine_step: float = 0.01,
    # Analysis options
    visualize_cell_types: bool = True,
    analyze_modality_alignment: bool = True,
    compute_corrected_pvalues: bool = False,
    verbose: bool = True
) -> Tuple[float, pd.DataFrame]:
    """
    Find optimal clustering resolution for integrated RNA+ATAC data - GPU/Linux version.
    
    Performs two-pass search to find the resolution that maximizes CCA correlation
    between DR embeddings and severity levels for a specified modality and DR type.
    
    Uses GPU-accelerated functions for faster computation:
    - cell_types_multiomics_linux for clustering
    - calculate_multiomics_sample_embedding with use_gpu=True for sample embedding
    
    Parameters
    ----------
    AnnData_integrated : AnnData
        Integrated AnnData with RNA and ATAC data
    output_dir : str
        Output directory
    optimization_target : str
        Which modality to optimize: "rna" or "atac"
    dr_type : str
        Which DR type to optimize: "expression" or "proportion"
    sev_col : str
        Severity column name
    batch_col : str or list, optional
        Batch column(s) for correction
    sample_col : str
        Sample identifier column
    celltype_col : str
        Cell type column name
    modality_col : str
        Modality label column
    use_rep : str
        Representation for clustering
    sample_hvg_number : int
        Number of highly variable genes for sample embedding
    n_expression_components : int
        Number of expression PCA components
    n_proportion_components : int
        Number of proportion PCA components
    harmony_for_proportion : bool
        Whether to apply Harmony correction to proportion
    preserve_cols : str or list, optional
        Columns to preserve through pseudobulk computation
    hvg_modality : str
        Which modality to use for HVG selection
    coarse_start : float
        Starting resolution for coarse search
    coarse_end : float
        Ending resolution for coarse search
    coarse_step : float
        Step size for coarse search
    fine_range : float
        Range around best for fine search
    fine_step : float
        Step size for fine search
    visualize_cell_types : bool
        Whether to create cell type visualizations
    analyze_modality_alignment : bool
        Whether to analyze RNA/ATAC score correlation
    compute_corrected_pvalues : bool
        Whether to compute corrected p-values
    verbose : bool
        Print verbose output
        
    Returns
    -------
    Tuple[float, pd.DataFrame]
        Optimal resolution and results dataframe
    """
    
    print("\n" + "=" * 80)
    print("MULTI-OMICS RESOLUTION OPTIMIZATION (GPU/Linux)")
    print("=" * 80)
    
    # Input validation
    if optimization_target not in ["rna", "atac"]:
        raise ValueError("optimization_target must be 'rna' or 'atac'")
    if dr_type not in ["expression", "proportion"]:
        raise ValueError("dr_type must be 'expression' or 'proportion'")
    
    # Convert batch_col to list format
    if isinstance(batch_col, str):
        batch_col = [batch_col]
    elif batch_col is None:
        batch_col = []
    
    # Ensure modality is included in batch columns for sample embedding
    batch_cols_for_embedding = batch_col.copy()
    if modality_col not in batch_cols_for_embedding:
        batch_cols_for_embedding.append(modality_col)
    
    # Determine num_PCs based on dr_type
    num_PCs = n_expression_components if dr_type == "expression" else n_proportion_components
    
    # Setup output directories
    main_output_dir = os.path.join(output_dir, f"Integration_optimization_{optimization_target}_{dr_type}")
    os.makedirs(main_output_dir, exist_ok=True)
    resolutions_dir = os.path.join(main_output_dir, "resolutions")
    os.makedirs(resolutions_dir, exist_ok=True)
    summary_dir = os.path.join(main_output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Print configuration
    if verbose:
        print(f"\nConfiguration:")
        print(f"  Optimization target: {optimization_target.upper()} {dr_type.upper()}")
        print(f"  Representation: {use_rep}")
        print(f"  Expression components: {n_expression_components}")
        print(f"  Proportion components: {n_proportion_components}")
        print(f"  HVG number: {sample_hvg_number}")
        print(f"  HVG modality: {hvg_modality}")
        print(f"  Cell type method: cell_types_multiomics_linux (GPU)")
        print(f"  Sample embedding: calculate_multiomics_sample_embedding (GPU)")
        print(f"  Cell type visualization: {visualize_cell_types}")
        if batch_col:
            print(f"  Batch columns: {batch_col}")
        print(f"  Output: {main_output_dir}")
    
    # Prepare data
    columns_to_check = [celltype_col, modality_col, sev_col, sample_col]
    if batch_col:
        columns_to_check.extend(batch_col)
    AnnData_integrated = ensure_non_categorical_columns(AnnData_integrated, columns_to_check)
    
    # Storage
    all_results = []
    all_null_results = []
    
    # Define DR column
    dr_column = f'X_DR_{dr_type}'
    
    def process_resolution(resolution: float, search_pass: str) -> Tuple[dict, dict]:
        """Process a single resolution using GPU-accelerated functions."""
        nonlocal AnnData_integrated
        
        if verbose:
            print(f"\n--- Resolution: {resolution:.3f} ({search_pass}) ---")
        
        resolution_dir = os.path.join(resolutions_dir, f"resolution_{resolution:.3f}_{dr_type}")
        os.makedirs(resolution_dir, exist_ok=True)
        
        result_dict = {
            'resolution': resolution,
            f'rna_cca_{dr_type}': np.nan,
            f'atac_cca_{dr_type}': np.nan,
            'optimization_score': np.nan,
            'p_value': np.nan,
            'corrected_pvalue': np.nan,
            'pass': search_pass,
            'n_clusters': 0
        }
        null_result = {'resolution': resolution, 'null_scores': None}
        
        try:
            # Clean up previous clustering
            if celltype_col in AnnData_integrated.obs.columns:
                AnnData_integrated.obs.drop(columns=[celltype_col], inplace=True, errors='ignore')
            if 'label_transfer_confidence' in AnnData_integrated.obs.columns:
                AnnData_integrated.obs.drop(columns=['label_transfer_confidence'], inplace=True, errors='ignore')
            if modality_col in AnnData_integrated.obs.columns:
                AnnData_integrated.obs[modality_col] = AnnData_integrated.obs[modality_col].astype(str)
            
            if 'X_umap' in AnnData_integrated.obsm:
                del AnnData_integrated.obsm['X_umap']
            
            # Step 1: Clustering using cell_types_multiomics_linux (GPU version)
            AnnData_integrated = cell_types_multiomics_linux(
                adata=AnnData_integrated,
                modality_column=modality_col,
                rna_modality_value="RNA",
                atac_modality_value="ATAC",
                cell_type_column=celltype_col,
                cluster_resolution=resolution,
                use_rep=use_rep,
                k_neighbors=15,
                transfer_metric="cosine",
                compute_umap=visualize_cell_types,
                save=False,
                output_dir=resolution_dir if visualize_cell_types else None,
                verbose=False,
                generate_plots=visualize_cell_types
            )
            
            n_clusters = AnnData_integrated.obs[celltype_col].nunique()
            result_dict['n_clusters'] = n_clusters
            if verbose:
                print(f"  Clusters: {n_clusters}")
            
            # Step 2: Sample embedding using calculate_multiomics_sample_embedding (GPU)
            pseudobulk_df, pseudobulk_adata = calculate_multiomics_sample_embedding(
                adata=AnnData_integrated,
                sample_col=sample_col,
                celltype_col=celltype_col,
                batch_col=batch_cols_for_embedding,
                output_dir=resolution_dir,
                sample_hvg_number=sample_hvg_number,
                n_expression_components=n_expression_components,
                n_proportion_components=n_proportion_components,
                harmony_for_proportion=harmony_for_proportion,
                preserve_cols_in_sample_embedding=preserve_cols,
                use_gpu=True,  # GPU version
                atac=False,
                save=True,
                verbose=False,
                hvg_modality=hvg_modality,
                modality_col=modality_col,
            )
            
            # Step 3: CCA analysis
            cca_results_df = batch_cca_analysis_multiomics(
                pseudobulk_adata=pseudobulk_adata,
                dr_column=dr_column,
                sev_col=sev_col,
                dr_type=dr_type,
                modalities=['RNA', 'ATAC'],
                n_components=1,
                num_PCs=num_PCs,
                output_dir=resolution_dir
            )
            
            for _, row in cca_results_df.iterrows():
                modality = row['modality'].lower()
                result_dict[f'{modality}_cca_{dr_type}'] = row['cca_score']
                if row['valid'] and not np.isnan(row['cca_score']) and verbose:
                    print(f"  {row['modality']} {dr_type}: {row['cca_score']:.4f}")
            
            # Step 4: CCA visualization for each modality
            for modality in ['RNA', 'ATAC']:
                try:
                    obs_standardized = pseudobulk_adata.obs.copy()
                    obs_standardized.index = obs_standardized.index.str.lower()
                    
                    uns_data_standardized = pseudobulk_adata.uns[dr_column].copy()
                    uns_data_standardized.index = uns_data_standardized.index.str.lower()
                    
                    modality_mask_std = obs_standardized['modality'] == modality
                    if not modality_mask_std.any():
                        continue
                    
                    pca_coords_modality = uns_data_standardized.loc[modality_mask_std].values[:, :num_PCs]
                    sev_levels_modality = obs_standardized.loc[modality_mask_std, sev_col].values
                    
                    plot_path = os.path.join(resolution_dir, f"cca_plot_{modality.lower()}_{dr_type}_res_{resolution:.3f}.png")
                    cca_score_viz, pc_indices_used, _ = plot_cca_on_2d_pca(
                        pca_coords_full=pca_coords_modality,
                        sev_levels=sev_levels_modality,
                        auto_select_best_2pc=True,
                        pc_indices=None,
                        output_path=plot_path,
                        sample_labels=None,
                        title_suffix=f"{modality} {dr_type.capitalize()} - Resolution {resolution:.3f}",
                        verbose=False,
                        create_contribution_plot=True
                    )
                    
                    if verbose:
                        print(f"  {modality} visualization: PC{pc_indices_used[0]+1} + PC{pc_indices_used[1]+1}")
                        
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Failed to create {modality} CCA visualization: {str(e)}")
            
            # Step 5: Generate null distribution if requested
            if compute_corrected_pvalues:
                try:
                    obs_standardized = pseudobulk_adata.obs.copy()
                    obs_standardized.index = obs_standardized.index.str.lower()
                    modality_mask = obs_standardized['modality'] == optimization_target.upper()
                    
                    if modality_mask.any():
                        null_dist = generate_null_distribution(
                            pseudobulk_adata=pseudobulk_adata,
                            column=dr_column,
                            trajectory_col=sev_col,
                            n_pcs=num_PCs,
                            save_path=os.path.join(resolution_dir, f"null_dist_{resolution:.3f}.npy"),
                            verbose=False,
                        )
                        null_result['null_scores'] = null_dist
                        
                        target_score = result_dict.get(f'{optimization_target}_cca_{dr_type}', np.nan)
                        if not np.isnan(target_score):
                            result_dict['p_value'] = np.mean(null_dist >= target_score)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Failed to generate null distribution: {e}")
            
            # Set optimization score
            target_metric = f"{optimization_target}_cca_{dr_type}"
            result_dict['optimization_score'] = result_dict.get(target_metric, np.nan)
            
            # Save results
            pd.DataFrame([result_dict]).to_csv(
                os.path.join(resolution_dir, f"results_{optimization_target}_{dr_type}_res_{resolution:.3f}.csv"),
                index=False
            )
                
        except Exception as e:
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return result_dict, null_result
    
    # === COARSE SEARCH ===
    if verbose:
        print("\n" + "=" * 60)
        print(f"PASS 1: Coarse Search ({coarse_start} - {coarse_end}, step {coarse_step})")
        print("=" * 60)
    
    for resolution in np.arange(coarse_start, coarse_end + coarse_step / 2, coarse_step):
        resolution = round(resolution, 3)
        result_dict, null_result = process_resolution(resolution, 'coarse')
        all_results.append(result_dict)
        all_null_results.append(null_result)
    
    # Find best from coarse search
    coarse_results = [r for r in all_results if not np.isnan(r['optimization_score'])]
    if not coarse_results:
        raise ValueError("No valid optimization scores in coarse search")
    
    best_coarse = max(coarse_results, key=lambda x: x['optimization_score'])
    best_resolution = best_coarse['resolution']
    
    if verbose:
        print(f"\nBest coarse resolution: {best_resolution:.2f}")
        print(f"Best {optimization_target.upper()} {dr_type} CCA: {best_coarse['optimization_score']:.4f}")
    
    # === FINE SEARCH ===
    if verbose:
        print("\n" + "=" * 60)
        print(f"PASS 2: Fine Search (±{fine_range} around best, step {fine_step})")
        print("=" * 60)
    
    search_range_start = max(coarse_start, best_resolution - fine_range)
    search_range_end = min(coarse_end, best_resolution + fine_range)
    
    if verbose:
        print(f"Search range: {search_range_start:.2f} to {search_range_end:.2f}")
    
    for resolution in np.arange(search_range_start, search_range_end + fine_step / 2, fine_step):
        resolution = round(resolution, 3)
        if any(abs(r['resolution'] - resolution) < 0.001 for r in all_results):
            continue
        
        result_dict, null_result = process_resolution(resolution, 'fine')
        all_results.append(result_dict)
        all_null_results.append(null_result)
    
    # Create results dataframe
    df_results = pd.DataFrame(all_results).sort_values("resolution")
    
    # === CORRECTED P-VALUES ===
    if compute_corrected_pvalues:
        valid_nulls = [r for r in all_null_results if r['null_scores'] is not None]
        
        if valid_nulls:
            if verbose:
                print("\n" + "=" * 60)
                print("GENERATING CORRECTED NULL DISTRIBUTION")
                print("=" * 60)
            
            corrected_null = generate_corrected_null_distribution(
                all_resolution_results=valid_nulls,
            )
            
            corrected_null_dir = os.path.join(main_output_dir, "corrected_null")
            os.makedirs(corrected_null_dir, exist_ok=True)
            np.save(os.path.join(corrected_null_dir, f"corrected_null_{dr_type}.npy"), corrected_null)
            
            for idx, row in df_results.iterrows():
                if not np.isnan(row['optimization_score']):
                    df_results.loc[idx, 'corrected_pvalue'] = np.mean(corrected_null >= row['optimization_score'])
    
    # === FIND FINAL BEST RESOLUTION ===
    valid_results = df_results[~df_results['optimization_score'].isna()]
    if valid_results.empty:
        raise ValueError("No valid results obtained")
    
    final_best_idx = valid_results['optimization_score'].idxmax()
    final_best_resolution = valid_results.loc[final_best_idx, 'resolution']
    final_best_score = valid_results.loc[final_best_idx, 'optimization_score']
    
    # === VISUALIZATIONS ===
    if verbose:
        print("\n--- Creating visualizations ---")
    
    plot_cca_vs_resolution_multiomics(
        df_results=df_results,
        optimization_target=optimization_target,
        dr_type=dr_type,
        best_resolution=final_best_resolution,
        output_dir=summary_dir
    )
    
    # === MODALITY ALIGNMENT ANALYSIS ===
    modality_correlation_results = None
    if analyze_modality_alignment:
        if verbose:
            print("\n" + "=" * 60)
            print("MODALITY ALIGNMENT ANALYSIS")
            print("=" * 60)
        modality_correlation_results = analyze_modality_correlation_multiomics(
            df_results=df_results,
            dr_type=dr_type,
            optimization_target=optimization_target,
            output_dir=main_output_dir,
            verbose=verbose
        )
    
    # === PRINT FINAL RESULTS ===
    if verbose:
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"\nOptimization Target: {optimization_target.upper()} {dr_type.upper()}")
        print(f"Best Resolution: {final_best_resolution:.3f}")
        print(f"Best CCA Score: {final_best_score:.4f}")
        print(f"Number of Clusters: {valid_results.loc[final_best_idx, 'n_clusters']}")
        
        best_row = valid_results.loc[final_best_idx]
        rna_score = best_row.get(f'rna_cca_{dr_type}', np.nan)
        atac_score = best_row.get(f'atac_cca_{dr_type}', np.nan)
        
        print(f"\nCCA scores at resolution {final_best_resolution:.3f} ({dr_type}):")
        print(f"  RNA:  {rna_score:.4f}" if not np.isnan(rna_score) else "  RNA:  N/A")
        print(f"  ATAC: {atac_score:.4f}" if not np.isnan(atac_score) else "  ATAC: N/A")
        
        if not np.isnan(best_row.get('p_value', np.nan)):
            print(f"  Standard p-value: {best_row['p_value']:.4f}")
        if compute_corrected_pvalues and not np.isnan(best_row.get('corrected_pvalue', np.nan)):
            print(f"  Corrected p-value: {best_row['corrected_pvalue']:.4f}")
        
        if modality_correlation_results:
            print(f"\nModality Alignment ({dr_type}):")
            print(f"  Pearson r = {modality_correlation_results['pearson_r']:.4f}")
            print(f"  {modality_correlation_results['interpretation']}")
    
    # === SAVE RESULTS ===
    results_csv = os.path.join(summary_dir, f"resolution_results_{optimization_target}_{dr_type}.csv")
    df_results.to_csv(results_csv, index=False)
    
    create_resolution_optimization_summary_multiomics(
        df_results=df_results,
        final_best_resolution=final_best_resolution,
        optimization_target=optimization_target,
        dr_type=dr_type,
        summary_dir=summary_dir
    )
    
    # Copy optimal pseudobulk
    optimal_res_dir = os.path.join(resolutions_dir, f"resolution_{final_best_resolution:.3f}_{dr_type}")
    optimal_pb_path = os.path.join(optimal_res_dir, "pseudobulk", "pseudobulk_sample.h5ad")
    if os.path.exists(optimal_pb_path):
        shutil.copy2(optimal_pb_path, os.path.join(summary_dir, f"optimal_{optimization_target}_{dr_type}.h5ad"))
    
    # === FINAL SUMMARY FILE ===
    _write_final_summary_multiomics(
        main_output_dir=main_output_dir,
        optimization_target=optimization_target,
        dr_type=dr_type,
        use_rep=use_rep,
        n_expression_components=n_expression_components,
        n_proportion_components=n_proportion_components,
        sample_hvg_number=sample_hvg_number,
        batch_col=batch_col,
        final_best_resolution=final_best_resolution,
        final_best_score=final_best_score,
        valid_results=valid_results,
        final_best_idx=final_best_idx,
        modality_correlation_results=modality_correlation_results,
        compute_corrected_pvalues=compute_corrected_pvalues,
        version="GPU/Linux",
    )
    
    if verbose:
        print(f"\nResults saved to: {results_csv}")
        print("\n" + "=" * 80)
    
    return final_best_resolution, df_results


def _write_final_summary_multiomics(
    main_output_dir: str,
    optimization_target: str,
    dr_type: str,
    use_rep: str,
    n_expression_components: int,
    n_proportion_components: int,
    sample_hvg_number: int,
    batch_col: List[str],
    final_best_resolution: float,
    final_best_score: float,
    valid_results: pd.DataFrame,
    final_best_idx: int,
    modality_correlation_results: Optional[dict],
    compute_corrected_pvalues: bool,
    version: str = "GPU/Linux",
) -> None:
    """Write final summary report to file."""
    final_summary_path = os.path.join(main_output_dir, f"FINAL_SUMMARY_{optimization_target}_{dr_type}.txt")
    
    best_row = valid_results.loc[final_best_idx]
    rna_score = best_row.get(f'rna_cca_{dr_type}', np.nan)
    atac_score = best_row.get(f'atac_cca_{dr_type}', np.nan)
    
    with open(final_summary_path, 'w') as f:
        f.write(f"MULTI-OMICS RESOLUTION OPTIMIZATION SUMMARY ({version})\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Analysis completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Optimization target: {optimization_target.upper()} {dr_type.upper()}\n")
        f.write(f"  Representation: {use_rep}\n")
        f.write(f"  Expression components: {n_expression_components}\n")
        f.write(f"  Proportion components: {n_proportion_components}\n")
        f.write(f"  HVG features: {sample_hvg_number}\n")
        f.write(f"  Cell type method: cell_types_multiomics ({version})\n")
        f.write(f"  Sample embedding: calculate_multiomics_sample_embedding ({version})\n")
        if batch_col:
            f.write(f"  Batch columns: {batch_col}\n")
        
        f.write(f"\nRESULTS:\n")
        f.write(f"  Optimal resolution: {final_best_resolution:.3f}\n")
        f.write(f"  Best CCA score: {final_best_score:.4f}\n")
        f.write(f"  Number of clusters: {best_row['n_clusters']}\n\n")
        
        f.write(f"CCA SCORES AT OPTIMAL RESOLUTION ({dr_type.upper()}):\n")
        f.write(f"  RNA:  {rna_score:.4f}\n" if not np.isnan(rna_score) else "  RNA:  N/A\n")
        f.write(f"  ATAC: {atac_score:.4f}\n" if not np.isnan(atac_score) else "  ATAC: N/A\n")
        
        if not np.isnan(best_row.get('p_value', np.nan)):
            f.write(f"  Standard p-value: {best_row['p_value']:.4f}\n")
        if compute_corrected_pvalues and not np.isnan(best_row.get('corrected_pvalue', np.nan)):
            f.write(f"  Corrected p-value: {best_row['corrected_pvalue']:.4f}\n")
        
        if modality_correlation_results:
            f.write(f"\nMODALITY ALIGNMENT:\n")
            f.write(f"  Pearson r: {modality_correlation_results['pearson_r']:.4f}\n")
            f.write(f"  Interpretation: {modality_correlation_results['interpretation']}\n\n")
        
        f.write("SEARCH SUMMARY:\n")
        f.write(f"  Total resolutions: {len(valid_results)}\n")
        f.write(f"  Coarse pass: {len(valid_results[valid_results['pass'] == 'coarse'])}\n")
        f.write(f"  Fine pass: {len(valid_results[valid_results['pass'] == 'fine'])}\n")
    
    print(f"Final summary: {final_summary_path}")