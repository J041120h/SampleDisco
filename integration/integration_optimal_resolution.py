"""
Multi-omics Optimal Resolution Finding Module

This module finds the optimal clustering resolution for integrated RNA+ATAC single-cell data
by maximizing CCA (Canonical Correlation Analysis) correlation between dimension reduction
embeddings and severity levels.

Key Features:
- Two-pass search strategy: coarse (0.1 steps) → fine (0.01 steps around best)
- Supports optimization on either RNA or ATAC modality
- Supports either expression or proportion DR type (only selected type is calculated)
- Uses cell_types_multiomics for RNA-based clustering with label transfer to ATAC
- CCA visualization plots with automatic best PC selection for both RNA and ATAC
- Optional modality correlation analysis to check if RNA and ATAC scores move together
- Comprehensive visualization and summary outputs

Author: Harry
"""

import anndata as ad
import pandas as pd
import numpy as np
import os
from anndata import AnnData
from typing import Optional, Union, List, Tuple
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.integration_cell_type import cell_types_multiomics
from integration.integration_visualization import *
from DR import dimension_reduction
from CCA import *
from CCA_test import *


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def suppress_warnings():
    """
    Suppress specific warnings that are expected during CCA analysis.
    These warnings occur due to numerical edge cases but don't affect results.
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
    
    This prevents pandas categorical errors during data manipulation,
    especially when merging or filtering data across different operations.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object to modify
    columns : list
        List of column names to check and convert. Can contain nested lists.
        
    Returns:
    --------
    AnnData
        Modified AnnData object with non-categorical columns
    """
    # Flatten the columns list to handle nested lists (e.g., batch_col could be a list)
    flat_columns = []
    for col in columns:
        if isinstance(col, list):
            flat_columns.extend(col)
        elif col is not None:
            flat_columns.append(col)
    
    # Convert each categorical column to string
    for col in flat_columns:
        if col in adata.obs.columns:
            if pd.api.types.is_categorical_dtype(adata.obs[col]):
                adata.obs[col] = adata.obs[col].astype(str)
    
    return adata


# =============================================================================
# CCA ANALYSIS FUNCTIONS
# =============================================================================

def cca_analysis(
    pseudobulk_adata: AnnData,
    modality: str,
    column: str,
    sev_col: str,
    n_components: int = 2,
    n_pcs: Optional[int] = None
) -> dict:
    """
    Perform CCA analysis between DR embeddings and severity levels for a specific modality.
    
    This function extracts the dimension reduction coordinates for a given modality,
    then computes the canonical correlation with severity levels.
    
    Parameters:
    -----------
    pseudobulk_adata : AnnData
        Pseudobulk AnnData object containing DR results in .uns
    modality : str
        Modality to analyze ('RNA' or 'ATAC')
    column : str
        Column name in .uns containing DR coordinates (e.g., 'X_DR_expression')
    sev_col : str
        Column name in .obs for severity levels
    n_components : int, default 2
        Number of CCA components to compute
    n_pcs : int, optional
        Number of PCs to use for CCA. If None, uses all available.
        
    Returns:
    --------
    dict
        Results dictionary containing:
        - cca_score: absolute correlation value
        - n_samples: number of samples used
        - n_features: number of PC dimensions used
        - n_pcs_used: actual number of PCs used
        - valid: whether analysis succeeded
        - error_message: error description if failed
        - X_weights, Y_weights: CCA weight vectors
        - X_loadings, Y_loadings: CCA loadings (correlations)
    """
    # Initialize result dictionary with default values
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
        'n_pcs_used': 0
    }
    
    try:
        # ---- Input validation ----
        if column not in pseudobulk_adata.uns:
            result['error_message'] = f"Column '{column}' not found in uns"
            return result
        if 'modality' not in pseudobulk_adata.obs.columns:
            result['error_message'] = "Column 'modality' not found in obs"
            return result
        if sev_col not in pseudobulk_adata.obs.columns:
            result['error_message'] = f"Column '{sev_col}' not found in obs"
            return result
        
        # ---- Standardize indices to lowercase ----
        # This handles potential case sensitivity issues in sample names
        obs_standardized = pseudobulk_adata.obs.copy()
        obs_standardized.index = obs_standardized.index.str.lower()
        
        uns_data_standardized = pseudobulk_adata.uns[column].copy()
        uns_data_standardized.index = uns_data_standardized.index.str.lower()
        
        # ---- Extract modality-specific samples ----
        modality_mask = obs_standardized['modality'] == modality
        
        if not modality_mask.any():
            result['error_message'] = f"No samples found for modality: {modality}"
            return result
        
        # Get DR coordinates and severity levels for this modality
        dr_coords_full = uns_data_standardized.loc[modality_mask].copy()
        sev_levels = obs_standardized.loc[modality_mask, sev_col].values
        
        # ---- Determine number of PCs to use ----
        max_pcs = dr_coords_full.shape[1]
        if n_pcs is None:
            n_pcs_to_use = max_pcs  # Use all available
        else:
            n_pcs_to_use = min(n_pcs, max_pcs)  # Don't exceed available
        
        # Subset to specified number of PCs
        dr_coords = dr_coords_full.iloc[:, :n_pcs_to_use]
        
        # ---- Basic validation ----
        if len(dr_coords) < 3:
            result['error_message'] = f"Insufficient samples: {len(dr_coords)}"
            return result
        if len(np.unique(sev_levels)) < 2:
            result['error_message'] = "Insufficient severity level variance"
            return result
        
        # ---- Prepare data for CCA ----
        X = dr_coords.values  # PC embeddings [n_samples, n_pcs]
        y = sev_levels.reshape(-1, 1)  # Severity [n_samples, 1]
        
        result['n_samples'] = len(X)
        result['n_features'] = X.shape[1]
        result['n_pcs_used'] = n_pcs_to_use
        
        # ---- Limit components based on data dimensions ----
        # CCA requires n_components <= min(n_features_X, n_features_y, n_samples - 1)
        max_components = min(X.shape[1], y.shape[1], X.shape[0] - 1)
        n_components_actual = min(n_components, max_components)
        
        if n_components_actual < 1:
            result['error_message'] = "Cannot compute CCA components"
            return result
        
        # ---- Standardize data ----
        # CCA requires standardized inputs for proper correlation computation
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        # ---- Fit CCA ----
        cca = CCA(n_components=n_components_actual, max_iter=1000, tol=1e-6)
        cca.fit(X_scaled, y_scaled)
        
        # Transform and compute correlation
        X_c, y_c = cca.transform(X_scaled, y_scaled)
        correlation = np.corrcoef(X_c[:, 0], y_c[:, 0])[0, 1]
        cca_score = abs(correlation)  # Use absolute value
        
        # ---- Store results ----
        result['cca_score'] = cca_score
        result['valid'] = True
        
        # Store weight vectors (define the canonical directions)
        result['X_weights'] = cca.x_weights_.flatten()
        result['Y_weights'] = cca.y_weights_.flatten()
        
        # Store loadings if available
        result['X_loadings'] = cca.x_loadings_ if hasattr(cca, 'x_loadings_') else None
        result['Y_loadings'] = cca.y_loadings_ if hasattr(cca, 'y_loadings_') else None
        
    except Exception as e:
        result['error_message'] = f"CCA failed: {str(e)}"
    
    return result


def batch_cca_analysis(
    pseudobulk_adata: AnnData,
    dr_column: str,
    sev_col: str,
    dr_type: str,
    modalities: Optional[List[str]] = None,
    n_components: int = 2,
    n_pcs: Optional[int] = None,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Run CCA analysis across modalities for a single DR column.
    
    Parameters:
    -----------
    pseudobulk_adata : AnnData
        Pseudobulk AnnData with DR results
    dr_column : str
        DR column name to analyze (e.g., 'X_DR_expression')
    sev_col : str
        Severity column name
    dr_type : str
        DR type for file naming ('expression' or 'proportion')
    modalities : list, optional
        Modalities to analyze. If None, uses all unique modalities.
    n_components : int, default 2
        Number of CCA components
    n_pcs : int, optional
        Number of PCs to use
    output_dir : str, optional
        Directory to save results
        
    Returns:
    --------
    pd.DataFrame
        Results with CCA scores and metadata for all modalities
    """
    import pickle
    
    # Get modalities if not specified
    if modalities is None:
        modalities = pseudobulk_adata.obs['modality'].unique()
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # ---- Iterate through modalities for the specified DR column ----
    for modality in modalities:
        if dr_column in pseudobulk_adata.uns:
            # Run CCA analysis for this modality
            result = cca_analysis(
                pseudobulk_adata=pseudobulk_adata,
                modality=modality,
                column=dr_column,
                sev_col=sev_col,
                n_components=n_components,
                n_pcs=n_pcs
            )
            results.append(result)
        else:
            # Column not found - add placeholder result
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
                'n_pcs_used': 0
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # ---- Save results if output directory specified ----
    if output_dir:
        # Save summary (human-readable CSV) - include dr_type in filename
        summary_df = results_df[['modality', 'column', 'cca_score', 'n_samples', 
                                'n_features', 'n_pcs_used', 'valid', 'error_message']]
        summary_path = os.path.join(output_dir, f'cca_results_{dr_type}_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Save complete results with vectors as pickle
        complete_path = os.path.join(output_dir, f'cca_results_{dr_type}_complete.pkl')
        with open(complete_path, 'wb') as f:
            pickle.dump(results_df, f)
    
    return results_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_cca_vs_resolution(
    df_results: pd.DataFrame,
    optimization_target: str,
    dr_type: str,
    best_resolution: float,
    output_dir: str
) -> None:
    """
    Create elegant visualization of CCA scores across resolutions.
    
    Creates a plot showing both modalities (RNA and ATAC) for the specified DR type.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Results with CCA scores for all resolutions
    optimization_target : str
        Target modality ('rna' or 'atac')
    dr_type : str
        Target DR type ('expression' or 'proportion')
    best_resolution : float
        Optimal resolution found
    output_dir : str
        Output directory for plots
    """
    # Sort by resolution
    df_sorted = df_results.sort_values('resolution').copy()
    resolutions = df_sorted['resolution'].values
    
    # Define colors
    colors = {'rna': '#2E86AB', 'atac': '#A23B72'}
    
    # ========== Main comparison plot ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot both modalities for the target DR type
    best_score = None
    for modality in ['rna', 'atac']:
        cca_col = f'{modality}_cca_{dr_type}'
        if cca_col not in df_sorted.columns:
            continue
        cca_values = df_sorted[cca_col].values
        
        # Determine line style based on optimization target
        linewidth = 2.5 if modality == optimization_target else 1.5
        alpha = 1.0 if modality == optimization_target else 0.7
        
        ax.plot(resolutions, cca_values, 
               color=colors[modality], 
               linewidth=linewidth, 
               alpha=alpha,
               marker='o', 
               markersize=6 if modality == optimization_target else 4,
               label=f'{modality.upper()}')
        
        # Get best score for optimization target
        if modality == optimization_target:
            best_score = df_sorted.loc[
                df_sorted['resolution'] == best_resolution, cca_col
            ].iloc[0]
    
    # Mark best resolution with vertical line
    ax.axvline(x=best_resolution, color='#E74C3C', linestyle='--', 
              linewidth=2, alpha=0.8, label=f'Best: {best_resolution:.3f}')
    
    # Mark best point with larger marker
    if best_score is not None:
        ax.scatter([best_resolution], [best_score], 
                  color='#E74C3C', s=150, zorder=5, edgecolors='white', linewidth=2)
    
    # Styling
    ax.set_xlabel('Resolution', fontsize=12, fontweight='medium')
    ax.set_ylabel('CCA Score', fontsize=12, fontweight='medium')
    ax.set_title(f'CCA Score vs Resolution\nTarget: {optimization_target.upper()} {dr_type.capitalize()}', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#FAFAFA')
    ax.set_xlim(resolutions.min() - 0.02, resolutions.max() + 0.02)
    
    plt.tight_layout()
    # Include both optimization_target and dr_type in filename
    plt.savefig(os.path.join(output_dir, f'cca_vs_resolution_{optimization_target}_{dr_type}.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def analyze_modality_correlation(
    df_results: pd.DataFrame,
    dr_type: str,
    optimization_target: str,
    output_dir: str,
    verbose: bool = True
) -> dict:
    """
    Analyze correlation between RNA and ATAC CCA scores across resolutions.
    
    Checks whether improving RNA CCA score also improves ATAC CCA score.
    High positive correlation → modalities are aligned
    Negative correlation → trade-off between modalities
    """
    from scipy import stats
    
    corr_dir = os.path.join(output_dir, "modality_correlation")
    os.makedirs(corr_dir, exist_ok=True)
    
    rna_col = f'rna_cca_{dr_type}'
    atac_col = f'atac_cca_{dr_type}'
    
    # Check if columns exist
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
    
    # Compute correlations
    pearson_r, pearson_p = stats.pearsonr(rna_scores, atac_scores)
    spearman_r, spearman_p = stats.spearmanr(rna_scores, atac_scores)
    
    # Interpretation
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
    
    # Scatter plot
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
    
    # Dual line plot
    ax2 = axes[1]
    ax2.plot(resolutions, rna_scores, 'o-', color='#2E86AB', linewidth=2, markersize=6, label='RNA')
    ax2.plot(resolutions, atac_scores, 's-', color='#A23B72', linewidth=2, markersize=6, label='ATAC')
    ax2.set_xlabel('Resolution', fontsize=11)
    ax2.set_ylabel('CCA Score', fontsize=11)
    ax2.set_title(f'{dr_type.capitalize()} DR Scores', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Include optimization_target and dr_type in filename
    plt.savefig(os.path.join(corr_dir, f'modality_correlation_{optimization_target}_{dr_type}.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save text results
    with open(os.path.join(corr_dir, f'correlation_results_{optimization_target}_{dr_type}.txt'), 'w') as f:
        f.write(f"Modality Correlation Analysis - {optimization_target.upper()} {dr_type.upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Pearson r:  {pearson_r:.4f} (p = {pearson_p:.4f})\n")
        f.write(f"Spearman r: {spearman_r:.4f} (p = {spearman_p:.4f})\n\n")
        f.write(f"Interpretation: {interpretation}\n")
    
    return results


def create_resolution_optimization_summary(
    df_results: pd.DataFrame,
    final_best_resolution: float,
    optimization_target: str,
    dr_type: str,
    summary_dir: str
) -> None:
    """
    Create summary text files for the selected modality × DR combination.
    Only creates files for the specified dr_type.
    """
    modalities = ['rna', 'atac']
    df_sorted = df_results.sort_values('resolution').copy()
    
    for modality in modalities:
        cca_col = f'{modality}_cca_{dr_type}'
        
        if cca_col not in df_sorted.columns:
            continue
        
        # Include both modality and dr_type in filename
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
# MAIN RESOLUTION OPTIMIZATION FUNCTION
# =============================================================================

def find_optimal_cell_resolution_integration(
    AnnData_integrated: AnnData,
    output_dir: str,
    optimization_target: str = "rna",
    dr_type: str = "expression",
    n_features: int = 40000,
    sev_col: str = "sev.level",
    batch_col: Optional[Union[str, List[str]]] = None,
    sample_col: str = "sample",
    modality_col: str = "modality",
    use_rep: str = 'X_glue',
    num_DR_components: int = 30,
    num_PCs: int = 20,
    n_pcs: int = 10,
    visualize_cell_types: bool = True,
    analyze_modality_alignment: bool = True,
    preserve_cols: Optional[Union[str, List[str]]] = None,
    verbose: bool = True
) -> Tuple[float, pd.DataFrame]:
    """
    Find optimal clustering resolution for integrated RNA+ATAC data.
    
    Performs two-pass search to find the resolution that maximizes CCA correlation
    between DR embeddings and severity levels for a specified modality and DR type.
    
    Uses cell_types_multiomics for clustering (RNA-based clustering with label transfer to ATAC).
    Only calculates and saves files for the selected dr_type (expression OR proportion).
    
    Search Strategy:
    ----------------
    Pass 1 (Coarse): Test resolutions 0.1, 0.2, ..., 1.0
    Pass 2 (Fine): Test resolutions best ± 0.05 in 0.01 steps
    
    Parameters:
    -----------
    AnnData_integrated : AnnData
        Integrated AnnData with RNA and ATAC data
    output_dir : str
        Output directory
    optimization_target : str, default "rna"
        Which modality to optimize: "rna" or "atac"
    dr_type : str, default "expression"
        Which DR type to optimize: "expression" or "proportion"
    n_features : int, default 40000
        Number of HVG features for pseudobulk
    sev_col : str, default "sev.level"
        Severity column name
    batch_col : str or list or None, default None
        Batch column(s) for correction
    sample_col : str, default "sample"
        Sample identifier column
    modality_col : str, default "modality"
        Modality label column
    use_rep : str, default 'X_glue'
        Representation for clustering
    num_DR_components : int, default 30
        Number of DR components
    num_PCs : int, default 20
        Number of PCs for clustering
    n_pcs : int, default 10
        Number of PCs for CCA analysis
    visualize_cell_types : bool, default True
        Whether to create cell type UMAP visualizations at each resolution
        (uses cell_types_multiomics built-in visualization)
    analyze_modality_alignment : bool, default True
        Whether to analyze RNA/ATAC score correlation
    preserve_cols : str or list, optional
        Columns to preserve through pseudobulk computation
    verbose : bool, default True
        Print verbose output
        
    Returns:
    --------
    tuple
        (optimal_resolution, results_dataframe)
    """
    
    print("\n" + "=" * 80)
    print("MULTI-OMICS RESOLUTION OPTIMIZATION")
    print("=" * 80)
    
    # ========== INPUT VALIDATION ==========
    if optimization_target not in ["rna", "atac"]:
        raise ValueError("optimization_target must be 'rna' or 'atac'")
    if dr_type not in ["expression", "proportion"]:
        raise ValueError("dr_type must be 'expression' or 'proportion'")
    
    # Convert batch_col to list format
    if isinstance(batch_col, str):
        batch_col = [batch_col]
    elif batch_col is None:
        batch_col = []
    
    # ========== SETUP OUTPUT DIRECTORIES ==========
    # Include both optimization_target and dr_type in directory name
    main_output_dir = os.path.join(output_dir, f"Integration_optimization_{optimization_target}_{dr_type}")
    os.makedirs(main_output_dir, exist_ok=True)
    resolutions_dir = os.path.join(main_output_dir, "resolutions")
    os.makedirs(resolutions_dir, exist_ok=True)
    summary_dir = os.path.join(main_output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # ========== PRINT CONFIGURATION ==========
    print(f"\nConfiguration:")
    print(f"  Optimization target: {optimization_target.upper()} {dr_type.upper()}")
    print(f"  Representation: {use_rep} ({num_PCs} components)")
    print(f"  DR components: {num_DR_components}")
    print(f"  CCA PCs: {n_pcs}")
    print(f"  Cell type method: cell_types_multiomics (RNA clustering + ATAC label transfer)")
    print(f"  Cell type visualization: {visualize_cell_types}")
    if batch_col:
        print(f"  Batch columns: {batch_col}")
    print(f"  Output: {main_output_dir}")
    
    # ========== PREPARE DATA ==========
    columns_to_check = ['cell_type', modality_col, sev_col, sample_col]
    if batch_col:
        columns_to_check.extend(batch_col)
    AnnData_integrated = ensure_non_categorical_columns(AnnData_integrated, columns_to_check)
    
    # ========== STORAGE ==========
    all_results = []
    
    # Define DR column based on dr_type
    dr_column = f'X_DR_{dr_type}'
    
    # ========== RESOLUTION PROCESSING FUNCTION ==========
    def process_resolution(resolution: float, search_pass: str) -> dict:
        """Process a single resolution: cluster → pseudobulk → DR → CCA → Visualize"""
        nonlocal AnnData_integrated
        
        print(f"\n--- Resolution: {resolution:.3f} ({search_pass}) ---")
        
        # Include dr_type in resolution directory name
        resolution_dir = os.path.join(resolutions_dir, f"resolution_{resolution:.3f}_{dr_type}")
        os.makedirs(resolution_dir, exist_ok=True)
        
        # Initialize results - only for the selected dr_type
        result_dict = {
            'resolution': resolution,
            f'rna_cca_{dr_type}': np.nan,
            f'atac_cca_{dr_type}': np.nan,
            'optimization_score': np.nan,
            'pass': search_pass,
            'n_clusters': 0
        }
        
        try:
            # Step 1: Clean up previous clustering
            if 'cell_type' in AnnData_integrated.obs.columns:
                AnnData_integrated.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            if 'label_transfer_confidence' in AnnData_integrated.obs.columns:
                AnnData_integrated.obs.drop(columns=['label_transfer_confidence'], inplace=True, errors='ignore')
            if modality_col in AnnData_integrated.obs.columns:
                AnnData_integrated.obs[modality_col] = AnnData_integrated.obs[modality_col].astype(str)
            
            # Remove UMAP for fresh computation at each resolution
            if 'X_umap' in AnnData_integrated.obsm:
                del AnnData_integrated.obsm['X_umap']
            
            # Step 2: Clustering using cell_types_multiomics
            # This clusters RNA cells and transfers labels to ATAC cells
            # Uses built-in visualization if visualize_cell_types=True
            AnnData_integrated = cell_types_multiomics(
                adata=AnnData_integrated,
                modality_column=modality_col,
                rna_modality_value="RNA",
                atac_modality_value="ATAC",
                cell_type_column='cell_type',
                cluster_resolution=resolution,
                use_rep=use_rep,
                k_neighbors=15,
                transfer_metric="cosine",
                compute_umap=visualize_cell_types,  # Compute UMAP if visualization is requested
                save=False,
                output_dir=resolution_dir if visualize_cell_types else None,  # Save plots to resolution dir
                verbose=False,
                generate_plots=visualize_cell_types  # Generate plots if requested
            )
            n_clusters = AnnData_integrated.obs['cell_type'].nunique()
            result_dict['n_clusters'] = n_clusters
            print(f"  Clusters: {n_clusters}")
            
            # Step 3: Pseudobulk
            from integration.integration_pseudobulk import compute_pseudobulk_adata_linux
            pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata_linux(
                adata=AnnData_integrated, batch_col=batch_col if batch_col else None,
                sample_col=sample_col, celltype_col='cell_type', n_features=n_features,
                output_dir=resolution_dir, save=False, verbose=False, preserve_cols=preserve_cols
            )
            
            # Step 4: Dimension reduction (using dimension_reduction from DR.py)
            dr_result = dimension_reduction(
                adata=AnnData_integrated, pseudobulk=pseudobulk_dict,
                pseudobulk_anndata=pseudobulk_adata, sample_col=sample_col,
                n_expression_components=num_DR_components, n_proportion_components=num_DR_components,
                batch_col=batch_col if batch_col else None, harmony_for_proportion=True, atac=False,
                output_dir=resolution_dir, not_save=True, verbose=False, preserve_cols=preserve_cols
            )
            
            # Step 5: CCA analysis - only for the selected dr_type
            # Import the visualization function from CCA_test
            from CCA_test import run_cca_on_pca_from_adata, plot_cca_on_2d_pca
            
            cca_results_df = batch_cca_analysis(
                pseudobulk_adata=pseudobulk_adata, 
                dr_column=dr_column, 
                sev_col=sev_col,
                dr_type=dr_type,
                modalities=['RNA', 'ATAC'], 
                n_components=1, 
                n_pcs=n_pcs, 
                output_dir=resolution_dir
            )
            
            for _, row in cca_results_df.iterrows():
                modality = row['modality'].lower()
                result_dict[f'{modality}_cca_{dr_type}'] = row['cca_score']
                if row['valid'] and not np.isnan(row['cca_score']):
                    print(f"  {row['modality']} {dr_type}: {row['cca_score']:.4f}")
            
            # Step 6: Create CCA visualization plots for each modality
            for modality in ['RNA', 'ATAC']:
                try:
                    # Standardize indices to lowercase for consistent matching
                    obs_standardized = pseudobulk_adata.obs.copy()
                    obs_standardized.index = obs_standardized.index.str.lower()
                    
                    uns_data_standardized = pseudobulk_adata.uns[dr_column].copy()
                    uns_data_standardized.index = uns_data_standardized.index.str.lower()
                    
                    # Filter to only the current modality
                    modality_mask_std = obs_standardized['modality'] == modality
                    if not modality_mask_std.any():
                        continue
                    
                    # Get modality-specific coordinates (use n_pcs dimensions)
                    pca_coords_modality = uns_data_standardized.loc[modality_mask_std].values[:, :n_pcs]
                    sev_levels_modality = obs_standardized.loc[modality_mask_std, sev_col].values
                    samples_modality = obs_standardized.loc[modality_mask_std].index.values
                    
                    # Create CCA visualization plot with automatic best PC selection
                    plot_path = os.path.join(resolution_dir, f"cca_plot_{modality.lower()}_{dr_type}_res_{resolution:.3f}.png")
                    cca_score_viz, pc_indices_used, cca_model_viz = plot_cca_on_2d_pca(
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
                        print(f"  {modality} visualization: PC{pc_indices_used[0]+1} + PC{pc_indices_used[1]+1} (viz score: {cca_score_viz:.4f})")
                        
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Failed to create {modality} CCA visualization: {str(e)}")
            
            # Set optimization score
            target_metric = f"{optimization_target}_cca_{dr_type}"
            result_dict['optimization_score'] = result_dict.get(target_metric, np.nan)
            
            # Save results - include dr_type in filename
            pd.DataFrame([result_dict]).to_csv(
                os.path.join(resolution_dir, f"results_{optimization_target}_{dr_type}_res_{resolution:.3f}.csv"), 
                index=False)
            pseudobulk_adata.write_h5ad(
                os.path.join(resolution_dir, f"pseudobulk_{dr_type}.h5ad"))
                
        except Exception as e:
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return result_dict
    
    # ========== PASS 1: COARSE SEARCH ==========
    print("\n" + "=" * 60)
    print("PASS 1: Coarse Search (0.1 - 1.0, step 0.1)")
    print("=" * 60)
    
    for resolution in np.arange(0.1, 1.01, 0.1):
        result_dict = process_resolution(resolution, 'coarse')
        all_results.append(result_dict)
    
    # Find best from coarse search
    coarse_results = [r for r in all_results if not np.isnan(r['optimization_score'])]
    if not coarse_results:
        raise ValueError("No valid optimization scores in coarse search")
    
    best_coarse = max(coarse_results, key=lambda x: x['optimization_score'])
    best_resolution = best_coarse['resolution']
    
    print(f"\nBest coarse resolution: {best_resolution:.2f}")
    print(f"Best {optimization_target.upper()} {dr_type} CCA: {best_coarse['optimization_score']:.4f}")
    
    # Print other modality score
    other_modality = 'atac' if optimization_target == 'rna' else 'rna'
    other_score = best_coarse.get(f'{other_modality}_cca_{dr_type}', np.nan)
    if not np.isnan(other_score):
        print(f"     {other_modality.upper()} {dr_type} CCA: {other_score:.4f}")
    
    # ========== PASS 2: FINE SEARCH ==========
    print("\n" + "=" * 60)
    print("PASS 2: Fine Search (±0.05 around best, step 0.01)")
    print("=" * 60)
    
    search_range_start = max(0.01, best_resolution - 0.05)
    search_range_end = min(1.00, best_resolution + 0.05)
    print(f"Search range: {search_range_start:.2f} to {search_range_end:.2f}")
    
    for resolution in np.arange(search_range_start, search_range_end + 0.001, 0.01):
        resolution = round(resolution, 3)
        if any(abs(r['resolution'] - resolution) < 0.001 for r in all_results):
            continue
        
        result_dict = process_resolution(resolution, 'fine')
        all_results.append(result_dict)
    
    # ========== CREATE RESULTS DATAFRAME ==========
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("resolution")
    
    # ========== FIND FINAL BEST RESOLUTION ==========
    valid_results = df_results[~df_results['optimization_score'].isna()]
    if valid_results.empty:
        raise ValueError("No valid results obtained")
    
    final_best_idx = valid_results['optimization_score'].idxmax()
    final_best_resolution = valid_results.loc[final_best_idx, 'resolution']
    final_best_score = valid_results.loc[final_best_idx, 'optimization_score']
    
    # ========== CCA VISUALIZATION ==========
    print("\n--- Creating CCA visualizations ---")
    plot_cca_vs_resolution(
        df_results=df_results, optimization_target=optimization_target,
        dr_type=dr_type, best_resolution=final_best_resolution, output_dir=summary_dir
    )
    
    # ========== MODALITY ALIGNMENT ANALYSIS ==========
    modality_correlation_results = None
    if analyze_modality_alignment:
        print("\n" + "=" * 60)
        print("MODALITY ALIGNMENT ANALYSIS")
        print("=" * 60)
        modality_correlation_results = analyze_modality_correlation(
            df_results=df_results, dr_type=dr_type, 
            optimization_target=optimization_target,
            output_dir=main_output_dir, verbose=verbose
        )
    
    # ========== PRINT FINAL RESULTS ==========
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nOptimization Target: {optimization_target.upper()} {dr_type.upper()}")
    print(f"Best Resolution: {final_best_resolution:.3f}")
    print(f"Best CCA Score: {final_best_score:.4f}")
    print(f"Number of Clusters: {valid_results.loc[final_best_idx, 'n_clusters']}")
    
    # All scores at best resolution (only for selected dr_type)
    best_row = valid_results.loc[final_best_idx]
    rna_score = best_row.get(f'rna_cca_{dr_type}', np.nan)
    atac_score = best_row.get(f'atac_cca_{dr_type}', np.nan)
    
    print(f"\nCCA scores at resolution {final_best_resolution:.3f} ({dr_type}):")
    print(f"  RNA:  {rna_score:.4f}" if not np.isnan(rna_score) else "  RNA:  N/A")
    print(f"  ATAC: {atac_score:.4f}" if not np.isnan(atac_score) else "  ATAC: N/A")
    
    if modality_correlation_results:
        print(f"\nModality Alignment ({dr_type}):")
        print(f"  Pearson r = {modality_correlation_results['pearson_r']:.4f}")
        print(f"  {modality_correlation_results['interpretation']}")
    
    # ========== SAVE RESULTS ==========
    # Include optimization_target and dr_type in filename
    results_csv = os.path.join(summary_dir, f"resolution_results_{optimization_target}_{dr_type}.csv")
    df_results.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")
    
    create_resolution_optimization_summary(
        df_results=df_results, final_best_resolution=final_best_resolution,
        optimization_target=optimization_target, dr_type=dr_type, summary_dir=summary_dir
    )
    
    # Copy optimal pseudobulk - include dr_type in filename
    optimal_res_dir = os.path.join(resolutions_dir, f"resolution_{final_best_resolution:.3f}_{dr_type}")
    optimal_pb_path = os.path.join(optimal_res_dir, f"pseudobulk_{dr_type}.h5ad")
    if os.path.exists(optimal_pb_path):
        import shutil
        shutil.copy2(optimal_pb_path, os.path.join(summary_dir, f"optimal_{optimization_target}_{dr_type}.h5ad"))
        print(f"Optimal pseudobulk saved to: {os.path.join(summary_dir, f'optimal_{optimization_target}_{dr_type}.h5ad')}")
    
    # ========== FINAL SUMMARY FILE ==========
    final_summary_path = os.path.join(main_output_dir, f"FINAL_SUMMARY_{optimization_target}_{dr_type}.txt")
    with open(final_summary_path, 'w') as f:
        f.write("MULTI-OMICS RESOLUTION OPTIMIZATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("CONFIGURATION:\n")
        f.write(f"  Optimization target: {optimization_target.upper()} {dr_type.upper()}\n")
        f.write(f"  Representation: {use_rep}\n")
        f.write(f"  Clustering PCs: {num_PCs}\n")
        f.write(f"  DR components: {num_DR_components}\n")
        f.write(f"  CCA PCs: {n_pcs}\n")
        f.write(f"  Features: {n_features}\n")
        f.write(f"  Cell type method: cell_types_multiomics\n")
        if batch_col:
            f.write(f"  Batch columns: {batch_col}\n")
        f.write(f"\nRESULTS:\n")
        f.write(f"  Optimal resolution: {final_best_resolution:.3f}\n")
        f.write(f"  Best CCA score: {final_best_score:.4f}\n")
        f.write(f"  Number of clusters: {valid_results.loc[final_best_idx, 'n_clusters']}\n\n")
        f.write(f"CCA SCORES AT OPTIMAL RESOLUTION ({dr_type.upper()}):\n")
        f.write(f"  RNA:  {rna_score:.4f}\n" if not np.isnan(rna_score) else "  RNA:  N/A\n")
        f.write(f"  ATAC: {atac_score:.4f}\n" if not np.isnan(atac_score) else "  ATAC: N/A\n")
        if modality_correlation_results:
            f.write(f"\nMODALITY ALIGNMENT:\n")
            f.write(f"  Pearson r: {modality_correlation_results['pearson_r']:.4f}\n")
            f.write(f"  Interpretation: {modality_correlation_results['interpretation']}\n\n")
        f.write("SEARCH SUMMARY:\n")
        f.write(f"  Total resolutions: {len(valid_results)}\n")
        f.write(f"  Coarse pass: {len(valid_results[valid_results['pass'] == 'coarse'])}\n")
        f.write(f"  Fine pass: {len(valid_results[valid_results['pass'] == 'fine'])}\n")
    
    print(f"Final summary: {final_summary_path}")
    print("\n" + "=" * 80)
    
    return final_best_resolution, df_results