"""
Multi-omics Optimal Resolution Finding Module

This module finds the optimal clustering resolution for integrated RNA+ATAC single-cell data
by maximizing CCA (Canonical Correlation Analysis) correlation between dimension reduction
embeddings and severity levels.

Key Features:
- Two-pass search strategy: coarse (0.1 steps) → fine (0.01 steps around best)
- Supports optimization on either RNA or ATAC modality
- Supports either expression or proportion DR type
- Optional modality correlation analysis to check if RNA and ATAC scores move together
- Corrected p-values accounting for resolution selection bias
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

from preparation.Cell_type_linux import cell_types_linux
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
    dr_columns: List[str],
    sev_col: str,
    modalities: Optional[List[str]] = None,
    n_components: int = 2,
    n_pcs: Optional[int] = None,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Run CCA analysis across multiple DR columns and modalities.
    
    This is the main batch processing function that iterates through all
    combinations of modalities and DR types.
    
    Parameters:
    -----------
    pseudobulk_adata : AnnData
        Pseudobulk AnnData with DR results
    dr_columns : list
        List of DR column names to analyze (e.g., ['X_DR_expression', 'X_DR_proportion'])
    sev_col : str
        Severity column name
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
        Results with CCA scores and metadata for all combinations
    """
    import pickle
    
    # Get modalities if not specified
    if modalities is None:
        modalities = pseudobulk_adata.obs['modality'].unique()
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # ---- Iterate through all modality × DR column combinations ----
    for modality in modalities:
        for column in dr_columns:
            if column in pseudobulk_adata.uns:
                # Run CCA analysis for this combination
                result = cca_analysis(
                    pseudobulk_adata=pseudobulk_adata,
                    modality=modality,
                    column=column,
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
                    'column': column,
                    'valid': False,
                    'error_message': f"Column '{column}' not found",
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
        # Save summary (human-readable CSV)
        summary_df = results_df[['modality', 'column', 'cca_score', 'n_samples', 
                                'n_features', 'n_pcs_used', 'valid', 'error_message']]
        summary_path = os.path.join(output_dir, 'cca_results_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Save complete results with vectors as pickle
        complete_path = os.path.join(output_dir, 'cca_results_complete.pkl')
        with open(complete_path, 'wb') as f:
            pickle.dump(results_df, f)
    
    return results_df


# =============================================================================
# NULL DISTRIBUTION FUNCTIONS
# =============================================================================

def generate_null_distribution(
    pseudobulk_adata: AnnData,
    modality: str,
    column: str,
    sev_col: str,
    n_permutations: int = 1000,
    n_pcs: Optional[int] = None,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Generate null distribution for CCA score using permutation testing.
    
    Algorithm:
    1. Extract DR coordinates for specified modality
    2. For each permutation:
       a. Randomly shuffle severity labels
       b. Compute CCA between embeddings and shuffled labels
       c. Record the correlation as one null sample
    
    Parameters:
    -----------
    pseudobulk_adata : AnnData
        Pseudobulk AnnData with DR results
    modality : str
        Modality to analyze ('RNA' or 'ATAC')
    column : str
        DR column name in .uns
    sev_col : str
        Severity column name
    n_permutations : int, default 1000
        Number of permutation samples
    n_pcs : int, optional
        Number of PCs to use. If None, uses all.
    save_path : str, optional
        Path to save null distribution as .npy file
    verbose : bool, default True
        Whether to print progress
        
    Returns:
    --------
    np.ndarray
        Array of null CCA scores
    """
    # ---- Extract data for specified modality ----
    modality_mask = pseudobulk_adata.obs['modality'] == modality
    if not modality_mask.any():
        raise ValueError(f"No samples found for modality: {modality}")
    
    # Get DR coordinates and severity levels
    dr_coords_full = pseudobulk_adata.uns[column].loc[modality_mask].copy()
    sev_levels = pseudobulk_adata.obs.loc[modality_mask, sev_col].values
    
    # ---- Determine number of PCs to use ----
    if n_pcs is None:
        dr_coords = dr_coords_full
        n_dims_used = dr_coords_full.shape[1]
    else:
        n_pcs = min(n_pcs, dr_coords_full.shape[1])
        dr_coords = dr_coords_full.iloc[:, :n_pcs]
        n_dims_used = n_pcs
    
    # ---- Validation ----
    if len(dr_coords) < 3:
        raise ValueError(f"Insufficient samples: {len(dr_coords)}")
    if len(np.unique(sev_levels)) < 2:
        raise ValueError("Insufficient severity level variance")
    
    # Prepare data
    X = dr_coords.values  # [n_samples, n_dims]
    y_original = sev_levels.copy()  # [n_samples]
    
    # ---- Run permutations ----
    null_scores = []
    failed_permutations = 0
    
    for perm in range(n_permutations):
        try:
            # Randomly shuffle severity labels
            permuted_sev = np.random.permutation(y_original)
            
            # Standardize data
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_permuted_scaled = scaler_y.fit_transform(permuted_sev.reshape(-1, 1))
            
            # Fit CCA with 1 component
            cca_perm = CCA(n_components=1, max_iter=1000, tol=1e-6)
            cca_perm.fit(X_scaled, y_permuted_scaled)
            
            # Compute correlation
            X_c_perm, y_c_perm = cca_perm.transform(X_scaled, y_permuted_scaled)
            perm_correlation = np.corrcoef(X_c_perm[:, 0], y_c_perm[:, 0])[0, 1]
            
            # Record score (handle invalid values)
            if np.isnan(perm_correlation) or np.isinf(perm_correlation):
                null_scores.append(0.0)
                failed_permutations += 1
            else:
                null_scores.append(abs(perm_correlation))
                
        except Exception:
            null_scores.append(0.0)
            failed_permutations += 1
    
    null_distribution = np.array(null_scores)
    
    # ---- Print status ----
    if verbose:
        success_rate = (n_permutations - failed_permutations) / n_permutations * 100
        print(f"  Null distribution ({modality}): {n_dims_used} PCs, "
              f"{success_rate:.1f}% success ({n_permutations - failed_permutations}/{n_permutations})")
    
    # ---- Save if path specified ----
    if save_path:
        np.save(save_path, null_distribution)
    
    return null_distribution


def generate_corrected_null_distribution_for_modality(
    all_resolution_results: List[dict],
    modality: str,
    dr_type: str,
    n_permutations: int = 1000
) -> np.ndarray:
    """
    Generate corrected null distribution accounting for resolution selection bias.
    
    When we select the "best" resolution, we're picking the maximum CCA score
    across all tested resolutions. This inflates our observed score compared to
    a single null distribution. To correct for this, we:
    
    1. For each permutation index i (from 0 to n_permutations-1)
    2. Collect the null score at index i from ALL resolutions
    3. Take the MAXIMUM across resolutions (mimicking our selection process)
    4. This gives us a "corrected" null score that accounts for multiple testing
    
    Parameters:
    -----------
    all_resolution_results : list
        List of dicts with 'resolution' and 'null_scores' for each resolution
    modality : str
        Target modality ("rna" or "atac")
    dr_type : str
        DR type ("expression" or "proportion")
    n_permutations : int
        Number of permutations
        
    Returns:
    --------
    np.ndarray
        Corrected null distribution
    """
    corrected_null_scores = []
    
    for perm_idx in range(n_permutations):
        # Collect the score at this permutation index from all resolutions
        perm_scores_across_resolutions = []
        
        for resolution_result in all_resolution_results:
            if 'null_scores' in resolution_result and resolution_result['null_scores'] is not None:
                if len(resolution_result['null_scores']) > perm_idx:
                    perm_scores_across_resolutions.append(resolution_result['null_scores'][perm_idx])
        
        # Take maximum (mimics selecting best resolution)
        if perm_scores_across_resolutions:
            max_score_for_this_perm = max(perm_scores_across_resolutions)
            corrected_null_scores.append(max_score_for_this_perm)
    
    return np.array(corrected_null_scores)


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
    
    Creates two plots:
    1. Main plot: Both modalities on same axes with best resolution marked
    2. 2x2 grid: All modality × DR type combinations
    
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
    
    # ========== Plot 1: Main comparison plot ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot both modalities for the target DR type
    for modality in ['rna', 'atac']:
        cca_col = f'{modality}_cca_{dr_type}'
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
    
    # Mark best resolution with vertical line
    ax.axvline(x=best_resolution, color='#E74C3C', linestyle='--', 
              linewidth=2, alpha=0.8, label=f'Best: {best_resolution:.3f}')
    
    # Mark best point with larger marker
    best_score = df_sorted.loc[
        df_sorted['resolution'] == best_resolution, 
        f'{optimization_target}_cca_{dr_type}'
    ].iloc[0]
    ax.scatter([best_resolution], [best_score], 
              color='#E74C3C', s=150, zorder=5, edgecolors='white', linewidth=2)
    
    # Styling
    ax.set_xlabel('Resolution', fontsize=12, fontweight='medium')
    ax.set_ylabel('CCA Score', fontsize=12, fontweight='medium')
    ax.set_title(f'CCA Score vs Resolution ({dr_type.capitalize()} DR)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#FAFAFA')
    ax.set_xlim(resolutions.min() - 0.02, resolutions.max() + 0.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cca_vs_resolution_{dr_type}.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # ========== Plot 2: 2x2 grid for all combinations ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    modalities = ['rna', 'atac']
    dr_types = ['expression', 'proportion']
    
    for i, modality in enumerate(modalities):
        for j, dr in enumerate(dr_types):
            ax = axes[i, j]
            
            cca_col = f'{modality}_cca_{dr}'
            cca_values = df_sorted[cca_col].values
            
            # Check if this is the optimization target
            is_target = (modality == optimization_target and dr == dr_type)
            
            # Plot line with fill
            color = colors[modality]
            ax.plot(resolutions, cca_values, 
                   color=color, linewidth=2, marker='o', markersize=5)
            ax.fill_between(resolutions, cca_values, alpha=0.1, color=color)
            
            # Mark best resolution if this is the target
            if is_target:
                ax.axvline(x=best_resolution, color='#E74C3C', linestyle='--', 
                          linewidth=2, alpha=0.8)
                ax.scatter([best_resolution], [best_score], 
                          color='#E74C3C', s=100, zorder=5, edgecolors='white', linewidth=2)
                ax.set_facecolor('#FFF9F0')
            else:
                ax.set_facecolor('#FAFAFA')
            
            ax.set_xlabel('Resolution', fontsize=10)
            ax.set_ylabel('CCA Score', fontsize=10)
            ax.set_title(f'{modality.upper()} - {dr.capitalize()}', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('CCA Scores Across All Modality × DR Combinations', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cca_vs_resolution_all.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def compute_all_corrected_pvalues_and_plots(
    df_results: pd.DataFrame,
    corrected_null_distributions: dict,
    main_output_dir: str,
    optimization_target: str,
    dr_type: str
) -> None:
    """
    Compute corrected p-values for all CCA scores and create visualization plots.
    """
    pvalue_dir = os.path.join(main_output_dir, "p_value")
    os.makedirs(pvalue_dir, exist_ok=True)
    
    modalities = ['rna', 'atac']
    dr_types = ['expression', 'proportion']
    
    df_results_copy = df_results.copy()
    
    # Initialize corrected p-value columns
    for modality in modalities:
        for dr_method in dr_types:
            corrected_pval_col = f'{modality}_corrected_pvalue_{dr_method}'
            if corrected_pval_col not in df_results_copy.columns:
                df_results_copy[corrected_pval_col] = np.nan
    
    # Process each resolution
    for idx, row in df_results_copy.iterrows():
        resolution = row['resolution']
        print(f"Computing corrected p-values for resolution {resolution:.3f}")
        
        res_dir = os.path.join(pvalue_dir, f"resolution_{resolution:.3f}")
        os.makedirs(res_dir, exist_ok=True)
        
        for modality in modalities:
            if modality not in corrected_null_distributions:
                continue
                
            corrected_null = corrected_null_distributions[modality]
            
            for dr_method in dr_types:
                cca_col = f'{modality}_cca_{dr_method}'
                
                if cca_col in row and not pd.isna(row[cca_col]):
                    cca_score = row[cca_col]
                    corrected_p_value = np.mean(corrected_null >= cca_score)
                    
                    corrected_pval_col = f'{modality}_corrected_pvalue_{dr_method}'
                    df_results_copy.loc[idx, corrected_pval_col] = corrected_p_value
                    
                    # Create clean p-value plot
                    plt.figure(figsize=(8, 5))
                    plt.hist(corrected_null, bins=40, alpha=0.7, color='#3498DB', 
                            density=True, edgecolor='white', linewidth=0.5)
                    plt.axvline(cca_score, color='#E74C3C', linestyle='--', linewidth=2.5)
                    plt.annotate(f'Observed: {cca_score:.4f}\np = {corrected_p_value:.4f}',
                                xy=(cca_score, plt.gca().get_ylim()[1] * 0.9),
                                fontsize=11, fontweight='medium',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                         edgecolor='#E74C3C', alpha=0.9))
                    plt.xlabel('CCA Score', fontsize=11)
                    plt.ylabel('Density', fontsize=11)
                    plt.title(f'{modality.upper()} {dr_method.capitalize()} | Resolution {resolution:.3f}',
                             fontsize=12, fontweight='bold')
                    plt.grid(True, alpha=0.3)
                    
                    plot_path = os.path.join(res_dir, f'pvalue_{modality}_{dr_method}.png')
                    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
                    plt.close()
                    
                    print(f"  {modality.upper()} {dr_method}: CCA={cca_score:.4f}, p={corrected_p_value:.4f}")
    
    # Update original dataframe
    for modality in modalities:
        for dr_method in dr_types:
            corrected_pval_col = f'{modality}_corrected_pvalue_{dr_method}'
            df_results[corrected_pval_col] = df_results_copy[corrected_pval_col]
    
    print(f"P-value plots saved to: {pvalue_dir}")

def analyze_modality_correlation(
    df_results: pd.DataFrame,
    dr_type: str,
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
    
    valid_mask = ~df_results[rna_col].isna() & ~df_results[atac_col].isna()
    valid_df = df_results[valid_mask].copy()
    
    if len(valid_df) < 3:
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
    plt.savefig(os.path.join(corr_dir, f'modality_correlation_{dr_type}.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save text results
    with open(os.path.join(corr_dir, f'correlation_results_{dr_type}.txt'), 'w') as f:
        f.write(f"Modality Correlation Analysis - {dr_type.upper()}\n")
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
    Create summary text files for each modality × DR combination.
    """
    modalities = ['rna', 'atac']
    dr_types = ['expression', 'proportion']
    df_sorted = df_results.sort_values('resolution').copy()
    
    for modality in modalities:
        for dr in dr_types:
            cca_col = f'{modality}_cca_{dr}'
            corrected_pval_col = f'{modality}_corrected_pvalue_{dr}'
            
            txt_path = os.path.join(summary_dir, f'{modality}_{dr}_results.txt')
            
            with open(txt_path, 'w') as f:
                f.write(f"{modality.upper()} - {dr.capitalize()} DR Results\n")
                f.write("=" * 60 + "\n\n")
                
                if modality == optimization_target and dr == dr_type:
                    f.write("*** OPTIMIZATION TARGET ***\n")
                    f.write(f"Best Resolution: {final_best_resolution:.3f}\n")
                    best_cca = df_sorted.loc[
                        df_sorted['resolution'] == final_best_resolution, cca_col
                    ].iloc[0]
                    f.write(f"Best CCA Score: {best_cca:.4f}\n\n")
                
                f.write(f"{'Resolution':<12} {'CCA Score':<12}")
                if corrected_pval_col in df_sorted.columns:
                    f.write(f" {'Corrected P':<12}")
                f.write("\n" + "-" * 40 + "\n")
                
                for _, row in df_sorted.iterrows():
                    cca_str = f"{row[cca_col]:.4f}" if not np.isnan(row[cca_col]) else "N/A"
                    f.write(f"{row['resolution']:<12.3f} {cca_str:<12}")
                    if corrected_pval_col in df_sorted.columns:
                        corr_p = row.get(corrected_pval_col, np.nan)
                        corr_p_str = f"{corr_p:.4f}" if not np.isnan(corr_p) else "N/A"
                        f.write(f" {corr_p_str:<12}")
                    f.write("\n")
                
                valid_mask = ~df_sorted[cca_col].isna()
                if valid_mask.any():
                    f.write("\n" + "-" * 40 + "\n")
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
    num_pvalue_simulations: int = 1000,
    n_pcs: int = 2,
    compute_pvalues: bool = True,
    visualize_embeddings: bool = True,
    analyze_modality_alignment: bool = True,
    preserve_cols: Optional[Union[str, List[str]]] = None,
    verbose: bool = True
) -> Tuple[float, pd.DataFrame]:
    """
    Find optimal clustering resolution for integrated RNA+ATAC data.
    
    Performs two-pass search to find the resolution that maximizes CCA correlation
    between DR embeddings and severity levels for a specified modality and DR type.
    
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
    num_pvalue_simulations : int, default 1000
        Permutations for p-value computation
    n_pcs : int, default 2
        Number of PCs for CCA analysis
    compute_pvalues : bool, default True
        Whether to compute corrected p-values
    visualize_embeddings : bool, default True
        Whether to create embedding visualizations
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
    all_resolution_null_results = {'rna': [], 'atac': []}
    
    # ========== RESOLUTION PROCESSING FUNCTION ==========
    def process_resolution(resolution: float, search_pass: str) -> Tuple[dict, dict]:
        """Process a single resolution: cluster → pseudobulk → DR → CCA"""
        nonlocal AnnData_integrated
        
        print(f"\n--- Resolution: {resolution:.3f} ({search_pass}) ---")
        
        resolution_dir = os.path.join(resolutions_dir, f"resolution_{resolution:.3f}")
        os.makedirs(resolution_dir, exist_ok=True)
        
        # Initialize results
        result_dict = {
            'resolution': resolution,
            'rna_cca_expression': np.nan, 'rna_cca_proportion': np.nan,
            'atac_cca_expression': np.nan, 'atac_cca_proportion': np.nan,
            'rna_pvalue_expression': np.nan, 'rna_pvalue_proportion': np.nan,
            'atac_pvalue_expression': np.nan, 'atac_pvalue_proportion': np.nan,
            'optimization_score': np.nan, 'pass': search_pass, 'n_clusters': 0
        }
        resolution_null_results = {
            'rna': {'resolution': resolution, 'null_scores': None},
            'atac': {'resolution': resolution, 'null_scores': None}
        }
        
        try:
            # Step 1: Clean up previous clustering
            if 'cell_type' in AnnData_integrated.obs.columns:
                AnnData_integrated.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            if modality_col in AnnData_integrated.obs.columns:
                AnnData_integrated.obs[modality_col] = AnnData_integrated.obs[modality_col].astype(str)
            
            # Step 2: Clustering
            AnnData_integrated = cell_types_linux(
                AnnData_integrated, cell_type_column='cell_type', existing_cell_types=False,
                save=False, output_dir=resolution_dir, cluster_resolution=resolution,
                use_rep=use_rep, markers=None, num_PCs=num_PCs, verbose=False
            )
            n_clusters = AnnData_integrated.obs['cell_type'].nunique()
            result_dict['n_clusters'] = n_clusters
            print(f"  Clusters: {n_clusters}")
            
            # Step 3: Pseudobulk
            pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata(
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
            
            # Step 5: Null distributions (for both modalities)
            if compute_pvalues:
                for modality in ['rna', 'atac']:
                    try:
                        null_dist = generate_null_distribution(
                            pseudobulk_adata=pseudobulk_adata, modality=modality.upper(),
                            column=f'X_DR_{dr_type}', sev_col=sev_col, n_pcs=n_pcs,
                            n_permutations=num_pvalue_simulations,
                            save_path=os.path.join(resolution_dir, f'null_dist_{modality}_{dr_type}.npy'),
                            verbose=verbose and (modality == optimization_target)
                        )
                        resolution_null_results[modality]['null_scores'] = null_dist
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Null dist failed for {modality}: {str(e)}")
            
            # Step 6: CCA analysis
            dr_columns = ['X_DR_expression', 'X_DR_proportion']
            cca_results_df = batch_cca_analysis(
                pseudobulk_adata=pseudobulk_adata, dr_columns=dr_columns, sev_col=sev_col,
                modalities=['RNA', 'ATAC'], n_components=1, n_pcs=n_pcs, output_dir=resolution_dir
            )
            
            for _, row in cca_results_df.iterrows():
                modality = row['modality'].lower()
                dr_method = row['column'].replace('X_DR_', '')
                result_dict[f'{modality}_cca_{dr_method}'] = row['cca_score']
                if row['valid'] and not np.isnan(row['cca_score']):
                    print(f"  {row['modality']} {dr_method}: {row['cca_score']:.4f}")
            
            # Set optimization score
            target_metric = f"{optimization_target}_cca_{dr_type}"
            result_dict['optimization_score'] = result_dict.get(target_metric, np.nan)
            
            # Step 7: Embedding visualizations
            if visualize_embeddings:
                try:
                    for viz_modality in ['RNA', 'ATAC']:
                        embedding_path = os.path.join(resolution_dir, f"embedding_{viz_modality}_{dr_type}")
                        visualize_multimodal_embedding_with_cca(
                            adata=pseudobulk_adata, modality_col=modality_col, color_col=sev_col,
                            target_modality=viz_modality, cca_results_df=cca_results_df,
                            output_dir=embedding_path, show_sample_names=False, verbose=False
                        )
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Embedding viz failed: {str(e)}")
            
            # Save results
            pd.DataFrame([result_dict]).to_csv(
                os.path.join(resolution_dir, f"results_res_{resolution:.3f}.csv"), index=False)
            pseudobulk_adata.write_h5ad(os.path.join(resolution_dir, "pseudobulk_sample.h5ad"))
                
        except Exception as e:
            print(f"  Error: {str(e)}")
        
        return result_dict, resolution_null_results
    
    # ========== PASS 1: COARSE SEARCH ==========
    print("\n" + "=" * 60)
    print("PASS 1: Coarse Search (0.1 - 1.0, step 0.1)")
    print("=" * 60)
    
    for resolution in np.arange(0.1, 1.01, 0.1):
        result_dict, resolution_null_results = process_resolution(resolution, 'coarse')
        all_results.append(result_dict)
        all_resolution_null_results['rna'].append(resolution_null_results['rna'])
        all_resolution_null_results['atac'].append(resolution_null_results['atac'])
    
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
        
        result_dict, resolution_null_results = process_resolution(resolution, 'fine')
        all_results.append(result_dict)
        all_resolution_null_results['rna'].append(resolution_null_results['rna'])
        all_resolution_null_results['atac'].append(resolution_null_results['atac'])
    
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
    
    # ========== CORRECTED P-VALUES ==========
    corrected_null_distributions = {}
    if compute_pvalues:
        print("\n" + "=" * 60)
        print("GENERATING CORRECTED NULL DISTRIBUTIONS")
        print("=" * 60)
        
        for modality in ['rna', 'atac']:
            valid_null_results = [r for r in all_resolution_null_results[modality] 
                                 if r['null_scores'] is not None]
            if valid_null_results:
                print(f"\nGenerating corrected null for {modality.upper()}...")
                corrected_null_distributions[modality] = generate_corrected_null_distribution_for_modality(
                    all_resolution_results=valid_null_results, modality=modality,
                    dr_type=dr_type, n_permutations=num_pvalue_simulations
                )
                null_path = os.path.join(summary_dir, f'corrected_null_{modality}_{dr_type}.npy')
                np.save(null_path, corrected_null_distributions[modality])
                print(f"  Saved to: {null_path}")
        
        if corrected_null_distributions:
            print("\nComputing corrected p-values...")
            compute_all_corrected_pvalues_and_plots(
                df_results=df_results, corrected_null_distributions=corrected_null_distributions,
                main_output_dir=main_output_dir, optimization_target=optimization_target, dr_type=dr_type
            )
    
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
            df_results=df_results, dr_type=dr_type, output_dir=main_output_dir, verbose=verbose
        )
    
    # ========== PRINT FINAL RESULTS ==========
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nOptimization Target: {optimization_target.upper()} {dr_type.upper()}")
    print(f"Best Resolution: {final_best_resolution:.3f}")
    print(f"Best CCA Score: {final_best_score:.4f}")
    print(f"Number of Clusters: {valid_results.loc[final_best_idx, 'n_clusters']}")
    
    # Corrected p-value
    corrected_pval_col = f"{optimization_target}_corrected_pvalue_{dr_type}"
    if corrected_pval_col in df_results.columns:
        corrected_p = df_results.loc[df_results['resolution'] == final_best_resolution, corrected_pval_col].iloc[0]
        if not pd.isna(corrected_p):
            print(f"Corrected P-value: {corrected_p:.4f}")
    
    # All scores at best resolution
    best_row = valid_results.loc[final_best_idx]
    print(f"\nAll CCA scores at resolution {final_best_resolution:.3f}:")
    print(f"  RNA Expression:  {best_row['rna_cca_expression']:.4f}")
    print(f"  RNA Proportion:  {best_row['rna_cca_proportion']:.4f}")
    print(f"  ATAC Expression: {best_row['atac_cca_expression']:.4f}")
    print(f"  ATAC Proportion: {best_row['atac_cca_proportion']:.4f}")
    
    if modality_correlation_results:
        print(f"\nModality Alignment ({dr_type}):")
        print(f"  Pearson r = {modality_correlation_results['pearson_r']:.4f}")
        print(f"  {modality_correlation_results['interpretation']}")
    
    # ========== SAVE RESULTS ==========
    results_csv = os.path.join(summary_dir, f"resolution_results_{optimization_target}_{dr_type}.csv")
    df_results.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")
    
    create_resolution_optimization_summary(
        df_results=df_results, final_best_resolution=final_best_resolution,
        optimization_target=optimization_target, dr_type=dr_type, summary_dir=summary_dir
    )
    
    # Copy optimal pseudobulk
    optimal_res_dir = os.path.join(resolutions_dir, f"resolution_{final_best_resolution:.3f}")
    optimal_pb_path = os.path.join(optimal_res_dir, "pseudobulk_sample.h5ad")
    if os.path.exists(optimal_pb_path):
        import shutil
        shutil.copy2(optimal_pb_path, os.path.join(summary_dir, "optimal.h5ad"))
        print(f"Optimal pseudobulk saved to: {os.path.join(summary_dir, 'optimal.h5ad')}")
    
    # ========== FINAL SUMMARY FILE ==========
    final_summary_path = os.path.join(main_output_dir, "FINAL_SUMMARY.txt")
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
        if batch_col:
            f.write(f"  Batch columns: {batch_col}\n")
        f.write(f"\nRESULTS:\n")
        f.write(f"  Optimal resolution: {final_best_resolution:.3f}\n")
        f.write(f"  Best CCA score: {final_best_score:.4f}\n")
        f.write(f"  Number of clusters: {valid_results.loc[final_best_idx, 'n_clusters']}\n\n")
        f.write("ALL CCA SCORES AT OPTIMAL RESOLUTION:\n")
        f.write(f"  RNA Expression:  {best_row['rna_cca_expression']:.4f}\n")
        f.write(f"  RNA Proportion:  {best_row['rna_cca_proportion']:.4f}\n")
        f.write(f"  ATAC Expression: {best_row['atac_cca_expression']:.4f}\n")
        f.write(f"  ATAC Proportion: {best_row['atac_cca_proportion']:.4f}\n\n")
        if modality_correlation_results:
            f.write("MODALITY ALIGNMENT:\n")
            f.write(f"  Pearson r: {modality_correlation_results['pearson_r']:.4f}\n")
            f.write(f"  Interpretation: {modality_correlation_results['interpretation']}\n\n")
        f.write("SEARCH SUMMARY:\n")
        f.write(f"  Total resolutions: {len(valid_results)}\n")
        f.write(f"  Coarse pass: {len(valid_results[valid_results['pass'] == 'coarse'])}\n")
        f.write(f"  Fine pass: {len(valid_results[valid_results['pass'] == 'fine'])}\n")
    
    print(f"Final summary: {final_summary_path}")
    print("\n" + "=" * 80)
    
    return final_best_resolution, df_results