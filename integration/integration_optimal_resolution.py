import anndata as ad
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from anndata import AnnData
import time
from DR import process_anndata_with_pca
from pseudo_adata import compute_pseudobulk_adata
from CCA import *
from CCA_test import *
from linux.CellType_linux import cell_types_linux
from integration_visualization import *
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

# Suppress specific warnings that are expected during CCA analysis
def suppress_warnings():
    """Suppress specific warnings that are expected during CCA analysis"""
    warnings.filterwarnings('ignore', category=UserWarning, 
                          message='.*y residual is constant at iteration.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, 
                          message='.*invalid value encountered in divide.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, 
                          message='.*All-NaN slice encountered.*')

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

def cca_analysis(pseudobulk_adata, modality, column, sev_col, n_components=2, n_pcs=None):
    """
    CCA analysis with weight vectors included in results.
    
    Parameters:
    -----------
    pseudobulk_adata : sc.AnnData
        AnnData object containing dimension reduction results
    modality : str
        Modality to analyze
    column : str
        Column name for dimension reduction coordinates
    sev_col : str
        Column name for severity levels
    n_components : int, default 2
        Number of CCA components
    n_pcs : int, optional
        Number of PCs to use for CCA analysis. If None, uses all available PCs.
        
    Returns:
    --------
    dict with CCA results including weight vectors
    """
    
    result = {
        'cca_score': np.nan,
        'n_samples': 0,
        'n_features': 0,
        'modality': modality,
        'column': column,
        'valid': False,
        'error_message': None,
        'X_weights': None,  # Added: weights for embedding space
        'Y_weights': None,  # Added: weights for severity space
        'X_loadings': None, # Added: loadings (correlations)
        'Y_loadings': None,  # Added: loadings (correlations)
        'n_pcs_used': 0  # Added: track number of PCs actually used
    }
    
    try:
        if column not in pseudobulk_adata.uns:
            result['error_message'] = f"Column '{column}' not found in uns"
            return result
        if 'modality' not in pseudobulk_adata.obs.columns:
            result['error_message'] = "Column 'modality' not found in obs"
            return result
        if sev_col not in pseudobulk_adata.obs.columns:
            result['error_message'] = f"Column '{sev_col}' not found in obs"
            return result
        
        # Standardize indices to lowercase to avoid case sensitivity issues
        obs_standardized = pseudobulk_adata.obs.copy()
        obs_standardized.index = obs_standardized.index.str.lower()
        
        uns_data_standardized = pseudobulk_adata.uns[column].copy()
        uns_data_standardized.index = uns_data_standardized.index.str.lower()
        
        # Extract modality samples using standardized indices
        modality_mask = obs_standardized['modality'] == modality
        
        if not modality_mask.any():
            result['error_message'] = f"No samples found for modality: {modality}"
            return result
        
        # Extract dimension reduction coordinates and severity levels
        dr_coords_full = uns_data_standardized.loc[modality_mask].copy()
        sev_levels = obs_standardized.loc[modality_mask, sev_col].values
        
        # Determine number of PCs to use
        max_pcs = dr_coords_full.shape[1]
        if n_pcs is None:
            # Use all available PCs by default
            n_pcs_to_use = max_pcs
        else:
            # Use specified number of PCs, but not more than available
            n_pcs_to_use = min(n_pcs, max_pcs)
        
        # Use only the specified number of PCs for CCA analysis
        dr_coords = dr_coords_full.iloc[:, :n_pcs_to_use]
        
        # Basic validation only
        if len(dr_coords) < 3:
            result['error_message'] = f"Insufficient samples: {len(dr_coords)}"
            return result
        if len(np.unique(sev_levels)) < 2:
            result['error_message'] = "Insufficient severity level variance"
            return result
        
        # Prepare data for CCA
        X = dr_coords.values  # Now using specified number of PCs
        y = sev_levels.reshape(-1, 1)
        
        result['n_samples'] = len(X)
        result['n_features'] = X.shape[1]  # Number of PCs used
        result['n_pcs_used'] = n_pcs_to_use
        
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
        
        # Transform and compute correlation
        X_c, y_c = cca.transform(X_scaled, y_scaled)
        correlation = np.corrcoef(X_c[:, 0], y_c[:, 0])[0, 1]
        cca_score = abs(correlation)
        
        # Store results
        result['cca_score'] = cca_score
        result['valid'] = True
        
        # Store weight vectors (these define the canonical directions)
        result['X_weights'] = cca.x_weights_.flatten()  # Shape: (n_features,)
        result['Y_weights'] = cca.y_weights_.flatten()  # Shape: (1,) for univariate y
        
        # Store loadings (correlations between original variables and canonical variates)
        result['X_loadings'] = cca.x_loadings_ if hasattr(cca, 'x_loadings_') else None
        result['Y_loadings'] = cca.y_loadings_ if hasattr(cca, 'y_loadings_') else None
        
    except Exception as e:
        result['error_message'] = f"CCA failed: {str(e)}"
    
    return result


def batch_cca_analysis(pseudobulk_adata, dr_columns, sev_col, modalities=None, 
                      n_components=2, n_pcs=None, output_dir=None):
    """
    Run CCA analysis across multiple dimension reduction results and modalities.
    Now includes weight vectors in the results.
    
    Parameters:
    -----------
    pseudobulk_adata : sc.AnnData
        AnnData object containing dimension reduction results
    dr_columns : list
        List of dimension reduction column names to analyze
    sev_col : str
        Column name for severity levels
    modalities : list, optional
        List of modalities to analyze. If None, uses all unique modalities
    n_components : int, default 2
        Number of CCA components
    n_pcs : int, optional
        Number of PCs to use for CCA analysis. If None, uses all available PCs.
    output_dir : str, optional
        Directory to save results
        
    Returns:
    --------
    pd.DataFrame
        Results DataFrame with CCA scores and weight vectors
    """
    
    import os
    
    if modalities is None:
        modalities = pseudobulk_adata.obs['modality'].unique()
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for modality in modalities:
        for column in dr_columns:
            if column in pseudobulk_adata.uns:
                result = cca_analysis(
                    pseudobulk_adata=pseudobulk_adata,
                    modality=modality,
                    column=column,
                    sev_col=sev_col,
                    n_components=n_components,
                    n_pcs=n_pcs  # Pass n_pcs parameter
                )
                
                results.append(result)
                
            else:
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
    
    # Save results table to output directory
    if output_dir:
        # Save summary (without vectors for readability)
        summary_df = results_df[['modality', 'column', 'cca_score', 'n_samples', 
                                'n_features', 'n_pcs_used', 'valid', 'error_message']]
        summary_path = os.path.join(output_dir, 'cca_results_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")
        
        # Save complete results (including vectors) as pickle for preservation
        import pickle
        complete_path = os.path.join(output_dir, 'cca_results_complete.pkl')
        with open(complete_path, 'wb') as f:
            pickle.dump(results_df, f)
        print(f"Complete results with vectors saved to: {complete_path}")
    
    return results_df


def generate_null_distribution(pseudobulk_adata, modality, column, sev_col,
                             n_permutations=1000, n_pcs=None,
                             save_path=None, verbose=True):
    """
    Generate null distribution using customizable number of PC dimensions.
    Steps:
    1. Extract specified number of DR components for the specified modality
    2. Randomly shuffle severity level labels for each sample
    3. Run CCA between PC embeddings and 1D severity (always 1 component)
    4. Record CCA correlation as one simulation
    
    Parameters:
    -----------
    n_pcs : int, optional
        Number of PC dimensions to use. If None, uses all available PCs.
    """
    # Extract data for specified modality
    modality_mask = pseudobulk_adata.obs['modality'] == modality
    if not modality_mask.any():
        raise ValueError(f"No samples found for modality: {modality}")
    
    # Get DR coordinates and severity levels
    dr_coords_full = pseudobulk_adata.uns[column].loc[modality_mask].copy()
    sev_levels = pseudobulk_adata.obs.loc[modality_mask, sev_col].values
    
    # Use specified number of DR components (default: all available)
    if n_pcs is None:
        dr_coords = dr_coords_full
        n_dims_used = dr_coords_full.shape[1]
    else:
        n_pcs = min(n_pcs, dr_coords_full.shape[1])  # Don't exceed available PCs
        dr_coords = dr_coords_full.iloc[:, :n_pcs]
        n_dims_used = n_pcs
    
    if len(dr_coords) < 3:
        raise ValueError(f"Insufficient samples: {len(dr_coords)}")
    if len(np.unique(sev_levels)) < 2:
        raise ValueError("Insufficient severity level variance")
    
    # Prepare data for CCA
    X = dr_coords.values  # PC embedding coordinates [n_samples, n_dims]
    y_original = sev_levels.copy()  # 1D severity levels [n_samples]
    
    # Import libraries
    from sklearn.cross_decomposition import CCA
    from sklearn.preprocessing import StandardScaler
    
    # Always use 1 CCA component
    n_components = 1
    
    # Run permutations
    null_scores = []
    failed_permutations = 0
    
    for perm in range(n_permutations):
        try:
            # Randomly shuffle severity level labels
            permuted_sev = np.random.permutation(y_original)
            
            # Run CCA between PC embeddings and shuffled 1D severity
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_permuted_scaled = scaler_y.fit_transform(permuted_sev.reshape(-1, 1))
            
            # Fit CCA with 1 component
            cca_perm = CCA(n_components=n_components, max_iter=1000, tol=1e-6)
            cca_perm.fit(X_scaled, y_permuted_scaled)
            
            # Transform and compute correlation
            X_c_perm, y_c_perm = cca_perm.transform(X_scaled, y_permuted_scaled)
            perm_correlation = np.corrcoef(X_c_perm[:, 0], y_c_perm[:, 0])[0, 1]
            
            # Record the CCA score
            if np.isnan(perm_correlation) or np.isinf(perm_correlation):
                null_scores.append(0.0)
                failed_permutations += 1
            else:
                null_scores.append(abs(perm_correlation))
                
        except Exception:
            null_scores.append(0.0)
            failed_permutations += 1
    
    null_distribution = np.array(null_scores)
    
    # Single status print
    if verbose:
        success_rate = (n_permutations - failed_permutations) / n_permutations * 100
        print(f"Null distribution generated using {n_dims_used} PC dimensions (1 CCA component): {success_rate:.1f}% success rate ({n_permutations - failed_permutations}/{n_permutations} permutations)")
    
    if save_path:
        np.save(save_path, null_distribution)
    
    return null_distribution

def ensure_non_categorical_columns(adata, columns):
    """Convert specified columns from categorical to string to avoid categorical errors"""
    for col in columns:
        if col in adata.obs.columns:
            if pd.api.types.is_categorical_dtype(adata.obs[col]):
                adata.obs[col] = adata.obs[col].astype(str)
    return adata

def compute_all_corrected_pvalues_and_plots(df_results, corrected_null_distributions, main_output_dir, 
                                          optimization_target, dr_type):
    """
    Compute corrected p-values for all CCA scores and create visualization plots.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Results dataframe with CCA scores for all resolutions
    corrected_null_distributions : dict
        Dictionary with corrected null distributions for each modality
        Format: {'rna': np.array, 'atac': np.array}
    main_output_dir : str
        Main output directory
    optimization_target : str
        Target modality for optimization
    dr_type : str
        Target DR type for optimization
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create p-value directory
    pvalue_dir = os.path.join(main_output_dir, "p_value")
    os.makedirs(pvalue_dir, exist_ok=True)
    
    # Define all modalities and DR types to process
    modalities = ['rna', 'atac']
    dr_types = ['expression', 'proportion']
    
    # Create a copy of the dataframe to avoid modifying the original
    df_results_copy = df_results.copy()
    
    # Add corrected p-value columns if they don't exist
    for modality in modalities:
        for dr_method in dr_types:
            corrected_pval_col = f'{modality}_corrected_pvalue_{dr_method}'
            if corrected_pval_col not in df_results_copy.columns:
                df_results_copy[corrected_pval_col] = np.nan
    
    # Process each resolution
    for idx, row in df_results_copy.iterrows():
        resolution = row['resolution']
        print(f"Computing corrected p-values for resolution {resolution}")
        
        # Create resolution-specific directory
        res_dir = os.path.join(pvalue_dir, f"resolution_{resolution}")
        os.makedirs(res_dir, exist_ok=True)
        
        # Process each modality and DR type combination
        for modality in modalities:
            # Use the modality-specific null distribution
            if modality not in corrected_null_distributions:
                print(f"  Warning: No null distribution available for {modality}")
                continue
                
            corrected_null_distribution = corrected_null_distributions[modality]
            
            for dr_method in dr_types:
                cca_col = f'{modality}_cca_{dr_method}'
                
                if cca_col in row and not pd.isna(row[cca_col]):
                    cca_score = row[cca_col]
                    
                    # Compute corrected p-value using modality-specific null distribution
                    corrected_p_value = np.mean(corrected_null_distribution >= cca_score)
                    
                    # Store corrected p-value directly using loc to ensure proper alignment
                    corrected_pval_col = f'{modality}_corrected_pvalue_{dr_method}'
                    df_results_copy.loc[idx, corrected_pval_col] = corrected_p_value
                    
                    # Create visualization plot
                    plt.figure(figsize=(10, 6))
                    
                    # Plot histogram of null distribution
                    plt.hist(corrected_null_distribution, bins=50, alpha=0.7, color='lightblue', 
                            density=True, label=f'Corrected Null Distribution ({modality.upper()})')
                    
                    # Plot vertical line for observed CCA score
                    plt.axvline(cca_score, color='red', linestyle='--', linewidth=2, 
                               label=f'Observed CCA Score: {cca_score:.4f}')
                    
                    # Add p-value text
                    plt.text(0.05, 0.95, f'Corrected p-value: {corrected_p_value:.4f}', 
                            transform=plt.gca().transAxes, fontsize=12, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Formatting
                    plt.xlabel('CCA Score')
                    plt.ylabel('Density')
                    plt.title(f'Corrected P-value Analysis\nResolution: {resolution}, {modality.upper()} {dr_method}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Save plot in resolution directory
                    plot_filename = f'pvalue_plot_res_{resolution}_{modality}_{dr_method}.png'
                    plot_path = os.path.join(res_dir, plot_filename)
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  {modality.upper()} {dr_method}: CCA={cca_score:.4f}, p={corrected_p_value:.4f}")
    
    # Update the original dataframe with the corrected p-values
    for modality in modalities:
        for dr_method in dr_types:
            corrected_pval_col = f'{modality}_corrected_pvalue_{dr_method}'
            df_results[corrected_pval_col] = df_results_copy[corrected_pval_col]
    
    print(f"All p-value plots saved to: {pvalue_dir}")


def generate_corrected_null_distribution_for_modality(all_resolution_results, modality, dr_type, n_permutations=1000):
    """
    Generate null distribution for a specific modality accounting for resolution selection bias.
    
    Parameters:
    -----------
    all_resolution_results : list
        List of dictionaries containing results from all resolutions for this modality
    modality : str
        Target modality ("rna" or "atac")
    dr_type : str
        DR type ("expression" or "proportion")
    n_permutations : int
        Number of permutations
        
    Returns:
    --------
    np.array
        Corrected null distribution for this modality
    """
    
    corrected_null_scores = []
    
    for perm_idx in range(n_permutations):
        # Collect the CCA score from this permutation across all resolutions
        perm_scores_across_resolutions = []
        
        for resolution_result in all_resolution_results:
            if 'null_scores' in resolution_result and resolution_result['null_scores'] is not None:
                if len(resolution_result['null_scores']) > perm_idx:
                    perm_scores_across_resolutions.append(resolution_result['null_scores'][perm_idx])
        
        # Select the maximum score (mimicking optimal resolution selection)
        if perm_scores_across_resolutions:
            max_score_for_this_perm = max(perm_scores_across_resolutions)
            corrected_null_scores.append(max_score_for_this_perm)
    
    return np.array(corrected_null_scores)

def create_resolution_optimization_summary(
    df_results: pd.DataFrame,
    final_best_resolution: float,
    optimization_target: str,
    dr_type: str,
    summary_dir: str
) -> None:
    """
    Create comprehensive summary visualizations and text files for resolution optimization results.
    
    Creates:
    - 4 line charts showing CCA values vs resolution for each modality/DR type combination
    - 4 text files containing CCA values and p-values for each combination
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        DataFrame containing all resolution results
    final_best_resolution : float
        The optimal resolution found
    optimization_target : str
        Which modality was optimized: "rna" or "atac"
    dr_type : str
        Which DR type was optimized: "expression" or "proportion"
    summary_dir : str
        Directory to save summary outputs
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    
    # Define the modality and DR type combinations
    modalities = ['rna', 'atac']
    dr_types = ['expression', 'proportion']
    
    # Create a figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Sort results by resolution for proper line plotting
    df_sorted = df_results.sort_values('resolution').copy()
    
    # Create plots and save text files for each combination
    plot_idx = 0
    for modality in modalities:
        for dr in dr_types:
            ax = axes[plot_idx]
            
            # Column names for CCA scores and p-values
            cca_col = f'{modality}_cca_{dr}'
            pval_col = f'{modality}_pvalue_{dr}'
            corrected_pval_col = f'{modality}_corrected_pvalue_{dr}'
            
            # Get data for plotting
            resolutions = df_sorted['resolution'].values
            cca_values = df_sorted[cca_col].values
            
            # Create the line plot
            ax.plot(resolutions, cca_values, 'b.-', linewidth=2, markersize=8)
            
            # Add vertical line at best resolution if this was the optimization target
            if modality == optimization_target and dr == dr_type:
                ax.axvline(x=final_best_resolution, color='red', linestyle='--', 
                          linewidth=2, label=f'Best resolution: {final_best_resolution:.3f}')
                # Highlight the optimization target plot with a different background
                ax.set_facecolor('#f0f0f0')
            
            # Set labels and title
            ax.set_xlabel('Resolution', fontsize=12)
            ax.set_ylabel('CCA Score', fontsize=12)
            ax.set_title(f'{modality.upper()} - {dr.capitalize()} DR', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add legend if there's a best resolution line
            if modality == optimization_target and dr == dr_type:
                ax.legend(fontsize=10)
            
            # Set y-axis limits with some padding
            valid_cca = cca_values[~np.isnan(cca_values)]
            if len(valid_cca) > 0:
                y_min, y_max = valid_cca.min(), valid_cca.max()
                y_padding = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - y_padding, y_max + y_padding)
            
            # Save text file with CCA values and p-values
            txt_filename = f'{modality}_{dr}_results.txt'
            txt_path = os.path.join(summary_dir, txt_filename)
            
            with open(txt_path, 'w') as f:
                f.write(f"Resolution Optimization Results: {modality.upper()} - {dr.capitalize()} DR\n")
                f.write("=" * 80 + "\n\n")
                
                # Write optimization target information
                if modality == optimization_target and dr == dr_type:
                    f.write(f"*** OPTIMIZATION TARGET ***\n")
                    f.write(f"Best Resolution: {final_best_resolution:.3f}\n")
                    best_cca = df_sorted.loc[df_sorted['resolution'] == final_best_resolution, cca_col].iloc[0]
                    f.write(f"Best CCA Score: {best_cca:.4f}\n\n")
                
                # Write header
                f.write(f"{'Resolution':<12} {'CCA Score':<12} {'P-value':<12}")
                if corrected_pval_col in df_sorted.columns:
                    f.write(f" {'Corrected P-value':<18}")
                f.write("\n")
                f.write("-" * 80 + "\n")
                
                # Write data for each resolution
                for idx, row in df_sorted.iterrows():
                    resolution = row['resolution']
                    cca_score = row[cca_col]
                    pvalue = row.get(pval_col, np.nan)
                    
                    # Format values
                    cca_str = f"{cca_score:.4f}" if not np.isnan(cca_score) else "N/A"
                    pval_str = f"{pvalue:.4f}" if not np.isnan(pvalue) else "N/A"
                    
                    f.write(f"{resolution:<12.3f} {cca_str:<12} {pval_str:<12}")
                    
                    # Add corrected p-value if available
                    if corrected_pval_col in df_sorted.columns:
                        corrected_pval = row.get(corrected_pval_col, np.nan)
                        corrected_pval_str = f"{corrected_pval:.4f}" if not np.isnan(corrected_pval) else "N/A"
                        f.write(f" {corrected_pval_str:<18}")
                    
                    f.write("\n")
                
                # Add summary statistics at the end
                f.write("\n" + "=" * 80 + "\n")
                f.write("Summary Statistics:\n")
                f.write(f"Total resolutions tested: {len(df_sorted)}\n")
                
                valid_cca_mask = ~df_sorted[cca_col].isna()
                if valid_cca_mask.any():
                    f.write(f"Min CCA Score: {df_sorted.loc[valid_cca_mask, cca_col].min():.4f}\n")
                    f.write(f"Max CCA Score: {df_sorted.loc[valid_cca_mask, cca_col].max():.4f}\n")
                    f.write(f"Mean CCA Score: {df_sorted.loc[valid_cca_mask, cca_col].mean():.4f}\n")
                    f.write(f"Std CCA Score: {df_sorted.loc[valid_cca_mask, cca_col].std():.4f}\n")
            
            print(f"Saved results to: {txt_path}")
            
            plot_idx += 1
    
    # Adjust layout and save the figure
    plt.suptitle(f'CCA Scores vs Resolution - All Modalities and DR Types\n(Optimization Target: {optimization_target.upper()} {dr_type.capitalize()})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    plot_filename = 'cca_scores_vs_resolution_all_modalities.png'
    plot_path = os.path.join(summary_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved CCA vs Resolution plot to: {plot_path}")
    
    # Create individual plots for each modality/DR combination for better visibility
    for modality in modalities:
        for dr in dr_types:
            plt.figure(figsize=(10, 6))
            
            cca_col = f'{modality}_cca_{dr}'
            resolutions = df_sorted['resolution'].values
            cca_values = df_sorted[cca_col].values
            
            plt.plot(resolutions, cca_values, 'b.-', linewidth=2, markersize=8)
            
            # Add vertical line at best resolution if this was the optimization target
            if modality == optimization_target and dr == dr_type:
                plt.axvline(x=final_best_resolution, color='red', linestyle='--', 
                           linewidth=2, label=f'Best resolution: {final_best_resolution:.3f}')
                plt.legend(fontsize=12)
            
            plt.xlabel('Resolution', fontsize=14)
            plt.ylabel('CCA Score', fontsize=14)
            plt.title(f'{modality.upper()} - {dr.capitalize()} DR: CCA Score vs Resolution', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Save individual plot
            individual_plot_filename = f'cca_scores_vs_resolution_{modality}_{dr}.png'
            individual_plot_path = os.path.join(summary_dir, individual_plot_filename)
            plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved individual plot to: {individual_plot_path}")
    
    print(f"\nAll summary files saved to: {summary_dir}")

# -----------------Main Function for Finding Optimal Cell Resolution-----------------

def find_optimal_cell_resolution_integration(
    AnnData_integrated: AnnData,
    output_dir: str,
    optimization_target: str = "rna",  # "rna" or "atac"
    dr_type: str = "expression",  # "expression" or "proportion"
    n_features: int = 40000,
    sev_col: str = "sev.level",
    batch_col: str = None,
    sample_col: str = "sample",
    modality_col: str = "modality",
    use_rep: str = 'X_glue',
    num_DR_components: int = 30,
    num_PCs: int = 20,
    num_pvalue_simulations: int = 1000,
    n_pcs = 2,
    compute_pvalues: bool = True,
    visualize_embeddings: bool = True,
    verbose: bool = True
) -> tuple:
    """
    Find optimal clustering resolution for integrated RNA+ATAC data by maximizing 
    CCA correlation between dimension reduction and severity levels.
    
    [Parameters remain the same]
    """
    start_time = time.time()
    
    print("\n\n Finding optimal resolution begins \n\n")

    # Validate optimization target and dr_type
    if optimization_target not in ["rna", "atac"]:
        raise ValueError("optimization_target must be 'rna' or 'atac'")
    
    if dr_type not in ["expression", "proportion"]:
        raise ValueError("dr_type must be 'expression' or 'proportion'")

    # Create main output directory
    main_output_dir = os.path.join(output_dir, f"Integration_optimization_{optimization_target}_{dr_type}")
    os.makedirs(main_output_dir, exist_ok=True)

    print(f"Starting integrated resolution optimization...")
    print(f"Optimization target: {optimization_target.upper()} {dr_type.upper()}")
    print(f"Using representation: {use_rep} with {num_PCs} components")
    print(f"Testing resolutions from 0.01 to 1.00...")
    if compute_pvalues:
        print(f"Computing p-values with {num_pvalue_simulations} simulations per resolution")

    # Ensure critical columns are not categorical to avoid errors
    columns_to_check = ['cell_type', modality_col, sev_col, sample_col]
    if batch_col:
        columns_to_check.append(batch_col)
    AnnData_integrated = ensure_non_categorical_columns(AnnData_integrated, columns_to_check)
    
    # Storage for all results
    all_results = []
    # Storage for null distribution results from each resolution for each modality
    all_resolution_null_results = {
        'rna': [],
        'atac': []
    }

    # First pass: coarse search
    print("\n=== FIRST PASS: Coarse Search ===")
    for resolution in np.arange(0.1, 1.01, 0.1):
        print(f"\n\nTesting resolution: {resolution:.2f}\n")
        
        # Create resolution-specific directory
        resolution_dir = os.path.join(main_output_dir, f"resolution_{resolution:.2f}")
        os.makedirs(resolution_dir, exist_ok=True)
        
        result_dict = {
            'resolution': resolution,
            'rna_cca_expression': np.nan,
            'rna_cca_proportion': np.nan,
            'atac_cca_expression': np.nan,
            'atac_cca_proportion': np.nan,
            'rna_pvalue_expression': np.nan,
            'rna_pvalue_proportion': np.nan,
            'atac_pvalue_expression': np.nan,
            'atac_pvalue_proportion': np.nan,
            'optimization_score': np.nan,
            'pass': 'coarse'
        }
        
        # Initialize null results for this resolution for both modalities
        resolution_null_results = {
            'rna': {'resolution': resolution, 'null_scores': None},
            'atac': {'resolution': resolution, 'null_scores': None}
        }
        
        try:
            # Clean up previous cell type assignments
            if 'cell_type' in AnnData_integrated.obs.columns:
                AnnData_integrated.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            
            # Ensure modality column is properly set up
            if modality_col in AnnData_integrated.obs.columns:
                # Convert to string type to avoid categorical issues
                AnnData_integrated.obs[modality_col] = AnnData_integrated.obs[modality_col].astype(str)
            
            # Perform clustering using the Linux version
            AnnData_integrated = cell_types_linux(
                AnnData_integrated,
                cell_column='cell_type',
                existing_cell_types=False,
                Save=False,
                output_dir=resolution_dir,
                cluster_resolution=resolution,
                use_rep=use_rep,
                markers=None,
                num_PCs=num_PCs,
                verbose=False
            )
            
            # Compute pseudobulk data
            pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata(
                adata=AnnData_integrated, 
                batch_col=batch_col, 
                sample_col=sample_col, 
                celltype_col='cell_type', 
                n_features=n_features, 
                output_dir=resolution_dir,
                Save=False,
                verbose=False
            )
            
            # Perform dimension reduction for integrated data
            process_anndata_with_pca(
                adata=AnnData_integrated,
                pseudobulk=pseudobulk_dict,
                pseudobulk_anndata=pseudobulk_adata,
                sample_col=sample_col,
                n_expression_pcs=num_DR_components,
                n_proportion_pcs=num_DR_components,
                atac=False,  # For integrated data, use RNA processing
                output_dir=resolution_dir,
                not_save=True,
                verbose=False
            )
            
            # Generate null distributions for BOTH modalities if computing p-values
            if compute_pvalues:
                for modality in ['rna', 'atac']:
                    try:
                        null_distribution = generate_null_distribution(
                            pseudobulk_adata=pseudobulk_adata,
                            modality=modality.upper(),
                            column=f'X_DR_{dr_type}',
                            sev_col=sev_col,
                            n_pcs=n_pcs,
                            n_permutations=num_pvalue_simulations,
                            save_path=os.path.join(resolution_dir, f'null_dist_{modality}_{dr_type}.npy'),
                            verbose=verbose and (modality == optimization_target)  # Only verbose for target
                        )
                        resolution_null_results[modality]['null_scores'] = null_distribution
                    except Exception as e:
                        print(f"Warning: Failed to generate null distribution for {modality} at resolution {resolution:.2f}: {str(e)}")
                        resolution_null_results[modality]['null_scores'] = None
            
            # Compute CCA for both modalities and both DR types using batch analysis
            dr_columns = ['X_DR_expression', 'X_DR_proportion']
            cca_results_df = batch_cca_analysis(
                pseudobulk_adata=pseudobulk_adata,
                dr_columns=dr_columns,
                sev_col=sev_col,
                modalities=['RNA', 'ATAC'],
                n_components=1,
                n_pcs=n_pcs,
                output_dir=resolution_dir
            )
            
            # Extract results into result_dict
            for _, row in cca_results_df.iterrows():
                modality = row['modality'].lower()
                dr_method = row['column'].replace('X_DR_', '')
                result_dict[f'{modality}_cca_{dr_method}'] = row['cca_score']
                
                if row['valid'] and not np.isnan(row['cca_score']):
                    print(f"{row['modality']} {dr_method} CCA Score: {row['cca_score']:.4f}")
            
            # Set optimization score based on the specified target modality and DR type
            target_metric = f"{optimization_target}_cca_{dr_type}"
            if target_metric in result_dict:
                result_dict['optimization_score'] = result_dict[target_metric]
            else:
                result_dict['optimization_score'] = np.nan
            
            print(f"Resolution {resolution:.2f}: Target {optimization_target.upper()} {dr_type} CCA Score = {result_dict['optimization_score']:.4f}")
            
            # Always create embedding visualizations
            try:
                # Create visualizations for both RNA and ATAC
                for viz_modality in ['RNA', 'ATAC']:
                    embedding_path = os.path.join(
                        resolution_dir, 
                        f"embedding_{viz_modality}_{dr_type}"
                    )
                    
                    # Create visualization for this modality
                    visualize_multimodal_embedding_with_cca(
                        adata=pseudobulk_adata,
                        modality_col=modality_col,
                        color_col=sev_col,
                        target_modality=viz_modality,  # Visualize current modality
                        cca_results_df=cca_results_df,
                        output_dir=embedding_path,
                        show_sample_names=False,
                        verbose=False
                    )
                    
                    if verbose:
                        print(f"Created {viz_modality} embedding visualization")
                        
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to create embedding visualization: {str(e)}")
            
            # Save resolution-specific results
            resolution_results_path = os.path.join(resolution_dir, f"results_res_{resolution:.2f}.csv")
            pd.DataFrame([result_dict]).to_csv(resolution_results_path, index=False)
                
        except Exception as e:
            print(f"Error at resolution {resolution:.2f}: {str(e)}")
        
        all_results.append(result_dict)
        all_resolution_null_results['rna'].append(resolution_null_results['rna'])
        all_resolution_null_results['atac'].append(resolution_null_results['atac'])

    # Find best resolution from first pass
    coarse_results = [r for r in all_results if not np.isnan(r['optimization_score'])]
    if not coarse_results:
        raise ValueError("No valid optimization scores obtained in coarse search.")
    
    best_coarse = max(coarse_results, key=lambda x: x['optimization_score'])
    best_resolution = best_coarse['resolution']
    print(f"\nBest resolution from first pass: {best_resolution:.2f}")
    print(f"Best {optimization_target.upper()} {dr_type} CCA score: {best_coarse['optimization_score']:.4f}")

    # Second pass: fine-tuned search
    print("\n=== SECOND PASS: Fine-tuned Search ===")
    search_range_start = max(0.01, best_resolution - 0.05)
    search_range_end = min(1.00, best_resolution + 0.05)
    
    print(f"Fine-tuning search from {search_range_start:.2f} to {search_range_end:.2f}...")

    for resolution in np.arange(search_range_start, search_range_end + 0.001, 0.01):
        resolution = round(resolution, 3)
        
        # Skip if already tested in coarse search
        if any(abs(r['resolution'] - resolution) < 0.001 for r in all_results):
            continue
        
        print(f"\nTesting fine-tuned resolution: {resolution:.3f}")
        
        # Create resolution-specific directory
        resolution_dir = os.path.join(main_output_dir, f"resolution_{resolution:.3f}")
        os.makedirs(resolution_dir, exist_ok=True)
        
        result_dict = {
            'resolution': resolution,
            'rna_cca_expression': np.nan,
            'rna_cca_proportion': np.nan,
            'atac_cca_expression': np.nan,
            'atac_cca_proportion': np.nan,
            'rna_pvalue_expression': np.nan,
            'rna_pvalue_proportion': np.nan,
            'atac_pvalue_expression': np.nan,
            'atac_pvalue_proportion': np.nan,
            'optimization_score': np.nan,
            'pass': 'fine'
        }
        
        # Initialize null results for this resolution for both modalities
        resolution_null_results = {
            'rna': {'resolution': resolution, 'null_scores': None},
            'atac': {'resolution': resolution, 'null_scores': None}
        }
        
        try:
            # Clean up previous cell type assignments
            if 'cell_type' in AnnData_integrated.obs.columns:
                AnnData_integrated.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            
            # Perform clustering
            AnnData_integrated = cell_types_linux(
                AnnData_integrated,
                cell_column='cell_type',
                existing_cell_types=False,
                Save=False,
                output_dir=resolution_dir,
                cluster_resolution=resolution,
                use_rep=use_rep,
                markers=None,
                num_PCs=num_PCs,
                verbose=False
            )
            
            # Compute pseudobulk data
            pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata(
                adata=AnnData_integrated, 
                batch_col=batch_col, 
                sample_col=sample_col, 
                celltype_col='cell_type', 
                n_features=n_features, 
                output_dir=resolution_dir,
                Save=False,
                verbose=False
            )
            
            # Perform dimension reduction
            process_anndata_with_pca(
                adata=AnnData_integrated,
                pseudobulk=pseudobulk_dict,
                pseudobulk_anndata=pseudobulk_adata,
                sample_col=sample_col,
                n_expression_pcs=num_DR_components,
                n_proportion_pcs=num_DR_components,
                atac=False,
                output_dir=resolution_dir,
                not_save=True,
                verbose=False
            )
            
            # Generate null distributions for BOTH modalities if computing p-values
            if compute_pvalues:
                for modality in ['rna', 'atac']:
                    try:
                        null_distribution = generate_null_distribution(
                            pseudobulk_adata=pseudobulk_adata,
                            modality=modality.upper(),
                            column=f'X_DR_{dr_type}',
                            sev_col=sev_col,
                            n_pcs=n_pcs,
                            n_permutations=num_pvalue_simulations,
                            save_path=os.path.join(resolution_dir, f'null_dist_{modality}_{dr_type}.npy'),
                            verbose=verbose and (modality == optimization_target)  # Only verbose for target
                        )
                        resolution_null_results[modality]['null_scores'] = null_distribution
                    except Exception as e:
                        print(f"Warning: Failed to generate null distribution for {modality} at resolution {resolution:.3f}: {str(e)}")
                        resolution_null_results[modality]['null_scores'] = None
            
            # Compute CCA for both modalities and DR types using batch analysis
            dr_columns = ['X_DR_expression', 'X_DR_proportion']
            cca_results_df = batch_cca_analysis(
                pseudobulk_adata=pseudobulk_adata,
                dr_columns=dr_columns,
                sev_col=sev_col,
                modalities=['RNA', 'ATAC'],
                n_components=1,
                n_pcs=n_pcs,
                output_dir=resolution_dir
            )
            
            # Extract results into result_dict
            for _, row in cca_results_df.iterrows():
                modality = row['modality'].lower()
                dr_method = row['column'].replace('X_DR_', '')
                result_dict[f'{modality}_cca_{dr_method}'] = row['cca_score']
            
            # Set optimization score based on target
            target_metric = f"{optimization_target}_cca_{dr_type}"
            if target_metric in result_dict:
                result_dict['optimization_score'] = result_dict[target_metric]
            else:
                result_dict['optimization_score'] = np.nan
            
            print(f"Fine-tuned Resolution {resolution:.3f}: Target Score {result_dict['optimization_score']:.4f}")
            
            # Always create embedding visualizations
            try:
                for viz_modality in ['RNA', 'ATAC']:
                    embedding_path = os.path.join(
                        resolution_dir, 
                        f"embedding_{viz_modality}_{dr_type}"
                    )
                    
                    # Create visualization for this modality
                    visualize_multimodal_embedding_with_cca(
                        adata=pseudobulk_adata,
                        modality_col=modality_col,
                        color_col=sev_col,
                        target_modality=viz_modality,  # Visualize current modality
                        cca_results_df=cca_results_df,
                        output_dir=embedding_path,
                        show_sample_names=False,
                        verbose=False
                    )
                    
                    if verbose:
                        print(f"Created {viz_modality} embedding visualization")
                        
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to create embedding visualization: {str(e)}")
            
            # Save resolution-specific results
            resolution_results_path = os.path.join(resolution_dir, f"results_res_{resolution:.3f}.csv")
            pd.DataFrame([result_dict]).to_csv(resolution_results_path, index=False)
                    
        except Exception as e:
            print(f"Error at fine-tuned resolution {resolution:.3f}: {str(e)}")
        
        all_results.append(result_dict)
        all_resolution_null_results['rna'].append(resolution_null_results['rna'])
        all_resolution_null_results['atac'].append(resolution_null_results['atac'])

    # Create comprehensive results dataframe BEFORE using it
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("resolution")

    # Generate corrected null distributions for BOTH modalities if computing p-values
    corrected_null_distributions = {}
    if compute_pvalues:
        print("\n=== GENERATING CORRECTED NULL DISTRIBUTIONS ===")
        print("Accounting for resolution selection bias...")
        
        for modality in ['rna', 'atac']:
            # Filter out null results that failed to generate
            valid_null_results = [r for r in all_resolution_null_results[modality] if r['null_scores'] is not None]
            
            if valid_null_results:
                print(f"\nGenerating corrected null distribution for {modality.upper()}...")
                corrected_null_distributions[modality] = generate_corrected_null_distribution_for_modality(
                    all_resolution_results=valid_null_results,
                    modality=modality,
                    dr_type=dr_type,
                    n_permutations=num_pvalue_simulations
                )
                
                # Save corrected null distribution
                summary_dir = os.path.join(main_output_dir, "summary")
                os.makedirs(summary_dir, exist_ok=True)
                corrected_null_path = os.path.join(summary_dir, f'corrected_null_distribution_{modality}_{dr_type}.npy')
                np.save(corrected_null_path, corrected_null_distributions[modality])
                print(f"Corrected null distribution for {modality.upper()} saved to: {corrected_null_path}")
            else:
                print(f"Warning: No valid null distributions generated for {modality.upper()}")
        
        if corrected_null_distributions:
            # Compute corrected p-values for all CCA scores and create visualization plots
            print("\n=== COMPUTING CORRECTED P-VALUES AND CREATING PLOTS ===")
            compute_all_corrected_pvalues_and_plots(
                df_results=df_results,
                corrected_null_distributions=corrected_null_distributions,
                main_output_dir=main_output_dir,
                optimization_target=optimization_target,
                dr_type=dr_type
            )
    
    # Find final best resolution based on target metric
    valid_results = df_results[~df_results['optimization_score'].isna()]
    if valid_results.empty:
        raise ValueError("No valid results obtained.")
    
    final_best_idx = valid_results['optimization_score'].idxmax()
    final_best_resolution = valid_results.loc[final_best_idx, 'resolution']
    final_best_score = valid_results.loc[final_best_idx, 'optimization_score']
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Optimization Target: {optimization_target.upper()} {dr_type.upper()}")
    print(f"Best resolution: {final_best_resolution:.3f}")
    print(f"Best CCA score: {final_best_score:.4f}")
    
    # Get corrected p-value if available
    corrected_pval_col = f"{optimization_target}_corrected_pvalue_{dr_type}"
    if corrected_pval_col in df_results.columns:
        corrected_p_value = df_results.loc[df_results['resolution'] == final_best_resolution, corrected_pval_col].iloc[0]
        if not pd.isna(corrected_p_value):
            print(f"Corrected p-value: {corrected_p_value:.4f}")
    
    # Display all scores at best resolution for context
    best_row = valid_results.loc[final_best_idx]
    print(f"\nAll CCA scores at best resolution {final_best_resolution:.3f}:")
    print(f"  RNA Expression: {best_row['rna_cca_expression']:.4f}")
    print(f"  RNA Proportion: {best_row['rna_cca_proportion']:.4f}")
    print(f"  ATAC Expression: {best_row['atac_cca_expression']:.4f}")
    print(f"  ATAC Proportion: {best_row['atac_cca_proportion']:.4f}")

    # Create summary directory for final results
    summary_dir = os.path.join(main_output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    # Save comprehensive results
    results_csv_path = os.path.join(summary_dir, f"resolution_scores_comprehensive_integration_{optimization_target}_{dr_type}.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nComprehensive results saved to: {results_csv_path}")

    create_resolution_optimization_summary(
        df_results=df_results,
        final_best_resolution=final_best_resolution,
        optimization_target=optimization_target,
        dr_type=dr_type,
        summary_dir=summary_dir
    )

    print(f"\n[Find Optimal Resolution Integration] Total runtime: {time.time() - start_time:.2f} seconds\n")

    return final_best_resolution, df_results

if __name__ == "__main__":
    integrated_adata = ad.read_h5ad("/dcl01/hongkai/data/data/hjiang/result/integration/preprocess/atac_rna_integrated.h5ad")
    output_dir = "/dcl01/hongkai/data/data/hjiang/result/integration"
    
    # integrated_adata = ad.read_h5ad("/dcl01/hongkai/data/data/hjiang/result/integration_test/subsample/atac_rna_integrated_subsampled_10pct.h5ad")
    # output_dir = "/dcl01/hongkai/data/data/hjiang/result/integration_test/subsample"
    suppress_warnings()

    try:
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator
        import cupy as cp
        
        rmm.reinitialize(
            managed_memory=True,
            pool_allocator=False,
        )
        cp.cuda.set_allocator(rmm_cupy_allocator)
    except:
        pass

    import torch
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"GPU Memory Available: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    optimal_res, results_df = find_optimal_cell_resolution_integration(
        AnnData_integrated=integrated_adata,
        output_dir=output_dir,
        optimization_target="atac",  # "rna" or "atac"
        dr_type="proportion",  # "expression" or "proportion"
        n_features=2000,
        sev_col="sev.level",
        batch_col="batch",
        sample_col="sample",
        modality_col="modality",
        use_rep='X_glue',
        num_DR_components=30,
        num_PCs=30,
        num_pvalue_simulations=1000,
        compute_pvalues=True,
        visualize_embeddings=True,
        n_pcs = 10,
        verbose=True
    )