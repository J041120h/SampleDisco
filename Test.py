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
from integration_visualization import visualize_multimodal_embedding
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
    

def cca_analysis(pseudobulk_adata, modality, column, sev_col, 
                            n_components=2, null_distribution=None):
    """
    Simplified CCA analysis using precomputed null distribution for p-value calculation.
    
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
    null_distribution : np.array, optional
        Precomputed null distribution of CCA scores for p-value calculation
        
    Assumptions based on fixed data type:
    - pseudobulk_adata.uns[column] contains pandas DataFrame with numeric coordinates
    - modality column exists and contains string values
    - severity column exists and contains numeric values
    - Data is already preprocessed and clean
    """
    
    result = {
        'cca_score': np.nan,
        'p_value': np.nan,
        'n_samples': 0,
        'n_features': 0,
        'modality': modality,
        'column': column,
        'valid': False,
        'error_message': None
    }
    
    try:
        # Extract modality samples (simplified - no categorical conversion needed)
        modality_mask = pseudobulk_adata.obs['modality'] == modality
        if not modality_mask.any():
            result['error_message'] = f"No samples found for modality: {modality}"
            return result
        
        # Get dimension reduction coordinates (simplified - assume DataFrame format)
        dr_coords = pseudobulk_adata.uns[column].loc[modality_mask].copy()
        
        # Get severity levels (simplified - assume numeric)
        sev_levels = pseudobulk_adata.obs.loc[modality_mask, sev_col].values
        
        # Basic validation only
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
        
        # Limit components based on data dimensions
        max_components = min(X.shape[1], X.shape[0] - 1)
        n_components_actual = min(n_components, max_components)
        
        if n_components_actual < 1:
            result['error_message'] = "Cannot compute CCA components"
            return result
        
        # Standardize data (simplified)
        from sklearn.preprocessing import StandardScaler
        from sklearn.cross_decomposition import CCA
        
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
        
        result['cca_score'] = cca_score
        
        # Calculate p-value using precomputed null distribution
        if null_distribution is not None:
            result['p_value'] = np.mean(null_distribution >= cca_score)
        else:
            result['p_value'] = np.nan
        
        result['valid'] = True
        
    except Exception as e:
        result['error_message'] = f"CCA failed: {str(e)}"
    
    return result


def batch_cca_analysis(pseudobulk_adata, dr_columns, sev_col, modalities=None, 
                      n_components=2, null_distribution=None):
    """
    Run CCA analysis across multiple dimension reduction results and modalities.
    
    Parameters:
    -----------
    pseudobulk_adata : sc.AnnData
        AnnData object containing dimension reduction results
    dr_columns : list
        List of dimension reduction column names to analyze (e.g., ['X_DR_expression', 'X_DR_proportion'])
    sev_col : str
        Column name for severity levels
    modalities : list, optional
        List of modalities to analyze. If None, uses all unique modalities
    n_components : int, default 2
        Number of CCA components
    null_distribution : np.array, optional
        Precomputed null distribution of CCA scores for p-value calculation
        
    Returns:
    --------
    pd.DataFrame
        Results DataFrame with CCA scores and p-values for each modality-column combination
    """
    
    if modalities is None:
        modalities = pseudobulk_adata.obs['modality'].unique()
    
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
                    null_distribution=null_distribution
                )
                results.append(result)
            else:
                # Add placeholder for missing data
                results.append({
                    'cca_score': np.nan,
                    'p_value': np.nan,
                    'n_samples': 0,
                    'n_features': 0,
                    'modality': modality,
                    'column': column,
                    'valid': False,
                    'error_message': f"Column '{column}' not found"
                })
    
    return pd.DataFrame(results)

def generate_null_distribution(pseudobulk_adata, modality, column, sev_col, 
                             n_components=2, n_trials=1000, simulations_per_trial=10, 
                             save_path=None, plot=True, verbose=True):
    
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_decomposition import CCA
    
    if verbose:
        print(f"Generating null distribution with {n_trials} trials, {simulations_per_trial} simulations per trial...")
    
    # Extract data (same as in simplified_cca_analysis)
    modality_mask = pseudobulk_adata.obs['modality'] == modality
    if not modality_mask.any():
        raise ValueError(f"No samples found for modality: {modality}")
    
    dr_coords = pseudobulk_adata.uns[column].loc[modality_mask].copy()
    sev_levels = pseudobulk_adata.obs.loc[modality_mask, sev_col].values
    
    if len(dr_coords) < 3:
        raise ValueError(f"Insufficient samples: {len(dr_coords)}")
    
    if len(np.unique(sev_levels)) < 2:
        raise ValueError("Insufficient severity level variance")
    
    # Prepare data
    X = dr_coords.values
    y_original = sev_levels.reshape(-1, 1)
    
    # Limit components
    max_components = min(X.shape[1], X.shape[0] - 1)
    n_components_actual = min(n_components, max_components)
    
    if n_components_actual < 1:
        raise ValueError("Cannot compute CCA components")
    
    # Pre-scale X (this won't change across permutations)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    null_scores = []
    
    for trial in range(n_trials):
        if verbose and (trial + 1) % 100 == 0:
            print(f"  Completed {trial + 1}/{n_trials} trials...")
        
        trial_scores = []
        
        # Run multiple simulations per trial
        for sim in range(simulations_per_trial):
            try:
                # Permute severity levels
                permuted_y = np.random.permutation(sev_levels).reshape(-1, 1)
                
                # Scale permuted y
                scaler_y = StandardScaler()
                permuted_y_scaled = scaler_y.fit_transform(permuted_y)
                
                # Fit CCA
                perm_cca = CCA(n_components=n_components_actual, max_iter=1000, tol=1e-6)
                perm_cca.fit(X_scaled, permuted_y_scaled)
                
                # Transform and compute correlation
                X_c_perm, y_c_perm = perm_cca.transform(X_scaled, permuted_y_scaled)
                perm_correlation = np.corrcoef(X_c_perm[:, 0], y_c_perm[:, 0])[0, 1]
                
                if not np.isnan(perm_correlation):
                    trial_scores.append(abs(perm_correlation))
                else:
                    trial_scores.append(0.0)
                    
            except Exception:
                trial_scores.append(0.0)
        
        # Select the best (highest) score from this trial
        if trial_scores:
            null_scores.append(max(trial_scores))
        else:
            null_scores.append(0.0)
    
    null_distribution = np.array(null_scores)

    if verbose:
            print(f"Simulation complete - generated {len(null_distribution)} null scores.")
            
    if save_path:
        np.save(save_path, null_distribution)
        if verbose:
            print(f"Saved null distribution to: {save_path}")
    
    # Generate plot
    if plot and save_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(null_distribution, bins=50, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black', linewidth=0.5)
        ax1.axvline(null_distribution.mean(), color='red', linestyle='--', 
                   label=f'Mean: {null_distribution.mean():.3f}')
        ax1.axvline(np.percentile(null_distribution, 95), color='orange', linestyle='--', 
                   label=f'95th percentile: {np.percentile(null_distribution, 95):.3f}')
        ax1.axvline(np.percentile(null_distribution, 99), color='purple', linestyle='--', 
                   label=f'99th percentile: {np.percentile(null_distribution, 99):.3f}')
        ax1.set_xlabel('CCA Score')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Null Distribution\n({n_trials} trials, best of {simulations_per_trial} each)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = save_path.replace('.npy', '_distribution_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Saved distribution plot to: {plot_path}")
        
        plt.show()
    
    return null_distribution

def create_modality_pvalue_plot(simulated_scores, observed_correlation, p_value, 
                               modality, dr_type, output_dir):
    """
    Create permutation test visualization plot
    
    Parameters:
    -----------
    simulated_scores : array
        Array of simulated correlation scores
    observed_correlation : float
        Observed CCA correlation
    p_value : float
        Calculated p-value
    modality : str
        Modality name (RNA/ATAC)
    dr_type : str
        DR type (expression/proportion)
    output_dir : str
        Output directory for saving plot
    """
    plt.figure(figsize=(8, 5))
    plt.hist(simulated_scores, bins=30, alpha=0.7, edgecolor='black', density=True)
    plt.axvline(observed_correlation, color='red', linestyle='dashed', linewidth=2,
                label=f'Observed: {observed_correlation:.3f} (p={p_value:.4f})')
    
    # Add percentile lines
    percentiles = [95, 99]
    for p in percentiles:
        threshold = np.percentile(simulated_scores, p)
        plt.axvline(threshold, color='orange', linestyle=':', alpha=0.5,
                   label=f'{p}th percentile: {threshold:.3f}')
    
    plt.xlabel('CCA Correlation Score')
    plt.ylabel('Density')
    plt.title(f'{modality} {dr_type.capitalize()} - Permutation Test (n={len(simulated_scores)})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, f"pvalue_dist_{modality}_{dr_type}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary statistics
    stats_path = os.path.join(output_dir, f"pvalue_stats_{modality}_{dr_type}.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Modality: {modality}\n")
        f.write(f"DR Type: {dr_type}\n")
        f.write(f"Observed correlation: {observed_correlation:.4f}\n")
        f.write(f"P-value: {p_value:.4f}\n")
        f.write(f"Number of simulations: {len(simulated_scores)}\n")
        f.write(f"Simulated mean: {np.mean(simulated_scores):.4f}\n")
        f.write(f"Simulated std: {np.std(simulated_scores):.4f}\n")
        f.write(f"Simulated min: {np.min(simulated_scores):.4f}\n")
        f.write(f"Simulated max: {np.max(simulated_scores):.4f}\n")
        
# Additional utility functions for data quality checks
def check_data_quality(adata, modality_col='modality', sev_col='sev.level'):
    """
    Check data quality before running CCA analysis
    """
    print("=== Data Quality Check ===")
    
    # Check for missing values
    print(f"Missing values in {sev_col}: {adata.obs[sev_col].isna().sum()}")
    print(f"Missing values in {modality_col}: {adata.obs[modality_col].isna().sum()}")
    
    # Check severity level distribution
    print(f"\nSeverity level distribution:")
    print(adata.obs[sev_col].value_counts())
    
    # Check modality distribution
    print(f"\nModality distribution:")
    print(adata.obs[modality_col].value_counts())
    
    # Check for variance in severity levels by modality
    for modality in adata.obs[modality_col].unique():
        modality_mask = adata.obs[modality_col] == modality
        modality_sev = adata.obs.loc[modality_mask, sev_col]
        if not modality_sev.empty:
            print(f"\n{modality} - Severity variance: {modality_sev.var():.6f}")
            print(f"{modality} - Unique severity levels: {modality_sev.nunique()}")

def ensure_non_categorical_columns(adata, columns):
    """Convert specified columns from categorical to string to avoid categorical errors"""
    for col in columns:
        if col in adata.obs.columns:
            if pd.api.types.is_categorical_dtype(adata.obs[col]):
                adata.obs[col] = adata.obs[col].astype(str)
    return adata

def save_comprehensive_cca_summary(df_results, optimization_target, dr_type, output_dir):
    """
    Save comprehensive CCA summary for all modalities and DR types
    """
    summary_path = os.path.join(output_dir, f"CCA_scores_comprehensive_summary_{optimization_target}_{dr_type}.txt")
    
    with open(summary_path, "w") as f:
        f.write(f"Comprehensive CCA Scores Summary - Integration ({optimization_target.upper()} {dr_type.upper()})\n")
        f.write("="*80 + "\n\n")
        
        # Summary statistics for each metric
        metrics = ['rna_cca_expression', 'rna_cca_proportion', 'atac_cca_expression', 'atac_cca_proportion']
        metric_names = ['RNA Expression', 'RNA Proportion', 'ATAC Expression', 'ATAC Proportion']
        
        for metric, name in zip(metrics, metric_names):
            valid_data = df_results[~df_results[metric].isna()]
            if not valid_data.empty:
                f.write(f"{name} CCA Scores:\n")
                f.write(f"  Mean: {valid_data[metric].mean():.4f}\n")
                f.write(f"  Std:  {valid_data[metric].std():.4f}\n")
                f.write(f"  Min:  {valid_data[metric].min():.4f} (at resolution {valid_data.loc[valid_data[metric].idxmin(), 'resolution']:.3f})\n")
                f.write(f"  Max:  {valid_data[metric].max():.4f} (at resolution {valid_data.loc[valid_data[metric].idxmax(), 'resolution']:.3f})\n")
                f.write(f"  Valid measurements: {len(valid_data)}\n\n")
        
        # Highlight the optimization target
        f.write(f"OPTIMIZATION TARGET: {optimization_target.upper()} {dr_type.upper()}\n")
        f.write("-" * 50 + "\n")
        target_metric = f"{optimization_target}_cca_{dr_type}"
        if target_metric in df_results.columns:
            valid_target = df_results[~df_results[target_metric].isna()]
            if not valid_target.empty:
                f.write(f"Target metric ({target_metric}) statistics:\n")
                f.write(f"  Best score: {valid_target[target_metric].max():.4f} (at resolution {valid_target.loc[valid_target[target_metric].idxmax(), 'resolution']:.3f})\n")
                f.write(f"  Mean score: {valid_target[target_metric].mean():.4f}\n")
                f.write(f"  Std: {valid_target[target_metric].std():.4f}\n\n")
        
        # Top 5 resolutions for each metric
        f.write("Top 5 Resolutions for Each Metric:\n")
        f.write("-" * 40 + "\n")
        
        for metric, name in zip(metrics, metric_names):
            valid_data = df_results[~df_results[metric].isna()]
            if not valid_data.empty:
                top5 = valid_data.nlargest(5, metric)
                f.write(f"\n{name}:\n")
                for idx, row in top5.iterrows():
                    f.write(f"  Resolution {row['resolution']:.3f}: {row[metric]:.4f}\n")
        
        # Correlation between metrics
        f.write("\n\nCorrelation Matrix Between CCA Scores:\n")
        f.write("-" * 40 + "\n")
        correlation_matrix = df_results[metrics].corr()
        f.write(correlation_matrix.to_string())
        f.write("\n")
    
    print(f"Comprehensive CCA summary saved to: {summary_path}")

def save_detailed_pvalue_summary(df_results, optimization_target, dr_type, output_dir):
    """
    Save detailed p-value summary including all resolutions for the chosen modality and DR type
    """
    pvalue_summary_path = os.path.join(output_dir, f"pvalue_summary_integration_{optimization_target}_{dr_type}.txt")
    
    # Find best resolution
    valid_results = df_results[~df_results['optimization_score'].isna()]
    if valid_results.empty:
        return
    
    final_best_idx = valid_results['optimization_score'].idxmax()
    final_best_resolution = valid_results.loc[final_best_idx, 'resolution']
    final_best_score = valid_results.loc[final_best_idx, 'optimization_score']
    
    with open(pvalue_summary_path, "w") as f:
        f.write(f"Detailed P-value Summary for Integration ({optimization_target.upper()} {dr_type.upper()})\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Optimization Target: {optimization_target.upper()} {dr_type.upper()}\n")
        f.write(f"Best Resolution: {final_best_resolution:.3f}\n")
        f.write(f"Best CCA Score: {final_best_score:.4f}\n\n")
        
        # Best resolution p-values
        f.write("P-values at Best Resolution:\n")
        f.write("-" * 30 + "\n")
        f.write(f"  RNA Expression: {valid_results.loc[final_best_idx, 'rna_pvalue_expression']:.4f}\n")
        f.write(f"  RNA Proportion: {valid_results.loc[final_best_idx, 'rna_pvalue_proportion']:.4f}\n")
        f.write(f"  ATAC Expression: {valid_results.loc[final_best_idx, 'atac_pvalue_expression']:.4f}\n")
        f.write(f"  ATAC Proportion: {valid_results.loc[final_best_idx, 'atac_pvalue_proportion']:.4f}\n\n")
        
        # P-values for chosen modality and DR type across all resolutions
        target_pvalue_metric = f"{optimization_target}_pvalue_{dr_type}"
        target_cca_metric = f"{optimization_target}_cca_{dr_type}"
        
        f.write(f"P-values and CCA Scores for {optimization_target.upper()} {dr_type.upper()} Across All Resolutions:\n")
        f.write("="*80 + "\n")
        f.write("Resolution\tCCA Score\tP-value\t\tSignificant (p<0.05)\n")
        f.write("-" * 60 + "\n")
        
        valid_data = df_results[~df_results[target_pvalue_metric].isna()].sort_values('resolution')
        for idx, row in valid_data.iterrows():
            is_significant = "Yes" if row[target_pvalue_metric] < 0.05 else "No"
            cca_score = row[target_cca_metric] if not pd.isna(row[target_cca_metric]) else "N/A"
            f.write(f"{row['resolution']:.3f}\t\t{cca_score:.4f}\t\t{row[target_pvalue_metric]:.4f}\t\t{is_significant}\n")
        
        # Summary statistics for target metric
        f.write(f"\n\nSummary Statistics for {optimization_target.upper()} {dr_type.upper()}:\n")
        f.write("="*60 + "\n")
        
        valid_pvalues = df_results[~df_results[target_pvalue_metric].isna()][target_pvalue_metric]
        valid_cca_scores = df_results[~df_results[target_cca_metric].isna()][target_cca_metric]
        
        if not valid_pvalues.empty:
            significant_count = (valid_pvalues < 0.05).sum()
            f.write(f"P-value Statistics:\n")
            f.write(f"  Total measurements: {len(valid_pvalues)}\n")
            f.write(f"  Significant (p<0.05): {significant_count} ({significant_count/len(valid_pvalues)*100:.1f}%)\n")
            f.write(f"  Mean p-value: {valid_pvalues.mean():.4f}\n")
            f.write(f"  Median p-value: {valid_pvalues.median():.4f}\n")
            f.write(f"  Min p-value: {valid_pvalues.min():.4f}\n")
            f.write(f"  Max p-value: {valid_pvalues.max():.4f}\n\n")
        
        if not valid_cca_scores.empty:
            f.write(f"CCA Score Statistics:\n")
            f.write(f"  Mean CCA score: {valid_cca_scores.mean():.4f}\n")
            f.write(f"  Median CCA score: {valid_cca_scores.median():.4f}\n")
            f.write(f"  Min CCA score: {valid_cca_scores.min():.4f}\n")
            f.write(f"  Max CCA score: {valid_cca_scores.max():.4f}\n")
            f.write(f"  Std CCA score: {valid_cca_scores.std():.4f}\n")
    
    print(f"Detailed P-value summary saved to: {pvalue_summary_path}")

def create_separate_modality_plots(df_results, best_resolution, optimization_target, dr_type, output_dir):
    """
    Create separate plots for each modality's CCA scores and p-values
    """
    valid_df = df_results[~df_results['optimization_score'].isna()]
    coarse_df = valid_df[valid_df['pass'] == 'coarse']
    fine_df = valid_df[valid_df['pass'] == 'fine']
    
    # RNA Modality Plots
    create_rna_modality_plots(coarse_df, fine_df, best_resolution, output_dir)
    
    # ATAC Modality Plots  
    create_atac_modality_plots(coarse_df, fine_df, best_resolution, output_dir)
    
    # Target modality optimization plot
    create_target_optimization_plot(coarse_df, fine_df, best_resolution, optimization_target, dr_type, output_dir)

def create_rna_modality_plots(coarse_df, fine_df, best_resolution, output_dir):
    """Create separate plots for RNA modality"""
    
    # RNA CCA Scores Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # CCA Scores
    ax1.scatter(coarse_df['resolution'], coarse_df['rna_cca_expression'], 
               color='darkblue', s=60, alpha=0.6, marker='o', label='RNA Expression (Coarse)')
    ax1.scatter(fine_df['resolution'], fine_df['rna_cca_expression'], 
               color='darkblue', s=40, alpha=0.8, marker='o', label='RNA Expression (Fine)')
    ax1.scatter(coarse_df['resolution'], coarse_df['rna_cca_proportion'], 
               color='lightblue', s=60, alpha=0.6, marker='^', label='RNA Proportion (Coarse)')
    ax1.scatter(fine_df['resolution'], fine_df['rna_cca_proportion'], 
               color='lightblue', s=40, alpha=0.8, marker='^', label='RNA Proportion (Fine)')
    
    ax1.axvline(x=best_resolution, color='r', linestyle='--', label=f'Best Resolution: {best_resolution:.3f}')
    ax1.set_xlabel("Resolution")
    ax1.set_ylabel("CCA Score")
    ax1.set_title("RNA Modality - CCA Scores")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # P-values
    valid_pval_rna_exp = pd.concat([coarse_df, fine_df])[~pd.concat([coarse_df, fine_df])['rna_pvalue_expression'].isna()]
    valid_pval_rna_prop = pd.concat([coarse_df, fine_df])[~pd.concat([coarse_df, fine_df])['rna_pvalue_proportion'].isna()]
    
    if not valid_pval_rna_exp.empty:
        ax2.scatter(valid_pval_rna_exp['resolution'], valid_pval_rna_exp['rna_pvalue_expression'], 
                   color='darkblue', s=40, alpha=0.7, marker='o', label='RNA Expression')
    if not valid_pval_rna_prop.empty:
        ax2.scatter(valid_pval_rna_prop['resolution'], valid_pval_rna_prop['rna_pvalue_proportion'], 
                   color='lightblue', s=40, alpha=0.7, marker='^', label='RNA Proportion')
    
    ax2.axvline(x=best_resolution, color='r', linestyle='--', label=f'Best Resolution: {best_resolution:.3f}')
    ax2.axhline(y=0.05, color='orange', linestyle=':', label='p=0.05 threshold')
    ax2.set_xlabel("Resolution")
    ax2.set_ylabel("P-value")
    ax2.set_title("RNA Modality - P-values")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    rna_plot_path = os.path.join(output_dir, "RNA_modality_analysis.png")
    plt.savefig(rna_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"RNA modality plot saved to: {rna_plot_path}")

def create_atac_modality_plots(coarse_df, fine_df, best_resolution, output_dir):
    """Create separate plots for ATAC modality"""
    
    # ATAC CCA Scores and P-values Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # CCA Scores
    ax1.scatter(coarse_df['resolution'], coarse_df['atac_cca_expression'], 
               color='darkgreen', s=60, alpha=0.6, marker='o', label='ATAC Expression (Coarse)')
    ax1.scatter(fine_df['resolution'], fine_df['atac_cca_expression'], 
               color='darkgreen', s=40, alpha=0.8, marker='o', label='ATAC Expression (Fine)')
    ax1.scatter(coarse_df['resolution'], coarse_df['atac_cca_proportion'], 
               color='lightgreen', s=60, alpha=0.6, marker='^', label='ATAC Proportion (Coarse)')
    ax1.scatter(fine_df['resolution'], fine_df['atac_cca_proportion'], 
               color='lightgreen', s=40, alpha=0.8, marker='^', label='ATAC Proportion (Fine)')
    
    ax1.axvline(x=best_resolution, color='r', linestyle='--', label=f'Best Resolution: {best_resolution:.3f}')
    ax1.set_xlabel("Resolution")
    ax1.set_ylabel("CCA Score")
    ax1.set_title("ATAC Modality - CCA Scores")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # P-values
    valid_pval_atac_exp = pd.concat([coarse_df, fine_df])[~pd.concat([coarse_df, fine_df])['atac_pvalue_expression'].isna()]
    valid_pval_atac_prop = pd.concat([coarse_df, fine_df])[~pd.concat([coarse_df, fine_df])['atac_pvalue_proportion'].isna()]
    
    if not valid_pval_atac_exp.empty:
        ax2.scatter(valid_pval_atac_exp['resolution'], valid_pval_atac_exp['atac_pvalue_expression'], 
                   color='darkgreen', s=40, alpha=0.7, marker='s', label='ATAC Expression')
    if not valid_pval_atac_prop.empty:
        ax2.scatter(valid_pval_atac_prop['resolution'], valid_pval_atac_prop['atac_pvalue_proportion'], 
                   color='lightgreen', s=40, alpha=0.7, marker='d', label='ATAC Proportion')
    
    ax2.axvline(x=best_resolution, color='r', linestyle='--', label=f'Best Resolution: {best_resolution:.3f}')
    ax2.axhline(y=0.05, color='orange', linestyle=':', label='p=0.05 threshold')
    ax2.set_xlabel("Resolution")
    ax2.set_ylabel("P-value")
    ax2.set_title("ATAC Modality - P-values")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    atac_plot_path = os.path.join(output_dir, "ATAC_modality_analysis.png")
    plt.savefig(atac_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ATAC modality plot saved to: {atac_plot_path}")

def create_target_optimization_plot(coarse_df, fine_df, best_resolution, optimization_target, dr_type, output_dir):
    """Create plot for the target optimization metric"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Target metric CCA scores
    target_metric = f"{optimization_target}_cca_{dr_type}"
    
    ax.scatter(coarse_df['resolution'], coarse_df[target_metric], 
               color='blue', s=60, alpha=0.6, label='Coarse Search')
    ax.scatter(fine_df['resolution'], fine_df[target_metric], 
               color='green', s=40, alpha=0.8, label='Fine Search')
    
    # Connect points with lines
    all_valid_df = pd.concat([coarse_df, fine_df])[~pd.concat([coarse_df, fine_df])[target_metric].isna()].sort_values('resolution')
    if not all_valid_df.empty:
        ax.plot(all_valid_df['resolution'], all_valid_df[target_metric], 
                'k-', linewidth=0.5, alpha=0.3)
    
    ax.axvline(x=best_resolution, color='r', linestyle='--', 
               label=f'Best Resolution: {best_resolution:.3f}')
    
    ax.set_xlabel("Resolution")
    ax.set_ylabel(f"CCA Score")
    ax.set_title(f"{optimization_target.upper()} {dr_type.capitalize()} - CCA Score Optimization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    opt_plot_path = os.path.join(output_dir, f"optimization_target_{optimization_target}_{dr_type}.png")
    plt.savefig(opt_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Target optimization plot saved to: {opt_plot_path}")

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
    num_pvalue_simulations: int = 100,
    compute_pvalues: bool = True,
    visualize_embeddings: bool = True,
    verbose: bool = True
) -> tuple:
    """
    Find optimal clustering resolution for integrated RNA+ATAC data by maximizing 
    CCA correlation between dimension reduction and severity levels.
    
    Parameters:
    -----------
    AnnData_integrated : AnnData
        Integrated AnnData object containing both RNA and ATAC data
    output_dir : str
        Output directory for results
    optimization_target : str
        Which modality to optimize: "rna" or "atac"
    dr_type : str
        Which DR type to optimize: "expression" or "proportion"
    n_features : int
        Number of features for pseudobulk computation
    sev_col : str
        Column name for severity levels in pseudobulk_anndata.obs
    batch_col : str
        Column name for batch information
    sample_col : str
        Column name for sample identifiers
    modality_col : str
        Column name containing modality information (RNA/ATAC)
    use_rep : str
        Representation to use for neighborhood graph
    num_DR_components : int
        Number of dimension reduction components
    num_PCs : int
        Number of PCs for neighborhood graph
    num_pvalue_simulations : int
        Number of simulations for p-value calculation
    compute_pvalues : bool
        Whether to compute p-values for each resolution
    visualize_embeddings : bool
        Whether to create embedding visualizations for each resolution
    verbose : bool
        Whether to print verbose output
        
    Returns:
    --------
    tuple: (optimal_resolution, results_dataframe)
    """
    start_time = time.time()
    
    print("\n\n Finding optimal resolution begins \n\n")

    # Validate optimization target and dr_type
    if optimization_target not in ["rna", "atac"]:
        raise ValueError("optimization_target must be 'rna' or 'atac'")
    
    if dr_type not in ["expression", "proportion"]:
        raise ValueError("dr_type must be 'expression' or 'proportion'")

    # Create subdirectories for different outputs
    main_output_dir = os.path.join(output_dir, f"CCA_resolution_optimization_integration_{optimization_target}_{dr_type}")
    resolution_plots_dir = os.path.join(main_output_dir, "resolution_plots")
    pvalue_results_dir = os.path.join(main_output_dir, "pvalue_results")
    embedding_plots_dir = os.path.join(main_output_dir, "embedding_visualizations")
    
    for dir_path in [main_output_dir, resolution_plots_dir, pvalue_results_dir, embedding_plots_dir]:
        os.makedirs(dir_path, exist_ok=True)

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

    # First pass: coarse search
    print("\n=== FIRST PASS: Coarse Search ===")
    for resolution in np.arange(0.1, 1.01, 0.1):
        print(f"\n\nTesting resolution: {resolution:.2f}\n")
        
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
                output_dir=output_dir,
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
                output_dir=output_dir,
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
                output_dir=output_dir,
                not_save=True,
                verbose=False
            )
            
            # Compute CCA for both modalities and both DR types
            for modality in ['RNA', 'ATAC']:
                for dr_method in ['expression', 'proportion']:
                    column = f'X_DR_{dr_method}'
                    if column in pseudobulk_adata.uns:
                        cca_score, p_value = improved_compute_modality_cca(
                            pseudobulk_adata, modality, column, sev_col
                        )
                        result_dict[f'{modality.lower()}_cca_{dr_method}'] = cca_score
                        result_dict[f'{modality.lower()}_pvalue_{dr_method}'] = p_value
                        
                        if not np.isnan(cca_score):
                            print(f"{modality} {dr_method} CCA Score: {cca_score:.4f}, p-value: {p_value:.4f}")
            
            # Set optimization score based on the specified target modality and DR type
            target_metric = f"{optimization_target}_cca_{dr_type}"
            if target_metric in result_dict:
                result_dict['optimization_score'] = result_dict[target_metric]
            else:
                result_dict['optimization_score'] = np.nan
            
            print(f"Resolution {resolution:.2f}: Target {optimization_target.upper()} {dr_type} CCA Score = {result_dict['optimization_score']:.4f}")
            
            # Create embedding visualizations if requested (only for target modality)
            if visualize_embeddings and not np.isnan(result_dict['optimization_score']):
                try:
                    embedding_path = os.path.join(
                        embedding_plots_dir, 
                        f"embedding_res_{resolution:.2f}_{optimization_target.upper()}_{dr_type}"
                    )
                    visualize_multimodal_embedding(
                        adata=pseudobulk_adata,
                        modality_col=modality_col,
                        color_col=sev_col,
                        target_modality=optimization_target.upper(),
                        output_dir=embedding_path,
                        show_sample_names=False,
                        verbose=False
                    )
                except Exception as e:
                    if verbose:
                        print(f"Warning: Failed to create embedding visualization: {str(e)}")
                
        except Exception as e:
            print(f"Error at resolution {resolution:.2f}: {str(e)}")
        
        all_results.append(result_dict)

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
                output_dir=output_dir,
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
                output_dir=output_dir,
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
                output_dir=output_dir,
                not_save=True,
                verbose=False
            )
            
            # Compute CCA for both modalities and DR types
            for modality in ['RNA', 'ATAC']:
                for dr_method in ['expression', 'proportion']:
                    column = f'X_DR_{dr_method}'
                    if column in pseudobulk_adata.uns:
                        cca_score, p_value = improved_compute_modality_cca(
                            pseudobulk_adata, modality, column, sev_col
                        )
                        result_dict[f'{modality.lower()}_cca_{dr_method}'] = cca_score
                        result_dict[f'{modality.lower()}_pvalue_{dr_method}'] = p_value
            
            # Set optimization score based on target
            target_metric = f"{optimization_target}_cca_{dr_type}"
            if target_metric in result_dict:
                result_dict['optimization_score'] = result_dict[target_metric]
            else:
                result_dict['optimization_score'] = np.nan
            
            print(f"Fine-tuned Resolution {resolution:.3f}: Target Score {result_dict['optimization_score']:.4f}")
                    
        except Exception as e:
            print(f"Error at fine-tuned resolution {resolution:.3f}: {str(e)}")
        
        all_results.append(result_dict)

    # Create comprehensive results dataframe
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("resolution")
    
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
    
    # Display all scores at best resolution for context
    best_row = valid_results.loc[final_best_idx]
    print(f"\nAll CCA scores at best resolution {final_best_resolution:.3f}:")
    print(f"  RNA Expression: {best_row['rna_cca_expression']:.4f}")
    print(f"  RNA Proportion: {best_row['rna_cca_proportion']:.4f}")
    print(f"  ATAC Expression: {best_row['atac_cca_expression']:.4f}")
    print(f"  ATAC Proportion: {best_row['atac_cca_proportion']:.4f}")

    # Save comprehensive results
    results_csv_path = os.path.join(main_output_dir, f"resolution_scores_comprehensive_integration_{optimization_target}_{dr_type}.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nComprehensive results saved to: {results_csv_path}")

    # Create separate modality visualizations
    create_separate_modality_plots(df_results, final_best_resolution, optimization_target, dr_type, main_output_dir)

    # Save comprehensive CCA summary
    save_comprehensive_cca_summary(df_results, optimization_target, dr_type, main_output_dir)

    # Save detailed p-value summary if computed
    if compute_pvalues:
        save_detailed_pvalue_summary(df_results, optimization_target, dr_type, pvalue_results_dir)

    print(f"\n[Find Optimal Resolution Integration] Total runtime: {time.time() - start_time:.2f} seconds\n")

    return final_best_resolution, df_results


if __name__ == "__main__":
    integrated_adata = ad.read_h5ad("/dcl01/hongkai/data/data/hjiang/result/integration_test/preprocess/atac_rna_integrated.h5ad")
    output_dir = "/dcl01/hongkai/data/data/hjiang/result/integration_test/"
    
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
        dr_type="expression",  # "expression" or "proportion"
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
        verbose=True
    )