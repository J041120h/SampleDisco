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
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA, PLSCanonical

import pandas as pd
import numpy as np
import re
from typing import Union

def convert_severity_to_numeric(data: Union[pd.Series, np.ndarray, list], 
                               verbose: bool = True) -> np.ndarray:
    """
    Convert severity level data to numeric values.
    
    Parameters:
    -----------
    data : pd.Series, np.ndarray, or list
        The severity level data to convert
    verbose : bool
        Whether to print conversion details
        
    Returns:
    --------
    numeric_data : np.ndarray
        Converted numeric array
    """
    
    # Convert to pandas Series for easier handling
    if isinstance(data, (np.ndarray, list)):
        data = pd.Series(data)
    elif isinstance(data, pd.Series):
        data = data.copy()
    else:
        raise ValueError("Data must be pandas Series, numpy array, or list")
    
    if verbose:
        print(f"Converting severity data to numeric...")
        print(f"Original unique values: {sorted(data.dropna().unique())}")
    
    # Convert categorical to string first if needed
    if pd.api.types.is_categorical_dtype(data):
        data = data.astype(str)
    
    # Strategy 1: Try direct numeric conversion
    try:
        numeric_data = pd.to_numeric(data, errors='coerce')
        if numeric_data.notna().sum() > 0:
            if verbose:
                print("✓ Direct numeric conversion successful")
            return numeric_data.values
    except:
        pass
    
    # Strategy 2: Extract numbers from strings (e.g., "level_2.5" -> 2.5)
    def extract_number(value):
        if pd.isna(value):
            return np.nan
        value_str = str(value).strip()
        # Find first number in string
        match = re.search(r'-?\d+\.?\d*', value_str)
        if match:
            try:
                return float(match.group())
            except:
                pass
        return np.nan
    
    extracted_numbers = data.apply(extract_number)
    if extracted_numbers.notna().sum() > 0:
        if verbose:
            print("✓ Extracted numbers from strings")
        return extracted_numbers.values
    
    # Strategy 3: Automatic ordinal ranking for categorical data
    unique_values = sorted([v for v in data.dropna().unique() if v is not None])
    
    if len(unique_values) <= 20:  # Only for reasonable number of categories
        ordinal_mapping = {val: i for i, val in enumerate(unique_values)}
        ordinal_values = data.map(ordinal_mapping)
        
        if verbose:
            print("✓ Applied automatic ordinal ranking")
            print(f"Mapping: {ordinal_mapping}")
        
        return ordinal_values.values
    
    # If all else fails, return NaN array
    if verbose:
        print("❌ Could not convert to numeric - returning NaN array")
    
    return np.full(len(data), np.nan)


def convert_adata_severity_column(adata, sev_col: str = "sev.level", verbose: bool = True):
    """
    Convert severity column in AnnData object to numeric values
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing the severity data
    sev_col : str
        Name of the severity column in adata.obs
    verbose : bool
        Whether to print conversion details
    """
    
    if sev_col not in adata.obs.columns:
        raise ValueError(f"Column '{sev_col}' not found in adata.obs")
    
    if verbose:
        print(f"\n=== Converting {sev_col} to numeric ===")
    
    # Convert the severity column
    numeric_data = convert_severity_to_numeric(adata.obs[sev_col], verbose=verbose)
    
    # Update the AnnData object
    adata.obs[sev_col] = numeric_data
    
    if verbose:
        valid_count = (~np.isnan(numeric_data)).sum()
        print(f"Conversion complete: {valid_count}/{len(numeric_data)} values converted")
        if valid_count > 0:
            print(f"Final range: [{np.nanmin(numeric_data):.3f}, {np.nanmax(numeric_data):.3f}]")
            print(f"Variance: {np.nanvar(numeric_data):.6f}")
            
def suppress_warnings():
    """Suppress specific warnings that are expected during CCA analysis"""
    warnings.filterwarnings('ignore', category=UserWarning, 
                          message='.*y residual is constant at iteration.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, 
                          message='.*invalid value encountered in divide.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, 
                          message='.*All-NaN slice encountered.*')

def validate_data_for_cca(X, y, min_variance_threshold=1e-6):
    """
    Validate and preprocess data before CCA analysis
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like  
        Target variable (e.g., severity levels)
    min_variance_threshold : float
        Minimum variance threshold for features
        
    Returns:
    --------
    X_valid, y_valid, is_valid : processed data and validity flag
    """
    X = np.array(X)
    y = np.array(y)
    
    # Check for NaN values
    X_nan_mask = np.isnan(X).any(axis=1)
    y_nan_mask = np.isnan(y)
    valid_mask = ~(X_nan_mask | y_nan_mask)
    
    if valid_mask.sum() < 3:  # Need at least 3 samples
        return None, None, False
    
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    # Check variance in features
    feature_vars = np.var(X_clean, axis=0)
    valid_features = feature_vars > min_variance_threshold
    
    if valid_features.sum() < 2:  # Need at least 2 features with variance
        return None, None, False
    
    X_clean = X_clean[:, valid_features]
    
    # Check variance in target
    if np.var(y_clean) < min_variance_threshold:
        return None, None, False
    
    # Check for constant values
    if len(np.unique(y_clean)) < 2:
        return None, None, False
    
    return X_clean, y_clean, True

def robust_cca_analysis(X, y, n_components=2):
    """
    Perform robust CCA analysis with proper error handling
    
    Parameters:
    -----------
    X : array-like
        Feature matrix (e.g., PCA coordinates)
    y : array-like
        Target variable (e.g., severity levels)
    n_components : int
        Number of CCA components
        
    Returns:
    --------
    cca_score : float
        CCA correlation score (NaN if analysis fails)
    cca_model : CCA model or None
    """
    try:
        # Validate data
        X_valid, y_valid, is_valid = validate_data_for_cca(X, y)
        
        if not is_valid:
            print("Warning: Data validation failed for CCA analysis")
            return np.nan, None
        
        # Reshape y if needed
        if y_valid.ndim == 1:
            y_valid = y_valid.reshape(-1, 1)
        
        # Determine number of components based on data dimensions
        max_components = min(X_valid.shape[1], y_valid.shape[1], X_valid.shape[0] - 1)
        n_components = min(n_components, max_components)
        
        if n_components < 1:
            return np.nan, None
        
        # Standardize data to avoid scaling issues
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X_valid)
        y_scaled = scaler_y.fit_transform(y_valid)
        
        # Check for zero variance after scaling
        if np.any(np.var(X_scaled, axis=0) < 1e-10) or np.any(np.var(y_scaled, axis=0) < 1e-10):
            return np.nan, None
        
        # Perform CCA with error handling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            cca = CCA(n_components=n_components, max_iter=1000, tol=1e-6)
            cca.fit(X_scaled, y_scaled)
            
            # Transform data
            X_c, y_c = cca.transform(X_scaled, y_scaled)
            
            # Calculate correlation for first canonical component
            if X_c.shape[1] > 0 and y_c.shape[1] > 0:
                correlation = np.corrcoef(X_c[:, 0], y_c[:, 0])[0, 1]
                
                # Return absolute correlation as CCA score
                cca_score = abs(correlation) if not np.isnan(correlation) else np.nan
            else:
                cca_score = np.nan
                
        return cca_score, cca
        
    except Exception as e:
        print(f"CCA analysis failed: {str(e)}")
        return np.nan, None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
import os

def improved_compute_modality_cca(pseudobulk_adata, modality, column, sev_col, 
                                  num_pvalue_simulations=1000, output_dir=None, verbose=False):
    """
    Improved version of compute_modality_cca function with proper PCA coordinate filtering
    and p-value calculation
    """
    try:
        # Ensure modality column is not categorical
        modality_col = 'modality'  # Assuming this is your modality column name
        if modality_col in pseudobulk_adata.obs.columns:
            if pd.api.types.is_categorical_dtype(pseudobulk_adata.obs[modality_col]):
                pseudobulk_adata.obs[modality_col] = pseudobulk_adata.obs[modality_col].astype(str)
        
        # Filter for specific modality
        modality_mask = pseudobulk_adata.obs[modality_col] == modality
        if not any(modality_mask):
            print(f"No samples found for modality: {modality}")
            return np.nan, np.nan
        
        # Get positions of the modality samples in the original data
        modality_positions = [i for i, is_modality in enumerate(modality_mask) if is_modality]
        
        # Filter the AnnData object
        modality_adata = pseudobulk_adata[modality_mask].copy()
        
        # CRITICAL: Filter PCA coordinates in .uns to match the filtered samples
        if column in pseudobulk_adata.uns:
            original_pca = pseudobulk_adata.uns[column]
            
            # Handle different data types for PCA coordinates
            if isinstance(original_pca, pd.DataFrame):
                # For pandas DataFrame
                modality_adata.uns[column] = original_pca.iloc[modality_positions].copy()
            elif isinstance(original_pca, np.ndarray):
                # For numpy array
                modality_adata.uns[column] = original_pca[modality_positions].copy()
            else:
                # Try generic indexing
                try:
                    modality_adata.uns[column] = original_pca[modality_positions]
                except Exception as e:
                    print(f"Warning: Could not filter PCA coordinates for {column}: {e}")
                    return np.nan, np.nan
            
            if verbose:
                print(f"Filtered {column} from {original_pca.shape[0]} to {modality_adata.uns[column].shape[0]} samples for {modality}")
        else:
            print(f"Column {column} not found in .uns for modality {modality}")
            return np.nan, np.nan
        
        # Validate dimensions match
        if modality_adata.uns[column].shape[0] != modality_adata.n_obs:
            print(f"Error: PCA coordinates ({modality_adata.uns[column].shape[0]}) don't match number of samples ({modality_adata.n_obs})")
            return np.nan, np.nan
        
        if sev_col not in modality_adata.obs.columns:
            print(f"Severity column {sev_col} not found for modality {modality}")
            return np.nan, np.nan
        
        # Get PCA coordinates and severity levels
        pca_coords = modality_adata.uns[column]
        if isinstance(pca_coords, pd.DataFrame):
            X = pca_coords.values
        else:
            X = np.array(pca_coords)
        
        # Get severity levels
        sev_levels = modality_adata.obs[sev_col].values
        
        # Convert severity levels to numeric if they're categorical/string
        if not pd.api.types.is_numeric_dtype(sev_levels):
            # Try to convert to numeric
            try:
                sev_levels = pd.to_numeric(sev_levels, errors='coerce')
            except:
                # Create numeric mapping for categorical data
                unique_levels = sorted(set(sev_levels))
                level_mapping = {level: i for i, level in enumerate(unique_levels)}
                sev_levels = np.array([level_mapping.get(level, np.nan) for level in sev_levels])
        
        # Perform robust CCA analysis
        cca_score, cca_model = robust_cca_analysis(X, sev_levels)
        
        # Calculate p-value if CCA score is valid
        if not np.isnan(cca_score) and num_pvalue_simulations > 0:
            p_value, simulated_scores = compute_cca_pvalue_with_plot(
                modality_adata, column, cca_score, sev_col, 
                num_pvalue_simulations, modality, column.split('_')[-1], 
                output_dir, verbose
            )
        else:
            p_value = np.nan
        
        return cca_score, p_value
        
    except Exception as e:
        print(f"Error in CCA computation for {modality}: {str(e)}")
        return np.nan, np.nan


def compute_cca_pvalue_with_plot(modality_adata, column, observed_correlation, sev_col, 
                                num_simulations=1000, modality="", dr_type="", 
                                output_dir=None, verbose=False):
    """
    Compute p-value for CCA correlation using permutation test and create visualization
    
    Returns:
    --------
    tuple : (p-value, simulated_scores)
    """
    try:
        # Get PCA coordinates
        pca_coords = modality_adata.uns[column]
        if isinstance(pca_coords, pd.DataFrame):
            pca_values = pca_coords.values
        else:
            pca_values = np.array(pca_coords)
        
        # Get severity levels and ensure they're numeric
        sev_levels = modality_adata.obs[sev_col].values
        
        # Convert to numeric if needed
        if not pd.api.types.is_numeric_dtype(sev_levels):
            if pd.api.types.is_categorical_dtype(modality_adata.obs[sev_col]):
                sev_levels = modality_adata.obs[sev_col].cat.codes.values
            else:
                # Try numeric conversion
                try:
                    sev_levels = pd.to_numeric(sev_levels, errors='coerce')
                except:
                    # Create ordinal mapping
                    unique_levels = sorted(set(sev_levels))
                    level_map = {level: i for i, level in enumerate(unique_levels)}
                    sev_levels = np.array([level_map.get(level, np.nan) for level in sev_levels])
        
        # Remove any NaN values
        valid_mask = ~np.isnan(sev_levels)
        if valid_mask.sum() < 3:  # Need at least 3 samples
            return np.nan, []
            
        pca_values = pca_values[valid_mask]
        sev_levels = sev_levels[valid_mask]
        
        # Reshape severity for CCA
        sev_levels_2d = sev_levels.reshape(-1, 1)
        
        # Perform permutation test
        simulated_scores = []
        
        for i in range(num_simulations):
            # Permute severity levels
            permuted_sev = np.random.permutation(sev_levels).reshape(-1, 1)
            
            # Perform CCA with permuted data
            try:
                # Validate and process data for this permutation
                X_valid, y_valid, is_valid = validate_data_for_cca(pca_values, permuted_sev.ravel())
                
                if not is_valid:
                    simulated_scores.append(0.0)  # Conservative approach: treat invalid as no correlation
                    continue
                
                # Reshape y if needed
                if y_valid.ndim == 1:
                    y_valid = y_valid.reshape(-1, 1)
                
                # Standardize
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_scaled = scaler_X.fit_transform(X_valid)
                y_scaled = scaler_y.fit_transform(y_valid)
                
                # Perform CCA
                n_components = min(X_scaled.shape[1], y_scaled.shape[1], X_scaled.shape[0] - 1)
                if n_components < 1:
                    simulated_scores.append(0.0)
                    continue
                    
                cca = CCA(n_components=n_components, max_iter=1000, tol=1e-6)
                cca.fit(X_scaled, y_scaled)
                
                # Get correlation
                X_c, y_c = cca.transform(X_scaled, y_scaled)
                if X_c.shape[1] > 0 and y_c.shape[1] > 0:
                    correlation = np.corrcoef(X_c[:, 0], y_c[:, 0])[0, 1]
                    simulated_scores.append(abs(correlation) if not np.isnan(correlation) else 0.0)
                else:
                    simulated_scores.append(0.0)
                    
            except Exception:
                simulated_scores.append(0.0)  # Conservative approach for any errors
        
        # Calculate p-value
        simulated_scores = np.array(simulated_scores)
        p_value = np.mean(simulated_scores >= observed_correlation)
        
        if verbose:
            print(f"P-value calculation: {np.sum(simulated_scores >= observed_correlation)}/{num_simulations} = {p_value:.4f}")
            print(f"Observed: {observed_correlation:.4f}, Simulated mean: {np.mean(simulated_scores):.4f}, std: {np.std(simulated_scores):.4f}")
        
        # Create visualization if output directory is provided
        if output_dir and len(simulated_scores) > 0:
            create_modality_pvalue_plot(simulated_scores, observed_correlation, 
                                      p_value, modality, dr_type, output_dir)
        
        return p_value, simulated_scores
        
    except Exception as e:
        if verbose:
            print(f"Error in p-value calculation: {str(e)}")
        return np.nan, []


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

def find_optimal_cell_resolution_integration(
    AnnData_integrated: AnnData,
    output_dir: str,
    optimization_target: str = "sum",  # "rna", "atac", or "sum"
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
        What to optimize: "rna" (RNA CCA only), "atac" (ATAC CCA only), 
        or "sum" (sum of both CCA scores)
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

    # Validate optimization target
    if optimization_target not in ["rna", "atac", "sum"]:
        raise ValueError("optimization_target must be 'rna', 'atac', or 'sum'")
    
    convert_adata_severity_column(integrated_adata, sev_col=sev_col, verbose=verbose)
    check_data_quality(integrated_adata)

    # Create subdirectories for different outputs
    main_output_dir = os.path.join(output_dir, f"CCA_resolution_optimization_integration_{optimization_target}")
    resolution_plots_dir = os.path.join(main_output_dir, "resolution_plots")
    pvalue_results_dir = os.path.join(main_output_dir, "pvalue_results")
    embedding_plots_dir = os.path.join(main_output_dir, "embedding_visualizations")
    
    for dir_path in [main_output_dir, resolution_plots_dir, pvalue_results_dir, embedding_plots_dir]:
        os.makedirs(dir_path, exist_ok=True)

    print(f"Starting integrated resolution optimization...")
    print(f"Optimization target: {optimization_target.upper()}")
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
            
            # Compute CCA for RNA modality
            for dr_type in ['expression', 'proportion']:
                column = f'X_DR_{dr_type}'
                if column in pseudobulk_adata.uns:
                    cca_score, p_value = improved_compute_modality_cca(
                        pseudobulk_adata, 'RNA', column, sev_col
                    )
                    result_dict[f'rna_cca_{dr_type}'] = cca_score
                    result_dict[f'rna_pvalue_{dr_type}'] = p_value
                    
                    if not np.isnan(cca_score):
                        print(f"RNA {dr_type} CCA Score: {cca_score:.4f}, p-value: {p_value:.4f}")
            
            # Compute CCA for ATAC modality
            for dr_type in ['expression', 'proportion']:
                column = f'X_DR_{dr_type}'
                if column in pseudobulk_adata.uns:
                    cca_score, p_value = improved_compute_modality_cca(
                        pseudobulk_adata, 'ATAC', column, sev_col
                    )
                    result_dict[f'atac_cca_{dr_type}'] = cca_score
                    result_dict[f'atac_pvalue_{dr_type}'] = p_value
                    
                    if not np.isnan(cca_score):
                        print(f"ATAC {dr_type} CCA Score: {cca_score:.4f}, p-value: {p_value:.4f}")
            
            # Calculate optimization score based on target
            if optimization_target == "rna":
                # Use maximum of RNA expression and proportion scores
                rna_scores = [result_dict['rna_cca_expression'], result_dict['rna_cca_proportion']]
                valid_scores = [s for s in rna_scores if not np.isnan(s)]
                result_dict['optimization_score'] = max(valid_scores) if valid_scores else np.nan
            elif optimization_target == "atac":
                # Use maximum of ATAC expression and proportion scores
                atac_scores = [result_dict['atac_cca_expression'], result_dict['atac_cca_proportion']]
                valid_scores = [s for s in atac_scores if not np.isnan(s)]
                result_dict['optimization_score'] = max(valid_scores) if valid_scores else np.nan
            else:  # sum
                # Sum of best scores from each modality
                rna_best = max([s for s in [result_dict['rna_cca_expression'], 
                               result_dict['rna_cca_proportion']] if not np.isnan(s)] or [0])
                atac_best = max([s for s in [result_dict['atac_cca_expression'], 
                                result_dict['atac_cca_proportion']] if not np.isnan(s)] or [0])
                result_dict['optimization_score'] = rna_best + atac_best
            
            print(f"Resolution {resolution:.2f}: Optimization Score = {result_dict['optimization_score']:.4f}")
            
            # Create embedding visualizations if requested
            if visualize_embeddings and not np.isnan(result_dict['optimization_score']):
                for modality in ['RNA', 'ATAC']:
                    try:
                        embedding_path = os.path.join(
                            embedding_plots_dir, 
                            f"embedding_res_{resolution:.2f}_{modality}"
                        )
                        visualize_multimodal_embedding(
                            adata=pseudobulk_adata,
                            modality_col=modality_col,
                            color_col=sev_col,
                            target_modality=modality,
                            output_dir=embedding_path,
                            show_sample_names=False,
                            verbose=False
                        )
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Failed to create embedding visualization for {modality}: {str(e)}")
                
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
    print(f"Best optimization score: {best_coarse['optimization_score']:.4f}")

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
                for dr_type in ['expression', 'proportion']:
                    column = f'X_DR_{dr_type}'
                    if column in pseudobulk_adata.uns:
                        cca_score, p_value = improved_compute_modality_cca(
                            pseudobulk_adata, modality, column, sev_col
                        )
                        result_dict[f'{modality.lower()}_cca_{dr_type}'] = cca_score
                        result_dict[f'{modality.lower()}_pvalue_{dr_type}'] = p_value
            
            # Calculate optimization score
            if optimization_target == "rna":
                rna_scores = [result_dict['rna_cca_expression'], result_dict['rna_cca_proportion']]
                valid_scores = [s for s in rna_scores if not np.isnan(s)]
                result_dict['optimization_score'] = max(valid_scores) if valid_scores else np.nan
            elif optimization_target == "atac":
                atac_scores = [result_dict['atac_cca_expression'], result_dict['atac_cca_proportion']]
                valid_scores = [s for s in atac_scores if not np.isnan(s)]
                result_dict['optimization_score'] = max(valid_scores) if valid_scores else np.nan
            else:  # sum
                rna_best = max([s for s in [result_dict['rna_cca_expression'], 
                               result_dict['rna_cca_proportion']] if not np.isnan(s)] or [0])
                atac_best = max([s for s in [result_dict['atac_cca_expression'], 
                                result_dict['atac_cca_proportion']] if not np.isnan(s)] or [0])
                result_dict['optimization_score'] = rna_best + atac_best
            
            print(f"Fine-tuned Resolution {resolution:.3f}: Score {result_dict['optimization_score']:.4f}")
                    
        except Exception as e:
            print(f"Error at fine-tuned resolution {resolution:.3f}: {str(e)}")
        
        all_results.append(result_dict)

    # Create comprehensive results dataframe
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("resolution")
    
    # Find final best resolution
    valid_results = df_results[~df_results['optimization_score'].isna()]
    if valid_results.empty:
        raise ValueError("No valid results obtained.")
    
    final_best_idx = valid_results['optimization_score'].idxmax()
    final_best_resolution = valid_results.loc[final_best_idx, 'resolution']
    final_best_score = valid_results.loc[final_best_idx, 'optimization_score']
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Best resolution: {final_best_resolution:.3f}")
    print(f"Best optimization score ({optimization_target}): {final_best_score:.4f}")
    print(f"Best RNA scores - Expression: {valid_results.loc[final_best_idx, 'rna_cca_expression']:.4f}, "
          f"Proportion: {valid_results.loc[final_best_idx, 'rna_cca_proportion']:.4f}")
    print(f"Best ATAC scores - Expression: {valid_results.loc[final_best_idx, 'atac_cca_expression']:.4f}, "
          f"Proportion: {valid_results.loc[final_best_idx, 'atac_cca_proportion']:.4f}")

    # Save comprehensive results
    results_csv_path = os.path.join(main_output_dir, f"resolution_scores_comprehensive_integration_{optimization_target}.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nComprehensive results saved to: {results_csv_path}")

    # Create main visualization plot
    create_integration_resolution_visualization(
        df_results, final_best_resolution, optimization_target, 
        main_output_dir, compute_pvalues
    )

    # Save p-value summary if computed
    if compute_pvalues:
        pvalue_summary_path = os.path.join(pvalue_results_dir, f"pvalue_summary_integration_{optimization_target}.txt")
        with open(pvalue_summary_path, "w") as f:
            f.write(f"Resolution Optimization P-value Summary for Integration ({optimization_target})\n")
            f.write("="*60 + "\n\n")
            f.write(f"Best Resolution: {final_best_resolution:.3f}\n")
            f.write(f"Best Optimization Score: {final_best_score:.4f}\n\n")
            f.write("Best P-values:\n")
            f.write(f"  RNA Expression: {valid_results.loc[final_best_idx, 'rna_pvalue_expression']:.4f}\n")
            f.write(f"  RNA Proportion: {valid_results.loc[final_best_idx, 'rna_pvalue_proportion']:.4f}\n")
            f.write(f"  ATAC Expression: {valid_results.loc[final_best_idx, 'atac_pvalue_expression']:.4f}\n")
            f.write(f"  ATAC Proportion: {valid_results.loc[final_best_idx, 'atac_pvalue_proportion']:.4f}\n")
        print(f"P-value summary saved to: {pvalue_summary_path}")

    print(f"\n[Find Optimal Resolution Integration] Total runtime: {time.time() - start_time:.2f} seconds\n")

    return final_best_resolution, df_results


def create_integration_resolution_visualization(df_results, best_resolution, optimization_target, 
                                              output_dir, include_pvalues):
    """Create comprehensive visualization of integration resolution search results"""
    
    # Determine number of subplots based on optimization target and p-values
    n_rows = 3 if optimization_target == "sum" else 2
    if include_pvalues:
        n_rows += 1
    
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4*n_rows))
    if n_rows == 1:
        axes = [axes]
    
    # Plot data
    valid_df = df_results[~df_results['optimization_score'].isna()]
    coarse_df = valid_df[valid_df['pass'] == 'coarse']
    fine_df = valid_df[valid_df['pass'] == 'fine']
    
    ax_idx = 0
    
    # Plot optimization score
    ax = axes[ax_idx]
    ax.scatter(coarse_df['resolution'], coarse_df['optimization_score'], 
               color='blue', s=60, alpha=0.6, label='Coarse Search')
    ax.scatter(fine_df['resolution'], fine_df['optimization_score'], 
               color='green', s=40, alpha=0.8, label='Fine Search')
    ax.plot(valid_df['resolution'], valid_df['optimization_score'], 
            'k-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=best_resolution, color='r', linestyle='--', 
               label=f'Best Resolution: {best_resolution:.3f}')
    ax.set_xlabel("Resolution")
    ax.set_ylabel(f"Optimization Score ({optimization_target.upper()})")
    ax.set_title(f"Integration Resolution Optimization - Target: {optimization_target.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax_idx += 1
    
    # Plot RNA CCA scores
    if optimization_target in ["rna", "sum"]:
        ax = axes[ax_idx]
        # Expression scores
        ax.scatter(coarse_df['resolution'], coarse_df['rna_cca_expression'], 
                   color='darkblue', s=60, alpha=0.6, marker='o', label='RNA Expression (Coarse)')
        ax.scatter(fine_df['resolution'], fine_df['rna_cca_expression'], 
                   color='darkblue', s=40, alpha=0.8, marker='o', label='RNA Expression (Fine)')
        # Proportion scores
        ax.scatter(coarse_df['resolution'], coarse_df['rna_cca_proportion'], 
                   color='lightblue', s=60, alpha=0.6, marker='^', label='RNA Proportion (Coarse)')
        ax.scatter(fine_df['resolution'], fine_df['rna_cca_proportion'], 
                   color='lightblue', s=40, alpha=0.8, marker='^', label='RNA Proportion (Fine)')
        
        ax.axvline(x=best_resolution, color='r', linestyle='--')
        ax.set_xlabel("Resolution")
        ax.set_ylabel("CCA Score")
        ax.set_title("RNA CCA Scores")
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)
        ax_idx += 1
    
    # Plot ATAC CCA scores
    if optimization_target in ["atac", "sum"]:
        ax = axes[ax_idx]
        # Expression scores
        ax.scatter(coarse_df['resolution'], coarse_df['atac_cca_expression'], 
                   color='darkgreen', s=60, alpha=0.6, marker='o', label='ATAC Expression (Coarse)')
        ax.scatter(fine_df['resolution'], fine_df['atac_cca_expression'], 
                   color='darkgreen', s=40, alpha=0.8, marker='o', label='ATAC Expression (Fine)')
        # Proportion scores
        ax.scatter(coarse_df['resolution'], coarse_df['atac_cca_proportion'], 
                   color='lightgreen', s=60, alpha=0.6, marker='^', label='ATAC Proportion (Coarse)')
        ax.scatter(fine_df['resolution'], fine_df['atac_cca_proportion'], 
                   color='lightgreen', s=40, alpha=0.8, marker='^', label='ATAC Proportion (Fine)')
        
        ax.axvline(x=best_resolution, color='r', linestyle='--')
        ax.set_xlabel("Resolution")
        ax.set_ylabel("CCA Score")
        ax.set_title("ATAC CCA Scores")
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)
        ax_idx += 1
    
    # Plot p-values if available
    if include_pvalues:
        ax = axes[ax_idx]
        
        # RNA p-values
        valid_pval_rna_exp = valid_df[~valid_df['rna_pvalue_expression'].isna()]
        valid_pval_rna_prop = valid_df[~valid_df['rna_pvalue_proportion'].isna()]
        
        if not valid_pval_rna_exp.empty:
            ax.scatter(valid_pval_rna_exp['resolution'], valid_pval_rna_exp['rna_pvalue_expression'], 
                       color='darkblue', s=40, alpha=0.7, marker='o', label='RNA Expression')
        if not valid_pval_rna_prop.empty:
            ax.scatter(valid_pval_rna_prop['resolution'], valid_pval_rna_prop['rna_pvalue_proportion'], 
                       color='lightblue', s=40, alpha=0.7, marker='^', label='RNA Proportion')
        
        # ATAC p-values
        valid_pval_atac_exp = valid_df[~valid_df['atac_pvalue_expression'].isna()]
        valid_pval_atac_prop = valid_df[~valid_df['atac_pvalue_proportion'].isna()]
        
        if not valid_pval_atac_exp.empty:
            ax.scatter(valid_pval_atac_exp['resolution'], valid_pval_atac_exp['atac_pvalue_expression'], 
                       color='darkgreen', s=40, alpha=0.7, marker='s', label='ATAC Expression')
        if not valid_pval_atac_prop.empty:
            ax.scatter(valid_pval_atac_prop['resolution'], valid_pval_atac_prop['atac_pvalue_proportion'], 
                       color='lightgreen', s=40, alpha=0.7, marker='d', label='ATAC Proportion')
        
        ax.axvline(x=best_resolution, color='r', linestyle='--')
        ax.axhline(y=0.05, color='orange', linestyle=':', label='p=0.05 threshold')
        
        ax.set_xlabel("Resolution")
        ax.set_ylabel("P-value")
        ax.set_title("P-values for All Modalities and DR Types")
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"resolution_optimization_comprehensive_integration_{optimization_target}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive plot saved to: {plot_path}")

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
        optimization_target="atac",  # or "rna" or "atac"
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
