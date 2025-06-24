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

def cca_analysis(pseudobulk_adata, modality, column, sev_col, n_components=2):
    """
    Simplified CCA analysis without p-value calculation or plotting.
    
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
        
    Assumptions based on fixed data type:
    - pseudobulk_adata.uns[column] contains pandas DataFrame with numeric coordinates
    - modality column exists and contains string values
    - severity column exists and contains numeric values
    - Data is already preprocessed and clean
    """
    
    result = {
        'cca_score': np.nan,
        'n_samples': 0,
        'n_features': 0,
        'modality': modality,
        'column': column,
        'valid': False,
        'error_message': None
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
        # Now indices should always match since we standardized them
        dr_coords_full = uns_data_standardized.loc[modality_mask].copy()
        sev_levels = obs_standardized.loc[modality_mask, sev_col].values
        
        # Use only the first 2 features for CCA analysis
        dr_coords = dr_coords_full.iloc[:, :2]
        
        # Basic validation only
        if len(dr_coords) < 3:
            result['error_message'] = f"Insufficient samples: {len(dr_coords)}"
            return result
        if len(np.unique(sev_levels)) < 2:
            result['error_message'] = "Insufficient severity level variance"
            return result
        
        # Prepare data for CCA
        X = dr_coords.values  # Now using only first 2 features
        y = sev_levels.reshape(-1, 1)
        
        result['n_samples'] = len(X)
        result['n_features'] = X.shape[1]  # Should be 2
        
        # Limit components based on data dimensions
        # CCA components are limited by min(X_features, y_features, n_samples-1)
        max_components = min(X.shape[1], y.shape[1], X.shape[0] - 1)
        n_components_actual = min(n_components, max_components)
        
        if n_components_actual < 1:
            result['error_message'] = "Cannot compute CCA components"
            return result
        
        # Standardize data (simplified)
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
        result['valid'] = True
        
    except Exception as e:
        result['error_message'] = f"CCA failed: {str(e)}"
    
    return result


def batch_cca_analysis(pseudobulk_adata, dr_columns, sev_col, modalities=None, 
                      n_components=2, output_dir=None):
    """
    Run CCA analysis across multiple dimension reduction results and modalities.
    Saves results to output directory without p-value calculation or plotting.
    
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
    output_dir : str, optional
        Directory to save results summary table
        
    Returns:
    --------
    pd.DataFrame
        Results DataFrame with CCA scores for each modality-column combination
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
                    n_components=n_components
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
                    'error_message': f"Column '{column}' not found"
                })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results table to output directory
    if output_dir:
        results_path = os.path.join(output_dir, 'cca_results_summary.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")
        
        # Save a detailed results table with additional info
        detailed_path = os.path.join(output_dir, 'cca_results_detailed.csv')
        results_df.to_csv(detailed_path, index=False)
        print(f"Detailed results saved to: {detailed_path}")
    
    return results_df

def generate_null_distribution(pseudobulk_adata, modality, column, sev_col, 
                                          n_components=2, n_permutations=1000, 
                                          save_path=None, verbose=True):
    """
    Generate null distribution using ONLY the first 2 dimensions of DR results.
    
    Steps:
    1. Extract first 2 DR components for the specified modality
    2. Randomly shuffle severity level labels for each sample  
    3. Run CCA between 2D embeddings and 1D severity
    4. Record CCA correlation as one simulation
    """
    if verbose:
        print(f"Generating null distribution with {n_permutations} permutations...")
        print("Method: Shuffle severity labels, CCA on first 2 DR components")
    
    # Extract data for specified modality
    modality_mask = pseudobulk_adata.obs['modality'] == modality
    if not modality_mask.any():
        raise ValueError(f"No samples found for modality: {modality}")
    
    # Get DR coordinates and severity levels
    dr_coords_full = pseudobulk_adata.uns[column].loc[modality_mask].copy()
    sev_levels = pseudobulk_adata.obs.loc[modality_mask, sev_col].values
    
    # IMPORTANT: Use only first 2 DR components
    dr_coords_2d = dr_coords_full.iloc[:, :2]  # First 2 columns only
    
    if verbose:
        print(f"Full DR shape: {dr_coords_full.shape}")
        print(f"Using 2D DR shape: {dr_coords_2d.shape}")
        print(f"Severity levels: {len(sev_levels)} samples")
        print(f"Unique severity values: {np.unique(sev_levels)}")
        print(f"DR column names used: {list(dr_coords_2d.columns)}")
    
    if len(dr_coords_2d) < 3:
        raise ValueError(f"Insufficient samples: {len(dr_coords_2d)}")
    
    if len(np.unique(sev_levels)) < 2:
        raise ValueError("Insufficient severity level variance")
    
    # Prepare data for CCA
    X = dr_coords_2d.values  # 2D embedding coordinates [n_samples, 2]
    y_original = sev_levels.copy()  # 1D severity levels [n_samples]
    
    if verbose:
        print(f"Final X shape: {X.shape} (should be [n_samples, 2])")
        print(f"Final y shape: {y_original.shape} (should be [n_samples])")
    
    # Import libraries
    from sklearn.cross_decomposition import CCA
    from sklearn.preprocessing import StandardScaler
    
    # Determine CCA components
    # For 2D X and 1D y: max components = min(2, 1, n_samples-1) = 1
    max_components = min(X.shape[1], 1, X.shape[0] - 1)  # 2, 1, n_samples-1
    n_comp_actual = min(n_components, max_components)
    
    if verbose:
        print(f"CCA components: requested={n_components}, max_possible={max_components}, using={n_comp_actual}")
    
    # Test original analysis first
    if verbose:
        print("\n=== Testing Original Analysis ===")
        try:
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y_original.reshape(-1, 1))
            
            cca_orig = CCA(n_components=n_comp_actual, max_iter=1000, tol=1e-6)
            cca_orig.fit(X_scaled, y_scaled)
            
            X_c_orig, y_c_orig = cca_orig.transform(X_scaled, y_scaled)
            orig_correlation = np.corrcoef(X_c_orig[:, 0], y_c_orig[:, 0])[0, 1]
            
            print(f"Original CCA correlation: {orig_correlation:.6f}")
            
        except Exception as e:
            print(f"Error in original analysis: {e}")
    
    # Run permutations
    print(f"\n=== Running {n_permutations} Permutations ===")
    
    null_scores = []
    failed_permutations = 0
    
    for perm in range(n_permutations):
        if verbose and (perm + 1) % 100 == 0:
            print(f"  Completed {perm + 1}/{n_permutations} permutations...")
        
        try:
            # Step 1 & 2: Randomly shuffle severity level labels
            permuted_sev = np.random.permutation(y_original)
            
            # Step 3: Run CCA between 2D embeddings and shuffled 1D severity
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_scaled = scaler_X.fit_transform(X)  # [n_samples, 2]
            y_permuted_scaled = scaler_y.fit_transform(permuted_sev.reshape(-1, 1))  # [n_samples, 1]
            
            # Fit CCA
            cca_perm = CCA(n_components=n_comp_actual, max_iter=1000, tol=1e-6)
            cca_perm.fit(X_scaled, y_permuted_scaled)
            
            # Transform and compute correlation
            X_c_perm, y_c_perm = cca_perm.transform(X_scaled, y_permuted_scaled)
            perm_correlation = np.corrcoef(X_c_perm[:, 0], y_c_perm[:, 0])[0, 1]
            
            # Step 4: Record the CCA score
            if np.isnan(perm_correlation) or np.isinf(perm_correlation):
                if verbose and perm < 3:
                    print(f"  Permutation {perm}: Invalid correlation = {perm_correlation}")
                null_scores.append(0.0)
                failed_permutations += 1
            else:
                null_scores.append(abs(perm_correlation))
                if verbose and perm < 5:
                    print(f"  Permutation {perm}: CCA correlation = {perm_correlation:.6f}")
                
        except Exception as e:
            if verbose and perm < 3:
                print(f"  Error in permutation {perm}: {str(e)}")
            null_scores.append(0.0)
            failed_permutations += 1
    
    null_distribution = np.array(null_scores)
    
    if np.all(null_distribution == 1.0):
        print(f"  WARNING: All permutation scores are 1.0! CCA might be overfitting.")
    elif np.all(null_distribution == 0.0):
        print(f"  WARNING: All permutation scores are 0.0! Check CCA computation.")
    elif np.std(null_distribution) < 1e-6:
        print(f"  WARNING: Very low variance in null distribution.")
    
    if save_path:
        np.save(save_path, null_distribution)
        if verbose:
            print(f"Saved null distribution to: {save_path}")
    
    return null_distribution


def ensure_non_categorical_columns(adata, columns):
    """Convert specified columns from categorical to string to avoid categorical errors"""
    for col in columns:
        if col in adata.obs.columns:
            if pd.api.types.is_categorical_dtype(adata.obs[col]):
                adata.obs[col] = adata.obs[col].astype(str)
    return adata

def compute_all_corrected_pvalues_and_plots(df_results, corrected_null_distribution, main_output_dir, 
                                          optimization_target, dr_type):
    """
    Compute corrected p-values for all CCA scores and create visualization plots.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Results dataframe with CCA scores for all resolutions
    corrected_null_distribution : np.array
        Corrected null distribution accounting for resolution selection
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
    
    # Track all corrected p-values to add to dataframe
    corrected_pvalues = {}
    
    # Process each resolution
    for idx, row in df_results.iterrows():
        resolution = row['resolution']
        print(f"Computing corrected p-values for resolution {resolution}")
        
        # Create resolution-specific directory
        res_dir = os.path.join(pvalue_dir, f"resolution_{resolution}")
        os.makedirs(res_dir, exist_ok=True)
        
        # Process each modality and DR type combination
        for modality in modalities:
            for dr_method in dr_types:
                cca_col = f'{modality}_cca_{dr_method}'
                
                if cca_col in row and not pd.isna(row[cca_col]):
                    cca_score = row[cca_col]
                    
                    # Compute corrected p-value
                    corrected_p_value = np.mean(corrected_null_distribution >= cca_score)
                    
                    # Store corrected p-value
                    corrected_pval_col = f'{modality}_corrected_pvalue_{dr_method}'
                    if corrected_pval_col not in corrected_pvalues:
                        corrected_pvalues[corrected_pval_col] = [np.nan] * len(df_results)
                    corrected_pvalues[corrected_pval_col][idx] = corrected_p_value
                    
                    # Create visualization plot
                    plt.figure(figsize=(10, 6))
                    
                    # Plot histogram of null distribution
                    plt.hist(corrected_null_distribution, bins=50, alpha=0.7, color='lightblue', 
                            density=True, label='Corrected Null Distribution')
                    
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
    
    # Add all corrected p-values to the dataframe
    for col, values in corrected_pvalues.items():
        df_results[col] = values
    
    print(f"All p-value plots saved to: {pvalue_dir}")


def generate_corrected_null_distribution(all_resolution_results, optimization_target, dr_type, n_permutations=1000):
    """
    Generate null distribution accounting for resolution selection bias (double dipping).
    
    For each permutation:
    1. Collect CCA scores from all resolutions for that permutation
    2. Select the maximum CCA score (mimicking optimal resolution selection)
    3. Use these maximum scores to form the corrected null distribution
    
    Parameters:
    -----------
    all_resolution_results : list
        List of dictionaries containing results from all resolutions, 
        where each dict has 'null_scores' key with array of permutation results
    optimization_target : str
        Target modality ("rna" or "atac")
    dr_type : str
        DR type ("expression" or "proportion")
    n_permutations : int
        Number of permutations (should match the number used in resolution testing)
        
    Returns:
    --------
    np.array
        Corrected null distribution accounting for resolution selection
    """
    
    corrected_null_scores = []
    
    for perm_idx in range(n_permutations):
        # Collect the CCA score from this permutation across all resolutions
        perm_scores_across_resolutions = []
        
        for resolution_result in all_resolution_results:
            if 'null_scores' in resolution_result and len(resolution_result['null_scores']) > perm_idx:
                perm_scores_across_resolutions.append(resolution_result['null_scores'][perm_idx])
        
        # Select the maximum score (mimicking optimal resolution selection)
        if perm_scores_across_resolutions:
            max_score_for_this_perm = max(perm_scores_across_resolutions)
            corrected_null_scores.append(max_score_for_this_perm)
    
    return np.array(corrected_null_scores)


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
    num_pvalue_simulations: int = 1000,  # Changed from 100 to 1000
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
        Number of simulations for p-value calculation (default 1000)
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
    # Storage for null distribution results from each resolution
    all_resolution_null_results = []

    # First pass: coarse search
    print("\n=== FIRST PASS: Coarse Search ===")
    for resolution in np.arange(0.1, 0.31, 0.1):
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
        
        # Initialize null results for this resolution
        resolution_null_result = {
            'resolution': resolution,
            'null_scores': None
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
            
            # Generate null distribution for this resolution if computing p-values
            if compute_pvalues:
                try:
                    null_distribution = generate_null_distribution(
                        pseudobulk_adata=pseudobulk_adata,
                        modality=optimization_target.upper(),
                        column=f'X_DR_{dr_type}',
                        sev_col=sev_col,
                        n_components=2,
                        n_permutations=num_pvalue_simulations,
                        save_path=os.path.join(resolution_dir, f'null_dist_{optimization_target}_{dr_type}.npy'),
                        verbose=verbose
                    )
                    resolution_null_result['null_scores'] = null_distribution
                except Exception as e:
                    print(f"Warning: Failed to generate null distribution for resolution {resolution:.2f}: {str(e)}")
                    resolution_null_result['null_scores'] = None
            
            # Compute CCA for both modalities and both DR types using batch analysis
            dr_columns = ['X_DR_expression', 'X_DR_proportion']
            cca_results_df = batch_cca_analysis(
                pseudobulk_adata=pseudobulk_adata,
                dr_columns=dr_columns,
                sev_col=sev_col,
                modalities=['RNA', 'ATAC'],
                n_components=2,
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
                embedding_path = os.path.join(
                    resolution_dir, 
                    f"embedding_{optimization_target.upper()}_{dr_type}"
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
            
            # Save resolution-specific results
            resolution_results_path = os.path.join(resolution_dir, f"results_res_{resolution:.2f}.csv")
            pd.DataFrame([result_dict]).to_csv(resolution_results_path, index=False)
                
        except Exception as e:
            print(f"Error at resolution {resolution:.2f}: {str(e)}")
        
        all_results.append(result_dict)
        all_resolution_null_results.append(resolution_null_result)

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
    search_range_start = max(0.01, best_resolution - 0.02)
    search_range_end = min(1.00, best_resolution + 0.02)
    
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
        
        # Initialize null results for this resolution
        resolution_null_result = {
            'resolution': resolution,
            'null_scores': None
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
            
            # Generate null distribution for this resolution if computing p-values
            if compute_pvalues:
                try:
                    null_distribution = generate_null_distribution(
                        pseudobulk_adata=pseudobulk_adata,
                        modality=optimization_target.upper(),
                        column=f'X_DR_{dr_type}',
                        sev_col=sev_col,
                        n_components=2,
                        n_permutations=num_pvalue_simulations,
                        save_path=os.path.join(resolution_dir, f'null_dist_{optimization_target}_{dr_type}.npy'),
                        verbose=verbose
                    )
                    resolution_null_result['null_scores'] = null_distribution
                except Exception as e:
                    print(f"Warning: Failed to generate null distribution for resolution {resolution:.3f}: {str(e)}")
                    resolution_null_result['null_scores'] = None
            
            # Compute CCA for both modalities and DR types using batch analysis
            dr_columns = ['X_DR_expression', 'X_DR_proportion']
            cca_results_df = batch_cca_analysis(
                pseudobulk_adata=pseudobulk_adata,
                dr_columns=dr_columns,
                sev_col=sev_col,
                modalities=['RNA', 'ATAC'],
                n_components=2,
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
                embedding_path = os.path.join(
                    resolution_dir, 
                    f"embedding_{optimization_target.upper()}_{dr_type}"
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
            
            # Save resolution-specific results
            resolution_results_path = os.path.join(resolution_dir, f"results_res_{resolution:.3f}.csv")
            pd.DataFrame([result_dict]).to_csv(resolution_results_path, index=False)
                    
        except Exception as e:
            print(f"Error at fine-tuned resolution {resolution:.3f}: {str(e)}")
        
        all_results.append(result_dict)
        all_resolution_null_results.append(resolution_null_result)

    # Create comprehensive results dataframe BEFORE using it
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("resolution")

    # Generate corrected null distribution if computing p-values
    corrected_null_distribution = None
    if compute_pvalues:
        print("\n=== GENERATING CORRECTED NULL DISTRIBUTION ===")
        print("Accounting for resolution selection bias...")
        
        # Filter out null results that failed to generate
        valid_null_results = [r for r in all_resolution_null_results if r['null_scores'] is not None]
        
        if valid_null_results:
            corrected_null_distribution = generate_corrected_null_distribution(
                all_resolution_results=valid_null_results,
                optimization_target=optimization_target,
                dr_type=dr_type,
                n_permutations=num_pvalue_simulations
            )
            
            # Save corrected null distribution
            summary_dir = os.path.join(main_output_dir, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            corrected_null_path = os.path.join(summary_dir, f'corrected_null_distribution_{optimization_target}_{dr_type}.npy')
            np.save(corrected_null_path, corrected_null_distribution)
            print(f"Corrected null distribution saved to: {corrected_null_path}")
            
            # Compute corrected p-values for all CCA scores and create visualization plots
            print("\n=== COMPUTING CORRECTED P-VALUES AND CREATING PLOTS ===")
            compute_all_corrected_pvalues_and_plots(
                df_results=df_results,  # Now df_results is defined!
                corrected_null_distribution=corrected_null_distribution,
                main_output_dir=main_output_dir,
                optimization_target=optimization_target,
                dr_type=dr_type
            )
        else:
            print("Warning: No valid null distributions generated, cannot create corrected null distribution")
    
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

    # # Create separate modality visualizations in summary directory
    # create_separate_modality_plots(df_results, final_best_resolution, optimization_target, dr_type, summary_dir)

    # # Save comprehensive CCA summary in summary directory
    # save_comprehensive_cca_summary(df_results, optimization_target, dr_type, summary_dir)

    # # Save detailed p-value summary if computed in summary directory
    # if compute_pvalues:
    #     save_detailed_pvalue_summary(df_results, optimization_target, dr_type, summary_dir)

    print(f"\n[Find Optimal Resolution Integration] Total runtime: {time.time() - start_time:.2f} seconds\n")

    return final_best_resolution, df_results


if __name__ == "__main__":
    integrated_adata = ad.read_h5ad("/dcl01/hongkai/data/data/hjiang/result/integration_test/subsample/atac_rna_integrated_subsampled_10pct.h5ad")
    output_dir = "/dcl01/hongkai/data/data/hjiang/result/integration_test/subsample"
    
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