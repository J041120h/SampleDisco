import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
import warnings
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from itertools import combinations
import traceback


def validate_and_fix_fit_object(fit, verbose=True):
    """
    Validate and fix data types in the fit object to prevent casting errors.
    """
    if verbose:
        print("Validating fit object...")
    
    # Check and fix X matrix
    if 'X' in fit:
        if verbose:
            print(f"X before conversion - Type: {type(fit['X'])}")
            if hasattr(fit['X'], 'dtypes'):
                print(f"X column dtypes: {fit['X'].dtypes}")
            elif hasattr(fit['X'], 'dtype'):
                print(f"X dtype: {fit['X'].dtype}")
        
        if not isinstance(fit['X'], pd.DataFrame):
            if verbose:
                print("Converting X to DataFrame...")
            fit['X'] = pd.DataFrame(fit['X'])
        
        # Convert all columns to numeric, handling any non-numeric values
        for col in fit['X'].columns:
            try:
                # First try direct conversion
                fit['X'][col] = pd.to_numeric(fit['X'][col], errors='coerce')
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not convert column {col} to numeric: {e}")
                # If conversion fails, try to extract numeric values
                try:
                    fit['X'][col] = fit['X'][col].astype(str).str.extract('(\d+\.?\d*)')[0].astype(float)
                except:
                    # Last resort: fill with zeros
                    if verbose:
                        print(f"Filling column {col} with zeros")
                    fit['X'][col] = 0
        
        # Ensure all data is float64
        fit['X'] = fit['X'].astype(np.float64)
        
        if verbose:
            print(f"X after conversion - Dtypes: {fit['X'].dtypes}")
            print(f"X shape: {fit['X'].shape}")
            print(f"X has NaN values: {fit['X'].isnull().any().any()}")
    
    # Check and fix means matrix
    if 'mean' in fit:
        if isinstance(fit['mean'], pd.DataFrame):
            fit['mean'] = fit['mean'].astype(np.float64)
        if verbose:
            print(f"Mean matrix dtypes: {fit['mean'].dtypes if hasattr(fit['mean'], 'dtypes') else fit['mean'].dtype}")
    
    # Check and fix Z matrix
    if 'Z' in fit:
        if isinstance(fit['Z'], pd.DataFrame):
            fit['Z'] = fit['Z'].astype(np.float64)
        if verbose:
            print(f"Z matrix dtypes: {fit['Z'].dtypes if hasattr(fit['Z'], 'dtypes') else fit['Z'].dtype}")
    
    # Check and fix omega2
    if 'omega2' in fit:
        if isinstance(fit['omega2'], pd.DataFrame):
            fit['omega2'] = fit['omega2'].astype(np.float64)
        if verbose:
            print(f"Omega2 matrix dtypes: {fit['omega2'].dtypes if hasattr(fit['omega2'], 'dtypes') else fit['omega2'].dtype}")
    
    # Check and fix sigma2
    if 'sigma2' in fit:
        if isinstance(fit['sigma2'], pd.DataFrame):
            # Only convert numeric columns to avoid errors with categorical data
            numeric_cols = fit['sigma2'].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fit['sigma2'][numeric_cols] = fit['sigma2'][numeric_cols].astype(np.float64)
        if verbose:
            print(f"Sigma2 matrix dtypes: {fit['sigma2'].dtypes}")
    
    if verbose:
        print("Fit object validation complete.")
    
    return fit

def raisintest(fit, coef=1, contrast=None, fdrmethod='fdr_bh', n_permutations=10, verbose=True):
    """
    Statistical testing for RAISIN model.
    
    Parameters:
    -----------
    fit : dict
        The output from the raisinfit function
    coef : int
        Index of coefficient to test (0-indexed, default=1)
    contrast : array-like or None
        Vector indicating combination of coefficients to test
    fdrmethod : str
        Method for FDR correction ('fdr_bh', 'fdr_by', 'bonferroni', etc.)
    n_permutations : int
        Number of permutations for estimating degrees of freedom
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    pandas.DataFrame
        Results with columns: Foldchange, stat, pvalue, FDR
    """
    
    try:
        # Validate and fix fit object first
        fit = validate_and_fix_fit_object(fit, verbose=verbose)
        
        # Extract components from fit
        X = fit['X'].values.astype(np.float64)  # Ensure float64 dtype
        means = fit['mean'].values.astype(np.float64)
        G = means.shape[0]  # number of genes
        group = fit['group'].values
        Z = fit['Z'].values.astype(np.float64)  # Ensure float64 dtype
        sigma2 = fit['sigma2']
        omega2 = fit['omega2'].values.astype(np.float64)
        
        if verbose:
            print(f"Testing {G} genes")
            print(f"Design matrix X shape: {X.shape}")
            print(f"X rank: {np.linalg.matrix_rank(X)}")
        
        # Set up contrast vector
        if contrast is None:
            contrast = np.zeros(X.shape[1])
            contrast[coef] = 1
        else:
            contrast = np.array(contrast)
        
        if len(contrast) != X.shape[1]:
            raise ValueError(f"Contrast length ({len(contrast)}) must match number of columns in X ({X.shape[1]})")
        
        if verbose:
            print(f"Contrast vector: {contrast}")
        
        # Function to safely invert X'X or permuted version
        def safe_invert_XTX(X_mat):
            """Safely invert X'X using pseudoinverse for singular matrices."""
            XTX = X_mat.T @ X_mat
            det = np.linalg.det(XTX)
            
            if abs(det) < 1e-10 or np.linalg.cond(XTX) > 1e12:
                # Use pseudoinverse for singular matrix
                return np.linalg.pinv(XTX)
            else:
                # Regular inverse
                return np.linalg.inv(XTX)
        
        # Check if X'X is singular
        XTX = X.T @ X
        det = np.linalg.det(XTX)
        
        if verbose:
            print(f"X'X determinant: {det}")
            print(f"X'X condition number: {np.linalg.cond(XTX)}")
        
        # Calculate test statistics using safe inversion
        XTX_inv = safe_invert_XTX(X)
        if verbose and (abs(det) < 1e-10 or np.linalg.cond(XTX) > 1e12):
            print("Design matrix X'X is singular or nearly singular. Using pseudoinverse.")
        
        # Calculate test statistics
        k = contrast.T @ XTX_inv @ X.T
        
        # Calculate fold changes
        b = means @ k.T
        if b.ndim > 1:
            b = b[:, 0]
        
        # Check if all variance components failed
        if np.all(np.isnan(sigma2.values) | (sigma2.values == 0)):
            warnings.warn('Unable to estimate variance for all random effects. Setting FDR to 1.')
            res = pd.DataFrame({
                'Foldchange': b,
                'FDR': 1.0
            })
            res = res.reindex(res['Foldchange'].abs().sort_values(ascending=False).index)
            return res
        
        # Calculate variance components for test statistic
        kZ = k @ Z
        a = np.zeros(G)
        for g in range(G):
            random_var = 0
            for i, g_name in enumerate(group):
                if g_name in sigma2.columns:
                    random_var += (kZ[i]**2) * sigma2.loc[g, g_name]
            fixed_var = np.sum(k**2 * omega2[g, :])
            a[g] = random_var + fixed_var
        
        # Calculate test statistics
        with np.errstate(divide='ignore', invalid='ignore'):
            stat = b / np.sqrt(a)
        stat = np.where(np.isfinite(stat), stat, 0)
        
        if verbose:
            print("Running permutation test to estimate degrees of freedom...")
        
        # Permutation test with safe matrix inversion
        perm_stats = []
        for sim_id in range(n_permutations):
            try:
                perm_idx = np.random.permutation(X.shape[0])
                perX = X[perm_idx, :].astype(np.float64)  # Ensure dtype
                
                # Use safe inversion for permuted matrix too
                perXTX_inv = safe_invert_XTX(perX)
                
                perm_k = contrast.T @ perXTX_inv @ perX.T
                perm_kZ = perm_k @ Z
                perm_a = np.zeros(G)
                for g in range(G):
                    random_var = 0
                    for i, g_name in enumerate(group):
                        if g_name in sigma2.columns:
                            random_var += (perm_kZ[i]**2) * sigma2.loc[g, g_name]
                    fixed_var = np.sum(perm_k**2 * omega2[g, :])
                    perm_a[g] = random_var + fixed_var
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    perm_stat = (means @ perm_k.T) / np.sqrt(perm_a)
                perm_stat = perm_stat[np.isfinite(perm_stat)]
                if perm_stat.ndim > 1:
                    perm_stat = perm_stat.flatten()
                perm_stats.extend(perm_stat)
                
            except Exception as e:
                if verbose:
                    print(f"Warning: Error in permutation {sim_id}: {e}")
                continue
        
        if not perm_stats:
            # If all permutations failed, use normal distribution
            warnings.warn("All permutations failed. Using normal distribution for p-values.")
            pval = 2 * stats.norm.sf(np.abs(stat))
            if verbose:
                print("Using normal distribution (fallback)")
        else:
            perm_stats = np.array(perm_stats)
            
            # Estimate degrees of freedom
            pnorm_ll = np.sum(stats.norm.logpdf(perm_stats))
            dfs = np.arange(1, 100.1, 0.1)
            pt_lls = np.array([np.sum(stats.t.logpdf(perm_stats, df=df)) for df in dfs])
            best_df_idx = np.argmax(pt_lls)
            
            if pt_lls[best_df_idx] > pnorm_ll:
                df = dfs[best_df_idx]
                pval = 2 * stats.t.sf(np.abs(stat), df=df)
                if verbose:
                    print(f"Using t-distribution with df={df:.1f}")
            else:
                pval = 2 * stats.norm.sf(np.abs(stat))
                if verbose:
                    print("Using normal distribution")
        
        # Multiple testing correction
        if fdrmethod == 'fdr_bh':
            _, fdr = fdrcorrection(pval, alpha=0.05, method='indep')
        elif fdrmethod == 'fdr_by':
            _, fdr = fdrcorrection(pval, alpha=0.05, method='negcorr')
        elif fdrmethod == 'bonferroni':
            fdr = np.minimum(pval * len(pval), 1.0)
        else:
            from statsmodels.stats.multitest import multipletests
            _, fdr, _, _ = multipletests(pval, alpha=0.05, method=fdrmethod)
        
        # Create results dataframe
        gene_names = fit['mean'].index
        res = pd.DataFrame({
            'Foldchange': b,
            'stat': stat,
            'pvalue': pval,
            'FDR': fdr
        }, index=gene_names)
        
        # Sort by FDR, then by absolute statistic
        # First sort by absolute statistic descending
        res['abs_stat'] = np.abs(res['stat'])
        res = res.sort_values('abs_stat', ascending=False)
        # Then sort by FDR ascending (stable sort preserves the abs_stat order for ties)
        res = res.sort_values('FDR', kind='stable')
        # Remove the temporary column
        res = res.drop('abs_stat', axis=1)
        
        if verbose:
            print(f"Testing complete. Found {np.sum(res['FDR'] < 0.05)} genes with FDR < 0.05")
        
        return res
        
    except Exception as e:
        print(f"ERROR in raisintest: {e}")
        traceback.print_exc()
        raise

def create_group_contrast(fit, group1, group2, verbose=False):
    """Create a contrast vector for comparing two groups."""
    try:
        X = fit['X']
        group_assignments = fit['group']
        
        # Add debug information
        if verbose:
            print(f"X type: {type(X)}, dtype: {X.dtype if hasattr(X, 'dtype') else 'N/A'}")
            print(f"X shape: {X.shape}")
            print(f"Looking for groups: {group1}, {group2}")
            if hasattr(X, 'columns'):
                print(f"X columns: {list(X.columns)}")
        
        # Method 1: Check if column names contain group names
        if hasattr(X, 'columns'):
            col_names = X.columns.tolist()
            group1_cols = [i for i, col in enumerate(col_names) if group1 in str(col)]
            group2_cols = [i for i, col in enumerate(col_names) if group2 in str(col)]
            
            if group1_cols and group2_cols:
                contrast = np.zeros(X.shape[1])
                for col_idx in group1_cols:
                    contrast[col_idx] = 1
                for col_idx in group2_cols:
                    contrast[col_idx] = -1
                
                if verbose:
                    print(f"Created contrast using column matching: {contrast}")
                return contrast
        
        # Method 2: Simple design with intercept and one grouping factor
        if X.shape[1] == 2:
            contrast = np.array([0, 1])
            if verbose:
                print(f"Created simple contrast for 2-column design: {contrast}")
            return contrast
        
        # Method 3: Multiple group columns (one-hot encoded)
        if X.shape[1] > 2:
            # Ensure X is a DataFrame for .iloc access
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
                
            sample_to_group = pd.Series(group_assignments.values, index=fit['mean'].columns)
            group1_samples = sample_to_group[sample_to_group == group1].index
            group2_samples = sample_to_group[sample_to_group == group2].index
            
            if verbose:
                print(f"Group1 ({group1}) samples: {len(group1_samples)}")
                print(f"Group2 ({group2}) samples: {len(group2_samples)}")
            
            contrast = np.zeros(X.shape[1])
            for col_idx in range(1, X.shape[1]):
                col_values = X.iloc[:, col_idx]
                
                # Ensure samples exist in the design matrix index
                group1_samples_in_X = group1_samples.intersection(X.index)
                group2_samples_in_X = group2_samples.intersection(X.index)
                
                if len(group1_samples_in_X) > 0 and len(group2_samples_in_X) > 0:
                    group1_vals = col_values[group1_samples_in_X].mean()
                    group2_vals = col_values[group2_samples_in_X].mean()
                    
                    if abs(group1_vals - group2_vals) > 0.5:
                        contrast[col_idx] = group1_vals - group2_vals
            
            if not np.allclose(contrast, 0):
                if verbose:
                    print(f"Created contrast using column analysis: {contrast}")
                return contrast.astype(np.float64)  # Ensure float64 dtype
        
        # Fallback
        warnings.warn(f"Could not automatically create contrast for {group1} vs {group2}. Using fallback.")
        contrast = np.zeros(X.shape[1])
        if X.shape[1] > 1:
            contrast[1] = 1
        
        if verbose:
            print(f"Created fallback contrast: {contrast}")
        return contrast.astype(np.float64)  # Ensure float64 dtype
        
    except Exception as e:
        print(f"ERROR creating contrast for {group1} vs {group2}: {e}")
        contrast = np.zeros(X.shape[1])
        if X.shape[1] > 1:
            contrast[1] = 1
        return contrast


def check_design_matrix(fit, verbose=True):
    """
    Check the design matrix structure and provide diagnostic information.
    """
    X = fit['X']
    group = fit['group']
    
    if verbose:
        print("\n" + "="*50)
        print("DESIGN MATRIX DIAGNOSTICS")
        print("="*50)
        print(f"X shape: {X.shape}")
        print(f"X type: {type(X)}")
        
        # Ensure X is numeric before computing rank
        if hasattr(X, 'values'):
            X_values = X.values
        else:
            X_values = X
        
        # Check if X_values is numeric
        if X_values.dtype == 'O':
            print("WARNING: X matrix has object dtype. This may cause issues.")
            print("X values sample:", X_values[:min(5, len(X_values))])
            # Try to identify which columns are problematic
            if hasattr(X, 'dtypes'):
                print("Column dtypes:")
                for col, dtype in X.dtypes.items():
                    print(f"  {col}: {dtype}")
        else:
            print(f"X rank: {np.linalg.matrix_rank(X_values)}")
        
        if hasattr(X, 'columns'):
            print(f"Column names: {list(X.columns)}")
        else:
            print("No column names (not a pandas DataFrame)")
        
        print(f"\nUnique groups: {group.unique()}")
        print(f"Group counts: {pd.Series(group).value_counts().to_dict()}")
        
        print(f"\nX matrix preview:")
        if hasattr(X, 'head'):
            print(X.head(10))
        else:
            print(X[:10] if len(X) > 10 else X)
        
        # Only compute X'X if X is numeric
        if X_values.dtype != 'O':
            print(f"\nX'X matrix:")
            XTX = X.T @ X
            print(XTX)
            print(f"X'X determinant: {np.linalg.det(XTX.values if hasattr(XTX, 'values') else XTX)}")
            print(f"X'X condition number: {np.linalg.cond(XTX.values if hasattr(XTX, 'values') else XTX)}")
            
            # Check for linear dependencies
            print(f"\nChecking for linear dependencies...")
            if X.shape[1] > 1:
                for i in range(X.shape[1]):
                    for j in range(i+1, X.shape[1]):
                        col1 = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]
                        col2 = X.iloc[:, j] if hasattr(X, 'iloc') else X[:, j]
                        corr = np.corrcoef(col1, col2)[0, 1]
                        if abs(corr) > 0.99:
                            print(f"WARNING: Columns {i} and {j} are highly correlated (r={corr:.4f})")
        else:
            print("\nSkipping X'X computation due to non-numeric dtype")
        
        print("="*50)


def run_pairwise_raisin_analysis(fit, output_dir, min_samples=2, fdrmethod='fdr_bh', 
                                 n_permutations=10, fdr_threshold=0.05, verbose=True):
    """
    Run pairwise RAISIN analysis between all groups with results and visualizations.
    
    Parameters:
    -----------
    fit : dict
        Output from raisinfit function
    output_dir : str
        Directory to save results and plots
    min_samples : int
        Minimum number of samples required per group
    fdrmethod : str
        Method for FDR correction
    n_permutations : int
        Number of permutations for degrees of freedom estimation
    fdr_threshold : float
        FDR threshold for significance
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    dict
        Dictionary containing all pairwise comparison results
    """
    
    try:
        # Validate and fix fit object first
        fit = validate_and_fix_fit_object(fit, verbose=verbose)
        
        # Check design matrix structure
        if verbose:
            check_design_matrix(fit, verbose=True)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"Output directory: {output_path.absolute()}")
        
        # Extract group information
        group_assignments = fit['group']
        unique_groups = group_assignments.unique()
        
        # Check sample counts per group
        group_counts = pd.Series(group_assignments).value_counts()
        valid_groups = group_counts[group_counts >= min_samples].index.tolist()
        
        if len(valid_groups) < 2:
            raise ValueError(f"Need at least 2 groups with >= {min_samples} samples each. "
                           f"Found {len(valid_groups)} valid groups: {valid_groups}")
        
        if verbose:
            print(f"Running pairwise comparisons between {len(valid_groups)} groups:")
            print(f"Groups: {valid_groups}")
            print(f"Sample counts: {group_counts[valid_groups].to_dict()}")
        
        # Generate all pairwise combinations
        group_pairs = list(combinations(valid_groups, 2))
        
        if verbose:
            print(f"\nTotal comparisons to perform: {len(group_pairs)}")
        
        # Store results for each comparison
        all_results = {}
        summary_data = []
        
        for i, (group1, group2) in enumerate(group_pairs):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Comparison {i+1}/{len(group_pairs)}: {group1} vs {group2}")
                print(f"{'='*60}")
            
            # Create contrast for this comparison
            contrast = create_group_contrast(fit, group1, group2, verbose=verbose)
            
            # Run test
            result = raisintest(fit, contrast=contrast, fdrmethod=fdrmethod, 
                              n_permutations=n_permutations, verbose=verbose)
            
            # Add comparison info
            result['comparison'] = f"{group1}_vs_{group2}"
            result['group1'] = group1
            result['group2'] = group2
            
            # Save results
            comparison_name = f"{group1}_vs_{group2}"
            all_results[comparison_name] = result
            
            # Save to CSV
            result.to_csv(output_path / f"{comparison_name}_results.csv")
            
            # Create individual volcano plot
            create_volcano_plot(result, group1, group2, output_path, fdr_threshold)
            
            # Collect summary data
            n_sig = (result['FDR'] < fdr_threshold).sum()
            n_up = ((result['Foldchange'] > 0) & (result['FDR'] < fdr_threshold)).sum()
            n_down = ((result['Foldchange'] < 0) & (result['FDR'] < fdr_threshold)).sum()
            
            summary_data.append({
                'comparison': comparison_name,
                'group1': group1,
                'group2': group2,
                'n_significant': n_sig,
                'n_upregulated': n_up,
                'n_downregulated': n_down,
                'max_abs_foldchange': result['Foldchange'].abs().max(),
                'min_pvalue': result['pvalue'].min(),
                'min_fdr': result['FDR'].min()
            })
        
        # Create summary dataframe
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / "pairwise_summary.csv", index=False)
        
        # Create visualizations
        create_summary_heatmap(summary_df, output_path, fdr_threshold)
        create_comparison_overview(summary_df, output_path)
        create_pvalue_distribution_plot(all_results, output_path)
        
        # Save top genes across all comparisons
        save_top_genes_across_comparisons(all_results, output_path, n_top=20)
        
        if verbose:
            print(f"\n{'='*60}")
            print("Analysis Complete!")
            print(f"Results saved to: {output_path.absolute()}")
            print(f"{'='*60}")
            print("Summary:")
            for _, row in summary_df.iterrows():
                print(f"{row['comparison']}: {row['n_significant']} significant genes "
                      f"({row['n_upregulated']} up, {row['n_downregulated']} down)")
        
        return all_results
        
    except Exception as e:
        print(f"ERROR in run_pairwise_raisin_analysis: {e}")
        traceback.print_exc()
        raise


def create_volcano_plot(result, group1, group2, output_path, fdr_threshold=0.05):
    """Create a volcano plot for a pairwise comparison."""
    plt.figure(figsize=(10, 8))
    
    # Calculate -log10(FDR)
    result['neg_log_fdr'] = -np.log10(result['FDR'])
    
    # Define colors based on significance and direction
    colors = []
    for _, row in result.iterrows():
        if row['FDR'] < fdr_threshold:
            if row['Foldchange'] > 0:
                colors.append('red')  # Upregulated
            else:
                colors.append('blue')  # Downregulated
        else:
            colors.append('gray')  # Not significant
    
    # Create scatter plot
    plt.scatter(result['Foldchange'], result['neg_log_fdr'], 
                c=colors, alpha=0.6, s=20)
    
    # Add horizontal line for significance threshold
    plt.axhline(y=-np.log10(fdr_threshold), color='black', linestyle='--', alpha=0.7)
    
    # Add vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Labels and title
    plt.xlabel('Log Fold Change', fontsize=12)
    plt.ylabel('-log10(FDR)', fontsize=12)
    plt.title(f'Volcano Plot: {group1} vs {group2}', fontsize=14)
    
    # Add legend
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='Upregulated')
    blue_patch = mpatches.Patch(color='blue', label='Downregulated')
    gray_patch = mpatches.Patch(color='gray', label='Not significant')
    plt.legend(handles=[red_patch, blue_patch, gray_patch])
    
    # Add text with counts
    n_up = np.sum((result['FDR'] < fdr_threshold) & (result['Foldchange'] > 0))
    n_down = np.sum((result['FDR'] < fdr_threshold) & (result['Foldchange'] < 0))
    plt.text(0.05, 0.95, f'Upregulated: {n_up}\nDownregulated: {n_down}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / f"{group1}_vs_{group2}_volcano.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_heatmap(summary_df, output_path, fdr_threshold):
    """Create a heatmap showing significant genes across all comparisons."""
    try:
        # Create a matrix for the heatmap
        groups = sorted(set(summary_df['group1']) | set(summary_df['group2']))
        n_groups = len(groups)
        
        # Initialize matrices as int type from the start
        sig_matrix = np.zeros((n_groups, n_groups), dtype=int)
        up_matrix = np.zeros((n_groups, n_groups), dtype=int)
        down_matrix = np.zeros((n_groups, n_groups), dtype=int)
        
        # Fill matrices
        for _, row in summary_df.iterrows():
            i = groups.index(row['group1'])
            j = groups.index(row['group2'])
            # Convert to int when assigning
            sig_matrix[i, j] = int(row['n_significant'])
            sig_matrix[j, i] = int(row['n_significant'])  # Make symmetric
            up_matrix[i, j] = int(row['n_upregulated'])
            down_matrix[i, j] = int(row['n_downregulated'])
            up_matrix[j, i] = int(row['n_downregulated'])  # Flip for symmetry
            down_matrix[j, i] = int(row['n_upregulated'])
        
        # Debug print to check matrix types
        print(f"Debug - sig_matrix dtype: {sig_matrix.dtype}, sample value: {sig_matrix[0,0]}, type: {type(sig_matrix[0,0])}")
        
        # Create the heatmap
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Use '.0f' format which works for both ints and floats and displays as integers
        # Significant genes heatmap
        sns.heatmap(sig_matrix, annot=True, fmt='.0f', ax=ax1, 
                    xticklabels=groups, yticklabels=groups,
                    cmap='YlOrRd', cbar_kws={'label': 'Number of significant genes'})
        ax1.set_title('Total Significant Genes')
        
        # Upregulated genes heatmap
        sns.heatmap(up_matrix, annot=True, fmt='.0f', ax=ax2,
                    xticklabels=groups, yticklabels=groups,
                    cmap='Reds', cbar_kws={'label': 'Number of upregulated genes'})
        ax2.set_title('Upregulated Genes (row vs column)')
        
        # Downregulated genes heatmap
        sns.heatmap(down_matrix, annot=True, fmt='.0f', ax=ax3,
                    xticklabels=groups, yticklabels=groups,
                    cmap='Blues', cbar_kws={'label': 'Number of downregulated genes'})
        ax3.set_title('Downregulated Genes (row vs column)')
        
        plt.tight_layout()
        plt.savefig(output_path / "summary_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"ERROR in create_summary_heatmap: {e}")
        print(f"Groups: {groups}")
        print(f"Summary dataframe columns: {summary_df.columns.tolist()}")
        print(f"Summary dataframe types: {summary_df.dtypes}")
        print(f"First row data: {summary_df.iloc[0].to_dict()}")
        traceback.print_exc()
        raise


def create_comparison_overview(summary_df, output_path):
    """Create overview plots of the comparison results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bar plot of significant genes per comparison
    ax1.bar(range(len(summary_df)), summary_df['n_significant'])
    ax1.set_xticks(range(len(summary_df)))
    ax1.set_xticklabels(summary_df['comparison'], rotation=45, ha='right')
    ax1.set_ylabel('Number of Significant Genes')
    ax1.set_title('Significant Genes per Comparison')
    
    # 2. Stacked bar plot of up/down regulated genes
    width = 0.8
    ax2.bar(range(len(summary_df)), summary_df['n_upregulated'], width, 
            label='Upregulated', color='red', alpha=0.7)
    ax2.bar(range(len(summary_df)), summary_df['n_downregulated'], width,
            bottom=summary_df['n_upregulated'], label='Downregulated', 
            color='blue', alpha=0.7)
    ax2.set_xticks(range(len(summary_df)))
    ax2.set_xticklabels(summary_df['comparison'], rotation=45, ha='right')
    ax2.set_ylabel('Number of Genes')
    ax2.set_title('Up/Down Regulated Genes per Comparison')
    ax2.legend()
    
    # 3. Scatter plot of max fold change vs number of significant genes
    ax3.scatter(summary_df['max_abs_foldchange'], summary_df['n_significant'])
    ax3.set_xlabel('Maximum |Fold Change|')
    ax3.set_ylabel('Number of Significant Genes')
    ax3.set_title('Fold Change vs Significance')
    
    # Add comparison names as labels
    for i, row in summary_df.iterrows():
        ax3.annotate(row['comparison'].replace('_vs_', '\nvs\n'), 
                    (row['max_abs_foldchange'], row['n_significant']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Distribution of minimum FDR values
    ax4.hist(summary_df['min_fdr'], bins=20, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Minimum FDR')
    ax4.set_ylabel('Number of Comparisons')
    ax4.set_title('Distribution of Minimum FDR Values')
    ax4.axvline(x=0.05, color='red', linestyle='--', label='FDR = 0.05')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "comparison_overview.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_pvalue_distribution_plot(all_results, output_path):
    """Create a plot showing p-value distributions across all comparisons."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Collect all p-values
    all_pvals = []
    for comparison, result in all_results.items():
        all_pvals.extend(result['pvalue'].values)
    
    # Histogram of p-values
    ax1.hist(all_pvals, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('P-value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of P-values (All Comparisons)')
    ax1.axvline(x=0.05, color='red', linestyle='--', label='p = 0.05')
    ax1.legend()
    
    # QQ-plot to check uniformity under null
    from scipy import stats as scipy_stats
    sorted_pvals = np.sort(all_pvals)
    n = len(sorted_pvals)
    expected = np.linspace(1/(n+1), n/(n+1), n)
    
    ax2.scatter(expected, sorted_pvals, alpha=0.5)
    ax2.plot([0, 1], [0, 1], 'r--', label='y=x (uniform distribution)')
    ax2.set_xlabel('Expected p-value (uniform)')
    ax2.set_ylabel('Observed p-value')
    ax2.set_title('QQ-plot: Expected vs Observed P-values')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "pvalue_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_top_genes_across_comparisons(all_results, output_path, n_top=20):
    """Save top genes across all comparisons."""
    # Collect top genes from each comparison
    top_genes_data = []
    
    for comparison, result in all_results.items():
        # Get top N genes
        top_genes = result.head(n_top)
        
        for gene_idx, row in top_genes.iterrows():
            top_genes_data.append({
                'gene': gene_idx,
                'comparison': comparison,
                'group1': row['group1'],
                'group2': row['group2'],
                'foldchange': row['Foldchange'],
                'pvalue': row['pvalue'],
                'fdr': row['FDR'],
                'rank': top_genes.index.get_loc(gene_idx) + 1
            })
    
    # Create dataframe and save
    top_genes_df = pd.DataFrame(top_genes_data)
    top_genes_df.to_csv(output_path / "top_genes_all_comparisons.csv", index=False)
    
    # Create summary of genes appearing in multiple comparisons
    gene_counts = top_genes_df['gene'].value_counts()
    frequent_genes = gene_counts[gene_counts > 1]
    
    if len(frequent_genes) > 0:
        # Create plot showing genes appearing in multiple comparisons
        plt.figure(figsize=(12, 8))
        frequent_genes.head(20).plot(kind='bar')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Gene')
        plt.ylabel('Number of Comparisons')
        plt.title(f'Top 20 Genes Appearing in Multiple Comparisons (Top {n_top})')
        plt.tight_layout()
        plt.savefig(output_path / "frequent_top_genes.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed information about frequent genes
        frequent_genes_detail = top_genes_df[top_genes_df['gene'].isin(frequent_genes.index)]
        frequent_genes_detail.to_csv(output_path / "frequent_top_genes_detail.csv", index=False)


# Example usage function
def example_usage():
    """
    Example of how to use the improved RAISIN test functions.
    """
    # This is just an example - replace with your actual fit object and paths
    print("Example usage:")
    print("1. First, run raisinfit to get your fit object")
    print("2. Then run the pairwise analysis:")
    print("""
    # Run pairwise analysis with outputs and visualizations
    results = run_pairwise_raisin_analysis(
        fit=your_fit_object,
        output_dir='./raisin_results',
        min_samples=2,
        fdrmethod='fdr_bh',
        n_permutations=10,
        fdr_threshold=0.05,
        verbose=True
    )
    """)
    print("3. Check the output directory for:")
    print("   - Individual comparison CSV files")
    print("   - Volcano plots for each comparison")
    print("   - Summary heatmaps")
    print("   - Comparison overview plots")
    print("   - P-value distribution plots")
    print("   - Top genes across all comparisons")


if __name__ == "__main__":
    example_usage()