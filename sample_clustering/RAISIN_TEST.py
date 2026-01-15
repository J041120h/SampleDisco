"""
RAISIN Testing - Python port of R RAISIN package

This module performs statistical testing for the RAISIN model.
Tests if a coefficient or combination of coefficients in the fixed effects
design matrix is 0, generating corresponding p-values and FDRs.

Author: Original R code by Zhicheng Ji, Wenpin Hou, Hongkai Ji
Python port maintains compatibility with R implementation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
import warnings
from statsmodels.stats.multitest import multipletests
import traceback


def raisintest(fit, coef=2, contrast=None, fdrmethod='fdr_bh', n_permutations=10, verbose=True):
    """
    Statistical testing for RAISIN model.
    
    This function performs statistical testing that tests if a coefficient or 
    a combination of coefficients in the design matrix of fixed effects is 0,
    and generates corresponding p-value and FDRs.
    
    Parameters
    ----------
    fit : dict
        The output from the raisinfit function.
    coef : int
        Index of coefficient to test (1-indexed like R, default=2).
        Only used if contrast is None.
    contrast : array-like or None
        Numeric vector indicating the combination of coefficients.
        Must have the same length as the number of columns in X.
    fdrmethod : str
        Method for FDR correction. Options: 'fdr_bh', 'fdr_by', 'bonferroni', etc.
        Default 'fdr_bh' corresponds to R's 'fdr' method.
    n_permutations : int
        Number of permutations for estimating degrees of freedom (default 10, matching R).
    verbose : bool
        Whether to print progress information.
    
    Returns
    -------
    pandas.DataFrame
        Results with columns: Foldchange, stat, pvalue, FDR
        Sorted by FDR ascending, then by |stat| descending.
    
    Notes
    -----
    The test statistic follows a t-distribution. A permutation procedure is used
    to estimate the degree of freedom. The p-value is calculated as the tail 
    probability of the t-distribution.
    """
    
    try:
        # -----------------------------------------------------------------
        # Extract components from fit (R lines 19-26)
        # -----------------------------------------------------------------
        X = fit['X']
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float64)
        else:
            X = np.array(X, dtype=np.float64)
        
        means = fit['mean']
        if isinstance(means, pd.DataFrame):
            gene_names = means.index
            means = means.values.astype(np.float64)
        else:
            gene_names = np.arange(means.shape[0])
            means = np.array(means, dtype=np.float64)
        
        G = means.shape[0]  # number of genes
        
        group = fit['group']
        if isinstance(group, pd.Series):
            group = group.values
        group = np.array(group, dtype=str)
        
        Z = fit['Z']
        if isinstance(Z, pd.DataFrame):
            Z = Z.values.astype(np.float64)
        else:
            Z = np.array(Z, dtype=np.float64)
        
        sigma2 = fit['sigma2']
        if isinstance(sigma2, pd.DataFrame):
            sigma2_values = sigma2.values.astype(np.float64)
            sigma2_columns = sigma2.columns
        else:
            sigma2_values = np.array(sigma2, dtype=np.float64)
            sigma2_columns = np.arange(sigma2_values.shape[1])
        
        omega2 = fit['omega2']
        if isinstance(omega2, pd.DataFrame):
            omega2 = omega2.values.astype(np.float64)
        else:
            omega2 = np.array(omega2, dtype=np.float64)
        
        failgroup = fit.get('failgroup', [])
        if failgroup is None:
            failgroup = []
        
        if verbose:
            print(f"Testing {G} genes")
            print(f"Design matrix X shape: {X.shape}")
            print(f"Random effects Z shape: {Z.shape}")
        
        # -----------------------------------------------------------------
        # Set up contrast vector (R lines 24-27)
        # -----------------------------------------------------------------
        if contrast is None:
            contrast = np.zeros(X.shape[1])
            # R uses 1-indexed, so coef=2 means second column
            if coef < 1 or coef > X.shape[1]:
                raise ValueError(f"coef must be between 1 and {X.shape[1]}")
            contrast[coef - 1] = 1
        else:
            contrast = np.array(contrast, dtype=np.float64)
        
        if len(contrast) != X.shape[1]:
            raise ValueError(f"Contrast length ({len(contrast)}) must match "
                           f"number of columns in X ({X.shape[1]})")
        
        if verbose:
            print(f"Contrast vector: {contrast}")
        
        # -----------------------------------------------------------------
        # Calculate k = t(contrast) %*% solve(t(X) %*% X) %*% t(X)
        # (R line 28)
        # -----------------------------------------------------------------
        XTX = X.T @ X
        try:
            XTX_inv = np.linalg.solve(XTX, np.eye(XTX.shape[0]))
        except np.linalg.LinAlgError:
            if verbose:
                print("Warning: X'X is singular, using pseudoinverse")
            XTX_inv = np.linalg.pinv(XTX)
        
        k = contrast @ XTX_inv @ X.T  # Shape: (n_samples,)
        
        # -----------------------------------------------------------------
        # Calculate fold changes: b = (means %*% t(k))[,1] (R line 29)
        # -----------------------------------------------------------------
        b = means @ k  # Shape: (G,)
        
        # -----------------------------------------------------------------
        # Check if all groups failed (R lines 30-33)
        # -----------------------------------------------------------------
        unique_groups = np.unique(group)
        if len(failgroup) > 0 and set(unique_groups) == set(failgroup):
            warnings.warn('Unable to estimate variance for all random effects. Setting FDR to 1.')
            res = pd.DataFrame({
                'Foldchange': b,
                'FDR': 1.0
            }, index=gene_names)
            # Sort by absolute fold change descending
            res = res.iloc[np.argsort(-np.abs(res['Foldchange'].values))]
            return res
        
        # -----------------------------------------------------------------
        # Calculate variance: a (R line 35)
        # a = colSums((k %*% Z)[1,]^2 * t(fit$sigma2[,group])) + 
        #     colSums(k[1,]^2 * t(fit$omega2))
        # -----------------------------------------------------------------
        # kZ = k %*% Z -> (1 x n_random_effects), but k is 1D so result is 1D
        kZ = k @ Z  # Shape: (n_random_effects,)
        
        # Build sigma2 indexed by group for each column of Z
        # sigma2[,group] selects columns of sigma2 according to group labels
        # This creates a (G x n_random_effects) matrix
        sigma2_by_group = np.zeros((G, len(group)))
        for i, g in enumerate(group):
            if g in sigma2_columns:
                col_idx = list(sigma2_columns).index(g)
                sigma2_by_group[:, i] = sigma2_values[:, col_idx]
        
        # a = sum over random effects of (kZ^2 * sigma2[,group]) + sum over samples of (k^2 * omega2)
        # First term: (kZ^2) is (n_random_effects,), sigma2_by_group is (G x n_random_effects)
        # Result should be (G,)
        random_var = np.sum((kZ ** 2) * sigma2_by_group, axis=1)  # (G,)
        
        # Second term: k^2 is (n_samples,), omega2 is (G x n_samples)
        fixed_var = np.sum((k ** 2) * omega2, axis=1)  # (G,)
        
        a = random_var + fixed_var  # (G,)
        
        # -----------------------------------------------------------------
        # Calculate test statistics (R line 36)
        # -----------------------------------------------------------------
        with np.errstate(divide='ignore', invalid='ignore'):
            stat = b / np.sqrt(a)
        stat = np.where(np.isfinite(stat), stat, 0)
        
        # -----------------------------------------------------------------
        # Permutation test to estimate df (R lines 38-43)
        # -----------------------------------------------------------------
        if verbose:
            print(f"Running {n_permutations} permutations to estimate degrees of freedom...")
        
        simu_stat = []
        for sim_id in range(n_permutations):
            # Permute rows of X (R line 39)
            perm_idx = np.random.permutation(X.shape[0])
            perX = X[perm_idx, :]
            
            # Recalculate k with permuted X (R line 40)
            perXTX = perX.T @ perX
            try:
                perXTX_inv = np.linalg.solve(perXTX, np.eye(perXTX.shape[0]))
            except np.linalg.LinAlgError:
                perXTX_inv = np.linalg.pinv(perXTX)
            
            perm_k = contrast @ perXTX_inv @ perX.T
            
            # Recalculate variance (R line 41)
            perm_kZ = perm_k @ Z
            perm_sigma2_by_group = np.zeros((G, len(group)))
            for i, g in enumerate(group):
                if g in sigma2_columns:
                    col_idx = list(sigma2_columns).index(g)
                    perm_sigma2_by_group[:, i] = sigma2_values[:, col_idx]
            
            perm_random_var = np.sum((perm_kZ ** 2) * perm_sigma2_by_group, axis=1)
            perm_fixed_var = np.sum((perm_k ** 2) * omega2, axis=1)
            perm_a = perm_random_var + perm_fixed_var
            
            # Calculate permuted statistics (R line 42)
            with np.errstate(divide='ignore', invalid='ignore'):
                perm_stat = (means @ perm_k) / np.sqrt(perm_a)
            
            perm_stat = perm_stat[np.isfinite(perm_stat)]
            simu_stat.extend(perm_stat.tolist())
        
        simu_stat = np.array(simu_stat)
        
        # -----------------------------------------------------------------
        # Determine distribution and calculate p-values (R lines 45-55)
        # -----------------------------------------------------------------
        if len(simu_stat) == 0:
            warnings.warn("All permutation statistics are invalid. Using normal distribution.")
            pval = 2 * stats.norm.sf(np.abs(stat))
            if verbose:
                print("Using normal distribution (fallback)")
        else:
            # Log-likelihood under normal distribution (R line 45)
            pnorm_ll = np.sum(stats.norm.logpdf(simu_stat))
            
            # Log-likelihood under t-distribution for various df (R lines 46-48)
            df_range = np.arange(1, 100.1, 0.1)
            pt_lls = np.array([np.sum(stats.t.logpdf(simu_stat, df=df)) for df in df_range])
            
            # Choose best distribution (R lines 50-55)
            if np.max(pt_lls) > pnorm_ll:
                best_df = df_range[np.argmax(pt_lls)]
                pval = 2 * stats.t.sf(np.abs(stat), df=best_df)
                if verbose:
                    print(f"Using t-distribution with df={best_df:.1f}")
            else:
                pval = 2 * stats.norm.sf(np.abs(stat))
                if verbose:
                    print("Using normal distribution")
        
        # -----------------------------------------------------------------
        # Multiple testing correction (R line 57)
        # -----------------------------------------------------------------
        # R's p.adjust with method='fdr' uses BH method
        if fdrmethod == 'fdr_bh' or fdrmethod == 'fdr':
            _, fdr, _, _ = multipletests(pval, method='fdr_bh')
        elif fdrmethod == 'fdr_by':
            _, fdr, _, _ = multipletests(pval, method='fdr_by')
        elif fdrmethod == 'bonferroni':
            _, fdr, _, _ = multipletests(pval, method='bonferroni')
        else:
            _, fdr, _, _ = multipletests(pval, method=fdrmethod)
        
        # -----------------------------------------------------------------
        # Create results dataframe (R lines 58-60)
        # -----------------------------------------------------------------
        res = pd.DataFrame({
            'Foldchange': b,
            'stat': stat,
            'pvalue': pval,
            'FDR': fdr
        }, index=gene_names)
        
        # Sort by FDR ascending, then by |stat| descending (R line 59)
        # R: res[order(res[,4], -abs(res[,2])),]
        sort_keys = np.lexsort((-np.abs(res['stat'].values), res['FDR'].values))
        res = res.iloc[sort_keys]
        
        if verbose:
            n_sig = np.sum(res['FDR'] < 0.05)
            print(f"Testing complete. Found {n_sig} genes with FDR < 0.05")
        
        return res
    
    except Exception as e:
        print(f"ERROR in raisintest: {e}")
        traceback.print_exc()
        raise


def create_contrast_for_groups(fit, group1, group2, verbose=False):
    """
    Create a contrast vector for comparing two groups.
    
    This is a helper function for when the design matrix has multiple groups
    and you want to test the difference between two specific groups.
    
    Parameters
    ----------
    fit : dict
        Output from raisinfit
    group1 : str
        First group name (will be positive in contrast)
    group2 : str
        Second group name (will be negative in contrast)
    verbose : bool
        Print diagnostic information
    
    Returns
    -------
    np.ndarray
        Contrast vector for use with raisintest
    """
    X = fit['X']
    if isinstance(X, pd.DataFrame):
        X_df = X
        X_cols = list(X.columns)
    else:
        X_df = pd.DataFrame(X)
        X_cols = list(range(X.shape[1]))
    
    # Try to find columns corresponding to the groups
    group1_cols = [i for i, col in enumerate(X_cols) if str(group1) in str(col)]
    group2_cols = [i for i, col in enumerate(X_cols) if str(group2) in str(col)]
    
    contrast = np.zeros(len(X_cols))
    
    if group1_cols and group2_cols:
        for i in group1_cols:
            contrast[i] = 1.0 / len(group1_cols)
        for i in group2_cols:
            contrast[i] = -1.0 / len(group2_cols)
    elif len(X_cols) == 2:
        # Simple design with intercept + one group indicator
        contrast[1] = 1
    else:
        # Try to infer from group assignments
        group_arr = fit['group']
        sample_names = fit.get('sample_names', X_df.index if hasattr(X_df, 'index') else None)
        
        if sample_names is not None and len(group_arr) == len(sample_names):
            # Find which samples belong to each group
            # This is more complex and depends on how groups are encoded
            pass
        
        # Default: test second coefficient
        if len(X_cols) > 1:
            contrast[1] = 1
    
    if verbose:
        print(f"Created contrast: {contrast}")
        print(f"X columns: {X_cols}")
    
    return contrast


def run_pairwise_tests(fit, output_dir=None, fdrmethod='fdr_bh', n_permutations=10, 
                       verbose=True):
    """
    Run pairwise comparisons between all unique groups in the fit object.
    
    Parameters
    ----------
    fit : dict
        Output from raisinfit
    output_dir : str, optional
        Directory to save results (if None, results are only returned)
    fdrmethod : str
        FDR correction method
    n_permutations : int
        Number of permutations for df estimation
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        Dictionary of results, keyed by comparison name
    """
    from itertools import combinations
    from pathlib import Path
    
    # Get unique groups from the group array
    unique_groups = np.unique(fit['group'])
    
    if len(unique_groups) < 2:
        warnings.warn("Less than 2 groups found. Cannot perform pairwise comparisons.")
        return {}
    
    if verbose:
        print(f"Found {len(unique_groups)} unique groups: {unique_groups}")
    
    results = {}
    
    for group1, group2 in combinations(unique_groups, 2):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Comparing {group1} vs {group2}")
            print(f"{'='*50}")
        
        contrast = create_contrast_for_groups(fit, group1, group2, verbose=verbose)
        
        try:
            result = raisintest(fit, contrast=contrast, fdrmethod=fdrmethod,
                               n_permutations=n_permutations, verbose=verbose)
            
            comparison_name = f"{group1}_vs_{group2}"
            result['comparison'] = comparison_name
            results[comparison_name] = result
            
            if output_dir is not None:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                result.to_csv(output_path / f"{comparison_name}_results.csv")
                
        except Exception as e:
            warnings.warn(f"Failed to compare {group1} vs {group2}: {e}")
            continue
    
    return results


# Visualization functions
def create_volcano_plot(result, group1, group2, output_path, fdr_threshold=0.05):
    """Create a volcano plot for a pairwise comparison."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    
    # Calculate -log10(FDR)
    neg_log_fdr = -np.log10(result['FDR'].values + 1e-300)  # Add small value to avoid log(0)
    foldchange = result['Foldchange'].values
    fdr = result['FDR'].values
    
    # Define colors
    colors = np.where(
        fdr < fdr_threshold,
        np.where(foldchange > 0, 'red', 'blue'),
        'gray'
    )
    
    plt.scatter(foldchange, neg_log_fdr, c=colors, alpha=0.6, s=20)
    
    # Add threshold line
    plt.axhline(y=-np.log10(fdr_threshold), color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.xlabel('Log Fold Change', fontsize=12)
    plt.ylabel('-log10(FDR)', fontsize=12)
    plt.title(f'Volcano Plot: {group1} vs {group2}', fontsize=14)
    
    # Add counts
    n_up = np.sum((fdr < fdr_threshold) & (foldchange > 0))
    n_down = np.sum((fdr < fdr_threshold) & (foldchange < 0))
    plt.text(0.05, 0.95, f'Up: {n_up}\nDown: {n_down}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("RAISIN Test Module")
    print("Usage:")
    print("  from raisintest import raisintest")
    print("  result = raisintest(fit, coef=2)")
    print("  # or with custom contrast:")
    print("  result = raisintest(fit, contrast=[0, 1, -1])")