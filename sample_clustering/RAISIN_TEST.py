"""
RAISIN Testing - Python port of R RAISIN package

This module performs statistical testing for the RAISIN model.
Tests if a coefficient or combination of coefficients in the fixed effects
design matrix is 0, generating corresponding p-values and FDRs.

Author: Original R code by Zhicheng Ji, Wenpin Hou, Hongkai Ji
Python port maintains compatibility with R implementation.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
import warnings
from statsmodels.stats.multitest import multipletests
import traceback

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
from statsmodels.stats.multitest import multipletests
import traceback
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path

def raisintest(
    fit,
    coef: int = 2,
    contrast=None,
    fdrmethod: str = 'fdr_bh',
    n_permutations: int = 100,  # UPDATED: Increased default from 10 to 100 for stability
    output_dir: str = None,
    min_samples: int = None,
    fdr_threshold: float = 0.05,
    make_volcano: bool = True,
    verbose: bool = True,
):
    """
    Statistical testing for RAISIN model.
    
    IMPROVEMENTS:
    - Default n_permutations increased to 100 to prevent df=1.0 (Cauchy) estimation.
    - Added checks for singular design matrices when no contrast is provided.
    """
    
    try:
        # -----------------------------------------------------------------
        # 1. Extract Data
        # -----------------------------------------------------------------
        X = fit['X']
        if isinstance(X, pd.DataFrame):
            X_df = X
            X = X.values.astype(np.float64)
            x_colnames = list(X_df.columns)
        else:
            X = np.array(X, dtype=np.float64)
            x_colnames = [f"coef_{i}" for i in range(X.shape[1])]
        
        means = fit['mean']
        if isinstance(means, pd.DataFrame):
            gene_names = np.array(means.index)
            means = means.values.astype(np.float64)
        else:
            gene_names = np.arange(means.shape[0])
            means = np.array(means, dtype=np.float64)
        
        G = means.shape[0]
        
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
            sigma2_columns = list(sigma2.columns)
        else:
            sigma2_values = np.array(sigma2, dtype=np.float64)
            sigma2_columns = list(range(sigma2_values.shape[1]))
        
        omega2 = fit['omega2']
        if isinstance(omega2, pd.DataFrame):
            omega2 = omega2.values.astype(np.float64)
        else:
            omega2 = np.array(omega2, dtype=np.float64)
            
        failgroup = fit.get('failgroup', [])

        # -----------------------------------------------------------------
        # 2. Filter Genes (Optional)
        # -----------------------------------------------------------------
        if min_samples is not None:
            # Gene is "detected" if mean > 0 (approximation for RAISIN mean output)
            sample_counts = np.sum(means > 0, axis=1)
            gene_mask = sample_counts >= min_samples
            
            if verbose:
                print(f"Filtering: {np.sum(~gene_mask)} genes removed (< {min_samples} samples)")
            
            means = means[gene_mask, :]
            gene_names = np.array(gene_names)[gene_mask]
            sigma2_values = sigma2_values[gene_mask, :]
            omega2 = omega2[gene_mask, :]
            G = means.shape[0]

        # -----------------------------------------------------------------
        # 3. Handle Contrast
        # -----------------------------------------------------------------
        if contrast is None:
            # Default behavior: test single coefficient
            contrast = np.zeros(X.shape[1])
            if coef < 1 or coef > X.shape[1]:
                raise ValueError(f"coef must be between 1 and {X.shape[1]}")
            contrast[coef - 1] = 1
            contrast_label = x_colnames[coef - 1]
            
            # Warning for intercept models
            if "(Intercept)" in x_colnames and np.sum(contrast) != 0:
                if verbose:
                    print(f"Warning: Testing single coefficient '{contrast_label}' in model with Intercept.")
                    print("Ensure this is what you want. For group comparisons, use a difference contrast.")
        else:
            contrast = np.array(contrast, dtype=np.float64)
            contrast_label = "custom_contrast"

        # -----------------------------------------------------------------
        # 4. Calculate Statistics
        # -----------------------------------------------------------------
        XTX = X.T @ X
        # Use pseudoinverse for rank-deficient matrices (common in over-parameterized designs)
        XTX_inv = np.linalg.pinv(XTX)
        
        k = contrast @ XTX_inv @ X.T
        b = means @ k  # Fold change
        
        # Check for failure groups
        unique_groups = np.unique(group)
        if failgroup and set(unique_groups).issubset(set(failgroup)):
            warnings.warn('All groups failed variance estimation. Returning FDR=1.')
            return pd.DataFrame({'Foldchange': b, 'FDR': 1.0}, index=gene_names)
        
        # Calculate Variance (a)
        kZ = k @ Z
        sigma2_by_group = np.zeros((G, len(group)))
        for i, g in enumerate(group):
            if g in sigma2_columns:
                col_idx = sigma2_columns.index(g)
                sigma2_by_group[:, i] = sigma2_values[:, col_idx]
        
        random_var = np.sum((kZ ** 2) * sigma2_by_group, axis=1)
        fixed_var = np.sum((k ** 2) * omega2, axis=1)
        a = random_var + fixed_var
        
        # t-statistic
        with np.errstate(divide='ignore', invalid='ignore'):
            stat = b / np.sqrt(a)
        stat = np.where(np.isfinite(stat), stat, 0)

        # -----------------------------------------------------------------
        # 5. Permutations (Degrees of Freedom Estimation)
        # -----------------------------------------------------------------
        if verbose:
            print(f"Running {n_permutations} permutations...")
            
        simu_stat = []
        # Pre-calculate inverse for speed if possible, but X changes per perm
        for _ in range(n_permutations):
            perm_idx = np.random.permutation(X.shape[0])
            perX = X[perm_idx, :]
            
            perXTX = perX.T @ perX
            perXTX_inv = np.linalg.pinv(perXTX)
            perm_k = contrast @ perXTX_inv @ perX.T
            
            perm_kZ = perm_k @ Z
            perm_random_var = np.sum((perm_kZ ** 2) * sigma2_by_group, axis=1)
            perm_fixed_var = np.sum((perm_k ** 2) * omega2, axis=1)
            perm_a = perm_random_var + perm_fixed_var
            
            with np.errstate(divide='ignore', invalid='ignore'):
                perm_stat = (means @ perm_k) / np.sqrt(perm_a)
            
            perm_stat = perm_stat[np.isfinite(perm_stat)]
            
            # Optimization: Don't store millions of points if G is huge
            # Just take a random subsample if G > 1000 to keep memory low
            if len(perm_stat) > 2000:
                 perm_stat = np.random.choice(perm_stat, 2000, replace=False)
            
            simu_stat.extend(perm_stat.tolist())
            
        simu_stat = np.array(simu_stat)
        
        # -----------------------------------------------------------------
        # 6. Fit Distribution (Normal vs t)
        # -----------------------------------------------------------------
        if len(simu_stat) == 0:
            pval = 2 * stats.norm.sf(np.abs(stat))
        else:
            pnorm_ll = np.sum(stats.norm.logpdf(simu_stat))
            
            # Scan df from 1 to 100
            df_range = np.arange(1, 100.1, 0.5)
            # Vectorize the logpdf calculation if possible, or loop
            best_ll = -np.inf
            best_df = 100
            
            for df_val in df_range:
                ll = np.sum(stats.t.logpdf(simu_stat, df=df_val))
                if ll > best_ll:
                    best_ll = ll
                    best_df = df_val
            
            if best_ll > pnorm_ll:
                if verbose:
                    print(f"Estimated distribution: t-distribution (df={best_df:.1f})")
                pval = 2 * stats.t.sf(np.abs(stat), df=best_df)
            else:
                if verbose:
                    print("Estimated distribution: Normal")
                pval = 2 * stats.norm.sf(np.abs(stat))

        # -----------------------------------------------------------------
        # 7. FDR and Output
        # -----------------------------------------------------------------
        _, fdr, _, _ = multipletests(pval, method=fdrmethod)
        
        res = pd.DataFrame({
            'Foldchange': b,
            'stat': stat,
            'pvalue': pval,
            'FDR': fdr
        }, index=gene_names)
        
        # Sort
        res = res.iloc[np.lexsort((-np.abs(res['stat'].values), res['FDR'].values))]
        
        if verbose:
            n_sig = np.sum(res['FDR'] < fdr_threshold)
            print(f"Found {n_sig} significant genes (FDR < {fdr_threshold})")

        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            res.to_csv(os.path.join(output_dir, "raisin_results.csv"))
            if make_volcano:
                _create_volcano_plot(
                    res, 
                    title=f"Contrast: {contrast_label}", 
                    output_path=os.path.join(output_dir, "volcano_plot.png"),
                    fdr_threshold=fdr_threshold
                )

        return res

    except Exception as e:
        print(f"ERROR in raisintest: {e}")
        traceback.print_exc()
        raise

def run_pairwise_tests(
    fit,
    output_dir,
    groups_to_compare=None,
    control_group=None,
    fdrmethod='fdr_bh',
    n_permutations=100,
    fdr_threshold=0.05,
    verbose=True
):
    """
    Automatically runs pairwise comparisons between groups.
    Fixed to handle integer column names.
    """
    
    # Extract column names from X to find indices
    X = fit['X']
    if isinstance(X, pd.DataFrame):
        x_cols = list(X.columns)
    else:
        # Fallback if X is numpy array
        unique_grps = np.unique(fit['group'])
        x_cols = ["(Intercept)"] + list(unique_grps)

    # Clean group names
    available_groups = np.unique(fit['group'])
    if groups_to_compare is None:
        groups_to_compare = available_groups
    
    # Generate pairs
    pairs = []
    if control_group:
        # Ensure control_group is compared as a string for safety
        control_group_str = str(control_group)
        avail_str = [str(g) for g in available_groups]
        
        if control_group_str not in avail_str:
            raise ValueError(f"Control group '{control_group}' not found in data.")
            
        for g in groups_to_compare:
            if str(g) != control_group_str:
                pairs.append((g, control_group)) # (Test, Control)
    else:
        pairs = list(combinations(groups_to_compare, 2))

    if verbose:
        print(f"Starting pairwise comparisons. Found {len(pairs)} pairs to test.")

    results_summary = {}

    for g1, g2 in pairs:
        # Define comparison name: g1_vs_g2 (Testing g1 - g2)
        comp_name = f"{g1}_vs_{g2}"
        if verbose:
            print(f"\n--- Testing: {g1} (Positive) vs {g2} (Negative) ---")
        
        # --- BUILD CONTRAST VECTOR ---
        idx_g1 = -1
        idx_g2 = -1
        
        # FIX: Convert everything to string for comparison to avoid AttributeError
        g1_str = str(g1)
        g2_str = str(g2)
        
        for i, col in enumerate(x_cols):
            col_str = str(col) # Convert column name to string safely
            
            # Check exact match or if column name ends with group name
            # (handles cases like 'feature_1' vs '1')
            if col_str == g1_str or col_str.endswith(g1_str):
                idx_g1 = i
            if col_str == g2_str or col_str.endswith(g2_str):
                idx_g2 = i
        
        # Check if we found the columns
        if idx_g1 == -1 or idx_g2 == -1:
            print(f"Skipping {comp_name}: Could not find corresponding columns in Design Matrix.")
            print(f"Available columns: {x_cols}")
            continue

        contrast = np.zeros(len(x_cols))
        contrast[idx_g1] = 1
        contrast[idx_g2] = -1
        
        # --- RUN TEST ---
        sub_dir = os.path.join(output_dir, comp_name)
        try:
            res = raisintest(
                fit,
                contrast=contrast,
                fdrmethod=fdrmethod,
                n_permutations=n_permutations,
                output_dir=sub_dir,
                fdr_threshold=fdr_threshold,
                verbose=True 
            )
            results_summary[comp_name] = np.sum(res['FDR'] < fdr_threshold)
            
        except Exception as e:
            print(f"Failed comparison {comp_name}: {e}")
            traceback.print_exc()

    if verbose:
        print("\n===== Pairwise Comparison Summary =====")
        for pair, count in results_summary.items():
            print(f"{pair}: {count} significant genes")
    
    return results_summary

def _create_volcano_plot(result, title, output_path, fdr_threshold=0.05):
    """Internal helper for volcano plots"""
    plt.figure(figsize=(6, 5))
    
    # Handle extremely small FDRs
    fdr = result['FDR'].values
    fdr = np.maximum(fdr, 1e-300) # avoid log(0)
    neg_log_fdr = -np.log10(fdr)
    fc = result['Foldchange'].values
    
    # Color logic
    colors = ['grey'] * len(fc)
    colors = np.array(colors, dtype=object)
    
    sig_mask = fdr < fdr_threshold
    up_mask = (fc > 0) & sig_mask
    down_mask = (fc < 0) & sig_mask
    
    colors[up_mask] = '#E64B35' # Red
    colors[down_mask] = '#3C5488' # Blue
    
    plt.scatter(fc, neg_log_fdr, c=colors, s=10, alpha=0.6, linewidths=0)
    
    plt.axhline(-np.log10(fdr_threshold), color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    
    plt.title(title)
    plt.xlabel("Log Fold Change")
    plt.ylabel("-log10(FDR)")
    
    # Add counts
    n_up = np.sum(up_mask)
    n_down = np.sum(down_mask)
    
    plt.text(0.02, 0.98, f"Up: {n_up}\nDown: {n_down}", 
             transform=plt.gca().transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
             
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

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
        # Try to infer from group assignments (left as in original)
        group_arr = fit['group']
        sample_names = fit.get('sample_names', X_df.index if hasattr(X_df, 'index') else None)
        # Default: test second coefficient if inference fails
        if len(X_cols) > 1:
            contrast[1] = 1
    
    if verbose:
        print(f"Created contrast: {contrast}")
        print(f"X columns: {X_cols}")
    
    return contrast


# Visualization functions
def create_volcano_plot(result, group1, group2, output_path, fdr_threshold=0.05):
    """Create a volcano plot for a comparison."""
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
    plt.text(
        0.05, 0.95,
        f'Up: {n_up}\nDown: {n_down}',
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
