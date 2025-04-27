import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.special import digamma, gamma, gammaln
import multiprocessing
import anndata as ad
import os
import warnings
from numpy.linalg import cholesky, inv, qr, svd
import os, warnings, multiprocessing as mp
from pathlib import Path
from typing import Dict, Tuple

def trigamma(x):
    """
    Implementation of the trigamma function (second derivative of the log of the gamma function)
    """
    return np.polygamma(1, x)  # polygamma(1, x) is equivalent to trigamma(x)

def trigamma_inverse(x):
    """
    Inverse of the trigamma function using numerical optimization
    """
    def objective(y):
        return trigamma(y) - x
    
    if x <= 0:
        return np.nan
    
    # Starting point based on approximation
    if x >= 1e-6:
        y0 = 0.5 + 1.0/x
    else:
        y0 = 1.0/x
    
    try:
        result = optimize.root(objective, y0)
        return result.x[0] if result.success else np.nan
    except:
        return np.nan

def generate_laguerre_quadrature(n):
    """
    Generate nodes and weights for Laguerre quadrature
    """
    from scipy.special import roots_laguerre
    x, w = roots_laguerre(n)
    return x, w

def raisin_fit(expr, sample, testtype, design, intercept=True, filtergene=False, 
               filtergenequantile=0.5, ncores=None):
    """
    Python implementation of RAISIN fitting method
    
    Parameters:
    - expr: A numpy array or matrix with genes as rows and cells as columns
    - sample: Array indicating which sample each cell comes from
    - testtype: Test type ('paired', 'unpaired', 'continuous', or 'custom')
    - design: Design information (format depends on test type)
    - intercept: Whether to add intercept to design matrix
    - filtergene: Whether to filter low expression genes
    - filtergenequantile: Quantile cutoff for gene filtering
    - ncores: Number of CPU cores to use
    
    Returns:
    Dictionary with fitted model components
    """
    if ncores is None:
        ncores = multiprocessing.cpu_count()
    
    # Check for duplicated row names
    if len(np.unique(expr.index)) < len(expr.index):
        print('Remove duplicated row names')
        expr = expr[~expr.index.duplicated()]
    
    # Setup design matrices based on test type
    if testtype != 'custom':
        samplename = design['sample'].values.astype(str)
        
        if testtype == 'unpaired':
            # Create design matrix for fixed effects
            if intercept:
                X = pd.get_dummies(design['feature'], drop_first=False)
                if intercept:
                    X.insert(0, 'intercept', 1)
            else:
                X = pd.get_dummies(design['feature'], drop_first=True)
            
            # Create design matrix for random effects
            Z = pd.get_dummies(design['sample'])
            group = design.loc[design['sample'].isin(Z.columns), 'feature'].values
        
        elif testtype == 'continuous':
            # For continuous features
            if intercept:
                X = pd.DataFrame({'intercept': np.ones(len(design)), 
                                  'feature': design['feature'].values})
            else:
                X = pd.DataFrame({'feature': design['feature'].values})
            
            # Random effects for samples
            Z = pd.get_dummies(design['sample'])
            group = np.repeat('group', Z.shape[1])
        
        elif testtype == 'paired':
            # Check if we have at least two pairs
            individual_counts = design['individual'].value_counts()
            n_pairs = sum(individual_counts == 2)
            
            if n_pairs < 2:
                print('Less than two pairs detected. Switch to unpaired test.')
                if intercept:
                    X = pd.get_dummies(design['feature'], drop_first=False)
                    if intercept:
                        X.insert(0, 'intercept', 1)
                else:
                    X = pd.get_dummies(design['feature'], drop_first=True)
                
                Z = pd.get_dummies(design['sample'])
                group = design.loc[design['sample'].isin(Z.columns), 'feature'].values
            else:
                # Set up for paired test
                if intercept:
                    X = pd.get_dummies(design['feature'], drop_first=False)
                    if intercept:
                        X.insert(0, 'intercept', 1)
                else:
                    X = pd.get_dummies(design['feature'], drop_first=True)
                
                # Create random effects for individuals
                Z = pd.get_dummies(design['individual'])
                
                # Create second random effect for differences within pairs
                tab = pd.Series(design['feature']).value_counts()
                max_group = tab.idxmax()
                Z2 = pd.get_dummies(design['sample'])
                Z2 = Z2.loc[:, design.loc[design['sample'].isin(Z2.columns), 'feature'] == max_group]
                
                # Combine the random effects
                Z = pd.concat([Z, Z2], axis=1)
                group = np.concatenate([
                    np.repeat('individual', Z.shape[1] - Z2.shape[1]),
                    np.repeat('difference', Z2.shape[1])
                ])
    else:
        # Custom test type
        X = design['X']
        Z = design['Z']
        group = design['group']
        samplename = X.index.values
    
    # Ensure all matrices have proper indices
    X.index = samplename
    Z.index = samplename
    
    # Calculate means across cells within each sample
    means = pd.DataFrame(index=expr.index, columns=samplename)
    for s in samplename:
        sample_cells = sample == s
        if sum(sample_cells) > 0:
            means[s] = expr.iloc[:, sample_cells].mean(axis=1)
    
    # Filter genes if requested
    if filtergene:
        m = means.values.flatten()
        m = m[~np.isnan(m)]
        m = np.quantile(m, filtergenequantile)
        gid = (means > m).sum(axis=1) > 0
        expr = expr.loc[gid]
        means = means.loc[gid]
    
    G = expr.shape[0]  # Number of genes
    
    # Generate Gaussian quadrature nodes and weights
    node, weight = generate_laguerre_quadrature(1000)
    useid = weight > 0
    node = node[useid]
    lognode = np.log(node)
    logweight = np.log(weight[useid])
    
    # Estimate cell-level variance (omega^2)
    omega2 = pd.DataFrame(index=expr.index, columns=samplename)
    for s in samplename:
        sample_cells = sample == s
        n_cells = sum(sample_cells)
        
        if n_cells > 1:
            # Calculate sample variance
            sample_expr = expr.iloc[:, sample_cells]
            sample_means = means[s]
            d = n_cells - 1
            
            # Calculate corrected sample variance
            s2 = ((sample_expr ** 2).mean(axis=1) - sample_means ** 2) * ((d + 1) / d)
            s2_positive = s2[s2 > 0]
            
            if len(s2_positive) > 0:
                stat = np.var(np.log(s2_positive)) - trigamma(d/2)
                
                if stat > 0:
                    theta = trigamma_inverse(stat)
                    phi = np.exp(np.mean(np.log(s2_positive)) - digamma(d/2) + digamma(theta)) * d/2
                    
                    if theta + d/2 > 1:
                        omega2[s] = (d * s2 / 2 + phi) / (theta + d/2 - 1)
                    else:
                        # Numerical integration for more complex cases
                        def calc_omega2(ss2):
                            alpha = theta + d/2
                            beta = d * ss2 / 2 + phi
                            return (beta ** alpha / gamma(alpha)) * np.sum(
                                np.exp(node - alpha * lognode - beta / node + logweight))
                        
                        omega2[s] = [calc_omega2(ss2) if ss2 > 0 else np.nan for ss2 in s2]
                else:
                    omega2[s] = np.exp(np.mean(np.log(s2_positive)))
        else:
            omega2[s] = np.nan
    
    # Handle samples with no variance estimates
    zid = [col for col in omega2.columns if omega2[col].isna().all()]
    nzid = [col for col in omega2.columns if col not in zid]
    
    if len(zid) > 0 and len(nzid) > 0:
        # Calculate distances between design points
        Xarr = X.values
        Xdist = np.zeros((len(samplename), len(samplename)))
        for i in range(len(samplename)):
            for j in range(len(samplename)):
                Xdist[i, j] = np.sqrt(np.sum((Xarr[i] - Xarr[j]) ** 2))
        
        Xdist_df = pd.DataFrame(Xdist, index=samplename, columns=samplename)
        
        # Fill in missing variances based on closest design points
        for sid in zid:
            if len(nzid) == 1:
                omega2[sid] = omega2[nzid[0]]
            else:
                dists = Xdist_df.loc[sid, nzid]
                tarid = dists[dists == dists.min()].index.tolist()
                omega2[sid] = omega2[tarid].mean(axis=1)
    
    # Adjust by number of cells in each sample
    wl = pd.Series({s: sum(sample == s) for s in samplename})
    for s in omega2.columns:
        omega2[s] = omega2[s] / wl[s]
    
    # Define function to estimate sample-level variance components
    def sigma2_func(currentgroup, controlgroup, donegroup, sigma2):
        print(f"Estimating sigma2 for group: {currentgroup}")
        
        # Combine fixed effects with already estimated random effects
        Xl = pd.concat([X, Z.loc[:, [c in controlgroup for c in group]]], axis=1)
        Zl = Z.loc[:, [g == currentgroup for g in group]]
        
        # Get rows that involve either current or control random effects
        Z_combined = Z.loc[:, [g in [currentgroup] + controlgroup for g in group]]
        lid = (Z_combined.sum(axis=1) > 0)
        
        if sum(lid) == 0:
            warnings.warn(f"No samples involved in group {currentgroup}, setting variance to 0")
            return np.zeros(G)
        
        Xl = Xl.loc[lid]
        Zl = Zl.loc[lid]
        
        # Make X full rank
        q, r = qr(Xl.values, mode='economic')
        rank = np.sum(np.abs(np.diag(r)) > 1e-10)
        Xl = Xl.iloc[:, :rank]
        
        n = sum(lid)
        p = n - Xl.shape[1]
        
        if p == 0:
            warnings.warn(f"Unable to estimate variance for group {currentgroup}, setting to 0")
            return np.zeros(G)
        
        # Generate contrast matrix K
        K = np.random.normal(size=(n, p))
        for i in range(p):
            b = Xl.values
            if i > 0:
                b = np.column_stack([b, K[:, :i]])
            
            # Orthogonalize K[:,i] with respect to b
            b_inv = np.linalg.pinv(b.T @ b)
            K[:, i] = K[:, i] - b @ (b_inv @ (b.T @ K[:, i]))
        
        # Normalize columns of K
        for i in range(p):
            K[:, i] = K[:, i] / np.sqrt(np.sum(K[:, i] ** 2))
        
        K = K.T  # p × n
        
        # Indices for selected samples
        sel_samples = [s for i, s in enumerate(samplename) if lid.iloc[i]]
        
        # Calculate components for variance estimation
        pl = np.dot(K, means.loc[:, sel_samples].T).T  # G × p
        qlm = np.dot(K, np.dot(Zl, Zl.T)) @ K.T  # p × p
        ql = np.diag(qlm)  # p
        
        # Calculate component related to cell-level variance
        rl = np.dot(omega2.loc[:, sel_samples], K.T ** 2)  # G × p
        
        # Add components from already estimated variance groups
        for sg in donegroup:
            KZmat = K @ Z.loc[lid, [g == sg for g in group]].values
            KZmat = KZmat @ KZmat.T
            
            for g in range(G):
                rl[g, :] += sigma2[g, sg] * np.diag(KZmat)
        
        # Estimate variance component
        # Method of moments estimation
        M = np.mean(np.maximum(0, (pl**2 - rl) / ql))
        V = np.mean(np.maximum(0, (pl**4 - 3*rl**2 - 6*M*ql*rl) / (3*ql**2)))
        
        try:
            alpha = M**2 / (V - M**2)
            gamma = M / (V - M**2)
            print(f"alpha={alpha} beta={gamma}")
            
            if np.isnan(alpha) or np.isnan(gamma) or alpha <= 0 or gamma <= 0:
                raise ValueError("Invalid hyperparameters")
                
            # Bayes estimation with valid hyperparameters
            def process_gene(gene_idx):
                p_vals = pl[gene_idx, :]
                r_vals = rl[gene_idx, :]
                
                # Compute denominator
                tmp_x = np.outer(p_vals, p_vals)
                tmp_w = omega2.iloc[gene_idx, [samplename.index(s) for s in sel_samples]].values
                t2 = np.dot(K.T, (tmp_w * K.T).T)
                
                # Numerical integration
                res = np.zeros_like(node)
                for i, gn in enumerate(node):
                    try:
                        cm = cholesky(gn * qlm + t2)
                        log_det = 2 * np.sum(np.log(np.diag(cm)))
                        cm_inv = inv(cm @ cm.T)
                        res[i] = -log_det - np.sum(tmp_x * cm_inv)
                    except:
                        res[i] = -np.inf
                
                # Calculate posterior expectation
                tmp = logweight + node + res/2 + (alpha-1) * lognode - gamma * node
                tmp_max = np.max(tmp)
                est = np.sum(np.exp(tmp + lognode - tmp_max)) / np.sum(np.exp(tmp - tmp_max))
                
                return est if not np.isnan(est) and est != np.inf else 0
            
            # Process all genes
            with multiprocessing.Pool(ncores) as pool:
                sigma2_est = pool.map(process_gene, range(G))
            
            return np.array(sigma2_est)
            
        except Exception as e:
            print(f"Error in estimating variance: {e}")
            print("Proceeding without variance pooling")
            
            # Alternative approach without pooling
            sigma2_est = np.zeros(G)
            
            def minimize_function(gene_idx, s2):
                p_vals = pl[gene_idx, :]
                r_vals = rl[gene_idx, :]
                return np.sum((s2*ql**2 + ql*r_vals - p_vals**2*ql) / (s2*ql + r_vals)**2)
            
            for g in range(G):
                try:
                    result = optimize.minimize_scalar(
                        lambda s2: minimize_function(g, s2), 
                        bounds=(0, 1000), 
                        method='bounded'
                    )
                    sigma2_est[g] = result.x if result.success else 0
                except:
                    sigma2_est[g] = 0
            
            return sigma2_est
    
    # Estimate sample-level variance components
    unique_groups = np.unique(group)
    sigma2 = pd.DataFrame(0, index=expr.index, columns=unique_groups)
    
    # Order groups by number of parameters
    group_npara = {ug: np.sum(Z.loc[:, [g == ug for g in group]].values != 0) for ug in unique_groups}
    group_order = sorted(unique_groups, key=lambda g: group_npara[g], reverse=True)
    
    tmp_control_group = list(unique_groups)
    tmp_done_group = []
    fail_group = []
    
    for ug in group_order:
        control_group = [g for g in tmp_control_group if g != ug]
        sigma2[ug] = sigma2_func(ug, control_group, tmp_done_group, sigma2)
        tmp_control_group.remove(ug)
        tmp_done_group.append(ug)
    
    # Prepare results dictionary
    results = {
        'mean': means,
        'sigma2': sigma2,
        'omega2': omega2,
        'X': X,
        'Z': Z,
        'group': group,
        'fail_group': fail_group
    }
    
    return results

def raisin_test(fit, coef=2, contrast=None, fdr_method='fdr'):
    """
    Perform statistical testing using the fitted RAISIN model
    Improved version based on the R implementation

    Parameters:
    - fit: Output from raisin_fit function
    - coef: Which coefficient to test if contrast is None
    - contrast: Contrast vector for hypothesis testing
    - fdr_method: Method for FDR calculation ('fdr', 'BH', etc.)
    
    Returns:
    DataFrame with fold changes, test statistics, p-values and FDR
    """
    # Extract components from fit
    X = fit['X']
    means = fit['mean']
    G = means.shape[0]  # Number of genes
    group = fit['group']
    Z = fit['Z']
    
    # Set up contrast if not provided
    if contrast is None:
        contrast = np.zeros(X.shape[1])
        contrast[coef-1] = 1  # Convert from 1-based to 0-based indexing
    
    # Check if we can estimate variance
    if len(set(group)) == len(fit['fail_group']):
        warnings.warn('Unable to estimate variance for all random effects. Setting FDR to 1.')
        # Calculate fold changes
        k = contrast @ np.linalg.inv(X.T @ X) @ X.T
        b = means @ k
        
        # Return results with FDR set to 1
        results = pd.DataFrame({
            'Foldchange': b,
            'FDR': np.ones_like(b)
        }, index=means.index)
        
        # Sort by absolute fold change
        return results.iloc[np.argsort(-np.abs(results['Foldchange'].values))]
    
    # Calculate the linear transformation matrix
    k = contrast @ np.linalg.inv(X.T @ X) @ X.T
    k = k.reshape(1, -1)  # Make k a row vector
    
    # Calculate fold changes
    b = (means @ k.T).values.flatten()
    
    # Calculate variances
    a = np.zeros(G)
    for g_idx, g in enumerate(group):
        Z_col = Z.iloc[:, g_idx].values.reshape(-1, 1)
        a += (k @ Z_col)**2 * fit['sigma2'][g].values
    
    # Add cell-level variance
    for s in fit['omega2'].columns:
        a += k[:, X.index.get_indexer([s])[0]]**2 * fit['omega2'][s].values
    
    # Calculate test statistics
    stat = b / np.sqrt(a)
    
    # Perform permutation test to determine degrees of freedom
    simu_stats = []
    n_perm = 10
    
    for i in range(n_perm):
        # Permute the design matrix
        perm_idx = np.random.permutation(X.shape[0])
        perm_X = X.values[perm_idx]
        
        # Calculate new k
        perm_k = contrast @ np.linalg.inv(perm_X.T @ perm_X) @ perm_X.T
        perm_k = perm_k.reshape(1, -1)
        
        # Calculate new a
        perm_a = np.zeros(G)
        for g_idx, g in enumerate(group):
            Z_col = Z.iloc[:, g_idx].values.reshape(-1, 1)
            perm_a += (perm_k @ Z_col)**2 * fit['sigma2'][g].values
        
        # Add cell-level variance
        for s in fit['omega2'].columns:
            col_idx = X.index.get_indexer([s])[0]
            if col_idx < len(perm_k[0]):
                perm_a += perm_k[0, col_idx]**2 * fit['omega2'][s].values
        
        # Calculate permuted statistics
        perm_b = (means @ perm_k.T).values.flatten()
        perm_stat = perm_b / np.sqrt(perm_a)
        simu_stats.extend(perm_stat)
    
    simu_stats = np.array(simu_stats)
    
    # Determine the best-fitting distribution
    # Calculate log-likelihood for normal distribution
    pnorm = np.sum(stats.norm.logpdf(simu_stats))
    
    # Try different degrees of freedom for t-distribution
    df_range = np.arange(1, 100.1, 0.1)
    pt_values = []
    
    for df in df_range:
        pt_values.append(np.sum(stats.t.logpdf(simu_stats, df=df)))
    
    pt_values = np.array(pt_values)
    
    # Determine if t or normal distribution fits better
    if np.max(pt_values) > pnorm:
        # Use t-distribution
        best_df = df_range[np.argmax(pt_values)]
        pval = 2 * stats.t.sf(np.abs(stat), df=best_df)
        print(f"Using t-distribution with df={best_df}")
    else:
        # Use normal distribution
        pval = 2 * stats.norm.sf(np.abs(stat))
        print("Using normal distribution")
    
    # Calculate FDR
    if fdr_method == 'fdr':
        from statsmodels.stats.multitest import fdrcorrection
        _, fdr = fdrcorrection(pval)
    else:
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(pval, method=fdr_method)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Foldchange': b,
        'stat': stat,
        'pvalue': pval,
        'FDR': fdr
    }, index=means.index)
    
    # Sort by FDR then absolute statistic
    return results.sort_values(['FDR', 'stat'], ascending=[True, False])

def RAISIN(
    base_path: str | Path,
    sample_to_clade: Dict[str, str],
    *,
    min_clade_size: int = 2,          # drop clades with <2 samples
    filtergenequantile: float = 0.50,
    ncores: int | None = None,
    verbose: bool = True
) -> Tuple[dict, pd.DataFrame]:
    """
    Run the RAISIN analysis end-to-end.

    Parameters
    ----------
    base_path
        Folder that contains `pseudobulk/expression.csv`
        and `harmony/adata_sample.h5ad`.
    sample_to_clade
        Map from sample → clade.  Samples missing here will be discarded.
    min_clade_size
        Clades represented by fewer than this many samples are discarded.
    filtergenequantile
        Passed straight into `raisin_fit`.
    ncores
        Cores for multiprocessing; default = all available.
    verbose
        If True, emits progress / warnings.

    Returns
    -------
    fit_results  : dict   -  output of `raisin_fit`
    test_results : DataFrame
    """
    base_path  = Path(base_path).expanduser()
    expression_path = base_path / "pseudobulk" / "expression.csv"
    adata_path      = base_path / "harmony"   / "adata_sample.h5ad"

    # ------------------------------------------------------------------ load —
    if verbose: print("🔹 Loading pseudobulk expression …")
    expression = pd.read_csv(expression_path, index_col=0).T     # genes × samples

    if verbose: print("🔹 Loading single-cell AnnData …")
    adata = ad.read_h5ad(adata_path, backed="r")                 # keep on disk
    cell_to_sample = pd.Series(adata.obs["sample"].values,
                               index=adata.obs_names, dtype="category")

    # ----------------------------------------------------------------- screen —
    samples = expression.columns.tolist()

    missing   = [s for s in samples           if s not in sample_to_clade]
    present   = [s for s in samples           if s in  sample_to_clade]
    if missing and verbose:
        warnings.warn(f"{len(missing)} sample(s) have no clade assignment "
                      f"and will be dropped: {missing}", stacklevel=2)
        expression = expression[present]                    # drop cols
        cell_mask  = cell_to_sample.isin(present)
        cell_to_sample = cell_to_sample[cell_mask]          # drop cells too

    # ----------------------------------------------------------------- design —
    design = pd.DataFrame({"sample": present,
                           "feature": [sample_to_clade[s] for s in present]})

    # drop clades that now have < min_clade_size samples
    clade_counts = design["feature"].value_counts()
    good_clades  = clade_counts[clade_counts >= min_clade_size].index
    if len(good_clades) < len(clade_counts) and verbose:
        bad = list(set(clade_counts.index) - set(good_clades))
        warnings.warn(f"Discarding {len(bad)} clade(s) with <{min_clade_size} "
                      f"samples: {bad}", stacklevel=2)

    design      = design[design["feature"].isin(good_clades)].reset_index(drop=True)
    expression  = expression[design["sample"]]               # align again
    cell_mask   = cell_to_sample.isin(design["sample"])
    cell_to_sample = cell_to_sample[cell_mask]

    # ----------------------------------------------------------------- fit  —
    if verbose: print("🔹 Fitting RAISIN …")
    fit_results = raisin_fit(
        expr                 = expression,
        sample               = cell_to_sample,
        testtype             = "unpaired",
        design               = design,
        filtergene           = True,
        filtergenequantile   = filtergenequantile,
        ncores               = ncores or mp.cpu_count()
    )

    # ----------------------------------------------------------------- test —
    if verbose: print("🔹 Testing fixed-effect contrast …")
    test_results = raisin_test(fit_results)

    out_file = base_path / "raisin_results.csv"
    test_results.to_csv(out_file)
    if verbose:
        print(f"\n✅ Results saved → {out_file}")
        print(f"   genes tested      : {len(test_results):>6}")
        print(f"   FDR < 0.05 genes  : {(test_results['FDR'] < 0.05).sum():>6}")

    return fit_results, test_results