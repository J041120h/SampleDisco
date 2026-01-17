"""
RAISIN Fitting - Python port of R RAISIN package

This module fits the RAISIN model and estimates the cell-level and sample-level variance.
The output will be used by raisintest function to generate p-values and FDRs.

Author: Original R code by Zhicheng Ji, Wenpin Hou, Hongkai Ji
Python port maintains compatibility with R implementation.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.special import digamma, gamma, polygamma
from scipy.spatial.distance import pdist, squareform
import warnings
from multiprocessing import cpu_count
import traceback
from joblib import Parallel, delayed


# ---------------------------------------------------------------------
#  Trigamma utilities (matching R's trigamma and trigammaInverse)
# ---------------------------------------------------------------------
def trigamma(x):
    """Wrapper around scipy.special.polygamma(1, x) - matches R's trigamma."""
    return polygamma(1, x)


def trigamma_inverse(x):
    """
    Inverse of trigamma via Newton iterations.
    Matches R's limma::trigammaInverse.
    """
    if x <= 0:
        raise ValueError("trigamma_inverse requires x > 0")

    # Starting guess based on asymptotic approximation
    if x >= 1e7:
        return 1.0 / np.sqrt(x)

    if x < 1e-6:
        return 1.0 / x

    # Newton-Raphson iteration
    y = 0.5 + 1.0 / x  # starting guess

    for _ in range(50):
        tri_y = trigamma(y)
        delta = (tri_y - x) / polygamma(2, y)
        y_new = y - delta

        # Keep positive
        if y_new <= 0:
            y_new = y / 2.0

        if abs(y_new - y) < 1e-8 * y:
            break
        y = y_new

    return y


# ---------------------------------------------------------------------
#  Laguerre–Gauss quadrature (matching R's statmod::gauss.quad)
# ---------------------------------------------------------------------
def gauss_quad_laguerre(n=1000):
    """
    Gauss-Laguerre quadrature rule.
    Matches R's statmod::gauss.quad(n, "laguerre").

    Returns nodes and weights for integrating f(x)*exp(-x) from 0 to infinity.
    """
    from numpy.polynomial.laguerre import laggauss

    # Try with requested number of points, fall back if overflow
    for n_try in [n, 500, 200, 100, 50]:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                nodes, weights = laggauss(n_try)
                return nodes, weights
        except (RuntimeWarning, ValueError, OverflowError):
            continue

    # Final fallback
    return laggauss(50)


# ---------------------------------------------------------------------
#  Design matrix construction helpers
# ---------------------------------------------------------------------
def model_matrix_no_intercept(factor_series):
    """
    Create design matrix without intercept (like R's model.matrix(~.-1, ...)).
    Returns a dummy-coded matrix with all levels.
    """
    dummies = pd.get_dummies(factor_series, prefix="", prefix_sep="")
    return dummies.values, dummies.columns.tolist()


def model_matrix_with_intercept(factor_series):
    """
    Create design matrix with intercept (like R's model.matrix(~., ...)).
    Returns intercept + dummy-coded matrix (drops first level for identifiability).
    """
    # For categorical: intercept + k-1 dummies
    if factor_series.dtype == "object" or isinstance(factor_series.iloc[0], str):
        dummies = pd.get_dummies(factor_series, drop_first=False)
        X = np.column_stack([np.ones(len(factor_series)), dummies.values])
        col_names = ["(Intercept)"] + list(dummies.columns)
    else:
        # For continuous: intercept + the variable itself
        X = np.column_stack([np.ones(len(factor_series)), factor_series.values])
        col_names = ["(Intercept)", "feature"]
    return X, col_names

# ---------------------------------------------------------------------
#  QR-based rank reduction (matching R's qr()$pivot logic)
# ---------------------------------------------------------------------
def qr_rank_reduce(X):
    """
    Reduce matrix to full column rank using QR decomposition.
    Matches R's: X[, qr(X)$pivot[seq_len(qr(X)$rank)]]
    """
    from scipy.linalg import qr as scipy_qr
    
    # Use pivoted QR to select linearly independent columns
    Q, R, P = scipy_qr(X, pivoting=True)
    
    # Determine numerical rank
    tol = max(X.shape) * np.finfo(float).eps * np.abs(R).max()
    rank = np.sum(np.abs(np.diag(R[: min(X.shape), :])) > tol)
    
    # Select first 'rank' pivoted columns
    selected_cols = P[:rank]
    return X[:, selected_cols]

# ---------------------------------------------------------------------
#  Cholesky-based inverse (matching R's chol2inv(chol(...)))
# ---------------------------------------------------------------------
def chol2inv(X):
    """
    Compute inverse using Cholesky decomposition.
    Matches R's chol2inv(chol(X)).
    """
    try:
        L = np.linalg.cholesky(X)
        L_inv = np.linalg.inv(L)
        return L_inv.T @ L_inv
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse for singular matrices
        return np.linalg.pinv(X)


# ---------------------------------------------------------------------
#  ComBat batch correction
# ---------------------------------------------------------------------
def apply_combat_correction(adata, batch_col, group_col=None, sample_col=None, verbose=True):
    """
    Apply ComBat batch correction to expression data.

    Parameters
    ----------
    adata : AnnData
        AnnData object with expression data
    batch_col : str
        Column in adata.obs containing batch information
    group_col : str, optional
        Column in adata.obs containing biological grouping to preserve
    sample_col : str, optional
        Column in adata.obs containing sample IDs (for pseudobulk data)
    verbose : bool
        Print progress messages

    Returns
    -------
    AnnData
        New AnnData object with batch-corrected expression in .X
    """
    try:
        from combat.pycombat import pycombat
    except ImportError:
        raise ImportError(
            "ComBat correction requires the 'combat' package. "
            "Install it with: pip install combat"
        )

    if verbose:
        print(f"\n===== Applying ComBat batch correction =====")
        print(f"Batch variable: {batch_col}")

    # Get expression matrix
    if adata.raw is not None and adata.raw.X is not None:
        expr = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else np.array(adata.raw.X)
        gene_names = adata.raw.var_names
    else:
        expr = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        gene_names = adata.var_names

    # Create expression dataframe (genes x cells for pycombat)
    expr_df = pd.DataFrame(expr.T, columns=adata.obs_names, index=gene_names)

    # Get batch information
    batch = adata.obs[batch_col].values

    # NOTE: pyComBat expects `mod` to be a *list* (or list of lists), not a NumPy array
    mod = []
    if group_col is not None and group_col in adata.obs.columns:
        if verbose:
            print(f"Preserving biological effects from: {group_col}")
        # Single covariate: pass one list of values (length = n_samples)
        covariate_values = list(adata.obs[group_col].astype(str))
        mod = covariate_values


    # Apply ComBat
    if verbose:
        print("Running ComBat correction...")

    corrected_expr = pycombat(expr_df, batch=batch, mod=mod)

    # Create new AnnData with corrected expression
    adata_corrected = adata.copy()
    adata_corrected.X = corrected_expr.T.values

    # Store original data in layers
    if adata.raw is not None:
        adata_corrected.layers["original_raw"] = adata.raw.X.copy()
    else:
        adata_corrected.layers["original"] = expr

    if verbose:
        print("ComBat correction complete")
        print(f"Original expression stored in layer: 'original' or 'original_raw'")

    return adata_corrected


# ---------------------------------------------------------------------
#  Main RAISIN function
# ---------------------------------------------------------------------
def raisinfit(
    adata,
    sample_col,
    testtype="unpaired",
    group_col=None,
    individual_col=None,
    batch_col=None,
    sample_to_clade=None,
    custom_design=None,
    intercept=True,
    filtergene=False,
    filtergenequantile=0.5,
    n_jobs=None,
    verbose=True,
):
    """
    Python port of RAISIN differential-expression model fitting.

    NOTE (change from previous behavior):
    - If group_col is provided (and exists in adata.obs), it is used for sample grouping
      and sample_to_clade is ignored (not needed / not used).

    Parameters
    ----------
    adata : AnnData
        AnnData object (single-cell or pseudobulk).
    sample_col : str
        Column in adata.obs identifying which sample each cell/observation belongs to.
    testtype : str
        One of 'paired', 'unpaired', 'continuous', or 'custom'.
    group_col : str, optional
        Column in adata.obs containing the grouping/feature variable.
    individual_col : str, optional
        Column in adata.obs containing individual/subject IDs.
    batch_col : str, optional
        Column in adata.obs containing batch IDs.
    sample_to_clade : dict, optional
        Alternative to group_col: explicit mapping {sample_id -> group}.
        Ignored if group_col is provided and exists in adata.obs.
    custom_design : dict, optional
        For testtype='custom': dict with keys 'X', 'Z', 'group'.
    intercept : bool
        Include intercept in fixed-effect design matrix (default True).
    filtergene : bool
        Filter out lowly expressed genes (default False).
    filtergenequantile : float
        Quantile threshold for gene filtering (default 0.5).
    n_jobs : int, optional
        Number of CPU cores for parallel computation. Default uses all cores.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Dictionary containing:
        - 'mean': gene x sample mean expression matrix
        - 'sigma2': gene x group variance components
        - 'omega2': gene x sample cell-level variance
        - 'X': fixed effects design matrix
        - 'Z': random effects design matrix
        - 'group': variance component grouping
        - 'failgroup': groups where variance estimation failed
        - 'sample_names': ordered sample names
        - 'batch_corrected': boolean indicating if ComBat was applied
    """

    if verbose:
        print("\n===== Starting RAISIN fitting =====")

    # -----------------------------------------------------------------
    #  Setup
    # -----------------------------------------------------------------
    if n_jobs in (None, -1):
        n_jobs = cpu_count()
    if verbose:
        print(f"Using {n_jobs} CPU cores")

    if verbose:
        print(f"Available columns in adata.obs: {list(adata.obs.columns)}")

    # Validate sample_col
    if sample_col not in adata.obs.columns:
        raise KeyError(
            f"Column '{sample_col}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    # Priority rule:
    # If group_col is provided and exists in obs, use it and ignore sample_to_clade.
    use_group_col = (group_col is not None) and (group_col in adata.obs.columns)
    use_sample_to_clade = (not use_group_col) and (sample_to_clade is not None)

    # Normalize "empty dict" to effectively None to avoid downstream errors
    if use_sample_to_clade and isinstance(sample_to_clade, dict) and len(sample_to_clade) == 0:
        sample_to_clade = None
        use_sample_to_clade = False

    if verbose and use_group_col and sample_to_clade is not None:
        print(
            f"Note: group_col='{group_col}' provided; sample_to_clade is ignored (not needed)."
        )

    # -----------------------------------------------------------------
    #  Apply ComBat batch correction if requested (and not custom design)
    # -----------------------------------------------------------------
    batch_corrected = False
    if batch_col is not None and testtype != "custom":
        if batch_col not in adata.obs.columns:
            raise KeyError(
                f"batch_col '{batch_col}' not found in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )

        if verbose:
            print(f"\nBatch correction requested using column: {batch_col}")

        # Determine biological grouping to preserve
        preserve_group = None

        # If group_col is provided, use it directly for biological covariate
        if use_group_col:
            preserve_group = group_col
            if verbose:
                print(f"Preserving biological effects from group_col: {group_col}")

        # If sample_to_clade is provided but no valid group_col, create temporary column
        elif use_sample_to_clade and sample_to_clade is not None:
            # Create sample-level mapping for biological groups
            sample_to_group_map = {str(k): v for k, v in sample_to_clade.items()}
            
            # Map each cell to its biological group via sample
            sample_ids = adata.obs[sample_col].astype(str)
            bio_groups = sample_ids.map(sample_to_group_map)
            
            # Check if mapping was successful
            if bio_groups.isna().any():
                unmapped = sample_ids[bio_groups.isna()].unique()
                if verbose:
                    print(f"Warning: {len(unmapped)} samples not found in sample_to_clade mapping")
            
            adata.obs["_combat_preserve_group"] = bio_groups
            preserve_group = "_combat_preserve_group"
            if verbose:
                print(f"Preserving biological effects from sample_to_clade mapping")

        # Apply ComBat
        adata = apply_combat_correction(
            adata,
            batch_col=batch_col,
            group_col=preserve_group,
            sample_col=sample_col,
            verbose=verbose,
        )
        batch_corrected = True

        # Clean up temporary column if created
        if preserve_group == "_combat_preserve_group":
            adata.obs.drop(columns=["_combat_preserve_group"], inplace=True)

    elif batch_col is not None and testtype == "custom":
        if verbose:
            print(
                f"\nWarning: batch_col provided but testtype='custom'. "
                f"Batch correction is skipped for custom designs. "
                f"Please apply batch correction manually if needed."
            )

    # -----------------------------------------------------------------
    #  Expression matrix (genes x cells)
    # -----------------------------------------------------------------
    if adata.raw is not None and adata.raw.X is not None:
        expr = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else np.array(adata.raw.X)
        gene_names = np.array(adata.raw.var_names)
        if verbose:
            print("Using raw counts from adata.raw.X")
    else:
        expr = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        gene_names = np.array(adata.var_names)
        if verbose:
            print("Using counts from adata.X")

    # Transpose to genes x cells (R convention)
    expr = expr.T

    # Sample IDs per cell
    sample = np.array(adata.obs[sample_col].values)

    # -----------------------------------------------------------------
    #  Remove duplicated genes (matching R line 47-51)
    # -----------------------------------------------------------------
    if len(gene_names) != len(set(gene_names)):
        if verbose:
            print("Removing duplicated gene names")
        _, keep_idx = np.unique(gene_names, return_index=True)
        keep_idx = np.sort(keep_idx)  # Preserve original order
        expr = expr[keep_idx, :]
        gene_names = gene_names[keep_idx]

    # -----------------------------------------------------------------
    #  Build design matrices based on testtype
    # -----------------------------------------------------------------
    failgroup = []

    if testtype == "custom":
        # -----------------------------------------
        # Custom design (R lines 85-90)
        # -----------------------------------------
        if custom_design is None:
            raise ValueError("custom_design dict required for testtype='custom'")

        X = np.array(custom_design["X"])
        Z = np.array(custom_design["Z"])
        group = np.array(custom_design["group"], dtype=str)

        if hasattr(custom_design["X"], "index"):
            sample_names = np.array(custom_design["X"].index)
        else:
            sample_names = np.array(list(range(X.shape[0])))

        if verbose:
            print("Using custom design matrices")

    else:
        # -----------------------------------------
        # Build design from adata metadata
        # -----------------------------------------

        # Get unique samples
        unique_samples = np.unique(sample)

        # Determine feature values for each sample
        if use_group_col:
            # Use group_col from adata.obs (preferred if provided)
            # Since batch is sample-level, we can simplify: just pick any cell from each sample
            sample_to_group = {}
            for s in unique_samples:
                # Get first cell from this sample
                first_cell_idx = np.where(sample == s)[0][0]
                sample_to_group[s] = adata.obs.iloc[first_cell_idx][group_col]
            
            feature_values = np.array([sample_to_group[s] for s in unique_samples])

            # Get individual values if needed for paired test
            if testtype == "paired":
                if individual_col is None:
                    raise ValueError("individual_col required for testtype='paired'")
                if individual_col not in adata.obs.columns:
                    raise KeyError(f"individual_col '{individual_col}' not found in adata.obs")

                sample_to_individual = {}
                for s in unique_samples:
                    # Get first cell from this sample
                    first_cell_idx = np.where(sample == s)[0][0]
                    sample_to_individual[s] = adata.obs.iloc[first_cell_idx][individual_col]
                
                individual_values = np.array([sample_to_individual[s] for s in unique_samples])
            else:
                individual_values = None

            if verbose:
                print(f"Using group_col='{group_col}' for grouping")

        elif use_sample_to_clade and sample_to_clade is not None:
            # Use explicit mapping
            sample_to_clade_str = {str(k): v for k, v in sample_to_clade.items()}
            if len(sample_to_clade_str) == 0:
                raise ValueError("sample_to_clade was provided but is empty.")

            # Convert all sample names to strings for consistent comparison
            unique_samples_str = [str(s) for s in unique_samples]

            common_samples = [
                s for s_str, s in zip(unique_samples_str, unique_samples) if s_str in sample_to_clade_str
            ]

            if len(common_samples) == 0:
                sample_examples = list(unique_samples[:5])
                clade_key_examples = list(sample_to_clade_str.keys())[:5]  # safe even if empty
                data_type = type(unique_samples[0]) if len(unique_samples) > 0 else None
                key_type = type(next(iter(sample_to_clade_str.keys()))) if len(sample_to_clade_str) > 0 else None
                raise ValueError(
                    f"No samples in data match keys in sample_to_clade.\n"
                    f"Sample names in data (first 5): {sample_examples}\n"
                    f"Sample names in sample_to_clade (first 5): {clade_key_examples}\n"
                    f"Sample types: data={data_type}, sample_to_clade keys={key_type}"
                )

            common_samples = np.array(common_samples)

            # Filter to only samples in mapping
            valid_mask = np.isin(sample, common_samples)
            expr = expr[:, valid_mask]
            sample = sample[valid_mask]
            unique_samples = common_samples

            # Get feature values using string keys
            feature_values = np.array([sample_to_clade_str[str(s)] for s in unique_samples])
            individual_values = None

            if verbose:
                print(f"Using sample_to_clade mapping for {len(unique_samples)} samples")

        else:
            # If group_col was specified but doesn't exist, fail fast with a clearer error
            if group_col is not None and group_col not in adata.obs.columns:
                raise KeyError(
                    f"group_col '{group_col}' not found in adata.obs. "
                    f"Available: {list(adata.obs.columns)}"
                )
            raise ValueError("Either a valid group_col must be provided, or a non-empty sample_to_clade mapping must be provided.")

        sample_names = unique_samples
        n_samples = len(sample_names)

        # -----------------------------------------
        # Build Z matrix based on testtype (R lines 55-78)
        # -----------------------------------------
        if testtype == "unpaired":
            # Z: identity matrix for samples (R lines 56-58)
            Z = np.eye(n_samples)
            Z_colnames = list(sample_names)
            # group: feature value for each sample
            group = np.array([str(f) for f in feature_values])

        elif testtype == "continuous":
            # Z: identity matrix, all in same variance group (R lines 60-62)
            Z = np.eye(n_samples)
            Z_colnames = list(sample_names)
            group = np.array(["group"] * n_samples)

        elif testtype == "paired":
            # Check if we have enough pairs (R lines 63-77)
            if individual_values is None:
                raise ValueError("individual_col required for paired test")

            ind_counts = pd.Series(individual_values).value_counts()
            n_pairs = (ind_counts == 2).sum()

            if n_pairs < 2:
                if verbose:
                    print("Less than two pairs detected. Switching to unpaired test.")
                # Fall back to unpaired (R lines 66-68)
                Z = np.eye(n_samples)
                Z_colnames = list(sample_names)
                group = np.array([str(f) for f in feature_values])
            else:
                # Paired design (R lines 69-77)
                # Z1: individual effects
                unique_individuals = np.unique(individual_values)
                Z1 = np.zeros((n_samples, len(unique_individuals)))
                for i, ind in enumerate(unique_individuals):
                    Z1[individual_values == ind, i] = 1

                # Z2: diagonal for the larger group (difference component)
                feature_counts = pd.Series(feature_values).value_counts()
                larger_group = feature_counts.idxmax()
                Z2_mask = feature_values == larger_group
                Z2 = np.eye(n_samples)[:, Z2_mask]

                # Combine Z matrices
                Z = np.hstack([Z1, Z2])
                Z_colnames = list(unique_individuals) + [f"diff_{i}" for i in range(Z2.shape[1])]

                # Group labels
                group = np.array(["individual"] * Z1.shape[1] + ["difference"] * Z2.shape[1])

                if verbose:
                    print(f"Paired design: {Z1.shape[1]} individuals, {Z2.shape[1]} difference terms")

        else:
            raise ValueError(f"Unknown testtype: {testtype}")

        # -----------------------------------------
        # Build X matrix (R lines 80-84)
        # -----------------------------------------
        if testtype == "continuous":
            # Continuous: intercept + continuous feature
            if intercept:
                X = np.column_stack([np.ones(n_samples), feature_values.astype(float)])
                X_colnames = ["(Intercept)", "feature"]
            else:
                X = feature_values.astype(float).reshape(-1, 1)
                X_colnames = ["feature"]
        else:
            # Categorical: dummy coding
            feature_df = pd.DataFrame({"feature": feature_values})
            if intercept:
                # With intercept: include all levels (R's model.matrix(~., ...))
                dummies = pd.get_dummies(feature_df["feature"], drop_first=False)
                X = np.column_stack([np.ones(n_samples), dummies.values])
                X_colnames = ["(Intercept)"] + list(dummies.columns)
            else:
                # No intercept: all levels (R's model.matrix(~.-1, ...))
                dummies = pd.get_dummies(feature_df["feature"], drop_first=True)
                X = dummies.values
                X_colnames = list(dummies.columns)

    G = expr.shape[0]  # Number of genes

    if verbose:
        print(f"Expression matrix: {G} genes x {expr.shape[1]} cells")
        print(f"Design matrix X: {X.shape[0]} samples x {X.shape[1]} features")
        print(f"Random effects Z: {Z.shape[0]} samples x {Z.shape[1]} effects")
        print(f"Unique variance groups: {np.unique(group)}")

    # -----------------------------------------------------------------
    #  Compute per-sample means (R lines 93-95)
    # -----------------------------------------------------------------
    means = np.zeros((G, len(sample_names)))
    for i, s in enumerate(sample_names):
        mask = sample == s
        if mask.any():
            means[:, i] = np.mean(expr[:, mask], axis=1)

    # -----------------------------------------------------------------
    #  Gene filtering (R lines 98-103)
    # -----------------------------------------------------------------
    if filtergene:
        m = np.quantile(means, filtergenequantile)
        gene_keep = np.any(means > m, axis=1)
        expr = expr[gene_keep, :]
        means = means[gene_keep, :]
        gene_names = gene_names[gene_keep]
        G = expr.shape[0]
        if verbose:
            print(f"After gene filtering: {G} genes retained")

    # -----------------------------------------------------------------
    #  Gauss-Laguerre quadrature (R lines 107-111)
    # -----------------------------------------------------------------
    node, weight = gauss_quad_laguerre(1000)
    pos_mask = weight > 0
    node = node[pos_mask]
    weight = weight[pos_mask]
    log_node = np.log(node)
    log_weight = np.log(weight)

    # -----------------------------------------------------------------
    #  Estimate cell-level variance w (R lines 122-146)
    # -----------------------------------------------------------------
    w = np.zeros((G, len(sample_names)))

    for i, s in enumerate(sample_names):
        sampid = np.where(sample == s)[0]
        n_cells = len(sampid)

        if n_cells > 1:
            d = n_cells - 1
            # Sample variance: s2 = (mean(x^2) - mean(x)^2) * (n/(n-1))
            s2 = (np.mean(expr[:, sampid] ** 2, axis=1) - means[:, i] ** 2) * ((d + 1) / d)

            # Variance of log(s2) for positive values
            s2_pos = s2[s2 > 0]
            if len(s2_pos) > 1:
                stat = np.var(np.log(s2_pos), ddof=1) - trigamma(d / 2)

                if stat > 0:
                    theta = trigamma_inverse(stat)
                    phi = np.exp(np.mean(np.log(s2_pos)) - digamma(d / 2) + digamma(theta)) * d / 2

                    if theta + d / 2 > 1:
                        # Closed form (R line 132)
                        w[:, i] = (d * s2 / 2 + phi) / (theta + d / 2 - 1)
                    else:
                        # Numerical integration (R lines 134-139)
                        alpha = theta + d / 2
                        for g in range(G):
                            if s2[g] > 0:
                                beta = d * s2[g] / 2 + phi
                                integrand = np.exp(node - alpha * log_node - beta / node + log_weight)
                                w[g, i] = (beta**alpha / gamma(alpha)) * np.sum(integrand)
                            else:
                                w[g, i] = 0
                else:
                    # stat <= 0: use geometric mean (R line 141)
                    w[:, i] = np.exp(np.mean(np.log(s2_pos)))
            else:
                w[:, i] = np.nan
        else:
            w[:, i] = np.nan

    # -----------------------------------------------------------------
    #  Fill missing w using nearest neighbor (R lines 148-161)
    # -----------------------------------------------------------------
    nan_cols = np.where(np.all(np.isnan(w), axis=0))[0]
    ok_cols = np.array([i for i in range(w.shape[1]) if i not in nan_cols])

    if len(nan_cols) > 0 and len(ok_cols) > 0:
        # Distance matrix based on X
        X_dist = squareform(pdist(X))

        for i in nan_cols:
            if len(ok_cols) == 1:
                w[:, i] = w[:, ok_cols[0]]
            else:
                # Find nearest neighbor(s) with minimum distance
                dists = X_dist[i, ok_cols]
                min_dist = dists.min()
                nearest = ok_cols[dists == min_dist]
                w[:, i] = np.mean(w[:, nearest], axis=1)

    # -----------------------------------------------------------------
    #  Normalize w by cell count (R lines 162-165)
    # -----------------------------------------------------------------
    n_per_sample = np.array([np.sum(sample == s) for s in sample_names])
    w = w / n_per_sample

    # Clear expression matrix to save memory
    del expr

    # -----------------------------------------------------------------
    #  Initialize sigma2 matrix (R lines 271-272)
    # -----------------------------------------------------------------
    unique_groups = np.unique(group)
    sigma2 = np.zeros((G, len(unique_groups)))
    sigma2_df = pd.DataFrame(sigma2, columns=unique_groups, index=range(G))

    # -----------------------------------------------------------------
    #  sigma2 estimation function (R lines 170-268)
    # -----------------------------------------------------------------
    def sigma2_func(current_group, control_groups, done_groups):
        nonlocal failgroup

        # Xl = cbind(X, Z[, group %in% controlgroup]) (R line 171)
        ctrl_mask = np.isin(group, control_groups)
        Xl = np.hstack([X, Z[:, ctrl_mask]]) if ctrl_mask.any() else X.copy()

        # Zl = Z[, group == currentgroup] (R line 172)
        curr_mask = group == current_group
        Zl = Z[:, curr_mask]

        # lid: rows with any random effect in current or control groups (R lines 173-174)
        involved_mask = np.isin(group, [current_group] + list(control_groups))
        lid = np.where(Z[:, involved_mask].sum(axis=1) > 0)[0]

        if len(lid) == 0:
            warnings.warn(f"No data for variance of group {current_group}")
            failgroup.append(current_group)
            return np.zeros(G)

        Xl = Xl[lid, :]
        Zl = Zl[lid, :]

        # Make Xl full rank (R line 178)
        Xl = qr_rank_reduce(Xl)

        n = len(lid)
        p = n - Xl.shape[1]

        if p == 0:
            failgroup.append(current_group)
            warnings.warn(f"Unable to estimate variance for group {current_group}, setting to 0.")
            return np.zeros(G)

        # -----------------------------------------
        # Construct orthogonal K matrix (R lines 187-198)
        # -----------------------------------------
        np.random.seed(42)  # For reproducibility
        K = np.random.normal(size=(n, p))

        for i in range(p):
            if i == 0:
                b = Xl
            else:
                b = np.hstack([Xl, K[:, :i]])

            btb = b.T @ b
            btb_inv = chol2inv(btb)
            K[:, i] = K[:, i] - b @ btb_inv @ b.T @ K[:, i]

        # Normalize columns
        K = K / np.sqrt(np.sum(K**2, axis=0, keepdims=True))
        K = K.T  # Now K is (p x n)

        # -----------------------------------------
        # Compute statistics (R lines 200-211)
        # -----------------------------------------
        means_lid = means[:, lid]  # (G x n)
        pl = (K @ means_lid.T).T  # (G x p)

        qlm = K @ Zl @ Zl.T @ K.T
        ql = np.diag(qlm)  # (p,)

        w_lid = w[:, lid]  # (G x n)
        rl = w_lid @ (K**2).T  # (G x p)

        # Add contribution from already-estimated groups (R lines 206-211)
        for sg in done_groups:
            sg_mask = group == sg
            Z_sg = Z[lid][:, sg_mask]
            KZmat = K @ Z_sg @ Z_sg.T @ K.T

            for g in range(G):
                rl[g, :] += sigma2_df.loc[g, sg] * np.diag(KZmat)

        # -----------------------------------------
        # Estimate hyperparameters (R lines 213-216)
        # -----------------------------------------
        pl2 = pl**2

        M_term = (pl2 - rl) / ql
        M = np.mean(np.maximum(0, M_term))

        V_term = (pl2**2 - 3 * rl**2 - 6 * M * ql * rl) / (3 * ql**2)
        V = np.mean(np.maximum(0, V_term))

        denom = V - M**2
        if denom <= 0 or M <= 0:
            alpha_hyper = np.nan
            gamma_hyper = np.nan
        else:
            alpha_hyper = M**2 / denom
            gamma_hyper = M / denom

        if verbose:
            print(f"  alpha={alpha_hyper:.4f}, gamma={gamma_hyper:.4f}")

        # -----------------------------------------
        # Check for valid hyperparameters (R lines 219-229)
        # -----------------------------------------
        if np.isnan(alpha_hyper) or np.isnan(gamma_hyper) or alpha_hyper <= 0 or gamma_hyper <= 0:
            if verbose:
                print("  Invalid hyperparameters. Proceeding without variance pooling.")

            def process_gene_simple(g):
                def root_func(s2):
                    return np.sum(
                        (s2 * ql**2 + ql * rl[g, :] - pl[g, :] ** 2 * ql)
                        / (s2 * ql + rl[g, :]) ** 2
                    )

                try:
                    from scipy.optimize import brentq

                    return brentq(root_func, 0, 1000)
                except Exception:
                    return 0.0

            if n_jobs > 1:
                est = Parallel(n_jobs=n_jobs)(delayed(process_gene_simple)(g) for g in range(G))
            else:
                est = [process_gene_simple(g) for g in range(G)]
            return np.array(est)

        # -----------------------------------------
        # EB estimation (R lines 230-267)
        # -----------------------------------------
        tK = K.T  # (n x p)

        def process_gene(g):
            tmpx = np.outer(pl[g, :], pl[g, :])
            tmpw = w[g, lid]

            t2 = tK.T @ (tmpw[:, None] * tK)

            res = np.zeros(len(node))
            max_retries = 10
            retry = 0

            while retry < max_retries:
                try:
                    for i_node, gn in enumerate(node):
                        cm_mat = gn * qlm + t2
                        try:
                            L = np.linalg.cholesky(cm_mat)
                            log_det = 2 * np.sum(np.log(np.diag(L)))
                            cm_inv = chol2inv(cm_mat)
                        except np.linalg.LinAlgError:
                            eigvals = np.linalg.eigvalsh(cm_mat)
                            eigvals = np.maximum(eigvals, 1e-10)
                            log_det = np.sum(np.log(eigvals))
                            cm_inv = np.linalg.pinv(cm_mat)

                        res[i_node] = -log_det - np.sum(tmpx * cm_inv)
                    break
                except Exception:
                    min_idx = np.argmin(tmpw)
                    tmpw[min_idx] *= 2
                    t2 = tK.T @ (tmpw[:, None] * tK)
                    retry += 1

            res = res / 2

            tmp = log_weight + node + res + (alpha_hyper - 1) * log_node - gamma_hyper * node

            num = np.exp(tmp + log_node)
            den = np.exp(tmp)

            est_val = np.sum(num) / np.sum(den)

            if not np.isfinite(est_val):
                mv = np.max(tmp)
                est_val = np.sum(np.exp(tmp + log_node - mv)) / np.sum(np.exp(tmp - mv))

            if np.isnan(est_val):
                est_val = 1.0

            return est_val

        if n_jobs > 1:
            est = Parallel(n_jobs=n_jobs)(delayed(process_gene)(g) for g in range(G))
        else:
            est = [process_gene(g) for g in range(G)]

        return np.array(est)

    # -----------------------------------------------------------------
    #  Iterate through groups (R lines 273-283)
    # -----------------------------------------------------------------
    control_groups = list(unique_groups)
    done_groups = []

    n_para = {ug: np.sum(Z[:, group == ug] != 0) for ug in unique_groups}
    sorted_groups = sorted(unique_groups, key=lambda u: n_para[u], reverse=True)

    for ug in sorted_groups:
        if verbose:
            print(f"\n===== Estimating variance for group: {ug} =====")

        sigma2_df[ug] = sigma2_func(ug, [g for g in control_groups if g != ug], done_groups)
        control_groups.remove(ug)
        done_groups.append(ug)

    # -----------------------------------------------------------------
    #  Assemble output (R line 284)
    # -----------------------------------------------------------------
    result = {
        "mean": pd.DataFrame(means, index=gene_names, columns=sample_names),
        "sigma2": sigma2_df,
        "omega2": pd.DataFrame(w, index=gene_names, columns=sample_names),
        "X": pd.DataFrame(X, index=sample_names),
        "Z": pd.DataFrame(Z, index=sample_names),
        "group": group,
        "failgroup": failgroup,
        "sample_names": sample_names,
        "batch_corrected": batch_corrected,
    }

    if verbose:
        print("\n===== Model fitting complete =====")
        if batch_corrected:
            print("Note: ComBat batch correction was applied")
        if failgroup:
            print(f"Warning: Variance estimation failed for groups: {failgroup}")

    return result