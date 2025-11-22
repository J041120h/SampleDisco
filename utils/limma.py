import pandas as pd
import patsy
import numpy as np
import sys

def limma(
    pheno: pd.DataFrame,
    exprs: np.ndarray,
    covariate_formula: str,
    design_formula: str = '1',
    rcond: float = 1e-8,
    verbose: bool = False
) -> np.ndarray:
    """
    Limma-like linear regression to remove batch (or other covariates).

    Parameters
    ----------
    pheno : DataFrame
        Sample-level metadata; rows must align with exprs rows.
    exprs : ndarray
        Expression matrix with shape (n_samples, n_genes).
    covariate_formula : str
        Patsy formula for covariates to regress out (e.g. '~ batch').
        We will remove the *non-intercept* part of these covariates.
    design_formula : str, optional
        Kept for API compatibility but not used in the new implementation.
    rcond : float
        rcond passed to np.linalg.lstsq.
    verbose : bool
        If True, prints brief diagnostics.

    Returns
    -------
    corrected : ndarray
        Batch/covariate-corrected expression matrix with same shape as exprs.
    """
    exprs = np.asarray(exprs)
    if exprs.ndim != 2:
        raise ValueError(f"exprs must be 2D (n_samples, n_genes); got shape {exprs.shape}")

    n_samples, n_genes = exprs.shape

    if verbose:
        print(f"[limma] exprs shape: {exprs.shape}, pheno shape: {pheno.shape}")
        print(f"[limma] covariate formula: {covariate_formula}")

    # Build covariate design matrix (includes intercept + dummy variables)
    covariate_design = patsy.dmatrix(covariate_formula, pheno, return_type='dataframe')
    X = np.asarray(covariate_design, dtype=float)  # (n_samples, p)

    if X.shape[0] != n_samples:
        raise ValueError(
            f"Rows in design ({X.shape[0]}) != samples in exprs ({n_samples})"
        )

    if verbose:
        print(f"[limma] design shape: {X.shape}")

    # Solve least squares: X (n_samples x p) -> exprs (n_samples x n_genes)
    coefficients, res, rank, s = np.linalg.lstsq(X, exprs, rcond=rcond)

    if verbose:
        print(f"[limma] coeff shape: {coefficients.shape}, rank: {rank}")

    # Identify non-intercept columns to remove (e.g. batch dummies)
    col_names = list(covariate_design.columns)
    batch_mask = [not name.startswith('Intercept') for name in col_names]

    if not any(batch_mask):
        if verbose:
            print("[limma] No non-intercept columns; returning exprs unchanged.")
        return exprs.copy()

    X_batch = X[:, batch_mask]               # (n_samples, p_batch)
    beta_batch = coefficients[batch_mask, :] # (p_batch, n_genes)

    correction = X_batch.dot(beta_batch)     # (n_samples, n_genes)
    corrected = exprs - correction

    if verbose:
        print(f"[limma] corrected shape: {corrected.shape}")

    return corrected