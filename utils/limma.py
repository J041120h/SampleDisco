import numpy as np
import pandas as pd
import patsy


# =============================================================================
# Limma-like correction (matches YOUR definition)
#
# pheno: metadata
# exprs: (n_samples, n_genes)  [AnnData.X convention]
# covariate_formula: variance to KEEP (preserve)
# design_formula: variance to REMOVE (batch)
# =============================================================================
def limma(
    pheno: pd.DataFrame,
    exprs: np.ndarray,                 # (n_samples, n_genes)
    covariate_formula: str = "1",      # KEEP (e.g. "~ condition + timepoint")
    design_formula: str | None = None, # REMOVE (e.g. "~ batch")
    rcond: float = 1e-8,
    verbose: bool = False
) -> np.ndarray:
    Y = np.asarray(exprs, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"exprs must be 2D (n_samples, n_genes); got shape {Y.shape}")
    n_samples, _ = Y.shape
    if len(pheno) != n_samples:
        raise ValueError(f"pheno rows ({len(pheno)}) != exprs samples ({n_samples})")

    def _ensure_formula(s: str) -> str:
        s = s.strip()
        if s in ("0", "1"):
            return s
        return s if s.startswith("~") else "~ " + s

    keep_f = _ensure_formula(covariate_formula) if covariate_formula else "1"
    Z_df = patsy.dmatrix(keep_f, pheno, return_type="dataframe")  # KEEP
    Z = np.asarray(Z_df, dtype=float)

    X = None
    if design_formula and design_formula.strip():
        rem_f = _ensure_formula(design_formula)
        X_df = patsy.dmatrix(rem_f, pheno, return_type="dataframe")
        rem_cols = [c for c in X_df.columns if "Intercept" not in c]
        if rem_cols:
            X = np.asarray(X_df[rem_cols], dtype=float)  # REMOVE (no intercept)

    if verbose:
        print(f"[limma] Y={Y.shape}")
        print(f"[limma] KEEP   {keep_f} -> Z={Z.shape}")
        print(f"[limma] REMOVE {design_formula} -> X={None if X is None else X.shape}")

    if X is None or X.shape[1] == 0:
        return Y.copy()

    # Fit full model: Y ≈ [Z, X] @ B
    W = np.hstack([Z, X])
    B, *_ = np.linalg.lstsq(W, Y, rcond=rcond)

    p_keep = Z.shape[1]
    Bx = B[p_keep:, :]       # coefficients for REMOVE block
    correction = X @ Bx      # nuisance contribution
    return Y - correction
