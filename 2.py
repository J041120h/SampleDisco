#!/usr/bin/env python3
"""
check_normalization.py

Load an .h5ad file, print a small sample of .X expression values, and
heuristically report whether the matrix looks like raw counts, normalized,
or log-transformed.

Usage:
    python check_normalization.py /path/to/data.h5ad --cells 5 --genes 5
"""

import argparse
import warnings
import numpy as np
import anndata as ad
import scanpy as sc
from scipy.sparse import issparse, csr_matrix, csc_matrix

def _safe_row_sum(X, axis=1):
    """Sum along rows or columns without densifying."""
    if issparse(X):
        return np.array(X.sum(axis=axis)).ravel()
    return X.sum(axis=axis)

def _sample_indices(n, k):
    k = min(k, n)
    if k <= 0:
        return np.array([], dtype=int)
    if k == n:
        return np.arange(n, dtype=int)
    return np.random.default_rng(42).choice(n, size=k, replace=False)

def _get_sample_block(X, row_idx, col_idx):
    """Return a small dense block of X[row_idx, col_idx] without densifying all of X."""
    if issparse(X):
        # Convert to CSR for efficient row slicing
        if not isinstance(X, (csr_matrix, csc_matrix)):
            X = X.tocsr()
        sub = X[row_idx][:, col_idx]
        return sub.toarray()
    return np.asarray(X[np.ix_(row_idx, col_idx)])

def _fraction_integers(arr, tol=1e-8):
    arr = arr.ravel()
    if arr.size == 0:
        return np.nan
    # Ignore NaNs
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    return np.mean(np.abs(arr - np.round(arr)) <= tol)

def _heuristic_normalization_report(adata, X_sample=None, libsizes=None):
    """
    Heuristically classify X as:
      - 'raw counts' (mostly integers, larger library sizes, wide variance)
      - 'normalized (CPM/TPM/size-factor)' (per-cell sums relatively uniform)
      - 'log-transformed' (values small, non-integer, presence of uns['log1p'])
    Returns a dict of clues and a one-line guess.
    """
    clues = {}

    # Clues from AnnData structure
    clues["has_raw"] = adata.raw is not None
    clues["has_counts_layer"] = "counts" in (adata.layers.keys() if adata.layers is not None else {})
    clues["has_log1p_uns"] = isinstance(adata.uns.get("log1p", None), dict)

    # Numeric clues from sample
    if X_sample is not None and X_sample.size > 0:
        clues["min"] = float(np.nanmin(X_sample))
        clues["max"] = float(np.nanmax(X_sample))
        clues["mean"] = float(np.nanmean(X_sample))
        clues["frac_integers"] = float(_fraction_integers(X_sample))
        clues["nonnegative"] = bool(np.nanmin(X_sample) >= -1e-8)  # allow tiny numerical noise
    else:
        clues["min"] = clues["max"] = clues["mean"] = np.nan
        clues["frac_integers"] = np.nan
        clues["nonnegative"] = True

    # Library size uniformity
    guess = "inconclusive"
    if libsizes is not None and libsizes.size > 1:
        lib_mean = np.mean(libsizes)
        lib_std = np.std(libsizes)
        lib_cv = lib_std / (lib_mean + 1e-12)
        clues["libsize_mean"] = float(lib_mean)
        clues["libsize_std"] = float(lib_std)
        clues["libsize_cv"] = float(lib_cv)
    else:
        clues["libsize_mean"] = clues["libsize_std"] = clues["libsize_cv"] = np.nan

    # Heuristic rules
    # 1) Strong hint for log-transformed: uns['log1p'] present or values small (< ~20) and clearly non-integer
    if clues["has_log1p_uns"] or (
        (not np.isnan(clues["frac_integers"]) and clues["frac_integers"] < 0.2)
        and clues["nonnegative"] and clues["max"] < 50
    ):
        guess = "likely LOG-transformed (e.g., log1p of normalized counts)"

    # 2) Hint for CPM/TPM/size-factor normalized: library sizes fairly uniform (small CV)
    #    Thresholds are heuristic; CV < 0.15 is often quite uniform.
    if guess == "inconclusive" and not np.isnan(clues["libsize_cv"]) and clues["libsize_cv"] < 0.15:
        guess = "likely library-normalized (e.g., CPM/TPM/size-factor)"

    # 3) Hint for raw counts: mostly integers & wide library size spread
    if guess == "inconclusive" and not np.isnan(clues["frac_integers"]) and clues["frac_integers"] > 0.95:
        if np.isnan(clues["libsize_cv"]) or clues["libsize_cv"] >= 0.15:
            guess = "likely RAW counts"

    # 4) If counts layer exists but X looks log/normalized, mention that
    if clues["has_counts_layer"] and "likely" in guess:
        guess += " (and a 'counts' layer is available)"

    return clues, guess

def main():
    parser = argparse.ArgumentParser(description="Inspect .h5ad .X and infer normalization status.")
    parser.add_argument("h5ad_path", type=str, help="Path to .h5ad file")
    parser.add_argument("--cells", type=int, default=5, help="Number of example cells to print")
    parser.add_argument("--genes", type=int, default=5, help="Number of example genes to print")
    parser.add_argument("--show-libsizes", action="store_true", help="Also print a small sample of per-cell library sizes")
    args = parser.parse_args()

    # Load AnnData
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.read_h5ad(args.h5ad_path)

    n_cells, n_genes = adata.n_obs, adata.n_vars
    X = adata.X

    print("="*80)
    print(f"File: {args.h5ad_path}")
    print(f"Shape: cells={n_cells:,}, genes={n_genes:,}")
    print(f"Sparse: {issparse(X)}")
    print(f"dtype : {getattr(X, 'dtype', type(X))}")
    print(f"Has .raw: {adata.raw is not None}")
    print(f"Layers: {list(adata.layers.keys()) if adata.layers is not None else []}")
    print(f"uns has 'log1p': {'log1p' in adata.uns}")
    print("="*80)

    # Sample a small block for display
    row_idx = _sample_indices(n_cells, args.cells)
    col_idx = _sample_indices(n_genes, args.genes)
    X_block = _get_sample_block(X, row_idx, col_idx) if row_idx.size and col_idx.size else np.array([[]])

    # Print row/col indices used
    print("Sampled cells (row indices):", row_idx.tolist())
    print("Sampled genes (col indices):", col_idx.tolist())

    # Print the small matrix
    if X_block.size:
        # Format a compact view
        with np.printoptions(precision=3, suppress=True, linewidth=140):
            print("\nExample .X values (rows=cells, cols=genes):")
            print(X_block)
    else:
        print("\n(No sample to display)")

    # Compute per-cell library sizes from X
    # NOTE: For log-transformed data, this sum is not a count total; we still use it as a diagnostic for uniformity.
    libsizes = _safe_row_sum(X, axis=1) if n_cells > 0 else np.array([])

    # Heuristic classification
    # Use a larger sample for statistics but still keep it cheap
    stat_rows = _sample_indices(n_cells, min(1000, max(50, n_cells//50)))
    stat_cols = _sample_indices(n_genes, min(1000, max(50, n_genes//50)))
    X_stat = _get_sample_block(X, stat_rows, stat_cols) if stat_rows.size and stat_cols.size else None

    clues, guess = _heuristic_normalization_report(adata, X_stat, libsizes)

    print("\n--- Normalization Heuristics ---")
    for k in ["has_raw", "has_counts_layer", "has_log1p_uns", "min", "max", "mean",
              "frac_integers", "nonnegative", "libsize_mean", "libsize_std", "libsize_cv"]:
        print(f"{k:>16}: {clues[k]}")
    print(f"\nNormalization guess: {guess}")

    # Optionally show a small sample of library sizes
    if args.show_libsizes and libsizes.size:
        ls_idx = _sample_indices(libsizes.size, min(10, libsizes.size))
        print("\nSample per-cell library sizes (sums over columns of .X):")
        with np.printoptions(precision=2, suppress=True, linewidth=140):
            for i, ridx in enumerate(ls_idx):
                print(f"  cell_idx={ridx:>7}: {libsizes[ridx]}")

    # Extra guidance
    print("\nTips:")
    print("  • If you see mostly integers with wide library-size variation ⇒ likely raw counts.")
    print("  • If per-cell sums are very similar (small CV) ⇒ likely library-normalized (CPM/TPM/size-factor).")
    print("  • If values are small/non-integers and 'uns[\"log1p\"]' exists ⇒ likely log1p-normalized.")
    print("  • If you need raw counts, check adata.layers['counts'] or adata.raw (if set).")
    print("="*80)

if __name__ == "__main__":
    main()
