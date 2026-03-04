#!/usr/bin/env python3
"""
Trajectory Differential Gene Analysis (GAM) - SINGLE pseudotime only [DENSE FIX]

SIMPLIFIED:
- You provide ONE pseudotime table (DataFrame or CSV/TSV path) with sample -> pseudotime.
- The script aligns it to `pseudobulk_adata.obs_names` (samples x genes).
- Fits one GAM per gene (smooth over pseudotime + optional covariates).
- BH-FDR correction + effect-size computation + optional pseudoDEG selection.
- Saves results + summary for this single pseudotime.
- Generates comprehensive Lamian-style visualizations.

FIX STRATEGY:
- Pre-emptively converts sparse gene matrices to dense NumPy arrays.
- This avoids 'csr_matrix' compatibility issues in pygam without needing patches.
"""

import os
import datetime
from typing import Dict, Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import anndata as ad

from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
from pygam import LinearGAM, s, f

# =============================================================================
# PSEUDOTIME LOADING (SINGLE PATH)
# =============================================================================

def _read_pseudotime_table(obj: Union[str, pd.DataFrame, Dict]) -> pd.DataFrame:
    """
    Coerce `obj` into a pandas DataFrame.
    """
    # Direct DataFrame
    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    # Dict-like input
    if isinstance(obj, dict):
        if "pseudotime_df" in obj and isinstance(obj["pseudotime_df"], pd.DataFrame):
            return obj["pseudotime_df"].copy()

        if "pseudotime_file" in obj and isinstance(obj["pseudotime_file"], str):
            obj = obj["pseudotime_file"]
        else:
            if len(obj) > 0 and all(np.isscalar(v) for v in obj.values()):
                return pd.DataFrame(
                    {
                        "sample": list(obj.keys()),
                        "pseudotime": list(obj.values()),
                    }
                )
            raise ValueError(
                "If `pseudotime_source` is a dict, it must either:\n"
                "  - contain 'pseudotime_df' (DataFrame), or\n"
                "  - contain 'pseudotime_file' (path str), or\n"
                "  - be a simple mapping {sample_id: pseudotime_value}."
            )

    # String → treat as path
    if isinstance(obj, str):
        if not os.path.exists(obj):
            raise FileNotFoundError(f"Pseudotime file not found: {obj}")

        ext = os.path.splitext(obj)[1].lower()
        if ext in [".tsv", ".txt"]:
            return pd.read_csv(obj, sep="\t")
        return pd.read_csv(obj)

    raise TypeError(
        f"Unsupported pseudotime_source type: {type(obj)}. "
        "Provide a DataFrame, a file path, a dict with 'pseudotime_df'/'pseudotime_file', "
        "or a simple mapping {sample_id: pseudotime}."
    )


def _infer_col(df: pd.DataFrame, candidates: List[str], contains: Optional[List[str]] = None) -> Optional[str]:
    """Find a column in df by exact match (case-insensitive), else by substring match."""
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    if contains:
        for c in cols:
            cl = c.lower()
            if any(sub in cl for sub in contains):
                return c

    return None


def load_sample_pseudotime(
    pseudobulk_adata: ad.AnnData,
    pseudotime_source: Union[str, pd.DataFrame, Dict],
    sample_col: str = "sample",
    pseudotime_col: str = "pseudotime",
    verbose: bool = False
) -> Dict[str, float]:
    """
    Load a SINGLE sample->pseudotime mapping from a pseudotime table, then align to pseudobulk samples.
    """
    df = _read_pseudotime_table(pseudotime_source)

    if df.shape[0] == 0:
        raise ValueError("Pseudotime table is empty.")

    # Infer pseudotime column
    ptime_colname = _infer_col(
        df,
        candidates=[pseudotime_col, "pseudotime", "ptime", "p_time", "pt"],
        contains=["pseudo", "ptime"]
    )
    # Infer sample column (or fallback to index)
    sample_colname = _infer_col(
        df,
        candidates=[sample_col, "sample", "sample_id", "sampleid", "obs", "obs_names"],
        contains=["sample"]
    )

    # If no pseudotime col but exactly 2 cols, assume col1=sample, col2=pseudotime
    if ptime_colname is None:
        if df.shape[1] == 2:
            sample_colname = df.columns[0]
            ptime_colname = df.columns[1]
        else:
            raise ValueError(
                f"Could not infer pseudotime column. Columns: {list(df.columns)}. "
                "Expected a column like 'pseudotime'/'ptime', or a 2-column table."
            )

    # If no sample col, try using index
    if sample_colname is None:
        if not isinstance(df.index, pd.RangeIndex):
            tmp = df.copy()
            tmp = tmp.reset_index().rename(columns={"index": "sample_id_inferred"})
            sample_colname = "sample_id_inferred"
            df = tmp
        else:
            raise ValueError(
                f"Could not infer sample column. Columns: {list(df.columns)}. "
                "Expected a column like 'sample'/'sample_id', or sample IDs in the index."
            )

    tmp = df[[sample_colname, ptime_colname]].copy()
    tmp[sample_colname] = tmp[sample_colname].astype(str)
    tmp[ptime_colname] = pd.to_numeric(tmp[ptime_colname], errors="coerce")

    before = tmp.shape[0]
    tmp = tmp.dropna(subset=[ptime_colname])
    if verbose:
        dropped = before - tmp.shape[0]
        if dropped > 0:
            print(f"  → Dropped {dropped} rows with invalid/missing pseudotime")

    # Deduplicate samples (keep first)
    if tmp[sample_colname].duplicated().any():
        ndup = tmp[sample_colname].duplicated().sum()
        if verbose:
            print(f"  → Found {ndup} duplicate samples; keeping first occurrence")
        tmp = tmp.drop_duplicates(subset=[sample_colname], keep="first")

    ptime_dict = dict(zip(tmp[sample_colname], tmp[ptime_colname].astype(float)))

    # Align to pseudobulk samples (obs_names)
    pb_samples = set(pseudobulk_adata.obs_names.astype(str))
    aligned = {str(s): float(t) for s, t in ptime_dict.items() if str(s) in pb_samples}

    if len(aligned) == 0:
        raise ValueError(
            "No overlapping samples between pseudotime table and pseudobulk AnnData.\n"
            f"- pseudobulk first 5: {list(pseudobulk_adata.obs_names.astype(str)[:5])}\n"
            f"- pseudotime first 5: {list(list(ptime_dict.keys())[:5])}"
        )

    if verbose:
        print(f"  → Loaded {len(ptime_dict)} pseudotime values; aligned {len(aligned)} to pseudobulk samples")

    return aligned


# =============================================================================
# SPLINE PARAMETER UTILS
# =============================================================================

def calculate_optimal_spline_parameters(
    n_samples: int,
    default_num_splines: int = 5,
    default_spline_order: int = 3,
    min_samples_per_spline: int = 2,
    verbose: bool = False
) -> Tuple[int, int]:
    """Calculate spline parameters based on sample size."""
    min_samples_needed = default_spline_order + 1

    if n_samples < min_samples_needed:
        spline_order = 1
        num_splines = min(2, max(1, n_samples - 2))
    elif n_samples < 6:
        spline_order = 2
        num_splines = max(2, min(default_num_splines, n_samples - spline_order - 1))
    elif n_samples < 10:
        spline_order = min(3, default_spline_order)
        max_splines = max(2, (n_samples - spline_order - 1) // min_samples_per_spline)
        num_splines = min(default_num_splines, max_splines)
    else:
        spline_order = default_spline_order
        max_feasible_splines = max(2, n_samples - spline_order - 4)
        max_splines_by_density = max(2, n_samples // min_samples_per_spline)
        max_splines = min(max_feasible_splines, max_splines_by_density)
        num_splines = min(default_num_splines, max_splines)

    if num_splines <= spline_order:
        num_splines = spline_order + 1

    num_splines = max(2, num_splines)
    spline_order = max(1, spline_order)
    
    return num_splines, spline_order


# =============================================================================
# GAM INPUT PREP
# =============================================================================

def prepare_gam_input_data_improved(
    pseudobulk_adata: ad.AnnData,
    ptime_expression: Dict[str, float],
    covariate_columns: Optional[List[str]] = None,
    sample_col: str = "sample",
    min_variance_threshold: float = 1e-6,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Prepare X (design) and Y (expression) for GAM fitting."""
    if ptime_expression is None or not ptime_expression:
        raise ValueError("Pseudotime values must be provided.")

    sample_meta = pseudobulk_adata.obs.copy()
    sample_names = pseudobulk_adata.obs_names.astype(str)

    sample_names_lower = pd.Series(sample_names.str.lower(), index=sample_names)
    ptime_expression_lower = {k.lower(): v for k, v in ptime_expression.items()}
    common_samples_lower = set(sample_names_lower.values) & set(ptime_expression_lower.keys())

    if len(common_samples_lower) == 0:
        # Try exact (case-sensitive) intersection as fallback
        common_samples = set(sample_names) & set(ptime_expression.keys())
        if len(common_samples) > 0:
            sample_mask = sample_names.isin(common_samples)
        else:
            raise ValueError(
                f"No common samples found.\n"
                f"AnnData first 5: {list(sample_names[:5])}\n"
                f"Pseudotime first 5: {list(list(ptime_expression.keys())[:5])}"
            )
    else:
        sample_mask = sample_names_lower.isin(common_samples_lower)

    filtered_adata = pseudobulk_adata[sample_mask].copy()
    filtered_sample_names = sample_names[sample_mask]
    filtered_meta = sample_meta.loc[filtered_sample_names].copy()

    # -------------------------------------------------------------------------
    # CRITICAL FIX: FORCE DENSE NUMPY ARRAY
    # -------------------------------------------------------------------------
    if issparse(filtered_adata.X):
        if verbose:
            print(f"  → Converting sparse expression matrix to dense")
        expression_matrix = filtered_adata.X.toarray()
    else:
        expression_matrix = np.asarray(filtered_adata.X)
    
    # Ensure it is a standard numpy array (not np.matrix or other subclass)
    expression_matrix = np.array(expression_matrix, copy=False)

    if np.any(np.isnan(expression_matrix)):
        n_nan = int(np.sum(np.isnan(expression_matrix)))
        if verbose:
            print(f"  → Replacing {n_nan} NaN values with 0")
        expression_matrix = np.nan_to_num(expression_matrix, nan=0.0)

    Y = pd.DataFrame(expression_matrix, index=filtered_sample_names, columns=filtered_adata.var_names)

    # Filter low-variance genes
    gene_variances = Y.var(axis=0)
    low_var = gene_variances < min_variance_threshold
    if low_var.any():
        if verbose:
            print(f"  → Filtering {int(low_var.sum())} low-variance genes (var < {min_variance_threshold})")
        Y = Y.loc[:, ~low_var]

    # Align pseudotime vector
    filtered_meta["pseudotime"] = np.nan
    for sname in filtered_sample_names:
        if sname in ptime_expression:
            filtered_meta.loc[sname, "pseudotime"] = ptime_expression[sname]
        else:
            s_lower = sname.lower()
            if s_lower in ptime_expression_lower:
                filtered_meta.loc[sname, "pseudotime"] = ptime_expression_lower[s_lower]

    if filtered_meta["pseudotime"].isna().any():
        missing = int(filtered_meta["pseudotime"].isna().sum())
        raise ValueError(f"Failed to assign pseudotime for {missing} samples")

    # Build X
    X = filtered_meta[["pseudotime"]].copy()

    # Add covariates if provided
    if covariate_columns:
        valid_covs = []
        for col in covariate_columns:
            if col in filtered_meta.columns and col != "pseudotime" and not filtered_meta[col].isna().all():
                valid_covs.append(col)

        if valid_covs:
            covs = filtered_meta[valid_covs].copy()
            cat_cols = covs.select_dtypes(include=["object", "category"]).columns
            if len(cat_cols) > 0:
                if verbose:
                    print(f"  → One-hot encoding categorical covariates: {list(cat_cols)}")
                covs = pd.get_dummies(covs, columns=list(cat_cols), drop_first=True)
            X = pd.concat([X, covs], axis=1)

    X.index = Y.index
    gene_names = list(Y.columns)

    if verbose:
        print(f"  → Prepared design matrix: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"  → Expression matrix: {Y.shape[0]} samples × {Y.shape[1]} genes")

    return X, Y, gene_names


# =============================================================================
# GAM FITTING
# =============================================================================
def fit_gam_models_for_genes(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    gene_names: List[str],
    *,
    spline_term: str = "pseudotime",
    num_splines: int = 5,
    spline_order: int = 3,
    fdr_threshold: float = 0.05,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, LinearGAM]]:
    """
    Fit GAM models for genes with spline parameter adjustment.

    IMPORTANT (Option B):
    For each gene, we only use samples where that gene is present
    (Y[gene] != 0) for fitting and testing.
    """
    # helper for ensuring everything is dense numpy array
    def _to_dense_2d(mat) -> np.ndarray:
        if isinstance(mat, pd.DataFrame):
            # If sparse dataframe
            if hasattr(mat, "sparse"):
                try:
                    mat = mat.sparse.to_dense()
                except Exception:
                    pass
            mat = mat.to_numpy()

        # Double check for sparse matrix type
        if issparse(mat):
            mat = mat.toarray()

        return np.asarray(mat, dtype=np.float64, order="C")

    n_samples = X.shape[0]
    adj_n_splines, adj_order = calculate_optimal_spline_parameters(
        n_samples=n_samples,
        default_num_splines=num_splines,
        default_spline_order=spline_order,
        verbose=verbose
    )

    if verbose:
        print(f"  → Spline parameters: n_splines={adj_n_splines}, order={adj_order}")

    X_dense = _to_dense_2d(X)
    Y_dense = _to_dense_2d(Y)

    # finite rows in X (same for all genes)
    finite_rows = np.isfinite(X_dense).all(axis=1)

    try:
        spline_idx = list(X.columns).index(spline_term)
    except ValueError:
        raise ValueError(f"spline_term '{spline_term}' not found in X.columns: {list(X.columns)}")

    terms = s(spline_idx, n_splines=adj_n_splines, spline_order=adj_order)
    for j in range(X_dense.shape[1]):
        if j != spline_idx:
            terms += f(j)

    gene_to_col = (
        {g: i for i, g in enumerate(Y.columns)}
        if isinstance(Y, pd.DataFrame)
        else {g: i for i, g in enumerate(gene_names)}
    )

    results = []
    gam_models: Dict[str, LinearGAM] = {}
    total = len(gene_names)
    good = 0

    # Skip counters
    skip_counts = {
        "no_column": 0,
        "no_nonzero_samples": 0,
        "too_few_samples": 0,
        "low_variance": 0,
        "fit_error": 0,
        "nonfinite_rows": 0,
    }

    for k, gene in enumerate(gene_names):
        if verbose and (k + 1) % 500 == 0:
            print(f"  → Processing gene {k + 1}/{total}...")

        col_idx = gene_to_col.get(gene, None)
        if col_idx is None or col_idx >= Y_dense.shape[1]:
            skip_counts["no_column"] += 1
            continue

        y_raw = Y_dense[:, col_idx]

        # Mask: finite in X, finite in y, AND gene present (non-zero)
        base_mask = finite_rows & np.isfinite(y_raw)
        nonzero_mask = y_raw != 0.0
        mask = base_mask & nonzero_mask

        if mask.sum() == 0:
            skip_counts["no_nonzero_samples"] += 1
            continue

        if base_mask.sum() == 0:
            skip_counts["nonfinite_rows"] += 1
            continue

        X_fit = X_dense[mask]
        y_fit = y_raw[mask]

        # require enough samples relative to spline parameters
        min_needed = adj_order + adj_n_splines + 2
        if y_fit.size < min_needed:
            skip_counts["too_few_samples"] += 1
            continue

        # variance check on the non-zero subset
        if np.var(y_fit) < 1e-10:
            skip_counts["low_variance"] += 1
            continue

        try:
            gam = LinearGAM(terms).fit(X_fit, y_fit)
            pval = gam.statistics_["p_values"][spline_idx]
            dev = gam.statistics_["pseudo_r2"]["explained_deviance"]
            if np.isfinite(pval) and np.isfinite(dev):
                results.append((gene, pval, dev))
                gam_models[gene] = gam
                good += 1
        except Exception:
            skip_counts["fit_error"] += 1
            continue

    if verbose:
        print(f"  → Successfully fitted {good}/{total} genes")
        total_skipped = sum(skip_counts.values())
        if total_skipped > 0:
            print(f"  → Skipped {total_skipped} genes:")
            for reason, count in skip_counts.items():
                if count > 0:
                    print(f"      {reason}: {count}")

    if not results:
        cols = ["gene", "pval", "dev_exp", "fdr", "significant"]
        return pd.DataFrame(columns=cols), {}

    res_df = pd.DataFrame(results, columns=["gene", "pval", "dev_exp"])
    try:
        _, fdrs, _, _ = multipletests(res_df["pval"], method="fdr_bh")
        res_df["fdr"] = fdrs
        res_df["significant"] = res_df["fdr"] < fdr_threshold
    except Exception:
        if verbose:
            print(f"  → FDR correction failed; using raw p-values")
        res_df["fdr"] = res_df["pval"]
        res_df["significant"] = res_df["fdr"] < fdr_threshold

    return res_df.sort_values("fdr").reset_index(drop=True), gam_models


# =============================================================================
# EFFECT SIZE + REGULATION DIRECTION
# =============================================================================
def calculate_effect_size_and_direction(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    gam_models: Dict[str, LinearGAM],
    genes: List[str],
    verbose: bool = False
) -> pd.DataFrame:
    """
    Calculate effect sizes AND regulation direction for genes.

    Direction is determined by correlation between pseudotime and GAM predictions.
    - Positive correlation → "UP" (upregulated along trajectory)
    - Negative correlation → "DOWN" (downregulated along trajectory)
    """
    effect_sizes = []
    error_count = 0

    # precompute dense X and finite mask once
    X_values = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    finite_X_rows = np.isfinite(X_values).all(axis=1)
    
    # Get pseudotime column index
    try:
        pt_idx = list(X.columns).index("pseudotime")
    except ValueError:
        pt_idx = 0  # Fallback to first column if "pseudotime" not found

    for gene in genes:
        if gene not in gam_models or gene not in Y.columns:
            continue
        try:
            gam = gam_models[gene]

            y_full = Y[gene].values
            finite_y = np.isfinite(y_full)
            nonzero = y_full != 0.0

            mask = finite_X_rows & finite_y & nonzero
            if mask.sum() == 0:
                continue

            X_used = X_values[mask]
            y_true = y_full[mask]

            y_pred = gam.predict(X_used)

            # Calculate effect size
            residuals = y_true - y_pred
            df_e = max(1, len(y_true) - gam.statistics_["edof"])
            rss = np.sum(residuals ** 2)

            if rss > 0 and df_e > 0:
                es = (np.max(y_pred) - np.min(y_pred)) / np.sqrt(rss / df_e)
                
                # Calculate direction via correlation
                pt_values = X_used[:, pt_idx]  # pseudotime values
                correlation = np.corrcoef(pt_values, y_pred)[0, 1]
                direction = "UP" if correlation > 0 else "DOWN"
                
                effect_sizes.append((gene, es, direction))
        except Exception as e:
            error_count += 1
            continue

    if verbose:
        print(f"  → Calculated effect sizes for {len(effect_sizes)} genes")
        if error_count > 0:
            print(f"  → Failed for {error_count} genes")

    return pd.DataFrame(effect_sizes, columns=["gene", "effect_size", "regulation"])


# =============================================================================
# DEG SELECTION
# =============================================================================
def determine_pseudoDEGs(
    results: pd.DataFrame,
    fdr_threshold: float,
    effect_size_threshold: float,
    top_n_genes: Optional[int],
    verbose: bool
) -> pd.DataFrame:
    if len(results) == 0:
        results["pseudoDEG"] = False
        return results

    if top_n_genes is not None:
        sig = results[results["fdr"] < fdr_threshold].copy()
        if len(sig) == 0:
            results["pseudoDEG"] = False
            if verbose:
                print("  → No genes below FDR threshold; pseudoDEG set to False for all")
            return results

        if "effect_size" in sig.columns and not sig["effect_size"].isna().all():
            if len(sig) > top_n_genes:
                top = sig.nlargest(top_n_genes, "effect_size")
                results["pseudoDEG"] = results["gene"].isin(top["gene"])
            else:
                results["pseudoDEG"] = results["fdr"] < fdr_threshold
        else:
            results["pseudoDEG"] = results["fdr"] < fdr_threshold

        if verbose:
            print(f"  → Selected {int(results['pseudoDEG'].sum())} pseudoDEGs (top_n={top_n_genes})")

    else:
        if "effect_size" in results.columns:
            results["pseudoDEG"] = (results["fdr"] < fdr_threshold) & (results["effect_size"] > effect_size_threshold)
        else:
            results["pseudoDEG"] = results["fdr"] < fdr_threshold

        if verbose:
            print(f"  → Selected {int(results['pseudoDEG'].sum())} pseudoDEGs (effect_size > {effect_size_threshold})")

    return results


# =============================================================================
# OUTPUT
# =============================================================================

def save_results(
    results_df: pd.DataFrame,
    output_dir: str,
    fdr_threshold: float,
    effect_size_threshold: float,
    top_n_genes: Optional[int] = None,
    verbose: bool = False
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    all_path = os.path.join(output_dir, f"gam_all_genes_{timestamp}.tsv")
    sig_path = os.path.join(output_dir, f"gam_significant_{timestamp}.tsv")
    deg_path = os.path.join(output_dir, f"gam_pseudoDEGs_{timestamp}.tsv")
    summary_file = os.path.join(output_dir, f"gam_summary_{timestamp}.txt")

    results_df.to_csv(all_path, sep="\t", index=False)

    if "fdr" in results_df.columns:
        results_df[results_df["fdr"] < fdr_threshold].to_csv(sig_path, sep="\t", index=False)

    if "pseudoDEG" in results_df.columns:
        results_df[results_df["pseudoDEG"]].to_csv(deg_path, sep="\t", index=False)

    with open(summary_file, "w") as f:
        f.write("===== GAM ANALYSIS SUMMARY =====\n\n")
        f.write(f"Analysis date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"FDR threshold: {fdr_threshold}\n")
        if top_n_genes is not None:
            f.write(f"Selection: Top {top_n_genes} genes by effect size\n")
        else:
            f.write(f"Effect size threshold: {effect_size_threshold}\n")
        f.write(f"\nTotal genes analyzed: {len(results_df)}\n")
        if "fdr" in results_df.columns:
            f.write(f"Significant genes (FDR<{fdr_threshold}): "
                    f"{int((results_df['fdr'] < fdr_threshold).sum())}\n")
        if "pseudoDEG" in results_df.columns:
            f.write(f"Selected pseudoDEGs: {int(results_df['pseudoDEG'].sum())}\n")
            if "regulation" in results_df.columns:
                deg_df = results_df[results_df["pseudoDEG"]]
                n_up = int((deg_df["regulation"] == "UP").sum())
                n_down = int((deg_df["regulation"] == "DOWN").sum())
                f.write(f"  - Upregulated: {n_up}\n")
                f.write(f"  - Downregulated: {n_down}\n")

    if verbose:
        print(f"  → Saved results to: {output_dir}")


def summarize_results(
    results: pd.DataFrame,
    top_n: int = 20,
    output_file: Optional[str] = None,
    verbose: bool = True,
    fdr_threshold: float = 0.05,
) -> None:
    if len(results) == 0:
        msg = "No genes were successfully analyzed."
        if verbose:
            print(msg)
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                f.write(msg)
        return

    lines = ["=== DIFFERENTIAL GENE EXPRESSION SUMMARY ==="]
    lines.append(f"Total genes analyzed: {len(results)}")
    
    if "fdr" in results.columns:
        n_sig = int((results['fdr'] < fdr_threshold).sum())
        lines.append(f"Significant genes (FDR < {fdr_threshold}): {n_sig}")

    if "pseudoDEG" in results.columns:
        n_deg = int(results["pseudoDEG"].sum())
        lines.append(f"Selected DEGs: {n_deg}")
        
        if "regulation" in results.columns:
            deg_df = results[results["pseudoDEG"]]
            n_up = int((deg_df["regulation"] == "UP").sum())
            n_down = int((deg_df["regulation"] == "DOWN").sum())
            lines.append(f"  - Upregulated: {n_up}")
            lines.append(f"  - Downregulated: {n_down}")
        
        if n_deg > 0:
            lines.append(f"\nTop {min(top_n, n_deg)} DEGs:")
            
            # Filter for pseudoDEGs
            deg_df = results[results["pseudoDEG"]].copy()
            
            # Sort by Effect Size (Descending) if available, otherwise by FDR (Ascending)
            if "effect_size" in deg_df.columns:
                top = deg_df.nlargest(min(top_n, n_deg), "effect_size")
                sort_criteria = "Effect Size"
            else:
                top = deg_df.nsmallest(min(top_n, n_deg), "fdr")
                sort_criteria = "FDR"
                
            lines.append(f"(Sorted by {sort_criteria})")

            for i, (_, row) in enumerate(top.iterrows(), 1):
                parts = [f"{i}. {row['gene']}: FDR={row['fdr']:.4e}"]
                if "effect_size" in row and pd.notna(row["effect_size"]):
                    parts.append(f"Effect={row['effect_size']:.3f}")
                if "regulation" in row and pd.notna(row["regulation"]):
                    parts.append(f"Direction={row['regulation']}")
                lines.append(", ".join(parts))

    out = "\n".join(lines)
    if verbose:
        print(out)
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(out)


def generate_gene_visualizations(
    gene_list: List[str],
    X: pd.DataFrame,
    Y: pd.DataFrame,
    gam_models: Dict[str, LinearGAM],
    results: pd.DataFrame,
    output_dir: str,
    verbose: bool = False
):
    viz_dir = os.path.join(output_dir, "gene_visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    try:
        from visualization.DEG_visualization import visualize_gene_expression
    except ImportError:
        if verbose:
            print("  → DEG_visualization module not available; skipping gene-level plots")
        return

    error_count = 0
    for gene in gene_list:
        if gene in gam_models and gene in Y.columns:
            try:
                visualize_gene_expression(
                    gene=gene,
                    X=X,
                    Y=Y,
                    gam_model=gam_models[gene],
                    stats_df=results,
                    output_dir=output_dir,
                    gene_subfolder="gene_visualizations",
                    verbose=False
                )
            except Exception:
                error_count += 1
    
    if verbose and error_count > 0:
        print(f"  → Visualization failed for {error_count} genes")


# =============================================================================
# SINGLE-TRAJECTORY RUNNER WITH INTEGRATED VISUALIZATION
# =============================================================================
def run_trajectory_gam_differential_gene_analysis(
    pseudobulk_adata: ad.AnnData,
    pseudotime_source: Union[str, pd.DataFrame, Dict],
    *,
    sample_col: str = "sample",
    pseudotime_col: str = "pseudotime",
    covariate_columns: Optional[List[str]] = None,
    fdr_threshold: float = 0.01,
    effect_size_threshold: float = 1.0,
    top_n_genes: int = 100,
    num_splines: int = 5,
    spline_order: int = 3,
    output_dir: str = "trajectory_diff_gene_results_single",
    visualization_gene_list: Optional[List[str]] = None,
    # New visualization parameters
    generate_visualizations: bool = True,
    group_col: Optional[str] = None,
    n_clusters: int = 3,
    top_n_genes_for_curves: int = 20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run trajectory differential gene analysis for ONE pseudotime vector.

    `pseudotime_source` can be:
      - path to CSV/TSV
      - DataFrame
      - dict with 'pseudotime_df'/'pseudotime_file'
      - dict mapping {sample_id: pseudotime}
    
    Parameters
    ----------
    pseudobulk_adata : AnnData
        Pseudobulk expression data (samples x genes)
    pseudotime_source : str, DataFrame, or dict
        Pseudotime information for samples
    sample_col : str
        Column name for sample IDs
    pseudotime_col : str
        Column name for pseudotime values
    covariate_columns : list, optional
        Additional covariates for GAM model
    fdr_threshold : float
        FDR threshold for significance
    effect_size_threshold : float
        Effect size threshold for pseudoDEG selection
    top_n_genes : int
        Number of top genes to select as pseudoDEGs
    num_splines : int
        Number of splines for GAM
    spline_order : int
        Order of spline basis
    output_dir : str
        Output directory
    visualization_gene_list : list, optional
        Specific genes to generate individual visualizations for
    generate_visualizations : bool
        Whether to generate Lamian-style visualization plots (default: True)
    group_col : str, optional
        Column for group comparison in XDE visualizations (e.g., 'condition', 'severity')
    n_clusters : int
        Number of clusters for heatmap visualization (default: 5)
    top_n_genes_for_curves : int
        Number of genes to show in expression curve plots (default: 20)
    verbose : bool
        Print progress messages
        
    Returns
    -------
    results : pd.DataFrame
        DataFrame with analysis results for all genes
    """
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("="*70)
        print("TRAJECTORY GAM DIFFERENTIAL GENE ANALYSIS")
        print("="*70)

    # 1. Load pseudotime
    if verbose:
        print("\n[1/5] Loading pseudotime...")
    ptime_dict = load_sample_pseudotime(
        pseudobulk_adata=pseudobulk_adata,
        pseudotime_source=pseudotime_source,
        sample_col=sample_col,
        pseudotime_col=pseudotime_col,
        verbose=verbose
    )

    # 2. Prepare GAM inputs
    if verbose:
        print("\n[2/5] Preparing GAM input matrices...")
    X, Y, gene_names = prepare_gam_input_data_improved(
        pseudobulk_adata=pseudobulk_adata,
        ptime_expression=ptime_dict,
        covariate_columns=covariate_columns,
        sample_col=sample_col,
        verbose=verbose
    )

    # 3. Fit GAMs
    if verbose:
        print(f"\n[3/5] Fitting GAM models for {len(gene_names)} genes...")
    stat_results, gam_models = fit_gam_models_for_genes(
        X=X,
        Y=Y,
        gene_names=gene_names,
        spline_term="pseudotime",
        num_splines=num_splines,
        spline_order=spline_order,
        fdr_threshold=fdr_threshold,
        verbose=verbose
    )

    if len(stat_results) == 0:
        if verbose:
            print("\n⚠ Warning: No genes were successfully analyzed")
        return pd.DataFrame()

    sig_genes = stat_results[stat_results["fdr"] < fdr_threshold]["gene"].tolist()
    if verbose:
        print(f"  → Found {len(sig_genes)} significant genes (FDR < {fdr_threshold})")

    # 4. Effect size & regulation direction
    if verbose:
        print("\n[4/5] Calculating effect sizes and regulation directions...")
    effect_sizes = calculate_effect_size_and_direction(
        X=X, Y=Y, gam_models=gam_models, genes=sig_genes, verbose=verbose
    )

    results = stat_results.merge(effect_sizes, on="gene", how="left")

    results = determine_pseudoDEGs(
        results=results,
        fdr_threshold=fdr_threshold,
        effect_size_threshold=effect_size_threshold,
        top_n_genes=top_n_genes,
        verbose=verbose
    )

    # Save to disk
    save_results(
        results_df=results,
        output_dir=output_dir,
        fdr_threshold=fdr_threshold,
        effect_size_threshold=effect_size_threshold,
        top_n_genes=top_n_genes,
        verbose=verbose
    )

    # Text summary
    if verbose:
        print("\n" + "="*70)
    summarize_results(
        results=results,
        top_n=min(20, len(results)),
        output_file=os.path.join(output_dir, "differential_gene_result.txt"),
        verbose=verbose,
        fdr_threshold=fdr_threshold
    )

    # Optional gene visualizations (legacy)
    if visualization_gene_list and len(gam_models) > 0:
        if verbose:
            print("\nGenerating gene-level visualizations...")
        generate_gene_visualizations(
            gene_list=visualization_gene_list,
            X=X,
            Y=Y,
            gam_models=gam_models,
            results=results,
            output_dir=output_dir,
            verbose=verbose
        )

    # ==========================================================================
    # Step 5: Generate Lamian-style visualizations
    # ==========================================================================
    if generate_visualizations:
        if verbose:
            print("\n[5/5] Generating Lamian-style visualizations...")
        try:
            from trajectory_DGE_visualization import generate_all_visualizations
            generate_all_visualizations(
                X=X,
                Y=Y,
                results=results,
                gam_models=gam_models,
                pseudobulk_adata=pseudobulk_adata,
                output_dir=output_dir,
                group_col=group_col,
                n_clusters=n_clusters,
                top_n_genes_for_curves=top_n_genes_for_curves,
                fdr_threshold=fdr_threshold,
                verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"  → Visualization generation failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        if verbose:
            print("\n[5/5] Skipping visualizations (generate_visualizations=False)")

    if verbose:
        print("\n" + "="*70)
        print(f"✓ Analysis complete. Results saved to: {output_dir}")
        print("="*70)

    return results