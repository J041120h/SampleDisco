#!/usr/bin/env python3
"""
Trajectory Differential Gene Analysis (GAM) - SINGLE pseudotime only

SIMPLIFIED:
- You provide ONE pseudotime table (DataFrame or CSV/TSV path) with sample -> pseudotime.
- The script aligns it to `pseudobulk_adata.obs_names` (samples x genes).
- Fits one GAM per gene (smooth over pseudotime + optional covariates).
- BH-FDR correction + effect-size computation + optional pseudoDEG selection.
- Saves results + summary for this single pseudotime.

Pseudotime input formats supported:
1) CSV/TSV with at least two columns: [sample, pseudotime]
2) DataFrame with the same
3) DataFrame where sample IDs are the index and one column is pseudotime

Example:
pseudotime_file = "/mnt/data/X_DR_expression_pseudotime.csv"
"""

import os
import datetime
from typing import Dict, Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import anndata as ad

from scipy.sparse import issparse
from pygam import LinearGAM, s, f
from statsmodels.stats.multitest import multipletests


# =============================================================================
# PSEUDOTIME LOADING (SINGLE PATH)
# =============================================================================

def _read_pseudotime_table(obj: Union[str, pd.DataFrame, Dict]) -> pd.DataFrame:
    """
    Coerce `obj` into a pandas DataFrame.

    Supported:
      - DataFrame
      - file path (csv/tsv/txt)
      - dict with key 'pseudotime_df' or 'pseudotime_file'
    """
    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    if isinstance(obj, dict):
        if "pseudotime_df" in obj and isinstance(obj["pseudotime_df"], pd.DataFrame):
            return obj["pseudotime_df"].copy()
        if "pseudotime_file" in obj and isinstance(obj["pseudotime_file"], str):
            obj = obj["pseudotime_file"]
        else:
            raise ValueError(
                "If `pseudotime_source` is a dict, it must contain 'pseudotime_df' (DataFrame) "
                "or 'pseudotime_file' (path str)."
            )

    if isinstance(obj, str):
        if not os.path.exists(obj):
            raise FileNotFoundError(f"Pseudotime file not found: {obj}")

        ext = os.path.splitext(obj)[1].lower()
        if ext in [".tsv", ".txt"]:
            return pd.read_csv(obj, sep="\t")
        return pd.read_csv(obj)

    raise TypeError(
        f"Unsupported pseudotime_source type: {type(obj)}. "
        "Provide a DataFrame, a file path, or a dict with 'pseudotime_df'/'pseudotime_file'."
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

    Returns:
      dict: {sample_id (str): pseudotime (float)}
    """
    df = _read_pseudotime_table(pseudotime_source)
    if df.shape[0] == 0:
        raise ValueError("Pseudotime table is empty.")

    # Infer pseudotime column
    ptime_colname = _infer_col(df, candidates=[pseudotime_col, "pseudotime", "ptime", "p_time", "pt"],
                              contains=["pseudo", "ptime"])
    # Infer sample column (or fallback to index)
    sample_colname = _infer_col(df, candidates=[sample_col, "sample", "sample_id", "sampleid", "obs", "obs_names"],
                                contains=["sample"])

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
        # Use index as sample IDs only if it looks meaningful (not RangeIndex)
        if not isinstance(df.index, pd.RangeIndex):
            tmp = df.copy()
            tmp = tmp.reset_index().rename(columns={"index": "sample_id_inferred"})
            sample_colname = "sample_id_inferred"
            df = tmp
        else:
            # If 2-column table we already set sample_colname above
            if sample_colname is None:
                raise ValueError(
                    f"Could not infer sample column. Columns: {list(df.columns)}. "
                    "Expected a column like 'sample'/'sample_id', or sample IDs in the index."
                )

    tmp = df[[sample_colname, ptime_colname]].copy()
    tmp[sample_colname] = tmp[sample_colname].astype(str)
    tmp[ptime_colname] = pd.to_numeric(tmp[ptime_colname], errors="coerce")
    before = tmp.shape[0]
    tmp = tmp.dropna(subset=[ptime_colname])
    if verbose and tmp.shape[0] < before:
        print(f"Dropped {before - tmp.shape[0]} rows with invalid/missing pseudotime.")

    # Deduplicate samples (keep first)
    if tmp[sample_colname].duplicated().any():
        if verbose:
            ndup = tmp[sample_colname].duplicated().sum()
            print(f"Warning: {ndup} duplicate sample rows found in pseudotime table. Keeping first occurrence.")
        tmp = tmp.drop_duplicates(subset=[sample_colname], keep="first")

    ptime_dict = dict(zip(tmp[sample_colname], tmp[ptime_colname].astype(float)))

    # Align to pseudobulk samples (obs_names)
    pb_samples = set(pseudobulk_adata.obs_names.astype(str))
    aligned = {str(s): float(t) for s, t in ptime_dict.items() if str(s) in pb_samples}

    if verbose:
        print(f"Pseudotime loaded: {len(ptime_dict)} samples in table")
        print(f"Pseudotime aligned: {len(aligned)} samples found in pseudobulk_adata.obs_names")

    if len(aligned) == 0:
        raise ValueError(
            "No overlapping samples between pseudotime table and pseudobulk AnnData.\n"
            f"- pseudobulk first 5: {list(pseudobulk_adata.obs_names.astype(str)[:5])}\n"
            f"- pseudotime first 5: {list(list(ptime_dict.keys())[:5])}"
        )

    return aligned


# =============================================================================
# SPLINE PARAMETER UTILS (UNCHANGED)
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
        if verbose:
            print(f"Very few samples ({n_samples}): using order={spline_order}, n_splines={num_splines}")
    elif n_samples < 6:
        spline_order = 2
        num_splines = max(2, min(default_num_splines, n_samples - spline_order - 1))
        if verbose:
            print(f"Few samples ({n_samples}): using order={spline_order}, n_splines={num_splines}")
    elif n_samples < 10:
        spline_order = min(3, default_spline_order)
        max_splines = max(2, (n_samples - spline_order - 1) // min_samples_per_spline)
        num_splines = min(default_num_splines, max_splines)
        if verbose:
            print(f"Moderate samples ({n_samples}): using order={spline_order}, n_splines={num_splines}")
    else:
        spline_order = default_spline_order
        max_feasible_splines = max(2, n_samples - spline_order - 4)
        max_splines_by_density = max(2, n_samples // min_samples_per_spline)
        max_splines = min(max_feasible_splines, max_splines_by_density)
        num_splines = min(default_num_splines, max_splines)

        if verbose and (num_splines != default_num_splines or spline_order != default_spline_order):
            print(f"Adjusted splines for {n_samples} samples: order={spline_order}, n_splines={num_splines}")

    if num_splines <= spline_order:
        num_splines = spline_order + 1
        if verbose:
            print(f"Safety adjustment: increased n_splines to {num_splines} (> order={spline_order})")

    num_splines = max(2, num_splines)
    spline_order = max(1, spline_order)
    return num_splines, spline_order


# =============================================================================
# GAM INPUT PREP (UNCHANGED)
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

    if verbose:
        print(f"AnnData: {pseudobulk_adata.n_obs} samples x {pseudobulk_adata.n_vars} genes")
        print(f"Pseudotime: {len(ptime_expression)} samples")

    sample_meta = pseudobulk_adata.obs.copy()
    sample_names = pseudobulk_adata.obs_names.astype(str)

    sample_names_lower = pd.Series(sample_names.str.lower(), index=sample_names)
    ptime_expression_lower = {k.lower(): v for k, v in ptime_expression.items()}
    common_samples_lower = set(sample_names_lower.values) & set(ptime_expression_lower.keys())

    if len(common_samples_lower) == 0:
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

    if verbose:
        print(f"Common samples: {int(sample_mask.sum())}")

    filtered_adata = pseudobulk_adata[sample_mask].copy()
    filtered_sample_names = sample_names[sample_mask]
    filtered_meta = sample_meta.loc[filtered_sample_names].copy()

    if issparse(filtered_adata.X):
        if verbose:
            print(f"Converting sparse matrix ({type(filtered_adata.X).__name__}) to dense")
        expression_matrix = filtered_adata.X.toarray()
    else:
        expression_matrix = np.array(filtered_adata.X)

    if np.any(np.isnan(expression_matrix)):
        n_nan = int(np.sum(np.isnan(expression_matrix)))
        if verbose:
            print(f"Warning: Found {n_nan} NaNs in expression; replacing with 0")
        expression_matrix = np.nan_to_num(expression_matrix, nan=0.0)

    Y = pd.DataFrame(expression_matrix, index=filtered_sample_names, columns=filtered_adata.var_names)

    gene_variances = Y.var(axis=0)
    low_var = gene_variances < min_variance_threshold
    if low_var.any():
        if verbose:
            print(f"Filtering {int(low_var.sum())} genes with var < {min_variance_threshold}")
        Y = Y.loc[:, ~low_var]

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

    X = filtered_meta[["pseudotime"]].copy()

    if covariate_columns:
        valid_covs = []
        for col in covariate_columns:
            if col in filtered_meta.columns and col != "pseudotime" and not filtered_meta[col].isna().all():
                valid_covs.append(col)
        if valid_covs:
            covs = filtered_meta[valid_covs].copy()
            cat_cols = covs.select_dtypes(include=["object", "category"]).columns
            if len(cat_cols) > 0:
                covs = pd.get_dummies(covs, columns=list(cat_cols), drop_first=True)
            X = pd.concat([X, covs], axis=1)

    X.index = Y.index
    gene_names = list(Y.columns)

    if verbose:
        print(f"Prepared X: {X.shape} (features={list(X.columns)})")
        print(f"Prepared Y: {Y.shape} (genes after filter={len(gene_names)})")

    return X, Y, gene_names


# =============================================================================
# GAM FITTING (UNCHANGED FROM YOUR SNIPPET)
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
    """Fit GAM models for genes with spline parameter adjustment."""
    import pygam.utils
    from scipy.linalg import cholesky as scipy_cholesky

    def patched_cholesky(A, sparse=None, verbose_chol=False):
        if issparse(A):
            A = A.toarray()
        elif hasattr(A, "A"):
            A = A.A
        return scipy_cholesky(A, lower=True)

    original_cholesky = pygam.utils.cholesky
    pygam.utils.cholesky = patched_cholesky

    try:
        def _to_dense_2d(mat) -> np.ndarray:
            if isinstance(mat, pd.DataFrame):
                if hasattr(mat, "sparse"):
                    try:
                        mat = mat.sparse.to_dense()
                    except Exception:
                        pass
                mat = mat.to_numpy()
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

        X_dense = _to_dense_2d(X)
        Y_dense = _to_dense_2d(Y)

        finite_rows = np.isfinite(X_dense).all(axis=1)

        try:
            spline_idx = list(X.columns).index(spline_term)
        except ValueError:
            raise ValueError(f"spline_term '{spline_term}' not found in X.columns: {list(X.columns)}")

        terms = s(spline_idx, n_splines=adj_n_splines, spline_order=adj_order)
        for j in range(X_dense.shape[1]):
            if j != spline_idx:
                terms += f(j)

        gene_to_col = {g: i for i, g in enumerate(Y.columns)} if isinstance(Y, pd.DataFrame) else {g: i for i, g in enumerate(gene_names)}

        results = []
        gam_models: Dict[str, LinearGAM] = {}
        total = len(gene_names)
        good = 0

        for k, gene in enumerate(gene_names):
            if verbose and (k + 1) % 100 == 0:
                print(f"Processing gene {k + 1}/{total}")

            col_idx = gene_to_col.get(gene, None)
            if col_idx is None or col_idx >= Y_dense.shape[1]:
                continue

            y_raw = Y_dense[:, col_idx]
            mask = finite_rows & np.isfinite(y_raw)

            X_fit = X_dense[mask]
            y_fit = y_raw[mask]

            min_needed = adj_order + adj_n_splines + 2
            if y_fit.size < min_needed:
                continue

            if np.var(y_fit) < 1e-10:
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
                continue

        if verbose:
            print(f"Fitted GAMs for {good}/{total} genes (splines n={adj_n_splines}, order={adj_order})")

        if not results:
            cols = ["gene", "pval", "dev_exp", "fdr", "significant"]
            return pd.DataFrame(columns=cols), {}

        res_df = pd.DataFrame(results, columns=["gene", "pval", "dev_exp"])
        try:
            _, fdrs, _, _ = multipletests(res_df["pval"], method="fdr_bh")
            res_df["fdr"] = fdrs
            res_df["significant"] = res_df["fdr"] < fdr_threshold
        except Exception:
            res_df["fdr"] = res_df["pval"]
            res_df["significant"] = res_df["fdr"] < fdr_threshold

        return res_df.sort_values("fdr").reset_index(drop=True), gam_models

    finally:
        pygam.utils.cholesky = original_cholesky


# =============================================================================
# EFFECT SIZE + DEG SELECTION (UNCHANGED)
# =============================================================================

def calculate_effect_size(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    gam_models: Dict[str, LinearGAM],
    genes: List[str],
    verbose: bool = False
) -> pd.DataFrame:
    effect_sizes = []
    for gene in genes:
        if gene not in gam_models or gene not in Y.columns:
            continue
        try:
            gam = gam_models[gene]
            X_values = X.values if isinstance(X, pd.DataFrame) else X
            y_pred = gam.predict(X_values)
            y_true = Y[gene].values

            residuals = y_true - y_pred
            df_e = max(1, len(y_true) - gam.statistics_["edof"])
            rss = np.sum(residuals**2)

            if rss > 0 and df_e > 0:
                es = (np.max(y_pred) - np.min(y_pred)) / np.sqrt(rss / df_e)
                effect_sizes.append((gene, es))
        except Exception:
            continue

    if verbose:
        print(f"Calculated effect sizes for {len(effect_sizes)} genes")

    return pd.DataFrame(effect_sizes, columns=["gene", "effect_size"])


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
            print(f"Selected {int(results['pseudoDEG'].sum())} pseudoDEGs (top_n={top_n_genes})")

    else:
        if "effect_size" in results.columns:
            results["pseudoDEG"] = (results["fdr"] < fdr_threshold) & (results["effect_size"] > effect_size_threshold)
        else:
            results["pseudoDEG"] = results["fdr"] < fdr_threshold

        if verbose:
            print(f"Selected {int(results['pseudoDEG'].sum())} pseudoDEGs (ES>{effect_size_threshold})")

    return results


# =============================================================================
# OUTPUT (UNCHANGED)
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

    results_df.to_csv(os.path.join(output_dir, f"gam_all_genes_{timestamp}.tsv"), sep="\t", index=False)

    if "fdr" in results_df.columns:
        results_df[results_df["fdr"] < fdr_threshold].to_csv(
            os.path.join(output_dir, f"gam_significant_{timestamp}.tsv"),
            sep="\t",
            index=False
        )

    if "pseudoDEG" in results_df.columns:
        results_df[results_df["pseudoDEG"]].to_csv(
            os.path.join(output_dir, f"gam_pseudoDEGs_{timestamp}.tsv"),
            sep="\t",
            index=False
        )

    summary_file = os.path.join(output_dir, f"gam_summary_{timestamp}.txt")
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
            f.write(f"Significant genes (FDR<{fdr_threshold}): {int((results_df['fdr'] < fdr_threshold).sum())}\n")
        if "pseudoDEG" in results_df.columns:
            f.write(f"Selected pseudoDEGs: {int(results_df['pseudoDEG'].sum())}\n")

    if verbose:
        print(f"Saved results to: {output_dir}")


def summarize_results(
    results: pd.DataFrame,
    top_n: int = 20,
    output_file: Optional[str] = None,
    verbose: bool = True
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
        lines.append(f"Significant genes (FDR < 0.05): {int((results['fdr'] < 0.05).sum())}")
    if "pseudoDEG" in results.columns:
        n_deg = int(results["pseudoDEG"].sum())
        lines.append(f"Selected DEGs: {n_deg}")
        if n_deg > 0:
            lines.append(f"\nTop {min(top_n, n_deg)} DEGs:")
            top = results[results["pseudoDEG"]].nsmallest(min(top_n, n_deg), "fdr")
            for i, (_, row) in enumerate(top.iterrows(), 1):
                if "effect_size" in row and pd.notna(row["effect_size"]):
                    lines.append(f"{i}. {row['gene']}: FDR={row['fdr']:.4e}, Effect={row['effect_size']:.3f}")
                else:
                    lines.append(f"{i}. {row['gene']}: FDR={row['fdr']:.4e}")

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
            print("DEG_visualization module not available; skipping.")
        return

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
                    verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"Visualization failed for {gene}: {e}")


# =============================================================================
# SINGLE-TRAJECTORY RUNNER
# =============================================================================

def run_trajectory_gam_differential_gene_analysis(
    pseudobulk_adata: ad.AnnData,
    pseudotime_source: Union[str, pd.DataFrame, Dict],
    *,
    sample_col: str = "sample",
    pseudotime_col: str = "pseudotime",
    covariate_columns: Optional[List[str]] = None,
    fdr_threshold: float = 0.05,
    effect_size_threshold: float = 1.0,
    top_n_genes: int = 100,
    num_splines: int = 5,
    spline_order: int = 3,
    output_dir: str = "trajectory_diff_gene_results_single",
    visualization_gene_list: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run trajectory differential gene analysis for ONE pseudotime vector.
    """
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("Loading and aligning pseudotime...")

    ptime_dict = load_sample_pseudotime(
        pseudobulk_adata=pseudobulk_adata,
        pseudotime_source=pseudotime_source,
        sample_col=sample_col,
        pseudotime_col=pseudotime_col,
        verbose=verbose
    )

    if verbose:
        print("Preparing GAM input matrices...")

    X, Y, gene_names = prepare_gam_input_data_improved(
        pseudobulk_adata=pseudobulk_adata,
        ptime_expression=ptime_dict,
        covariate_columns=covariate_columns,
        sample_col=sample_col,
        verbose=verbose
    )

    if verbose:
        print(f"Fitting GAM models for {len(gene_names)} genes...")

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
            print("Warning: No genes were successfully analyzed.")
        empty = pd.DataFrame()
        return empty

    sig_genes = stat_results[stat_results["fdr"] < fdr_threshold]["gene"].tolist()

    if verbose:
        print(f"Calculating effect sizes for {len(sig_genes)} significant genes...")

    effect_sizes = calculate_effect_size(
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

    save_results(
        results_df=results,
        output_dir=output_dir,
        fdr_threshold=fdr_threshold,
        effect_size_threshold=effect_size_threshold,
        top_n_genes=top_n_genes,
        verbose=verbose
    )

    summarize_results(
        results=results,
        top_n=min(20, len(results)),
        output_file=os.path.join(output_dir, "differential_gene_result.txt"),
        verbose=verbose
    )

    if visualization_gene_list and len(gam_models) > 0:
        generate_gene_visualizations(
            gene_list=visualization_gene_list,
            X=X,
            Y=Y,
            gam_models=gam_models,
            results=results,
            output_dir=output_dir,
            verbose=verbose
        )

    if verbose:
        print(f"Done. Results saved to: {output_dir}")

    return results