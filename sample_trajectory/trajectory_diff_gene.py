#!/usr/bin/env python3
"""
Trajectory Differential Gene Analysis (GAM) — SINGLE pseudotime only.

Builds the per-sample pseudobulk **on the fly** from the cell-level
`adata_preprocessed` so we don't need a separate sample-level adata in the
pipeline. The recipe mirrors the original `compute_pseudobulk_adata`:

  1. Aggregate cells per (sample × celltype) by mean (no double normalization —
     `.X` is already normalized + log1p from preprocessing).
  2. Per cell type: optional Limma batch correction; optional first-round HVG
     selection for noise reduction.
  3. Concatenate per-celltype HVGs → `samples × (celltype-gene)` matrix.

The existing GAM stack operates on that samples × features matrix unchanged.
"""

import os
import datetime
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from scipy.sparse import csr_matrix, issparse
from statsmodels.stats.multitest import multipletests
from pygam import LinearGAM, s, f


# =============================================================================
# Pseudobulk-on-the-fly helper
# =============================================================================

def _to_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _attempt_limma(temp_adata, batch_col, preserve_cols, verbose=False):
    """Apply Limma correction in-place on `temp_adata.X`.

    On failure, emits a WARNING (unconditional, not gated by verbose) so that
    batch-confounded results are not silently produced, and returns False.
    """
    try:
        from utils.limma import limma
        X = temp_adata.X.toarray() if issparse(temp_adata.X) else np.asarray(temp_adata.X)
        if preserve_cols:
            quoted = [f'Q("{c}")' for c in preserve_cols]
            covariate_formula = "~ " + " + ".join(quoted)
        else:
            covariate_formula = "1"
        temp_adata.X = limma(
            pheno=temp_adata.obs, exprs=X,
            covariate_formula=covariate_formula,
            design_formula=f'~ Q("{batch_col}")',
            rcond=1e-8, verbose=False,
        )
        return True
    except Exception as exc:
        warnings.warn(
            f"[trajectory_diff_gene] Limma batch correction (batch_col={batch_col!r}) "
            f"FAILED: {type(exc).__name__}: {exc}. Pseudobulk will be returned WITHOUT "
            f"batch correction — downstream results may be batch-confounded.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False


def _build_sample_pseudobulk(
    adata: ad.AnnData,
    sample_col: str,
    celltype_col: Optional[str] = "cell_type",
    batch_col: Optional[Union[str, List[str]]] = None,
    n_features_per_celltype: Optional[int] = 2000,
    columns_to_preserve: Optional[Union[str, List[str]]] = None,
    verbose: bool = False,
) -> ad.AnnData:
    """Aggregate cell-level `adata` into a `samples × (celltype-gene)` AnnData.

    Mirrors the recipe used by the legacy ``compute_pseudobulk_adata`` (per-
    cell-type aggregation + optional Limma + optional first-round HVG), but
    omits the second normalization pass (``adata.X`` is already normalized /
    log-transformed by preprocessing) and the second HVG round.
    """
    if sample_col not in adata.obs.columns:
        raise KeyError(f"sample_col '{sample_col}' not in adata.obs")

    preserve_cols = [c for c in _to_list(columns_to_preserve)
                     if c in adata.obs.columns]
    if isinstance(batch_col, list):
        batch_cols = [c for c in batch_col if c and c in adata.obs.columns]
        if not batch_cols:
            batch_col_to_use = None
        elif len(batch_cols) == 1:
            batch_col_to_use = batch_cols[0]
        else:
            # Multiple batch cols: combine into a synthetic interaction column.
            adata.obs["__pb_batch__"] = adata.obs[batch_cols].astype(str).agg("|".join, axis=1)
            batch_col_to_use = "__pb_batch__"
    else:
        batch_col_to_use = batch_col if (batch_col and batch_col in adata.obs.columns) else None

    samples_sorted = sorted(adata.obs[sample_col].astype(str).unique())
    sample_idx = {s: i for i, s in enumerate(samples_sorted)}
    n_samples = len(samples_sorted)
    n_genes = adata.n_vars

    if celltype_col is None or celltype_col not in adata.obs.columns:
        # No celltype split — simple per-sample mean of all cells.
        sample_arr = adata.obs[sample_col].astype(str).values
        cell_to_sample = np.asarray([sample_idx[s] for s in sample_arr])
        valid = cell_to_sample >= 0
        if not valid.all():
            sample_arr = sample_arr[valid]
            cell_to_sample = cell_to_sample[valid]
        n_cells = len(cell_to_sample)
        indicator = csr_matrix(
            (np.ones(n_cells, dtype=np.float32),
             (cell_to_sample, np.arange(n_cells))),
            shape=(n_samples, n_cells),
        )
        cells_per_sample = np.array(indicator.sum(axis=1)).flatten()
        cells_per_sample[cells_per_sample == 0] = 1
        X = adata.X.tocsr() if issparse(adata.X) else adata.X
        summed = indicator @ (X[valid] if not valid.all() else X)
        if issparse(summed):
            summed = np.asarray(summed.todense())
        mean_X = (summed / cells_per_sample[:, None]).astype(np.float32)
        pb_obs = pd.DataFrame(index=pd.Index(samples_sorted, name=sample_col))
        sample_to_meta = adata.obs.drop_duplicates(subset=[sample_col]).set_index(
            adata.obs[sample_col].drop_duplicates().values)
        sample_to_meta = adata.obs.groupby(sample_col, observed=True).first()
        pb_obs = pb_obs.join(sample_to_meta, how="left")
        out = ad.AnnData(X=mean_X, obs=pb_obs, var=adata.var.copy())
        return out

    cell_types_sorted = sorted(adata.obs[celltype_col].astype(str).unique())
    if verbose:
        print(f"[pseudobulk] {len(samples_sorted)} samples × {len(cell_types_sorted)} cell types")
    sample_arr = adata.obs[sample_col].astype(str).values
    celltype_arr = adata.obs[celltype_col].astype(str).values

    pb_obs = pd.DataFrame(index=pd.Index(samples_sorted, name=sample_col))
    sample_to_meta = adata.obs.groupby(sample_col, observed=True).first()
    pb_obs = pb_obs.join(sample_to_meta, how="left")

    X_full = adata.X.tocsr() if issparse(adata.X) else adata.X

    feature_blocks = []
    feature_names = []
    for celltype in cell_types_sorted:
        ct_mask = celltype_arr == celltype
        if ct_mask.sum() == 0:
            continue
        ct_samples = sample_arr[ct_mask]
        ct_sample_idx = np.array([sample_idx[s] for s in ct_samples])
        n_cells_ct = len(ct_sample_idx)
        indicator = csr_matrix(
            (np.ones(n_cells_ct, dtype=np.float32),
             (ct_sample_idx, np.arange(n_cells_ct))),
            shape=(n_samples, n_cells_ct),
        )
        cells_per = np.array(indicator.sum(axis=1)).flatten()
        cells_per_safe = cells_per.copy()
        cells_per_safe[cells_per_safe == 0] = 1
        X_ct = X_full[ct_mask]
        summed = indicator @ X_ct
        if issparse(summed):
            summed = np.asarray(summed.todense())
        ct_pb = (summed / cells_per_safe[:, None]).astype(np.float32)

        temp = sc.AnnData(
            X=ct_pb.copy(),
            obs=pb_obs.copy(),
            var=adata.var.copy(),
        )
        gene_keep = np.asarray((ct_pb != 0).any(axis=0)).flatten()
        if gene_keep.sum() == 0:
            continue
        temp = temp[:, gene_keep].copy()

        if batch_col_to_use is not None and batch_col_to_use in temp.obs.columns:
            nb = temp.obs[batch_col_to_use].nunique(dropna=True)
            if nb >= 2:
                _attempt_limma(temp, batch_col_to_use, preserve_cols, verbose=verbose)

        if issparse(temp.X):
            mask = np.isnan(temp.X.data) | np.isinf(temp.X.data)
            if mask.any():
                temp.X.data[mask] = 0.0
                temp.X.eliminate_zeros()
        else:
            np.nan_to_num(temp.X, copy=False)

        if n_features_per_celltype is not None and 0 < n_features_per_celltype < temp.n_vars:
            try:
                sc.pp.highly_variable_genes(
                    temp, n_top_genes=n_features_per_celltype, subset=False,
                )
                hvg_mask = temp.var["highly_variable"].values
                temp = temp[:, hvg_mask].copy()
            except Exception as exc:
                if verbose:
                    print(f"    HVG selection failed for celltype '{celltype}': {exc}")

        if temp.n_vars == 0:
            continue
        block = temp.X.toarray() if issparse(temp.X) else np.asarray(temp.X)
        feature_blocks.append(block.astype(np.float32))
        feature_names.extend(f"{celltype} - {g}" for g in temp.var_names)

    if not feature_blocks:
        raise RuntimeError("No features survived pseudobulk aggregation")

    concat_X = np.concatenate(feature_blocks, axis=1)
    var = pd.DataFrame(index=pd.Index(feature_names, name="feature"))
    out = ad.AnnData(X=concat_X, obs=pb_obs.copy(), var=var)
    if verbose:
        print(f"[pseudobulk] result shape: {out.shape}")
    return out

def _read_pseudotime_table(obj: Union[str, pd.DataFrame, Dict]) -> pd.DataFrame:
    """Coerce a pseudotime source (DataFrame, dict, or file path) into a DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj.copy()

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

    ptime_colname = _infer_col(
        df,
        candidates=[pseudotime_col, "pseudotime", "ptime", "p_time", "pt"],
        contains=["pseudo", "ptime"]
    )
    sample_colname = _infer_col(
        df,
        candidates=[sample_col, "sample", "sample_id", "sampleid", "obs", "obs_names"],
        contains=["sample"]
    )

    if ptime_colname is None:
        if df.shape[1] == 2:
            sample_colname = df.columns[0]
            ptime_colname = df.columns[1]
        else:
            raise ValueError(
                f"Could not infer pseudotime column. Columns: {list(df.columns)}. "
                "Expected a column like 'pseudotime'/'ptime', or a 2-column table."
            )

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

    if tmp[sample_colname].duplicated().any():
        ndup = tmp[sample_colname].duplicated().sum()
        if verbose:
            print(f"  → Found {ndup} duplicate samples; keeping first occurrence")
        tmp = tmp.drop_duplicates(subset=[sample_colname], keep="first")

    ptime_dict = dict(zip(tmp[sample_colname], tmp[ptime_colname].astype(float)))

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
    Fit per-gene LinearGAM models with adaptive spline parameters.

    Per-gene masking (Option B): each gene is fitted only on the samples where
    that gene is non-zero, so structural zeros don't bias the spline toward zero.
    """
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

    if verbose:
        print(f"  → Spline parameters: n_splines={adj_n_splines}, order={adj_order}")

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

        # Per-gene mask: finite X + finite y + gene non-zero (Option B per-gene masking)
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

        min_needed = adj_order + adj_n_splines + 2  # minimum identifiable samples for this spline
        if y_fit.size < min_needed:
            skip_counts["too_few_samples"] += 1
            continue

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

def calculate_effect_size_and_direction(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    gam_models: Dict[str, LinearGAM],
    genes: List[str],
    verbose: bool = False
) -> pd.DataFrame:
    """
    Compute effect size and regulation direction for each gene's GAM fit.

    Effect size: (max_pred - min_pred) / residual_std — the fitted dynamic range
    relative to noise. Direction: "UP" if corr(pseudotime, y_pred) > 0, else "DOWN".
    Both are computed on the non-zero subset used for fitting (Option B masking).
    """
    effect_sizes = []
    error_count = 0

    X_values = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    finite_X_rows = np.isfinite(X_values).all(axis=1)

    try:
        pt_idx = list(X.columns).index("pseudotime")
    except ValueError:
        pt_idx = 0

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

            residuals = y_true - y_pred
            df_e = max(1, len(y_true) - gam.statistics_["edof"])
            rss = np.sum(residuals ** 2)

            if rss > 0 and df_e > 0:
                es = (np.max(y_pred) - np.min(y_pred)) / np.sqrt(rss / df_e)

                pt_values = X_used[:, pt_idx]
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

def run_trajectory_gam_differential_gene_analysis(
    adata: ad.AnnData,
    pseudotime_source: Union[str, pd.DataFrame, Dict],
    *,
    sample_col: str = "sample",
    celltype_col: Optional[str] = "cell_type",
    batch_col: Optional[Union[str, List[str]]] = None,
    n_features_per_celltype: Optional[int] = 2000,
    columns_to_preserve: Optional[Union[str, List[str]]] = None,
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
    # anchor_col: numeric obs column used to orient pseudotime so that
    # "UP" always means increasing with the biological axis (e.g. sev.level).
    # When provided, pseudotime is flipped if corr(pseudotime, anchor) < 0.
    anchor_col: Optional[str] = None,
    verbose: bool = True,
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

    if verbose:
        print("\n[0/5] Building per-sample pseudobulk from cell-level adata...")
    pseudobulk_adata = _build_sample_pseudobulk(
        adata,
        sample_col=sample_col,
        celltype_col=celltype_col,
        batch_col=batch_col,
        n_features_per_celltype=n_features_per_celltype,
        columns_to_preserve=columns_to_preserve,
        verbose=verbose,
    )

    if verbose:
        print("\n[1/5] Loading pseudotime...")
    ptime_dict = load_sample_pseudotime(
        pseudobulk_adata=pseudobulk_adata,
        pseudotime_source=pseudotime_source,
        sample_col=sample_col,
        pseudotime_col=pseudotime_col,
        verbose=verbose
    )

    # Flip pseudotime if negatively correlated with anchor_col so that UP/DOWN
    # labels always reflect increasing biological gradient, independent of which
    # MST endpoint TSCAN chose as origin.
    if anchor_col is not None and anchor_col in pseudobulk_adata.obs.columns:
        anchor = pseudobulk_adata.obs[anchor_col]
        shared = [s for s in ptime_dict if s in anchor.index]
        if shared:
            pt_vals = np.array([ptime_dict[s] for s in shared], dtype=float)
            anc_vals = anchor.loc[shared].values.astype(float)
            valid = np.isfinite(pt_vals) & np.isfinite(anc_vals)
            if valid.sum() >= 2:
                corr = np.corrcoef(pt_vals[valid], anc_vals[valid])[0, 1]
                if np.isfinite(corr) and corr < 0:
                    max_pt = max(ptime_dict.values())
                    ptime_dict = {s: max_pt - v for s, v in ptime_dict.items()}
                    if verbose:
                        print(f"  → Flipped pseudotime to align with anchor '{anchor_col}' "
                              f"(corr before flip: {corr:.3f})")
    elif anchor_col is not None:
        import warnings
        warnings.warn(
            f"anchor_col '{anchor_col}' not found in pseudobulk_adata.obs; "
            "pseudotime orientation is not anchored.",
            RuntimeWarning,
            stacklevel=2,
        )

    if verbose:
        print("\n[2/5] Preparing GAM input matrices...")
    X, Y, gene_names = prepare_gam_input_data_improved(
        pseudobulk_adata=pseudobulk_adata,
        ptime_expression=ptime_dict,
        covariate_columns=covariate_columns,
        sample_col=sample_col,
        verbose=verbose
    )

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

    save_results(
        results_df=results,
        output_dir=output_dir,
        fdr_threshold=fdr_threshold,
        effect_size_threshold=effect_size_threshold,
        top_n_genes=top_n_genes,
        verbose=verbose
    )

    if verbose:
        print("\n" + "="*70)
    summarize_results(
        results=results,
        top_n=min(20, len(results)),
        output_file=os.path.join(output_dir, "differential_gene_result.txt"),
        verbose=verbose,
        fdr_threshold=fdr_threshold
    )

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

    if generate_visualizations:
        if verbose:
            print("\n[5/5] Generating Lamian-style visualizations...")
        try:
            from .trajectory_DGE_visualization import generate_all_visualizations
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