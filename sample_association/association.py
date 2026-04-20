"""
Dimension association analysis — per-PC variance-explained decomposition.

For every sample-embedding component (expression / proportion PCs) and every
sample-level metadata column in ``pseudo_adata.obs``, estimate how much of
the component's variance that variable explains.

The test is unified for continuous and categorical variables:

    PC_k  ~  design(variable)        (OLS)
    R²    =  1 - SSR / SST           ∈ [0, 1]
    p     =  permutation p-value     (shuffle variable, refit)

Design for continuous variables is [1, x]; for categorical variables it is
[1, one-hot(levels, drop-first)]. R² is therefore directly comparable across
types and across components: for continuous it is the squared Pearson
correlation; for categorical it is η² (one-way ANOVA effect size).

Outputs (per embedding):
    <output_dir>/variance_explained_<embedding>.csv
    <output_dir>/figures/<embedding>_variance_heatmap.pdf
    <output_dir>/figures/<embedding>_top_associations.pdf

All figures are saved as PDF only.

Public entrypoint: ``run_dimension_association_analysis()``.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anndata import AnnData
from typing import Optional, List, Tuple, Dict, Sequence
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Journal-style plotting defaults
# =============================================================================

_JOURNAL_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _sig_stars(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


# =============================================================================
# Embedding helpers
# =============================================================================

def _get_dr_matrix(pseudo_adata: AnnData, embedding_key: str) -> Tuple[np.ndarray, List[str]]:
    """Return (ndarray, component_names) for a DR embedding key."""
    if embedding_key in pseudo_adata.uns and isinstance(pseudo_adata.uns[embedding_key], pd.DataFrame):
        df = pseudo_adata.uns[embedding_key]
        return df.values, list(df.columns)
    if embedding_key in pseudo_adata.obsm:
        X = np.asarray(pseudo_adata.obsm[embedding_key])
        tag = embedding_key.replace("X_DR_", "")
        return X, [f"{tag}_{i+1}" for i in range(X.shape[1])]
    raise KeyError(f"Embedding '{embedding_key}' not found in pseudo_adata.uns or obsm")


def _available_embeddings(pseudo_adata: AnnData) -> List[str]:
    """Return DR embeddings present (expression, proportion)."""
    out = []
    for key in ("X_DR_expression", "X_DR_proportion"):
        try:
            _get_dr_matrix(pseudo_adata, key)
            out.append(key)
        except KeyError:
            continue
    return out


# =============================================================================
# Variable classification (continuous vs categorical)
#
# Continuous and categorical variables are tested by the same R² formula, but
# their OLS design differs (numeric column vs. one-hot). We still need to
# decide which encoding to use for each column.
# =============================================================================

_INTERNAL_COL_PATTERNS = (
    r"^pseudotime(_.*)?$",
    r"^cluster_.*_kmeans$",
    r"^X_DR_.*$",
    r"^_",
)


def _is_internal_col(col: str) -> bool:
    return any(re.match(p, col) for p in _INTERNAL_COL_PATTERNS)


def _classify_variables(
    obs: pd.DataFrame,
    sample_col: str,
    min_unique: int = 2,
    categorical_max_levels: int = 10,
) -> Tuple[List[str], List[str]]:
    """Split obs columns into (continuous, categorical)."""
    n = len(obs)
    continuous, categorical = [], []
    for col in obs.columns:
        if col == sample_col or _is_internal_col(col):
            continue
        s = obs[col].dropna()
        n_unique = s.nunique()
        if n_unique < min_unique:
            continue
        if pd.api.types.is_bool_dtype(s):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            is_float = pd.api.types.is_float_dtype(s)
            if n_unique > categorical_max_levels:
                continuous.append(col)
            elif is_float and n_unique > max(5, int(0.3 * n)):
                continuous.append(col)
            else:
                categorical.append(col)
        else:
            if n_unique <= max(categorical_max_levels, int(0.5 * len(s)) + 1):
                categorical.append(col)
    return continuous, categorical


# =============================================================================
# OLS per-PC variance decomposition
# =============================================================================

def _design_matrix(values: np.ndarray, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build [intercept, predictors] design matrix and a row-valid mask.

    Returns (design, valid_mask). `design` has the intercept as column 0.
    """
    n = len(values)
    if kind == "continuous":
        y = pd.to_numeric(pd.Series(values), errors="coerce").values.astype(float)
        valid = ~np.isnan(y)
        x = y[valid].reshape(-1, 1)
        design = np.hstack([np.ones((x.shape[0], 1)), x])
        return design, valid
    # categorical: one-hot, drop-first for identifiability
    s = pd.Series(values)
    valid = ~s.isna().values
    vals = s[valid].astype(str)
    levels = pd.Index(sorted(vals.unique()))
    if len(levels) < 2:
        return np.ones((valid.sum(), 1)), valid  # intercept-only → R² = 0
    dummies = pd.get_dummies(vals, prefix="lvl")[[f"lvl_{lv}" for lv in levels[1:]]]
    design = np.hstack([np.ones((len(vals), 1)), dummies.values.astype(float)])
    return design, valid


def _r2_from_design(y: np.ndarray, design: np.ndarray) -> float:
    """OLS R² = 1 - SSR/SST. Intercept-only design returns 0."""
    if design.shape[1] <= 1:
        return 0.0
    y_c = y - y.mean()
    sst = float((y_c ** 2).sum())
    if sst <= 0:
        return np.nan
    # lstsq via SVD; well-conditioned even for small n
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    ssr = float((resid ** 2).sum())
    return float(max(0.0, 1.0 - ssr / sst))


def _permutation_p_r2(
    y: np.ndarray,
    raw_values: np.ndarray,
    kind: str,
    observed_r2: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Null: shuffle raw_values, rebuild design, recompute R²."""
    if n_permutations <= 0 or np.isnan(observed_r2):
        return np.nan
    count = 0
    valid_count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(raw_values)
        d, mask = _design_matrix(perm, kind)
        if mask.sum() < 3 or d.shape[1] <= 1:
            continue
        r2_null = _r2_from_design(y[mask], d)
        if np.isnan(r2_null):
            continue
        valid_count += 1
        if r2_null >= observed_r2:
            count += 1
    if valid_count == 0:
        return np.nan
    return (count + 1) / (valid_count + 1)


def _variance_explained_one_embedding(
    X: np.ndarray,
    comp_names: List[str],
    obs: pd.DataFrame,
    continuous_cols: List[str],
    categorical_cols: List[str],
    n_permutations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Per-PC R² for every (variable, component). One row per cell."""
    rows = []
    plan: List[Tuple[str, str]] = (
        [(v, "continuous") for v in continuous_cols]
        + [(v, "categorical") for v in categorical_cols]
    )
    for var, kind in plan:
        if var not in obs.columns:
            continue
        raw = obs[var].values
        design_full, valid = _design_matrix(raw, kind)
        n_valid = int(valid.sum())
        p_preds = design_full.shape[1] - 1  # predictors beyond intercept
        if n_valid < max(4, p_preds + 2) or p_preds < 1:
            continue
        n_levels = int(pd.Series(raw).dropna().astype(str).nunique()) if kind == "categorical" else np.nan

        for ci, comp in enumerate(comp_names):
            y = X[valid, ci].astype(float)
            if np.nanstd(y) == 0:
                continue
            r2 = _r2_from_design(y, design_full)
            perm_p = _permutation_p_r2(
                y=X[:, ci].astype(float),
                raw_values=raw,
                kind=kind,
                observed_r2=r2,
                n_permutations=n_permutations,
                rng=rng,
            )

            row = {
                "variable": var,
                "component": comp,
                "kind": kind,
                "n_levels": n_levels,
                "r2": r2,
                "perm_p": perm_p,
                "n": n_valid,
            }
            if kind == "continuous":
                y_num = pd.to_numeric(obs[var], errors="coerce").values[valid].astype(float)
                if np.std(y_num) > 0 and np.std(y) > 0:
                    row["pearson_r"], _ = pearsonr(y, y_num)
                    row["spearman_r"], _ = spearmanr(y, y_num)
                else:
                    row["pearson_r"] = np.nan
                    row["spearman_r"] = np.nan
            else:
                row["pearson_r"] = np.nan
                row["spearman_r"] = np.nan
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "variable", "component", "kind", "n_levels",
            "r2", "perm_p", "fdr", "pearson_r", "spearman_r", "n",
        ])

    df = pd.DataFrame(rows)
    # BH-FDR across components within each variable
    df["fdr"] = np.nan
    for var, idx in df.groupby("variable").groups.items():
        idx_list = list(idx)
        pvals = df.loc[idx_list, "perm_p"].values
        mask = ~np.isnan(pvals)
        if mask.sum() == 0:
            continue
        _, fdr, _, _ = multipletests(pvals[mask], method="fdr_bh")
        out = np.full(len(pvals), np.nan)
        out[mask] = fdr
        df.loc[idx_list, "fdr"] = out

    return df


# =============================================================================
# Figures — one per embedding, PDF only
# =============================================================================

def _plot_variance_heatmap(
    df: pd.DataFrame,
    embedding_key: str,
    output_path: str,
) -> None:
    """Heatmap of R² across variables × components for a single embedding."""
    if df.empty:
        return
    with plt.rc_context(_JOURNAL_RC):
        comps = list(dict.fromkeys(df["component"].tolist()))
        # Sort variables by peak R² (categorical first in tie: they usually dominate)
        rank = df.groupby("variable")["r2"].max().sort_values(ascending=False)
        var_order = rank.index.tolist()

        r2_mat = pd.DataFrame(np.nan, index=var_order, columns=comps)
        fdr_mat = pd.DataFrame(np.nan, index=var_order, columns=comps)
        kind_map: Dict[str, str] = {}
        for _, row in df.iterrows():
            r2_mat.loc[row["variable"], row["component"]] = row["r2"]
            fdr_mat.loc[row["variable"], row["component"]] = row["fdr"]
            kind_map[row["variable"]] = row["kind"]

        # Any significant hits? Only then is the FDR-star legend useful.
        any_stars = bool(np.any(fdr_mat.values < 0.05))

        n_vars, n_cols = r2_mat.shape
        width = max(4.5, 0.55 * n_cols + 2.0)
        height = max(3.0, 0.42 * n_vars + 1.4)
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

        vmax = max(0.1, float(np.nanmax(r2_mat.values))) if r2_mat.size else 0.1
        im = ax.imshow(r2_mat.values, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(comps, rotation=45, ha="right")
        ax.set_yticks(range(n_vars))
        ax.set_yticklabels([f"{v}  [{kind_map[v][:3]}]" for v in var_order])

        for i, var in enumerate(var_order):
            for j, comp in enumerate(comps):
                val = r2_mat.iat[i, j]
                if np.isnan(val):
                    continue
                stars = _sig_stars(fdr_mat.iat[i, j])
                txt = f"{val:.2f}"
                if stars:
                    txt += f"\n{stars}"
                color = "white" if val > 0.55 * vmax else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

        ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_vars, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", length=0)

        title = embedding_key.replace("X_DR_", "").capitalize()
        # Fold the FDR-legend into the title so it cannot collide with tick labels.
        if any_stars:
            full_title = (
                f"Variance explained — {title} embedding\n"
                f"FDR: * < 0.05   ** < 0.01   *** < 0.001"
            )
        else:
            full_title = f"Variance explained — {title} embedding"
        ax.set_title(full_title, pad=8)

        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("R² (per-PC)", rotation=270, labelpad=12)
        cbar.outline.set_linewidth(0.5)

        fig.savefig(output_path)
        plt.close(fig)


def _plot_top_associations(
    pseudo_adata: AnnData,
    df: pd.DataFrame,
    embedding_key: str,
    output_path: str,
    top_n: int = 6,
) -> None:
    """Top (variable, PC) hits for one embedding: scatter or strip+box."""
    if df.empty:
        return
    d = df.dropna(subset=["fdr"]).copy()
    if d.empty:
        # Fall back to raw p if nothing has FDR (e.g. n_permutations=0)
        d = df.dropna(subset=["r2"]).copy()
        if d.empty:
            return
        d["_score"] = d["r2"]
    else:
        d["_score"] = -np.log10(d["fdr"].clip(lower=1e-10))
    # Dedupe: keep the best PC for each variable
    d = d.sort_values("_score", ascending=False)
    d = d.drop_duplicates(subset=["variable"], keep="first").head(top_n)
    if d.empty:
        return

    X, comps = _get_dr_matrix(pseudo_adata, embedding_key)
    ncols = min(3, len(d))
    nrows = int(np.ceil(len(d) / ncols))

    with plt.rc_context(_JOURNAL_RC):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.6 * ncols, 3.2 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        for ax_ in axes.flat[len(d):]:
            ax_.axis("off")

        for ax, (_, row) in zip(axes.flat, d.iterrows()):
            var = row["variable"]
            try:
                ci = comps.index(row["component"])
            except ValueError:
                ax.axis("off")
                continue
            y_emb = X[:, ci]
            if row["kind"] == "continuous":
                y_raw = pd.to_numeric(pseudo_adata.obs[var], errors="coerce").values
                valid = ~np.isnan(y_raw)
                xv, yv = y_emb[valid], y_raw[valid]
                ax.scatter(xv, yv, s=18, alpha=0.78,
                           edgecolor="white", linewidth=0.4, color="#3B7DB3")
                if len(xv) > 2 and np.std(xv) > 0:
                    z = np.polyfit(xv, yv, 1)
                    xs = np.linspace(xv.min(), xv.max(), 80)
                    ax.plot(xs, np.poly1d(z)(xs), color="#222222", lw=1.0, ls="--")
                ax.set_xlabel(row["component"])
                ax.set_ylabel(var)
                r_txt = (f"r={row['spearman_r']:.2f}  "
                         if not np.isnan(row.get("spearman_r", np.nan)) else "")
                ax.set_title(
                    f"{r_txt}R²={row['r2']:.2f}  FDR={row.get('fdr', np.nan):.2g}",
                    fontsize=9,
                )
            else:
                labels_raw = pseudo_adata.obs[var].astype(str).values
                mask = (labels_raw != "nan") & (~pd.isna(pseudo_adata.obs[var]))
                labs = labels_raw[mask]
                vals = y_emb[mask]
                groups = sorted(pd.unique(labs))
                positions = np.arange(len(groups))
                data = [vals[labs == g] for g in groups]
                ax.boxplot(
                    data, positions=positions, widths=0.55,
                    showfliers=False, patch_artist=True,
                    medianprops={"color": "black", "lw": 1},
                    boxprops={"facecolor": "#E7EEF4", "edgecolor": "#2E5984", "lw": 0.8},
                    whiskerprops={"color": "#2E5984", "lw": 0.8},
                    capprops={"color": "#2E5984", "lw": 0.8},
                )
                rng = np.random.default_rng(0)
                for pos, dat in zip(positions, data):
                    jitter = rng.uniform(-0.12, 0.12, size=len(dat))
                    ax.scatter(pos + jitter, dat, s=10, alpha=0.75,
                               color="#2E5984", edgecolor="white", linewidth=0.3)
                ax.set_xticks(positions)
                ax.set_xticklabels(groups, rotation=30, ha="right")
                ax.set_ylabel(row["component"])
                ax.set_title(
                    f"{var}  R²={row['r2']:.2f}  FDR={row.get('fdr', np.nan):.2g}",
                    fontsize=9,
                )

        title = embedding_key.replace("X_DR_", "").capitalize()
        fig.suptitle(f"Top phenotype ↔ {title} associations", fontsize=11)
        fig.savefig(output_path)
        plt.close(fig)


# =============================================================================
# Public entrypoint
# =============================================================================

def run_dimension_association_analysis(
    pseudo_adata: AnnData,
    output_dir: str,
    continuous_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    n_permutations: int = 999,
    sample_col: str = "sample",
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Per-PC variance-explained analysis across sample-level metadata.

    For each DR embedding present in ``pseudo_adata`` (expression,
    proportion), a table of (variable, component) R² values is computed.
    Both continuous and categorical variables are tested with a common
    OLS → R² formulation and a common permutation null, producing one
    directly-comparable metric per dimension.

    Outputs (under ``output_dir``)::

        variance_explained_expression.csv
        variance_explained_proportion.csv
        figures/expression_variance_heatmap.pdf
        figures/proportion_variance_heatmap.pdf
        figures/expression_top_associations.pdf
        figures/proportion_top_associations.pdf

    Args:
        pseudo_adata: AnnData with ``X_DR_expression`` / ``X_DR_proportion``
            in ``.uns`` or ``.obsm`` and per-sample metadata in ``.obs``.
        output_dir: root directory for outputs; created if missing.
        continuous_cols / categorical_cols: optional overrides for variable
            classification. If both are None, auto-classification is used.
        n_permutations: permutations for the R² null; ``0`` disables.
        sample_col: name of the sample-id column to exclude from testing.
        random_state: seed for the permutation RNG.

    Returns:
        Dict with keys ``results`` (dict of embedding → DataFrame),
        ``continuous_cols``, ``categorical_cols``.
    """
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    auto_cont, auto_cat = _classify_variables(pseudo_adata.obs, sample_col=sample_col)
    if continuous_cols is None:
        continuous_cols = auto_cont
    if categorical_cols is None:
        categorical_cols = auto_cat
    continuous_cols = [c for c in dict.fromkeys(continuous_cols) if c in pseudo_adata.obs.columns]
    categorical_cols = [c for c in dict.fromkeys(categorical_cols) if c in pseudo_adata.obs.columns]

    if verbose:
        print(f"[Association] Continuous variables ({len(continuous_cols)}): {continuous_cols}")
        print(f"[Association] Categorical variables ({len(categorical_cols)}): {categorical_cols}")

    results: Dict[str, pd.DataFrame] = {}
    for emb in _available_embeddings(pseudo_adata):
        X, comps = _get_dr_matrix(pseudo_adata, emb)
        rng = np.random.default_rng(random_state)
        df = _variance_explained_one_embedding(
            X=X, comp_names=comps,
            obs=pseudo_adata.obs,
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            n_permutations=n_permutations,
            rng=rng,
        )
        results[emb] = df

        tag = emb.replace("X_DR_", "")
        csv_path = os.path.join(output_dir, f"variance_explained_{tag}.csv")
        df.to_csv(csv_path, index=False)
        if verbose:
            print(f"[Association] {emb}: {len(df)} rows → {csv_path}")

        try:
            _plot_variance_heatmap(
                df, emb,
                os.path.join(figures_dir, f"{tag}_variance_heatmap.pdf"),
            )
        except Exception as e:
            print(f"[Association] Warning: variance heatmap failed for {emb}: {e}")

        try:
            _plot_top_associations(
                pseudo_adata, df, emb,
                os.path.join(figures_dir, f"{tag}_top_associations.pdf"),
            )
        except Exception as e:
            print(f"[Association] Warning: top-associations plot failed for {emb}: {e}")

    return {
        "continuous_cols": continuous_cols,
        "categorical_cols": categorical_cols,
        "results": results,
    }
