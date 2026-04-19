import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from typing import Optional, List
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_dr_matrix(pseudo_adata: AnnData, embedding_key: str):
    """Return (ndarray, component_names) for a DR embedding key."""
    if embedding_key in pseudo_adata.uns and isinstance(pseudo_adata.uns[embedding_key], pd.DataFrame):
        df = pseudo_adata.uns[embedding_key]
        return df.values, list(df.columns)
    if embedding_key in pseudo_adata.obsm:
        X = pseudo_adata.obsm[embedding_key]
        tag = embedding_key.replace("X_DR_", "")
        return X, [f"{tag}_{i+1}" for i in range(X.shape[1])]
    raise KeyError(f"Embedding '{embedding_key}' not found in pseudo_adata.uns or obsm")


# ---------------------------------------------------------------------------
# B2.1  Continuous-variable correlation
# ---------------------------------------------------------------------------

def correlate_embedding_with_variable(
    pseudo_adata: AnnData,
    continuous_cols: List[str],
    output_dir: str,
    embedding_keys: Optional[List[str]] = None,
    n_permutations: int = 999,
    verbose: bool = True,
) -> dict:
    """
    Correlate each DR component with continuous phenotype variables.

    For every (embedding, variable) pair, computes Pearson + Spearman r and a
    permutation p-value, then applies BH-FDR across components within each pair.

    Parameters
    ----------
    pseudo_adata : AnnData
        Sample-level AnnData with DR embeddings in obsm/uns and metadata in obs.
    continuous_cols : list of str
        Columns in pseudo_adata.obs to test (continuous / ordinal).
    output_dir : str
        Directory to write results.
    embedding_keys : list of str, optional
        DR keys to test. Defaults to ["X_DR_expression", "X_DR_proportion"].
    n_permutations : int
        Permutations for null p-value (default 999).
    verbose : bool

    Returns
    -------
    dict mapping embedding_key -> tidy results DataFrame.
    """
    if embedding_keys is None:
        embedding_keys = ["X_DR_expression", "X_DR_proportion"]

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for emb_key in embedding_keys:
        try:
            X, comp_names = _get_dr_matrix(pseudo_adata, emb_key)
        except KeyError as e:
            print(f"[Association] Skipping {emb_key}: {e}")
            continue

        emb_label = emb_key.replace("X_DR_", "")
        rows = []
        rng = np.random.default_rng(42)

        for var_col in continuous_cols:
            if var_col not in pseudo_adata.obs.columns:
                print(f"[Association] Column '{var_col}' not found in obs, skipping.")
                continue

            y_raw = pseudo_adata.obs[var_col].values
            valid = ~pd.isna(pd.Series(y_raw))
            if valid.sum() < 4:
                print(f"[Association] Too few valid samples for '{var_col}', skipping.")
                continue

            y_v = y_raw[valid].astype(float)
            X_v = X[valid]

            for ci, comp_name in enumerate(comp_names):
                x_c = X_v[:, ci]
                pr, pp = pearsonr(x_c, y_v)
                sr, sp = spearmanr(x_c, y_v)
                null_r = np.array([
                    pearsonr(rng.permutation(x_c), y_v)[0]
                    for _ in range(n_permutations)
                ])
                perm_p = float(np.mean(np.abs(null_r) >= np.abs(pr)))
                rows.append({
                    "variable": var_col,
                    "component": comp_name,
                    "pearson_r": pr,
                    "pearson_p": pp,
                    "spearman_r": sr,
                    "spearman_p": sp,
                    "perm_p": perm_p,
                })

        if not rows:
            continue

        result_df = pd.DataFrame(rows)

        # BH-FDR per (variable, embedding) pair
        fdr_vals = []
        for var_col in result_df["variable"].unique():
            mask = result_df["variable"] == var_col
            pvals = result_df.loc[mask, "perm_p"].values
            _, fdr, _, _ = multipletests(pvals, method="fdr_bh")
            fdr_vals.extend(fdr.tolist())
        result_df["fdr_perm"] = fdr_vals

        csv_path = os.path.join(output_dir, f"association_correlation_{emb_label}.csv")
        result_df.to_csv(csv_path, index=False)
        all_results[emb_key] = result_df

        if verbose:
            print(f"[Association] {emb_key}: correlation results saved to {csv_path}")

        try:
            _plot_association_heatmap(result_df, continuous_cols, emb_label, output_dir)
            _plot_association_scatter_top3(
                result_df, pseudo_adata, X, comp_names, continuous_cols, emb_label, output_dir
            )
        except Exception as e:
            print(f"[Association] Warning: plot failed for {emb_key}: {e}")

    return all_results


def _plot_association_heatmap(result_df, continuous_cols, emb_label, output_dir):
    """Heatmap: rows = variables, cols = DR components; color = Spearman r; * = FDR."""
    vars_in = [v for v in continuous_cols if v in result_df["variable"].values]
    comps = list(result_df["component"].unique())
    if not vars_in or not comps:
        return

    r_mat = pd.DataFrame(np.nan, index=vars_in, columns=comps)
    sig_mat = pd.DataFrame("", index=vars_in, columns=comps)

    for var_col in vars_in:
        sub = result_df[result_df["variable"] == var_col].set_index("component")
        for comp in comps:
            if comp in sub.index:
                r_mat.loc[var_col, comp] = sub.loc[comp, "spearman_r"]
                fdr = sub.loc[comp, "fdr_perm"]
                sig_mat.loc[var_col, comp] = (
                    "***" if fdr < 0.001 else ("**" if fdr < 0.01 else ("*" if fdr < 0.05 else ""))
                )

    fig, ax = plt.subplots(figsize=(max(6, len(comps) * 0.6), max(4, len(vars_in) * 0.8)))
    sns.heatmap(
        r_mat.astype(float), ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=sig_mat.values, fmt="s", linewidths=0.5,
        cbar_kws={"label": "Spearman r"},
    )
    ax.set_title(f"Association Heatmap — {emb_label}")
    ax.set_xlabel("DR Component")
    ax.set_ylabel("Variable")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"association_heatmap_{emb_label}.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()


def _plot_association_scatter_top3(result_df, pseudo_adata, X, comp_names,
                                   continuous_cols, emb_label, output_dir):
    """Top-3 component scatter per variable with regression line."""
    for var_col in continuous_cols:
        if var_col not in pseudo_adata.obs.columns:
            continue
        sub = result_df[result_df["variable"] == var_col].copy()
        if sub.empty:
            continue
        sub = sub.iloc[sub["pearson_r"].abs().argsort()[::-1].values]
        top3 = sub.head(3)

        y_raw = pseudo_adata.obs[var_col].values.astype(float)
        valid = ~np.isnan(y_raw)
        n_panels = min(3, len(top3))
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
        if n_panels == 1:
            axes = [axes]

        for ax, (_, row) in zip(axes, top3.iterrows()):
            ci = comp_names.index(row["component"])
            x_c = X[:, ci]
            ax.scatter(x_c[valid], y_raw[valid], alpha=0.7, edgecolors="k", linewidths=0.5)
            if valid.sum() > 2:
                z = np.polyfit(x_c[valid], y_raw[valid], 1)
                xs = np.linspace(x_c[valid].min(), x_c[valid].max(), 100)
                ax.plot(xs, np.poly1d(z)(xs), "r--", lw=1.5)
            ax.set_xlabel(row["component"])
            ax.set_ylabel(var_col)
            ax.set_title(f"r={row['pearson_r']:.2f}, FDR={row['fdr_perm']:.3f}")

        safe = var_col.replace("/", "_").replace(" ", "_")
        plt.suptitle(f"Top Associations — {var_col} ({emb_label})", y=1.02)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"association_scatter_top3_{emb_label}_{safe}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()


# ---------------------------------------------------------------------------
# B2.2  PERMANOVA on distance matrices
# ---------------------------------------------------------------------------

def permanova_on_distance(
    pseudo_adata: AnnData,
    categorical_cols: List[str],
    distance_dir: str,
    output_dir: str,
    distance_methods: Optional[List[str]] = None,
    n_permutations: int = 999,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run PERMANOVA for categorical phenotype variables × sample distance matrices.

    Uses skbio.stats.distance.permanova if available; otherwise falls back to a
    numpy-only pseudo-F permutation test.

    Parameters
    ----------
    pseudo_adata : AnnData
        Sample-level AnnData with categorical_cols in obs.
    categorical_cols : list of str
        Categorical columns in pseudo_adata.obs.
    distance_dir : str
        Root of the Sample_distance/ tree produced by sample_distance.py.
    output_dir : str
        Directory to write PERMANOVA results.
    distance_methods : list of str, optional
        Method names to search for. Defaults to ["cosine"].
    n_permutations : int
        Permutations (default 999).
    verbose : bool

    Returns
    -------
    pd.DataFrame with columns: distance_method, grouping_column, pseudo_F, R2, p_value, n_samples.
    """
    if distance_methods is None:
        distance_methods = ["cosine"]

    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for method in distance_methods:
        dist_matrix = _load_distance_matrix(distance_dir, method)
        if dist_matrix is None:
            print(f"[PERMANOVA] No distance matrix found for '{method}' in {distance_dir}")
            continue

        common = [s for s in pseudo_adata.obs_names if s in dist_matrix.index]
        if len(common) < 4:
            print(f"[PERMANOVA] Too few overlapping samples for '{method}', skipping.")
            continue

        D = dist_matrix.loc[common, common].values.astype(float)

        for grp_col in categorical_cols:
            if grp_col not in pseudo_adata.obs.columns:
                print(f"[PERMANOVA] Column '{grp_col}' not in obs, skipping.")
                continue

            labels_raw = pseudo_adata.obs.loc[common, grp_col].values
            valid = ~pd.isna(pd.Series(labels_raw))
            if valid.sum() < 4:
                continue

            labels = labels_raw[valid].astype(str)
            D_v = D[np.ix_(valid, valid)]

            pseudo_F, R2, p_value = _permanova_test(D_v, labels, n_permutations)

            rows.append({
                "distance_method": method,
                "grouping_column": grp_col,
                "pseudo_F": pseudo_F,
                "R2": R2,
                "p_value": p_value,
                "n_samples": int(valid.sum()),
            })

            if verbose:
                print(
                    f"[PERMANOVA] {method} × {grp_col}: "
                    f"pseudo_F={pseudo_F:.3f}, R²={R2:.3f}, p={p_value:.4f}"
                )

            try:
                _plot_permanova_null(D_v, labels, method, grp_col, pseudo_F, p_value,
                                     n_permutations, output_dir)
                _plot_within_between_boxplot(D_v, labels, method, grp_col, output_dir)
            except Exception as e:
                print(f"[PERMANOVA] Warning: plot failed for {method}×{grp_col}: {e}")

    summary_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["distance_method", "grouping_column", "pseudo_F", "R2", "p_value", "n_samples"]
    )
    summary_path = os.path.join(output_dir, "permanova_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    if len(summary_df) > 1:
        try:
            _plot_permanova_summary_heatmap(summary_df, output_dir)
        except Exception as e:
            print(f"[PERMANOVA] Warning: summary heatmap failed: {e}")

    if verbose:
        print(f"[PERMANOVA] Summary saved to: {summary_path}")

    return summary_df


def _load_distance_matrix(distance_dir: str, method: str) -> Optional[pd.DataFrame]:
    """Search distance_dir recursively for a square distance matrix CSV matching method."""
    candidates = []
    for root, _, files in os.walk(distance_dir):
        if method.lower() not in root.lower():
            continue
        for fname in files:
            if fname.startswith("distance_matrix") and fname.endswith(".csv"):
                candidates.append(os.path.join(root, fname))

    for path in sorted(set(candidates)):
        try:
            df = pd.read_csv(path, index_col=0)
            if df.shape[0] == df.shape[1] and df.shape[0] > 1:
                return df
        except Exception:
            continue
    return None


def _pseudo_F_stat(D: np.ndarray, labels: np.ndarray):
    """PERMANOVA pseudo-F and R² from a symmetric distance matrix."""
    n = len(labels)
    groups = np.unique(labels)
    k = len(groups)
    if k < 2:
        return np.nan, np.nan

    # Gower centering
    A = -0.5 * (D ** 2)
    row_mean = A.mean(axis=1, keepdims=True)
    col_mean = A.mean(axis=0, keepdims=True)
    G = A - row_mean - col_mean + A.mean()

    SS_total = np.trace(G)
    SS_between = sum(
        G[np.ix_(np.where(labels == g)[0], np.where(labels == g)[0])].sum() / len(np.where(labels == g)[0])
        for g in groups
    )
    SS_within = SS_total - SS_between

    df_between = k - 1
    df_within = n - k
    if df_within <= 0 or SS_within == 0:
        return np.nan, np.nan

    pseudo_F = (SS_between / df_between) / (SS_within / df_within)
    R2 = SS_between / SS_total if SS_total != 0 else np.nan
    return float(pseudo_F), float(R2)


def _permanova_test(D: np.ndarray, labels: np.ndarray, n_permutations: int):
    """Run PERMANOVA, preferring skbio; fallback to in-house numpy implementation."""
    try:
        from skbio.stats.distance import permanova as skbio_permanova, DistanceMatrix
        dm = DistanceMatrix(D)
        grouping = pd.Series(labels, name="group")
        res = skbio_permanova(dm, grouping, permutations=n_permutations)
        _, R2 = _pseudo_F_stat(D, labels)
        return float(res["test statistic"]), R2, float(res["p-value"])
    except ImportError:
        pass

    obs_F, R2 = _pseudo_F_stat(D, labels)
    if np.isnan(obs_F):
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(42)
    null_F = []
    for _ in range(n_permutations):
        f, _ = _pseudo_F_stat(D, rng.permutation(labels))
        if not np.isnan(f):
            null_F.append(f)

    p_value = (np.sum(np.array(null_F) >= obs_F) + 1) / (len(null_F) + 1)
    return obs_F, R2, float(p_value)


def _plot_permanova_null(D, labels, method, grp_col, obs_F, p_value, n_permutations, output_dir):
    rng = np.random.default_rng(123)
    null_F = []
    for _ in range(min(n_permutations, 299)):
        f, _ = _pseudo_F_stat(D, rng.permutation(labels))
        if not np.isnan(f):
            null_F.append(f)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_F, bins=30, alpha=0.7, label="Null distribution")
    ax.axvline(obs_F, color="r", linestyle="--", lw=2, label=f"Observed F (p={p_value:.3f})")
    ax.set_xlabel("Pseudo-F")
    ax.set_ylabel("Count")
    ax.set_title(f"PERMANOVA Null — {method} × {grp_col}")
    ax.legend()
    plt.tight_layout()
    safe = grp_col.replace("/", "_").replace(" ", "_")
    plt.savefig(
        os.path.join(output_dir, f"permanova_null_{method}_{safe}.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()


def _plot_within_between_boxplot(D, labels, method, grp_col, output_dir):
    n = len(labels)
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if labels[i] == labels[j] else between).append(D[i, j])

    df = pd.DataFrame({
        "distance": within + between,
        "type": ["Within"] * len(within) + ["Between"] * len(between),
    })
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=df, x="type", y="distance", ax=ax, palette=["#4C72B0", "#DD8452"])
    safe = grp_col.replace("/", "_").replace(" ", "_")
    ax.set_title(f"Within vs Between Group Distance\n{method} × {grp_col}")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"distance_by_group_boxplot_{method}_{safe}.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()


def _plot_permanova_summary_heatmap(summary_df: pd.DataFrame, output_dir: str):
    pivot = summary_df.pivot(index="distance_method", columns="grouping_column", values="p_value")
    log_p = -np.log10(pivot.astype(float).clip(lower=1e-10))

    fig, ax = plt.subplots(figsize=(max(4, log_p.shape[1] * 1.2), max(3, log_p.shape[0] * 0.8)))
    sns.heatmap(
        log_p, ax=ax, cmap="YlOrRd",
        annot=pivot.round(3).values, fmt="s",
        linewidths=0.5, cbar_kws={"label": "−log₁₀(p)"},
    )
    ax.set_title("PERMANOVA Summary (−log₁₀ p-value)")
    ax.set_xlabel("Grouping Variable")
    ax.set_ylabel("Distance Method")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "permanova_summary_heatmap.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
