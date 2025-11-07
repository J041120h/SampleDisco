from __future__ import annotations
import os
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# ---------------- Utilities ---------------- #

def _bootstrap_ci_mean(vals: np.ndarray, iters: int = 2000, alpha: float = 0.05, seed: int = 42) -> Tuple[float,float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)
    if len(vals) == 0:
        return (np.nan, np.nan)
    bs = rng.choice(vals, size=(iters, len(vals)), replace=True).mean(axis=1)
    return float(np.quantile(bs, alpha/2)), float(np.quantile(bs, 1 - alpha/2))

def _transform_severity(sev: np.ndarray, mode: str = "raw") -> np.ndarray:
    sev = np.asarray(sev, dtype=float)
    if mode == "raw":
        return sev
    if mode == "scaled":
        mn, mx = np.nanmin(sev), np.nanmax(sev)
        return (sev - mn) / (mx - mn) if mx > mn else np.zeros_like(sev)
    if mode == "rank":
        order = sev.argsort(kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(sev), dtype=float)
        ranks /= (len(sev) - 1) if len(sev) > 1 else 1.0
        return ranks
    raise ValueError(f"Unknown severity_transform='{mode}'")

def _aggregate(vals: np.ndarray, dist: Optional[np.ndarray], how: str) -> np.ndarray:
    """Aggregate k-NN per-row values to a single score."""
    if vals.ndim == 1:
        return vals
    how = how.lower()
    if how == "mean":
        return vals.mean(axis=1)
    if how == "median":
        return np.median(vals, axis=1)
    if how == "distance_weighted":
        if dist is None:
            raise ValueError("distance_weighted aggregation requires distances.")
        w = 1.0 / np.clip(dist, 1e-8, None)
        w /= w.sum(axis=1, keepdims=True)
        return (vals * w).sum(axis=1)
    raise ValueError(f"Unknown nn_agg='{how}'")

# -------------- Main Benchmark -------------- #

def benchmark_pseudotime_embeddings_custom(
    df: pd.DataFrame,
    embedding: np.ndarray,
    method_name: str = "embedding",
    *,
    anchor_batch: str = "Su",
    batch_col: str = "batch",
    sev_col: str = "sev.level",
    pseudotime_col: Optional[str] = None,
    severity_transform: str = "raw",
    neighbor_batches_include: Optional[Sequence[str]] = None,
    neighbor_batches_exclude: Optional[Sequence[str]] = None,
    k_neighbors: int = 1,
    metric: str = "euclidean",
    nn_agg: str = "mean",
    standardize_embedding: bool = True,
    make_plots: bool = True,
    save_dir: str = "benchmark_out",
    random_state: int = 0,
) -> Dict[str, object]:
    """
    Two-part benchmark for a SINGLE embedding:
      (1) Su-only (anchor batch) ANOVA: pseudotime ~ C(severity), plotting p-value and effect size (η², ω²)
      (2) For each Su sample, find k nearest neighbors from OTHER batches; compute |Δseverity| and plot ONE histogram

    Note: sev_col may include dots/spaces; we quote it in the formula using Patsy's Q('...').
    """
    os.makedirs(save_dir, exist_ok=True)
    out = {"paths": {}}

    if batch_col not in df or sev_col not in df:
        raise ValueError("df must contain batch_col and sev_col.")

    anchor_mask = (df[batch_col] == anchor_batch).values
    if not anchor_mask.any():
        raise ValueError(f"No rows for anchor batch '{anchor_batch}'.")

    sev_raw = df[sev_col].to_numpy()
    sev_for_nn = _transform_severity(sev_raw, severity_transform)

    # ---------- Part 1: Su-only ANOVA (with effect size) ----------
    print(f"\n{'='*60}")
    print("DEBUG: ANOVA Analysis")
    print(f"{'='*60}")
    print(f"pseudotime_col = {pseudotime_col}")
    print(f"make_plots = {make_plots}")
    
    if pseudotime_col is not None:
        if pseudotime_col not in df.columns:
            raise ValueError(f"pseudotime_col '{pseudotime_col}' not in df.")
        
        df_anchor = df.loc[anchor_mask].copy()
        print(f"Anchor batch '{anchor_batch}' has {len(df_anchor)} samples")
        print(f"Unique severity levels: {sorted(df_anchor[sev_col].dropna().unique())}")

        # Safely reference severity column with Patsy's Q(...)
        formula = f"{pseudotime_col} ~ C(Q('{sev_col}'))"
        ols = smf.ols(formula, data=df_anchor).fit()
        aov = anova_lm(ols, typ=2)
        out["anova_anchor"] = aov
        print(f"\nANOVA table:\n{aov}")

        # Identify rows and compute effect sizes
        # One factor + Residual in typ=2 one-way ANOVA
        # Row name of factor will look like "C(Q('sev.level'))"
        effect_row = next((idx for idx in aov.index if idx != "Residual"), None)
        ss_effect = float(aov.loc[effect_row, "sum_sq"]) if effect_row else np.nan
        df_effect = float(aov.loc[effect_row, "df"]) if effect_row else np.nan
        ss_resid  = float(aov.loc["Residual", "sum_sq"])
        df_resid  = float(aov.loc["Residual", "df"])
        ss_total  = ss_effect + ss_resid
        mse       = ss_resid / df_resid if df_resid > 0 else np.nan

        # Eta-squared and Omega-squared (unbiased)
        eta_sq = ss_effect / ss_total if ss_total > 0 else np.nan
        omega_sq = ((ss_effect - df_effect * mse) /
                    (ss_total + mse)) if (not np.isnan(mse) and (ss_total + mse) > 0) else np.nan

        # p-value (if available)
        p_val = aov.loc[effect_row, "PR(>F)"] if (effect_row and "PR(>F)" in aov.columns) else np.nan

        # Parallel SciPy check (optional)
        groups = [g[pseudotime_col].values for _, g in df_anchor.groupby(sev_col)]
        if sum(len(g) > 1 for g in groups) >= 2:
            F, p = f_oneway(*groups)
        else:
            F, p = (np.nan, np.nan)
        out["anova_anchor_scipy"] = {"F": F, "p": p}
        print(f"SciPy F-test: F={F:.4f}, p={p:.4g}")
        print(f"Effect sizes: eta²={eta_sq:.4f}, omega²={omega_sq:.4f}")

        # Plot: ANOVA box/jitter + effect sizes in title
        if make_plots:
            print("\n>>> Generating ANOVA plot with effect size...")
            fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=160)
            order = sorted(df_anchor[sev_col].dropna().unique())
            data_to_plot = [df_anchor.loc[df_anchor[sev_col] == s, pseudotime_col].values for s in order]
            ax.boxplot(data_to_plot, positions=np.arange(len(order)) + 1, widths=0.6, showfliers=False)
            for i, s in enumerate(order, start=1):
                y = df_anchor.loc[df_anchor[sev_col] == s, pseudotime_col].values
                x = np.random.default_rng(random_state + i).normal(i, 0.045, size=len(y))
                ax.scatter(x, y, s=12, alpha=0.7)

            title = (f"{anchor_batch} only — ANOVA: {pseudotime_col} ~ C({sev_col})"
                     f"\n p={p_val:.3g} • η²={eta_sq:.3f} • ω²={omega_sq:.3f}")
            ax.set_title(title)
            ax.set_xlabel("Severity Level")
            ax.set_ylabel(pseudotime_col)
            ax.set_xticks(np.arange(len(order)) + 1)
            ax.set_xticklabels([str(s) for s in order])
            p1 = os.path.join(save_dir, f"{anchor_batch.lower()}_anova_box_effectsize.png")
            plt.tight_layout()
            plt.savefig(p1)
            plt.close(fig)
            out["paths"]["anchor_anova_box"] = p1
            print(f">>> ANOVA plot saved to: {p1}")
        else:
            print(">>> make_plots=False, skipping ANOVA plot")
    else:
        print(">>> pseudotime_col is None, skipping ANOVA analysis")
        out["anova_anchor"] = None
        out["anova_anchor_scipy"] = None

    # ---------- Part 2: Su-anchored nearest-neighbor severity gap (ONE plot only) ----------
    print(f"\n{'='*60}")
    print("DEBUG: Nearest Neighbor Severity Gap Analysis")
    print(f"{'='*60}")
    
    pool_mask = ~anchor_mask
    if neighbor_batches_include is not None:
        pool_mask &= df[batch_col].isin(set(neighbor_batches_include)).values
    if neighbor_batches_exclude is not None:
        pool_mask &= ~df[batch_col].isin(set(neighbor_batches_exclude)).values
    if not pool_mask.any():
        raise ValueError("No samples left in neighbor pool after filters.")

    X = np.asarray(embedding)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != len(df):
        raise ValueError(f"Embedding has {X.shape[0]} rows; df has {len(df)}.")

    X_use = X.copy()
    if standardize_embedding:
        X_use = StandardScaler().fit_transform(X_use)

    X_anchor = X_use[anchor_mask]
    sev_anchor = sev_for_nn[anchor_mask]

    X_pool = X_use[pool_mask]
    sev_pool = sev_for_nn[pool_mask]
    
    print(f"Anchor samples: {len(X_anchor)}")
    print(f"Pool samples: {len(X_pool)}")
    print(f"k_neighbors: {k_neighbors}")

    if X_pool.shape[0] < k_neighbors:
        raise ValueError(f"Neighbor pool smaller than k ({X_pool.shape[0]} < {k_neighbors}).")

    nn = NearestNeighbors(n_neighbors=k_neighbors, metric=metric)
    nn.fit(X_pool)
    dist, idx = nn.kneighbors(X_anchor)

    sev_nn = sev_pool[idx]
    delta = np.abs(sev_nn - sev_anchor[:, None])

    per_su = _aggregate(delta, dist, nn_agg)
    mean_gap = float(per_su.mean())
    lo, hi = _bootstrap_ci_mean(per_su)

    print(f"\nResults for '{method_name}':")
    print(f"  Mean |Δseverity|: {mean_gap:.4f}")
    print(f"  95% CI: [{lo:.4f}, {hi:.4f}]")

    summary = {
        "method": method_name,
        "mean_|Δsev|": mean_gap,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "k": k_neighbors,
        "metric": metric,
        "nn_agg": nn_agg,
        "severity_transform": severity_transform,
        "n_anchor": int(len(per_su)),
        "neighbor_pool_n": int(X_pool.shape[0]),
    }

    out["nn_gap_summary"] = pd.DataFrame([summary])
    out["nn_gap_per_sample"] = per_su
    
    csv_path = os.path.join(save_dir, "nn_severity_gap_summary.csv")
    out["nn_gap_summary"].to_csv(csv_path, index=False)
    out["paths"]["nn_gap_summary_csv"] = csv_path
    print(f"\nSummary CSV saved to: {csv_path}")

    if make_plots:
        # ONE PLOT ONLY: Histogram of per-sample gaps (no ECDF)
        fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=160)
        ax.hist(per_su, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(mean_gap, linestyle='--', linewidth=2, label=f'Mean={mean_gap:.2f}')
        ax.set_xlabel("|Δ severity|")
        ax.set_ylabel("Count")
        ax.set_title(f"{method_name}: Per-{anchor_batch} NN |Δ severity| (k={k_neighbors})\n"
                     f"Mean={mean_gap:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")
        ax.legend()
        p2 = os.path.join(save_dir, "nn_severity_gap_hist.png")
        plt.tight_layout(); plt.savefig(p2); plt.close(fig)
        out["paths"]["nn_gap_hist"] = p2
        print(f"Histogram saved to: {p2}")

    print(f"{'='*60}\n")
    return out
