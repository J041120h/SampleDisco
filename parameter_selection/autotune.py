"""Sample-embedding hyperparameter autotune.

Bayesian search over the CMD weight (`alpha_only` scope by default) using the
`multi_metric_proxy` ensemble. Adaptive — the proxy ensemble drops components
the data can't support:

  - If the dataset has no usable batch column → drop iLISI(batch) and
    ASW(batch) proxies (they need a discrete batch label).
  - If the dataset has no grouping/trajectory label → drop supervised proxies
    (CCA, SPS, CV-kNN, pseudotime-Spearman).
  - If neither → emit a warning and short-circuit with fixed defaults.

Generalized version of `wire_autotune_dualembed_v2.py`. No dataset-specific
paths; all data flows through the same `compute_sample_embedding` primitives
in `sample_embedding/blocks.py`.
"""

from __future__ import annotations

import math
import os
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm, spearmanr

from sample_embedding.blocks import (
    assemble_units,
    build_emb_from_blocks,
    composition_per_unit,
    derive_weights,
    loo_cmd,
    soft_assign,
)


DEFAULT_ALPHA_BOUNDS = (0.1, 10.0)


# ============================================================ #
# Pre-compute blocks (composition + CMD) once; sweep weights    #
# ============================================================ #
def build_blocks(
    adata: AnnData,
    sample_col: str,
    celltype_col: str,
    cluster_emb_key: str,
    cmd_emb_key: Optional[str] = None,
    modality_col: Optional[str] = None,
    batch_col: Optional[str] = None,
    grouping_col: Optional[str] = None,
    medium_K: int = 120,
    fine_K: int = 300,
    cmd_dim: int = 8,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Build composition + CMD blocks once. Returns a dict the inner loop reuses."""
    cmd_key = cmd_emb_key if cmd_emb_key and cmd_emb_key in adata.obsm else (
        f"{cluster_emb_key}_nosamp" if f"{cluster_emb_key}_nosamp" in adata.obsm
        else cluster_emb_key
    )

    units, unit_cellids, unit_ids, unit_groups, unit_batches, all_cellids, Z_clust = \
        assemble_units(adata, sample_col, cluster_emb_key,
                       modality_col=modality_col, batch_col=batch_col)
    n_units = len(units)
    cellid_idx = {cid: i for i, cid in enumerate(all_cellids)}

    cell_type = adata.obs[celltype_col].astype(str).values
    unique_cts = sorted(set(cell_type))
    K_c = len(unique_cts)

    # A1
    L1 = {ct: i for i, ct in enumerate(unique_cts)}
    soft1 = np.zeros((Z_clust.shape[0], K_c), dtype=np.float32)
    for i, ct in enumerate(cell_type):
        soft1[i, L1[ct]] = 1.0
    unit_cellids_list = [unit_cellids[uid] for uid in unit_ids]
    A1 = composition_per_unit(unit_cellids_list, soft1, cellid_idx)

    K_med = min(medium_K, max(2, Z_clust.shape[0] // 200))
    if verbose:
        print(f"[autotune.build_blocks] K-means K={K_med}...")
    km_med = MiniBatchKMeans(n_clusters=K_med, random_state=seed,
                              batch_size=4096, n_init=5, max_iter=200).fit(Z_clust)
    soft2 = soft_assign(Z_clust, km_med.cluster_centers_)
    A2 = composition_per_unit(unit_cellids_list, soft2, cellid_idx)

    K_fine = min(fine_K, max(2, Z_clust.shape[0] // 100))
    if verbose:
        print(f"[autotune.build_blocks] K-means K={K_fine}...")
    km_fine = MiniBatchKMeans(n_clusters=K_fine, random_state=seed + 1,
                                batch_size=4096, n_init=5, max_iter=200).fit(Z_clust)
    soft3 = soft_assign(Z_clust, km_fine.cluster_centers_)
    A3 = composition_per_unit(unit_cellids_list, soft3, cellid_idx)

    # CMD
    Z_cmd = np.asarray(adata.obsm[cmd_key], dtype=np.float32)
    cmd_units = []
    for uid, group in zip(unit_ids, unit_groups):
        cids = unit_cellids[uid]
        idxs = [cellid_idx[c] for c in cids if c in cellid_idx]
        cmd_units.append((uid, group, Z_cmd[idxs]))
    coarse_label_map = dict(zip(all_cellids, cell_type))
    CMD = loo_cmd(cmd_units, unit_cellids, coarse_label_map,
                    max_dim_per_cluster=cmd_dim, seed=seed,
                    loo=True, verbose=False)

    # Sample-level metadata for scoring
    if grouping_col is not None and grouping_col in adata.obs.columns:
        grp_series = adata.obs.groupby(sample_col, observed=True)[grouping_col].agg(
            lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan
        )
        # Align to unit_ids — for MO uids are "{sample}_{modality}", so try suffix-strip
        grouping = []
        for uid in unit_ids:
            val = grp_series.get(uid, np.nan)
            if pd.isna(val):
                # try without modality suffix
                for s in grp_series.index:
                    if uid.startswith(str(s) + "_"):
                        val = grp_series[s]
                        break
            grouping.append(val)
        grouping_arr = np.asarray(grouping)
    else:
        grouping_arr = None

    batch_arr = (np.asarray(unit_batches) if unit_batches is not None
                  else np.asarray(unit_groups))
    has_batch = len(set(batch_arr.tolist())) > 1
    has_grouping = grouping_arr is not None and len(
        set(x for x in grouping_arr.tolist() if pd.notna(x))
    ) > 1

    return dict(
        A1=A1, A2=A2, A3=A3, CMD=CMD,
        K_c=K_c, K_med=K_med, K_fine=K_fine,
        unit_ids=unit_ids, unit_groups=unit_groups, unit_batches=unit_batches,
        n_units=n_units, grouping=grouping_arr, batch=batch_arr,
        has_batch=has_batch, has_grouping=has_grouping,
        cluster_emb_key=cluster_emb_key, cmd_emb_key=cmd_key,
    )


# ============================================================ #
# Scoring functions                                              #
# ============================================================ #
def _cca_corr(emb, target):
    e = np.asarray(emb)
    t = np.asarray(target, dtype=float).reshape(-1, 1)
    keep = ~np.isnan(t.flatten())
    if keep.sum() < 4:
        return 0.0
    e = e[keep]
    t = t[keep]
    n_pc = min(10, e.shape[1], e.shape[0] - 1)
    Xr = PCA(n_components=n_pc, random_state=42).fit_transform(e)
    try:
        c = CCA(n_components=1, max_iter=500).fit(Xr, t)
        U, V = c.transform(Xr, t)
        r = float(abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1]))
        return r if np.isfinite(r) else 0.0
    except Exception:
        return 0.0


def _ilisi_norm(emb, labels, k=15):
    e = np.asarray(emb)
    labs = np.array([str(l) for l in labels])
    n = e.shape[0]
    uniq = sorted(set(labs))
    if len(uniq) < 2 or n < k + 1:
        return 0.0
    D = squareform(pdist(e))
    np.fill_diagonal(D, np.inf)
    lis = np.zeros(n)
    for i in range(n):
        nn = np.argpartition(D[i], k)[:k]
        _, counts = np.unique(labs[nn], return_counts=True)
        p = counts / counts.sum()
        lis[i] = 1.0 / np.sum(p ** 2)
    return float(np.clip(np.mean(lis) / len(uniq), 0.0, 1.0))


def _asw_safe(emb, labels):
    e = np.asarray(emb)
    labs = np.array([str(l) for l in labels])
    if len(set(labs)) < 2 or e.shape[0] < 4:
        return 0.0
    if any(list(labs).count(b) < 2 for b in set(labs)):
        return 0.0
    try:
        return float(silhouette_score(e, labs, metric="euclidean"))
    except Exception:
        return 0.0


def _sps_continuous(emb, target, q=4):
    e = np.asarray(emb)
    t = np.asarray(target, dtype=float)
    keep = ~np.isnan(t)
    if keep.sum() < 4:
        return 0.0
    e = e[keep]
    t = t[keep]
    try:
        bins = pd.qcut(pd.Series(t), q=q, labels=False, duplicates="drop").values
    except Exception:
        return 0.0
    if len(set(bins)) < 2:
        return 0.0
    D = squareform(pdist(e))
    n = e.shape[0]
    iu = np.triu_indices(n, k=1)
    eq = bins[iu[0]] == bins[iu[1]]
    if eq.sum() < 1 or (~eq).sum() < 1:
        return 0.0
    w = D[iu][eq].mean()
    b = D[iu][~eq].mean()
    return float(b / max(w, 1e-9))


def _cv_knn_neg_mae(emb, target, k=3, n_splits=5):
    e = np.asarray(emb)
    t = np.asarray(target, dtype=float)
    keep = ~np.isnan(t)
    if keep.sum() < n_splits + k:
        return 0.0
    e = e[keep]
    t = t[keep]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_sum = 0.0
    n_sum = 0
    for tr, te in kf.split(e):
        D = np.linalg.norm(e[te][:, None] - e[tr][None, :], axis=-1)
        idx = np.argpartition(D, min(k, D.shape[1] - 1), axis=1)[:, :k]
        preds = t[tr][idx].mean(axis=1)
        mae_sum += float(np.sum(np.abs(preds - t[te])))
        n_sum += len(te)
    mae = mae_sum / max(n_sum, 1)
    std = float(np.std(t) + 1e-9)
    return -float(mae / std)


def _pseudotime_spearman(emb, target):
    e = np.asarray(emb)
    t = np.asarray(target, dtype=float)
    keep = ~np.isnan(t)
    if keep.sum() < 4:
        return 0.0
    e = e[keep]
    t = t[keep]
    pt = PCA(n_components=1, random_state=42).fit_transform(e).flatten()
    rho, _ = spearmanr(pt, t)
    return float(abs(rho)) if np.isfinite(rho) else 0.0


def _minmax(x, lo, hi):
    if hi <= lo:
        return 0.5
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


SCORING_BOUNDS = {
    "cca":            (0.0, 1.0),
    "ilisi_norm":     (0.0, 1.0),
    "sps":            (1.0, 3.0),
    "neg_asw_batch":  (-0.5, 0.5),
}


def make_scorer(name: str, meta: Dict, lam: float = 0.5) -> Callable[[np.ndarray], float]:
    """Closure: returns score_fn(emb_array) → float.

    Gates the underlying components based on data availability:
      - `multi_metric_proxy` only includes proxies whose required data is present.
      - If both batch and grouping are missing, the scorer returns 0 (caller
        should detect this and short-circuit).
    """
    grouping = meta.get("grouping")
    batch = meta.get("batch")
    has_grouping = bool(meta.get("has_grouping"))
    has_batch = bool(meta.get("has_batch"))

    if name == "cca":
        if not has_grouping:
            return lambda emb: 0.0
        return lambda emb: _cca_corr(emb, grouping)
    if name == "ilisi_batch":
        if not has_batch:
            return lambda emb: 0.0
        return lambda emb: _ilisi_norm(emb, batch)
    if name == "sps":
        if not has_grouping:
            return lambda emb: 0.0
        return lambda emb: _sps_continuous(emb, grouping)
    if name == "neg_asw_batch":
        if not has_batch:
            return lambda emb: 0.0
        return lambda emb: -_asw_safe(emb, batch)
    if name == "cv_knn_severity":
        if not has_grouping:
            return lambda emb: 0.0
        return lambda emb: _cv_knn_neg_mae(emb, grouping)
    if name == "pseudotime_spearman":
        if not has_grouping:
            return lambda emb: 0.0
        return lambda emb: _pseudotime_spearman(emb, grouping)

    if name == "sev_minus_batch":
        return lambda emb: (
            (_cca_corr(emb, grouping) if has_grouping else 0.0)
            - lam * (_asw_safe(emb, batch) if has_batch else 0.0)
        )

    if name in ("multi_metric_proxy", "auto"):
        components: List[Callable[[np.ndarray], float]] = []
        if has_grouping:
            components.append(lambda emb: _minmax(_cca_corr(emb, grouping), *SCORING_BOUNDS["cca"]))
            components.append(lambda emb: _minmax(_sps_continuous(emb, grouping), *SCORING_BOUNDS["sps"]))
        if has_batch:
            components.append(lambda emb: _minmax(_ilisi_norm(emb, batch), *SCORING_BOUNDS["ilisi_norm"]))
            components.append(lambda emb: _minmax(-_asw_safe(emb, batch), *SCORING_BOUNDS["neg_asw_batch"]))
        if not components:
            return lambda emb: 0.0

        def f(emb):
            vals = [c(emb) for c in components]
            return float(np.mean(vals))
        return f

    raise ValueError(f"unknown scoring: {name}")


# ============================================================ #
# Search strategies                                              #
# ============================================================ #
def search_grid(objective: Callable, alpha_grid: List[float]):
    trace = []
    best = None
    for a in alpha_grid:
        s = objective(a)
        trace.append((a, s))
        if best is None or s > best[1]:
            best = (a, s)
    return best[0], best[1], trace


def search_golden(objective: Callable, bounds=(0.1, 10.0), max_iter=12):
    phi = (1 + math.sqrt(5)) / 2
    a, b = bounds
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = objective(x1)
    f2 = objective(x2)
    trace = [(x1, f1), (x2, f2)]
    for _ in range(max_iter):
        if f1 > f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = objective(x1)
            trace.append((x1, f1))
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = objective(x2)
            trace.append((x2, f2))
    best = max(trace, key=lambda x: x[1])
    return best[0], best[1], trace


def _gp_ei(gp, X_grid, y_best, xi=0.01):
    mu, sigma = gp.predict(X_grid, return_std=True)
    with np.errstate(divide="warn"):
        imp = mu - y_best - xi
        Z = imp / np.where(sigma > 1e-12, sigma, 1e-12)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-12] = 0
    return ei


def search_bayesian(objective: Callable, bounds=(0.1, 10.0),
                     n_init=5, n_iter=10, seed=42):
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    X_init = np.linspace(lo, hi, n_init).reshape(-1, 1)
    trace = []
    for x in X_init:
        s = objective(float(x[0]))
        trace.append((float(x[0]), s))
    X = np.array([[a] for a, _ in trace])
    y = np.array([s for _, s in trace])
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1)))
    grid = np.linspace(lo, hi, 500).reshape(-1, 1)
    for _ in range(n_iter):
        try:
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                            n_restarts_optimizer=2,
                                            random_state=seed).fit(X, y)
            ei = _gp_ei(gp, grid, y.max())
            x_next = float(grid[int(np.argmax(ei))][0])
        except Exception:
            x_next = float(rng.uniform(lo, hi))
        if any(abs(x_next - x) < 1e-3 for x, _ in trace):
            x_next = float(rng.uniform(lo, hi))
        s = objective(x_next)
        trace.append((x_next, s))
        X = np.vstack([X, [[x_next]]])
        y = np.append(y, s)
    best = max(trace, key=lambda x: x[1])
    return best[0], best[1], trace


SEARCH_FUNCS = {
    "grid":            lambda obj, b: search_grid(obj, [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]),
    "golden_section":  lambda obj, b: search_golden(obj, bounds=b, max_iter=12),
    "bayesian":        lambda obj, b: search_bayesian(obj, bounds=b, n_init=5, n_iter=10),
}


# ============================================================ #
# Public entry                                                   #
# ============================================================ #
def run_autotune(
    adata: AnnData,
    output_dir: str,
    *,
    sample_col: str = "sample",
    celltype_col: str = "cell_type",
    cluster_emb_key: str = "X_pca_harmony",
    cmd_emb_key: Optional[str] = None,
    modality_col: Optional[str] = None,
    batch_col: Optional[Union[str, List[str]]] = None,
    grouping_col: Optional[str] = None,
    medium_K: int = 120,
    fine_K: int = 300,
    cmd_dim: int = 8,
    pca_components: int = 10,
    batch_method: str = "harmony",
    scoring: str = "auto",
    search: str = "bayesian",
    scope: str = "alpha_only",
    alpha_bounds: Tuple[float, float] = DEFAULT_ALPHA_BOUNDS,
    seed: int = 42,
    save: bool = True,
    verbose: bool = True,
) -> Dict:
    """Run autotune and return the best params + final sample-AnnData."""
    t0 = time.time()
    primary_batch = batch_col[0] if isinstance(batch_col, (list, tuple)) and batch_col else batch_col
    if isinstance(primary_batch, list):
        primary_batch = primary_batch[0] if primary_batch else None

    blocks = build_blocks(
        adata, sample_col=sample_col, celltype_col=celltype_col,
        cluster_emb_key=cluster_emb_key, cmd_emb_key=cmd_emb_key,
        modality_col=modality_col, batch_col=primary_batch,
        grouping_col=grouping_col, medium_K=medium_K, fine_K=fine_K,
        cmd_dim=cmd_dim, seed=seed, verbose=verbose,
    )

    if not blocks["has_batch"] and not blocks["has_grouping"]:
        if verbose:
            print("[autotune] no batch and no grouping column → using fixed defaults; "
                  "no search performed.")
        weights = derive_weights(blocks["K_c"], blocks["K_med"], blocks["K_fine"],
                                   cmd_weight=0.60, n_blocks=4)
        final_emb = build_emb_from_blocks(
            [blocks["A1"], blocks["A2"], blocks["A3"], blocks["CMD"]],
            weights,
            unit_ids=blocks["unit_ids"],
            unit_groups=blocks["unit_groups"],
            unit_batches=blocks["unit_batches"],
            pca_components=pca_components, batch_method=batch_method,
            seed=seed, verbose=verbose,
        )
        return _finalize(adata, blocks, final_emb, weights,
                         best_params={"cmd_weight": 0.60},
                         best_score=float("nan"),
                         trace=[], search=search, scoring=scoring,
                         scope=scope, alpha_bounds=alpha_bounds,
                         pca_components=pca_components,
                         batch_method=batch_method,
                         output_dir=output_dir, save=save,
                         t_start=t0, verbose=verbose)

    if scope != "alpha_only":
        raise ValueError(
            f"only scope='alpha_only' is supported in this generalized port "
            f"(got '{scope}')")

    score_fn = make_scorer(scoring, blocks)

    def objective(alpha: float) -> float:
        weights = derive_weights(blocks["K_c"], blocks["K_med"], blocks["K_fine"],
                                   cmd_weight=alpha, n_blocks=4)
        emb_df = build_emb_from_blocks(
            [blocks["A1"], blocks["A2"], blocks["A3"], blocks["CMD"]],
            weights,
            unit_ids=blocks["unit_ids"],
            unit_groups=blocks["unit_groups"],
            unit_batches=blocks["unit_batches"],
            pca_components=pca_components, batch_method=batch_method,
            seed=seed, verbose=False,
        )
        return float(score_fn(emb_df.values))

    if search not in SEARCH_FUNCS:
        raise ValueError(f"unknown search '{search}' (choices: {list(SEARCH_FUNCS)})")
    if verbose:
        print(f"[autotune] search={search}  scoring={scoring}  "
              f"scope={scope}  bounds={alpha_bounds}")
    best_alpha, best_score, trace = SEARCH_FUNCS[search](objective, alpha_bounds)
    if verbose:
        print(f"[autotune] best cmd_weight={best_alpha:.4f}  score={best_score:.4f}  "
              f"({len(trace)} evals)")

    final_weights = derive_weights(blocks["K_c"], blocks["K_med"], blocks["K_fine"],
                                     cmd_weight=best_alpha, n_blocks=4)
    final_emb = build_emb_from_blocks(
        [blocks["A1"], blocks["A2"], blocks["A3"], blocks["CMD"]],
        final_weights,
        unit_ids=blocks["unit_ids"],
        unit_groups=blocks["unit_groups"],
        unit_batches=blocks["unit_batches"],
        pca_components=pca_components, batch_method=batch_method,
        seed=seed, verbose=verbose,
    )

    return _finalize(adata, blocks, final_emb, final_weights,
                     best_params={"cmd_weight": float(best_alpha)},
                     best_score=float(best_score),
                     trace=trace, search=search, scoring=scoring,
                     scope=scope, alpha_bounds=alpha_bounds,
                     pca_components=pca_components,
                     batch_method=batch_method,
                     output_dir=output_dir, save=save,
                     t_start=t0, verbose=verbose)


# Proxy-component descriptions for the human-readable report.
_PROXY_DESCRIPTIONS = {
    "cca":             ("supervised", "CCA(emb, grouping_col) — canonical correlation between embedding and grouping label"),
    "sps":             ("supervised", "SPS(emb, grouping_col) — between/within-quartile distance ratio"),
    "cv_knn_severity": ("supervised", "CV-kNN — 5-fold cross-validated MAE of k=3 kNN regression on grouping"),
    "pseudotime_spearman": ("supervised", "|Spearman(PC1, grouping)| — pseudotime alignment"),
    "ilisi_batch":     ("unsupervised", "iLISI(emb, batch) — k-NN batch mixing"),
    "neg_asw_batch":   ("unsupervised", "−ASW(emb, batch) — negative silhouette of batch labels"),
}


def _active_proxies(scoring: str, has_batch: bool, has_grouping: bool):
    """Return the list of proxy names the scorer actually evaluates."""
    if scoring in ("auto", "multi_metric_proxy"):
        names = []
        if has_grouping:
            names += ["cca", "sps"]
        if has_batch:
            names += ["ilisi_batch", "neg_asw_batch"]
        return names
    return [scoring] if scoring in _PROXY_DESCRIPTIONS else [scoring]


def _format_autotune_report(*, best_params, best_score, trace, weights,
                              blocks, search, scoring, scope, alpha_bounds,
                              pca_components, batch_method, elapsed_s):
    """Build the human-readable autotune_record.txt content."""
    has_batch = bool(blocks.get("has_batch"))
    has_grouping = bool(blocks.get("has_grouping"))
    n_units = int(blocks.get("n_units", 0))
    n_groups = len(set(blocks.get("unit_groups") or []))
    proxies = _active_proxies(scoring, has_batch, has_grouping)

    lines = []
    lines.append("Sample-embedding autotune — composition + CMD")
    lines.append("=" * 68)
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 68)
    lines.append(f"  search algorithm   : {search}")
    lines.append(f"  scoring strategy   : {scoring}")
    lines.append(f"  scope              : {scope}")
    lines.append(f"  α bounds           : [{alpha_bounds[0]:g}, {alpha_bounds[1]:g}]")
    lines.append(f"  PCA components     : {pca_components}")
    lines.append(f"  sample Harmony     : {batch_method}")
    lines.append(f"  number of units    : {n_units}")
    lines.append(f"  unique groups      : {n_groups}")
    lines.append(f"  has batch column   : {has_batch}")
    lines.append(f"  has grouping col   : {has_grouping}")
    lines.append("")
    lines.append("Block setup (inverse-variance weights from K values)")
    lines.append("-" * 68)
    lines.append(f"  K_c   (cell types) : {int(blocks['K_c'])}")
    lines.append(f"  K_med (k-means)    : {int(blocks['K_med'])}")
    lines.append(f"  K_fine (k-means)   : {int(blocks['K_fine'])}")
    lines.append(f"  cluster_emb_key    : {blocks.get('cluster_emb_key', '?')}")
    lines.append(f"  cmd_emb_key        : {blocks.get('cmd_emb_key', '?')}")
    lines.append("")
    lines.append("Active scoring proxies (gated by data availability)")
    lines.append("-" * 68)
    if not proxies:
        lines.append("  (none — no batch or grouping column available; defaults used)")
    else:
        for name in proxies:
            kind, desc = _PROXY_DESCRIPTIONS.get(name, ("?", name))
            lines.append(f"  - [{kind:<12s}] {name:<20s}  {desc}")
        if scoring in ("auto", "multi_metric_proxy"):
            lines.append("  ensemble: multi_metric_proxy = mean of the proxies above (each min-max scaled).")
    lines.append("")
    lines.append("Result")
    lines.append("-" * 68)
    for k, v in best_params.items():
        if isinstance(v, float):
            lines.append(f"  best {k:<14s}: {v:.6f}")
        else:
            lines.append(f"  best {k:<14s}: {v}")
    lines.append(f"  best score        : {best_score:.6f}")
    lines.append(f"  block weights     : "
                  + ", ".join(f"{x:.4f}" for x in weights)
                  + "  (A1, A2, A3, CMD)")
    lines.append(f"  total evaluations : {len(trace)}")
    lines.append(f"  wall time         : {elapsed_s:.2f} s")
    lines.append("")
    lines.append(f"Search trace ({len(trace)} evals, top 25 by score)")
    lines.append("-" * 68)
    lines.append(f"  {'rank':>4s}  {'α (cmd_weight)':>16s}  {'score':>10s}")
    sorted_trace = sorted(trace, key=lambda r: r[1], reverse=True)
    for i, (a, s) in enumerate(sorted_trace[:25], 1):
        marker = "  ★ best" if i == 1 else ""
        lines.append(f"  {i:>4d}  {a:>16.6f}  {s:>10.4f}{marker}")
    if len(sorted_trace) > 25:
        lines.append(f"  ... ({len(sorted_trace) - 25} more)")
    lines.append("")
    lines.append(f"Full chronological trace ({len(trace)} evals)")
    lines.append("-" * 68)
    lines.append(f"  {'step':>4s}  {'α (cmd_weight)':>16s}  {'score':>10s}")
    for i, (a, s) in enumerate(trace, 1):
        lines.append(f"  {i:>4d}  {a:>16.6f}  {s:>10.4f}")
    lines.append("")
    return "\n".join(lines)


def _finalize(adata, blocks, final_emb, weights, *,
               best_params, best_score, trace,
               search, scoring, scope, alpha_bounds,
               pca_components, batch_method,
               output_dir, save, t_start, verbose):
    """Write the autotuned embedding into the cell-level adata and persist artifacts."""
    elapsed = time.time() - t_start
    adata.uns["X_DR_sample"] = final_emb.copy()
    adata.uns["sample_embedding_params"] = {
        "best_params": best_params,
        "best_score": best_score,
        "search": search,
        "scoring": scoring,
        "scope": scope,
        "block_weights": list(map(float, weights)),
        "K_c": int(blocks["K_c"]),
        "K_med": int(blocks["K_med"]),
        "K_fine": int(blocks["K_fine"]),
        "cluster_emb_key": str(blocks.get("cluster_emb_key", "")),
        "cmd_emb_key": str(blocks.get("cmd_emb_key", "")),
        "n_evals": len(trace),
        "autotuned": True,
        "wall_time_s": float(elapsed),
    }

    if save:
        out_dir = os.path.join(output_dir, "sample_embedding")
        os.makedirs(out_dir, exist_ok=True)
        emb_csv = os.path.join(out_dir, "sample_embedding.csv")
        final_emb.to_csv(emb_csv)

        preprocessed_h5 = os.path.join(output_dir, "preprocess", "adata_preprocessed.h5ad")
        if os.path.exists(preprocessed_h5):
            try:
                sc.write(preprocessed_h5, adata)
            except Exception as exc:
                if verbose:
                    print(f"[autotune] WARNING: could not re-save "
                          f"{preprocessed_h5}: {exc}")

        report = _format_autotune_report(
            best_params=best_params, best_score=best_score, trace=trace,
            weights=weights, blocks=blocks, search=search, scoring=scoring,
            scope=scope, alpha_bounds=alpha_bounds,
            pca_components=pca_components, batch_method=batch_method,
            elapsed_s=elapsed,
        )
        report_path = os.path.join(out_dir, "autotune_record.txt")
        with open(report_path, "w") as f:
            f.write(report)
        if verbose:
            print(f"[autotune] wrote {emb_csv}")
            print(f"[autotune] wrote {report_path}")

    if verbose:
        print(f"[autotune] done in {elapsed:.2f}s")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "trace": trace,
        "block_weights": list(map(float, weights)),
        "adata": adata,
        "sample_embedding": final_emb,
    }
