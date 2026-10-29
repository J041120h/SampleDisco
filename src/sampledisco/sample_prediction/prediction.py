"""Supervised use of the sample embedding: phenotype prediction and label transfer.

Both functions take a one-row-per-sample AnnData (``build_sample_adata``), read the
first ``n_pcs`` components of ``uns['X_DR_sample']`` (``obsm`` as fallback) and the
labels from ``.obs``.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _embedding(pseudo_adata: AnnData, n_pcs: Optional[int]) -> np.ndarray:
    emb = pseudo_adata.uns.get("X_DR_sample")
    if isinstance(emb, pd.DataFrame):
        X = emb.loc[pseudo_adata.obs_names].to_numpy(dtype=float)
    elif "X_DR_sample" in pseudo_adata.obsm:
        X = np.asarray(pseudo_adata.obsm["X_DR_sample"], dtype=float)
    else:
        raise KeyError("'X_DR_sample' not found in .uns or .obsm; build the AnnData with build_sample_adata.")
    return X[:, :n_pcs]


def predict_phenotype(
    pseudo_adata: AnnData,
    label_col: str,
    *,
    group_col: Optional[str] = None,
    n_pcs: Optional[int] = None,
    task: str = "classification",
    n_splits: int = 5,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, Dict]:
    """Out-of-fold prediction of a sample phenotype from the first ``n_pcs`` embedding PCs.

    Standardized PCs feed an L2 logistic regression (``task="classification"``, binary or
    multiclass) or a ridge regression (``task="regression"``). The penalty (C in
    {0.01, 0.1, 1, 10}, or alpha in {0.1, 1, 10, 100}) is chosen by an inner 3-fold
    grouped CV inside each training split (scored by AUC, balanced accuracy or R²). The
    outer split is ``n_splits``-fold, stratified by class for classification and seeded by
    ``random_state``. With ``group_col`` (e.g. donor) every sample of a group falls in the
    same fold, outer and inner, so draws of one donor are never in training and test at
    once. Samples with a missing label are left out. This is the recipe of the paper's
    Sound Life trait prediction.

    Returns ``(predictions, metrics)``. ``predictions`` is indexed by sample with columns
    ``label``, ``predicted``, ``fold`` and, for classification, ``prob_<class>``.
    ``metrics`` holds ``n_samples``, ``n_groups``, ``n_pcs``, the penalty chosen in each
    fold (``C`` or ``alpha``) and ``auc`` (binary; the second class in sorted order is
    positive), ``accuracy`` and ``balanced_accuracy`` (classification) or ``r2``
    (regression), all computed on the out-of-fold predictions of every sample.
    """
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'.")
    obs = pseudo_adata.obs
    keep = obs[label_col].notna().to_numpy()
    X = _embedding(pseudo_adata, n_pcs)[keep]
    labels = obs[label_col].to_numpy()[keep]
    groups = (obs[group_col] if group_col else obs.index.to_series()).astype(str).to_numpy()[keep]

    if task == "classification":
        classes, y = np.unique(labels, return_inverse=True)
        model, param = LogisticRegression(max_iter=3000), "C"
        grid, scoring = [0.01, 0.1, 1, 10], "roc_auc" if len(classes) == 2 else "balanced_accuracy"
        outer = StratifiedGroupKFold(n_splits, shuffle=True, random_state=random_state).split(X, y, groups)
        P = np.zeros((len(y), len(classes)))
    else:
        y = labels.astype(float)
        model, param, grid, scoring = Ridge(), "alpha", [0.1, 1, 10, 100], "r2"
        codes = np.unique(groups, return_inverse=True)[1]
        shuffled = np.random.default_rng(random_state).permutation(codes.max() + 1)[codes]
        outer = GroupKFold(n_splits).split(X, y, shuffled)
        pred = np.zeros(len(y))

    fold, chosen = np.zeros(len(y), dtype=int), []
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    for i, (tr, te) in enumerate(outer):
        gs = GridSearchCV(pipe, {f"model__{param}": grid}, cv=GroupKFold(3), scoring=scoring)
        gs.fit(X[tr], y[tr], groups=groups[tr])
        chosen.append(gs.best_params_[f"model__{param}"])
        fold[te] = i
        if task == "classification":
            P[np.ix_(te, gs.classes_)] = gs.predict_proba(X[te])
        else:
            pred[te] = gs.predict(X[te])

    metrics = {"task": task, "n_samples": len(y), "n_groups": len(np.unique(groups)), "n_pcs": X.shape[1],
               param: chosen}
    out = pd.DataFrame({"label": labels}, index=obs.index[keep])
    if task == "classification":
        hard = P.argmax(1)
        out["predicted"] = classes[hard]
        for j, c in enumerate(classes):
            out[f"prob_{c}"] = P[:, j]
        if len(classes) == 2:
            metrics["auc"] = float(roc_auc_score(y, P[:, 1]))
        metrics["accuracy"] = float(accuracy_score(y, hard))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y, hard))
    else:
        out["predicted"] = pred
        metrics["r2"] = float(r2_score(y, pred))
    out["fold"] = fold
    return out.infer_objects(), metrics


def _knn_vote(D: np.ndarray, y: np.ndarray, k: int, weighted: bool) -> Tuple[np.ndarray, np.ndarray]:
    pred, conf = np.full(len(D), np.nan, dtype=object), np.full(len(D), np.nan)
    for i, d in enumerate(D):
        nn = np.argsort(d, kind="stable")[:k]
        nn = nn[np.isfinite(d[nn])]
        if nn.size == 0:
            continue
        w = 1.0 / np.maximum(d[nn], 1e-12) if weighted else np.ones(nn.size)
        votes = pd.Series(w).groupby(y[nn], sort=False).sum()  # labels in order of first (nearest) neighbour
        pred[i], conf[i] = votes.idxmax(), votes.max() / w.sum()
    return pred, conf


def transfer_labels(
    pseudo_adata: AnnData,
    label_col: str,
    *,
    query: Optional[Sequence[bool]] = None,
    n_neighbors: int = 5,
    weighted: bool = False,
    n_pcs: Optional[int] = None,
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Label query samples by a k-nearest-neighbour vote in the sample embedding.

    References are the samples with a label in ``label_col`` that are not in the boolean
    ``query`` mask; every other sample is a query (by default, the unlabelled samples).
    Each query takes the majority label of its ``n_neighbors`` nearest references
    (Euclidean distance on the first ``n_pcs`` PCs); with ``weighted=True`` each vote
    counts 1/distance. Ties go to the label of the nearest tied neighbour. The confidence
    is the (weighted) share of the votes won by the assigned label.

    To judge reliability, every reference is also labelled from the other references
    (leave-one-out). With ``group_col`` (a donor, or ``sample`` for multi-omics units) the
    references of the same group are left out as well, so a unit is never matched to
    another unit of the same donor or sample.

    Returns ``(table, metrics)``. ``table`` is indexed by sample with columns ``label``,
    ``query``, ``predicted`` (the transferred label for queries, the leave-one-out label for
    references) and ``confidence``. ``metrics`` holds ``loo_accuracy``,
    ``loo_balanced_accuracy``, ``n_reference``, ``n_query``, ``n_neighbors`` and ``n_pcs``.
    """
    obs = pseudo_adata.obs
    X = _embedding(pseudo_adata, n_pcs)
    labels = obs[label_col]
    ref = labels.notna().to_numpy()
    if query is not None:
        ref &= ~np.asarray(query, dtype=bool)
    y = labels.to_numpy()[ref]

    D = cdist(X, X[ref])
    g = (obs[group_col] if group_col else obs.index.to_series()).astype(str).to_numpy()[ref]
    D[ref] = np.where(g[:, None] == g[None, :], np.inf, D[ref])
    pred, conf = _knn_vote(D, y, n_neighbors, weighted)

    table = pd.DataFrame({"label": labels.to_numpy(), "query": ~ref, "predicted": pred, "confidence": conf},
                         index=obs.index).infer_objects()  # numeric labels back to a numeric dtype
    loo = table[ref & pd.notna(pred)]
    metrics = {"loo_accuracy": float(accuracy_score(loo["label"], loo["predicted"])),
               "loo_balanced_accuracy": float(balanced_accuracy_score(loo["label"], loo["predicted"])),
               "n_reference": int(ref.sum()), "n_query": int((~ref).sum()),
               "n_neighbors": n_neighbors, "n_pcs": X.shape[1]}
    return table, metrics
