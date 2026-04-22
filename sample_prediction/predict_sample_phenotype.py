import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from typing import Optional, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _get_feature_matrix(pseudo_adata: AnnData, feature_source: str):
    """Return (X ndarray, feature_names list) for the requested source."""
    if feature_source == "expression":
        if "X_DR_expression" not in pseudo_adata.obsm:
            raise KeyError("X_DR_expression not found in pseudo_adata.obsm")
        X = pseudo_adata.obsm["X_DR_expression"]
        return X, [f"Expr_PC{i+1}" for i in range(X.shape[1])]

    if feature_source == "proportion":
        if "X_DR_proportion" not in pseudo_adata.obsm:
            raise KeyError("X_DR_proportion not found in pseudo_adata.obsm")
        X = pseudo_adata.obsm["X_DR_proportion"]
        return X, [f"Prop_PC{i+1}" for i in range(X.shape[1])]

    if feature_source in ("cluster_expression", "cluster_proportion"):
        suffix = feature_source.split("_")[1]
        col = f"cluster_{suffix}_kmeans"
        if col not in pseudo_adata.obs.columns:
            raise KeyError(f"'{col}' not in pseudo_adata.obs — run sample clustering first.")
        dummies = pd.get_dummies(pseudo_adata.obs[col].astype(str))
        return dummies.values.astype(float), list(dummies.columns)

    if feature_source in ("pseudotime_expression", "pseudotime_proportion"):
        suffix = feature_source.split("_")[1]
        col = f"pseudotime_{suffix}"
        if col not in pseudo_adata.obs.columns:
            raise KeyError(f"'{col}' not in pseudo_adata.obs — run trajectory analysis first.")
        X = pseudo_adata.obs[col].values.reshape(-1, 1).astype(float)
        return X, [col]

    raise ValueError(
        f"Unknown feature_source '{feature_source}'. Choose from: "
        "expression, proportion, cluster_expression, cluster_proportion, "
        "pseudotime_expression, pseudotime_proportion"
    )


# ---------------------------------------------------------------------------
# Task / model helpers
# ---------------------------------------------------------------------------

def _detect_task_type(y: np.ndarray) -> str:
    if pd.api.types.is_numeric_dtype(y) and len(np.unique(y)) > 10:
        return "regression"
    return "classification"


def _build_model(task_type: str, random_state: int = 42):
    from sklearn.linear_model import Ridge, LogisticRegression
    if task_type == "regression":
        return Ridge(alpha=1.0)
    return LogisticRegression(max_iter=1000, random_state=random_state)


def _get_cv_splitter(n_samples: int, task_type: str, cv: str = "auto"):
    from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold

    if cv == "auto":
        if n_samples < 30:
            return LeaveOneOut(), "LOOCV"
        k = 5
    else:
        k = int(cv)

    if task_type == "classification":
        return StratifiedKFold(n_splits=k, shuffle=True, random_state=42), f"{k}-fold stratified"
    return KFold(n_splits=k, shuffle=True, random_state=42), f"{k}-fold"


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _cross_validate(estimator, X: np.ndarray, y: np.ndarray, cv_splitter, task_type: str):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import r2_score, accuracy_score

    le = None
    y_fit = y.copy()
    if task_type == "classification":
        le = LabelEncoder()
        y_fit = le.fit_transform(y.astype(str))
    else:
        y_fit = y.astype(float)

    y_pred_cv = np.full(len(y), np.nan, dtype=object if task_type == "classification" else float)
    fold_scores = []

    for train_idx, test_idx in cv_splitter.split(X, y_fit):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        est = copy.deepcopy(estimator)
        est.fit(X_tr, y_fit[train_idx])

        if task_type == "regression":
            preds = est.predict(X_te)
            y_pred_cv[test_idx] = preds
            if len(test_idx) > 1:
                fold_scores.append(r2_score(y_fit[test_idx], preds))
        else:
            preds = est.predict(X_te)
            y_pred_cv[test_idx] = le.inverse_transform(preds)
            fold_scores.append(accuracy_score(y_fit[test_idx], preds))

    return y_pred_cv, fold_scores, le


def _feature_importance(estimator, X: np.ndarray, y: np.ndarray,
                         feature_names: list, task_type: str) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    y_fit = y.copy()
    if task_type == "classification":
        y_fit = LabelEncoder().fit_transform(y.astype(str))
    else:
        y_fit = y.astype(float)

    est = copy.deepcopy(estimator)
    est.fit(X_s, y_fit)

    coefs = est.coef_
    imp = np.abs(coefs.mean(axis=0) if coefs.ndim > 1 else coefs)

    return (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

_SKIP_IMP = {"cluster_expression", "cluster_proportion", "pseudotime_expression", "pseudotime_proportion"}


def _save_regression_plots(y_true, y_pred_cv, feature_imp_df,
                            feature_source, target_col, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    from sklearn.metrics import r2_score

    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred_cv, dtype=float)
    valid = ~np.isnan(yp)

    # predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt[valid], yp[valid], alpha=0.7, edgecolors="k", linewidths=0.5)
    lo, hi = min(yt[valid].min(), yp[valid].min()), max(yt[valid].max(), yp[valid].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    r2 = r2_score(yt[valid], yp[valid]) if valid.sum() > 1 else np.nan
    ax.set_xlabel(f"Actual {target_col}")
    ax.set_ylabel(f"Predicted {target_col}")
    ax.set_title(f"Predicted vs Actual (CV R²={r2:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "predicted_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # residuals
    resid = yt[valid] - yp[valid]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(yp[valid], resid, alpha=0.7, edgecolors="k", linewidths=0.5)
    ax.axhline(0, color="r", linestyle="--", lw=1.5)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "residuals.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # feature importance
    if feature_source not in _SKIP_IMP and feature_imp_df is not None and len(feature_imp_df):
        top = feature_imp_df.head(20)
        fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
        ax.barh(top["feature"][::-1], top["importance"][::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Top Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "feature_importance_top20.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # pseudotime scatter
    if "pseudotime" in feature_source and valid.sum() > 2:
        from scipy.stats import pearsonr
        r, p = pearsonr(yt[valid], yp[valid])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(yt[valid], yp[valid], alpha=0.7)
        ax.set_xlabel(f"Actual {target_col}")
        ax.set_ylabel(f"Predicted {target_col}")
        ax.set_title(f"Pseudotime prediction (r={r:.3f}, p={p:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "pseudotime_vs_target.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # cluster boxplot
    if "cluster" in feature_source:
        fig, ax = plt.subplots(figsize=(7, 4))
        df_b = pd.DataFrame({"cluster": np.array(y_true)[valid], "value": yt[valid]})
        sns.boxplot(data=df_b, x="cluster", y="value", ax=ax)
        ax.set_title(f"{target_col} by Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "target_by_cluster_boxplot.png"), dpi=150, bbox_inches="tight")
        plt.close()


def _save_classification_plots(y_true, y_pred_cv, feature_imp_df,
                                feature_source, target_col, plots_dir, classes):
    os.makedirs(plots_dir, exist_ok=True)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    valid = ~pd.isna(pd.Series(y_pred_cv))
    yt = np.array(y_true)[valid].astype(str)
    yp = np.array(y_pred_cv)[valid].astype(str)
    class_labels = [str(c) for c in classes]

    # confusion matrix
    cm = confusion_matrix(yt, yp, labels=class_labels)
    fig, ax = plt.subplots(figsize=(max(4, len(class_labels)), max(3, len(class_labels))))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {target_col}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # feature importance
    if feature_source not in _SKIP_IMP and feature_imp_df is not None and len(feature_imp_df):
        top = feature_imp_df.head(20)
        fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
        ax.barh(top["feature"][::-1], top["importance"][::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Top Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "feature_importance_top20.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # cluster composition
    if "cluster" in feature_source:
        df_cc = pd.crosstab(pd.Series(yt, name="cluster"), pd.Series(yp, name=target_col))
        fig, ax = plt.subplots(figsize=(max(6, df_cc.shape[1] * 0.8), 4))
        df_cc.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"Class Distribution by Cluster — {target_col}")
        ax.set_xlabel("Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "class_composition_by_cluster.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # pseudotime density
    if "pseudotime" in feature_source:
        fig, ax = plt.subplots(figsize=(7, 4))
        for cls in class_labels:
            mask = yt == cls
            if mask.sum() > 1:
                sns.kdeplot(yp[mask].astype(float), ax=ax, label=cls, fill=True, alpha=0.3)
        ax.set_xlabel("Pseudotime prediction")
        ax.set_title(f"Target Density by Pseudotime — {target_col}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "target_density_by_pseudotime.png"), dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def predict_sample_phenotype(
    pseudo_adata: AnnData,
    target_col: str,
    feature_source: str = "expression",
    task_type: str = "auto",
    cv: str = "auto",
    n_permutations: int = 0,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Predict a sample-level phenotype from one of the pipeline's sample-level embeddings.

    Uses Ridge regression for regression targets and Logistic Regression for
    classification targets.

    Parameters
    ----------
    pseudo_adata : AnnData
        Sample-level AnnData. Requires X_DR_expression / X_DR_proportion in obsm
        and target_col in obs.
    target_col : str
        Column in pseudo_adata.obs to predict.
    feature_source : str
        One of: expression (default), proportion, cluster_expression,
        cluster_proportion, pseudotime_expression, pseudotime_proportion.
    task_type : str
        "auto", "regression", or "classification".
    cv : str
        "auto" → LOOCV if n<30 else 5-fold. Or integer string for explicit k-fold.
    n_permutations : int
        If >0, run permutation test on mean CV score.
    output_dir : str, optional
        Outputs go to {output_dir}/prediction_{target}_{source}/.
    verbose : bool

    Returns
    -------
    dict with keys: cv_score_mean, cv_score_std, task_type, predictions_df,
                    feature_importance_df, perm_p, output_subdir.
    """
    if target_col not in pseudo_adata.obs.columns:
        raise KeyError(f"target_col '{target_col}' not in pseudo_adata.obs")

    X, feature_names = _get_feature_matrix(pseudo_adata, feature_source)
    y_raw = pseudo_adata.obs[target_col].values
    sample_ids = pseudo_adata.obs_names.tolist()

    valid_mask = ~pd.isna(pd.Series(y_raw))
    if valid_mask.sum() < 3:
        raise ValueError(
            f"Too few non-missing samples for '{target_col}' (n={valid_mask.sum()})"
        )
    X = X[valid_mask]
    y = y_raw[valid_mask]
    sample_ids = [s for s, m in zip(sample_ids, valid_mask) if m]
    n_samples = len(y)

    if task_type == "auto":
        task_type = _detect_task_type(y)

    if verbose:
        print(
            f"[Prediction] target={target_col}, source={feature_source}, "
            f"task={task_type}, n={n_samples}"
        )

    estimator = _build_model(task_type)
    cv_splitter, cv_label = _get_cv_splitter(n_samples, task_type, cv)

    if verbose:
        print(f"[Prediction] CV strategy: {cv_label}")

    y_pred_cv, fold_scores, le = _cross_validate(estimator, X, y, cv_splitter, task_type)
    mean_score = float(np.nanmean(fold_scores)) if fold_scores else np.nan
    std_score = float(np.nanstd(fold_scores)) if fold_scores else np.nan

    if verbose:
        metric = "R²" if task_type == "regression" else "Accuracy"
        print(f"[Prediction] CV {metric}: {mean_score:.4f} ± {std_score:.4f}")

    feature_imp_df = _feature_importance(estimator, X, y, feature_names, task_type)

    pred_df = pd.DataFrame({"sample_id": sample_ids, "y_true": y, "y_pred_cv": y_pred_cv})
    scores_df = pd.DataFrame({
        "fold": list(range(len(fold_scores))) + ["mean", "std"],
        "score": fold_scores + [mean_score, std_score],
    })

    # permutation test
    perm_df = None
    perm_p = np.nan
    null_scores = None
    if n_permutations > 0:
        if verbose:
            print(f"[Prediction] Running {n_permutations} permutations...")
        rng = np.random.default_rng(42)
        null_scores = []
        for _ in range(n_permutations):
            y_perm = rng.permutation(y)
            _, fs_perm, _ = _cross_validate(estimator, X, y_perm, cv_splitter, task_type)
            null_scores.append(float(np.nanmean(fs_perm)))
        null_scores = np.array(null_scores)
        perm_p = float(np.mean(null_scores >= mean_score))
        perm_df = pd.DataFrame({
            "null_cv_score": null_scores,
            "observed_score": mean_score,
            "p_value": perm_p,
        })
        if verbose:
            print(f"[Prediction] Permutation p-value: {perm_p:.4f}")

    result = {
        "cv_score_mean": mean_score,
        "cv_score_std": std_score,
        "task_type": task_type,
        "predictions_df": pred_df,
        "feature_importance_df": feature_imp_df,
        "perm_p": perm_p,
        "output_subdir": None,
    }

    if output_dir is not None:
        subdir = os.path.join(output_dir, f"prediction_{target_col}_{feature_source}")
        plots_dir = os.path.join(subdir, "plots")
        os.makedirs(subdir, exist_ok=True)

        scores_df.to_csv(os.path.join(subdir, "cv_scores.csv"), index=False)
        pred_df.to_csv(os.path.join(subdir, "predictions.csv"), index=False)
        feature_imp_df.to_csv(os.path.join(subdir, "feature_importance.csv"), index=False)
        if perm_df is not None:
            perm_df.to_csv(os.path.join(subdir, "permutation_null.csv"), index=False)

        try:
            if task_type == "regression":
                _save_regression_plots(
                    y, y_pred_cv, feature_imp_df, feature_source, target_col, plots_dir
                )
            else:
                classes = le.classes_ if le is not None else np.unique(y).astype(str)
                _save_classification_plots(
                    y, y_pred_cv, feature_imp_df, feature_source, target_col, plots_dir, classes
                )

            if null_scores is not None:
                os.makedirs(plots_dir, exist_ok=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(null_scores, bins=30, alpha=0.7, label="Null distribution")
                ax.axvline(mean_score, color="r", linestyle="--", lw=2,
                           label=f"Observed (p={perm_p:.3f})")
                ax.set_xlabel("CV Score")
                ax.set_ylabel("Count")
                ax.set_title(f"Permutation Test — {target_col}")
                ax.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plots_dir, "permutation_null_distribution.png"),
                    dpi=150, bbox_inches="tight"
                )
                plt.close()
        except Exception as e:
            print(f"[Prediction] Warning: plot failed: {e}")

        result["output_subdir"] = subdir
        if verbose:
            print(f"[Prediction] Results saved to: {subdir}")

    return result


# ---------------------------------------------------------------------------
# Cross-modality prediction
# ---------------------------------------------------------------------------

def predict_cross_modality(
    pseudo_adata: AnnData,
    target_col: str,
    train_modality: str,
    test_modality: str,
    modality_col: str = "modality",
    feature_source: str = "expression",
    task_type: str = "auto",
    integer_output: bool = False,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Train on samples from one modality and predict on samples from another, using a
    shared sample-level embedding (e.g. GLUE-aligned X_DR_expression).

    Uses Ridge for regression and Logistic Regression for classification.

    If integer_output=True, forces regression and rounds+clips predictions to
    the observed integer class range of the training target (useful when the
    target is ordinal, e.g. severity levels 1..4).
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (
        r2_score, accuracy_score, mean_absolute_error,
        confusion_matrix, ConfusionMatrixDisplay,
    )

    if modality_col not in pseudo_adata.obs.columns:
        raise KeyError(f"modality_col '{modality_col}' not in pseudo_adata.obs")
    if target_col not in pseudo_adata.obs.columns:
        raise KeyError(f"target_col '{target_col}' not in pseudo_adata.obs")

    X_all, feature_names = _get_feature_matrix(pseudo_adata, feature_source)
    modality = pseudo_adata.obs[modality_col].astype(str).values
    y_all = pseudo_adata.obs[target_col].values
    ids = np.array(pseudo_adata.obs_names)

    train_mask = (modality == str(train_modality)) & ~pd.isna(pd.Series(y_all))
    test_mask = (modality == str(test_modality)) & ~pd.isna(pd.Series(y_all))
    if train_mask.sum() < 3:
        raise ValueError(f"Too few training samples (n={train_mask.sum()}) for modality={train_modality}")
    if test_mask.sum() < 1:
        raise ValueError(f"No test samples for modality={test_modality}")

    X_tr_raw, X_te_raw = X_all[train_mask], X_all[test_mask]
    y_tr_raw, y_te_raw = y_all[train_mask], y_all[test_mask]
    ids_te = ids[test_mask]

    if integer_output:
        task_type = "regression"
    elif task_type == "auto":
        task_type = _detect_task_type(y_tr_raw)

    if verbose:
        mode = " [ordinal-integer]" if integer_output else ""
        print(
            f"[CrossModality] train={train_modality} (n={train_mask.sum()}) → "
            f"test={test_modality} (n={test_mask.sum()}), source={feature_source}, "
            f"task={task_type}{mode}"
        )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    le = None
    if task_type == "classification":
        le = LabelEncoder()
        y_tr = le.fit_transform(y_tr_raw.astype(str))
        test_classes = set(np.array(y_te_raw).astype(str))
        unseen = test_classes - set(le.classes_.astype(str))
        if unseen and verbose:
            print(f"[CrossModality] Warning: test classes not in train set: {unseen}")
    else:
        y_tr = y_tr_raw.astype(float)

    estimator = _build_model(task_type)
    estimator.fit(X_tr, y_tr)
    preds_raw = estimator.predict(X_te)

    y_pred_int = None
    r2 = mae = np.nan
    if task_type == "classification":
        y_pred = le.inverse_transform(preds_raw)
        y_true = np.array(y_te_raw).astype(str)
        y_pred = y_pred.astype(str)
        score = accuracy_score(y_true, y_pred)
        metric = "Accuracy"
    elif integer_output:
        y_true_float = np.array(y_te_raw).astype(float)
        y_min = int(np.floor(y_tr_raw.astype(float).min()))
        y_max = int(np.ceil(y_tr_raw.astype(float).max()))
        y_pred_int = np.clip(np.rint(preds_raw), y_min, y_max).astype(int)
        y_true_int = y_true_float.astype(int)
        score = accuracy_score(y_true_int.astype(str), y_pred_int.astype(str))
        r2 = r2_score(y_true_float, preds_raw) if len(y_true_float) > 1 else np.nan
        mae = mean_absolute_error(y_true_float, preds_raw)
        y_true = y_true_float
        y_pred = preds_raw
        metric = "Accuracy"
    else:
        y_pred = preds_raw
        y_true = np.array(y_te_raw).astype(float)
        score = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
        metric = "R²"

    if verbose:
        if integer_output:
            print(
                f"[CrossModality] Test Accuracy (integerized): {score:.4f} | "
                f"R²={r2:.4f} | MAE={mae:.4f}"
            )
        else:
            print(f"[CrossModality] Test {metric}: {score:.4f}")

    pred_df = pd.DataFrame({"sample_id": ids_te, "y_true": y_true, "y_pred": y_pred})
    if y_pred_int is not None:
        pred_df["y_pred_int"] = y_pred_int
    feature_imp_df = _feature_importance(estimator, X_tr_raw, y_tr_raw, feature_names, task_type)

    result = {
        "score": float(score) if not np.isnan(score) else np.nan,
        "metric": metric,
        "task_type": task_type,
        "predictions_df": pred_df,
        "feature_importance_df": feature_imp_df,
        "r2": float(r2) if not np.isnan(r2) else np.nan,
        "mae": float(mae) if not np.isnan(mae) else np.nan,
        "output_subdir": None,
    }

    if output_dir is not None:
        suffix = "_ordinal" if integer_output else ""
        subdir = os.path.join(
            output_dir,
            f"cross_{train_modality}_to_{test_modality}_{target_col}_{feature_source}{suffix}",
        )
        plots_dir = os.path.join(subdir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        pred_df.to_csv(os.path.join(subdir, "predictions.csv"), index=False)
        feature_imp_df.to_csv(os.path.join(subdir, "feature_importance.csv"), index=False)
        if integer_output:
            pd.DataFrame({"accuracy": [score], "r2": [r2], "mae": [mae],
                          "n_train": [int(train_mask.sum())], "n_test": [int(test_mask.sum())]}
                         ).to_csv(os.path.join(subdir, "score.csv"), index=False)
        else:
            pd.DataFrame({"metric": [metric], "score": [score],
                          "n_train": [int(train_mask.sum())], "n_test": [int(test_mask.sum())]}
                         ).to_csv(os.path.join(subdir, "score.csv"), index=False)

        try:
            if task_type == "classification":
                class_labels = [str(c) for c in le.classes_]
                cm = confusion_matrix(y_true, y_pred, labels=class_labels)
                fig, ax = plt.subplots(figsize=(max(4, len(class_labels)), max(3, len(class_labels))))
                ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot(ax=ax, colorbar=False)
                ax.set_title(f"{train_modality}→{test_modality}: {target_col} (acc={score:.3f})")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
                plt.close()
            else:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5)
                lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
                ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
                ax.set_xlabel(f"Actual {target_col}")
                ax.set_ylabel(f"Predicted {target_col}")
                title_score = f"acc={score:.3f}, R²={r2:.3f}" if integer_output else f"R²={score:.3f}"
                ax.set_title(f"{train_modality}→{test_modality} ({title_score})")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "predicted_vs_actual.png"), dpi=150, bbox_inches="tight")
                plt.close()

                if integer_output:
                    class_labels = sorted({int(v) for v in np.unique(np.concatenate([y_true.astype(int), y_pred_int]))})
                    class_labels = [str(c) for c in class_labels]
                    cm = confusion_matrix(y_true.astype(int).astype(str), y_pred_int.astype(str),
                                          labels=class_labels)
                    fig, ax = plt.subplots(figsize=(max(4, len(class_labels)), max(3, len(class_labels))))
                    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot(
                        ax=ax, colorbar=False
                    )
                    ax.set_title(
                        f"{train_modality}→{test_modality}: {target_col} (acc={score:.3f}, integerized)"
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
                    plt.close()

            if feature_source not in _SKIP_IMP and len(feature_imp_df):
                top = feature_imp_df.head(20)
                fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
                ax.barh(top["feature"][::-1], top["importance"][::-1])
                ax.set_xlabel("Importance")
                ax.set_title("Top Feature Importances")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "feature_importance_top20.png"), dpi=150, bbox_inches="tight")
                plt.close()
        except Exception as e:
            print(f"[CrossModality] Warning: plot failed: {e}")

        result["output_subdir"] = subdir
        if verbose:
            print(f"[CrossModality] Results saved to: {subdir}")

    return result
