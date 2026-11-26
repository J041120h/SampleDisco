"""Trajectory-DGE variance-ratio gene filter, minimum cells per group, zero handling and the
batch covariate in the GAM. Synthetic data, CPU only.

Run: python tests/test_trajdge_filter.py
"""
import os
import tempfile
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import sampledisco.sample_trajectory.trajectory_diff_gene as tdg
from sampledisco.sample_trajectory.trajectory_diff_gene import (
    _build_sample_pseudobulk, calculate_effect_size_and_direction, fit_gam_models_for_genes,
    prepare_gam_input_data_improved, fit_linear_models_for_genes, trajectory_test_features)
from sampledisco.sample_trajectory.trajectory_DGE_visualization import _adjusted_curve

N_SAMPLES, N_GENES = 16, 40


def _cells(seed=0, small=(), scale=1.0, batches=True, batch_of=lambda si: "b1" if si < 8 else "b2"):
    """16 samples x 2 cell types x 30 cells. Gene g0 carries extra between-sample variance."""
    rng = np.random.default_rng(seed)
    rows, X = [], []
    shift = rng.normal(0, 1.0, N_SAMPLES)
    for si in range(N_SAMPLES):
        for ct in ("T", "B"):
            n = 4 if (f"S{si}", ct) in small else 30
            x = rng.normal(2.0, 1.0, size=(n, N_GENES))
            x[:, 0] += shift[si]
            X.append(x * scale)
            rows += [(f"S{si}", ct, batch_of(si))] * n
    obs = pd.DataFrame(rows, columns=["sample", "cell_type", "batch"])
    obs.index = [f"c{i}" for i in range(len(obs))]
    if not batches:
        obs = obs.drop(columns="batch")
    return ad.AnnData(np.vstack(X).astype(np.float32), obs=obs,
                      var=pd.DataFrame(index=[f"g{j}" for j in range(N_GENES)]))


def _build(a, **kw):
    return _build_sample_pseudobulk(a, "sample", "cell_type", batch_col="batch" if "batch" in a.obs else None, **kw)


def _unbalanced(si):
    return "b1" if si < 6 else "b2" if si < 11 else "b3" if si < 15 else "b4"      # sizes 6, 5, 4, 1


def _expected_ratio(a):
    """Independent hat-matrix computation of the variance ratio for cell type T."""
    T = a[(a.obs["cell_type"] == "T").values]
    names = sorted(T.obs["sample"].unique())
    M = np.array([T[T.obs["sample"] == s].X.mean(0) for s in names], dtype=np.float64)
    V = np.array([T[T.obs["sample"] == s].X.var(0, ddof=1) for s in names], dtype=np.float64)
    b = pd.Series([_unbalanced(int(s[1:])) for s in names])
    D = np.column_stack([np.ones(len(names))] + [(b == k).values.astype(float) for k in sorted(b.unique())[1:]])
    H = D @ np.linalg.pinv(D)
    dof = len(names) - np.linalg.matrix_rank(D)
    total = (((np.eye(len(names)) - H) @ M) ** 2).sum(0) / dof
    tech = ((1 - np.diag(H))[:, None] * V / 30).sum(0) / dof
    return total / tech


def test_ratio_filter_keeps_signal_gene_and_rejects_noise():
    pb = _build(_cells())
    assert "T - g0" in pb.var_names and "B - g0" in pb.var_names
    assert sum(n.startswith("T - ") for n in pb.var_names) <= 4          # noise genes pass at ~1%


def test_ratio_filter_is_scale_invariant():
    a, b = _build(_cells()), _build(_cells(scale=7.0))
    assert list(a.var_names) == list(b.var_names)


def test_filter_matches_hat_matrix_with_unbalanced_batches_and_a_single_sample_batch():
    a = _cells(batch_of=_unbalanced)
    ratio = _expected_ratio(a)
    for thr in (1.0, 2.0, 3.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pb = _build(a, min_variance_ratio=thr)
        got = {n for n in pb.var_names if n.startswith("T - ")}
        assert got == {f"T - g{j}" for j in np.flatnonzero(ratio >= thr)}, thr


def test_null_calibration_pure_noise_passes_at_the_expected_rate():
    at1, at2, n = 0, 0, 0
    for seed in range(3):
        a = _cells(seed=seed, batch_of=_unbalanced)
        for thr in (1.0, 2.0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pb = _build(a, min_variance_ratio=thr)
            k = sum(1 for v in pb.var_names if v.startswith("T - ") and v != "T - g0")
            at1, at2 = (at1 + k, at2) if thr == 1.0 else (at1, at2 + k)
        n += N_GENES - 1
    assert 0.3 < at1 / n < 0.65 and at2 / n < 0.06, (at1 / n, at2 / n)


def test_small_groups_are_nan_and_do_not_change_the_other_groups():
    a = _cells(small={("S3", "T"), ("S12", "T")})
    pb = _build(a)
    col = pb.var_names.get_loc("T - g0")
    rows = pb.obs_names.get_indexer(["S3", "S12"])
    assert np.isnan(pb.X[rows, col]).all() and np.isfinite(np.delete(pb.X[:, col], rows)).all()
    gone = a.obs["sample"].isin(["S3", "S12"]) & (a.obs["cell_type"] == "T")
    pb_drop = _build(a[~gone.values].copy())
    assert list(pb.var_names) == list(pb_drop.var_names)
    np.testing.assert_allclose(pb.X, pb_drop.X, atol=1e-5)


def test_empty_group_is_nan_not_zero():
    a = _cells()
    a = a[~((a.obs["sample"] == "S5") & (a.obs["cell_type"] == "B")).values].copy()
    pb = _build(a)
    assert np.isnan(pb[["S5"], [i for i in pb.var_names if i.startswith("B - ")]].X).all()


def test_single_sample_batch_is_kept_without_warning():
    a = _cells(batch_of=_unbalanced)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        pb = _build(a, min_variance_ratio=0)
    assert not any("single" in str(w.message) for w in rec)
    assert np.isfinite(pb.X).all()


def test_pseudobulk_values_are_not_batch_corrected():
    a = _cells()
    corrected = _build(a, min_variance_ratio=0)
    plain = _build_sample_pseudobulk(a, "sample", "cell_type", batch_col=None, min_variance_ratio=0)
    np.testing.assert_allclose(corrected.X, plain.X)


def test_celltype_with_no_degrees_of_freedom_is_skipped_with_warning():
    a = _cells(batch_of=lambda si: f"b{si}")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            _build(a)
        except RuntimeError:
            pass
    assert sum("skipped" in str(w.message) for w in rec) == 2


def test_sparse_input_equals_dense_input():
    a = _cells()
    b = a.copy()
    b.X = sp.csr_matrix(a.X)
    x, y = _build(a), _build(b)
    assert list(x.var_names) == list(y.var_names)
    np.testing.assert_allclose(x.X, y.X, atol=1e-5)


def test_no_batch_and_one_batch_agree():
    one = _cells()
    one.obs["batch"] = "b1"
    r_none, r_one = _build(_cells(batches=False)), _build(one)
    assert list(r_none.var_names) == list(r_one.var_names)
    np.testing.assert_allclose(r_none.X, r_one.X)


def test_min_cells_below_two_rejected():
    try:
        _build(_cells(), min_cells_per_group=1)
    except ValueError:
        return
    raise AssertionError("min_cells_per_group=1 accepted")


def test_tested_set_is_identical_when_pseudotime_is_permuted():
    a = _cells()
    pt = pd.Series(np.linspace(0, 1, N_SAMPLES), index=[f"S{i}" for i in range(N_SAMPLES)])
    genes = []
    for ptime in (pt, pd.Series(np.random.default_rng(1).permutation(pt.values), index=pt.index)):
        with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = tdg.run_trajectory_gam_differential_gene_analysis(
                a, ptime.to_dict(), batch_col="batch", n_jobs=1, output_dir=d,
                generate_visualizations=False, verbose=False)
        genes.append(set(res["gene"]))
    assert genes[0] == genes[1] and "T - g0" in genes[0]


def _fit_recording_n(X, Y):
    seen, real = [], tdg.LinearGAM

    class _Rec(real):
        def fit(self, Xf, yf, *a, **k):
            seen.append(len(yf))
            return super().fit(Xf, yf, *a, **k)
    tdg.LinearGAM = _Rec
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res, models = fit_gam_models_for_genes(X, Y, ["g"], n_jobs=1)
    finally:
        tdg.LinearGAM = real
    return res, models, seen


def test_true_zero_is_fitted_and_nan_is_not():
    rng = np.random.default_rng(3)
    idx = [f"s{i}" for i in range(20)]
    pt = np.linspace(0, 1, 20)
    y = 1 + np.sin(3 * pt) + rng.normal(0, 0.1, 20)
    y[[2, 11]] = 0.0
    X = pd.DataFrame({"pseudotime": pt}, index=idx)
    Y = pd.DataFrame({"g": y}, index=idx)
    assert _fit_recording_n(X, Y)[2] == [20]
    Y.iloc[[0, 5], 0] = np.nan
    assert _fit_recording_n(X, Y)[2] == [18]


def test_effect_size_uses_finite_samples_and_keeps_true_zeros():
    rng = np.random.default_rng(4)
    idx = [f"s{i}" for i in range(30)]
    pt = np.linspace(0, 1, 30)
    y = 1 + 3 * pt + rng.normal(0, 0.2, 30)
    y[[3, 9]] = 0.0
    y[[0, 20]] = np.nan
    X = pd.DataFrame({"pseudotime": pt}, index=idx)
    Y = pd.DataFrame({"g": y}, index=idx)
    _, models, seen = _fit_recording_n(X, Y)
    es = calculate_effect_size_and_direction(X, Y, models, ["g"])
    m = np.isfinite(y)
    pred = models["g"].predict(X.values[m])
    df_e = m.sum() - models["g"].statistics_["edof"]
    expect = (pred.max() - pred.min()) / np.sqrt(((y[m] - pred) ** 2).sum() / df_e)
    assert seen == [28]
    np.testing.assert_allclose(es["effect_size"].iloc[0], expect)
    assert es["regulation"].iloc[0] == "UP"


def _batch_design(n=24, seed=0, confounded=True):
    """Pseudotime strongly tied to a 3-level batch; genes carry a batch shift and no pseudotime effect."""
    rng = np.random.default_rng(seed)
    batch = np.repeat(["b1", "b2", "b3"], n // 3)
    pt = (np.repeat([0.0, 1.0, 2.0], n // 3) if confounded else 0) + rng.uniform(0, 1, n)
    idx = [f"s{i}" for i in range(n)]
    d = pd.get_dummies(pd.Series(batch, index=idx), drop_first=True).astype(float)
    Y = pd.DataFrame(np.outer(d.sum(axis=1) * 3 + d["b3"], np.ones(60)) + rng.normal(0, 1, (n, 60)),
                     index=idx, columns=[f"g{j}" for j in range(60)])
    return pd.DataFrame({"pseudotime": pt}, index=idx), d, Y


def test_batch_covariate_removes_batch_driven_false_positives():
    X, d, Y = _batch_design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bare, _ = fit_gam_models_for_genes(X, Y, list(Y.columns), n_jobs=1)
        adj, _ = fit_gam_models_for_genes(pd.concat([X, d], axis=1), Y, list(Y.columns), n_jobs=1,
                                          unpenalised_columns=list(d.columns))
    assert (bare["pval"] < 0.05).mean() > 0.8
    assert (adj["pval"] < 0.05).mean() < 0.15            # pyGAM p-values are mildly liberal


def test_unpenalised_batch_contrast_is_recovered_exactly():
    X, d, _ = _batch_design(seed=1, confounded=False)
    y = 1.0 + 2.0 * d["b2"].to_numpy() - 3.0 * d["b3"].to_numpy()
    XX = pd.concat([X, d], axis=1)
    _, models = fit_gam_models_for_genes(XX, pd.DataFrame({"g": y}, index=X.index), ["g"], n_jobs=1,
                                         unpenalised_columns=list(d.columns))
    gam = models["g"]
    for j, truth in ((1, 2.0), (2, -3.0)):
        on = XX.to_numpy().copy()
        off = on.copy()
        on[:, j] = 1.0
        off[:, j] = 0.0
        assert abs((gam.partial_dependence(term=j, X=on) - gam.partial_dependence(term=j, X=off)).mean() - truth) < 1e-6


def test_effect_size_and_direction_ignore_a_pure_batch_shift():
    rng = np.random.default_rng(5)
    n = 24
    idx = [f"s{i}" for i in range(n)]
    batch = pd.Series(np.tile(["b1", "b2"], n // 2), index=idx)
    pt = np.linspace(0, 1, n)
    base = -2 * pt + rng.normal(0, 0.1, n)
    d = pd.get_dummies(batch, drop_first=True).astype(float)
    X = pd.concat([pd.DataFrame({"pseudotime": pt}, index=idx), d], axis=1)
    out = []
    for shift in (0.0, 8.0):
        Y = pd.DataFrame({"g": base + shift * d["b2"]}, index=idx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, models = fit_gam_models_for_genes(X, Y, ["g"], n_jobs=1, unpenalised_columns=list(d.columns))
        out.append(calculate_effect_size_and_direction(X, Y, models, ["g"]).iloc[0])
    assert out[0]["regulation"] == out[1]["regulation"] == "DOWN"
    np.testing.assert_allclose(out[0]["effect_size"], out[1]["effect_size"], rtol=1e-6)


def test_covariate_design_is_float_and_pipeline_runs_with_batch_and_covariate():
    a = _cells()
    a.obs["sex"] = a.obs["sample"].map(lambda s: "F" if int(s[1:]) % 2 else "M")
    pt = dict(zip([f"S{i}" for i in range(N_SAMPLES)], np.random.default_rng(0).normal(0, 1.0, N_SAMPLES)))   # g0's sample shift
    pb = _build(a)
    pb.obs["sex"] = pb.obs["sex"].astype(str)
    X, Y, _ = prepare_gam_input_data_improved(pb, pt, covariate_columns=["sex", "batch"], unpenalised_columns=["batch"])
    assert (X.dtypes == float).all() and np.isfinite(X.to_numpy()).all()
    assert X.attrs["unpenalised"] == ["batch_b2"]
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = tdg.run_trajectory_gam_differential_gene_analysis(
            a, pt, batch_col="batch", covariate_columns=["sex", "batch"], n_jobs=1, output_dir=d,
            generate_visualizations=False, verbose=False, top_n_genes=5)
    top = res[res["gene"].str.endswith(" - g0")]
    assert (top["fdr"] < 0.05).all() and top["effect_size"].notna().all() and (top["regulation"] == "UP").all()


def test_adjusted_curve_equals_predict_without_covariates_and_covariate_mean_with():
    rng = np.random.default_rng(6)
    X, d, _ = _batch_design(seed=2, confounded=False)
    y = np.sin(3 * X["pseudotime"].to_numpy()) + 2 * d["b2"].to_numpy() + rng.normal(0, 0.1, len(X))
    grid = np.linspace(X["pseudotime"].min(), X["pseudotime"].max(), 15)
    _, m0 = fit_gam_models_for_genes(X, pd.DataFrame({"g": y}, index=X.index), ["g"], n_jobs=1)
    np.testing.assert_allclose(_adjusted_curve(m0["g"], grid, X, y), m0["g"].predict(grid.reshape(-1, 1)), atol=1e-10)
    XX = pd.concat([X, d], axis=1)
    _, m1 = fit_gam_models_for_genes(XX, pd.DataFrame({"g": y}, index=X.index), ["g"], n_jobs=1,
                                     unpenalised_columns=list(d.columns))
    expect = [np.mean(m1["g"].predict(np.column_stack([np.full(len(XX), t), d.to_numpy()]))) for t in grid]
    np.testing.assert_allclose(_adjusted_curve(m1["g"], grid, XX, y), expect, atol=1e-8)


def _null_pvalues(n, n_batches, confounded, genes=600, seed=0, rename=None):
    """Pure-noise genes (plus a batch shift per gene when batched); pseudotime tied to batch if `confounded`."""
    rng = np.random.default_rng(seed)
    idx = [f"s{i}" for i in range(n)]
    bi = np.arange(n) % n_batches
    pt = rng.uniform(0, 1, n) + (0.7 * bi if confounded else 0)
    Y = rng.normal(0, 1, (n, genes)) + (bi[:, None] * rng.normal(0, 1.5, genes) if n_batches > 1 else 0)
    X, un = pd.DataFrame({"pseudotime": pt}, index=idx), []
    if n_batches > 1:
        lab = pd.Series([f"b{i}" for i in bi], index=idx).replace(rename or {})
        d = pd.get_dummies(lab, drop_first=True).astype(float)
        X, un = pd.concat([X, d], axis=1), list(d.columns)
    names = [f"g{j}" for j in range(genes)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res, _ = fit_gam_models_for_genes(X, pd.DataFrame(Y, index=idx, columns=names), names, n_jobs=1,
                                          unpenalised_columns=un)
    return res.set_index("gene").sort_index()["pval"]


def test_pvalue_is_calibrated_without_batch():
    for n in (26, 85, 405):
        assert (_null_pvalues(n, 1, False) < 0.05).mean() <= 0.07, n


def test_pvalue_is_calibrated_with_a_confounded_batch():
    for n, nb in ((60, 3), (120, 4)):
        assert (_null_pvalues(n, nb, True) < 0.05).mean() <= 0.07, (n, nb)


def test_pvalue_is_invariant_to_the_batch_labels():
    a = _null_pvalues(60, 3, True, genes=60)
    b = _null_pvalues(60, 3, True, genes=60, rename={"b0": "z0"})
    np.testing.assert_allclose(a, b, rtol=1e-6)


def _batch_pb(batch):
    a = _cells()
    a.obs["batch"] = a.obs["sample"].map(lambda s: batch(int(s[1:])))
    return a, dict(zip([f"S{i}" for i in range(N_SAMPLES)], np.random.default_rng(0).normal(0, 1.0, N_SAMPLES)))


def _run(a, pt, **kw):
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tdg.run_trajectory_gam_differential_gene_analysis(
            a, pt, n_jobs=1, output_dir=d, generate_visualizations=False, verbose=False, **kw).set_index("gene").sort_index()


def test_string_integer_and_object_batch_dtypes_give_the_same_result():
    lab = lambda si: "b1" if si < 8 else "b2"
    ref = None
    for cast in (lambda s: s.astype(object), lambda s: s.astype("string"), lambda s: s.astype("category"),
                 lambda s: s.map({"b1": 1, "b2": 2}).astype(int)):
        a, pt = _batch_pb(lab)
        a.obs["batch"] = cast(a.obs["batch"])
        res = _run(a, pt, batch_col="batch")
        ref = res if ref is None else ref
        np.testing.assert_allclose(res["pval"], ref["pval"], rtol=1e-6)


def test_batch_dummies_have_lam_zero_and_other_covariates_the_default():
    X, d, Y = _batch_design(seed=3)
    X["sex"] = (np.arange(len(X)) % 2).astype(float)
    XX = pd.concat([X, d], axis=1)
    _, models = fit_gam_models_for_genes(XX, Y[["g0"]], ["g0"], n_jobs=1, unpenalised_columns=list(d.columns))
    lams = {c: models["g0"].terms[j].lam[0] for j, c in enumerate(XX.columns) if j}
    assert all(lams[c] == 0 for c in d.columns) and lams["sex"] == 0.6


def test_categorical_batch_with_an_unused_category_and_missing_batch():
    a, pt = _batch_pb(lambda si: "b1" if si < 8 else "b2")
    a.obs["batch"] = pd.Categorical(a.obs["batch"], categories=["b1", "b2", "unused"])
    assert len(_run(a, pt, batch_col="batch")) > 0
    a, pt = _batch_pb(lambda si: "b1" if si < 8 else "b2")
    a.obs["batch"] = a.obs["batch"].where(a.obs["sample"] != "S3")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        pb = _build(a, min_variance_ratio=0)
    assert sum("missing 'batch'" in str(w.message) for w in rec) == 1
    assert np.isnan(pb[["S3"]].X).all() and np.isfinite(pb[[f"S{i}" for i in range(N_SAMPLES) if i != 3]].X).all()
    X, Y, _ = prepare_gam_input_data_improved(pb, {f"S{i}": float(i) for i in range(N_SAMPLES)},
                                              covariate_columns=["batch"], unpenalised_columns=["batch"])
    assert X.loc["S3"].isna()["batch_b2"] and X.drop(index="S3").notna().all().all()


def test_adjusted_curve_for_a_gene_missing_one_batch():
    X, d, Y = _batch_design(seed=4, confounded=False)
    XX = pd.concat([X, d], axis=1)
    y = Y["g0"].copy()
    y[d["b3"] == 1.0] = np.nan                                  # gene never measured in batch b3
    _, m = fit_gam_models_for_genes(XX, y.to_frame(), ["g0"], n_jobs=1, unpenalised_columns=list(d.columns))
    grid = np.linspace(X["pseudotime"].min(), X["pseudotime"].max(), 10)
    fit = np.isfinite(y.to_numpy())
    expect = [np.mean(m["g0"].predict(np.column_stack([np.full(fit.sum(), t), d.to_numpy()[fit]]))) for t in grid]
    np.testing.assert_allclose(_adjusted_curve(m["g0"], grid, XX, y), expect, atol=1e-8)


def _ols_design(n=30, n_batches=3, seed=0, slope=0.0):
    rng = np.random.default_rng(seed)
    idx = [f"s{i}" for i in range(n)]
    pt = rng.uniform(0, 1, n)
    d = pd.get_dummies(pd.Series([f"b{i % n_batches}" for i in range(n)], index=idx), drop_first=True).astype(float)
    Y = pd.DataFrame(rng.normal(0, 1, (n, 25)) + slope * pt[:, None] + 2 * d.sum(axis=1).to_numpy()[:, None],
                     index=idx, columns=[f"g{j}" for j in range(25)])
    Y.iloc[[2, 7], 3] = np.nan
    return pd.DataFrame({"pseudotime": pt}, index=idx), d, Y


def test_linear_test_equals_statsmodels_ols_with_and_without_batch():
    import statsmodels.api as sm
    X0, d, Y = _ols_design(slope=0.8)
    for X in (X0, pd.concat([X0, d], axis=1)):
        res, models = fit_linear_models_for_genes(X, Y, list(Y.columns))
        for g, row in res.set_index("gene").iterrows():
            m = Y[g].notna().to_numpy()
            fit = sm.OLS(Y[g].to_numpy()[m], sm.add_constant(X.to_numpy()[m])).fit()
            np.testing.assert_allclose(row["pval"], fit.pvalues[1], rtol=1e-8)
            np.testing.assert_allclose(row["dev_exp"], fit.rsquared, rtol=1e-8)
            np.testing.assert_allclose(models[g].coef_[0], fit.params[1], rtol=1e-8)


def test_linear_effect_size_and_direction():
    X, d, Y = _ols_design(slope=-1.5)
    XX = pd.concat([X, d], axis=1)
    res, models = fit_linear_models_for_genes(XX, Y, ["g0"])
    es = calculate_effect_size_and_direction(XX, Y, models, ["g0"]).iloc[0]
    fit = models["g0"]
    resid = Y["g0"].to_numpy() - fit.predict(XX.to_numpy())
    sd = np.sqrt((resid ** 2).sum() / (len(Y) - XX.shape[1] - 1))
    np.testing.assert_allclose(es["effect_size"], abs(fit.coef_[0]) * np.ptp(X["pseudotime"]) / sd, rtol=1e-10)
    assert es["regulation"] == "DOWN"


def test_linear_pvalue_is_calibrated():
    for n, nb, conf in ((26, 1, False), (85, 1, False), (60, 3, True)):
        X, _, _ = _ols_design(n=n, n_batches=max(nb, 2))
        rng = np.random.default_rng(1)
        bi = np.arange(n) % nb
        X = X.assign(pseudotime=rng.uniform(0, 1, n) + (0.7 * bi if conf else 0))
        if nb > 1:
            X = pd.concat([X, pd.get_dummies(pd.Series(bi, index=X.index).astype(str), drop_first=True).astype(float)], axis=1)
        Y = pd.DataFrame(rng.normal(0, 1, (n, 600)) + (bi[:, None] * rng.normal(0, 1.5, 600) if nb > 1 else 0), index=X.index)
        res, _ = fit_linear_models_for_genes(X, Y, list(Y.columns))
        assert (res["pval"] < 0.05).mean() <= 0.07, (n, nb)


def _pb_and_pt():
    a = _cells()
    pt = dict(zip([f"S{i}" for i in range(N_SAMPLES)], np.random.default_rng(0).normal(0, 1.0, N_SAMPLES)))
    return a, pt, _build(a, min_variance_ratio=0)


def test_gam_test_equals_the_stepwise_pipeline():
    a, pt, pb = _pb_and_pt()
    X, Y, names = prepare_gam_input_data_improved(pb, pt, covariate_columns=["batch"], unpenalised_columns=["batch"])
    res, models = fit_gam_models_for_genes(X, Y, names, n_jobs=1, unpenalised_columns=X.attrs["unpenalised"])
    es = calculate_effect_size_and_direction(X, Y, models, list(res.loc[res["fdr"] < 0.05, "gene"]))
    exp = res.merge(es, on="gene", how="left").set_index("gene").sort_index()
    for kw in ({}, {"test": "gam"}):
        got = _run(a, pt, batch_col="batch", min_variance_ratio=0, **kw)
        np.testing.assert_array_equal(got["pval"], exp["pval"])
        np.testing.assert_array_equal(got["effect_size"].fillna(-1), exp["effect_size"].fillna(-1))


def test_trajectory_test_features_reproduces_the_run_function():
    a, pt, pb = _pb_and_pt()
    for test in ("linear", "gam"):
        ref = _run(a, pt, batch_col="batch", min_variance_ratio=0, test=test, top_n_genes=5)
        with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = trajectory_test_features(pb, pt, test=test, batch_col="batch", n_jobs=1, output_dir=d,
                                           top_n_genes=5, verbose=False).set_index("gene").sort_index()
        assert list(got.index) == list(ref.index)
        for c in ("pval", "fdr", "dev_exp", "effect_size"):
            np.testing.assert_allclose(got[c].astype(float), ref[c].astype(float), rtol=1e-10, equal_nan=True)
        assert (got["regulation"].fillna("") == ref["regulation"].fillna("")).all()
        assert (got["pseudoDEG"] == ref["pseudoDEG"]).all()


def test_linear_run_with_visualizations_and_invalid_test_rejected():
    a, pt, _ = _pb_and_pt()
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = tdg.run_trajectory_gam_differential_gene_analysis(
            a, pt, batch_col="batch", test="linear", n_jobs=1, output_dir=d, verbose=True, top_n_genes=5)
        assert os.path.exists(os.path.join(d, "visualizations", "05_gene_curves.png"))
    assert (res.set_index("gene").loc["T - g0", "regulation"]) == "UP"
    try:
        _run(a, pt, test="quadratic")
    except ValueError:
        return
    raise AssertionError("invalid test accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
