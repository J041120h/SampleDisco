"""Trajectory-DGE variance-ratio gene filter, minimum cells per group and zero handling.
Synthetic data, CPU only.

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
    _build_sample_pseudobulk, calculate_effect_size_and_direction, fit_gam_models_for_genes)

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


def test_small_groups_are_nan_and_do_not_change_limma_estimates():
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


def test_single_sample_batch_row_is_nan_and_warns_per_celltype():
    a = _cells(batch_of=_unbalanced)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        pb = _build(a, min_variance_ratio=0)
    msgs = [str(w.message) for w in rec if "single retained sample" in str(w.message)]
    assert len(msgs) == 2 and all("['b4']" in m for m in msgs)
    assert "'B'" in "".join(msgs) and "'T'" in "".join(msgs)
    assert np.isnan(pb[["S15"]].X).all() and np.isfinite(pb[[f"S{i}" for i in range(15)]].X).all()


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
