"""Regression tests for tutorial-breaking bugs fixed in 0.5.1.

1. ``from sampledisco import wrapper`` must give a callable (it gave the subpackage).
2. An explicit canonical obsm key falls back to its legacy alias.
3. The CCA permutation-null histogram never crashes the test.
6. The distance group summary works when given a cell-level AnnData.
7. downstream_analysis never deletes a user-supplied summary CSV that already existed.
10. The multi-omics wrapper never writes the user's integrated input file.
11. After cell typing, the embedding uses the labels cell typing wrote.
12. A non-'sample' sample column survives the sample-metadata merge.
13. A missing grouping column skips the group check with a clear message.
14/15. chi_square / jensen_shannon run and receive grouping_columns.
16. The trajectory-DGE visualization module does not change global warning filters.
17. RAISIN's automatic worker count never exceeds SLURM_CPUS_PER_TASK.
18. RAISIN never reports significance for a comparison using a single-sample group.
19. Multi-omics cell typing compares the modalities it was given (e.g. RNA + FLOW).
20. CPU-path Harmony is pinned to device="cpu" (harmonypy >= 0.2 would pick a GPU itself);
    so is the GPU path's tiny sample-level Harmony.
21. The CCA permutation p-value is (1 + k) / (N + 1), never 0.
22. Supervised CCA pseudotime increases with the trajectory column.
23. harmony-pytorch gets the allocated CPU count, not every physical core.
24. Multi-omics DGE / RAISIN are skipped (with a warning) when there is no RNA expression file.
25. Trajectory-DGE pseudobulk warns about single-sample batches before limma.
"""
import inspect
import os
import subprocess
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import sampledisco
from sampledisco.sample_distance.distance_test import distanceCheck
from sampledisco.sample_trajectory.CCA_test import cca_pvalue_test
from sampledisco.utils.embedding_keys import resolve_comp_key, resolve_rmd_key


# --------------------------------------------------------------------------- 1
_SRC = str(Path(sampledisco.__file__).resolve().parents[1])

# Serve a stand-in for the heavy sampledisco.wrapper.wrapper module (so the fresh
# interpreter does not import scanpy & co.) through the normal import machinery,
# which still imports and binds the real sampledisco.wrapper package first.
_FAKE = """
import sys, importlib.abc, importlib.machinery
def _fn(a, *, b=1):
    return ("ran", a, b)
class _Fake(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name == "sampledisco.wrapper.wrapper":
            return importlib.machinery.ModuleSpec(name, self)
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.wrapper = _fn
sys.meta_path.insert(0, _Fake())
"""


@pytest.mark.parametrize("prelude", [
    "",                                  # `from sampledisco import wrapper` first
    "import sampledisco.wrapper\n",      # subpackage imported first
    "import sampledisco\nsampledisco.wrapper\n",  # attribute accessed first
])
def test_from_sampledisco_import_wrapper_is_callable(prelude):
    code = _FAKE + prelude + (
        "import inspect, sampledisco\n"
        "from sampledisco import wrapper\n"
        "assert wrapper is sampledisco.wrapper\n"
        "assert wrapper(5, b=2) == ('ran', 5, 2)\n"
        "assert sampledisco.wrapper(7) == ('ran', 7, 1)\n"
        "assert sampledisco.wrapper(8) == ('ran', 8, 1)\n"  # second access too
        "assert list(inspect.signature(wrapper).parameters) == ['a', 'b']\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [_SRC] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr


def test_existing_wrapper_import_paths_still_work():
    from sampledisco.wrapper.wrapper import wrapper as fn
    import sampledisco.wrapper.wrapper as mod
    from sampledisco.wrapper.rna_wrapper import rna_wrapper  # noqa: F401
    from sampledisco import wrapper

    assert inspect.isfunction(fn) and mod.wrapper is fn
    assert sampledisco.wrapper.wrapper is mod          # dotted attribute access
    assert sampledisco.wrapper.wrapper.wrapper is fn
    assert callable(wrapper) and wrapper is sampledisco.wrapper
    assert inspect.signature(wrapper) == inspect.signature(fn)


# --------------------------------------------------------------------------- 2
def _obsm_adata(keys, n=6):
    a = ad.AnnData(np.zeros((n, 2), dtype=np.float32))
    for i, k in enumerate(keys):
        a.obsm[k] = np.full((n, 3), float(i), dtype=np.float32)
    return a


def test_explicit_canonical_key_uses_legacy_alias():
    a = _obsm_adata(["X_glue", "X_umap", "Z_clust", "Z_cmd"])  # the multiomics demo file
    with pytest.warns(FutureWarning, match="legacy alias 'Z_clust'"):
        assert resolve_comp_key(a, "Z_comp", context="t_explicit") == "Z_clust"
    with pytest.warns(FutureWarning, match="legacy alias 'Z_cmd'"):
        assert resolve_rmd_key(a, "Z_rmd", context="t_explicit") == "Z_cmd"
    # Same answer as auto-resolution.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert resolve_comp_key(a, None, context="t_auto") == "Z_clust"
        assert resolve_rmd_key(a, None, context="t_auto") == "Z_cmd"


def test_explicit_canonical_key_alias_priority():
    a = _obsm_adata(["X_pca_harmony", "Z_clust"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert resolve_comp_key(a, "Z_comp", context="t_prio") == "Z_clust"


def test_explicit_key_present_is_unchanged():
    a = _obsm_adata(["Z_comp", "Z_clust", "Z_rmd", "Z_cmd", "X_glue"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_comp_key(a, "Z_comp") == "Z_comp"
        assert resolve_rmd_key(a, "Z_rmd") == "Z_rmd"
        assert resolve_rmd_key(a, "X_glue") == "X_glue"


def test_explicit_key_still_raises_when_nothing_matches():
    a = _obsm_adata(["X_glue", "X_umap"])
    with pytest.raises(KeyError, match="'Z_comp' not in adata.obsm"):
        resolve_comp_key(a, "Z_comp")
    with pytest.raises(KeyError, match="'Z_rmd' not in adata.obsm"):
        resolve_rmd_key(a, "Z_rmd")
    # A non-canonical explicit key never falls back.
    b = _obsm_adata(["Z_clust"])
    with pytest.raises(KeyError, match="'X_custom' not in adata.obsm"):
        resolve_comp_key(b, "X_custom")


# --------------------------------------------------------------------------- 3
def test_null_histogram_degenerate_range_does_not_raise(tmp_path):
    from sampledisco.sample_trajectory.CCA_test import _plot_null_histogram
    # 8 samples + n_pcs >= 7: every permuted |r| == 1 up to ~1e-16.
    scores = np.array([1.0, np.nextafter(1.0, 0.0)] * 500)
    try:
        np.histogram(scores, bins=30)
        numpy_raises = False
    except ValueError:
        numpy_raises = True
    path = tmp_path / "null.png"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _plot_null_histogram(scores, 1.0, 0.0, str(path))
    assert path.exists()
    if numpy_raises:
        assert any("degenerate" in str(w.message) for w in rec)


def test_null_histogram_unplottable_is_skipped_with_warning(tmp_path):
    from sampledisco.sample_trajectory.CCA_test import _plot_null_histogram
    path = tmp_path / "null.png"
    with pytest.warns(RuntimeWarning, match="could not draw the null histogram"):
        _plot_null_histogram(np.array([np.nan, np.nan]), 0.5, 1.0, str(path))
    assert not path.exists()


def test_cca_pvalue_test_saturated_pcs_completes(tmp_path):
    rng = np.random.default_rng(0)
    n = 8
    idx = [f"s{i}" for i in range(n)]
    coords = pd.DataFrame(rng.normal(size=(n, n - 1)), index=idx)
    obs = pd.DataFrame({"sev.level": np.arange(n, dtype=float)}, index=idx)
    pa = ad.AnnData(np.zeros((n, 1), dtype=np.float32), obs=obs)
    pa.uns["X_DR_sample"] = coords
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p = cca_pvalue_test(pa, "X_DR_sample", input_correlation=1.0,
                            output_directory=str(tmp_path), num_simulations=50,
                            n_pcs=n - 1, verbose=False)
    assert 0.0 <= p <= 1.0
    assert (tmp_path / "CCA_test" / "cca_pvalue_result_X_DR_sample.txt").exists()


# --------------------------------------------------------------------------- 6
_SAMPLES = ["A1", "A2", "A3", "B1", "B2", "B3"]
_GROUP = {s: ("mild" if s[0] == "A" else "severe") for s in _SAMPLES}


def _distance_df():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(len(_SAMPLES), 4))
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    return pd.DataFrame(d, index=_SAMPLES, columns=_SAMPLES)


def _expected_score(D):
    inn, btw = [], []
    for i, a in enumerate(_SAMPLES):
        for b in _SAMPLES[i + 1:]:
            (inn if _GROUP[a] == _GROUP[b] else btw).append(D.loc[a, b])
    return np.mean(btw) / np.mean(inn)


def _sample_adata():
    obs = pd.DataFrame({"sev.level": [_GROUP[s] for s in _SAMPLES]}, index=_SAMPLES)
    return ad.AnnData(np.zeros((len(_SAMPLES), 1), dtype=np.float32), obs=obs)


def _cell_adata(sample_col="sample", per_sample=3):
    cells = [(s, f"{s}_c{k}") for s in _SAMPLES for k in range(per_sample)]
    obs = pd.DataFrame({sample_col: [s for s, _ in cells],
                        "sev.level": [_GROUP[s] for s, _ in cells]},
                       index=[c for _, c in cells])
    obs[sample_col] = obs[sample_col].astype("category")
    return ad.AnnData(np.zeros((len(cells), 1), dtype=np.float32), obs=obs)


def test_distance_check_sample_level_unchanged(tmp_path):
    D = _distance_df()
    score = distanceCheck(D, "r", "m", str(tmp_path), adata=_sample_adata(),
                          grouping_columns=["sev.level"])
    assert score == pytest.approx(_expected_score(D), rel=1e-12)


def test_distance_check_accepts_cell_level_adata(tmp_path):
    D = _distance_df()
    expected = distanceCheck(D, "r", "m", str(tmp_path / "s"), adata=_sample_adata(),
                             grouping_columns=["sev.level"])
    got = distanceCheck(D, "r", "m", str(tmp_path / "c"), adata=_cell_adata(),
                        grouping_columns=["sev.level"])
    assert got == pytest.approx(expected, rel=1e-12)
    assert (tmp_path / "c" / "distance_check_results_r_m.txt").exists()
    got_donor = distanceCheck(D, "r", "m", str(tmp_path / "d"),
                              adata=_cell_adata(sample_col="donor"),
                              grouping_columns=["sev.level"], sample_column="donor")
    assert got_donor == pytest.approx(expected, rel=1e-12)


def test_distance_check_cell_level_without_sample_column_says_what_to_pass(tmp_path):
    with pytest.raises(ValueError, match="sample-level \\(pseudobulk\\) AnnData"):
        distanceCheck(_distance_df(), "r", "m", str(tmp_path),
                      adata=_cell_adata(sample_col="donor"), grouping_columns=["sev.level"])


def test_sample_distance_with_cell_level_adata_writes_group_summary(tmp_path):
    from sampledisco.sample_distance.sample_distance import sample_distance
    cell = _cell_adata()
    rng = np.random.default_rng(2)
    cell.uns["X_DR_sample"] = pd.DataFrame(rng.normal(size=(len(_SAMPLES), 3)),
                                           index=_SAMPLES, columns=["PC1", "PC2", "PC3"])
    sample_distance(adata=cell, output_dir=str(tmp_path), method="cosine",
                    data_type="RNA", grouping_columns=["sev.level"])
    assert (tmp_path / "cosine" / "distance_check_results_sample_DR_cosine.txt").exists()


# --------------------------------------------------------------------------- 7
def _run_distance_only(out_dir, summary_path):
    from sampledisco.wrapper.wrapper import downstream_analysis
    pa = _sample_adata()
    rng = np.random.default_rng(3)
    pa.uns["X_DR_sample"] = pd.DataFrame(rng.normal(size=(len(_SAMPLES), 3)),
                                         index=_SAMPLES, columns=["PC1", "PC2", "PC3"])
    downstream_analysis(
        pseudo_adata=pa, output_dir=str(out_dir), modality="rna",
        status_flags={"rna": {}}, sample_distance_methods=["cosine"],
        grouping_columns=["sev.level"], summary_sample_csv_path=summary_path,
        trajectory_analysis=False, sample_cluster=False, visualize_data=False,
    )


def test_user_supplied_existing_summary_csv_is_kept(tmp_path):
    user_csv = tmp_path / "my_summary.csv"
    pd.DataFrame({"cosine": [0.5]}, index=["earlier_run"]).to_csv(user_csv)
    _run_distance_only(tmp_path / "out", str(user_csv))
    assert user_csv.exists()
    kept = pd.read_csv(user_csv, index_col=0)
    assert "earlier_run" in kept.index and "sample_DR" in kept.index


def test_summary_csv_created_by_the_run_is_still_removed(tmp_path):
    new_csv = tmp_path / "new_summary.csv"
    _run_distance_only(tmp_path / "out1", str(new_csv))
    assert not new_csv.exists()
    _run_distance_only(tmp_path / "out2", None)                  # default path
    assert not (tmp_path / "out2" / "summary_sample.csv").exists()


# -------------------------------------------------------------------------- 10
def _md5(path):
    import hashlib
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _union_adata(n=480, seed=0):
    """Pre-integrated multi-omics union like the demo file: legacy obsm names."""
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({"sample": rng.choice([f"S{i}" for i in range(1, 5)], n),
                        "modality": rng.choice(["RNA", "ATAC"], n),
                        "cell_type": rng.choice(["T", "B", "NK"], n)},
                       index=[f"c{i}" for i in range(n)])
    obs["sev.level"] = obs["sample"].map({"S1": 1.0, "S2": 1.0, "S3": 4.0, "S4": 4.0})
    a = ad.AnnData(np.zeros((n, 2), dtype=np.float32), obs=obs)
    a.obsm["Z_clust"] = rng.normal(size=(n, 10)).astype(np.float32)
    a.obsm["X_glue"] = rng.normal(size=(n, 10)).astype(np.float32)
    return a


def test_multiomics_wrapper_never_writes_the_input_file(tmp_path):
    from sampledisco.wrapper.multiomics_wrapper import multiomics_wrapper
    inp = tmp_path / "input_integrated.h5ad"
    _union_adata().write_h5ad(inp)
    before = _md5(inp)
    out = tmp_path / "multiomics"
    kw = dict(rna_file="unused", atac_file="unused", multiomics_output_dir=str(out),
              integration=False, integrated_h5ad_path=str(inp), cell_type_cluster=False,
              use_gpu=False, multiomics_verbose=False,
              sample_embedding_medium_K=4, sample_embedding_fine_K=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multiomics_wrapper(derive_sample_embedding=True, **kw)
        assert _md5(inp) == before
        union = out / "preprocess" / "adata_sample.h5ad"
        assert "X_DR_sample" in ad.read_h5ad(union).uns
        # derive-only resume: reads the output union (with the embedding), not the input
        res = multiomics_wrapper(derive_sample_embedding=False, **kw)
    assert "X_DR_sample" in res["adata"].uns
    assert _md5(inp) == before


# -------------------------------------------------------------------------- 11
def _typed_cell_adata(n=600, seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({"sample": rng.choice([f"S{i}" for i in range(1, 7)], n),
                        "celltype": rng.choice([f"L{i}" for i in range(6)], n)},
                       index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(np.zeros((n, 2), dtype=np.float32), obs=obs)
    a.obsm["Z_comp"] = rng.normal(size=(n, 20)).astype(np.float32)
    a.obsm["Z_rmd"] = rng.normal(size=(n, 20)).astype(np.float32)
    return a


def test_embedding_uses_the_labels_cell_typing_wrote(tmp_path):
    from sampledisco.wrapper.rna_wrapper import rna_wrapper
    inp = tmp_path / "typed.h5ad"
    _typed_cell_adata().write_h5ad(inp)
    kw = dict(rna_count_data_path="unused", preprocessing=False, adata_path=str(inp),
              celltype_col="celltype", use_gpu=False, verbose=False,
              sample_embedding_medium_K=4, sample_embedding_fine_K=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        typed = rna_wrapper(rna_output_dir=str(tmp_path / "o1"), cell_type_cluster=True,
                            existing_cell_types=True, n_target_cell_clusters=3, **kw)
        untyped = rna_wrapper(rna_output_dir=str(tmp_path / "o2"), cell_type_cluster=False, **kw)
    assert typed["celltype_col"] == "cell_type"
    n_typed = typed["adata"].obs["cell_type"].nunique()
    assert n_typed <= 3
    assert typed["adata"].uns["sample_embedding_params"]["K_c"] == n_typed
    # cell typing not run: the user's column is used, as before
    assert untyped["celltype_col"] == "celltype"
    assert untyped["adata"].uns["sample_embedding_params"]["K_c"] == 6


# -------------------------------------------------------------------------- 12
def test_merge_sample_metadata_keeps_custom_sample_column(tmp_path):
    from sampledisco.utils.merge_sample_meta import merge_sample_metadata
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"donor": ["d1", "d2"], "sev.level": [1, 4]}).to_csv(meta, index=False)
    a = ad.AnnData(np.zeros((3, 1), dtype=np.float32),
                   obs=pd.DataFrame({"donor": ["d1", "d1", "d2"]}, index=["c1", "c2", "c3"]))
    merge_sample_metadata(a, meta, sample_column="donor", verbose=False)
    assert list(a.obs["donor"]) == ["d1", "d1", "d2"]
    assert list(a.obs["sample"]) == ["d1", "d1", "d2"]
    assert list(a.obs["sev.level"]) == [1, 1, 4]
    # default key name: unchanged, no extra columns
    meta2 = tmp_path / "meta2.csv"
    pd.DataFrame({"sample": ["d1", "d2"], "sev.level": [1, 4]}).to_csv(meta2, index=False)
    b = ad.AnnData(np.zeros((3, 1), dtype=np.float32),
                   obs=pd.DataFrame({"sample": ["d1", "d1", "d2"]}, index=["c1", "c2", "c3"]))
    merge_sample_metadata(b, meta2, verbose=False)
    assert list(b.obs.columns) == ["sample", "sev.level"]


# -------------------------------------------------------------------------- 13
def test_missing_grouping_column_skips_group_check_cleanly(tmp_path, capsys):
    from sampledisco.sample_distance.sample_distance import sample_distance
    pa = _sample_adata()
    rng = np.random.default_rng(4)
    pa.uns["X_DR_sample"] = pd.DataFrame(rng.normal(size=(len(_SAMPLES), 3)),
                                         index=_SAMPLES, columns=["PC1", "PC2", "PC3"])
    sample_distance(adata=pa, output_dir=str(tmp_path), method="cosine",
                    data_type="RNA", grouping_columns=["not_a_column"])
    out = capsys.readouterr().out
    assert "NoneType" not in out
    assert "Skipping group distance check" in out
    assert (tmp_path / "cosine" / "distance_matrix_sample_DR.csv").exists()
    assert not (tmp_path / "cosine" / "distance_check_results_sample_DR_cosine.txt").exists()


# ------------------------------------------------------------------- 14 / 15
@pytest.mark.parametrize("method,sub", [("chi_square", "chi_square"),
                                        ("jensen_shannon", "jensen_shannon")])
def test_proportion_distances_run_and_score_groups(tmp_path, method, sub):
    from sampledisco.sample_distance.sample_distance import sample_distance
    cell = _cell_adata(per_sample=5)
    rng = np.random.default_rng(5)
    cell.obs["cell_type"] = rng.choice(["T", "B", "NK"], cell.n_obs)
    sample_distance(adata=_sample_adata(), output_dir=str(tmp_path), method=method,
                    cell_adata=cell, pseudobulk_adata=_sample_adata(),
                    sample_column="sample", cell_type_column="cell_type",
                    grouping_columns=["sev.level"])
    out = tmp_path / sub / "proportion_DR_distance"
    assert (out / "distance_matrix_proportion_DR.csv").exists()
    assert (out / f"distance_check_results_proportion_DR_{method}.txt").exists()


# -------------------------------------------------------------------------- 16
def test_dge_visualization_leaves_global_warning_filters_unchanged(tmp_path):
    import importlib
    import matplotlib.pyplot, seaborn, sklearn.cluster, scipy.ndimage  # noqa: F401  (pre-load deps)
    name = "sampledisco.sample_trajectory.trajectory_DGE_visualization"
    sys.modules.pop(name, None)
    before = list(warnings.filters)
    viz = importlib.import_module(name)                    # module-level code runs here
    assert list(warnings.filters) == before
    rng = np.random.default_rng(6)
    idx = [f"s{i}" for i in range(8)]
    X = pd.DataFrame({"pseudotime": np.linspace(0, 1, 8)}, index=idx)
    Y = pd.DataFrame(rng.normal(size=(8, 3)), index=idx, columns=["g1", "g2", "g3"])
    res = pd.DataFrame({"gene": ["g1", "g2", "g3"], "pval": [0.01, 0.5, 0.9],
                        "fdr": [0.03, 0.6, 0.9], "dev_exp": [0.5, 0.1, 0.0],
                        "significant": [True, False, False]})
    pa = ad.AnnData(np.zeros((8, 1), dtype=np.float32), obs=pd.DataFrame(index=idx))
    viz.generate_all_visualizations(X, Y, res, {}, pa, str(tmp_path), verbose=False)
    assert list(warnings.filters) == before


# -------------------------------------------------------------------------- 17
def test_raisin_auto_worker_count_capped_by_slurm(monkeypatch):
    from sampledisco.sample_clustering import RAISIN
    monkeypatch.setattr(RAISIN.os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert RAISIN._resolve_n_jobs(-1) == 8
    assert RAISIN._resolve_n_jobs(None) == 8
    assert RAISIN._resolve_n_jobs(3) == 3            # explicit values untouched
    assert RAISIN._resolve_n_jobs(12) == 12
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    assert RAISIN._resolve_n_jobs(-1) == 16          # only ever caps
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert RAISIN._resolve_n_jobs(-1) == 16          # no SLURM variable: unchanged


# -------------------------------------------------------------------------- 18
def _raisin_fit():
    """Unpaired RAISIN fit with groups A (3 samples), B (3), C (1 sample)."""
    from sampledisco.sample_clustering.RAISIN import raisinfit
    rng = np.random.default_rng(7)
    clade = {"S1": "A", "S2": "A", "S3": "A", "S4": "B", "S5": "B", "S6": "B", "S7": "C"}
    cells = [(s, f"{s}_c{i}") for s in clade for i in range(30)]
    obs = pd.DataFrame({"sample": [s for s, _ in cells]}, index=[c for _, c in cells])
    X = rng.poisson(3.0, size=(len(cells), 200)).astype(np.float32)
    a = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(200)]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return raisinfit(a, sample_col="sample", sample_to_clade=clade, testtype="unpaired",
                         n_jobs=1, verbose=False)


def test_raisin_single_sample_group_is_not_tested(tmp_path):
    from sampledisco.sample_clustering.RAISIN_TEST import raisintest, run_pairwise_tests
    fit = _raisin_fit()
    assert [str(g) for g in fit["failgroup"]] == ["C"]
    with pytest.warns(RuntimeWarning, match=r"\['C'\]"):
        n_sig, tables = run_pairwise_tests(fit, str(tmp_path), n_permutations=10,
                                           make_summary_plots=False, verbose=False)
    assert set(tables) == {"A_vs_B"} and set(n_sig) == {"A_vs_B"}
    for comp in ("A_vs_C", "B_vs_C"):
        assert "variance not estimable" in (tmp_path / comp / "FAILED").read_text()
    summ = pd.read_csv(tmp_path / "summary_plots" / "raisin_summary.csv")
    assert summ.loc[summ.comparison == "A_vs_C", "n_sig"].isna().all()
    # A_vs_B (both groups >= 2 samples) is exactly the unguarded raisintest result
    cols = [str(c) for c in fit["X"].columns]
    ab = np.zeros(len(cols)); ab[cols.index("A")] = 1; ab[cols.index("B")] = -1
    pd.testing.assert_frame_equal(tables["A_vs_B"],
                                  raisintest(fit, contrast=ab, n_permutations=10, verbose=False))
    # a direct raisintest on a contrast that uses C reports NaN, never significance
    ac = np.zeros(len(cols)); ac[cols.index("A")] = 1; ac[cols.index("C")] = -1
    with pytest.warns(RuntimeWarning, match=r"\['C'\]"):
        r = raisintest(fit, contrast=ac, n_permutations=10, verbose=False)
    assert r[["stat", "pvalue", "FDR"]].isna().all().all() and r["Foldchange"].notna().all()


# -------------------------------------------------------------------------- 19
@pytest.mark.parametrize("second", ["FLOW", "ATAC"])
def test_multiomics_cell_typing_keeps_cells_for_any_second_modality(second):
    from sampledisco.preparation.multi_omics_cell_type_cpu import cell_types_multiomics
    rng = np.random.default_rng(8)
    n = 300                                              # cells per modality, 3 shared blobs
    centers = rng.normal(scale=8.0, size=(3, 10))
    lab = np.tile(np.arange(3), 2 * n // 3)
    z = (centers[lab] + rng.normal(size=(2 * n, 10))).astype(np.float32)
    obs = pd.DataFrame({"modality": ["RNA"] * n + [second] * n},
                       index=[f"c{i}" for i in range(2 * n)])
    a = ad.AnnData(np.zeros((2 * n, 1), dtype=np.float32), obs=obs)
    a.obsm["Z_comp"] = z
    out = cell_types_multiomics(a, rna_modality_value="RNA", atac_modality_value=second,
                                num_PCs=10, compute_umap=False, generate_plots=False,
                                verbose=False)
    assert out.n_obs == 2 * n
    assert (out.obs["modality"] == second).sum() == n


# -------------------------------------------------------------------------- 20
class _FakeHarmonyOut:
    def __init__(self, Z):
        self.Z_corr = Z


def _recording_run_harmony(calls, with_device=True):
    """Stand-in for harmonypy.run_harmony (0.2.0 signature, or pre-0.2 without device)."""
    if with_device:
        def run_harmony(data_mat, meta_data, vars_use, nclust=None, max_iter_harmony=10,
                        random_state=0, device=None):
            calls.append(device)
            return _FakeHarmonyOut(np.asarray(data_mat).T)
    else:
        def run_harmony(data_mat, meta_data, vars_use, nclust=None, max_iter_harmony=10,
                        random_state=0):
            calls.append("no-device-arg")
            return _FakeHarmonyOut(np.asarray(data_mat).T)
    return run_harmony


def test_harmony_compat_pins_cpu_unless_gpu(monkeypatch):
    import harmonypy
    from sampledisco.utils.harmony_compat import harmonize_embedding
    monkeypatch.setitem(sys.modules, "harmony", None)       # force the harmonypy branch
    obs = pd.DataFrame({"batch": ["a", "b"] * 10})
    X = np.random.default_rng(9).normal(size=(20, 5))
    calls = []
    monkeypatch.setattr(harmonypy, "run_harmony", _recording_run_harmony(calls))
    harmonize_embedding(X, obs, "batch", use_gpu=False)
    harmonize_embedding(X, obs, "batch", use_gpu=True)
    assert calls == ["cpu", None]                           # GPU path: harmonypy decides
    old = []                                                # pre-0.2 harmonypy: no device kwarg
    monkeypatch.setattr(harmonypy, "run_harmony", _recording_run_harmony(old, with_device=False))
    harmonize_embedding(X, obs, "batch", use_gpu=False)
    assert old == ["no-device-arg"]


def test_sample_level_harmony_runs_on_cpu_when_use_gpu_false(monkeypatch, tmp_path):
    import harmonypy
    from sampledisco.sample_embedding import compute_sample_embedding
    calls = []
    monkeypatch.setattr(harmonypy, "run_harmony", _recording_run_harmony(calls))
    rng = np.random.default_rng(10)
    n = 2000
    obs = pd.DataFrame({"sample": rng.choice([f"S{i}" for i in range(1, 11)], n),
                        "batch": rng.choice(["b1", "b2"], n),
                        "cell_type": rng.choice(["T", "B", "NK", "Mono"], n)},
                       index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(np.zeros((n, 2), dtype=np.float32), obs=obs)
    a.obsm["Z_comp"] = rng.normal(size=(n, 10)).astype(np.float32)
    a.obsm["Z_rmd"] = rng.normal(size=(n, 10)).astype(np.float32)
    compute_sample_embedding(a, str(tmp_path), use_gpu=False, sample_col="sample",
                             celltype_col="cell_type", batch_col="batch", medium_K=4,
                             fine_K=6, save=False, verbose=False)
    assert calls and set(calls) == {"cpu"}


def test_gpu_path_sample_harmony_is_pinned_to_cpu(monkeypatch):
    import harmonypy
    from sampledisco.sample_embedding.sample_embedding_gpu import _gpu_harmonize
    calls = []
    monkeypatch.setattr(harmonypy, "run_harmony", _recording_run_harmony(calls))
    Fp = np.random.default_rng(13).normal(size=(10, 4))
    _gpu_harmonize(Fp, [f"u{i}" for i in range(10)], ["a", "b"] * 5, 10, seed=0)
    assert calls == ["cpu"]


# -------------------------------------------------------------------------- 21
def test_cca_permutation_pvalue_is_never_zero(tmp_path):
    rng = np.random.default_rng(11)
    n, N = 12, 50
    idx = [f"s{i}" for i in range(n)]
    pa = ad.AnnData(np.zeros((n, 1), dtype=np.float32),
                    obs=pd.DataFrame({"sev.level": np.arange(n, dtype=float)}, index=idx))
    pa.uns["X_DR_sample"] = pd.DataFrame(rng.normal(size=(n, 2)), index=idx)
    kw = dict(num_simulations=N, n_pcs=2, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_none = cca_pvalue_test(pa, "X_DR_sample", 2.0, str(tmp_path / "a"), **kw)  # unreachable
        p_all = cca_pvalue_test(pa, "X_DR_sample", -1.0, str(tmp_path / "b"), **kw)  # always reached
        p_mid = cca_pvalue_test(pa, "X_DR_sample", 0.5, str(tmp_path / "c"), **kw)
    assert p_none == pytest.approx(1 / (N + 1))
    assert p_all == pytest.approx(1.0)
    k = p_mid * (N + 1) - 1
    assert abs(k - round(k)) < 1e-9 and 0 <= round(k) <= N
    txt = (tmp_path / "a" / "CCA_test" / "cca_pvalue_result_X_DR_sample.txt").read_text()
    assert "(1 + k) / (N + 1)" in txt and "k = 0 of N = 50" in txt


# -------------------------------------------------------------------------- 22
def test_supervised_cca_pseudotime_increases_with_trajectory(capsys):
    from scipy.stats import spearmanr
    from sampledisco.sample_trajectory.CCA import CCA_Call
    rng = np.random.default_rng(12)
    n = 10
    idx = [f"s{i}" for i in range(n)]
    sev = np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 2)
    base = sev + rng.normal(scale=0.1, size=n)
    other = rng.normal(scale=0.1, size=n)
    scores = []
    for sign in (1.0, -1.0):                   # mirror images: one comes out reversed
        pa = ad.AnnData(np.zeros((n, 1), dtype=np.float32),
                        obs=pd.DataFrame({"sev.level": sev}, index=idx))
        pa.uns["X_DR_sample"] = pd.DataFrame(np.column_stack([sign * base, other]),
                                             index=idx, columns=["PC1", "PC2"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score, _, ptime, _ = CCA_Call(pa, output_dir=None, trajectory_col="sev.level",
                                          n_components=2)
        assert spearmanr([ptime[s] for s in idx], sev)[0] > 0
        assert min(ptime.values()) == pytest.approx(0.0) and max(ptime.values()) == pytest.approx(1.0)
        scores.append(score)
    assert "Flipped pseudotime" in capsys.readouterr().out
    assert scores[0] == pytest.approx(scores[1])           # |r| unaffected


# -------------------------------------------------------------------------- 23
def test_harmony_pytorch_gets_the_allocated_cpu_count(monkeypatch):
    import types
    from sampledisco.utils.harmony_compat import harmonize_embedding
    seen = []

    def harmonize(X, batch_mat, batch_key, n_clusters=None, max_iter_harmony=10,
                  random_state=0, use_gpu=False, n_jobs=-1, verbose=True):
        seen.append(n_jobs)
        return np.asarray(X)

    fake = types.ModuleType("harmony")
    fake.harmonize = harmonize
    monkeypatch.setitem(sys.modules, "harmony", fake)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    obs = pd.DataFrame({"batch": ["a", "b"] * 10})
    X = np.random.default_rng(14).normal(size=(20, 5))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    harmonize_embedding(X, obs, "batch", use_gpu=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    harmonize_embedding(X, obs, "batch", use_gpu=False)
    assert seen == [3, 16]                                  # never harmony's -1 (all cores)


# -------------------------------------------------------------------------- 24
def test_multiomics_dge_skipped_without_rna_expression_file(capsys):
    from sampledisco.wrapper.wrapper import _skip_multiomics_dge
    assert _skip_multiomics_dge(True, True, "out/preprocess/adata_rna_preprocessed.h5ad") == (False, False)
    out = capsys.readouterr().out
    assert out.count("WARNING: skipping multi-omics trajectory DGE and cluster DGE") == 1
    assert "adata_rna_preprocessed.h5ad" in out
    assert _skip_multiomics_dge(False, False, "x") == (False, False)
    assert capsys.readouterr().out == ""                    # nothing requested: no warning


# -------------------------------------------------------------------------- 25
def test_single_sample_batches_are_kept_without_warning():
    from sampledisco.sample_trajectory.trajectory_diff_gene import _build_sample_pseudobulk
    rng = np.random.default_rng(15)
    samples = ["S1", "S2", "S3", "S4", "S5"]
    cells = [(s, f"{s}_c{i}") for s in samples for i in range(20)]
    obs = pd.DataFrame({"sample": [s for s, _ in cells],
                        "cell_type": rng.choice(["T", "B"], len(cells))},
                       index=[c for _, c in cells])
    a = ad.AnnData(rng.gamma(2.0, size=(len(cells), 30)).astype(np.float32), obs=obs,
                   var=pd.DataFrame(index=[f"g{j}" for j in range(30)]))
    a.obs["batch"] = a.obs["sample"].map({"S1": "b1", "S2": "b1", "S3": "b2", "S4": "b2", "S5": "b3"})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _build_sample_pseudobulk(a, "sample", "cell_type", batch_col="batch",
                                 min_cells_per_group=2, min_variance_ratio=0)
    assert not any("single" in str(w.message) for w in rec)
