"""Regression tests for the 0.6.1 fixes (items A-M). Synthetic data, CPU only."""
import ast
import os
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import sampledisco

PKG = Path(sampledisco.__file__).resolve().parent


# ---------------------------------------------------------------------------- A
def _cells_with_meta(modality=False):
    samples = ["S1", "S2", "S3"]
    cells = [(s, i) for s in samples for i in range(10)]
    obs = pd.DataFrame({
        "sample": [s for s, _ in cells],
        "sev.level": [{"S1": 1.0, "S2": 2.0, "S3": 3.0}[s] for s, _ in cells],
        "batch": [{"S1": "a", "S2": "a", "S3": "b"}[s] for s, _ in cells],
        "n_genes": [100 + 10 * i for _, i in cells],              # per cell
        "cell_type": ["T" if i % 2 else "B" for _, i in cells],   # per cell
    }, index=[f"c{k}" for k in range(len(cells))])
    obs.loc["c0", "sev.level"] = np.nan        # NaN mixed with the sample's value
    if modality:
        obs["modality"] = ["RNA" if i < 5 else "ATAC" for _, i in cells]
    a = ad.AnnData(np.zeros((len(cells), 1), dtype=np.float32), obs=obs)
    units = ([f"{s}_{m}" for s in samples for m in ("ATAC", "RNA")] if modality else samples)
    a.uns["X_DR_sample"] = pd.DataFrame(np.random.default_rng(0).normal(size=(len(units), 2)),
                                        index=units, columns=["PC1", "PC2"])
    return a


def test_build_sample_adata_keeps_only_unit_level_columns(capsys):
    from sampledisco.sample_embedding.sample_embedding import build_sample_adata
    pa = build_sample_adata(_cells_with_meta(), sample_col="sample")
    assert set(pa.obs.columns) == {"sample", "sev.level", "batch"}
    assert pa.obs.loc["S1", "sev.level"] == 1.0 and pa.obs.loc["S3", "batch"] == "b"
    out = capsys.readouterr().out
    assert "dropped 2 obs column(s)" in out and "n_genes" in out and "cell_type" in out


def test_build_sample_adata_multiomics_units_keep_modality():
    from sampledisco.sample_embedding.sample_embedding import build_sample_adata
    pa = build_sample_adata(_cells_with_meta(modality=True), sample_col="sample",
                            modality_col="modality")
    assert {"sample", "modality", "sev.level", "batch"} <= set(pa.obs.columns)
    assert "n_genes" not in pa.obs.columns and "cell_type" not in pa.obs.columns
    assert pa.obs.loc["S2_ATAC", "modality"] == "ATAC"


# ---------------------------------------------------------------------------- B
def test_cell_types_atac_merges_leiden_overshoot():
    from sampledisco.preparation.ATAC_cell_type import cell_types_atac
    rng = np.random.default_rng(17)
    centers = rng.normal(scale=10.0, size=(8, 10))
    lab = np.repeat(np.arange(8), 60)
    a = ad.AnnData(np.zeros((480, 1), dtype=np.float32),
                   obs=pd.DataFrame(index=[f"c{i}" for i in range(480)]))
    a.obsm["Z_comp"] = (centers[lab] + rng.normal(size=(480, 10))).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = cell_types_atac(a, n_target_clusters=3, umap=False, generate_plots=False,
                              verbose=False)
    assert 1 < out.obs["cell_type"].nunique() <= 3
    assert "cell_type_original" in out.obs and "cluster_mapping" in out.uns


# ---------------------------------------------------------------------------- C
def test_atac_gene_activity_cpu_worker():
    from sampledisco.gene_activity import ATAC_gene_activity as ga
    assert "peak_to_gene_activity_weighted import" not in (
        PKG / "gene_activity" / "ATAC_gene_activity.py").read_text()
    X = sp.csr_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32))
    g2p = {"G1": [{"peak": "p0", "combined_weight": 2.0, "gene_name": "A"},
                  {"peak": "p2", "combined_weight": 1.0}],
           "G2": [{"peak": "p1"}]}
    mat, stats = ga.process_gene_batch((["G1", "G2", "G3"], g2p, {"p0": 0, "p1": 1, "p2": 2},
                                        X, "weighted_sum", {}))
    np.testing.assert_allclose(mat.toarray(), [[4.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    assert [s["n_peaks"] for s in stats] == [2, 1, 0] and stats[0]["gene_name"] == "A"
    mat, _ = ga.process_gene_batch((["G1"], g2p, {"p0": 0, "p1": 1, "p2": 2}, X, "weighted_mean", {}))
    np.testing.assert_allclose(mat.toarray().ravel(), [4.0 / 3, 0.0])


# ---------------------------------------------------------------------------- D
@pytest.mark.parametrize("mod", ["rna_preprocess_gpu", "atac_preprocess_gpu"])
def test_gpu_preprocess_sample_level_batch_default_is_none(mod):
    tree = ast.parse((PKG / "preparation" / f"{mod}.py").read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names = [a.arg for a in node.args.args]
            defaults = dict(zip(names[len(names) - len(node.args.defaults):], node.args.defaults))
            if "sample_level_batch_key" in defaults:
                found.append(defaults["sample_level_batch_key"])
    assert found and all(isinstance(d, ast.Constant) and d.value is None for d in found)


# ---------------------------------------------------------------------------- E
@pytest.mark.parametrize("viz_cols", [["sev.level"], None])
def test_multiomics_target_modality_adds_highlight_plots(tmp_path, viz_cols):
    # viz_cols=None is the demo config: no colour/visualization column, so the
    # highlight colour falls back to grouping_columns (default ['sev.level']).
    from sampledisco.wrapper.wrapper import downstream_analysis
    rng = np.random.default_rng(18)
    units = [f"S{i}_{m}" for i in range(1, 5) for m in ("RNA", "ATAC")]
    obs = pd.DataFrame({"modality": [u.split("_")[1] for u in units],
                        "sev.level": [float(u[1]) for u in units]}, index=units)
    emb = pd.DataFrame(rng.normal(size=(len(units), 3)), index=units, columns=["PC1", "PC2", "PC3"])
    pa = ad.AnnData(emb.values.astype(np.float32), obs=obs)
    pa.uns["X_DR_sample"] = emb
    pa.obsm["X_DR_sample"] = emb.values.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        downstream_analysis(
            pseudo_adata=pa, output_dir=str(tmp_path), modality="multiomics",
            status_flags={"multiomics": {}}, sample_distance_calculation=False,
            trajectory_analysis=False, sample_cluster=False, visualize_data=False,
            visualize_embedding=True, multiomics_modality_col="modality",
            multiomics_target_modality="ATAC",
            multiomics_visualization_grouping_column=viz_cols, verbose=False)
    viz = tmp_path / "visualization"
    if viz_cols:
        assert (viz / "all_samples_expression_by_sev.level.png").exists()  # default output
    assert (viz / "ATAC_sev.level_expression.png").exists()                # highlight plots
    assert (viz / "ATAC_sev.level_proportion.png").exists()


# ---------------------------------------------------------------------------- F
def test_multiomics_wrapper_needs_no_rna_atac_files_without_integration(tmp_path):
    from sampledisco.wrapper.multiomics_wrapper import multiomics_wrapper
    rng = np.random.default_rng(19)
    n = 200
    obs = pd.DataFrame({"sample": rng.choice(["S1", "S2"], n), "modality": rng.choice(["RNA", "ATAC"], n),
                        "cell_type": rng.choice(["T", "B"], n)}, index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(np.zeros((n, 1), dtype=np.float32), obs=obs)
    a.obsm["Z_comp"] = rng.normal(size=(n, 5)).astype(np.float32)
    a.obsm["X_glue"] = rng.normal(size=(n, 5)).astype(np.float32)
    inp = tmp_path / "integrated.h5ad"
    a.write_h5ad(inp)
    res = multiomics_wrapper(rna_file=None, atac_file=None, multiomics_output_dir=str(tmp_path / "mo"),
                             integration=False, integrated_h5ad_path=str(inp),
                             cell_type_cluster=False, derive_sample_embedding=False,
                             use_gpu=False, multiomics_verbose=False)
    assert res["adata"].n_obs == n
    with pytest.raises(ValueError, match="rna_file and atac_file"):
        multiomics_wrapper(rna_file=None, atac_file=None, multiomics_output_dir=str(tmp_path / "x"),
                           integration=True)


# ---------------------------------------------------------------------------- G
def test_gam_worker_pool_capped_by_slurm(monkeypatch):
    import joblib
    from sampledisco.sample_trajectory.trajectory_diff_gene import fit_gam_models_for_genes
    seen = []

    class _RecordingParallel:
        def __init__(self, n_jobs=None, **kw):
            seen.append(n_jobs)

        def __call__(self, tasks):
            return [f(*a, **k) for f, a, k in tasks]

    monkeypatch.setattr(joblib, "Parallel", _RecordingParallel)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    rng = np.random.default_rng(20)
    idx = [f"s{i}" for i in range(12)]
    X = pd.DataFrame({"pseudotime": np.linspace(0, 1, 12)}, index=idx)
    Y = pd.DataFrame(rng.gamma(2.0, size=(12, 3)), index=idx, columns=["g1", "g2", "g3"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gam_models_for_genes(X, Y, ["g1", "g2", "g3"], n_jobs=-1)
    assert seen == [2]


# ---------------------------------------------------------------------------- H
def test_requirements_name_igraph_not_python_igraph():
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    if not req.exists():
        pytest.skip("requirements.txt not available (installed package)")
    lines = [l.split("#")[0].strip() for l in req.read_text().splitlines()]
    assert "igraph>=0.10" in lines and not any(l.startswith("python-igraph") for l in lines)


# ---------------------------------------------------------------------------- I
def test_unknown_batch_method_is_rejected():
    from sampledisco.cli import validate_config
    from sampledisco.sample_embedding import compute_sample_embedding
    from sampledisco.sample_embedding.blocks import check_batch_method
    for ok in ("harmony", "linear", "none"):
        check_batch_method(ok)
    for bad in ("Harmony", "combat", None):
        with pytest.raises(ValueError, match="batch_method"):
            check_batch_method(bad)
    a = _cells_with_meta()
    a.obsm["Z_comp"] = np.zeros((a.n_obs, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="batch_method"):
        compute_sample_embedding(a, "unused", sample_col="sample", celltype_col="cell_type",
                                 batch_method="combat", save=False, verbose=False)

    def f(rna_sample_embedding_batch_method="harmony"):
        pass
    with pytest.raises(ValueError, match="rna_sample_embedding_batch_method"):
        validate_config({"rna_sample_embedding_batch_method": "combat"}, f)


# ---------------------------------------------------------------------------- J
def test_existing_labels_merge_warns():
    from sampledisco.preparation.cell_type_cpu import cell_types
    rng = np.random.default_rng(21)
    n = 300
    a = ad.AnnData(np.zeros((n, 1), dtype=np.float32),
                   obs=pd.DataFrame({"celltype": rng.choice([f"L{i}" for i in range(6)], n)},
                                    index=[f"c{i}" for i in range(n)]))
    a.obsm["Z_comp"] = rng.normal(size=(n, 20)).astype(np.float32)
    with pytest.warns(UserWarning, match="merged into 3 groups"):
        cell_types(a, cell_type_column="celltype", existing_cell_types=True, n_target_clusters=3,
                   umap=False, verbose=False)
    assert a.obs["celltype"].nunique() == 6          # original labels untouched


# ---------------------------------------------------------------------------- K
_K_MODULES = ["gene_activity/RNA_name_convertor.py", "gene_activity/ATAC_gene_activity.py",
              "gene_activity/ATAC_peak_annotation.py", "gene_activity/validation.py",
              "gene_activity/test_pseudo_correlation.py", "utils/subsample_generator.py"]


@pytest.mark.parametrize("rel", _K_MODULES)
def test_no_module_level_blanket_warning_filter(rel):
    tree = ast.parse((PKG / rel).read_text())
    for node in tree.body:                         # module level only
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            args = [a.value for a in node.value.args if isinstance(a, ast.Constant)]
            assert not (name in ("filterwarnings", "simplefilter") and args[:1] == ["ignore"]), rel


@pytest.mark.parametrize("modname,needs", [
    ("sampledisco.gene_activity.validation", None),
    ("sampledisco.utils.subsample_generator", None),
    ("sampledisco.gene_activity.ATAC_gene_activity", None),
    ("sampledisco.gene_activity.RNA_name_convertor", "pyensembl"),
    ("sampledisco.gene_activity.ATAC_peak_annotation", "pyensembl"),
])
def test_importing_leaves_global_warning_filters_unchanged(modname, needs):
    import importlib
    if needs:
        pytest.importorskip(needs)
    import matplotlib.pyplot, scanpy, seaborn, tqdm  # noqa: F401  (pre-load third-party deps)
    sys.modules.pop(modname, None)
    before = list(warnings.filters)
    importlib.import_module(modname)
    assert list(warnings.filters) == before


# ---------------------------------------------------------------------------- L
def test_corrected_resolution_pvalues_never_zero(tmp_path):
    from sampledisco.sample_trajectory.CCA_test import (compute_corrected_pvalues,
                                                        compute_corrected_pvalues_rna)
    df = pd.DataFrame({"resolution": [0.1, 0.2], "cca_score": [0.99, 0.1]})
    null = np.linspace(0, 0.5, 20)                  # 16 values >= 0.1, none >= 0.99
    for fn in (compute_corrected_pvalues, compute_corrected_pvalues_rna):
        out = fn(df.copy(), null, str(tmp_path), "X")
        assert out["corrected_pvalue"].tolist() == pytest.approx([1 / 21, 17 / 21])


# ---------------------------------------------------------------------------- M
def test_sample_embedding_params_record_pcs_used(tmp_path):
    from sampledisco.sample_embedding import compute_sample_embedding
    rng = np.random.default_rng(22)
    n = 600
    obs = pd.DataFrame({"sample": rng.choice([f"S{i}" for i in range(1, 7)], n),
                        "cell_type": rng.choice(["T", "B", "NK"], n)}, index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(np.zeros((n, 1), dtype=np.float32), obs=obs)
    a.obsm["Z_comp"] = rng.normal(size=(n, 10)).astype(np.float32)
    a.obsm["Z_rmd"] = rng.normal(size=(n, 10)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compute_sample_embedding(a, str(tmp_path), sample_col="sample", celltype_col="cell_type",
                                 medium_K=4, fine_K=6, pca_components=10, save=False, verbose=False)
    p = a.uns["sample_embedding_params"]
    assert p["pca_components"] == a.uns["X_DR_sample"].shape[1] == 5     # 6 units -> at most 5
    assert p["pca_components_requested"] == 10


def test_single_plot_tde_heatmap_definition():
    tree = ast.parse((PKG / "sample_trajectory" / "trajectory_DGE_visualization.py").read_text())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "plot_tde_heatmap"]
    assert len(defs) == 1


# ======================================================= 0.6.1 follow-ups (1-6)
# ---------------------------------------------------------------------------- 1
@pytest.mark.parametrize("rel", ["preparation/cell_type_cpu.py", "preparation/ATAC_cell_type_gpu.py"])
def test_cell_typing_h5ad_writes_are_gzip(rel):
    tree = ast.parse((PKG / rel).read_text())
    writes = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr in ("write", "write_h5ad")]
    assert writes
    for n in writes:
        kw = {k.arg: k.value for k in n.keywords}
        assert isinstance(kw.get("compression"), ast.Constant) and kw["compression"].value == "gzip", \
            f"{rel}:{n.lineno}"


def test_cell_types_save_writes_gzip_with_identical_values(tmp_path):
    import h5py
    from sampledisco.preparation.cell_type_cpu import cell_types
    rng = np.random.default_rng(23)
    n = 120
    a = ad.AnnData(rng.poisson(2.0, size=(n, 30)).astype(np.float32),
                   obs=pd.DataFrame({"celltype": rng.choice(["A", "B"], n)},
                                    index=[f"c{i}" for i in range(n)]))
    a.obsm["Z_comp"] = rng.normal(size=(n, 20)).astype(np.float32)   # >= cell_embedding_num_PCs
    cell_types(a, cell_type_column="celltype", existing_cell_types=True, umap=False,
               save=True, output_dir=str(tmp_path), verbose=False)
    p = tmp_path / "preprocess" / "adata_preprocessed.h5ad"
    with h5py.File(p, "r") as f:
        assert f["X"].compression == "gzip"
    np.testing.assert_array_equal(ad.read_h5ad(p).X, a.X)


# ---------------------------------------------------------------------------- 2/3
def _raisin_toy(with_raw=False, raw_extra_genes=0):
    """8 samples x 20 cells, batches a/b crossed with groups g1/g2; batch b is shifted."""
    rng = np.random.default_rng(24)
    samples = [f"S{i}" for i in range(1, 9)]
    batch = dict(zip(samples, "aaaabbbb"))
    group = dict(zip(samples, ["g1", "g1", "g2", "g2", "g1", "g1", "g2", "g2"]))
    cells = [(s, j) for s in samples for j in range(20)]
    obs = pd.DataFrame({"sample": [s for s, _ in cells], "batch": [batch[s] for s, _ in cells],
                        "grp": [group[s] for s, _ in cells]}, index=[f"c{k}" for k in range(len(cells))])
    G = 25 + raw_extra_genes
    full = rng.gamma(5.0, 1.0, size=(len(cells), G)) + 3.0 * (obs["batch"].values == "b")[:, None]
    genes = [f"g{i}" for i in range(G)]
    if not with_raw:
        return ad.AnnData(full[:, :25].astype(np.float64), obs=obs, var=pd.DataFrame(index=genes[:25]))
    raw = ad.AnnData(full.astype(np.float64), obs=obs, var=pd.DataFrame(index=genes))
    a = raw[:, genes[:25]].copy()
    a.X = np.log1p(a.X)                 # .X differs from .raw, as after normalisation
    a.raw = raw
    return a


def test_combat_keeps_sparse_original_layer():
    from sampledisco.sample_clustering.RAISIN import apply_combat_correction
    a = _raisin_toy()
    dense = a.X.copy()
    a.X = sp.csr_matrix(a.X)
    out = apply_combat_correction(a, batch_col="batch", group_col="grp", verbose=False)
    assert sp.issparse(out.layers["original"])
    np.testing.assert_array_equal(out.layers["original"].toarray(), dense)
    assert not np.allclose(out.X, dense)


@pytest.mark.parametrize("raw_extra_genes", [0, 5])
def test_raisin_uses_combat_values_when_raw_present(raw_extra_genes):
    from sampledisco.sample_clustering.RAISIN import raisinfit
    kw = dict(sample_col="sample", testtype="unpaired", group_col="grp", n_jobs=1, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_raw = raisinfit(_raisin_toy(True, raw_extra_genes), batch_col="batch", **kw)
        plain = _raisin_toy(False, raw_extra_genes)
        plain_raw = ad.AnnData(_raisin_toy(True, raw_extra_genes).raw.X.copy(),
                               obs=plain.obs.copy(),
                               var=pd.DataFrame(index=_raisin_toy(True, raw_extra_genes).raw.var_names))
        fit_ref = raisinfit(plain_raw.copy(), batch_col="batch", **kw)       # ComBat on the same values
        fit_unc = raisinfit(plain_raw.copy(), batch_col=None, **kw)          # no correction
    assert fit_raw["batch_corrected"]
    assert list(fit_raw["mean"].index) == list(fit_ref["mean"].index)       # the raw genes
    np.testing.assert_allclose(fit_raw["mean"].values, fit_ref["mean"].values)
    assert not np.allclose(fit_raw["mean"].values, fit_unc["mean"].values)


# ---------------------------------------------------------------------------- 4
def test_n_worker_threads_capped_by_slurm(monkeypatch):
    from sampledisco.sample_embedding.blocks import n_worker_threads
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert n_worker_threads() == 2
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert n_worker_threads(cap=8) == 8


# ---------------------------------------------------------------------------- 5
def test_multiomics_integration_off_warning_is_accurate():
    src = (PKG / "wrapper" / "multiomics_wrapper.py").read_text()
    assert "silently skipped" not in src


# ---------------------------------------------------------------------------- 6
def test_atac_gene_activity_import_is_quiet(capsys):
    import importlib
    pytest.importorskip("tqdm")
    sys.modules.pop("sampledisco.gene_activity.ATAC_gene_activity", None)
    capsys.readouterr()
    ga = importlib.import_module("sampledisco.gene_activity.ATAC_gene_activity")
    assert "CuPy not available" not in capsys.readouterr().out
    if not ga.GPU_AVAILABLE:          # the message moved to the use_gpu=True path
        assert "GPU requested but CuPy not available" in (
            PKG / "gene_activity" / "ATAC_gene_activity.py").read_text()


# ======================================================= reproducible Harmony (0.6.1)
class _FakeHarmonyOut:
    def __init__(self, Z):
        self.Z_corr = Z


def _thread_recording_run_harmony(seen):
    import torch

    def run_harmony(data_mat, meta_data, vars_use, nclust=None, max_iter_harmony=10,
                    random_state=0, device=None):
        seen.append(torch.get_num_threads())
        return _FakeHarmonyOut(np.asarray(data_mat).T)
    return run_harmony


def test_requirements_make_harmony_pytorch_core():
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    if not req.exists():
        pytest.skip("requirements.txt not available (installed package)")
    lines = [l.split("#")[0].strip() for l in req.read_text().splitlines()]
    assert "harmony-pytorch==0.1.8" in lines and "harmonypy==0.2.0" in lines


def test_single_threaded_sets_and_restores_thread_counts():
    torch = pytest.importorskip("torch")
    from threadpoolctl import threadpool_info
    from sampledisco.utils.cpus import single_threaded
    orig = torch.get_num_threads()
    torch.set_num_threads(3)
    try:
        with single_threaded():
            assert torch.get_num_threads() == 1
            assert all(p["num_threads"] == 1 for p in threadpool_info() if p["user_api"] == "blas")
        assert torch.get_num_threads() == 3
        with pytest.raises(RuntimeError):
            with single_threaded():
                raise RuntimeError("boom")
        assert torch.get_num_threads() == 3                 # restored on error too
        with single_threaded():                             # after torch.set_num_threads(3),
            assert all(p["num_threads"] == 1 for p in threadpool_info() if p["user_api"] == "blas")
    finally:
        torch.set_num_threads(orig)


def test_all_harmonypy_calls_run_single_threaded(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    import harmonypy
    from sampledisco.utils.harmony_compat import harmonize_embedding
    from sampledisco.sample_embedding.sample_embedding_gpu import _gpu_harmonize
    from sampledisco.sample_embedding import compute_sample_embedding
    seen = []
    monkeypatch.setattr(harmonypy, "run_harmony", _thread_recording_run_harmony(seen))
    orig = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        # (1) cell-level fallback when harmony-pytorch cannot be imported
        monkeypatch.setitem(sys.modules, "harmony", None)
        obs = pd.DataFrame({"batch": ["a", "b"] * 10})
        harmonize_embedding(np.random.default_rng(25).normal(size=(20, 5)), obs, "batch")
        # (2) GPU-path sample-level Harmony
        _gpu_harmonize(np.random.default_rng(26).normal(size=(10, 4)),
                       [f"u{i}" for i in range(10)], ["a", "b"] * 5, 10, seed=0)
        # (3) CPU sample-level Harmony (blocks.build_emb_from_blocks)
        rng = np.random.default_rng(27)
        n = 2000
        cobs = pd.DataFrame({"sample": rng.choice([f"S{i}" for i in range(1, 11)], n),
                             "batch": rng.choice(["b1", "b2"], n),
                             "cell_type": rng.choice(["T", "B", "NK", "Mono"], n)},
                            index=[f"c{i}" for i in range(n)])
        a = ad.AnnData(np.zeros((n, 2), dtype=np.float32), obs=cobs)
        a.obsm["Z_comp"] = rng.normal(size=(n, 10)).astype(np.float32)
        a.obsm["Z_rmd"] = rng.normal(size=(n, 10)).astype(np.float32)
        compute_sample_embedding(a, str(tmp_path), use_gpu=False, sample_col="sample",
                                 celltype_col="cell_type", batch_col="batch", medium_K=4,
                                 fine_K=6, save=False, verbose=False)
        assert seen == [1, 1, 1]
        assert torch.get_num_threads() == 2                 # caller's setting restored
    finally:
        torch.set_num_threads(orig)


@pytest.mark.parametrize("backend", ["harmony-pytorch", "harmonypy"])
def test_cell_level_harmony_repeatable(monkeypatch, backend):
    pytest.importorskip("torch")
    from sampledisco.utils.harmony_compat import harmonize_embedding
    if backend == "harmony-pytorch":
        pytest.importorskip("harmony")
    else:
        monkeypatch.setitem(sys.modules, "harmony", None)
    rng = np.random.default_rng(28)
    X = rng.normal(size=(600, 10)) + np.repeat([[0.0], [2.0]], 300, axis=0)
    obs = pd.DataFrame({"batch": ["a"] * 300 + ["b"] * 300})
    z1 = harmonize_embedding(X, obs, "batch", max_iter_harmony=5)
    z2 = harmonize_embedding(X, obs, "batch", max_iter_harmony=5)
    assert np.array_equal(z1, z2)


@pytest.mark.parametrize("rel,call", [("preparation/rna_preprocess_cpu.py", "pca"),
                                      ("preparation/atac_preprocess_cpu.py", "lsi")])
def test_cell_embedding_svd_runs_single_threaded(rel, call):
    """The RNA PCA and ATAC LSI run inside utils.cpus.single_threaded() (multi-threaded MKL
    made X_pca differ between identical runs)."""
    tree = ast.parse((PKG / rel).read_text())
    calls, guarded = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == call:
            calls.append(node)
        if isinstance(node, ast.With) and any(
                isinstance(i.context_expr, ast.Call) and getattr(i.context_expr.func, "id", "") == "single_threaded"
                for i in node.items):
            guarded |= {id(n) for n in ast.walk(node)}
    assert calls and all(id(c) in guarded for c in calls)


# ======================================================= GAM workers (N, O)
def _gam_toy():
    rng = np.random.default_rng(29)
    n = 40
    idx = [f"s{i}" for i in range(n)]
    pt = np.linspace(0, 1, n)
    X = pd.DataFrame({"pseudotime": pt}, index=idx)
    Y = pd.DataFrame({
        "early": np.where(pt < 0.5, 2 + 3 * pt + rng.normal(0, 0.3, n), 0.0),   # non-zero on [0, .5)
        "late": np.where(pt > 0.3, 1 + np.sin(4 * pt) + rng.normal(0, 0.3, n), 0.0),
        "all": 2 + np.cos(3 * pt) + rng.normal(0, 0.3, n),
    }, index=idx)
    return X, Y


def test_gam_pvalues_do_not_depend_on_order_or_n_jobs():
    from sampledisco.sample_trajectory.trajectory_diff_gene import fit_gam_models_for_genes
    X, Y = _gam_toy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, _ = fit_gam_models_for_genes(X, Y, ["early", "late", "all"], n_jobs=1)
        b, _ = fit_gam_models_for_genes(X, Y, ["all", "late", "early"], n_jobs=1)
        c, _ = fit_gam_models_for_genes(X, Y, ["late", "early", "all"], n_jobs=2)
    pa, pb, pc = (d.set_index("gene")["pval"].sort_index() for d in (a, b, c))
    assert len(pa) == 3
    np.testing.assert_array_equal(pa.values, pb.values)     # bit-identical, any gene order
    np.testing.assert_array_equal(pa.values, pc.values)     # and with worker processes


def test_gam_fits_run_with_one_blas_thread(monkeypatch):
    import joblib
    from threadpoolctl import threadpool_info
    import sampledisco.sample_trajectory.trajectory_diff_gene as tdg
    seen = []
    real = tdg.LinearGAM

    class _Recording(real):
        def fit(self, *a, **k):
            seen.append(max([p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"] or [1]))
            return super().fit(*a, **k)
    monkeypatch.setattr(tdg, "LinearGAM", _Recording)
    X, Y = _gam_toy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tdg.fit_gam_models_for_genes(X, Y, ["early", "late", "all"], n_jobs=1)
    assert seen and set(seen) == {1}                          # serial path

    inner = []

    class _RecordingParallel:
        def __init__(self, n_jobs=None, **kw):
            backend, _ = joblib.parallel.get_active_backend()
            inner.append(getattr(backend, "inner_max_num_threads", None))

        def __call__(self, tasks):
            return [fn(*a, **k) for fn, a, k in tasks]
    monkeypatch.setattr(joblib, "Parallel", _RecordingParallel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tdg.fit_gam_models_for_genes(X, Y, ["early", "late", "all"], n_jobs=2)
    assert inner == [1]                                       # loky workers: 1 inner thread


# ======================================================= dendrogram cut to exactly k
def test_cut_linkage_gives_exactly_k_and_matches_fcluster_when_fcluster_does():
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import pdist
    from sampledisco.utils.linkage_cut import cut_linkage
    agree = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 13))
        X = rng.normal(size=(n, 6))
        for method in ("average", "complete", "single", "weighted", "centroid", "median"):
            Z = sch.linkage(pdist(X), method=method) if method not in ("centroid", "median") \
                else sch.linkage(X, method=method)
            for k in range(1, n + 1):
                lab = cut_linkage(Z, k)
                assert sorted(set(lab)) == list(range(1, k + 1)), (seed, method, k)
                if sch.is_monotonic(Z):                         # the cut_tree partition
                    ct = sch.cut_tree(Z, n_clusters=k).ravel()
                    assert len(set(zip(lab.tolist(), ct.tolist()))) == k
                fc = sch.fcluster(Z, t=k, criterion="maxclust")
                if sch.is_monotonic(Z) and len(set(fc)) == k:
                    np.testing.assert_array_equal(lab, fc)      # same partition AND numbering
                    agree += 1
    assert agree > 1000


def test_cut_linkage_nonmonotonic_stops_agglomeration_at_k():
    import scipy.cluster.hierarchy as sch
    from sampledisco.utils.linkage_cut import cut_linkage
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.9], [5.0, 5.0], [5.2, 5.1]])
    Z = sch.linkage(X, method="centroid")
    assert not sch.is_monotonic(Z)                 # an inversion
    for k in range(1, 6):
        assert len(set(cut_linkage(Z, k))) == k
    lab = cut_linkage(Z, 2)
    assert len({lab[0], lab[1], lab[2]}) == 1 and len({lab[3], lab[4]}) == 1 and lab[0] != lab[3]


def _nine_types(seed=31):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5.0, size=(9, 29))
    lab = np.repeat(np.arange(9), 40)
    a = ad.AnnData(np.zeros((lab.size, 1), dtype=np.float32),
                   obs=pd.DataFrame({"cell_type": [str(i + 1) for i in lab]},
                                    index=[f"c{i}" for i in range(lab.size)]))
    a.obsm["Z_comp"] = (centers[lab] + rng.normal(scale=0.1, size=(lab.size, 29))).astype(np.float32)
    return a


@pytest.mark.parametrize("which", ["cell_type_cpu", "ATAC_cell_type"])
def test_dendrogram_merge_nine_to_eight_gives_eight(which):
    """The release check's ATAC case: 9 clusters, n_target_clusters=8 -> 8 cell types
    (fcluster 'maxclust' returned 7)."""
    if which == "cell_type_cpu":
        from sampledisco.preparation.cell_type_cpu import cell_type_dendrogram as fn
        out = fn(_nine_types(), 8, groupby="cell_type", is_atac=True)
    else:
        from sampledisco.preparation.ATAC_cell_type import cell_type_dendrogram_atac as fn
        out = fn(_nine_types(), 8, groupby="cell_type", use_rep="Z_comp", verbose=False)
    assert out.obs["cell_type"].nunique() == 8
    assert sorted(out.obs["cell_type"].astype(str).unique(), key=int) == [str(i) for i in range(1, 9)]
    assert sum(len(v) for v in out.uns["cluster_mapping"].values()) == 9


def test_cut_linkage_when_maxclust_returns_too_few(monkeypatch):
    """Independent of the scipy version: if fcluster 'maxclust' returns fewer than k groups
    (scipy 1.14.1 does for the release check's 9 -> 8 ATAC case), the cut still has k."""
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import pdist
    import sampledisco.utils.linkage_cut as lc
    real = sch.fcluster

    def short_maxclust(Z, t, criterion="inconsistent", **kw):
        if criterion == "maxclust":
            return real(Z, t=max(1, t - 1), criterion="maxclust")
        return real(Z, t=t, criterion=criterion, **kw)
    monkeypatch.setattr(lc.sch, "fcluster", short_maxclust)
    Z = sch.linkage(pdist(np.random.default_rng(32).normal(size=(9, 29))), method="average")
    for k in range(2, 10):
        lab = lc.cut_linkage(Z, k)
        assert sorted(set(lab)) == list(range(1, k + 1))
        ct = sch.cut_tree(Z, n_clusters=k).ravel()
        assert len(set(zip(lab.tolist(), ct.tolist()))) == k     # the cut_tree partition
