"""The in-place .uns update stores the same embedding as a full h5ad re-write and leaves everything
else in the file untouched; rmd_weight='equal' resolves to sqrt(w_A1^2 + w_A2^2 + w_A3^2) and gives the
same embedding as passing that number. Synthetic data, CPU only.
Run with `pytest tests/` or `python tests/test_inplace_uns_and_equal_weight.py`."""
import math
import os
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sampledisco.sample_embedding import compute_sample_embedding
from sampledisco.sample_embedding.blocks import check_rmd_weight, resolve_rmd_weight, save_embedding_to_h5ad

KW = dict(sample_col="sample", celltype_col="cell_type", batch_col="batch", medium_K=4, fine_K=6, verbose=False)


def make_adata(n=2000, g=30, seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({"sample": pd.Categorical(rng.choice([f"S{i}" for i in range(1, 11)], n)),
                        "batch": rng.choice(["b1", "b2"], n),
                        "cell_type": pd.Categorical(rng.choice(["T", "B", "NK", "Mono"], n))},
                       index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(X=sp.random(n, g, density=0.2, format="csr", random_state=seed, dtype=np.float32), obs=obs,
                   var=pd.DataFrame(index=[f"g{j}" for j in range(g)]))
    a.layers["counts"] = a.X.copy()
    a.obsm["Z_comp"] = rng.normal(size=(n, 10)).astype(np.float32)
    a.obsm["Z_rmd"] = rng.normal(size=(n, 10)).astype(np.float32)
    a.uns["kept"] = {"note": "untouched"}
    return a


def write_cell_h5ad(out, a):
    path = os.path.join(out, "preprocess", "adata_preprocessed.h5ad")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a.write_h5ad(path, compression="gzip")
    return path


def assert_same_adata(x, y):
    assert (x.X != y.X).nnz == 0 and (x.layers["counts"] != y.layers["counts"]).nnz == 0
    pd.testing.assert_frame_equal(x.obs, y.obs)
    pd.testing.assert_frame_equal(x.var, y.var)
    assert set(x.obsm) == set(y.obsm) and all(np.array_equal(x.obsm[k], y.obsm[k]) for k in x.obsm)
    pd.testing.assert_frame_equal(x.uns["X_DR_sample"], y.uns["X_DR_sample"])
    assert x.uns["sample_embedding_params"].keys() == y.uns["sample_embedding_params"].keys()
    assert x.uns["kept"] == y.uns["kept"] == {"note": "untouched"}


def test_inplace_update_equals_full_rewrite():
    with tempfile.TemporaryDirectory() as out:
        a = make_adata()
        path = write_cell_h5ad(out, a)
        a.uns["X_DR_sample"] = pd.DataFrame({"PC1": [1.0]}, index=["old"])  # stale value already on disk
        a.uns["sample_embedding_params"] = {"old": 1}
        save_embedding_to_h5ad(path, a)
        compute_sample_embedding(a, out, save=True, save_cell_adata=True, **KW)
        on_disk = ad.read_h5ad(path)
        full = os.path.join(out, "full.h5ad")
        a.write_h5ad(full, compression="gzip")  # what the previous full re-write would store
        assert_same_adata(on_disk, ad.read_h5ad(full))
        pd.testing.assert_frame_equal(on_disk.uns["X_DR_sample"], a.uns["X_DR_sample"])
        assert on_disk.uns["sample_embedding_params"]["rmd_weight"] == a.uns["sample_embedding_params"]["rmd_weight"]


def test_inplace_falls_back_to_rewrite_when_layout_differs():
    with tempfile.TemporaryDirectory() as out:
        a = make_adata()
        path = write_cell_h5ad(out, a)
        compute_sample_embedding(a, out, save=False, **KW)
        assert save_embedding_to_h5ad(path, a) == "in-place"
        a.obs["new_col"] = 1  # in-memory change the file lacks -> whole object is re-written
        assert save_embedding_to_h5ad(path, a) == "rewrite"
        assert "new_col" in ad.read_h5ad(path).obs


def test_same_layout_changed_labels_rewrites():
    with tempfile.TemporaryDirectory() as out:
        a = make_adata()
        path = write_cell_h5ad(out, a)
        a.obs["cell_type"] = a.obs["cell_type"].cat.rename_categories({"T": "T_cell"})  # same columns, new labels
        compute_sample_embedding(a, out, save=True, save_cell_adata=True, **KW)
        on_disk = ad.read_h5ad(path)
        assert "T_cell" in set(on_disk.obs["cell_type"]) and "T" not in set(on_disk.obs["cell_type"])
        pd.testing.assert_frame_equal(on_disk.uns["X_DR_sample"], a.uns["X_DR_sample"])
        c = make_adata()
        write_cell_h5ad(out, c)
        compute_sample_embedding(c, out, save=False, **KW)
        cats = list(c.obs["cell_type"].cat.categories)  # same codes, categories swapped -> different labels
        c.obs["cell_type"] = pd.Categorical.from_codes(c.obs["cell_type"].cat.codes, categories=cats[::-1])
        assert save_embedding_to_h5ad(path, c) == "rewrite"
        b = make_adata()
        write_cell_h5ad(out, b)
        compute_sample_embedding(b, out, save=False, **KW)
        b.obsm["Z_rmd"][5, 3] += 1.0  # a changed value in a matrix (row 5 is not a probe row; row 0 is)
        b.obsm["Z_rmd"][0, 3] += 1.0
        assert save_embedding_to_h5ad(path, b) == "rewrite"


def test_old_format_falls_back_and_interrupted_update_recovers():
    import h5py
    with tempfile.TemporaryDirectory() as out:
        a = make_adata()
        path = write_cell_h5ad(out, a)
        compute_sample_embedding(a, out, save=False, **KW)
        with h5py.File(path, "r+") as f:
            del f["obs"].attrs["column-order"]  # unreadable / unexpected layout -> full re-write, not a skipped save
        assert save_embedding_to_h5ad(path, a) == "rewrite"
        with h5py.File(path, "r+") as f:  # leftovers of an interrupted update
            f["uns"].move("X_DR_sample", "__old_X_DR_sample")
            f["uns"].create_dataset("__new_X_DR_sample", data=[0])
        assert save_embedding_to_h5ad(path, a) == "in-place"
        with h5py.File(path, "r") as f:
            assert not any(k.startswith(("__new_", "__old_")) for k in f["uns"])
        pd.testing.assert_frame_equal(ad.read_h5ad(path).uns["X_DR_sample"], a.uns["X_DR_sample"])


def test_invalid_rmd_weight_rejected_early():
    from sampledisco.cli import validate_config
    from sampledisco.wrapper.wrapper import wrapper
    import yaml
    from importlib.resources import files
    cfg = yaml.safe_load((files("sampledisco") / "config" / "config_demo.yaml").read_text())
    for bad in ("eqaul", True, -1.0, 0, float("nan"), float("inf"), None):
        try:
            check_rmd_weight(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
        try:
            validate_config({**cfg, "rna_sample_embedding_rmd_weight": bad}, wrapper)
        except ValueError as e:
            assert "rna_sample_embedding_rmd_weight" in str(e)
        else:
            raise AssertionError(bad)
        try:  # use_gpu=True must raise, not fall back to a CPU rerun
            compute_sample_embedding(make_adata(n=300), "", use_gpu=True, save=False, rmd_weight=bad, **KW)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    for good in ("equal", 0.6, 1, np.float32(2.5)):
        check_rmd_weight(good)


def test_equal_rmd_weight_rule():
    for K_c, K_med, K_fine in ((5, 120, 300), (17, 120, 300), (4, 6, 9)):
        assert resolve_rmd_weight("equal", K_c, K_med, K_fine) == round(
            math.sqrt(K_fine / K_c + K_fine / K_med + 1.0), 2)
    assert resolve_rmd_weight(0.6, 5, 120, 300) == 0.6
    try:
        resolve_rmd_weight("auto", 5, 120, 300)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown rmd_weight string must raise")


def test_equal_default_matches_numeric_alpha():
    a = make_adata()
    compute_sample_embedding(a, "", save=False, **KW)  # default rmd_weight="equal"
    p = a.uns["sample_embedding_params"]
    alpha = resolve_rmd_weight("equal", p["K_c"], p["medium_K"], p["fine_K"])
    assert p["rmd_weight"] == alpha == p["block_weights"][3]
    eq = a.uns["X_DR_sample"].copy()
    compute_sample_embedding(a, "", save=False, rmd_weight=alpha, **KW)
    assert np.array_equal(eq.values, a.uns["X_DR_sample"].values)


def test_autotune_without_labels_uses_equal_default():
    from sampledisco.parameter_selection.autotune import run_autotune
    a = make_adata()
    run_autotune(a, "", sample_col="sample", celltype_col="cell_type", batch_col=None, medium_K=4, fine_K=6,
                       save=False, verbose=False)
    p = a.uns["sample_embedding_params"]
    assert p["best_params"]["rmd_weight"] == resolve_rmd_weight("equal", p["K_c"], p["K_med"], p["K_fine"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
