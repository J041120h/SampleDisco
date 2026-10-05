"""The index-based sample-embedding helpers reproduce the dict-based ones exactly, and the
save_cell_adata switch leaves the cell-level h5ad untouched. Synthetic data, CPU only.
Run with `pytest tests/` or `python tests/test_sample_embedding_equivalence.py`."""
import os
import tempfile
import warnings

import anndata as ad
import h5py
import numpy as np
import pandas as pd

from sampledisco.sample_embedding import compute_sample_embedding
from sampledisco.sample_embedding.blocks import (
    assemble_units, composition_per_unit, loo_rmd, loo_rmd_from_index, soft_assign,
    soft_composition, sorted_codes, unit_index)


def make_adata(multi=False, n=3001, seed=0):
    rng = np.random.default_rng(seed)
    if multi:  # sample ids with and without modality suffixes, upper- and lower-case (no id collisions)
        samples = rng.choice(["D1_RNA", "D1_ATAC", "D2_rna", "D2_atac", "D3", "D4"], n)
        modality = np.array(["RNA" if s.lower().endswith("_rna") else "ATAC" if s.lower().endswith("_atac")
                             else rng.choice(["RNA", "ATAC"]) for s in samples])
    else:
        samples = rng.choice([f"S{i}" for i in range(1, 14)], n)
        modality = None
    batch = pd.Series(rng.choice(["b1", "b2", "b3"], n), dtype=object)
    batch[rng.random(n) < 0.05] = np.nan
    ct = pd.Categorical(rng.choice(["T", "B", "NK", "Mono", "10"], n),
                        categories=["unused", "T", "B", "NK", "Mono", "10"])
    obs = pd.DataFrame({"sample": pd.Categorical(samples), "batch": batch.values, "cell_type": ct},
                       index=[f"c{i}" for i in range(n)])
    if multi:
        obs["modality"] = pd.Categorical(modality)
    a = ad.AnnData(obs=obs)
    a.obsm["Z_comp"] = rng.normal(size=(n, 10)).astype(np.float32)
    a.obsm["Z_rmd"] = rng.normal(size=(n, 10)).astype(np.float32)
    return a


def test_sorted_codes_matches_np_unique():
    a = make_adata()
    for col in ("sample", "batch", "cell_type"):
        lab, codes = sorted_codes(a.obs[col])
        ref_lab, ref_codes = np.unique(a.obs[col].astype(str).values, return_inverse=True)
        assert list(lab) == list(ref_lab) and np.array_equal(codes, ref_codes), col


def check_units(a, modality_col):
    _, old_cids, old_ids, old_groups, old_batches, _, _ = assemble_units(
        a, "sample", "Z_comp", modality_col=modality_col, batch_col="batch")
    ids, groups, batches, unit_of_cell, rows = unit_index(a, "sample", modality_col=modality_col,
                                                          batch_col="batch")
    assert ids == old_ids and groups == old_groups and batches == old_batches
    names = a.obs_names.values
    for uid, r in zip(ids, rows):
        assert list(names[r]) == old_cids[uid]
    assert (unit_of_cell >= 0).all()


def test_unit_index_single_and_multi():
    check_units(make_adata(), None)
    check_units(make_adata(multi=True), "modality")


def test_loo_rmd_from_index_bit_identical():
    for multi in (False, True):
        a = make_adata(multi=multi)
        mcol = "modality" if multi else None
        _, cids, ids, groups, _, all_ids, _ = assemble_units(a, "sample", "Z_comp", modality_col=mcol,
                                                             batch_col="batch")
        idx = {c: i for i, c in enumerate(all_ids)}
        Z = a.obsm["Z_rmd"]
        units = [(u, g, Z[[idx[c] for c in cids[u]]]) for u, g in zip(ids, groups)]
        old = loo_rmd(units, cids, dict(zip(all_ids, a.obs["cell_type"].astype(str).values)), seed=42)
        ids2, groups2, _, unit_of_cell, _ = unit_index(a, "sample", modality_col=mcol, batch_col="batch")
        lab, codes = sorted_codes(a.obs["cell_type"])
        new = loo_rmd_from_index(Z, unit_of_cell, codes, len(lab), groups2, seed=42)
        assert np.array_equal(old, new)


def test_soft_composition_bit_identical_odd_and_even():
    for n, K in ((3001, 7), (3001, 8), (3000, 7)):  # n*K odd, even, even
        a = make_adata(n=n)
        Z = a.obsm["Z_comp"]
        C = Z[np.random.default_rng(1).choice(n, K, replace=False)].copy()
        _, cids, ids, _, _, all_ids, _ = assemble_units(a, "sample", "Z_comp", batch_col="batch")
        old = composition_per_unit([cids[u] for u in ids], soft_assign(Z, C),
                                   {c: i for i, c in enumerate(all_ids)})
        _, _, _, _, rows = unit_index(a, "sample", batch_col="batch")
        for threads in (1, 4):
            assert np.array_equal(old, soft_composition(Z, C, rows, n_threads=threads)), (n, K, threads)


def test_kmeans_centers_pair_matches_serial_fits():
    from sampledisco.sample_embedding.blocks import kmeans_centers, kmeans_centers_pair
    Z = np.random.default_rng(3).normal(size=(5000, 8)).astype(np.float32)
    pair = kmeans_centers_pair(Z, 6, 12, 0, n_threads=4)
    assert np.array_equal(pair[0], kmeans_centers(Z, 6, 0)) and np.array_equal(pair[1], kmeans_centers(Z, 12, 1))


def _write_old_cell_h5ad(out):
    a = make_adata()
    a.uns["X_DR_sample"] = pd.DataFrame({"PC1": [1.0]}, index=["old"])
    path = os.path.join(out, "preprocess", "adata_preprocessed.h5ad")
    os.makedirs(os.path.dirname(path))
    a.write_h5ad(path)
    os.utime(path, (1_000_000_000, 1_000_000_000))
    return a, path


def _old_uns_intact(path):
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem
    if os.path.getmtime(path) != 1_000_000_000:
        return False
    with h5py.File(path, "r") as f:
        return read_elem(f["uns/X_DR_sample"]).index.tolist() == ["old"]


def test_save_cell_adata_false_leaves_h5ad_untouched():
    kw = dict(sample_col="sample", celltype_col="cell_type", batch_col="batch", medium_K=4, fine_K=6,
              verbose=False)
    with tempfile.TemporaryDirectory() as out:
        a, path = _write_old_cell_h5ad(out)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_sample_embedding(a, out, save=True, save_cell_adata=False, **kw)
        assert _old_uns_intact(path)
        assert any("OLDER .uns['X_DR_sample']" in str(x.message) for x in w)
        assert os.path.exists(os.path.join(out, "sample_embedding", "sample_embedding.csv"))
        compute_sample_embedding(a, out, save=True, save_cell_adata=True, **kw)
        assert os.path.getmtime(path) != 1_000_000_000


def test_run_autotune_save_cell_adata_false():
    from sampledisco.parameter_selection.autotune import run_autotune
    with tempfile.TemporaryDirectory() as out:
        a, path = _write_old_cell_h5ad(out)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_autotune(a, out, sample_col="sample", celltype_col="cell_type", batch_col="batch",
                         medium_K=4, fine_K=6, search="grid", save=True, save_cell_adata=False, verbose=False)
        assert "X_DR_sample" in a.uns and _old_uns_intact(path)
        assert any("OLDER .uns['X_DR_sample']" in str(x.message) for x in w)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
