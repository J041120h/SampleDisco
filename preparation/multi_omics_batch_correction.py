"""Harmony post-pass on the GLUE embedding (`X_glue`).

scGLUE handles cross-modality alignment and batch removal in one step. Its
output ``adata.obsm['X_glue']`` is batch-removed but sample-preserved — it
serves directly as the per-(sample, modality) CMD displacement embedding
downstream.

For the sample-removed embedding required by cell typing + A1/A2/A3
composition blocks, ``harmonize_xglue`` runs a single Harmony iteration on
``X_glue`` with ``sample_col`` (and ``batch_col`` if available) as
batch_keys, writing ``adata.obsm['X_glue_harmony']``. Off by default;
opt-in via the wrapper's ``glue_batch_correction=True``.

Alternative source for ``X_glue_harmony`` (also opt-in): a second scGLUE
training run configured with ``treat_sample_as_batch=True``, merged into
the primary RNA + ATAC h5ads by ``multi_omics_glue.py``. Both paths
produce the same obsm key and downstream code treats them identically.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from anndata import AnnData


XGLUE_KEY         = "X_glue"          # sample-preserved → CMD role
XGLUE_HARMONY_KEY = "X_glue_harmony"  # sample-removed   → cluster role


def _has_signal(adata: AnnData, col: Optional[str]) -> bool:
    """True when `col` is in obs and has ≥ 2 unique values."""
    if not col or col not in adata.obs.columns:
        return False
    return adata.obs[col].astype(str).nunique() >= 2


def harmonize_xglue(
    adata: AnnData,
    *,
    sample_col: str,
    batch_col: Optional[str] = None,
    in_key: str = XGLUE_KEY,
    out_key: str = XGLUE_HARMONY_KEY,
    use_gpu: bool = False,
    max_iter: int = 50,
    random_state: int = 0,
    verbose: bool = True,
) -> AnnData:
    """Run one Harmony pass on ``adata.obsm[in_key]`` to remove per-sample
    variance, writing ``adata.obsm[out_key]`` — the sample-removed cluster
    / composition embedding.

    ``batch_keys`` defaults to ``[sample_col]``. When ``batch_col`` is
    present with ≥ 2 levels, it is added (belt-and-suspenders against any
    residual batch structure GLUE left behind).

    No-op (with a log line) when ``in_key`` is missing or when
    ``sample_col`` has fewer than 2 levels.
    """
    if in_key not in adata.obsm:
        if verbose:
            print(f"[xglue-harmony] {in_key!r} missing from obsm — skipping.")
        return adata
    if not _has_signal(adata, sample_col):
        if verbose:
            print(f"[xglue-harmony] sample_col={sample_col!r} has <2 levels — skipping.")
        return adata

    batch_keys: List[str] = (
        [batch_col, sample_col] if _has_signal(adata, batch_col) else [sample_col]
    )

    from harmony import harmonize

    n_cells, n_dims = adata.obsm[in_key].shape
    if verbose:
        print(f"[xglue-harmony] {out_key}: {n_cells:,} × {n_dims} dims, "
              f"batch_keys={batch_keys}, gpu={use_gpu}, max_iter={max_iter}")
    t0 = time.time()
    X_corr = harmonize(
        np.asarray(adata.obsm[in_key], dtype=np.float32),
        adata.obs,
        batch_key=batch_keys if len(batch_keys) > 1 else batch_keys[0],
        max_iter_harmony=max_iter,
        use_gpu=use_gpu,
        random_state=random_state,
        verbose=verbose,
    )
    adata.obsm[out_key] = np.asarray(X_corr, dtype=np.float32)
    if verbose:
        print(f"[xglue-harmony] {out_key}: done in {time.time() - t0:.1f}s")
    return adata
