"""Harmony post-pass that produces the paper's ``Z_clust`` from ``Z_rmd``.

SampleDisco (Stage 2 of Fig. 1) uses two cell-level views:

  ``Z_rmd``   — sample-PRESERVED; preserves the per-sample variance the
                RMD displacement block needs.
  ``Z_clust`` — sample-REMOVED; shared cell-state geometry used by cell
                typing and the A1 / A2 / A3 composition blocks.

scGLUE produces a single embedding (``obsm['X_glue']``) which IS the
sample-preserved view, so it becomes ``Z_rmd`` once GLUE finishes. The
sample-removed ``Z_clust`` is produced by ONE of two paths:

  (default) Harmony post-pass on ``Z_rmd`` (this module) with the sample
            column (and batch column, if present) as batch_keys.
  (opt-in)  A second scGLUE training run with sample as use_batch,
            merged into the primary h5ads by ``multi_omics_glue.py``.
            When the 2-run output is present, the Harmony post-pass
            auto-skips.

Either way the result is written to ``obsm['Z_clust']`` and the upstream
``obsm['X_glue']`` is aliased to ``obsm['Z_rmd']`` so downstream code can
read paper-aligned keys uniformly.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from anndata import AnnData

from sampledisco.utils.harmony_compat import harmonize_embedding


# Paper-aligned obsm keys (Fig. 1 / Stage 2).
Z_RMD_KEY   = "Z_rmd"     # sample-PRESERVED → RMD role
Z_CLUST_KEY = "Z_clust"   # sample-REMOVED   → cluster / composition role

# scGLUE's own training output. Kept as a constant for internal callers
# (glue_train, the merge helper) that interface directly with scGLUE.
# Downstream pipelines should use the Z_* keys above.
XGLUE_KEY = "X_glue"


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
    out_key: str = Z_CLUST_KEY,
    use_gpu: bool = False,
    max_iter: int = 50,
    random_state: int = 0,
    verbose: bool = True,
) -> AnnData:
    """Run one Harmony pass on ``adata.obsm[in_key]`` (= Z_rmd / X_glue) to
    remove per-sample variance, writing ``adata.obsm[out_key]`` = Z_clust.

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

    n_cells, n_dims = adata.obsm[in_key].shape
    if verbose:
        print(f"[xglue-harmony] {out_key}: {n_cells:,} × {n_dims} dims, "
              f"batch_keys={batch_keys}, gpu={use_gpu}, max_iter={max_iter}")
    t0 = time.time()
    X_corr = harmonize_embedding(
        np.asarray(adata.obsm[in_key], dtype=np.float32),
        adata.obs,
        batch_key=batch_keys,
        max_iter_harmony=max_iter,
        use_gpu=use_gpu,
        seed=random_state,
    )
    adata.obsm[out_key] = np.asarray(X_corr, dtype=np.float32)
    # Alias the input as Z_rmd (sample-preserved) so downstream sees both
    # paper-aligned keys regardless of whether the cluster embedding came
    # from this Harmony pass or from a 2-run scGLUE merge.
    if Z_RMD_KEY != in_key and Z_RMD_KEY not in adata.obsm:
        adata.obsm[Z_RMD_KEY] = adata.obsm[in_key]
    if verbose:
        print(f"[xglue-harmony] {out_key}: done in {time.time() - t0:.1f}s "
              f"(also aliased obsm[{in_key!r}] → obsm[{Z_RMD_KEY!r}])")
    return adata
