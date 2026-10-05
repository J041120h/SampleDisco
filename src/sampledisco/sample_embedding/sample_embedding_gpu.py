"""GPU sample-embedding entry point.

Same API as `sample_embedding.compute_sample_embedding`, but the hot
primitives (k-means, RBF soft-assign, final-stack PCA) run on the GPU via
`cuml` / `cupy` / `rapids_singlecell`. The sample-level Harmony step runs on
the CPU via `harmonypy` — identical to `sample_embedding.py` — because that
matrix is tiny (n_units x PCs) and a GPU Harmony gives no speedup; using the
same implementation keeps the GPU and CPU sample embeddings consistent.

The recipe is identical to `sample_embedding.py` — only the array backend
for the heavy intermediate computations changes.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from sampledisco.sample_embedding.blocks import (
    clr_transform,
    composite_batch_labels,
    composition_from_rows,
    derive_weights,
    frobenius_stack,
    harmony_meta_from_index,
    kmeans_centers_pair,
    loo_rmd_from_index,
    n_worker_threads,
    resolve_rmd_weight,
    save_embedding_to_h5ad,
    regress_out_batch_linear,
    sorted_codes,
    unit_index,
    warn_if_stale_embedding,
)
from sampledisco.sample_embedding.sample_embedding import (
    _aggregate_obs,
    _resolve_rmd_emb_key,
)
from sampledisco.utils.embedding_keys import resolve_comp_key, resolve_rmd_key


def _gpu_kmeans_centers_pair(Z_np: np.ndarray, Z_gpu, K_med: int, K_fine: int, seed: int,
                             n_threads: int):
    """MiniBatchKMeans centres at K_med (seed) and K_fine (seed + 1): cuML when it provides
    MiniBatchKMeans, otherwise sklearn on the CPU (the two fits run concurrently there)."""
    import cupy as cp
    try:
        from cuml.cluster import MiniBatchKMeans as cuMiniBatchKMeans
        centers = []
        for K, sd in ((K_med, seed), (K_fine, seed + 1)):
            km = cuMiniBatchKMeans(n_clusters=K, random_state=sd,
                                   batch_size=4096, n_init=5, max_iter=200)
            km.fit(Z_gpu)
            centers.append(cp.asarray(km.cluster_centers_))
        return centers
    except Exception:
        return [cp.asarray(c) for c in kmeans_centers_pair(Z_np, K_med, K_fine, seed, n_threads)]


def _gpu_soft(Z_gpu, centers):
    """RBF soft assignment on the GPU; returns the n_cells x K matrix as numpy."""
    import cupy as cp
    Z_sq = (Z_gpu * Z_gpu).sum(axis=1, keepdims=True)
    A_sq = (centers * centers).sum(axis=1, keepdims=True).T
    D2 = Z_sq + A_sq - 2.0 * (Z_gpu @ centers.T)
    D2 = cp.maximum(D2, 0)
    D = cp.sqrt(D2)
    sigma = float(cp.median(D).get())
    logits = -D2 / (2.0 * sigma * sigma + 1e-12)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = cp.exp(logits)
    soft_gpu = e / cp.maximum(e.sum(axis=1, keepdims=True), 1e-12)
    return cp.asnumpy(soft_gpu)


def _gpu_pca(F_np: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    """GPU PCA on the Frobenius-stack matrix (samples × features)."""
    try:
        import cupy as cp
        from cuml.decomposition import PCA as cuPCA
        F_gpu = cp.asarray(F_np)
        pca = cuPCA(n_components=n_components, random_state=seed)
        Fp_gpu = pca.fit_transform(F_gpu)
        return cp.asnumpy(Fp_gpu)
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=seed).fit_transform(F_np)


def _gpu_harmonize(
    Fp: np.ndarray,
    unit_ids: List[str],
    batch_labels,
    n_units: int,
    seed: int = 42,
    verbose: bool = False,
    multi_meta_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Sample-level batch correction via `harmonypy` — the SAME implementation
    the CPU path uses (`blocks.build_emb_from_blocks`).

    The input is the sample-level matrix (n_units x PCs), which is tiny, so a GPU
    Harmony gives no speedup here; running the identical `harmonypy` correction
    instead keeps the GPU and CPU sample embeddings consistent. (harmony-pytorch's
    `harmonize` defaults n_clusters to int(N/30) = 0 for small sample counts and
    crashes, which previously forced a silent fall back to a *different* estimator
    on GPU while CPU ran real Harmony.)

    Two modes:
      - Single-batch (backward compatible): `batch_labels` is a list of per-unit
        strings; Harmony runs with `batch_key="batch"`.
      - Multi-covariate: `multi_meta_df` (one col per covariate, indexed by
        unit_ids); Harmony runs with `batch_key=list(meta.columns)`.
    """
    if multi_meta_df is not None and len(multi_meta_df.columns) >= 1:
        meta = multi_meta_df.copy()
        meta.index = pd.Index(unit_ids, name="sample")
        batch_keys = list(meta.columns)
    else:
        meta = pd.DataFrame({"batch": batch_labels}, index=pd.Index(unit_ids, name="sample"))
        batch_keys = "batch"

    try:
        import harmonypy as hm
        nclust = max(2, min(meta.nunique().max() if isinstance(batch_keys, list)
                            else len(set(meta["batch"])),
                            n_units // 2))
        ho = hm.run_harmony(np.asarray(Fp, dtype=np.float32), meta,
                            batch_keys,  # str OR list[str]
                            nclust=nclust,
                            max_iter_harmony=30,
                            random_state=seed)
        Zc = ho.Z_corr
        if Zc.shape[0] != n_units:
            Zc = Zc.T
        return np.asarray(Zc, dtype=np.float32)
    except Exception as exc:
        print(f"  [Harmony] FAILED ({exc!r}); trying linear batch regression "
              f"before falling back to raw PCA")
        try:
            collapsed = (meta[batch_keys].astype(str).agg("__".join, axis=1).values
                         if isinstance(batch_keys, list) else meta["batch"].values)
            return np.asarray(regress_out_batch_linear(Fp, collapsed), dtype=np.float32)
        except Exception as exc2:
            print(f"  [Harmony] linear regression fallback FAILED ({exc2!r}); using raw PCA "
                  f"— sample embedding will NOT be batch-corrected")
            return np.asarray(Fp, dtype=np.float32)


def compute_sample_embedding(
    adata: AnnData,
    output_dir: str,
    *,
    sample_col: str = "sample",
    celltype_col: str = "cell_type",
    comp_emb_key: Optional[str] = None,
    rmd_emb_key: Optional[str] = None,
    modality_col: Optional[str] = None,
    batch_col: Optional[Union[str, List[str]]] = None,
    medium_K: int = 120,
    fine_K: int = 300,
    rmd_dim_per_cluster: int = 8,
    use_clr: bool = False,
    use_rmd: bool = True,
    block_weights: Optional[List[float]] = None,
    rmd_weight: Union[float, str] = "equal",
    pca_components: int = 10,
    batch_method: str = "harmony",
    save: bool = True,
    verbose: bool = True,
    seed: int = 42,
    cluster_emb_key: Optional[str] = None,
    save_cell_adata: bool = True,
) -> AnnData:
    """GPU compute_sample_embedding — see CPU version for full docstring."""
    start_time = time.time() if verbose else None

    if cluster_emb_key is not None:
        warnings.warn("cluster_emb_key= is deprecated and will be removed in 1.0; "
                      "use comp_emb_key=.", FutureWarning, stacklevel=2)
        if comp_emb_key is None:
            comp_emb_key = cluster_emb_key

    comp_key = resolve_comp_key(adata, comp_emb_key,
                                context="compute_sample_embedding_gpu")
    if celltype_col not in adata.obs.columns:
        raise KeyError(
            f"celltype_col '{celltype_col}' not in adata.obs")

    rmd_key = resolve_rmd_key(adata, rmd_emb_key, comp_key=comp_key,
                              required=use_rmd,
                              context="compute_sample_embedding_gpu")
    if verbose:
        print(f"[sample_embedding_gpu] comp_emb={comp_key}, rmd_emb={rmd_key}")

    # primary_batch: first col → unit_index (group labelling); batch_cols_multi → Harmony multi-cov
    if isinstance(batch_col, (list, tuple)):
        batch_cols_multi = [c for c in batch_col if c]
    elif batch_col:
        batch_cols_multi = [batch_col]
    else:
        batch_cols_multi = []
    primary_batch = batch_cols_multi[0] if batch_cols_multi else None

    unit_ids, unit_groups, unit_batches, unit_of_cell, unit_rows = unit_index(
        adata, sample_col, modality_col=modality_col, batch_col=primary_batch)
    Z_comp = np.asarray(adata.obsm[comp_key], dtype=np.float32)
    n_units = len(unit_ids)
    if n_units < 2:
        raise ValueError(f"need ≥2 units, got {n_units}")
    if verbose:
        print(f"[sample_embedding_gpu] {n_units} units; "
              f"{Z_comp.shape[0]} cells; comp_emb dim={Z_comp.shape[1]}")

    unique_cts, ct_codes = sorted_codes(adata.obs[celltype_col])
    K_c = len(unique_cts)
    if K_c < 2:
        raise ValueError(f"need ≥2 cell types, got {K_c}")

    # ---- A1: coarse cell-type composition (one-hot, mean per unit) ----------
    soft1 = np.zeros((Z_comp.shape[0], K_c), dtype=np.float32)
    soft1[np.arange(Z_comp.shape[0]), ct_codes] = 1.0
    A1 = composition_from_rows(unit_rows, soft1)
    del soft1
    if use_clr:
        A1 = clr_transform(A1)
    if verbose:
        print(f"[A1] shape={A1.shape}")

    # ---- A2 / A3: k-means centres, soft assignment on the GPU ----
    import cupy as cp
    K_med = min(medium_K, max(2, Z_comp.shape[0] // 200))
    K_fine = min(fine_K, max(2, Z_comp.shape[0] // 100))
    if verbose:
        print(f"[A2/A3] MiniBatchKMeans K={K_med} and K={K_fine}; GPU soft assignment...", flush=True)
    Z_gpu = cp.asarray(Z_comp)
    C_med, C_fine = _gpu_kmeans_centers_pair(Z_comp, Z_gpu, K_med, K_fine, seed,
                                             n_worker_threads())
    A2 = composition_from_rows(unit_rows, _gpu_soft(Z_gpu, C_med))
    A3 = composition_from_rows(unit_rows, _gpu_soft(Z_gpu, C_fine))
    del Z_gpu
    if use_clr:
        A2 = clr_transform(A2)
        A3 = clr_transform(A3)
    if verbose:
        print(f"[A2] shape={A2.shape}")
        print(f"[A3] shape={A3.shape}")

    blocks = [A1, A2, A3]

    # ---- RMD: per-(group, coarse cluster) LOO displacement — CPU (per-cluster PCA is small) ----
    RMD = np.empty((n_units, 0), dtype=np.float32)
    if use_rmd:
        if verbose:
            print(f"[RMD] LOO displacement on rmd_emb...", flush=True)
        Z_rmd = np.asarray(adata.obsm[rmd_key], dtype=np.float32)
        RMD = loo_rmd_from_index(
            Z_rmd, unit_of_cell, ct_codes, K_c, unit_groups,
            max_dim_per_cluster=rmd_dim_per_cluster, seed=seed, loo=True,
            verbose=verbose,
        )
        if RMD.shape[1] > 0:
            blocks.append(RMD)

    # ---- Weights (auto-scaled by K_c/K_med/K_fine when not overridden) ----
    rmd_weight = resolve_rmd_weight(rmd_weight, K_c, K_med, K_fine)
    if block_weights is None:
        weights = derive_weights(K_c, K_med, K_fine,
                                   rmd_weight=rmd_weight,
                                   n_blocks=len(blocks))
    else:
        if len(block_weights) != len(blocks):
            raise ValueError(
                f"block_weights length {len(block_weights)} != blocks {len(blocks)}")
        weights = list(block_weights)
    if verbose:
        print(f"[sample_embedding_gpu] weights={[round(w, 3) for w in weights]}")

    # ---- Final: Frobenius stack + GPU PCA + sample-level Harmony ----
    F = frobenius_stack(blocks, weights)
    n_pc_full = min(pca_components, F.shape[0] - 1, F.shape[1])
    if n_pc_full < 1:
        raise ValueError(
            f"insufficient data for PCA (shape={F.shape}, requested {pca_components})")
    Fp = _gpu_pca(F, n_pc_full, seed)

    # Sample-level Harmony — multi-covariate when >=2 batch_cols given, else legacy
    multi_meta_df = (
        harmony_meta_from_index(adata, unit_of_cell, unit_ids, batch_cols_multi)
        if len(batch_cols_multi) >= 2 else None
    )

    batch_labels, used_composite = composite_batch_labels(unit_groups, unit_batches)
    if multi_meta_df is not None:
        per_col = ", ".join([f"{c}={multi_meta_df[c].nunique()}" for c in multi_meta_df.columns])
        if verbose:
            print(f"  [batch correction] multi-covariate Harmony: keys={list(multi_meta_df.columns)} ({per_col})")
        do_harmony = n_units >= 8 and any(multi_meta_df[c].nunique() > 1 for c in multi_meta_df.columns)
    else:
        if verbose:
            tag = "composite (group+batch)" if used_composite else "group only"
            print(f"  [batch correction] {len(set(batch_labels))} groups ({tag})")
        do_harmony = len(set(batch_labels)) > 1 and n_units >= 8

    if do_harmony and batch_method == "none":
        # explicit no-op: skip sample-level batch correction, keep raw PCA
        Zc = Fp
    elif do_harmony:
        if batch_method == "linear":
            collapsed = (multi_meta_df.astype(str).agg("__".join, axis=1).values
                         if multi_meta_df is not None else batch_labels)
            Zc = regress_out_batch_linear(Fp, collapsed)
        else:
            Zc = _gpu_harmonize(Fp, unit_ids, batch_labels,
                                n_units=n_units, seed=seed, verbose=verbose,
                                multi_meta_df=multi_meta_df)
    else:
        Zc = Fp

    emb_df = pd.DataFrame(
        np.asarray(Zc, dtype=np.float32),
        index=pd.Index(unit_ids, name="sample"),
        columns=[f"PC{i+1}" for i in range(Zc.shape[1])],
    )

    adata.uns["X_DR_sample"] = emb_df.copy()
    adata.uns["sample_embedding_params"] = {
        "medium_K": int(K_med),
        "fine_K": int(K_fine),
        "K_c": int(K_c),
        "use_clr": bool(use_clr),
        "use_rmd": bool(use_rmd),
        "rmd_weight": float(rmd_weight),
        "block_weights": list(map(float, weights)),
        "pca_components": int(pca_components),
        "batch_method": str(batch_method),
        "comp_emb_key": str(comp_key),
        "rmd_emb_key": str(rmd_key),
        # Deprecated 0.2.0 spelling, kept for readers of this dict; removed in 1.0.
        "cluster_emb_key": str(comp_key),
        "modality_col": str(modality_col) if modality_col else "",
        "batch_col": str(primary_batch) if primary_batch else "",
        "batch_cols_multi": list(batch_cols_multi),
        "seed": int(seed),
        "backend": "gpu",
    }

    if save:
        out_dir = os.path.join(output_dir, "sample_embedding")
        os.makedirs(out_dir, exist_ok=True)
        emb_csv = os.path.join(out_dir, "sample_embedding.csv")
        emb_df.to_csv(emb_csv)
        blocks_npz = os.path.join(out_dir, "sample_embedding_blocks.npz")
        np.savez_compressed(
            blocks_npz,
            unit_ids=np.asarray(unit_ids, dtype=str),
            A1_cell_types=np.asarray(unique_cts, dtype=str),
            A1=A1,
            A2=A2,
            A3=A3,
            RMD=RMD,
        )
        preprocessed_h5 = os.path.join(output_dir, "preprocess", "adata_preprocessed.h5ad")
        resave = save_cell_adata and os.path.exists(preprocessed_h5)
        if not save_cell_adata:
            warn_if_stale_embedding(preprocessed_h5, emb_csv)
        if resave:
            try:
                how = save_embedding_to_h5ad(preprocessed_h5, adata)
                if verbose:
                    print(f"[sample_embedding_gpu] updated {preprocessed_h5} (.uns['X_DR_sample'], {how})")
            except Exception as exc:
                if verbose:
                    print(f"[sample_embedding_gpu] WARNING: could not re-save "
                          f"{preprocessed_h5}: {exc}")
        if verbose:
            print(f"[sample_embedding_gpu] wrote {emb_csv}")
            print(f"[sample_embedding_gpu] wrote {blocks_npz}")

    if verbose and start_time is not None:
        print(f"[sample_embedding_gpu] done in {time.time() - start_time:.2f}s; "
              f"shape={emb_df.shape}")

    return adata
