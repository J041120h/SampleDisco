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
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from sampledisco.sample_embedding.blocks import (
    assemble_units,
    clr_transform,
    composition_per_unit,
    composite_batch_labels,
    derive_weights,
    frobenius_stack,
    loo_rmd,
    regress_out_batch_linear,
)
from sampledisco.sample_embedding.sample_embedding import (
    _aggregate_obs,
    _resolve_rmd_emb_key,
)


def _gpu_kmeans_soft(Z_np: np.ndarray, K: int, seed: int):
    """GPU MiniBatchKMeans + RBF soft assignment.

    Returns:
        soft (np.ndarray, n_cells × K) — soft assignment probabilities
    """
    import cupy as cp
    try:
        from cuml.cluster import MiniBatchKMeans as cuMiniBatchKMeans
        Z_gpu = cp.asarray(Z_np)
        km = cuMiniBatchKMeans(n_clusters=K, random_state=seed,
                                batch_size=4096, n_init=5, max_iter=200)
        km.fit(Z_gpu)
        centers = cp.asarray(km.cluster_centers_)
    except Exception:
        # cuml unavailable: sklearn k-means on CPU, then move data to GPU for soft-assign
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=K, random_state=seed,
                              batch_size=4096, n_init=5, max_iter=200).fit(Z_np)
        Z_gpu = cp.asarray(Z_np)
        centers = cp.asarray(km.cluster_centers_)

    # Pairwise sq distances on GPU
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
    cluster_emb_key: str = "Z_clust",
    rmd_emb_key: Optional[str] = None,
    modality_col: Optional[str] = None,
    batch_col: Optional[Union[str, List[str]]] = None,
    medium_K: int = 120,
    fine_K: int = 300,
    rmd_dim_per_cluster: int = 8,
    use_clr: bool = False,
    use_rmd: bool = True,
    block_weights: Optional[List[float]] = None,
    rmd_weight: float = 0.60,
    pca_components: int = 10,
    batch_method: str = "harmony",
    save: bool = True,
    verbose: bool = True,
    seed: int = 42,
) -> AnnData:
    """GPU compute_sample_embedding — see CPU version for full docstring."""
    start_time = time.time() if verbose else None

    if cluster_emb_key not in adata.obsm:
        raise KeyError(
            f"cluster_emb_key '{cluster_emb_key}' not in adata.obsm")
    if celltype_col not in adata.obs.columns:
        raise KeyError(
            f"celltype_col '{celltype_col}' not in adata.obs")

    rmd_key = _resolve_rmd_emb_key(adata, cluster_emb_key, rmd_emb_key)
    if rmd_key not in adata.obsm:
        raise KeyError(f"rmd_emb_key '{rmd_key}' not in adata.obsm")
    if verbose:
        print(f"[sample_embedding_gpu] cluster_emb={cluster_emb_key}, "
              f"rmd_emb={rmd_key}")

    # primary_batch: first col → assemble_units (group labelling); batch_cols_multi → Harmony multi-cov
    if isinstance(batch_col, (list, tuple)):
        batch_cols_multi = [c for c in batch_col if c]
    elif batch_col:
        batch_cols_multi = [batch_col]
    else:
        batch_cols_multi = []
    primary_batch = batch_cols_multi[0] if batch_cols_multi else None

    units, unit_cellids, unit_ids, unit_groups, unit_batches, all_cellids, Z_clust = \
        assemble_units(adata, sample_col, cluster_emb_key,
                       modality_col=modality_col, batch_col=primary_batch)
    n_units = len(units)
    if n_units < 2:
        raise ValueError(f"need ≥2 units, got {n_units}")
    cellid_idx = {cid: i for i, cid in enumerate(all_cellids)}
    if verbose:
        print(f"[sample_embedding_gpu] {n_units} units; "
              f"{Z_clust.shape[0]} cells; cluster_emb dim={Z_clust.shape[1]}")

    cell_type = adata.obs[celltype_col].astype(str).values
    unique_cts = sorted(set(cell_type))
    K_c = len(unique_cts)
    if K_c < 2:
        raise ValueError(f"need ≥2 cell types, got {K_c}")

    # ---- A1: coarse cell-type composition (one-hot, mean per unit) ----------
    L1 = {ct: i for i, ct in enumerate(unique_cts)}
    soft1 = np.zeros((Z_clust.shape[0], K_c), dtype=np.float32)
    for i, ct in enumerate(cell_type):
        soft1[i, L1[ct]] = 1.0
    unit_cellids_list = [unit_cellids[uid] for uid in unit_ids]
    A1 = composition_per_unit(unit_cellids_list, soft1, cellid_idx)
    if use_clr:
        A1 = clr_transform(A1)
    if verbose:
        print(f"[A1] shape={A1.shape}")

    # ---- A2: soft k-means at K_med (GPU) ----
    K_med = min(medium_K, max(2, Z_clust.shape[0] // 200))
    if verbose:
        print(f"[A2] GPU MiniBatchKMeans K={K_med}...", flush=True)
    soft2 = _gpu_kmeans_soft(Z_clust, K_med, seed)
    A2 = composition_per_unit(unit_cellids_list, soft2, cellid_idx)
    if use_clr:
        A2 = clr_transform(A2)
    if verbose:
        print(f"[A2] shape={A2.shape}")

    # ---- A3: soft k-means at K_fine (GPU) ----
    K_fine = min(fine_K, max(2, Z_clust.shape[0] // 100))
    if verbose:
        print(f"[A3] GPU MiniBatchKMeans K={K_fine}...", flush=True)
    soft3 = _gpu_kmeans_soft(Z_clust, K_fine, seed + 1)
    A3 = composition_per_unit(unit_cellids_list, soft3, cellid_idx)
    if use_clr:
        A3 = clr_transform(A3)
    if verbose:
        print(f"[A3] shape={A3.shape}")

    blocks = [A1, A2, A3]

    # ---- RMD: per-(group, coarse cluster) LOO displacement — CPU (per-cluster PCA is small) ----
    if use_rmd:
        if verbose:
            print(f"[RMD] LOO displacement on rmd_emb...", flush=True)
        Z_rmd = np.asarray(adata.obsm[rmd_key], dtype=np.float32)
        rmd_units = []
        for uid, group in zip(unit_ids, unit_groups):
            cids = unit_cellids[uid]
            idxs = [cellid_idx[c] for c in cids if c in cellid_idx]
            rmd_units.append((uid, group, Z_rmd[idxs]))
        coarse_label_map = dict(zip(all_cellids, cell_type))
        RMD = loo_rmd(
            rmd_units, unit_cellids, coarse_label_map,
            max_dim_per_cluster=rmd_dim_per_cluster, seed=seed, loo=True,
            verbose=verbose,
        )
        if RMD.shape[1] > 0:
            blocks.append(RMD)

    # ---- Weights (auto-scaled by K_c/K_med/K_fine when not overridden) ----
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
    from sampledisco.sample_embedding.blocks import build_harmony_meta_df
    multi_meta_df = (
        build_harmony_meta_df(adata, unit_cellids, unit_ids, batch_cols_multi)
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
        "cluster_emb_key": str(cluster_emb_key),
        "rmd_emb_key": str(rmd_key),
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
        preprocessed_h5 = os.path.join(output_dir, "preprocess", "adata_preprocessed.h5ad")
        if os.path.exists(preprocessed_h5):
            try:
                sc.write(preprocessed_h5, adata)
            except Exception as exc:
                if verbose:
                    print(f"[sample_embedding_gpu] WARNING: could not re-save "
                          f"{preprocessed_h5}: {exc}")
        if verbose:
            print(f"[sample_embedding_gpu] wrote {emb_csv}")
            if os.path.exists(preprocessed_h5):
                print(f"[sample_embedding_gpu] updated {preprocessed_h5} (.uns['X_DR_sample'])")

    if verbose and start_time is not None:
        print(f"[sample_embedding_gpu] done in {time.time() - start_time:.2f}s; "
              f"shape={emb_df.shape}")

    return adata
