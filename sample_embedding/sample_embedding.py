"""CPU sample-embedding entry point.

`compute_sample_embedding(adata, ...)` takes a cell-level AnnData with a
cluster-emb obsm key (and optionally a sample-preserved emb for RMD) and
**mutates it in place**: the sample-level embedding is written back to the
cell-level adata under ``.uns['X_DR_sample']`` (DataFrame, samples × PCs) and
``.uns['sample_embedding_params']`` (dict). No separate sample-level h5ad is
written; the cell-level ``adata_preprocessed.h5ad`` is re-saved with the new
``.uns`` keys. The embedding is also written to a human-readable CSV.

The function returns the cell-level adata (mutated). Use
``build_sample_adata`` to materialize a small sample-level AnnData in memory
when downstream code requires per-sample obs (e.g. ``pseudo_adata``).

Algorithm (matches the singleRMD recipe; no CLR by default):
    1. A1  — coarse cell-type composition (one-hot, mean per unit)
    2. A2  — soft k-means composition at K_med
    3. A3  — soft k-means composition at K_fine
    4. RMD — per-(group, cluster) LOO displacement from the rmd_emb;
             group = modality_col (MO) or batch_col (single-omics).
    5. Frobenius weighted stack → PCA → composite-batch Harmony.

When `block_weights=None`, weights are auto-rescaled via inverse-variance
schedule using the actual K_c, K_med, K_fine values so user-modified cluster
counts don't drift the relative balance among blocks.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.cluster import MiniBatchKMeans

from sample_embedding.blocks import (
    assemble_units,
    build_emb_from_blocks,
    clr_transform,
    composition_per_unit,
    derive_weights,
    loo_rmd,
    soft_assign,
)


def _resolve_rmd_emb_key(adata, cluster_emb_key: str,
                          rmd_emb_key: Optional[str]) -> str:
    if rmd_emb_key is not None and rmd_emb_key in adata.obsm:
        return rmd_emb_key
    if "Z_rmd" in adata.obsm:
        return "Z_rmd"
    return cluster_emb_key


def build_sample_adata(adata, sample_col: str = "sample",
                        modality_col: Optional[str] = None) -> AnnData:
    """Materialize an in-memory sample-level AnnData from a cell-level adata.

    Reads the sample-level embedding from ``adata.uns['X_DR_sample']`` and
    aggregates per-unit metadata from ``adata.obs``. The result is suitable
    for the downstream consumers (sample_distance, sample_association,
    sample_trajectory, etc.) that expect one row per sample (or per
    sample × modality for multi-omics).
    """
    if "X_DR_sample" not in adata.uns:
        raise KeyError(
            "adata.uns['X_DR_sample'] not found — run compute_sample_embedding first.")
    emb_df = adata.uns["X_DR_sample"]
    if not isinstance(emb_df, pd.DataFrame):
        emb_df = pd.DataFrame(np.asarray(emb_df))
    unit_ids = list(emb_df.index.astype(str))
    obs_df = _aggregate_obs(adata, sample_col, modality_col, unit_ids)

    sample_adata = AnnData(
        X=emb_df.values.astype(np.float32),
        obs=obs_df,
    )
    sample_adata.uns["X_DR_sample"] = emb_df.copy()
    sample_adata.obsm["X_DR_sample"] = emb_df.values.astype(np.float32)
    if "sample_embedding_params" in adata.uns:
        sample_adata.uns["sample_embedding_params"] = dict(adata.uns["sample_embedding_params"])
    return sample_adata


def _aggregate_obs(adata, sample_col: str, modality_col: Optional[str],
                    unit_ids: List[str]) -> pd.DataFrame:
    """Aggregate cell-level metadata into per-unit (sample or sample×modality)
    metadata for the output sample-AnnData."""
    obs = adata.obs.copy()
    obs["__sample__"] = obs[sample_col].astype(str).values
    if modality_col is not None and modality_col in obs.columns:
        obs["__modality__"] = obs[modality_col].astype(str).values
    else:
        obs["__modality__"] = ""
    grouped = obs.groupby(["__sample__", "__modality__"], observed=True)
    rec = {}
    for col in obs.columns:
        if col in ("__sample__", "__modality__"):
            continue
        rec[col] = grouped[col].agg(
            lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan
        )
    if rec:
        agg = pd.concat(rec, axis=1)
    else:
        agg = pd.DataFrame(index=grouped.size().index)

    # Map back to unit_ids (handling both "_{modality}" suffixed and unsuffixed cases)
    rows = []
    for uid in unit_ids:
        # Try direct (sample, modality) lookup using suffix-stripping
        match = None
        for (s, m), row in agg.iterrows():
            candidate = f"{s}_{m}" if m else s
            if candidate == uid or (m and s == uid) or s == uid:
                match = row
                if candidate == uid:
                    break
        if match is None:
            # Fall back to empty row (preserves index alignment)
            match = pd.Series({c: np.nan for c in agg.columns})
        rows.append(match)
    out = pd.DataFrame(rows, index=pd.Index(unit_ids, name="sample"))
    return out


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
    """Compute sample-level embedding (singleRMD recipe).

    Returns a sample-level AnnData with `.uns['X_DR_sample']` (DataFrame,
    samples × pca_components) and `.obsm['X_DR_sample']` (ndarray).
    """
    start_time = time.time() if verbose else None

    if cluster_emb_key not in adata.obsm:
        raise KeyError(
            f"cluster_emb_key '{cluster_emb_key}' not in adata.obsm "
            f"(available: {list(adata.obsm.keys())})")
    if celltype_col not in adata.obs.columns:
        raise KeyError(
            f"celltype_col '{celltype_col}' not in adata.obs (available "
            f"columns: {list(adata.obs.columns)[:20]}...)")

    rmd_key = _resolve_rmd_emb_key(adata, cluster_emb_key, rmd_emb_key)
    if rmd_key not in adata.obsm:
        raise KeyError(f"rmd_emb_key '{rmd_key}' not in adata.obsm")
    if verbose:
        print(f"[sample_embedding] cluster_emb={cluster_emb_key}, "
              f"rmd_emb={rmd_key}")

    # Normalize batch_col -> primary (single str for assemble_units) + multi list (for Harmony multi-cov)
    if isinstance(batch_col, (list, tuple)):
        batch_cols_multi = [c for c in batch_col if c]
    elif batch_col:
        batch_cols_multi = [batch_col]
    else:
        batch_cols_multi = []
    primary_batch = batch_cols_multi[0] if batch_cols_multi else None

    # Build units (one per sample or sample×modality)
    units, unit_cellids, unit_ids, unit_groups, unit_batches, all_cellids, Z_clust = \
        assemble_units(adata, sample_col, cluster_emb_key,
                       modality_col=modality_col, batch_col=primary_batch)
    n_units = len(units)
    if n_units < 2:
        raise ValueError(f"need ≥2 units to compute embedding, got {n_units}")

    cellid_idx = {cid: i for i, cid in enumerate(all_cellids)}
    if verbose:
        print(f"[sample_embedding] {n_units} units; "
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
        print(f"[A1] coarse cell-type composition: shape={A1.shape}")

    # ---- A2: soft k-means at K_med ----
    K_med = min(medium_K, max(2, Z_clust.shape[0] // 200))
    if verbose:
        print(f"[A2] MiniBatchKMeans K={K_med}...", flush=True)
    km_med = MiniBatchKMeans(n_clusters=K_med, random_state=seed,
                              batch_size=4096, n_init=5, max_iter=200).fit(Z_clust)
    soft2 = soft_assign(Z_clust, km_med.cluster_centers_)
    A2 = composition_per_unit(unit_cellids_list, soft2, cellid_idx)
    if use_clr:
        A2 = clr_transform(A2)
    if verbose:
        print(f"[A2] shape={A2.shape}")

    # ---- A3: soft k-means at K_fine ----
    K_fine = min(fine_K, max(2, Z_clust.shape[0] // 100))
    if verbose:
        print(f"[A3] MiniBatchKMeans K={K_fine}...", flush=True)
    km_fine = MiniBatchKMeans(n_clusters=K_fine, random_state=seed + 1,
                                batch_size=4096, n_init=5, max_iter=200).fit(Z_clust)
    soft3 = soft_assign(Z_clust, km_fine.cluster_centers_)
    A3 = composition_per_unit(unit_cellids_list, soft3, cellid_idx)
    if use_clr:
        A3 = clr_transform(A3)
    if verbose:
        print(f"[A3] shape={A3.shape}")

    blocks = [A1, A2, A3]

    # ---- RMD: per-(group, coarse cluster) LOO displacement ----
    if use_rmd:
        if verbose:
            print(f"[RMD] LOO displacement on rmd_emb...", flush=True)
        # Build per-unit cells from the rmd_emb (might differ from cluster_emb)
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

    # ---- Weights ----
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
        print(f"[sample_embedding] weights={[round(w, 3) for w in weights]} "
              f"(n_blocks={len(blocks)})")

    # Multi-covariate Harmony meta (only when >=2 batch_cols)
    from sample_embedding.blocks import build_harmony_meta_df
    harmony_meta_df = (
        build_harmony_meta_df(adata, unit_cellids, unit_ids, batch_cols_multi)
        if len(batch_cols_multi) >= 2 else None
    )
    if verbose and harmony_meta_df is not None:
        print(f"[sample_embedding] multi-covariate Harmony meta: cols={list(harmony_meta_df.columns)}, "
              f"strata_per_col={[harmony_meta_df[c].nunique() for c in harmony_meta_df.columns]}")

    # ---- Final: Frobenius stack + PCA + sample-level Harmony ----
    emb_df = build_emb_from_blocks(
        blocks, weights,
        unit_ids=unit_ids,
        unit_groups=unit_groups,
        unit_batches=unit_batches,
        harmony_meta_df=harmony_meta_df,
        pca_components=pca_components,
        batch_method=batch_method,
        seed=seed,
        verbose=verbose,
    )

    # ---- Write the sample embedding back to the cell-level adata ----
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
        "backend": "cpu",
    }

    if save:
        out_dir = os.path.join(output_dir, "sample_embedding")
        os.makedirs(out_dir, exist_ok=True)
        emb_csv = os.path.join(out_dir, "sample_embedding.csv")
        emb_df.to_csv(emb_csv)

        # Re-save the cell-level adata_preprocessed.h5ad now that we have added
        # X_DR_sample to its .uns. Look for it at the canonical preprocess path.
        preprocessed_h5 = os.path.join(output_dir, "preprocess", "adata_preprocessed.h5ad")
        if os.path.exists(preprocessed_h5):
            try:
                sc.write(preprocessed_h5, adata)
            except Exception as exc:
                if verbose:
                    print(f"[sample_embedding] WARNING: could not re-save "
                          f"{preprocessed_h5}: {exc}")

        if verbose:
            print(f"[sample_embedding] wrote {emb_csv}")
            if os.path.exists(preprocessed_h5):
                print(f"[sample_embedding] updated {preprocessed_h5} (.uns['X_DR_sample'])")

    if verbose and start_time is not None:
        print(f"[sample_embedding] done in {time.time() - start_time:.2f}s; "
              f"shape={emb_df.shape}")

    return adata
