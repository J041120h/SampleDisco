"""Shared math primitives for sample embedding.

Backend-agnostic: works with numpy arrays and (if available) cupy arrays.
Routing is done by checking the input array's module — no explicit `use_gpu`
flag needed at this layer.

The recipe ports the `wire_singleRMD` / `wire_singleRMD_dualembed` variants
(no CLR by default, raw compositions, inverse-variance block weights).
"""

from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Backend helpers                                                             #
# --------------------------------------------------------------------------- #

def _xp(arr):
    """Return the array module (numpy or cupy) for the given array."""
    mod = type(arr).__module__
    if mod.startswith("cupy"):
        import cupy as cp
        return cp
    return np


def _to_numpy(arr):
    """Move array to CPU (numpy)."""
    if hasattr(arr, "get") and type(arr).__module__.startswith("cupy"):
        return arr.get()
    return np.asarray(arr)


# --------------------------------------------------------------------------- #
# Composition primitives                                                       #
# --------------------------------------------------------------------------- #

def soft_assign(Z, anchors, sigma: Optional[float] = None):
    """Gaussian-RBF soft assignment of cells to k-means anchors.

    Returns an (n_cells, n_anchors) matrix of probabilities.
    Works with numpy or cupy (dispatches via _xp).
    """
    xp = _xp(Z)
    # Pairwise distances via the (a-b)^2 = a^2 + b^2 - 2ab expansion
    # to keep memory predictable and stay on whichever device Z lives on.
    Z_sq = (Z * Z).sum(axis=1, keepdims=True)
    A_sq = (anchors * anchors).sum(axis=1, keepdims=True).T
    D2 = Z_sq + A_sq - 2.0 * (Z @ anchors.T)
    D2 = xp.maximum(D2, 0)
    if sigma is None:
        D = xp.sqrt(D2)
        sigma_val = float(xp.median(D))
    else:
        sigma_val = float(sigma)
    logits = -D2 / (2.0 * sigma_val * sigma_val + 1e-12)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = xp.exp(logits)
    return e / xp.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def composition_per_unit(unit_cellids, soft, cellid_idx) -> np.ndarray:
    """Per-unit composition: mean of `soft` rows over each unit's cells.

    `soft` may be cupy or numpy; output is always numpy (small matrix,
    CPU-side downstream handling).
    """
    soft_np = _to_numpy(soft)
    K = soft_np.shape[1]
    comp = np.zeros((len(unit_cellids), K), dtype=np.float32)
    for i, cell_ids in enumerate(unit_cellids):
        idxs = [cellid_idx[c] for c in cell_ids if c in cellid_idx]
        if idxs:
            comp[i] = soft_np[idxs].mean(axis=0)
    return comp


def composition_from_rows(unit_rows: List[np.ndarray], soft) -> np.ndarray:
    """`composition_per_unit` with per-unit row-index arrays (same reduction, no cell-id lookups)."""
    soft_np = _to_numpy(soft)
    comp = np.zeros((len(unit_rows), soft_np.shape[1]), dtype=np.float32)
    for i, rows in enumerate(unit_rows):
        if len(rows):
            comp[i] = soft_np[rows].mean(axis=0)
    return comp


def soft_composition(Z: np.ndarray, anchors: np.ndarray, unit_rows: List[np.ndarray],
                     n_threads: int = 1) -> np.ndarray:
    """`composition_from_rows(unit_rows, soft_assign(Z, anchors))` without building the
    n_cells x K soft matrix: the row-wise softmax and the per-unit mean run unit by unit
    (identical arithmetic per row), threaded over units. sigma = median(sqrt(D2)) is read
    from a partition of D2 (sqrt is monotone), as `np.median` does on sqrt(D2)."""
    G = Z @ anchors.T
    G *= 2.0
    D2 = (Z * Z).sum(axis=1, keepdims=True) + (anchors * anchors).sum(axis=1, keepdims=True).T
    D2 -= G
    del G
    np.maximum(D2, 0, out=D2)
    flat = D2.ravel()
    k = flat.size // 2
    if flat.size % 2:
        sigma = float(np.sqrt(np.partition(flat, k)[k]))
    else:
        part = np.partition(flat, [k - 1, k])
        sigma = float(np.mean(np.sqrt(part[[k - 1, k]])))
    denom = 2.0 * sigma * sigma + 1e-12

    def unit_mean(rows):
        if not len(rows):
            return np.zeros(D2.shape[1], dtype=np.float32)
        L = D2[rows]
        np.negative(L, out=L)
        L /= denom
        L -= L.max(axis=1, keepdims=True)
        np.exp(L, out=L)
        L /= np.maximum(L.sum(axis=1, keepdims=True), 1e-12)
        return L.mean(axis=0)

    if n_threads > 1:
        with ThreadPoolExecutor(n_threads) as ex:
            rows_out = list(ex.map(unit_mean, unit_rows))
    else:
        rows_out = [unit_mean(r) for r in unit_rows]
    return np.vstack(rows_out).astype(np.float32)


def n_worker_threads(cap: int = 16) -> int:
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, min(cap, n))


def kmeans_centers(Z: np.ndarray, K: int, seed: int) -> np.ndarray:
    """MiniBatchKMeans centres. compute_labels=False skips the final full labelling pass,
    which runs after the centres are fixed and whose output the embedding never uses."""
    from sklearn.cluster import MiniBatchKMeans
    return MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=4096, n_init=5,
                           max_iter=200, compute_labels=False).fit(Z).cluster_centers_


def kmeans_centers_pair(Z: np.ndarray, K_med: int, K_fine: int, seed: int,
                        n_threads: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Centres at K_med (seed) and K_fine (seed + 1). With n_threads > 1 the two independent
    fits run concurrently (each keeps its own RandomState). BLAS (process-wide) is held at one
    thread because k-means++ issues thousands of tiny GEMMs; the OpenMP thread count is a
    per-thread setting, so each worker sets its own share."""
    if n_threads < 2:
        return kmeans_centers(Z, K_med, seed), kmeans_centers(Z, K_fine, seed + 1)
    from threadpoolctl import threadpool_limits

    def fit(K, sd):
        with threadpool_limits(limits=max(1, n_threads // 2), user_api="openmp"):
            return kmeans_centers(Z, K, sd)

    with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(2) as ex:
        f_med = ex.submit(fit, K_med, seed)
        f_fine = ex.submit(fit, K_fine, seed + 1)
        return f_med.result(), f_fine.result()


UNS_EMBEDDING_KEYS = ("X_DR_sample", "sample_embedding_params")
_N_PROBE = 64  # rows (dense) or stored entries (sparse) compared per matrix


def _probe(n: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(n, _N_PROBE)).astype(np.int64)) if n > 0 else np.empty(0, np.int64)


def _same_matrix(elem, mem) -> bool:
    """Same shape, dtype, format and a fixed sample of stored values (on-disk X / layer / obsm / obsp / varm vs memory)."""
    import h5py
    import scipy.sparse as sp
    if isinstance(elem, h5py.Dataset):
        if sp.issparse(mem) or not isinstance(mem, np.ndarray) or elem.shape != mem.shape or elem.dtype != mem.dtype:
            return False
        rows = _probe(mem.shape[0])
        return np.array_equal(elem[rows], mem[rows], equal_nan=mem.dtype.kind in "fc")
    enc = elem.attrs.get("encoding-type")
    if enc not in ("csr_matrix", "csc_matrix") or not sp.issparse(mem) or mem.format != enc[:3]:
        return False
    if tuple(elem.attrs["shape"]) != mem.shape or elem["data"].dtype != mem.data.dtype or elem["data"].shape[0] != mem.nnz:
        return False
    pos, ptr = _probe(mem.nnz), _probe(len(mem.indptr))
    return (np.array_equal(elem["data"][pos], mem.data[pos], equal_nan=True)
            and np.array_equal(elem["indices"][pos], mem.indices[pos])
            and np.array_equal(elem["indptr"][ptr], mem.indptr[ptr]))


def _same_frame(disk: pd.DataFrame, mem: pd.DataFrame) -> bool:
    """Same index, same column names and identical values and dtypes per column (column order ignored)."""
    if not disk.index.equals(mem.index) or set(map(str, disk.columns)) != set(map(str, mem.columns)):
        return False
    for c in mem.columns:
        d, m = disk[str(c)], mem[c]
        if d.dtype != m.dtype:
            return False
        if isinstance(m.dtype, pd.CategoricalDtype):  # unordered dtype equality ignores category order: check it
            if not (d.cat.categories.equals(m.cat.categories)
                    and np.array_equal(d.cat.codes.to_numpy(), m.cat.codes.to_numpy())):
                return False
        elif not d.equals(m):
            return False
    return True


def _file_matches(f, adata) -> bool:
    """True when the open h5ad holds the same object as `adata`, apart from the two embedding entries:
    identical obs and var (names, columns and values), the same obsm/layers/obsp/varm keys, X and every
    obsm/layers/obsp/varm entry with the same shape, dtype and a sample of values, and the same uns key names
    (uns values are not compared). Any doubt -> False."""
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem
    try:
        if not (_same_frame(read_elem(f["obs"]), adata.obs) and _same_frame(read_elem(f["var"]), adata.var)):
            return False
        disk_uns = {k for k in f["uns"] if not k.startswith(("__new_", "__old_"))} if "uns" in f else set()
        if disk_uns - set(UNS_EMBEDDING_KEYS) != set(adata.uns) - set(UNS_EMBEDDING_KEYS):
            return False
        if ("X" in f) != (adata.X is not None) or (adata.X is not None and not _same_matrix(f["X"], adata.X)):
            return False
        for name, mem in (("obsm", adata.obsm), ("layers", adata.layers), ("obsp", adata.obsp), ("varm", adata.varm)):
            disk = f[name] if name in f else {}
            if set(disk) != set(mem.keys()) or not all(_same_matrix(disk[k], mem[k]) for k in mem.keys()):
                return False
        return True
    except Exception:  # old/unknown h5ad format, unexpected types: fall back to the full re-write
        return False


def write_embedding_uns(path: str, adata) -> bool:
    """Persist adata.uns['X_DR_sample'] / ['sample_embedding_params'] into an existing .h5ad in place,
    replacing only those two uns entries (X, layers, obs, obsm are not rewritten).

    Only done when the file holds the same object as `adata` (`_file_matches`); otherwise returns False
    without touching the file and the caller re-writes the whole file as before. In-memory changes to obs and
    var (names, columns, values) and to X / layers / obsm / obsp / varm (shape, dtype, sampled values) are
    detected and trigger the full re-write. For the other uns entries only the key names are compared, not
    their values: a changed value under an existing uns key is NOT detected and is not saved.
    Each entry is written under a temporary key and then moved into place, so an interrupted update leaves
    the old or the new value, never neither."""
    import h5py
    try:
        from anndata.io import write_elem
    except ImportError:
        from anndata.experimental import write_elem
    with h5py.File(path, "r+") as f:
        if not _file_matches(f, adata):
            return False
        uns = f.require_group("uns")
        for key in UNS_EMBEDDING_KEYS:
            new, old = f"__new_{key}", f"__old_{key}"
            if new in uns:
                del uns[new]
            write_elem(uns, new, adata.uns[key])
            if key in uns:
                if old in uns:
                    del uns[old]
                uns.move(key, old)
            uns.move(new, key)
            if old in uns:
                del uns[old]
    return True


def save_embedding_to_h5ad(path: str, adata) -> str:
    """Store the sample embedding in the cell-level h5ad: in place when the file matches `adata`,
    otherwise a full re-write (previous behaviour). Returns 'in-place' or 'rewrite'."""
    if write_embedding_uns(path, adata):
        return "in-place"
    import scanpy as sc
    sc.write(path, adata)
    return "rewrite"


def warn_if_stale_embedding(path: str, csv_path: str) -> None:
    """Called when the cell-level h5ad is NOT re-written: warn if it still carries an older
    .uns['X_DR_sample'], which a later run with derive_sample_embedding=False would reuse."""
    if not (path and os.path.exists(path)):
        return
    import h5py
    try:
        with h5py.File(path, "r") as f:
            stale = "uns" in f and "X_DR_sample" in f["uns"]
    except OSError:
        return
    if stale:
        import warnings
        warnings.warn(
            f"save_cell_adata=False: {path} still holds an OLDER .uns['X_DR_sample']. A later run "
            f"with derive_sample_embedding=False would silently reuse that stale embedding; the "
            f"current one is only in {csv_path}.", UserWarning, stacklevel=3)


def clr_transform(comp: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Aitchison centred-log-ratio. Optional; the singleRMD variant does NOT
    use this transform, but it is exposed for callers that want to opt in."""
    p = comp + eps
    p = p / p.sum(axis=1, keepdims=True)
    log_p = np.log(p)
    return (log_p - log_p.mean(axis=1, keepdims=True)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Reference-relative mean displacement (RMD)                                   #
# --------------------------------------------------------------------------- #

def loo_rmd(
    units: List[Tuple[str, str, np.ndarray]],
    units_uid_to_cellids: Dict[str, List[str]],
    label_for_cellid: Dict[str, str],
    *,
    max_dim_per_cluster: int = 8,
    seed: int = 42,
    loo: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Leave-One-Out per-(group, cluster) reference-relative mean displacement.

    `units[i] = (uid, group_label, cells_in_latent)` — `group_label` is
    typically `modality` for multi-omics or `batch` for single-omics.

    For each unit `u` (group `g`) and cluster `k`:
        μ_{u,k}   = mean of u's cells in cluster k
        ref_{u,k} = mean over other units u' with group(u')=g of μ_{u',k}
                      (if loo) OR the full-group mean (if not loo)
        d_{u,k}   = μ_{u,k} − ref_{u,k}

    Each cluster's displacement matrix is reduced via PCA to at most
    `max_dim_per_cluster` PCs and the per-cluster blocks are concatenated.
    """
    cluster_labels = sorted(set(label_for_cellid.values()),
                             key=lambda s: str(s))
    K = len(cluster_labels)
    L_idx = {lab: i for i, lab in enumerate(cluster_labels)}

    groups = sorted({u[1] for u in units})
    G = len(groups)
    G_idx = {g: i for i, g in enumerate(groups)}

    d_latent = units[0][2].shape[1]
    n_units = len(units)
    sums_smk = np.zeros((n_units, K, d_latent), dtype=np.float64)
    cnts_smk = np.zeros((n_units, K), dtype=np.int64)
    units_groupidx = np.zeros(n_units, dtype=np.int64)
    for ui, (uid, group, cells) in enumerate(units):
        units_groupidx[ui] = G_idx[group]
        cell_ids = units_uid_to_cellids[uid]
        for cid, cv in zip(cell_ids, cells):
            lab = label_for_cellid.get(cid)
            if lab is None:
                continue
            ki = L_idx[lab]
            sums_smk[ui, ki] += cv
            cnts_smk[ui, ki] += 1

    out = _loo_rmd_from_sums(sums_smk, cnts_smk, units_groupidx, G,
                             max_dim_per_cluster=max_dim_per_cluster, seed=seed, loo=loo)
    if verbose:
        print(f"  [RMD] shape={out.shape}")
    return out


def loo_rmd_from_index(
    Z_rmd: np.ndarray,
    unit_of_cell: np.ndarray,
    label_codes: np.ndarray,
    n_labels: int,
    unit_groups: List[str],
    *,
    max_dim_per_cluster: int = 8,
    seed: int = 42,
    loo: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """`loo_rmd` from integer codes (cell -> unit, cell -> sorted label). The per-(unit, label)
    sums use np.bincount, which accumulates in float64 in cell order exactly like the
    per-cell loop of `loo_rmd`; the rest is shared with it."""
    n_units = len(unit_groups)
    groups = sorted(set(unit_groups))
    G_idx = {g: i for i, g in enumerate(groups)}
    units_groupidx = np.array([G_idx[g] for g in unit_groups], dtype=np.int64)
    keep = unit_of_cell >= 0
    key = unit_of_cell[keep] * n_labels + label_codes[keep]
    Zk = Z_rmd[keep]
    sums_smk = np.stack([np.bincount(key, weights=Zk[:, j], minlength=n_units * n_labels)
                         for j in range(Zk.shape[1])], axis=1).reshape(n_units, n_labels, -1)
    cnts_smk = np.bincount(key, minlength=n_units * n_labels).reshape(n_units, n_labels).astype(np.int64)
    out = _loo_rmd_from_sums(sums_smk, cnts_smk, units_groupidx, len(groups),
                             max_dim_per_cluster=max_dim_per_cluster, seed=seed, loo=loo)
    if verbose:
        print(f"  [RMD] shape={out.shape}")
    return out


def _loo_rmd_from_sums(sums_smk, cnts_smk, units_groupidx, G, *,
                       max_dim_per_cluster, seed, loo):
    from sklearn.decomposition import PCA

    n_units, K, d_latent = sums_smk.shape
    grand_sum = np.zeros((G, K, d_latent), dtype=np.float64)
    grand_cnt = np.zeros((G, K), dtype=np.int64)
    for ui in range(n_units):
        grand_sum[units_groupidx[ui]] += sums_smk[ui]
        grand_cnt[units_groupidx[ui]] += cnts_smk[ui]

    # Overall latent centroids provide a deterministic notion of the nearest
    # batch/group for leave-one-out edge cases.  In particular, a sample that
    # is alone in its batch has no within-batch reference; using zero would
    # turn its RMD into an absolute position rather than a displacement.
    group_sum = grand_sum.sum(axis=1)
    group_cnt = grand_cnt.sum(axis=1)
    group_mean = group_sum / np.maximum(group_cnt[:, None], 1)
    unit_sum = sums_smk.sum(axis=1)
    unit_cnt = cnts_smk.sum(axis=1)
    unit_mean = unit_sum / np.maximum(unit_cnt[:, None], 1)

    per_disp = np.zeros((n_units, K, d_latent), dtype=np.float32)
    for ui in range(n_units):
        gi = units_groupidx[ui]
        if loo:
            ref_sum = grand_sum[gi] - sums_smk[ui]
            ref_cnt = grand_cnt[gi] - cnts_smk[ui]
        else:
            ref_sum = grand_sum[gi]
            ref_cnt = grand_cnt[gi]
        ref = ref_sum / np.maximum(ref_cnt[:, None], 1)
        missing = ref_cnt == 0
        if np.any(missing):
            distances = np.linalg.norm(group_mean - unit_mean[ui], axis=1)
            distances[gi] = np.inf
            nearest_groups = np.argsort(distances)
            for ki in np.flatnonzero(missing):
                for gj in nearest_groups:
                    if np.isfinite(distances[gj]) and grand_cnt[gj, ki] > 0:
                        ref[ki] = grand_sum[gj, ki] / grand_cnt[gj, ki]
                        break
                else:
                    # The cell type exists only in this unit/group.  A zero
                    # displacement is safer than an absolute-position feature.
                    if cnts_smk[ui, ki] > 0:
                        ref[ki] = sums_smk[ui, ki] / cnts_smk[ui, ki]
                    else:
                        ref[ki] = 0.0
        ref = ref.astype(np.float32)
        own_cnt = cnts_smk[ui]
        own_mean = np.where(own_cnt[:, None] > 0,
                              sums_smk[ui] / np.maximum(own_cnt[:, None], 1),
                              ref).astype(np.float32)
        per_disp[ui] = own_mean - ref

    rel = np.sqrt(cnts_smk.astype(np.float32))
    rel /= np.maximum(rel.max(axis=0, keepdims=True), 1e-6)
    per_disp *= rel[:, :, None]

    out_blocks = []
    for ki in range(K):
        sub = per_disp[:, ki, :]
        if sub.std() < 1e-8:
            continue
        nc = min(max_dim_per_cluster, sub.shape[1], n_units - 1)
        if nc < 1:
            continue
        try:
            pcs = PCA(n_components=nc, random_state=seed).fit_transform(sub)
            out_blocks.append(pcs.astype(np.float32))
        except Exception as exc:
            print(f"  [RMD] PCA failed for cluster {ki} (shape={sub.shape}, nc={nc}): "
                  f"{type(exc).__name__}: {exc}; cluster dropped from RMD block")
            continue
    return (np.concatenate(out_blocks, axis=1) if out_blocks
            else np.zeros((n_units, 0), dtype=np.float32))


# --------------------------------------------------------------------------- #
# Block weights                                                                #
# --------------------------------------------------------------------------- #

def derive_weights(
    K_c: int,
    K_med: int,
    K_fine: int,
    rmd_weight: float = 0.60,
    n_blocks: int = 4,
) -> List[float]:
    """Inverse-variance composition weights.

    Returns [w_A1, w_A2, w_A3, w_RMD] (or [w_A1, w_A2, w_A3] if n_blocks=3,
    no RMD block).

        w_A1 = √(K_fine / K_c)
        w_A2 = √(K_fine / K_med)
        w_A3 = 1.0
        w_RMD = rmd_weight  (literal; not scaled)

    When the user changes any of `medium_K`, `fine_K`, or the data's number
    of cell-type labels (`K_c`), composition weights auto-rescale so the
    relative balance among A1/A2/A3 stays meaningful. `rmd_weight` here must be numeric; the
    package default ``"equal"`` is resolved by `resolve_rmd_weight` before this is called.
    """
    K_c = max(int(K_c), 2)
    K_med = max(int(K_med), 2)
    K_fine = max(int(K_fine), 2)
    w_A1 = math.sqrt(K_fine / K_c)
    w_A2 = math.sqrt(K_fine / K_med)
    w_A3 = 1.0
    weights = [w_A1, w_A2, w_A3]
    if n_blocks >= 4:
        weights.append(float(rmd_weight))
    return weights


def check_rmd_weight(rmd_weight) -> None:
    """Accept ``"equal"`` or a finite positive number; raise ValueError otherwise (typos, bools, nan, <= 0)."""
    if isinstance(rmd_weight, str):
        ok = rmd_weight == "equal"
    else:
        ok = (isinstance(rmd_weight, (int, float, np.integer, np.floating))
              and not isinstance(rmd_weight, (bool, np.bool_))
              and math.isfinite(rmd_weight) and rmd_weight > 0)
    if not ok:
        raise ValueError(f"rmd_weight must be 'equal' or a finite positive number (got {rmd_weight!r})")


def resolve_rmd_weight(rmd_weight: Union[float, str], K_c: int, K_med: int, K_fine: int) -> float:
    """Numeric RMD weight α. ``"equal"`` gives the RMD block the same energy as the three
    composition blocks together: α² = w_A1² + w_A2² + w_A3² (so α depends only on K_c, K_med and
    K_fine), rounded to 2 decimals as reported. A number is used as given."""
    check_rmd_weight(rmd_weight)
    if isinstance(rmd_weight, str):
        return round(math.sqrt(sum(w * w for w in derive_weights(K_c, K_med, K_fine, n_blocks=3))), 2)
    return float(rmd_weight)


# --------------------------------------------------------------------------- #
# Frobenius stack + PCA + sample-level Harmony                                 #
# --------------------------------------------------------------------------- #

def frobenius_stack(blocks: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """Center, scale to ‖B‖_F = √N · w_b, concatenate columns."""
    if len(blocks) != len(weights):
        raise ValueError(
            f"weights length {len(weights)} != number of blocks {len(blocks)}")
    norm_blocks = []
    for blk, w in zip(blocks, weights):
        c = blk - blk.mean(axis=0, keepdims=True)
        fr = np.linalg.norm(c)
        if fr > 1e-8:
            c = c / fr * math.sqrt(blk.shape[0]) * w
        norm_blocks.append(c.astype(np.float32))
    F = np.concatenate(norm_blocks, axis=1).astype(np.float32)
    np.nan_to_num(F, copy=False)
    return F


def regress_out_batch_linear(X: np.ndarray, batch_labels) -> np.ndarray:
    """Per-PC linear regression batch removal. Used as a Harmony fallback."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder

    X = np.asarray(X, dtype=np.float32)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    B = enc.fit_transform(np.asarray(batch_labels).reshape(-1, 1))
    if B.shape[1] < 2:
        return X
    reg = LinearRegression(fit_intercept=True).fit(B, X)
    return (X - reg.predict(B)).astype(np.float32)


def composite_batch_labels(
    unit_groups: List[str],
    unit_batches: Optional[List[str]],
) -> Tuple[List[str], bool]:
    """Build per-unit composite-batch labels for Harmony.

    Returns group-only labels when batches are absent or map 1:1 to units;
    otherwise returns `f"{group}__{batch}"` composite labels.
    Returns (labels, used_composite_bool).
    """
    if unit_batches is None or len(unit_batches) != len(unit_groups):
        return list(unit_groups), False
    composite = [f"{g}__{b}" for g, b in zip(unit_groups, unit_batches)]
    n_units = len(unit_groups)
    n_groups = len(set(composite))
    if n_groups >= n_units:
        return list(unit_groups), False
    return composite, True


def build_emb_from_blocks(
    blocks: List[np.ndarray],
    weights: List[float],
    unit_ids: List[str],
    unit_groups: List[str],
    *,
    unit_batches: Optional[List[str]] = None,
    harmony_meta_df: Optional[pd.DataFrame] = None,
    pca_components: int = 10,
    batch_method: str = "harmony",
    seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """Frobenius-weighted stack + PCA + (optional) sample-level Harmony.

    Returns a pandas DataFrame indexed by `unit_ids`, columns PC1..PC{N}.

    If `harmony_meta_df` is provided (>=1 cols), Harmony is called with
    `batch_key=list(meta_df.columns)` for true multi-covariate correction.
    Otherwise falls back to the legacy single-key path using composite labels
    from (unit_groups, unit_batches).
    """
    from sklearn.decomposition import PCA

    F = frobenius_stack(blocks, weights)
    n_units = F.shape[0]
    n_pc_full = min(pca_components, F.shape[0] - 1, F.shape[1])
    if n_pc_full < 1:
        raise ValueError(
            f"insufficient data for PCA (shape={F.shape}, requested {pca_components})")
    Fp = PCA(n_components=n_pc_full, random_state=seed).fit_transform(F)

    use_multi = (
        harmony_meta_df is not None
        and len(harmony_meta_df.columns) >= 1
        and len(harmony_meta_df) == n_units
    )

    if use_multi:
        meta = harmony_meta_df.copy()
        meta.index = pd.Index(unit_ids, name="sample")
        batch_keys = list(meta.columns)
        n_groups = int(np.prod([meta[c].nunique() for c in batch_keys]))
        if verbose:
            per_col = ", ".join([f"{c}={meta[c].nunique()}" for c in batch_keys])
            print(f"  [batch correction] multi-covariate Harmony: keys={batch_keys}  ({per_col})")
        do_harmony = n_units >= 8 and any(meta[c].nunique() > 1 for c in batch_keys)
    else:
        batch_labels, used_composite = composite_batch_labels(unit_groups, unit_batches)
        if verbose:
            tag = "composite (group+batch)" if used_composite else "group only"
            print(f"  [batch correction] {len(set(batch_labels))} groups ({tag})")
        meta = pd.DataFrame({"batch": batch_labels}, index=pd.Index(unit_ids, name="sample"))
        batch_keys = "batch"
        do_harmony = len(set(batch_labels)) > 1 and n_units >= 8

    if do_harmony and batch_method == "none":
        # explicit no-op: skip sample-level batch correction, keep raw PCA
        Zc = Fp
    elif do_harmony:
        if batch_method == "linear":
            # `linear` regression only supports single-key composite labels
            if isinstance(batch_keys, list):
                # collapse to composite for the linear path
                joint = meta[batch_keys].astype(str).agg("__".join, axis=1).values
            else:
                joint = meta["batch"].values
            Zc = regress_out_batch_linear(Fp, joint)
        else:
            if isinstance(batch_keys, list):
                joint = meta[batch_keys].astype(str).agg("__".join, axis=1).values
            else:
                joint = meta["batch"].values
            try:
                import harmonypy as hm
                nclust = max(2, min(meta.nunique().max() if isinstance(batch_keys, list)
                                    else len(set(meta["batch"])),
                                    n_units // 2))
                ho = hm.run_harmony(Fp, meta,
                                    batch_keys,  # str OR list[str]
                                    nclust=nclust,
                                    max_iter_harmony=30,
                                    random_state=seed)
                Zc = ho.Z_corr
                if Zc.shape[0] != n_units:
                    Zc = Zc.T
            except Exception as exc:
                print(f"  [Harmony] FAILED ({exc!r}); falling back to linear "
                      f"regression batch removal", file=sys.stderr)
                try:
                    Zc = regress_out_batch_linear(Fp, joint)
                    print("  [Harmony fallback] linear regression succeeded",
                          file=sys.stderr)
                except Exception as exc2:
                    print(f"  [Harmony fallback] linear regression FAILED ({exc2!r}); "
                          f"using raw PCA — sample embedding will NOT be batch-corrected",
                          file=sys.stderr)
                    Zc = Fp
    else:
        Zc = Fp

    return pd.DataFrame(
        np.asarray(Zc, dtype=np.float32),
        index=pd.Index(unit_ids, name="sample"),
        columns=[f"PC{i+1}" for i in range(Zc.shape[1])],
    )


# --------------------------------------------------------------------------- #
# Unit assembly                                                                #
# --------------------------------------------------------------------------- #

def assemble_units(
    adata,
    sample_col: str,
    comp_emb_key: str,
    modality_col: Optional[str] = None,
    batch_col: Optional[str] = None,
) -> Tuple[
        List[Tuple[str, str, np.ndarray]],  # units: (uid, group, cells)
        Dict[str, List[str]],                # uid -> list of cell ids
        List[str],                            # unit_ids
        List[str],                            # groups per unit
        Optional[List[str]],                  # batches per unit (or None)
        List[str],                            # ordered comp_emb cell ids
        np.ndarray,                           # stacked Z (n_cells, d_emb)
]:
    """Build (unit_id, group_label, cells_in_emb) tuples from an AnnData.

    - Multi-omics:  units = (sample, modality), uid = f"{sample}_{modality}",
                    group = modality.
    - Single-omics: units = sample, uid = sample,
                    group = batch (if batch_col given) else "single".

    Returns rich tuple for downstream wiring.
    """
    Z = np.asarray(adata.obsm[comp_emb_key], dtype=np.float32)
    cell_ids = adata.obs_names.astype(str).values
    sample_arr = adata.obs[sample_col].astype(str).values

    if modality_col is not None and modality_col in adata.obs.columns:
        # Multi-omics
        modality_arr = adata.obs[modality_col].astype(str).values
        mods = sorted(set(modality_arr))
        unit_cellids_d: Dict[str, List[str]] = {}
        units: List[Tuple[str, str, np.ndarray]] = []
        unit_ids: List[str] = []
        unit_groups: List[str] = []
        for s_uniq in sorted(set(sample_arr)):
            for m in mods:
                mask = (sample_arr == s_uniq) & (modality_arr == m)
                if mask.sum() == 0:
                    continue
                # Strip modality suffix if user used `{sample}_{modality}` IDs
                bio = s_uniq
                for suf in (f"_{m}", f"_{m.lower()}"):
                    if bio.endswith(suf):
                        bio = bio[: -len(suf)]
                        break
                uid = bio if bio.endswith(f"_{m}") else f"{bio}_{m}"
                cids = cell_ids[mask].tolist()
                units.append((uid, m, Z[mask]))
                unit_cellids_d[uid] = cids
                unit_ids.append(uid)
                unit_groups.append(m)
        unit_batches = _per_unit_batch(adata, unit_cellids_d, batch_col)
    else:
        # Single-omics — group label = majority batch; falls back to "single"
        if batch_col is not None and batch_col in adata.obs.columns:
            batch_arr = adata.obs[batch_col].astype(str).values
        else:
            batch_arr = np.array(["single"] * len(sample_arr), dtype=object)
        unit_cellids_d = {}
        units = []
        unit_ids = []
        unit_groups = []
        for s_uniq in sorted(set(sample_arr)):
            mask = sample_arr == s_uniq
            if mask.sum() == 0:
                continue
            # Majority batch for the sample
            sample_batches = batch_arr[mask]
            grp = max(sorted(set(sample_batches)), key=list(sample_batches).count)
            uid = s_uniq
            cids = cell_ids[mask].tolist()
            units.append((uid, grp, Z[mask]))
            unit_cellids_d[uid] = cids
            unit_ids.append(uid)
            unit_groups.append(grp)
        unit_batches = None  # single-omics: batch IS the group; no separate list needed

    return units, unit_cellids_d, unit_ids, unit_groups, unit_batches, list(cell_ids), Z


def _per_unit_batch(
    adata,
    unit_cellids: Dict[str, List[str]],
    batch_col: Optional[str],
) -> Optional[List[str]]:
    if batch_col is None or batch_col not in adata.obs.columns:
        return None
    cellid_to_batch = dict(zip(
        adata.obs_names.astype(str).values,
        adata.obs[batch_col].astype(str).values,
    ))
    out = []
    for uid, cids in unit_cellids.items():
        bs = [cellid_to_batch.get(c) for c in cids if c in cellid_to_batch]
        bs = [b for b in bs if b is not None and b != "nan"]
        if not bs:
            out.append("UNK")
        else:
            out.append(max(sorted(set(bs)), key=bs.count))
    return out


# --------------------------------------------------------------------------- #
# Integer-index unit assembly (same units as assemble_units, no cell-id dicts) #
# --------------------------------------------------------------------------- #

def sorted_codes(series) -> Tuple[np.ndarray, np.ndarray]:
    """Equivalent to ``np.unique(series.astype(str).values, return_inverse=True)``,
    computed from the categorical codes (no string conversion of every cell)."""
    cat = series if isinstance(series.dtype, pd.CategoricalDtype) else series.astype("category")
    codes = cat.cat.codes.to_numpy().astype(np.int64)
    if (codes < 0).any():  # missing values become the label 'nan', as with astype(str)
        return np.unique(series.astype(str).values, return_inverse=True)
    labels = np.asarray(cat.cat.categories.astype(str), dtype=object)
    if len(set(labels)) != len(labels):  # distinct categories with equal str()
        return np.unique(series.astype(str).values, return_inverse=True)
    present = np.flatnonzero(np.bincount(codes, minlength=len(labels)) > 0)
    order = present[np.argsort(labels[present].astype(str), kind="stable")]
    remap = np.full(len(labels), -1, dtype=np.int64)
    remap[order] = np.arange(len(order))
    return labels[order].astype(str), remap[codes]


def _majority_per_unit(series, unit_of_cell: np.ndarray, n_units: int, *,
                       skip_nan: bool, empty: Optional[str]) -> List[str]:
    """Most frequent str label per unit; ties go to the first label in sorted order
    (``max(sorted(set(labels)), key=labels.count)``). With skip_nan, the label 'nan' is
    ignored and a unit with no other label gets `empty`."""
    labels, codes = sorted_codes(series)
    keep = unit_of_cell >= 0
    cnt = np.bincount(unit_of_cell[keep] * len(labels) + codes[keep],
                      minlength=n_units * len(labels)).reshape(n_units, len(labels))
    if skip_nan:
        cnt[:, labels == "nan"] = 0
    best = cnt.argmax(axis=1)
    return [str(labels[b]) if cnt[i, b] > 0 else empty for i, b in enumerate(best)]


def unit_index(
    adata,
    sample_col: str,
    modality_col: Optional[str] = None,
    batch_col: Optional[str] = None,
) -> Tuple[List[str], List[str], Optional[List[str]], np.ndarray, List[np.ndarray]]:
    """Integer-index twin of `assemble_units`: the same unit ids, order, groups and batches,
    plus ``unit_of_cell`` (cell -> unit position) and ``unit_rows`` (unit -> row indices in
    obs order). Rows are positions, so repeated obs_names are handled per cell."""
    s_labels, s_codes = sorted_codes(adata.obs[sample_col])
    n_cells = len(s_codes)
    multi = modality_col is not None and modality_col in adata.obs.columns
    if multi:
        m_labels, m_codes = sorted_codes(adata.obs[modality_col])
        pair = s_codes * len(m_labels) + m_codes
        present = np.flatnonzero(np.bincount(pair, minlength=len(s_labels) * len(m_labels)) > 0)
        unit_ids, unit_groups = [], []
        for p in present:
            s_uniq, m = str(s_labels[p // len(m_labels)]), str(m_labels[p % len(m_labels)])
            bio = s_uniq
            for suf in (f"_{m}", f"_{m.lower()}"):
                if bio.endswith(suf):
                    bio = bio[: -len(suf)]
                    break
            unit_ids.append(bio if bio.endswith(f"_{m}") else f"{bio}_{m}")
            unit_groups.append(m)
        remap = np.full(len(s_labels) * len(m_labels), -1, dtype=np.int64)
        remap[present] = np.arange(len(present))
        unit_of_cell = remap[pair]
    else:
        unit_ids = [str(x) for x in s_labels]
        unit_of_cell = s_codes
    n_units = len(unit_ids)
    order = np.argsort(unit_of_cell, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(unit_of_cell, minlength=n_units))]
    unit_rows = [order[bounds[i]:bounds[i + 1]] for i in range(n_units)]
    has_batch = batch_col is not None and batch_col in adata.obs.columns
    if multi:
        unit_batches = (_majority_per_unit(adata.obs[batch_col], unit_of_cell, n_units,
                                           skip_nan=True, empty="UNK") if has_batch else None)
    else:
        unit_groups = (_majority_per_unit(adata.obs[batch_col], unit_of_cell, n_units,
                                          skip_nan=False, empty=None)
                       if has_batch else ["single"] * n_units)
        unit_batches = None
    assert len(unit_of_cell) == n_cells
    return unit_ids, unit_groups, unit_batches, unit_of_cell, unit_rows


def harmony_meta_from_index(adata, unit_of_cell: np.ndarray, unit_ids: List[str],
                            batch_cols: Optional[Sequence[str]]) -> Optional[pd.DataFrame]:
    """Per-unit majority label of each batch column ('nan' ignored, 'UNK' if none) for
    multi-covariate sample-level Harmony."""
    if not batch_cols:
        return None
    cols = [batch_cols] if isinstance(batch_cols, str) else list(batch_cols)
    cols = [c for c in cols if c in adata.obs.columns]
    if not cols:
        return None
    return pd.DataFrame({c: _majority_per_unit(adata.obs[c], unit_of_cell, len(unit_ids),
                                               skip_nan=True, empty="UNK") for c in cols},
                        index=unit_ids)
