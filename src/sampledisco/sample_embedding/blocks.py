"""Shared math primitives for sample embedding.

Backend-agnostic: works with numpy arrays and (if available) cupy arrays.
Routing is done by checking the input array's module — no explicit `use_gpu`
flag needed at this layer.

The recipe ports the `wire_singleRMD` / `wire_singleRMD_dualembed` variants
(no CLR by default, raw compositions, inverse-variance block weights).
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Sequence, Tuple

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


def clr_transform(comp: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Aitchison centred-log-ratio. Optional; the singleRMD variant does NOT
    use this transform, but it is exposed for callers that want to opt in."""
    p = comp + eps
    p = p / p.sum(axis=1, keepdims=True)
    log_p = np.log(p)
    return (log_p - log_p.mean(axis=1, keepdims=True)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Counterfactual displacement (RMD)                                            #
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
    """Leave-One-Out per-(group, cluster) counterfactual displacement.

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
    from sklearn.decomposition import PCA

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

    grand_sum = np.zeros((G, K, d_latent), dtype=np.float64)
    grand_cnt = np.zeros((G, K), dtype=np.int64)
    for ui in range(n_units):
        grand_sum[units_groupidx[ui]] += sums_smk[ui]
        grand_cnt[units_groupidx[ui]] += cnts_smk[ui]

    per_disp = np.zeros((n_units, K, d_latent), dtype=np.float32)
    for ui in range(n_units):
        gi = units_groupidx[ui]
        if loo:
            ref_sum = grand_sum[gi] - sums_smk[ui]
            ref_cnt = grand_cnt[gi] - cnts_smk[ui]
        else:
            ref_sum = grand_sum[gi]
            ref_cnt = grand_cnt[gi]
        ref = np.where(ref_cnt[:, None] > 0,
                        ref_sum / np.maximum(ref_cnt[:, None], 1),
                        0.0).astype(np.float32)
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
    out = (np.concatenate(out_blocks, axis=1) if out_blocks
            else np.zeros((n_units, 0), dtype=np.float32))
    if verbose:
        print(f"  [RMD] shape={out.shape}")
    return out


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
    relative balance among A1/A2/A3 stays meaningful. The default rmd_weight
    (0.60) is taken from the winning variant.
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


def build_harmony_meta_df(
    adata,
    unit_cellids: Dict[str, List[str]],
    unit_ids: List[str],
    batch_cols: Optional[Sequence[str]],
) -> Optional[pd.DataFrame]:
    """Build a per-unit DataFrame for multi-covariate Harmony.

    Each row is a unit (sample), each column a batch covariate, value = majority
    label across the unit's cells. Returns None when batch_cols is empty or no
    cols match adata.obs. Used when >=2 batch_cols are passed; single batch_col
    goes through the legacy single-key code path.
    """
    if not batch_cols:
        return None
    cols = [batch_cols] if isinstance(batch_cols, str) else list(batch_cols)
    cols = [c for c in cols if c in adata.obs.columns]
    if not cols:
        return None
    cellid_to_batches = {
        c: dict(zip(
            adata.obs_names.astype(str).values,
            adata.obs[c].astype(str).values,
        )) for c in cols
    }
    rows = {c: [] for c in cols}
    for uid in unit_ids:
        cids = unit_cellids.get(uid, [])
        for c in cols:
            mapper = cellid_to_batches[c]
            bs = [mapper.get(cid) for cid in cids if cid in mapper]
            bs = [b for b in bs if b is not None and b != "nan"]
            rows[c].append(max(sorted(set(bs)), key=bs.count) if bs else "UNK")
    return pd.DataFrame(rows, index=unit_ids)


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
    cluster_emb_key: str,
    modality_col: Optional[str] = None,
    batch_col: Optional[str] = None,
) -> Tuple[
        List[Tuple[str, str, np.ndarray]],  # units: (uid, group, cells)
        Dict[str, List[str]],                # uid -> list of cell ids
        List[str],                            # unit_ids
        List[str],                            # groups per unit
        Optional[List[str]],                  # batches per unit (or None)
        List[str],                            # ordered cluster_emb cell ids
        np.ndarray,                           # stacked Z (n_cells, d_emb)
]:
    """Build (unit_id, group_label, cells_in_emb) tuples from an AnnData.

    - Multi-omics:  units = (sample, modality), uid = f"{sample}_{modality}",
                    group = modality.
    - Single-omics: units = sample, uid = sample,
                    group = batch (if batch_col given) else "single".

    Returns rich tuple for downstream wiring.
    """
    Z = np.asarray(adata.obsm[cluster_emb_key], dtype=np.float32)
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
