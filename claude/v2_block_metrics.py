"""Recompute per-block (A1/A2/A3/CMD) batch metrics for the V2 unpaired_test
SE run (cell typing on X_glue_harmony at res=0.8). Mirrors the layout of
debug_unpaired_test.py PART 2 so results are directly comparable to the
"original" block_metrics.csv produced earlier on the bug pipeline.
"""
from __future__ import annotations
import os, sys, time, gc
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import numpy as np, pandas as pd, scanpy as sc, anndata as ad
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H5      = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/preprocess/atac_rna_integrated.h5ad'
OUT_DIR = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/debug/block_analysis_v2'
os.makedirs(OUT_DIR, exist_ok=True)

CLUSTER_KEY = 'X_glue_harmony'
CMD_KEY     = 'X_glue_harmony_nosamp'
MED_K, FINE_K, CMD_DIM = 120, 300, 8
N_PCS_R2 = 10
SEED = 42

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def mean_pc_r2(block: np.ndarray, label: np.ndarray, n_pcs: int) -> float:
    n_pcs = min(n_pcs, min(block.shape) - 1)
    pcs = PCA(n_components=n_pcs, random_state=SEED).fit_transform(block)
    Y = pd.get_dummies(pd.Series(label, dtype=str)).values
    return float(np.mean([LinearRegression().fit(Y, pcs[:, i]).score(Y, pcs[:, i])
                          for i in range(n_pcs)]))


def soft_cluster_assign(X: np.ndarray, K: int, seed: int) -> np.ndarray:
    """Soft assignment via 1/(1+d^2) on MiniBatchKMeans centroids, normalised."""
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=3, batch_size=10_000)
    km.fit(X)
    d2 = ((X[:, None, :] - km.cluster_centers_[None, :, :])**2).sum(-1)   # (N, K)
    # Memory-friendly chunked computation
    return None  # placeholder — we compute per-unit below


# --------------------------------------------------------------------------- #
log(f"loading {H5} (X dropped)")
a = ad.read_h5ad(H5, backed='r')
n = a.shape[0]
obs = pd.DataFrame({
    'sample':   a.obs['sample'].astype(str).values,
    'modality': a.obs['modality'].astype(str).values,
    'batch':    a.obs['batch'].astype(str).values,
    'cell_type': a.obs['cell_type'].astype(str).values,
})
Xc = np.asarray(a.obsm[CLUSTER_KEY])
Xd = np.asarray(a.obsm[CMD_KEY])
a.file.close()
log(f"  {n:,} cells; K_c={obs['cell_type'].nunique()}")

units = obs['sample'] + "__" + obs['modality']
unit_order = sorted(units.unique())
unit_idx   = {u: np.flatnonzero(units == u) for u in unit_order}

unit_meta = (obs.assign(unit=units)
                .groupby('unit')
                .agg(batch=('batch', lambda s: s.mode().iloc[0]))
                .reindex(unit_order))
batch_per_unit = unit_meta['batch'].values
log(f"  {len(unit_order)} units")

cell_types_sorted = sorted(obs['cell_type'].unique(), key=lambda x: (len(x), x))
ct_to_i = {ct: i for i, ct in enumerate(cell_types_sorted)}
K_c = len(cell_types_sorted)


# ---- A1: per-unit composition over K_c cell types ------------------------- #
log(f"A1 (composition, K_c={K_c})")
A1 = np.zeros((len(unit_order), K_c), dtype=np.float32)
for ui, u in enumerate(unit_order):
    idx = unit_idx[u]
    labs = obs['cell_type'].values[idx]
    cnts = np.bincount([ct_to_i[c] for c in labs], minlength=K_c).astype(np.float32)
    A1[ui] = cnts / max(cnts.sum(), 1.0)
A1 = A1 / np.clip(np.linalg.norm(A1, axis=1, keepdims=True), 1e-12, None)


# ---- A2 / A3: per-unit soft assignment (proportion over K) ---------------- #
def build_soft_proportion_block(X: np.ndarray, K: int) -> np.ndarray:
    log(f"  fitting MiniBatchKMeans K={K} on cluster embedding ({X.shape})")
    km = MiniBatchKMeans(n_clusters=K, random_state=SEED, n_init=3, batch_size=10_000)
    km.fit(X)
    centers = km.cluster_centers_.astype(np.float32)
    M = np.zeros((len(unit_order), K), dtype=np.float32)
    for ui, u in enumerate(unit_order):
        idx = unit_idx[u]
        Xi = X[idx].astype(np.float32)
        # softmax-style soft assignment via inverse-distance
        d2 = np.sum(Xi[:, None, :]**2, -1) + np.sum(centers**2, 1)[None] - 2.0 * (Xi @ centers.T)
        w  = 1.0 / (1.0 + d2)
        w  = w / np.clip(w.sum(1, keepdims=True), 1e-12, None)
        M[ui] = w.sum(0) / max(len(idx), 1)
    return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)

log(f"A2 (soft, K={MED_K})")
A2 = build_soft_proportion_block(Xc, MED_K)
log(f"A3 (soft, K={FINE_K})")
A3 = build_soft_proportion_block(Xc, FINE_K)


# ---- CMD: per-unit per-cluster LOO displacement (top-CMD_DIM PCs of CMD emb) #
log(f"CMD block (per cluster top-{CMD_DIM} PCs on {CMD_KEY})")
# Use the same K_c clusters as A1 (cell-type level). Per (unit, modality) we
# take the mean residual against the population mean within that cluster, then
# concatenate K_c × CMD_DIM dims per unit.
CMD = np.zeros((len(unit_order), K_c * CMD_DIM), dtype=np.float32)
# For memory: compute per-cluster PCA on a subsample, then project all cells
import scipy.sparse as sp
np.random.seed(SEED)
for ck, ct in enumerate(cell_types_sorted):
    mask = obs['cell_type'].values == ct
    n_ct = int(mask.sum())
    if n_ct < CMD_DIM + 1:
        continue
    Xct = Xd[mask].astype(np.float32)
    sub = np.random.choice(n_ct, min(n_ct, 30_000), replace=False)
    pca = PCA(n_components=CMD_DIM, random_state=SEED).fit(Xct[sub])
    Z   = pca.transform(Xct)            # (n_ct, CMD_DIM)
    mu  = Z.mean(0)
    # per-unit mean residual within this cluster
    abs_idx = np.flatnonzero(mask)
    units_in = (obs['sample'].values[abs_idx] + "__" + obs['modality'].values[abs_idx])
    df = pd.DataFrame({'unit': units_in}).reset_index(drop=True)
    for ui, u in enumerate(unit_order):
        sel = df['unit'].values == u
        if sel.sum() == 0:
            continue
        CMD[ui, ck*CMD_DIM:(ck+1)*CMD_DIM] = Z[sel].mean(0) - mu
CMD = CMD / np.clip(np.linalg.norm(CMD, axis=1, keepdims=True), 1e-12, None)


# ---- metrics ------------------------------------------------------------- #
rows = []
for name, M in [('A1', A1), ('A2', A2), ('A3', A3), ('CMD', CMD)]:
    asw = float(silhouette_score(M, batch_per_unit)) if len(set(batch_per_unit)) >= 2 else float('nan')
    r2  = mean_pc_r2(M, batch_per_unit, N_PCS_R2)
    log(f"  {name}: shape={M.shape} ASW_batch={asw:+.4f} mean_PC_R2_batch={r2:.4f}")
    rows.append({'block': name, 'shape': f"{M.shape[0]}x{M.shape[1]}",
                 'ASW_batch': asw, 'mean_PC_R2_batch': r2})

    # PCA scatter (batch-coloured) for visual confirmation
    pcs2 = PCA(n_components=2, random_state=SEED).fit_transform(M)
    fig, ax = plt.subplots(figsize=(6, 5))
    for bv in sorted(set(batch_per_unit)):
        msk = batch_per_unit == bv
        ax.scatter(pcs2[msk, 0], pcs2[msk, 1], s=20, alpha=0.7, label=f"{bv} (n={msk.sum()})")
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title(f"{name} (V2, K_c={K_c}) — batch  R²={r2:.3f}, ASW={asw:+.3f}")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc='best')
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}_batch_pca.png", dpi=130); plt.close(fig)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}/block_metrics.csv", index=False)
log(f"saved {OUT_DIR}/block_metrics.csv")
log("\n" + df.to_string(index=False))
log("DONE")
