"""Resolution sweep for cell-type clustering on X_glue_harmony, on unpaired_test.

Question: does re-clustering cell_type on the batch-CORRECTED embedding
(X_glue_harmony) at higher Leiden resolutions reduce A1's batch-explained
variance (currently 0.32 with K_c=9 on the raw, batch-contaminated X_glue)?

For each resolution in RESOLUTIONS:
    - sc.tl.leiden on the (precomputed) KNN graph of X_glue_harmony
    - Build A1 = row-L2-normalized per-(sample, modality) composition over the
      new cell-type labels
    - Compute: K_c, ASW(batch), mean PC R^2(batch) on top-10 PCs of A1
    - Save a per-block PCA scatter coloured by batch

Output: /dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/debug/cluster_resolution/
    - sweep_metrics.csv
    - A1_res<R>_batch_pca.png  (one per resolution)
    - summary.md
"""
from __future__ import annotations
import os, sys, time, gc
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import h5py
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H5      = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/preprocess/atac_rna_integrated.h5ad'
OUT_DIR = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/debug/cluster_resolution'
os.makedirs(OUT_DIR, exist_ok=True)

RESOLUTIONS = [0.5, 1.0, 1.5, 2.5, 4.0]
EMB_KEY     = 'X_glue_harmony'   # batch-corrected, sample-removed
N_NEIGHBORS = 15
N_PCS_FOR_R2 = 10                # top-K PCs of A1 used for mean R^2 metric
ASW_MAX_N    = 20000              # subsample for silhouette (sample-level → fine)
SEED         = 0


def log(m: str) -> None: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def mean_pc_r2(block: np.ndarray, label: np.ndarray, n_pcs: int) -> float:
    """Mean R^2 from regressing each of top-`n_pcs` PCs of `block` on one-hot label."""
    n_pcs = min(n_pcs, min(block.shape) - 1)
    pcs = PCA(n_components=n_pcs, random_state=SEED).fit_transform(block)
    Y = pd.get_dummies(pd.Series(label, dtype=str)).values
    r2s = []
    for i in range(n_pcs):
        lr = LinearRegression().fit(Y, pcs[:, i])
        r2s.append(lr.score(Y, pcs[:, i]))
    return float(np.mean(r2s))


def build_A1(unit_cells_idx: dict, labels: np.ndarray, K: int) -> np.ndarray:
    """Per-unit composition (proportion) over `K` clusters, row-L2-normalized."""
    A1 = np.zeros((len(unit_cells_idx), K), dtype=np.float32)
    for ui, idx in enumerate(unit_cells_idx.values()):
        if len(idx) == 0:
            continue
        counts = np.bincount(labels[idx], minlength=K).astype(np.float32)
        A1[ui] = counts / max(counts.sum(), 1.0)
    norms = np.linalg.norm(A1, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return A1 / norms


# --------------------------------------------------------------------------- #
log(f"loading h5 (backed; obs + X_glue_harmony only)")
a = ad.read_h5ad(H5, backed='r')
n = a.shape[0]
obs = pd.DataFrame({
    'sample':   a.obs['sample'].astype(str).values,
    'modality': a.obs['modality'].astype(str).values,
    'batch':    a.obs['batch'].astype(str).values,
})
Xh = np.asarray(a.obsm[EMB_KEY])
a.file.close()
log(f"  {n:,} cells × {Xh.shape[1]} dims, batches={sorted(obs['batch'].unique())}")

# Build an in-memory adata for scanpy's neighbours/leiden
log("constructing minimal adata for scanpy")
adata = sc.AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs)
adata.obsm[EMB_KEY] = Xh.astype(np.float32)
del Xh; gc.collect()

log(f"sc.pp.neighbors(use_rep={EMB_KEY!r}, n_neighbors={N_NEIGHBORS}) — pynndescent backend")
t0 = time.time()
sc.pp.neighbors(adata, use_rep=EMB_KEY, n_neighbors=N_NEIGHBORS, random_state=SEED)
log(f"  KNN graph built in {time.time()-t0:.1f}s")

# Pre-compute the per-(sample, modality) -> cell-index lookup used by build_A1
log("indexing cells per (sample, modality) unit")
units = (obs['sample'] + "__" + obs['modality']).values
unit_meta = (obs.assign(unit=units)
                .groupby('unit').agg(sample=('sample', 'first'),
                                     modality=('modality', 'first'),
                                     batch=('batch', lambda s: s.mode().iloc[0]))
                .reset_index())
unit_order = unit_meta['unit'].tolist()
unit_idx_map = {u: np.flatnonzero(units == u) for u in unit_order}
batch_per_unit = unit_meta['batch'].values
log(f"  {len(unit_order)} units; batch counts:\n{pd.Series(batch_per_unit).value_counts().to_string()}")

# --------------------------------------------------------------------------- #
records = []
for res in RESOLUTIONS:
    log(f"=== resolution {res} ===")
    t0 = time.time()
    sc.tl.leiden(adata, resolution=res, random_state=SEED,
                 key_added=f'leiden_{res}',
                 flavor='igraph', n_iterations=2, directed=False)
    labels = adata.obs[f'leiden_{res}'].astype(int).values
    K = int(labels.max()) + 1
    log(f"  leiden in {time.time()-t0:.1f}s; K_c={K}")

    A1 = build_A1(unit_idx_map, labels, K)
    auto_w_A1 = float(np.sqrt(300.0 / max(K, 1)))   # canonical auto weight w_A1 = sqrt(K_med / K_c) with K_med=300

    # ASW on a sample-level subsample (units, not cells)
    asw_b = float(silhouette_score(A1, batch_per_unit)) if len(set(batch_per_unit)) >= 2 else float('nan')
    r2_b  = mean_pc_r2(A1, batch_per_unit, N_PCS_FOR_R2)
    log(f"  A1 shape={A1.shape}  ASW(batch)={asw_b:+.4f}  mean_PC_R2_batch={r2_b:.4f}  auto_w_A1={auto_w_A1:.3f}")

    # Per-resolution A1 PCA scatter (batch-coloured) — top 2 PCs
    pcs2 = PCA(n_components=2, random_state=SEED).fit_transform(A1)
    fig, ax = plt.subplots(figsize=(6, 5))
    for bv in sorted(set(batch_per_unit)):
        m = batch_per_unit == bv
        ax.scatter(pcs2[m, 0], pcs2[m, 1], s=20, alpha=0.7, label=f"{bv} (n={m.sum()})")
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title(f'A1 (res={res}, K_c={K}) — colour=batch\n'
                 f'mean_PC_R²(batch)={r2_b:.3f}  ASW(batch)={asw_b:+.3f}')
    ax.legend(fontsize=7, ncol=2, frameon=False, loc='best')
    fig.tight_layout()
    p = f"{OUT_DIR}/A1_res{res}_batch_pca.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    log(f"  saved {p}")

    records.append({
        'resolution':    res,
        'K_c':           K,
        'auto_w_A1':     auto_w_A1,
        'A1_shape':      f"{A1.shape[0]}x{A1.shape[1]}",
        'ASW_batch':     asw_b,
        'mean_PC_R2_batch': r2_b,
    })

df = pd.DataFrame.from_records(records)
df.to_csv(f"{OUT_DIR}/sweep_metrics.csv", index=False)
log(f"saved {OUT_DIR}/sweep_metrics.csv")
log("\n" + df.to_string(index=False))

# Summary markdown
md = ["# Cell-type Leiden resolution sweep on X_glue_harmony — unpaired_test\n",
      f"- Input embedding: `{EMB_KEY}` (batch-corrected, sample-removed)",
      f"- KNN: scanpy default (pynndescent), k={N_NEIGHBORS}",
      f"- Resolutions: {RESOLUTIONS}",
      f"- A1 block: per-(sample, modality) composition over Leiden clusters, row-L2-normalised",
      f"- Metric: mean R² when each of top-{N_PCS_FOR_R2} PCs of A1 is regressed on one-hot batch (lower = less batch-confounded)\n",
      "## Sweep results\n",
      df.to_markdown(index=False),
      "\n",
      "## Reference\n",
      "- Original A1 (cell_type built on RAW X_glue, K_c=9): mean_PC_R²(batch) = 0.319",
      ""]
with open(f"{OUT_DIR}/summary.md", 'w') as f:
    f.write("\n".join(md))
log(f"saved {OUT_DIR}/summary.md")
log("DONE")
