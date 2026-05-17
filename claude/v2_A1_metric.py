"""Lightweight comparison of A1 batch contamination between:
  · OLD pipeline (cell_type built on raw X_glue, K_c=9) — saved metric=0.319
  · NEW pipeline V2 (cell_type built on X_glue_harmony at res=0.8)

Reads cell_type from the now-updated h5ad (V2 has overwritten it) and rebuilds
A1 per (sample, modality). Reports K_c, ASW(batch), mean_PC_R²(batch),
auto_w_A1=sqrt(300/K_c), and the effective ≈ w² · R² proxy.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import numpy as np, pandas as pd, anndata as ad
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H5      = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/preprocess/atac_rna_integrated.h5ad'
OUT_DIR = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/debug/v2'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_PCS_R2 = 10


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def mean_pc_r2(M, label, n):
    n = min(n, min(M.shape) - 1)
    pcs = PCA(n_components=n, random_state=SEED).fit_transform(M)
    Y = pd.get_dummies(pd.Series(label, dtype=str)).values
    return float(np.mean([LinearRegression().fit(Y, pcs[:, i]).score(Y, pcs[:, i])
                          for i in range(n)]))


log(f"reading obs from {H5} (backed)")
a = ad.read_h5ad(H5, backed='r')
obs = pd.DataFrame({
    'sample':    a.obs['sample'].astype(str).values,
    'modality':  a.obs['modality'].astype(str).values,
    'batch':     a.obs['batch'].astype(str).values,
    'cell_type': a.obs['cell_type'].astype(str).values,
})
a.file.close()

K_c = obs['cell_type'].nunique()
log(f"  V2 cell_type: K_c={K_c}; distribution:\n{obs['cell_type'].value_counts().to_string()}")

units = (obs['sample'] + "__" + obs['modality']).values
unit_order = sorted(set(units))
unit_idx = {u: np.flatnonzero(units == u) for u in unit_order}
batch_per_unit = np.array([
    obs['batch'].values[unit_idx[u][0]] for u in unit_order  # any cell in unit
])

cell_types_sorted = sorted(obs['cell_type'].unique(), key=lambda x: (len(x), x))
ct_to_i = {c: i for i, c in enumerate(cell_types_sorted)}

log("building A1 (per-(sample, modality) composition over K_c cell types, L2 normalised)")
A1 = np.zeros((len(unit_order), K_c), dtype=np.float32)
for ui, u in enumerate(unit_order):
    idx = unit_idx[u]
    labs = obs['cell_type'].values[idx]
    counts = np.bincount([ct_to_i[c] for c in labs], minlength=K_c).astype(np.float32)
    A1[ui] = counts / max(counts.sum(), 1.0)
A1 = A1 / np.clip(np.linalg.norm(A1, axis=1, keepdims=True), 1e-12, None)

asw = float(silhouette_score(A1, batch_per_unit)) if len(set(batch_per_unit)) >= 2 else float('nan')
r2  = mean_pc_r2(A1, batch_per_unit, N_PCS_R2)
w   = float(np.sqrt(300.0 / max(K_c, 1)))
eff = w * w * r2

# Plot
pcs2 = PCA(n_components=2, random_state=SEED).fit_transform(A1)
fig, ax = plt.subplots(figsize=(6, 5))
for bv in sorted(set(batch_per_unit)):
    msk = batch_per_unit == bv
    ax.scatter(pcs2[msk, 0], pcs2[msk, 1], s=20, alpha=0.7, label=f"{bv} (n={msk.sum()})")
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title(f"V2 A1 (K_c={K_c}, cell_type on X_glue_harmony @ res=0.8)\n"
             f"mean_PC_R²(batch)={r2:.3f}  ASW(batch)={asw:+.3f}  w_A1={w:.2f}  eff={eff:.2f}")
ax.legend(fontsize=7, ncol=2, frameon=False, loc='best')
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/A1_v2_batch_pca.png", dpi=130); plt.close(fig)

# Save comparison
old = {'K_c': 9, 'ASW_batch': -0.2198, 'mean_PC_R2_batch': 0.3194,
       'auto_w_A1': float(np.sqrt(300.0 / 9)),
       'effective_w2_R2': float(np.sqrt(300.0 / 9))**2 * 0.3194}
new = {'K_c': int(K_c), 'ASW_batch': asw, 'mean_PC_R2_batch': r2,
       'auto_w_A1': w, 'effective_w2_R2': eff}
cmp = pd.DataFrame([{'version': 'OLD (raw X_glue, res=0.8)', **old},
                    {'version': 'V2 (X_glue_harmony, res=0.8)', **new}])
cmp.to_csv(f"{OUT_DIR}/A1_compare.csv", index=False)
log("\n" + cmp.to_string(index=False))
log(f"saved {OUT_DIR}/A1_compare.csv and A1_v2_batch_pca.png")
