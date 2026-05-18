"""Re-autotune the V2 unpaired_test SE and report per-modality CCA + p-value.

Reuses the cached cell-level h5ad (has X_glue_harmony / X_glue_harmony_nosamp
/ cell_type / sample / modality / batch / sev.level). Runs:

  1. autotune (15-eval Bayesian over cmd_weight in [0.1, 10], grouping='sev.level')
  2. figure3/embedding/1.py-style analysis on the tuned sample embedding:
     - joint best 2-PC pair (maximises r_RNA + r_ATAC)
     - per-modality CCA r against sev.level
     - 1000-permutation p-value per modality
     - cosine similarity of CCA direction vectors

Output: prints to stdout AND writes a summary CSV / txt to
        /dcs07/.../multi_omics_unpaired_test/multiomics/sampledisco_tuned_v2_RETUNE/
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import numpy as np, pandas as pd, anndata as ad
from itertools import combinations
from sklearn.cross_decomposition import CCA

H5      = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/preprocess/atac_rna_integrated.h5ad"
OUT_DIR = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/sampledisco_tuned_v2_RETUNE"
os.makedirs(OUT_DIR, exist_ok=True)
N_PERM  = 1000
SEED    = 42

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── 1. Load adata (memory-safe; drop X) ──────────────────────────────────
log(f"loading {H5} (backed; no X)")
a = ad.read_h5ad(H5, backed='r')
n = a.shape[0]
obs = a.obs.copy()
obsm = {k: np.asarray(a.obsm[k]) for k in a.obsm.keys()}
a.file.close()
import scanpy as sc
adata = sc.AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs)
for k, v in obsm.items():
    adata.obsm[k] = v
log(f"  shape={adata.shape}  obsm={list(adata.obsm.keys())}  "
    f"K_c={adata.obs['cell_type'].nunique()}")


# ── 2. Re-run autotune ──────────────────────────────────────────────────
from parameter_selection.autotune import run_autotune
log("running autotune (Bayesian, 15 evals)")
t0 = time.time()
run_autotune(
    adata, OUT_DIR,
    sample_col="sample", celltype_col="cell_type",
    cluster_emb_key="X_glue_harmony",
    cmd_emb_key="X_glue_harmony_nosamp",
    modality_col="modality", batch_col="batch",
    grouping_col="sev.level",
    save=True, verbose=True,
)
log(f"  autotune done in {time.time()-t0:.1f}s")


# ── 3. Load tuned SE + sev.level + modality ─────────────────────────────
emb_csv = f"{OUT_DIR}/sample_embedding/sample_embedding.csv"
emb = pd.read_csv(emb_csv, index_col=0)
log(f"  tuned SE shape: {emb.shape}")

unit = emb.index.to_series()
mod = unit.str.rsplit('_', n=1).str[1]   # last underscore separator: <sample>_<modality>
sample = unit.str.rsplit('_', n=1).str[0]

# Sample-level sev.level (from cell-level adata, take majority per sample)
def majority(s):
    s = s.dropna()
    return s.mode().iloc[0] if not s.empty else np.nan
sev_per_sample = (adata.obs[['sample','sev.level']]
                  .groupby('sample')['sev.level']
                  .apply(majority).astype(float))
sl = sample.map(sev_per_sample).astype(float).values
log(f"  sev.level coverage: {(~pd.isna(sl)).sum()}/{len(sl)} units")


# ── 4. Joint best 2-PC pair (maximise r_RNA + r_ATAC) ──────────────────
def cca_r(X, y):
    if X.shape[1] < 1:
        return 0.0
    c = CCA(n_components=1, max_iter=500).fit(X, y.reshape(-1, 1))
    U, V = c.transform(X, y.reshape(-1, 1))
    return float(abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1]))

E_rna  = emb.values[mod.values == 'RNA']
E_atac = emb.values[mod.values == 'ATAC']
y_rna  = sl[mod.values == 'RNA']
y_atac = sl[mod.values == 'ATAC']
n_pc = emb.shape[1]
log(f"  n_RNA={len(E_rna)}  n_ATAC={len(E_atac)}  n_PCs={n_pc}")

log("scanning all joint PC pairs for max (r_RNA + r_ATAC)")
best = None
for i, j in combinations(range(n_pc), 2):
    r_r = cca_r(E_rna[:, [i, j]], y_rna)
    r_a = cca_r(E_atac[:, [i, j]], y_atac)
    s = r_r + r_a
    if best is None or s > best[0]:
        best = (s, i, j, r_r, r_a)
sum_best, pc_i, pc_j, r_rna_best, r_atac_best = best
log(f"  joint best: PC{pc_i+1} + PC{pc_j+1}  r_RNA={r_rna_best:.4f}  r_ATAC={r_atac_best:.4f}  sum={sum_best:.4f}")


# ── 5. Permutation p-values on the joint best plane ────────────────────
log(f"running {N_PERM}-permutation p-value on joint plane (PC{pc_i+1}, PC{pc_j+1})")
rng = np.random.default_rng(SEED)
out_rows = []
for tag, X, y in (('RNA', E_rna[:, [pc_i, pc_j]], y_rna),
                  ('ATAC', E_atac[:, [pc_i, pc_j]], y_atac)):
    r_obs = cca_r(X, y)
    perms = np.array([cca_r(X, rng.permutation(y)) for _ in range(N_PERM)])
    pval = (perms >= r_obs).mean()
    out_rows.append({
        'modality': tag, 'n': len(X), 'pc_pair': f'PC{pc_i+1}+PC{pc_j+1}',
        'r_obs': r_obs, 'p_perm_1sided': pval,
        'perm_mean': perms.mean(), 'perm_95th': np.quantile(perms, 0.95),
    })
res = pd.DataFrame(out_rows)
log("\n" + res.to_string(index=False))


# ── 6. Cosine similarity in the joint plane + full-D ───────────────────
def cca_direction(X, y):
    c = CCA(n_components=1, max_iter=500).fit(X, y.reshape(-1, 1))
    return c.x_weights_[:, 0]  # weight vector of x → linear comb of input PCs

def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

w_rna_full  = cca_direction(E_rna,  y_rna)
w_atac_full = cca_direction(E_atac, y_atac)
w_rna_first2  = cca_direction(E_rna[:, :2], y_rna)
w_atac_first2 = cca_direction(E_atac[:, :2], y_atac)
w_rna_joint   = cca_direction(E_rna[:, [pc_i, pc_j]],  y_rna)
w_atac_joint  = cca_direction(E_atac[:, [pc_i, pc_j]], y_atac)

cosines = {
    'full embedding (all PCs)': cosine(w_rna_full, w_atac_full),
    'first 2 PCs':              cosine(w_rna_first2, w_atac_first2),
    f'joint best plane (PC{pc_i+1},PC{pc_j+1})': cosine(w_rna_joint, w_atac_joint),
}
log("Cosine similarity of CCA direction vectors:")
for k, v in cosines.items():
    log(f"  {k:42s}  {v:+.4f}")


# ── 7. Save a summary file ─────────────────────────────────────────────
with open(f"{OUT_DIR}/cca_summary.txt", 'w') as f:
    f.write("Re-autotune V2 (unpaired_test) — per-modality CCA + permutation p-value\n")
    f.write("=" * 70 + "\n")
    f.write(f"input h5ad      : {H5}\n")
    f.write(f"tuned SE        : {emb_csv}\n\n")
    f.write(f"Joint best 2-PC pair (max r_RNA + r_ATAC)\n")
    f.write(f"  pair  : PC{pc_i+1}, PC{pc_j+1}\n")
    f.write(f"  r_RNA  : {r_rna_best:.4f}\n")
    f.write(f"  r_ATAC : {r_atac_best:.4f}\n")
    f.write(f"  sum    : {sum_best:.4f}\n\n")
    f.write(f"Permutation p-values ({N_PERM} 1-sided shuffles of sev.level)\n")
    f.write(res.to_string(index=False) + "\n\n")
    f.write("Cosine similarity of CCA direction vectors\n")
    for k, v in cosines.items():
        f.write(f"  {k:42s}  {v:+.4f}\n")
res.to_csv(f"{OUT_DIR}/cca_per_modality.csv", index=False)
log(f"  wrote summary → {OUT_DIR}/cca_summary.txt")
log("DONE")
