"""V2 of the unpaired_test pipeline — GPU-accelerated.

Pipeline-ordering bug fix: cell typing now runs on `X_glue_harmony` (the
batch-corrected, sample-removed embedding produced by harmonize_xglue),
not on the raw, batch-contaminated `X_glue`. Resolution kept at the
pipeline default (0.8) per user request.

GPU details:
    - The standard `cell_types_multiomics_linux` GPU path uses
      `rapids_singlecell` / `cuml`, but RAPIDS is broken on this node
      (CUDA driver mismatch).
    - We replicate the SAME Jaccard-SNN label-transfer math, but the
      slow 4 KNN queries are done with torch on the V100 (chunked
      pairwise cosine on the GPU). Leiden runs on CPU with igraph
      (~2 min on 898K cells).

Output dirs:
    ROOT/sampledisco_default_v2_celltype-on-harmony   (default α)
    ROOT/sampledisco_tuned_v2_celltype-on-harmony     (autotuned α)
"""
from __future__ import annotations
import os, sys, time, gc
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import h5py
import torch
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

H5   = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/preprocess/atac_rna_integrated.h5ad'
ROOT = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics'

CLUSTER_KEY = 'X_glue_harmony'
CMD_KEY     = 'X_glue_harmony_nosamp'
RESOLUTION  = 0.8
K_TRANSFER  = 3                  # matches pipeline default (multiomics_wrapper)
SEED        = 42


def log(m: str) -> None: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
def load_minimal(h5: str) -> sc.AnnData:
    a = ad.read_h5ad(h5, backed='r')
    n = a.shape[0]
    obs_df = a.obs.copy()
    obsm   = {k: np.asarray(a.obsm[k]) for k in a.obsm.keys()}
    a.file.close()
    out = sc.AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs_df)
    for k, v in obsm.items():
        out.obsm[k] = v
    return out


def h5_write_obs_string_column(h5: str, col: str, values: np.ndarray) -> None:
    cats = np.array(sorted(set(values)))
    code_of = {c: i for i, c in enumerate(cats)}
    codes = np.array([code_of[v] for v in values], dtype='int8')
    with h5py.File(h5, 'a') as f:
        target = f"obs/{col}"
        if target in f:
            del f[target]
        grp = f.create_group(target)
        grp.attrs['encoding-type']    = 'categorical'
        grp.attrs['encoding-version'] = '0.2.0'
        grp.attrs['ordered']          = False
        grp.create_dataset('codes',      data=codes)
        grp.create_dataset('categories', data=np.array(cats, dtype='S'))
        obs_grp = f['obs']
        order = list(obs_grp.attrs.get('column-order', []))
        if col not in order:
            order.append(col)
            obs_grp.attrs['column-order'] = np.array(order, dtype='O')


def gpu_knn_cosine(query: np.ndarray, ref: np.ndarray, k: int,
                   q_chunk: int = 8_192, r_chunk: int = 65_536) -> np.ndarray:
    """Top-k cosine neighbours of `query` rows among `ref` rows on the GPU.
    Double-chunked (over query AND ref) with a running top-k merge so the
    peak GPU tensor is roughly q_chunk x r_chunk x 4 bytes.
    Returns int32 array of shape (n_query, k) with indices into `ref`."""
    dev = torch.device("cuda")
    # Normalise on CPU once (cheap) so cosine = inner product on GPU
    qn = torch.nn.functional.normalize(
            torch.from_numpy(np.ascontiguousarray(query, dtype=np.float32)), dim=1)
    rn = torch.nn.functional.normalize(
            torch.from_numpy(np.ascontiguousarray(ref,   dtype=np.float32)), dim=1)
    n_q, n_r = qn.shape[0], rn.shape[0]
    out = np.empty((n_q, k), dtype=np.int32)
    NEG_INF = float('-inf')
    for qi in range(0, n_q, q_chunk):
        qj = min(qi + q_chunk, n_q)
        b = qj - qi
        q_gpu = qn[qi:qj].to(dev, non_blocking=True)
        # running top-k for this query chunk
        best_sim = torch.full((b, k), NEG_INF, device=dev, dtype=torch.float32)
        best_idx = torch.full((b, k), -1,      device=dev, dtype=torch.int64)
        for ri in range(0, n_r, r_chunk):
            rj = min(ri + r_chunk, n_r)
            r_gpu = rn[ri:rj].to(dev, non_blocking=True)
            sims = q_gpu @ r_gpu.T                    # (b, r_chunk)
            tk = min(k, sims.shape[1])
            v_loc, i_loc = sims.topk(tk, dim=1)
            i_loc = i_loc + ri                        # offset to global ref index
            # merge with running top-k via concat + topk
            merged_sim = torch.cat([best_sim, v_loc], dim=1)
            merged_idx = torch.cat([best_idx, i_loc], dim=1)
            v_new, sel = merged_sim.topk(k, dim=1)
            best_sim = v_new
            best_idx = torch.gather(merged_idx, 1, sel)
            del r_gpu, sims, v_loc, i_loc, merged_sim, merged_idx, v_new, sel
        out[qi:qj] = best_idx.cpu().numpy().astype(np.int32)
        del q_gpu, best_sim, best_idx
        torch.cuda.empty_cache()
    return out


def knn_to_sparse(indices: np.ndarray, n_samples: int, n_features: int) -> sparse.csr_matrix:
    k = indices.shape[1]
    row_idx = np.repeat(np.arange(n_samples), k)
    col_idx = indices.ravel()
    data    = np.ones(n_samples * k, dtype=np.float32)
    return sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n_samples, n_features))


# --------------------------------------------------------------------------- #
log(f"loading {H5} (X dropped)")
adata = load_minimal(H5)
log(f"  shape={adata.shape}; obsm={list(adata.obsm.keys())}")

for key in (CLUSTER_KEY, CMD_KEY):
    if key not in adata.obsm:
        raise RuntimeError(f"required obsm key '{key}' is missing")
log(f"  torch cuda? {torch.cuda.is_available()}  device: {torch.cuda.get_device_name(0)}")

log(f"old cell_type distribution (built on RAW X_glue, K_c={adata.obs['cell_type'].nunique()}):")
print(adata.obs['cell_type'].value_counts().to_string())

# --------------------------------------------------------------------------- #
log(f"=== re-typing cell_type on {CLUSTER_KEY} at resolution {RESOLUTION} ===")
modality_col = 'modality'
rna_mask  = (adata.obs[modality_col] == 'RNA').values
atac_mask = (adata.obs[modality_col] == 'ATAC').values
n_rna, n_atac = int(rna_mask.sum()), int(atac_mask.sum())
log(f"  RNA={n_rna:,}  ATAC={n_atac:,}")

emb = np.asarray(adata.obsm[CLUSTER_KEY], dtype=np.float32)
rna_emb  = emb[rna_mask]
atac_emb = emb[atac_mask]

# --- Step 1: Leiden on RNA via scanpy + igraph backend (CPU but fast) ----- #
log(f"  Step 1: sc.pp.neighbors + Leiden on RNA (n={n_rna:,}, igraph)")
t0 = time.time()
rna_adata = sc.AnnData(X=np.zeros((n_rna, 1), dtype=np.float32))
rna_adata.obsm[CLUSTER_KEY] = rna_emb
sc.pp.neighbors(rna_adata, use_rep=CLUSTER_KEY, n_neighbors=15, random_state=SEED)
sc.tl.leiden(rna_adata, resolution=RESOLUTION, random_state=SEED,
             key_added='cell_type', flavor='igraph', n_iterations=2, directed=False)
rna_labels_int = rna_adata.obs['cell_type'].astype(int).values
rna_labels = (rna_labels_int + 1).astype(str)
K_c = int(rna_labels_int.max() + 1)
log(f"  Leiden done in {time.time()-t0:.1f}s; K_c={K_c}")
del rna_adata; gc.collect()

# --- Step 2: GPU KNN for Jaccard-SNN label transfer ----------------------- #
log(f"  Step 2: GPU KNN (k={K_TRANSFER}) — 4 builds via torch on V100")
t0 = time.time()
rna_rna_idx   = gpu_knn_cosine(rna_emb,  rna_emb,  K_TRANSFER)   ; log(f"    rna→rna  {time.time()-t0:.1f}s")
t0 = time.time()
rna_atac_idx  = gpu_knn_cosine(rna_emb,  atac_emb, K_TRANSFER)   ; log(f"    rna→atac {time.time()-t0:.1f}s")
t0 = time.time()
atac_rna_idx  = gpu_knn_cosine(atac_emb, rna_emb,  K_TRANSFER)   ; log(f"    atac→rna {time.time()-t0:.1f}s")
t0 = time.time()
atac_atac_idx = gpu_knn_cosine(atac_emb, atac_emb, K_TRANSFER)   ; log(f"    atac→atac {time.time()-t0:.1f}s")

log("  Step 3: Jaccard-SNN matrices + label predictions")
xx = knn_to_sparse(rna_rna_idx,   n_rna,  n_rna)
xy = knn_to_sparse(rna_atac_idx,  n_rna,  n_atac)
yx = knn_to_sparse(atac_rna_idx,  n_atac, n_rna)
yy = knn_to_sparse(atac_atac_idx, n_atac, n_atac)
jaccard = (xx @ yx.T) + (xy @ yy.T)
jaccard.data /= (4 * K_TRANSFER - jaccard.data)
row_sums = np.asarray(jaccard.sum(axis=0)).ravel()
row_sums[row_sums == 0] = 1
normalized_jaccard = jaccard.multiply(1.0 / row_sums)

try:
    onehot = OneHotEncoder(sparse_output=True)
except TypeError:
    onehot = OneHotEncoder(sparse=True)
rna_onehot = onehot.fit_transform(rna_labels.reshape(-1, 1))
atac_scores = normalized_jaccard.T @ rna_onehot
atac_pred_idx = np.asarray(atac_scores.argmax(axis=1)).ravel()
atac_labels = onehot.categories_[0][atac_pred_idx]
mean_conf = float(np.asarray(atac_scores.max(axis=1).toarray() if sparse.issparse(atac_scores)
                             else atac_scores.max(axis=1)).mean())
log(f"  ATAC label transfer mean confidence: {mean_conf:.3f}")

# --- Combine + filter modality-imbalanced clusters ------------------------ #
adata.obs['cell_type'] = pd.NA
adata.obs.loc[rna_mask,  'cell_type'] = rna_labels
adata.obs.loc[atac_mask, 'cell_type'] = atac_labels
adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
log(f"  cell_type assigned. Distribution:")
print(adata.obs['cell_type'].value_counts().to_string())

from utils.imbalance_cell_type_handeler import filter_modality_imbalanced_clusters
adata = filter_modality_imbalanced_clusters(
    adata=adata,
    modality_column=modality_col,
    cluster_column='cell_type',
    min_proportion_of_expected=0.05,
    verbose=True,
)
K_c_post = adata.obs['cell_type'].nunique()
log(f"  K_c after imbalanced-cluster filter: {K_c_post}")

# --- Save cell_type back into h5ad via h5py (no X rewrite) ---------------- #
log(f"writing 'cell_type' into {H5}")
h5_write_obs_string_column(H5, 'cell_type', adata.obs['cell_type'].astype(str).values)

gc.collect(); torch.cuda.empty_cache()

# --- SE: default α + autotuned ------------------------------------------- #
from sample_embedding import compute_sample_embedding
from parameter_selection.autotune import run_autotune

out_default = f"{ROOT}/sampledisco_default_v2_celltype-on-harmony"
log(f"default-α SE → {out_default}  (cluster={CLUSTER_KEY}, cmd={CMD_KEY})")
os.makedirs(out_default, exist_ok=True)
compute_sample_embedding(
    adata, out_default,
    sample_col="sample", celltype_col="cell_type",
    cluster_emb_key=CLUSTER_KEY, cmd_emb_key=CMD_KEY,
    modality_col="modality", batch_col="batch",
    save=True, verbose=True,
)

out_tuned = f"{ROOT}/sampledisco_tuned_v2_celltype-on-harmony"
log(f"autotuned SE → {out_tuned}")
os.makedirs(out_tuned, exist_ok=True)
run_autotune(
    adata, out_tuned,
    sample_col="sample", celltype_col="cell_type",
    cluster_emb_key=CLUSTER_KEY, cmd_emb_key=CMD_KEY,
    modality_col="modality", batch_col="batch",
    grouping_col="sev.level",
    save=True, verbose=True,
)
log("DONE")
