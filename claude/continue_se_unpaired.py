"""Continue the unpaired_diemb pipeline from after gene_activity.

Loads adata_sample.h5ad directly (which has Z_clust / Z_cmd / X_glue
obsm and full cell-level obs). We bypass integrate_preprocess for this
run because (a) it was leaving an obsm-stripped, 5-obs-column artifact
on disk for this dataset, and (b) the slim X in adata_sample.h5ad is
all-zero (upstream gene_activity bug — see task #31), so any X-based
QC would drop all cells anyway. Cell typing + SE + autotune all run on
obsm + obs only, so we can proceed safely without X.

Torch-GPU Leiden + Jaccard-SNN label transfer on Z_clust replaces the
wrapper's sklearn brute-force path (which took hours on ~1M cells).
The original cuml/cupy GPU path is broken on this node (CUDA driver
mismatch); torch works because scglue and harmony already use it.
"""
from __future__ import annotations
import os, sys, time, gc
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import torch
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

BASE      = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics"
ADATA_PATH = f"{BASE}/preprocess/adata_sample.h5ad"
OUT_ADATA  = f"{BASE}/preprocess/adata_sample_celltyped.h5ad"
ROOT      = BASE
CLUSTER_KEY = "Z_clust"
CMD_KEY     = "Z_cmd"
RESOLUTION  = 0.8
K_TRANSFER  = 3
SEED        = 42


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gpu_knn_cosine(query: np.ndarray, ref: np.ndarray, k: int,
                   q_chunk: int = 8_192, r_chunk: int = 65_536) -> np.ndarray:
    """Top-k cosine neighbours on GPU via torch, double-chunked top-k merge."""
    dev = torch.device("cuda")
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
        best_sim = torch.full((b, k), NEG_INF, device=dev, dtype=torch.float32)
        best_idx = torch.full((b, k), -1,      device=dev, dtype=torch.int64)
        for ri in range(0, n_r, r_chunk):
            rj = min(ri + r_chunk, n_r)
            r_gpu = rn[ri:rj].to(dev, non_blocking=True)
            sims = q_gpu @ r_gpu.T
            tk = min(k, sims.shape[1])
            v_loc, i_loc = sims.topk(tk, dim=1)
            i_loc = i_loc + ri
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


def cell_typing_torch(adata: sc.AnnData, cluster_key: str, modality_col: str,
                      resolution: float = RESOLUTION, k_transfer: int = K_TRANSFER) -> sc.AnnData:
    """Leiden on RNA via igraph + torch-GPU Jaccard-SNN label transfer to ATAC."""
    rna_mask  = (adata.obs[modality_col] == 'RNA').values
    atac_mask = (adata.obs[modality_col] == 'ATAC').values
    n_rna, n_atac = int(rna_mask.sum()), int(atac_mask.sum())
    log(f"  cell typing on {cluster_key}: RNA={n_rna:,}  ATAC={n_atac:,}")

    emb = np.asarray(adata.obsm[cluster_key], dtype=np.float32)
    rna_emb  = emb[rna_mask]
    atac_emb = emb[atac_mask]

    log("  Step 1: sc.pp.neighbors + Leiden on RNA (igraph backend)")
    rna_a = sc.AnnData(X=np.zeros((n_rna, 1), dtype=np.float32))
    rna_a.obsm[cluster_key] = rna_emb
    t0 = time.time()
    sc.pp.neighbors(rna_a, use_rep=cluster_key, n_neighbors=15, random_state=SEED)
    sc.tl.leiden(rna_a, resolution=resolution, random_state=SEED,
                 key_added='cell_type', flavor='igraph', n_iterations=2, directed=False)
    rna_lab_int = rna_a.obs['cell_type'].astype(int).values
    rna_lab = (rna_lab_int + 1).astype(str)
    K_c = int(rna_lab_int.max() + 1)
    log(f"  Leiden done in {time.time()-t0:.1f}s; K_c={K_c}")
    del rna_a; gc.collect()

    if n_atac > 0:
        log(f"  Step 2: GPU KNN (k={k_transfer}, 4 builds via torch on {torch.cuda.get_device_name(0)})")
        t = time.time(); rr = gpu_knn_cosine(rna_emb,  rna_emb,  k_transfer); log(f"    rna→rna  {time.time()-t:.1f}s")
        t = time.time(); ra = gpu_knn_cosine(rna_emb,  atac_emb, k_transfer); log(f"    rna→atac {time.time()-t:.1f}s")
        t = time.time(); ar = gpu_knn_cosine(atac_emb, rna_emb,  k_transfer); log(f"    atac→rna {time.time()-t:.1f}s")
        t = time.time(); aa = gpu_knn_cosine(atac_emb, atac_emb, k_transfer); log(f"    atac→atac {time.time()-t:.1f}s")

        log("  Step 3: Jaccard-SNN + label transfer")
        xx = knn_to_sparse(rr,  n_rna,  n_rna)
        xy = knn_to_sparse(ra,  n_rna,  n_atac)
        yx = knn_to_sparse(ar,  n_atac, n_rna)
        yy = knn_to_sparse(aa,  n_atac, n_atac)
        jac = (xx @ yx.T) + (xy @ yy.T)
        jac.data /= (4 * k_transfer - jac.data)
        rs = np.asarray(jac.sum(axis=0)).ravel(); rs[rs == 0] = 1
        njac = jac.multiply(1.0 / rs)
        try:
            ohe = OneHotEncoder(sparse_output=True)
        except TypeError:
            ohe = OneHotEncoder(sparse=True)
        rna_oh = ohe.fit_transform(rna_lab.reshape(-1, 1))
        atac_scores = njac.T @ rna_oh
        atac_lab = ohe.categories_[0][np.asarray(atac_scores.argmax(axis=1)).ravel()]
    else:
        atac_lab = np.array([], dtype=object)

    adata.obs['cell_type'] = pd.NA
    adata.obs.loc[rna_mask,  'cell_type'] = rna_lab
    if n_atac > 0:
        adata.obs.loc[atac_mask, 'cell_type'] = atac_lab
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')

    from utils.imbalance_cell_type_handler import filter_modality_imbalanced_clusters
    adata = filter_modality_imbalanced_clusters(
        adata=adata, modality_column=modality_col, cluster_column='cell_type',
        min_proportion_of_expected=0.05, verbose=True)
    log(f"  K_c post-filter: {adata.obs['cell_type'].nunique()}")
    return adata


# ───────────────────────────────────────────────────────────────────────── #
log(f"Loading {ADATA_PATH}")
adata = sc.read_h5ad(ADATA_PATH)
log(f"  shape={adata.shape}  obsm={list(adata.obsm.keys())}")
for k in (CLUSTER_KEY, CMD_KEY):
    if k not in adata.obsm:
        raise KeyError(f"required obsm key {k!r} missing from {ADATA_PATH}")
log(f"  torch cuda? {torch.cuda.is_available()}  device: {torch.cuda.get_device_name(0)}")

# Cell typing
adata = cell_typing_torch(adata, cluster_key=CLUSTER_KEY, modality_col='modality')

# Save to a sibling file rather than overwriting adata_sample.h5ad (avoids
# corrupting the gene-activity output if something below fails).
log(f"Writing cell_type-annotated adata to {OUT_ADATA}")
adata.write(OUT_ADATA, compression='gzip')
log("  saved")

gc.collect(); torch.cuda.empty_cache()

# SE + autotune
from sample_embedding import compute_sample_embedding
from parameter_selection.autotune import run_autotune

out_dir = ROOT
os.makedirs(f"{out_dir}/sample_embedding", exist_ok=True)
log(f"Sample embedding + autotune → {out_dir}/sample_embedding/")
run_autotune(
    adata, out_dir,
    sample_col='sample', celltype_col='cell_type',
    cluster_emb_key=CLUSTER_KEY, cmd_emb_key=CMD_KEY,
    modality_col='modality', batch_col='batch',
    grouping_col='sev.level',
    save=True, verbose=True,
)
log("DONE")
