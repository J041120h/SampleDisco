"""V2 multi-omics pipeline + benchmarks on 4 datasets.

For each of {ENCODE, eye_retina, eye_lutea, heart}:
    1. Memory-safe load (obs + obsm only, X dropped).
    2. Dual GPU Harmony on X_glue (harmony-pytorch on V100):
         X_glue_harmony         — Harmony with [batch, sample]
         X_glue_harmony_nosamp  — Harmony with [batch] only
    3. Re-cluster cell_type on X_glue_harmony at resolution 0.8
       (igraph Leiden + torch-GPU-KNN label transfer to ATAC).
    4. SE default-α and SE autotuned (cluster=X_glue_harmony, cmd=X_glue_harmony_nosamp).
    5. Benchmark V2 SE against all existing competitors (same scoring code as upstream).

Outputs are written under /dcs07/hongkai/data/harry/result/test/test_run_multiomics/<dataset>/.
The source h5ads under /dcs07/.../multi_omics_*/ are NEVER modified.
"""
from __future__ import annotations
import os, sys, time, gc, json, shutil
sys.path.insert(0, "/users/hjiang/GenoDistance/code")
sys.path.insert(0, "/users/hjiang/GenoDistance/code/Benchmark_multiomics")

import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import torch
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

TEST_ROOT = "/dcs07/hongkai/data/harry/result/test/test_run_multiomics"
os.makedirs(TEST_ROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset configs.  meta_csv / tissue_col / age_col / disease_state_col are
# the upstream benchmark conventions.  cluster_emb_key / cmd_emb_key are
# set after harmonize_xglue runs (auto-fallback in _resolve_cmd_emb_key).
# ---------------------------------------------------------------------------
DATASETS = {
    "ENCODE": dict(
        h5         = "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/preprocess/adata_sample.h5ad",
        sample_col = "sample",
        modality_col = "modality",
        batch_col  = None,                            # ENCODE has no batch col
        grouping_col = "tissue",                      # autotune target & benchmark col
        bench_module = "benchmark_metircs_ENCODE",
        bench_meta_csv = "/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        bench_eval_kwargs = dict(k_neighbors=5, n_permutations=1000),
        competitors = {
            "SD_expression": "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/embeddings/sample_expression_embedding.csv",
            "SD_proportion": "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/embeddings/sample_proportion_embedding.csv",
            "pilot":         "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/pilot/pilot_native_embedding.csv",
            "pseudobulk":    "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/pseudobulk/pseudobulk/pca_embeddings.csv",
            "QOT":           "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/QOT/88_qot_distance_matrix_mds_10d.csv",
            "GEDI":          "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/GEDI/gedi_sample_embedding.csv",
            "Gloscope":      "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/Gloscope/knn_divergence_mds_10d.csv",
            "MFA":           "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/MFA/sample_embeddings.csv",
            "mustard":       "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/mustard/sample_embedding.csv",
            "scPoli":        "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/scPoli/sample_embeddings_full.csv",
        },
    ),
    "eye_retina": dict(
        h5         = "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/preprocess/atac_rna_integrated.h5ad",
        sample_col = "sample",
        modality_col = "modality",
        batch_col  = "batch",
        grouping_col = "age",
        bench_module = "benchmark_eye",
        bench_meta_csv = "/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        bench_eval_kwargs = dict(k_neighbors=3),
        competitors = {
            "SD_expression": "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/embeddings/sample_expression_embedding.csv",
            "SD_proportion": "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/embeddings/sample_proportion_embedding.csv",
            "pilot":         "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/pilot/wasserstein_distance_mds_10d.csv",
            "pseudobulk":    "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/pseudobulk/pseudobulk/pca_embeddings.csv",
            "QOT":           "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/QOT/24_qot_distance_matrix_mds_10d.csv",
            "GEDI":          "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/GEDI/gedi_sample_embedding.csv",
            "Gloscope":      "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/Gloscope/knn_divergence_mds_10d.csv",
            "MFA":           "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/MFA/sample_embeddings.csv",
            "mustard":       "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/mustard/sample_embedding.csv",
            "scPoli":        "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/scPoli/sample_embeddings_full.csv",
        },
    ),
    "eye_lutea": dict(
        h5         = "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/preprocess/atac_rna_integrated.h5ad",
        sample_col = "sample",
        modality_col = "modality",
        batch_col  = "batch",
        grouping_col = "age",
        bench_module = "benchmark_eye",
        bench_meta_csv = "/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        bench_eval_kwargs = dict(k_neighbors=3),
        competitors = {
            "SD_expression": "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/embeddings/sample_expression_embedding.csv",
            "SD_proportion": "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/embeddings/sample_proportion_embedding.csv",
            "pilot":         "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/pilot/wasserstein_distance_mds_10d.csv",
            "pseudobulk":    "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/pseudobulk/pseudobulk/pca_embeddings.csv",
            "QOT":           "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/QOT/24_qot_distance_matrix_mds_10d.csv",
            "GEDI":          "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/GEDI/gedi_sample_embedding.csv",
            "Gloscope":      "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/Gloscope/knn_divergence_mds_10d.csv",
            "MFA":           "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/MFA/sample_embeddings.csv",
            "mustard":       "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/mustard/sample_embedding.csv",
            "scPoli":        "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/scPoli/sample_embeddings_full.csv",
        },
    ),
    "heart": dict(
        h5         = "/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/preprocess/atac_rna_integrated.h5ad",
        sample_col = "sample",
        modality_col = "modality",
        batch_col  = "batch",
        grouping_col = "disease_state",
        bench_module = "benchmark_heart",
        bench_meta_csv = "/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        bench_eval_kwargs = dict(k_neighbors=15),
        competitors = {
            "SD_expression": "/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/embeddings/sample_expression_embedding.csv",
            "SD_proportion": "/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/embeddings/sample_proportion_embedding.csv",
            "pilot":         "/dcs07/hongkai/data/harry/result/multi_omics_heart/pilot/wasserstein_distance_mds_10d.csv",
            "pseudobulk":    "/dcs07/hongkai/data/harry/result/multi_omics_heart/pseudobulk/pseudobulk/pca_embeddings.csv",
            "QOT":           "/dcs07/hongkai/data/harry/result/multi_omics_heart/QOT/44_qot_distance_matrix_mds_10d.csv",
            "GEDI":          "/dcs07/hongkai/data/harry/result/multi_omics_heart/GEDI/gedi_sample_embedding.csv",
            "Gloscope":      "/dcs07/hongkai/data/harry/result/multi_omics_heart/Gloscope/knn_divergence_mds_10d.csv",
            "MFA":           "/dcs07/hongkai/data/harry/result/multi_omics_heart/MFA/sample_embeddings.csv",
            "mustard":       "/dcs07/hongkai/data/harry/result/multi_omics_heart/mustard/sample_embedding.csv",
            "scPoli":        "/dcs07/hongkai/data/harry/result/multi_omics_heart/scPoli/sample_embeddings_full.csv",
        },
    ),
}

SEED = 42
RESOLUTION = 0.8
K_TRANSFER = 3


def log(m: str) -> None: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# helpers reused from pipeline_unpaired_test_v2
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


def gpu_knn_cosine(query: np.ndarray, ref: np.ndarray, k: int,
                   q_chunk: int = 8_192, r_chunk: int = 65_536) -> np.ndarray:
    """Top-k cosine neighbours of query rows among ref rows on the GPU."""
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


def cell_typing_v2(adata: sc.AnnData, cluster_key: str, modality_col: str,
                   resolution: float = RESOLUTION, k_transfer: int = K_TRANSFER) -> sc.AnnData:
    """Re-cluster cell_type on `cluster_key` via Leiden on RNA + GPU-KNN
    Jaccard-SNN label transfer to ATAC. Mirrors the standard multi-omics
    cell typing function but with GPU KNN.

    Returns the (possibly filtered) AnnData with cell_type filled. Caller
    MUST replace its adata reference with the return value."""
    rna_mask  = (adata.obs[modality_col] == 'RNA').values
    atac_mask = (adata.obs[modality_col] == 'ATAC').values
    n_rna, n_atac = int(rna_mask.sum()), int(atac_mask.sum())
    log(f"  cell-typing: RNA={n_rna:,}  ATAC={n_atac:,}  use_rep={cluster_key}")
    emb = np.asarray(adata.obsm[cluster_key], dtype=np.float32)

    # Step 1 — Leiden on RNA (igraph)
    rna_a = sc.AnnData(X=np.zeros((n_rna, 1), dtype=np.float32))
    rna_a.obsm[cluster_key] = emb[rna_mask]
    t0 = time.time()
    sc.pp.neighbors(rna_a, use_rep=cluster_key, n_neighbors=15, random_state=SEED)
    sc.tl.leiden(rna_a, resolution=resolution, random_state=SEED, key_added='cell_type',
                 flavor='igraph', n_iterations=2, directed=False)
    rna_lab_int = rna_a.obs['cell_type'].astype(int).values
    rna_lab = (rna_lab_int + 1).astype(str)
    K_c = int(rna_lab_int.max() + 1)
    log(f"    Leiden done in {time.time()-t0:.1f}s; K_c={K_c}")
    del rna_a; gc.collect()

    if n_atac > 0:
        rna_emb  = emb[rna_mask]
        atac_emb = emb[atac_mask]
        log("    Step 2: GPU KNN (4 builds via torch)")
        t = time.time(); rr = gpu_knn_cosine(rna_emb,  rna_emb,  k_transfer); log(f"      rna→rna  {time.time()-t:.1f}s")
        t = time.time(); ra = gpu_knn_cosine(rna_emb,  atac_emb, k_transfer); log(f"      rna→atac {time.time()-t:.1f}s")
        t = time.time(); ar = gpu_knn_cosine(atac_emb, rna_emb,  k_transfer); log(f"      atac→rna {time.time()-t:.1f}s")
        t = time.time(); aa = gpu_knn_cosine(atac_emb, atac_emb, k_transfer); log(f"      atac→atac {time.time()-t:.1f}s")

        log("    Step 3: Jaccard-SNN + label transfer")
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

    from utils.imbalance_cell_type_handeler import filter_modality_imbalanced_clusters
    adata = filter_modality_imbalanced_clusters(
        adata=adata, modality_column=modality_col, cluster_column='cell_type',
        min_proportion_of_expected=0.05, verbose=True)
    log(f"    K_c post-filter: {adata.obs['cell_type'].nunique()}")
    return adata


# --------------------------------------------------------------------------- #
# Per-dataset SE run
# --------------------------------------------------------------------------- #
def run_se_one_dataset(name: str, cfg: dict, out_root: str) -> dict:
    log(f"========== SE: {name} ==========")
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    timings = {}

    t0 = time.time()
    a = load_minimal(cfg['h5'])
    log(f"  shape={a.shape}  obsm={list(a.obsm.keys())}")
    timings['load'] = time.time() - t0

    # Sanity / derive batch
    batch_col = cfg.get('batch_col')
    if batch_col is not None and batch_col not in a.obs.columns:
        raise RuntimeError(f"{name}: batch_col={batch_col} not in obs")
    if batch_col is None:
        log("  (no batch col: harmonize_xglue will run sample-only pass)")

    # Dual Harmony
    from preparation.multi_omics_batch_correction import (
        harmonize_xglue, XGLUE_HARMONY_KEY, XGLUE_HARMONY_NOSAMP,
    )
    t0 = time.time()
    a = harmonize_xglue(a, batch_col=batch_col, sample_col=cfg['sample_col'],
                        use_gpu=True, max_iter=50, random_state=SEED, verbose=True)
    timings['harmony'] = time.time() - t0
    log(f"  harmony: {timings['harmony']:.1f}s; obsm now {list(a.obsm.keys())}")

    cluster_emb_key = XGLUE_HARMONY_KEY if XGLUE_HARMONY_KEY in a.obsm else 'X_glue'
    cmd_emb_key     = XGLUE_HARMONY_NOSAMP if XGLUE_HARMONY_NOSAMP in a.obsm else cluster_emb_key

    # Cell typing on cluster_emb_key
    t0 = time.time()
    a = cell_typing_v2(a, cluster_key=cluster_emb_key, modality_col=cfg['modality_col'])
    timings['cell_typing'] = time.time() - t0

    # SE default
    from sample_embedding import compute_sample_embedding
    from parameter_selection.autotune import run_autotune
    out_default = os.path.join(out_dir, "sampledisco_default_v2")
    os.makedirs(out_default, exist_ok=True)
    t0 = time.time()
    compute_sample_embedding(
        a, out_default,
        sample_col=cfg['sample_col'], celltype_col='cell_type',
        cluster_emb_key=cluster_emb_key, cmd_emb_key=cmd_emb_key,
        modality_col=cfg['modality_col'], batch_col=batch_col,
        save=True, verbose=True,
    )
    timings['se_default'] = time.time() - t0
    log(f"  default-α SE saved: {out_default}/sample_embedding/sample_embedding.csv")

    # SE autotuned
    out_tuned = os.path.join(out_dir, "sampledisco_tuned_v2")
    os.makedirs(out_tuned, exist_ok=True)
    t0 = time.time()
    run_autotune(
        a, out_tuned,
        sample_col=cfg['sample_col'], celltype_col='cell_type',
        cluster_emb_key=cluster_emb_key, cmd_emb_key=cmd_emb_key,
        modality_col=cfg['modality_col'], batch_col=batch_col,
        grouping_col=cfg.get('grouping_col'),
        save=True, verbose=True,
    )
    timings['se_tuned'] = time.time() - t0
    log(f"  tuned SE saved: {out_tuned}/sample_embedding/sample_embedding.csv")

    K_c = int(a.obs['cell_type'].nunique())
    with open(os.path.join(out_dir, "pipeline_meta.json"), 'w') as f:
        json.dump({"K_c": K_c, "timings_sec": timings, "config": {k: v for k, v in cfg.items() if k != 'competitors'}}, f, indent=2)
    del a; gc.collect(); torch.cuda.empty_cache()
    return {"name": name, "K_c": K_c, "timings": timings,
            "default_csv": f"{out_default}/sample_embedding/sample_embedding.csv",
            "tuned_csv":   f"{out_tuned}/sample_embedding/sample_embedding.csv"}


# --------------------------------------------------------------------------- #
# Per-dataset benchmark run (V2 + competitors → fresh summary.csv)
# --------------------------------------------------------------------------- #
def run_bench_one_dataset(name: str, cfg: dict, se_result: dict, out_root: str) -> str:
    log(f"========== BENCH: {name} ==========")
    bench_root = os.path.join(out_root, name, "Benchmark_result")
    os.makedirs(bench_root, exist_ok=True)
    summary_csv = os.path.join(bench_root, "summary.csv")
    # general_outdir is what each benchmark uses to write per-method folders
    general_outdir = os.path.join(out_root, name)

    mod = __import__(cfg['bench_module'])
    evaluate_fn = mod.evaluate_multimodal_integration
    save_fn     = mod.save_to_summary_csv

    methods = dict(cfg['competitors'])
    methods["SD_v2_tuned"]   = se_result["tuned_csv"]
    methods["SD_v2_default"] = se_result["default_csv"]

    for mname, ecsv in methods.items():
        if not os.path.exists(ecsv):
            log(f"  SKIP {mname} (missing csv: {ecsv})")
            continue
        log(f"  evaluating {mname} ← {ecsv}")
        try:
            res = evaluate_fn(
                meta_csv=cfg['bench_meta_csv'],
                embedding_csv=ecsv,
                method_name=mname,
                general_outdir=general_outdir,
                **cfg.get('bench_eval_kwargs', {}),
            )
            save_fn(res, summary_csv)
        except Exception as e:
            log(f"    !! {mname} failed: {type(e).__name__}: {e}")
    log(f"  summary at: {summary_csv}")
    return summary_csv


# --------------------------------------------------------------------------- #
# Master
# --------------------------------------------------------------------------- #
def main():
    log(f"torch cuda? {torch.cuda.is_available()}  device: "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    se_results = {}
    for name, cfg in DATASETS.items():
        try:
            se_results[name] = run_se_one_dataset(name, cfg, TEST_ROOT)
        except Exception as e:
            log(f"!! {name} SE failed: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue

    bench_csvs = {}
    for name, cfg in DATASETS.items():
        if name not in se_results:
            continue
        try:
            bench_csvs[name] = run_bench_one_dataset(name, cfg, se_results[name], TEST_ROOT)
        except Exception as e:
            log(f"!! {name} BENCH failed: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    log("========== DONE ALL ==========")
    log(f"summaries: {bench_csvs}")


if __name__ == "__main__":
    main()
