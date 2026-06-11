#!/usr/bin/env python
"""Sample-embedding ABLATION on the health-aging PBMC benchmark (single-omics).

Same 5 variants as COVID. Cell adata has obsm Z_clust (sample-removed ->
composition) and Z_cmd (sample-preserved -> RMD). Reads the adata LITE
(obsm + obs only; the 16679-gene X is never needed) to avoid OOM on 1.9M cells.
Scores each variant with the dataset's own 4-test suite
(P1_age / P2_cd4cd8_ratio / P3_mait / G1_batch) and writes per-variant JSONs.
"""
import argparse, json, os, subprocess, sys
import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, "/users/hjiang/GenoDistance/code")
from sample_embedding.blocks import (
    assemble_units, composition_per_unit, soft_assign, loo_rmd,
    derive_weights, build_emb_from_blocks,
)
from sklearn.cluster import MiniBatchKMeans

ADATA = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/preprocess/adata_preprocessed.h5ad"
BENCH = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/benchmark"
META = f"{BENCH}/meta_pbmc.csv"
SCRIPTS = f"{BENCH}/scripts"
CLUSTER_KEY, RMD_KEY = "Z_clust", "Z_cmd"
# Tube_id is the benchmark sample unit (316, == meta 'sample'); Donor_id (166) is coarser.
SAMPLE_COL, CELLTYPE_COL, BATCH_COL = "Tube_id", "Cluster_names", "Batch"
MEDIUM_K, FINE_K, RMD_DIM, RMD_WEIGHT, PCA_N, SEED = 120, 300, 8, 0.60, 10, 42
VARIANTS = ["proportion_only", "rmd_only", "no_batch_removal",
            "linear_regression", "original"]
TESTS = [("P1_age", "P1"), ("P2_cd4cd8_ratio", "P2"),
         ("P3_mait", "P3"), ("G1_batch", "G1")]


def read_lite(path):
    """Load only obsm[cluster,rmd] + obs[sample,celltype,batch] (skip X)."""
    ab = ad.read_h5ad(path, backed="r")
    obs = ab.obs[[SAMPLE_COL, CELLTYPE_COL, BATCH_COL]].astype(str).copy()
    lite = ad.AnnData(obs=obs)
    lite.obs_names = ab.obs_names.astype(str)
    lite.obsm[CLUSTER_KEY] = np.asarray(ab.obsm[CLUSTER_KEY], dtype=np.float32)
    lite.obsm[RMD_KEY] = np.asarray(ab.obsm[RMD_KEY], dtype=np.float32)
    return lite


def build_blocks(adata):
    units, unit_cellids, unit_ids, unit_groups, unit_batches, all_cellids, Z = \
        assemble_units(adata, SAMPLE_COL, CLUSTER_KEY,
                       modality_col=None, batch_col=BATCH_COL)
    cellid_idx = {c: i for i, c in enumerate(all_cellids)}
    ucl = [unit_cellids[u] for u in unit_ids]
    ct = adata.obs[CELLTYPE_COL].astype(str).values
    uniq = sorted(set(ct)); K_c = len(uniq); L1 = {c: i for i, c in enumerate(uniq)}
    soft1 = np.zeros((Z.shape[0], K_c), dtype=np.float32)
    for i, c in enumerate(ct):
        soft1[i, L1[c]] = 1.0
    A1 = composition_per_unit(ucl, soft1, cellid_idx)
    K_med = min(MEDIUM_K, max(2, Z.shape[0] // 200))
    km = MiniBatchKMeans(n_clusters=K_med, random_state=SEED, batch_size=4096,
                         n_init=5, max_iter=200).fit(Z)
    A2 = composition_per_unit(ucl, soft_assign(Z, km.cluster_centers_), cellid_idx)
    K_fine = min(FINE_K, max(2, Z.shape[0] // 100))
    km = MiniBatchKMeans(n_clusters=K_fine, random_state=SEED + 1, batch_size=4096,
                         n_init=5, max_iter=200).fit(Z)
    A3 = composition_per_unit(ucl, soft_assign(Z, km.cluster_centers_), cellid_idx)
    Zr = adata.obsm[RMD_KEY]
    rmd_units = []
    for uid, grp in zip(unit_ids, unit_groups):
        idxs = [cellid_idx[c] for c in unit_cellids[uid] if c in cellid_idx]
        rmd_units.append((uid, grp, Zr[idxs]))
    coarse = dict(zip(all_cellids, ct))
    RMD = loo_rmd(rmd_units, unit_cellids, coarse,
                  max_dim_per_cluster=RMD_DIM, seed=SEED, loo=True, verbose=False)
    return dict(A1=A1, A2=A2, A3=A3, RMD=RMD, K_c=K_c, K_med=K_med, K_fine=K_fine,
                unit_ids=unit_ids, unit_groups=unit_groups, unit_batches=unit_batches)


def assemble(variant, B):
    common = dict(unit_ids=B["unit_ids"], unit_groups=B["unit_groups"],
                  unit_batches=B["unit_batches"], pca_components=PCA_N,
                  seed=SEED, verbose=False)
    w3 = derive_weights(B["K_c"], B["K_med"], B["K_fine"], n_blocks=3)
    w4 = derive_weights(B["K_c"], B["K_med"], B["K_fine"], rmd_weight=RMD_WEIGHT, n_blocks=4)
    full = [B["A1"], B["A2"], B["A3"], B["RMD"]]
    if variant == "proportion_only":
        return build_emb_from_blocks([B["A1"], B["A2"], B["A3"]], w3, batch_method="harmony", **common)
    if variant == "rmd_only":
        if B["RMD"].shape[1] == 0:
            raise ValueError("RMD block empty")
        return build_emb_from_blocks([B["RMD"]], [1.0], batch_method="harmony", **common)
    if variant == "no_batch_removal":
        return build_emb_from_blocks(full, w4, batch_method="none", **common)
    if variant == "linear_regression":
        return build_emb_from_blocks(full, w4, batch_method="linear", **common)
    if variant == "original":
        return build_emb_from_blocks(full, w4, batch_method="harmony", **common)
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", required=True)
    a = ap.parse_args()
    out = a.outroot; os.makedirs(out, exist_ok=True)
    print("[ha] reading adata lite", flush=True)
    adata = read_lite(ADATA)
    print(f"[ha] {adata.n_obs} cells; building blocks", flush=True)
    B = build_blocks(adata)
    print(f"[ha] K_c={B['K_c']} K_med={B['K_med']} K_fine={B['K_fine']} "
          f"RMD_dim={B['RMD'].shape[1]} n_units={len(B['unit_ids'])}", flush=True)

    env = dict(os.environ, PYTHONNOUSERSITE="1")
    for v in VARIANTS:
        vout = os.path.join(out, v); os.makedirs(vout, exist_ok=True)
        emb = assemble(v, B)
        emb_csv = os.path.join(vout, "embedding.csv"); emb.to_csv(emb_csv)
        for tag, short in TESTS:
            subprocess.run(
                ["/users/hjiang/.conda/envs/hongkai/bin/python", "-u", f"{tag}.py",
                 "--embedding", emb_csv, "--meta", META,
                 "--out", os.path.join(vout, f"{short}.json")],
                cwd=SCRIPTS, env=env, check=True)
        print(f"[ha] {v} done", flush=True)
    print(f"[ha] ALL DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
