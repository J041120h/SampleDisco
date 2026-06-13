#!/usr/bin/env python
"""Sample-embedding ABLATION on the PAIRED multi-omics benchmarks
(ENCODE / heart / retina / lutea).

Same 5 variants as COVID, but units = sample x modality and the RMD/Harmony
GROUP is modality (RNA vs ATAC). 'no_batch_removal' therefore skips the
MODALITY-MIXING Harmony (expected to hurt modality mixing / paired matching).
Each dataset is scored with its OWN existing scorer module
(benchmark_metrics_ENCODE / benchmark_heart / benchmark_eye), whose metrics are
paired_partner_rank, ASW_modality, iLISI, and a condition-preservation score
(tissue / disease_state / cca-vs-age).
"""
import argparse, importlib, os, sys
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, "/users/hjiang/GenoDistance/code")
sys.path.insert(0, "/users/hjiang/GenoDistance/code/Benchmark_multiomics")

from sampledisco.sample_embedding.blocks import (
    assemble_units, composition_per_unit, soft_assign, loo_rmd,
    derive_weights, build_emb_from_blocks,
)
from sklearn.cluster import MiniBatchKMeans

MEDIUM_K, FINE_K, RMD_DIM, RMD_WEIGHT, PCA_N, SEED = 120, 300, 8, 0.60, 10, 42
VARIANTS = ["proportion_only", "rmd_only", "no_batch_removal",
            "linear_regression", "original"]

# per-dataset: input adata, cluster/rmd obsm keys, scorer module, meta, cond kwarg
DATASETS = {
    "ENCODE": dict(
        adata="/dcs07/hongkai/data/harry/result/Benchmark_multiomics/adata_cell.h5ad",
        cluster_key="X_pca_harmony", rmd_key="X_pca",
        scorer="benchmark_metrics_ENCODE",
        meta="/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        cond=dict(tissue_col="tissue")),
    "heart": dict(
        adata="/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/preprocess/atac_rna_integrated.h5ad",
        cluster_key="X_glue", rmd_key="X_glue",
        scorer="benchmark_heart",
        meta="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        cond=dict(disease_state_col="disease_state")),
    "retina": dict(
        adata="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/preprocess/atac_rna_integrated.h5ad",
        cluster_key="X_glue", rmd_key="X_glue", scorer="benchmark_eye",
        meta="BUILD", cond=dict(age_col="age")),
    "lutea": dict(
        adata="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/preprocess/atac_rna_integrated.h5ad",
        cluster_key="X_glue", rmd_key="X_glue", scorer="benchmark_eye",
        meta="BUILD", cond=dict(age_col="age")),
}


def build_meta_from_obs(adata, out_csv):
    """Build a per-(bio)sample meta CSV from adata.obs for the eye scorer."""
    obs = adata.obs.copy()
    obs["__bio__"] = obs["sample"].astype(str)
    keep = [c for c in ["age", "tissue", "disease_state", "sex"] if c in obs.columns]
    meta = obs.groupby("__bio__")[keep].first()
    meta.index.name = "sample"
    meta.to_csv(out_csv)
    return out_csv


def build_blocks(adata, cluster_key, rmd_key):
    units, unit_cellids, unit_ids, unit_groups, unit_batches, all_cellids, Z = \
        assemble_units(adata, "sample", cluster_key,
                       modality_col="modality", batch_col=None)
    cellid_idx = {c: i for i, c in enumerate(all_cellids)}
    ucl = [unit_cellids[u] for u in unit_ids]

    ct = adata.obs["cell_type"].astype(str).values
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

    Zr = np.asarray(adata.obsm[rmd_key], dtype=np.float32)
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
    w4 = derive_weights(B["K_c"], B["K_med"], B["K_fine"],
                        rmd_weight=RMD_WEIGHT, n_blocks=4)
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
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--outroot", required=True)
    a = ap.parse_args()
    cfg = DATASETS[a.dataset]
    out = os.path.join(a.outroot, a.dataset)
    os.makedirs(out, exist_ok=True)

    print(f"[mo {a.dataset}] loading {cfg['adata']}", flush=True)
    adata = sc.read_h5ad(cfg["adata"])
    for k in (cfg["cluster_key"], cfg["rmd_key"]):
        if k not in adata.obsm:
            raise KeyError(f"{k} not in obsm {list(adata.obsm)}")

    meta = cfg["meta"]
    if meta == "BUILD":
        meta = build_meta_from_obs(adata, os.path.join(out, "meta_built.csv"))
        print(f"[mo {a.dataset}] built meta -> {meta}", flush=True)

    scorer = importlib.import_module(cfg["scorer"])
    evaluate = scorer.evaluate_multimodal_integration
    save_sum = scorer.save_to_summary_csv

    print(f"[mo {a.dataset}] building blocks ({adata.n_obs} cells)", flush=True)
    B = build_blocks(adata, cfg["cluster_key"], cfg["rmd_key"])
    print(f"[mo {a.dataset}] K_c={B['K_c']} K_med={B['K_med']} K_fine={B['K_fine']} "
          f"RMD_dim={B['RMD'].shape[1]} n_units={len(B['unit_ids'])}", flush=True)

    summary = os.path.join(out, f"ablation_summary_mo_{a.dataset}.csv")
    for v in VARIANTS:
        vout = os.path.join(out, v); os.makedirs(vout, exist_ok=True)
        emb = assemble(v, B)
        emb_csv = os.path.join(vout, "embedding.csv"); emb.to_csv(emb_csv)
        res = evaluate(meta_csv=meta, embedding_csv=emb_csv, method_name=v,
                       general_outdir=vout, n_permutations=100,
                       create_visualizations=False, **cfg["cond"])
        save_sum(res, summary)
        print(f"[mo {a.dataset}] {v} done", flush=True)
    print(f"[mo {a.dataset}] ALL DONE -> {summary}", flush=True)


if __name__ == "__main__":
    main()
