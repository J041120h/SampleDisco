#!/usr/bin/env python
"""Sample-embedding ABLATION on the single-omics COVID benchmark.

Starts from the ALREADY-preprocessed cell adata (obsm: X_pca_harmony [sample-
removed -> composition], X_pca_harmony_nosamp [sample-preserved -> RMD]); it
re-runs ONLY the sample-embedding derivation in 5 variants and scores each with
the EXISTING BenchmarkWrapper metric suite. Blocks (A1/A2/A3/RMD) are built once
per size and reused across variants.

Variants:
  proportion_only    composition blocks A1/A2/A3 only            + Harmony
  rmd_only           RMD displacement block only                  + Harmony
  no_batch_removal   A1/A2/A3 + RMD, NO sample-level correction
  linear_regression  A1/A2/A3 + RMD, linear (regress-out) batch removal
  original           A1/A2/A3 + RMD, Harmony (current pipeline)
"""
import argparse, os, sys
import numpy as np
import scanpy as sc

sys.path.insert(0, "/users/hjiang/GenoDistance/code")
sys.path.insert(0, "/users/hjiang/GenoDistance/code/Benchmark_covid")

from sample_embedding.blocks import (
    assemble_units, composition_per_unit, soft_assign, loo_rmd,
    derive_weights, build_emb_from_blocks,
)
from sklearn.cluster import MiniBatchKMeans
from embedding_trajectory import compute_trajectory_from_embedding
from other_benchmark_wrapper import run_benchmarks

CLUSTER_KEY = "X_pca_harmony"          # sample-removed  -> composition
RMD_KEY     = "X_pca_harmony_nosamp"   # sample-preserved -> RMD displacement
SAMPLE_COL, CELLTYPE_COL, BATCH_COL = "sample", "cell_type", "batch"
MEDIUM_K, FINE_K, RMD_DIM, RMD_WEIGHT, PCA_N, SEED = 120, 300, 8, 0.60, 10, 42
VARIANTS = ["proportion_only", "rmd_only", "no_batch_removal",
            "linear_regression", "original"]


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

    Zr = np.asarray(adata.obsm[RMD_KEY], dtype=np.float32)
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
        return build_emb_from_blocks([B["A1"], B["A2"], B["A3"]], w3,
                                     batch_method="harmony", **common)
    if variant == "rmd_only":
        if B["RMD"].shape[1] == 0:
            raise ValueError("RMD block is empty; cannot run rmd_only")
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
    ap.add_argument("--size", required=True)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--outroot", required=True)
    a = ap.parse_args()

    print(f"[ablation covid {a.size}] loading {a.adata}", flush=True)
    adata = sc.read_h5ad(a.adata)
    for k in (CLUSTER_KEY, RMD_KEY):
        if k not in adata.obsm:
            raise KeyError(f"{k} not in obsm: {list(adata.obsm)}")

    print(f"[ablation covid {a.size}] building blocks "
          f"({adata.n_obs} cells)", flush=True)
    B = build_blocks(adata)
    print(f"[ablation covid {a.size}] K_c={B['K_c']} K_med={B['K_med']} "
          f"K_fine={B['K_fine']} RMD_dim={B['RMD'].shape[1]} "
          f"n_units={len(B['unit_ids'])}", flush=True)

    os.makedirs(a.outroot, exist_ok=True)
    summary = os.path.join(a.outroot, f"ablation_summary_covid_{a.size}.csv")
    for v in VARIANTS:
        outdir = os.path.join(a.outroot, f"covid_{a.size}", v)
        os.makedirs(outdir, exist_ok=True)
        emb = assemble(v, B)
        emb_csv = os.path.join(outdir, "embedding.csv")
        emb.to_csv(emb_csv)
        compute_trajectory_from_embedding(
            emb_csv, a.meta, severity_column="sev.level",
            sample_column="sample", save_plot=False, verbose=False)
        pt_csv = os.path.join(outdir, "trajectory", "pseudotime_results.csv")
        run_benchmarks(
            meta_csv_path=a.meta, pseudotime_csv_path=pt_csv,
            embedding_csv_path=emb_csv, method_name=v,
            output_base_dir=outdir, summary_csv_path=summary,
            ari_clustering={"label_col": "sev.level", "k_neighbors": 20,
                            "n_clusters": None, "create_plots": False},
            batch_removal={"k": 15, "include_self": False},
            batch_mixing={"k": 20},
            embedding_visualization={"dpi": 150, "figsize": (10, 4)},
        )
        print(f"[ablation covid {a.size}] {v} done", flush=True)
    print(f"[ablation covid {a.size}] ALL DONE -> {summary}", flush=True)


if __name__ == "__main__":
    main()
