"""Run a single (dataset, variant) of the SampleDisco sweep.
Designed to be invoked as a subprocess so each variant gets fresh memory.

Usage: python sd_param_sweep_one.py <dataset> <variant>
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, "/users/hjiang/GenoDistance/code")
sys.path.insert(0, "/users/hjiang/GenoDistance/code/Benchmark_multiomics")

import numpy as np, pandas as pd, scanpy as sc
from sklearn.cluster import MiniBatchKMeans

from sample_embedding import compute_sample_embedding
from sample_embedding.blocks import (
    assemble_units, soft_assign, composition_per_unit, derive_weights,
    loo_cmd, build_emb_from_blocks,
)

OUT = "/dcs07/hongkai/data/harry/result/CLAUDE_SD_SWEEP"

DATASETS = {
    "ENCODE": dict(
        h5="/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/preprocess/adata_sample.h5ad",
        meta="/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        outroot="/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics",
        evaluator="encode", k=5,
    ),
    "Lutea": dict(
        h5="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/preprocess/atac_rna_integrated.h5ad",
        meta="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        outroot="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea",
        evaluator="eye", k=3,
    ),
    "Retina": dict(
        h5="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/preprocess/atac_rna_integrated.h5ad",
        meta="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        outroot="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina",
        evaluator="eye", k=3,
    ),
    "Heart": dict(
        h5="/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/preprocess/atac_rna_integrated.h5ad",
        meta="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        outroot="/dcs07/hongkai/data/harry/result/multi_omics_heart",
        evaluator="heart", k=15,
    ),
}


def load_minimal(h5: str) -> sc.AnnData:
    """Load adata WITHOUT the X matrix to save memory (we only need obs + obsm)."""
    import anndata as ad
    a = ad.read_h5ad(h5, backed='r')
    # Materialize obs and obsm into memory; discard X
    n = a.shape[0]
    obs_df = a.obs.copy()
    obsm = {k: np.asarray(a.obsm[k]) for k in a.obsm.keys()}
    a.file.close()
    new = sc.AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs_df)
    for k, v in obsm.items():
        new.obsm[k] = v
    return new


def make_se(adata, variant: str, outdir: str):
    cluster_emb_key, cmd_emb_key = "X_glue", "X_glue"
    sample_col, celltype_col, modality_col = "sample", "cell_type", "modality"

    if variant == "A1_soft":
        units, unit_cellids, unit_ids, unit_groups, unit_batches, all_cellids, Z = \
            assemble_units(adata, sample_col, cluster_emb_key,
                           modality_col=modality_col, batch_col=None)
        cellid_idx = {c: i for i, c in enumerate(all_cellids)}
        ct = adata.obs[celltype_col].astype(str).values
        K_c = len(set(ct))
        K_med = min(120, max(2, Z.shape[0] // 200))
        K_fine = min(300, max(2, Z.shape[0] // 100))
        km1 = MiniBatchKMeans(n_clusters=K_c, random_state=42, batch_size=4096,
                                n_init=5, max_iter=200).fit(Z)
        unit_cellids_list = [unit_cellids[u] for u in unit_ids]
        A1 = composition_per_unit(unit_cellids_list,
                                    soft_assign(Z, km1.cluster_centers_), cellid_idx)
        km2 = MiniBatchKMeans(n_clusters=K_med, random_state=42, batch_size=4096,
                                n_init=5, max_iter=200).fit(Z)
        A2 = composition_per_unit(unit_cellids_list,
                                    soft_assign(Z, km2.cluster_centers_), cellid_idx)
        km3 = MiniBatchKMeans(n_clusters=K_fine, random_state=43, batch_size=4096,
                                n_init=5, max_iter=200).fit(Z)
        A3 = composition_per_unit(unit_cellids_list,
                                    soft_assign(Z, km3.cluster_centers_), cellid_idx)
        cmd_units = [(uid, g, Z[[cellid_idx[c] for c in unit_cellids[uid] if c in cellid_idx]])
                      for uid, g in zip(unit_ids, unit_groups)]
        coarse_label_map = dict(zip(all_cellids, ct))
        CMD = loo_cmd(cmd_units, unit_cellids, coarse_label_map,
                       max_dim_per_cluster=8, seed=42, loo=True, verbose=False)
        weights = derive_weights(K_c, K_med, K_fine, cmd_weight=0.60, n_blocks=4)
        emb_df = build_emb_from_blocks([A1, A2, A3, CMD], weights,
                                         unit_ids=unit_ids, unit_groups=unit_groups,
                                         unit_batches=None, pca_components=10,
                                         batch_method="harmony", seed=42, verbose=False)
        os.makedirs(f"{outdir}/sample_embedding", exist_ok=True)
        emb_df.to_csv(f"{outdir}/sample_embedding/sample_embedding.csv")
        return

    kw = dict(sample_col=sample_col, celltype_col=celltype_col,
                cluster_emb_key=cluster_emb_key, cmd_emb_key=cmd_emb_key,
                modality_col=modality_col, batch_col=None,
                save=True, verbose=False)
    if variant == "baseline":
        pass
    elif variant.startswith("cw_"):
        kw["cmd_weight"] = float(variant.split("_")[1])
    elif variant == "fixed_w":
        kw["block_weights"] = [3.0, 1.55, 1.0, 0.60]
    elif variant == "K_finer":
        kw["medium_K"] = 200; kw["fine_K"] = 500
    elif variant == "K_coarser":
        kw["medium_K"] = 60;  kw["fine_K"] = 150
    compute_sample_embedding(adata, outdir, **kw)


def evaluate(ds_name: str, ds_cfg: dict, emb_csv: str, method_label: str):
    if ds_cfg["evaluator"] == "encode":
        from benchmark_metrics_ENCODE import evaluate_multimodal_integration as ev
        bio_key = "tissue_preservation_score"
    elif ds_cfg["evaluator"] == "eye":
        from benchmark_eye import evaluate_multimodal_integration as ev
        bio_key = "cca_score"
    else:
        from benchmark_heart import evaluate_multimodal_integration as ev
        bio_key = "disease_state_preservation_score"
    r = ev(meta_csv=ds_cfg["meta"], embedding_csv=emb_csv,
             method_name=method_label, general_outdir=ds_cfg["outroot"],
             k_neighbors=ds_cfg["k"], n_permutations=200,
             create_visualizations=False)
    return {"paired_v2": r["paired_v2_score"],
            "bio":       r[bio_key],
            "ASW":       r["ASW_modality_overall"]}


def main():
    if len(sys.argv) != 3:
        print("usage: sd_param_sweep_one.py <dataset> <variant>", file=sys.stderr)
        sys.exit(2)
    ds_name, variant = sys.argv[1], sys.argv[2]
    if ds_name not in DATASETS:
        print(f"unknown dataset {ds_name}", file=sys.stderr); sys.exit(2)
    ds_cfg = DATASETS[ds_name]
    outdir = f"{OUT}/{ds_name}_{variant}"
    os.makedirs(outdir, exist_ok=True)
    method_label = f"sweep_{ds_name}_{variant}"
    print(f"[{ds_name}/{variant}] loading {ds_cfg['h5']}", flush=True)
    adata = load_minimal(ds_cfg["h5"])
    print(f"[{ds_name}/{variant}] adata loaded shape={adata.shape}", flush=True)
    make_se(adata, variant, outdir)
    emb_csv = f"{outdir}/sample_embedding/sample_embedding.csv"
    metrics = evaluate(ds_name, ds_cfg, emb_csv, method_label)
    metrics["dataset"] = ds_name
    metrics["variant"] = variant
    out_json = f"{outdir}/metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{ds_name}/{variant}] paired_v2={metrics['paired_v2']:.3f} "
          f"bio={metrics['bio']:.3f} ASW={metrics['ASW']:.3f}  → {out_json}",
          flush=True)


if __name__ == "__main__":
    main()
