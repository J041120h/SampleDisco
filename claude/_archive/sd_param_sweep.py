"""SampleDisco parameter sweep on the 4 multi-omics datasets.

For each variant, generate sample embedding then evaluate via the existing
multi-omics benchmark scripts to get the canonical 3 metrics:
  - paired_v2_score (smaller = better)
  - biology score (tissue/disease preservation OR cca_score; larger = better)
  - ASW_modality (larger = better)

Variants (single-factor changes from baseline = current default):
  baseline   : cmd_weight=0.60, K_med=120, K_fine=300, auto block_weights, A1=one-hot
  cw_0.30    : cmd_weight=0.30
  cw_1.00    : cmd_weight=1.00
  cw_2.00    : cmd_weight=2.00
  fixed_w    : block_weights=[3.0, 1.55, 1.0, 0.60]
  K_finer    : K_med=200, K_fine=500
  K_coarser  : K_med=60,  K_fine=150
  A1_soft    : A1 uses Gaussian-soft k-means at K=K_c (instead of one-hot)
"""
from __future__ import annotations
import os, sys, time, traceback
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
os.makedirs(OUT, exist_ok=True)

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

VARIANTS = ["baseline", "cw_0.30", "cw_1.00", "cw_2.00", "fixed_w",
            "K_finer", "K_coarser", "A1_soft"]


def make_se(adata, variant: str, outdir: str, verbose=False):
    """Generate one variant's sample embedding."""
    cluster_emb_key, cmd_emb_key = "X_glue", "X_glue"
    sample_col, celltype_col, modality_col = "sample", "cell_type", "modality"

    if variant == "A1_soft":
        # Custom path: replace A1 (one-hot per cell type) with soft k-means at K=K_c.
        units, unit_cellids, unit_ids, unit_groups, unit_batches, all_cellids, Z = \
            assemble_units(adata, sample_col, cluster_emb_key,
                           modality_col=modality_col, batch_col=None)
        cellid_idx = {c: i for i, c in enumerate(all_cellids)}
        ct = adata.obs[celltype_col].astype(str).values
        K_c = len(set(ct))
        K_med = min(120, max(2, Z.shape[0] // 200))
        K_fine = min(300, max(2, Z.shape[0] // 100))

        # A1 — Gaussian-soft k-means at K_c (replaces one-hot composition)
        km1 = MiniBatchKMeans(n_clusters=K_c, random_state=42, batch_size=4096,
                                n_init=5, max_iter=200).fit(Z)
        unit_cellids_list = [unit_cellids[u] for u in unit_ids]
        A1 = composition_per_unit(unit_cellids_list,
                                    soft_assign(Z, km1.cluster_centers_), cellid_idx)
        # A2, A3 normal
        km2 = MiniBatchKMeans(n_clusters=K_med, random_state=42, batch_size=4096,
                                n_init=5, max_iter=200).fit(Z)
        A2 = composition_per_unit(unit_cellids_list,
                                    soft_assign(Z, km2.cluster_centers_), cellid_idx)
        km3 = MiniBatchKMeans(n_clusters=K_fine, random_state=43, batch_size=4096,
                                n_init=5, max_iter=200).fit(Z)
        A3 = composition_per_unit(unit_cellids_list,
                                    soft_assign(Z, km3.cluster_centers_), cellid_idx)
        # CMD — same as default
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
                save=True, verbose=verbose)
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


def evaluate(name: str, ds_cfg: dict, emb_csv: str, method_label: str) -> dict:
    """Run the right benchmark evaluator on the embedding, return the 3-metric dict."""
    if ds_cfg["evaluator"] == "encode":
        from benchmark_metrics_ENCODE import evaluate_multimodal_integration as ev
        r = ev(meta_csv=ds_cfg["meta"], embedding_csv=emb_csv,
                 method_name=method_label, general_outdir=ds_cfg["outroot"],
                 k_neighbors=ds_cfg["k"], n_permutations=200,
                 create_visualizations=False)
        return {"paired_v2": r["paired_v2_score"],
                "bio":       r["tissue_preservation_score"],
                "ASW":       r["ASW_modality_overall"]}
    elif ds_cfg["evaluator"] == "eye":
        from benchmark_eye import evaluate_multimodal_integration as ev
        r = ev(meta_csv=ds_cfg["meta"], embedding_csv=emb_csv,
                 method_name=method_label, general_outdir=ds_cfg["outroot"],
                 k_neighbors=ds_cfg["k"], n_permutations=200,
                 create_visualizations=False)
        return {"paired_v2": r["paired_v2_score"],
                "bio":       r["cca_score"],
                "ASW":       r["ASW_modality_overall"]}
    else:
        from benchmark_heart import evaluate_multimodal_integration as ev
        r = ev(meta_csv=ds_cfg["meta"], embedding_csv=emb_csv,
                 method_name=method_label, general_outdir=ds_cfg["outroot"],
                 k_neighbors=ds_cfg["k"], n_permutations=200,
                 create_visualizations=False)
        return {"paired_v2": r["paired_v2_score"],
                "bio":       r["disease_state_preservation_score"],
                "ASW":       r["ASW_modality_overall"]}


def main():
    rows = []
    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'='*78}\n  {ds_name}\n{'='*78}")
        adata = sc.read(ds_cfg["h5"])
        for variant in VARIANTS:
            t0 = time.time()
            outdir = f"{OUT}/{ds_name}_{variant}"
            os.makedirs(outdir, exist_ok=True)
            method_label = f"sweep_{ds_name}_{variant}"
            try:
                make_se(adata, variant, outdir)
                emb_csv = f"{outdir}/sample_embedding/sample_embedding.csv"
                metrics = evaluate(ds_name, ds_cfg, emb_csv, method_label)
                metrics.update({"dataset": ds_name, "variant": variant,
                                "wall_s": round(time.time() - t0, 1)})
                rows.append(metrics)
                print(f"  [{ds_name}/{variant:<11s}] paired_v2={metrics['paired_v2']:.3f} "
                      f"bio={metrics['bio']:.3f} ASW={metrics['ASW']:.3f}  "
                      f"({metrics['wall_s']}s)")
            except Exception:
                print(f"  [{ds_name}/{variant}] FAIL")
                traceback.print_exc()
                rows.append({"dataset": ds_name, "variant": variant,
                              "paired_v2": np.nan, "bio": np.nan, "ASW": np.nan,
                              "wall_s": round(time.time() - t0, 1)})

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/sweep_metrics.csv", index=False)
    print(f"\nWrote {OUT}/sweep_metrics.csv")
    print(df.to_string())


if __name__ == "__main__":
    main()
