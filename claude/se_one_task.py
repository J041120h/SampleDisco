"""Run ONLY the default-α sample-embedding step for one dataset.

Designed to be wrapped by Benchmark_covid/monitor_wrapper.py so the resource
usage of `compute_sample_embedding` is measured in isolation. No preprocess,
no autotune, no downstream analysis.

Usage:
    python -u se_one_task.py --task <key> --outdir <dir>

Tasks:
    covid_rna_25  covid_rna_50  covid_rna_100
    covid_rna_200 covid_rna_279 covid_rna_400
    covid_atac
    mo_encode  mo_lutea  mo_retina  mo_heart
    unpaired
"""
from __future__ import annotations

import argparse
import os
import sys
import time

CODE_DIR = "/users/hjiang/GenoDistance/code"
sys.path.insert(0, CODE_DIR)

import scanpy as sc

from sample_embedding import compute_sample_embedding


# Per-task config.  h5 path, embedding keys, modality + batch settings.
TASKS = {
    # ------------- COVID RNA (single-omics, 6 sample sizes) -------------
    "covid_rna_25":  dict(
        h5="/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_25_sample/rna/preprocess/adata_cell.h5ad",
        cluster_key="Z_clust", cmd_key="Z_cmd",
        modality_col=None, batch_col="batch"),
    "covid_rna_50":  dict(
        h5="/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_50_sample/rna/preprocess/adata_cell.h5ad",
        cluster_key="Z_clust", cmd_key="Z_cmd",
        modality_col=None, batch_col="batch"),
    "covid_rna_100": dict(
        h5="/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_100_sample/rna/preprocess/adata_cell.h5ad",
        cluster_key="Z_clust", cmd_key="Z_cmd",
        modality_col=None, batch_col="batch"),
    "covid_rna_200": dict(
        h5="/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_200_sample/rna/preprocess/adata_cell.h5ad",
        cluster_key="Z_clust", cmd_key="Z_cmd",
        modality_col=None, batch_col="batch"),
    "covid_rna_279": dict(
        h5="/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_279_sample/rna/preprocess/adata_cell.h5ad",
        cluster_key="Z_clust", cmd_key="Z_cmd",
        modality_col=None, batch_col="batch"),
    "covid_rna_400": dict(
        h5="/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_400_sample/rna/preprocess/adata_cell.h5ad",
        cluster_key="Z_clust", cmd_key="Z_cmd",
        modality_col=None, batch_col="batch"),

    # ------------- COVID ATAC (single-omics) -------------
    "covid_atac": dict(
        h5="/dcs07/hongkai/data/harry/result/Benchmark_covid/ATAC/preprocess/adata_preprocessed.h5ad",
        cluster_key="Z_clust", cmd_key=None,  # auto: nosamp falls back to Z_clust
        modality_col=None, batch_col=None),

    # ------------- Multi-omics (default-α) -------------
    # ENCODE: integrated h5ad has X stored as dict (broken). Use sister
    # adata_sample.h5ad which is also cell-level + has X_glue.
    "mo_encode": dict(
        h5="/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/preprocess/adata_sample.h5ad",
        cluster_key="X_glue", cmd_key="X_glue",
        modality_col="modality", batch_col=None),
    "mo_lutea": dict(
        h5="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/preprocess/atac_rna_integrated.h5ad",
        cluster_key="X_glue", cmd_key="X_glue",
        modality_col="modality", batch_col=None),
    "mo_retina": dict(
        h5="/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/preprocess/atac_rna_integrated.h5ad",
        cluster_key="X_glue", cmd_key="X_glue",
        modality_col="modality", batch_col=None),
    "mo_heart": dict(
        h5="/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/preprocess/atac_rna_integrated.h5ad",
        cluster_key="X_glue", cmd_key="X_glue",
        modality_col="modality", batch_col=None),

    # ------------- Unpaired (default-α) -------------
    "unpaired": dict(
        h5="/dcs07/hongkai/data/harry/result/multi_omics_unpaired_paper/multiomics/preprocess/atac_rna_integrated.h5ad",
        cluster_key="X_glue", cmd_key="X_glue",
        modality_col="modality", batch_col="batch"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS.keys()))
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = TASKS[args.task]
    h5 = cfg["h5"]
    if not os.path.exists(h5):
        print(f"FATAL: input h5ad missing: {h5}", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)

    t_load_start = time.time()
    print(f"[se_one_task] task={args.task}", flush=True)
    print(f"[se_one_task] loading {h5}", flush=True)
    adata = sc.read(h5)
    t_load_done = time.time()
    print(f"[se_one_task] loaded {adata.shape} in {t_load_done - t_load_start:.2f}s",
          flush=True)
    print(f"[se_one_task] obsm={list(adata.obsm.keys())}", flush=True)

    if "cell_type" not in adata.obs.columns:
        print(f"FATAL: cell_type missing from adata.obs", file=sys.stderr)
        return 3

    t_se_start = time.time()
    compute_sample_embedding(
        adata, args.outdir,
        sample_col="sample",
        celltype_col="cell_type",
        cluster_emb_key=cfg["cluster_key"],
        cmd_emb_key=cfg["cmd_key"],
        modality_col=cfg["modality_col"],
        batch_col=cfg["batch_col"],
        save=True, verbose=True,
    )
    t_se_done = time.time()
    print(f"[se_one_task] compute_sample_embedding: {t_se_done - t_se_start:.2f}s",
          flush=True)
    print(f"[se_one_task] total: {t_se_done - t_load_start:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
