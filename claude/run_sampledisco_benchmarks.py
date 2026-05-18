"""Benchmark the new SampleDisco embedding against the existing competing-method
results. Adds (or updates) a single column named "SampleDisco" in each
dataset's existing summary CSV.

Decisions (confirmed with user):
  - One column "SampleDisco" per dataset (replaces SD_expression / SD_proportion).
  - COVID single-omics RNA: use AUTOTUNED embedding; derive pseudotime via the
    existing CCA code (sample_trajectory.CCA.CCA_Call).
  - Multi-omics (ENCODE / Lutea / Retina / Heart): use DEFAULT-α embedding.
  - Unpaired / Long COVID: out of scope for this run.

Outputs:
  - Per-method benchmark directories under each dataset's existing common
    output root.
  - Updated summary CSVs at the canonical paths the figure scripts already
    consume.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict

CODE_DIR = "/users/hjiang/GenoDistance/code"
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "Benchmark_covid"))
sys.path.insert(0, os.path.join(CODE_DIR, "Benchmark_multiomics"))

import numpy as np
import pandas as pd
import scanpy as sc

from sample_trajectory.CCA import CCA_Call
from sample_embedding.sample_embedding import build_sample_adata

# Lazy-imported below so the COVID and multi-omics scripts only load when needed:
#   from other_benchmark_wrapper import run_benchmarks
#   from benchmark_metrics_ENCODE import (evaluate_multimodal_integration as encode_eval, save_to_summary_csv as encode_save)
#   from benchmark_eye import (evaluate_multimodal_integration as eye_eval, save_to_summary_csv as eye_save)
#   from benchmark_heart import (evaluate_multimodal_integration as heart_eval, save_to_summary_csv as heart_save)


# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
COVID_BENCH_ROOT  = "/dcs07/hongkai/data/harry/result/Benchmark_covid"
COVID_OUT_ROOT    = f"{COVID_BENCH_ROOT}/ALL_BENCHMARK_OUTPUTS"
COVID_META        = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv"
COVID_SUMMARY     = f"{COVID_OUT_ROOT}/benchmark_summary_all_methods.csv"
COVID_SAMPLE_SIZES = [25, 50, 100, 200, 279, 400]   # directory naming

ENCODE_META       = "/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv"
ENCODE_OUTDIR     = "/dcs07/hongkai/data/harry/result/Benchmark_multiomics"
ENCODE_EMB        = "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/sampledisco_default/sample_embedding/sample_embedding.csv"
ENCODE_SUMMARY    = f"{ENCODE_OUTDIR}/summary.csv"

EYE_META          = "/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv"
LUTEA_OUTDIR      = "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea"
LUTEA_EMB         = "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/sampledisco_default/sample_embedding/sample_embedding.csv"
LUTEA_SUMMARY     = f"{LUTEA_OUTDIR}/Benchmark_result/summary.csv"

RETINA_OUTDIR     = "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina"
RETINA_EMB        = "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/sampledisco_default/sample_embedding/sample_embedding.csv"
RETINA_SUMMARY    = f"{RETINA_OUTDIR}/Benchmark_result/summary.csv"

HEART_META        = "/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv"
HEART_OUTDIR      = "/dcs07/hongkai/data/harry/result/multi_omics_heart"
HEART_EMB         = "/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/sampledisco_default/sample_embedding/sample_embedding.csv"
HEART_SUMMARY     = f"{HEART_OUTDIR}/summary.csv"


# --------------------------------------------------------------------------- #
# Logging helpers                                                             #
# --------------------------------------------------------------------------- #
def _hdr(msg: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(msg, flush=True)
    print("=" * 78, flush=True)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# COVID single-omics RNA                                                      #
# --------------------------------------------------------------------------- #
def _derive_covid_pseudotime(size: int) -> str:
    """Inject the autotuned embedding into adata.uns['X_DR_sample'] and run
    CCA_Call to produce pseudotime + visualization PDFs. Returns the path to
    the pseudotime CSV."""
    base       = f"{COVID_BENCH_ROOT}/covid_{size}_sample/rna"
    cell_h5    = f"{base}/preprocess/adata_cell.h5ad"
    tuned_dir  = f"{base}/sampledisco_tuned"
    tuned_csv  = f"{tuned_dir}/sample_embedding/sample_embedding.csv"
    pseudo_csv = f"{tuned_dir}/CCA/pseudotime_sample.csv"

    if os.path.exists(pseudo_csv):
        _log(f"covid_{size}_sample: pseudotime already exists → reuse ({pseudo_csv})")
        return pseudo_csv

    _log(f"covid_{size}_sample: loading {cell_h5}")
    adata = sc.read(cell_h5)

    _log(f"covid_{size}_sample: loading tuned embedding {tuned_csv}")
    emb_df = pd.read_csv(tuned_csv, index_col=0)
    adata.uns["X_DR_sample"] = emb_df

    # CCA_Call expects an adata whose obs is sample-level. Build it via
    # build_sample_adata; that aggregates per-sample metadata (incl. sev.level).
    _log(f"covid_{size}_sample: building sample-level adata for CCA")
    sample_adata = build_sample_adata(adata, sample_col="sample", modality_col=None)
    if "sev.level" not in sample_adata.obs.columns:
        raise KeyError(f"covid_{size}_sample: sev.level missing after sample-level aggregation")

    _log(f"covid_{size}_sample: running CCA_Call (trajectory_col='sev.level')")
    CCA_Call(
        adata=sample_adata,
        output_dir=tuned_dir,
        trajectory_col="sev.level",
        n_components=10,
        verbose=True,
    )
    if not os.path.exists(pseudo_csv):
        raise FileNotFoundError(f"CCA did not produce {pseudo_csv}")
    _log(f"covid_{size}_sample: wrote {pseudo_csv}")
    return pseudo_csv


def _run_covid_benchmark(size: int, pseudo_csv: str) -> None:
    from other_benchmark_wrapper import run_benchmarks
    base       = f"{COVID_BENCH_ROOT}/covid_{size}_sample/rna"
    tuned_csv  = f"{base}/sampledisco_tuned/sample_embedding/sample_embedding.csv"
    output_dir = f"{COVID_OUT_ROOT}/SampleDisco-{size}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    _log(f"covid_{size}_sample: running 7-test benchmark → {output_dir}")
    common_kwargs = dict(
        embedding_visualization={"dpi": 300, "figsize": (12, 5)},
        ari_clustering={
            "k_neighbors": 20,
            "n_clusters": None,
            "create_plots": True,
            "label_col": "sev.level",
        },
        batch_removal={"k": 15, "include_self": False},
        batch_mixing={"k": 20},
    )
    run_benchmarks(
        meta_csv_path=COVID_META,
        pseudotime_csv_path=pseudo_csv,
        embedding_csv_path=tuned_csv,
        summary_csv_path=COVID_SUMMARY,
        method_name="SampleDisco",
        output_base_dir=output_dir,
        **common_kwargs,
    )


def block_covid_rna() -> None:
    _hdr("COVID single-omics RNA — SampleDisco (autotuned)")
    Path(COVID_OUT_ROOT).mkdir(parents=True, exist_ok=True)
    for size in COVID_SAMPLE_SIZES:
        try:
            pseudo = _derive_covid_pseudotime(size)
            _run_covid_benchmark(size, pseudo)
            _log(f"covid_{size}_sample: DONE")
        except Exception:
            _log(f"covid_{size}_sample: FAIL")
            traceback.print_exc()


# --------------------------------------------------------------------------- #
# Multi-omics (default-α only)                                                #
# --------------------------------------------------------------------------- #
def block_mo_encode() -> None:
    _hdr("Multi-omics ENCODE — SampleDisco (default-α)")
    from benchmark_metrics_ENCODE import (
        evaluate_multimodal_integration as encode_eval,
        save_to_summary_csv as encode_save,
    )
    try:
        results = encode_eval(
            meta_csv=ENCODE_META,
            embedding_csv=ENCODE_EMB,
            method_name="SampleDisco",
            general_outdir=ENCODE_OUTDIR,
            k_neighbors=5,
            n_permutations=1000,
        )
        encode_save(results, ENCODE_SUMMARY)
        _log("ENCODE: DONE")
    except Exception:
        _log("ENCODE: FAIL")
        traceback.print_exc()


def block_mo_eye(name: str, emb: str, outdir: str, summary: str) -> None:
    _hdr(f"Multi-omics {name} — SampleDisco (default-α)")
    from benchmark_eye import (
        evaluate_multimodal_integration as eye_eval,
        save_to_summary_csv as eye_save,
    )
    try:
        results = eye_eval(
            meta_csv=EYE_META,
            embedding_csv=emb,
            method_name="SampleDisco",
            general_outdir=outdir,
            k_neighbors=3,
        )
        eye_save(results, summary)
        _log(f"{name}: DONE")
    except Exception:
        _log(f"{name}: FAIL")
        traceback.print_exc()


def block_mo_heart() -> None:
    _hdr("Multi-omics Heart — SampleDisco (default-α)")
    from benchmark_heart import (
        evaluate_multimodal_integration as heart_eval,
        save_to_summary_csv as heart_save,
    )
    try:
        results = heart_eval(
            meta_csv=HEART_META,
            embedding_csv=HEART_EMB,
            method_name="SampleDisco",
            general_outdir=HEART_OUTDIR,
            k_neighbors=15,
        )
        heart_save(results, HEART_SUMMARY)
        _log("Heart: DONE")
    except Exception:
        _log("Heart: FAIL")
        traceback.print_exc()


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main(argv) -> int:
    blocks = {
        "covid":  block_covid_rna,
        "encode": block_mo_encode,
        "lutea":  lambda: block_mo_eye("Lutea", LUTEA_EMB, LUTEA_OUTDIR, LUTEA_SUMMARY),
        "retina": lambda: block_mo_eye("Retina", RETINA_EMB, RETINA_OUTDIR, RETINA_SUMMARY),
        "heart":  block_mo_heart,
    }
    selected = argv[1:] if len(argv) > 1 else list(blocks.keys())
    bad = [b for b in selected if b not in blocks]
    if bad:
        print(f"Unknown blocks: {bad}; valid: {list(blocks.keys())}", file=sys.stderr)
        return 2
    t0 = time.time()
    for b in selected:
        try:
            blocks[b]()
        except Exception:
            _log(f"BLOCK {b} crashed:")
            traceback.print_exc()
    _log(f"All blocks done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
