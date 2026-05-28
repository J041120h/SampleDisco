"""Re-run the COVID single-omics RNA benchmark using SampleDisco's
DEFAULT-α (untuned) sample embedding instead of the autotuned one.

Decisions (confirmed with user):
  * Column name in the new summary CSV stays "SampleDisco" (so the plot
    treats it identically to the tuned run; only the underlying SE differs).
  * Per-method-size output directories use a "SampleDisco-default-{size}"
    name to avoid colliding with the existing tuned SampleDisco-{size}
    folders under ALL_BENCHMARK_OUTPUTS/.
  * Summary CSV goes to a NEW location next to the original so neither
    overwrites the other. Competitor (non-SampleDisco) columns are seeded
    from the existing tuned-run summary so we don't waste compute
    re-benchmarking embeddings that haven't changed.
  * Pseudotime is recomputed from the default SE via CCA_Call and saved
    under sampledisco_default/CCA/pseudotime_sample.csv.
"""
from __future__ import annotations
import os, sys, time, traceback
from pathlib import Path

CODE_DIR = "/users/hjiang/GenoDistance/code"
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "Benchmark_covid"))

import pandas as pd
import scanpy as sc

from sample_trajectory.CCA import CCA_Call
from sample_embedding.sample_embedding import build_sample_adata

# Paths — mirror run_sampledisco_benchmarks.py but flip tuned→default + new summary
COVID_BENCH_ROOT   = "/dcs07/hongkai/data/harry/result/Benchmark_covid"
COVID_OUT_ROOT     = f"{COVID_BENCH_ROOT}/ALL_BENCHMARK_OUTPUTS"
COVID_META         = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv"
COVID_SUMMARY_OLD  = f"{COVID_OUT_ROOT}/benchmark_summary_all_methods.csv"
COVID_SUMMARY_NEW  = f"{COVID_OUT_ROOT}/benchmark_summary_all_methods_default.csv"
COVID_SAMPLE_SIZES = [25, 50, 100, 200, 279, 400]


def _log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _seed_summary_from_tuned():
    """Copy the tuned-run summary to the new path, then drop all
    SampleDisco-* columns so run_benchmarks can populate them fresh."""
    Path(COVID_OUT_ROOT).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(COVID_SUMMARY_OLD):
        raise FileNotFoundError(COVID_SUMMARY_OLD)
    df = pd.read_csv(COVID_SUMMARY_OLD, index_col=0)
    sd_cols = [c for c in df.columns if c.startswith("SampleDisco-")]
    df = df.drop(columns=sd_cols)
    df.to_csv(COVID_SUMMARY_NEW, index_label="Metric")
    _log(f"seeded {COVID_SUMMARY_NEW}  (dropped {len(sd_cols)} tuned cols, "
         f"kept {df.shape[1]} competitor cols)")


def _derive_default_pseudotime(size: int) -> str:
    """Run CCA on the DEFAULT SE and save pseudotime_sample.csv next to it."""
    base       = f"{COVID_BENCH_ROOT}/covid_{size}_sample/rna"
    cell_h5    = f"{base}/preprocess/adata_cell.h5ad"
    default_dir = f"{base}/sampledisco_default"
    default_csv = f"{default_dir}/sample_embedding/sample_embedding.csv"
    pseudo_csv  = f"{default_dir}/CCA/pseudotime_sample.csv"

    if not os.path.exists(default_csv):
        raise FileNotFoundError(default_csv)
    if os.path.exists(pseudo_csv):
        _log(f"covid_{size}_sample: pseudotime already exists → reuse ({pseudo_csv})")
        return pseudo_csv

    _log(f"covid_{size}_sample: loading {cell_h5}")
    adata = sc.read(cell_h5)
    _log(f"covid_{size}_sample: loading DEFAULT embedding {default_csv}")
    adata.uns["X_DR_sample"] = pd.read_csv(default_csv, index_col=0)

    sample_adata = build_sample_adata(adata, sample_col="sample", modality_col=None)
    if "sev.level" not in sample_adata.obs.columns:
        raise KeyError(f"covid_{size}_sample: sev.level missing after sample-level aggregation")

    _log(f"covid_{size}_sample: running CCA_Call (trajectory_col='sev.level')")
    CCA_Call(adata=sample_adata, output_dir=default_dir,
             trajectory_col="sev.level", n_components=10, verbose=True)
    if not os.path.exists(pseudo_csv):
        raise FileNotFoundError(f"CCA did not produce {pseudo_csv}")
    return pseudo_csv


def _run_covid_default_benchmark(size: int, pseudo_csv: str) -> None:
    from other_benchmark_wrapper import run_benchmarks
    base        = f"{COVID_BENCH_ROOT}/covid_{size}_sample/rna"
    default_csv = f"{base}/sampledisco_default/sample_embedding/sample_embedding.csv"
    # Per-method-size dir: "default" suffix to avoid colliding with the
    # tuned SampleDisco-{size} folders under ALL_BENCHMARK_OUTPUTS/.
    output_dir  = f"{COVID_OUT_ROOT}/SampleDisco-default-{size}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    _log(f"covid_{size}_sample: 7-test benchmark → {output_dir}")
    common_kwargs = dict(
        embedding_visualization={"dpi": 300, "figsize": (12, 5)},
        ari_clustering={"k_neighbors": 20, "n_clusters": None,
                         "create_plots": True, "label_col": "sev.level"},
        batch_removal={"k": 15, "include_self": False},
        batch_mixing={"k": 20},
    )
    run_benchmarks(
        meta_csv_path=COVID_META,
        pseudotime_csv_path=pseudo_csv,
        embedding_csv_path=default_csv,
        summary_csv_path=COVID_SUMMARY_NEW,    # NEW summary
        method_name="SampleDisco",              # column stays "SampleDisco-{n}"
        output_base_dir=output_dir,
        **common_kwargs,
    )


def main():
    print("=" * 78); print("COVID single-omics RNA — SampleDisco (DEFAULT-α)"); print("=" * 78)
    _seed_summary_from_tuned()
    for size in COVID_SAMPLE_SIZES:
        try:
            pseudo = _derive_default_pseudotime(size)
            _run_covid_default_benchmark(size, pseudo)
            _log(f"covid_{size}_sample: DONE")
        except Exception:
            _log(f"covid_{size}_sample: FAIL")
            traceback.print_exc()
    _log(f"ALL DONE. New summary at: {COVID_SUMMARY_NEW}")


if __name__ == "__main__":
    main()
