"""For each MO weight-fix strategy, run the existing multi-omics benchmark
on its sample_embedding.csv and append SampleDisco_<strategy> column to the
existing summary CSVs. Then compute the SampleDisco-vs-QOT focused ranking.

Strategies (already produced by /tmp/fix_mo_weights.py):
  - fixed:  block_weights = [3.0, 1.55, 1.0, 0.60]   (WIRE-style)
  - no_A1:  block_weights = [1e-6, 1.55, 1.0, 0.60]  (drop coarse composition)
  - capped: auto-derived but A1 capped at 3.0

Existing reference:
  - default: original auto-derived weights (already in summary CSVs as 'SampleDisco')
"""
from __future__ import annotations
import os, sys, time, traceback
from pathlib import Path

CODE_DIR = "/users/hjiang/GenoDistance/code"
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "Benchmark_multiomics"))

import numpy as np, pandas as pd

FIX_BASE = "/dcs07/hongkai/data/harry/result/CLAUDE_MO_FIX"
STRATEGIES = ["fixed", "no_A1", "capped"]

DATASETS = [
    ("ENCODE", {
        "meta_csv": "/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        "outdir":   "/dcs07/hongkai/data/harry/result/Benchmark_multiomics",
        "summary":  "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/summary.csv",
        "k":        5,
        "evaluator":"encode",
    }),
    ("Lutea", {
        "meta_csv": "/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        "outdir":   "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea",
        "summary":  "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/Benchmark_result/summary.csv",
        "k":        3,
        "evaluator":"eye",
    }),
    ("Retina", {
        "meta_csv": "/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
        "outdir":   "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina",
        "summary":  "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/Benchmark_result/summary.csv",
        "k":        3,
        "evaluator":"eye",
    }),
    ("Heart", {
        "meta_csv": "/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
        "outdir":   "/dcs07/hongkai/data/harry/result/multi_omics_heart",
        "summary":  "/dcs07/hongkai/data/harry/result/multi_omics_heart/summary.csv",
        "k":        15,
        "evaluator":"heart",
    }),
]


def _emb_path(name: str, strat: str) -> str:
    return f"{FIX_BASE}/{name}_{strat}/sample_embedding/sample_embedding.csv"


def run_one(name: str, ds_cfg: dict, strat: str) -> None:
    method_label = f"SampleDisco_{strat}"
    emb = _emb_path(name, strat)
    if not os.path.exists(emb):
        print(f"  [{name}/{strat}] MISSING {emb}")
        return
    print(f"\n=== {name} / {strat} ===")
    print(f"  embedding: {emb}")
    print(f"  -> {method_label} column in {ds_cfg['summary']}")
    if ds_cfg["evaluator"] == "encode":
        from benchmark_metircs_ENCODE import (
            evaluate_multimodal_integration as ev,
            save_to_summary_csv as sv,
        )
        results = ev(meta_csv=ds_cfg["meta_csv"], embedding_csv=emb,
                       method_name=method_label,
                       general_outdir=ds_cfg["outdir"],
                       k_neighbors=ds_cfg["k"], n_permutations=1000)
        sv(results, ds_cfg["summary"])
    elif ds_cfg["evaluator"] == "eye":
        from benchmark_eye import (
            evaluate_multimodal_integration as ev,
            save_to_summary_csv as sv,
        )
        results = ev(meta_csv=ds_cfg["meta_csv"], embedding_csv=emb,
                       method_name=method_label,
                       general_outdir=ds_cfg["outdir"],
                       k_neighbors=ds_cfg["k"])
        sv(results, ds_cfg["summary"])
    elif ds_cfg["evaluator"] == "heart":
        from benchmark_heart import (
            evaluate_multimodal_integration as ev,
            save_to_summary_csv as sv,
        )
        results = ev(meta_csv=ds_cfg["meta_csv"], embedding_csv=emb,
                       method_name=method_label,
                       general_outdir=ds_cfg["outdir"],
                       k_neighbors=ds_cfg["k"])
        sv(results, ds_cfg["summary"])


def main():
    for name, cfg in DATASETS:
        for strat in STRATEGIES:
            try:
                run_one(name, cfg, strat)
            except Exception:
                print(f"  [{name}/{strat}] FAIL")
                traceback.print_exc()
    print("\nALL FIX-VARIANT BENCHMARKS DONE")


if __name__ == "__main__":
    main()
