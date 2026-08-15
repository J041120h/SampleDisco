#!/usr/bin/env python
"""Probe the remaining RNA / ATAC / unpaired-multiomics candidates.

Same report as probe_inputs.py, over every non-throwaway cell-level object in
the result root that is not already wired into build_similarity_panel.DATASETS.
"""
import sys
import traceback

import anndata as ad

R = "/dcs07/hongkai/data/harry/result"

CANDIDATES = {
    # --- ATAC ---
    "covid_ATAC": f"{R}/Benchmark_covid/ATAC/preprocess/adata_preprocessed.h5ad",
    "unpaired_paper_ATAC": f"{R}/multi_omics_unpaired_paper/atac/preprocess/adata_cell.h5ad",
    # --- RNA (single-omics arms of the multi-omics datasets) ---
    "ENCODE_rna": f"{R}/archived_benchmarks/Benchmark_ENCODE_rna/rna/preprocess/adata_cell.h5ad",
    "heart_rna": f"{R}/archived_benchmarks/Benchmark_heart_rna/rna/preprocess/adata_cell.h5ad",
    "heart_SD_rna": f"{R}/multi_omics_heart/SD/rna/preprocess/adata_cell.h5ad",
    "ENCODE_bench_cell": f"{R}/Benchmark_multiomics/adata_cell.h5ad",
    # --- unpaired multi-omics (RNA + ATAC, not paired) ---
    "unpaired_diemb": f"{R}/multi_omics_unpaired_diemb/multiomics/preprocess/adata_preprocessed.h5ad",
    "unpaired_paper_mo": f"{R}/multi_omics_unpaired_paper/multiomics/preprocess/atac_rna_integrated.h5ad",
    "unpaired_test_mo": f"{R}/multi_omics_unpaired_test/multiomics/preprocess/atac_rna_integrated.h5ad",
    # --- other cohorts ---
    "1M_scBloodNL": f"{R}/1M-scBloodNL/rna/preprocess/adata_cell.h5ad",
    "1M_scBloodNL_V3": f"{R}/1M-scBloodNL/V3/rna/preprocess/adata_cell.h5ad",
    "long_covid_analysis": f"{R}/long_covid/analysis/preprocess/adata_cell.h5ad",
    "health_aging_r3": f"{R}/health_aging_PBMC/round3_filename/preprocess/adata_preprocessed.h5ad",
    "heart_standalone_mo": f"{R}/multi_omics_heart/heart/multiomics/preprocess/atac_rna_integrated.h5ad",
}

INTEREST = [
    "sample", "Tube_id", "Donor_id", "cell_type", "Cluster_names", "celltype",
    "modality", "batch", "Batch", "sev.level", "severity", "condition",
    "disease_state", "tissue", "age", "Age", "Age_group", "sex", "Sex",
    "group", "study", "dataset", "stim", "time", "month",
]


def probe(name, path):
    print(f"\n{'=' * 78}\n### {name}\n{path}", flush=True)
    try:
        a = ad.read_h5ad(path, backed="r")
    except Exception as e:
        print(f"FAILED TO OPEN: {type(e).__name__}: {e}")
        return
    try:
        print(f"n_obs={a.n_obs}  n_vars={a.n_vars}")
        print("obsm:", {k: tuple(a.obsm[k].shape) for k in a.obsm.keys()})
        print("obs columns:", list(a.obs.columns))
        for c in INTEREST:
            if c not in a.obs.columns:
                continue
            s = a.obs[c]
            print(f"  - {c!r}: nunique={s.nunique(dropna=True)} "
                  f"e.g. {sorted(map(str, s.dropna().unique()))[:8]}")
    except Exception:
        traceback.print_exc()
    finally:
        try:
            a.file.close()
        except Exception:
            pass


def main():
    for k in (sys.argv[1:] or list(CANDIDATES)):
        probe(k, CANDIDATES[k])
    return 0


if __name__ == "__main__":
    sys.exit(main())
