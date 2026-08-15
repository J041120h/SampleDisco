#!/usr/bin/env python
"""Probe candidate cell-level h5ads for the fig2d similarity panel.

Reports n_obs, obsm keys (+dims), obs columns, and the cardinality of the
candidate sample / cell-type / group columns, so the per-dataset recipe can be
fixed before any block is built. Read-only; opens every file backed.
"""
import sys
import traceback

import anndata as ad

CANDIDATES = {
    "covid_400": "/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_400_sample/rna/preprocess/adata_cell.h5ad",
    "long_covid": "/dcs07/hongkai/data/harry/result/long_covid/rna/preprocess/adata_cell.h5ad",
    "health_aging": "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/preprocess/adata_preprocessed.h5ad",
    "heart": "/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/preprocess/atac_rna_integrated.h5ad",
    "retina": "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/preprocess/atac_rna_integrated.h5ad",
    "lutea": "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/preprocess/atac_rna_integrated.h5ad",
    "ENCODE": "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/preprocess/adata_sample.h5ad",
}

# columns worth reporting cardinality/values for
INTEREST = [
    "sample", "Tube_id", "Donor_id", "cell_type", "Cluster_names", "celltype",
    "modality", "batch", "Batch", "sev.level", "severity", "condition",
    "disease_state", "tissue", "age", "Age", "Age at enrollment", "sex", "group",
]


def probe(name, path):
    print(f"\n{'=' * 78}\n### {name}\n{path}", flush=True)
    try:
        a = ad.read_h5ad(path, backed="r")
    except Exception:
        print("FAILED TO OPEN:")
        traceback.print_exc()
        return
    print(f"n_obs={a.n_obs}  n_vars={a.n_vars}")
    print("obsm:", {k: tuple(a.obsm[k].shape) for k in a.obsm.keys()})
    print("uns keys:", list(a.uns.keys())[:20])
    print("obs columns:", list(a.obs.columns))
    for c in INTEREST:
        if c not in a.obs.columns:
            continue
        s = a.obs[c]
        nu = s.nunique(dropna=True)
        head = sorted(map(str, s.dropna().unique()))[:12]
        print(f"  - {c!r}: nunique={nu} dtype={s.dtype} e.g. {head}")
    a.file.close()


def main():
    keys = sys.argv[1:] or list(CANDIDATES)
    for k in keys:
        probe(k, CANDIDATES[k])
    return 0


if __name__ == "__main__":
    sys.exit(main())
