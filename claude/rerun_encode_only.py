"""Re-run ENCODE multi-omics sample embedding (default-α + association).

ENCODE's atac_rna_integrated.h5ad has X stored as a dict (unreadable),
so we use the sister file adata_sample.h5ad which is also cell-level
(835 934 cells × 17 504 features) and carries X_glue, sample, modality,
and cell_type — everything compute_sample_embedding needs.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

CODE_DIR = "/users/hjiang/GenoDistance/code"
sys.path.insert(0, CODE_DIR)

import scanpy as sc

from rerun_sample_embedding import _run_assoc_for_pseudo, _hdr, _log
from sample_embedding import compute_sample_embedding


ENCODE_H5 = "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/preprocess/adata_sample.h5ad"
ENCODE_OUT = "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics"


def main() -> int:
    _hdr("ENCODE — fix-up run (default-α + association)")
    if not os.path.exists(ENCODE_H5):
        _log(f"MISSING {ENCODE_H5}")
        return 1
    _log(f"Loading {ENCODE_H5}")
    adata = sc.read(ENCODE_H5)
    _log(f"adata: shape={adata.shape}, obsm={list(adata.obsm.keys())}, "
         f"n_samples={adata.obs['sample'].nunique()}")

    if "cell_type" not in adata.obs.columns:
        _log("ERROR: cell_type missing")
        return 2
    if "X_glue" not in adata.obsm:
        _log("ERROR: X_glue missing")
        return 3

    out_default = f"{ENCODE_OUT}/sampledisco_default"
    os.makedirs(out_default, exist_ok=True)
    _log(f"Default-α sample embedding → {out_default}")
    try:
        compute_sample_embedding(
            adata, out_default,
            sample_col="sample",
            celltype_col="cell_type",
            cluster_emb_key="X_glue",
            cmd_emb_key="X_glue",
            modality_col="modality",
            batch_col=None,
            save=True, verbose=True,
        )
    except Exception:
        _log("FAIL: sample embedding")
        traceback.print_exc()
        return 4

    try:
        _run_assoc_for_pseudo(adata, out_default, modality_col="modality")
    except Exception:
        _log("FAIL: association analysis (sample embedding still ok)")
        traceback.print_exc()
        return 5

    _log("ENCODE fix-up done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
