"""Run TSCAN trajectory + visualization on the multi_omics_unpaired_paper
multiomics pseudobulk.

The pseudobulk h5ad already holds the V2 sample embeddings (composition +
CMD output by compute_sample_embedding) in adata.uns as DataFrames keyed
``X_DR_expression`` (431x30) and ``X_DR_proportion`` (431x11). We run the
prewritten ``TSCAN`` function from sample_trajectory.TSCAN on both keys
with the multiomics_unpaired config defaults (origin auto-picked, BIC
selects clusters, rank pseudotime, ``sev.level`` for visualization).
"""
from __future__ import annotations

import os
import sys
import time

CODE_DIR = "/users/hjiang/GenoDistance/code"
sys.path.insert(0, CODE_DIR)

import scanpy as sc

from sample_trajectory.TSCAN import TSCAN


PSEUDOBULK = (
    "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test"
    "/multiomics/pseudobulk/pseudobulk_sample.h5ad"
)
OUT_ROOT = (
    "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test"
    "/multiomics/trajectory"
)

EMBEDDING_KEYS = ("X_DR_expression", "X_DR_proportion")
GROUPING = ["sev.level"]


def main() -> int:
    if not os.path.exists(PSEUDOBULK):
        print(f"FATAL: pseudobulk missing: {PSEUDOBULK}", file=sys.stderr)
        return 2

    print(f"[tscan] loading {PSEUDOBULK}", flush=True)
    adata = sc.read(PSEUDOBULK)
    print(f"[tscan] shape={adata.shape}  uns keys={list(adata.uns.keys())}",
          flush=True)

    os.makedirs(OUT_ROOT, exist_ok=True)

    for key in EMBEDDING_KEYS:
        sub = key.replace("X_DR_", "")
        outdir = os.path.join(OUT_ROOT, sub)
        os.makedirs(outdir, exist_ok=True)

        print("\n" + "=" * 78, flush=True)
        print(f"[tscan] embedding={key} → {outdir}", flush=True)
        print("=" * 78, flush=True)

        t0 = time.time()
        TSCAN(
            AnnData_sample=adata,
            column=key,
            n_clusters=None,          # auto via BIC (config default)
            output_dir=outdir,
            grouping_columns=GROUPING,
            verbose=True,
            origin=None,              # auto pick endpoint
            pseudotime_mode="rank",
        )
        print(f"[tscan] {key}: {time.time() - t0:.2f}s", flush=True)

    print("\n[tscan] all done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
