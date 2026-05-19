"""TSCAN + KMeans on the V2 single-embedding outputs of
multi_omics_unpaired_test.

V2 SampleDisc produces a single combined sample embedding per variant
(``sampledisco_*/sample_embedding/sample_embedding.csv``), replacing the
older split expression/proportion pair. For each surviving variant we:

  1. Load the per-sample embedding CSV.
  2. Build a minimal AnnData using the pseudobulk's obs metadata (so the
     ``sev.level`` grouping label is available for visualization).
  3. Stash the embedding into ``adata.uns['X_DR_sample']`` (TSCAN reads
     from .uns) and ``adata.obsm['X_DR_sample']`` (cluster.cluster reads
     from .obsm).
  4. Run TSCAN (BIC-selected k, rank pseudotime, random endpoint origin,
     ``sev.level`` for grouping plot).
  5. Run KMeans with k=4 (config_unpaired default).

Outputs per variant:
  {variant}/trajectory/TSCAN/  — clusters_by_cluster_X_DR_sample.png,
                                 clusters_by_grouping_X_DR_sample.png,
                                 X_DR_sample_pseudotime.csv
  {variant}/sample_cluster/    — kmeans_clusters_sample.csv,
                                 kmeans_sample_embedding.png
"""
from __future__ import annotations

import glob
import os
import sys
import time

CODE_DIR = "/users/hjiang/GenoDistance/code"
sys.path.insert(0, CODE_DIR)

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from sample_trajectory.TSCAN import TSCAN
from sample_clustering.cluster import cluster as kmeans_cluster


ROOT = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics"
PSEUDOBULK = f"{ROOT}/pseudobulk/pseudobulk_sample.h5ad"
GROUPING = ["sev.level"]
EMB_KEY = "X_DR_sample"
KMEANS_K = 4


def _build_variant_adata(emb_csv: str, pseudobulk: ad.AnnData) -> ad.AnnData:
    """Load a SampleDisc embedding CSV and pair it with pseudobulk metadata.

    Returns a tiny AnnData (samples x 1 dummy var) carrying the embedding
    in both ``.uns[EMB_KEY]`` (DataFrame) and ``.obsm[EMB_KEY]`` (ndarray).
    """
    df = pd.read_csv(emb_csv)
    if "sample" not in df.columns:
        raise ValueError(f"missing 'sample' column in {emb_csv}")
    df = df.set_index("sample")

    common = pseudobulk.obs.index.intersection(df.index)
    if len(common) == 0:
        raise ValueError(
            f"no sample overlap between pseudobulk and {emb_csv}")
    df = df.loc[common]
    obs = pseudobulk.obs.loc[common].copy()

    X = np.zeros((len(common), 1), dtype=np.float32)
    adata = ad.AnnData(X=X, obs=obs)
    adata.uns[EMB_KEY] = df
    adata.obsm[EMB_KEY] = df.values.astype(np.float32)
    return adata


def _process_variant(variant_dir: str, pseudobulk: ad.AnnData) -> None:
    emb_csv = os.path.join(variant_dir, "sample_embedding", "sample_embedding.csv")
    if not os.path.exists(emb_csv):
        print(f"[skip] no sample_embedding.csv in {variant_dir}", flush=True)
        return

    name = os.path.basename(variant_dir.rstrip("/"))
    print("\n" + "=" * 78, flush=True)
    print(f"[run] variant={name}", flush=True)
    print(f"      emb_csv={emb_csv}", flush=True)
    print("=" * 78, flush=True)

    adata = _build_variant_adata(emb_csv, pseudobulk)
    print(f"      samples={adata.n_obs}  dims={adata.uns[EMB_KEY].shape[1]}",
          flush=True)

    # ----- TSCAN -----
    traj_dir = os.path.join(variant_dir, "trajectory")
    os.makedirs(traj_dir, exist_ok=True)
    t0 = time.time()
    TSCAN(
        AnnData_sample=adata,
        column=EMB_KEY,
        n_clusters=None,
        output_dir=traj_dir,
        grouping_columns=GROUPING,
        verbose=True,
        origin=None,
        pseudotime_mode="rank",
    )
    print(f"      tscan: {time.time() - t0:.2f}s", flush=True)

    # ----- KMeans (k=4) -----
    t0 = time.time()
    kmeans_cluster(
        pseudobulk_adata=adata,
        output_dir=variant_dir,
        number_of_clusters=KMEANS_K,
        random_state=0,
    )
    print(f"      kmeans: {time.time() - t0:.2f}s", flush=True)


def main() -> int:
    if not os.path.exists(PSEUDOBULK):
        print(f"FATAL: pseudobulk missing: {PSEUDOBULK}", file=sys.stderr)
        return 2

    print(f"[load] {PSEUDOBULK}", flush=True)
    pseudobulk = sc.read(PSEUDOBULK)
    print(f"[load] pseudobulk shape={pseudobulk.shape}", flush=True)

    variants = sorted(glob.glob(os.path.join(ROOT, "sampledisco_*")))
    variants = [v for v in variants if os.path.isdir(v)]
    print(f"[scan] found {len(variants)} sampledisco variant(s): "
          f"{[os.path.basename(v) for v in variants]}", flush=True)

    for v in variants:
        _process_variant(v, pseudobulk)

    print("\n[done] all variants processed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
