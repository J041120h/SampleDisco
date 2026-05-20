"""
Re-encode pbmc_benchmark_ready.h5ad to be ~3.6x smaller, lossless.

Source: /dcs07/hongkai/data/harry/result/health_aging_PBMC/benchmark_ready/pbmc_benchmark_ready.h5ad
  X/data    : float64 -> float32  (ecosystem default; scanpy/scvi-tools/cellxgene-census,
                                   HCA, Tabula Sapiens all use float32; float32 exactly
                                   represents integers up to 2^24, our max is 4611)
  X/indices : int64   -> int32    (true max=36,600 < 2^31)
  X/indptr  : int64   (kept; nnz=2.4e9 > 2^31)
  /raw      : dropped (byte-identical duplicate of /X)

Output: pbmc_benchmark_ready_slim.h5ad   (expect ~20 GB)
Memory peak: ~25 GB (compute node has 745 GB, fine).
"""

import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp

DATA = Path("/dcs07/hongkai/data/harry/result/health_aging_PBMC")
SRC = DATA / "benchmark_ready" / "pbmc_benchmark_ready.h5ad"
DST = DATA / "benchmark_ready" / "pbmc_benchmark_ready_slim.h5ad"


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def main():
    log(f"src  = {SRC}  ({SRC.stat().st_size / 1e9:.2f} GB)")
    log(f"dst  = {DST}")

    # ------------------------------------------------------------------
    # 1. read obs/var (small) with anndata so we keep all encodings/categoricals
    # ------------------------------------------------------------------
    log("reading obs/var via anndata (backed) ...")
    a_src = ad.read_h5ad(SRC, backed="r")
    obs = a_src.obs.copy()
    var = a_src.var.copy()
    uns = dict(a_src.uns) if hasattr(a_src, "uns") else {}
    a_src.file.close()
    log(f"  obs: shape={obs.shape}  var: shape={var.shape}  uns keys: {list(uns.keys())}")

    # ------------------------------------------------------------------
    # 2. stream-load X with dtype conversion via h5py
    # ------------------------------------------------------------------
    log("stream-loading X with type conversion ...")
    t0 = time.time()
    with h5py.File(SRC, "r") as h:
        nnz = h["X/data"].shape[0]
        n_obs = h["X/indptr"].shape[0] - 1
        n_var = h["var/_index"].shape[0]
        log(f"  n_obs={n_obs:,}  n_var={n_var:,}  nnz={nnz:,}")

        data = np.empty(nnz, dtype=np.float32)
        indices = np.empty(nnz, dtype=np.int32)
        CHUNK = 200_000_000
        for s in range(0, nnz, CHUNK):
            e = min(s + CHUNK, nnz)
            data[s:e] = h["X/data"][s:e].astype(np.float32)
            indices[s:e] = h["X/indices"][s:e].astype(np.int32)
            log(f"    {e / nnz * 100:5.1f}%  t={time.time() - t0:.1f}s")
        indptr = h["X/indptr"][:].astype(np.int64)

    log(f"  X arrays in memory:  "
        f"data {data.nbytes / 1e9:.2f} GB ({data.dtype}), "
        f"indices {indices.nbytes / 1e9:.2f} GB ({indices.dtype}), "
        f"indptr {indptr.nbytes / 1e9:.3f} GB ({indptr.dtype})")

    log("building scipy CSR ...")
    X = sp.csr_matrix((data, indices, indptr), shape=(n_obs, n_var), copy=False)
    # enforce dtypes (scipy sometimes promotes indices to int64)
    if X.indices.dtype != np.int32:
        log(f"  forcing indices back to int32 (was {X.indices.dtype})")
        X.indices = X.indices.astype(np.int32, copy=False)
    log(f"  X: {X.shape}  nnz={X.nnz:,}  "
        f"data={X.data.dtype} indices={X.indices.dtype} indptr={X.indptr.dtype}")

    # ------------------------------------------------------------------
    # 3. build new AnnData and write
    # ------------------------------------------------------------------
    log("constructing AnnData ...")
    a_new = ad.AnnData(X=X, obs=obs, var=var)
    a_new.uns["benchmark_ready_provenance"] = (
        "X re-encoded from pbmc_benchmark_ready.h5ad: "
        "data float64->float32 (scanpy/scvi-tools standard; integer counts max=4611 "
        "exact in float32), indices int64->int32 (n_var=36,601); "
        "raw/X duplicate dropped. obs unchanged. Built by slim_benchmark_ready_h5ad.py"
    )
    # copy other uns entries if any (e.g., Cluster_names_colors)
    for k, v in uns.items():
        if k != "benchmark_ready_provenance":
            a_new.uns[k] = v
    log(f"  AnnData: {a_new}")

    if DST.exists():
        log(f"removing existing {DST}")
        DST.unlink()

    log("writing slim h5ad ...")
    t0 = time.time()
    a_new.write_h5ad(DST, compression=None)
    log(f"  write done in {time.time() - t0:.1f}s  -> {DST.stat().st_size / 1e9:.2f} GB")

    # ------------------------------------------------------------------
    # 4. verify
    # ------------------------------------------------------------------
    log("verifying slim file ...")
    a_check = ad.read_h5ad(DST, backed="r")
    print(f"  shape = {a_check.shape}")
    print(f"  obs cols ({len(a_check.obs.columns)}): {list(a_check.obs.columns)}")
    print(f"  uns keys: {list(a_check.uns.keys())}")
    # spot-check X dtypes via h5py
    with h5py.File(DST, "r") as h:
        print(f"  X/data    dtype={h['X/data'].dtype}    shape={h['X/data'].shape}")
        print(f"  X/indices dtype={h['X/indices'].dtype} shape={h['X/indices'].shape}")
        print(f"  X/indptr  dtype={h['X/indptr'].dtype}  shape={h['X/indptr'].shape}")
        print(f"  raw present? {'raw' in h}")
    a_check.file.close()

    # ------------------------------------------------------------------
    # 5. compare slim X vs source X on a few rows (full reconstruction)
    # ------------------------------------------------------------------
    log("cross-checking: read first 10 cells from both files and compare X values ...")
    with h5py.File(SRC, "r") as hs, h5py.File(DST, "r") as hd:
        ip_s = hs["X/indptr"][:11]
        ip_d = hd["X/indptr"][:11]
        if not np.array_equal(ip_s, ip_d):
            print("  !! indptr differs in first 11 rows")
        for row in range(10):
            s0, s1 = ip_s[row], ip_s[row + 1]
            d0, d1 = ip_d[row], ip_d[row + 1]
            ds = hs["X/data"][s0:s1]
            dd = hd["X/data"][d0:d1]
            ixs = hs["X/indices"][s0:s1]
            ixd = hd["X/indices"][d0:d1]
            ok = np.array_equal(ds.astype(np.float32), dd) and np.array_equal(ixs.astype(np.int32), ixd)
            if not ok:
                print(f"  !! row {row} mismatch")
                break
        else:
            print(f"  first 10 rows: data and indices match between SRC and DST")

    print(f"\nsource:  {SRC.stat().st_size / 1e9:7.2f} GB")
    print(f"slim:    {DST.stat().st_size / 1e9:7.2f} GB")
    print(f"ratio:   {SRC.stat().st_size / DST.stat().st_size:.2f}x smaller")


if __name__ == "__main__":
    main()
