"""
Concrete check (not suspicion) of how much Batch / File_name / Age signal
remains in the cell-level Harmony output from round 1.

Inputs
  rna/preprocess/adata_preprocessed.h5ad
    - obsm['Z_clust']  (n_cells x 20)  -- pass-1 Harmony (sample-removed)
    - obsm['Z_cmd']    (n_cells x 20)  -- pass-2 Harmony (sample-preserved)
    - obs has Batch, File_name, Tube_id, Age, Sex (after merge)

Methods
  - For each embedding (Z_clust, Z_cmd), compute per-dim R^2 of:
      Age           : linear regression at the cell level   (R^2 = r^2)
      Batch (14)    : one-way ANOVA eta^2 = SS_between / SS_total
      File_name(N)  : same
      Sex (2)       : same

This is the right diagnostic because the sample embedding aggregates cells
within each sample; any per-cell embedding dim whose ANOVA eta^2 against
File_name is high will leak into the sample-level mean and survive
sample-level Harmony.

Output  rna/preprocess/cell_level_batch_leak.csv  (long form: dim, variable, eta2/r2)
        plus a console summary.
"""

from pathlib import Path
import time

import anndata as ad
import numpy as np
import pandas as pd

H5AD = Path("/dcs07/hongkai/data/harry/result/health_aging_PBMC/rna/preprocess/adata_preprocessed.h5ad")
OUT_CSV = Path("/dcs07/hongkai/data/harry/result/health_aging_PBMC/rna/preprocess/cell_level_batch_leak.csv")


def banner(s):
    print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)


def eta2_categorical(values: np.ndarray, group_codes: np.ndarray) -> float:
    """ANOVA eta^2 for a 1D numeric `values` against an integer-coded categorical `group_codes`.
       Fast O(n + K)."""
    mean_all = values.mean()
    ss_total = float(((values - mean_all) ** 2).sum())
    if ss_total == 0:
        return 0.0
    # SS_between = sum_k n_k (mean_k - mean_all)^2
    K = int(group_codes.max()) + 1
    # group sums and counts
    sums = np.bincount(group_codes, weights=values, minlength=K)
    cnts = np.bincount(group_codes, minlength=K).astype(np.float64)
    means = np.divide(sums, cnts, out=np.zeros_like(sums), where=cnts > 0)
    ss_between = float((cnts * (means - mean_all) ** 2).sum())
    return ss_between / ss_total


def main():
    banner("Loading adata (backed) and obs")
    t0 = time.time()
    a = ad.read_h5ad(H5AD, backed="r")
    print(f"  shape={a.shape}  obsm keys={list(a.obsm.keys())}  load t={time.time() - t0:.1f}s")
    obs = a.obs.copy()
    print(f"  obs cols: {list(obs.columns)[:25]}...")
    age = obs["Age"].astype(float).values
    sex = obs["Sex"].astype(str)
    batch = obs["Batch"].astype(str)
    fname = obs["File_name"].astype(str)
    print(f"  unique Batch     = {batch.nunique()}")
    print(f"  unique File_name = {fname.nunique()}")
    print(f"  unique Sex       = {sex.nunique()}")
    print(f"  Age range        = [{age.min()}, {age.max()}]")

    batch_codes = batch.astype("category").cat.codes.values.astype(np.int64)
    fname_codes = fname.astype("category").cat.codes.values.astype(np.int64)
    sex_codes   = sex.astype("category").cat.codes.values.astype(np.int64)

    results = []
    for emb_key in ["Z_clust", "Z_cmd"]:
        banner(f"Embedding: {emb_key}")
        t = time.time()
        # load this obsm into memory (n_cells x 20 = ~150 MB)
        E = np.asarray(a.obsm[emb_key])
        print(f"  shape={E.shape}  dtype={E.dtype}  load t={time.time() - t:.1f}s")
        n_dim = E.shape[1]

        for j in range(n_dim):
            v = E[:, j].astype(np.float64)
            # Age: linear R^2 = r^2
            cov = float(((v - v.mean()) * (age - age.mean())).sum())
            denom = float(np.sqrt(((v - v.mean()) ** 2).sum() * ((age - age.mean()) ** 2).sum()))
            r2_age = (cov / denom) ** 2 if denom > 0 else 0.0
            r2_batch = eta2_categorical(v, batch_codes)
            r2_fname = eta2_categorical(v, fname_codes)
            r2_sex = eta2_categorical(v, sex_codes)
            results.append(dict(embedding=emb_key, dim=j + 1,
                                age=r2_age, batch=r2_batch,
                                file_name=r2_fname, sex=r2_sex))

        # short summary
        dim_results = [r for r in results if r["embedding"] == emb_key]
        print(f"  {'dim':>4s}  {'age':>8s}  {'batch':>8s}  {'file_name':>9s}  {'sex':>8s}")
        for r in dim_results:
            print(f"  {r['dim']:>4d}  {r['age']:8.4f}  {r['batch']:8.4f}  {r['file_name']:9.4f}  {r['sex']:8.4f}")
        df = pd.DataFrame(dim_results)
        print(f"  ROW SUMS (across 20 dims) — file_name={df['file_name'].sum():.3f}  "
              f"batch={df['batch'].sum():.3f}  age={df['age'].sum():.4f}  sex={df['sex'].sum():.4f}")
        print(f"  MAX-dim    file_name={df['file_name'].max():.3f}  "
              f"batch={df['batch'].max():.3f}  age={df['age'].max():.4f}  sex={df['sex'].max():.4f}")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    a.file.close()


if __name__ == "__main__":
    main()
