#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
import scanpy as sc


def add_raw_counts_layer_by_obs_names(
    preprocessed_h5ad_path,
    raw_h5ad_path,
    layer_name: str = "raw_counts",
    verbose: bool = True,
):
    """
    Add a raw-counts layer to a preprocessed AnnData by aligning cells
    and genes using obs_names / var_names.

    Assumes:
      - adata_pre.obs_names are a subset of (or equal to) adata_raw.obs_names
      - adata_pre.var_names are a subset of (or equal to) adata_raw.var_names
    """

    if verbose:
        print(f"[add_raw_counts_layer_by_obs_names] Reading preprocessed: {preprocessed_h5ad_path}")
        print(f"[add_raw_counts_layer_by_obs_names] Reading raw:          {raw_h5ad_path}")

    adata_pre = sc.read_h5ad(preprocessed_h5ad_path)
    adata_raw = sc.read_h5ad(raw_h5ad_path)

    # ------------------------------------------------------------------
    # 1) Check obs_names alignment (cells)
    # ------------------------------------------------------------------
    if not adata_pre.obs_names.is_unique:
        raise ValueError("preprocessed AnnData has non-unique obs_names; cannot align safely.")

    if not adata_raw.obs_names.is_unique:
        raise ValueError("raw AnnData has non-unique obs_names; cannot align safely.")

    # Cells present in pre that are missing in raw
    missing_cells = adata_pre.obs_names.difference(adata_raw.obs_names)
    if len(missing_cells) > 0:
        # Hard fail so you don't silently misalign anything
        n_show = min(10, len(missing_cells))
        raise ValueError(
            f"{len(missing_cells)} cells in preprocessed AnnData are not found in raw AnnData.\n"
            f"Examples: {list(missing_cells[:n_show])}"
        )

    if verbose:
        print(f"  Cells in preprocessed: {adata_pre.n_obs}")
        print(f"  Cells in raw:         {adata_raw.n_obs}")
        print("  All preprocessed cells are present in raw. Aligning in that order...")

    # Reorder raw AnnData rows to match preprocessed cells exactly
    adata_raw = adata_raw[adata_pre.obs_names, :]

    # ------------------------------------------------------------------
    # 2) Check var_names alignment (genes)
    # ------------------------------------------------------------------
    genes_pre = adata_pre.var_names
    genes_raw = adata_raw.var_names

    missing_genes = genes_pre.difference(genes_raw)
    if len(missing_genes) > 0:
        n_show = min(10, len(missing_genes))
        raise ValueError(
            f"{len(missing_genes)} genes in preprocessed AnnData are not found in raw AnnData.\n"
            f"Examples: {list(missing_genes[:n_show])}"
        )

    if verbose:
        print(f"  Genes in preprocessed: {len(genes_pre)}")
        print(f"  Genes in raw:          {len(genes_raw)}")
        print("  All preprocessed genes are present in raw. Aligning columns...")

    # Map gene order: pre → raw indices
    gene_to_raw = pd.Series(np.arange(len(genes_raw)), index=genes_raw)
    gene_idx = gene_to_raw.loc[genes_pre].values

    # ------------------------------------------------------------------
    # 3) Slice raw.X to [cells in pre order, genes in pre order]
    # ------------------------------------------------------------------
    raw_X = adata_raw.X
    if isinstance(raw_X, spmatrix):
        # For sparse matrices, rely on AnnData's own slicing (keeps sparsity)
        # Rows already aligned by obs; just align genes by label
        aligned_raw = adata_raw[:, genes_pre].X
    else:
        # Dense NumPy array – index by integer positions
        aligned_raw = raw_X[:, gene_idx]

    # ------------------------------------------------------------------
    # 4) Add as new layer and write back
    # ------------------------------------------------------------------
    adata_pre.layers[layer_name] = aligned_raw

    if verbose:
        print(f"  Added layer '{layer_name}' with shape {aligned_raw.shape} to adata_pre.")
        print(f"  Writing updated AnnData back to: {preprocessed_h5ad_path}")

    adata_pre.write(preprocessed_h5ad_path)

    if verbose:
        print("Done: added raw counts layer by obs_names / var_names alignment.")


def print_batch_sample_counts(h5ad_path, batch_col="batch"):
    """
    Load a pseudobulk AnnData (samples × genes) and print how many samples
    exist in each batch.

    Parameters
    ----------
    h5ad_path : str
        Path to pseudobulk_sample.h5ad (samples are rows).
    batch_col : str
        Column name in .obs containing batch labels.
    """
    print(f"Loading pseudobulk AnnData from: {h5ad_path}")
    adata = sc.read(h5ad_path)

    if batch_col not in adata.obs.columns:
        print(f"[ERROR] '{batch_col}' not found in adata.obs!")
        print(f"Available columns: {list(adata.obs.columns)}")
        return

    batch_counts = adata.obs[batch_col].value_counts()

    print("\n=== Batch → Sample Counts ===")
    for batch, count in batch_counts.items():
        print(f"{batch:15s} : {count} samples")

    print("\nTotal samples:", adata.n_obs)
    print("Total batches:", len(batch_counts))



# if __name__ == "__main__":
#     add_raw_counts_layer_by_obs_names(
#         preprocessed_h5ad_path="/dcs07/hongkai/data/harry/result/processed_data/100_adata_cell.h5ad",
#         raw_h5ad_path="/dcl01/hongkai/data/data/hjiang/Data/covid_data/Benchmark/count_data_subsample_100samples.h5ad",
#         layer_name="raw_counts",
#         verbose=True,
#     )
    # print_batch_sample_counts("/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_50_sample/rna/pseudobulk/pseudobulk_sample.h5ad")

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse, csr_matrix


def debug_inspect_adata_for_gpu(
    adata: sc.AnnData,
    sample_column: str,
    batch_key: str | None = None,
    num_features: int = 2000,
    name: str = "adata",
    cpu_hvg_test: bool = True,
    max_cells_for_hvg: int = 20000,
):
    """Sanity-check an AnnData object before sending it to RAPIDS/GPUs."""
    print(f"\n========== DEBUG INSPECT: {name} ==========")
    print(f"Shape: {adata.n_obs} cells × {adata.n_vars} genes")

    # ---------- obs checks ----------
    if sample_column not in adata.obs.columns:
        print(f"[WARN] `{sample_column}` not in adata.obs.columns")
    else:
        n_missing = adata.obs[sample_column].isna().sum()
        print(f"[META] `{sample_column}` levels: {adata.obs[sample_column].nunique()} (missing={n_missing})")
        print("[META] sample counts (top 10):")
        print(adata.obs[sample_column].value_counts().head(10))

    if batch_key is not None:
        if batch_key in adata.obs.columns:
            n_missing_b = adata.obs[batch_key].isna().sum()
            print(f"[META] `{batch_key}` levels: {adata.obs[batch_key].nunique()} (missing={n_missing_b})")
            print("[META] batch counts (top 10):")
            print(adata.obs[batch_key].value_counts().head(10))
        else:
            print(f"[WARN] `{batch_key}` not in adata.obs.columns")

    # ---------- X structure ----------
    X = adata.X
    print(f"[X] type: {type(X)}")
    print(f"[X] dtype: {getattr(X, 'dtype', 'N/A')}")
    sparse_flag = issparse(X)
    print(f"[X] is sparse: {sparse_flag}")

    if sparse_flag:
        if not isinstance(X, csr_matrix):
            print("[X] Converting sparse matrix to CSR format (CPU)...")
            X = csr_matrix(X)
            adata.X = X

        nnz = X.nnz
        density = nnz / (X.shape[0] * X.shape[1])
        print(f"[X] nnz: {nnz} (density ~ {density:.4e})")

        indptr = X.indptr
        indices = X.indices

        if not np.all(indptr[:-1] <= indptr[1:]):
            print("[X][ERROR] indptr not monotonically increasing")

        if indptr[-1] != nnz:
            print(f"[X][ERROR] indptr[-1]={indptr[-1]} != nnz={nnz}")

        if indices.size > 0:
            if indices.min() < 0 or indices.max() >= X.shape[1]:
                print(
                    f"[X][ERROR] indices out of bounds: "
                    f"min={indices.min()}, max={indices.max()}, n_vars={X.shape[1]}"
                )
            else:
                print(f"[X] indices OK, range=[{indices.min()}, {indices.max()}]")

        data = X.data
    else:
        data = np.asarray(X)
        nnz = (data != 0).sum()
        density = nnz / data.size
        print(f"[X] nnz (dense): {nnz} (density ~ {density:.4e})")

    # ---------- Value checks ----------
    if data.size > 0:
        n_nan = np.isnan(data).sum()
        n_posinf = np.isposinf(data).sum()
        n_neginf = np.isneginf(data).sum()
        n_neg = (data < 0).sum()

        print(f"[VAL] NaN count: {int(n_nan)}")
        print(f"[VAL] +inf: {int(n_posinf)}, -inf: {int(n_neginf)}")
        print(f"[VAL] negative entries: {int(n_neg)}")

        nz = data[data != 0]
        if nz.size > 0:
            print(
                f"[VAL] nonzero stats: "
                f"min={nz.min():.3g}, max={nz.max():.3g}, mean={nz.mean():.3g}"
            )
        else:
            print("[VAL][WARN] all values are zero")

    # ---------- CPU HVG test ----------
    if cpu_hvg_test:
        print("\n[HVG-CPU-TEST] Running Scanpy HVG on random subset...")
        n_cells = min(max_cells_for_hvg, adata.n_obs)

        if n_cells < 500:
            print("[HVG-CPU-TEST] Too few cells, skipping.")
        else:
            idx = np.random.choice(adata.n_obs, size=n_cells, replace=False)
            adata_sub = adata[idx].copy()
            sc.pp.filter_genes(adata_sub, min_cells=10)

            print(f"[HVG-CPU-TEST] subset shape: {adata_sub.n_obs} × {adata_sub.n_vars}")

            try:
                sc.pp.highly_variable_genes(
                    adata_sub,
                    n_top_genes=min(num_features, adata_sub.n_vars - 1),
                    flavor="seurat_v3",
                    batch_key=sample_column if sample_column in adata_sub.obs.columns else None,
                )
                print("[HVG-CPU-TEST] success")
                print("[HVG-CPU-TEST] HVGs:", int(adata_sub.var["highly_variable"].sum()))
            except Exception as e:
                print(f"[HVG-CPU-TEST][ERROR] {repr(e)}")

    print("========== END DEBUG INSPECT ==========\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect AnnData for GPU/RAPIDS issues")
    parser.add_argument("--h5ad", required=True, help="Path to input .h5ad")
    parser.add_argument("--sample_column", required=True, help="Sample column in obs")
    parser.add_argument("--batch_key", default=None, help="Batch column in obs")
    parser.add_argument("--num_features", type=int, default=2000)
    parser.add_argument("--no_cpu_hvg_test", action="store_true")
    parser.add_argument("--max_cells_for_hvg", type=int, default=20000)

    args = parser.parse_args()

    print("\n=== Loading AnnData ===")
    adata = sc.read_h5ad(args.h5ad)

    debug_inspect_adata_for_gpu(
        adata=adata,
        sample_column=args.sample_column,
        batch_key=args.batch_key,
        num_features=args.num_features,
        name=args.h5ad,
        cpu_hvg_test=not args.no_cpu_hvg_test,
        max_cells_for_hvg=args.max_cells_for_hvg,
    )


if __name__ == "__main__":
    main()
