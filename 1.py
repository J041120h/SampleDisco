import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

def add_raw_counts_layer_by_integer_index(preprocessed_h5ad_path, raw_h5ad_path,
                                          layer_name="raw_counts", verbose=True):
    import scanpy as sc

    adata_pre = sc.read_h5ad(preprocessed_h5ad_path)
    adata_raw = sc.read_h5ad(raw_h5ad_path)

    # 1) Make raw obs_names = integer row IDs (as strings)
    adata_raw.obs_names = pd.Index(map(str, range(adata_raw.n_obs)))

    # 2) Get pre indices as a NumPy array (NOT a pandas Index)
    pre_idx = adata_pre.obs_names.astype(int).to_numpy()

    # 3) Map genes
    genes_pre = adata_pre.var_names
    genes_raw = adata_raw.var_names

    if not np.all(np.isin(genes_pre, genes_raw)):
        missing = genes_pre[~np.isin(genes_pre, genes_raw)]
        raise ValueError(f"Preprocessed genes missing in raw, e.g. {missing[:10]}")

    gene_to_raw = pd.Series(np.arange(len(genes_raw)), index=genes_raw)
    gene_idx = gene_to_raw.loc[genes_pre].values

    raw_X = adata_raw.X
    if isinstance(raw_X, spmatrix):
        # use AnnData slicing for sparse
        aligned_raw = adata_raw[pre_idx, :][:, genes_pre].X
    else:
        aligned_raw = raw_X[np.ix_(pre_idx, gene_idx)]

    adata_pre.layers[layer_name] = aligned_raw
    adata_pre.write(preprocessed_h5ad_path)

    if verbose:
        print("Done: added raw counts layer by integer index.")

if __name__ == "__main__":
    add_raw_counts_layer_by_integer_index(
        preprocessed_h5ad_path = "/dcs07/hongkai/data/harry/result/processed_data/279_adata_cell.h5ad",
        raw_h5ad_path = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/Benchmark/count_data_Su_subset.h5ad",
        layer_name="raw_counts",
        verbose=True,
    )