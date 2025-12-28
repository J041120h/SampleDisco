import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
from harmony import harmonize
from scipy import sparse
from scipy.sparse import issparse

from utils.safe_save import safe_h5ad_write
from utils.random_seed import set_global_seed
from utils.merge_sample_meta import merge_sample_metadata


def anndata_cluster(
    adata_cluster,
    output_dir,
    sample_column="sample",
    num_features=2000,
    num_PCs=20,
    num_harmony=30,
    vars_to_regress_for_harmony=None,
    verbose=True,
):
    if verbose:
        print("=== [CPU] Processing data for clustering ===")

    sc.pp.highly_variable_genes(
        adata_cluster,
        n_top_genes=num_features,
        flavor="seurat_v3",
        batch_key=sample_column,
    )
    adata_cluster = adata_cluster[:, adata_cluster.var["highly_variable"]].copy()
    sc.pp.normalize_total(adata_cluster, target_sum=1e4)
    sc.pp.log1p(adata_cluster)

    sc.tl.pca(adata_cluster, n_comps=num_PCs, svd_solver="arpack")

    if verbose:
        print("=== [CPU] Running Harmony integration ===")
        print("Variables to regress:", ", ".join(vars_to_regress_for_harmony or []))

    Z = harmonize(
        adata_cluster.obsm["X_pca"],
        adata_cluster.obs,
        batch_key=vars_to_regress_for_harmony,
        max_iter_harmony=num_harmony,
        use_gpu=False,
    )
    adata_cluster.obsm["X_pca_harmony"] = Z

    if verbose:
        print("End of harmony for adata_cluster.")

    save_path = os.path.join(output_dir, "adata_cell.h5ad")
    safe_h5ad_write(adata_cluster, save_path, verbose=verbose)

    return adata_cluster


def anndata_sample(
    adata_sample_diff,
    output_dir,
    num_PCs=20,
    verbose=True,
):
    if verbose:
        print("=== [CPU] Processing data for sample differences ===")

    adata_sample_diff.layers["counts"] = adata_sample_diff.X.copy()

    sc.pp.normalize_total(adata_sample_diff, target_sum=1e4)
    sc.pp.log1p(adata_sample_diff)
    sc.tl.pca(adata_sample_diff, n_comps=num_PCs, svd_solver="arpack")

    adata_sample_diff.X = adata_sample_diff.layers["counts"].copy()
    del adata_sample_diff.layers["counts"]

    save_path = os.path.join(output_dir, "adata_sample.h5ad")
    safe_h5ad_write(adata_sample_diff, save_path, verbose=verbose)

    return adata_sample_diff


def preprocess(
    h5ad_path,
    sample_meta_path,
    output_dir,
    sample_column="sample",
    cell_meta_path=None,
    batch_key=None,
    num_PCs=20,
    num_harmony=30,
    num_features=2000,
    min_cells=500,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    vars_to_regress=None,
    verbose=True,
):
    """
    End-to-end preprocessing pipeline that reads an input AnnData (H5AD), attaches cell/sample metadata,
    performs QC filtering (genes/cells, mitochondrial %, excluded genes, minimum cells per sample),
    then produces two outputs: (1) a clustering-ready AnnData with HVG->normalize/log->PCA->Harmony,
    and (2) a sample-difference AnnData with normalize/log->PCA while preserving original counts in X.
    Both outputs are safely written to disk in output_dir/preprocess.
    """
    start_time = time.time()
    set_global_seed(seed=42, verbose=verbose)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Created output directory")

    output_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=== Reading input dataset ===")

    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f"Raw shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # ----------------------------
    # 1) Attach cell-level metadata
    # ----------------------------
    if cell_meta_path is None:
        if sample_column not in adata.obs.columns:
            if verbose:
                print(f"   ℹ️ No '{sample_column}' column in adata.obs; inferring from obs_names")
            adata.obs[sample_column] = adata.obs_names.str.split(":").str[0]
    else:
        if verbose:
            print(f"   📄 Merging cell-level metadata from: {cell_meta_path}")
        cell_meta = pd.read_csv(cell_meta_path).set_index("barcode")
        adata.obs = adata.obs.join(cell_meta, how="left")

        if sample_column not in adata.obs.columns:
            if verbose:
                print(f"   ℹ️ Still no '{sample_column}' column after cell_meta merge; "
                      f"inferring from obs_names")
            adata.obs[sample_column] = adata.obs_names.str.split(":").str[0]

    # --------------------------------------
    # 2) Attach *sample-level* metadata (NEW)
    # --------------------------------------
    if sample_meta_path is not None:
        if verbose:
            print("=== Merging sample-level metadata into adata.obs ===")
        adata = merge_sample_metadata(
            adata=adata,
            metadata_path=sample_meta_path,
            sample_column=sample_column,
            verbose=verbose,
        )
        
    # ------------------------------------------------
    # 3) Build vars_to_regress & batch keys *after* all
    #    metadata merges so required columns exist
    # ------------------------------------------------
    vars_to_regress = vars_to_regress or []
    flat_vars = []
    for v in vars_to_regress:
        if isinstance(v, (list, tuple, np.ndarray, pd.Index)):
            flat_vars.extend(map(str, list(v)))
        else:
            flat_vars.append(str(v))

    vars_to_regress_for_harmony = flat_vars.copy()
    if sample_column not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append(sample_column)

    flat_batch_keys = []
    if batch_key is not None:
        if isinstance(batch_key, (list, tuple, np.ndarray, pd.Index)):
            flat_batch_keys.extend([str(b) for b in list(batch_key)])
        else:
            flat_batch_keys.append(str(batch_key))

    required = list(dict.fromkeys(flat_vars + flat_batch_keys))
    if required:
        missing_vars = sorted(set(required) - set(map(str, adata.obs.columns)))
    else:
        missing_vars = []

    if missing_vars:
        raise KeyError(f"The following variables are missing from adata.obs: {missing_vars}")
    else:
        if verbose:
            print("All required columns are present in adata.obs.")

    # ---------------------------
    # 4) QC, filtering, and genes
    # ---------------------------
    print("Type of adata.X:", type(adata.X))
    print("adata.X dtype:", adata.X.dtype)
    print("adata.X is sparse:", issparse(adata.X))

    if adata.X.dtype != np.float32:
        print(f"Converting adata.X from {adata.X.dtype} to float32")
        if issparse(adata.X):
            adata.X = adata.X.astype(np.float32)
        else:
            adata.X = np.asarray(adata.X, dtype=np.float32)

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_features)
    if verbose:
        print(f"After initial filtering: {adata.shape[0]} cells × {adata.shape[1]} genes")

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    adata = adata[adata.obs["pct_counts_mt"] < pct_mito_cutoff].copy()
    if verbose:
        print(f"After mitochondrial filtering: {adata.shape[0]} cells × {adata.shape[1]} genes")

    mt_genes = adata.var_names[adata.var_names.str.startswith("MT-")]
    genes_to_exclude = set(mt_genes) | set(exclude_genes or [])
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()
    if verbose:
        print(f"After gene exclusion filtering: {adata.shape[0]} cells × {adata.shape[1]} genes")

    cell_counts = adata.obs.groupby(sample_column).size()
    keep = cell_counts[cell_counts >= min_cells].index
    adata = adata[adata.obs[sample_column].isin(keep)].copy()
    if verbose:
        print(f"After sample filtering: {adata.shape[0]} cells × {adata.shape[1]} genes")
        print(f"Number of samples remaining: {len(keep)}")

    sc.pp.filter_genes(adata, min_cells=int(0.001 * adata.n_obs))
    if verbose:
        print(f"After final gene filtering: {adata.shape[0]} cells × {adata.shape[1]} genes")
        print(f"Processed shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

    adata.raw = adata.copy()
    if verbose:
        print("Preprocessing complete!")

    # -----------------------------------
    # 5) Split into cluster / sample views
    # -----------------------------------
    adata_cluster = adata.copy()
    adata_sample_diff = adata.copy()

    adata_cluster = anndata_cluster(
        adata_cluster=adata_cluster,
        output_dir=output_dir,
        sample_column=sample_column,
        num_features=num_features,
        num_PCs=num_PCs,
        num_harmony=num_harmony,
        vars_to_regress_for_harmony=vars_to_regress_for_harmony,
        verbose=verbose,
    )

    adata_sample_diff = anndata_sample(
        adata_sample_diff=adata_sample_diff,
        output_dir=output_dir,
        num_PCs=num_PCs,
        verbose=verbose,
    )

    if verbose:
        print(f"Total runtime: {time.time() - start_time:.2f} seconds")

    return adata_cluster, adata_sample_diff