import os
import contextlib
import io
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from harmony import harmonize
import rapids_singlecell as rsc
from scipy import sparse
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.safe_save import safe_h5ad_write  # Importing the new safe save method

def anndata_cluster(
    adata_cluster,
    output_dir,
    sample_column='sample',
    num_features=2000,
    num_PCs=20,
    num_harmony=30,
    vars_to_regress_for_harmony=None,
    verbose=True
):
    if verbose:
        print('=== Processing data for clustering ===')

    # Ensure on CPU for HVG (memory-intensive on full matrix)
    rsc.get.anndata_to_CPU(adata_cluster)
    
    # Step A1: HVG on CPU (avoids GPU memory issue on large datasets)
    if verbose:
        print('Running HVG selection on CPU...')
    sc.pp.highly_variable_genes(
        adata_cluster,
        n_top_genes=num_features,
        flavor='seurat_v3',
        batch_key=sample_column
    )
    adata_cluster = adata_cluster[:, adata_cluster.var['highly_variable']].copy()
    
    if verbose:
        print(f'After HVG: {adata_cluster.shape[0]} cells × {adata_cluster.shape[1]} genes')
    
    # Now safe to use GPU (matrix is much smaller after HVG subset)
    if verbose:
        print('Moving to GPU for normalization, PCA, and Harmony...')
    rsc.get.anndata_to_GPU(adata_cluster)
    rsc.pp.normalize_total(adata_cluster, target_sum=1e4)
    rsc.pp.log1p(adata_cluster)

    # Step A2: GPU PCA
    rsc.pp.pca(adata_cluster, n_comps=num_PCs)

    if verbose:
        print('=== [GPU] Running Harmony integration ===')
        print('Variables to regress:', ', '.join(vars_to_regress_for_harmony or []))

    # Step A3: GPU Harmony
    Z = harmonize(
        adata_cluster.obsm['X_pca'],
        adata_cluster.obs,
        batch_key=vars_to_regress_for_harmony,  # allow list (covariates) or a single key
        max_iter_harmony=num_harmony,
        use_gpu=True
    )
    adata_cluster.obsm['X_pca_harmony'] = Z

    if verbose:
        print("End of harmony for adata_cluster.")

    # Back to CPU before saving
    rsc.get.anndata_to_CPU(adata_cluster)
    
    # Use the safe save function to write the file
    save_path = os.path.join(output_dir, 'adata_cell.h5ad')
    safe_h5ad_write(adata_cluster, save_path, verbose=verbose)

    return adata_cluster

def anndata_sample(
    adata_sample_diff,
    output_dir,
    batch_key,
    num_PCs=20,
    num_harmony=30,
    verbose=True
):
    if verbose:
        print('=== [CPU] Processing data for sample differences ===')

    # Ensure on CPU (full matrix too large for GPU)
    rsc.get.anndata_to_CPU(adata_sample_diff)
    
    # Store original counts
    adata_sample_diff.layers["counts"] = adata_sample_diff.X.copy()

    # CPU normalize/log/PCA (full matrix operations)
    sc.pp.normalize_total(adata_sample_diff, target_sum=1e4)
    sc.pp.log1p(adata_sample_diff)
    sc.pp.pca(adata_sample_diff, n_comps=num_PCs)

    # Restore counts
    adata_sample_diff.X = adata_sample_diff.layers["counts"].copy()
    del adata_sample_diff.layers["counts"]
    
    # Use the safe save function to write the file
    save_path = os.path.join(output_dir, 'adata_sample.h5ad')
    safe_h5ad_write(adata_sample_diff, save_path, verbose=verbose)

    return adata_sample_diff


from scipy.sparse import issparse

def preprocess_linux(
    h5ad_path,
    sample_meta_path,
    output_dir,
    sample_column='sample',
    cell_meta_path=None,
    batch_key='batch',
    num_PCs=20,
    num_harmony=30,
    num_features=2000,
    min_cells=500,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
    vars_to_regress=None,
    verbose=True
):
    from utils.random_seed import set_global_seed
    set_global_seed(seed=42, verbose=verbose)
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=== Reading input dataset ===")

    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f"Raw shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # ---------- vars_to_regress & Harmony covariates ----------
    if vars_to_regress is None:
        vars_to_regress = []
    flat_vars = []
    for v in vars_to_regress:
        if isinstance(v, (list, tuple, np.ndarray, pd.Index)):
            flat_vars.extend(map(str, list(v)))
        else:
            flat_vars.append(str(v))

    vars_to_regress_for_harmony = flat_vars.copy()
    if sample_column not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append(sample_column)

    # ---------- Attach metadata ----------
    if cell_meta_path is None:
        if sample_column not in adata.obs.columns:
            adata.obs[sample_column] = adata.obs_names.str.split(":").str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path).set_index("barcode")
        adata.obs = adata.obs.join(cell_meta, how="left")

    if sample_meta_path is not None:
        sample_meta = pd.read_csv(sample_meta_path).set_index(sample_column)
        adata.obs = adata.obs.join(sample_meta, on=sample_column, how="left")

    # ---------- Required columns check ----------
    required = flat_vars.copy()
    if batch_key:
        required.append(str(batch_key))
    required = list(dict.fromkeys(required))
    missing_vars = sorted(set(required) - set(map(str, adata.obs.columns)))
    if missing_vars:
        raise KeyError(
            f"The following variables are missing from adata.obs: {missing_vars}"
        )
    else:
        print("All required columns are present in adata.obs.")

    # ---------- CPU-side dtype + QC filtering ----------
    print("Before processing. Type of adata.X:", type(adata.X))
    print("adata.X dtype:", adata.X.dtype)
    print("adata.X is sparse:", issparse(adata.X))

    if adata.X.dtype != np.float32:
        print(f"Converting adata.X from {adata.X.dtype} to float32 (CPU)")
        if issparse(adata.X):
            adata.X = adata.X.astype(np.float32)
        else:
            adata.X = np.asarray(adata.X, dtype=np.float32)

    # 1) Filter genes/cells on CPU
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_features)
    if verbose:
        print(f"After initial filtering (CPU): {adata.shape[0]} cells × {adata.shape[1]} genes")

    # 2) QC metrics + mito on CPU
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], log1p=False, inplace=True)

    adata = adata[adata.obs["pct_counts_mt"] < pct_mito_cutoff].copy()
    if verbose:
        print(f"After mitochondrial filtering (CPU): {adata.shape[0]} cells × {adata.shape[1]} genes")

    # 3) Remove MT genes + excluded genes on CPU
    mt_genes = adata.var_names[adata.var_names.str.startswith("MT-")]
    genes_to_exclude = set(mt_genes) | set(exclude_genes or [])
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()
    if verbose:
        print(f"After gene exclusion filtering (CPU): {adata.shape[0]} cells × {adata.shape[1]} genes")

    # 4) Filter samples with too few cells on CPU
    cell_counts = adata.obs.groupby(sample_column).size()
    keep = cell_counts[cell_counts >= min_cells].index
    adata = adata[adata.obs[sample_column].isin(keep)].copy()
    if verbose:
        print(f"After sample filtering (CPU): {adata.shape[0]} cells × {adata.shape[1]} genes")
        print(f"Number of samples remaining: {len(keep)}")

    # 5) Final gene filtering on CPU (instead of rsc.pp.filter_genes)
    min_cells_final = int(0.001 * adata.n_obs)
    if min_cells_final > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells_final)
    if verbose:
        print(f"After final gene filtering (CPU): {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Optionally: scrublet here (CPU) if desired
    # if doublet:
    #     with contextlib.redirect_stdout(io.StringIO()):
    #         rsc.pp.scrublet(adata)
    #     adata = adata[~adata.obs["predicted_doublet"]].copy()

    adata.raw = adata.copy()
    if verbose:
        print("Preprocessing complete on CPU. Proceeding to clustering & sample analysis.")

    # ---------- Stay on CPU; let subfunctions handle GPU transitions ----------
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
        batch_key=batch_key,
        num_PCs=num_PCs,
        num_harmony=num_harmony,
        verbose=verbose,
    )

    if verbose:
        print(f"Total runtime: {time.time() - start_time:.2f} seconds")

    return adata_cluster, adata_sample_diff