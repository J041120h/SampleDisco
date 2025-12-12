import os
import contextlib
import io
import numpy as np
import pandas as pd
import scanpy as sc
from harmony import harmonize
import time
from scipy import sparse
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
        print('=== [CPU] Processing data for clustering ===')

    # Step A1: HVG selection (before normalization)
    sc.pp.highly_variable_genes(
        adata_cluster,
        n_top_genes=num_features,
        flavor='seurat_v3',
        batch_key=sample_column
    )
    adata_cluster = adata_cluster[:, adata_cluster.var['highly_variable']].copy()
    sc.pp.normalize_total(adata_cluster, target_sum=1e4)
    sc.pp.log1p(adata_cluster)

    # Step A2: CPU PCA
    sc.tl.pca(adata_cluster, n_comps=num_PCs, svd_solver='arpack')

    if verbose:
        print('=== [CPU] Running Harmony integration ===')
        print('Variables to regress:', ', '.join(vars_to_regress_for_harmony or []))

    # Step A3: Harmony
    Z = harmonize(
        adata_cluster.obsm['X_pca'],
        adata_cluster.obs,
        batch_key=vars_to_regress_for_harmony,  # allow list (covariates) or a single key
        max_iter_harmony=num_harmony,
        use_gpu=False
    )
    adata_cluster.obsm['X_pca_harmony'] = Z

    if verbose:
        print("End of harmony for adata_cluster.")

    # Use safe saving method instead of direct write
    save_path = os.path.join(output_dir, 'adata_cell.h5ad')
    safe_h5ad_write(adata_cluster, save_path, verbose=verbose)
    
    return adata_cluster

def anndata_sample(
    adata_sample_diff,
    output_dir,
    num_PCs=20,
    verbose=True
):
    if verbose:
        print('=== [CPU] Processing data for sample differences ===')

    # Store original counts
    adata_sample_diff.layers["counts"] = adata_sample_diff.X.copy()

    # Preprocess
    sc.pp.normalize_total(adata_sample_diff, target_sum=1e4)
    sc.pp.log1p(adata_sample_diff)
    sc.tl.pca(adata_sample_diff, n_comps=num_PCs, svd_solver='arpack')

    # Restore original counts
    adata_sample_diff.X = adata_sample_diff.layers["counts"].copy()
    del adata_sample_diff.layers["counts"]

    # Use safe saving method instead of direct write
    save_path = os.path.join(output_dir, 'adata_sample.h5ad')
    safe_h5ad_write(adata_sample_diff, save_path, verbose=verbose)

    return adata_sample_diff

def preprocess(
    h5ad_path,
    sample_meta_path,
    output_dir,
    sample_column='sample',
    cell_meta_path=None,
    batch_key=None,  # Changed to None default, will accept string or list
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
    start_time = time.time()
    from utils.random_seed import set_global_seed
    set_global_seed(seed = 42, verbose = verbose)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Created output directory")

    output_dir = os.path.join(output_dir, 'preprocess')
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print('=== Reading input dataset ===')

    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f'Raw shape: {adata.shape[0]} cells × {adata.shape[1]} genes')

    # ---------- Normalize vars_to_regress, build Harmony covariates ----------
    if vars_to_regress is None:
        vars_to_regress = []
    # Flatten and coerce to strings (guards against arrays/lists)
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
            adata.obs[sample_column] = adata.obs_names.str.split(':').str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path)
        cell_meta.set_index('barcode', inplace=True)
        adata.obs = adata.obs.join(cell_meta, how='left')

    if sample_meta_path is not None:
        sample_meta = pd.read_csv(sample_meta_path)
        sample_meta = sample_meta.set_index(sample_column)
        adata.obs = adata.obs.join(
            sample_meta,
            on=sample_column, 
            how='left'
        )

    # ---------- Required columns check (robust to None/unhashables) ----------
    required = flat_vars.copy()
    
    # Handle batch_key as either string or list
    if batch_key is not None:
        if isinstance(batch_key, (list, tuple)):
            required.extend([str(b) for b in batch_key])
        else:
            required.append(str(batch_key))
    
    # De-duplicate while preserving order
    required = list(dict.fromkeys(required))
    missing_vars = sorted(set(required) - set(map(str, adata.obs.columns)))
    if missing_vars:
        raise KeyError(f"The following variables are missing from adata.obs: {missing_vars}")
    else:
        print("All required columns are present in adata.obs.")

    print("Type of adata.X:", type(adata.X))
    print("adata.X dtype:", adata.X.dtype)
    print("adata.X is sparse:", sparse.issparse(adata.X))

    if adata.X.dtype != np.float32:
        print(f"Converting adata.X from {adata.X.dtype} to float32")
        adata.X = adata.X.astype(np.float32)

    # Filter genes and cells
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_features)
    if verbose:
        print(f'After initial filtering: {adata.shape[0]} cells × {adata.shape[1]} genes')

    # Calculate mitochondrial gene percentage
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Filter based on mitochondrial percentage
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()
    if verbose:
        print(f'After mitochondrial filtering: {adata.shape[0]} cells × {adata.shape[1]} genes')

    # Filter MT genes and excluded genes
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    genes_to_exclude = set(mt_genes) | set(exclude_genes or [])
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()
    if verbose:
        print(f'After gene exclusion filtering: {adata.shape[0]} cells × {adata.shape[1]} genes')

    # Filter samples with too few cells
    cell_counts = adata.obs.groupby(sample_column).size()
    keep = cell_counts[cell_counts >= min_cells].index
    adata = adata[adata.obs[sample_column].isin(keep)].copy()
    if verbose:
        print(f'After sample filtering: {adata.shape[0]} cells × {adata.shape[1]} genes')
        print(f'Number of samples remaining: {len(keep)}')

    # Additional gene filtering based on cell percentage
    sc.pp.filter_genes(adata, min_cells=int(0.001 * adata.n_obs))
    if verbose:
        print(f'After final gene filtering: {adata.shape[0]} cells × {adata.shape[1]} genes')

    if verbose:
        print(f'Processed shape: {adata.shape[0]} cells × {adata.shape[1]} genes')

    if doublet:
        # Run scrublet quietly and APPLY the filter
        with contextlib.redirect_stdout(io.StringIO()):
            sc.pp.scrublet(adata, batch_key=sample_column)
        adata = adata[~adata.obs['predicted_doublet']].copy()

    adata.raw = adata.copy()
    if verbose:
        print("Preprocessing complete!")

    adata_cluster = adata.copy()
    adata_sample_diff = adata.copy()

    # Updated function call for anndata_cluster - matching its actual signature
    adata_cluster = anndata_cluster(
        adata_cluster=adata_cluster,
        output_dir=output_dir,
        sample_column=sample_column,
        num_features=num_features,
        num_PCs=num_PCs,
        num_harmony=num_harmony,
        vars_to_regress_for_harmony=vars_to_regress_for_harmony,
        verbose=verbose
    )

    # Updated function call for anndata_sample - matching its actual signature
    adata_sample_diff = anndata_sample(
        adata_sample_diff=adata_sample_diff,
        output_dir=output_dir,
        num_PCs=num_PCs,
        verbose=verbose
    )

    if verbose:
        print(f"Total runtime: {time.time() - start_time:.2f} seconds")

    return adata_cluster, adata_sample_diff