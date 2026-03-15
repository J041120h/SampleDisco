import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
from harmony import harmonize
from scipy.sparse import issparse

from utils.safe_save import safe_h5ad_write
from utils.random_seed import set_global_seed
from utils.merge_sample_meta import merge_sample_metadata


def anndata_cluster(
    adata_cluster,
    output_dir,
    sample_column="sample",
    num_cell_hvgs=2000,
    cell_embedding_num_PCs=20,
    num_harmony_iterations=30,
    vars_to_regress_for_harmony=None,
    verbose=True,
):
    if verbose:
        print("=== [CPU] Processing data for clustering ===")

    sc.pp.highly_variable_genes(
        adata_cluster,
        n_top_genes=num_cell_hvgs,
        flavor="seurat_v3",
        batch_key=sample_column,
    )
    adata_cluster = adata_cluster[:, adata_cluster.var["highly_variable"]].copy()

    if verbose:
        print(f"After HVG selection: {adata_cluster.shape[0]} cells × {adata_cluster.shape[1]} genes")

    sc.pp.normalize_total(adata_cluster, target_sum=1e4)
    sc.pp.log1p(adata_cluster)
    sc.tl.pca(adata_cluster, n_comps=cell_embedding_num_PCs, svd_solver="arpack")

    if verbose:
        print("=== [CPU] Running Harmony integration ===")
        print("Variables to regress:", ", ".join(vars_to_regress_for_harmony or []))

    harmony_embeddings = harmonize(
        adata_cluster.obsm["X_pca"],
        adata_cluster.obs,
        batch_key=vars_to_regress_for_harmony,
        max_iter_harmony=num_harmony_iterations,
        use_gpu=False,
    )
    adata_cluster.obsm["X_pca_harmony"] = harmony_embeddings

    if verbose:
        print("Harmony integration complete.")

    save_path = os.path.join(output_dir, "adata_cell.h5ad")
    safe_h5ad_write(adata_cluster, save_path)

    return adata_cluster


def anndata_sample(
    adata_sample_diff,
    output_dir,
    sample_embedding_num_PCs=20,
    verbose=True,
):
    if verbose:
        print("=== [CPU] Processing data for sample differences ===")

    X_original_counts = adata_sample_diff.X.copy()

    sc.pp.normalize_total(adata_sample_diff, target_sum=1e4)
    sc.pp.log1p(adata_sample_diff)
    sc.tl.pca(adata_sample_diff, n_comps=sample_embedding_num_PCs, svd_solver="arpack")

    adata_sample_diff.X = X_original_counts
    del X_original_counts

    save_path = os.path.join(output_dir, "adata_sample.h5ad")
    safe_h5ad_write(adata_sample_diff, save_path)

    return adata_sample_diff

def _flatten_to_strings(values):
    """Flatten nested iterables to a list of strings."""
    flattened = []
    for value in values:
        if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
            flattened.extend(str(x) for x in value)
        else:
            flattened.append(str(value))
    return flattened

def _ensure_sample_column(adata, sample_column, verbose=True):
    """Infer sample column from obs_names if not present."""
    if sample_column not in adata.obs.columns:
        if verbose:
            print(f"   ℹ️ No '{sample_column}' column in adata.obs; inferring from obs_names")
        adata.obs[sample_column] = adata.obs_names.str.split(":").str[0]


def preprocess(
    h5ad_path,
    sample_meta_path,
    output_dir,
    sample_column="sample",
    cell_meta_path=None,
    batch_key=None,
    cell_embedding_num_PCs=20,
    num_harmony_iterations=30,
    num_cell_hvgs=2000,
    min_cells=500,
    min_genes=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    vars_to_regress=None,
    verbose=True,
):
    """
    End-to-end preprocessing pipeline for single-cell RNA-seq data.

    Reads an input AnnData (H5AD), attaches cell/sample metadata, performs QC filtering
    (genes/cells, mitochondrial %, excluded genes, minimum cells per sample), then produces
    two outputs:
      (1) A clustering-ready AnnData with HVG -> normalize/log -> PCA -> Harmony
      (2) A sample-difference AnnData with normalize/log -> PCA while preserving original counts in X

    Both outputs are safely written to disk in output_dir/preprocess.

    Parameters
    ----------
    h5ad_path : str
        Path to the input h5ad file.
    sample_meta_path : str
        Path to the sample-level metadata CSV file.
    output_dir : str
        Directory to save output files.
    sample_column : str, default="sample"
        Column name in adata.obs that identifies samples.
    cell_meta_path : str, optional
        Path to the cell-level metadata CSV file.
    batch_key : str, optional
        Column name(s) for batch information.
    cell_embedding_num_PCs : int, default=20
        Number of principal components for cell-level PCA.
    num_harmony_iterations : int, default=30
        Maximum number of Harmony iterations.
    num_cell_hvgs : int, default=2000
        Number of highly variable genes to select.
    min_cells : int, default=500
        Minimum number of cells for a gene/sample to be kept.
    min_genes : int, default=500
        Minimum number of genes expressed in a cell for it to be kept.
    pct_mito_cutoff : float, default=20
        Maximum percentage of mitochondrial counts allowed per cell.
    exclude_genes : list, optional
        List of gene names to exclude from analysis.
    vars_to_regress : list, optional
        Variables to regress out during Harmony integration.
    verbose : bool, default=True
        Whether to print progress messages.

    Returns
    -------
    tuple
        (adata_cluster, adata_sample_diff) - Two AnnData objects for
        clustering and sample-level differential analysis respectively.
    """
    start_time = time.time()
    set_global_seed(seed=42)

    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=== Reading input dataset ===")

    adata = sc.read_h5ad(h5ad_path)

    if verbose:
        print(f"Raw shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if cell_meta_path is None:
        _ensure_sample_column(adata, sample_column, verbose)
    else:
        if verbose:
            print(f"   📄 Merging cell-level metadata from: {cell_meta_path}")
        cell_metadata = pd.read_csv(cell_meta_path).set_index("barcode")
        adata.obs = adata.obs.join(cell_metadata, how="left")
        _ensure_sample_column(adata, sample_column, verbose)

    if sample_meta_path is not None:
        if verbose:
            print("=== Merging sample-level metadata into adata.obs ===")
        adata = merge_sample_metadata(
            adata=adata,
            metadata_path=sample_meta_path,
            sample_column=sample_column,
            verbose=verbose,
        )

    flattened_vars_to_regress = _flatten_to_strings(vars_to_regress or [])
    vars_to_regress_for_harmony = flattened_vars_to_regress.copy()
    if sample_column not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append(sample_column)

    flattened_batch_keys = _flatten_to_strings([batch_key] if batch_key else [])

    required_columns = list(dict.fromkeys(flattened_vars_to_regress + flattened_batch_keys))
    if required_columns:
        missing_columns = sorted(set(required_columns) - set(adata.obs.columns.astype(str)))
        if missing_columns:
            raise KeyError(f"The following variables are missing from adata.obs: {missing_columns}")

    if verbose:
        print("All required columns are present in adata.obs.")

    if verbose:
        print(f"adata.X type: {type(adata.X).__name__}, dtype: {adata.X.dtype}, sparse: {issparse(adata.X)}")

    if adata.X.dtype != np.float32:
        if verbose:
            print(f"Converting adata.X from {adata.X.dtype} to float32")
        adata.X = adata.X.astype(np.float32) if issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_genes)

    if verbose:
        print(f"After initial filtering: {adata.shape[0]} cells × {adata.shape[1]} genes")

    mito_gene_mask = adata.var_names.str.startswith(("MT-", "mt-"))
    adata.var["mt"] = mito_gene_mask
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < pct_mito_cutoff].copy()

    if verbose:
        print(f"After mitochondrial filtering: {adata.shape[0]} cells × {adata.shape[1]} genes")

    mito_genes = adata.var_names[adata.var_names.str.startswith("MT-")]
    genes_to_exclude = set(mito_genes) | set(exclude_genes or [])
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()

    if verbose:
        print(f"After gene exclusion: {adata.shape[0]} cells × {adata.shape[1]} genes")

    cells_per_sample = adata.obs.groupby(sample_column).size()
    samples_to_keep = cells_per_sample[cells_per_sample >= min_cells].index
    adata = adata[adata.obs[sample_column].isin(samples_to_keep)].copy()

    if verbose:
        print(f"After sample filtering: {adata.shape[0]} cells × {adata.shape[1]} genes")
        print(f"Samples remaining: {len(samples_to_keep)}")

    min_cells_for_gene = int(0.001 * adata.n_obs)
    sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)

    if verbose:
        print(f"Final shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
        print("Preprocessing complete!")

    adata_cluster = adata.copy()
    adata_sample_diff = adata.copy()
    del adata

    adata_cluster = anndata_cluster(
        adata_cluster=adata_cluster,
        output_dir=output_dir,
        sample_column=sample_column,
        num_cell_hvgs=num_cell_hvgs,
        cell_embedding_num_PCs=cell_embedding_num_PCs,
        num_harmony_iterations=num_harmony_iterations,
        vars_to_regress_for_harmony=vars_to_regress_for_harmony,
        verbose=verbose,
    )

    adata_sample_diff = anndata_sample(
        adata_sample_diff=adata_sample_diff,
        output_dir=output_dir,
        sample_embedding_num_PCs=cell_embedding_num_PCs,
        verbose=verbose,
    )

    if verbose:
        print(f"Total runtime: {time.time() - start_time:.2f} seconds")

    return adata_cluster, adata_sample_diff