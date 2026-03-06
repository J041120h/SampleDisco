import os
import time
import contextlib
import io
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
from muon import atac as ac
from harmony import harmonize
from scipy.sparse import issparse

from utils.safe_save import safe_h5ad_write
from utils.random_seed import set_global_seed
from utils.merge_sample_meta import merge_sample_metadata


def anndata_cluster(
    adata_cluster,
    output_dir,
    sample_column="sample",
    num_cell_hvfs=50000,
    cell_embedding_num_PCs=50,
    num_harmony_iterations=30,
    vars_to_regress_for_harmony=None,
    tfidf_scale_factor=1e4,
    log_transform=True,
    drop_first_lsi=True,
    verbose=True,
):
    """
    Process AnnData for clustering: TF-IDF, HVF selection, LSI, and Harmony integration.

    Parameters
    ----------
    adata_cluster : AnnData
        AnnData object to process for clustering.
    output_dir : str
        Directory to save output files.
    sample_column : str, default="sample"
        Column name identifying samples.
    num_cell_hvfs : int, default=50000
        Number of highly variable features to select.
    cell_embedding_num_PCs : int, default=50
        Number of LSI components.
    num_harmony_iterations : int, default=30
        Maximum Harmony iterations.
    vars_to_regress_for_harmony : list, optional
        Variables for Harmony batch correction.
    tfidf_scale_factor : float, default=1e4
        Scale factor for TF-IDF normalization.
    log_transform : bool, default=True
        Whether to apply log1p transformation after TF-IDF.
    drop_first_lsi : bool, default=True
        Whether to drop the first LSI component.
    verbose : bool, default=True
        Whether to print progress messages.

    Returns
    -------
    AnnData
        Processed AnnData object with LSI and Harmony embeddings.
    """
    if verbose:
        print("=== [CPU] Processing data for clustering ===")

    # TF-IDF normalization (ATAC-specific)
    if verbose:
        print("Running TF-IDF normalization...")
    ac.pp.tfidf(adata_cluster, scale_factor=tfidf_scale_factor)

    if log_transform:
        if verbose:
            print("Applying log1p transformation...")
        sc.pp.log1p(adata_cluster)

    # HVF selection
    if verbose:
        print("Running HVF selection...")
    sc.pp.highly_variable_genes(
        adata_cluster,
        n_top_genes=num_cell_hvfs,
        flavor="seurat_v3",
        batch_key=sample_column,
    )
    adata_cluster = adata_cluster[:, adata_cluster.var["highly_variable"]].copy()

    if verbose:
        print(f"After HVF selection: {adata_cluster.shape[0]} cells × {adata_cluster.shape[1]} features")

    # LSI dimensionality reduction (ATAC-specific, analogous to PCA for RNA)
    if verbose:
        print(f"Running LSI with {cell_embedding_num_PCs} components...")
    ac.tl.lsi(adata_cluster, n_comps=cell_embedding_num_PCs)

    if drop_first_lsi:
        if verbose:
            print("Dropping first LSI component...")
        adata_cluster.obsm["X_lsi"] = adata_cluster.obsm["X_lsi"][:, 1:]
        adata_cluster.varm["LSI"] = adata_cluster.varm["LSI"][:, 1:]
        adata_cluster.uns["lsi"]["stdev"] = adata_cluster.uns["lsi"]["stdev"][1:]

    # Harmony integration
    if verbose:
        print("=== [CPU] Running Harmony integration ===")
        print("Variables to regress:", ", ".join(vars_to_regress_for_harmony or []))

    if vars_to_regress_for_harmony:
        harmony_embeddings = harmonize(
            adata_cluster.obsm["X_lsi"],
            adata_cluster.obs,
            batch_key=vars_to_regress_for_harmony,
            max_iter_harmony=num_harmony_iterations,
            use_gpu=False,
        )
        adata_cluster.obsm["X_lsi_harmony"] = harmony_embeddings
    else:
        adata_cluster.obsm["X_lsi_harmony"] = adata_cluster.obsm["X_lsi"].copy()

    if verbose:
        print("Harmony integration complete.")

    save_path = os.path.join(output_dir, "adata_cell.h5ad")
    safe_h5ad_write(adata_cluster, save_path)

    return adata_cluster


def anndata_sample(
    adata_sample_diff,
    output_dir,
    sample_embedding_num_PCs=50,
    tfidf_scale_factor=1e4,
    log_transform=True,
    drop_first_lsi=True,
    verbose=True,
):
    """
    Process AnnData for sample-level differential analysis.

    Parameters
    ----------
    adata_sample_diff : AnnData
        AnnData object for sample-level analysis.
    output_dir : str
        Directory to save output files.
    sample_embedding_num_PCs : int, default=50
        Number of LSI components.
    tfidf_scale_factor : float, default=1e4
        Scale factor for TF-IDF normalization.
    log_transform : bool, default=True
        Whether to apply log1p transformation.
    drop_first_lsi : bool, default=True
        Whether to drop the first LSI component.
    verbose : bool, default=True
        Whether to print progress messages.

    Returns
    -------
    AnnData
        Processed AnnData object with LSI embeddings.
    """
    if verbose:
        print("=== [CPU] Processing data for sample differences ===")

    X_original_counts = adata_sample_diff.X.copy()

    # TF-IDF normalization (ATAC-specific)
    if verbose:
        print("Running TF-IDF normalization...")
    ac.pp.tfidf(adata_sample_diff, scale_factor=tfidf_scale_factor)

    if log_transform:
        if verbose:
            print("Applying log1p transformation...")
        sc.pp.log1p(adata_sample_diff)

    # LSI dimensionality reduction (ATAC-specific)
    if verbose:
        print(f"Running LSI with {sample_embedding_num_PCs} components...")
    ac.tl.lsi(adata_sample_diff, n_comps=sample_embedding_num_PCs)

    if drop_first_lsi:
        adata_sample_diff.obsm["X_lsi"] = adata_sample_diff.obsm["X_lsi"][:, 1:]
        adata_sample_diff.varm["LSI"] = adata_sample_diff.varm["LSI"][:, 1:]
        adata_sample_diff.uns["lsi"]["stdev"] = adata_sample_diff.uns["lsi"]["stdev"][1:]

    # Restore original counts
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
    cell_embedding_num_PCs=50,
    num_harmony_iterations=30,
    num_cell_hvfs=50000,
    min_cells=1,
    min_features=2000,
    max_features=15000,
    min_cells_per_sample=1,
    exclude_features=None,
    vars_to_regress=None,
    doublet_detection=True,
    tfidf_scale_factor=1e4,
    log_transform=True,
    drop_first_lsi=True,
    verbose=True,
):
    """
    End-to-end preprocessing pipeline for scATAC-seq data.

    Reads an input AnnData (H5AD), attaches cell/sample metadata, performs QC filtering
    (features/cells, doublet detection, excluded features, minimum cells per sample), then
    produces two outputs:
      (1) A clustering-ready AnnData with TF-IDF -> HVF -> LSI -> Harmony
      (2) A sample-difference AnnData with TF-IDF -> LSI while preserving original counts in X

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
    batch_key : str or list, optional
        Column name(s) for batch information.
    cell_embedding_num_PCs : int, default=50
        Number of LSI components for cell-level analysis.
    num_harmony_iterations : int, default=30
        Maximum number of Harmony iterations.
    num_cell_hvfs : int, default=50000
        Number of highly variable features to select.
    min_cells : int, default=1
        Minimum number of cells with a feature for it to be kept.
    min_features : int, default=2000
        Minimum number of features per cell.
    max_features : int, default=15000
        Maximum number of features per cell.
    min_cells_per_sample : int, default=1
        Minimum cells per sample to keep the sample.
    exclude_features : list, optional
        List of feature names to exclude from analysis.
    vars_to_regress : list, optional
        Variables to regress out during Harmony integration.
    doublet_detection : bool, default=True
        Whether to run doublet detection.
    tfidf_scale_factor : float, default=1e4
        Scale factor for TF-IDF normalization.
    log_transform : bool, default=True
        Whether to apply log1p transformation after TF-IDF.
    drop_first_lsi : bool, default=True
        Whether to drop the first LSI component.
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
        print(f"Raw shape: {adata.shape[0]} cells × {adata.shape[1]} features")

    # Handle cell metadata and sample column
    if cell_meta_path is None:
        _ensure_sample_column(adata, sample_column, verbose)
    else:
        if verbose:
            print(f"   📄 Merging cell-level metadata from: {cell_meta_path}")
        cell_metadata = pd.read_csv(cell_meta_path).set_index("barcode")
        adata.obs = adata.obs.join(cell_metadata, how="left")
        _ensure_sample_column(adata, sample_column, verbose)

    # Merge sample metadata
    if sample_meta_path is not None:
        if verbose:
            print("=== Merging sample-level metadata into adata.obs ===")
        adata = merge_sample_metadata(
            adata=adata,
            metadata_path=sample_meta_path,
            sample_column=sample_column,
            verbose=verbose,
        )

    # Process vars_to_regress
    flattened_vars_to_regress = _flatten_to_strings(vars_to_regress or [])
    vars_to_regress_for_harmony = flattened_vars_to_regress.copy()
    if sample_column not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append(sample_column)

    # Process batch_key
    flattened_batch_keys = _flatten_to_strings([batch_key] if batch_key else [])

    # Validate required columns
    required_columns = list(dict.fromkeys(flattened_vars_to_regress + flattened_batch_keys))
    if required_columns:
        missing_columns = sorted(set(required_columns) - set(adata.obs.columns.astype(str)))
        if missing_columns:
            raise KeyError(f"The following variables are missing from adata.obs: {missing_columns}")

    if verbose:
        print("All required columns are present in adata.obs.")

    # Ensure float32 dtype
    if verbose:
        print(f"adata.X type: {type(adata.X).__name__}, dtype: {adata.X.dtype}, sparse: {issparse(adata.X)}")

    if adata.X.dtype != np.float32:
        if verbose:
            print(f"Converting adata.X from {adata.X.dtype} to float32")
        adata.X = adata.X.astype(np.float32) if issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)

    # === QC Filtering (ATAC-specific) ===
    if verbose:
        print("=== QC filtering ===")

    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    # Filter features by minimum cells
    mu.pp.filter_var(adata, "n_cells_by_counts", lambda x: x >= min_cells)

    # Filter cells by feature count range (ATAC-specific: both min and max)
    mu.pp.filter_obs(adata, "n_genes_by_counts",
                     lambda x: (x >= min_features) & (x <= max_features))

    if verbose:
        print(f"After initial filtering: {adata.shape[0]} cells × {adata.shape[1]} features")

    # Doublet detection
    if doublet_detection:
        if adata.n_vars >= 50:
            try:
                if verbose:
                    print("Running doublet detection...")
                with contextlib.redirect_stdout(io.StringIO()):
                    n_prin = min(30, adata.n_vars - 1, adata.n_obs - 1)
                    sc.pp.scrublet(adata, batch_key=sample_column, n_prin_comps=n_prin)
                    n_doublets = adata.obs["predicted_doublet"].sum()
                    adata = adata[~adata.obs["predicted_doublet"]].copy()
                if verbose:
                    print(f"After doublet removal: {adata.shape[0]} cells (removed {n_doublets} doublets)")
            except (ValueError, RuntimeError) as e:
                if verbose:
                    print(f"Scrublet failed ({e}) – continuing without doublet removal.")

    # Exclude specified features
    if exclude_features:
        adata = adata[:, ~adata.var_names.isin(exclude_features)].copy()
        if verbose:
            print(f"After feature exclusion: {adata.shape[0]} cells × {adata.shape[1]} features")

    # Sample-level filtering
    cells_per_sample = adata.obs.groupby(sample_column).size()
    samples_to_keep = cells_per_sample[cells_per_sample >= min_cells_per_sample].index
    adata = adata[adata.obs[sample_column].isin(samples_to_keep)].copy()

    if verbose:
        print(f"After sample filtering: {adata.shape[0]} cells × {adata.shape[1]} features")
        print(f"Samples remaining: {len(samples_to_keep)}")

    # Final feature filtering (keep features in at least 0.1% of cells)
    min_cells_for_feature = int(0.001 * adata.n_obs)
    sc.pp.filter_genes(adata, min_cells=min_cells_for_feature)

    if verbose:
        print(f"Final shape: {adata.shape[0]} cells × {adata.shape[1]} features")
        print("QC filtering complete!")

    # Create copies for parallel processing
    adata_cluster = adata.copy()
    adata_sample_diff = adata.copy()
    del adata

    # Process for clustering
    adata_cluster = anndata_cluster(
        adata_cluster=adata_cluster,
        output_dir=output_dir,
        sample_column=sample_column,
        num_cell_hvfs=num_cell_hvfs,
        cell_embedding_num_PCs=cell_embedding_num_PCs,
        num_harmony_iterations=num_harmony_iterations,
        vars_to_regress_for_harmony=vars_to_regress_for_harmony,
        tfidf_scale_factor=tfidf_scale_factor,
        log_transform=log_transform,
        drop_first_lsi=drop_first_lsi,
        verbose=verbose,
    )

    # Process for sample-level analysis
    adata_sample_diff = anndata_sample(
        adata_sample_diff=adata_sample_diff,
        output_dir=output_dir,
        sample_embedding_num_PCs=cell_embedding_num_PCs,
        tfidf_scale_factor=tfidf_scale_factor,
        log_transform=log_transform,
        drop_first_lsi=drop_first_lsi,
        verbose=verbose,
    )

    if verbose:
        print(f"Total runtime: {time.time() - start_time:.2f} seconds")

    return adata_cluster, adata_sample_diff