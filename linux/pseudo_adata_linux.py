import os
import time
import gc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse, csr_matrix
from typing import Tuple, Dict, List
import contextlib

# GPU imports
import torch

# Import the TF-IDF function (keep as in your original setup)
from tf_idf import tfidf_memory_efficient
from utils.batch_regress import simple_batch_regression


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0


def _extract_sample_metadata(
    cell_adata: sc.AnnData,
    sample_adata: sc.AnnData,
    sample_col: str,
    exclude_cols: List[str] | None = None,
) -> sc.AnnData:
    """Detect and copy sample-level metadata (identical to CPU version)."""
    if exclude_cols is None:
        exclude_cols = []
    exclude_cols = set(exclude_cols) | {sample_col}

    grouped = cell_adata.obs.groupby(sample_col)
    meta_dict: Dict[str, pd.Series] = {}

    for col in cell_adata.obs.columns:
        if col in exclude_cols:
            continue
        uniques_per_sample = grouped[col].apply(lambda x: x.dropna().unique())
        # Keep if every sample shows ≤1 unique value
        if uniques_per_sample.apply(lambda u: len(u) <= 1).all():
            meta_dict[col] = uniques_per_sample.apply(lambda u: u[0] if len(u) else np.nan)

    if meta_dict:
        meta_df = pd.DataFrame(meta_dict)
        meta_df.index.name = "sample"
        sample_adata.obs = sample_adata.obs.join(meta_df, how="left")

    return sample_adata


def compute_pseudobulk_layers_gpu(
    adata: sc.AnnData,
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './',
    n_features: int = 2000,
    normalize: bool = True,
    target_sum: float = 1e4,
    atac: bool = False,
    verbose: bool = False,
    batch_size: int = 100,
    max_cells_per_batch: int = 50000,
    combat_timeout: float = 1800.0  # timeout for ComBat (seconds)
) -> Tuple[pd.DataFrame, pd.DataFrame, sc.AnnData]:
    """
    GPU-accelerated pseudobulk computation matching CPU version output exactly.

    Parameters
    ----------
    combat_timeout : float
        Maximum time in seconds to wait for ComBat before falling back to regression.
        Set to None to disable timeout.
    """
    start_time = time.time() if verbose else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clear_gpu_memory()

    if verbose and torch.cuda.is_available():
        print(f"Using PyTorch with GPU: {torch.cuda.get_device_name(0)}")
        alloc, reserved = get_gpu_memory_info()
        print(f"Initial GPU memory: {alloc:.2f}/{reserved:.2f} GB allocated/reserved")

    # Create output directory
    pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
    os.makedirs(pseudobulk_dir, exist_ok=True)

    # Check if batch correction should be applied
    batch_correction = (
        batch_col is not None and
        batch_col in adata.obs.columns and
        not adata.obs[batch_col].isnull().all()
    )
    if batch_correction and adata.obs[batch_col].isnull().any():
        adata.obs[batch_col] = adata.obs[batch_col].fillna("Unknown")

    # Get unique samples and cell types
    samples = sorted(adata.obs[sample_col].unique())
    cell_types = sorted(adata.obs[celltype_col].unique())

    if verbose:
        print(f"Processing {len(cell_types)} cell types across {len(samples)} samples")

    # Phase 1: Create pseudobulk AnnData with layers (matching CPU structure)
    pseudobulk_adata = _create_pseudobulk_layers_gpu(
        adata, samples, cell_types, sample_col, celltype_col,
        batch_col, batch_size, max_cells_per_batch, verbose
    )

    # Phase 2: Process each cell type layer (matching CPU logic exactly)
    all_hvg_data = {}
    all_gene_names = []
    cell_types_to_remove = []

    for cell_type in cell_types:
        if verbose:
            print(f"\nProcessing cell type: {cell_type}")

        try:
            # Create temporary AnnData for this cell type
            layer_data = pseudobulk_adata.layers[cell_type]
            temp_adata = sc.AnnData(
                X=layer_data.copy(),
                obs=pseudobulk_adata.obs.copy(),
                var=pseudobulk_adata.var.copy()
            )

            # Filter out genes with zero expression across all samples
            sc.pp.filter_genes(temp_adata, min_cells=1)

            if temp_adata.n_vars == 0:
                if verbose:
                    print(f"  No expressed genes for {cell_type}, skipping")
                cell_types_to_remove.append(cell_type)
                continue

            # Apply normalization (using GPU acceleration where possible)
            if normalize:
                if atac:
                    tfidf_memory_efficient(temp_adata, scale_factor=target_sum)
                else:
                    _normalize_gpu(temp_adata, target_sum, device, max_cells_per_batch)

            # Check for NaN values and filter
            if issparse(temp_adata.X):
                nan_genes = np.array(np.isnan(temp_adata.X.toarray()).any(axis=0)).flatten()
            else:
                nan_genes = np.isnan(temp_adata.X).any(axis=0)

            if nan_genes.any():
                if verbose:
                    print(f"  Found {nan_genes.sum()} genes with NaN values, removing them")
                temp_adata = temp_adata[:, ~nan_genes].copy()

            # Batch correction if needed
            if batch_correction and len(temp_adata.obs[batch_col].unique()) > 1:
                min_batch_size = temp_adata.obs[batch_col].value_counts().min()
                if min_batch_size >= 2:
                    combat_succeeded = False
                    try:
                        if verbose:
                            print(f"  Applying ComBat batch correction on {temp_adata.n_vars} genes")

                        import signal

                        def timeout_handler(signum, frame):
                            raise TimeoutError("ComBat timed out")

                        # Set up timeout if specified and platform supports it
                        old_handler = None
                        if combat_timeout is not None:
                            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(int(combat_timeout))

                        try:
                            # Suppress ComBat output
                            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                sc.pp.combat(temp_adata, key=batch_col)
                            combat_succeeded = True
                        finally:
                            # Disable the alarm and restore handler
                            if combat_timeout is not None:
                                signal.alarm(0)
                                if old_handler is not None:
                                    signal.signal(signal.SIGALRM, old_handler)

                        # Check for NaN values after ComBat
                        if combat_succeeded:
                            if issparse(temp_adata.X):
                                has_nan = np.any(np.isnan(temp_adata.X.data))
                                if has_nan:
                                    nan_genes_post = np.array(np.isnan(temp_adata.X.toarray()).any(axis=0)).flatten()
                                else:
                                    nan_genes_post = np.zeros(temp_adata.n_vars, dtype=bool)
                            else:
                                nan_genes_post = np.isnan(temp_adata.X).any(axis=0)
                                has_nan = nan_genes_post.any()

                            if has_nan:
                                if verbose:
                                    print(f"  Found {nan_genes_post.sum()} genes with NaN after ComBat, removing them")
                                temp_adata = temp_adata[:, ~nan_genes_post].copy()

                            if verbose:
                                print(f"  ComBat completed successfully, {temp_adata.n_vars} genes remaining")

                    except (TimeoutError, Exception) as e:
                        if verbose:
                            if isinstance(e, TimeoutError):
                                print(f"  ComBat timed out for {cell_type} after {combat_timeout}s")
                            else:
                                print(f"  ComBat failed for {cell_type}: {str(e)}")
                            print(f"  Falling back to regression-based batch correction")

                        # Fall back to simple regression-based batch correction (the provided helper)
                        temp_adata = simple_batch_regression(
                            temp_adata,
                            batch_col=batch_col,
                            verbose=verbose
                        )

                        # Check for NaN values after regression
                        if issparse(temp_adata.X):
                            nan_genes_post = np.array(np.isnan(temp_adata.X.toarray()).any(axis=0)).flatten()
                        else:
                            nan_genes_post = np.isnan(temp_adata.X).any(axis=0)

                        if nan_genes_post.any():
                            if verbose:
                                print(f"  Found {nan_genes_post.sum()} genes with NaN after regression, removing them")
                            temp_adata = temp_adata[:, ~nan_genes_post].copy()

            if verbose:
                print(f"  Selecting top {n_features} HVGs")

            # Ensure we don't request more genes than available
            n_hvgs = min(n_features, temp_adata.n_vars)

            # Use scanpy's HVG selection (matching CPU version)
            sc.pp.highly_variable_genes(
                temp_adata,
                n_top_genes=n_hvgs,
                subset=False
            )

            # Extract HVG data
            hvg_mask = temp_adata.var['highly_variable']
            hvg_genes = temp_adata.var.index[hvg_mask].tolist()

            if len(hvg_genes) == 0:
                if verbose:
                    print(f"  No HVGs found for {cell_type}, skipping")
                cell_types_to_remove.append(cell_type)
                continue

            # Extract expression values for HVGs
            hvg_expr = temp_adata[:, hvg_mask].X
            if issparse(hvg_expr):
                hvg_expr = hvg_expr.toarray()

            # Create prefixed gene names and store data (matching CPU format)
            prefixed_genes = [f"{cell_type} - {g}" for g in hvg_genes]
            all_gene_names.extend(prefixed_genes)

            # Store expression data
            for i, sample in enumerate(temp_adata.obs.index):
                if sample not in all_hvg_data:
                    all_hvg_data[sample] = {}
                for j, gene in enumerate(prefixed_genes):
                    all_hvg_data[sample][gene] = float(hvg_expr[i, j])

            if verbose:
                print(f"  Selected {len(hvg_genes)} HVGs")

        except Exception as e:
            if verbose:
                print(f"  Failed to process {cell_type}: {e}")
            cell_types_to_remove.append(cell_type)

    # Remove failed cell types from the list
    cell_types = [ct for ct in cell_types if ct not in cell_types_to_remove]

    # Phase 3: Create concatenated AnnData with all cell type HVGs
    if verbose:
        print("\nConcatenating all cell type HVGs into single AnnData")

    # Create sample x gene matrix from all HVG data
    all_unique_genes = sorted(list(set(all_gene_names)))
    concat_matrix = np.zeros((len(samples), len(all_unique_genes)), dtype=float)

    for i, sample in enumerate(samples):
        for j, gene in enumerate(all_unique_genes):
            if sample in all_hvg_data and gene in all_hvg_data[sample]:
                concat_matrix[i, j] = all_hvg_data[sample][gene]

    # Create concatenated AnnData
    concat_adata = sc.AnnData(
        X=concat_matrix,
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=all_unique_genes)
    )

    # Final HVG selection on concatenated data
    if verbose:
        print(f"Applying final HVG selection on {concat_adata.n_vars} concatenated genes")

    # Filter out genes with zero expression across all samples
    sc.pp.filter_genes(concat_adata, min_cells=1)

    # Apply final HVG selection
    final_n_hvgs = min(n_features, concat_adata.n_vars)
    sc.pp.highly_variable_genes(
        concat_adata,
        n_top_genes=final_n_hvgs,
        subset=True  # keep only final HVGs
    )

    if verbose:
        print(f"Final HVG selection: {concat_adata.n_vars} genes")

    # Create final expression DataFrame in original format
    final_expression_df = _create_final_expression_matrix(
        all_hvg_data,
        concat_adata.var.index.tolist(),  # Only final HVGs
        samples,
        cell_types,
        verbose
    )

    # Phase 4: Compute cell proportions
    cell_proportion_df = _compute_cell_proportions(
        adata, samples, cell_types, sample_col, celltype_col
    )

    # Add cell proportions to concat_adata
    concat_adata.uns['cell_proportions'] = cell_proportion_df

    # Save final outputs (matching CPU version exactly)
    final_expression_df.to_csv(
        os.path.join(pseudobulk_dir, "expression_hvg.csv")
    )
    cell_proportion_df.to_csv(
        os.path.join(pseudobulk_dir, "proportion.csv")
    )

    clear_gpu_memory()

    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\nTotal runtime: {elapsed_time:.2f} seconds")
        print(f"Final HVG matrix shape: {final_expression_df.shape}")
        print(f"Final AnnData shape: {concat_adata.shape}")

    return final_expression_df, cell_proportion_df, concat_adata


def _create_pseudobulk_layers_gpu(
    adata: sc.AnnData,
    samples: list,
    cell_types: list,
    sample_col: str,
    celltype_col: str,
    batch_col: str,
    batch_size: int,
    max_cells_per_batch: int,
    verbose: bool
) -> sc.AnnData:
    """Create pseudobulk AnnData with cell type layers using GPU acceleration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create new AnnData with samples as observations
    obs_df = pd.DataFrame(index=samples)
    obs_df.index.name = 'sample'

    # Add batch information if available
    if batch_col is not None and batch_col in adata.obs.columns:
        sample_batch_map = (
            adata.obs[[sample_col, batch_col]]
            .drop_duplicates()
            .set_index(sample_col)[batch_col]
            .to_dict()
        )
        obs_df[batch_col] = [sample_batch_map.get(s, 'Unknown') for s in samples]

    # Use all genes from original data
    var_df = adata.var.copy()

    # Initialize empty pseudobulk AnnData
    n_samples = len(samples)
    n_genes = adata.n_vars

    # Create zero matrix for main X
    X_main = np.zeros((n_samples, n_genes), dtype=float)
    pseudobulk_adata = sc.AnnData(X=X_main, obs=obs_df, var=var_df)

    # Add layers for each cell type
    for cell_type in cell_types:
        if verbose:
            print(f"Creating layer for {cell_type}")

        # Initialize layer matrix
        layer_matrix = np.zeros((n_samples, n_genes), dtype=float)

        # Compute pseudobulk expression for each sample
        for sample_idx, sample in enumerate(samples):
            # Get cells for this sample and cell type
            mask = (adata.obs[sample_col] == sample) & (adata.obs[celltype_col] == cell_type)
            cell_indices = np.where(mask)[0]

            if len(cell_indices) > 0:
                # Process in chunks for memory efficiency
                if len(cell_indices) > max_cells_per_batch:
                    expr_sum = np.zeros(n_genes, dtype=float)
                    for chunk_start in range(0, len(cell_indices), max_cells_per_batch):
                        chunk_end = min(chunk_start + max_cells_per_batch, len(cell_indices))
                        chunk_indices = cell_indices[chunk_start:chunk_end]

                        if issparse(adata.X):
                            chunk_data = adata.X[chunk_indices, :].toarray()
                        else:
                            chunk_data = adata.X[chunk_indices, :]

                        # Move to GPU for sum
                        chunk_gpu = torch.from_numpy(np.asarray(chunk_data)).float().to(device)
                        chunk_sum = chunk_gpu.sum(dim=0)
                        expr_sum += chunk_sum.cpu().numpy()

                        del chunk_gpu
                        clear_gpu_memory()

                    layer_matrix[sample_idx, :] = expr_sum / len(cell_indices)
                else:
                    # Process all at once
                    if issparse(adata.X):
                        expr_values = np.asarray(adata.X[cell_indices, :].mean(axis=0)).flatten()
                    else:
                        expr_values = np.asarray(adata.X[cell_indices, :].mean(axis=0)).flatten()
                    layer_matrix[sample_idx, :] = expr_values

        # Add as layer
        pseudobulk_adata.layers[cell_type] = layer_matrix

    return pseudobulk_adata


def _normalize_gpu(temp_adata: sc.AnnData, target_sum: float, device: torch.device,
                   max_cells_per_batch: int):
    """GPU-accelerated normalization matching scanpy's normalize_total and log1p."""
    n_obs = temp_adata.n_obs

    # Convert to dense if sparse
    if issparse(temp_adata.X):
        temp_adata.X = temp_adata.X.toarray()

    # Process in chunks
    for start_idx in range(0, n_obs, max_cells_per_batch):
        end_idx = min(start_idx + max_cells_per_batch, n_obs)
        chunk = np.asarray(temp_adata.X[start_idx:end_idx])
        X_chunk = torch.from_numpy(chunk).float().to(device)

        # Normalize total (matching scanpy)
        row_sums = X_chunk.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        X_chunk = X_chunk * (target_sum / row_sums)

        # Log1p transformation
        X_chunk = torch.log1p(X_chunk)

        # Copy back to CPU
        temp_adata.X[start_idx:end_idx] = X_chunk.cpu().numpy()

        # Clean up
        del X_chunk
        clear_gpu_memory()


def _create_final_expression_matrix(
    all_hvg_data: dict,
    all_gene_names: list,
    samples: list,
    cell_types: list,
    verbose: bool
) -> pd.DataFrame:
    """Create final expression matrix in original format (identical to CPU version)."""
    if verbose:
        print("\nCreating final HVG expression matrix")

    # Group genes by cell type
    cell_type_groups = {}
    for gene in set(all_gene_names):
        cell_type = gene.split(' - ')[0]
        if cell_type not in cell_type_groups:
            cell_type_groups[cell_type] = []
        cell_type_groups[cell_type].append(gene)

    # Sort genes within each cell type
    for ct in cell_type_groups:
        cell_type_groups[ct].sort()

    # Create final format (cell types as rows, samples as columns)
    final_expression_df = pd.DataFrame(
        index=sorted(cell_types),
        columns=samples,
        dtype=object
    )

    for cell_type in final_expression_df.index:
        ct_genes = cell_type_groups.get(cell_type, [])
        for sample in final_expression_df.columns:
            values = []
            gene_names = []
            for gene in ct_genes:
                if sample in all_hvg_data and gene in all_hvg_data[sample]:
                    values.append(all_hvg_data[sample][gene])
                    gene_names.append(gene)
            if values:
                final_expression_df.at[cell_type, sample] = pd.Series(values, index=gene_names)
            else:
                final_expression_df.at[cell_type, sample] = pd.Series(dtype=float)

    return final_expression_df


def _compute_cell_proportions(
    adata: sc.AnnData,
    samples: list,
    cell_types: list,
    sample_col: str,
    celltype_col: str
) -> pd.DataFrame:
    """Compute cell type proportions for each sample (identical to CPU version)."""
    proportion_df = pd.DataFrame(index=cell_types, columns=samples, dtype=float)

    for sample in samples:
        sample_mask = (adata.obs[sample_col] == sample)
        total_cells = int(sample_mask.sum())
        for cell_type in cell_types:
            ct_mask = sample_mask & (adata.obs[celltype_col] == cell_type)
            n_cells = int(ct_mask.sum())
            proportion = n_cells / total_cells if total_cells > 0 else 0.0
            proportion_df.loc[cell_type, sample] = proportion

    return proportion_df


def compute_pseudobulk_adata_linux(
    adata: sc.AnnData,
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './',
    Save: bool = True,
    n_features: int = 2000,
    normalize: bool = True,
    target_sum: float = 1e4,
    atac: bool = False,
    verbose: bool = False,
    batch_size: int = 100,
    max_cells_per_batch: int = 50000,
    combat_timeout: float = 1800.0
) -> Tuple[Dict, sc.AnnData]:
    """
    Explicit GPU version with additional control parameters.

    Additional Parameters
    ---------------------
    batch_size : int
        Number of samples to process simultaneously on GPU
    max_cells_per_batch : int
        Maximum number of cells to load to GPU at once
    combat_timeout : float
        Maximum time in seconds to wait for ComBat before falling back to regression
    """
    # Call the main GPU function
    from utils.random_seed import set_global_seed
    set_global_seed(seed = 42, verbose = verbose)
    cell_expression_hvg_df, cell_proportion_df, final_adata = compute_pseudobulk_layers_gpu(
        adata=adata,
        batch_col=batch_col,
        sample_col=sample_col,
        celltype_col=celltype_col,
        output_dir=output_dir,
        n_features=n_features,
        normalize=normalize,
        target_sum=target_sum,
        atac=atac,
        verbose=verbose,
        batch_size=batch_size,
        max_cells_per_batch=max_cells_per_batch,
        combat_timeout=combat_timeout
    )

    # Extract sample metadata (matching CPU version - do this ALWAYS)
    final_adata = _extract_sample_metadata(
        cell_adata=adata,
        sample_adata=final_adata,
        sample_col=sample_col,
    )

    # Add cell proportions (matching CPU version - do this ALWAYS)
    final_adata.uns['cell_proportions'] = cell_proportion_df

    # Save only if requested
    if Save:
        pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
        sc.write(os.path.join(pseudobulk_dir, "pseudobulk_sample.h5ad"), final_adata)

    # Create backward-compatible dictionary
    pseudobulk = {
        "cell_expression_corrected": cell_expression_hvg_df,
        "cell_proportion": cell_proportion_df.T
    }

    return pseudobulk, final_adata