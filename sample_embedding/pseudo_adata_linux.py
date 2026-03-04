#!/usr/bin/env python3
"""
GPU-accelerated pseudobulk computation using rapids_singlecell.

Cleaner refactoring that leverages rapids_singlecell's native GPU functions
for normalization, log1p, and HVG selection.
"""

import os
import gc
import signal
import io
import warnings
import contextlib
from typing import Tuple, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse, csr_matrix

import cupy as cp
import rapids_singlecell as rsc

from tf_idf import tfidf_memory_efficient
from utils.random_seed import set_global_seed


def clear_gpu_memory():
    """Clear GPU memory (CuPy + PyTorch if available)."""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def _convert_to_list(input_value: Optional[Union[str, List[str]]]) -> List[str]:
    """Convert None/str/list to list."""
    if input_value is None:
        return []
    return [input_value] if isinstance(input_value, str) else list(input_value)


def _check_if_normalized_and_log_transformed(adata: sc.AnnData) -> bool:
    """
    Check if data appears to be already normalized and log-transformed.
    
    Heuristics:
    - Log-transformed data typically has max values < 20 (since log1p(10000) ~ 9.2)
    - Raw counts often have very large values (thousands or more)
    - Log-transformed data has a more compressed range
    """
    if adata.n_obs == 0 or adata.n_vars == 0:
        return False
    
    sample_size = min(1000, adata.n_obs)
    sample_indices = np.random.choice(adata.n_obs, sample_size, replace=False)
    
    sample_data = adata.X[sample_indices, :]
    if issparse(sample_data):
        sample_data = sample_data.toarray()
    
    return np.nanmax(sample_data) < 20


def _get_nan_gene_mask(expression_matrix) -> np.ndarray:
    """Get mask of genes with any NaN values."""
    if issparse(expression_matrix):
        expression_matrix = expression_matrix.toarray()
    return np.isnan(expression_matrix).any(axis=0)


def _extract_sample_metadata(
    cell_level_adata: sc.AnnData,
    sample_level_adata: sc.AnnData,
    sample_column: str,
    columns_to_exclude: Optional[List[str]] = None,
) -> sc.AnnData:
    """Extract sample-level metadata from cell-level obs."""
    excluded_columns = set(columns_to_exclude or []) | {sample_column}
    grouped_by_sample = cell_level_adata.obs.groupby(sample_column)

    sample_metadata = {}
    for column_name in cell_level_adata.obs.columns:
        if column_name in excluded_columns:
            continue
        unique_values_per_sample = grouped_by_sample[column_name].apply(lambda x: x.dropna().unique())
        if unique_values_per_sample.apply(lambda unique_vals: len(unique_vals) <= 1).all():
            sample_metadata[column_name] = unique_values_per_sample.apply(
                lambda unique_vals: unique_vals[0] if len(unique_vals) else np.nan
            )

    if sample_metadata:
        metadata_dataframe = pd.DataFrame(sample_metadata)
        overlapping_columns = sample_level_adata.obs.columns.intersection(metadata_dataframe.columns)
        if len(overlapping_columns) > 0:
            metadata_dataframe = metadata_dataframe.drop(columns=list(overlapping_columns))
        if metadata_dataframe.shape[1] > 0:
            sample_level_adata.obs = sample_level_adata.obs.join(metadata_dataframe, how="left")

    return sample_level_adata


def _attempt_combat_correction(
    adata: sc.AnnData,
    batch_column: str,
    columns_to_preserve: List[str],
    timeout_seconds: float,
    verbose: bool
) -> bool:
    """Attempt ComBat correction with timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutError("ComBat timed out")

    try:
        previous_handler = signal.signal(signal.SIGALRM, timeout_handler) if timeout_seconds else None
        if timeout_seconds:
            signal.alarm(int(timeout_seconds))

        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                sc.pp.combat(
                    adata,
                    key=batch_column,
                    covariates=columns_to_preserve or None,
                    inplace=True
                )
            if verbose:
                preserved_info = f" (kept={columns_to_preserve})" if columns_to_preserve else ""
                print(f"    Applied ComBat{preserved_info}")
            return True
        finally:
            if timeout_seconds:
                signal.alarm(0)
                if previous_handler:
                    signal.signal(signal.SIGALRM, previous_handler)

    except (TimeoutError, Exception) as error:
        if verbose:
            print(f"    ComBat failed: {type(error).__name__}: {error}")
        return False


def _attempt_limma_correction(
    adata: sc.AnnData,
    batch_column: str,
    columns_to_preserve: List[str],
    verbose: bool
) -> bool:
    """Attempt limma-style regression correction."""
    try:
        from utils.limma import limma

        expression_matrix = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)

        if columns_to_preserve:
            quoted_columns = [f'Q("{col}")' for col in columns_to_preserve]
            covariate_formula = "~ " + " + ".join(quoted_columns)
        else:
            covariate_formula = "1"

        adata.X = limma(
            pheno=adata.obs,
            exprs=expression_matrix,
            covariate_formula=covariate_formula,
            design_formula=f'~ Q("{batch_column}")',
            rcond=1e-8,
            verbose=False
        )

        if verbose:
            preserved_info = f" (kept={columns_to_preserve})" if columns_to_preserve else ""
            print(f"    Applied limma{preserved_info}")
        return True

    except Exception as error:
        if verbose:
            print(f"    Limma failed: {type(error).__name__}: {error}")
        return False


def apply_batch_correction(
    adata: sc.AnnData,
    batch_column: str,
    columns_to_preserve: List[str],
    combat_timeout_seconds: float,
    verbose: bool
) -> bool:
    """Apply batch correction, trying ComBat first, then limma."""
    samples_per_batch = adata.obs[batch_column].value_counts()
    num_batches = len(samples_per_batch)

    if verbose:
        print(f"    Batches: {num_batches}, sizes: {samples_per_batch.min()}-{samples_per_batch.max()}")

    if num_batches <= 1:
        if verbose:
            print("    Skipping: only 1 batch")
        return False

    if samples_per_batch.min() < 2:
        if verbose:
            print(f"    Skipping ComBat: batches <2 samples")
        return _attempt_limma_correction(adata, batch_column, columns_to_preserve, verbose)

    if _attempt_combat_correction(adata, batch_column, columns_to_preserve, combat_timeout_seconds, verbose):
        return True
    return _attempt_limma_correction(adata, batch_column, columns_to_preserve, verbose)


def aggregate_pseudobulk(
    adata: sc.AnnData,
    sample_column: str,
    celltype_column: str,
    batch_columns: List[str],
    verbose: bool
) -> sc.AnnData:
    """Aggregate cells into pseudobulk samples with cell-type layers."""
    unique_samples = sorted(adata.obs[sample_column].unique())
    unique_cell_types = sorted(adata.obs[celltype_column].unique())
    num_samples, num_genes = len(unique_samples), adata.n_vars

    if verbose:
        print(f"  Aggregating {adata.n_obs} cells -> {num_samples} samples x {len(unique_cell_types)} cell types")

    sample_to_index = {sample: idx for idx, sample in enumerate(unique_samples)}
    celltype_to_index = {celltype: idx for idx, celltype in enumerate(unique_cell_types)}

    sample_obs = pd.DataFrame(index=unique_samples)
    sample_obs.index.name = 'sample'
    for batch_col in batch_columns:
        sample_batch_mapping = adata.obs[[sample_column, batch_col]].drop_duplicates().set_index(sample_column)[batch_col].to_dict()
        sample_obs[batch_col] = [sample_batch_mapping.get(sample, 'Unknown') for sample in unique_samples]

    pseudobulk_adata = sc.AnnData(
        X=np.zeros((num_samples, num_genes), dtype=np.float32),
        obs=sample_obs,
        var=adata.var.copy()
    )

    cell_sample_indices = adata.obs[sample_column].map(sample_to_index).values
    cell_celltype_indices = adata.obs[celltype_column].map(celltype_to_index).values
    valid_cell_mask = ~(pd.isna(cell_sample_indices) | pd.isna(cell_celltype_indices))
    valid_cell_positions = np.where(valid_cell_mask)[0]
    valid_sample_indices = cell_sample_indices[valid_cell_mask].astype(int)
    valid_celltype_indices = cell_celltype_indices[valid_cell_mask].astype(int)

    expression_matrix = adata.X.tocsr() if issparse(adata.X) else adata.X

    for celltype_idx, celltype_name in enumerate(unique_cell_types):
        cells_of_this_type_mask = valid_celltype_indices == celltype_idx
        cell_positions_for_type = valid_cell_positions[cells_of_this_type_mask]
        sample_indices_for_type = valid_sample_indices[cells_of_this_type_mask]

        if len(cell_positions_for_type) == 0:
            pseudobulk_adata.layers[celltype_name] = np.zeros((num_samples, num_genes), dtype=np.float32)
            continue

        sample_cell_indicator = csr_matrix(
            (np.ones(len(cell_positions_for_type), dtype=np.float32),
             (sample_indices_for_type, np.arange(len(cell_positions_for_type)))),
            shape=(num_samples, len(cell_positions_for_type))
        )

        cells_per_sample = np.array(sample_cell_indicator.sum(axis=1)).flatten()
        cells_per_sample[cells_per_sample == 0] = 1

        expression_for_type = expression_matrix[cell_positions_for_type, :]
        summed_expression = sample_cell_indicator @ expression_for_type
        if issparse(summed_expression):
            summed_expression = np.asarray(summed_expression.todense())

        pseudobulk_adata.layers[celltype_name] = (summed_expression / cells_per_sample[:, None]).astype(np.float32)

    if verbose:
        print(f"  Created {len(unique_cell_types)} cell type layers")

    return pseudobulk_adata


def _remove_nan_genes(adata: sc.AnnData, verbose: bool, context: str = "") -> sc.AnnData:
    """Remove genes with NaN values from AnnData."""
    nan_gene_mask = _get_nan_gene_mask(adata.X)
    if nan_gene_mask.any():
        if verbose and context:
            print(f"    Removing {nan_gene_mask.sum()} NaN genes {context}")
        return adata[:, ~nan_gene_mask].copy()
    return adata


def process_celltype_layer(
    layer_expression: np.ndarray,
    sample_obs: pd.DataFrame,
    gene_var: pd.DataFrame,
    cell_type_name: str,
    batch_column: Optional[str],
    columns_to_preserve: List[str],
    num_features: int,
    should_normalize: bool,
    normalization_target_sum: float,
    is_atac_data: bool,
    combat_timeout_seconds: float,
    verbose: bool
) -> Optional[Tuple[List[str], np.ndarray]]:
    """Process a single cell-type layer on GPU."""
    temp_adata = sc.AnnData(X=layer_expression.copy(), obs=sample_obs.copy(), var=gene_var.copy())

    sc.pp.filter_genes(temp_adata, min_cells=1)
    if temp_adata.n_vars == 0:
        if verbose:
            print(f"    Skipping: no genes after filtering")
        return None

    if should_normalize:
        if is_atac_data:
            tfidf_memory_efficient(temp_adata, scale_factor=normalization_target_sum)
        else:
            rsc.get.anndata_to_GPU(temp_adata)
            rsc.pp.normalize_total(temp_adata, target_sum=normalization_target_sum)
            rsc.pp.log1p(temp_adata)
            rsc.get.anndata_to_CPU(temp_adata)

    temp_adata = _remove_nan_genes(temp_adata, verbose)

    if batch_column and batch_column in temp_adata.obs.columns:
        if apply_batch_correction(temp_adata, batch_column, columns_to_preserve, combat_timeout_seconds, verbose):
            temp_adata = _remove_nan_genes(temp_adata, verbose, "post-correction")

    num_hvgs_to_select = min(num_features, temp_adata.n_vars)
    rsc.get.anndata_to_GPU(temp_adata)
    rsc.pp.highly_variable_genes(temp_adata, n_top_genes=num_hvgs_to_select)
    rsc.get.anndata_to_CPU(temp_adata)

    hvg_mask = temp_adata.var["highly_variable"].values
    hvg_gene_names = temp_adata.var.index[hvg_mask].tolist()

    if not hvg_gene_names:
        if verbose:
            print(f"    Skipping: no HVGs found")
        return None

    hvg_expression = temp_adata[:, hvg_mask].X
    if issparse(hvg_expression):
        hvg_expression = hvg_expression.toarray()

    if verbose:
        print(f"    Selected {len(hvg_gene_names)} HVGs")

    return [f"{cell_type_name} - {gene}" for gene in hvg_gene_names], hvg_expression


def compute_cell_type_proportions(
    adata: sc.AnnData,
    sample_names: List[str],
    cell_type_names: List[str],
    sample_column: str,
    celltype_column: str
) -> pd.DataFrame:
    """Compute cell type proportions per sample."""
    proportion_dataframe = pd.DataFrame(index=cell_type_names, columns=sample_names, dtype=float)

    for sample_name in sample_names:
        sample_mask = adata.obs[sample_column] == sample_name
        total_cells_in_sample = sample_mask.sum()
        for cell_type_name in cell_type_names:
            cells_of_type_in_sample = (sample_mask & (adata.obs[celltype_column] == cell_type_name)).sum()
            proportion_dataframe.loc[cell_type_name, sample_name] = (
                cells_of_type_in_sample / total_cells_in_sample if total_cells_in_sample > 0 else 0.0
            )

    return proportion_dataframe


def _build_expression_dataframe(
    unique_cell_types: List[str],
    unique_samples: List[str],
    genes_grouped_by_celltype: Dict[str, List[str]],
    hvg_expression_by_sample: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Build the final expression dataframe with cell type rows and sample columns."""
    expression_dataframe = pd.DataFrame(index=sorted(unique_cell_types), columns=unique_samples, dtype=object)
    
    for cell_type_name in expression_dataframe.index:
        genes_for_celltype = genes_grouped_by_celltype.get(cell_type_name, [])
        for sample_name in expression_dataframe.columns:
            sample_data = hvg_expression_by_sample.get(sample_name, {})
            expression_values = []
            gene_names = []
            for gene_name in genes_for_celltype:
                if gene_name in sample_data:
                    expression_values.append(sample_data[gene_name])
                    gene_names.append(gene_name)
            expression_dataframe.at[cell_type_name, sample_name] = (
                pd.Series(expression_values, index=gene_names) if expression_values else pd.Series(dtype=float)
            )
    
    return expression_dataframe


def compute_pseudobulk_gpu(
    adata: sc.AnnData,
    batch_col: Optional[Union[str, List[str]]] = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './',
    n_features: int = 2000,
    normalize: bool = True,
    target_sum: float = 1e4,
    atac: bool = False,
    verbose: bool = False,
    combat_timeout: float = 20.0,
    preserve_cols: Optional[Union[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, sc.AnnData]:
    """GPU-accelerated pseudobulk computation."""
    if verbose:
        print(f"[Pseudobulk] Using GPU (rapids_singlecell)")
    clear_gpu_memory()

    os.makedirs(os.path.join(output_dir, "pseudobulk"), exist_ok=True)

    valid_batch_columns = [
        col for col in _convert_to_list(batch_col)
        if col and col in adata.obs.columns and not adata.obs[col].isnull().all()
    ]

    columns_to_preserve = [col for col in _convert_to_list(preserve_cols) if col in adata.obs.columns]

    for column_name in valid_batch_columns + columns_to_preserve:
        if adata.obs[column_name].isnull().any():
            adata.obs[column_name] = adata.obs[column_name].fillna("Unknown")

    unique_samples = sorted(adata.obs[sample_col].unique())
    unique_cell_types = sorted(adata.obs[celltype_col].unique())

    if verbose:
        print(f"[Pseudobulk] {len(unique_samples)} samples, {len(unique_cell_types)} cell types")
        print(f"[Pseudobulk] Batch correction: {valid_batch_columns if valid_batch_columns else 'disabled'}")
        print("[Pseudobulk] Phase 1: Aggregating...")

    pseudobulk_adata = aggregate_pseudobulk(adata, sample_col, celltype_col, valid_batch_columns, verbose)

    if columns_to_preserve:
        pseudobulk_adata = _extract_sample_metadata(
            cell_level_adata=adata,
            sample_level_adata=pseudobulk_adata,
            sample_column=sample_col,
            columns_to_exclude=None,
        )
        columns_to_preserve = [col for col in columns_to_preserve if col in pseudobulk_adata.obs.columns]

    combined_batch_column = None
    if valid_batch_columns:
        combined_batch_column = "_combined_batch_"
        if len(valid_batch_columns) == 1:
            pseudobulk_adata.obs[combined_batch_column] = pseudobulk_adata.obs[valid_batch_columns[0]].astype(str)
        else:
            pseudobulk_adata.obs[combined_batch_column] = pseudobulk_adata.obs[valid_batch_columns].astype(str).agg("|".join, axis=1)

    if verbose:
        print("[Pseudobulk] Phase 2: Processing cell types...")

    hvg_expression_by_sample: Dict[str, Dict[str, float]] = {}
    all_hvg_gene_names: List[str] = []
    successfully_processed_cell_types: List[str] = []

    for celltype_idx, cell_type_name in enumerate(unique_cell_types):
        if verbose:
            print(f"  [{celltype_idx+1}/{len(unique_cell_types)}] {cell_type_name}")

        try:
            processing_result = process_celltype_layer(
                pseudobulk_adata.layers[cell_type_name],
                pseudobulk_adata.obs,
                pseudobulk_adata.var,
                cell_type_name,
                combined_batch_column,
                columns_to_preserve,
                n_features,
                normalize,
                target_sum,
                atac,
                combat_timeout,
                verbose
            )

            if processing_result is None:
                continue

            prefixed_gene_names, hvg_expression_matrix = processing_result
            all_hvg_gene_names.extend(prefixed_gene_names)
            successfully_processed_cell_types.append(cell_type_name)

            for sample_idx, sample_name in enumerate(pseudobulk_adata.obs.index):
                hvg_expression_by_sample.setdefault(sample_name, {}).update(
                    {gene: float(hvg_expression_matrix[sample_idx, gene_idx])
                     for gene_idx, gene in enumerate(prefixed_gene_names)}
                )

            clear_gpu_memory()

        except Exception as error:
            if verbose:
                print(f"    Error: {type(error).__name__}: {error}")

    unique_cell_types = successfully_processed_cell_types
    if verbose:
        print(f"[Pseudobulk] Retained {len(unique_cell_types)} cell types")
        print("[Pseudobulk] Phase 3: Building final matrix...")

    unique_hvg_genes = sorted(set(all_hvg_gene_names))
    gene_name_to_index = {gene: idx for idx, gene in enumerate(unique_hvg_genes)}

    concatenated_expression_matrix = np.zeros((len(unique_samples), len(unique_hvg_genes)), dtype=np.float32)
    for sample_idx, sample_name in enumerate(unique_samples):
        if sample_name in hvg_expression_by_sample:
            for gene_name, expression_value in hvg_expression_by_sample[sample_name].items():
                if gene_name in gene_name_to_index:
                    concatenated_expression_matrix[sample_idx, gene_name_to_index[gene_name]] = expression_value

    concatenated_adata = sc.AnnData(
        X=concatenated_expression_matrix,
        obs=pd.DataFrame(index=unique_samples),
        var=pd.DataFrame(index=unique_hvg_genes)
    )

    sc.pp.filter_genes(concatenated_adata, min_cells=1)
    num_final_features = min(n_features, concatenated_adata.n_vars)
    rsc.get.anndata_to_GPU(concatenated_adata)
    rsc.pp.highly_variable_genes(concatenated_adata, n_top_genes=num_final_features)
    rsc.get.anndata_to_CPU(concatenated_adata)

    concatenated_adata = concatenated_adata[:, concatenated_adata.var["highly_variable"]].copy()

    if verbose:
        print(f"[Pseudobulk] Final features: {concatenated_adata.n_vars}")

    final_selected_genes = set(concatenated_adata.var.index)
    genes_grouped_by_celltype: Dict[str, List[str]] = {}
    for gene_name in final_selected_genes:
        cell_type_prefix = gene_name.split(' - ')[0]
        genes_grouped_by_celltype.setdefault(cell_type_prefix, []).append(gene_name)
    for cell_type_prefix in genes_grouped_by_celltype:
        genes_grouped_by_celltype[cell_type_prefix].sort()

    expression_dataframe = _build_expression_dataframe(
        unique_cell_types, unique_samples, genes_grouped_by_celltype, hvg_expression_by_sample
    )

    if verbose:
        print("[Pseudobulk] Phase 4: Computing proportions...")

    cell_type_proportions = compute_cell_type_proportions(
        adata, unique_samples, unique_cell_types, sample_col, celltype_col
    )
    concatenated_adata.uns["cell_proportions"] = cell_type_proportions

    clear_gpu_memory()

    if verbose:
        print("[Pseudobulk] Complete")

    return expression_dataframe, cell_type_proportions, concatenated_adata


def compute_pseudobulk_adata_linux(
    adata: sc.AnnData,
    batch_col: Optional[Union[str, List[str]]] = None,
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './',
    save: bool = True,
    sample_hvg_number: int = 2000,
    atac: bool = False,
    verbose: bool = False,
    preserve_covarient_in_sample_embedding: Optional[Union[str, List[str]]] = None,
) -> Tuple[Dict, sc.AnnData]:
    """Wrapper returning backward-compatible dict plus final AnnData."""
    if verbose:
        print(f"[Pseudobulk] Input: {adata.n_obs} cells, {adata.n_vars} genes")

    set_global_seed(seed=42)

    is_already_normalized = _check_if_normalized_and_log_transformed(adata)
    
    if verbose:
        if is_already_normalized:
            print("[Pseudobulk] Data appears already normalized/log-transformed, skipping normalization")
        else:
            print("[Pseudobulk] Data appears to be raw counts, will normalize after pseudobulk aggregation")

    expression_dataframe, cell_type_proportions, final_adata = compute_pseudobulk_gpu(
        adata=adata,
        batch_col=batch_col,
        sample_col=sample_col,
        celltype_col=celltype_col,
        output_dir=output_dir,
        n_features=sample_hvg_number,
        normalize=not is_already_normalized,
        target_sum=1e4,
        atac=atac,
        verbose=verbose,
        combat_timeout=20.0,
        preserve_cols=preserve_covarient_in_sample_embedding,
    )

    if verbose:
        print("[Pseudobulk] Extracting sample metadata...")

    final_adata = _extract_sample_metadata(adata, final_adata, sample_col)
    final_adata.uns["cell_proportions"] = cell_type_proportions

    if save:
        output_path = os.path.join(output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
        if verbose:
            print(f"[Pseudobulk] Saving to {output_path}")
        sc.write(output_path, final_adata)

    if verbose:
        print(f"[Pseudobulk] Output: {final_adata.n_obs} samples, {final_adata.n_vars} features")

    return {
        "cell_expression_corrected": expression_dataframe,
        "cell_proportion": cell_type_proportions.T
    }, final_adata