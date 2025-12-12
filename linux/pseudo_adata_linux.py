import os
import gc
import signal
import io
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from typing import Tuple, Dict, List
import contextlib

# GPU imports
import torch

# Import the TF-IDF function (keep as in your original setup)
from tf_idf import tfidf_memory_efficient
from utils.random_seed import set_global_seed
from utils.limma import limma


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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
        if uniques_per_sample.apply(lambda u: len(u) <= 1).all():
            meta_dict[col] = uniques_per_sample.apply(lambda u: u[0] if len(u) else np.nan)

    if meta_dict:
        meta_df = pd.DataFrame(meta_dict)
        meta_df.index.name = "sample"
        sample_adata.obs = sample_adata.obs.join(meta_df, how="left")

    return sample_adata

def compute_pseudobulk_layers_gpu(
    adata: sc.AnnData,
    batch_col: str | List[str] | None = 'batch',
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
    combat_timeout: float = 20.0
) -> Tuple[pd.DataFrame, pd.DataFrame, sc.AnnData]:
    """
    GPU-accelerated pseudobulk computation matching CPU version output exactly.
    
    Parameters:
    -----------
    batch_col : str | List[str] | None
        Single batch column name, list of batch column names, or None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"[Pseudobulk] Using device: {device}")
    clear_gpu_memory()

    pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
    os.makedirs(pseudobulk_dir, exist_ok=True)

    # Convert batch_col to list for uniform handling
    if batch_col is None:
        batch_cols = []
    elif isinstance(batch_col, str):
        batch_cols = [batch_col] if batch_col else []
    else:
        batch_cols = list(batch_col)
    
    # Filter to only valid batch columns
    valid_batch_cols = []
    for bc in batch_cols:
        if bc in adata.obs.columns and not adata.obs[bc].isnull().all():
            valid_batch_cols.append(bc)
    
    batch_correction = len(valid_batch_cols) > 0
    
    # Fill NaN values in batch columns
    if batch_correction:
        for bc in valid_batch_cols:
            if adata.obs[bc].isnull().any():
                adata.obs[bc] = adata.obs[bc].fillna("Unknown")
    
    if verbose:
        if batch_correction:
            print(f"[Pseudobulk] Batch columns {valid_batch_cols} detected for correction")
        else:
            print(f"[Pseudobulk] No valid batch columns found; skipping batch correction")

    samples = sorted(adata.obs[sample_col].unique())
    cell_types = sorted(adata.obs[celltype_col].unique())

    if verbose:
        print(f"[Pseudobulk] Found {len(samples)} samples, {len(cell_types)} cell types")
        print(f"[Pseudobulk] Batch correction: {'enabled' if batch_correction else 'disabled'}")

    # Phase 1: Create pseudobulk AnnData with layers
    if verbose:
        print("[Pseudobulk] Phase 1: Creating pseudobulk layers...")
    pseudobulk_adata = _create_pseudobulk_layers_gpu(
        adata, samples, cell_types, sample_col, celltype_col,
        valid_batch_cols,verbose
    )

    # Phase 2: Process each cell type layer
    if verbose:
        print("[Pseudobulk] Phase 2: Processing cell type layers...")
    all_hvg_data = {}
    all_gene_names = []
    cell_types_to_remove = []

    for ct_idx, cell_type in enumerate(cell_types):
        if verbose:
            print(f"  Processing cell type {ct_idx + 1}/{len(cell_types)}: {cell_type}")
        try:
            layer_data = pseudobulk_adata.layers[cell_type]
            temp_adata = sc.AnnData(
                X=layer_data.copy(),
                obs=pseudobulk_adata.obs.copy(),
                var=pseudobulk_adata.var.copy()
            )

            sc.pp.filter_genes(temp_adata, min_cells=1)
            
            if temp_adata.n_vars == 0:
                if verbose:
                    print(f"    Skipping {cell_type}: no genes remaining after filtering")
                cell_types_to_remove.append(cell_type)
                continue

            if normalize:
                if atac:
                    tfidf_memory_efficient(temp_adata, scale_factor=target_sum)
                else:
                    _normalize_gpu(temp_adata, target_sum, device, max_cells_per_batch)

            if issparse(temp_adata.X):
                nan_genes = np.array(np.isnan(temp_adata.X.toarray()).any(axis=0)).flatten()
            else:
                nan_genes = np.isnan(temp_adata.X).any(axis=0)

            if nan_genes.any():
                temp_adata = temp_adata[:, ~nan_genes].copy()

            batch_correction_applied = False
            
            # Batch correction logic with multiple batch columns
            if batch_correction:
                # Create combined batch column
                combined_batch_col = '_combined_batch_'
                if len(valid_batch_cols) == 1:
                    temp_adata.obs[combined_batch_col] = temp_adata.obs[valid_batch_cols[0]]
                else:
                    # Combine multiple batch columns with separator
                    batch_values = temp_adata.obs[valid_batch_cols].astype(str)
                    temp_adata.obs[combined_batch_col] = batch_values.apply(
                        lambda row: '|'.join(row), axis=1
                    )
                
                batch_counts = temp_adata.obs[combined_batch_col].value_counts()
                n_batches = len(batch_counts)
                min_batch_size = batch_counts.min()
                max_batch_size = batch_counts.max()
                
                if verbose:
                    print(f"    Batch info: {n_batches} batches, sizes range {min_batch_size}-{max_batch_size}")
                
                if n_batches <= 1:
                    if verbose:
                        print(f"    Skipping batch correction: only {n_batches} batch(es) present")
                elif min_batch_size < 2:
                    small_batches = batch_counts[batch_counts < 2].to_dict()
                    if verbose:
                        print(f"    Skipping ComBat: batches with <2 samples: {small_batches}")
                    # Try limma directly since ComBat won't work
                    batch_correction_applied = _try_limma_correction(
                        temp_adata, combined_batch_col, verbose
                    )
                else:
                    # Try ComBat first
                    batch_correction_applied = _try_combat_correction(
                        temp_adata, combined_batch_col, combat_timeout, verbose
                    )
                    
                    # Fallback to limma if ComBat failed
                    if not batch_correction_applied:
                        batch_correction_applied = _try_limma_correction(
                            temp_adata, combined_batch_col, verbose
                        )
                
                # Clean up NaN genes after batch correction
                if batch_correction_applied:
                    if issparse(temp_adata.X):
                        nan_genes_post = np.array(
                            np.isnan(temp_adata.X.toarray()).any(axis=0)
                        ).flatten()
                    else:
                        nan_genes_post = np.isnan(temp_adata.X).any(axis=0)
                    
                    if nan_genes_post.any():
                        n_nan = nan_genes_post.sum()
                        if verbose:
                            print(f"    Removing {n_nan} genes with NaN after batch correction")
                        temp_adata = temp_adata[:, ~nan_genes_post].copy()

            n_hvgs = min(n_features, temp_adata.n_vars)
            sc.pp.highly_variable_genes(
                temp_adata,
                n_top_genes=n_hvgs,
                subset=False
            )

            hvg_mask = temp_adata.var['highly_variable']
            hvg_genes = temp_adata.var.index[hvg_mask].tolist()

            if len(hvg_genes) == 0:
                if verbose:
                    print(f"    Skipping {cell_type}: no HVGs found")
                cell_types_to_remove.append(cell_type)
                continue

            hvg_expr = temp_adata[:, hvg_mask].X
            if issparse(hvg_expr):
                hvg_expr = hvg_expr.toarray()

            prefixed_genes = [f"{cell_type} - {g}" for g in hvg_genes]
            all_gene_names.extend(prefixed_genes)

            for i, sample in enumerate(temp_adata.obs.index):
                if sample not in all_hvg_data:
                    all_hvg_data[sample] = {}
                for j, gene in enumerate(prefixed_genes):
                    all_hvg_data[sample][gene] = float(hvg_expr[i, j])

            if verbose:
                print(f"    Selected {len(hvg_genes)} HVGs")

        except Exception as e:
            if verbose:
                print(f"    Error processing {cell_type}: {type(e).__name__}: {e}")
            cell_types_to_remove.append(cell_type)

    cell_types = [ct for ct in cell_types if ct not in cell_types_to_remove]
    
    if verbose:
        print(f"[Pseudobulk] Retained {len(cell_types)} cell types after filtering")

    # Phase 3: Create concatenated AnnData with all cell type HVGs
    if verbose:
        print("[Pseudobulk] Phase 3: Creating concatenated expression matrix...")
    all_unique_genes = sorted(list(set(all_gene_names)))
    concat_matrix = np.zeros((len(samples), len(all_unique_genes)), dtype=float)

    for i, sample in enumerate(samples):
        for j, gene in enumerate(all_unique_genes):
            if sample in all_hvg_data and gene in all_hvg_data[sample]:
                concat_matrix[i, j] = all_hvg_data[sample][gene]

    concat_adata = sc.AnnData(
        X=concat_matrix,
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=all_unique_genes)
    )

    sc.pp.filter_genes(concat_adata, min_cells=1)
    final_n_hvgs = min(n_features, concat_adata.n_vars)
    sc.pp.highly_variable_genes(
        concat_adata,
        n_top_genes=final_n_hvgs,
        subset=True
    )

    if verbose:
        print(f"[Pseudobulk] Final feature count: {concat_adata.n_vars}")

    final_expression_df = _create_final_expression_matrix(
        all_hvg_data,
        concat_adata.var.index.tolist(),
        samples,
        cell_types,
        verbose
    )

    # Phase 4: Compute cell proportions
    if verbose:
        print("[Pseudobulk] Phase 4: Computing cell proportions...")
    cell_proportion_df = _compute_cell_proportions(
        adata, samples, cell_types, sample_col, celltype_col
    )

    concat_adata.uns['cell_proportions'] = cell_proportion_df

    clear_gpu_memory()

    if verbose:
        print("[Pseudobulk] Pseudobulk computation complete")

    return final_expression_df, cell_proportion_df, concat_adata

def _try_combat_correction(
    temp_adata: sc.AnnData,
    batch_col: str,
    combat_timeout: float,
    verbose: bool
) -> bool:
    """Attempt ComBat batch correction with timeout and error handling."""
    try:
        def timeout_handler(signum, frame):
            raise TimeoutError("ComBat timed out")

        old_handler = None
        if combat_timeout is not None:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(combat_timeout))

        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                sc.pp.combat(temp_adata, key=batch_col)
            
            if verbose:
                print(f"    Applied ComBat batch correction successfully")
            return True
            
        finally:
            if combat_timeout is not None:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

    except TimeoutError:
        if verbose:
            print(f"    ComBat failed: timed out after {combat_timeout}s")
        return False
    except ValueError as e:
        if verbose:
            print(f"    ComBat failed: ValueError: {e}")
        return False
    except np.linalg.LinAlgError as e:
        if verbose:
            print(f"    ComBat failed: LinAlgError (singular matrix): {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"    ComBat failed: {type(e).__name__}: {e}")
        return False


def _try_limma_correction(
    temp_adata: sc.AnnData,
    batch_col: str,
    verbose: bool
) -> bool:
    """Attempt limma batch correction with error handling."""
    try:
        if issparse(temp_adata.X):
            X_dense = temp_adata.X.toarray()
        else:
            X_dense = temp_adata.X.copy()
        
        covariate_formula = f'~ Q("{batch_col}")'

        X_corrected = limma(
            pheno=temp_adata.obs,
            exprs=X_dense,
            covariate_formula=covariate_formula,
            design_formula='1',
            rcond=1e-8,
            verbose=False
        )
    
        temp_adata.X = X_corrected
        if verbose:
            print(f"    Applied limma batch correction successfully")
        return True

    except np.linalg.LinAlgError as e:
        if verbose:
            print(f"    Limma failed: LinAlgError (singular matrix): {e}")
        return False
    except ValueError as e:
        if verbose:
            print(f"    Limma failed: ValueError: {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"    Limma failed: {type(e).__name__}: {e}")
        return False

def _create_pseudobulk_layers_gpu(
    adata: sc.AnnData,
    samples: list,
    cell_types: list,
    sample_col: str,
    celltype_col: str,
    batch_cols: List[str],
    verbose: bool
) -> sc.AnnData:
    """Optimized pseudobulk creation using sparse indicator matrix multiplication."""
    from scipy.sparse import csr_matrix
    
    n_samples = len(samples)
    n_celltypes = len(cell_types)
    n_genes = adata.n_vars
    
    if verbose:
        print(f"  Aggregating {adata.n_obs} cells into {n_samples} samples x {n_celltypes} cell types")
    
    sample_to_idx = {s: i for i, s in enumerate(samples)}
    ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    
    obs_df = pd.DataFrame(index=samples)
    obs_df.index.name = 'sample'
    
    # Handle multiple batch columns
    if batch_cols:
        for bc in batch_cols:
            sample_batch_map = (
                adata.obs[[sample_col, bc]]
                .drop_duplicates()
                .set_index(sample_col)[bc]
                .to_dict()
            )
            obs_df[bc] = [sample_batch_map.get(s, 'Unknown') for s in samples]
    
    var_df = adata.var.copy()
    
    X_main = np.zeros((n_samples, n_genes), dtype=np.float32)
    pseudobulk_adata = sc.AnnData(X=X_main, obs=obs_df, var=var_df)
    
    cell_sample_idx = adata.obs[sample_col].map(sample_to_idx).values
    cell_ct_idx = adata.obs[celltype_col].map(ct_to_idx).values
    
    valid_mask = ~(pd.isna(cell_sample_idx) | pd.isna(cell_ct_idx))
    cell_sample_idx = cell_sample_idx.astype(int)
    cell_ct_idx = cell_ct_idx.astype(int)
    
    if issparse(adata.X):
        X = adata.X.tocsr() if not isinstance(adata.X, csr_matrix) else adata.X
    else:
        X = adata.X
    
    for ct_idx, cell_type in enumerate(cell_types):
        ct_mask = (cell_ct_idx == ct_idx) & valid_mask
        ct_cell_indices = np.where(ct_mask)[0]
        
        if len(ct_cell_indices) == 0:
            pseudobulk_adata.layers[cell_type] = np.zeros((n_samples, n_genes), dtype=np.float32)
            continue
        
        ct_sample_idx = cell_sample_idx[ct_cell_indices]
        
        row_idx = ct_sample_idx
        col_idx = np.arange(len(ct_cell_indices))
        data = np.ones(len(ct_cell_indices), dtype=np.float32)
        
        indicator = csr_matrix(
            (data, (row_idx, col_idx)),
            shape=(n_samples, len(ct_cell_indices))
        )
        
        counts = np.array(indicator.sum(axis=1)).flatten()
        counts[counts == 0] = 1
        
        if issparse(X):
            X_ct = X[ct_cell_indices, :]
        else:
            X_ct = X[ct_cell_indices, :]
        
        if issparse(X_ct):
            sums = indicator @ X_ct
            sums = np.asarray(sums.todense())
        else:
            sums = indicator @ X_ct
        
        layer_matrix = sums / counts[:, np.newaxis]
        
        pseudobulk_adata.layers[cell_type] = layer_matrix.astype(np.float32)
    
    if verbose:
        print(f"  Created {len(cell_types)} cell type layers")
    
    return pseudobulk_adata

def _normalize_gpu(
    temp_adata: sc.AnnData,
    target_sum: float,
    device: torch.device,
    max_cells_per_batch: int
):
    """GPU-accelerated normalization matching scanpy's normalize_total and log1p."""
    n_obs = temp_adata.n_obs

    if issparse(temp_adata.X):
        temp_adata.X = temp_adata.X.toarray()

    for start_idx in range(0, n_obs, max_cells_per_batch):
        end_idx = min(start_idx + max_cells_per_batch, n_obs)
        chunk = np.asarray(temp_adata.X[start_idx:end_idx])
        X_chunk = torch.from_numpy(chunk).float().to(device)

        row_sums = X_chunk.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1
        X_chunk = X_chunk * (target_sum / row_sums)

        X_chunk = torch.log1p(X_chunk)

        temp_adata.X[start_idx:end_idx] = X_chunk.cpu().numpy()

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
    cell_type_groups = {}
    for gene in set(all_gene_names):
        cell_type = gene.split(' - ')[0]
        cell_type_groups.setdefault(cell_type, []).append(gene)

    for ct in cell_type_groups:
        cell_type_groups[ct].sort()

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
    combat_timeout: float = 20.0
) -> Tuple[Dict, sc.AnnData]:
    """
    Explicit GPU version with additional control parameters.
    """
    if verbose:
        print(f"[Pseudobulk] Starting pseudobulk computation...")
        print(f"[Pseudobulk] Input: {adata.n_obs} cells, {adata.n_vars} genes")
    
    set_global_seed(seed=42, verbose=verbose)

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

    if verbose:
        print("[Pseudobulk] Extracting sample metadata...")
    
    final_adata = _extract_sample_metadata(
        cell_adata=adata,
        sample_adata=final_adata,
        sample_col=sample_col,
    )

    final_adata.uns['cell_proportions'] = cell_proportion_df

    if Save:
        pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
        if verbose:
            print(f"[Pseudobulk] Saving to {pseudobulk_dir}/pseudobulk_sample.h5ad")
        sc.write(os.path.join(pseudobulk_dir, "pseudobulk_sample.h5ad"), final_adata)

    pseudobulk = {
        "cell_expression_corrected": cell_expression_hvg_df,
        "cell_proportion": cell_proportion_df.T
    }

    if verbose:
        print(f"[Pseudobulk] Done. Output: {final_adata.n_obs} samples, {final_adata.n_vars} features")

    return pseudobulk, final_adata