import os
import time
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import rapids_singlecell as rsc
from scipy.sparse import issparse, csr_matrix
from typing import Tuple, Dict, List, Optional
import contextlib
import io
import gc

# GPU imports
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import the TF-IDF function - assuming it exists
from tf_idf import tfidf_memory_efficient


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0


def sparse_to_torch_sparse(matrix, device='cuda'):
    """Convert scipy sparse matrix to PyTorch sparse tensor."""
    if not issparse(matrix):
        return torch.from_numpy(matrix).float().to(device)
    
    coo = matrix.tocoo()
    indices = torch.LongTensor(np.vstack((coo.row, coo.col))).to(device)
    values = torch.FloatTensor(coo.data).to(device)
    shape = torch.Size(coo.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32, device=device)


def process_in_chunks(X, chunk_size=10000):
    """Process large matrix in chunks to avoid memory issues."""
    n_rows = X.shape[0]
    results = []
    
    for i in range(0, n_rows, chunk_size):
        end_idx = min(i + chunk_size, n_rows)
        if issparse(X):
            chunk = X[i:end_idx].toarray()
        else:
            chunk = X[i:end_idx]
        results.append(chunk)
    
    return np.vstack(results) if results else np.array([])


def batch_process_gpu_memory_efficient(X_gpu, indices_list, batch_size=100, operation='mean', 
                                      max_cells_per_batch=50000, verbose=False):
    """
    Process data in batches on GPU with memory-efficient approach.
    
    Args:
        X_gpu: GPU tensor or numpy array (will be processed in chunks)
        indices_list: List of index arrays for different groups
        batch_size: Number of groups to process simultaneously
        operation: 'mean' or 'sum'
        max_cells_per_batch: Maximum number of cells to process at once
        verbose: Print memory usage
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    
    # If X_gpu is numpy array, we'll load chunks to GPU as needed
    is_numpy = isinstance(X_gpu, np.ndarray) or issparse(X_gpu)
    
    for i in range(0, len(indices_list), batch_size):
        if verbose and i % 100 == 0:
            alloc, reserved = get_gpu_memory_info()
            print(f"  Processing batch {i}/{len(indices_list)}, GPU memory: {alloc:.2f}/{reserved:.2f} GB")
        
        batch_indices = indices_list[i:i + batch_size]
        batch_results = []
        
        # Collect all indices for this batch
        all_batch_indices = []
        group_boundaries = [0]
        
        for indices in batch_indices:
            if len(indices) > 0:
                all_batch_indices.extend(indices)
                group_boundaries.append(len(all_batch_indices))
        
        if len(all_batch_indices) == 0:
            # Empty batch - create zero results
            for _ in batch_indices:
                n_features = X_gpu.shape[1] if hasattr(X_gpu, 'shape') else X_gpu.size(1)
                batch_results.append(np.zeros(n_features))
            results.extend(batch_results)
            continue
        
        # Process in sub-chunks if too many cells
        unique_indices = sorted(list(set(all_batch_indices)))
        
        if len(unique_indices) > max_cells_per_batch:
            # Need to process in smaller chunks
            sub_results = {}
            
            for sub_start in range(0, len(unique_indices), max_cells_per_batch):
                sub_end = min(sub_start + max_cells_per_batch, len(unique_indices))
                sub_indices = unique_indices[sub_start:sub_end]
                
                # Load data to GPU
                if is_numpy:
                    if issparse(X_gpu):
                        sub_data = torch.from_numpy(X_gpu[sub_indices].toarray()).float().to(device)
                    else:
                        sub_data = torch.from_numpy(X_gpu[sub_indices]).float().to(device)
                else:
                    sub_data = X_gpu[torch.tensor(sub_indices, device=device)]
                
                # Store results
                for idx, data_idx in enumerate(sub_indices):
                    sub_results[data_idx] = sub_data[idx].cpu().numpy()
                
                # Free GPU memory
                del sub_data
                clear_gpu_memory()
            
            # Reconstruct results for each group
            for j, indices in enumerate(batch_indices):
                if len(indices) == 0:
                    n_features = X_gpu.shape[1]
                    batch_results.append(np.zeros(n_features))
                else:
                    group_data = np.stack([sub_results[idx] for idx in indices])
                    if operation == 'mean':
                        batch_results.append(np.mean(group_data, axis=0))
                    else:
                        batch_results.append(np.sum(group_data, axis=0))
            
            # Clean up
            del sub_results
            
        else:
            # Can process all at once
            if is_numpy:
                if issparse(X_gpu):
                    subset_data = torch.from_numpy(X_gpu[unique_indices].toarray()).float().to(device)
                else:
                    subset_data = torch.from_numpy(X_gpu[unique_indices]).float().to(device)
            else:
                idx_tensor = torch.tensor(unique_indices, dtype=torch.long, device=device)
                subset_data = X_gpu[idx_tensor]
            
            # Create mapping from original to subset indices
            idx_mapping = {orig: new for new, orig in enumerate(unique_indices)}
            
            # Process each group
            for j, indices in enumerate(batch_indices):
                if len(indices) == 0:
                    n_features = subset_data.shape[1]
                    batch_results.append(torch.zeros(n_features, device=device).cpu().numpy())
                else:
                    # Map to subset indices
                    mapped_indices = [idx_mapping[idx] for idx in indices]
                    group_data = subset_data[mapped_indices]
                    
                    if operation == 'mean':
                        result = torch.mean(group_data, dim=0)
                    else:
                        result = torch.sum(group_data, dim=0)
                    
                    batch_results.append(result.cpu().numpy())
            
            # Free GPU memory
            del subset_data
            if not is_numpy:
                del idx_tensor
            clear_gpu_memory()
        
        results.extend(batch_results)
    
    return np.array(results) if results else np.array([])


def compute_pseudobulk_layers_torch(
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
    use_mixed_precision: bool = False,  # Disabled by default to save memory
    max_cells_per_batch: int = 50000
) -> Tuple[pd.DataFrame, pd.DataFrame, sc.AnnData]:
    """
    PyTorch GPU-accelerated compute pseudobulk expression with aggressive memory management.
    
    Additional parameters:
        batch_size: Number of samples to process simultaneously (reduced default)
        use_mixed_precision: Use FP16 for faster computation (disabled by default)
        max_cells_per_batch: Maximum cells to load to GPU at once
    """
    start_time = time.time() if verbose else None
    
    # Set up PyTorch with memory optimization
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires GPU.")
    
    # Set memory allocation settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    device = torch.device('cuda')
    clear_gpu_memory()
    
    if verbose:
        print(f"Using PyTorch with GPU: {torch.cuda.get_device_name(0)}")
        print(f"Batch size: {batch_size}, Mixed precision: {use_mixed_precision}")
        alloc, reserved = get_gpu_memory_info()
        print(f"Initial GPU memory: {alloc:.2f}/{reserved:.2f} GB allocated/reserved")
    
    # Create output directory
    pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
    os.makedirs(pseudobulk_dir, exist_ok=True)
    
    # Check if batch correction should be applied
    batch_correction = (batch_col is not None and 
                       batch_col in adata.obs.columns and 
                       not adata.obs[batch_col].isnull().all())
    
    if batch_correction and adata.obs[batch_col].isnull().any():
        adata.obs[batch_col].fillna("Unknown", inplace=True)
    
    # Get unique samples and cell types
    samples = sorted(adata.obs[sample_col].unique())
    cell_types = sorted(adata.obs[celltype_col].unique())
    
    if verbose:
        print(f"Processing {len(cell_types)} cell types across {len(samples)} samples")
    
    # Phase 1: Create pseudobulk AnnData with layers
    # Keep data on CPU and process in chunks
    pseudobulk_adata = _create_pseudobulk_layers_torch_memory_efficient(
        adata, samples, cell_types, sample_col, celltype_col, 
        batch_col, batch_size, max_cells_per_batch, verbose
    )
    
    # Phase 2: Process each cell type layer
    all_hvg_data = {}
    all_gene_names = []
    cell_types_to_remove = []
    
    for ct_idx, cell_type in enumerate(cell_types):
        if verbose:
            print(f"\nProcessing cell type {ct_idx+1}/{len(cell_types)}: {cell_type}")
            alloc, reserved = get_gpu_memory_info()
            print(f"  GPU memory before processing: {alloc:.2f}/{reserved:.2f} GB")
        
        try:
            # Create temporary AnnData for this cell type
            layer_data = pseudobulk_adata.layers[cell_type]
            temp_adata = sc.AnnData(
                X=layer_data.copy(),
                obs=pseudobulk_adata.obs.copy(),
                var=pseudobulk_adata.var.copy()
            )
            
            # Process with PyTorch for HVG selection
            hvg_genes, hvg_expr = _select_hvgs_torch_memory_efficient(
                temp_adata, n_features, normalize, target_sum, atac,
                batch_col, batch_correction, device, batch_size, 
                use_mixed_precision, max_cells_per_batch, verbose
            )
            
            if len(hvg_genes) == 0:
                if verbose:
                    print(f"  No HVGs found for {cell_type}, skipping")
                cell_types_to_remove.append(cell_type)
                continue
            
            # Create prefixed gene names and store data
            prefixed_genes = [f"{cell_type} - {g}" for g in hvg_genes]
            all_gene_names.extend(prefixed_genes)
            
            # Store expression data
            for i, sample in enumerate(temp_adata.obs.index):
                if sample not in all_hvg_data:
                    all_hvg_data[sample] = {}
                
                for j, gene in enumerate(prefixed_genes):
                    all_hvg_data[sample][gene] = hvg_expr[i, j]
            
            if verbose:
                print(f"  Selected {len(hvg_genes)} HVGs")
            
            # Clean up
            del temp_adata
            del layer_data
            del hvg_expr
            gc.collect()
                
        except Exception as e:
            if verbose:
                print(f"  Failed to process {cell_type}: {e}")
            cell_types_to_remove.append(cell_type)
        
        # Clear GPU memory after each cell type
        clear_gpu_memory()
    
    # Remove failed cell types
    cell_types = [ct for ct in cell_types if ct not in cell_types_to_remove]
    
    # Clean up pseudobulk_adata layers to free memory
    del pseudobulk_adata
    gc.collect()
    clear_gpu_memory()
    
    # Phase 3: Create concatenated AnnData
    # Phase 3: Create concatenated AnnData
    concat_adata = _create_concat_adata_memory_efficient(
        all_hvg_data, all_gene_names, samples, n_features, 
        batch_size, verbose,
        original_adata=adata,  # Pass the original adata
        sample_col=sample_col,  # Pass the sample column name
        batch_col=batch_col  # Pass the batch column name
    )
    
    # Compute cell proportions
    cell_proportion_df = _compute_cell_proportions_memory_efficient(
        adata, samples, cell_types, sample_col, celltype_col, 
        batch_size, verbose
    )
    
    # Create final expression matrix
    cell_expression_hvg_df = _create_final_expression_matrix(
        all_hvg_data, all_gene_names, samples, cell_types, verbose
    )
    
    # Final cleanup
    clear_gpu_memory()
    
    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
        alloc, reserved = get_gpu_memory_info()
        print(f"Final GPU memory: {alloc:.2f}/{reserved:.2f} GB")
    
    return cell_expression_hvg_df, cell_proportion_df, concat_adata


def _create_pseudobulk_layers_torch_memory_efficient(
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
    """
    Create pseudobulk AnnData with cell type layers using memory-efficient processing.
    """
    
    # Create observation dataframe
    obs_df = pd.DataFrame(index=samples)
    obs_df.index.name = 'sample'
    
    if batch_col is not None and batch_col in adata.obs.columns:
        sample_batch_map = (
            adata.obs[[sample_col, batch_col]]
            .drop_duplicates()
            .set_index(sample_col)[batch_col]
            .to_dict()
        )
        obs_df[batch_col] = [sample_batch_map.get(s, 'Unknown') for s in samples]
    
    var_df = adata.var.copy()
    
    n_samples = len(samples)
    n_genes = adata.n_vars
    
    # Initialize pseudobulk AnnData
    X_main = np.zeros((n_samples, n_genes), dtype=np.float32)
    pseudobulk_adata = sc.AnnData(X=X_main, obs=obs_df, var=var_df)
    
    # Pre-compute sample indices for efficiency
    sample_indices = {sample: idx for idx, sample in enumerate(samples)}
    
    # Add layers for each cell type
    for ct_idx, cell_type in enumerate(cell_types):
        if verbose:
            print(f"Creating layer for cell type {ct_idx+1}/{len(cell_types)}: {cell_type}")
        
        layer_matrix = np.zeros((n_samples, n_genes), dtype=np.float32)
        
        # Get all cells for this cell type
        ct_mask = adata.obs[celltype_col] == cell_type
        ct_indices = np.where(ct_mask)[0]
        
        if len(ct_indices) == 0:
            pseudobulk_adata.layers[cell_type] = layer_matrix
            continue
        
        # Group by sample
        ct_samples = adata.obs.loc[ct_mask, sample_col].values
        sample_groups = {}
        
        for idx, sample in zip(ct_indices, ct_samples):
            if sample not in sample_groups:
                sample_groups[sample] = []
            sample_groups[sample].append(idx)
        
        # Process samples in batches
        sample_list = list(sample_groups.keys())
        indices_list = [sample_groups[s] for s in sample_list]
        
        # Batch process with memory efficiency
        results = batch_process_gpu_memory_efficient(
            adata.X, indices_list, batch_size, 
            operation='mean', max_cells_per_batch=max_cells_per_batch,
            verbose=False
        )
        
        # Fill in the layer matrix
        for i, sample in enumerate(sample_list):
            sample_idx = sample_indices[sample]
            layer_matrix[sample_idx, :] = results[i]
        
        pseudobulk_adata.layers[cell_type] = layer_matrix
        
        # Clean up
        del sample_groups
        del indices_list
        gc.collect()
    
    return pseudobulk_adata


def _select_hvgs_torch_memory_efficient(
    temp_adata: sc.AnnData,
    n_features: int,
    normalize: bool,
    target_sum: float,
    atac: bool,
    batch_col: str,
    batch_correction: bool,
    device: torch.device,
    batch_size: int,
    use_mixed_precision: bool,
    max_cells_per_batch: int,
    verbose: bool
) -> Tuple[List[str], np.ndarray]:
    """
    Select highly variable genes using PyTorch with memory-efficient approach.
    """
    
    # Filter out genes with zero expression
    gene_expr_sum = np.array(temp_adata.X.sum(axis=0)).flatten()
    expressed_mask = gene_expr_sum > 0
    temp_adata = temp_adata[:, expressed_mask].copy()
    
    if temp_adata.n_vars == 0:
        return [], np.array([])
    
    # Apply normalization
    if normalize:
        if atac:
            # TF-IDF normalization (requires CPU)
            tfidf_memory_efficient(temp_adata, scale_factor=target_sum)
        else:
            # Normalize in chunks to avoid memory issues
            n_obs = temp_adata.n_obs
            chunk_size = min(max_cells_per_batch, n_obs)
            
            if issparse(temp_adata.X):
                temp_adata.X = temp_adata.X.toarray()
            
            for start_idx in range(0, n_obs, chunk_size):
                end_idx = min(start_idx + chunk_size, n_obs)
                
                # Process chunk on GPU
                chunk = temp_adata.X[start_idx:end_idx]
                X_chunk = torch.from_numpy(chunk).float().to(device)
                
                # Normalize total
                row_sums = X_chunk.sum(dim=1, keepdim=True)
                row_sums[row_sums == 0] = 1
                X_chunk = X_chunk * (target_sum / row_sums)
                
                # Log1p
                X_chunk = torch.log1p(X_chunk)
                
                # Copy back
                temp_adata.X[start_idx:end_idx] = X_chunk.cpu().numpy()
                
                # Clean up
                del X_chunk
                clear_gpu_memory()
    
    # Check for NaN values
    nan_genes = np.isnan(temp_adata.X).any(axis=0)
    if nan_genes.any():
        temp_adata = temp_adata[:, ~nan_genes].copy()
    
    # Apply batch correction if needed
    if batch_correction and len(temp_adata.obs[batch_col].unique()) > 1:
        min_batch_size = temp_adata.obs[batch_col].value_counts().min()
        if min_batch_size >= 2:
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                     warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    sc.pp.combat(temp_adata, key=batch_col)
                
                # Remove any NaN values after Combat
                nan_genes_post = np.isnan(temp_adata.X).any(axis=0)
                if nan_genes_post.any():
                    temp_adata = temp_adata[:, ~nan_genes_post].copy()
            
            except Exception as e:
                if verbose:
                    print(f"  Combat failed: {str(e)}")
    
    # Select HVGs using chunked variance calculation
    n_hvgs = min(n_features, temp_adata.n_vars)
    
    # Calculate variance in chunks
    n_genes = temp_adata.n_vars
    variances = np.zeros(n_genes)
    means = np.zeros(n_genes)
    
    # First pass: calculate means
    for start_idx in range(0, temp_adata.n_obs, max_cells_per_batch):
        end_idx = min(start_idx + max_cells_per_batch, temp_adata.n_obs)
        chunk = temp_adata.X[start_idx:end_idx]
        
        X_chunk = torch.from_numpy(chunk).float().to(device)
        chunk_mean = X_chunk.mean(dim=0)
        
        # Incremental mean update
        weight = (end_idx - start_idx) / temp_adata.n_obs
        means += chunk_mean.cpu().numpy() * weight
        
        del X_chunk
        clear_gpu_memory()
    
    # Second pass: calculate variance
    for start_idx in range(0, temp_adata.n_obs, max_cells_per_batch):
        end_idx = min(start_idx + max_cells_per_batch, temp_adata.n_obs)
        chunk = temp_adata.X[start_idx:end_idx]
        
        X_chunk = torch.from_numpy(chunk).float().to(device)
        means_tensor = torch.from_numpy(means).float().to(device)
        
        # Calculate squared differences
        diff_squared = (X_chunk - means_tensor).pow(2)
        chunk_var_sum = diff_squared.sum(dim=0)
        
        # Add to total variance
        variances += chunk_var_sum.cpu().numpy()
        
        del X_chunk, means_tensor, diff_squared
        clear_gpu_memory()
    
    # Finalize variance calculation
    variances = variances / (temp_adata.n_obs - 1)
    
    # Get top variable genes
    hvg_indices = np.argsort(variances)[-n_hvgs:][::-1]
    
    # Extract HVG data
    hvg_genes = temp_adata.var.index[hvg_indices].tolist()
    hvg_expr = temp_adata[:, hvg_indices].X
    
    return hvg_genes, hvg_expr


def _create_concat_adata_memory_efficient(
    all_hvg_data: dict,
    all_gene_names: list,
    samples: list,
    n_features: int,
    batch_size: int,
    verbose: bool,
    original_adata: sc.AnnData = None,  # Add this parameter
    sample_col: str = 'sample',  # Add this parameter
    batch_col: str = None  # Add this parameter
) -> sc.AnnData:
    """
    Create concatenated AnnData with memory-efficient approach.
    """
    if verbose:
        print("\nConcatenating all cell type HVGs into single AnnData")
    
    all_unique_genes = sorted(list(set(all_gene_names)))
    
    if len(all_unique_genes) == 0:
        raise ValueError("No HVGs found across all cell types")
    
    # Create matrix on CPU
    n_samples = len(samples)
    n_genes = len(all_unique_genes)
    concat_matrix = np.zeros((n_samples, n_genes), dtype=np.float32)
    
    # Create gene index mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_unique_genes)}
    
    # Fill matrix
    for i, sample in enumerate(samples):
        if sample in all_hvg_data:
            for gene, value in all_hvg_data[sample].items():
                if gene in gene_to_idx:
                    concat_matrix[i, gene_to_idx[gene]] = value
    
    # Create obs dataframe with metadata
    obs_df = pd.DataFrame(index=samples)
    obs_df.index.name = 'sample'
    
    # Transfer sample metadata from original adata if provided
    if original_adata is not None:
        # Get unique sample metadata
        sample_metadata = {}
        for col in original_adata.obs.columns:
            if col != sample_col:  # Skip the sample column itself
                # Get unique values per sample
                sample_values = (original_adata.obs
                               .groupby(sample_col)[col]
                               .apply(lambda x: x.dropna().unique()))
                
                # If each sample has a unique value, it's sample-level metadata
                if sample_values.apply(lambda x: len(x) <= 1).all():
                    for sample in samples:
                        if sample in sample_values.index:
                            vals = sample_values[sample]
                            if len(vals) > 0:
                                if sample not in sample_metadata:
                                    sample_metadata[sample] = {}
                                sample_metadata[sample][col] = vals[0]
        
        # Add metadata to obs_df
        for sample in samples:
            if sample in sample_metadata:
                for col, value in sample_metadata[sample].items():
                    obs_df.loc[sample, col] = value
    
    concat_adata = sc.AnnData(
        X=concat_matrix,
        obs=obs_df,  # Now includes metadata
        var=pd.DataFrame(index=all_unique_genes)
    )
    
    # Apply final HVG selection
    if verbose:
        print(f"Applying final HVG selection on {concat_adata.n_vars} concatenated genes")
    
    sc.pp.filter_genes(concat_adata, min_cells=1)
    
    final_n_hvgs = min(n_features, concat_adata.n_vars)
    if final_n_hvgs < concat_adata.n_vars:
        # Calculate variance on CPU
        variances = np.var(concat_adata.X, axis=0)
        hvg_indices = np.argsort(variances)[-final_n_hvgs:][::-1]
        concat_adata = concat_adata[:, hvg_indices].copy()
    
    return concat_adata


def _compute_cell_proportions_memory_efficient(
    adata: sc.AnnData,
    samples: list,
    cell_types: list,
    sample_col: str,
    celltype_col: str,
    batch_size: int,
    verbose: bool
) -> pd.DataFrame:
    """Compute cell type proportions per sample with memory efficiency."""
    
    proportion_df = pd.DataFrame(
        index=cell_types,
        columns=samples,
        dtype=float
    )
    
    # Process in batches to avoid memory issues
    for batch_start in range(0, len(samples), batch_size):
        batch_end = min(batch_start + batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]
        
        for sample in batch_samples:
            sample_mask = adata.obs[sample_col] == sample
            total_cells = sample_mask.sum()
            
            if total_cells > 0:
                for cell_type in cell_types:
                    ct_count = ((adata.obs[sample_col] == sample) & 
                               (adata.obs[celltype_col] == cell_type)).sum()
                    proportion_df.loc[cell_type, sample] = ct_count / total_cells
            else:
                proportion_df.loc[:, sample] = 0.0
    
    return proportion_df


def _create_final_expression_matrix(
    all_hvg_data: dict,
    all_gene_names: list,
    samples: list,
    cell_types: list,
    verbose: bool
) -> pd.DataFrame:
    """Create final expression matrix in the expected format."""
    
    if verbose:
        print("\nCreating final HVG expression matrix")
    
    # Group genes by cell type
    cell_type_groups = {}
    for gene in set(all_gene_names):
        cell_type = gene.split(' - ')[0]
        if cell_type not in cell_type_groups:
            cell_type_groups[cell_type] = []
        cell_type_groups[cell_type].append(gene)
    
    for ct in cell_type_groups:
        cell_type_groups[ct].sort()
    
    # Create final format
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
                final_expression_df.at[cell_type, sample] = pd.Series(
                    values, index=gene_names
                )
            else:
                final_expression_df.at[cell_type, sample] = pd.Series(dtype=float)
    
    return final_expression_df


def _extract_sample_metadata(
    cell_adata: sc.AnnData,
    sample_adata: sc.AnnData,
    sample_col: str,
    exclude_cols: List[str] | None = None,
) -> sc.AnnData:
    """Detect and copy sample‑level metadata."""
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


# Main entry point - choose between implementations
def compute_pseudobulk_adata_optimized(
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
    use_pytorch: bool = True,
    batch_size: int = 100,
    use_mixed_precision: bool = False,
    max_cells_per_batch: int = 50000
) -> Tuple[Dict, sc.AnnData]:
    """
    Optimized pseudobulk computation with choice of backend and memory management.
    
    Args:
        use_pytorch: If True, use PyTorch implementation; otherwise use rapids_singlecell
        batch_size: Batch size for PyTorch processing (reduced for memory)
        use_mixed_precision: Use FP16 for faster PyTorch computation (disabled by default)
        max_cells_per_batch: Maximum cells to process at once on GPU
    """
    
    if use_pytorch and torch.cuda.is_available():
        # Use PyTorch implementation
        cell_expression_hvg_df, cell_proportion_df, final_adata = compute_pseudobulk_layers_torch(
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
            use_mixed_precision=use_mixed_precision,
            max_cells_per_batch=max_cells_per_batch
        )
    
    if Save:
        pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
        final_adata = _extract_sample_metadata(
            cell_adata=adata,
            sample_adata=final_adata,
            sample_col=sample_col,
        )
        
        sc.write(os.path.join(pseudobulk_dir, "pseudobulk_sample.h5ad"), final_adata)
    
    # Create backward-compatible dictionary
    pseudobulk = {
        "cell_expression_corrected": cell_expression_hvg_df,
        "cell_proportion": cell_proportion_df.T
    }
    
    return pseudobulk, final_adata


# Backward compatibility wrapper
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
    verbose: bool = False
) -> Tuple[Dict, sc.AnnData]:
    """
    GPU-accelerated backward compatibility wrapper.
    Uses PyTorch by default with memory-efficient settings.
    """
    
    return compute_pseudobulk_adata_optimized(
        adata=adata,
        batch_col=batch_col,
        sample_col=sample_col,
        celltype_col=celltype_col,
        output_dir=output_dir,
        Save=Save,
        n_features=n_features,
        normalize=normalize,
        target_sum=target_sum,
        atac=atac,
        verbose=verbose,
        use_pytorch=True,
        batch_size=100,  # Conservative batch size
        use_mixed_precision=False,  # Disabled for memory
        max_cells_per_batch=50000  # Limit cells per GPU batch
    )