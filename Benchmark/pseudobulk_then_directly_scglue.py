import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
import gc
from typing import Optional, List

# GPU imports
import torch


def get_gpu_memory_info():
    """Get GPU memory information."""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        free = total - allocated
        return {
            'total': total,
            'allocated': allocated, 
            'cached': cached,
            'free': free
        }
    return None


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def estimate_memory_needed(n_cells, n_genes, dtype=np.float32):
    """Estimate memory needed in GB."""
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = n_cells * n_genes * bytes_per_element
    return total_bytes / (1024**3)


def adaptive_batch_size(n_genes, available_memory_gb, safety_factor=0.7):
    """Calculate adaptive batch size based on available GPU memory."""
    bytes_per_element = 4  # float32
    available_bytes = available_memory_gb * (1024**3) * safety_factor
    max_cells = int(available_bytes / (n_genes * bytes_per_element))
    return max(1000, max_cells)  # Minimum 1000 cells


def compute_naive_pseudobulk(
    adata_path: str,
    output_dir: str,
    hvg_file_path: Optional[str] = None,
    sample_col: str = 'sample',
    use_gpu: bool = True,
    sample_batch_size: int = 10,
    cell_batch_size: Optional[int] = None,  # Now optional for adaptive sizing
    verbose: bool = False,
    output_filename: Optional[str] = None,
    memory_safety_factor: float = 0.6  # Use 60% of available GPU memory
) -> sc.AnnData:
    """
    Compute naive sample-level pseudobulk without normalization or batch correction.
    
    Parameters
    ----------
    adata_path : str
        Path to the AnnData file
    output_dir : str
        Directory to save the resulting pseudobulk AnnData
    hvg_file_path : str or None
        Path to txt file containing HVG gene names (one per line). 
        If provided, only these genes will be kept.
    sample_col : str
        Column name in obs containing sample IDs
    use_gpu : bool
        Whether to use GPU acceleration
    sample_batch_size : int
        Number of samples to process together in each batch
    cell_batch_size : int or None
        Maximum cells to process at once. If None, will be calculated adaptively.
    verbose : bool
        Print progress messages
    output_filename : str or None
        Custom filename for output. If None, uses 'naive_pseudobulk.h5ad'
    memory_safety_factor : float
        Fraction of available GPU memory to use (0.6 = 60%)
        
    Returns
    -------
    sc.AnnData
        Sample-level pseudobulk AnnData (raw counts, no normalization)
    """
    
    start_time = time.time() if verbose else None
    
    # Load the AnnData
    if verbose:
        print(f"Loading AnnData from {adata_path}")
    adata = sc.read_h5ad(adata_path)
    
    if verbose:
        print(f"Loaded data: {adata.shape[0]} cells × {adata.shape[1]} genes")
        print(f"Samples: {adata.obs[sample_col].nunique()}")
    
    # Filter to HVG if provided
    if hvg_file_path is not None:
        if verbose:
            print(f"Loading HVG list from {hvg_file_path}")
        
        # Read HVG list
        with open(hvg_file_path, 'r') as f:
            hvg_list = [line.strip() for line in f.readlines() if line.strip()]
        
        if verbose:
            print(f"Found {len(hvg_list)} genes in HVG file")
        
        # Filter genes to keep only HVGs
        gene_mask = adata.var.index.isin(hvg_list)
        genes_found = gene_mask.sum()
        
        if genes_found == 0:
            raise ValueError("No genes from HVG list found in the data")
        
        if verbose:
            print(f"Keeping {genes_found} HVG genes out of {len(hvg_list)} requested")
        
        adata = adata[:, gene_mask].copy()
    
    # Check GPU availability and setup adaptive batch sizing
    device = None
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        clear_gpu_memory()
        
        # Get GPU memory info
        gpu_info = get_gpu_memory_info()
        if verbose:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory - Total: {gpu_info['total']:.2f} GB, "
                  f"Free: {gpu_info['free']:.2f} GB")
        
        # Calculate adaptive batch size if not provided
        if cell_batch_size is None:
            cell_batch_size = adaptive_batch_size(
                adata.n_vars, 
                gpu_info['free'], 
                safety_factor=memory_safety_factor
            )
            if verbose:
                print(f"Adaptive cell batch size: {cell_batch_size}")
        else:
            # Check if provided batch size is feasible
            estimated_memory = estimate_memory_needed(cell_batch_size, adata.n_vars)
            if estimated_memory > gpu_info['free'] * memory_safety_factor:
                new_batch_size = adaptive_batch_size(
                    adata.n_vars, 
                    gpu_info['free'], 
                    safety_factor=memory_safety_factor
                )
                if verbose:
                    print(f"Warning: Requested batch size would need {estimated_memory:.2f} GB, "
                          f"reducing to {new_batch_size}")
                cell_batch_size = new_batch_size
                
    elif use_gpu and verbose:
        print("GPU requested but not available, using CPU")
        if cell_batch_size is None:
            cell_batch_size = 50000  # Default for CPU
    else:
        if cell_batch_size is None:
            cell_batch_size = 50000  # Default for CPU
    
    # Get unique samples
    samples = sorted(adata.obs[sample_col].unique())
    n_samples = len(samples)
    n_genes = adata.n_vars
    
    if verbose:
        print(f"\nCreating naive pseudobulk for {n_samples} samples")
        print(f"Sample batch size: {sample_batch_size}")
        print(f"Cell batch size: {cell_batch_size}")
    
    # Initialize pseudobulk expression matrix
    pseudobulk_matrix = np.zeros((n_samples, n_genes))
    
    # Process samples in batches
    n_batches = (n_samples + sample_batch_size - 1) // sample_batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * sample_batch_size
        end_idx = min(start_idx + sample_batch_size, n_samples)
        batch_samples = samples[start_idx:end_idx]
        
        if verbose:
            print(f"\nProcessing batch {batch_idx + 1}/{n_batches} "
                  f"(samples {start_idx + 1}-{end_idx})")
        
        # Process samples in current batch
        batch_results = _process_sample_batch(
            adata=adata,
            samples=batch_samples,
            sample_col=sample_col,
            device=device,
            cell_batch_size=cell_batch_size,
            verbose=verbose
        )
        
        # Store results
        for local_idx, sample_idx in enumerate(range(start_idx, end_idx)):
            pseudobulk_matrix[sample_idx, :] = batch_results[local_idx, :]
        
        # Clear memory after each batch
        del batch_results
        if use_gpu:
            clear_gpu_memory()
        else:
            gc.collect()
    
    # Create pseudobulk AnnData
    obs_df = pd.DataFrame(index=samples)
    obs_df.index.name = sample_col
    
    # Copy any sample-level metadata
    grouped = adata.obs.groupby(sample_col)
    exclude_cols = {sample_col}
    
    for col in adata.obs.columns:
        if col not in exclude_cols:
            uniques_per_sample = grouped[col].apply(lambda x: x.dropna().unique())
            # Keep if every sample has ≤1 unique value
            if uniques_per_sample.apply(lambda u: len(u) <= 1).all():
                obs_df[col] = uniques_per_sample.apply(
                    lambda u: u[0] if len(u) else np.nan
                )
    
    # Add cell count per sample
    cell_counts = adata.obs[sample_col].value_counts()
    obs_df['n_cells'] = [cell_counts.get(s, 0) for s in samples]
    
    # Create the pseudobulk AnnData
    pseudobulk_adata = sc.AnnData(
        X=pseudobulk_matrix,
        obs=obs_df,
        var=adata.var.copy()
    )
    
    if verbose:
        print(f"\nPseudobulk shape: {pseudobulk_adata.shape}")
        print(f"Cell counts per sample: min={obs_df['n_cells'].min()}, "
              f"max={obs_df['n_cells'].max()}, mean={obs_df['n_cells'].mean():.1f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the pseudobulk
    if output_filename is None:
        output_filename = 'naive_pseudobulk.h5ad'
    
    save_path = os.path.join(output_dir, output_filename)
    pseudobulk_adata.write(save_path)
    
    if verbose:
        print(f"\nSaved naive pseudobulk to {save_path}")
    
    # Clear GPU memory
    if use_gpu:
        clear_gpu_memory()
    
    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\nTotal runtime: {elapsed_time:.2f} seconds")
    
    return pseudobulk_adata


def _process_sample_batch(
    adata: sc.AnnData,
    samples: List[str],
    sample_col: str,
    device: Optional[torch.device],
    cell_batch_size: int,
    verbose: bool = False
) -> np.ndarray:
    """
    Process a batch of samples to compute their pseudobulk expressions.
    
    Parameters
    ----------
    adata : sc.AnnData
        The full AnnData object
    samples : List[str]
        List of sample IDs to process
    sample_col : str
        Column name for sample IDs
    device : torch.device or None
        GPU device if available
    cell_batch_size : int
        Maximum cells to process at once
    verbose : bool
        Print progress messages
        
    Returns
    -------
    np.ndarray
        Pseudobulk expression matrix for the batch (n_samples × n_genes)
    """
    n_batch_samples = len(samples)
    n_genes = adata.n_vars
    batch_matrix = np.zeros((n_batch_samples, n_genes))
    
    use_gpu = device is not None
    
    for local_idx, sample in enumerate(samples):
        if verbose and len(samples) > 5:
            print(f"    Processing sample {local_idx + 1}/{n_batch_samples}: {sample}")
        
        # Get cells for this sample
        sample_mask = adata.obs[sample_col] == sample
        cell_indices = np.where(sample_mask)[0]
        
        if len(cell_indices) == 0:
            continue
        
        # Compute sum expression (naive pseudobulk)
        if use_gpu and len(cell_indices) > 100:
            # GPU processing with cell batching
            expr_sum = np.zeros(n_genes, dtype=np.float64)  # Use float64 for accumulation
            n_cells = len(cell_indices)
            
            # Calculate actual chunk size based on memory constraints
            actual_chunk_size = min(cell_batch_size, n_cells)
            
            if verbose and n_cells > cell_batch_size:
                n_chunks = (n_cells + actual_chunk_size - 1) // actual_chunk_size
                print(f"      Sample has {n_cells} cells, processing in {n_chunks} chunks")
            
            for chunk_start in range(0, n_cells, actual_chunk_size):
                chunk_end = min(chunk_start + actual_chunk_size, n_cells)
                chunk_indices = cell_indices[chunk_start:chunk_end]
                
                try:
                    # Get expression data
                    if issparse(adata.X):
                        chunk_data = adata.X[chunk_indices, :].toarray()
                    else:
                        chunk_data = adata.X[chunk_indices, :]
                    
                    # Check memory requirement before GPU transfer
                    chunk_memory_gb = estimate_memory_needed(
                        chunk_data.shape[0], chunk_data.shape[1]
                    )
                    
                    if use_gpu:
                        gpu_info = get_gpu_memory_info()
                        if chunk_memory_gb > gpu_info['free'] * 0.8:  # 80% safety margin
                            if verbose:
                                print(f"      Chunk too large for GPU ({chunk_memory_gb:.2f} GB), "
                                      f"using CPU")
                            # Fall back to CPU for this chunk
                            chunk_sum = chunk_data.sum(axis=0)
                        else:
                            # Move to GPU for sum calculation
                            chunk_gpu = torch.from_numpy(chunk_data.astype(np.float32)).to(device)
                            chunk_sum = chunk_gpu.sum(dim=0).cpu().numpy()
                            del chunk_gpu
                            clear_gpu_memory()
                    else:
                        chunk_sum = chunk_data.sum(axis=0)
                    
                    expr_sum += chunk_sum.astype(np.float64)
                    
                    # Clean up
                    del chunk_data, chunk_sum
                    
                except torch.cuda.OutOfMemoryError as e:
                    if verbose:
                        print(f"      GPU OOM error, falling back to CPU for this chunk")
                    clear_gpu_memory()
                    
                    # Fall back to CPU
                    if issparse(adata.X):
                        chunk_data = adata.X[chunk_indices, :].toarray()
                    else:
                        chunk_data = adata.X[chunk_indices, :]
                    
                    chunk_sum = chunk_data.sum(axis=0)
                    expr_sum += chunk_sum.astype(np.float64)
                    
                    del chunk_data, chunk_sum
                
                # Force garbage collection after each chunk
                gc.collect()
            
            batch_matrix[local_idx, :] = expr_sum
        else:
            # CPU processing
            if issparse(adata.X):
                expr_values = np.asarray(
                    adata.X[cell_indices, :].sum(axis=0)
                ).flatten()
            else:
                expr_values = adata.X[cell_indices, :].sum(axis=0)
            
            batch_matrix[local_idx, :] = expr_values
    
    return batch_matrix


# Memory-efficient alternative using accumulator approach
def compute_naive_pseudobulk_memory_efficient(
    adata_path: str,
    output_dir: str,
    hvg_file_path: Optional[str] = None,
    sample_col: str = 'sample',
    use_gpu: bool = True,
    chunk_size: int = 10000,  # Much smaller chunks
    verbose: bool = False,
    output_filename: Optional[str] = None
) -> sc.AnnData:
    """
    Ultra memory-efficient version that processes very small chunks and accumulates results.
    """
    start_time = time.time() if verbose else None
    
    # Load data
    if verbose:
        print(f"Loading AnnData from {adata_path}")
    adata = sc.read_h5ad(adata_path)
    
    # Filter to HVG if provided
    if hvg_file_path is not None:
        with open(hvg_file_path, 'r') as f:
            hvg_list = [line.strip() for line in f.readlines() if line.strip()]
        gene_mask = adata.var.index.isin(hvg_list)
        adata = adata[:, gene_mask].copy()
        if verbose:
            print(f"Filtered to {gene_mask.sum()} HVG genes")
    
    # Setup
    samples = sorted(adata.obs[sample_col].unique())
    n_samples = len(samples)
    n_genes = adata.n_vars
    n_cells = adata.n_obs
    
    # Create sample mapping
    sample_to_idx = {sample: idx for idx, sample in enumerate(samples)}
    
    # Initialize result - use disk-backed array for very large datasets
    pseudobulk_matrix = np.zeros((n_samples, n_genes), dtype=np.float64)
    
    if verbose:
        print(f"Processing {n_cells} cells in chunks of {chunk_size}")
    
    # Process in small chunks
    n_chunks = (n_cells + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_cells)
        
        if verbose and chunk_idx % 100 == 0:
            print(f"Processing chunk {chunk_idx + 1}/{n_chunks}")
        
        # Get chunk data
        chunk_samples = adata.obs[sample_col].iloc[start_idx:end_idx]
        
        if issparse(adata.X):
            chunk_expr = adata.X[start_idx:end_idx, :].toarray()
        else:
            chunk_expr = adata.X[start_idx:end_idx, :]
        
        # Accumulate by sample
        for i, sample in enumerate(chunk_samples):
            sample_idx = sample_to_idx[sample]
            pseudobulk_matrix[sample_idx, :] += chunk_expr[i, :]
        
        # Clean up
        del chunk_expr
        if chunk_idx % 10 == 0:  # Clean up every 10 chunks
            gc.collect()
    
    # Create result AnnData
    obs_df = pd.DataFrame(index=samples)
    obs_df.index.name = sample_col
    
    # Add metadata
    grouped = adata.obs.groupby(sample_col)
    for col in adata.obs.columns:
        if col != sample_col:
            uniques_per_sample = grouped[col].apply(lambda x: x.dropna().unique())
            if uniques_per_sample.apply(lambda u: len(u) <= 1).all():
                obs_df[col] = uniques_per_sample.apply(
                    lambda u: u[0] if len(u) else np.nan
                )
    
    cell_counts = adata.obs[sample_col].value_counts()
    obs_df['n_cells'] = [cell_counts.get(s, 0) for s in samples]
    
    # Create AnnData
    pseudobulk_adata = sc.AnnData(
        X=pseudobulk_matrix,
        obs=obs_df,
        var=adata.var.copy()
    )
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    if output_filename is None:
        output_filename = 'naive_pseudobulk_memory_efficient.h5ad'
    
    save_path = os.path.join(output_dir, output_filename)
    pseudobulk_adata.write(save_path)
    
    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\nCompleted in {elapsed_time:.2f} seconds")
        print(f"Saved to {save_path}")
    
    return pseudobulk_adata

if __name__ == "__main__":
    pseudobulk = compute_naive_pseudobulk(
        adata_path="/dcl01/hongkai/data/data/hjiang/Data/paired/rna/all.h5ad",
        output_dir="/dcl01/hongkai/data/data/hjiang/Data/multi_omics_testing",
        hvg_file_path="/dcl01/hongkai/data/data/hjiang/Data/multi_omics_testing/unique_genes.txt",
        sample_col='sample',
        use_gpu=True,
        sample_batch_size=1,
        cell_batch_size=5000,  # Reduced from 50000 to 5000
        verbose=True
    )