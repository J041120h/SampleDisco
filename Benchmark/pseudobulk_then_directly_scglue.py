import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
import gc
from typing import Optional

# GPU imports
import torch


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_naive_pseudobulk(
    adata_path: str,
    output_dir: str,
    hvg_file_path: Optional[str] = None,
    sample_col: str = 'sample',
    use_gpu: bool = True,
    max_cells_per_batch: int = 50000,
    verbose: bool = False,
    output_filename: Optional[str] = None
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
    max_cells_per_batch : int
        Maximum cells to process at once (for memory management)
    verbose : bool
        Print progress messages
    output_filename : str or None
        Custom filename for output. If None, uses 'naive_pseudobulk.h5ad'
        
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
    
    # Check GPU availability
    device = None
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        clear_gpu_memory()
        if verbose:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif use_gpu and verbose:
        print("GPU requested but not available, using CPU")
    
    # Get unique samples
    samples = sorted(adata.obs[sample_col].unique())
    n_samples = len(samples)
    n_genes = adata.n_vars
    
    if verbose:
        print(f"\nCreating naive pseudobulk for {n_samples} samples")
    
    # Initialize pseudobulk expression matrix
    pseudobulk_matrix = np.zeros((n_samples, n_genes))
    
    # Aggregate expression for each sample (sum, not mean - naive pseudobulk)
    for idx, sample in enumerate(samples):
        if verbose and idx % 10 == 0:
            print(f"  Processing sample {idx+1}/{n_samples}")
        
        # Get cells for this sample
        sample_mask = adata.obs[sample_col] == sample
        cell_indices = np.where(sample_mask)[0]
        
        if len(cell_indices) == 0:
            continue
        
        # Compute sum expression (naive pseudobulk)
        if use_gpu and device is not None and len(cell_indices) > 100:
            # Process in chunks for GPU
            expr_sum = np.zeros(n_genes)
            n_cells = len(cell_indices)
            
            for chunk_start in range(0, n_cells, max_cells_per_batch):
                chunk_end = min(chunk_start + max_cells_per_batch, n_cells)
                chunk_indices = cell_indices[chunk_start:chunk_end]
                
                # Get expression data
                if issparse(adata.X):
                    chunk_data = adata.X[chunk_indices, :].toarray()
                else:
                    chunk_data = adata.X[chunk_indices, :]
                
                # Move to GPU for sum calculation
                chunk_gpu = torch.from_numpy(chunk_data.astype(np.float32)).to(device)
                chunk_sum = chunk_gpu.sum(dim=0)
                expr_sum += chunk_sum.cpu().numpy()
                
                del chunk_gpu
                if use_gpu:
                    clear_gpu_memory()
            
            pseudobulk_matrix[idx, :] = expr_sum
        else:
            # CPU processing
            if issparse(adata.X):
                expr_values = np.asarray(
                    adata.X[cell_indices, :].sum(axis=0)
                ).flatten()
            else:
                expr_values = adata.X[cell_indices, :].sum(axis=0)
            
            pseudobulk_matrix[idx, :] = expr_values
    
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


# Example usage
if __name__ == "__main__":
    # Example call without HVG filtering
    pseudobulk = compute_naive_pseudobulk(
        adata_path="/path/to/your/data.h5ad",
        output_dir="/path/to/output/",
        sample_col='sample',
        use_gpu=True,
        verbose=True
    )
    
    # Example call with HVG filtering
    # pseudobulk_hvg = compute_naive_pseudobulk(
    #     adata_path="/dcl01/hongkai/data/data/hjiang/Data/count_data.h5ad",
    #     output_dir="/dcl01/hongkai/data/data/hjiang/Data/multi_omics_testing",
    #     hvg_file_path="/dcl01/hongkai/data/data/hjiang/result/pseudobulk/unique_gene_names.txt",
    #     sample_col='sample',
    #     use_gpu=True,
    #     verbose=True,
    #     output_filename='naive_pseudobulk_rna.h5ad'
    # )