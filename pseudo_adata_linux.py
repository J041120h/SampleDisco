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

# GPU imports
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import the TF-IDF function - assuming it exists
from tf_idf import tfidf_memory_efficient


def sparse_to_torch_sparse(matrix, device='cuda'):
    """Convert scipy sparse matrix to PyTorch sparse tensor."""
    if not issparse(matrix):
        return torch.from_numpy(matrix).float().to(device)
    
    coo = matrix.tocoo()
    indices = torch.LongTensor(np.vstack((coo.row, coo.col))).to(device)
    values = torch.FloatTensor(coo.data).to(device)
    shape = torch.Size(coo.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32, device=device)


def torch_sparse_mean(sparse_tensor, dim=0, keepdim=False):
    """Compute mean of sparse tensor along dimension."""
    # Sum along dimension
    sum_result = torch.sparse.sum(sparse_tensor, dim=dim)
    
    # Count non-zero elements along dimension
    ones = torch.ones_like(sparse_tensor.values())
    indices = sparse_tensor.indices()
    shape = sparse_tensor.shape
    
    ones_tensor = torch.sparse_coo_tensor(indices, ones, shape, device=sparse_tensor.device)
    count = torch.sparse.sum(ones_tensor, dim=dim)
    
    # Avoid division by zero
    count = torch.where(count == 0, torch.ones_like(count), count)
    
    if sparse_tensor.is_sparse:
        # Convert to dense for division
        mean_result = sum_result.to_dense() / count.to_dense()
    else:
        mean_result = sum_result / count
    
    if not keepdim and mean_result.dim() > 1:
        mean_result = mean_result.squeeze(dim)
    
    return mean_result


def batch_process_gpu(X_gpu, indices_list, batch_size=1000, operation='mean'):
    """
    Process data in batches on GPU using PyTorch.
    
    Args:
        X_gpu: GPU tensor (can be sparse or dense)
        indices_list: List of index arrays for different groups
        batch_size: Number of groups to process simultaneously
        operation: 'mean' or 'sum'
    
    Returns:
        List of results for each group
    """
    device = X_gpu.device if hasattr(X_gpu, 'device') else 'cuda'
    results = []
    
    # Process in batches
    for i in range(0, len(indices_list), batch_size):
        batch_indices = indices_list[i:i + batch_size]
        batch_results = []
        
        for indices in batch_indices:
            if len(indices) == 0:
                # Empty result
                if X_gpu.is_sparse:
                    result = torch.zeros(X_gpu.shape[1], device=device)
                else:
                    result = torch.zeros(X_gpu.shape[1], device=device)
            else:
                # Convert indices to tensor
                idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                
                # Extract subset
                if X_gpu.is_sparse:
                    # For sparse tensors, we need to handle differently
                    subset = torch.index_select(X_gpu, 0, idx_tensor)
                    if operation == 'mean':
                        result = torch_sparse_mean(subset, dim=0)
                    else:  # sum
                        result = torch.sparse.sum(subset, dim=0).to_dense()
                else:
                    subset = X_gpu[idx_tensor]
                    if operation == 'mean':
                        result = torch.mean(subset, dim=0)
                    else:  # sum
                        result = torch.sum(subset, dim=0)
                
            batch_results.append(result)
        
        # Stack batch results
        if batch_results:
            batch_tensor = torch.stack(batch_results)
            results.append(batch_tensor.cpu().numpy())
    
    # Concatenate all results
    if results:
        return np.vstack(results)
    else:
        return np.array([])


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
    batch_size: int = 1000,
    use_mixed_precision: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, sc.AnnData]:
    """
    PyTorch GPU-accelerated compute pseudobulk expression with batching.
    
    Additional parameters:
        batch_size: Number of samples to process simultaneously
        use_mixed_precision: Use FP16 for faster computation
    """
    start_time = time.time() if verbose else None
    
    # Set up PyTorch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires GPU.")
    
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    if verbose:
        print(f"Using PyTorch with GPU: {torch.cuda.get_device_name(0)}")
        print(f"Batch size: {batch_size}, Mixed precision: {use_mixed_precision}")
    
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
    
    # Convert data to PyTorch tensor on GPU
    if verbose:
        print("Transferring data to GPU using PyTorch...")
    
    # Convert to appropriate format
    if issparse(adata.X):
        X_torch = sparse_to_torch_sparse(adata.X, device=device)
    else:
        X_torch = torch.from_numpy(adata.X).float().to(device)
    
    if use_mixed_precision and not X_torch.is_sparse:
        X_torch = X_torch.half()
    
    # Phase 1: Create pseudobulk AnnData with layers using PyTorch
    pseudobulk_adata = _create_pseudobulk_layers_torch(
        adata, X_torch, samples, cell_types, sample_col, celltype_col, 
        batch_col, batch_size, verbose
    )
    
    # Phase 2: Process each cell type layer with batching
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
            
            # Process with PyTorch for HVG selection
            hvg_genes, hvg_expr = _select_hvgs_torch(
                temp_adata, n_features, normalize, target_sum, atac,
                batch_col, batch_correction, device, batch_size, 
                use_mixed_precision, verbose
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
                
        except Exception as e:
            if verbose:
                print(f"  Failed to process {cell_type}: {e}")
            cell_types_to_remove.append(cell_type)
    
    # Remove failed cell types
    cell_types = [ct for ct in cell_types if ct not in cell_types_to_remove]
    
    # Phase 3: Create concatenated AnnData with PyTorch optimization
    concat_adata = _create_concat_adata_torch(
        all_hvg_data, all_gene_names, samples, n_features, 
        device, batch_size, verbose
    )
    
    # Compute cell proportions using PyTorch
    cell_proportion_df = _compute_cell_proportions_torch(
        adata, samples, cell_types, sample_col, celltype_col, 
        device, batch_size
    )
    
    # Create final expression matrix
    cell_expression_hvg_df = _create_final_expression_matrix(
        all_hvg_data, all_gene_names, samples, cell_types, verbose
    )
    
    # Clear GPU memory
    del X_torch
    torch.cuda.empty_cache()
    
    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
    
    return cell_expression_hvg_df, cell_proportion_df, concat_adata


def _create_pseudobulk_layers_torch(
    adata: sc.AnnData,
    X_torch: torch.Tensor,
    samples: list,
    cell_types: list,
    sample_col: str,
    celltype_col: str,
    batch_col: str,
    batch_size: int,
    verbose: bool
) -> sc.AnnData:
    """
    Create pseudobulk AnnData with cell type layers using PyTorch batching.
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
    X_main = np.zeros((n_samples, n_genes))
    pseudobulk_adata = sc.AnnData(X=X_main, obs=obs_df, var=var_df)
    
    # Pre-compute sample indices for efficiency
    sample_indices = {sample: idx for idx, sample in enumerate(samples)}
    
    # Add layers for each cell type
    for cell_type in cell_types:
        if verbose:
            print(f"Creating layer for {cell_type}")
        
        layer_matrix = np.zeros((n_samples, n_genes), dtype=np.float32)
        
        # Get all cells for this cell type
        ct_mask = adata.obs[celltype_col] == cell_type
        ct_indices = np.where(ct_mask)[0]
        
        if len(ct_indices) == 0:
            pseudobulk_adata.layers[cell_type] = layer_matrix
            continue
        
        # Group by sample and prepare batches
        ct_samples = adata.obs.loc[ct_mask, sample_col].values
        sample_groups = {}
        
        for idx, sample in zip(ct_indices, ct_samples):
            if sample not in sample_groups:
                sample_groups[sample] = []
            sample_groups[sample].append(idx)
        
        # Process samples in batches
        sample_list = list(sample_groups.keys())
        indices_list = [sample_groups[s] for s in sample_list]
        
        # Batch process
        results = batch_process_gpu(X_torch, indices_list, batch_size, operation='mean')
        
        # Fill in the layer matrix
        for i, sample in enumerate(sample_list):
            sample_idx = sample_indices[sample]
            layer_matrix[sample_idx, :] = results[i]
        
        pseudobulk_adata.layers[cell_type] = layer_matrix
    
    return pseudobulk_adata


def _select_hvgs_torch(
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
    verbose: bool
) -> Tuple[List[str], np.ndarray]:
    """
    Select highly variable genes using PyTorch for acceleration.
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
            # Use PyTorch for normalization
            X_torch = torch.from_numpy(temp_adata.X if not issparse(temp_adata.X) 
                                     else temp_adata.X.toarray()).float().to(device)
            
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    # Normalize total
                    row_sums = X_torch.sum(dim=1, keepdim=True)
                    row_sums[row_sums == 0] = 1  # Avoid division by zero
                    X_torch = X_torch * (target_sum / row_sums)
                    
                    # Log1p
                    X_torch = torch.log1p(X_torch)
            else:
                # Normalize total
                row_sums = X_torch.sum(dim=1, keepdim=True)
                row_sums[row_sums == 0] = 1
                X_torch = X_torch * (target_sum / row_sums)
                
                # Log1p
                X_torch = torch.log1p(X_torch)
            
            temp_adata.X = X_torch.cpu().numpy()
            del X_torch
    
    # Check for NaN values
    if issparse(temp_adata.X):
        nan_genes = np.array(np.isnan(temp_adata.X.toarray()).any(axis=0)).flatten()
    else:
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
                if issparse(temp_adata.X):
                    nan_genes_post = np.array(np.isnan(temp_adata.X.toarray()).any(axis=0)).flatten()
                else:
                    nan_genes_post = np.isnan(temp_adata.X).any(axis=0)
                
                if nan_genes_post.any():
                    temp_adata = temp_adata[:, ~nan_genes_post].copy()
            
            except Exception as e:
                if verbose:
                    print(f"  Combat failed: {str(e)}")
    
    # Select HVGs using PyTorch-accelerated variance calculation
    n_hvgs = min(n_features, temp_adata.n_vars)
    
    # Convert to PyTorch for variance calculation
    X_torch = torch.from_numpy(temp_adata.X if not issparse(temp_adata.X) 
                             else temp_adata.X.toarray()).float().to(device)
    
    # Calculate mean and variance in batches
    with torch.no_grad():
        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                mean = X_torch.mean(dim=0)
                variance = X_torch.var(dim=0)
        else:
            mean = X_torch.mean(dim=0)
            variance = X_torch.var(dim=0)
        
        # Calculate coefficient of variation or dispersion
        cv = torch.sqrt(variance) / (mean + 1e-6)
        
        # Get top variable genes
        _, top_indices = torch.topk(cv, k=n_hvgs)
        hvg_indices = top_indices.cpu().numpy()
    
    del X_torch
    
    # Extract HVG data
    hvg_genes = temp_adata.var.index[hvg_indices].tolist()
    hvg_expr = temp_adata[:, hvg_indices].X
    
    if issparse(hvg_expr):
        hvg_expr = hvg_expr.toarray()
    
    return hvg_genes, hvg_expr


def _create_concat_adata_torch(
    all_hvg_data: dict,
    all_gene_names: list,
    samples: list,
    n_features: int,
    device: torch.device,
    batch_size: int,
    verbose: bool
) -> sc.AnnData:
    """
    Create concatenated AnnData using PyTorch for efficiency.
    """
    if verbose:
        print("\nConcatenating all cell type HVGs into single AnnData")
    
    all_unique_genes = sorted(list(set(all_gene_names)))
    
    if len(all_unique_genes) == 0:
        raise ValueError("No HVGs found across all cell types")
    
    # Create matrix in batches on GPU
    n_samples = len(samples)
    n_genes = len(all_unique_genes)
    
    # Process in chunks to avoid memory issues
    concat_matrix = torch.zeros((n_samples, n_genes), dtype=torch.float32, device=device)
    
    # Create gene index mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_unique_genes)}
    
    # Fill matrix in batches
    for i in range(0, n_samples, batch_size):
        batch_samples = samples[i:i + batch_size]
        batch_data = []
        
        for sample in batch_samples:
            sample_vec = torch.zeros(n_genes, device=device)
            if sample in all_hvg_data:
                for gene, value in all_hvg_data[sample].items():
                    if gene in gene_to_idx:
                        sample_vec[gene_to_idx[gene]] = value
            batch_data.append(sample_vec)
        
        if batch_data:
            batch_tensor = torch.stack(batch_data)
            concat_matrix[i:i + len(batch_samples)] = batch_tensor
    
    # Convert to numpy
    concat_matrix_np = concat_matrix.cpu().numpy()
    del concat_matrix
    
    concat_adata = sc.AnnData(
        X=concat_matrix_np,
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=all_unique_genes)
    )
    
    # Apply final HVG selection
    if verbose:
        print(f"Applying final HVG selection on {concat_adata.n_vars} concatenated genes")
    
    sc.pp.filter_genes(concat_adata, min_cells=1)
    
    final_n_hvgs = min(n_features, concat_adata.n_vars)
    if final_n_hvgs < concat_adata.n_vars:
        # Use PyTorch for final HVG selection
        X_torch = torch.from_numpy(concat_adata.X).float().to(device)
        
        with torch.no_grad():
            variance = X_torch.var(dim=0)
            _, top_indices = torch.topk(variance, k=final_n_hvgs)
            hvg_indices = top_indices.cpu().numpy()
        
        del X_torch
        
        concat_adata = concat_adata[:, hvg_indices].copy()
    
    return concat_adata


def _compute_cell_proportions_torch(
    adata: sc.AnnData,
    samples: list,
    cell_types: list,
    sample_col: str,
    celltype_col: str,
    device: torch.device,
    batch_size: int
) -> pd.DataFrame:
    """Compute cell type proportions per sample using PyTorch."""
    
    proportion_df = pd.DataFrame(
        index=cell_types,
        columns=samples,
        dtype=float
    )
    
    # Create one-hot encodings for efficient computation
    sample_to_idx = {s: i for i, s in enumerate(samples)}
    celltype_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    
    # Create sample and cell type indices
    sample_indices = torch.tensor([sample_to_idx.get(s, -1) for s in adata.obs[sample_col]], 
                                 device=device)
    celltype_indices = torch.tensor([celltype_to_idx.get(ct, -1) for ct in adata.obs[celltype_col]], 
                                   device=device)
    
    # Remove invalid indices
    valid_mask = (sample_indices >= 0) & (celltype_indices >= 0)
    sample_indices = sample_indices[valid_mask]
    celltype_indices = celltype_indices[valid_mask]
    
    # Create sparse matrix for counting
    n_cells = len(sample_indices)
    indices = torch.stack([celltype_indices, sample_indices])
    values = torch.ones(n_cells, device=device)
    count_matrix = torch.sparse_coo_tensor(
        indices, values, (len(cell_types), len(samples)), 
        dtype=torch.float32, device=device
    ).to_dense()
    
    # Normalize to get proportions
    sample_totals = count_matrix.sum(dim=0, keepdim=True)
    sample_totals[sample_totals == 0] = 1  # Avoid division by zero
    proportions = count_matrix / sample_totals
    
    # Convert to DataFrame
    proportion_df.iloc[:, :] = proportions.cpu().numpy()
    
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
    use_pytorch: bool = True,
    batch_size: int = 1000,
    use_mixed_precision: bool = True
) -> Tuple[Dict, sc.AnnData]:
    """
    Optimized pseudobulk computation with choice of backend.
    
    Args:
        use_pytorch: If True, use PyTorch implementation; otherwise use rapids_singlecell
        batch_size: Batch size for PyTorch processing
        use_mixed_precision: Use FP16 for faster PyTorch computation
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
            use_mixed_precision=use_mixed_precision
        )
    else:
        # Fall back to rapids_singlecell implementation
        if verbose and use_pytorch:
            print("PyTorch requested but CUDA not available, falling back to rapids_singlecell")
        
        cell_expression_hvg_df, cell_proportion_df, final_adata = compute_pseudobulk_layers_gpu_rsc(
            adata=adata,
            batch_col=batch_col,
            sample_col=sample_col,
            celltype_col=celltype_col,
            output_dir=output_dir,
            n_features=n_features,
            normalize=normalize,
            target_sum=target_sum,
            atac=atac,
            verbose=verbose
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