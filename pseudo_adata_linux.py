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
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix

# Import the TF-IDF function - assuming it exists
from tf_idf import tfidf_memory_efficient


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


def ensure_cpu_arrays(adata):
    """
    Ensure all arrays in AnnData object are on CPU (not GPU).
    This prevents "Implicit conversion to NumPy array" errors.
    """
    # Convert main matrix
    if hasattr(adata.X, 'get'):
        adata.X = adata.X.get()
    
    # Convert layers
    if hasattr(adata, 'layers'):
        for key in list(adata.layers.keys()):
            if hasattr(adata.layers[key], 'get'):
                adata.layers[key] = adata.layers[key].get()
    
    # Convert obsm (embeddings)
    if hasattr(adata, 'obsm'):
        for key in list(adata.obsm.keys()):
            if hasattr(adata.obsm[key], 'get'):
                adata.obsm[key] = adata.obsm[key].get()
    
    # Convert varm
    if hasattr(adata, 'varm'):
        for key in list(adata.varm.keys()):
            if hasattr(adata.varm[key], 'get'):
                adata.varm[key] = adata.varm[key].get()
    
    # Convert obsp (pairwise arrays)
    if hasattr(adata, 'obsp'):
        for key in list(adata.obsp.keys()):
            if hasattr(adata.obsp[key], 'get'):
                adata.obsp[key] = adata.obsp[key].get()
    
    # Convert varp
    if hasattr(adata, 'varp'):
        for key in list(adata.varp.keys()):
            if hasattr(adata.varp[key], 'get'):
                adata.varp[key] = adata.varp[key].get()
    
    return adata


def compute_pseudobulk_layers_gpu_rsc(
    adata: sc.AnnData,
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './',
    n_features: int = 2000,
    normalize: bool = True,
    target_sum: float = 1e4,
    atac: bool = False,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, sc.AnnData]:
    """
    GPU-accelerated compute pseudobulk expression using rapids_singlecell.
    
    This version uses rsc functions instead of custom GPU implementations.
    """
    start_time = time.time() if verbose else None
    
    if verbose:
        print(f"Using GPU with rapids_singlecell")
        if cp.cuda.is_available():
            print(f"GPU Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    
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
    
    # Transfer main adata to GPU once
    if verbose:
        print("Transferring data to GPU...")
    rsc.get.anndata_to_GPU(adata)
    
    # Phase 1: Create pseudobulk AnnData with layers
    pseudobulk_adata = _create_pseudobulk_layers_rsc(
        adata, samples, cell_types, sample_col, celltype_col, batch_col, verbose
    )
    
    # Phase 2: Process each cell type layer
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
            
            # Transfer to GPU for processing
            rsc.get.anndata_to_GPU(temp_adata)
            
            # Filter out genes with zero expression using rsc
            rsc.pp.filter_genes(temp_adata, min_cells=1)
            
            if temp_adata.n_vars == 0:
                if verbose:
                    print(f"  No expressed genes for {cell_type}, skipping")
                cell_types_to_remove.append(cell_type)
                continue
            
            # Apply normalization using rsc
            if normalize:
                if atac:
                    # For ATAC, we need to transfer back to CPU for TF-IDF
                    rsc.get.anndata_to_CPU(temp_adata)
                    tfidf_memory_efficient(temp_adata, scale_factor=target_sum)
                    rsc.get.anndata_to_GPU(temp_adata)
                else:
                    # Use rsc normalization functions
                    rsc.pp.normalize_total(temp_adata, target_sum=target_sum)
                    rsc.pp.log1p(temp_adata)
            
            # Transfer to CPU for NaN checking and batch correction
            rsc.get.anndata_to_CPU(temp_adata)
            
            # Check for NaN values
            if issparse(temp_adata.X):
                nan_genes = np.array(np.isnan(temp_adata.X.toarray()).any(axis=0)).flatten()
            else:
                nan_genes = np.isnan(temp_adata.X).any(axis=0)
            
            if nan_genes.any():
                if verbose:
                    print(f"  Found {nan_genes.sum()} genes with NaN values, removing them")
                temp_adata = temp_adata[:, ~nan_genes].copy()
            
            # Apply batch correction if needed (Combat requires CPU)
            if batch_correction and len(temp_adata.obs[batch_col].unique()) > 1:
                min_batch_size = temp_adata.obs[batch_col].value_counts().min()
                if min_batch_size >= 2:
                    try:
                        if verbose:
                            print(f"  Applying batch correction on {temp_adata.n_vars} genes")
                        
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
                            if verbose:
                                print(f"  Found {nan_genes_post.sum()} genes with NaN after Combat, removing them")
                            temp_adata = temp_adata[:, ~nan_genes_post].copy()
                        
                        if verbose:
                            print(f"  Combat completed successfully, {temp_adata.n_vars} genes remaining")
                    
                    except Exception as e:
                        if verbose:
                            print(f"  Combat failed for {cell_type}: {str(e)}")
                            print(f"  Proceeding without batch correction for this cell type")
            
            # Transfer back to GPU for HVG selection
            rsc.get.anndata_to_GPU(temp_adata)
            
            # Select highly variable genes using rsc
            if verbose:
                print(f"  Selecting top {n_features} HVGs using rapids_singlecell")
            
            n_hvgs = min(n_features, temp_adata.n_vars)
            
            # Use rsc's highly_variable_genes function
            rsc.pp.highly_variable_genes(
                temp_adata,
                n_top_genes=n_hvgs,
                subset=False,  # Don't subset yet, just mark
                flavor='seurat_v3' if normalize else 'seurat'
            )
            
            # Transfer back to CPU for data extraction
            rsc.get.anndata_to_CPU(temp_adata)
            
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
    
    # Phase 3: Create concatenated AnnData
    if verbose:
        print("\nConcatenating all cell type HVGs into single AnnData")
    
    all_unique_genes = sorted(list(set(all_gene_names)))
    
    if len(all_unique_genes) == 0:
        raise ValueError("No HVGs found across all cell types")
    
    concat_matrix = np.zeros((len(samples), len(all_unique_genes)))
    
    for i, sample in enumerate(samples):
        for j, gene in enumerate(all_unique_genes):
            if sample in all_hvg_data and gene in all_hvg_data[sample]:
                concat_matrix[i, j] = all_hvg_data[sample][gene]
    
    concat_adata = sc.AnnData(
        X=concat_matrix,
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=all_unique_genes)
    )
    
    # Apply final HVG selection using rsc
    if verbose:
        print(f"Applying final HVG selection on {concat_adata.n_vars} concatenated genes")
    
    # Transfer to GPU for final processing
    rsc.get.anndata_to_GPU(concat_adata)
    
    rsc.pp.filter_genes(concat_adata, min_cells=1)
    
    final_n_hvgs = min(n_features, concat_adata.n_vars)
    if final_n_hvgs < concat_adata.n_vars:
        rsc.pp.highly_variable_genes(
            concat_adata,
            n_top_genes=final_n_hvgs,
            subset=True
        )
    
    # Transfer back to CPU
    rsc.get.anndata_to_CPU(concat_adata)
    concat_adata = ensure_cpu_arrays(concat_adata)
    
    if verbose:
        print(f"Final AnnData has {concat_adata.n_obs} samples and {concat_adata.n_vars} genes")
    
    # Compute cell proportions
    cell_proportion_df = _compute_cell_proportions(
        adata, samples, cell_types, sample_col, celltype_col
    )
    
    # Create final expression matrix
    cell_expression_hvg_df = _create_final_expression_matrix(
        all_hvg_data, all_gene_names, samples, cell_types, verbose
    )
    
    # Transfer original adata back to CPU
    rsc.get.anndata_to_CPU(adata)
    
    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
    
    return cell_expression_hvg_df, cell_proportion_df, concat_adata


def _create_pseudobulk_layers_rsc(
    adata: sc.AnnData,
    samples: list,
    cell_types: list,
    sample_col: str,
    celltype_col: str,
    batch_col: str,
    verbose: bool
) -> sc.AnnData:
    """
    Create pseudobulk AnnData with cell type layers using GPU acceleration.
    Note: adata should already be on GPU when this is called.
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
    
    # Get the GPU matrix
    X_gpu = adata.X
    
    # Add layers for each cell type
    for cell_type in cell_types:
        if verbose:
            print(f"Creating layer for {cell_type}")
        
        layer_matrix = np.zeros((n_samples, n_genes))
        
        # Get all cells for this cell type
        ct_mask = adata.obs[celltype_col] == cell_type
        ct_indices = np.where(ct_mask)[0]
        
        if len(ct_indices) == 0:
            pseudobulk_adata.layers[cell_type] = layer_matrix
            continue
        
        # Group by sample
        ct_samples = adata.obs.loc[ct_mask, sample_col].values
        
        for sample in np.unique(ct_samples):
            sample_idx = sample_indices[sample]
            sample_ct_mask = ct_indices[ct_samples == sample]
            
            if len(sample_ct_mask) > 0:
                # GPU computation using CuPy
                indices_gpu = cp.array(sample_ct_mask)
                
                # Extract subset and compute mean
                if hasattr(X_gpu, 'data'):  # Sparse matrix
                    subset = X_gpu[indices_gpu, :]
                    expr_mean = subset.mean(axis=0)
                    # Convert sparse output to dense array
                    if hasattr(expr_mean, 'toarray'):
                        expr_mean = expr_mean.toarray().flatten()
                    elif hasattr(expr_mean, 'A1'):
                        expr_mean = expr_mean.A1
                    layer_matrix[sample_idx, :] = cp.asnumpy(expr_mean)
                else:  # Dense matrix
                    subset = X_gpu[indices_gpu, :]
                    expr_mean = cp.mean(subset, axis=0)
                    layer_matrix[sample_idx, :] = cp.asnumpy(expr_mean)
        
        pseudobulk_adata.layers[cell_type] = layer_matrix
    
    return pseudobulk_adata


def _compute_cell_proportions(
    adata: sc.AnnData,
    samples: list,
    cell_types: list,
    sample_col: str,
    celltype_col: str
) -> pd.DataFrame:
    """Compute cell type proportions per sample."""
    
    proportion_df = pd.DataFrame(
        index=cell_types,
        columns=samples,
        dtype=float
    )
    
    # Compute proportions
    for sample in samples:
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
    GPU-accelerated backward compatibility wrapper using rapids_singlecell.
    """
    
    # Call the rsc-based GPU function
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
        
        # Ensure CPU arrays before saving
        final_adata = ensure_cpu_arrays(final_adata)
        sc.write(os.path.join(pseudobulk_dir, "pseudobulk_sample.h5ad"), final_adata)
    
    # Create backward-compatible dictionary
    pseudobulk = {
        "cell_expression_corrected": cell_expression_hvg_df,
        "cell_proportion": cell_proportion_df.T
    }
    
    return pseudobulk, final_adata