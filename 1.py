import anndata as ad
import pandas as pd
import numpy as np
import h5py
from typing import Dict, Any, Optional
import warnings

def inspect_h5ad_file(filepath: str, verbose: bool = True, max_display: int = 10) -> Dict[str, Any]:
    """
    Comprehensively inspect an h5ad file and return detailed information about its structure.
    
    Parameters:
    -----------
    filepath : str
        Path to the h5ad file
    verbose : bool, default True
        Whether to print detailed information
    max_display : int, default 10
        Maximum number of items to display in lists/arrays
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing all inspection results
    """
    
    inspection_results = {}
    
    try:
        # Load the AnnData object
        adata = ad.read_h5ad(filepath)
        
        if verbose:
            print("="*80)
            print(f"COMPREHENSIVE H5AD FILE INSPECTION: {filepath}")
            print("="*80)
        
        # Basic file information
        basic_info = {
            'file_path': filepath,
            'n_obs': adata.n_obs,  # number of cells/observations
            'n_vars': adata.n_vars,  # number of genes/variables
            'data_type': str(type(adata.X)),
            'data_shape': adata.X.shape if adata.X is not None else None,
        }
        inspection_results['basic_info'] = basic_info
        
        if verbose:
            print("\n1. BASIC INFORMATION")
            print("-" * 40)
            print(f"Number of cells (observations): {adata.n_obs:,}")
            print(f"Number of genes (variables): {adata.n_vars:,}")
            print(f"Data matrix shape: {adata.X.shape if adata.X is not None else 'None'}")
            print(f"Data matrix type: {type(adata.X)}")
            if hasattr(adata.X, 'dtype'):
                print(f"Data matrix dtype: {adata.X.dtype}")
        
        # Observations (cells) information
        obs_info = {
            'obs_names_preview': list(adata.obs_names[:max_display]),
            'obs_columns': list(adata.obs.columns),
            'n_obs_columns': len(adata.obs.columns),
            'obs_dtypes': dict(adata.obs.dtypes),
        }
        
        # Get unique values for categorical columns
        obs_categorical_info = {}
        for col in adata.obs.columns:
            if adata.obs[col].dtype == 'category' or adata.obs[col].dtype == 'object':
                unique_vals = adata.obs[col].unique()
                obs_categorical_info[col] = {
                    'n_unique': len(unique_vals),
                    'values': list(unique_vals[:max_display]) if len(unique_vals) > max_display else list(unique_vals)
                }
        
        obs_info['categorical_info'] = obs_categorical_info
        inspection_results['observations'] = obs_info
        
        if verbose:
            print("\n2. OBSERVATIONS (CELLS)")
            print("-" * 40)
            print(f"Cell names (first {max_display}): {adata.obs_names[:max_display].tolist()}")
            print(f"Number of cell metadata columns: {len(adata.obs.columns)}")
            print(f"Cell metadata columns: {list(adata.obs.columns)}")
            
            if obs_categorical_info:
                print("\nCategorical/Object columns details:")
                for col, info in obs_categorical_info.items():
                    print(f"  {col}: {info['n_unique']} unique values")
                    if info['n_unique'] <= max_display:
                        print(f"    Values: {info['values']}")
                    else:
                        print(f"    Sample values: {info['values']}")
        
        # Variables (genes) information
        var_info = {
            'var_names_preview': list(adata.var_names[:max_display]),
            'var_columns': list(adata.var.columns),
            'n_var_columns': len(adata.var.columns),
            'var_dtypes': dict(adata.var.dtypes),
        }
        
        # Get info about gene metadata
        var_categorical_info = {}
        for col in adata.var.columns:
            if adata.var[col].dtype == 'category' or adata.var[col].dtype == 'object':
                unique_vals = adata.var[col].unique()
                var_categorical_info[col] = {
                    'n_unique': len(unique_vals),
                    'values': list(unique_vals[:max_display]) if len(unique_vals) > max_display else list(unique_vals)
                }
        
        var_info['categorical_info'] = var_categorical_info
        inspection_results['variables'] = var_info
        
        if verbose:
            print("\n3. VARIABLES (GENES)")
            print("-" * 40)
            print(f"Gene names (first {max_display}): {adata.var_names[:max_display].tolist()}")
            print(f"Number of gene metadata columns: {len(adata.var.columns)}")
            print(f"Gene metadata columns: {list(adata.var.columns)}")
            
            if var_categorical_info:
                print("\nCategorical/Object columns details:")
                for col, info in var_categorical_info.items():
                    print(f"  {col}: {info['n_unique']} unique values")
                    if info['n_unique'] <= max_display:
                        print(f"    Values: {info['values']}")
                    else:
                        print(f"    Sample values: {info['values']}")
        
        # Data matrix information
        data_matrix_info = {}
        if adata.X is not None:
            data_matrix_info = {
                'shape': adata.X.shape,
                'dtype': str(adata.X.dtype),
                'is_sparse': hasattr(adata.X, 'toarray'),
                'matrix_type': str(type(adata.X)),
            }
            
            # Get some basic statistics (safely)
            try:
                if hasattr(adata.X, 'toarray'):
                    # Sparse matrix
                    data_matrix_info['n_nonzero'] = adata.X.nnz
                    data_matrix_info['sparsity'] = 1 - (adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]))
                    # Sample a small portion for stats to avoid memory issues
                    sample_data = adata.X[:min(1000, adata.X.shape[0]), :min(1000, adata.X.shape[1])].toarray()
                else:
                    sample_data = adata.X[:min(1000, adata.X.shape[0]), :min(1000, adata.X.shape[1])]
                
                data_matrix_info['min_value'] = float(np.min(sample_data))
                data_matrix_info['max_value'] = float(np.max(sample_data))
                data_matrix_info['mean_value'] = float(np.mean(sample_data))
            except Exception as e:
                data_matrix_info['stats_error'] = str(e)
        
        inspection_results['data_matrix'] = data_matrix_info
        
        if verbose:
            print("\n4. DATA MATRIX")
            print("-" * 40)
            if adata.X is not None:
                print(f"Shape: {adata.X.shape}")
                print(f"Data type: {adata.X.dtype}")
                print(f"Matrix type: {type(adata.X)}")
                if hasattr(adata.X, 'toarray'):
                    print(f"Sparse matrix: Yes")
                    print(f"Non-zero elements: {adata.X.nnz:,}")
                    print(f"Sparsity: {(1 - adata.X.nnz/(adata.X.shape[0]*adata.X.shape[1])):.2%}")
                else:
                    print(f"Sparse matrix: No")
                
                if 'min_value' in data_matrix_info:
                    print(f"Value range (sample): {data_matrix_info['min_value']:.3f} to {data_matrix_info['max_value']:.3f}")
                    print(f"Mean value (sample): {data_matrix_info['mean_value']:.3f}")
            else:
                print("No data matrix found (X is None)")
        
        # Layers information
        layers_info = {}
        if adata.layers:
            for layer_name, layer_data in adata.layers.items():
                layers_info[layer_name] = {
                    'shape': layer_data.shape,
                    'dtype': str(layer_data.dtype),
                    'is_sparse': hasattr(layer_data, 'toarray'),
                    'type': str(type(layer_data))
                }
        
        inspection_results['layers'] = layers_info
        
        if verbose:
            print("\n5. LAYERS")
            print("-" * 40)
            if adata.layers:
                print(f"Number of layers: {len(adata.layers)}")
                for layer_name, layer_info in layers_info.items():
                    print(f"  {layer_name}:")
                    print(f"    Shape: {layer_info['shape']}")
                    print(f"    Type: {layer_info['type']}")
                    print(f"    Dtype: {layer_info['dtype']}")
            else:
                print("No layers found")
        
        # Unstructured annotations
        uns_info = {}
        if adata.uns:
            for key, value in adata.uns.items():
                uns_info[key] = {
                    'type': str(type(value)),
                    'description': str(value)[:200] + ('...' if len(str(value)) > 200 else '')
                }
        
        inspection_results['unstructured'] = uns_info
        
        if verbose:
            print("\n6. UNSTRUCTURED ANNOTATIONS")
            print("-" * 40)
            if adata.uns:
                print(f"Number of unstructured annotations: {len(adata.uns)}")
                for key, info in uns_info.items():
                    print(f"  {key}: {info['type']}")
                    if len(info['description']) < 100:
                        print(f"    Value: {info['description']}")
            else:
                print("No unstructured annotations found")
        
        # Obsm (cell embeddings/reductions)
        obsm_info = {}
        if adata.obsm:
            for key, value in adata.obsm.items():
                obsm_info[key] = {
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'type': str(type(value))
                }
        
        inspection_results['obsm'] = obsm_info
        
        if verbose:
            print("\n7. OBSM (CELL EMBEDDINGS/DIMENSIONALITY REDUCTIONS)")
            print("-" * 40)
            if adata.obsm:
                for key, info in obsm_info.items():
                    print(f"  {key}: {info['shape']} ({info['dtype']})")
            else:
                print("No obsm data found")
        
        # Varm (gene embeddings)
        varm_info = {}
        if adata.varm:
            for key, value in adata.varm.items():
                varm_info[key] = {
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'type': str(type(value))
                }
        
        inspection_results['varm'] = varm_info
        
        if verbose:
            print("\n8. VARM (GENE EMBEDDINGS)")
            print("-" * 40)
            if adata.varm:
                for key, info in varm_info.items():
                    print(f"  {key}: {info['shape']} ({info['dtype']})")
            else:
                print("No varm data found")
        
        # Obsp (cell pairwise relationships)
        obsp_info = {}
        if adata.obsp:
            for key, value in adata.obsp.items():
                obsp_info[key] = {
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'type': str(type(value)),
                    'is_sparse': hasattr(value, 'toarray')
                }
        
        inspection_results['obsp'] = obsp_info
        
        if verbose:
            print("\n9. OBSP (CELL PAIRWISE RELATIONSHIPS)")
            print("-" * 40)
            if adata.obsp:
                for key, info in obsp_info.items():
                    print(f"  {key}: {info['shape']} ({info['dtype']}) - Sparse: {info['is_sparse']}")
            else:
                print("No obsp data found")
        
        # Varp (gene pairwise relationships)
        varp_info = {}
        if adata.varp:
            for key, value in adata.varp.items():
                varp_info[key] = {
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'type': str(type(value)),
                    'is_sparse': hasattr(value, 'toarray')
                }
        
        inspection_results['varp'] = varp_info
        
        if verbose:
            print("\n10. VARP (GENE PAIRWISE RELATIONSHIPS)")
            print("-" * 40)
            if adata.varp:
                for key, info in varp_info.items():
                    print(f"  {key}: {info['shape']} ({info['dtype']}) - Sparse: {info['is_sparse']}")
            else:
                print("No varp data found")
        
        if verbose:
            print("\n" + "="*80)
            print("INSPECTION COMPLETE")
            print("="*80)
        
        return inspection_results
        
    except Exception as e:
        error_info = {
            'error': str(e),
            'error_type': type(e).__name__
        }
        inspection_results['error'] = error_info
        
        if verbose:
            print(f"Error inspecting file: {e}")
        
        return inspection_results


def quick_inspect_h5ad(filepath: str) -> None:
    """
    Quick inspection function that prints only the most essential information.
    """
    try:
        adata = ad.read_h5ad(filepath)
        print(f"File: {filepath}")
        print(f"Cells: {adata.n_obs:,}")
        print(f"Genes: {adata.n_vars:,}")
        print(f"Cell metadata columns: {list(adata.obs.columns)}")
        print(f"Gene metadata columns: {list(adata.var.columns)}")
        if adata.layers:
            print(f"Layers: {list(adata.layers.keys())}")
        if adata.obsm:
            print(f"Cell embeddings: {list(adata.obsm.keys())}")
    except Exception as e:
        print(f"Error: {e}")


import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse

def inspect_gene_activity(adata_path, n_genes_to_show=10, n_cells_to_show=100):
    """
    Comprehensive inspection of gene activity h5ad file to diagnose integration issues.
    
    Parameters:
    -----------
    adata_path : str
        Path to the gene activity h5ad file
    n_genes_to_show : int
        Number of genes to show in detailed analysis
    n_cells_to_show : int
        Number of cells to sample for visualization
    """
    
    print("=" * 80)
    print("GENE ACTIVITY DATA INSPECTION")
    print("=" * 80)
    
    # Load data
    adata = sc.read(adata_path)
    print(f"\n1. BASIC INFORMATION:")
    print(f"   Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
    print(f"   Data type: {type(adata.X)}")
    if issparse(adata.X):
        print(f"   Sparse matrix format: {adata.X.format}")
        print(f"   Sparsity: {1 - adata.X.nnz / (adata.shape[0] * adata.shape[1]):.2%}")
    
    # Check for layers
    print(f"\n2. LAYERS PRESENT:")
    if adata.layers:
        for layer_name in adata.layers.keys():
            print(f"   - {layer_name}")
    else:
        print(f"   No layers found")
    
    # Data statistics
    print(f"\n3. DATA STATISTICS (X matrix):")
    
    # Convert to dense for statistics if sparse
    if issparse(adata.X):
        # Sample for memory efficiency
        sample_size = min(1000, adata.shape[0])
        sample_idx = np.random.choice(adata.shape[0], sample_size, replace=False)
        X_sample = adata.X[sample_idx].toarray()
    else:
        X_sample = adata.X[:min(1000, adata.shape[0])]
    
    print(f"   Overall statistics (sampled):")
    print(f"   - Min value: {np.min(X_sample):.6f}")
    print(f"   - Max value: {np.max(X_sample):.6f}")
    print(f"   - Mean value: {np.mean(X_sample):.6f}")
    print(f"   - Median value: {np.median(X_sample):.6f}")
    print(f"   - Std dev: {np.std(X_sample):.6f}")
    
    # Check for zero inflation
    zero_proportion = np.mean(X_sample == 0)
    print(f"   - Proportion of zeros: {zero_proportion:.2%}")
    print(f"   - Proportion of non-zeros: {1-zero_proportion:.2%}")
    
    # Non-zero value statistics
    non_zero_vals = X_sample[X_sample != 0]
    if len(non_zero_vals) > 0:
        print(f"\n   Non-zero value statistics:")
        print(f"   - Min (non-zero): {np.min(non_zero_vals):.6f}")
        print(f"   - Max (non-zero): {np.max(non_zero_vals):.6f}")
        print(f"   - Mean (non-zero): {np.mean(non_zero_vals):.6f}")
        print(f"   - Median (non-zero): {np.median(non_zero_vals):.6f}")
    else:
        print(f"\n   WARNING: No non-zero values found in sample!")
    
    # Per-gene statistics
    print(f"\n4. PER-GENE STATISTICS:")
    
    # Calculate gene-wise statistics
    if issparse(adata.X):
        gene_means = np.array(adata.X.mean(axis=0)).flatten()
        gene_vars = np.array(adata.X.power(2).mean(axis=0)).flatten() - gene_means**2
        gene_zeros = np.array((adata.X == 0).sum(axis=0)).flatten() / adata.shape[0]
    else:
        gene_means = np.mean(adata.X, axis=0)
        gene_vars = np.var(adata.X, axis=0)
        gene_zeros = np.mean(adata.X == 0, axis=0)
    
    print(f"   Gene expression levels:")
    print(f"   - Genes with mean = 0: {np.sum(gene_means == 0)} ({np.mean(gene_means == 0):.1%})")
    print(f"   - Genes with mean < 0.001: {np.sum(gene_means < 0.001)} ({np.mean(gene_means < 0.001):.1%})")
    print(f"   - Genes with mean < 0.01: {np.sum(gene_means < 0.01)} ({np.mean(gene_means < 0.01):.1%})")
    print(f"   - Genes with mean < 0.1: {np.sum(gene_means < 0.1)} ({np.mean(gene_means < 0.1):.1%})")
    
    print(f"\n   Gene detection rates:")
    print(f"   - Genes detected in 0% of cells: {np.sum(gene_zeros == 1)}")
    print(f"   - Genes detected in <1% of cells: {np.sum(gene_zeros > 0.99)}")
    print(f"   - Genes detected in <10% of cells: {np.sum(gene_zeros > 0.90)}")
    print(f"   - Genes detected in >50% of cells: {np.sum(gene_zeros < 0.50)}")
    
    # Per-cell statistics
    print(f"\n5. PER-CELL STATISTICS:")
    
    if issparse(adata.X):
        cell_sums = np.array(adata.X.sum(axis=1)).flatten()
        cell_detected = np.array((adata.X > 0).sum(axis=1)).flatten()
    else:
        cell_sums = np.sum(adata.X, axis=1)
        cell_detected = np.sum(adata.X > 0, axis=1)
    
    print(f"   Total counts per cell:")
    print(f"   - Min: {np.min(cell_sums):.2f}")
    print(f"   - Max: {np.max(cell_sums):.2f}")
    print(f"   - Mean: {np.mean(cell_sums):.2f}")
    print(f"   - Median: {np.median(cell_sums):.2f}")
    
    print(f"\n   Detected genes per cell:")
    print(f"   - Min: {np.min(cell_detected)}")
    print(f"   - Max: {np.max(cell_detected)}")
    print(f"   - Mean: {np.mean(cell_detected):.1f}")
    print(f"   - Median: {np.median(cell_detected):.1f}")
    
    # Sample specific genes
    print(f"\n6. SAMPLE GENES (Top {n_genes_to_show} by mean expression):")
    
    # Get top expressed genes
    top_gene_idx = np.argsort(gene_means)[-n_genes_to_show:][::-1]
    
    for idx in top_gene_idx:
        gene_name = adata.var_names[idx]
        print(f"\n   {gene_name}:")
        print(f"   - Mean: {gene_means[idx]:.6f}")
        print(f"   - Variance: {gene_vars[idx]:.6f}")
        print(f"   - Zero rate: {gene_zeros[idx]:.2%}")
        print(f"   - Detection rate: {1-gene_zeros[idx]:.2%}")
    
    # Check for common marker genes
    print(f"\n7. COMMON MARKER GENES CHECK:")
    common_markers = ['CD3D', 'CD4', 'CD8A', 'MS4A1', 'CD19', 'CD14', 'FCGR3A', 
                     'NCAM1', 'CD34', 'PPBP', 'EPCAM', 'PECAM1']
    
    found_markers = [gene for gene in common_markers if gene in adata.var_names]
    print(f"   Found {len(found_markers)}/{len(common_markers)} common markers")
    
    for gene in found_markers[:5]:  # Show first 5
        idx = np.where(adata.var_names == gene)[0][0]
        print(f"   {gene}: mean={gene_means[idx]:.6f}, detection={1-gene_zeros[idx]:.1%}")
    
    # Visualization
    print(f"\n8. GENERATING VISUALIZATIONS...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Distribution of gene means
    ax = axes[0, 0]
    ax.hist(gene_means[gene_means > 0], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Gene Mean Expression (non-zero)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Gene Mean Expression')
    ax.set_yscale('log')
    
    # Distribution of gene detection rates
    ax = axes[0, 1]
    ax.hist(1 - gene_zeros, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Gene Detection Rate')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Gene Detection Rates')
    
    # Cell total counts distribution
    ax = axes[0, 2]
    ax.hist(cell_sums, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Total Counts per Cell')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Cell Total Counts')
    
    # Detected genes per cell
    ax = axes[1, 0]
    ax.hist(cell_detected, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Detected Genes')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Detected Genes per Cell')
    
    # Sample heatmap of top genes
    ax = axes[1, 1]
    sample_cells = np.random.choice(adata.shape[0], min(n_cells_to_show, adata.shape[0]), replace=False)
    if issparse(adata.X):
        heatmap_data = adata.X[sample_cells][:, top_gene_idx[:10]].toarray()
    else:
        heatmap_data = adata.X[sample_cells][:, top_gene_idx[:10]]
    
    im = ax.imshow(heatmap_data.T, aspect='auto', cmap='viridis')
    ax.set_xlabel('Cells (sampled)')
    ax.set_ylabel('Top 10 Genes')
    ax.set_title('Expression Heatmap (Top Genes)')
    plt.colorbar(im, ax=ax)
    
    # Mean vs variance plot
    ax = axes[1, 2]
    ax.scatter(gene_means, gene_vars, alpha=0.3, s=1)
    ax.set_xlabel('Mean Expression')
    ax.set_ylabel('Variance')
    ax.set_title('Mean-Variance Relationship')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Recommendations
    print(f"\n9. DIAGNOSTIC SUMMARY & RECOMMENDATIONS:")
    
    if zero_proportion > 0.95:
        print(f"   ⚠️  EXTREMELY SPARSE DATA ({zero_proportion:.1%} zeros)")
        print(f"      → Consider different gene activity scoring method")
        print(f"      → Check peak-to-gene assignment parameters")
    
    if np.sum(gene_means == 0) > adata.shape[1] * 0.5:
        print(f"   ⚠️  MANY SILENT GENES ({np.mean(gene_means == 0):.1%} with zero expression)")
        print(f"      → Filter genes before integration")
        print(f"      → Check if gene names match between modalities")
    
    if np.mean(cell_detected) < 100:
        print(f"   ⚠️  LOW GENES PER CELL (mean: {np.mean(cell_detected):.0f})")
        print(f"      → May need to adjust gene activity calculation")
        print(f"      → Consider using broader peak windows")
    
    if np.max(X_sample) < 1:
        print(f"   ⚠️  LOW DYNAMIC RANGE (max value: {np.max(X_sample):.6f})")
        print(f"      → Data may be over-normalized")
        print(f"      → Consider using raw counts or different normalization")
    
    print(f"\n   Suggested preprocessing steps:")
    print(f"   1. Use TF-IDF normalization for ATAC data")
    print(f"   2. Filter genes detected in <1% of cells")
    print(f"   3. Consider log1p transformation with smaller pseudocount")
    print(f"   4. Use specialized integration methods (Seurat WNN, LIGER)")
    
    return adata

def fix_integrated_h5ad_with_obsm_varm(
    integrated_h5ad_path: str,
    glue_dir: str,
    raw_rna_path: str,
    output_path: str = None,
    chunk_size: int = 10000,
    enable_chunking: bool = True,
    verbose: bool = True
) -> ad.AnnData:
    """
    Fix an already generated integrated h5ad file by appending missing obsm and varm information.
    Now supports chunked processing for memory efficiency.
    
    Parameters
    ----------
    integrated_h5ad_path : str
        Path to the already generated integrated h5ad file
    glue_dir : str
        Directory containing the GLUE output files
    raw_rna_path : str
        Path to the raw RNA counts h5ad file
    output_path : str, optional
        Output path for the fixed file. If None, overwrites the input file
    chunk_size : int, default 10000
        Number of cells to process in each chunk
    enable_chunking : bool, default True
        Whether to use chunked processing (recommended for large datasets)
    memory_threshold_gb : float, default 8.0
        Estimated memory usage threshold (GB) to trigger chunked processing
    verbose : bool
        Whether to print progress messages
    
    Returns
    -------
    ad.AnnData
        Fixed AnnData object with obsm and varm appended
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import os
    import gc
    import psutil
    from scipy import sparse
    from typing import Dict, Any, Tuple
    
    def process_obsm_chunk(
        integrated_chunk: ad.AnnData,
        source_adata: ad.AnnData,
        source_cells: pd.Index,
        chunk_cells: pd.Index,
        cell_mapping: pd.Series,
        obsm_key: str,
        global_idx_offset: int
    ) -> np.ndarray:
        """Process obsm data for a chunk of cells"""
        common_cells = chunk_cells.intersection(source_cells).intersection(source_adata.obs.index)
        
        if len(common_cells) == 0:
            return None
            
        # Get source data for common cells
        source_data = source_adata[common_cells].obsm[obsm_key]
        
        # Create chunk array
        if len(source_data.shape) == 2:
            chunk_array = np.zeros((len(chunk_cells), source_data.shape[1]), dtype=source_data.dtype)
        else:
            chunk_array = np.zeros((len(chunk_cells),) + source_data.shape[1:], dtype=source_data.dtype)
        
        # Fill in values
        for i, cell_id in enumerate(common_cells):
            if cell_id in cell_mapping.index:
                local_idx = chunk_cells.get_loc(cell_id)
                chunk_array[local_idx] = source_data[i]
        
        return chunk_array
    
    # Set output path
    if output_path is None:
        output_path = integrated_h5ad_path
    
    # Construct file paths
    rna_processed_path = os.path.join(glue_dir, "glue-rna-emb.h5ad")
    atac_path = os.path.join(glue_dir, "glue-atac-emb.h5ad")
    
    if verbose:
        print("🔧 Fixing integrated h5ad file with missing obsm/varm...")
        print(f"   Input: {integrated_h5ad_path}")
        print(f"   Output: {output_path}")
        print(f"   Chunked processing: {'Enabled' if enable_chunking else 'Disabled'}")
        if enable_chunking:
            print(f"   Chunk size: {chunk_size:,} cells")
    
    # Load integrated data (check if we should use backed mode)
    if verbose:
        print("\n📂 Loading integrated data...")
    
    # First, get basic info about the dataset
    integrated_info = ad.read_h5ad(integrated_h5ad_path, backed='r')
    n_obs, n_vars = integrated_info.shape
    
    if verbose:
        print(f"   Dataset size: {n_obs:,} cells × {n_vars:,} features")
    
    if enable_chunking:
        if verbose:
            print("   Using chunked processing for memory efficiency")
        integrated = integrated_info  # Keep in backed mode
    else:
        if verbose:
            print("   Loading full dataset into memory")
        integrated_info.file.close()
        integrated = ad.read_h5ad(integrated_h5ad_path)
    
    # Identify RNA and ATAC cells
    rna_mask = integrated.obs['modality'] == 'RNA'
    atac_mask = integrated.obs['modality'] == 'ATAC'
    rna_cells = integrated.obs.index[rna_mask]
    atac_cells = integrated.obs.index[atac_mask]
    
    if verbose:
        print(f"   Found {rna_mask.sum():,} RNA cells and {atac_mask.sum():,} ATAC cells")
    
    # Create cell mappings
    rna_cell_mapping = pd.Series(range(len(rna_cells)), index=rna_cells)
    atac_cell_mapping = pd.Series(
        range(len(rna_cells), len(rna_cells) + len(atac_cells)), 
        index=atac_cells
    )
    
    # Load source data files
    if verbose:
        print("\n📂 Loading source embeddings...")
    
    rna_processed = ad.read_h5ad(rna_processed_path)
    atac_processed = ad.read_h5ad(atac_path)
    
    # Collect all obsm keys to process
    all_obsm_keys = set()
    if hasattr(rna_processed, 'obsm'):
        all_obsm_keys.update(rna_processed.obsm.keys())
    if hasattr(atac_processed, 'obsm'):
        all_obsm_keys.update(atac_processed.obsm.keys())
    
    # Initialize output arrays
    obsm_arrays = {}
    
    if enable_chunking:
        # Process in chunks
        n_chunks = (n_obs + chunk_size - 1) // chunk_size
        if verbose:
            print(f"\n🔄 Processing {n_chunks} chunks...")
        
        # Initialize arrays for each obsm key
        for key in all_obsm_keys:
            # Determine array dimensions from source data
            if key in rna_processed.obsm:
                sample_shape = rna_processed.obsm[key].shape[1:]
                dtype = rna_processed.obsm[key].dtype
            elif key in atac_processed.obsm:
                sample_shape = atac_processed.obsm[key].shape[1:]
                dtype = atac_processed.obsm[key].dtype
            else:
                continue
                
            if key not in integrated.obsm or key == 'X_umap':
                obsm_arrays[key] = np.zeros((n_obs,) + sample_shape, dtype=dtype)
            else:
                obsm_arrays[key] = integrated.obsm[key].copy()
        
        # Process chunks
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_obs)
            
            if verbose and chunk_idx % max(1, n_chunks // 10) == 0:
                print(f"   Processing chunk {chunk_idx + 1}/{n_chunks} "
                      f"(cells {start_idx:,}-{end_idx-1:,})")
            
            # Get chunk cell indices
            chunk_cells = integrated.obs.index[start_idx:end_idx]
            chunk_rna_cells = chunk_cells.intersection(rna_cells)
            chunk_atac_cells = chunk_cells.intersection(atac_cells)
            
            # Process RNA obsm for this chunk
            for key in rna_processed.obsm.keys():
                if key in obsm_arrays:
                    chunk_data = process_obsm_chunk(
                        None, rna_processed, rna_cells, chunk_rna_cells,
                        rna_cell_mapping, key, start_idx
                    )
                    if chunk_data is not None:
                        # Map chunk data to global indices
                        for i, cell_id in enumerate(chunk_rna_cells):
                            global_idx = rna_cell_mapping[cell_id]
                            obsm_arrays[key][global_idx] = chunk_data[i]
            
            # Process ATAC obsm for this chunk
            for key in atac_processed.obsm.keys():
                if key in obsm_arrays:
                    chunk_data = process_obsm_chunk(
                        None, atac_processed, atac_cells, chunk_atac_cells,
                        atac_cell_mapping, key, start_idx
                    )
                    if chunk_data is not None:
                        # Map chunk data to global indices
                        for i, cell_id in enumerate(chunk_atac_cells):
                            global_idx = atac_cell_mapping[cell_id]
                            obsm_arrays[key][global_idx] = chunk_data[i]
            
            # Trigger garbage collection periodically
            if chunk_idx % 10 == 0:
                gc.collect()
        
        # Convert to full AnnData object for final processing
        if verbose:
            print("\n📝 Converting to full AnnData object...")
        
        integrated.file.close()
        integrated = ad.read_h5ad(integrated_h5ad_path)
        
        # Add processed obsm arrays
        for key, array in obsm_arrays.items():
            integrated.obsm[key] = array
            if verbose:
                print(f"   Added obsm['{key}'] with shape {array.shape}")
    
    else:
        # Non-chunked processing (original logic)
        if verbose:
            print("\n🔄 Processing RNA embeddings...")
        
        common_rna_cells = rna_cells.intersection(rna_processed.obs.index)
        if len(common_rna_cells) > 0:
            for key in rna_processed.obsm.keys():
                if key not in integrated.obsm or key == 'X_umap':
                    if verbose:
                        print(f"   Adding RNA obsm['{key}'] with shape {rna_processed.obsm[key].shape}")
                    
                    rna_data = rna_processed[common_rna_cells].obsm[key]
                    
                    if len(rna_data.shape) == 2:
                        full_array = np.zeros((integrated.n_obs, rna_data.shape[1]), dtype=rna_data.dtype)
                    else:
                        full_array = np.zeros((integrated.n_obs,) + rna_data.shape[1:], dtype=rna_data.dtype)
                    
                    for i, cell_id in enumerate(common_rna_cells):
                        if cell_id in rna_cell_mapping.index:
                            full_idx = rna_cell_mapping[cell_id]
                            full_array[full_idx] = rna_data[i]
                    
                    integrated.obsm[key] = full_array
        
        if verbose:
            print("\n🔄 Processing ATAC embeddings...")
        
        common_atac_cells = atac_cells.intersection(atac_processed.obs.index)
        if len(common_atac_cells) > 0:
            for key in atac_processed.obsm.keys():
                if key not in integrated.obsm:
                    if verbose:
                        print(f"   Adding ATAC obsm['{key}'] with shape {atac_processed.obsm[key].shape}")
                    
                    atac_data = atac_processed[common_atac_cells].obsm[key]
                    
                    if len(atac_data.shape) == 2:
                        full_array = np.zeros((integrated.n_obs, atac_data.shape[1]), dtype=atac_data.dtype)
                    else:
                        full_array = np.zeros((integrated.n_obs,) + atac_data.shape[1:], dtype=atac_data.dtype)
                    
                    for i, cell_id in enumerate(common_atac_cells):
                        if cell_id in atac_cell_mapping.index:
                            full_idx = atac_cell_mapping[cell_id]
                            full_array[full_idx] = atac_data[i]
                    
                    integrated.obsm[key] = full_array
                elif key != 'X_umap':
                    if verbose:
                        print(f"   Updating ATAC portion of obsm['{key}']")
                    
                    atac_data = atac_processed[common_atac_cells].obsm[key]
                    
                    for i, cell_id in enumerate(common_atac_cells):
                        if cell_id in atac_cell_mapping.index:
                            full_idx = atac_cell_mapping[cell_id]
                            integrated.obsm[key][full_idx] = atac_data[i]
    
    # Process varm (same for both chunked and non-chunked)
    if verbose:
        print("\n📝 Processing varm data...")
    
    # Extract RNA varm
    rna_varm_dict = {}
    if hasattr(rna_processed, 'varm'):
        for key in rna_processed.varm.keys():
            if verbose:
                print(f"   Storing RNA varm['{key}'] with shape {rna_processed.varm[key].shape}")
            rna_varm_dict[key] = rna_processed.varm[key].copy()
    
    # Extract ATAC varm
    atac_varm_dict = {}
    if hasattr(atac_processed, 'varm'):
        for key in atac_processed.varm.keys():
            if verbose:
                print(f"   Storing ATAC varm['{key}'] with shape {atac_processed.varm[key].shape}")
            atac_varm_dict[key] = atac_processed.varm[key].copy()
    
    # Clean up source data
    del rna_processed, atac_processed
    gc.collect()
    
    # Load raw RNA for varm
    if verbose:
        print("\n📂 Loading raw RNA for varm...")
    try:
        rna_raw = ad.read_h5ad(raw_rna_path, backed='r')
        
        if hasattr(rna_raw, 'varm'):
            for key in rna_raw.varm.keys():
                if verbose:
                    print(f"   Storing raw RNA varm['{key}'] with shape {rna_raw.varm[key].shape}")
                rna_varm_dict[key] = rna_raw.varm[key].copy()
        
        rna_raw.file.close()
        del rna_raw
    except Exception as e:
        if verbose:
            print(f"   Warning: Could not load raw RNA varm: {e}")
    
    # Apply varm to integrated data
    all_varm = {**atac_varm_dict, **rna_varm_dict}  # RNA takes precedence
    
    if verbose and len(all_varm) > 0:
        print("\n📝 Applying varm to integrated data...")
    
    for key, value in all_varm.items():
        if value.shape[0] == integrated.n_vars:
            integrated.varm[key] = value
            if verbose:
                print(f"   Added varm['{key}'] with shape {value.shape}")
        else:
            if verbose:
                print(f"   Skipped varm['{key}'] - dimension mismatch "
                      f"(expected {integrated.n_vars}, got {value.shape[0]})")
    
    # Final garbage collection
    gc.collect()
    
    # Save the fixed file
    if verbose:
        print(f"\n💾 Saving fixed integrated data to {output_path}...")
    
    # Use chunked writing for large datasets
    if n_obs > 50000:
        if verbose:
            print("   Using optimized writing for large dataset...")
        integrated.write(output_path, compression='gzip', compression_opts=4)
    else:
        integrated.write(output_path, compression='gzip', compression_opts=4)
    
    if verbose:
        print("\n✅ Fix complete!")
        print(f"\n📊 Summary:")
        print(f"   Total cells: {integrated.n_obs:,}")
        print(f"   Total genes: {integrated.n_vars:,}")
        print(f"   obsm keys: {list(integrated.obsm.keys())}")
        print(f"   varm keys: {list(integrated.varm.keys())}")
        print(f"   obs columns: {list(integrated.obs.columns)}")
        print(f"   var columns: {list(integrated.var.columns)}")
    
    return integrated

def consume_memory():
    """
    WARNING: This will consume all available RAM and likely crash your system.
    Use only in VMs or controlled environments.
    """
    memory_hog = []
    try:
        while True:
            # Allocate 100MB chunks
            chunk = bytearray(100 * 1024 * 1024)  # 100MB
            memory_hog.append(chunk)
            print(f"Allocated {len(memory_hog) * 100}MB")
    except MemoryError:
        print("Memory exhausted!")
    
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, csc_matrix
import h5py
import os
from typing import Dict, Any, List, Tuple
import warnings

def format_bytes(bytes_size: float) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def get_object_memory(obj: Any) -> int:
    """Estimate memory usage of an object."""
    if obj is None:
        return 0
    elif issparse(obj):
        # For sparse matrices
        return obj.data.nbytes + obj.indices.nbytes + obj.indptr.nbytes
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum()
    elif isinstance(obj, pd.Series):
        return obj.memory_usage(deep=True)
    elif isinstance(obj, dict):
        return sum(get_object_memory(v) for v in obj.values())
    elif isinstance(obj, (list, tuple)):
        return sum(get_object_memory(item) for item in obj)
    else:
        # Rough estimate for other objects
        import sys
        return sys.getsizeof(obj)

def analyze_matrix_efficiency(matrix: Any, name: str) -> Dict[str, Any]:
    """Analyze if a matrix is stored efficiently."""
    analysis = {
        'name': name,
        'issues': [],
        'recommendations': []
    }
    
    if matrix is None:
        return analysis
    
    # Check if dense matrix could be sparse
    if isinstance(matrix, np.ndarray):
        total_elements = matrix.size
        if total_elements > 0:
            zero_fraction = np.sum(matrix == 0) / total_elements
            analysis['zero_fraction'] = zero_fraction
            
            if zero_fraction > 0.5:
                analysis['issues'].append(f"Dense matrix with {zero_fraction:.1%} zeros")
                analysis['recommendations'].append(f"Convert to sparse format (could save ~{zero_fraction:.0%} memory)")
                
        # Check data type
        if matrix.dtype == np.float64:
            analysis['issues'].append("Using float64 precision")
            analysis['recommendations'].append("Consider float32 if precision allows")
            
    elif issparse(matrix):
        # Check sparse matrix format
        if not isinstance(matrix, csr_matrix) and not isinstance(matrix, csc_matrix):
            analysis['issues'].append(f"Using {type(matrix).__name__} sparse format")
            analysis['recommendations'].append("Consider CSR or CSC format for better performance")
            
        # Check if sparse matrix is actually dense
        density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
        analysis['density'] = density
        if density > 0.5:
            analysis['issues'].append(f"Sparse matrix with {density:.1%} density")
            analysis['recommendations'].append("Consider dense format for better performance")
    
    return analysis

def inspect_h5ad_memory(filepath: str) -> None:
    """
    Comprehensive h5ad file memory inspection and optimization suggestions.
    
    Parameters:
    -----------
    filepath : str
        Path to the h5ad file
    """
    
    print("=" * 80)
    print(f"H5AD MEMORY INSPECTION REPORT")
    print(f"File: {filepath}")
    print(f"File size on disk: {format_bytes(os.path.getsize(filepath))}")
    print("=" * 80)
    
    # Load the anndata object
    print("\nLoading file...")
    adata = sc.read_h5ad(filepath)
    
    # Basic information
    print(f"\nBasic Information:")
    print(f"  Number of observations (cells): {adata.n_obs:,}")
    print(f"  Number of variables (genes): {adata.n_vars:,}")
    
    # Memory breakdown
    memory_breakdown = {}
    
    # 1. Main data matrix (X)
    print("\n" + "-" * 40)
    print("1. MAIN DATA MATRIX (X)")
    print("-" * 40)
    x_memory = get_object_memory(adata.X)
    memory_breakdown['X'] = x_memory
    print(f"  Memory usage: {format_bytes(x_memory)}")
    print(f"  Type: {type(adata.X)}")
    if hasattr(adata.X, 'dtype'):
        print(f"  Data type: {adata.X.dtype}")
    if hasattr(adata.X, 'shape'):
        print(f"  Shape: {adata.X.shape}")
    
    x_analysis = analyze_matrix_efficiency(adata.X, 'X')
    if x_analysis['issues']:
        print(f"  ⚠️  Issues found:")
        for issue in x_analysis['issues']:
            print(f"    - {issue}")
        print(f"  💡 Recommendations:")
        for rec in x_analysis['recommendations']:
            print(f"    - {rec}")
    
    # 2. Layers
    print("\n" + "-" * 40)
    print("2. LAYERS")
    print("-" * 40)
    if adata.layers:
        total_layers_memory = 0
        for layer_name, layer_data in adata.layers.items():
            layer_memory = get_object_memory(layer_data)
            total_layers_memory += layer_memory
            memory_breakdown[f'layer_{layer_name}'] = layer_memory
            print(f"  '{layer_name}':")
            print(f"    Memory: {format_bytes(layer_memory)}")
            print(f"    Type: {type(layer_data)}")
            
            layer_analysis = analyze_matrix_efficiency(layer_data, layer_name)
            if layer_analysis['issues']:
                print(f"    ⚠️  Issues:")
                for issue in layer_analysis['issues']:
                    print(f"      - {issue}")
        
        memory_breakdown['layers_total'] = total_layers_memory
        print(f"\n  Total layers memory: {format_bytes(total_layers_memory)}")
        
        # Check for duplicate layers
        if len(adata.layers) > 1:
            print("\n  Checking for duplicate layers...")
            duplicates = []
            layer_names = list(adata.layers.keys())
            for i in range(len(layer_names)):
                for j in range(i+1, len(layer_names)):
                    layer1 = adata.layers[layer_names[i]]
                    layer2 = adata.layers[layer_names[j]]
                    if type(layer1) == type(layer2):
                        if issparse(layer1) and issparse(layer2):
                            if (layer1 != layer2).nnz == 0:
                                duplicates.append((layer_names[i], layer_names[j]))
                        elif isinstance(layer1, np.ndarray) and isinstance(layer2, np.ndarray):
                            if np.array_equal(layer1, layer2):
                                duplicates.append((layer_names[i], layer_names[j]))
            
            if duplicates:
                print(f"  ⚠️  Found duplicate layers:")
                for dup in duplicates:
                    print(f"    - '{dup[0]}' and '{dup[1]}' are identical")
    else:
        print("  No layers found")
    
    # 3. Observations metadata (obs)
    print("\n" + "-" * 40)
    print("3. OBSERVATIONS METADATA (obs)")
    print("-" * 40)
    obs_memory = get_object_memory(adata.obs)
    memory_breakdown['obs'] = obs_memory
    print(f"  Total memory: {format_bytes(obs_memory)}")
    print(f"  Number of columns: {len(adata.obs.columns)}")
    
    if len(adata.obs.columns) > 0:
        print("\n  Top memory-consuming columns:")
        col_memories = []
        for col in adata.obs.columns:
            col_memory = adata.obs[col].memory_usage(deep=True)
            col_memories.append((col, col_memory))
        
        col_memories.sort(key=lambda x: x[1], reverse=True)
        for col, mem in col_memories[:10]:
            dtype = str(adata.obs[col].dtype)
            print(f"    '{col}': {format_bytes(mem)} (dtype: {dtype})")
            
            # Check for optimization opportunities
            if adata.obs[col].dtype == 'object':
                unique_ratio = len(adata.obs[col].unique()) / len(adata.obs[col])
                if unique_ratio < 0.5:
                    print(f"      💡 Consider converting to categorical (unique ratio: {unique_ratio:.1%})")
    
    # 4. Variables metadata (var)
    print("\n" + "-" * 40)
    print("4. VARIABLES METADATA (var)")
    print("-" * 40)
    var_memory = get_object_memory(adata.var)
    memory_breakdown['var'] = var_memory
    print(f"  Total memory: {format_bytes(var_memory)}")
    print(f"  Number of columns: {len(adata.var.columns)}")
    
    if len(adata.var.columns) > 0:
        print("\n  Top memory-consuming columns:")
        col_memories = []
        for col in adata.var.columns:
            col_memory = adata.var[col].memory_usage(deep=True)
            col_memories.append((col, col_memory))
        
        col_memories.sort(key=lambda x: x[1], reverse=True)
        for col, mem in col_memories[:10]:
            dtype = str(adata.var[col].dtype)
            print(f"    '{col}': {format_bytes(mem)} (dtype: {dtype})")
    
    # 5. Dimensional reductions (obsm)
    print("\n" + "-" * 40)
    print("5. DIMENSIONAL REDUCTIONS (obsm)")
    print("-" * 40)
    if adata.obsm:
        total_obsm_memory = 0
        for key, value in adata.obsm.items():
            obsm_memory = get_object_memory(value)
            total_obsm_memory += obsm_memory
            memory_breakdown[f'obsm_{key}'] = obsm_memory
            print(f"  '{key}':")
            print(f"    Memory: {format_bytes(obsm_memory)}")
            if hasattr(value, 'shape'):
                print(f"    Shape: {value.shape}")
            if hasattr(value, 'dtype'):
                print(f"    Data type: {value.dtype}")
                if value.dtype == np.float64:
                    print(f"    💡 Consider float32 for embeddings")
        
        memory_breakdown['obsm_total'] = total_obsm_memory
        print(f"\n  Total obsm memory: {format_bytes(total_obsm_memory)}")
    else:
        print("  No dimensional reductions found")
    
    # 6. Variable embeddings (varm)
    print("\n" + "-" * 40)
    print("6. VARIABLE EMBEDDINGS (varm)")
    print("-" * 40)
    if adata.varm:
        total_varm_memory = 0
        for key, value in adata.varm.items():
            varm_memory = get_object_memory(value)
            total_varm_memory += varm_memory
            memory_breakdown[f'varm_{key}'] = varm_memory
            print(f"  '{key}': {format_bytes(varm_memory)}")
        
        memory_breakdown['varm_total'] = total_varm_memory
        print(f"\n  Total varm memory: {format_bytes(total_varm_memory)}")
    else:
        print("  No variable embeddings found")
    
    # 7. Unstructured data (uns)
    print("\n" + "-" * 40)
    print("7. UNSTRUCTURED DATA (uns)")
    print("-" * 40)
    if adata.uns:
        def explore_uns(uns_dict, prefix=""):
            items = []
            for key, value in uns_dict.items():
                mem = get_object_memory(value)
                full_key = f"{prefix}{key}"
                items.append((full_key, mem, type(value).__name__))
                
                if isinstance(value, dict) and mem > 1024*1024:  # Explore large nested dicts
                    nested_items = explore_uns(value, f"{full_key}.")
                    items.extend(nested_items)
            return items
        
        uns_items = explore_uns(adata.uns)
        uns_items.sort(key=lambda x: x[1], reverse=True)
        
        total_uns_memory = sum(item[1] for item in uns_items if '.' not in item[0])
        memory_breakdown['uns_total'] = total_uns_memory
        
        print(f"  Total memory: {format_bytes(total_uns_memory)}")
        print(f"\n  Top memory-consuming items:")
        for key, mem, dtype in uns_items[:15]:
            print(f"    '{key}': {format_bytes(mem)} ({dtype})")
    else:
        print("  No unstructured data found")
    
    # 8. Pairwise matrices (obsp, varp)
    print("\n" + "-" * 40)
    print("8. PAIRWISE MATRICES")
    print("-" * 40)
    
    if adata.obsp:
        print("  Observation pairwise (obsp):")
        total_obsp_memory = 0
        for key, value in adata.obsp.items():
            obsp_memory = get_object_memory(value)
            total_obsp_memory += obsp_memory
            memory_breakdown[f'obsp_{key}'] = obsp_memory
            print(f"    '{key}': {format_bytes(obsp_memory)}")
        memory_breakdown['obsp_total'] = total_obsp_memory
    else:
        print("  No observation pairwise matrices found")
    
    if adata.varp:
        print("  Variable pairwise (varp):")
        total_varp_memory = 0
        for key, value in adata.varp.items():
            varp_memory = get_object_memory(value)
            total_varp_memory += varp_memory
            memory_breakdown[f'varp_{key}'] = varp_memory
            print(f"    '{key}': {format_bytes(varp_memory)}")
        memory_breakdown['varp_total'] = total_varp_memory
    else:
        print("  No variable pairwise matrices found")
    
    # Summary
    print("\n" + "=" * 80)
    print("MEMORY SUMMARY")
    print("=" * 80)
    
    # Sort components by memory usage
    main_components = [
        ('X', memory_breakdown.get('X', 0)),
        ('layers', memory_breakdown.get('layers_total', 0)),
        ('obs', memory_breakdown.get('obs', 0)),
        ('var', memory_breakdown.get('var', 0)),
        ('obsm', memory_breakdown.get('obsm_total', 0)),
        ('varm', memory_breakdown.get('varm_total', 0)),
        ('uns', memory_breakdown.get('uns_total', 0)),
        ('obsp', memory_breakdown.get('obsp_total', 0)),
        ('varp', memory_breakdown.get('varp_total', 0)),
    ]
    
    main_components.sort(key=lambda x: x[1], reverse=True)
    
    total_memory = sum(x[1] for x in main_components)
    
    print(f"\nTotal estimated memory in RAM: {format_bytes(total_memory)}")
    print(f"\nBreakdown by component:")
    for name, mem in main_components:
        if mem > 0:
            percentage = (mem / total_memory) * 100
            print(f"  {name:10s}: {format_bytes(mem):>12s} ({percentage:5.1f}%)")
    
    # General recommendations
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    # Check overall data type usage
    if adata.X is not None and hasattr(adata.X, 'dtype'):
        if adata.X.dtype == np.float64:
            potential_savings = x_memory * 0.5
            recommendations.append(
                f"Convert main matrix X from float64 to float32 (save ~{format_bytes(potential_savings)})"
            )
    
    # Check for large uns data
    if 'uns_total' in memory_breakdown and memory_breakdown['uns_total'] > total_memory * 0.2:
        recommendations.append(
            "Large 'uns' data detected (>20% of total). Consider:\n" +
            "    - Removing intermediate analysis results\n" +
            "    - Storing large plots/figures separately\n" +
            "    - Cleaning up temporary computation data"
        )
    
    # Check for multiple large layers
    if 'layers_total' in memory_breakdown and len(adata.layers) > 3:
        if memory_breakdown['layers_total'] > memory_breakdown.get('X', 0):
            recommendations.append(
                "Layers consume more memory than main matrix. Consider:\n" +
                "    - Keeping only essential layers\n" +
                "    - Storing layers in separate files if rarely used"
            )
    
    # Check embedding precision
    high_precision_embeddings = []
    for key in adata.obsm.keys():
        if hasattr(adata.obsm[key], 'dtype') and adata.obsm[key].dtype == np.float64:
            high_precision_embeddings.append(key)
    
    if high_precision_embeddings:
        recommendations.append(
            f"Convert embeddings to float32: {', '.join(high_precision_embeddings)}"
        )
    
    # Print recommendations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    else:
        print("\nNo major optimization opportunities detected.")
    
    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)

import scanpy as sc
import numpy as np
from scipy.sparse import csc_matrix
import os
import time

def convert_X_to_sparse_csc(filepath: str, overwrite: bool = True) -> None:
    """
    Convert the dense .X matrix in an h5ad file to sparse CSC format.
    
    Parameters:
    -----------
    filepath : str
        Path to the h5ad file
    overwrite : bool
        If True, overwrites the original file. If False, creates a backup.
    """
    
    print("=" * 60)
    print("CONVERTING DENSE MATRIX TO SPARSE CSC FORMAT")
    print("=" * 60)
    
    # Get file info
    original_size = os.path.getsize(filepath)
    print(f"\nOriginal file: {filepath}")
    print(f"Original file size: {original_size / (1024**3):.2f} GB")
    
    # Create backup path if not overwriting
    if not overwrite:
        backup_path = filepath.replace('.h5ad', '_backup.h5ad')
        import shutil
        print(f"\nCreating backup at: {backup_path}")
        shutil.copy2(filepath, backup_path)
    
    # Load the file
    print("\nLoading file...")
    start_time = time.time()
    adata = sc.read_h5ad(filepath)
    load_time = time.time() - start_time
    print(f"File loaded in {load_time:.1f} seconds")
    
    # Check current X format
    print("\nCurrent .X format:")
    print(f"  Type: {type(adata.X)}")
    print(f"  Shape: {adata.X.shape}")
    print(f"  Data type: {adata.X.dtype}")
    
    if isinstance(adata.X, np.ndarray):
        # Calculate sparsity
        total_elements = adata.X.size
        zero_elements = np.sum(adata.X == 0)
        sparsity = zero_elements / total_elements
        print(f"  Sparsity: {sparsity:.1%} zeros")
        print(f"  Memory usage (estimated): {adata.X.nbytes / (1024**3):.2f} GB")
        
        # Convert to sparse CSC
        print("\nConverting to sparse CSC format...")
        start_time = time.time()
        
        # Convert to CSC matrix maintaining float64 dtype
        adata.X = csc_matrix(adata.X, dtype=np.float64)
        
        conversion_time = time.time() - start_time
        print(f"Conversion completed in {conversion_time:.1f} seconds")
        
        # Verify conversion
        print("\nNew .X format:")
        print(f"  Type: {type(adata.X)}")
        print(f"  Shape: {adata.X.shape}")
        print(f"  Data type: {adata.X.dtype}")
        print(f"  Number of stored values: {adata.X.nnz:,}")
        print(f"  Density: {(adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1])):.1%}")
        
        # Estimate memory usage of sparse matrix
        sparse_memory = (adata.X.data.nbytes + 
                        adata.X.indices.nbytes + 
                        adata.X.indptr.nbytes)
        print(f"  Memory usage (estimated): {sparse_memory / (1024**3):.2f} GB")
        
        # Calculate savings
        original_memory = total_elements * 8  # 8 bytes per float64
        savings_ratio = 1 - (sparse_memory / original_memory)
        print(f"\nEstimated memory savings: {savings_ratio:.1%}")
        
    else:
        print(f"\nWarning: .X is already in {type(adata.X)} format, not a dense numpy array.")
        print("No conversion needed.")
        return
    
    # Save the file
    print(f"\nSaving file to: {filepath}")
    start_time = time.time()
    
    # Save with compression for better disk space usage
    adata.write_h5ad(filepath, compression='gzip')
    
    save_time = time.time() - start_time
    print(f"File saved in {save_time:.1f} seconds")
    
    # Check new file size
    new_size = os.path.getsize(filepath)
    print(f"\nNew file size: {new_size / (1024**3):.2f} GB")
    size_reduction = (1 - new_size/original_size) * 100
    print(f"File size reduction: {size_reduction:.1f}%")
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    # Verification step
    print("\nVerifying saved file...")
    adata_verify = sc.read_h5ad(filepath)
    print(f"Verified .X type: {type(adata_verify.X)}")
    print(f"Verified .X dtype: {adata_verify.X.dtype}")
    print(f"Verified shape: {adata_verify.X.shape}")
    
    print("\n✅ File has been successfully converted and saved!")


# Main execution
if __name__ == "__main__":
    # Replace with your actual file path
    filepath = "/dcs07/hongkai/data/harry/result/multiomics/preprocess/atac_rna_integrated.h5ad"
    
    # Run the conversion (this will overwrite the original file)
    convert_X_to_sparse_csc(filepath, overwrite=True)
    
    # If you want to keep a backup, use:
    # convert_X_to_sparse_csc(filepath, overwrite=False)