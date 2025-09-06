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

if __name__ == "__main__":
    # fix_integrated_h5ad_with_obsm_varm(
    #     integrated_h5ad_path="/dcs07/hongkai/data/harry/result/multiomics/preprocess/atac_rna_integrated.h5ad",
    #     glue_dir="/dcs07/hongkai/data/harry/result/multiomics/integration/glue",
    #     raw_rna_path="/dcl01/hongkai/data/data/hjiang/Data/paired/rna/all.h5ad",
    #     output_path="/dcs07/hongkai/data/harry/result/multiomics/preprocess/atac_rna_integrated_fixed.h5ad",
    #     chunk_size=10000,  # Use chunked processing with 10K cells per chunk
    #     enable_chunking=True,  # Enable chunking (default)
    #     verbose=True
    # )

    filepath = "/dcs07/hongkai/data/harry/result/heart/multiomics/preprocess/adata_sample.h5ad"
    # adata = inspect_gene_activity(filepath)

    # Comprehensive inspection
    results = inspect_h5ad_file(filepath, verbose=True)
    
    # Quick inspection
    # quick_inspect_h5ad(filepath)
    
    # Access specific information from results dictionary
    # print(f"Number of cells: {results['basic_info']['n_obs']}")
    # print(f"Cell types: {results['observations']['categorical_info'].get('cell_type', {}).get('values', [])}")