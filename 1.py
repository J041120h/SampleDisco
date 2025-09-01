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


# Example usage:
if __name__ == "__main__":
    # Replace with your h5ad file path
    filepath = "/dcl01/hongkai/data/data/hjiang/Test/gene_activity/rna_gene_id.h5ad"
    
    # Comprehensive inspection
    results = inspect_h5ad_file(filepath, verbose=True)
    
    # Quick inspection
    # quick_inspect_h5ad(filepath)
    
    # Access specific information from results dictionary
    # print(f"Number of cells: {results['basic_info']['n_obs']}")
    # print(f"Cell types: {results['observations']['categorical_info'].get('cell_type', {}).get('values', [])}")