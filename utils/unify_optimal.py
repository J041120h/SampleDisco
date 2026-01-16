import os
import scanpy as sc
import pandas as pd
import numpy as np


def replace_optimal_dimension_reduction(
    base_path: str,
    verbose: bool = True
) -> None:
    """
    Replaces dimension reduction results in pseudobulk_sample.h5ad with optimal results
    from resolution optimization.
    
    This function:
    1. Loads the optimal expression and proportion h5ad files
    2. Loads the pseudobulk_sample.h5ad file
    3. Replaces X_DR_expression with optimal expression results
    4. Replaces X_DR_proportion with optimal proportion results
    5. Saves the updated pseudobulk_sample.h5ad
    6. Updates the unified embedding CSV files in <base_path>/embeddings
       (sample_expression_embedding.csv, sample_proportion_embedding.csv)
    
    Parameters:
    -----------
    base_path : str
        Base path to the output directory 
        Example: '/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_25_sample/rna'
    verbose : bool, default True
        Whether to print verbose output
        
    Example:
    --------
    >>> replace_optimal_dimension_reduction(
    ...     base_path='/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_25_sample/rna'
    ... )
    """
    
    # Construct file paths based on the pattern provided
    optimal_expression_path = os.path.join(
        base_path, 
        "RNA_resolution_optimization_expression", 
        "summary", 
        "optimal.h5ad"
    )
    
    optimal_proportion_path = os.path.join(
        base_path, 
        "RNA_resolution_optimization_proportion", 
        "summary", 
        "optimal.h5ad"
    )
    
    pseudobulk_path = os.path.join(
        base_path, 
        "pseudobulk", 
        "pseudobulk_sample.h5ad"
    )
    
    if verbose:
        print(f"[Replace DR] Starting dimension reduction replacement...")
        print(f"[Replace DR] File paths:")
        print(f"  - Optimal expression: {optimal_expression_path}")
        print(f"  - Optimal proportion: {optimal_proportion_path}")
        print(f"  - Pseudobulk sample: {pseudobulk_path}")
    
    # Check if files exist
    missing_files = []
    if not os.path.exists(optimal_expression_path):
        missing_files.append(f"Optimal expression: {optimal_expression_path}")
    if not os.path.exists(optimal_proportion_path):
        missing_files.append(f"Optimal proportion: {optimal_proportion_path}")
    if not os.path.exists(pseudobulk_path):
        missing_files.append(f"Pseudobulk: {pseudobulk_path}")
    
    if missing_files:
        error_msg = "The following files were not found:\n" + "\n".join([f"  - {f}" for f in missing_files])
        raise FileNotFoundError(error_msg)
    
    # Load all three files
    try:
        if verbose:
            print(f"\n[Replace DR] Loading files...")
        
        optimal_expression = sc.read_h5ad(optimal_expression_path)
        optimal_proportion = sc.read_h5ad(optimal_proportion_path)
        pseudobulk_sample = sc.read_h5ad(pseudobulk_path)
        
        if verbose:
            print(f"  ✓ Optimal expression loaded: {optimal_expression.shape}")
            print(f"  ✓ Optimal proportion loaded: {optimal_proportion.shape}")
            print(f"  ✓ Pseudobulk sample loaded: {pseudobulk_sample.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load h5ad files: {str(e)}")
    
    # Replace X_DR_expression from optimal_expression
    if verbose:
        print(f"\n[Replace DR] Replacing expression dimension reduction results...")
    
    # Keys to copy for expression DR
    expression_uns_keys = [
        'X_DR_expression',
        'X_DR_expression_variance',
        'X_DR_expression_variance_ratio',
        'X_pca_expression_method',
        'X_lsi_expression_method',
        'X_spectral_expression_method'
    ]
    
    expression_obsm_keys = [
        'X_DR_expression',
        'X_pca_expression_method',
        'X_lsi_expression_method',
        'X_spectral_expression_method'
    ]
    
    copied_expression_count = 0
    for key in expression_uns_keys:
        if key in optimal_expression.uns:
            pseudobulk_sample.uns[key] = optimal_expression.uns[key].copy()
            copied_expression_count += 1
            if verbose:
                shape_info = ""
                if hasattr(optimal_expression.uns[key], 'shape'):
                    shape_info = f" (shape: {optimal_expression.uns[key].shape})"
                print(f"  ✓ Copied .uns['{key}']{shape_info}")
    
    for key in expression_obsm_keys:
        if key in optimal_expression.obsm:
            pseudobulk_sample.obsm[key] = optimal_expression.obsm[key].copy()
            copied_expression_count += 1
            if verbose:
                print(f"  ✓ Copied .obsm['{key}'] (shape: {optimal_expression.obsm[key].shape})")
    
    if copied_expression_count == 0:
        print(f"  ⚠ Warning: No expression DR keys found in optimal_expression file")
    
    # Replace X_DR_proportion from optimal_proportion
    if verbose:
        print(f"\n[Replace DR] Replacing proportion dimension reduction results...")
    
    # Keys to copy for proportion DR
    proportion_uns_keys = [
        'X_DR_proportion',
        'X_DR_proportion_variance_ratio',
        'pca_proportion_variance_ratio'
    ]
    
    proportion_obsm_keys = [
        'X_DR_proportion',
        'X_pca_proportion'
    ]
    
    copied_proportion_count = 0
    for key in proportion_uns_keys:
        if key in optimal_proportion.uns:
            pseudobulk_sample.uns[key] = optimal_proportion.uns[key].copy()
            copied_proportion_count += 1
            if verbose:
                shape_info = ""
                if hasattr(optimal_proportion.uns[key], 'shape'):
                    shape_info = f" (shape: {optimal_proportion.uns[key].shape})"
                print(f"  ✓ Copied .uns['{key}']{shape_info}")
    
    for key in proportion_obsm_keys:
        if key in optimal_proportion.obsm:
            pseudobulk_sample.obsm[key] = optimal_proportion.obsm[key].copy()
            copied_proportion_count += 1
            if verbose:
                print(f"  ✓ Copied .obsm['{key}'] (shape: {optimal_proportion.obsm[key].shape})")
    
    if copied_proportion_count == 0:
        print(f"  ⚠ Warning: No proportion DR keys found in optimal_proportion file")
    
    # Save the updated pseudobulk_sample
    try:
        if verbose:
            print(f"\n[Replace DR] Saving updated pseudobulk_sample...")
            print(f"  Destination: {pseudobulk_path}")
        
        sc.write(pseudobulk_path, pseudobulk_sample)
        
        if os.path.exists(pseudobulk_path):
            file_size = os.path.getsize(pseudobulk_path)
            if verbose:
                print(f"  ✓ Successfully saved ({file_size / (1024*1024):.1f} MB)")
                print(f"\n[Replace DR] === SUMMARY ===")
                print(f"  Expression DR keys copied: {copied_expression_count}")
                print(f"  Proportion DR keys copied: {copied_proportion_count}")
                print(f"  File updated: {pseudobulk_path}")
        else:
            raise RuntimeError("File was not created after save operation")
    
    except Exception as e:
        raise RuntimeError(f"Failed to save updated pseudobulk file: {str(e)}")
    
    # -------------------------------------------------
    # NEW: Update CSV embeddings like the main pipeline
    # -------------------------------------------------
    try:
        if verbose:
            print(f"\n[Replace DR] Updating embedding CSV files...")
        
        embedding_dir = os.path.join(base_path, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
        
        def _save_embedding_csv_from_uns(uns_key: str, filename: str, desc: str) -> bool:
            if uns_key not in pseudobulk_sample.uns:
                if verbose:
                    print(f"  ⚠ {desc} not found in pseudobulk_sample.uns['{uns_key}']; skipping CSV update")
                return False
            
            data = pseudobulk_sample.uns[uns_key]
            
            # Convert to DataFrame
            if isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                try:
                    df = pd.DataFrame(data)
                except Exception:
                    if verbose:
                        print(f"  ⚠ Skipping {desc}: could not convert type {type(data)} to DataFrame")
                    return False
            
            out_path = os.path.join(embedding_dir, filename)
            df.to_csv(out_path)
            
            if verbose:
                print(f"  ✓ Saved {desc} to {out_path} (shape: {df.shape})")
            return True
        
        _save_embedding_csv_from_uns(
            "X_DR_expression",
            "sample_expression_embedding.csv",
            "expression embedding"
        )
        _save_embedding_csv_from_uns(
            "X_DR_proportion",
            "sample_proportion_embedding.csv",
            "proportion embedding"
        )
    
    except Exception as e:
        if verbose:
            print(f"  ⚠ Failed to update embedding CSV files: {e}")
    
    return pseudobulk_sample
