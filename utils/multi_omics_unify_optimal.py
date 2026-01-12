import os
import scanpy as sc
import numpy as np
from typing import Optional, Union


def replace_optimal_dimension_reduction(
    base_path: str,
    expression_resolution_dir: Optional[str] = None,
    proportion_resolution_dir: Optional[str] = None,
    pseudobulk_path: Optional[str] = None,
    optimization_target: str = "rna",
    verbose: bool = True
) -> sc.AnnData:
    """
    Replaces dimension reduction results in pseudobulk_sample.h5ad with optimal results
    from resolution optimization for BOTH expression and proportion.
    
    This function:
    1. Loads the optimal expression and proportion h5ad files from Integration_optimization_{target}_{dr_type}/summary/optimal_{target}_{dr_type}.h5ad
    2. Loads the pseudobulk_sample.h5ad file
    3. Replaces X_DR_expression with optimal expression results
    4. Replaces X_DR_proportion with optimal proportion results
    5. Saves the updated pseudobulk_sample.h5ad
    
    Parameters:
    -----------
    base_path : str
        Base path to the output directory
        Example: '/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_25_sample/multiomics'
    expression_resolution_dir : str, optional
        Path to the expression resolution optimization directory
        If None, defaults to base_path/resolution_optimization_expression
    proportion_resolution_dir : str, optional
        Path to the proportion resolution optimization directory
        If None, defaults to base_path/resolution_optimization_proportion
    pseudobulk_path : str, optional
        Path to the pseudobulk_sample.h5ad file
        If None, defaults to base_path/pseudobulk/pseudobulk_sample.h5ad
    optimization_target : str, default "rna"
        Optimization target ('rna' or 'atac') - used to construct filenames
    verbose : bool, default True
        Whether to print verbose output
        
    Returns:
    --------
    sc.AnnData
        Updated pseudobulk AnnData with optimal embeddings
        
    Example:
    --------
    >>> # Using default paths
    >>> adata = replace_optimal_dimension_reduction(
    ...     base_path='/path/to/multiomics_output',
    ...     optimization_target='rna'
    ... )
    
    >>> # Using custom paths
    >>> adata = replace_optimal_dimension_reduction(
    ...     base_path='/path/to/multiomics_output',
    ...     expression_resolution_dir='/custom/expression/path',
    ...     proportion_resolution_dir='/custom/proportion/path',
    ...     pseudobulk_path='/custom/pseudobulk/path.h5ad',
    ...     optimization_target='atac'
    ... )
    """
    
    # Construct default file paths if not provided
    if expression_resolution_dir is None:
        expression_resolution_dir = os.path.join(
            base_path, 
            "resolution_optimization_expression"
        )
    
    if proportion_resolution_dir is None:
        proportion_resolution_dir = os.path.join(
            base_path, 
            "resolution_optimization_proportion"
        )
    
    if pseudobulk_path is None:
        pseudobulk_path = os.path.join(
            base_path, 
            "pseudobulk", 
            "pseudobulk_sample.h5ad"
        )
    
    # Construct paths to optimal h5ad files with the correct directory structure
    # The actual structure is: resolution_optimization_{dr_type}/Integration_optimization_{target}_{dr_type}/summary/optimal_{target}_{dr_type}.h5ad
    optimal_expression_path = os.path.join(
        expression_resolution_dir,
        f"Integration_optimization_{optimization_target}_expression",
        "summary", 
        f"optimal_{optimization_target}_expression.h5ad"
    )
    
    optimal_proportion_path = os.path.join(
        proportion_resolution_dir,
        f"Integration_optimization_{optimization_target}_proportion",
        "summary", 
        f"optimal_{optimization_target}_proportion.h5ad"
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Replacing Dimension Reduction with Optimal Results")
        print(f"{'='*70}")
        print(f"\n[Replace DR] Configuration:")
        print(f"  - Optimization target: {optimization_target.upper()}")
        print(f"\n[Replace DR] File paths:")
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
        error_msg = "\n[ERROR] The following required files were not found:\n"
        error_msg += "\n".join([f"  ✗ {f}" for f in missing_files])
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
        raise RuntimeError(f"[ERROR] Failed to load h5ad files: {str(e)}")
    
    # Verify sample alignment
    if not np.array_equal(optimal_expression.obs.index, pseudobulk_sample.obs.index):
        raise ValueError("[ERROR] Sample indices in optimal_expression do not match pseudobulk_sample")
    if not np.array_equal(optimal_proportion.obs.index, pseudobulk_sample.obs.index):
        raise ValueError("[ERROR] Sample indices in optimal_proportion do not match pseudobulk_sample")
    
    # Replace X_DR_expression from optimal_expression
    if verbose:
        print(f"\n[Replace DR] Replacing EXPRESSION dimension reduction results...")
    
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
                elif hasattr(optimal_expression.uns[key], '__len__'):
                    shape_info = f" (length: {len(optimal_expression.uns[key])})"
                print(f"  ✓ Copied .uns['{key}']{shape_info}")
    
    for key in expression_obsm_keys:
        if key in optimal_expression.obsm:
            pseudobulk_sample.obsm[key] = optimal_expression.obsm[key].copy()
            copied_expression_count += 1
            if verbose:
                print(f"  ✓ Copied .obsm['{key}'] (shape: {optimal_expression.obsm[key].shape})")
    
    if copied_expression_count == 0:
        print(f"  ⚠ Warning: No expression DR keys found in optimal_expression file")
    elif verbose:
        print(f"  → Total expression keys copied: {copied_expression_count}")
    
    # Replace X_DR_proportion from optimal_proportion
    if verbose:
        print(f"\n[Replace DR] Replacing PROPORTION dimension reduction results...")
    
    # Keys to copy for proportion DR
    proportion_uns_keys = [
        'X_DR_proportion',
        'X_DR_proportion_variance',
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
                elif hasattr(optimal_proportion.uns[key], '__len__'):
                    shape_info = f" (length: {len(optimal_proportion.uns[key])})"
                print(f"  ✓ Copied .uns['{key}']{shape_info}")
    
    for key in proportion_obsm_keys:
        if key in optimal_proportion.obsm:
            pseudobulk_sample.obsm[key] = optimal_proportion.obsm[key].copy()
            copied_proportion_count += 1
            if verbose:
                print(f"  ✓ Copied .obsm['{key}'] (shape: {optimal_proportion.obsm[key].shape})")
    
    if copied_proportion_count == 0:
        print(f"  ⚠ Warning: No proportion DR keys found in optimal_proportion file")
    elif verbose:
        print(f"  → Total proportion keys copied: {copied_proportion_count}")
    
    # Save the updated pseudobulk_sample
    try:
        if verbose:
            print(f"\n[Replace DR] Saving updated pseudobulk_sample...")
            print(f"  Destination: {pseudobulk_path}")
        
        # Create backup of original file
        backup_path = pseudobulk_path + ".backup"
        if os.path.exists(backup_path):
            if verbose:
                print(f"  Note: Backup file already exists: {backup_path}")
        else:
            import shutil
            shutil.copy2(pseudobulk_path, backup_path)
            if verbose:
                print(f"  ✓ Created backup: {backup_path}")
        
        # Write updated file
        pseudobulk_sample.write_h5ad(pseudobulk_path)
        
        if os.path.exists(pseudobulk_path):
            file_size = os.path.getsize(pseudobulk_path)
            if verbose:
                print(f"  ✓ Successfully saved ({file_size / (1024*1024):.1f} MB)")
                print(f"\n{'-'*70}")
                print(f"SUMMARY")
                print(f"{'-'*70}")
                print(f"  Optimization target: {optimization_target.upper()}")
                print(f"  Expression DR keys copied: {copied_expression_count}")
                print(f"  Proportion DR keys copied: {copied_proportion_count}")
                print(f"  Total keys updated: {copied_expression_count + copied_proportion_count}")
                print(f"  File updated: {pseudobulk_path}")
                print(f"  Backup saved: {backup_path}")
                print(f"{'='*70}\n")
        else:
            raise RuntimeError("File was not created after save operation")
    
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to save updated pseudobulk file: {str(e)}")
    
    return pseudobulk_sample


def verify_optimal_embeddings(
    pseudobulk_path: str,
    verbose: bool = True
) -> dict:
    """
    Verify that the pseudobulk file contains the expected optimal embeddings.
    
    Parameters:
    -----------
    pseudobulk_path : str
        Path to the pseudobulk_sample.h5ad file
    verbose : bool, default True
        Whether to print verification results
        
    Returns:
    --------
    dict
        Dictionary containing verification results with keys:
        - 'has_expression_dr': bool
        - 'has_proportion_dr': bool
        - 'expression_shape': tuple or None
        - 'proportion_shape': tuple or None
        - 'missing_keys': list
        
    Example:
    --------
    >>> results = verify_optimal_embeddings(
    ...     pseudobulk_path='/path/to/pseudobulk_sample.h5ad'
    ... )
    >>> print(f"Expression DR present: {results['has_expression_dr']}")
    """
    
    if not os.path.exists(pseudobulk_path):
        raise FileNotFoundError(f"Pseudobulk file not found: {pseudobulk_path}")
    
    adata = sc.read_h5ad(pseudobulk_path)
    
    # Check for required keys
    required_obsm_keys = ['X_DR_expression', 'X_DR_proportion']
    required_uns_keys = [
        'X_DR_expression_variance_ratio',
        'X_DR_proportion_variance_ratio'
    ]
    
    results = {
        'has_expression_dr': 'X_DR_expression' in adata.obsm,
        'has_proportion_dr': 'X_DR_proportion' in adata.obsm,
        'expression_shape': adata.obsm.get('X_DR_expression', np.array([])).shape if 'X_DR_expression' in adata.obsm else None,
        'proportion_shape': adata.obsm.get('X_DR_proportion', np.array([])).shape if 'X_DR_proportion' in adata.obsm else None,
        'missing_keys': []
    }
    
    # Check for missing keys
    for key in required_obsm_keys:
        if key not in adata.obsm:
            results['missing_keys'].append(f"obsm['{key}']")
    
    for key in required_uns_keys:
        if key not in adata.uns:
            results['missing_keys'].append(f"uns['{key}']")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Verification Results for: {os.path.basename(pseudobulk_path)}")
        print(f"{'='*70}")
        print(f"Expression DR present: {results['has_expression_dr']}")
        if results['expression_shape']:
            print(f"  Shape: {results['expression_shape']}")
        print(f"\nProportion DR present: {results['has_proportion_dr']}")
        if results['proportion_shape']:
            print(f"  Shape: {results['proportion_shape']}")
        
        if results['missing_keys']:
            print(f"\n⚠ Missing keys:")
            for key in results['missing_keys']:
                print(f"  - {key}")
        else:
            print(f"\n✓ All required keys present")
        print(f"{'='*70}\n")
    
    return results