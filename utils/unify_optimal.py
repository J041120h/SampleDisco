import os
import scanpy as sc
import pandas as pd
import numpy as np


def replace_optimal_dimension_reduction(
    base_path: str,
    verbose: bool = True
) -> sc.AnnData:
    """
    Replaces dimension reduction results AND pseudobulk expression in
    pseudobulk_sample.h5ad with optimal results from resolution optimization.
    
    This function:
    1. Loads the optimal expression and proportion h5ad files
    2. Loads the pseudobulk_sample.h5ad file
    3. Replaces the pseudobulk expression matrix (X and layers) with the one
       from the optimal expression AnnData (after checking sample/gene alignment)
    4. Replaces X_DR_expression with optimal expression DR results
    5. Replaces X_DR_proportion with optimal proportion DR results
    6. Saves the updated pseudobulk_sample.h5ad (with a backup)
    7. Updates the unified embedding CSV files in <base_path>/embeddings
       (sample_expression_embedding.csv, sample_proportion_embedding.csv)
    
    Parameters
    ----------
    base_path : str
        Base path to the output directory 
        Example: '/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_25_sample/rna'
    verbose : bool, default True
        Whether to print verbose output
        
    Returns
    -------
    sc.AnnData
        Updated pseudobulk AnnData with optimal expression and embeddings
    
    Example
    -------
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
        print(f"[Replace DR] Starting dimension reduction + expression replacement...")
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
    
    # ------------------------------------------------------------------
    # Safety: check sample alignment between optimal and base pseudobulk
    # ------------------------------------------------------------------
    if not np.array_equal(optimal_expression.obs.index, pseudobulk_sample.obs.index):
        raise ValueError(
            "[ERROR] Sample indices in optimal_expression do not match pseudobulk_sample"
        )
    if not np.array_equal(optimal_proportion.obs.index, pseudobulk_sample.obs.index):
        raise ValueError(
            "[ERROR] Sample indices in optimal_proportion do not match pseudobulk_sample"
        )
    
    # ------------------------------------------------------------------
    # NEW: Replace pseudobulk expression matrix with optimal expression
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[Replace DR] Replacing pseudobulk EXPRESSION matrix with optimal expression...")
    
    # Require gene (var) identity + order to match for safe replacement
    if not np.array_equal(optimal_expression.var.index, pseudobulk_sample.var.index):
        msg = (
            "[ERROR] Gene (var) indices in optimal_expression do not match pseudobulk_sample.\n"
            "        Cannot safely replace expression matrix.\n"
            f"        optimal_expression genes: {optimal_expression.var.shape[0]}\n"
            f"        pseudobulk_sample genes:  {pseudobulk_sample.var.shape[0]}"
        )
        raise ValueError(msg)
    
    # Replace the main expression matrix
    pseudobulk_sample.X = optimal_expression.X.copy()
    if verbose:
        print(f"  ✓ Replaced pseudobulk_sample.X with optimal_expression.X "
              f"(shape: {pseudobulk_sample.X.shape})")
    
    # Copy layers from optimal_expression (if any)
    if len(optimal_expression.layers) > 0:
        for layer_name, layer_data in optimal_expression.layers.items():
            pseudobulk_sample.layers[layer_name] = layer_data.copy()
            if verbose:
                print(f"  ✓ Copied layer '{layer_name}' "
                      f"(shape: {layer_data.shape}) from optimal_expression")
    else:
        if verbose:
            print("  • No layers found in optimal_expression; skipping layer copy")
    
    # Copy var annotations from optimal_expression
    if optimal_expression.var.shape[1] > 0:
        for col in optimal_expression.var.columns:
            pseudobulk_sample.var[col] = optimal_expression.var[col].copy()
        if verbose:
            print(f"  ✓ Copied var annotations from optimal_expression "
                  f"(columns: {list(optimal_expression.var.columns)})")
    else:
        if verbose:
            print("  • No additional var annotations in optimal_expression; skipping var copy")
    
    # ------------------------------------------------------------------
    # Replace X_DR_expression from optimal_expression
    # ------------------------------------------------------------------
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
    
    # ------------------------------------------------------------------
    # Replace X_DR_proportion from optimal_proportion
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[Replace DR] Replacing PROPORTION dimension reduction results...")
    
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
    
    # ------------------------------------------------------------------
    # Save the updated pseudobulk_sample (with backup)
    # ------------------------------------------------------------------
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
                print(f"\n[Replace DR] === SUMMARY ===")
                print(f"  Expression DR keys copied:  {copied_expression_count}")
                print(f"  Proportion DR keys copied: {copied_proportion_count}")
                print(f"  Expression matrix replaced: True")
                print(f"  File updated: {pseudobulk_path}")
                print(f"  Backup saved: {backup_path}")
        else:
            raise RuntimeError("File was not created after save operation")
    
    except Exception as e:
        raise RuntimeError(f"Failed to save updated pseudobulk file: {str(e)}")
    
    # -------------------------------------------------
    # Update CSV embeddings like the main pipeline
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
