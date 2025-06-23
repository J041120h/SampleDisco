import os
import anndata as ad
from CCA_test import *
from CCA import *

def integration_CCA_test(pseudobulk_anndata_path,
                        output_dir,
                        sev_col="sev.level",
                        num_simulations=1000,
                        ptime=True,
                        verbose=False):
    """
    Perform CCA integration test on pseudobulk data with both RNA and ATAC modalities.
    Each modality is processed separately with its own CCA analysis and p-value tests.
    
    Parameters:
    -----------
    pseudobulk_anndata_path : str
        Path to the pseudobulk AnnData file (.h5ad)
    output_dir : str
        Base output directory for results
    sev_col : str, default "severity"
        Column name for severity/condition in the data
    num_simulations : int, default 1000
        Number of simulations for p-value testing
    ptime : bool, default True
        Whether to compute pseudotime
    verbose : bool, default False
        Whether to print verbose output
        
    Returns:
    --------
    dict
        Dictionary containing CCA results and p-values for both modalities
    """
    
    # Create output directories
    output_dir = os.path.join(output_dir, "CCA_test")
    rna_output_dir = os.path.join(output_dir, "RNA")
    atac_output_dir = os.path.join(output_dir, "ATAC")
    
    # Create subdirectories for proportion and expression tests
    rna_proportion_dir = os.path.join(rna_output_dir, "proportion")
    rna_expression_dir = os.path.join(rna_output_dir, "expression")
    atac_proportion_dir = os.path.join(atac_output_dir, "proportion")
    atac_expression_dir = os.path.join(atac_output_dir, "expression")
    
    for dir_path in [output_dir, rna_output_dir, atac_output_dir, 
                     rna_proportion_dir, rna_expression_dir, 
                     atac_proportion_dir, atac_expression_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Load data
    try:
        pseudobulk_anndata = ad.read_h5ad(pseudobulk_anndata_path)
    except Exception as e:
        raise FileNotFoundError(f"Could not load AnnData file: {e}")
    
    # Validate required columns
    if 'modality' not in pseudobulk_anndata.obs.columns:
        raise ValueError("'modality' column not found in pseudobulk_anndata.obs")
    
    if sev_col not in pseudobulk_anndata.obs.columns:
        raise ValueError(f"Severity column '{sev_col}' not found in pseudobulk_anndata.obs")
    
    # Filter data by modality
    rna_mask = pseudobulk_anndata.obs['modality'] == 'RNA'
    atac_mask = pseudobulk_anndata.obs['modality'] == 'ATAC'
    
    # Get indices for proper PCA coordinate filtering
    rna_indices = pseudobulk_anndata.obs.index[rna_mask]
    atac_indices = pseudobulk_anndata.obs.index[atac_mask]
    
    # Get positions in the original data for PCA coordinate filtering
    rna_positions = [i for i, idx in enumerate(pseudobulk_anndata.obs.index) if idx in rna_indices]
    atac_positions = [i for i, idx in enumerate(pseudobulk_anndata.obs.index) if idx in atac_indices]
    
    # Filter RNA and ATAC data - AnnData filtering automatically handles obsm
    pseudobulk_rna = pseudobulk_anndata[rna_mask].copy()
    pseudobulk_atac = pseudobulk_anndata[atac_mask].copy()
    
    # Fix PCA coordinates in .uns to match filtered samples
    def filter_pca_coordinates(adata, positions, modality_name):
        """Filter PCA coordinates in .uns to match the samples in the filtered data"""
        import pandas as pd
        import numpy as np
        
        pca_keys = ["X_DR_expression", "X_DR_proportion"]
        
        for key in pca_keys:
            if key in adata.uns:
                original_shape = adata.uns[key].shape
                if original_shape[0] != adata.n_obs:
                    # Handle different data types
                    if isinstance(adata.uns[key], pd.DataFrame):
                        # pandas DataFrame - use iloc for row indexing
                        adata.uns[key] = adata.uns[key].iloc[positions]
                        if verbose:
                            print(f"{modality_name} {key}: filtered DataFrame from {original_shape} to {adata.uns[key].shape}")
                    elif isinstance(adata.uns[key], np.ndarray):
                        # numpy array
                        adata.uns[key] = adata.uns[key][positions]
                        if verbose:
                            print(f"{modality_name} {key}: filtered numpy array from {original_shape} to {adata.uns[key].shape}")
                    else:
                        # Try generic indexing (works for most array-like objects)
                        try:
                            adata.uns[key] = adata.uns[key][positions]
                            if verbose:
                                print(f"{modality_name} {key}: filtered {type(adata.uns[key])} from {original_shape} to {adata.uns[key].shape}")
                        except Exception as e:
                            print(f"Warning: Could not filter {modality_name} {key}: {e}")
                            print(f"  Data type: {type(adata.uns[key])}")
                            print(f"  Shape: {adata.uns[key].shape}")
                else:
                    if verbose:
                        print(f"{modality_name} {key}: already correct shape {original_shape}")
            else:
                if verbose:
                    print(f"Warning: {key} not found in {modality_name} .uns")
        
        return adata
    
    # Filter PCA coordinates for each modality
    pseudobulk_rna = filter_pca_coordinates(pseudobulk_rna, rna_positions, "RNA")
    pseudobulk_atac = filter_pca_coordinates(pseudobulk_atac, atac_positions, "ATAC")
    
    if verbose:
        print(f"RNA samples: {pseudobulk_rna.n_obs}")
        print(f"ATAC samples: {pseudobulk_atac.n_obs}")
        print(f"RNA obsm shapes: {[(k, v.shape) for k, v in pseudobulk_rna.obsm.items()]}")
        print(f"ATAC obsm shapes: {[(k, v.shape) for k, v in pseudobulk_atac.obsm.items()]}")
        
        # Check PCA coordinates in .uns
        for modality_name, adata in [("RNA", pseudobulk_rna), ("ATAC", pseudobulk_atac)]:
            print(f"\n{modality_name} PCA coordinates in .uns:")
            for key in ["X_DR_expression", "X_DR_proportion"]:
                if key in adata.uns:
                    print(f"  {key} shape: {adata.uns[key].shape}")
                else:
                    print(f"  {key}: not found")
        
        # Debug: Check if obsm matrices have correct dimensions
        for modality_name, adata in [("RNA", pseudobulk_rna), ("ATAC", pseudobulk_atac)]:
            print(f"\n{modality_name} validation:")
            print(f"  n_obs: {adata.n_obs}")
            print(f"  severity column length: {len(adata.obs[sev_col])}")
            for key, matrix in adata.obsm.items():
                print(f"  {key} shape: {matrix.shape}")
                if matrix.shape[0] != adata.n_obs:
                    print(f"    WARNING: {key} has {matrix.shape[0]} rows but should have {adata.n_obs}")
    
    # Validate that we have both modalities
    if pseudobulk_rna.n_obs == 0:
        raise ValueError("No RNA samples found in the data")
    if pseudobulk_atac.n_obs == 0:
        raise ValueError("No ATAC samples found in the data")
    
    # Additional validation: Check obsm consistency
    def validate_obsm_consistency(adata, modality_name):
        """Validate that obsm matrices have consistent dimensions with the filtered data"""
        inconsistent_keys = []
        for key, matrix in adata.obsm.items():
            if matrix.shape[0] != adata.n_obs:
                inconsistent_keys.append((key, matrix.shape[0], adata.n_obs))
        
        if inconsistent_keys:
            print(f"Warning: {modality_name} has inconsistent obsm matrices:")
            for key, matrix_rows, expected_rows in inconsistent_keys:
                print(f"  {key}: {matrix_rows} rows, expected {expected_rows}")
            
            # Fix inconsistent obsm matrices by removing them or fixing them
            # Option 1: Remove inconsistent matrices (conservative approach)
            for key, _, _ in inconsistent_keys:
                print(f"  Removing inconsistent matrix: {key}")
                del adata.obsm[key]
        
        return adata
    
    # Validate and fix obsm consistency
    pseudobulk_rna = validate_obsm_consistency(pseudobulk_rna, "RNA")
    pseudobulk_atac = validate_obsm_consistency(pseudobulk_atac, "ATAC")
    
    # Final validation: Check that PCA coordinates match sample counts
    def validate_pca_coordinates(adata, modality_name):
        """Final validation that PCA coordinates match the number of samples"""
        pca_keys = ["X_DR_expression", "X_DR_proportion"]
        for key in pca_keys:
            if key in adata.uns:
                if adata.uns[key].shape[0] != adata.n_obs:
                    raise ValueError(f"{modality_name} {key} has {adata.uns[key].shape[0]} rows but should have {adata.n_obs}")
                else:
                    if verbose:
                        print(f"✓ {modality_name} {key} validation passed: {adata.uns[key].shape[0]} rows")
    
    validate_pca_coordinates(pseudobulk_rna, "RNA")
    validate_pca_coordinates(pseudobulk_atac, "ATAC")
    
    # Initialize results dictionary
    results = {
        'rna_results': {
            'cca_scores': {},
            'pvalue_tests': {}
        },
        'atac_results': {
            'cca_scores': {},
            'pvalue_tests': {}
        }
    }
    
    # Process RNA modality
    if verbose:
        print("\n=== Processing RNA modality ===")
    
    try:
        rna_first_component_score_proportion, rna_first_component_score_expression, rna_ptime_proportion, rna_ptime_expression = CCA_Call(
            adata=pseudobulk_rna, 
            output_dir=rna_output_dir, 
            sev_col=sev_col, 
            ptime=ptime, 
            verbose=verbose
        )
        
        # Store RNA CCA results
        results['rna_results']['cca_scores'] = {
            'first_component_score_proportion': rna_first_component_score_proportion,
            'first_component_score_expression': rna_first_component_score_expression,
            'ptime_proportion': rna_ptime_proportion,
            'ptime_expression': rna_ptime_expression
        }
        
        if verbose:
            print("RNA CCA analysis completed successfully")
            
    except Exception as e:
        print(f"Error: RNA CCA analysis failed: {e}")
        results['rna_results']['cca_scores'] = None
        rna_first_component_score_proportion = None
        rna_first_component_score_expression = None
    
    # Process ATAC modality
    if verbose:
        print("\n=== Processing ATAC modality ===")
    
    try:
        atac_first_component_score_proportion, atac_first_component_score_expression, atac_ptime_proportion, atac_ptime_expression = CCA_Call(
            adata=pseudobulk_atac, 
            output_dir=atac_output_dir, 
            sev_col=sev_col, 
            ptime=ptime, 
            verbose=verbose
        )
        
        # Store ATAC CCA results
        results['atac_results']['cca_scores'] = {
            'first_component_score_proportion': atac_first_component_score_proportion,
            'first_component_score_expression': atac_first_component_score_expression,
            'ptime_proportion': atac_ptime_proportion,
            'ptime_expression': atac_ptime_expression
        }
        
        if verbose:
            print("ATAC CCA analysis completed successfully")
            
    except Exception as e:
        print(f"Error: ATAC CCA analysis failed: {e}")
        results['atac_results']['cca_scores'] = None
        atac_first_component_score_proportion = None
        atac_first_component_score_expression = None
    
    # RNA p-value tests
    if verbose:
        print("\n=== Running RNA p-value tests ===")
    
    # RNA proportion test
    if rna_first_component_score_proportion is not None:
        try:
            rna_proportion_pvalue = cca_pvalue_test(
                pseudo_adata=pseudobulk_rna,
                column="X_DR_proportion",
                input_correlation=rna_first_component_score_proportion,
                output_directory=rna_proportion_dir,
                num_simulations=num_simulations,
                sev_col=sev_col,
                verbose=verbose
            )
            results['rna_results']['pvalue_tests']['proportion'] = rna_proportion_pvalue
            
            if verbose:
                print(f"RNA proportion p-value test completed. Results saved to {rna_proportion_dir}")
                
        except Exception as e:
            print(f"Warning: RNA proportion p-value test failed: {e}")
            results['rna_results']['pvalue_tests']['proportion'] = None
    
    # RNA expression test
    if rna_first_component_score_expression is not None:
        try:
            rna_expression_pvalue = cca_pvalue_test(
                pseudo_adata=pseudobulk_rna,
                column="X_DR_expression",
                input_correlation=rna_first_component_score_expression,
                output_directory=rna_expression_dir,
                num_simulations=num_simulations,
                sev_col=sev_col,
                verbose=verbose
            )
            results['rna_results']['pvalue_tests']['expression'] = rna_expression_pvalue
            
            if verbose:
                print(f"RNA expression p-value test completed. Results saved to {rna_expression_dir}")
                
        except Exception as e:
            print(f"Warning: RNA expression p-value test failed: {e}")
            results['rna_results']['pvalue_tests']['expression'] = None
    
    # ATAC p-value tests
    if verbose:
        print("\n=== Running ATAC p-value tests ===")
    
    # ATAC proportion test
    if atac_first_component_score_proportion is not None:
        try:
            atac_proportion_pvalue = cca_pvalue_test(
                pseudo_adata=pseudobulk_atac,
                column="X_DR_proportion",
                input_correlation=atac_first_component_score_proportion,
                output_directory=atac_proportion_dir,
                num_simulations=num_simulations,
                sev_col=sev_col,
                verbose=verbose
            )
            results['atac_results']['pvalue_tests']['proportion'] = atac_proportion_pvalue
            
            if verbose:
                print(f"ATAC proportion p-value test completed. Results saved to {atac_proportion_dir}")
                
        except Exception as e:
            print(f"Warning: ATAC proportion p-value test failed: {e}")
            results['atac_results']['pvalue_tests']['proportion'] = None
    
    # ATAC expression test
    if atac_first_component_score_expression is not None:
        try:
            atac_expression_pvalue = cca_pvalue_test(
                pseudo_adata=pseudobulk_atac,
                column="X_DR_expression",
                input_correlation=atac_first_component_score_expression,
                output_directory=atac_expression_dir,
                num_simulations=num_simulations,
                sev_col=sev_col,
                verbose=verbose
            )
            results['atac_results']['pvalue_tests']['expression'] = atac_expression_pvalue
            
            if verbose:
                print(f"ATAC expression p-value test completed. Results saved to {atac_expression_dir}")
                
        except Exception as e:
            print(f"Warning: ATAC expression p-value test failed: {e}")
            results['atac_results']['pvalue_tests']['expression'] = None
    
    # Print final summary
    if verbose:
        print(f"\n=== CCA Integration Test Summary ===")
        print(f"RNA CCA: {'✓' if results['rna_results']['cca_scores'] else '✗'}")
        print(f"- RNA proportion test: {'✓' if results['rna_results']['pvalue_tests'].get('proportion') else '✗'}")
        print(f"- RNA expression test: {'✓' if results['rna_results']['pvalue_tests'].get('expression') else '✗'}")
        print(f"ATAC CCA: {'✓' if results['atac_results']['cca_scores'] else '✗'}")
        print(f"- ATAC proportion test: {'✓' if results['atac_results']['pvalue_tests'].get('proportion') else '✗'}")
        print(f"- ATAC expression test: {'✓' if results['atac_results']['pvalue_tests'].get('expression') else '✗'}")
        print(f"All results saved to {output_dir}")
    
    return results
