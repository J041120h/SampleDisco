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

import anndata as ad
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from anndata import AnnData
import time
from DR import process_anndata_with_pca
from pseudo_adata import compute_pseudobulk_adata
from CCA import *
from CCA_test import *
from linux.CellType_linux import cell_types_linux
from integration_visualization import visualize_multimodal_embedding

def ensure_non_categorical_columns(adata, columns):
    """Convert specified columns from categorical to string to avoid categorical errors"""
    for col in columns:
        if col in adata.obs.columns:
            if pd.api.types.is_categorical_dtype(adata.obs[col]):
                adata.obs[col] = adata.obs[col].astype(str)
    return adata

def find_optimal_cell_resolution_integration(
    AnnData_integrated: AnnData,
    output_dir: str,
    optimization_target: str = "sum",  # "rna", "atac", or "sum"
    n_features: int = 40000,
    sev_col: str = "sev.level",
    batch_col: str = None,
    sample_col: str = "sample",
    modality_col: str = "modality",
    use_rep: str = 'X_glue',
    num_DR_components: int = 30,
    num_PCs: int = 20,
    num_pvalue_simulations: int = 100,
    compute_pvalues: bool = True,
    visualize_embeddings: bool = True,
    verbose: bool = True
) -> tuple:
    """
    Find optimal clustering resolution for integrated RNA+ATAC data by maximizing 
    CCA correlation between dimension reduction and severity levels.
    
    Parameters:
    -----------
    AnnData_integrated : AnnData
        Integrated AnnData object containing both RNA and ATAC data
    output_dir : str
        Output directory for results
    optimization_target : str
        What to optimize: "rna" (RNA CCA only), "atac" (ATAC CCA only), 
        or "sum" (sum of both CCA scores)
    n_features : int
        Number of features for pseudobulk computation
    sev_col : str
        Column name for severity levels in pseudobulk_anndata.obs
    batch_col : str
        Column name for batch information
    sample_col : str
        Column name for sample identifiers
    modality_col : str
        Column name containing modality information (RNA/ATAC)
    use_rep : str
        Representation to use for neighborhood graph
    num_DR_components : int
        Number of dimension reduction components
    num_PCs : int
        Number of PCs for neighborhood graph
    num_pvalue_simulations : int
        Number of simulations for p-value calculation
    compute_pvalues : bool
        Whether to compute p-values for each resolution
    visualize_embeddings : bool
        Whether to create embedding visualizations for each resolution
    verbose : bool
        Whether to print verbose output
        
    Returns:
    --------
    tuple: (optimal_resolution, results_dataframe)
    """
    start_time = time.time()
    
    # Validate optimization target
    if optimization_target not in ["rna", "atac", "sum"]:
        raise ValueError("optimization_target must be 'rna', 'atac', or 'sum'")
    
    # Create subdirectories for different outputs
    main_output_dir = os.path.join(output_dir, f"CCA_resolution_optimization_integration_{optimization_target}")
    resolution_plots_dir = os.path.join(main_output_dir, "resolution_plots")
    pvalue_results_dir = os.path.join(main_output_dir, "pvalue_results")
    embedding_plots_dir = os.path.join(main_output_dir, "embedding_visualizations")
    
    for dir_path in [main_output_dir, resolution_plots_dir, pvalue_results_dir, embedding_plots_dir]:
        os.makedirs(dir_path, exist_ok=True)

    print(f"Starting integrated resolution optimization...")
    print(f"Optimization target: {optimization_target.upper()}")
    print(f"Using representation: {use_rep} with {num_PCs} components")
    print(f"Testing resolutions from 0.01 to 1.00...")
    if compute_pvalues:
        print(f"Computing p-values with {num_pvalue_simulations} simulations per resolution")

    # Ensure critical columns are not categorical to avoid errors
    columns_to_check = ['cell_type', modality_col, sev_col, sample_col]
    if batch_col:
        columns_to_check.append(batch_col)
    AnnData_integrated = ensure_non_categorical_columns(AnnData_integrated, columns_to_check)
    
    # Storage for all results
    all_results = []

    # Helper function to compute CCA for both modalities
    def compute_modality_cca(pseudobulk_adata, modality, column, sev_col):
        """Compute CCA for a specific modality"""
        # Ensure modality column is not categorical
        if modality_col in pseudobulk_adata.obs.columns:
            if pd.api.types.is_categorical_dtype(pseudobulk_adata.obs[modality_col]):
                pseudobulk_adata.obs[modality_col] = pseudobulk_adata.obs[modality_col].astype(str)
        
        # Filter for specific modality
        modality_mask = pseudobulk_adata.obs[modality_col] == modality
        if not any(modality_mask):
            return np.nan, np.nan
        
        modality_adata = pseudobulk_adata[modality_mask].copy()
        
        # Filter PCA coordinates in .uns
        positions = [i for i, is_mod in enumerate(modality_mask) if is_mod]
        if column in pseudobulk_adata.uns and column in modality_adata.uns:
            original_coords = pseudobulk_adata.uns[column]
            if isinstance(original_coords, pd.DataFrame):
                modality_adata.uns[column] = original_coords.iloc[positions]
            else:
                modality_adata.uns[column] = original_coords[positions]
        
        try:
            (pca_coords_2d, sev_levels, cca_model, 
             cca_score, samples) = run_cca_on_2d_pca_from_adata(
                adata=modality_adata,
                column=column,
                sev_col=sev_col
            )
            
            # Compute p-value if requested
            p_value = np.nan
            if compute_pvalues:
                p_value = cca_pvalue_test(
                    pseudo_adata=modality_adata,
                    column=column,
                    input_correlation=cca_score,
                    output_directory=pvalue_results_dir,
                    num_simulations=num_pvalue_simulations,
                    sev_col=sev_col,
                    verbose=False
                )
            
            return cca_score, p_value
            
        except Exception as e:
            if verbose:
                print(f"Warning: CCA failed for {modality} {column}: {str(e)}")
            return np.nan, np.nan

    # First pass: coarse search
    print("\n=== FIRST PASS: Coarse Search ===")
    for resolution in np.arange(0.1, 1.01, 0.1):
        print(f"\n\nTesting resolution: {resolution:.2f}\n")
        
        result_dict = {
            'resolution': resolution,
            'rna_cca_expression': np.nan,
            'rna_cca_proportion': np.nan,
            'atac_cca_expression': np.nan,
            'atac_cca_proportion': np.nan,
            'rna_pvalue_expression': np.nan,
            'rna_pvalue_proportion': np.nan,
            'atac_pvalue_expression': np.nan,
            'atac_pvalue_proportion': np.nan,
            'optimization_score': np.nan,
            'pass': 'coarse'
        }
        
        try:
            # Clean up previous cell type assignments
            if 'cell_type' in AnnData_integrated.obs.columns:
                AnnData_integrated.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            
            # Ensure modality column is properly set up
            if modality_col in AnnData_integrated.obs.columns:
                # Convert to string type to avoid categorical issues
                AnnData_integrated.obs[modality_col] = AnnData_integrated.obs[modality_col].astype(str)
            
            # Ensure modality column is properly set up
            if modality_col in AnnData_integrated.obs.columns:
                # Convert to string type to avoid categorical issues
                AnnData_integrated.obs[modality_col] = AnnData_integrated.obs[modality_col].astype(str)
            
            # Perform clustering using the Linux version
            AnnData_integrated = cell_types_linux(
                AnnData_integrated,
                cell_column='cell_type',
                existing_cell_types=False,
                Save=False,
                output_dir=output_dir,
                cluster_resolution=resolution,
                use_rep=use_rep,
                markers=None,
                num_PCs=num_PCs,
                verbose=False
            )
            
            # Compute pseudobulk data
            pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata(
                adata=AnnData_integrated, 
                batch_col=batch_col, 
                sample_col=sample_col, 
                celltype_col='cell_type', 
                n_features=n_features, 
                output_dir=output_dir,
                Save=False,
                verbose=False
            )
            
            # Perform dimension reduction for integrated data
            process_anndata_with_pca(
                adata=AnnData_integrated,
                pseudobulk=pseudobulk_dict,
                pseudobulk_anndata=pseudobulk_adata,
                sample_col=sample_col,
                n_expression_pcs=num_DR_components,
                n_proportion_pcs=num_DR_components,
                atac=False,  # For integrated data, use RNA processing
                output_dir=output_dir,
                not_save=True,
                verbose=False
            )
            
            # Compute CCA for RNA modality
            for dr_type in ['expression', 'proportion']:
                column = f'X_DR_{dr_type}'
                if column in pseudobulk_adata.uns:
                    cca_score, p_value = compute_modality_cca(
                        pseudobulk_adata, 'RNA', column, sev_col
                    )
                    result_dict[f'rna_cca_{dr_type}'] = cca_score
                    result_dict[f'rna_pvalue_{dr_type}'] = p_value
                    
                    if not np.isnan(cca_score):
                        print(f"RNA {dr_type} CCA Score: {cca_score:.4f}, p-value: {p_value:.4f}")
            
            # Compute CCA for ATAC modality
            for dr_type in ['expression', 'proportion']:
                column = f'X_DR_{dr_type}'
                if column in pseudobulk_adata.uns:
                    cca_score, p_value = compute_modality_cca(
                        pseudobulk_adata, 'ATAC', column, sev_col
                    )
                    result_dict[f'atac_cca_{dr_type}'] = cca_score
                    result_dict[f'atac_pvalue_{dr_type}'] = p_value
                    
                    if not np.isnan(cca_score):
                        print(f"ATAC {dr_type} CCA Score: {cca_score:.4f}, p-value: {p_value:.4f}")
            
            # Calculate optimization score based on target
            if optimization_target == "rna":
                # Use maximum of RNA expression and proportion scores
                rna_scores = [result_dict['rna_cca_expression'], result_dict['rna_cca_proportion']]
                valid_scores = [s for s in rna_scores if not np.isnan(s)]
                result_dict['optimization_score'] = max(valid_scores) if valid_scores else np.nan
            elif optimization_target == "atac":
                # Use maximum of ATAC expression and proportion scores
                atac_scores = [result_dict['atac_cca_expression'], result_dict['atac_cca_proportion']]
                valid_scores = [s for s in atac_scores if not np.isnan(s)]
                result_dict['optimization_score'] = max(valid_scores) if valid_scores else np.nan
            else:  # sum
                # Sum of best scores from each modality
                rna_best = max([s for s in [result_dict['rna_cca_expression'], 
                               result_dict['rna_cca_proportion']] if not np.isnan(s)] or [0])
                atac_best = max([s for s in [result_dict['atac_cca_expression'], 
                                result_dict['atac_cca_proportion']] if not np.isnan(s)] or [0])
                result_dict['optimization_score'] = rna_best + atac_best
            
            print(f"Resolution {resolution:.2f}: Optimization Score = {result_dict['optimization_score']:.4f}")
            
            # Create embedding visualizations if requested
            if visualize_embeddings and not np.isnan(result_dict['optimization_score']):
                for modality in ['RNA', 'ATAC']:
                    try:
                        embedding_path = os.path.join(
                            embedding_plots_dir, 
                            f"embedding_res_{resolution:.2f}_{modality}"
                        )
                        visualize_multimodal_embedding(
                            adata=pseudobulk_adata,
                            modality_col=modality_col,
                            color_col=sev_col,
                            target_modality=modality,
                            output_dir=embedding_path,
                            show_sample_names=False,
                            verbose=False
                        )
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Failed to create embedding visualization for {modality}: {str(e)}")
                
        except Exception as e:
            print(f"Error at resolution {resolution:.2f}: {str(e)}")
        
        all_results.append(result_dict)

    # Find best resolution from first pass
    coarse_results = [r for r in all_results if not np.isnan(r['optimization_score'])]
    if not coarse_results:
        raise ValueError("No valid optimization scores obtained in coarse search.")
    
    best_coarse = max(coarse_results, key=lambda x: x['optimization_score'])
    best_resolution = best_coarse['resolution']
    print(f"\nBest resolution from first pass: {best_resolution:.2f}")
    print(f"Best optimization score: {best_coarse['optimization_score']:.4f}")

    # Second pass: fine-tuned search
    print("\n=== SECOND PASS: Fine-tuned Search ===")
    search_range_start = max(0.01, best_resolution - 0.05)
    search_range_end = min(1.00, best_resolution + 0.05)
    
    print(f"Fine-tuning search from {search_range_start:.2f} to {search_range_end:.2f}...")

    for resolution in np.arange(search_range_start, search_range_end + 0.001, 0.01):
        resolution = round(resolution, 3)
        
        # Skip if already tested in coarse search
        if any(abs(r['resolution'] - resolution) < 0.001 for r in all_results):
            continue
        
        print(f"\nTesting fine-tuned resolution: {resolution:.3f}")
        
        result_dict = {
            'resolution': resolution,
            'rna_cca_expression': np.nan,
            'rna_cca_proportion': np.nan,
            'atac_cca_expression': np.nan,
            'atac_cca_proportion': np.nan,
            'rna_pvalue_expression': np.nan,
            'rna_pvalue_proportion': np.nan,
            'atac_pvalue_expression': np.nan,
            'atac_pvalue_proportion': np.nan,
            'optimization_score': np.nan,
            'pass': 'fine'
        }
        
        try:
            # Clean up previous cell type assignments
            if 'cell_type' in AnnData_integrated.obs.columns:
                AnnData_integrated.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            
            # Perform clustering
            AnnData_integrated = cell_types_linux(
                AnnData_integrated,
                cell_column='cell_type',
                existing_cell_types=False,
                Save=False,
                output_dir=output_dir,
                cluster_resolution=resolution,
                use_rep=use_rep,
                markers=None,
                num_PCs=num_PCs,
                verbose=False
            )
            
            # Compute pseudobulk data
            pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata(
                adata=AnnData_integrated, 
                batch_col=batch_col, 
                sample_col=sample_col, 
                celltype_col='cell_type', 
                n_features=n_features, 
                output_dir=output_dir,
                Save=False,
                verbose=False
            )
            
            # Perform dimension reduction
            process_anndata_with_pca(
                adata=AnnData_integrated,
                pseudobulk=pseudobulk_dict,
                pseudobulk_anndata=pseudobulk_adata,
                sample_col=sample_col,
                n_expression_pcs=num_DR_components,
                n_proportion_pcs=num_DR_components,
                atac=False,
                output_dir=output_dir,
                not_save=True,
                verbose=False
            )
            
            # Compute CCA for both modalities and DR types
            for modality in ['RNA', 'ATAC']:
                for dr_type in ['expression', 'proportion']:
                    column = f'X_DR_{dr_type}'
                    if column in pseudobulk_adata.uns:
                        cca_score, p_value = compute_modality_cca(
                            pseudobulk_adata, modality, column, sev_col
                        )
                        result_dict[f'{modality.lower()}_cca_{dr_type}'] = cca_score
                        result_dict[f'{modality.lower()}_pvalue_{dr_type}'] = p_value
            
            # Calculate optimization score
            if optimization_target == "rna":
                rna_scores = [result_dict['rna_cca_expression'], result_dict['rna_cca_proportion']]
                valid_scores = [s for s in rna_scores if not np.isnan(s)]
                result_dict['optimization_score'] = max(valid_scores) if valid_scores else np.nan
            elif optimization_target == "atac":
                atac_scores = [result_dict['atac_cca_expression'], result_dict['atac_cca_proportion']]
                valid_scores = [s for s in atac_scores if not np.isnan(s)]
                result_dict['optimization_score'] = max(valid_scores) if valid_scores else np.nan
            else:  # sum
                rna_best = max([s for s in [result_dict['rna_cca_expression'], 
                               result_dict['rna_cca_proportion']] if not np.isnan(s)] or [0])
                atac_best = max([s for s in [result_dict['atac_cca_expression'], 
                                result_dict['atac_cca_proportion']] if not np.isnan(s)] or [0])
                result_dict['optimization_score'] = rna_best + atac_best
            
            print(f"Fine-tuned Resolution {resolution:.3f}: Score {result_dict['optimization_score']:.4f}")
                    
        except Exception as e:
            print(f"Error at fine-tuned resolution {resolution:.3f}: {str(e)}")
        
        all_results.append(result_dict)

    # Create comprehensive results dataframe
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("resolution")
    
    # Find final best resolution
    valid_results = df_results[~df_results['optimization_score'].isna()]
    if valid_results.empty:
        raise ValueError("No valid results obtained.")
    
    final_best_idx = valid_results['optimization_score'].idxmax()
    final_best_resolution = valid_results.loc[final_best_idx, 'resolution']
    final_best_score = valid_results.loc[final_best_idx, 'optimization_score']
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Best resolution: {final_best_resolution:.3f}")
    print(f"Best optimization score ({optimization_target}): {final_best_score:.4f}")
    print(f"Best RNA scores - Expression: {valid_results.loc[final_best_idx, 'rna_cca_expression']:.4f}, "
          f"Proportion: {valid_results.loc[final_best_idx, 'rna_cca_proportion']:.4f}")
    print(f"Best ATAC scores - Expression: {valid_results.loc[final_best_idx, 'atac_cca_expression']:.4f}, "
          f"Proportion: {valid_results.loc[final_best_idx, 'atac_cca_proportion']:.4f}")

    # Save comprehensive results
    results_csv_path = os.path.join(main_output_dir, f"resolution_scores_comprehensive_integration_{optimization_target}.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nComprehensive results saved to: {results_csv_path}")

    # Create main visualization plot
    create_integration_resolution_visualization(
        df_results, final_best_resolution, optimization_target, 
        main_output_dir, compute_pvalues
    )

    # Save p-value summary if computed
    if compute_pvalues:
        pvalue_summary_path = os.path.join(pvalue_results_dir, f"pvalue_summary_integration_{optimization_target}.txt")
        with open(pvalue_summary_path, "w") as f:
            f.write(f"Resolution Optimization P-value Summary for Integration ({optimization_target})\n")
            f.write("="*60 + "\n\n")
            f.write(f"Best Resolution: {final_best_resolution:.3f}\n")
            f.write(f"Best Optimization Score: {final_best_score:.4f}\n\n")
            f.write("Best P-values:\n")
            f.write(f"  RNA Expression: {valid_results.loc[final_best_idx, 'rna_pvalue_expression']:.4f}\n")
            f.write(f"  RNA Proportion: {valid_results.loc[final_best_idx, 'rna_pvalue_proportion']:.4f}\n")
            f.write(f"  ATAC Expression: {valid_results.loc[final_best_idx, 'atac_pvalue_expression']:.4f}\n")
            f.write(f"  ATAC Proportion: {valid_results.loc[final_best_idx, 'atac_pvalue_proportion']:.4f}\n")
        print(f"P-value summary saved to: {pvalue_summary_path}")

    print(f"\n[Find Optimal Resolution Integration] Total runtime: {time.time() - start_time:.2f} seconds\n")

    return final_best_resolution, df_results


def create_integration_resolution_visualization(df_results, best_resolution, optimization_target, 
                                              output_dir, include_pvalues):
    """Create comprehensive visualization of integration resolution search results"""
    
    # Determine number of subplots based on optimization target and p-values
    n_rows = 3 if optimization_target == "sum" else 2
    if include_pvalues:
        n_rows += 1
    
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4*n_rows))
    if n_rows == 1:
        axes = [axes]
    
    # Plot data
    valid_df = df_results[~df_results['optimization_score'].isna()]
    coarse_df = valid_df[valid_df['pass'] == 'coarse']
    fine_df = valid_df[valid_df['pass'] == 'fine']
    
    ax_idx = 0
    
    # Plot optimization score
    ax = axes[ax_idx]
    ax.scatter(coarse_df['resolution'], coarse_df['optimization_score'], 
               color='blue', s=60, alpha=0.6, label='Coarse Search')
    ax.scatter(fine_df['resolution'], fine_df['optimization_score'], 
               color='green', s=40, alpha=0.8, label='Fine Search')
    ax.plot(valid_df['resolution'], valid_df['optimization_score'], 
            'k-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=best_resolution, color='r', linestyle='--', 
               label=f'Best Resolution: {best_resolution:.3f}')
    ax.set_xlabel("Resolution")
    ax.set_ylabel(f"Optimization Score ({optimization_target.upper()})")
    ax.set_title(f"Integration Resolution Optimization - Target: {optimization_target.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax_idx += 1
    
    # Plot RNA CCA scores
    if optimization_target in ["rna", "sum"]:
        ax = axes[ax_idx]
        # Expression scores
        ax.scatter(coarse_df['resolution'], coarse_df['rna_cca_expression'], 
                   color='darkblue', s=60, alpha=0.6, marker='o', label='RNA Expression (Coarse)')
        ax.scatter(fine_df['resolution'], fine_df['rna_cca_expression'], 
                   color='darkblue', s=40, alpha=0.8, marker='o', label='RNA Expression (Fine)')
        # Proportion scores
        ax.scatter(coarse_df['resolution'], coarse_df['rna_cca_proportion'], 
                   color='lightblue', s=60, alpha=0.6, marker='^', label='RNA Proportion (Coarse)')
        ax.scatter(fine_df['resolution'], fine_df['rna_cca_proportion'], 
                   color='lightblue', s=40, alpha=0.8, marker='^', label='RNA Proportion (Fine)')
        
        ax.axvline(x=best_resolution, color='r', linestyle='--')
        ax.set_xlabel("Resolution")
        ax.set_ylabel("CCA Score")
        ax.set_title("RNA CCA Scores")
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)
        ax_idx += 1
    
    # Plot ATAC CCA scores
    if optimization_target in ["atac", "sum"]:
        ax = axes[ax_idx]
        # Expression scores
        ax.scatter(coarse_df['resolution'], coarse_df['atac_cca_expression'], 
                   color='darkgreen', s=60, alpha=0.6, marker='o', label='ATAC Expression (Coarse)')
        ax.scatter(fine_df['resolution'], fine_df['atac_cca_expression'], 
                   color='darkgreen', s=40, alpha=0.8, marker='o', label='ATAC Expression (Fine)')
        # Proportion scores
        ax.scatter(coarse_df['resolution'], coarse_df['atac_cca_proportion'], 
                   color='lightgreen', s=60, alpha=0.6, marker='^', label='ATAC Proportion (Coarse)')
        ax.scatter(fine_df['resolution'], fine_df['atac_cca_proportion'], 
                   color='lightgreen', s=40, alpha=0.8, marker='^', label='ATAC Proportion (Fine)')
        
        ax.axvline(x=best_resolution, color='r', linestyle='--')
        ax.set_xlabel("Resolution")
        ax.set_ylabel("CCA Score")
        ax.set_title("ATAC CCA Scores")
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)
        ax_idx += 1
    
    # Plot p-values if available
    if include_pvalues:
        ax = axes[ax_idx]
        
        # RNA p-values
        valid_pval_rna_exp = valid_df[~valid_df['rna_pvalue_expression'].isna()]
        valid_pval_rna_prop = valid_df[~valid_df['rna_pvalue_proportion'].isna()]
        
        if not valid_pval_rna_exp.empty:
            ax.scatter(valid_pval_rna_exp['resolution'], valid_pval_rna_exp['rna_pvalue_expression'], 
                       color='darkblue', s=40, alpha=0.7, marker='o', label='RNA Expression')
        if not valid_pval_rna_prop.empty:
            ax.scatter(valid_pval_rna_prop['resolution'], valid_pval_rna_prop['rna_pvalue_proportion'], 
                       color='lightblue', s=40, alpha=0.7, marker='^', label='RNA Proportion')
        
        # ATAC p-values
        valid_pval_atac_exp = valid_df[~valid_df['atac_pvalue_expression'].isna()]
        valid_pval_atac_prop = valid_df[~valid_df['atac_pvalue_proportion'].isna()]
        
        if not valid_pval_atac_exp.empty:
            ax.scatter(valid_pval_atac_exp['resolution'], valid_pval_atac_exp['atac_pvalue_expression'], 
                       color='darkgreen', s=40, alpha=0.7, marker='s', label='ATAC Expression')
        if not valid_pval_atac_prop.empty:
            ax.scatter(valid_pval_atac_prop['resolution'], valid_pval_atac_prop['atac_pvalue_proportion'], 
                       color='lightgreen', s=40, alpha=0.7, marker='d', label='ATAC Proportion')
        
        ax.axvline(x=best_resolution, color='r', linestyle='--')
        ax.axhline(y=0.05, color='orange', linestyle=':', label='p=0.05 threshold')
        
        ax.set_xlabel("Resolution")
        ax.set_ylabel("P-value")
        ax.set_title("P-values for All Modalities and DR Types")
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"resolution_optimization_comprehensive_integration_{optimization_target}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive plot saved to: {plot_path}")

if __name__ == "__main__":
    integrated_adata = ad.read_h5ad("/dcl01/hongkai/data/data/hjiang/result/integration/glue/atac_rna_integrated.h5ad")
    output_dir = "/dcl01/hongkai/data/data/hjiang/result/integration/CCA"
    try:
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator
        import cupy as cp
        
        rmm.reinitialize(
            managed_memory=True,
            pool_allocator=False,
        )
        cp.cuda.set_allocator(rmm_cupy_allocator)
    except:
        pass

    import torch
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"GPU Memory Available: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    optimal_res, results_df = find_optimal_cell_resolution_integration(
        AnnData_integrated=integrated_adata,
        output_dir="/dcl01/hongkai/data/data/hjiang/result/integration_test/",
        optimization_target="atac",  # or "rna" or "atac"
        n_features=2000,
        sev_col="sev.level",
        batch_col="batch",
        sample_col="sample",
        modality_col="modality",
        use_rep='X_glue',
        num_DR_components=30,
        num_PCs=30,
        num_pvalue_simulations=1000,
        compute_pvalues=True,
        visualize_embeddings=True,
        verbose=True
    )
