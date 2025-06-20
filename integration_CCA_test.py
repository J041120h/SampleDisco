import os
import anndata as ad
from CCA_test import *
from CCA import *

def integration_CCA_test(pseudobulk_anndata_path,
                        output_dir,
                        sev_col="severity",
                        num_simulations=1000,
                        ptime=True,
                        verbose=False):
    """
    Perform CCA integration test on pseudobulk data with both RNA and ATAC modalities.
    
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
        Dictionary containing CCA results and p-values
    """
    
    # Create output directories
    output_dir = os.path.join(output_dir, "CCA_test")
    rna_output_dir = os.path.join(output_dir, "RNA")
    atac_output_dir = os.path.join(output_dir, "ATAC")
    
    for dir_path in [output_dir, rna_output_dir, atac_output_dir]:
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
    pseudobulk_rna = pseudobulk_anndata[pseudobulk_anndata.obs['modality'] == 'RNA'].copy()
    pseudobulk_atac = pseudobulk_anndata[pseudobulk_anndata.obs['modality'] == 'ATAC'].copy()
    
    if verbose:
        print(f"RNA samples: {pseudobulk_rna.n_obs}")
        print(f"ATAC samples: {pseudobulk_atac.n_obs}")
    
    # Validate that we have both modalities
    if pseudobulk_rna.n_obs == 0:
        raise ValueError("No RNA samples found in the data")
    if pseudobulk_atac.n_obs == 0:
        raise ValueError("No ATAC samples found in the data")
    
    # Perform CCA analysis
    try:
        first_component_score_proportion, first_component_score_expression, ptime_proportion, ptime_expression = CCA_Call(
            adata=pseudobulk_anndata, 
            output_dir=output_dir, 
            sev_col=sev_col, 
            ptime=ptime, 
            verbose=verbose
        )
    except Exception as e:
        raise RuntimeError(f"CCA_Call failed: {e}")
    
    # Initialize results dictionary
    results = {
        'first_component_score_proportion': first_component_score_proportion,
        'first_component_score_expression': first_component_score_expression,
        'ptime_proportion': ptime_proportion,
        'ptime_expression': ptime_expression
    }
    
    # Perform p-value tests for proportion data (RNA)
    try:
        rna_pvalue_results = cca_pvalue_test(
            pseudo_adata=pseudobulk_anndata,
            column="X_DR_proportion",
            input_correlation=first_component_score_proportion,
            output_directory=rna_output_dir,
            num_simulations=num_simulations,
            sev_col=sev_col,
            verbose=verbose
        )
        results['rna_pvalue_results'] = rna_pvalue_results
        
        if verbose:
            print(f"RNA proportion p-value test completed. Results saved to {rna_output_dir}")
            
    except Exception as e:
        print(f"Warning: RNA p-value test failed: {e}")
        results['rna_pvalue_results'] = None
    
    # Perform p-value tests for expression data (ATAC)
    try:
        atac_pvalue_results = cca_pvalue_test(
            pseudo_adata=pseudobulk_anndata,
            column="X_DR_expression",
            input_correlation=first_component_score_expression,
            output_directory=atac_output_dir,
            num_simulations=num_simulations,
            sev_col=sev_col,
            verbose=verbose
        )
        results['atac_pvalue_results'] = atac_pvalue_results
        
        if verbose:
            print(f"ATAC expression p-value test completed. Results saved to {atac_output_dir}")
            
    except Exception as e:
        print(f"Warning: ATAC p-value test failed: {e}")
        results['atac_pvalue_results'] = None
    
    if verbose:
        print(f"CCA integration test completed. All results saved to {output_dir}")
    
    return results