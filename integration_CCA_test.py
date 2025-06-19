from CCA_test import *
from CCA import *

def integration_CCA_test(pseudobulk_anndata_path,
                         output_dir
                         ):
    
    pseudobulk_anndata = ad.read_h5ad(pseudobulk_anndata_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    adata_rna = [pseudobulk_anndata.obs['modality'] == 'RNA'].copy()
    first_component_score_proportion, first_component_score_expression, ptime_proportion, ptime_expression = CCA_Call(adata = pseudobulk_anndata, output_dir=cca_output_dir, sev_col = sev_col_cca, ptime = True, verbose = trajectory_verbose)
    cca_pvalue_test(
        pseudo_adata = pseudobulk_anndata,
        column = "X_DR_proportion",
        input_correlation = first_component_score_proportion,
        output_directory = atac_cca_output_dir,
        num_simulations = 1000,
        sev_col = sev_col_cca,
        verbose = trajectory_verbose
    )

    cca_pvalue_test(
        pseudo_adata = pseudobulk_anndata,
        column = "X_DR_expression",
        input_correlation = first_component_score_expression,
        output_directory = atac_cca_output_dir,
        num_simulations = 1000,
        sev_col = sev_col_cca,
        verbose = trajectory_verbose
    )