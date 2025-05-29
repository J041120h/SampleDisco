
import scbean.model.vipcca as vip
import scbean.tools.utils as tl
import scbean.tools.plotting as pl

import matplotlib
import scanpy as sc
matplotlib.use('TkAgg')

def integration_RNA_ATAC(RNA_pseudobulk, ATAC_pseudobulk, output_dir):
    """
    Integrates RNA and ATAC pseudobulk data.
    
    Parameters:
    - RNA_pseudobulk: Anndata 
    - ATAC_pseudobulk: Anndata
    - output_dir: Directory to save the integrated results.
    
    Returns:
    - None
    """
    adata_all = tl.preprocessing([RNA_pseudobulk, ATAC_pseudobulk], hvg = False, index_unique="-")
    handle = vip.VIPCCA(
        adata_all=adata_all,
        res_path=output_dir,
        mode='CVAE',
        split_by="_batch",
        epochs=20,
        lambda_regulizer=5,
        batch_input_size=128,
        batch_input_size2=16
    )
    # do integration and return an AnnData object
    adata_integrate = handle.fit_integrate()
    sc.pp.neighbors(adata_integrate, use_rep='X_vipcca')
    sc.tl.umap(adata_integrate)
    sc.set_figure_params(figsize=[5.5,4.5])
    sc.pl.umap(adata_integrate, color=['_batch'], use_raw=False, show=True,)

