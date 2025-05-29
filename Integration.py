import os
import scbean.model.vipcca as vip
import scbean.tools.utils as tl
import scbean.tools.plotting as pl
import matplotlib
import matplotlib.pyplot as plt
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
    os.makedirs(output_dir, exist_ok=True)
    RNA_pseudobulk.obs['_batch'] = 'RNA'  # or 0
    ATAC_pseudobulk.obs['_batch'] = 'ATAC'  # or 1
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
    
    # Save the plot locally
    plot_path = os.path.join(output_dir, "integration_umap.png")
    sc.pl.umap(adata_integrate, color=['_batch'], use_raw=False, show=False, save=False)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"UMAP plot saved to: {plot_path}")

if __name__ == "__main__":
    # Example usage
    RNA_pseudobulk = sc.read_h5ad("/Users/harry/Desktop/GenoDistance/result/pseudobulk/pseudobulk_adata.h5ad")
    ATAC_pseudobulk = sc.read_h5ad("/Users/harry/Desktop/GenoDistance/result/ATAC/pseudobulk/pseudobulk_adata.h5ad")
    output_dir = "/Users/harry/Desktop/GenoDistance/result/integration"
    
    integration_RNA_ATAC(RNA_pseudobulk, ATAC_pseudobulk, output_dir)