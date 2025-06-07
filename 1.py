import anndata as ad

def print_first_ten_genes(adata_path):
    # Load the AnnData object
    adata = ad.read_h5ad(adata_path)
    
    # Try to access gene names from common locations
    if 'gene_name' in adata.var.columns:
        genes = adata.var['gene_name']
    else:
        genes = adata.var.index  # fallback to index if gene_name column doesn't exist

    print("First 10 gene names:")
    print(genes[:10].tolist())

if __name__ == "__main__":

    print_first_ten_genes("/Users/harry/Desktop/GenoDistance/Data/count_data.h5ad")
