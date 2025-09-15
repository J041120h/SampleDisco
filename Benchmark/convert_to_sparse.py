import anndata as ad
import scipy.sparse as sp

def convert_to_sparse(h5ad_path):
    """
    Convert the count matrix (.X) in an h5ad file to sparse format if it's not already sparse.
    
    Parameters:
    -----------
    h5ad_path : str
        Path to the h5ad file
    """
    # Read the h5ad file
    adata = ad.read_h5ad(h5ad_path)
    
    # Check if .X is already sparse
    if not sp.issparse(adata.X):
        print(f"Converting dense matrix to sparse format...")
        # Convert to sparse (CSR format is commonly used for count matrices)
        adata.X = sp.csr_matrix(adata.X)
        
        # Overwrite the original file
        adata.write_h5ad(h5ad_path)
        print(f"Successfully converted and saved to {h5ad_path}")
    else:
        print(f"Matrix is already sparse (type: {type(adata.X).__name__}). No conversion needed.")

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "/dcl01/hongkai/data/data/hjiang/Data/paired/rna/all.h5ad"
    convert_to_sparse(file_path)