import scanpy as sc

def print_example_cell_names(h5ad_path: str, n_examples: int = 10):
    """
    Print a few example cell names (obs index) from an AnnData .h5ad file.
    
    Parameters
    ----------
    h5ad_path : str
        Path to the .h5ad file.
    n_examples : int
        Number of cell names to display (default: 10).
    """
    try:
        print(f"🔍 Loading AnnData from: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path, backed=None)
        
        cell_names = adata.obs_names.tolist()
        total_cells = len(cell_names)
        
        print(f"\n✅ Total cells: {total_cells}")
        print(f"📋 Showing {min(n_examples, total_cells)} example cell names:\n")
        
        for name in cell_names[:n_examples]:
            print("  -", name)
        
    except Exception as e:
        print(f"❌ Error reading {h5ad_path}: {e}")

if __name__ == "__main__":
    print_example_cell_names(h5ad_path = '/dcl01/hongkai/data/data/hjiang/Data/paired/atac/all.h5ad')