import scanpy as sc
import sys

def print_h5ad_metadata(h5ad_path, preview_n=3):
    # Load the AnnData object
    try:
        adata = sc.read_h5ad(h5ad_path)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    # Print .obs column names and preview
    print("\n.obs columns and sample values:")
    for col in adata.obs.columns:
        values = adata.obs[col].unique()
        preview = values[:preview_n] if len(values) > preview_n else values
        print(f"  {col}: {list(preview)}")

    # Print .var column names and preview
    print("\n.var columns and sample values:")
    for col in adata.var.columns:
        values = adata.var[col].unique()
        preview = values[:preview_n] if len(values) > preview_n else values
        print(f"  {col}: {list(preview)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_file.h5ad>")
    else:
        print_h5ad_metadata(sys.argv[1])
