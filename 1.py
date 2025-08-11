import scanpy as sc
import sys
import os
from pathlib import Path

def add_sample_column(file_path):
    # Read the h5ad file
    adata = sc.read_h5ad(file_path)
    
    # Extract sample names from cell IDs
    adata.obs['sample'] = [cell_id.split('_')[0] for cell_id in adata.obs.index]
    
    # Show results
    print("Sample distribution:")
    print(adata.obs['sample'].value_counts())
    
    # Save back to the same file
    adata.write_h5ad(file_path)
    print(f"Updated {file_path} with sample column")

def analyze_h5ad(file_path):
    # Read the h5ad file
    adata = sc.read_h5ad(file_path)
    
    # Print basic info
    print(f"AnnData object: {adata}")
    print(f"Shape: {adata.shape} (cells × genes)")
    print(f"Data type: {adata.X.dtype}")
    
    # Print obs (cell metadata)
    print(f"\nObservations (obs) - {adata.obs.shape[0]} cells:")
    print(adata.obs.info())
    print("\nFirst few obs entries:")
    print(adata.obs.head())
    
    # Print var (gene metadata)
    print(f"\nVariables (var) - {adata.var.shape[0]} genes:")
    print(adata.var.info())
    print("\nFirst few var entries:")
    print(adata.var.head())

def process_directory(directory_path):
    """Process all h5ad files in the given directory"""
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist")
        return
    
    if not directory.is_dir():
        print(f"Error: {directory_path} is not a directory")
        return
    
    # Find all h5ad files
    h5ad_files = list(directory.glob("*.h5ad"))
    
    if not h5ad_files:
        print(f"No h5ad files found in {directory_path}")
        return
    
    print(f"Found {len(h5ad_files)} h5ad files in {directory_path}")
    print("=" * 80)
    
    for i, h5ad_file in enumerate(h5ad_files, 1):
        print(f"\nProcessing file {i}/{len(h5ad_files)}: {h5ad_file.name}")
        print("-" * 60)
        
        try:
            analyze_h5ad(str(h5ad_file))
            add_sample_column(str(h5ad_file))
            print(f"✓ Successfully processed {h5ad_file.name}")
        except Exception as e:
            print(f"✗ Error processing {h5ad_file.name}: {e}")
        
        print("=" * 80)

if __name__ == "__main__":
    
    directory_path =  "/dcl01/hongkai/data/data/hjiang/Data/paired/atac"
    process_directory(directory_path)