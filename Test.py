#!/usr/bin/env python
# Script to merge multiple h5ad files into one large h5ad file

import os
import scanpy as sc
import pandas as pd
import numpy as np
import anndata
from tqdm import tqdm
import glob

def merge_h5ad_files():
    """
    Merge specific h5ad files into a test h5ad file.
    """
    # Set paths
    input_dir = "/users/hjiang/GenoDistance/Data/h5ad_files/"
    output_file = "/users/hjiang/GenoDistance/Data/h5ad_files/test_ATAC.h5ad"
    
    # Specify the exact files to merge
    target_files = [
        "re_SRR14466469.h5ad",
        "SRR14466469.h5ad", 
        "re_SRR14466471.h5ad",
        "SRR14466471.h5ad"
    ]
    
    # Get full paths for the target files
    h5ad_files = []
    for filename in target_files:
        full_path = os.path.join(input_dir, filename)
        if os.path.exists(full_path):
            h5ad_files.append(full_path)
        else:
            print(f"Warning: File {filename} not found in {input_dir}")
    
    if len(h5ad_files) == 0:
        print("No target files found. Available files:")
        available_files = glob.glob(os.path.join(input_dir, "*.h5ad"))
        for f in available_files:
            print(f"  {os.path.basename(f)}")
        return
    
    print(f"Found {len(h5ad_files)} target files to merge:")
    for f in h5ad_files:
        print(f"  {os.path.basename(f)}")
    
    if len(h5ad_files) == 0:
        print("No target files found to merge. Exiting.")
        return
    
    # Load the first file to get started
    print(f"Loading first file: {os.path.basename(h5ad_files[0])}")
    combined_adata = sc.read_h5ad(h5ad_files[0])
    
    # Add file origin to obs if not present
    sample_id = os.path.splitext(os.path.basename(h5ad_files[0]))[0]
    if 'sample' not in combined_adata.obs.columns:
        combined_adata.obs['sample'] = sample_id
    
    # Always add sample prefix to cell names for the first file too
    print(f"Adding sample prefix '{sample_id}_' to all cells from the first file")
    combined_adata.obs_names = [f"{sample_id}_{cell}" for cell in combined_adata.obs_names]
    
    # Process remaining files
    for file_path in tqdm(h5ad_files[1:], desc="Merging files"):
        try:
            # Load the current file
            current_adata = sc.read_h5ad(file_path)
            sample_id = os.path.splitext(os.path.basename(file_path))[0]
            
            # Add file origin to obs if not present
            if 'sample' not in current_adata.obs.columns:
                current_adata.obs['sample'] = sample_id
                
            # Always add sample_id prefix to all cells to ensure uniqueness
            # This handles cases where the same sample was processed twice
            print(f"Adding sample prefix '{sample_id}_' to all cells from this file")
            current_adata.obs_names = [f"{sample_id}_{cell}" for cell in current_adata.obs_names]
            
            # Handle potentially different feature sets
            if not np.array_equal(current_adata.var_names, combined_adata.var_names):
                print(f"Note: {sample_id} has different features. Finding common features...")
                # Find common features
                common_features = current_adata.var_names.intersection(combined_adata.var_names)
                
                if len(common_features) == 0:
                    print(f"Warning: No common features with {sample_id}. Skipping this file.")
                    continue
                    
                print(f"Found {len(common_features)} common features")
                
                # Subset both objects to common features
                current_adata = current_adata[:, common_features]
                combined_adata = combined_adata[:, common_features]
            
            # Concatenate the data
            combined_adata = anndata.concat(
                [combined_adata, current_adata],
                join='outer',  # Use outer join to keep all features
                merge='first',  # For conflicting elements in .uns, .varm, .obsm, keep info from first object
                index_unique=None  # We already made sure cell names are unique
            )
            
            print(f"Added {current_adata.shape[0]} cells from {sample_id}. "
                  f"Combined shape: {combined_adata.shape}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Save the merged object
    print(f"\nSaving merged data to {output_file}")
    combined_adata.write_h5ad(output_file, compression='gzip')
    
    print("\nMerging complete!")
    print(f"Test dataset saved to: {output_file}")
    print(f"Final dataset dimensions: {combined_adata.shape[0]} cells × {combined_adata.shape[1]} features")
    print(f"Sample distribution:")
    print(combined_adata.obs['sample'].value_counts())


import scanpy as sc

def print_anndata_columns(h5ad_path):
    """
    Load an AnnData object from an .h5ad file and print the column names of .obs and .var.
    
    Parameters
    ----------
    h5ad_path : str
        Path to the AnnData .h5ad file
    """
    # Load the AnnData object
    adata = sc.read_h5ad(h5ad_path)
    
    # Print .obs and .var column names
    print("obs columns:")
    print(adata.obs.columns.tolist())
    
    print("\nvar columns:")
    print(adata.var.columns.tolist())

# Example usage
if __name__ == "__main__":
    print("Beginning to merge h5ad files...")
    h5ad_file = "/Users/harry/Desktop/GenoDistance/result/ATAC/harmony/ATAC_sample.h5ad"
    # ---- Load the AnnData object ----
    adata = sc.read_h5ad(h5ad_file)

    # ---- Check if 'sample' column exists and count cells per sample ----
    if 'sample' not in adata.obs.columns:
        raise ValueError("The 'sample' column is not present in adata.obs")

    cell_counts = adata.obs['sample'].value_counts().sort_index()

    # ---- Display the result ----
    print("Number of cells per sample:")
    print(cell_counts)
