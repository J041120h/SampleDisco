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
    print(adata.uns.columns.keys())
    

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

def plot_tss_enrichment(adata, enrichment_key='TSS.enrichment', groupby=None, figsize=(8, 6), save_path=None):
    """
    Generate a violin plot of TSS enrichment values from an AnnData object.
    
    Parameters:
    - adata: AnnData object with TSS enrichment data in `obs`.
    - enrichment_key: The column in `adata.obs` that contains TSS enrichment values.
    - groupby: Optional. Column in `adata.obs` to group the violin plot by.
    - figsize: Tuple for figure size.
    - save_path: Optional. Path to save the figure (e.g., 'tss_enrichment.png').
    """

    print("Available `adata.obs` columns:")
    print(list(adata.obs.columns))
    print("\nAvailable `adata.var` columns:")
    print(list(adata.var.columns))

    if enrichment_key not in adata.obs.columns:
        raise ValueError(f"'{enrichment_key}' not found in adata.obs. Please provide valid enrichment_key.")
    
    plt.figure(figsize=figsize)
    if groupby and groupby in adata.obs.columns:
        sns.violinplot(data=adata.obs, x=groupby, y=enrichment_key, inner='box')
        plt.title(f"TSS Enrichment by {groupby}")
    else:
        sns.violinplot(data=adata.obs, y=enrichment_key, inner='box')
        plt.title("TSS Enrichment")
    
    plt.ylabel("TSS Enrichment Score")
    plt.xlabel(groupby if groupby else "")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


import pandas as pd
import anndata as ad
import scanpy as sc
import sys

def process_anndata(file_path):
    """
    Load AnnData object, print obs columns, and convert disease severity to numeric values.
    Overwrites the original file with converted data.
    
    Args:
        file_path (str): Path to the AnnData object file (.h5ad)
    
    Returns:
        anndata.AnnData: The modified AnnData object
    """
    try:
        # Load the AnnData object
        print(f"Loading AnnData object from: {file_path}")
        adata = ad.read_h5ad(file_path)
        
        # Print all obs column names
        print("\nAll obs columns:")
        print("-" * 40)
        for i, col in enumerate(adata.obs.columns, 1):
            print(f"{i:2d}. {col}")
        
        # Check if 'disease severity' column exists (case-insensitive search)
        disease_col = None
        for col in adata.obs.columns:
            if 'disease' in col.lower() and 'severity' in col.lower():
                disease_col = col
                break
        
        if disease_col is None:
            print("\nWarning: No column containing 'disease severity' found!")
            print("Available columns:", list(adata.obs.columns))
            return adata
        
        print(f"\nFound disease severity column: '{disease_col}'")
        
        # Print unique values in the disease severity column
        unique_values = adata.obs[disease_col].unique()
        print(f"Unique values in '{disease_col}': {unique_values}")
        
        # Define the mapping
        severity_mapping = {
            'Healthy': 1,
            'Mild': 2,
            'Moderate': 3,
            'Severe': 4,
            'Fatal': 5
        }
        
        # Convert the categorical values to numeric
        print(f"\nConverting '{disease_col}' using mapping:")
        for key, value in severity_mapping.items():
            print(f"  {key} -> {value}")
        
        # Store original values for comparison
        original_values = adata.obs[disease_col].copy()
        
        # Apply the mapping (replace the original column)
        adata.obs[disease_col] = adata.obs[disease_col].map(severity_mapping)
        
        # Handle any unmapped values
        unmapped_mask = adata.obs[disease_col].isna()
        if unmapped_mask.any():
            unmapped = original_values[unmapped_mask].unique()
            print(f"\nWarning: Some values could not be mapped: {unmapped}")
        
        # Print the converted column
        print(f"\nConverted column '{disease_col}' (original -> numeric):")
        print("-" * 50)
        comparison_df = pd.DataFrame({
            'original': original_values,
            'converted': adata.obs[disease_col]
        })
        print(comparison_df.value_counts().sort_index())
        
        # Print summary statistics
        print(f"\nSummary of converted values:")
        print(adata.obs[disease_col].value_counts().sort_index())
        
        # Overwrite the original file
        print(f"\nOverwriting original file: {file_path}")
        adata.write(file_path)
        print("File successfully overwritten with converted data!")
        
        return adata
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None
    
def remove_atac_uns_entry(h5ad_path: str):
    """
    Load an AnnData object from the specified .h5ad file, remove `uns['atac']` if it exists,
    and overwrite the original file.

    Parameters:
    -----------
    h5ad_path : str
        Path to the .h5ad file.
    """
    # Load AnnData object
    adata = ad.read_h5ad(h5ad_path)

    # Remove uns['atac'] if it exists
    if 'atac' in adata.uns:
        print("Removing uns['atac'] from AnnData.")
        del adata.uns['atac']
    else:
        print("uns['atac'] not found in AnnData.")

    # Overwrite the original file
    adata.write_h5ad(h5ad_path)
    print(f"Updated AnnData object saved to: {h5ad_path}")

# Example usage
if __name__ == "__main__":
    print("Beginning to process h5ad file...")
    file_path = "/Users/harry/Desktop/GenoDistance/Data/test_ATAC.h5ad"
    
    remove_atac_uns_entry(file_path)