import scanpy as sc
import sys
import pandas as pd
from collections import defaultdict

def display_samples_by_group(adata):
    """
    Organizes and displays samples by their group in the format:
    Group1: Sample1 Sample2 Sample3
    Group2: Sample4 Sample5
    """
    # Check if both Sample and Group columns exist
    if "Sample" not in adata.obs.columns or "Group" not in adata.obs.columns:
        print("Error: Sample or Group column not found in .obs")
        return None
    
    # Create a mapping of groups to their samples
    group_to_samples = defaultdict(list)
    
    # Use drop_duplicates to eliminate redundancy
    sample_group_pairs = adata.obs[["Sample", "Group"]].drop_duplicates()
    
    # Fill the dictionary
    for _, row in sample_group_pairs.iterrows():
        group_to_samples[row["Group"]].append(row["Sample"])
    
    # Print the mapping in the requested format
    print("\nSamples by Group:")
    for group, samples in group_to_samples.items():
        samples_str = " ".join(samples)
        print(f"{group}: {samples_str}")
    
    return group_to_samples

def map_sample_to_group(adata):
    """
    Maps samples to their corresponding groups, removing redundancy.
    Returns a dictionary with sample IDs as keys and group assignments as values.
    """
    # Check if both Sample and Group columns exist
    if "Sample" not in adata.obs.columns or "Group" not in adata.obs.columns:
        print("Error: Sample or Group column not found in .obs")
        return None
    
    # Create a mapping dictionary, removing redundancy by using drop_duplicates
    sample_to_group_df = adata.obs[["Sample", "Group"]].drop_duplicates()
    sample_to_group_dict = dict(zip(sample_to_group_df["Sample"], sample_to_group_df["Group"]))
    
    # Print the mapping
    print("\nSample to Group mapping:")
    for sample, group in sample_to_group_dict.items():
        print(f" {sample}: {group}")
    
    return sample_to_group_dict

def print_h5ad_metadata(h5ad_path, preview_n=3):
    # Load the AnnData object
    try:
        adata = sc.read_h5ad(h5ad_path)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return
    
    # Print .obs column names and sample values
    print("\n.obs columns and sample values:")
    for col in adata.obs.columns:
        values = adata.obs[col].unique()
        preview = values[:preview_n] if len(values) > preview_n else values
        print(f" {col}: {list(preview)}")
    
    # Specifically print the 'Group' column
    if "Group" in adata.obs.columns:
        print("\nSpecifically printing 'Group' column values:")
        group_values = adata.obs["Group"].unique()
        print(f" Group: {list(group_values)}")
    else:
        print("\n'Group' column not found in .obs.")
    
    # Print .var column names and sample values
    print("\n.var columns and sample values:")
    for col in adata.var.columns:
        values = adata.var[col].unique()
        preview = values[:preview_n] if len(values) > preview_n else values
        print(f" {col}: {list(preview)}")
    
    # Display samples organized by groups
    if "Sample" in adata.obs.columns and "Group" in adata.obs.columns:
        group_to_samples = display_samples_by_group(adata)
        sample_to_group_dict = map_sample_to_group(adata)
        return sample_to_group_dict, group_to_samples
    else:
        print("\nCannot create Sample to Group mapping - required columns not found.")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_file.h5ad>")
    else:
        print_h5ad_metadata(sys.argv[1])