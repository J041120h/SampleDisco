#!/usr/bin/env python3
"""
Script to merge multiple h5ad files from a directory into a single h5ad file.
Each file's name (before .h5ad) is used as tissue identifier for cells.
"""

import os
import sys
from pathlib import Path
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_h5ad_files(directory: str) -> List[tuple]:
    """
    Load all h5ad files from the specified directory.
    
    Args:
        directory: Path to directory containing h5ad files
        
    Returns:
        List of tuples (tissue_name, AnnData object)
    """
    h5ad_files = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise ValueError(f"Directory {directory} does not exist")
    
    if not dir_path.is_dir():
        raise ValueError(f"{directory} is not a directory")
    
    # Find all h5ad files
    h5ad_file_paths = list(sorted(dir_path.glob("*.h5ad")))
    total_files = len(h5ad_file_paths)
    
    print(f"\n{'='*60}")
    print(f"Found {total_files} h5ad files to process")
    print(f"{'='*60}\n")
    
    for idx, file_path in enumerate(h5ad_file_paths, 1):
        tissue_name = file_path.stem  # Get filename without extension
        print(f"[{idx}/{total_files}] Loading {file_path.name} (tissue: {tissue_name})...")
        logger.info(f"Loading {file_path.name} (tissue: {tissue_name})")
        
        try:
            adata = sc.read_h5ad(file_path)
            h5ad_files.append((tissue_name, adata))
            print(f"    ✓ Successfully loaded: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
            logger.info(f"  Loaded {adata.n_obs} cells, {adata.n_vars} genes")
        except Exception as e:
            print(f"    ✗ Failed to load: {e}")
            logger.error(f"  Failed to load {file_path.name}: {e}")
            continue
    
    if not h5ad_files:
        raise ValueError(f"No valid h5ad files found in {directory}")
    
    print(f"\n{'='*60}")
    print(f"Successfully loaded {len(h5ad_files)}/{total_files} files")
    print(f"{'='*60}\n")
    
    return h5ad_files


def merge_h5ad_files(h5ad_list: List[tuple], batch_key: str = "tissue") -> ad.AnnData:
    """
    Merge multiple AnnData objects into one, adding tissue information.
    
    Args:
        h5ad_list: List of tuples (tissue_name, AnnData object)
        batch_key: Name of the column to store tissue information
        
    Returns:
        Merged AnnData object
    """
    if not h5ad_list:
        raise ValueError("No h5ad files to merge")
    
    print(f"Starting to process {len(h5ad_list)} files for merging...")
    print(f"Adding '{batch_key}' annotation to each dataset...\n")
    
    # Add tissue information to each AnnData object
    annotated_adatas = []
    total_cells = 0
    
    for idx, (tissue_name, adata) in enumerate(h5ad_list, 1):
        print(f"[{idx}/{len(h5ad_list)}] Processing {tissue_name}...")
        
        # Create a copy to avoid modifying the original
        adata_copy = adata.copy()
        
        # Add tissue information to obs
        adata_copy.obs[batch_key] = tissue_name
        
        # Also add a unique cell ID prefix based on tissue
        print(f"    - Adding tissue prefix to {len(adata_copy.obs_names):,} cell IDs...")
        adata_copy.obs_names = [f"{tissue_name}_{cell_id}" for cell_id in adata_copy.obs_names]
        
        annotated_adatas.append(adata_copy)
        total_cells += adata_copy.n_obs
        print(f"    ✓ Done! Cumulative cells: {total_cells:,}\n")
    
    print(f"{'='*60}")
    print(f"Concatenating all {len(annotated_adatas)} datasets...")
    print(f"Total cells to merge: {total_cells:,}")
    print(f"{'='*60}\n")
    
    logger.info(f"Merging {len(annotated_adatas)} h5ad files...")
    
    # Concatenate all AnnData objects
    # outer join to keep all genes, even if they're not in all datasets
    print("Performing concatenation (this may take a while for large datasets)...")
    merged_adata = ad.concat(
        annotated_adatas,
        join='outer',  # Keep all genes
        merge='same',  # Merge same columns in obs/var
        uns_merge='unique',  # Keep unique uns entries
        label=batch_key,
        keys=[tissue for tissue, _ in h5ad_list],
        index_unique=None  # We already made indices unique
    )
    print("✓ Concatenation complete!\n")
    
    # Fill NaN values with 0 for missing genes
    if merged_adata.X is not None and hasattr(merged_adata.X, 'toarray'):
        # For sparse matrices
        print("Converting to CSR format for efficient storage...")
        merged_adata.X = merged_adata.X.tocsr()
        print("✓ Conversion complete!\n")
    
    logger.info(f"Merged dataset: {merged_adata.n_obs} cells, {merged_adata.n_vars} genes")
    
    # Print tissue distribution
    print(f"{'='*60}")
    print("TISSUE DISTRIBUTION IN MERGED DATASET:")
    print(f"{'='*60}")
    tissue_counts = merged_adata.obs[batch_key].value_counts()
    logger.info("Tissue distribution:")
    for tissue, count in tissue_counts.items():
        print(f"  {tissue}: {count:,} cells ({count/merged_adata.n_obs*100:.1f}%)")
        logger.info(f"  {tissue}: {count} cells")
    print(f"{'='*60}\n")
    
    return merged_adata


def main(directory: str, output_path: Optional[str] = None, batch_key: str = "tissue"):
    """
    Main function to merge h5ad files.
    
    Args:
        directory: Path to directory containing h5ad files
        output_path: Path for output file (default: directory/all.h5ad)
        batch_key: Name of the column to store tissue information
    """
    try:
        print("\n" + "="*60)
        print("H5AD FILE MERGER")
        print("="*60)
        print(f"Directory: {directory}")
        print(f"Tissue column name: {batch_key}")
        print("="*60 + "\n")
        
        # Load all h5ad files
        print("STEP 1: Loading h5ad files")
        print("-"*40)
        logger.info(f"Loading h5ad files from {directory}")
        h5ad_list = load_h5ad_files(directory)
        logger.info(f"Found {len(h5ad_list)} h5ad files")
        
        # Merge files
        print("STEP 2: Merging datasets")
        print("-"*40)
        merged_adata = merge_h5ad_files(h5ad_list, batch_key=batch_key)
        
        # Determine output path
        if output_path is None:
            output_path = os.path.join(directory, "all.h5ad")
        
        # Save merged file
        print("STEP 3: Saving merged dataset")
        print("-"*40)
        print(f"Output path: {output_path}")
        print("Writing h5ad file (this may take a while for large datasets)...")
        logger.info(f"Saving merged file to {output_path}")
        merged_adata.write_h5ad(output_path, compression='gzip')
        print("✓ File saved successfully!\n")
        logger.info(f"Successfully saved merged h5ad file!")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("MERGE COMPLETE - SUMMARY")
        print("="*60)
        print(f"✓ Total cells: {merged_adata.n_obs:,}")
        print(f"✓ Total genes: {merged_adata.n_vars:,}")
        print(f"✓ Number of tissues: {merged_adata.obs[batch_key].nunique()}")
        print(f"✓ Output file: {output_path}")
        print(f"✓ File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")
        print("="*60 + "\n")
        
        return merged_adata
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR OCCURRED")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print(f"{'='*60}\n")
        logger.error(f"Error during merging: {e}")
        raise


if __name__ == "__main__":
    
    # Run the merge
    main("/dcl01/hongkai/data/data/hjiang/Data/paired/atac")