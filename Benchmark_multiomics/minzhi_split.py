import os
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Optional

def split_and_concat_multiome_data(
    folder_path: str,
    output_rna_path: str = "concatenated_rna.h5ad",
    output_atac_path: str = "concatenated_atac.h5ad",
    file_pattern: str = "*.h5ad",
    join: str = "outer"
) -> tuple[ad.AnnData, ad.AnnData]:
    """
    Split combined RNA+ATAC h5ad files into separate modalities and concatenate.
    
    Parameters
    ----------
    folder_path : str
        Path to folder containing h5ad files with combined RNA+ATAC data
    output_rna_path : str
        Path to save concatenated RNA h5ad file
    output_atac_path : str
        Path to save concatenated ATAC h5ad file
    file_pattern : str
        Glob pattern to match h5ad files (default: "*.h5ad")
    join : str
        How to handle var indices during concatenation ('inner' or 'outer')
        - 'outer': keep all features (union)
        - 'inner': keep only common features (intersection)
    
    Returns
    -------
    rna_adata : AnnData
        Concatenated RNA data
    atac_adata : AnnData
        Concatenated ATAC data
    """
    
    folder = Path(folder_path)
    h5ad_files = sorted(folder.glob(file_pattern))
    
    if not h5ad_files:
        raise ValueError(f"No h5ad files found in {folder_path} matching pattern {file_pattern}")
    
    print(f"Found {len(h5ad_files)} h5ad files")
    
    rna_datasets = []
    atac_datasets = []
    
    for i, file_path in enumerate(h5ad_files, 1):
        print(f"\n[{i}/{len(h5ad_files)}] Processing: {file_path.name}")
        
        # Read the combined data
        adata = sc.read_h5ad(file_path)
        print(f"  Shape: {adata.shape}")
        
        # Check if 'mod' column exists
        if 'mod' not in adata.var.columns:
            print(f"  WARNING: 'mod' column not found in {file_path.name}, skipping...")
            continue
        
        # Split by modality
        rna_mask = adata.var['mod'] == 'rna'
        atac_mask = adata.var['mod'] == 'atac'
        
        rna_data = adata[:, rna_mask].copy()
        atac_data = adata[:, atac_mask].copy()
        
        print(f"  RNA features: {rna_data.n_vars}")
        print(f"  ATAC features: {atac_data.n_vars}")
        
        rna_datasets.append(rna_data)
        atac_datasets.append(atac_data)
    
    if not rna_datasets or not atac_datasets:
        raise ValueError("No valid RNA or ATAC datasets found")
    
    # Concatenate all RNA datasets
    print(f"\n🧬 Concatenating {len(rna_datasets)} RNA datasets...")
    rna_concat = ad.concat(
        rna_datasets,
        join=join,
        label="source_file",
        keys=[f.stem for f in h5ad_files[:len(rna_datasets)]],
        index_unique="-"
    )
    print(f"  Final RNA shape: {rna_concat.shape}")
    
    # Concatenate all ATAC datasets
    print(f"\n🔬 Concatenating {len(atac_datasets)} ATAC datasets...")
    atac_concat = ad.concat(
        atac_datasets,
        join=join,
        label="source_file",
        keys=[f.stem for f in h5ad_files[:len(atac_datasets)]],
        index_unique="-"
    )
    print(f"  Final ATAC shape: {atac_concat.shape}")
    
    # Save concatenated datasets
    print(f"\n💾 Saving concatenated RNA to: {output_rna_path}")
    rna_concat.write_h5ad(output_rna_path)
    
    print(f"💾 Saving concatenated ATAC to: {output_atac_path}")
    atac_concat.write_h5ad(output_atac_path)
    
    print("\n✅ Done!")
    
    return rna_concat, atac_concat


# Example usage:
if __name__ == "__main__":
    folder_path = "/dcs07/hongkai/data/mjiang/scMultiomics/database/HCA_hyt/processed/PMID_31436334"
    
    rna_adata, atac_adata = split_and_concat_multiome_data(
        folder_path=folder_path,
        output_rna_path="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/rna_raw.h5ad",
        output_atac_path="/dcs07/hongkai/data/harry/result/multi_omics_eye/data/atac_raw.h5ad",
        join="outer"  # or "inner" if you want only common features
    )
    
    print(f"\nRNA data: {rna_adata.shape}")
    print(f"ATAC data: {atac_adata.shape}")