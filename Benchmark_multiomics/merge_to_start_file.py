import os
from pathlib import Path
from typing import List, Optional

import anndata as ad
import numpy as np


RNA_DIR = Path("/dcl01/hongkai/data/data/hjiang/Data/paired/rna")
ATAC_DIR = Path("/dcs07/hongkai/data/harry/result/Benchmark_omics/gene_activity/h5ad")


def merge_paired_rna_atac(
    rna_dir: Path = RNA_DIR,
    atac_dir: Path = ATAC_DIR,
    output_path: Optional[Path] = None,
    join_genes: str = "inner",  # "inner" = intersection of genes between RNA/ATAC per tissue
) -> ad.AnnData:
    """
    Merge paired RNA and ATAC (gene activity) h5ad files into one big AnnData.

    Steps:
      1) For each tissue with both RNA and ATAC files:
         - Load RNA and ATAC
         - Add obs['tissue'] from filename
      2) Check pairing:
         - For each tissue, keep only cells present in BOTH RNA and ATAC
         - Print summary of overlaps/mismatches
      3) For each tissue, create RNA/ATAC combined AnnData:
         - Add obs['modality'] = 'RNA' / 'ATAC'
         - Add obs['original_barcode'] = original obs_names
         - Rename obs_names to "<original_barcode>_RNA" / "_ATAC"
      4) Concatenate all tissues into one big AnnData.
    """
    print("============================================================")
    print(" Merging paired RNA + ATAC (gene activity) h5ad files")
    print("============================================================")
    print(f"RNA directory : {rna_dir}")
    print(f"ATAC directory: {atac_dir}")
    print(f"Gene join mode per tissue (RNA vs ATAC): {join_genes!r}")
    print("------------------------------------------------------------\n")

    if not rna_dir.exists():
        raise FileNotFoundError(f"RNA directory does not exist: {rna_dir}")
    if not atac_dir.exists():
        raise FileNotFoundError(f"ATAC directory does not exist: {atac_dir}")

    rna_files = sorted(rna_dir.glob("*.h5ad"))
    print(f"Found {len(rna_files)} RNA h5ad files")

    per_tissue_adatas: List[ad.AnnData] = []
    tissues_processed: List[str] = []

    for rna_path in rna_files:
        tissue = rna_path.stem  # e.g. "adrenal_gland"
        atac_path = atac_dir / f"{tissue}.h5ad"

        print("\n------------------------------------------------------------")
        print(f"Tissue: {tissue}")
        print(f"  RNA file : {rna_path}")
        print(f"  ATAC file: {atac_path}")

        if not atac_path.exists():
            print(f"  [WARNING] ATAC file for tissue '{tissue}' not found. Skipping this tissue.")
            continue

        # 1) Load RNA and ATAC for this tissue
        print("  Loading RNA AnnData...")
        rna = ad.read_h5ad(rna_path)
        print(f"    RNA shape: {rna.shape}")

        print("  Loading ATAC AnnData...")
        atac = ad.read_h5ad(atac_path)
        print(f"    ATAC shape: {atac.shape}")

        # Add 'tissue' column
        print("  Adding 'tissue' column to obs...")
        rna.obs["tissue"] = tissue
        atac.obs["tissue"] = tissue

        # 2) Check pairing: keep only cells present in BOTH modalities
        rna_cells = set(rna.obs_names)
        atac_cells = set(atac.obs_names)
        common_cells = sorted(rna_cells & atac_cells)
        rna_only = rna_cells - atac_cells
        atac_only = atac_cells - rna_cells

        print("  Cell pairing check:")
        print(f"    RNA cells total   : {len(rna_cells)}")
        print(f"    ATAC cells total  : {len(atac_cells)}")
        print(f"    Common cells      : {len(common_cells)}")
        print(f"    RNA-only cells    : {len(rna_only)}")
        print(f"    ATAC-only cells   : {len(atac_only)}")

        if len(common_cells) == 0:
            print("  [ERROR] No overlapping cells between RNA and ATAC for this tissue. Skipping.")
            continue

        if len(atac_only) > 0:
            print("  [WARNING] Some ATAC cells do not have a matching RNA cell. They will be dropped.")
        if len(rna_only) > 0:
            print("  [WARNING] Some RNA cells do not have a matching ATAC cell. They will be dropped.")

        # Subset to common cells (same order in both objects)
        print("  Subsetting RNA and ATAC to common cells...")
        rna = rna[common_cells].copy()
        atac = atac[common_cells].copy()
        print(f"    RNA subset shape : {rna.shape}")
        print(f"    ATAC subset shape: {atac.shape}")

        # 3) Ensure genes align reasonably (inner join across modalities for this tissue)
        # We will handle alignment via ad.concat with join=join_genes.
        print(f"  Preparing to merge RNA & ATAC for tissue '{tissue}'...")

        # Add original_barcode + modality; then rename obs_names
        print("  Adding 'original_barcode' and 'modality', renaming obs_names with _RNA/_ATAC suffixes...")
        rna.obs["original_barcode"] = rna.obs_names.astype(str)
        atac.obs["original_barcode"] = atac.obs_names.astype(str)

        rna.obs["modality"] = "RNA"
        atac.obs["modality"] = "ATAC"

        rna.obs_names = [f"{bc}_RNA" for bc in rna.obs["original_barcode"].tolist()]
        atac.obs_names = [f"{bc}_ATAC" for bc in atac.obs["original_barcode"].tolist()]

        print("    Example RNA barcode mapping (first 3):")
        for old, new in list(zip(rna.obs["original_barcode"], rna.obs_names))[:3]:
            print(f"      {old} -> {new}")
        print("    Example ATAC barcode mapping (first 3):")
        for old, new in list(zip(atac.obs["original_barcode"], atac.obs_names))[:3]:
            print(f"      {old} -> {new}")

        # 4) Merge RNA + ATAC for this tissue along cells (axis=0)
        # join=join_genes: how to handle gene sets (var)
        print(f"  Concatenating RNA + ATAC for tissue '{tissue}' (join={join_genes!r})...")
        merged_tissue = ad.concat(
            [rna, atac],
            axis=0,
            join=join_genes,    # usually "inner" to keep genes present in both modalities
            label=None,
            index_unique=None,
        )
        print(f"    Merged tissue AnnData shape: {merged_tissue.shape}")

        per_tissue_adatas.append(merged_tissue)
        tissues_processed.append(tissue)

    if not per_tissue_adatas:
        raise RuntimeError("No tissues were successfully processed. Check your inputs.")

    print("\n============================================================")
    print(" Concatenating all tissues into one big AnnData object")
    print("============================================================")
    print(f"Tissues processed: {tissues_processed}")
    big_adata = ad.concat(
        per_tissue_adatas,
        axis=0,
        join="outer",     # across tissues, use outer to keep union of genes
        label=None,
        index_unique=None,
    )
    print(f"Final merged AnnData shape: {big_adata.shape}")
    print("Columns in obs:")
    print(f"  {list(big_adata.obs.columns)}")
    print("------------------------------------------------------------")

    # Optional: save to disk
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving merged AnnData to: {output_path}")
        big_adata.write_h5ad(output_path)
        print("Save complete.")

    return big_adata

from pathlib import Path
from typing import Optional

import anndata as ad


def merge_single_rna_atac(
    rna_path: Path,
    atac_path: Path,
    output_path: Optional[Path] = None,
    join_genes: str = "inner",  # how to align genes between RNA & ATAC
) -> ad.AnnData:
    """
    Merge a single RNA h5ad and a single ATAC (gene activity) h5ad into one AnnData.

    Steps:
      1) Load RNA and ATAC AnnData
      2) Keep only cells present in BOTH RNA and ATAC (paired barcodes)
      3) Add obs['modality'] = 'RNA' / 'ATAC'
         Add obs['original_barcode'] = original obs_names
         Rename obs_names to "<barcode>_RNA" / "<barcode>_ATAC"
      4) Concatenate along cells (axis=0) with gene join given by `join_genes`
      5) Optionally save to `output_path`
    """

    print("============================================================")
    print(" Merging single RNA + ATAC (gene activity) h5ad files")
    print("============================================================")
    print(f"RNA file : {rna_path}")
    print(f"ATAC file: {atac_path}")
    print(f"Gene join mode (RNA vs ATAC): {join_genes!r}")
    print("------------------------------------------------------------\n")

    # --- 1) Load RNA & ATAC ---
    if not rna_path.exists():
        raise FileNotFoundError(f"RNA file does not exist: {rna_path}")
    if not atac_path.exists():
        raise FileNotFoundError(f"ATAC file does not exist: {atac_path}")

    print("Loading RNA AnnData...")
    rna = ad.read_h5ad(rna_path)
    print(f"  RNA shape: {rna.shape}")

    print("Loading ATAC AnnData...")
    atac = ad.read_h5ad(atac_path)
    print(f"  ATAC shape: {atac.shape}")

    # --- 2) Cell pairing: keep only overlapping barcodes ---
    rna_cells = set(rna.obs_names)
    atac_cells = set(atac.obs_names)
    common_cells = sorted(rna_cells & atac_cells)
    rna_only = rna_cells - atac_cells
    atac_only = atac_cells - rna_cells

    print("\nCell pairing check:")
    print(f"  RNA cells total   : {len(rna_cells)}")
    print(f"  ATAC cells total  : {len(atac_cells)}")
    print(f"  Common cells      : {len(common_cells)}")
    print(f"  RNA-only cells    : {len(rna_only)}")
    print(f"  ATAC-only cells   : {len(atac_only)}")

    if len(common_cells) == 0:
        raise RuntimeError("No overlapping cells between RNA and ATAC. Cannot merge.")

    if len(atac_only) > 0:
        print("  [WARNING] Some ATAC cells do not have matching RNA cells. They will be dropped.")
    if len(rna_only) > 0:
        print("  [WARNING] Some RNA cells do not have matching ATAC cells. They will be dropped.")

    print("\nSubsetting to common cells...")
    rna = rna[common_cells].copy()
    atac = atac[common_cells].copy()
    print(f"  RNA subset shape : {rna.shape}")
    print(f"  ATAC subset shape: {atac.shape}")

    # --- 3) Add metadata + rename obs_names with modality suffix ---
    print("\nAdding 'original_barcode' and 'modality', renaming obs_names...")
    rna.obs["original_barcode"] = rna.obs_names.astype(str)
    atac.obs["original_barcode"] = atac.obs_names.astype(str)

    rna.obs["modality"] = "RNA"
    atac.obs["modality"] = "ATAC"

    rna.obs_names = [f"{bc}_RNA" for bc in rna.obs["original_barcode"].tolist()]
    atac.obs_names = [f"{bc}_ATAC" for bc in atac.obs["original_barcode"].tolist()]

    print("  Example RNA barcode mapping (first 3):")
    for old, new in list(zip(rna.obs["original_barcode"], rna.obs_names))[:3]:
        print(f"    {old} -> {new}")
    print("  Example ATAC barcode mapping (first 3):")
    for old, new in list(zip(atac.obs["original_barcode"], atac.obs_names))[:3]:
        print(f"    {old} -> {new}")

    # --- 4) Concatenate along cells (axis=0) ---
    print(f"\nConcatenating RNA + ATAC (join={join_genes!r})...")
    merged = ad.concat(
        [rna, atac],
        axis=0,
        join=join_genes,   # e.g. "inner" to keep genes present in both
        label=None,
        index_unique=None,
    )
    print(f"  Merged AnnData shape: {merged.shape}")
    print("  obs columns:", list(merged.obs.columns))

    # --- 5) Optional save ---
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving merged AnnData to: {output_path}")
        merged.write_h5ad(output_path)
        print("Save complete.")

    return merged


if __name__ == "__main__":
    # out_file = Path("/dcs07/hongkai/data/harry/result/Benchmark_omics/paired_rna_atac_merged.h5ad")
    # merged = merge_paired_rna_atac(output_path=out_file)
    rna_file = Path("/dcs07/hongkai/data/harry/result/multi_omics_heart/data/rna_raw.h5ad")
    atac_file = Path("/dcs07/hongkai/data/harry/result/multi_omics_heart/data/heart_gene_activity.h5ad")
    out_file = Path("/dcs07/hongkai/data/harry/result/multi_omics_heart/data/paired_rna_atac_merged.h5ad")

    merged = merge_single_rna_atac(
        rna_path=rna_file,
        atac_path=atac_file,
        output_path=out_file,
        join_genes="inner",   # or "outer" if you want union of genes
    )
