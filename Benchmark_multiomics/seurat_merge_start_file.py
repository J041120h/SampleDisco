#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge a single paired RNA h5ad and a single ATAC gene-activity h5ad into one AnnData,
in the SAME "stacked-modality" manner as your tissue-based merge:

- keep only overlapping barcodes (paired cells)
- add obs['modality'] and obs['original_barcode']
- rename obs_names to "<barcode>_RNA" / "<barcode>_ATAC"
- concatenate along cells (axis=0)
- (optional) save to disk

NOTE: Does NOT change .X (no normalization/log/etc.). Only subsets and adds obs columns.
"""

from pathlib import Path
from typing import Optional

import anndata as ad


def merge_rna_atac_two_files(
    rna_path: Path,
    atac_path: Path,
    output_path: Optional[Path] = None,
    join_genes: str = "inner",  # "inner" keeps genes present in BOTH; "outer" keeps union
) -> ad.AnnData:
    print("============================================================")
    print(" Merging RNA + ATAC gene-activity (two-file stacked merge)")
    print("============================================================")
    print(f"RNA file : {rna_path}")
    print(f"ATAC file: {atac_path}")
    print(f"Gene join mode (RNA vs ATAC): {join_genes!r}")
    print("------------------------------------------------------------\n")

    # --- Validate paths ---
    rna_path = Path(rna_path)
    atac_path = Path(atac_path)
    if not rna_path.exists():
        raise FileNotFoundError(f"RNA file does not exist: {rna_path}")
    if not atac_path.exists():
        raise FileNotFoundError(f"ATAC file does not exist: {atac_path}")

    # --- 1) Load ---
    print("Loading RNA AnnData...")
    rna = ad.read_h5ad(rna_path)
    print(f"  RNA shape: {rna.shape}")

    print("Loading ATAC AnnData...")
    atac = ad.read_h5ad(atac_path)
    print(f"  ATAC shape: {atac.shape}")

    # --- 2) Pairing by barcode intersection ---
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

    print("\nSubsetting RNA and ATAC to common cells (same order)...")
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
    print(f"\nConcatenating RNA + ATAC (axis=0, join={join_genes!r})...")
    merged = ad.concat(
        [rna, atac],
        axis=0,
        join=join_genes,
        label=None,
        index_unique=None,
    )
    print(f"  Merged AnnData shape: {merged.shape}")
    print(f"  obs columns: {list(merged.obs.columns)}")

    # --- 5) Optional save ---
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving merged AnnData to: {output_path}")
        merged.write_h5ad(output_path)
        print("Save complete.")

    return merged


if __name__ == "__main__":
    # Example usage (edit paths)
    rna_file = Path("/dcs07/hongkai/data/harry/result/multi_omics_eye/data/rna/lutea.h5ad")
    atac_file = Path("/dcs07/hongkai/data/harry/result/multi_omics_eye/data/merge/lutea_atac_integrated.h5ad")
    out_file = Path("/dcs07/hongkai/data/harry/result/multi_omics_eye/data/merge/lutea_rna_atac_merged.h5ad")

    _ = merge_rna_atac_two_files(
        rna_path=rna_file,
        atac_path=atac_file,
        output_path=out_file,
        join_genes="inner",  # "inner" or "outer"
    )
