from pathlib import Path
from typing import Union
import pandas as pd
from anndata import AnnData
def merge_sample_metadata(
    adata: AnnData,
    metadata_path: Union[str, Path],
    sample_column: str = "sample",
    verbose: bool = True,
) -> AnnData:
    """
    Merge sample-level metadata (CSV file) with AnnData.obs and standardize
    the sample column to 'sample'.
    Assumes `metadata_path` is a CSV- or CSV-like text file.
    """
    metadata_path = Path(metadata_path)
    
    # ------------------------
    # Load metadata (CSV only)
    # ------------------------
    if verbose:
        print(f"   📄 Reading CSV metadata file: {metadata_path}")
    
    # encoding='utf-8-sig' strips a potential BOM at the start of the file
    meta = pd.read_csv(metadata_path, sep=None, engine="python", encoding="utf-8-sig")
    
    # Normalize column names: remove BOMs and surrounding whitespace
    meta.columns = (
        meta.columns.astype(str)
        .str.replace(r"^\ufeff", "", regex=True)  # remove any leading BOM char
        .str.strip()
    )
    
    if sample_column not in meta.columns:
        raise ValueError(
            f"❌ Sample column '{sample_column}' not in metadata.\n"
            f"   Columns found: {list(meta.columns)}"
        )
    
    # Use sample column as index for merge
    meta = meta.set_index(sample_column)
    
    # Track original number of columns
    original_cols = adata.obs.shape[1]
    
    # Clean metadata string columns
    for col in meta.columns:
        if meta[col].dtype == "object":
            meta[col] = meta[col].fillna("Unknown").astype(str)
    
    # If the AnnData already has the sample column, use it; otherwise warn
    if sample_column in adata.obs.columns:
        sample_vals = adata.obs[sample_column]
    elif "sample" in adata.obs.columns:
        sample_vals = adata.obs["sample"]
    else:
        sample_vals = None
        if verbose:
            print("   ⚠️ No explicit sample column in adata.obs — merge may fail for some rows")
    
    # **FIX: Handle overlapping columns**
    # Find columns that exist in both dataframes (excluding the join key)
    overlapping_cols = adata.obs.columns.intersection(meta.columns)
    
    if len(overlapping_cols) > 0:
        if verbose:
            print(f"   🔄 Dropping {len(overlapping_cols)} overlapping columns from adata.obs: {list(overlapping_cols)}")
            print(f"      Will use metadata versions instead")
        # Drop overlapping columns from adata.obs (keep metadata versions)
        adata.obs = adata.obs.drop(columns=overlapping_cols)
    
    # Perform merge
    adata.obs = adata.obs.join(meta, on=sample_column, how="left")
    
    # Standardize final column to "sample"
    if sample_column != "sample":
        if sample_column in adata.obs.columns:
            adata.obs["sample"] = adata.obs[sample_column]
            adata.obs = adata.obs.drop(columns=[sample_column])
            if verbose:
                print(f"   🔄 Renamed '{sample_column}' ➝ 'sample'")
    
    # Reporting
    added_cols = adata.obs.shape[1] - original_cols
    total_cells = adata.obs.shape[0]
    
    if sample_vals is not None:
        matched = sample_vals.isin(meta.index).sum()
        if verbose:
            print(f"   ✅ Added {added_cols} new metadata columns")
            print(f"   🔗 Matched {matched}/{total_cells} entries ({matched/total_cells*100:.1f}%)")
            if matched < total_cells:
                print(f"   ⚠️ {total_cells - matched} rows have missing metadata")
    else:
        if verbose:
            print(f"   ⚠️ Added {added_cols} columns (could not verify sample matching)")
    
    return adata