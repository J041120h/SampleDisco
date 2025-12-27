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
    Merge sample-level metadata with AnnData.obs and standardize sample column to 'sample'.
    Automatically detects file format (CSV, TSV, TXT, Excel).
    """
    metadata_path = Path(metadata_path)
    ext = metadata_path.suffix.lower()

    # ------------------------
    # Load metadata (auto mode)
    # ------------------------
    if ext in {".xls", ".xlsx"}:
        if verbose: print(f"   📄 Reading Excel file: {metadata_path}")
        meta = pd.read_excel(metadata_path)
    else:
        if verbose: print(f"   📄 Reading text file (auto-detect sep): {metadata_path}")
        # sep=None + engine="python" makes pandas detect the delimiter automatically
        meta = pd.read_csv(metadata_path, sep=None, engine="python")

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
