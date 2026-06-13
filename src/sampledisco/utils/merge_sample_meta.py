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
    Merge sample-level metadata (CSV file) into AnnData.obs, then ensure the
    join-key column is named 'sample'.

    The join is a left join on ``sample_column``. Overlapping columns in
    ``adata.obs`` are dropped in favour of the metadata file's version.
    If ``sample_column != 'sample'``, the column is renamed after the join.

    Parameters
    ----------
    adata : AnnData
    metadata_path : str or Path
        CSV file (separator auto-detected). UTF-8-BOM files are handled.
    sample_column : str
        Column identifying samples in both the CSV and ``adata.obs``.
    verbose : bool

    Returns
    -------
    AnnData  (obs modified in-place, same object returned)
    """
    metadata_path = Path(metadata_path)

    if verbose:
        print(f"   📄 Reading CSV metadata file: {metadata_path}")
    
    # utf-8-sig strips a BOM that Excel sometimes writes at the start of CSVs.
    meta = pd.read_csv(metadata_path, sep=None, engine="python", encoding="utf-8-sig")

    # Belt-and-suspenders: also strip BOM from column names (some tools write it mid-stream).
    meta.columns = (
        meta.columns.astype(str)
        .str.replace(r"^\ufeff", "", regex=True)
        .str.strip()
    )
    
    if sample_column not in meta.columns:
        raise ValueError(
            f"❌ Sample column '{sample_column}' not in metadata.\n"
            f"   Columns found: {list(meta.columns)}"
        )
    
    meta = meta.set_index(sample_column)

    original_cols = adata.obs.shape[1]

    for col in meta.columns:
        if meta[col].dtype == "object":
            meta[col] = meta[col].fillna("Unknown").astype(str)

    if sample_column in adata.obs.columns:
        sample_vals = adata.obs[sample_column]
    elif "sample" in adata.obs.columns:
        sample_vals = adata.obs["sample"]
    else:
        sample_vals = None
        if verbose:
            print("   ⚠️ No explicit sample column in adata.obs — merge may fail for some rows")
    
    # Drop overlapping columns before the join so metadata versions win.
    overlapping_cols = adata.obs.columns.intersection(meta.columns)

    if len(overlapping_cols) > 0:
        if verbose:
            print(f"   🔄 Dropping {len(overlapping_cols)} overlapping columns from adata.obs: {list(overlapping_cols)}")
            print(f"      Will use metadata versions instead")
        adata.obs = adata.obs.drop(columns=overlapping_cols)

    adata.obs = adata.obs.join(meta, on=sample_column, how="left")

    if sample_column != "sample":
        if sample_column in adata.obs.columns:
            adata.obs["sample"] = adata.obs[sample_column]
            adata.obs = adata.obs.drop(columns=[sample_column])
            if verbose:
                print(f"   🔄 Renamed '{sample_column}' ➝ 'sample'")
    
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