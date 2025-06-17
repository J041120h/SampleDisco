#!/usr/bin/env python3
import anndata as ad
import numpy as np
import pandas as pd

def inspect_anndata(adata_path, verbose=True):
    """
    Comprehensively inspect an AnnData object, showing contents of .var, .obs, .uns, and .obsm
    
    Parameters:
    -----------
    adata_path : str
        Path to the h5ad file
    verbose : bool
        If True, show detailed information about data types and shapes
    """
    # Load AnnData
    adata = ad.read_h5ad(adata_path)
    
    print("="*80)
    print(f"AnnData object overview")
    print("="*80)
    print(f"Shape: {adata.shape[0]} observations × {adata.shape[1]} variables")
    print()
    
    # Print .var columns
    print("Columns in .var:")
    print("-"*40)
    if len(adata.var.columns) > 0:
        for col in adata.var.columns:
            if verbose:
                dtype = adata.var[col].dtype
                n_unique = adata.var[col].nunique() if dtype == 'object' else 'N/A'
                print(f"  - {col}: {dtype} (unique values: {n_unique})")
            else:
                print(f"  - {col}")
    else:
        print("  (No columns in .var)")
    
    # Print .obs columns
    print("\nColumns in .obs:")
    print("-"*40)
    if len(adata.obs.columns) > 0:
        for col in adata.obs.columns:
            if verbose:
                dtype = adata.obs[col].dtype
                n_unique = adata.obs[col].nunique() if dtype in ['object', 'category'] else 'N/A'
                print(f"  - {col}: {dtype} (unique values: {n_unique})")
            else:
                print(f"  - {col}")
    else:
        print("  (No columns in .obs)")
    
    # Print .uns keys and their contents
    print("\nContents of .uns (unstructured annotations):")
    print("-"*40)
    if len(adata.uns.keys()) > 0:
        for key in sorted(adata.uns.keys()):
            value = adata.uns[key]
            if isinstance(value, pd.DataFrame):
                print(f"  - {key}: DataFrame with shape {value.shape}")
                if verbose:
                    print(f"      Columns: {list(value.columns)[:5]}{'...' if len(value.columns) > 5 else ''}")
                    print(f"      Index: {list(value.index)[:3]}{'...' if len(value.index) > 3 else ''}")
            elif isinstance(value, np.ndarray):
                print(f"  - {key}: numpy array with shape {value.shape} and dtype {value.dtype}")
            elif isinstance(value, dict):
                print(f"  - {key}: dict with {len(value)} keys")
                if verbose and len(value) <= 10:
                    for k in list(value.keys())[:5]:
                        print(f"      '{k}': {type(value[k]).__name__}")
            elif isinstance(value, (list, tuple)):
                print(f"  - {key}: {type(value).__name__} with {len(value)} elements")
            else:
                print(f"  - {key}: {type(value).__name__}")
                if verbose and hasattr(value, 'shape'):
                    print(f"      Shape: {value.shape}")
    else:
        print("  (No entries in .uns)")
    
    # Print .obsm keys (multi-dimensional observations)
    print("\nContents of .obsm (multi-dimensional observations):")
    print("-"*40)
    if len(adata.obsm.keys()) > 0:
        for key in sorted(adata.obsm.keys()):
            value = adata.obsm[key]
            if isinstance(value, np.ndarray):
                print(f"  - {key}: array with shape {value.shape} and dtype {value.dtype}")
                if verbose:
                    print(f"      Range: [{np.min(value):.3f}, {np.max(value):.3f}]")
            elif hasattr(value, 'shape'):
                print(f"  - {key}: {type(value).__name__} with shape {value.shape}")
            else:
                print(f"  - {key}: {type(value).__name__}")
    else:
        print("  (No entries in .obsm)")
    
    # Print .varm keys (multi-dimensional variables)
    print("\nContents of .varm (multi-dimensional variables):")
    print("-"*40)
    if len(adata.varm.keys()) > 0:
        for key in sorted(adata.varm.keys()):
            value = adata.varm[key]
            if isinstance(value, np.ndarray):
                print(f"  - {key}: array with shape {value.shape} and dtype {value.dtype}")
            else:
                print(f"  - {key}: {type(value).__name__}")
    else:
        print("  (No entries in .varm)")
    
    # Print .layers keys
    print("\nLayers:")
    print("-"*40)
    if len(adata.layers.keys()) > 0:
        for key in sorted(adata.layers.keys()):
            layer = adata.layers[key]
            if hasattr(layer, 'shape'):
                print(f"  - {key}: shape {layer.shape}")
            else:
                print(f"  - {key}")
    else:
        print("  (No layers)")
    
    print("\n" + "="*80)
    
    return adata


def main(adata_path):
    """
    Main function to inspect AnnData file
    """
    inspect_anndata(adata_path, verbose=True)


# Additional utility function to specifically check for dimension reduction results
def check_dimred_results(adata_path):
    """
    Specifically check for dimension reduction results in an AnnData object
    """
    adata = ad.read_h5ad(adata_path)
    
    print("\nChecking for dimension reduction results:")
    print("-"*50)
    
    # Common dimension reduction keys
    common_dr_keys = ['X_pca', 'X_umap', 'X_tsne', 'X_lsi', 'X_spectral', 
                      'X_DR_expression', 'X_DR_proportion', 'X_pca_expression',
                      'X_lsi_expression', 'X_spectral_expression']
    
    # Check in .obsm
    print("In .obsm:")
    found_in_obsm = []
    for key in common_dr_keys:
        if key in adata.obsm:
            shape = adata.obsm[key].shape
            print(f"  ✓ {key}: shape {shape}")
            found_in_obsm.append(key)
    
    # Check in .uns
    print("\nIn .uns:")
    found_in_uns = []
    for key in common_dr_keys:
        if key in adata.uns:
            value = adata.uns[key]
            if isinstance(value, pd.DataFrame):
                print(f"  ✓ {key}: DataFrame with shape {value.shape}")
            elif isinstance(value, np.ndarray):
                print(f"  ✓ {key}: array with shape {value.shape}")
            else:
                print(f"  ✓ {key}: {type(value).__name__}")
            found_in_uns.append(key)
    
    # Check for any other potential DR results
    print("\nOther potential dimension reduction results:")
    for key in adata.obsm.keys():
        if key not in common_dr_keys and ('pca' in key.lower() or 'umap' in key.lower() or 
                                          'tsne' in key.lower() or 'lsi' in key.lower() or
                                          'dr' in key.lower() or 'spectral' in key.lower()):
            print(f"  - In .obsm: {key} (shape: {adata.obsm[key].shape})")
    
    for key in adata.uns.keys():
        if key not in common_dr_keys and ('pca' in key.lower() or 'umap' in key.lower() or 
                                          'tsne' in key.lower() or 'lsi' in key.lower() or
                                          'dr' in key.lower() or 'spectral' in key.lower()):
            value = adata.uns[key]
            if isinstance(value, pd.DataFrame):
                print(f"  - In .uns: {key} (DataFrame, shape: {value.shape})")
            elif isinstance(value, np.ndarray):
                print(f"  - In .uns: {key} (array, shape: {value.shape})")
            else:
                print(f"  - In .uns: {key} ({type(value).__name__})")
    
    if not found_in_obsm and not found_in_uns:
        print("  ⚠ No standard dimension reduction results found!")
    
    return adata

import pandas as pd
import anndata as ad

def update_anndata_with_cell_meta(cell_meta, adata_path, index_col='cell_id', overwrite=True):
    """
    Update `.obs` in an AnnData object with external metadata.
    Drops or overwrites overlapping columns. Removes '_index' column if present.

    Parameters:
    - cell_meta: str or pd.DataFrame
    - adata_path: str
    - index_col: str — column in metadata that matches AnnData obs_names
    - overwrite: bool — if True, overwrite overlapping columns

    Returns:
    - None
    """
    # Load metadata
    if isinstance(cell_meta, str):
        if cell_meta.endswith(".csv"):
            meta_df = pd.read_csv(cell_meta, index_col=index_col)
        elif cell_meta.endswith(".tsv"):
            meta_df = pd.read_csv(cell_meta, sep='\t', index_col=index_col)
        else:
            raise ValueError("Unsupported file format.")
    elif isinstance(cell_meta, pd.DataFrame):
        meta_df = cell_meta.set_index(index_col) if index_col in cell_meta.columns else cell_meta
    else:
        raise TypeError("cell_meta must be a path or DataFrame")

    # Load AnnData
    adata = ad.read_h5ad(adata_path)

    # Subset metadata
    matching_meta = meta_df.loc[meta_df.index.intersection(adata.obs_names)]

    # Drop overlapping columns if overwriting
    overlapping_cols = adata.obs.columns.intersection(matching_meta.columns)
    if overwrite:
        adata.obs = adata.obs.drop(columns=overlapping_cols)

    # Join metadata
    adata.obs = adata.obs.join(matching_meta, how='left')

    # Drop invalid column if present
    if '_index' in adata.obs.columns:
        adata.obs.drop(columns=['_index'], inplace=True)

    # Convert object-type columns to string
    for col in adata.obs.select_dtypes(include='object').columns:
        adata.obs[col] = adata.obs[col].astype(str)

    # Save back to disk
    adata.write(adata_path, compression='gzip')

import scanpy as sc
import numpy as np
import anndata as ad
import pandas as pd

def subsample_rna_and_merge_with_atac(
    input_path: str,
    output_path: str,
    modality_col: str = "modality",
    sample_col: str = "sample",
    rna_label: str = "RNA",
    atac_label: str = "ATAC",
    sample_fraction: float = 0.1,
    random_seed: int = 42
):
    """
    Subsample RNA data by selecting a fraction of samples and keeping all cells
    from those selected samples, then merge with ATAC cells.
    
    Parameters:
    -----------
    input_path : str
        Path to input h5ad file
    output_path : str
        Path to save output h5ad file
    modality_col : str
        Column name indicating modality (default: "modality")
    sample_col : str
        Column name indicating sample origin (default: "sample")
    rna_label : str
        Label for RNA cells in modality column (default: "RNA")
    atac_label : str
        Label for ATAC cells in modality column (default: "ATAC")
    sample_fraction : float
        Fraction of samples to keep (default: 0.1)
    random_seed : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    combined_adata : anndata.AnnData
        Combined AnnData object with RNA cells from selected samples and all ATAC cells
    """
    # Load the full AnnData object
    adata = sc.read_h5ad(input_path)
    
    # Filter RNA and ATAC cells
    rna_adata = adata[adata.obs[modality_col] == rna_label].copy()
    atac_adata = adata[adata.obs[modality_col] == atac_label].copy()
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get unique samples in RNA data
    unique_samples = rna_adata.obs[sample_col].unique()
    total_samples = len(unique_samples)
    
    # Calculate number of samples to keep
    n_samples_to_keep = max(1, int(sample_fraction * total_samples))  # Keep at least 1 sample
    
    # Randomly select samples
    if n_samples_to_keep >= total_samples:
        # Keep all samples if fraction would require more samples than available
        selected_samples = unique_samples
    else:
        selected_samples = np.random.choice(
            unique_samples,
            size=n_samples_to_keep,
            replace=False
        )
    
    # Filter RNA data to keep only cells from selected samples
    sample_mask = rna_adata.obs[sample_col].isin(selected_samples)
    rna_subsampled = rna_adata[sample_mask].copy()
    
    # Print summary statistics
    print(f"Total samples in RNA data: {total_samples}")
    print(f"Selected samples: {n_samples_to_keep}")
    print(f"Sample selection ratio: {n_samples_to_keep / total_samples:.3f}")
    print(f"Selected sample names: {sorted(selected_samples)}")
    print(f"\nOriginal RNA cells: {rna_adata.n_obs}")
    print(f"RNA cells after sample selection: {rna_subsampled.n_obs}")
    print(f"Cell retention ratio: {rna_subsampled.n_obs / rna_adata.n_obs:.3f}")
    print(f"ATAC cells: {atac_adata.n_obs}")
    
    # Show sample composition
    print("\nSample composition (cells per sample):")
    original_counts = rna_adata.obs[sample_col].value_counts().sort_index()
    subsampled_counts = rna_subsampled.obs[sample_col].value_counts().sort_index()
    
    for sample in original_counts.index:
        orig_count = original_counts[sample]
        if sample in selected_samples:
            print(f"  {sample}: {orig_count} (KEPT)")
        else:
            print(f"  {sample}: {orig_count} (REMOVED)")
    
    # Concatenate the ATAC cells with the subsampled RNA cells
    combined_adata = ad.concat([rna_subsampled, atac_adata], axis=0, join='outer', label='source', fill_value=0)
    
    # Save the result
    combined_adata.write_h5ad(output_path)
    
    return combined_adata

def print_genes_per_cell(adata_path):
    # Load AnnData
    adata = sc.read_h5ad(adata_path)

    # Compute number of genes per cell
    if hasattr(adata.X, "A1"):  # sparse matrix
        gene_counts = (adata.X > 0).sum(axis=1).A1
    else:
        gene_counts = (adata.X > 0).sum(axis=1)

    # Print results
    for cell, count in zip(adata.obs_names, gene_counts):
        print(f"{cell}\t{count}")

import pandas as pd
import anndata as ad
import numpy as np
from typing import Optional, List, Union

def merge_sample_metadata(
    adata: ad.AnnData,
    metadata_path: str,
    sample_column: str = "sample",
    overwrite_existing: bool = False,
    verbose: bool = True
) -> ad.AnnData:
    """
    Merge sample-level metadata from CSV file into AnnData object.
    Each row in the CSV represents a sample, and metadata will be added 
    to all cells belonging to that sample.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        The AnnData object that already contains a sample column in adata.obs
    metadata_path : str
        Path to the CSV file containing sample metadata (e.g., ATAC_Metadata.csv)
        First column should contain sample names (used as index)
    sample_column : str, optional
        Name of the existing sample column in adata.obs (default: "sample")
    overwrite_existing : bool, optional
        Whether to overwrite existing columns in adata.obs (default: False)
    verbose : bool, optional
        Whether to print information about the merge process
        
    Returns:
    --------
    anndata.AnnData
        AnnData object with merged sample-level metadata
    """
    
    # Check if sample column exists in adata.obs
    if sample_column not in adata.obs.columns:
        raise ValueError(f"Sample column '{sample_column}' not found in adata.obs. "
                        f"Available columns: {list(adata.obs.columns)}")
    
    # Read sample metadata CSV (first column as index = sample names)
    try:
        sample_metadata = pd.read_csv(metadata_path, index_col=0)
        if verbose:
            print(f"Loaded sample metadata with shape: {sample_metadata.shape}")
            print(f"Sample metadata columns: {list(sample_metadata.columns)}")
            print(f"Samples in metadata: {list(sample_metadata.index)}")
    except Exception as e:
        raise FileNotFoundError(f"Could not read metadata file: {e}")
    
    # Get unique samples from adata
    adata_samples = set(adata.obs[sample_column].unique())
    metadata_samples = set(sample_metadata.index)
    
    # Check overlap
    common_samples = adata_samples.intersection(metadata_samples)
    missing_samples = adata_samples - metadata_samples
    
    if verbose:
        print(f"Samples in AnnData: {len(adata_samples)} - {sorted(adata_samples)}")
        print(f"Samples in metadata: {len(metadata_samples)} - {sorted(metadata_samples)}")
        print(f"Common samples: {len(common_samples)} - {sorted(common_samples)}")
        if missing_samples:
            print(f"WARNING: Samples in AnnData but not in metadata: {sorted(missing_samples)}")
    
    if len(common_samples) == 0:
        print("ERROR: No common samples found!")
        return adata
    
    # Identify which columns to add
    new_cols = []
    existing_cols_to_update = []
    skipped_cols = []
    
    for col in sample_metadata.columns:
        if col in adata.obs.columns:
            if overwrite_existing:
                existing_cols_to_update.append(col)
            else:
                skipped_cols.append(col)
        else:
            new_cols.append(col)
    
    # Add sample metadata to cells
    for col in new_cols + existing_cols_to_update:
        # Create a mapping from sample to metadata value
        sample_to_value = sample_metadata[col].to_dict()
        
        # Map each cell's sample to the corresponding metadata value
        adata.obs[col] = adata.obs[sample_column].map(sample_to_value)
        
        # Handle cells from samples not in metadata
        if missing_samples:
            mask = adata.obs[sample_column].isin(missing_samples)
            adata.obs.loc[mask, col] = np.nan
    
    if verbose:
        print(f"\nSuccessfully merged sample metadata:")
        if new_cols:
            print(f"Added {len(new_cols)} new columns: {new_cols}")
        if existing_cols_to_update:
            print(f"Updated {len(existing_cols_to_update)} existing columns: {existing_cols_to_update}")
        if skipped_cols:
            print(f"Skipped {len(skipped_cols)} existing columns: {skipped_cols}")
            print("  (Use overwrite_existing=True to update these)")
        
        # Show sample distribution
        print(f"\nCells per sample:")
        sample_counts = adata.obs[sample_column].value_counts()
        for sample, count in sample_counts.items():
            metadata_status = "✓" if sample in metadata_samples else "✗ (no metadata)"
            print(f"  {sample}: {count} cells {metadata_status}")
    
    return adata


def inspect_sample_metadata_overlap(
    adata: ad.AnnData,
    metadata_path: str,
    sample_column: str = "sample"
) -> dict:
    """
    Inspect the overlap between AnnData samples and sample metadata without merging.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        The AnnData object with existing sample column
    metadata_path : str
        Path to the CSV file containing sample metadata
    sample_column : str, optional
        Name of the sample column in adata.obs (default: "sample")
        
    Returns:
    --------
    dict
        Dictionary with overlap information
    """
    
    # Check if sample column exists
    if sample_column not in adata.obs.columns:
        raise ValueError(f"Sample column '{sample_column}' not found in adata.obs")
    
    # Read sample metadata
    sample_metadata = pd.read_csv(metadata_path, index_col=0)
    
    # Compare samples
    adata_samples = set(adata.obs[sample_column].unique())
    metadata_samples = set(sample_metadata.index)
    common_samples = adata_samples.intersection(metadata_samples)
    
    # Compare column names with existing adata columns
    adata_cols = set(adata.obs.columns)
    metadata_cols = set(sample_metadata.columns)
    common_cols = adata_cols.intersection(metadata_cols)
    new_cols = metadata_cols - adata_cols
    
    overlap_info = {
        'adata_samples': sorted(adata_samples),
        'metadata_samples': sorted(metadata_samples),
        'common_samples': sorted(common_samples),
        'missing_samples': sorted(adata_samples - metadata_samples),
        'sample_overlap_ratio': len(common_samples) / len(adata_samples) if adata_samples else 0,
        'adata_columns': list(adata_cols),
        'metadata_columns': list(metadata_cols),
        'common_columns': list(common_cols),
        'new_columns': list(new_cols),
        'sample_metadata_preview': sample_metadata.head(),
        'cells_per_sample': adata.obs[sample_column].value_counts().to_dict()
    }
    
    return overlap_info

import pandas as pd
import anndata as ad
import numpy as np

def merge_pseudobulk_metadata(
    adata: ad.AnnData,
    metadata_path: str,
    overwrite_existing: bool = False,
    verbose: bool = True
) -> ad.AnnData:
    """
    Merge sample metadata from CSV file into pseudobulked AnnData object.
    Handles cases where metadata is only available for a subset of samples.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        The pseudobulked AnnData object where adata.obs.index contains sample names
    metadata_path : str
        Path to the CSV file containing sample metadata (e.g., ATAC_Metadata.csv)
        First column should contain sample names (used as index)
        Note: CSV may only contain metadata for a subset of samples
    overwrite_existing : bool, optional
        Whether to overwrite existing columns in adata.obs (default: False)
    verbose : bool, optional
        Whether to print information about the merge process
        
    Returns:
    --------
    anndata.AnnData
        AnnData object with merged sample metadata (NaN for samples without metadata)
    """
    
    # Read sample metadata CSV (first column as index = sample names)
    try:
        sample_metadata = pd.read_csv(metadata_path, index_col=0)
        if verbose:
            print(f"Loaded sample metadata with shape: {sample_metadata.shape}")
            print(f"Sample metadata columns: {list(sample_metadata.columns)}")
            print(f"Samples in metadata: {list(sample_metadata.index)}")
    except Exception as e:
        raise FileNotFoundError(f"Could not read metadata file: {e}")
    
    # Get sample names from adata and metadata
    adata_samples = set(adata.obs.index)
    metadata_samples = set(sample_metadata.index)
    
    # Check overlap
    common_samples = adata_samples.intersection(metadata_samples)
    missing_in_metadata = adata_samples - metadata_samples
    missing_in_adata = metadata_samples - adata_samples
    
    if verbose:
        print(f"\nSample overlap analysis:")
        print(f"Samples in AnnData: {len(adata_samples)} - {sorted(adata_samples)}")
        print(f"Samples in metadata: {len(metadata_samples)} - {sorted(metadata_samples)}")
        print(f"Common samples: {len(common_samples)} - {sorted(common_samples)}")
        if missing_in_metadata:
            print(f"Samples in AnnData but not in metadata: {sorted(missing_in_metadata)}")
        if missing_in_adata:
            print(f"Samples in metadata but not in AnnData: {sorted(missing_in_adata)}")
    
    if len(common_samples) == 0:
        print("ERROR: No common samples found!")
        return adata
    
    # Identify which columns to add/update
    new_cols = []
    existing_cols_to_update = []
    skipped_cols = []
    
    for col in sample_metadata.columns:
        if col in adata.obs.columns:
            if overwrite_existing:
                existing_cols_to_update.append(col)
            else:
                skipped_cols.append(col)
        else:
            new_cols.append(col)
    
    # Add/update metadata columns
    for col in new_cols + existing_cols_to_update:
        # Create new column with NaN for all samples first
        adata.obs[col] = np.nan
        
        # Only fill values for samples that exist in metadata
        common_sample_list = list(common_samples)
        adata.obs.loc[common_sample_list, col] = sample_metadata.loc[common_sample_list, col]
    
    if verbose:
        print(f"\nSuccessfully merged sample metadata:")
        if new_cols:
            print(f"Added {len(new_cols)} new columns: {new_cols}")
        if existing_cols_to_update:
            print(f"Updated {len(existing_cols_to_update)} existing columns: {existing_cols_to_update}")
        if skipped_cols:
            print(f"Skipped {len(skipped_cols)} existing columns: {skipped_cols}")
            print("  (Use overwrite_existing=True to update these)")
        
        # Show final sample info with metadata status
        print(f"\nFinal pseudobulked data:")
        print(f"Total samples: {adata.n_obs}")
        print(f"Samples with metadata: {len(common_samples)} ({len(common_samples)/adata.n_obs:.1%})")
        if missing_in_metadata:
            print(f"Samples without metadata (filled with NaN): {len(missing_in_metadata)}")
            if len(missing_in_metadata) <= 10:  # Show names if not too many
                print(f"  {sorted(missing_in_metadata)}")
            else:
                print(f"  First few: {sorted(list(missing_in_metadata))[:5]}...")
        print(f"Total genes/features: {adata.n_vars}")
    
    return adata

def merge_pseudobulk_metadata(
    adata: ad.AnnData,
    metadata_path: str,
    overwrite_existing: bool = False,
    verbose: bool = True
) -> ad.AnnData:
    """
    Merge sample metadata from CSV file into pseudobulked AnnData object.
    In pseudobulked data, each observation (row) represents a sample.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        The pseudobulked AnnData object where adata.obs.index contains sample names
    metadata_path : str
        Path to the CSV file containing sample metadata (e.g., ATAC_Metadata.csv)
        First column should contain sample names (used as index)
    overwrite_existing : bool, optional
        Whether to overwrite existing columns in adata.obs (default: False)
    verbose : bool, optional
        Whether to print information about the merge process
        
    Returns:
    --------
    anndata.AnnData
        AnnData object with merged sample metadata
    """
    
    # Read sample metadata CSV (first column as index = sample names)
    try:
        sample_metadata = pd.read_csv(metadata_path, index_col=0)
        if verbose:
            print(f"Loaded sample metadata with shape: {sample_metadata.shape}")
            print(f"Sample metadata columns: {list(sample_metadata.columns)}")
            print(f"Samples in metadata: {list(sample_metadata.index)}")
    except Exception as e:
        raise FileNotFoundError(f"Could not read metadata file: {e}")
    
    # Get sample names from adata and metadata
    adata_samples = set(adata.obs.index)
    metadata_samples = set(sample_metadata.index)
    
    # Check overlap
    common_samples = adata_samples.intersection(metadata_samples)
    missing_in_metadata = adata_samples - metadata_samples
    missing_in_adata = metadata_samples - adata_samples
    
    if verbose:
        print(f"\nSample overlap analysis:")
        print(f"Samples in AnnData: {len(adata_samples)} - {sorted(adata_samples)}")
        print(f"Samples in metadata: {len(metadata_samples)} - {sorted(metadata_samples)}")
        print(f"Common samples: {len(common_samples)} - {sorted(common_samples)}")
        if missing_in_metadata:
            print(f"Samples in AnnData but not in metadata: {sorted(missing_in_metadata)}")
        if missing_in_adata:
            print(f"Samples in metadata but not in AnnData: {sorted(missing_in_adata)}")
    
    if len(common_samples) == 0:
        print("ERROR: No common samples found!")
        return adata
    
    # Identify which columns to add/update
    new_cols = []
    existing_cols_to_update = []
    skipped_cols = []
    
    for col in sample_metadata.columns:
        if col in adata.obs.columns:
            if overwrite_existing:
                existing_cols_to_update.append(col)
            else:
                skipped_cols.append(col)
        else:
            new_cols.append(col)
    
    if verbose:
        print(f"\nColumn analysis:")
        print(f"New columns to add: {new_cols}")
        if existing_cols_to_update:
            print(f"Existing columns to update: {existing_cols_to_update}")
        if skipped_cols:
            print(f"Existing columns skipped: {skipped_cols}")
    
    # Add/update metadata columns - FIX: Use reindex instead of loc
    for col in new_cols + existing_cols_to_update:
        # Use reindex to handle missing samples gracefully (fills with NaN)
        adata.obs[col] = sample_metadata[col].reindex(adata.obs.index)
    
    if verbose:
        # Show final sample info
        print(f"\nFinal pseudobulked data:")
        print(f"Total samples: {adata.n_obs}")
        print(f"Total genes/features: {adata.n_vars}")
        if missing_in_metadata:
            print(f"Samples without metadata (will have NaN values): {len(missing_in_metadata)}")
            print(f"These samples: {sorted(missing_in_metadata)}")
    
    return adata

def inspect_pseudobulk_metadata_overlap(
    adata: ad.AnnData,
    metadata_path: str
) -> dict:
    """
    Inspect the overlap between pseudobulked AnnData samples and sample metadata.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        The pseudobulked AnnData object
    metadata_path : str
        Path to the CSV file containing sample metadata
        
    Returns:
    --------
    dict
        Dictionary with overlap information
    """
    
    # Read sample metadata
    sample_metadata = pd.read_csv(metadata_path, index_col=0)
    
    # Compare samples
    adata_samples = set(adata.obs.index)
    metadata_samples = set(sample_metadata.index)
    common_samples = adata_samples.intersection(metadata_samples)
    
    # Compare column names
    adata_cols = set(adata.obs.columns)
    metadata_cols = set(sample_metadata.columns)
    common_cols = adata_cols.intersection(metadata_cols)
    new_cols = metadata_cols - adata_cols
    
    overlap_info = {
        'adata_samples': sorted(adata_samples),
        'metadata_samples': sorted(metadata_samples),
        'common_samples': sorted(common_samples),
        'missing_in_metadata': sorted(adata_samples - metadata_samples),
        'missing_in_adata': sorted(metadata_samples - adata_samples),
        'sample_overlap_ratio': len(common_samples) / len(adata_samples) if adata_samples else 0,
        'adata_columns': list(adata_cols),
        'metadata_columns': list(metadata_cols),
        'common_columns': list(common_cols),
        'new_columns': list(new_cols),
        'sample_metadata_preview': sample_metadata.head(),
        'adata_shape': (adata.n_obs, adata.n_vars)
    }
    
    return overlap_info


def validate_pseudobulk_structure(adata: ad.AnnData, verbose: bool = True) -> bool:
    """
    Validate that the AnnData object has the expected pseudobulk structure.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        The AnnData object to validate
    verbose : bool, optional
        Whether to print validation results
        
    Returns:
    --------
    bool
        True if structure looks like pseudobulk data
    """
    
    is_valid = True
    issues = []
    
    # Check if obs index contains sample-like names
    sample_names = adata.obs.index.tolist()
    
    # Basic checks
    if len(sample_names) < 2:
        issues.append("Very few observations - expected multiple samples")
        is_valid = False
    
    if len(set(sample_names)) != len(sample_names):
        issues.append("Duplicate sample names found in obs.index")
        is_valid = False
    
    # Check for typical single-cell patterns (which shouldn't be in pseudobulk)
    cell_like_patterns = sum(1 for name in sample_names[:10] if any(sep in str(name) for sep in ['_', '-']) and len(str(name)) > 10)
    if cell_like_patterns > len(sample_names) * 0.5:
        issues.append("Sample names look like cell barcodes - might not be pseudobulked")
        is_valid = False
    
    if verbose:
        print(f"Pseudobulk validation:")
        print(f"Number of samples: {len(sample_names)}")
        print(f"Number of features: {adata.n_vars}")
        print(f"Sample names preview: {sample_names[:5]}")
        
        if issues:
            print(f"Potential issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ Structure looks appropriate for pseudobulked data")
    
    return is_valid


import pandas as pd

def update_sev_level_values(
    metadata_path: str,
    sample_updates: dict,
    output_path: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Update sev.level values for specific samples in the metadata CSV.
    
    Parameters:
    -----------
    metadata_path : str
        Path to the CSV file containing sample metadata
    sample_updates : dict
        Dictionary mapping sample names to new sev.level values
        Example: {'sample1': 'severe', 'sample2': 'mild'}
    output_path : str, optional
        Path to save the updated CSV. If None, overwrites the original file
    verbose : bool, optional
        Whether to print update information
        
    Returns:
    --------
    pd.DataFrame
        Updated metadata DataFrame
    """
    
    # Read the metadata CSV
    metadata = pd.read_csv(metadata_path, index_col=0)
    
    if verbose:
        print(f"Original metadata shape: {metadata.shape}")
        if 'sev.level' in metadata.columns:
            print(f"Original sev.level values:")
            print(metadata['sev.level'].value_counts())
        else:
            print("WARNING: 'sev.level' column not found in metadata!")
            return metadata
    
    # Track changes
    updated_samples = []
    missing_samples = []
    
    # Update the values
    for sample_name, new_value in sample_updates.items():
        if sample_name in metadata.index:
            old_value = metadata.loc[sample_name, 'sev.level']
            metadata.loc[sample_name, 'sev.level'] = new_value
            updated_samples.append((sample_name, old_value, new_value))
            if verbose:
                print(f"Updated {sample_name}: '{old_value}' → '{new_value}'")
        else:
            missing_samples.append(sample_name)
            if verbose:
                print(f"WARNING: Sample '{sample_name}' not found in metadata")
    
    if verbose:
        print(f"\nSummary:")
        print(f"Successfully updated: {len(updated_samples)} samples")
        if missing_samples:
            print(f"Missing samples: {missing_samples}")
        
        print(f"\nUpdated sev.level values:")
        print(metadata['sev.level'].value_counts())
    
    # Save the updated metadata
    output_file = output_path if output_path else metadata_path
    metadata.to_csv(output_file)
    
    if verbose:
        print(f"\nSaved updated metadata to: {output_file}")
    
    return metadata


def update_sev_level_from_mapping_file(
    adata: ad.AnnData,
    mapping_file_path: str,
    sample_col: str = 'sample',
    sev_level_col: str = 'sev.level',
    verbose: bool = True
) -> ad.AnnData:
    """
    Update sev.level values in AnnData using a separate mapping file.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        The AnnData object to update
    mapping_file_path : str
        Path to CSV file with sample names and new sev.level values
    sample_col : str
        Column name in mapping file containing sample names
    sev_level_col : str
        Column name in mapping file containing new sev.level values
    verbose : bool, optional
        Whether to print update information
        
    Returns:
    --------
    anndata.AnnData
        Updated AnnData object
    """
    
    # Read mapping file
    mapping = pd.read_csv(mapping_file_path)
    
    if verbose:
        print(f"AnnData shape: {adata.shape}")
        print(f"Mapping file shape: {mapping.shape}")
    
    # Check if required columns exist in mapping file
    if sample_col not in mapping.columns or sev_level_col not in mapping.columns:
        print(f"ERROR: Required columns not found in mapping file!")
        print(f"Available columns: {list(mapping.columns)}")
        return adata
    
    # Create update dictionary from mapping file
    sample_updates = dict(zip(mapping[sample_col], mapping[sev_level_col]))
    
    if verbose:
        print(f"Found {len(sample_updates)} sample updates in mapping file")
        if 'sev.level' in adata.obs.columns:
            print(f"Original sev.level values in AnnData:")
            print(adata.obs['sev.level'].value_counts())
        else:
            print("'sev.level' column not found in AnnData - will be created")
    
    # Track changes
    updated_samples = []
    missing_samples = []
    
    # Update the values in AnnData
    for sample_name, new_value in sample_updates.items():
        if sample_name in adata.obs.index:
            if 'sev.level' in adata.obs.columns:
                old_value = adata.obs.loc[sample_name, 'sev.level']
            else:
                old_value = 'N/A'
                # Initialize the column if it doesn't exist
                if 'sev.level' not in adata.obs.columns:
                    adata.obs['sev.level'] = pd.NA
            
            adata.obs.loc[sample_name, 'sev.level'] = new_value
            updated_samples.append((sample_name, old_value, new_value))
            if verbose:
                print(f"Updated {sample_name}: '{old_value}' → '{new_value}'")
        else:
            missing_samples.append(sample_name)
            if verbose:
                print(f"WARNING: Sample '{sample_name}' not found in AnnData")
    
    if verbose:
        print(f"\nSummary:")
        print(f"Successfully updated: {len(updated_samples)} samples")
        if missing_samples:
            print(f"Missing samples: {missing_samples}")
        
        print(f"\nUpdated sev.level values in AnnData:")
        print(adata.obs['sev.level'].value_counts())
    
    return adata

import scanpy as sc

def print_srr14466476_obs_column_values(adata_path):
    # Load the AnnData object
    adata = sc.read(adata_path)
    
    # Check if the object 'SRR14466476' exists in obs
    if 'SRR14466476' in adata.obs.index:
        # Retrieve the row corresponding to 'SRR14466476'
        srr14466476_data = adata.obs.loc['SRR14466476']
        
        # Print all columns and their values for 'SRR14466476'
        print(f"Obs columns and values for SRR14466476:")
        for column, value in srr14466476_data.items():
            print(f"{column}: {value}")
    else:
        print("Object 'SRR14466476' not found in obs.")


if __name__ == "__main__":
    adata = sc.read("/dcl01/hongkai/data/data/hjiang/result/ATAC/harmony/ATAC_sample.h5ad")
    
    # updated_adata = update_sev_level_from_mapping_file(
    #     adata=adata,
    #     mapping_file_path="/dcl01/hongkai/data/data/hjiang/Data/ATAC_Metadata.csv"
    # )
    
    # # Save the updated AnnData
    # sc.write("/dcl01/hongkai/data/data/hjiang/result/integration/pseudobulk/pseudobulk_sample.h5ad", updated_adata)
    
    sample_counts = adata.obs['sample'].value_counts()

    # Print the results
    print("Number of cells per sample:")
    print(sample_counts)

    # Or if you want it sorted by sample name instead of count:
    sample_counts_sorted = adata.obs['sample'].value_counts().sort_index()
    print("\nNumber of cells per sample (sorted by sample name):")
    print(sample_counts_sorted)