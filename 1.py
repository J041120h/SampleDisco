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


if __name__ == "__main__":
    # main("/users/hjiang/GenoDistance/result/harmony/pseudobulk_sample.h5ad")
    # adata = inspect_anndata("/users/hjiang/GenoDistance/result/integration/pseudobulk/pseudobulk_sample.h5ad", verbose=True)
    # print_genes_per_cell("/dcl01/hongkai/data/data/hjiang/Data/ATAC.h5ad")
    combined = subsample_rna_and_merge_with_atac("/dcl01/hongkai/data/data/hjiang/result/integration/glue/atac_rna_integrated.h5ad", "/dcl01/hongkai/data/data/hjiang/result/integration/glue/atac_rna_integrated_test.h5ad")

