import os
import scanpy as sc
import pandas as pd

def treecor_harmony_with_check(
    h5ad_path,
    sample_meta_path,
    output_dir,
    cell_meta_path=None,
    markers=None,
    cluster_resolution=1,
    num_PCs=20,
    num_harmony=3,
    num_features=1000,
    min_cells=0,
    min_features=0,
    pct_mito_cutoff=20,
    exclude_genes=None,
    method='average',
    metric='euclidean',
    distance_mode='centroid',
    vars_to_regress=None,
    verbose=True
):
    """
    Reads and processes a raw-count H5AD file and appends sample metadata. 
    Includes basic filtering, mitochondrial gene removal, and final checks on 
    the number of cells, genes, and unique samples.

    Parameters
    ----------
    h5ad_path : str
        Path to the H5AD file (raw count data).
    sample_meta_path : str
        Path to the CSV file containing sample metadata.
    output_dir : str
        Path to the directory for saving outputs (will create a "harmony" 
        subdirectory).
    cell_meta_path : str, optional
        CSV file containing cell-level metadata with 'barcode' as index or column. 
        Default is None, in which case 'sample' is inferred from cell barcodes.
    markers : list, optional
        Not currently used in this pipeline but can be provided for future expansions.
    cluster_resolution : float, optional
        Not currently used in this pipeline but can be provided for future expansions.
    num_PCs : int, optional
        Number of principal components (placeholder).
    num_harmony : int, optional
        Number of Harmony integration steps (placeholder).
    num_features : int, optional
        Number of highly variable features (placeholder).
    min_cells : int, optional
        Minimum number of cells in which a gene must be expressed to be retained.
    min_features : int, optional
        Minimum number of genes that a cell must express to be retained.
    pct_mito_cutoff : float, optional
        Mitochondrial reads percentage cutoff for filtering cells.
    exclude_genes : list, optional
        List of genes to exclude from the dataset (e.g., remove cell cycle genes).
    method : str, optional
        Placeholder for distance/aggregation method.
    metric : str, optional
        Placeholder for distance metric.
    distance_mode : str, optional
        Placeholder for distance calculation mode.
    vars_to_regress : list, optional
        Genes or metadata to regress out during normalization (placeholder).
    verbose : bool, optional
        Whether to print status messages during processing.

    Returns
    -------
    adata : AnnData
        The processed AnnData object.
    """

    if vars_to_regress is None:
        vars_to_regress = []

    # 0. Create output directories if not present
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating output directory")

    # Append 'harmony' subdirectory
    harmony_dir = os.path.join(output_dir, 'harmony')
    if not os.path.exists(harmony_dir):
        os.makedirs(harmony_dir)
        if verbose:
            print("Automatically generating harmony subdirectory")

    # 1. Read the raw count data from an existing H5AD
    if verbose:
        print('=== Reading input dataset ===')
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f'Dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')


    if verbose:
        print(f'Dimension of processed data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # 4. Attach cell metadata if provided
    if cell_meta_path is None:
        # Infer 'sample' by splitting on a delimiter in obs_names, e.g. "SAMPLE:BARCODE"
        adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path)
        # Ensure we use 'barcode' as the index if it's a column
        if 'barcode' in cell_meta.columns:
            cell_meta.set_index('barcode', inplace=True)
        adata.obs = adata.obs.join(cell_meta, how='left')

    # 5. Attach sample metadata
    sample_meta = pd.read_csv(sample_meta_path)
    adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')

    # === Final check & printout ===
    n_cells, n_genes = adata.shape
    if verbose:
        print("\n=== Final AnnData dimensions and sample counts ===")
        print(f"Number of cells: {n_cells}")
        print(f"Number of genes: {n_genes}")
        if "sample" in adata.obs:
            n_samples = adata.obs["sample"].nunique()
            print(f"Number of unique samples: {n_samples}")
            print("Unique sample IDs:")
            print(adata.obs["sample"].unique())
        else:
            print("Warning: 'sample' column not found in adata.obs.")

    # (Optional) Save the processed AnnData to disk
    # adata.write_h5ad(os.path.join(harmony_dir, "processed_adata.h5ad"))

    return adata


if __name__ == '__main__':
    adata_processed = treecor_harmony_with_check(
    h5ad_path= "/users/hjiang/GenoDistance/Data/count_data.h5ad",
    sample_meta_path= "/users/hjiang/GenoDistance/Data/sample_data.csv",
    output_dir="/users/hjiang/r",
    cell_meta_path="/users/hjiang/GenoDistance/Data/cell_data.csv",
    min_cells=3,
    min_features=200,
    pct_mito_cutoff=20  # Example excluded genes
)
    
import os
import numpy as np
import anndata
import pandas as pd
from datetime import datetime

def sample_anndata_by_sample(h5ad_path, n, cell_meta_path, sample_meta_path, output_dir=None):
    """
    Reads an AnnData object from the specified .h5ad file, randomly selects n unique samples based on 
    the 'sample' column in adata.obs, subsets all cells corresponding to these samples, saves the subset 
    AnnData locally, and returns the path to the saved file.
    
    Parameters:
        h5ad_path (str): Path to the input .h5ad file.
        n (int): Number of unique samples to randomly select.
        cell_meta_path (str or None): Path to the cell metadata CSV file. 
                                      If None, sample IDs will be parsed from cell names.
        sample_meta_path (str): Path to the sample metadata CSV file.
        output_dir (str, optional): Directory to save the subset file. If None, uses the directory of h5ad_path.
    
    Returns:
        str: Path to the saved subset AnnData file.
    """
    # Step 1: Load the AnnData object from file
    print("Step 1: Loading AnnData object from file...")
    adata = anndata.read_h5ad(h5ad_path)
    print(f"Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} features.")
    
    # Step 2: Process cell metadata
    if cell_meta_path is None:
        print("Step 2: No cell metadata provided. Parsing sample IDs from cell names...")
        # If no cell metadata, parse sample IDs from cell names (e.g. "SAMPLE1:ATCGG")
        adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
        print("Parsed 'sample' column from cell names.")
    else:
        print("Step 2: Loading cell metadata from file...")
        cell_meta = pd.read_csv(cell_meta_path)
        print(f"Loaded cell metadata with {cell_meta.shape[0]} rows and {cell_meta.shape[1]} columns.")
        # Assuming 'barcode' is the column with cell IDs
        cell_meta.set_index('barcode', inplace=True)
        print("Merging cell metadata with AnnData.obs...")
        adata.obs = adata.obs.join(cell_meta, how='left')
        print("Merged cell metadata.")
    
    # Step 3: Attach sample metadata
    print("Step 3: Loading sample metadata from file...")
    sample_meta = pd.read_csv(sample_meta_path)
    print(f"Loaded sample metadata with {sample_meta.shape[0]} rows and {sample_meta.shape[1]} columns.")
    print("Merging sample metadata with AnnData.obs...")
    adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')
    print("Merged sample metadata.")
    
    # Check if 'sample' column exists
    if 'sample' not in adata.obs.columns:
        raise ValueError("The AnnData object does not contain a 'sample' column in .obs.")
    
    # Step 4: Extract unique sample labels
    print("Step 4: Extracting unique sample labels...")
    unique_samples = adata.obs['sample'].unique()
    print(f"Found {len(unique_samples)} unique samples.")
    
    # Ensure that n does not exceed the total number of unique samples
    if n > len(unique_samples):
        raise ValueError(f"Requested sample size n ({n}) exceeds the number of unique samples ({len(unique_samples)}).")
    
    # Step 5: Randomly select n unique sample labels without replacement
    print("Step 5: Randomly selecting samples...")
    selected_samples = np.random.choice(unique_samples, size=n, replace=False)
    print(f"Selected samples: {selected_samples}")
    
    # Step 6: Subset all cells corresponding to the selected sample labels
    print("Step 6: Subsetting cells corresponding to selected samples...")
    mask = adata.obs['sample'].isin(selected_samples)
    subset_adata = adata[mask, :].copy()
    print(f"Subset AnnData contains {subset_adata.n_obs} cells.")
    
    # Step 7: Determine and prepare the output directory
    print("Step 7: Preparing output directory...")
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(h5ad_path))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory is set to: {output_dir}")
    
    # Step 8: Create a unique filename and save the subset AnnData object
    print("Step 8: Creating unique filename and saving subset AnnData...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(os.path.basename(h5ad_path))[0]
    output_filename = f"{base_filename}_sample_{n}_{timestamp}.h5ad"
    output_path = os.path.join(output_dir, output_filename)
    subset_adata.write_h5ad(output_path)
    print(f"Saved subset AnnData file to: {output_path}")
    
    print("Process completed successfully.")
    return output_path
