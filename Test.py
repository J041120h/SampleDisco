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