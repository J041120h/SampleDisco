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


import os
import scanpy as sc
import pandas as pd
import numpy as np

def treecor_seurat_mapping(
    h5ad_path,
    sample_meta_path,
    output_dir,
    cell_meta_path=None,
    after_process_h5ad_path=None,
    num_hvg=2000,
    min_cells=500,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
    n_pcs=20,
    vars_to_regress=[],
    verbose=True
):
    """
    A Seurat-like HVG-based workflow in Scanpy. 
    - Performs QC, filtering, HVG detection (Seurat v3 flavor), PCA, neighbors, and UMAP. 
    - Plots UMAP colored by 'batch' in output_dir/comparison.
    - Also (optionally) reads another H5AD file (after_process_h5ad_path) and 
      plots its UMAP colored by 'batch' to the same directory.
    - Does NOT save the final AnnData object to disk.
    """

    # Create main output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"Main output directory: {output_dir}")

    # Create comparison subdirectory for results
    comparison_dir = os.path.join(output_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    if verbose:
        print(f"All results will be saved in: {comparison_dir}")

    # Point Scanpy's default figure directory to the comparison folder
    sc.settings.figdir = comparison_dir

    # 1. Read the raw count data
    if verbose:
        print('=== Reading input dataset ===')
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f'Initial dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # 2. Merge cell metadata if provided; otherwise infer 'sample' from cell names
    if cell_meta_path is None:
        adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path)
        if 'barcode' in cell_meta.columns:
            cell_meta.set_index('barcode', inplace=True)
        adata.obs = adata.obs.join(cell_meta, how='left')

    # 3. Merge sample-level metadata
    sample_meta = pd.read_csv(sample_meta_path)
    # Must contain columns 'sample' and 'batch' for final usage
    if 'sample' not in sample_meta.columns:
        raise ValueError("sample_meta must contain a 'sample' column.")
    if 'batch' not in sample_meta.columns:
        raise ValueError("sample_meta must contain a 'batch' column to use as category.")

    adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')

    # 4. Basic filtering of cells and genes
    sc.pp.filter_genes(adata, min_cells=min_cells)     # keep genes in >= min_cells
    sc.pp.filter_cells(adata, min_genes=min_features)  # keep cells with >= min_features

    # 4a. Mito filtering
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()

    # 4b. Exclude specific genes (if provided) and remove MT- genes
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()

    # 4c. Keep only samples with at least min_cells
    cell_counts_per_patient = adata.obs.groupby('sample').size()
    patients_to_keep = cell_counts_per_patient[cell_counts_per_patient >= min_cells].index
    adata = adata[adata.obs['sample'].isin(patients_to_keep)].copy()

    # 4d. Filter out genes not present in at least 1% of cells
    min_cells_for_gene = int(0.01 * adata.n_obs)
    sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)
    
    if verbose:
        print(f'Dimension of processed data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # 5. Optional doublet detection
    if doublet:
        if verbose:
            print("Performing scrublet-based doublet detection...")
        sc.pp.scrublet(adata, batch_key="sample")

    # 6. Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 7. Identify highly variable genes (HVG) using Seurat v3
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=num_hvg)
    if verbose:
        print(f"Selected {np.sum(adata.var['highly_variable'])} HVGs using Seurat v3 flavor.")

    # 8. (Optional) Regress out unwanted variables
    if len(vars_to_regress) > 0:
        if verbose:
            print(f"Regressing out variables: {vars_to_regress}")
        sc.pp.scale(adata, max_value=10)
        sc.pp.regress_out(adata, keys=vars_to_regress)
        sc.pp.scale(adata, max_value=10)
    else:
        sc.pp.scale(adata, max_value=10)

    # 9. PCA on the HVGs
    sc.tl.pca(adata, n_comps=n_pcs, use_highly_variable=True)
    if verbose:
        print(f"Computed PCA with {n_pcs} components on HVGs.")

    # 10. Build neighbors graph and compute UMAP
    sc.pp.neighbors(adata, n_pcs=n_pcs)
    sc.tl.umap(adata)
    if verbose:
        print("Computed UMAP.")

    # 11. Plot UMAP colored by batch and save
    if verbose:
        print("Plotting UMAP colored by 'batch'...")
    sc.pl.umap(adata, color=['batch'], save='_batch_umap.png', show=False)
    if verbose:
        umap_path = os.path.join(sc.settings.figdir, 'umap_batch_umap.png')
        print(f"UMAP plot for main data saved to: {umap_path}")

    # 12. (Optional) Plot UMAP for the 'after_process' dataset if provided
    if after_process_h5ad_path is not None:
        if verbose:
            print(f"Reading additional dataset from {after_process_h5ad_path} ...")
        adata_after = sc.read_h5ad(after_process_h5ad_path)
        # We assume 'batch' is already present in adata_after.obs
        if 'batch' not in adata_after.obs:
            print("Warning: 'batch' not found in after_process dataset. No coloring by batch possible.")
        else:
            if verbose:
                print("Plotting UMAP colored by 'batch' for after_process data...")
            sc.pl.umap(adata_after, color=['batch'], save='_batch_umap_after_process.png', show=False)
            if verbose:
                after_umap_path = os.path.join(sc.settings.figdir, 'umap_batch_umap_after_process.png')
                print(f"UMAP plot for after_process data saved to: {after_umap_path}")

    # 13. Do not save the final AnnData object; simply return it
    if verbose:
        print("Analysis complete. The AnnData object is returned in memory only.")

    return adata

import scanpy as sc
import pandas as pd

def count_samples_in_adata(h5ad_path, sample_meta_path, cell_meta_path=None, verbose=True):
    """
    Reads an AnnData object, combines sample and cell metadata, and counts unique sample entries.

    Parameters:
    h5ad_path (str): Path to the AnnData (.h5ad) file.
    sample_meta_path (str): Path to the sample metadata CSV file.
    cell_meta_path (str, optional): Path to the cell metadata CSV file. Defaults to None.
    verbose (bool, optional): Whether to print status messages. Defaults to True.

    Returns:
    int: The number of unique samples in the dataset.
    """
    if verbose:
        print('=== Reading input dataset ===')
    adata = sc.read_h5ad(h5ad_path)
    
    if verbose:
        print(f'Dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')
    
    # Assign sample IDs to cells
    if cell_meta_path is None:
        adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
    else:
        if verbose:
            print('=== Reading cell metadata ===')
        cell_meta = pd.read_csv(cell_meta_path)
        cell_meta.set_index('barcode', inplace=True)
        adata.obs = adata.obs.join(cell_meta, how='left')
    
    if verbose:
        print('=== Reading sample metadata ===')
    sample_meta = pd.read_csv(sample_meta_path)
    
    # Merge sample metadata with observation data
    adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')
    
    # Count unique samples
    unique_samples = adata.obs['sample'].nunique()
    
    if verbose:
        print(f'Number of unique samples in dataset: {unique_samples}')
    
    return unique_samples

import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from harmony import harmonize

# Local imports from your project
from Visualization import visualization_harmony  # if you use it later
from pseudobulk import compute_pseudobulk_dataframes
from HierarchicalConstruction import cell_type_dendrogram
from HVG import find_hvgs
from Grouping import find_sample_grouping

def test_harmony(
    h5ad_path,
    anndata_cluster_path,
    sample_meta_path,
    output_dir,
    cell_meta_path=None,
    markers=None,
    cluster_resolution=0.8,
    num_PCs=20,
    num_harmony=30,
    num_features=2000,
    min_cells=500,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
    combat=True,  # Note: not fully implemented in example
    method='average',
    metric='euclidean',
    distance_mode='centroid',
    vars_to_regress=[],
    verbose=True
):
    """
    Harmony Integration with proportional HVG selection by cell type,
    now reading an existing H5AD file that only contains raw counts (no meta).

    This function:
      1. Reads and preprocesses the data (filter genes/cells, remove MT genes, etc.).
      2. Splits into two branches for:
         (a) adata_cluster used for clustering with Harmony
         (b) adata_sample_diff used for sample-level analysis (minimal batch correction).
      3. Returns both AnnData objects.
    """

    # 0. Create output directories if not present
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating output directory")

    # Append 'harmony' subdirectory
    output_dir = os.path.join(output_dir, 'harmony')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating harmony subdirectory")

    # 1. Read the raw count data from an existing H5AD
    if verbose:
        print('=== Read input dataset ===')
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f'Dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # Attach sample info
    if cell_meta_path is None:
        # If no cell metadata provided, assume the obs_names "barcode" has "sample" in front
        adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path)
        cell_meta.set_index('barcode', inplace=True)
        adata.obs = adata.obs.join(cell_meta, how='left')

    sample_meta = pd.read_csv(sample_meta_path)
    adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')

    # Basic filtering
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_features)

    # Mito QC
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()

    # Exclude genes if needed
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()

    # Only keep samples with enough cells
    cell_counts_per_patient = adata.obs.groupby('sample').size()
    patients_to_keep = cell_counts_per_patient[cell_counts_per_patient >= min_cells].index
    adata = adata[adata.obs['sample'].isin(patients_to_keep)].copy()

    # Drop genes that are too rare in these final cells
    min_cells_for_gene = int(0.01 * adata.n_obs)  # e.g., gene must appear in 1% of cells
    sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)

    if verbose:
        print(f'Dimension of processed data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # Optional doublet detection
    if doublet:
        sc.pp.scrublet(adata, batch_key="sample")

    # Normalization & log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if verbose:
        print("Preprocessing complete!")

    # Split data for two separate analyses
    adata_cluster = sc.read_h5ad(anndata_cluster_path)
    adata_sample_diff = adata.copy()  # used for sample-level analysis

    # 2(b). Sample-level analysis
    # Carry over the cell_type from the cluster object if needed
    if 'cell_type' not in adata_cluster.obs.columns or adata_cluster.obs['cell_type'].nunique() == 0:
        adata_cluster.obs['cell_type'] = '1'
    adata_sample_diff.obs['cell_type'] = adata_cluster.obs['cell_type']

    return adata_cluster, adata_sample_diff