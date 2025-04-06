# import os
# import scanpy as sc
# import pandas as pd

# def treecor_harmony_with_check(
#     h5ad_path,
#     sample_meta_path,
#     output_dir,
#     cell_meta_path=None,
#     markers=None,
#     cluster_resolution=1,
#     num_PCs=20,
#     num_harmony=3,
#     num_features=1000,
#     min_cells=0,
#     min_features=0,
#     pct_mito_cutoff=20,
#     exclude_genes=None,
#     method='average',
#     metric='euclidean',
#     distance_mode='centroid',
#     vars_to_regress=None,
#     verbose=True
# ):
#     """
#     Reads and processes a raw-count H5AD file and appends sample metadata. 
#     Includes basic filtering, mitochondrial gene removal, and final checks on 
#     the number of cells, genes, and unique samples.

#     Parameters
#     ----------
#     h5ad_path : str
#         Path to the H5AD file (raw count data).
#     sample_meta_path : str
#         Path to the CSV file containing sample metadata.
#     output_dir : str
#         Path to the directory for saving outputs (will create a "harmony" 
#         subdirectory).
#     cell_meta_path : str, optional
#         CSV file containing cell-level metadata with 'barcode' as index or column. 
#         Default is None, in which case 'sample' is inferred from cell barcodes.
#     markers : list, optional
#         Not currently used in this pipeline but can be provided for future expansions.
#     cluster_resolution : float, optional
#         Not currently used in this pipeline but can be provided for future expansions.
#     num_PCs : int, optional
#         Number of principal components (placeholder).
#     num_harmony : int, optional
#         Number of Harmony integration steps (placeholder).
#     num_features : int, optional
#         Number of highly variable features (placeholder).
#     min_cells : int, optional
#         Minimum number of cells in which a gene must be expressed to be retained.
#     min_features : int, optional
#         Minimum number of genes that a cell must express to be retained.
#     pct_mito_cutoff : float, optional
#         Mitochondrial reads percentage cutoff for filtering cells.
#     exclude_genes : list, optional
#         List of genes to exclude from the dataset (e.g., remove cell cycle genes).
#     method : str, optional
#         Placeholder for distance/aggregation method.
#     metric : str, optional
#         Placeholder for distance metric.
#     distance_mode : str, optional
#         Placeholder for distance calculation mode.
#     vars_to_regress : list, optional
#         Genes or metadata to regress out during normalization (placeholder).
#     verbose : bool, optional
#         Whether to print status messages during processing.

#     Returns
#     -------
#     adata : AnnData
#         The processed AnnData object.
#     """

#     if vars_to_regress is None:
#         vars_to_regress = []

#     # 0. Create output directories if not present
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         if verbose:
#             print("Automatically generating output directory")

#     # Append 'harmony' subdirectory
#     harmony_dir = os.path.join(output_dir, 'harmony')
#     if not os.path.exists(harmony_dir):
#         os.makedirs(harmony_dir)
#         if verbose:
#             print("Automatically generating harmony subdirectory")

#     # 1. Read the raw count data from an existing H5AD
#     if verbose:
#         print('=== Reading input dataset ===')
#     adata = sc.read_h5ad(h5ad_path)
#     if verbose:
#         print(f'Dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')


#     if verbose:
#         print(f'Dimension of processed data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

#     # 4. Attach cell metadata if provided
#     if cell_meta_path is None:
#         # Infer 'sample' by splitting on a delimiter in obs_names, e.g. "SAMPLE:BARCODE"
#         adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
#     else:
#         cell_meta = pd.read_csv(cell_meta_path)
#         # Ensure we use 'barcode' as the index if it's a column
#         if 'barcode' in cell_meta.columns:
#             cell_meta.set_index('barcode', inplace=True)
#         adata.obs = adata.obs.join(cell_meta, how='left')

#     # 5. Attach sample metadata
#     sample_meta = pd.read_csv(sample_meta_path)
#     adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')

#     # === Final check & printout ===
#     n_cells, n_genes = adata.shape
#     if verbose:
#         print("\n=== Final AnnData dimensions and sample counts ===")
#         print(f"Number of cells: {n_cells}")
#         print(f"Number of genes: {n_genes}")
#         if "sample" in adata.obs:
#             n_samples = adata.obs["sample"].nunique()
#             print(f"Number of unique samples: {n_samples}")
#             print("Unique sample IDs:")
#             print(adata.obs["sample"].unique())
#         else:
#             print("Warning: 'sample' column not found in adata.obs.")

#     # (Optional) Save the processed AnnData to disk
#     # adata.write_h5ad(os.path.join(harmony_dir, "processed_adata.h5ad"))

#     return adata
    
# import os
# import numpy as np
# import anndata
# import pandas as pd
# from datetime import datetime

# def sample_anndata_by_sample(h5ad_path, n, cell_meta_path, sample_meta_path, output_dir=None):
#     """
#     Reads an AnnData object from the specified .h5ad file, randomly selects n unique samples based on 
#     the 'sample' column in adata.obs, subsets all cells corresponding to these samples, saves the subset 
#     AnnData locally, and returns the path to the saved file.
    
#     Parameters:
#         h5ad_path (str): Path to the input .h5ad file.
#         n (int): Number of unique samples to randomly select.
#         cell_meta_path (str or None): Path to the cell metadata CSV file. 
#                                       If None, sample IDs will be parsed from cell names.
#         sample_meta_path (str): Path to the sample metadata CSV file.
#         output_dir (str, optional): Directory to save the subset file. If None, uses the directory of h5ad_path.
    
#     Returns:
#         str: Path to the saved subset AnnData file.
#     """
#     # Step 1: Load the AnnData object from file
#     print("Step 1: Loading AnnData object from file...")
#     adata = anndata.read_h5ad(h5ad_path)
#     print(f"Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} features.")
    
#     # Step 2: Process cell metadata
#     if cell_meta_path is None:
#         print("Step 2: No cell metadata provided. Parsing sample IDs from cell names...")
#         # If no cell metadata, parse sample IDs from cell names (e.g. "SAMPLE1:ATCGG")
#         adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
#         print("Parsed 'sample' column from cell names.")
#     else:
#         print("Step 2: Loading cell metadata from file...")
#         cell_meta = pd.read_csv(cell_meta_path)
#         print(f"Loaded cell metadata with {cell_meta.shape[0]} rows and {cell_meta.shape[1]} columns.")
#         # Assuming 'barcode' is the column with cell IDs
#         cell_meta.set_index('barcode', inplace=True)
#         print("Merging cell metadata with AnnData.obs...")
#         adata.obs = adata.obs.join(cell_meta, how='left')
#         print("Merged cell metadata.")
    
#     # Step 3: Attach sample metadata
#     print("Step 3: Loading sample metadata from file...")
#     sample_meta = pd.read_csv(sample_meta_path)
#     print(f"Loaded sample metadata with {sample_meta.shape[0]} rows and {sample_meta.shape[1]} columns.")
#     print("Merging sample metadata with AnnData.obs...")
#     adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')
#     print("Merged sample metadata.")
    
#     # Check if 'sample' column exists
#     if 'sample' not in adata.obs.columns:
#         raise ValueError("The AnnData object does not contain a 'sample' column in .obs.")
    
#     # Step 4: Extract unique sample labels
#     print("Step 4: Extracting unique sample labels...")
#     unique_samples = adata.obs['sample'].unique()
#     print(f"Found {len(unique_samples)} unique samples.")
    
#     # Ensure that n does not exceed the total number of unique samples
#     if n > len(unique_samples):
#         raise ValueError(f"Requested sample size n ({n}) exceeds the number of unique samples ({len(unique_samples)}).")
    
#     # Step 5: Randomly select n unique sample labels without replacement
#     print("Step 5: Randomly selecting samples...")
#     selected_samples = np.random.choice(unique_samples, size=n, replace=False)
#     print(f"Selected samples: {selected_samples}")
    
#     # Step 6: Subset all cells corresponding to the selected sample labels
#     print("Step 6: Subsetting cells corresponding to selected samples...")
#     mask = adata.obs['sample'].isin(selected_samples)
#     subset_adata = adata[mask, :].copy()
#     print(f"Subset AnnData contains {subset_adata.n_obs} cells.")
    
#     # Step 7: Determine and prepare the output directory
#     print("Step 7: Preparing output directory...")
#     if output_dir is None:
#         output_dir = os.path.dirname(os.path.abspath(h5ad_path))
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"Output directory is set to: {output_dir}")
    
#     # Step 8: Create a unique filename and save the subset AnnData object
#     print("Step 8: Creating unique filename and saving subset AnnData...")
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     base_filename = os.path.splitext(os.path.basename(h5ad_path))[0]
#     output_filename = f"{base_filename}_sample_{n}_{timestamp}.h5ad"
#     output_path = os.path.join(output_dir, output_filename)
#     subset_adata.write_h5ad(output_path)
#     print(f"Saved subset AnnData file to: {output_path}")
    
#     print("Process completed successfully.")
#     return output_path


# import os
# import scanpy as sc
# import pandas as pd
# import numpy as np

# def treecor_seurat_mapping(
#     h5ad_path,
#     sample_meta_path,
#     output_dir,
#     cell_meta_path=None,
#     after_process_h5ad_path=None,
#     num_hvg=2000,
#     min_cells=500,
#     min_features=500,
#     pct_mito_cutoff=20,
#     exclude_genes=None,
#     doublet=True,
#     n_pcs=20,
#     vars_to_regress=[],
#     verbose=True
# ):
#     """
#     A Seurat-like HVG-based workflow in Scanpy. 
#     - Performs QC, filtering, HVG detection (Seurat v3 flavor), PCA, neighbors, and UMAP. 
#     - Plots UMAP colored by 'batch' in output_dir/comparison.
#     - Also (optionally) reads another H5AD file (after_process_h5ad_path) and 
#       plots its UMAP colored by 'batch' to the same directory.
#     - Does NOT save the final AnnData object to disk.
#     """

#     # Create main output directory if needed
#     os.makedirs(output_dir, exist_ok=True)
#     if verbose:
#         print(f"Main output directory: {output_dir}")

#     # Create comparison subdirectory for results
#     comparison_dir = os.path.join(output_dir, 'comparison')
#     os.makedirs(comparison_dir, exist_ok=True)
#     if verbose:
#         print(f"All results will be saved in: {comparison_dir}")

#     # Point Scanpy's default figure directory to the comparison folder
#     sc.settings.figdir = comparison_dir

#     # 1. Read the raw count data
#     if verbose:
#         print('=== Reading input dataset ===')
#     adata = sc.read_h5ad(h5ad_path)
#     if verbose:
#         print(f'Initial dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

#     # 2. Merge cell metadata if provided; otherwise infer 'sample' from cell names
#     if cell_meta_path is None:
#         adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
#     else:
#         cell_meta = pd.read_csv(cell_meta_path)
#         if 'barcode' in cell_meta.columns:
#             cell_meta.set_index('barcode', inplace=True)
#         adata.obs = adata.obs.join(cell_meta, how='left')

#     # 3. Merge sample-level metadata
#     sample_meta = pd.read_csv(sample_meta_path)
#     # Must contain columns 'sample' and 'batch' for final usage
#     if 'sample' not in sample_meta.columns:
#         raise ValueError("sample_meta must contain a 'sample' column.")
#     if 'batch' not in sample_meta.columns:
#         raise ValueError("sample_meta must contain a 'batch' column to use as category.")

#     adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')

#     # 4. Basic filtering of cells and genes
#     sc.pp.filter_genes(adata, min_cells=min_cells)     # keep genes in >= min_cells
#     sc.pp.filter_cells(adata, min_genes=min_features)  # keep cells with >= min_features

#     # 4a. Mito filtering
#     adata.var['mt'] = adata.var_names.str.startswith('MT-')
#     sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
#     adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()

#     # 4b. Exclude specific genes (if provided) and remove MT- genes
#     mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
#     if exclude_genes is not None:
#         genes_to_exclude = set(exclude_genes) | set(mt_genes)
#     else:
#         genes_to_exclude = set(mt_genes)
#     adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()

#     # 4c. Keep only samples with at least min_cells
#     cell_counts_per_patient = adata.obs.groupby('sample').size()
#     patients_to_keep = cell_counts_per_patient[cell_counts_per_patient >= min_cells].index
#     adata = adata[adata.obs['sample'].isin(patients_to_keep)].copy()

#     # 4d. Filter out genes not present in at least 1% of cells
#     min_cells_for_gene = int(0.01 * adata.n_obs)
#     sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)
    
#     if verbose:
#         print(f'Dimension of processed data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

#     # 5. Optional doublet detection
#     if doublet:
#         if verbose:
#             print("Performing scrublet-based doublet detection...")
#         sc.pp.scrublet(adata, batch_key="sample")

#     # 6. Normalize and log-transform
#     sc.pp.normalize_total(adata, target_sum=1e4)
#     sc.pp.log1p(adata)

#     # 7. Identify highly variable genes (HVG) using Seurat v3
#     sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=num_hvg)
#     if verbose:
#         print(f"Selected {np.sum(adata.var['highly_variable'])} HVGs using Seurat v3 flavor.")

#     # 8. (Optional) Regress out unwanted variables
#     if len(vars_to_regress) > 0:
#         if verbose:
#             print(f"Regressing out variables: {vars_to_regress}")
#         sc.pp.scale(adata, max_value=10)
#         sc.pp.regress_out(adata, keys=vars_to_regress)
#         sc.pp.scale(adata, max_value=10)
#     else:
#         sc.pp.scale(adata, max_value=10)

#     # 9. PCA on the HVGs
#     sc.tl.pca(adata, n_comps=n_pcs, use_highly_variable=True)
#     if verbose:
#         print(f"Computed PCA with {n_pcs} components on HVGs.")

#     # 10. Build neighbors graph and compute UMAP
#     sc.pp.neighbors(adata, n_pcs=n_pcs)
#     sc.tl.umap(adata)
#     if verbose:
#         print("Computed UMAP.")

#     # 11. Plot UMAP colored by batch and save
#     if verbose:
#         print("Plotting UMAP colored by 'batch'...")
#     sc.pl.umap(adata, color=['batch'], save='_batch_umap.png', show=False)
#     if verbose:
#         umap_path = os.path.join(sc.settings.figdir, 'umap_batch_umap.png')
#         print(f"UMAP plot for main data saved to: {umap_path}")

#     # 12. (Optional) Plot UMAP for the 'after_process' dataset if provided
#     if after_process_h5ad_path is not None:
#         if verbose:
#             print(f"Reading additional dataset from {after_process_h5ad_path} ...")
#         adata_after = sc.read_h5ad(after_process_h5ad_path)
#         # We assume 'batch' is already present in adata_after.obs
#         if 'batch' not in adata_after.obs:
#             print("Warning: 'batch' not found in after_process dataset. No coloring by batch possible.")
#         else:
#             if verbose:
#                 print("Plotting UMAP colored by 'batch' for after_process data...")
#             sc.pl.umap(adata_after, color=['batch'], save='_batch_umap_after_process.png', show=False)
#             if verbose:
#                 after_umap_path = os.path.join(sc.settings.figdir, 'umap_batch_umap_after_process.png')
#                 print(f"UMAP plot for after_process data saved to: {after_umap_path}")

#     # 13. Do not save the final AnnData object; simply return it
#     if verbose:
#         print("Analysis complete. The AnnData object is returned in memory only.")

#     return adata

# import scanpy as sc
# import pandas as pd

# def count_samples_in_adata(h5ad_path, sample_meta_path, cell_meta_path=None, verbose=True):
#     """
#     Reads an AnnData object, combines sample and cell metadata, and counts unique sample entries.

#     Parameters:
#     h5ad_path (str): Path to the AnnData (.h5ad) file.
#     sample_meta_path (str): Path to the sample metadata CSV file.
#     cell_meta_path (str, optional): Path to the cell metadata CSV file. Defaults to None.
#     verbose (bool, optional): Whether to print status messages. Defaults to True.

#     Returns:
#     int: The number of unique samples in the dataset.
#     """
#     if verbose:
#         print('=== Reading input dataset ===')
#     adata = sc.read_h5ad(h5ad_path)
    
#     if verbose:
#         print(f'Dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')
    
#     # Assign sample IDs to cells
#     if cell_meta_path is None:
#         adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
#     else:
#         if verbose:
#             print('=== Reading cell metadata ===')
#         cell_meta = pd.read_csv(cell_meta_path)
#         cell_meta.set_index('barcode', inplace=True)
#         adata.obs = adata.obs.join(cell_meta, how='left')
    
#     if verbose:
#         print('=== Reading sample metadata ===')
#     sample_meta = pd.read_csv(sample_meta_path)
    
#     # Merge sample metadata with observation data
#     adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')
    
#     # Count unique samples
#     unique_samples = adata.obs['sample'].nunique()
    
#     if verbose:
#         print(f'Number of unique samples in dataset: {unique_samples}')
    
#     return unique_samples

# import os
# import numpy as np
# import pandas as pd
# import scanpy as sc
# import harmonypy as hm
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from harmony import harmonize

# # Local imports from your project
# from Visualization import visualization  # if you use it later
# from pseudobulk import compute_pseudobulk_dataframes
# from CellType_Win import cell_type_dendrogram
# from HVG import find_hvgs
# from Grouping import find_sample_grouping

# def test_harmony(
#     h5ad_path,
#     anndata_cluster_path,
#     sample_meta_path,
#     output_dir,
#     cell_meta_path=None,
#     markers=None,
#     cluster_resolution=0.8,
#     num_PCs=20,
#     num_harmony=30,
#     num_features=2000,
#     min_cells=500,
#     min_features=500,
#     pct_mito_cutoff=20,
#     exclude_genes=None,
#     doublet=True,
#     combat=True,  # Note: not fully implemented in example
#     method='average',
#     metric='euclidean',
#     distance_mode='centroid',
#     vars_to_regress=[],
#     verbose=True
# ):
#     """
#     Harmony Integration with proportional HVG selection by cell type,
#     now reading an existing H5AD file that only contains raw counts (no meta).

#     This function:
#       1. Reads and preprocesses the data (filter genes/cells, remove MT genes, etc.).
#       2. Splits into two branches for:
#          (a) adata_cluster used for clustering with Harmony
#          (b) adata_sample_diff used for sample-level analysis (minimal batch correction).
#       3. Returns both AnnData objects.
#     """

#     # 0. Create output directories if not present
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         if verbose:
#             print("Automatically generating output directory")

#     # Append 'harmony' subdirectory
#     output_dir = os.path.join(output_dir, 'harmony')
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         if verbose:
#             print("Automatically generating harmony subdirectory")

#     # 1. Read the raw count data from an existing H5AD
#     if verbose:
#         print('=== Read input dataset ===')
#     adata = sc.read_h5ad(h5ad_path)
#     if verbose:
#         print(f'Dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

#     # Attach sample info
#     if cell_meta_path is None:
#         # If no cell metadata provided, assume the obs_names "barcode" has "sample" in front
#         adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
#     else:
#         cell_meta = pd.read_csv(cell_meta_path)
#         cell_meta.set_index('barcode', inplace=True)
#         adata.obs = adata.obs.join(cell_meta, how='left')

#     sample_meta = pd.read_csv(sample_meta_path)
#     adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')

#     # Basic filtering
#     sc.pp.filter_genes(adata, min_cells=min_cells)
#     sc.pp.filter_cells(adata, min_genes=min_features)

#     # Mito QC
#     adata.var['mt'] = adata.var_names.str.startswith('MT-')
#     sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
#     adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()

#     # Exclude genes if needed
#     mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
#     if exclude_genes is not None:
#         genes_to_exclude = set(exclude_genes) | set(mt_genes)
#     else:
#         genes_to_exclude = set(mt_genes)
#     adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()

#     # Only keep samples with enough cells
#     cell_counts_per_patient = adata.obs.groupby('sample').size()
#     patients_to_keep = cell_counts_per_patient[cell_counts_per_patient >= min_cells].index
#     adata = adata[adata.obs['sample'].isin(patients_to_keep)].copy()

#     # Drop genes that are too rare in these final cells
#     min_cells_for_gene = int(0.01 * adata.n_obs)  # e.g., gene must appear in 1% of cells
#     sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)

#     if verbose:
#         print(f'Dimension of processed data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

#     # Optional doublet detection
#     if doublet:
#         sc.pp.scrublet(adata, batch_key="sample")

#     # Normalization & log transform
#     sc.pp.normalize_total(adata, target_sum=1e4)
#     sc.pp.log1p(adata)

#     if verbose:
#         print("Preprocessing complete!")

#     # Split data for two separate analyses
#     adata_cluster = sc.read_h5ad(anndata_cluster_path)
#     adata_sample_diff = adata.copy()  # used for sample-level analysis

#     # 2(b). Sample-level analysis
#     # Carry over the cell_type from the cluster object if needed
#     if 'cell_type' not in adata_cluster.obs.columns or adata_cluster.obs['cell_type'].nunique() == 0:
#         adata_cluster.obs['cell_type'] = '1'
#     adata_sample_diff.obs['cell_type'] = adata_cluster.obs['cell_type']
#     sc.write(os.path.join(output_dir, 'adata_sample.h5ad'), adata_sample_diff)

#     return adata_cluster, adata_sample_diff

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from anndata import AnnData
# from umap import UMAP

# def plot_pseudobulk_umap(
#     adata: AnnData,
#     output_dir: str,
#     pseudobulk: dict,
#     grouping_columns: list = ['sev.level'],
#     age_bin_size: int = None,
#     verbose: bool = False
# ) -> None:
#     """
#     Constructs a sample-by-feature dataframe from pseudobulk-corrected expression data,
#     selects the top 2000 highly variable genes (HVGs), and performs UMAP to embed samples in 2D.
#     Samples are colored by severity level.

#     Parameters:
#     - adata: AnnData object containing sample metadata in adata.obs.
#     - output_dir: Directory to save the UMAP plot.
#     - pseudobulk: Dictionary containing 'cell_expression_corrected' with expression data.
#     - grouping_columns: Columns to use for grouping. Default is ['sev.level'].
#     - age_bin_size: Integer for age binning if required. Default is None.
#     - verbose: If True, prints additional information.
#     """
#     # Extract corrected cell expression data
#     cell_expr = pseudobulk['cell_expression']
    
#     # Construct sample-by-feature matrix by concatenating expression across cell types
#     sample_df = pd.DataFrame({
#         sample: np.concatenate([cell_expr.loc[ct, sample] for ct in cell_expr.index])
#         for sample in cell_expr.columns
#     }).T
#     sample_df.index.name = 'sample'
#     sample_df.columns = [f"feature_{i}" for i in range(sample_df.shape[1])]
    
#     # Select top 2000 most variable features
#     top_features = sample_df.var(axis=0).nlargest(2000).index
#     sample_df = sample_df[top_features]
    
#     # Get sample grouping information
#     diff_groups = find_sample_grouping(adata, adata.obs['sample'].unique(), grouping_columns, age_bin_size)
#     if isinstance(diff_groups, dict):
#         diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])
#     if 'plot_group' not in diff_groups.columns:
#         raise KeyError("Column 'plot_group' is missing in diff_groups.")
    
#     # Ensure consistent formatting and merge grouping information
#     diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
#     sample_df.index = sample_df.index.astype(str).str.strip().str.lower()
#     sample_df = sample_df.merge(diff_groups.reset_index().rename(columns={'index': 'sample'}), on='sample', how='left')
    
#     # Identify numeric columns (excluding the 'plot_group' column)
#     numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    
#     # Perform UMAP on top HVGs
#     umap_model = UMAP(n_components=2, random_state=42)
#     umap_coords = umap_model.fit_transform(sample_df[numeric_cols])
#     umap_df = pd.DataFrame(umap_coords, index=sample_df.index, columns=['UMAP1', 'UMAP2'])
#     umap_df['plot_group'] = sample_df['plot_group']
    
#     # Extract numeric severity levels for coloring
#     umap_df['sev_level'] = umap_df['plot_group'].str.extract(r'(\d+\.\d+)').astype(float)
#     if umap_df['sev_level'].isna().sum() > 0:
#         raise ValueError("Some plot_group values could not be parsed for severity levels.")
    
#     # Normalize severity levels for color mapping
#     norm_severity = (
#         (umap_df['sev_level'] - umap_df['sev_level'].min()) 
#         / (umap_df['sev_level'].max() - umap_df['sev_level'].min())
#     )
    
#     # Plot UMAP results
#     plt.figure(figsize=(8, 6))
#     sc = plt.scatter(
#         umap_df['UMAP1'], 
#         umap_df['UMAP2'], 
#         c=norm_severity, 
#         cmap='coolwarm', 
#         s=80, 
#         alpha=0.8, 
#         edgecolors='k'
#     )
#     plt.xlabel('UMAP1')
#     plt.ylabel('UMAP2')
#     plt.title('2D UMAP of HVG Expression')
#     plt.colorbar(sc, label='Severity Level')
    
#     # Save plot
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(output_dir, 'sample_relationship_umap_2D_sample.pdf'))
#     plt.close()
    
#     if verbose:
#         print(f"UMAP plot saved to {output_dir}/sample_relationship_umap_2D_sample.pdf")


# def plot_pseudobulk_batch_umap(
#     adata: AnnData,
#     output_dir: str,
#     pseudobulk: dict,
#     grouping_columns: list = ['batch'],
#     age_bin_size: int = None,
#     verbose: bool = False
# ) -> None:
#     """
#     Constructs a sample-by-feature dataframe from pseudobulk-corrected expression data,
#     selects the top 2000 highly variable genes (HVGs), and performs UMAP to embed samples in 2D.
#     Samples are colored by batch.

#     Parameters:
#     - adata: AnnData object containing sample metadata in adata.obs.
#     - output_dir: Directory to save the UMAP plot.
#     - pseudobulk: Dictionary containing 'cell_expression_corrected' with expression data.
#     - grouping_columns: List of columns in adata.obs to use for grouping. Default is ['batch'].
#     - age_bin_size: Integer for age binning if required. Default is None.
#     - verbose: If True, prints additional information.
#     """
#     # Ensure required data exists
#     if 'cell_expression_corrected' not in pseudobulk:
#         raise KeyError("Missing 'cell_expression_corrected' key in pseudobulk dictionary.")

#     # Extract corrected cell expression data
#     cell_expr = pseudobulk['cell_expression_corrected']

#     # Construct sample-by-feature matrix
#     sample_df = pd.DataFrame({
#         sample: np.concatenate([cell_expr.loc[ct, sample] for ct in cell_expr.index])
#         for sample in cell_expr.columns
#     }).T
#     sample_df.index.name = 'sample'
#     sample_df.columns = [f"feature_{i}" for i in range(sample_df.shape[1])]

#     # Select top 2000 most variable features (if applicable)
#     if sample_df.shape[1] > 2000:
#         top_features = sample_df.var(axis=0).nlargest(2000).index
#         sample_df = sample_df[top_features]

#     # Retrieve batch (or grouping) information
#     diff_groups = find_sample_grouping(adata, adata.obs['sample'].unique(), grouping_columns, age_bin_size)

#     # Convert to DataFrame if necessary
#     if isinstance(diff_groups, dict):
#         diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])

#     if 'plot_group' not in diff_groups.columns:
#         raise KeyError("Column 'plot_group' is missing in diff_groups.")

#     # Format index for merging
#     sample_df.index = sample_df.index.astype(str).str.strip().str.lower()
#     diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
#     diff_groups = diff_groups.reset_index().rename(columns={'index': 'sample'})

#     # Merge grouping information
#     sample_df = sample_df.merge(diff_groups, on='sample', how='left')

#     # Identify numeric columns (excluding 'plot_group')
#     numeric_cols = sample_df.select_dtypes(include=[np.number]).columns

#     # Perform UMAP
#     umap_model = UMAP(n_components=2, random_state=42)
#     umap_coords = umap_model.fit_transform(sample_df[numeric_cols])
#     umap_df = pd.DataFrame(umap_coords, index=sample_df.index, columns=['UMAP1', 'UMAP2'])
#     umap_df['batch'] = sample_df['plot_group']

#     # Drop NaN values from batch column
#     umap_df = umap_df.dropna(subset=['batch'])

#     # Plot UMAP results with batch coloring
#     plt.figure(figsize=(8, 6))

#     unique_batches = umap_df['batch'].unique()
#     cmap = plt.cm.get_cmap('tab10', min(len(unique_batches), 10))  # Up to 10 unique colors

#     has_legend = False

#     for i, batch in enumerate(unique_batches):
#         subset = umap_df[umap_df['batch'] == batch]
#         if not subset.empty:
#             plt.scatter(
#                 subset['UMAP1'],
#                 subset['UMAP2'],
#                 label=str(batch),
#                 color=cmap(i % 10),
#                 s=80,
#                 alpha=0.8,
#                 edgecolors='k'
#             )
#             has_legend = True

#     plt.xlabel('UMAP1')
#     plt.ylabel('UMAP2')
#     plt.title('2D UMAP of HVG Expression (Colored by Batch)')

#     if has_legend:
#         plt.legend(title='Batch', bbox_to_anchor=(1.05, 1), loc='upper left')

#     # Save plot
#     os.makedirs(output_dir, exist_ok=True)
#     save_path = os.path.join(output_dir, 'sample_relationship_umap_2D_batch.pdf')
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()

#     if verbose:
#         print(f"UMAP plot saved to {save_path}")

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.sparse import issparse
# from sklearn.decomposition import PCA

# # Note: Ensure that find_sample_grouping is imported or defined in your environment

# def plot_avg_hvg_expression_pca(adata_sample_diff, output_dir, grouping_columns=['sev.level'], age_bin_size=None, verbose=False):
#     """
#     Computes and plots a 2D PCA of the average HVG expression per sample.
#     The PCA points are colored based on severity levels extracted from the 'plot_group'
#     field, which is derived by calling find_sample_grouping on the input data.

#     Parameters:
#     -----------
#     adata_sample_diff : AnnData-like object
#         Object with attributes .X (expression data) and .obs (observation metadata).
#     output_dir : str
#         Directory where the output plot will be saved.
#     grouping_columns : list, optional
#         List of columns to use for grouping samples (default is ['sev.level']).
#     age_bin_size : int or None, optional
#         Age bin size for grouping, if applicable.
#     verbose : bool, optional
#         If True, prints additional debugging information.
#     """
#     if verbose:
#         print("[visualization_harmony] Computing sample-level PCA from average HVG expression.")
    
#     # Debug prints for data shapes
#     print("adata_sample_diff shape:", adata_sample_diff.shape)
#     print("adata_sample_diff.X shape:", adata_sample_diff.X.shape)
#     print("Is adata_sample_diff.X sparse?", issparse(adata_sample_diff.X))
    
#     # Convert expression data to a DataFrame (handling sparse and dense cases)
#     if issparse(adata_sample_diff.X):
#         df = pd.DataFrame(
#             adata_sample_diff.X.toarray(),
#             index=adata_sample_diff.obs_names,
#             columns=adata_sample_diff.var_names
#         )
#     else:
#         df = pd.DataFrame(
#             adata_sample_diff.X,
#             index=adata_sample_diff.obs_names,
#             columns=adata_sample_diff.var_names
#         )
    
#     # Add sample information and compute average HVG expression per sample
#     df['sample'] = adata_sample_diff.obs['sample']
#     sample_means = df.groupby('sample').mean()
    
#     # Derive the plot grouping using the find_sample_grouping function
#     samples = adata_sample_diff.obs['sample'].unique()
#     diff_groups = find_sample_grouping(adata_sample_diff, samples, grouping_columns, age_bin_size)
    
#     if isinstance(diff_groups, dict):
#         diff_groups = pd.DataFrame.from_dict(diff_groups, orient='index', columns=['plot_group'])
    
#     if not isinstance(diff_groups, pd.DataFrame):
#         raise TypeError(f"Expected diff_groups to be a DataFrame, but got {type(diff_groups)}")
#     if 'plot_group' not in diff_groups.columns:
#         raise KeyError("Column 'plot_group' is missing in diff_groups.")
    
#     # Standardize index formats for merging
#     sample_means.index = sample_means.index.astype(str).str.strip().str.lower()
#     diff_groups.index = diff_groups.index.astype(str).str.strip().str.lower()
    
#     # Reset index and merge grouping information with sample_means
#     diff_groups = diff_groups.reset_index().rename(columns={'index': 'sample'})
#     sample_means = sample_means.reset_index().rename(columns={'index': 'sample'})
#     sample_means = sample_means.merge(diff_groups, on='sample', how='left').set_index('sample')
    
#     # Perform PCA on the sample-averaged data (excluding the plot_group column)
#     pca = PCA(n_components=2)
#     pca_coords = pca.fit_transform(sample_means.drop(columns=['plot_group']))
#     pca_df = pd.DataFrame(pca_coords, index=sample_means.index, columns=['PC1', 'PC2'])
#     pca_df = pca_df.join(sample_means[['plot_group']])
    
#     # Extract severity level as a numeric value from the plot_group string (e.g., "sev.level_X.XX")
#     pca_df['sev_level'] = pca_df['plot_group'].str.extract(r'(\d+\.\d+)').astype(float)
    
#     if pca_df['sev_level'].isna().sum() > 0:
#         raise ValueError("Some plot_group values could not be parsed for severity levels.")
    
#     # Normalize severity values based on the observed range
#     sev_min = pca_df['sev_level'].min()
#     sev_max = pca_df['sev_level'].max()
#     norm_severity = (pca_df['sev_level'] - sev_min) / (sev_max - sev_min)
    
#     # Define the colormap (blue for low severity, red for high)
#     colormap = plt.cm.coolwarm
    
#     plt.figure(figsize=(8, 6))
#     sc = plt.scatter(
#         pca_df['PC1'], pca_df['PC2'],
#         c=norm_severity, cmap=colormap, s=80, alpha=0.8, edgecolors='k'
#     )
#     plt.xlabel('PC1')
#     plt.ylabel('PC2')
#     plt.title('2D PCA of Avg HVG Expression')
#     plt.grid(True)
#     plt.tight_layout()
    
#     # Add a colorbar to show severity levels
#     cbar = plt.colorbar(sc)
#     cbar.set_label('Severity Level')
    
#     # Save the plot
#     os.makedirs(output_dir, exist_ok=True)
#     plot_path = os.path.join(output_dir, 'sample_relationship_pca_2D_sample.pdf')
#     plt.savefig(plot_path)
#     plt.close()
    
#     if verbose:
#         print(f"[visualization_harmony] PCA plot saved to: {plot_path}")