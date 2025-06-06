import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from harmony import harmonize
import time
import contextlib
import io
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu
import warnings

def combine_rna_and_activity_data(
    rna_h5ad_path,
    activity_h5ad_path,
    rna_cell_meta_path=None,
    activity_cell_meta_path=None,
    rna_sample_meta_path=None,
    activity_sample_meta_path=None,
    rna_sample_column='sample',
    activity_sample_column='sample',
    unified_sample_column='sample',
    rna_batch_key='batch',
    activity_batch_key='batch',
    unified_batch_key='batch',
    rna_prefix='RNA',
    activity_prefix='ATAC',
    verbose=True
):
    """
    Combine RNA and gene activity data into a single AnnData object.
    
    Parameters:
    -----------
    rna_h5ad_path : str
        Path to RNA H5AD file
    activity_h5ad_path : str
        Path to gene activity H5AD file
    rna_cell_meta_path : str, optional
        Path to RNA cell metadata CSV (if None, will extract from obs_names)
    activity_cell_meta_path : str, optional
        Path to gene activity cell metadata CSV (if None, will extract from obs_names)
    rna_sample_meta_path : str, optional
        Path to RNA sample metadata CSV (if None, no additional sample metadata)
    activity_sample_meta_path : str, optional
        Path to gene activity sample metadata CSV (if None, no additional sample metadata)
    rna_sample_column : str
        Column name for sample identification in RNA data
    activity_sample_column : str
        Column name for sample identification in gene activity data
    unified_sample_column : str
        Column name for sample identification in combined data
    rna_batch_key : str
        Column name for batch identification in RNA data
    activity_batch_key : str
        Column name for batch identification in gene activity data
    unified_batch_key : str
        Column name for batch identification in combined data
    rna_prefix : str
        Prefix to add to RNA cell barcodes
    activity_prefix : str
        Prefix to add to gene activity cell barcodes
    verbose : bool
        Print progress messages
    
    Returns:
    --------
    adata_combined : AnnData
        Combined AnnData object with both RNA and gene activity data
    """
    
    if verbose:
        print("=== Loading RNA and Gene Activity Data ===")
    
    # Load the raw count data
    adata_rna = sc.read_h5ad(rna_h5ad_path)
    adata_activity = sc.read_h5ad(activity_h5ad_path)
    
    if verbose:
        print(f"RNA data shape: {adata_rna.shape}")
        print(f"Gene activity data shape: {adata_activity.shape}")
    
    # Load cell metadata
    if verbose:
        print("=== Loading Cell Metadata ===")
    
    # Handle RNA cell metadata
    if rna_cell_meta_path is not None:
        rna_cell_meta = pd.read_csv(rna_cell_meta_path)
        # Ensure barcode column exists or create from index
        if 'barcode' not in rna_cell_meta.columns:
            rna_cell_meta['barcode'] = rna_cell_meta.index.astype(str)
    else:
        # Create minimal cell metadata from obs_names
        if verbose:
            print("No RNA cell metadata provided, creating from obs_names")
        rna_cell_meta = pd.DataFrame({
            'barcode': adata_rna.obs_names.astype(str)
        })
        # Extract sample from barcode if sample column not already in obs
        if rna_sample_column not in adata_rna.obs.columns:
            rna_cell_meta[rna_sample_column] = adata_rna.obs_names.str.split(':').str[0]
    
    # Handle gene activity cell metadata
    if activity_cell_meta_path is not None:
        activity_cell_meta = pd.read_csv(activity_cell_meta_path)
        # Ensure barcode column exists or create from index
        if 'barcode' not in activity_cell_meta.columns:
            activity_cell_meta['barcode'] = activity_cell_meta.index.astype(str)
    else:
        # Create minimal cell metadata from obs_names
        if verbose:
            print("No gene activity cell metadata provided, creating from obs_names")
        activity_cell_meta = pd.DataFrame({
            'barcode': adata_activity.obs_names.astype(str)
        })
        # Extract sample from barcode if sample column not already in obs
        if activity_sample_column not in adata_activity.obs.columns:
            activity_cell_meta[activity_sample_column] = adata_activity.obs_names.str.split(':').str[0]
    
    # Add data type column to distinguish RNA vs gene activity cells
    rna_cell_meta['data_type'] = 'RNA'
    activity_cell_meta['data_type'] = 'Gene_Activity'
    
    # Add prefixes to cell barcodes to make them unique
    rna_cell_meta['barcode'] = rna_prefix + '_' + rna_cell_meta['barcode'].astype(str)
    activity_cell_meta['barcode'] = activity_prefix + '_' + activity_cell_meta['barcode'].astype(str)
    
    # Update AnnData obs_names
    adata_rna.obs_names = [rna_prefix + '_' + str(x) for x in adata_rna.obs_names]
    adata_activity.obs_names = [activity_prefix + '_' + str(x) for x in adata_activity.obs_names]
    
    # Attach cell metadata
    rna_cell_meta.set_index('barcode', inplace=True)
    activity_cell_meta.set_index('barcode', inplace=True)
    
    adata_rna.obs = adata_rna.obs.join(rna_cell_meta, how='left')
    adata_activity.obs = adata_activity.obs.join(activity_cell_meta, how='left')
    
    # Load sample metadata (optional)
    if verbose:
        print("=== Loading Sample Metadata ===")
    
    if rna_sample_meta_path is not None:
        rna_sample_meta = pd.read_csv(rna_sample_meta_path)
        adata_rna.obs = adata_rna.obs.merge(rna_sample_meta, on=rna_sample_column, how='left')
        if verbose:
            print("RNA sample metadata loaded and merged")
    else:
        if verbose:
            print("No RNA sample metadata provided")
    
    if activity_sample_meta_path is not None:
        activity_sample_meta = pd.read_csv(activity_sample_meta_path)
        adata_activity.obs = adata_activity.obs.merge(activity_sample_meta, on=activity_sample_column, how='left')
        if verbose:
            print("Gene activity sample metadata loaded and merged")
    else:
        if verbose:
            print("No gene activity sample metadata provided")
    
    # Standardize column names to unified names
    if verbose:
        print("=== Standardizing Column Names ===")
    
    # Ensure required columns exist (create default values if needed)
    # Handle sample column
    if rna_sample_column not in adata_rna.obs.columns:
        if verbose:
            print(f"RNA sample column '{rna_sample_column}' not found, creating from data_type")
        adata_rna.obs[rna_sample_column] = 'RNA_sample'
    
    if activity_sample_column not in adata_activity.obs.columns:
        if verbose:
            print(f"Gene activity sample column '{activity_sample_column}' not found, creating from data_type")
        adata_activity.obs[activity_sample_column] = 'ATAC_sample'
    
    # Handle batch column (create default if not provided)
    if rna_batch_key not in adata_rna.obs.columns:
        if verbose:
            print(f"RNA batch column '{rna_batch_key}' not found, creating default")
        adata_rna.obs[rna_batch_key] = 'RNA_batch'
    
    if activity_batch_key not in adata_activity.obs.columns:
        if verbose:
            print(f"Gene activity batch column '{activity_batch_key}' not found, creating default")
        adata_activity.obs[activity_batch_key] = 'ATAC_batch'
    
    # Rename sample columns to unified name
    if rna_sample_column != unified_sample_column:
        if unified_sample_column in adata_rna.obs.columns and unified_sample_column != rna_sample_column:
            adata_rna.obs.drop(columns=[unified_sample_column], inplace=True)
        adata_rna.obs[unified_sample_column] = adata_rna.obs[rna_sample_column]
    
    if activity_sample_column != unified_sample_column:
        if unified_sample_column in adata_activity.obs.columns and unified_sample_column != activity_sample_column:
            adata_activity.obs.drop(columns=[unified_sample_column], inplace=True)
        adata_activity.obs[unified_sample_column] = adata_activity.obs[activity_sample_column]
    
    # Rename batch columns to unified name
    if rna_batch_key != unified_batch_key:
        if unified_batch_key in adata_rna.obs.columns and unified_batch_key != rna_batch_key:
            adata_rna.obs.drop(columns=[unified_batch_key], inplace=True)
        adata_rna.obs[unified_batch_key] = adata_rna.obs[rna_batch_key]
    
    if activity_batch_key != unified_batch_key:
        if unified_batch_key in adata_activity.obs.columns and unified_batch_key != activity_batch_key:
            adata_activity.obs.drop(columns=[unified_batch_key], inplace=True)
        adata_activity.obs[unified_batch_key] = adata_activity.obs[activity_batch_key]
    
    if verbose:
        print("=== Combining Datasets ===")
        print(f"RNA data shape: {adata_rna.shape}")
        print(f"Gene activity data shape: {adata_activity.shape}")
    
    # Combine the datasets using scanpy concat with outer join
    # This automatically handles missing genes by filling with zeros
    adata_combined = sc.concat([adata_rna, adata_activity], axis=0, join='outer')
    
    if verbose:
        print(f"Combined data shape: {adata_combined.shape}")
        print(f"RNA cells: {sum(adata_combined.obs['data_type'] == 'RNA')}")
        print(f"Gene activity cells: {sum(adata_combined.obs['data_type'] == 'Gene_Activity')}")
        print(f"Total unique genes: {adata_combined.n_vars}")
    
    # Return the combined AnnData object
    return adata_combined


def clean_obs_for_writing(adata):
    """
    Clean obs dataframe to ensure it can be written to H5AD format.
    Converts problematic data types and handles missing values.
    """
    import pandas as pd
    import numpy as np
    
    # Create a copy of obs to avoid modifying the original
    obs_clean = adata.obs.copy()
    
    for col in obs_clean.columns:
        # Check if column has object dtype
        if obs_clean[col].dtype == 'object':
            # Replace NaN values with empty string
            obs_clean[col] = obs_clean[col].fillna('')
            
            # Convert all values to string
            obs_clean[col] = obs_clean[col].astype(str)
            
            # Replace 'nan' string with empty string
            obs_clean[col] = obs_clean[col].replace('nan', '')
        
        # Handle boolean columns
        elif obs_clean[col].dtype == 'bool':
            # Convert to string representation
            obs_clean[col] = obs_clean[col].astype(str)
        
        # Handle any other problematic dtypes
        elif obs_clean[col].dtype.name.startswith('category'):
            # Convert categorical to string
            obs_clean[col] = obs_clean[col].astype(str)
    
    # Update the adata obs
    adata.obs = obs_clean
    
    return adata



def combined_harmony_analysis(
    rna_h5ad_path,
    activity_h5ad_path,
    rna_cell_meta_path=None,
    activity_cell_meta_path=None,
    rna_sample_meta_path=None,
    activity_sample_meta_path=None,
    output_dir=None,
    rna_sample_column='sample',
    activity_sample_column='sample',
    unified_sample_column='sample',
    rna_batch_key='batch',
    activity_batch_key='batch',
    unified_batch_key='batch',
    rna_prefix='RNA',
    activity_prefix='ATAC',
    num_PCs=20,
    num_harmony=30,
    num_features=2000,
    min_cells=500,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
    vars_to_regress=[],
    verbose=True
):
    start_time = time.time()
    
    # Create output directories (optional)
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            if verbose:
                print("Created output directory")
        
        output_dir = os.path.join(output_dir, 'combined_harmony')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            if verbose:
                print("Created combined_harmony subdirectory")
    
    # Step 1: Combine RNA and gene activity data
    adata_combined = combine_rna_and_activity_data(
        rna_h5ad_path=rna_h5ad_path,
        activity_h5ad_path=activity_h5ad_path,
        rna_cell_meta_path=rna_cell_meta_path,
        activity_cell_meta_path=activity_cell_meta_path,
        rna_sample_meta_path=rna_sample_meta_path,
        activity_sample_meta_path=activity_sample_meta_path,
        rna_sample_column=rna_sample_column,
        activity_sample_column=activity_sample_column,
        unified_sample_column=unified_sample_column,
        rna_batch_key=rna_batch_key,
        activity_batch_key=activity_batch_key,
        unified_batch_key=unified_batch_key,
        rna_prefix=rna_prefix,
        activity_prefix=activity_prefix,
        verbose=verbose
    )
    
    # Prepare vars_to_regress for harmony - automatically add data_type and sample
    vars_to_regress_for_harmony = vars_to_regress.copy()
    if unified_sample_column not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append(unified_sample_column)
    if 'data_type' not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append('data_type')
    
    # Error checking for required columns
    all_required_columns = vars_to_regress_for_harmony + [unified_batch_key]
    missing_vars = [col for col in all_required_columns if col not in adata_combined.obs.columns]
    if missing_vars:
        raise KeyError(f"The following variables are missing from adata_combined.obs: {missing_vars}")
    
    if verbose:
        print("=== Starting Quality Control and Filtering ===")
    
    # Basic filtering
    sc.pp.filter_cells(adata_combined, min_genes=min_features)
    sc.pp.filter_genes(adata_combined, min_cells=min_cells)
    if verbose:
        print(f"After basic filtering -- Cells: {adata_combined.n_obs}, Genes: {adata_combined.n_vars}")
    
    # Mitochondrial QC
    adata_combined.var['mt'] = adata_combined.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_combined, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata_combined = adata_combined[adata_combined.obs['pct_counts_mt'] < pct_mito_cutoff].copy()
    
    # Exclude genes
    mt_genes = adata_combined.var_names[adata_combined.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    adata_combined = adata_combined[:, ~adata_combined.var_names.isin(genes_to_exclude)].copy()
    
    if verbose:
        print(f"After MT filtering -- Cells: {adata_combined.n_obs}, Genes: {adata_combined.n_vars}")
    
    # Sample filtering
    cell_counts_per_sample = adata_combined.obs.groupby(unified_sample_column).size()
    if verbose:
        print("Sample counts before filtering:")
        print(cell_counts_per_sample.sort_values(ascending=False))
    
    samples_to_keep = cell_counts_per_sample[cell_counts_per_sample >= min_cells].index
    adata_combined = adata_combined[adata_combined.obs[unified_sample_column].isin(samples_to_keep)].copy()
    
    if verbose:
        print(f"Samples retained: {list(samples_to_keep)}")
        print("Sample counts after filtering:")
        print(adata_combined.obs[unified_sample_column].value_counts().sort_values(ascending=False))
    
    # Final gene filtering
    min_cells_for_gene = int(0.01 * adata_combined.n_obs)
    sc.pp.filter_genes(adata_combined, min_cells=min_cells_for_gene)
    
    if verbose:
        print(f"Final dimensions -- Cells: {adata_combined.n_obs}, Genes: {adata_combined.n_vars}")
    
    # Optional doublet detection
    if doublet:
        if verbose:
            print("=== Running Doublet Detection ===")
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            sc.pp.scrublet(adata_combined)
        adata_combined = adata_combined[~adata_combined.obs['predicted_doublet']].copy()
        if verbose:
            print(f"After doublet removal -- Cells: {adata_combined.n_obs}")
    
    # Save raw data
    adata_combined.raw = adata_combined.copy()
    
    # Step 2: Preprocessing and Harmony integration
    if verbose:
        print('=== Processing Combined Data for Integration ===')
    
    # Normalization and log transformation
    sc.pp.normalize_total(adata_combined, target_sum=1e4)
    sc.pp.log1p(adata_combined)
    
    # HVG selection
    sc.pp.highly_variable_genes(
        adata_combined,
        n_top_genes=num_features,
        flavor='seurat_v3',
        batch_key=unified_sample_column
    )
    adata_combined = adata_combined[:, adata_combined.var['highly_variable']].copy()
    
    # PCA
    sc.tl.pca(adata_combined, n_comps=num_PCs, svd_solver='arpack')
    
    if verbose:
        print('=== Running Harmony Integration ===')
        print(f'Variables to regress: {", ".join(vars_to_regress_for_harmony)}')
    
    # Harmony integration
    Z = harmonize(
        adata_combined.obsm['X_pca'],
        adata_combined.obs,
        batch_key=vars_to_regress_for_harmony,
        max_iter_harmony=num_harmony,
        use_gpu=True
    )
    adata_combined.obsm['X_pca_harmony'] = Z
    
    # Neighbors and UMAP (always compute these)
    if verbose:
        print('=== Computing Neighbors and UMAP ===')

    clean_obs_for_writing(adata_combined)
    sc.write(os.path.join(output_dir, 'adata_combined.h5ad'), adata_combined)
    # Print summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if verbose:
        print(f"\n=== Analysis Complete ===")
        print(f"Execution time: {elapsed_time:.2f} seconds")
        print(f"Final data shape: {adata_combined.shape}")


# Example usage:
if __name__ == "__main__":
    adata_integrated = combined_harmony_analysis(
        rna_h5ad_path="/Users/harry/Desktop/GenoDistance/Data/count_data.h5ad",
        activity_h5ad_path="/Users/harry/Desktop/GenoDistance/result/gene_activity/gene_activity_weighted.h5ad",
        rna_cell_meta_path=None,
        activity_cell_meta_path=None,
        rna_sample_meta_path="/Users/harry/Desktop/GenoDistance/Data/sample_data.csv",
        activity_sample_meta_path="/Users/harry/Desktop/GenoDistance/Data/ATAC_Metadata.csv",
        output_dir="/Users/harry/Desktop/GenoDistance/result",
        verbose=True
    )