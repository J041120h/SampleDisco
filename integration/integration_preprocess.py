import os
import scanpy as sc
import pandas as pd
import numpy as np
import time
import contextlib
import io
from scipy import sparse
import gc
import psutil


def sanitize_sparse_matrix(X, verbose=False):
    """
    Ensure a sparse matrix is properly formatted and safe for operations.
    Handles both CSR and CSC matrices.
    """
    if not sparse.issparse(X):
        return X
    
    if verbose:
        print(f"   Sanitizing sparse matrix...")
        print(f"     Format: {X.format}")
        print(f"     Shape: {X.shape}")
        print(f"     NNZ: {X.nnz:,}")
    
    # Convert to CSR if not already
    if not isinstance(X, sparse.csr_matrix):
        if verbose:
            print(f"     Converting from {X.format} to CSR...")
        X = X.tocsr()
    
    # Ensure proper operations in correct order
    if verbose:
        print(f"     Removing duplicates...")
    X.sum_duplicates()
    
    if verbose:
        print(f"     Sorting indices...")
    X.sort_indices()
    
    # Check if we need int64 for large matrices
    max_nnz = X.shape[0] * X.shape[1]
    needs_int64 = X.nnz > np.iinfo(np.int32).max or max_nnz > np.iinfo(np.int32).max
    
    if needs_int64:
        if verbose:
            print(f"     Large matrix detected, using int64 indices")
        X.indices = X.indices.astype(np.int64, copy=False)
        X.indptr = X.indptr.astype(np.int64, copy=False)
    else:
        # For smaller matrices, ensure int32
        if X.indices.dtype != np.int32:
            if verbose:
                print(f"     Converting indices from {X.indices.dtype} to int32")
            X.indices = X.indices.astype(np.int32, copy=False)
        if X.indptr.dtype != np.int32:
            if verbose:
                print(f"     Converting indptr from {X.indptr.dtype} to int32")
            X.indptr = X.indptr.astype(np.int32, copy=False)
    
    if verbose:
        print(f"     Sanitization complete!")
        print(f"     Final format: {X.format}")
        print(f"     Indices dtype: {X.indices.dtype}")
        print(f"     Sorted: {X.has_sorted_indices}")
    
    return X


def fill_obs_nan_with_unknown(
    adata: sc.AnnData,
    fill_value: str = "unKnown",
    verbose: bool = False,
) -> None:
    """
    Replace NaN values in all .obs columns with `fill_value`.
    """
    for col in adata.obs.columns:
        ser = adata.obs[col]

        # Skip if the column has no missing values
        if not ser.isnull().any():
            continue

        # Handle categoricals
        if pd.api.types.is_categorical_dtype(ser):
            if fill_value not in ser.cat.categories:
                ser = ser.cat.add_categories([fill_value])
            ser = ser.fillna(fill_value)
        # Handle everything else
        else:
            if pd.api.types.is_numeric_dtype(ser):
                ser = ser.astype("object")
            ser = ser.fillna(fill_value)

        # Write back to AnnData
        adata.obs[col] = ser

        if verbose:
            print(f"✓ Filled NaNs in .obs['{col}'] with '{fill_value}'")


def integrate_preprocess(
    output_dir,
    h5ad_path = None,
    sample_column = 'sample',
    modality_col = 'modality',
    min_cells_sample=1,
    min_cell_gene=10,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
    verbose=True
):
    """
    Integration preprocessing with proper sparse matrix handling.
    
    FIXED: Ensures sparse matrices are properly sorted before operations.
    """
    # Start timing
    start_time = time.time()

    if h5ad_path is None:
        h5ad_path = os.path.join(output_dir, 'preprocess/atac_rna_integrated.h5ad')
        
    # Check if file exists
    if not os.path.exists(h5ad_path):
        # Try alternate path
        alt_path = os.path.join(output_dir, 'glue/atac_rna_integrated.h5ad')
        if os.path.exists(alt_path):
            h5ad_path = alt_path
            if verbose:
                print(f"Using alternate path: {h5ad_path}")
        else:
            raise FileNotFoundError(f"Input file not found: {h5ad_path}")

    # Create output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating output_dir")
    
    output_dir = os.path.join(output_dir, 'preprocess')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating preprocess subdirectory")

    if doublet and min_cells_sample < 30:
        min_cells_sample = 30
        if verbose:
            print("Minimum dimension requested by scrublet is 30, raised min_cells_sample accordingly")
    
    # 1. Read the raw count data from an existing H5AD
    if verbose:
        print('=== Reading input dataset ===')
        print(f'Input file: {h5ad_path}')
    
    adata = sc.read_h5ad(h5ad_path)
    
    if verbose:
        print(f'Dimension of raw data (cells x genes): {adata.shape[0]:,} x {adata.shape[1]:,}')
    
    # CRITICAL FIX: Sanitize the sparse matrix immediately after loading
    if sparse.issparse(adata.X):
        if verbose:
            print('\n=== Sanitizing sparse matrix ===')
        adata.X = sanitize_sparse_matrix(adata.X, verbose=verbose)
    
    # Also sanitize layers if they exist
    if adata.layers:
        if verbose:
            print('\n=== Checking layers ===')
        for layer_name in list(adata.layers.keys()):
            if sparse.issparse(adata.layers[layer_name]):
                if verbose:
                    print(f'Sanitizing layer: {layer_name}')
                adata.layers[layer_name] = sanitize_sparse_matrix(
                    adata.layers[layer_name], 
                    verbose=False  # Less verbose for layers
                )

    # Modify sample IDs by adding modality information
    if modality_col is not None and modality_col in adata.obs.columns:
        adata.obs[sample_column] = adata.obs[sample_column].astype(str) + '_' + adata.obs[modality_col].astype(str)
        if verbose:
            print(f"Modified sample IDs by adding modality information from '{modality_col}' column")

    # Basic filtering
    if verbose:
        print('\n=== Basic filtering ===')
        print(f'Before filtering: {adata.n_obs:,} cells, {adata.n_vars:,} genes')
    
    sc.pp.filter_cells(adata, min_genes=min_features)
    if verbose:
        print(f'After min_genes={min_features}: {adata.n_obs:,} cells')
    
    sc.pp.filter_genes(adata, min_cells=min_cell_gene)
    if verbose:
        print(f'After min_cells={min_cell_gene}: {adata.n_vars:,} genes')

    # Mito QC
    if verbose:
        print('\n=== Mitochondrial QC ===')
    
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    n_cells_before = adata.n_obs
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()
    
    if verbose:
        n_removed = n_cells_before - adata.n_obs
        print(f'Removed {n_removed:,} cells with >={pct_mito_cutoff}% mitochondrial reads')
        print(f'Cells remaining: {adata.n_obs:,}')

    # Exclude genes if needed
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    
    n_genes_before = adata.n_vars
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()
    
    if verbose:
        n_removed = n_genes_before - adata.n_vars
        print(f'Removed {n_removed} genes (MT genes and user-specified)')
        print(f"After gene removal -- Cells: {adata.n_obs:,}, Genes: {adata.n_vars:,}")

    # Re-sanitize after subsetting (subsetting can sometimes unsort indices)
    if sparse.issparse(adata.X):
        adata.X = sanitize_sparse_matrix(adata.X, verbose=False)

    # Sample filtering
    cell_counts_per_patient = adata.obs.groupby(sample_column).size()
    if verbose:
        print("\n=== Sample filtering ===")
        print("Sample counts BEFORE filtering:")
        print(cell_counts_per_patient.sort_values(ascending=False))
    
    patients_to_keep = cell_counts_per_patient[cell_counts_per_patient >= min_cells_sample].index
    if verbose:
        print(f"\nSamples retained (>= {min_cells_sample} cells): {len(patients_to_keep)}")
    
    adata = adata[adata.obs[sample_column].isin(patients_to_keep)].copy()
    
    cell_counts_after = adata.obs[sample_column].value_counts()
    if verbose:
        print("\nSample counts AFTER filtering:")
        print(cell_counts_after.sort_values(ascending=False))

    # Re-sanitize after sample filtering
    if sparse.issparse(adata.X):
        adata.X = sanitize_sparse_matrix(adata.X, verbose=False)

    # Drop genes that are too rare in these final cells
    min_cells_for_gene = int(0.01 * adata.n_obs)
    sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)
    if verbose:
        print(f"\nFinal filtering -- Cells: {adata.n_obs:,}, Genes: {adata.n_vars:,}")

    # Final sanitization after all filtering
    if sparse.issparse(adata.X):
        adata.X = sanitize_sparse_matrix(adata.X, verbose=False)

    # Optional doublet detection
    if doublet:
        if verbose:
            print(f"\n=== Doublet detection ===")
            print(f"Running scrublet on {adata.n_obs:,} cells...")
        
        try:
            # Store original cell count for comparison
            original_n_cells = adata.n_obs
            
            # Create a copy for scrublet to avoid modifying original
            adata_scrub = adata.copy()
            
            # Ensure the copy also has sanitized matrix
            if sparse.issparse(adata_scrub.X):
                adata_scrub.X = sanitize_sparse_matrix(adata_scrub.X, verbose=False)
            
            # Run scrublet with suppressed output
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                sc.pp.scrublet(adata_scrub, batch_key=sample_column)
            
            # Check if scrublet results match our data
            if 'predicted_doublet' not in adata_scrub.obs.columns:
                if verbose:
                    print("Warning: Scrublet did not add 'predicted_doublet' column. Skipping doublet removal.")
            elif adata_scrub.n_obs != original_n_cells:
                if verbose:
                    print(f"Warning: Scrublet changed cell count. Using original data without doublet removal.")
            else:
                # Successfully ran scrublet, now filter doublets
                n_doublets = adata_scrub.obs['predicted_doublet'].sum()
                if verbose:
                    print(f"Detected {n_doublets:,} doublets out of {original_n_cells:,} cells")
                
                # Copy the scrublet results back to original adata
                adata.obs['predicted_doublet'] = adata_scrub.obs['predicted_doublet']
                adata.obs['doublet_score'] = adata_scrub.obs.get('doublet_score', 0)
                
                # Filter out doublets
                adata = adata[~adata.obs['predicted_doublet']].copy()
                
                if verbose:
                    print(f"After doublet removal: {adata.n_obs:,} cells remaining")
                
                # Sanitize after doublet removal
                if sparse.issparse(adata.X):
                    adata.X = sanitize_sparse_matrix(adata.X, verbose=False)
        
        except Exception as e:
            if verbose:
                print(f"Warning: Scrublet failed with error: {str(e)}")
                print("Continuing without doublet detection...")

    # Fill NaN values
    fill_obs_nan_with_unknown(adata, verbose=verbose)
    
    # Store raw data
    adata.raw = adata.copy()
    
    if verbose:
        print("\n=== Preprocessing complete! ===")
        print(f"Final dimensions: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Save to new file instead of overwriting original
    output_h5ad_path = os.path.join(output_dir, 'adata_sample.h5ad')
    
    # Final check before saving
    if sparse.issparse(adata.X):
        if verbose:
            print(f"\n=== Final matrix check before saving ===")
            print(f"  Format: {adata.X.format}")
            print(f"  Sorted indices: {adata.X.has_sorted_indices}")
            print(f"  Indices dtype: {adata.X.indices.dtype}")
            print(f"  NNZ: {adata.X.nnz:,}")
        
        # One final sanitization to be absolutely sure
        adata.X = sanitize_sparse_matrix(adata.X, verbose=False)
    
    sc.write(output_h5ad_path, adata)
    if verbose:
        print(f"\nPreprocessed data saved to: {output_h5ad_path}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print execution time
    if verbose:
        print(f"\nFunction execution time: {elapsed_time:.2f} seconds")

    return adata