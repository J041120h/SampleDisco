import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Visualization import visualization_harmony
from combat.pycombat import pycombat
from HVG import highly_variable_gene_selection, select_hvf_loess

def check_nan_and_negative_in_lists(df: pd.DataFrame, verbose=False) -> bool:
    found_nan = False
    found_negative = False
    for row_index, row in df.iterrows():
        for col in df.columns:
            cell = row[col]
            if isinstance(cell, np.ndarray):
                if np.isnan(cell).any():
                    if verbose:
                        print(f"Found NaN value in cell at row {row_index}, column '{col}'.")
                    found_nan = True
                if (cell < 0).any():
                    if verbose:
                        print(f"Found negative value(s) in cell at row {row_index}, column '{col}'.")
                    found_negative = True
    if verbose and not found_nan and not found_negative:
        print("No NaN or negative values found in any numpy array cell.")
    return found_nan or found_negative


def vector_to_string(vector):
    """Convert a vector (list, np.array, etc.) to a full string representation without truncation."""
    arr = np.array(vector)
    return np.array2string(arr, threshold=np.inf, separator=', ')

def save_dataframe_as_strings(df: pd.DataFrame, pseudobulk_dir: str, filename: str, verbose=False):
    """Convert all cells in a DataFrame (where each cell is a vector) to strings without truncation, and save to a CSV file."""
    
    np.set_printoptions(threshold=np.inf)
    df_as_strings = df.applymap(lambda x: vector_to_string(x) if isinstance(x, (list, np.ndarray)) else str(x))

    file_path = os.path.join(pseudobulk_dir, filename)
    df_as_strings.to_csv(file_path, index=True)
    
    if verbose:
        print(f"DataFrame saved as strings to {file_path}")

def contains_nan_in_lists(df: pd.DataFrame, verbose=False) -> bool:
    found_nan = False
    for row_index, row in df.iterrows():
        for col in df.columns:
            cell = row[col]
            if isinstance(cell, np.ndarray) and np.isnan(cell).any():
                if verbose:
                    print(f"Found NaN in row {row_index}, column '{col}'.")
                found_nan = True
    return found_nan

def combat_correct_cell_expressions(
    adata: sc.AnnData,
    cell_expression_df: pd.DataFrame,
    cell_proportion_df: pd.DataFrame,
    pseudobulk_dir: str, 
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    parametric: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """Applies ComBat batch correction to cell expression data efficiently."""
    
    # Check for invalid values
    check_nan_and_negative_in_lists(cell_expression_df, verbose=verbose)

    # Efficiently create sample-batch mapping
    sample_batch_map = (
        adata.obs.groupby(sample_col)[batch_col].first().to_dict()
    )

    # Identify number of genes efficiently
    example_array = next(
        (arr for arr in cell_expression_df.iloc[0].dropna() if isinstance(arr, (list, np.ndarray)) and len(arr) > 0),
        None
    )
    if example_array is None:
        raise ValueError("No valid arrays found in cell_expression_df.")
    n_genes = len(example_array)

    # Initialize corrected DataFrame
    corrected_df = cell_expression_df.copy()

    # Precompute batch counts to avoid recomputation
    batch_counts = pd.Series(list(sample_batch_map.values())).value_counts()
    cell_types_to_drop = set()

    for ctype, row_data in corrected_df.iterrows():
        # Extract valid samples and expression data
        valid_mask = row_data.notna() & row_data.apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0)
        valid_sample_ids = row_data.index[valid_mask]

        if valid_sample_ids.empty:
            if verbose:
                print(f"Skipping '{ctype}' (no valid samples).")
            cell_types_to_drop.add(ctype)
            continue

        # Extract expression data in bulk
        arrays_for_this_ctype = np.array([np.array(row_data[sample_id], dtype=float) for sample_id in valid_sample_ids])
        arrays_for_this_ctype = np.nan_to_num(arrays_for_this_ctype, nan=0, posinf=0, neginf=0)

        # Assign batch labels
        batch_labels = np.array([sample_batch_map.get(sample_id, "missing_batch") for sample_id in valid_sample_ids])
        unique_batches = np.unique(batch_labels)

        # Ensure batch diversity
        if len(unique_batches) < 2 or np.any(batch_counts[unique_batches] < 2):
            if verbose:
                print(f"Skipping '{ctype}' due to batch diversity issues.")
            cell_types_to_drop.add(ctype)
            continue

        # Convert to gene-by-sample matrix
        expr_matrix = arrays_for_this_ctype.T
        var_per_gene = np.var(expr_matrix, axis=1)
        non_zero_var_mask = var_per_gene > 0

        if not non_zero_var_mask.any():
            if verbose:
                print(f"Skipping '{ctype}' (all genes have zero variance).")
            cell_types_to_drop.add(ctype)
            continue

        # Filter out zero-variance genes before batch correction
        expr_matrix_sub = expr_matrix[non_zero_var_mask]

        # Run ComBat
        expr_df_t = pd.DataFrame(expr_matrix_sub, columns=valid_sample_ids)
        batch_series = pd.Series(batch_labels, index=valid_sample_ids, name='batch')

        corrected_df_sub = pycombat(expr_df_t, batch=batch_series, parametric=parametric)

        # Restore the original zero-variance genes
        corrected_expr_matrix = expr_matrix.copy()
        corrected_expr_matrix[non_zero_var_mask] = corrected_df_sub.values
        corrected_expr_matrix_t = corrected_expr_matrix.T

        # Store corrected values efficiently
        corrected_df.loc[ctype, valid_sample_ids] = list(corrected_expr_matrix_t)

        if verbose:
            print(f"ComBat correction applied for '{ctype}'.")

    # Drop invalid cell types
    if cell_types_to_drop:
        if verbose:
            print(f"Removing {len(cell_types_to_drop)} cell types: {cell_types_to_drop}")
        corrected_df.drop(labels=cell_types_to_drop, inplace=True, errors='ignore')
        cell_proportion_df.drop(labels=cell_types_to_drop, inplace=True, errors='ignore')

    # Save corrected expression data
    save_dataframe_as_strings(corrected_df, pseudobulk_dir, "corrected_expression.csv", verbose=verbose)

    if verbose:
        print("ComBat correction completed.")
    
    return corrected_df

def compute_pseudobulk_dataframes(
    adata: sc.AnnData,
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './',
    n_features: int = 2000, 
    frac: float = 0.3,
    verbose: bool = False
):
    """
    Computes pseudobulk expression and cell proportion data from an AnnData object.
    
    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object containing single-cell expression data.
    batch_col : str, default 'batch'
        Column name for batch labels in `adata.obs`.
    sample_col : str, default 'sample'
        Column name for sample identifiers in `adata.obs`.
    celltype_col : str, default 'cell_type'
        Column name for cell type annotations in `adata.obs`.
    output_dir : str, default './'
        Directory to save the computed pseudobulk data.
    n_features : int, default 2000
        Number of highly variable features to retain.
    frac : float, default 0.3
        Fraction parameter for LOESS smoothing in HVG selection.
    verbose : bool, default False
        If True, prints additional progress messages.

    Returns:
    --------
    dict
        A dictionary containing:
        - "cell_expression": Raw cell expression data.
        - "cell_proportion": Cell type proportions per sample.
        - "cell_expression_corrected": Batch-corrected expression data.
    """

    start_time = time.time()  # Start timing
    
    pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
    os.makedirs(pseudobulk_dir, exist_ok=True)

    if verbose:
        print("\nProcessing pseudobulk data...")

    # Handle missing batch labels
    if adata.obs[batch_col].isnull().any():
        print("Warning: Missing batch labels found. Filling missing values with 'Unknown'.")
        adata.obs[batch_col].fillna("Unknown", inplace=True)

    batch_counts = adata.obs[batch_col].value_counts()
    small_batches = batch_counts[batch_counts < 5]
    if not small_batches.empty:
        print(f"Warning: The following batches have fewer than 5 samples: {small_batches.to_dict()}")

    X_data = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X

    # Check for invalid values
    if np.isnan(X_data).any() or (X_data < 0).any() or np.isinf(X_data).any():
        print("Warning: Found NaN, negative, or Inf values in expression data. Replacing with zeros.")
        X_data = np.nan_to_num(X_data, nan=0, posinf=0, neginf=0)

    # Filter genes with zero variance
    gene_variances = np.var(X_data, axis=0)
    nonzero_variance_mask = gene_variances > 0
    if not np.all(nonzero_variance_mask):
        print("Warning: Found genes with zero variance. Excluding these genes.")
    
    gene_names = adata.var_names[nonzero_variance_mask]
    X_data = X_data[:, nonzero_variance_mask]
    
    # Initialize pseudobulk data structures
    samples = adata.obs[sample_col].unique()
    cell_types = adata.obs[celltype_col].unique()
    cell_expression_df = pd.DataFrame(index=cell_types, columns=samples, dtype=object)
    cell_proportion_df = pd.DataFrame(index=cell_types, columns=samples, dtype=float)

    if verbose:
        print(f"Found {len(samples)} samples and {len(cell_types)} cell types. Computing pseudobulk data...")

    # Compute pseudobulk expression and proportions
    for sample in samples:
        sample_mask = adata.obs[sample_col] == sample
        total_cells = np.sum(sample_mask)
        for ctype in cell_types:
            ctype_mask = sample_mask & (adata.obs[celltype_col] == ctype)
            num_cells = np.sum(ctype_mask)
            expr_values = X_data[ctype_mask, :].mean(axis=0) if num_cells > 0 else np.zeros(len(gene_names))
            proportion = num_cells / total_cells if total_cells > 0 else 0.0
            cell_expression_df.loc[ctype, sample] = expr_values
            cell_proportion_df.loc[ctype, sample] = proportion
    
    if verbose:
        print("Successfully computed pseudobulk data.")

    # Apply batch correction
    cell_expression_corrected_df = combat_correct_cell_expressions(adata, cell_expression_df, cell_proportion_df, pseudobulk_dir)
    
    # Select highly variable genes
    cell_expression_corrected_df = highly_variable_gene_selection(cell_expression_corrected_df, 2000)
    cell_expression_corrected_df, top_features = select_hvf_loess(cell_expression_corrected_df, n_features=n_features, frac=frac)
    
    # Transpose proportions for output format
    proportion_df = cell_proportion_df.T
    
    # Store results in dictionary
    pseudobulk = {
        "cell_expression": cell_expression_df,
        "cell_proportion": proportion_df,
        "cell_expression_corrected": cell_expression_corrected_df
    }

    # Save results
    save_dataframe_as_strings(cell_expression_df, pseudobulk_dir, "expression.csv")
    save_dataframe_as_strings(cell_proportion_df, pseudobulk_dir, "proportion.csv")

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time  # Compute elapsed time

    if verbose:
        print(f"\nTime Report:")
        print(f"- Total execution time: {elapsed_time:.2f} seconds")
    
    return pseudobulk
