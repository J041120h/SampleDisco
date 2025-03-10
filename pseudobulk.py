import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Visualization import visualization_harmony
from combat.pycombat import pycombat
from HVG import highly_variable_gene_selection

def check_nan_and_negative_in_lists(df: pd.DataFrame) -> bool:
    found_nan = False
    found_negative = False
    for row_index, row in df.iterrows():
        for col in df.columns:
            cell = row[col]
            if isinstance(cell, np.ndarray):
                if np.isnan(cell).any():
                    print(f"Found NaN value in cell at row {row_index}, column '{col}'.")
                    found_nan = True
                if (cell < 0).any():
                    print(f"Found negative value(s) in cell at row {row_index}, column '{col}'.")
                    found_negative = True
    if not found_nan and not found_negative:
        print("No NaN or negative values found in any numpy array cell.")
    return found_nan or found_negative


import numpy as np
import pandas as pd
from combat.pycombat import pycombat
import os

def vector_to_string(vector):
    """
    Convert a vector (list, np.array, etc.) to a full string representation without truncation.
    """
    # Ensure the vector is a NumPy array and set the threshold to infinity to prevent truncation.
    arr = np.array(vector)
    return np.array2string(arr, threshold=np.inf, separator=', ')

def save_dataframe_as_strings(df: pd.DataFrame, pseudobulk_dir: str, filename: str):
    """
    Convert all cells in a DataFrame (where each cell is a vector) to strings without truncation,
    and save to a CSV file in a 'pseudobulk' subdirectory within output_dir.
    """

    # Set numpy print options to ensure full vectors are printed without truncation.
    np.set_printoptions(threshold=np.inf)
    
    # Convert each cell to a string: use vector_to_string for lists/arrays; otherwise, convert to string.
    df_as_strings = df.applymap(lambda x: vector_to_string(x) if isinstance(x, (list, np.ndarray)) else str(x))

    # Create the full file path.
    file_path = os.path.join(pseudobulk_dir, filename)

    # Save the resulting DataFrame to CSV.
    df_as_strings.to_csv(file_path, index=True)
    print(f"DataFrame saved as strings to {file_path}")



def contains_nan_in_lists(df: pd.DataFrame) -> bool:
    found_nan = False
    for row_index, row in df.iterrows():
        for col in df.columns:
            cell = row[col]
            if isinstance(cell, np.ndarray) and np.isnan(cell).any():
                print(f"Found NaN in row {row_index}, column '{col}'.")
                found_nan = True
    return found_nan

import numpy as np
import pandas as pd
import scanpy as sc

def combat_correct_cell_expressions(
    adata: sc.AnnData,
    cell_expression_df: pd.DataFrame,
    cell_proportion_df: pd.DataFrame,
    pseudobulk_dir: str, 
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    parametric: bool = True
) -> pd.DataFrame:
    # Check for problematic values (NaN or negatives) in the data
    check_nan_and_negative_in_lists(cell_expression_df)
    
    sample_batch_map = (
        adata.obs[[sample_col, batch_col]]
        .drop_duplicates()
        .set_index(sample_col)[batch_col]
        .to_dict()
    )
    
    example_array = next(
        (arr for arr in cell_expression_df.iloc[0].dropna() if arr is not None and len(arr) > 0),
        None
    )
    if example_array is None:
        raise ValueError("No valid arrays found in cell_expression_df.")
    n_genes = len(example_array)
    
    # Make a deep copy of the original DataFrame to store corrected data
    corrected_df = cell_expression_df.copy(deep=True)
    
    # We'll collect any cell types we need to remove entirely
    cell_types_to_drop = set()
    
    # Process each cell type separately
    for ctype in corrected_df.index:
        if ctype in cell_types_to_drop:
            continue  # Already marked for removal
        
        row_data = corrected_df.loc[ctype]
        batch_labels = []
        arrays_for_this_ctype = []
        valid_sample_ids = []  # keep track of samples with non-zero proportion
        
        # Loop over samples for the given cell type
        for sample_id in row_data.index:
            expr_array = row_data[sample_id]
            # Check if the cell type is absent in the sample by testing if its expression vector is all zeros.
            if expr_array is None or len(expr_array) == 0 or np.allclose(expr_array, 0):
                continue
            else:
                expr_array = np.array(expr_array, dtype=float)
            # Replace NaN or infinite values with zero
            expr_array = np.nan_to_num(expr_array, nan=0, posinf=0, neginf=0)
            arrays_for_this_ctype.append(expr_array)
            batch_labels.append(sample_batch_map.get(sample_id, "missing_batch"))
            valid_sample_ids.append(sample_id)
        
        # If no samples with non-zero expression were found, mark for removal
        if len(arrays_for_this_ctype) == 0:
            print(f"Deleting '{ctype}' because no samples with non-zero expression were found.")
            cell_types_to_drop.add(ctype)
            continue
        
        batch_labels_array = np.array(batch_labels)
        unique_batches = pd.unique(batch_labels_array)
        batch_counts = pd.Series(batch_labels).value_counts()
        
        # If we have too few batches or an insufficient number of samples per batch, drop this cell type
        if len(unique_batches) < 2 or any(batch_counts < 2):
            print(
                f"\nDeleting '{ctype}' due to insufficient batch diversity or small batch sizes: "
                f"{batch_counts.to_dict()}.\n"
            )
            cell_types_to_drop.add(ctype)
            continue
        
        # Create the expression matrix for Combat: genes x samples
        expr_matrix = np.vstack(arrays_for_this_ctype).T
        if np.any(expr_matrix < 0):
            print(f"\nWarning: Negative values detected in '{ctype}' expression matrix.\n")
        
        # Identify genes with zero variance (can cause errors in Combat)
        var_per_gene = np.var(expr_matrix, axis=1)
        zero_var_idx = np.where(var_per_gene == 0)[0]
        if len(zero_var_idx) == expr_matrix.shape[0]:
            print(f"\nDeleting '{ctype}' because all genes have zero variance.\n")
            cell_types_to_drop.add(ctype)
            continue
        
        # Remove zero-variance genes for Combat, then run pycombat
        expr_matrix_sub = np.delete(expr_matrix, zero_var_idx, axis=0)
        expr_df_t = pd.DataFrame(expr_matrix_sub, columns=valid_sample_ids)
        batch_series = pd.Series(batch_labels, index=valid_sample_ids, name='batch')
        
        if expr_df_t.isnull().values.any():
            print(f"\nWarning: NaN values detected in expression data for '{ctype}' before ComBat.\n")
        
        corrected_df_sub = pycombat(
            expr_df_t,
            batch=batch_series,
            parametric=parametric
        )
        
        print(f"After pycombat: {np.isnan(corrected_df_sub.values).sum()} NaN values for '{ctype}'")
        corrected_values_sub = corrected_df_sub.values
        
        # Reconstruct the full corrected expression matrix by reinserting zero-variance genes
        corrected_expr_matrix = expr_matrix.copy()
        corrected_expr_matrix[zero_var_idx, :] = expr_matrix[zero_var_idx, :]
        non_zero_idx = np.delete(np.arange(n_genes), zero_var_idx)
        corrected_expr_matrix[non_zero_idx, :] = corrected_values_sub
        corrected_expr_matrix_t = corrected_expr_matrix.T
        
        # Update the corrected DataFrame for the samples included in Combat correction
        for i, sample_id in enumerate(valid_sample_ids):
            corrected_df.loc[ctype, sample_id] = corrected_expr_matrix_t[i]
        
        print(f"ComBat correction applied for '{ctype}'.")
    
    # First drop any cell types that were flagged for removal
    if cell_types_to_drop:
        print(f"\nRemoving {len(cell_types_to_drop)} cell types that failed checks: {cell_types_to_drop}\n")
        corrected_df.drop(labels=cell_types_to_drop, inplace=True, errors='ignore')
        cell_proportion_df.drop(labels=cell_types_to_drop, inplace=True, errors='ignore')
    
    # Next, check for any NaNs in the corrected DataFrame
    # If a row contains NaNs, drop that entire row from both dataframes
    for idx in corrected_df.index:
        row = corrected_df.loc[idx]
        if any(isinstance(cell, np.ndarray) and np.isnan(cell).any() for cell in row):
            print(f"\nRow '{idx}' contains NaNs after ComBat correction. Removing this cell type.\n")
            cell_types_to_drop.add(idx)
    
    # Drop any additional rows found to have NaNs
    if cell_types_to_drop:
        print(f"\nDrop '{cell_types_to_drop}'\n")
        corrected_df.drop(labels=cell_types_to_drop, inplace=True, errors='ignore')
        cell_proportion_df.drop(labels=cell_types_to_drop, inplace=True, errors='ignore')
    
    # Finally, save the corrected data (if needed)
    save_dataframe_as_strings(corrected_df, pseudobulk_dir, "corrected_expression.csv")
    
    print("ComBat correction completed.")
    return corrected_df

def compute_pseudobulk_dataframes(
    adata: sc.AnnData,
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './'
):
    pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
    os.makedirs(pseudobulk_dir, exist_ok=True)

    if adata.obs[batch_col].isnull().any():
        print("\n\n\n\nWarning: Missing batch labels found. Filling missing values with 'Unknown'.\n\n\n\n")
        adata.obs[batch_col].fillna("Unknown", inplace=True)
    batch_counts = adata.obs[batch_col].value_counts()
    small_batches = batch_counts[batch_counts < 5]
    if not small_batches.empty:
        print(f"\n\n\n\nWarning: The following batches have fewer than 5 samples: {small_batches.to_dict()}. Consider merging these batches.\n\n\n\n")

    X_data = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X

    if np.isnan(X_data).any() or (X_data < 0).any() or np.isinf(X_data).any():
        if np.isnan(X_data).any():
            print("\n\n\n\nWarning: X_data contains NaN values.\n\n\n\n")
        if (X_data < 0).any():
            print("\n\n\n\nWarning: X_data contains negative values.\n\n\n\n")
        if np.isinf(X_data).any():
            print("\n\n\n\nWarning: X_data contains Inf values.\n\n\n\n")
        print("\n\n\n\nWarning: Found NaN or Inf values in expression data. Replacing with zeros.\n\n\n\n")
        X_data = np.nan_to_num(X_data, nan=0, posinf=0, neginf=0)
    else:
        print("\n\n\n\nNo NaN, negative, or Inf values found in X_data.\n\n\n\n")

    gene_variances = np.var(X_data, axis=0)
    nonzero_variance_mask = gene_variances > 0
    if not np.all(nonzero_variance_mask):
        print("\n\n\n\nWarning: Found genes with zero variance. Excluding these genes from analysis.\n\n\n\n")
    gene_names = adata.var_names[nonzero_variance_mask]
    X_data = X_data[:, nonzero_variance_mask]
    
    samples = adata.obs[sample_col].unique()
    cell_types = adata.obs[celltype_col].unique()
    cell_expression_df = pd.DataFrame(index=cell_types, columns=samples, dtype=object)
    cell_proportion_df = pd.DataFrame(index=cell_types, columns=samples, dtype=float)
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
    print("\n\n\n\nSuccessfully computed pseudobulk data.\n\n\n\n")

    cell_expression_corrected_df = combat_correct_cell_expressions(adata, cell_expression_df, cell_proportion_df, pseudobulk_dir)
    # Then we calculate the HVG for each cell type after combat correction
    cell_expression_corrected_df = highly_variable_gene_selection(cell_expression_corrected_df, 2000)
    pseudobulk = {
        "cell_expression": cell_expression_df,
        "cell_proportion": cell_proportion_df,
        "cell_expression_corrected": cell_expression_corrected_df
    }
    save_dataframe_as_strings(cell_expression_df, pseudobulk_dir, "expression.csv")
    save_dataframe_as_strings(cell_proportion_df, pseudobulk_dir, "proportion.csv")
    return pseudobulk
