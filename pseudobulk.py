import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Visualization import visualization
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
    
    start_time = time.time() if verbose else None

    check_nan_and_negative_in_lists(cell_expression_df)
    
    sample_batch_map = (
        adata.obs[[sample_col, batch_col]]
        .drop_duplicates()
        .set_index(sample_col)[batch_col]
        .to_dict()
    )
    
    example_series = next(
        (s for s in cell_expression_df.iloc[0].dropna() if s is not None and isinstance(s, pd.Series) and len(s) > 0),
        None
    )
    if example_series is None:
        raise ValueError("No valid Series found in cell_expression_df.")
    gene_names = example_series.index.tolist()
    n_genes = len(gene_names)
    
    corrected_df = cell_expression_df.copy(deep=True)
    cell_types_to_drop = set()
    
    for ctype in corrected_df.index:
        if ctype in cell_types_to_drop:
            continue
        
        row_data = corrected_df.loc[ctype]
        batch_labels = []
        arrays_for_this_ctype = []
        valid_sample_ids = []
        
        for sample_id in row_data.index:
            expr_series = row_data[sample_id]
            if expr_series is None or len(expr_series) == 0 or np.allclose(expr_series.values, 0):
                continue
            expr_array = np.nan_to_num(expr_series.values, nan=0, posinf=0, neginf=0)
            arrays_for_this_ctype.append(expr_array)
            batch_labels.append(sample_batch_map.get(sample_id, "missing_batch"))
            valid_sample_ids.append(sample_id)
        
        if len(arrays_for_this_ctype) == 0:
            if verbose:
                print(f"Deleting '{ctype}' because no samples with non-zero expression were found.")
            cell_types_to_drop.add(ctype)
            continue
        
        batch_labels_array = np.array(batch_labels)
        unique_batches = pd.unique(batch_labels_array)
        batch_counts = pd.Series(batch_labels).value_counts()
        
        if len(unique_batches) < 2 or any(batch_counts < 2):
            if verbose:
                print(f"Deleting '{ctype}' due to insufficient batch diversity or small batch sizes: {batch_counts.to_dict()}.")
            cell_types_to_drop.add(ctype)
            continue
        
        expr_matrix = np.vstack(arrays_for_this_ctype).T
        var_per_gene = np.var(expr_matrix, axis=1)
        zero_var_idx = np.where(var_per_gene == 0)[0]
        if len(zero_var_idx) == expr_matrix.shape[0]:
            if verbose:
                print(f"Deleting '{ctype}' because all genes have zero variance.")
            cell_types_to_drop.add(ctype)
            continue
        
        expr_matrix_sub = np.delete(expr_matrix, zero_var_idx, axis=0)
        expr_df_t = pd.DataFrame(expr_matrix_sub, columns=valid_sample_ids)
        batch_series = pd.Series(batch_labels, index=valid_sample_ids, name='batch')
        
        if expr_df_t.isnull().values.any() and verbose:
            print(f"Warning: NaN values detected in expression data for '{ctype}' before ComBat.")
        
        corrected_df_sub = pycombat(
            expr_df_t,
            batch=batch_series,
            parametric=parametric
        )
        corrected_values_sub = corrected_df_sub.values
        
        corrected_expr_matrix = expr_matrix.copy()
        corrected_expr_matrix[zero_var_idx, :] = expr_matrix[zero_var_idx, :]
        non_zero_idx = np.delete(np.arange(n_genes), zero_var_idx)
        corrected_expr_matrix[non_zero_idx, :] = corrected_values_sub
        corrected_expr_matrix_t = corrected_expr_matrix.T
        
        for i, sample_id in enumerate(valid_sample_ids):
            corrected_df.loc[ctype, sample_id] = pd.Series(corrected_expr_matrix_t[i], index=gene_names)
        if verbose:
            print(f"ComBat correction applied for '{ctype}'.")
    
    if cell_types_to_drop:
        if verbose:
            print(f"Removing {len(cell_types_to_drop)} cell types that failed checks: {cell_types_to_drop}")
        corrected_df.drop(labels=cell_types_to_drop, inplace=True, errors='ignore')
        cell_proportion_df.drop(labels=cell_types_to_drop, inplace=True, errors='ignore')
    
    for idx in corrected_df.index:
        row = corrected_df.loc[idx]
        if any(isinstance(cell, pd.Series) and cell.isnull().any() for cell in row):
            if verbose:
                print(f"Row '{idx}' contains NaNs after ComBat correction. Removing this cell type.")
            cell_types_to_drop.add(idx)
    
    if cell_types_to_drop:
        if verbose:
            print(f"Drop '{cell_types_to_drop}'")
        corrected_df.drop(labels=cell_types_to_drop, inplace=True, errors='ignore')
        cell_proportion_df.drop(labels=cell_types_to_drop, inplace=True, errors='ignore')
    
    save_dataframe_as_strings(corrected_df, pseudobulk_dir, "corrected_expression.csv")
    
    if verbose:
        print("ComBat correction completed.")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[combat] Total runtime: {elapsed_time:.2f} seconds")
    
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
    start_time = time.time() if verbose else None

    pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
    os.makedirs(pseudobulk_dir, exist_ok=True)

    if adata.obs[batch_col].isnull().any():
        if verbose:
            print("\n\n\n\nWarning: Missing batch labels found. Filling missing values with 'Unknown'.\n\n\n\n")
        adata.obs[batch_col].fillna("Unknown", inplace=True)

    batch_counts = adata.obs[batch_col].value_counts()
    small_batches = batch_counts[batch_counts < 5]
    if not small_batches.empty and verbose:
        print(f"\n\n\n\nWarning: The following batches have fewer than 5 samples: {small_batches.to_dict()}. Consider merging these batches.\n\n\n\n")

    X_data = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X

    if np.isnan(X_data).any() or (X_data < 0).any() or np.isinf(X_data).any():
        if verbose:
            if np.isnan(X_data).any():
                print("\n\n\n\nWarning: X_data contains NaN values.\n\n\n\n")
            if (X_data < 0).any():
                print("\n\n\n\nWarning: X_data contains negative values.\n\n\n\n")
            if np.isinf(X_data).any():
                print("\n\n\n\nWarning: X_data contains Inf values.\n\n\n\n")
            print("\n\n\n\nWarning: Found NaN or Inf values in expression data. Replacing with zeros.\n\n\n\n")
        X_data = np.nan_to_num(X_data, nan=0, posinf=0, neginf=0)
    else:
        if verbose:
            print("\n\n\n\nNo NaN, negative, or Inf values found in X_data.\n\n\n\n")

    gene_variances = np.var(X_data, axis=0)
    nonzero_variance_mask = gene_variances > 0
    if not np.all(nonzero_variance_mask) and verbose:
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

            # Prepend cell type to each gene name
            modified_gene_names = [f"{ctype} - {g}" for g in gene_names]
            expr_series = pd.Series(expr_values, index=modified_gene_names)

            cell_expression_df.loc[ctype, sample] = expr_series
            cell_proportion_df.loc[ctype, sample] = proportion

    if verbose:
        print("\n\n\n\nSuccessfully computed pseudobulk data.\n\n\n\n")

    if verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\n[pseudobulk] Total runtime: {elapsed_time:.2f} seconds\n\n")

    cell_expression_corrected_df = combat_correct_cell_expressions(
        adata, cell_expression_df, cell_proportion_df, pseudobulk_dir, verbose=verbose
    )

    cell_expression_corrected_df = highly_variable_gene_selection(cell_expression_corrected_df, n_features)
    cell_expression_corrected_df, top_features = select_hvf_loess(
        cell_expression_corrected_df, n_features=n_features, frac=frac
    )

    proportion_df = cell_proportion_df.T

    pseudobulk = {
        "cell_expression": cell_expression_df,
        "cell_proportion": proportion_df,
        "cell_expression_corrected": cell_expression_corrected_df
    }

    save_dataframe_as_strings(cell_expression_df, pseudobulk_dir, "expression.csv")
    save_dataframe_as_strings(cell_proportion_df, pseudobulk_dir, "proportion.csv")

    return pseudobulk
