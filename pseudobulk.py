import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Visualization import visualization_harmony
from combat.pycombat import pycombat


def contains_nan_in_lists(df: pd.DataFrame) -> bool:
    """
    Check if any NumPy arrays inside a DataFrame contain NaN values.
    """
    for row in df.itertuples(index=False):
        for array in row:
            if isinstance(array, np.ndarray) and np.isnan(array).any():
                return True
    return False


def combat_correct_cell_expressions(
    adata: sc.AnnData,
    cell_expression_df: pd.DataFrame,
    batch_col: str = 'batch',
    sample_col: str = 'sample'
) -> pd.DataFrame:
    """
    Apply ComBat batch correction to each cell type across samples.

    Parameters:
    - adata: AnnData object containing batch information.
    - cell_expression_df: DataFrame (rows: cell types, columns: samples, values: 1D gene expression arrays).
    - batch_col: Column name for batch labels in adata.obs.
    - sample_col: Column name for sample identifiers in adata.obs.

    Returns:
    - corrected_df: Batch-corrected DataFrame (same structure as input).
    """
    # Map sample IDs to batch labels
    sample_batch_map = adata.obs[[sample_col, batch_col]].drop_duplicates().set_index(sample_col)[batch_col].to_dict()

    # Determine number of genes from first valid array
    example_array = next((arr for arr in cell_expression_df.iloc[0].dropna() if arr is not None and len(arr) > 0), None)
    if example_array is None:
        raise ValueError("No valid arrays found in cell_expression_df.")
    n_genes = len(example_array)

    corrected_df = cell_expression_df.copy(deep=True)

    for ctype in corrected_df.index:
        row_data = corrected_df.loc[ctype]
        batch_labels, arrays_for_this_ctype = [], []

        for sample_id in row_data.index:
            expr_array = row_data[sample_id] if row_data[sample_id] is not None else np.zeros(n_genes)
            arrays_for_this_ctype.append(expr_array)
            batch_labels.append(sample_batch_map.get(sample_id, "missing_batch"))

        unique_batches = pd.unique(batch_labels)
        if len(batch_labels) < 2 or len(unique_batches) < 2:
            print(f"Skipping ComBat for '{ctype}' due to insufficient batch diversity.")
            continue

        expr_matrix = np.vstack(arrays_for_this_ctype).T  # Transpose for pycombat
        expr_df_t = pd.DataFrame(expr_matrix, columns=row_data.index)
        batch_series = pd.Series(batch_labels, index=row_data.index, name='batch')

        corrected_df_t = pycombat(expr_df_t, batch=batch_series).values.T  # Transpose back
        for i, sample_id in enumerate(row_data.index):
            corrected_df.loc[ctype, sample_id] = corrected_df_t[i]

        print(f"ComBat correction applied for '{ctype}'.")

    if contains_nan_in_lists(corrected_df):
        print("Warning: NaN values detected. Returning uncorrected data.")
        return cell_expression_df

    print("ComBat correction completed.")
    return corrected_df


def compute_pseudobulk_dataframes(
    adata: sc.AnnData,
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './'
):
    """
    Compute pseudobulk data: average gene expression and cell type proportions per sample.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - batch_col: Column name for batch information.
    - sample_col: Column for sample identifiers in adata.obs.
    - celltype_col: Column for cell type assignments in adata.obs.
    - output_dir: Directory to save results.

    Returns:
    - pseudobulk: Dictionary containing:
        - "cell_expression": DataFrame with average gene expression per (cell type, sample).
        - "cell_proportion": DataFrame with proportions of each cell type per sample.
        - "cell_expression_corrected": ComBat-corrected gene expression DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)

    samples, cell_types, gene_names = adata.obs[sample_col].unique(), adata.obs[celltype_col].unique(), adata.var_names
    X_data = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X

    # Initialize DataFrames
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

    print("Successfully computed pseudobulk data.")

    cell_expression_corrected_df = combat_correct_cell_expressions(adata, cell_expression_df)
    pseudobulk = {
        "cell_expression": cell_expression_df,
        "cell_proportion": cell_proportion_df,
        "cell_expression_corrected": cell_expression_corrected_df
    }

    return pseudobulk
