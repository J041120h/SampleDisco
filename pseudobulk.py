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
    Checks if any of the lists (numpy arrays) inside the DataFrame contain NaN values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each cell contains a 1D numpy array.

    Returns
    -------
    bool
        True if any of the arrays contain NaN, False otherwise.
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
    Applies ComBat batch correction to each cell type across samples for a
    DataFrame where:
      - Rows = cell types
      - Columns = sample IDs
      - Each cell is a 1D array of shape (n_genes,).

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object that includes `batch_col` and `sample_col` in `adata.obs`
        for each cell. We'll look up the batch for each sample from here.
    cell_expression_df : DataFrame
        Rows = cell types, columns = samples.
        Each cell in the table is a 1D numpy array of length n_genes.
    batch_col : str
        The name of the column in `adata.obs` that indicates batch.
    sample_col : str
        The name of the column in `adata.obs` that indicates sample ID.

    Returns
    -------
    corrected_df : DataFrame
        Same shape as `cell_expression_df` (rows = cell types, columns = samples),
        but each cell's array is now ComBat-corrected.
    """

    sample_batch_map = (adata.obs[[sample_col, batch_col]]
                        .drop_duplicates()
                        .set_index(sample_col)[batch_col]
                        .to_dict())

    # Extract the number of genes from the first valid array
    example_row = cell_expression_df.iloc[0].dropna()
    example_array = next((arr for arr in example_row if arr is not None and len(arr) > 0), None)
    if example_array is None:
        raise ValueError("Unable to find a non-empty array in cell_expression_df.")
    n_genes = len(example_array)

    # Make a copy to store corrected values
    corrected_df = cell_expression_df.copy(deep=True)

    # ---------------------------
    # 2. Loop over cell types
    # ---------------------------
    for ctype in corrected_df.index:  # each row
        # Extract the row as a Series: index=sample IDs, values=arrays of shape (n_genes,)
        row_data = corrected_df.loc[ctype]

        # Build an (n_samples x n_genes) matrix by stacking the arrays in row order
        # Also collect the batch labels in the same order
        arrays_for_this_ctype = []
        batch_labels = []
        samples_in_row = row_data.index

        for sample_id in samples_in_row:
            expr_array = row_data[sample_id]
            # It might be None or an empty array if data is missing
            if expr_array is None or len(expr_array) == 0:
                expr_array = np.zeros(n_genes, dtype=float)
                print(f"\nWarning: Missing data for cell type '{ctype}' in sample '{sample_id}'.\n")
            
            arrays_for_this_ctype.append(expr_array)

            # Lookup the batch for this sample. If not found, use a placeholder or skip.
            batch = sample_batch_map.get(sample_id, "missing_batch")
            batch_labels.append(batch)

        # Convert to shape (n_samples, n_genes)
        expr_matrix = np.vstack(arrays_for_this_ctype)  # shape: (n_samples, n_genes)

        # If there's only one sample or only one unique batch, ComBat doesn't do anything meaningful.
        # We can skip or at least check for that here:
        unique_batches = pd.unique(batch_labels)
        if len(batch_labels) < 2 or len(unique_batches) < 2:
            # Skip ComBat, leave row as is
            print(f"Skipping ComBat correction for cell type '{ctype}' due to insufficient samples or batches.")
            continue

        # ---------------------------
        # 3. Apply ComBat
        # ---------------------------
        # pycombat expects shape (n_genes x n_samples), so transpose
        expr_matrix_t = expr_matrix.T  # shape: (n_genes, n_samples)
        expr_df_t = pd.DataFrame(
            expr_matrix_t,
            columns=samples_in_row,      # sample IDs
            index=range(n_genes)         # or gene names if available
        )

        # Convert the batch labels into a Series, indexed by sample:
        batch_series = pd.Series(batch_labels, index=samples_in_row, name='batch')

        # Apply ComBat
        corrected_df_t = pycombat(expr_df_t, batch=batch_series)

        # Convert the DataFrame back to a NumPy array if needed
        corrected_matrix_t = corrected_df_t.values

        # ---------------------------
        # 4. Write corrected arrays back to corrected_df
        # ---------------------------
        for i, sample_id in enumerate(samples_in_row):
            corrected_df.loc[ctype, sample_id] = corrected_matrix_t[i]

        print(f"ComBat correction complete for cell type '{ctype}'.")

    # ---------------------------
    # 5. Check for NaNs
    # ---------------------------
    if contains_nan_in_lists(corrected_df):
        print("\n\n\n\nWarning: NaN values detected in corrected data. Returning original cell_expression_df.\n\n\n\n")
        return cell_expression_df  # Return the uncorrected version

    print("ComBat correction completed successfully without NaNs.")
    return corrected_df


def compute_pseudobulk_dataframes(
    adata: sc.AnnData,
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './'
):
    """
    Creates two DataFrames:

    1) `cell_expression_df` with rows = cell types, columns = samples.
       Each cell is a vector of average gene expressions for that cell type in that sample.
    2) `cell_proportion_df` with rows = cell types, columns = samples.
       Each cell is a single float for proportion of that cell type in that sample.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cells x genes).
        Must have `sample_col` and `celltype_col` in .obs.
    sample_col : str
        Column in `adata.obs` indicating sample ID.
    celltype_col : str
        Column in `adata.obs` indicating cell type.
    output_dir : str
        Directory where the output might be saved. 
        (Optional in this snippet; you can omit if not saving.)

    Returns
    -------
    cell_expression_df : DataFrame
        Rows = cell types, columns = samples, each element is a 1D numpy array of shape (n_genes,).
    cell_proportion_df : DataFrame
        Rows = cell types, columns = samples, each element is a float (the proportion).
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract relevant columns
    samples = adata.obs[sample_col].unique()
    cell_types = adata.obs[celltype_col].unique()
    gene_names = adata.var_names

    # Convert sparse matrix to dense if needed
    X_data = adata.X
    if not isinstance(X_data, np.ndarray):
        X_data = X_data.toarray()

    # Create empty DataFrames
    # Each cell in cell_expression_df is initially set to None (or np.nan).
    # We'll store arrays in them, so we use dtype=object for the expression DF.
    cell_expression_df = pd.DataFrame(
        index=cell_types,
        columns=samples,
        dtype=object
    )
    cell_proportion_df = pd.DataFrame(
        index=cell_types,
        columns=samples,
        dtype=float
    )

    for sample in samples:
        # Mask: all cells from this sample
        sample_mask = (adata.obs[sample_col] == sample)
        total_cells = np.sum(sample_mask)

        for ctype in cell_types:
            # Further subset to this cell type
            ctype_mask = sample_mask & (adata.obs[celltype_col] == ctype)
            num_cells = np.sum(ctype_mask)

            if num_cells > 0:
                # Average expression across genes for the subset
                expr_values = X_data[ctype_mask, :].mean(axis=0)
                proportion = num_cells / total_cells
            else:
                # No cells of this (sample, cell_type) combination
                expr_values = np.zeros(len(gene_names))
                proportion = 0.0

            # Store results in the DataFrames
            cell_expression_df.loc[ctype, sample] = expr_values
            cell_proportion_df.loc[ctype, sample] = proportion
    
    print("Successfuly computed pseudobulk dataframes.")

    cell_expression_corrected_df = combat_correct_cell_expressions(adata, cell_expression_df)
    # Create a dictionary to store DataFrames
    pseudobulk = {
        "cell_expression": cell_expression_df,
        "cell_proportion": cell_proportion_df,
        "cell_expression_corrected": cell_expression_corrected_df
    }

    return pseudobulk
