import os
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_categorical_dtype


def find_sample_grouping(
    adata,
    samples,
    grouping_columns=None,
    age_bin_size=None
):
    """
    Returns a dictionary that maps each sample to a group label based on
    the requested grouping columns in `adata.obs`.
    
    Parameters
    ----------
    adata : anndata.AnnData or None
        The AnnData object containing per-cell metadata in adata.obs.
    samples : list of str
        The samples of interest (must match entries in `adata.obs['sample']` if adata is provided).
    grouping_columns : list of str, optional
        Which columns in `adata.obs` to use for grouping.
        If None or if adata is None, fallback to first two letters of sample name.
    age_bin_size : int or None
        If one of the grouping columns is 'age', this integer determines
        the bin width. For example, if age_bin_size = 10, then ages will be
        grouped in intervals of 10 years, starting from the min age.
        
    Returns
    -------
    dict
        A dictionary mapping {sample: group_label}.
    """

    # If adata is None or grouping_columns not provided, fallback to first-two-letter grouping
    if adata is None or not grouping_columns:
        return {sample: sample[:2] for sample in samples}

    # We will need the 'sample' column in adata.obs
    if 'sample' not in adata.obs.columns:
        raise KeyError("'sample' column is missing in adata.obs. Cannot build groups by sample.")

    # If 'age' is one of the grouping columns, precompute the minimum age for binning
    if 'age' in grouping_columns:
        if 'age' not in adata.obs.columns:
            raise KeyError("'age' column is specified but not present in adata.obs.")
        min_age = adata.obs['age'].min()

    groups = {}

    def get_column_value_for_sample(column, sample_df):
        # Extract the data for this column (all cells that match the sample)
        values = sample_df[column].dropna()

        if column == 'age':
            if age_bin_size is None:
                if len(values) == 0:
                    return "age_NoData"
                avg_age = values.mean()
                return f"age_{int(avg_age)}"
            else:
                if len(values) == 0:
                    return "ageBin_NoData"
                avg_age = values.mean()
                bin_index = int((avg_age - min_age) // age_bin_size)
                return f"ageBin_{bin_index}"
        else:
            # Check if numeric or categorical
            if is_numeric_dtype(values):
                if len(values) == 0:
                    # No numeric values
                    return f"{column}_NoData"
                return f"{column}_{values.mean():.2f}"
            else:
                # Categorical/string -> use mode
                if len(values) == 0:
                    return f"{column}_NoData"  # or some other fallback
                modes = values.mode()  # could be multiple, just take first
                if len(modes) == 0:
                    return f"{column}_NoMode"
                return f"{column}_{modes.iloc[0]}"

    # Build the mapping for each sample
    for sample in samples:
        # Extract the sub-dataframe for this sample
        mask = (adata.obs['sample'] == sample)
        if not mask.any():
            # If no cells for this sample in adata, you can skip or define a fallback:
            groups[sample] = "Unknown"
            continue

        sample_df = adata.obs.loc[mask, grouping_columns]

        # Compute grouping label as combination of each column's value
        col_values = []
        for col in grouping_columns:
            col_val = get_column_value_for_sample(col, sample_df)
            col_values.append(col_val)

        # Join them into a single group label
        group_label = "_".join(col_values)
        groups[sample] = group_label

    return groups