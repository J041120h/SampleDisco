import os
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

def find_sample_grouping(
    adata,
    samples,
    grouping_columns=None,
    age_bin_size=None,
    sample_column='sample'
):
    if adata is None or not grouping_columns:
        print("[INFO] No adata or grouping columns provided. Using default prefix grouping.")
        return {sample: sample[:2] for sample in samples}

    if sample_column not in adata.obs.columns:
        raise KeyError(f"[ERROR] '{sample_column}' column is missing in adata.obs. Cannot group by sample.")

    # Normalize sample column in adata to lowercase for matching
    adata.obs[sample_column] = adata.obs[sample_column].astype(str).str.lower()
    # Create lowercase version of samples for matching, but keep original mapping
    sample_map = {s.lower(): s for s in samples}
    samples_lower = list(sample_map.keys())

    # Validate that all grouping columns exist in adata.obs
    for col in grouping_columns:
        if col not in adata.obs.columns:
            raise KeyError(f"[ERROR] Grouping column '{col}' is missing in adata.obs.")

    if 'age' in grouping_columns:
        if 'age' not in adata.obs.columns:
            raise KeyError("[ERROR] 'age' column is specified but not present in adata.obs.")
        min_age = adata.obs['age'].min()

    groups = {}

    def get_column_value_for_sample(column, sample_df):
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
            if is_numeric_dtype(values):
                if len(values) == 0:
                    return f"{column}_NoData"
                return f"{column}_{values.mean():.2f}"
            else:
                if len(values) == 0:
                    return f"{column}_NoData"
                modes = values.mode()
                if len(modes) == 0:
                    return f"{column}_NoMode"
                return f"{column}_{modes.iloc[0]}"

    for sample_lower in samples_lower:
        original_sample = sample_map[sample_lower]
        mask = (adata.obs[sample_column] == sample_lower)

        if not mask.any():
            print(f"[WARNING] No cells found for sample '{original_sample}' in adata.obs['{sample_column}'].")
            print(f"           Existing unique values in '{sample_column}': {adata.obs[sample_column].unique().tolist()}")
            groups[original_sample] = "Unknown"
            continue

        sample_df = adata.obs.loc[mask, grouping_columns]
        col_values = []
        for col in grouping_columns:
            col_val = get_column_value_for_sample(col, sample_df)
            col_values.append(col_val)

        group_label = "_".join(col_values)
        groups[original_sample] = group_label

    return groups
