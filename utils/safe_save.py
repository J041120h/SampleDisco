import os
import pandas as pd
import scanpy as sc
import numpy as np


def _convert_value_to_string(val):
    if pd.isna(val):
        return 'Unknown'
    elif isinstance(val, (bool, np.bool_)):
        return 'True' if val else 'False'
    elif isinstance(val, (int, np.integer, float, np.floating)):
        return str(val).replace('.0', '') if float(val).is_integer() else str(val)
    elif isinstance(val, str):
        return val if val.strip() else 'Unknown'
    else:
        return str(val)


def _clean_na_strings(series):
    return series.replace(['None', 'nan', 'NaN', 'NULL', '', '<NA>'], 'Unknown')


def _make_categories_unique(categories):
    if len(categories) == len(set(categories)):
        return categories
    
    seen = {}
    unique_categories = []
    for cat in categories:
        if cat in seen:
            seen[cat] += 1
            unique_categories.append(f"{cat}_{seen[cat]}")
        else:
            seen[cat] = 0
            unique_categories.append(cat)
    return unique_categories


def clean_obs_for_saving(adata):
    obs_copy = adata.obs.copy()
    
    for col in obs_copy.columns:
        col_data = obs_copy[col]
        
        if pd.api.types.is_categorical_dtype(col_data):
            new_categories = [_convert_value_to_string(cat) for cat in col_data.cat.categories]
            new_categories = _make_categories_unique(new_categories)
            mapping = dict(zip(col_data.cat.categories, new_categories))
            new_values = [mapping.get(val, 'Unknown') if not pd.isna(val) else 'Unknown' 
                         for val in col_data.to_numpy()]
            col_data = pd.Categorical(new_values, categories=new_categories)

        elif col_data.dtype == 'object':
            new_values = [_convert_value_to_string(val) for val in col_data]
            col_data = _clean_na_strings(pd.Series(new_values, index=col_data.index))
            col_data = pd.Categorical(col_data)
        
        elif col_data.dtype in ['bool', np.bool_]:
            new_values = [_convert_value_to_string(val) for val in col_data]
            col_data = pd.Categorical(new_values)
        
        elif pd.api.types.is_numeric_dtype(col_data):
            n_unique = col_data.nunique()
            if 0 < n_unique < 20:
                col_data = col_data.fillna(-999).astype(str)
                col_data = col_data.replace(['-999', '-999.0'], 'Unknown')
                col_data = pd.Categorical(col_data)
            elif col_data.isna().any():
                col_data = col_data.fillna(-1)
        
        else:
            col_data = _clean_na_strings(col_data.astype(str).fillna('Unknown'))
            col_data = pd.Categorical(col_data)
        
        obs_copy[col] = col_data
    
    adata.obs = obs_copy
    return adata


def ensure_cpu_arrays(adata):
    if hasattr(adata.X, 'get'):
        adata.X = adata.X.get()
    
    for attr_name in ['layers', 'obsm', 'varm', 'obsp', 'varp']:
        container = getattr(adata, attr_name)
        for key in list(container.keys()):
            if hasattr(container[key], 'get'):
                container[key] = container[key].get()
    
    return adata


def safe_h5ad_write(adata, filepath):
    try:
        adata_copy = adata.copy()
        adata_copy = ensure_cpu_arrays(adata_copy)
        adata_copy = clean_obs_for_saving(adata_copy)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        sc.write(filepath, adata_copy)
    except Exception as e:
        print(f"Error saving H5AD file: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"adata.obs shape: {adata.obs.shape}")
        print("Non-string categories found:")
        for col in adata.obs.columns:
            if pd.api.types.is_categorical_dtype(adata.obs[col]):
                cats = adata.obs[col].cat.categories
                non_string = [c for c in cats if not isinstance(c, str)]
                if non_string:
                    print(f"  {col}: {len(non_string)} non-string categories, examples: {non_string[:3]}")
        raise