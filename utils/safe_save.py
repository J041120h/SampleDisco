import os
import pandas as pd
import scanpy as sc
import numpy as np

def clean_obs_for_saving(adata, verbose=True):
    """
    Clean adata.obs to prevent string conversion errors during H5AD saving.
    Simplified version that handles all common data types efficiently.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to clean
    verbose : bool
        Whether to print cleaning operations
        
    Returns
    -------
    adata : AnnData
        Cleaned AnnData object
    """
    if verbose:
        print("[clean_obs_for_saving] Cleaning observation metadata for H5AD compatibility...")
    
    # Create a copy of obs to avoid modifying during iteration
    obs_copy = adata.obs.copy()
    
    for col in obs_copy.columns:
        col_data = obs_copy[col].copy()
        
        # Handle categorical columns
        if pd.api.types.is_categorical_dtype(col_data):
            # Convert categories to strings
            new_categories = []
            for cat in col_data.cat.categories:
                if pd.isna(cat):
                    new_cat = 'Unknown'
                elif isinstance(cat, bool) or isinstance(cat, np.bool_):
                    new_cat = 'True' if cat else 'False'
                elif isinstance(cat, (int, np.integer, float, np.floating)):
                    new_cat = str(cat).replace('.0', '') if float(cat).is_integer() else str(cat)
                elif isinstance(cat, str):
                    new_cat = cat if cat.strip() else 'Unknown'
                else:
                    new_cat = str(cat)
                new_categories.append(new_cat)
            
            # Handle duplicates if any
            if len(new_categories) != len(set(new_categories)):
                seen = {}
                final_categories = []
                for cat in new_categories:
                    if cat in seen:
                        seen[cat] += 1
                        final_categories.append(f"{cat}_{seen[cat]}")
                    else:
                        seen[cat] = 0
                        final_categories.append(cat)
                new_categories = final_categories
            
            # Map values to new categories
            mapping = dict(zip(col_data.cat.categories, new_categories))
            col_values = col_data.to_numpy()
            new_values = [mapping.get(val, 'Unknown') if not pd.isna(val) else 'Unknown' 
                         for val in col_values]
            
            col_data = pd.Categorical(new_values, categories=new_categories)
        
        # Handle object columns
        elif col_data.dtype == 'object':
            new_values = []
            for val in col_data:
                if pd.isna(val):
                    new_val = 'Unknown'
                elif isinstance(val, bool):
                    new_val = 'True' if val else 'False'
                elif isinstance(val, (int, float)):
                    new_val = str(val).replace('.0', '') if isinstance(val, float) and val.is_integer() else str(val)
                elif isinstance(val, str):
                    new_val = val if val.strip() else 'Unknown'
                else:
                    new_val = str(val)
                new_values.append(new_val)
            
            col_data = pd.Series(new_values, index=col_data.index)
            col_data = col_data.replace(['None', 'nan', 'NaN', 'NULL', '', '<NA>'], 'Unknown')
            col_data = pd.Categorical(col_data)
        
        # Handle boolean columns
        elif col_data.dtype in ['bool', np.bool_]:
            new_values = ['True' if val else 'False' if not pd.isna(val) else 'Unknown' 
                         for val in col_data]
            col_data = pd.Categorical(new_values)
        
        # Handle numeric columns that might be categorical
        elif pd.api.types.is_numeric_dtype(col_data):
            n_unique = col_data.nunique()
            if n_unique < 20 and n_unique > 0:  # Likely categorical
                if col_data.isna().any():
                    col_data = col_data.fillna(-999)
                col_data = col_data.astype(str).replace(['-999', '-999.0'], 'Unknown')
                col_data = pd.Categorical(col_data)
            else:
                # Keep as numeric but handle NaN
                if col_data.isna().any():
                    col_data = col_data.fillna(-1)
        
        else:
            # Any other dtype - convert to string categorical
            col_data = col_data.astype(str).fillna('Unknown')
            col_data = col_data.replace(['None', 'nan', 'NaN', 'NULL', '', '<NA>'], 'Unknown')
            col_data = pd.Categorical(col_data)
        
        # Update the column
        obs_copy[col] = col_data
    
    # Replace the entire obs dataframe
    adata.obs = obs_copy
    
    if verbose:
        print("[clean_obs_for_saving] Cleaning completed successfully")
        # Quick summary of categorical columns
        cat_cols = [col for col in adata.obs.columns if pd.api.types.is_categorical_dtype(adata.obs[col])]
        print(f"[clean_obs_for_saving] {len(cat_cols)} categorical columns processed")
    
    return adata


def ensure_cpu_arrays(adata):
    """
    Ensure all arrays in AnnData object are on CPU (not GPU).
    This prevents "Implicit conversion to NumPy array" errors.
    """
    # Convert main matrix
    if hasattr(adata.X, 'get'):
        adata.X = adata.X.get()
    
    # Convert layers
    if hasattr(adata, 'layers'):
        for key in list(adata.layers.keys()):
            if hasattr(adata.layers[key], 'get'):
                adata.layers[key] = adata.layers[key].get()
    
    # Convert obsm (embeddings)
    if hasattr(adata, 'obsm'):
        for key in list(adata.obsm.keys()):
            if hasattr(adata.obsm[key], 'get'):
                adata.obsm[key] = adata.obsm[key].get()
    
    # Convert varm
    if hasattr(adata, 'varm'):
        for key in list(adata.varm.keys()):
            if hasattr(adata.varm[key], 'get'):
                adata.varm[key] = adata.varm[key].get()
    
    # Convert obsp (pairwise arrays)
    if hasattr(adata, 'obsp'):
        for key in list(adata.obsp.keys()):
            if hasattr(adata.obsp[key], 'get'):
                adata.obsp[key] = adata.obsp[key].get()
    
    # Convert varp
    if hasattr(adata, 'varp'):
        for key in list(adata.varp.keys()):
            if hasattr(adata.varp[key], 'get'):
                adata.varp[key] = adata.varp[key].get()
    
    return adata


def safe_h5ad_write(adata, filepath, verbose=True):
    """
    Safely write AnnData object to H5AD format with proper error handling.
    Simplified version without extensive debugging.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to save
    filepath : str
        Path to save the file
    verbose : bool
        Whether to print progress messages
    """
    try:
        if verbose:
            print(f"[safe_h5ad_write] Preparing to save to: {filepath}")
        
        # Create a copy to avoid modifying the original
        adata_copy = adata.copy()
        
        # Ensure CPU arrays
        adata_copy = ensure_cpu_arrays(adata_copy)
        
        # Clean observation metadata
        adata_copy = clean_obs_for_saving(adata_copy, verbose=verbose)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write the file
        if verbose:
            print(f"[safe_h5ad_write] Writing H5AD file...")
        
        sc.write(filepath, adata_copy)
        
        if verbose:
            print(f"[safe_h5ad_write] Successfully saved to: {filepath}")
            
    except Exception as e:
        if verbose:
            print(f"[safe_h5ad_write] Error saving H5AD file: {str(e)}")
            
            # Provide basic diagnostic information
            print("\n=== DIAGNOSTIC INFORMATION ===")
            print(f"Error type: {type(e).__name__}")
            print(f"adata.obs shape: {adata.obs.shape}")
            
            # Check for non-string categories
            print("\nChecking for non-string categories:")
            for col in adata.obs.columns:
                if pd.api.types.is_categorical_dtype(adata.obs[col]):
                    cats = adata.obs[col].cat.categories
                    non_string = [c for c in cats if not isinstance(c, str)]
                    if non_string:
                        print(f"  - {col}: Found {len(non_string)} non-string categories")
                        print(f"    Examples: {non_string[:3]}")
        
        raise e
