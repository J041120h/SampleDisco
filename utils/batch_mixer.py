def get_mixed_batch_column(adata, batch_column=None):
    """
    Helper function to handle batch column for functions that only take one batch column.
    
    Parameters:
    -----------
    adata : AnnData
        The AnnData object containing the data
    batch_column : str or None
        The original batch column name. If None, returns "modality"
    
    Returns:
    --------
    adata : AnnData
        The (potentially modified) AnnData object with new mixed column if applicable
    str
        The name of the batch column to use (either "modality" or the mixed column name)
    """
    
    # If no batch column specified, just return "modality"
    if batch_column is None:
        print("No batch column specified, using 'modality' as the batch column.")
        return adata, "modality"
    
    # Create new mixed batch column name
    mixed_column_name = f"{batch_column}_modality"
    
    # Create the mixed batch column by combining original batch with modality
    # Assuming both columns exist in adata.obs
    if batch_column in adata.obs.columns and 'modality' in adata.obs.columns:
        # Create combined categories with "_" separator and modality as suffix
        adata.obs[mixed_column_name] = (
            adata.obs[batch_column].astype(str) + "_" + 
            adata.obs['modality'].astype(str)
        )
        print(f"Created mixed batch column '{mixed_column_name}' by combining '{batch_column}' and 'modality'.")
    else:
        # Handle missing columns gracefully
        if batch_column not in adata.obs.columns:
            print(f"Warning: batch column '{batch_column}' not found in adata.obs")
            return adata, "modality"
        if 'modality' not in adata.obs.columns:
            print(f"Warning: 'modality' column not found in adata.obs")
            return adata, batch_column
    
    return adata, mixed_column_name