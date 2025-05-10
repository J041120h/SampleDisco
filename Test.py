import anndata as ad

def print_sample_column(h5ad_path: str, sample_col: str = "Sample"):
    """
    Read an h5ad file and print the sample column for each cell.
    
    Parameters:
    -----------
    h5ad_path : str
        Path to the h5ad file
    sample_col : str, optional
        Name of the sample column, default is "Sample"
    """
    try:
        # Load the h5ad file
        print(f"Loading h5ad file from: {h5ad_path}")
        adata = ad.read_h5ad(h5ad_path)
        
        # Check if the sample column exists in the observation annotations
        if sample_col not in adata.obs.columns:
            raise ValueError(f"Column '{sample_col}' not found in the observations. "
                           f"Available columns: {list(adata.obs.columns)}")
        
        # Print the number of cells
        print(f"Total number of cells: {adata.n_obs}")
        
        # Print the sample for each cell
        print(f"\nSample information for each cell (showing first 20 cells):")
        for i, (idx, sample) in enumerate(zip(adata.obs.index, adata.obs[sample_col])):
            if i < 20:  # Limit output to first 20 cells to avooid excessive printing
                print(f"Cell {idx}: {sample}")
            else:
                break
        
        # Print a summary of unique samples and their counts
        sample_counts = adata.obs[sample_col].value_counts()
        print(f"\nSummary of samples (total: {len(sample_counts)} unique samples):")
        for sample, count in sample_counts.items():
            print(f"Sample '{sample}': {count} cells")
            
    except Exception as e:
        print(f"Error processing h5ad file: {str(e)}")

# Example usage
if __name__ == "__main__":
    h5ad_path = "/users/hjiang/SingleCell/count_data.h5ad"
    print_sample_column(h5ad_path)