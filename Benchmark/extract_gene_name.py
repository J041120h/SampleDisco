import os
import scanpy as sc

def extract_unique_genes(h5ad_path: str, output_dir: str, output_filename: str = "unique_genes.txt"):
    """
    Extracts unique gene names from a h5ad file where .var_names are formatted as 'celltype - gene_name'.
    Saves them into a text file.

    Parameters
    ----------
    h5ad_path : str
        Path to the input .h5ad file.
    output_dir : str
        Directory where the output file will be saved.
    output_filename : str, optional
        Name of the output text file (default: "unique_genes.txt").
    """
    # Load the h5ad file
    print(f"Loading AnnData from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    # Extract raw gene names (after the dash)
    raw_genes = [name.split(" - ", 1)[-1] for name in adata.var_names]

    # Get unique ones
    unique_genes = sorted(set(raw_genes))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to file
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        for gene in unique_genes:
            f.write(gene + "\n")

    print(f"✅ Saved {len(unique_genes)} unique genes to {output_path}")


if __name__ == "__main__":
    # Example usage
    extract_unique_genes("/dcl01/hongkai/data/data/hjiang/Data/multi_omics_testing/rna/pseudobulk/pseudobulk_sample.h5ad", "/dcl01/hongkai/data/data/hjiang/Data/multi_omics_testing")