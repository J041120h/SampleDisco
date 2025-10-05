import anndata as ad

def compare_celltypes(h5ad_path1, h5ad_path2, celltype_col='cell_type'):
    """
    Compare if all matching cells belong to the same cell type across two h5ad files.

    Parameters
    ----------
    h5ad_path1 : str
        Path to first .h5ad file
    h5ad_path2 : str
        Path to second .h5ad file
    celltype_col : str
        Column name in .obs containing cell type annotations

    Returns
    -------
    bool
        True if all overlapping cells have the same cell type, False otherwise
    """
    adata1 = ad.read_h5ad(h5ad_path1)
    adata2 = ad.read_h5ad(h5ad_path2)

    # Intersect cell IDs
    common_cells = adata1.obs_names.intersection(adata2.obs_names)
    if len(common_cells) == 0:
        print("No overlapping cells found.")
        return False

    # Extract cell types
    ct1 = adata1.obs.loc[common_cells, celltype_col].astype(str)
    ct2 = adata2.obs.loc[common_cells, celltype_col].astype(str)

    # Compare equality
    same = (ct1 == ct2)
    all_match = same.all()

    print(f"Total overlapping cells: {len(common_cells)}")
    print(f"Matching cell types: {same.sum()}")
    print(f"Mismatched cell types: {(~same).sum()}")

    if not all_match:
        mismatches = common_cells[~same]
        print("Example mismatches:")
        print(pd.DataFrame({
            "cell_id": mismatches,
            "file1_celltype": ct1.loc[mismatches].values,
            "file2_celltype": ct2.loc[mismatches].values
        }).head())

    return all_match

if __name__ == "__main__":
    compare_celltypes("/dcs07/hongkai/data/harry/result/Benchmark/covid_25_sample/rna/preprocess/adata_sample_test.h5ad", "/dcs07/hongkai/data/harry/result/Benchmark/covid_25_sample/rna/preprocess/adata_sample.h5ad", celltype_col="celltype")
