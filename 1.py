import anndata as ad

def print_cell_names(adata_path: str):
    """
    Load an AnnData object, show first 10 cell names before/after removing tissue prefixes,
    and overwrite the same file with cleaned cell names.
    """
    print(f"📂 Loading AnnData from: {adata_path}")
    adata = ad.read_h5ad(adata_path)
    print(f"✅ Loaded AnnData with {adata.n_obs} cells × {adata.n_vars} genes")

    # --- Before renaming ---
    print("\n🧬 First 10 cell/sample names (before):")
    for name in adata.obs_names[:10]:
        print(f"  - {name}")

    # --- Perform renaming ---
    old_names = adata.obs_names.to_list()
    new_names = []
    for name in old_names:
        i = name.find("ENC")
        if i >= 0:
            new_names.append(name[i:])
        else:
            new_names.append(name)  # leave unchanged if no "ENC"

    # --- After renaming ---
    print("\n🧬 First 10 cell/sample names (after):")
    for name in new_names[:10]:
        print(f"  - {name}")

    # --- Apply and save ---
    adata.obs_names = new_names
    adata.obs_names_make_unique()
    adata.write_h5ad(adata_path)
    print(f"\n💾 Overwritten cleaned AnnData file: {adata_path}")


if __name__ == "__main__":
    # print_cell_names('/dcs07/hongkai/data/harry/result/Benchmark/multiomics/preprocess/atac_rna_integrated.h5ad')
