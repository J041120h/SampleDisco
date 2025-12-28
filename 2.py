from pathlib import Path
import anndata as ad

def append_modality_to_sample(h5ad_path: str, sample_col: str = "sample", modality_col: str = "modality"):
    """
    Load .h5ad → sample := sample + "_" + modality → overwrite the same file.

    Example result: MA9_heart_ATAC
    """
    h5ad_path = Path(h5ad_path)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"File not found: {h5ad_path}")

    print(f"🔧 Loading: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)

    # check columns
    if sample_col not in adata.obs:
        raise KeyError(f"'{sample_col}' column not in adata.obs")
    if modality_col not in adata.obs:
        raise KeyError(f"'{modality_col}' column not in adata.obs")

    # build new sample values
    new_sample = (
        adata.obs[sample_col].astype(str).str.replace("^PMID_\\d+_", "", regex=True) # optional cleanup
        + "_" +
        adata.obs[modality_col].astype(str)
    )

    adata.obs[sample_col] = new_sample.astype("category")

    print("📌 Updated sample column example:")
    print(adata.obs[sample_col].head())

    # overwrite file
    adata.write_h5ad(h5ad_path)
    print(f"💾 Saved & overwritten: {h5ad_path}")

append_modality_to_sample(
    "/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/rna/preprocess/adata_cell.h5ad"
)