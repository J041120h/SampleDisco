"""Step 0: generate adata_rna_preprocessed.h5ad for multi_omics_unpaired_diemb.

unpaired_diemb was built with the old pipeline; it has no per-modality RNA
expression h5ad. This produces one from the cached glue-rna-emb.h5ad via the
new preparation.multi_omics_merge.preprocess_rna_for_downstream — QC + log1p,
raw counts kept in layers['counts']. Used as CellTypist input + RAISIN/DGE
input. hongkai env (no cuml needed).
"""
import sys
sys.path.insert(0, "/users/hjiang/GenoDistance/code")
from preparation.multi_omics_merge import preprocess_rna_for_downstream

BASE = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics"
preprocess_rna_for_downstream(
    rna_emb_path=f"{BASE}/integration/glue/glue-rna-emb.h5ad",
    output_path=f"{BASE}/preprocess/adata_rna_preprocessed.h5ad",
    sample_column="sample",
    sample_meta_path="/dcl01/hongkai/data/data/hjiang/Data/merged_rna_atac_metadata.csv",
    verbose=True,
)
print("STEP0_DONE")
