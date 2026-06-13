"""Build a per-cell-type RNA pseudobulk (samples x 'celltype - gene') from the
diemb RNA cells, using the MERGED cell_type names so feature names match the
diemb trajectory DEG. Needed by the figure5 concordance (1.py) gene-GAM step.
"""
import sys

sys.path.insert(0, "/users/hjiang/GenoDistance/code")
import scanpy as sc
from sampledisco.sample_trajectory.trajectory_diff_gene import _build_sample_pseudobulk

RNA = ("/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics/"
       "preprocess/adata_rna_preprocessed.h5ad")
OUT = ("/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics/"
       "sample_embedding_tune-on-RNA/cluster_severity_deg/rna_pseudobulk_celltype.h5ad")

print("loading", RNA, flush=True)
a = sc.read_h5ad(RNA)
print("building per-celltype pseudobulk (merged names, all genes)...", flush=True)
pb = _build_sample_pseudobulk(a, sample_col="sample", celltype_col="cell_type",
                              batch_col=None, n_features_per_celltype=None,
                              verbose=True)
print("pseudobulk shape:", pb.shape, flush=True)
print("var[:3]:", list(pb.var_names[:3]), flush=True)
pb.write(OUT)
print("PB_DONE wrote", OUT, flush=True)
