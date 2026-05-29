"""Step 1: CellTypist annotation of diemb RNA cells (Claude conda env).

CellTypist's hongkai-env import pulls in cuml→cudf which fails on this node's
driver; the `Claude` env (rapids 24.12) imports cleanly, so this script is run
with /users/hjiang/.conda/envs/Claude/bin/python.

Input is adata_rna_preprocessed.h5ad whose X is already normalize_total(1e4)+
log1p — exactly CellTypist's expected input. We emit per-cell predicted labels
(no over-clustering); majority-vote per existing Leiden cluster happens in the
hongkai orchestration so the existing 17-cluster Leiden structure is preserved
and only NAMED.
"""
import sys
import pandas as pd
import scanpy as sc
import celltypist

BASE = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics"
RNA  = f"{BASE}/preprocess/adata_rna_preprocessed.h5ad"
OUT  = f"{BASE}/sample_embedding_tune-on-RNA/cluster_severity_deg/celltypist_per_cell.csv"
MODEL = "Immune_All_High.pkl"

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)

print(f"[celltypist] reading {RNA}", flush=True)
adata = sc.read_h5ad(RNA)
print(f"[celltypist] shape={adata.shape}  X.max={float(adata.X.max()):.3f} (expect log1p)", flush=True)

print(f"[celltypist] annotating with {MODEL}", flush=True)
pred = celltypist.annotate(adata, model=MODEL, majority_voting=False)
labels = pred.predicted_labels
col = "predicted_labels" if "predicted_labels" in labels.columns else labels.columns[0]
out = pd.DataFrame({"cell_id": adata.obs_names.astype(str),
                    "celltypist_label": labels[col].astype(str).values})
out.to_csv(OUT, index=False)
print(f"[celltypist] wrote {OUT}  ({out.shape[0]} cells, "
      f"{out['celltypist_label'].nunique()} unique labels)", flush=True)
print("STEP1_DONE", flush=True)
