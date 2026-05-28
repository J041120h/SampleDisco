"""
Re-run dimension association for round5 / round6 after deleting Batch_AgeGroup
from the cell-level preprocessed adata (no longer needed; round6 uses proper
multi-covariate Harmony instead of the joint-strata workaround).

Loads cell adata once (heavy), then for each round:
  1. read sample_embedding.csv
  2. stick it into adata.uns['X_DR_sample'] (per build_sample_adata contract)
  3. build_sample_adata -> per-sample AnnData
  4. run_dimension_association_analysis (writes variance_explained_sample.csv
     + figures under <round_dir>/sample_association/)
"""

import sys
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import anndata as ad
import pandas as pd

from sample_embedding.sample_embedding import build_sample_adata
from sample_association.association import run_dimension_association_analysis

CELL_H5AD = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/preprocess/adata_preprocessed.h5ad"

TARGETS = [
    ("round5",
     "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round5_batch_age/sample_embedding/sample_embedding/sample_embedding.csv",
     "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round5_batch_age/sample_embedding/sample_association"),
    ("round6",
     "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round6_multicov_batch_age/sample_embedding/sample_embedding/sample_embedding.csv",
     "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round6_multicov_batch_age/sample_embedding/sample_association"),
]
SAMPLE_COL = "Tube_id"


def main():
    print("Loading cell-level adata (this is heavy) ...", flush=True)
    a = ad.read_h5ad(CELL_H5AD)
    print(f"  shape: {a.shape}; obs cols: {len(a.obs.columns)}")
    assert "Batch_AgeGroup" not in a.obs.columns, "Batch_AgeGroup still present!"
    print("  confirmed Batch_AgeGroup is gone.")

    for tag, emb_csv, out_dir in TARGETS:
        print(f"\n=== {tag} ===", flush=True)
        emb = pd.read_csv(emb_csv, index_col=0)
        emb.index = emb.index.astype(str)
        emb.index.name = SAMPLE_COL
        print(f"  embedding {emb.shape} from {emb_csv}")
        a.uns["X_DR_sample"] = emb.copy()
        sample_adata = build_sample_adata(a, sample_col=SAMPLE_COL)
        print(f"  sample_adata: {sample_adata.shape}; obs cols: {list(sample_adata.obs.columns)[:8]} ...")
        run_dimension_association_analysis(
            pseudo_adata=sample_adata,
            output_dir=out_dir,
            n_permutations=999,
            sample_col=SAMPLE_COL,
            verbose=True,
        )
        print(f"  wrote results under {out_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
