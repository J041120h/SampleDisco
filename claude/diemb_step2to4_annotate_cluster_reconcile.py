"""Steps 2-4 for the diemb cluster/severity downstream (hongkai env).

Step 2  Name the existing 17 Leiden clusters by CellTypist majority vote.
        Leiden labels (cell_type 1..17) live in the embedding union; the
        per-cell CellTypist labels come from diemb_step1_celltypist.py. We
        match RNA cells by bare barcode (rna.obs_names == union.original_barcode)
        and assign each Leiden cluster the most frequent CellTypist label.
        Collisions (two Leiden clusters voting the same label) are kept distinct
        with an "(L<id>)" suffix so the 17 clusters are preserved, only renamed.

Step 3  Package KMeans (sample_clustering.cluster.cluster, k=4) on the RNA-tune
        sample embedding (RNA units only, 405).

Step 4  Hungarian 1:1 reconciliation of KMeans clusters -> severity
        (sample_clustering.cluster_severity_reconcile), writing a report and a
        per-sample reconciled-severity grouping onto the RNA cell h5ad.

Writes (under sample_embedding_tune-on-RNA/cluster_severity_deg/):
  celltype_naming.txt, sample_cluster/ (KMeans csv+png),
  cluster_severity_reconciliation.txt, pseudo_sample_embedding.h5ad
Overwrites preprocess/adata_rna_preprocessed.h5ad with obs cols:
  leiden, celltypist, cell_type (named), reconciled_severity.
"""
import os
import sys

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

sys.path.insert(0, "/users/hjiang/GenoDistance/code")
from sample_clustering.cluster import cluster
from sample_clustering.cluster_severity_reconcile import reconcile_clusters_to_label
from utils.safe_save import safe_h5ad_write

BASE = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics"
OUT = f"{BASE}/sample_embedding_tune-on-RNA/cluster_severity_deg"
RNA_PRE = f"{BASE}/preprocess/adata_rna_preprocessed.h5ad"
UNION = f"{BASE}/preprocess/adata_sample.h5ad"
EMB_CSV = f"{BASE}/sample_embedding_tune-on-RNA/sample_embedding.csv"
CT_CSV = f"{OUT}/celltypist_per_cell.csv"
os.makedirs(OUT, exist_ok=True)

# ── load RNA cell adata + union Leiden labels ─────────────────────────────── #
print("[load] adata_rna_preprocessed", flush=True)
rna = sc.read_h5ad(RNA_PRE)
print(f"  shape {rna.shape}", flush=True)
bc = rna.obs_names.astype(str)

print("[load] union Leiden cell_type (backed)", flush=True)
u = sc.read_h5ad(UNION, backed="r")
um = u.obs[u.obs["modality"] == "RNA"]
bc_to_leiden = pd.Series(um["cell_type"].astype(str).values,
                         index=um["original_barcode"].astype(str).values)

# ── Step 2: CellTypist majority vote per Leiden cluster ───────────────────── #
print("[step2] majority-vote CellTypist label per Leiden cluster", flush=True)
rna.obs["leiden"] = bc_to_leiden.reindex(bc).values
ct = pd.read_csv(CT_CSV)
ct_map = pd.Series(ct["celltypist_label"].astype(str).values,
                   index=ct["cell_id"].astype(str).values)
rna.obs["celltypist"] = ct_map.reindex(bc).values
n_na_leiden = int(pd.isna(rna.obs["leiden"]).sum())
n_na_ct = int(pd.isna(rna.obs["celltypist"]).sum())
print(f"  unmatched leiden={n_na_leiden}  celltypist={n_na_ct}", flush=True)

vote_tbl = (rna.obs.dropna(subset=["leiden", "celltypist"])
            .groupby("leiden")["celltypist"].value_counts()
            .rename("n").reset_index())
top = vote_tbl.sort_values("n", ascending=False).drop_duplicates("leiden")
top = top.set_index("leiden")
# Name each Leiden cluster by its majority CellTypist label. Leiden clusters
# that share a label are intentionally merged into one cell type (no Leiden-id
# disambiguation), so cell_type is the biological label only.
name_map = {str(lid): row["celltypist"] for lid, row in top.iterrows()}

leiden_levels = sorted(top.index, key=lambda x: int(x))
with open(f"{OUT}/celltype_naming.txt", "w") as fh:
    fh.write("Leiden cluster -> CellTypist majority-vote name\n")
    fh.write("=" * 60 + "\n\n")
    for lid in leiden_levels:
        n_cells = int((rna.obs["leiden"].astype(str) == str(lid)).sum())
        top_n = int(top.loc[lid, "n"])
        fh.write(f"  Leiden {lid:>2}  (n={n_cells:>7})  ->  {name_map[str(lid)]}"
                 f"   [{top_n}/{n_cells} = {top_n / max(n_cells,1):.1%} agree]\n")
    fh.write("\nFull contingency (Leiden x CellTypist counts):\n")
    fh.write(pd.crosstab(rna.obs["leiden"], rna.obs["celltypist"]).to_string())
    fh.write("\n")
print(f"  wrote {OUT}/celltype_naming.txt", flush=True)
rna.obs["cell_type"] = (rna.obs["leiden"].astype(str).map(name_map)
                        .astype("category"))
print(f"  named cell types: {list(rna.obs['cell_type'].cat.categories)}", flush=True)

# ── Step 3: KMeans k=4 on RNA-tune sample embedding (RNA units only) ──────── #
print("[step3] KMeans k=4 on RNA-tune sample embedding", flush=True)
emb = pd.read_csv(EMB_CSV)
emb = emb[emb["sample"].astype(str).str.endswith("_RNA")].reset_index(drop=True)
units = emb["sample"].astype(str).values
pc_cols = [c for c in emb.columns if c.upper().startswith("PC")]
X_dr = emb[pc_cols].values.astype(np.float32)
print(f"  {len(units)} RNA units x {len(pc_cols)} PCs", flush=True)

unit_sev = (u.obs.drop_duplicates("sample").set_index("sample")["sev.level"]
            .astype(str))
pseudo = ad.AnnData(
    X=np.zeros((len(units), 1), dtype=np.float32),
    obs=pd.DataFrame({"sev.level": pd.Series(units, index=units).map(unit_sev)},
                     index=units),
)
pseudo.obsm["X_DR_sample"] = X_dr
pseudo.uns["X_DR_sample"] = X_dr  # CCA_Call reads the embedding from .uns
expr_results, _ = cluster(pseudobulk_adata=pseudo, output_dir=OUT,
                          number_of_clusters=4, random_state=0)

# ── Step 4: Hungarian reconcile clusters -> severity ──────────────────────── #
print("[step4] reconcile KMeans clusters -> severity (Hungarian 1:1)", flush=True)
sample_to_label = pseudo.obs["sev.level"].to_dict()
sample_to_pred, cluster_to_sev, stats = reconcile_clusters_to_label(
    sample_to_cluster=expr_results,
    sample_to_label=sample_to_label,
    label_name="severity",
    output_txt=f"{OUT}/cluster_severity_reconciliation.txt",
)
print(f"  reconciliation accuracy = {stats['accuracy']:.4f}", flush=True)

# Per-unit reconciled severity onto pseudo; persist for the trajectory step.
pseudo.obs["kmeans_cluster"] = pd.Series(expr_results).reindex(units).astype(str).values
pseudo.obs["reconciled_severity"] = (pd.Series(sample_to_pred).reindex(units)
                                     .astype(str).values)
safe_h5ad_write(pseudo, f"{OUT}/pseudo_sample_embedding.h5ad")
print(f"  wrote {OUT}/pseudo_sample_embedding.h5ad", flush=True)

# Per-sample reconciled severity onto the RNA cells (strip _RNA -> bare sample).
bare_to_recon = {unit[:-4]: sev for unit, sev in sample_to_pred.items()}
rna.obs["reconciled_severity"] = (rna.obs["sample"].astype(str)
                                  .map(bare_to_recon).astype(str))
n_grouped = int((rna.obs["reconciled_severity"] != "nan").sum())
print(f"  reconciled_severity set for {n_grouped}/{rna.n_obs} cells", flush=True)

print("[write] overwriting adata_rna_preprocessed with new obs cols", flush=True)
safe_h5ad_write(rna, RNA_PRE)
print("STEP2TO4_DONE", flush=True)
