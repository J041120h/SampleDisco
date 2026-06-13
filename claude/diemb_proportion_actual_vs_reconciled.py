"""Run the differential cell-proportion test with the ACTUAL sev.level label and
compare it to the unsupervised reconciled-KMeans grouping already computed.

Reports, per pairwise comparison and overall:
  - direction agreement: sign(logFC_actual) == sign(logFC_reconciled)
  - agreement restricted to cell types significant under the actual label
  - significant-set overlap between the two groupings
"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, "/users/hjiang/GenoDistance/code")
from sampledisco.sample_clustering.proportion_test import proportion_test

BASE = ("/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics/"
        "sample_embedding_tune-on-RNA/cluster_severity_deg")
RNA = ("/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics/"
       "preprocess/adata_rna_preprocessed.h5ad")
RECON_DIR = f"{BASE}/proportion_test"               # existing (reconciled-severity)
ACTUAL_DIR = f"{BASE}/proportion_test_actual"        # new (true sev.level)
FDR = 0.05

# ── minimal obs-only adata (proportion test only needs obs) ───────────────── #
print("[load] obs of adata_rna_preprocessed", flush=True)
a = sc.read_h5ad(RNA, backed="r")
obs = a.obs[["sample", "cell_type", "sev.level", "reconciled_severity"]].copy()
obs["sev.level"] = obs["sev.level"].astype(str)
mini = ad.AnnData(X=np.zeros((obs.shape[0], 1), dtype=np.float32), obs=obs)

print("[run] proportion test on ACTUAL sev.level", flush=True)
proportion_test(adata=mini, sample_col="sample", group_col="sev.level",
                celltype_col="cell_type", output_dir=ACTUAL_DIR, verbose=False)

# ── compare directions per comparison ─────────────────────────────────────── #
comps = ["1_vs_2", "1_vs_3", "1_vs_4", "2_vs_3", "2_vs_4", "3_vs_4"]
rows, all_pairs = [], []
for c in comps:
    fa = f"{ACTUAL_DIR}/proportion_test_{c}.csv"
    fr = f"{RECON_DIR}/proportion_test_{c}.csv"
    if not (os.path.exists(fa) and os.path.exists(fr)):
        print(f"  [skip] {c}: missing file"); continue
    da = pd.read_csv(fa).set_index("celltype")
    dr = pd.read_csv(fr).set_index("celltype")
    m = da.join(dr, lsuffix="_act", rsuffix="_rec", how="inner").dropna(subset=["logFC_act", "logFC_rec"])
    m["agree"] = np.sign(m["logFC_act"]) == np.sign(m["logFC_rec"])
    m["sig_act"] = m["FDR_act"] < FDR
    m["sig_rec"] = m["FDR_rec"] < FDR
    m["comparison"] = c
    all_pairs.append(m.reset_index())
    n = len(m)
    rows.append({
        "comparison": c, "n_celltypes": n,
        "dir_agree_all": f"{int(m['agree'].sum())}/{n}",
        "dir_agree_all_pct": round(100 * m["agree"].mean(), 1),
        "n_sig_actual": int(m["sig_act"].sum()),
        "dir_agree_in_sig_actual": (f"{int(m.loc[m['sig_act'],'agree'].sum())}/{int(m['sig_act'].sum())}"
                                     if m["sig_act"].any() else "0/0"),
        "sig_both": int((m["sig_act"] & m["sig_rec"]).sum()),
    })

summary = pd.DataFrame(rows)
allp = pd.concat(all_pairs, ignore_index=True)
tot = len(allp)
agree_all = allp["agree"].mean()
sigA = allp[allp["sig_act"]]
agree_sig = sigA["agree"].mean() if len(sigA) else float("nan")

out_txt = f"{BASE}/proportion_actual_vs_reconciled.txt"
with open(out_txt, "w") as fh:
    fh.write("Differential proportion: ACTUAL sev.level vs unsupervised reconciled-KMeans\n")
    fh.write("=" * 78 + "\n\n")
    fh.write(summary.to_string(index=False) + "\n\n")
    fh.write(f"OVERALL direction agreement (all celltype x comparison): "
             f"{int(allp['agree'].sum())}/{tot} = {100*agree_all:.1f}%\n")
    fh.write(f"OVERALL direction agreement among cell types significant under ACTUAL label: "
             f"{int(sigA['agree'].sum())}/{len(sigA)} = {100*agree_sig:.1f}%\n\n")
    fh.write("Per (cell type x comparison): logFC_actual | logFC_recon | agree | sig_actual\n")
    for _, r in allp.sort_values(["comparison", "celltype"]).iterrows():
        fh.write(f"  {r['comparison']}  {r['celltype']:26s} "
                 f"act={r['logFC_act']:+.3f}  rec={r['logFC_rec']:+.3f}  "
                 f"{'AGREE' if r['agree'] else 'DISAGREE'}"
                 f"{'  *sig_act*' if r['sig_act'] else ''}\n")
allp.to_csv(f"{BASE}/proportion_actual_vs_reconciled.csv", index=False)

print("\n" + summary.to_string(index=False), flush=True)
print(f"\nOVERALL direction agreement (all): {int(allp['agree'].sum())}/{tot} = {100*agree_all:.1f}%", flush=True)
print(f"OVERALL direction agreement (sig under actual): {int(sigA['agree'].sum())}/{len(sigA)} = {100*agree_sig:.1f}%", flush=True)
print(f"wrote {out_txt}", flush=True)
print("PROP_COMPARE_DONE", flush=True)
