"""Steps 5-7 for the diemb cluster/severity downstream (hongkai env).

Runs on the reconciled-severity grouping written by
diemb_step2to4_annotate_cluster_reconcile.py, RNA expression only, named cell
types. All compute uses package functions (mirrors wrapper.downstream_analysis).

Step 5  Cluster DEG via RAISIN (group_col = reconciled_severity, batch = batch).
Step 6  Differential cell proportion (proportion_test, same grouping).
Step 7  Semi-supervised trajectory (CCA_Call on sev.level) + trajectory DGE.

Outputs under sample_embedding_tune-on-RNA/cluster_severity_deg/:
  raisin_results/, proportion_test/, trajectory/, trajectoryDEG/
"""
import gc
import os
import sys
import traceback

import pandas as pd
import scanpy as sc

sys.path.insert(0, "/users/hjiang/GenoDistance/code")
from sampledisco.sample_clustering.RAISIN import raisinfit
from sampledisco.sample_clustering.RAISIN_TEST import run_pairwise_tests
from sampledisco.sample_clustering.proportion_test import proportion_test
from sampledisco.sample_trajectory.CCA import CCA_Call
from sampledisco.sample_trajectory.trajectory_diff_gene import run_trajectory_gam_differential_gene_analysis

BASE = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics"
OUT = f"{BASE}/sample_embedding_tune-on-RNA/cluster_severity_deg"
RNA_PRE = f"{BASE}/preprocess/adata_rna_preprocessed.h5ad"
PSEUDO = f"{OUT}/pseudo_sample_embedding.h5ad"

SAMPLE_COL = "sample"
BATCH_COL = "batch"
CELLTYPE_COL = "cell_type"
GROUP_COL = "reconciled_severity"
TRAJ_COL = "sev.level"

print("[load] adata_rna_preprocessed", flush=True)
rna = sc.read_h5ad(RNA_PRE)
# Drop cells without a reconciled-severity group (unmatched samples).
keep = rna.obs[GROUP_COL].astype(str) != "nan"
if int((~keep).sum()) > 0:
    print(f"  dropping {int((~keep).sum())} cells lacking {GROUP_COL}", flush=True)
    rna = rna[keep].copy()
print(f"  shape {rna.shape} | groups: "
      f"{rna.obs[GROUP_COL].value_counts().to_dict()}", flush=True)
groups = sorted(rna.obs[GROUP_COL].astype(str).unique())
control = "1" if "1" in groups else None  # mildest severity as baseline

# ── Step 6: differential cell proportion ──────────────────────────────────── #
print("\n[step6] proportion test", flush=True)
try:
    proportion_test(
        adata=rna,
        sample_col=SAMPLE_COL,
        group_col=GROUP_COL,
        celltype_col=CELLTYPE_COL,
        output_dir=os.path.join(OUT, "proportion_test"),
        verbose=True,
    )
    print("STEP6_DONE", flush=True)
except Exception:
    traceback.print_exc()

# ── Step 7: semi-supervised trajectory + trajectory DGE ───────────────────── #
print("\n[step7] supervised trajectory (CCA) + trajectory DGE", flush=True)
try:
    pseudo = sc.read_h5ad(PSEUDO)
    cca_a, cca_b, ptime_a, ptime_b = CCA_Call(
        adata=pseudo,
        output_dir=os.path.join(OUT, "trajectory"),
        trajectory_col=TRAJ_COL,
        n_components=2,
        verbose=True,
    )
    ptime = ptime_a if ptime_a else ptime_b
    # sample-embedding units are "<sample>_RNA"; map to bare sample (float values
    # so _read_pseudotime_table accepts the simple {sample: pseudotime} mapping).
    ptime_bare = {str(u)[:-4]: float(v) for u, v in ptime.items()
                  if str(u).endswith("_RNA")}
    print(f"  pseudotime for {len(ptime_bare)} RNA samples; "
          f"CCA score={cca_a if cca_a is not None else cca_b}", flush=True)
    if not ptime_bare:
        raise ValueError("CCA produced no pseudotime (empty); skipping trajectory DGE.")
    run_trajectory_gam_differential_gene_analysis(
        adata=rna,
        pseudotime_source=ptime_bare,
        sample_col=SAMPLE_COL,
        celltype_col=CELLTYPE_COL,
        batch_col=BATCH_COL,
        output_dir=os.path.join(OUT, "trajectoryDEG"),
        verbose=True,
    )
    print("STEP7_DONE", flush=True)
except Exception:
    traceback.print_exc()

# ── Step 5: RAISIN cluster DEG, per named cell type ───────────────────────── #
# raisinfit has no native cell-type split and ComBat on all 895k cells exceeds
# 200G, so we run it per cell type (each fits easily) — which also matches the
# "per named cell type" DEG plan. n_jobs=16 = the logical CPUs the cgroup grants
# (8 physical cores x HT); BLAS is pinned to 1 thread/worker at launch so the
# joblib per-gene fitting doesn't nest-oversubscribe.
print("\n[step5] RAISIN cluster DEG per cell type", flush=True)
celltypes = sorted(rna.obs[CELLTYPE_COL].astype(str).unique())
ok, failed = [], []
for ct in celltypes:
    ct_safe = ct.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    out_ct = os.path.join(OUT, "raisin_results", ct_safe)
    sub = rna[rna.obs[CELLTYPE_COL].astype(str) == ct].copy()
    print(f"\n  --- {ct}  ({sub.n_obs} cells, "
          f"{sub.obs[SAMPLE_COL].nunique()} samples) ---", flush=True)
    try:
        fit = raisinfit(
            adata=sub, sample_col=SAMPLE_COL, testtype="unpaired",
            batch_col=BATCH_COL, group_col=GROUP_COL,
            intercept=True, n_jobs=16, verbose=True,
        )
        run_pairwise_tests(
            fit=fit, output_dir=out_ct, control_group=control,
            fdrmethod="fdr_bh", fdr_threshold=0.05, verbose=True,
        )
        ok.append(ct)
    except Exception:
        failed.append(ct)
        traceback.print_exc()
    finally:
        del sub
        gc.collect()
print(f"\n[step5] RAISIN done — {len(ok)} cell types ok, {len(failed)} failed: {failed}",
      flush=True)
print("STEP5_DONE", flush=True)

# Consolidated cross-cell-type RAISIN summary.
import subprocess
subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(__file__), "diemb_raisin_summary.py"),
     os.path.join(OUT, "raisin_results")],
    check=False,
)

print("STEP5TO7_DONE", flush=True)
