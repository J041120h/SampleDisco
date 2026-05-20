"""Mode-B α sweep: per-modality CCA across the cmd_weight grid.

Mirrors claude/_archive/cca_vs_alpha_sweep.py (which built the same table
for Mode A on unpaired_test). Difference: this one runs on the
unpaired_diemb pipeline (2-run scGLUE, Z_clust + Z_cmd) so we can answer
the question "do r_RNA(α) and r_ATAC(α) move together in Mode B?" and
compare the answer to Mode A.

For each α (Mode-A grid + autotune's 15 evals): build SE, split by
modality, compute (a) full 10-PC CCA per modality, (b) joint best-2-PC
CCA per modality, (c) joint composite. Saves alpha_sweep.csv and a 2-pane
plot side-by-side.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import numpy as np, pandas as pd, anndata as ad
import scanpy as sc
from itertools import combinations
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr, spearmanr

H5      = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics/preprocess/adata_sample_celltyped.h5ad"
OUT_DIR = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics/alpha_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

CLUSTER_EMB = "Z_clust"
CMD_EMB     = "Z_cmd"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── load + memory-safe adata ──────────────────────────────────────────────
log(f"loading {H5}")
a = ad.read_h5ad(H5, backed='r')
n = a.shape[0]
obs = a.obs.copy()
obsm = {k: np.asarray(a.obsm[k]) for k in a.obsm.keys()}
a.file.close()
adata = sc.AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs)
for k, v in obsm.items():
    adata.obsm[k] = v
log(f"  shape={adata.shape}  K_c={adata.obs['cell_type'].nunique()}  obsm={list(adata.obsm.keys())}")


# ── build_blocks ONCE ─────────────────────────────────────────────────────
from parameter_selection.autotune import build_blocks, build_emb_from_blocks, derive_weights
log("build_blocks (one-shot K-means K=120, K=300)")
blocks = build_blocks(
    adata,
    sample_col="sample", celltype_col="cell_type",
    cluster_emb_key=CLUSTER_EMB,
    cmd_emb_key=CMD_EMB,
    modality_col="modality", batch_col="batch",
    grouping_col="sev.level",
    verbose=True,
)


# ── unit-level modality + sev.level ───────────────────────────────────────
# unit_id is built as `{bio}_{modality}` and matches adata.obs["sample"] when
# that column already carries the modality suffix (the case here after the
# gene-activity merge). Use unit_id directly for sev lookup; take modality
# from blocks["unit_groups"] which assemble_units already records.
unit_ids = blocks["unit_ids"]
unit_df = pd.DataFrame({
    "unit":     unit_ids,
    "modality": blocks["unit_groups"],
})

sev_per_sample = (adata.obs.groupby("sample", observed=True)["sev.level"]
                  .apply(lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else np.nan))
unit_df["sev"] = pd.to_numeric(unit_df["unit"].map(sev_per_sample), errors="coerce")
log(f"  modality counts: {unit_df['modality'].value_counts().to_dict()}  "
    f"sev non-NaN: {int(unit_df['sev'].notna().sum())}/{len(unit_df)}")


# ── CCA helpers ───────────────────────────────────────────────────────────
def cca_r_full(X: np.ndarray, y: np.ndarray) -> float:
    if X.shape[1] < 1: return 0.0
    y = np.asarray(y, dtype=float)
    keep = ~np.isnan(y)
    if keep.sum() < 5: return 0.0
    X = X[keep]; y = y[keep]
    c = CCA(n_components=1, max_iter=500).fit(X, y.reshape(-1, 1))
    U, V = c.transform(X, y.reshape(-1, 1))
    return float(abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1]))


def joint_best_pair_cca(E_rna, y_rna, E_atac, y_atac):
    n_pc = E_rna.shape[1]
    best = (None, 0.0, 0.0, -np.inf)
    for i, j in combinations(range(n_pc), 2):
        r_r = cca_r_full(E_rna[:,  [i, j]], y_rna)
        r_a = cca_r_full(E_atac[:, [i, j]], y_atac)
        s = r_r + r_a
        if s > best[3]:
            best = ((i, j), r_r, r_a, s)
    return best


# ── α grid: same as Mode A (15 autotune α + 25 log-spaced) ───────────────
# Mode-B autotune actually visited these 15 (from autotune_record.txt):
autotune_alphas = [
    0.1, 2.302204, 2.500601, 2.575, 2.679158, 2.758517, 2.818036,
    2.917234, 4.980561, 5.05, 5.178958, 7.420842, 7.525, 7.639078, 10.0,
]
extra_alphas = list(np.round(np.logspace(np.log10(0.1), np.log10(10), 25), 4))
alphas = sorted(set(autotune_alphas + extra_alphas))
log(f"sweeping {len(alphas)} α values from {min(alphas)} to {max(alphas)}")


# ── sweep ─────────────────────────────────────────────────────────────────
rna_mask  = (unit_df["modality"] == "RNA").values
atac_mask = (unit_df["modality"] == "ATAC").values
y_rna  = unit_df.loc[rna_mask,  "sev"].values
y_atac = unit_df.loc[atac_mask, "sev"].values

records = []
t_start = time.time()
for k, alpha in enumerate(alphas, 1):
    weights = derive_weights(blocks["K_c"], blocks["K_med"], blocks["K_fine"],
                              cmd_weight=alpha, n_blocks=4)
    emb_df = build_emb_from_blocks(
        [blocks["A1"], blocks["A2"], blocks["A3"], blocks["CMD"]],
        weights, unit_ids=unit_ids,
        unit_groups=blocks["unit_groups"], unit_batches=blocks["unit_batches"],
        pca_components=10, batch_method="harmony", seed=42, verbose=False,
    )
    E = emb_df.values
    E_rna, E_atac = E[rna_mask], E[atac_mask]

    r_rna_full  = cca_r_full(E_rna,  y_rna)
    r_atac_full = cca_r_full(E_atac, y_atac)
    (pc_i, pc_j), r_rna_j, r_atac_j, sumj = joint_best_pair_cca(E_rna, y_rna, E_atac, y_atac)
    keep = ~pd.isna(unit_df["sev"]).values
    r_joint_full = cca_r_full(E[keep], unit_df.loc[keep, "sev"].values)

    records.append({
        "alpha": float(alpha),
        "r_rna_full":  r_rna_full,
        "r_atac_full": r_atac_full,
        "r_rna_jointBP":  r_rna_j,
        "r_atac_jointBP": r_atac_j,
        "joint_best_pair": f"PC{pc_i+1}+PC{pc_j+1}",
        "r_joint_full": r_joint_full,
        "sum_jointBP": sumj,
        "was_autotune_eval": float(alpha) in {round(x, 6) for x in autotune_alphas},
    })
    log(f"  α={alpha:7.3f}  r_RNA_full={r_rna_full:.4f}  r_ATAC_full={r_atac_full:.4f}  "
        f"r_RNA_BP={r_rna_j:.4f}  r_ATAC_BP={r_atac_j:.4f}  joint={r_joint_full:.4f}  "
        f"best_pair=PC{pc_i+1}+PC{pc_j+1}")
log(f"sweep done in {time.time()-t_start:.1f}s")

df = pd.DataFrame.from_records(records).sort_values("alpha").reset_index(drop=True)
df.to_csv(f"{OUT_DIR}/alpha_sweep.csv", index=False)
log(f"saved {OUT_DIR}/alpha_sweep.csv")


# ── correlation analysis vs Mode A ────────────────────────────────────────
pr_full, ppv_full = pearsonr(df["r_rna_full"],  df["r_atac_full"])
sr_full, spv_full = spearmanr(df["r_rna_full"], df["r_atac_full"])
pr_bp,   ppv_bp   = pearsonr(df["r_rna_jointBP"],  df["r_atac_jointBP"])
sr_bp,   spv_bp   = spearmanr(df["r_rna_jointBP"], df["r_atac_jointBP"])

log("=" * 70)
log("Mode B (Z_clust + Z_cmd) — correlation between r_RNA(α) and r_ATAC(α)")
log("=" * 70)
log(f"  Full 10-PC CCA   :  Pearson r = {pr_full:+.4f} (p={ppv_full:.3g})   "
    f"Spearman ρ = {sr_full:+.4f} (p={spv_full:.3g})")
log(f"  Joint best 2-PC  :  Pearson r = {pr_bp:+.4f} (p={ppv_bp:.3g})   "
    f"Spearman ρ = {sr_bp:+.4f} (p={spv_bp:.3g})")

# Side-by-side comparison: load Mode A's sweep table
A_PATH = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/sampledisco_tuned_v2_RETUNE/alpha_sweep/alpha_sweep.csv"
if os.path.exists(A_PATH):
    dA = pd.read_csv(A_PATH)
    prA, _ = pearsonr(dA["r_rna_full"], dA["r_atac_full"])
    srA, _ = spearmanr(dA["r_rna_full"], dA["r_atac_full"])
    prAbp, _ = pearsonr(dA["r_rna_jointBP"], dA["r_atac_jointBP"])
    srAbp, _ = spearmanr(dA["r_rna_jointBP"], dA["r_atac_jointBP"])
    log("")
    log("Mode A (X_glue_harmony + nosamp) — reference (recomputed from saved sweep)")
    log(f"  Full 10-PC CCA   :  Pearson r = {prA:+.4f}   Spearman ρ = {srA:+.4f}")
    log(f"  Joint best 2-PC  :  Pearson r = {prAbp:+.4f}   Spearman ρ = {srAbp:+.4f}")


# ── plot: 2x2 panel: Mode A top, Mode B bottom; full / jointBP left/right ─
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
panels = [
    (df,  "Mode B (diemb: Z_clust + Z_cmd)",  axes[0]),
]
if os.path.exists(A_PATH):
    panels.insert(0, (dA, "Mode A (test: X_glue_harmony + nosamp)", axes[1]))

for (d, title, row) in panels:
    for axi, (cols, label) in enumerate((
        (("r_rna_full", "r_atac_full"), "Full 10-PC CCA"),
        (("r_rna_jointBP", "r_atac_jointBP"), "Joint best 2-PC pair"),
    )):
        a_, b_ = cols
        ax = row[axi]
        ax.plot(d["alpha"], d[a_], 'o-', color='tab:blue',   label='RNA',  ms=4)
        ax.plot(d["alpha"], d[b_], 's-', color='tab:orange', label='ATAC', ms=4)
        ax.set_xscale("log")
        ax.set_xlabel("cmd_weight α (log)")
        ax.set_ylabel("CCA r vs sev.level")
        pr, _ = pearsonr(d[a_], d[b_])
        sr, _ = spearmanr(d[a_], d[b_])
        ax.set_title(f"{title}\n{label}   Pearson(RNA,ATAC)={pr:+.3f}  Spearman={sr:+.3f}",
                     fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/alpha_sweep.png", dpi=130)
plt.close(fig)
log(f"saved {OUT_DIR}/alpha_sweep.png")
log("DONE")
