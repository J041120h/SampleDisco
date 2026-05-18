"""Sweep autotune's cmd_weight α and record per-modality CCA at each step.

Answers: do r_RNA and r_ATAC rise / fall together across α, or do they
diverge? Same blocks (A1/A2/A3/CMD) are reused for every α — only the
relative weight on CMD changes.

For each α (dense log grid + the 15 α-values that autotune actually
visited): build the SE, split by modality, compute (a) full-D CCA per
modality, (b) joint best-2-PC CCA per modality, (c) joint composite (what
autotune optimises). Save table + plot.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import numpy as np, pandas as pd, anndata as ad
import scanpy as sc
from itertools import combinations
from sklearn.cross_decomposition import CCA

H5      = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/preprocess/atac_rna_integrated.h5ad"
OUT_DIR = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/sampledisco_tuned_v2_RETUNE/alpha_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

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
log(f"  shape={adata.shape}  K_c={adata.obs['cell_type'].nunique()}")


# ── build_blocks ONCE (kmeans / composition / CMD) ────────────────────────
from parameter_selection.autotune import build_blocks, build_emb_from_blocks, derive_weights
log("build_blocks (one-shot K-means K=120, K=300)")
blocks = build_blocks(
    adata,
    sample_col="sample", celltype_col="cell_type",
    cluster_emb_key="X_glue_harmony",
    cmd_emb_key="X_glue_harmony_nosamp",
    modality_col="modality", batch_col="batch",
    grouping_col="sev.level",
    verbose=True,
)


# Map unit_id ("<sample>__<modality>") → modality + sev.level
unit_ids = blocks["unit_ids"]
# unit_id format from sample_embedding: "<sample>_<modality>"
unit_df = pd.DataFrame({"unit": unit_ids})
unit_df["modality"] = unit_df["unit"].str.rsplit("_", n=1).str[1]
unit_df["sample"]   = unit_df["unit"].str.rsplit("_", n=1).str[0]

# sev.level per sample (majority of cell rows in adata.obs)
sev_per_sample = (adata.obs[["sample", "sev.level"]]
                  .groupby("sample")["sev.level"]
                  .apply(lambda s: s.dropna().mode().iloc[0])
                  .astype(float))
unit_df["sev"] = unit_df["sample"].map(sev_per_sample).astype(float)
log(f"  modality counts: {unit_df['modality'].value_counts().to_dict()}")


# ── CCA helpers ───────────────────────────────────────────────────────────
def cca_r_full(X: np.ndarray, y: np.ndarray) -> float:
    """CCA on ALL columns of X vs scalar y. Returns abs(r)."""
    if X.shape[1] < 1:
        return 0.0
    c = CCA(n_components=1, max_iter=500).fit(X, y.reshape(-1, 1))
    U, V = c.transform(X, y.reshape(-1, 1))
    return float(abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1]))


def joint_best_pair_cca(E_rna: np.ndarray, y_rna: np.ndarray,
                         E_atac: np.ndarray, y_atac: np.ndarray) -> tuple:
    """Return (best_pair, r_rna, r_atac, sum) over all PC pairs."""
    n_pc = E_rna.shape[1]
    best = (None, 0.0, 0.0, -np.inf)
    for i, j in combinations(range(n_pc), 2):
        r_r = cca_r_full(E_rna[:,  [i, j]], y_rna)
        r_a = cca_r_full(E_atac[:, [i, j]], y_atac)
        s = r_r + r_a
        if s > best[3]:
            best = ((i, j), r_r, r_a, s)
    return best


# ── α grid: dense log-spaced + autotune's actual 15 evaluations ──────────
autotune_alphas = [0.10, 0.616, 1.032, 1.528, 1.588, 1.786, 2.575, 3.056,
                   3.612, 4.246, 5.05, 6.250, 7.525, 8.849, 10.0]
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

    # full-D CCA per modality (uses all 10 PCs)
    r_rna_full  = cca_r_full(E_rna,  y_rna)
    r_atac_full = cca_r_full(E_atac, y_atac)

    # joint best 2-PC pair
    (pc_i, pc_j), r_rna_j, r_atac_j, sumj = joint_best_pair_cca(E_rna, y_rna, E_atac, y_atac)

    # joint CCA (mixed RNA+ATAC vs sev, what autotune optimises in part)
    keep = ~pd.isna(unit_df["sev"]).values
    r_joint_full = cca_r_full(E[keep], unit_df.loc[keep, "sev"].values)

    rec = {
        "alpha": float(alpha),
        "r_rna_full":  r_rna_full,
        "r_atac_full": r_atac_full,
        "r_rna_jointBP":  r_rna_j,
        "r_atac_jointBP": r_atac_j,
        "joint_best_pair": f"PC{pc_i+1}+PC{pc_j+1}",
        "r_joint_full": r_joint_full,
        "sum_jointBP": sumj,
        "was_autotune_eval": float(alpha) in {round(x, 4) for x in autotune_alphas},
    }
    records.append(rec)
    log(f"  α={alpha:7.3f}  r_RNA_full={r_rna_full:.4f}  r_ATAC_full={r_atac_full:.4f}  "
        f"r_RNA_BP={r_rna_j:.4f}  r_ATAC_BP={r_atac_j:.4f}  joint={r_joint_full:.4f}  "
        f"best_pair={rec['joint_best_pair']}")
log(f"sweep done in {time.time()-t_start:.1f}s")

df = pd.DataFrame.from_records(records)
df = df.sort_values("alpha").reset_index(drop=True)
df.to_csv(f"{OUT_DIR}/alpha_sweep.csv", index=False)
log(f"saved {OUT_DIR}/alpha_sweep.csv")


# ── analysis: do r_RNA and r_ATAC move together? ─────────────────────────
from scipy.stats import pearsonr, spearmanr
pr_full, ppv_full = pearsonr(df["r_rna_full"],  df["r_atac_full"])
sr_full, spv_full = spearmanr(df["r_rna_full"], df["r_atac_full"])
pr_bp,   ppv_bp   = pearsonr(df["r_rna_jointBP"],  df["r_atac_jointBP"])
sr_bp,   spv_bp   = spearmanr(df["r_rna_jointBP"], df["r_atac_jointBP"])

log("=" * 70)
log("Correlation between r_RNA(α) and r_ATAC(α) across the α grid")
log("=" * 70)
log(f"  Full 10-PC CCA   :  Pearson r = {pr_full:+.4f} (p={ppv_full:.3g})   "
    f"Spearman ρ = {sr_full:+.4f} (p={spv_full:.3g})")
log(f"  Joint best 2-PC  :  Pearson r = {pr_bp:+.4f} (p={ppv_bp:.3g})   "
    f"Spearman ρ = {sr_bp:+.4f} (p={spv_bp:.3g})")
log("")
log("Interpretation:")
log("  +1 = RNA and ATAC CCA always move together with α")
log("  -1 = they trade off (one up when the other is down)")
log("   0 = independent of each other")

# ── plot ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for axi, (cols, label) in enumerate((
    (("r_rna_full", "r_atac_full"), "Full 10-PC CCA"),
    (("r_rna_jointBP", "r_atac_jointBP"), "Joint best 2-PC pair"),
)):
    a_, b_ = cols
    ax[axi].plot(df["alpha"], df[a_], 'o-', color='tab:blue', label='RNA')
    ax[axi].plot(df["alpha"], df[b_], 's-', color='tab:orange', label='ATAC')
    ax[axi].set_xscale("log")
    ax[axi].set_xlabel("cmd_weight α (log)")
    ax[axi].set_ylabel("CCA r vs sev.level")
    pr, _ = pearsonr(df[a_], df[b_])
    ax[axi].set_title(f"{label}\nPearson(RNA, ATAC across α) = {pr:+.3f}")
    ax[axi].legend()
    ax[axi].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/alpha_sweep.png", dpi=130)
plt.close(fig)
log(f"saved {OUT_DIR}/alpha_sweep.png")
log("DONE")
