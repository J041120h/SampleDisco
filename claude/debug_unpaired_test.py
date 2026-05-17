"""Debug pipeline for /dcs07/hongkai/data/harry/result/multi_omics_unpaired_test

Two parts:

  PART 1 — cell-level UMAP × {X_glue, X_glue_harmony, X_glue_harmony_nosamp}
           coloured by batch and by modality (6 PNG).

  PART 2 — sample-level block decomposition. Replicate the SampleDisco
           pipeline up to (but not including) the Frobenius stack, then for
           each block (A1, A2, A3, CMD) compute:
             * unit-level ASW(batch)                — higher = more batch-separated
             * top-2 PC scatter coloured by batch   — visual sanity check
             * explained-variance fraction tied to batch (R² of OLS on one-hot batch)
           Write all numbers to a CSV + a short markdown summary, plot per-block
           PCA scatters as PNG.

All outputs into:
   /dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/debug/
"""
from __future__ import annotations
import os, sys, time, gc
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from sample_embedding.blocks import (
    assemble_units, composition_per_unit, soft_assign, loo_cmd,
)

H5  = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/preprocess/atac_rna_integrated.h5ad'
OUT = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/debug'
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/cell_umap", exist_ok=True)
os.makedirs(f"{OUT}/block_analysis", exist_ok=True)

BATCH_COLORS = {
    'Aruna':'#1f77b4', 'Guo':'#ff7f0e', 'Lee':'#2ca02c', 'Mudd':'#d62728',
    'SS1':'#9467bd', 'SS2':'#e377c2', 'Silvin':'#7f7f7f', 'Su':'#bcbd22',
    'Wen':'#17becf', 'Wilk':'#aec7e8', 'Yu':'#ffbb78', 'Zhu':'#98df8a',
    'UNK':'#cccccc',
}
MOD_COLORS = {'RNA':'#4c9fe5', 'ATAC':'#ed6a5a'}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def plot_scatter(xy, colors_arr, color_map, title, path, point_size=4, alpha=0.6):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label in sorted(color_map.keys()):
        mask = colors_arr == label
        if mask.sum() == 0:
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1], s=point_size, alpha=alpha,
                   c=color_map[label], label=f"{label} (n={int(mask.sum())})",
                   edgecolors='none')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False,
              fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# =========================================================================
# Load minimal adata (no X)
# =========================================================================
log(f"loading {H5} (no X)")
a = ad.read_h5ad(H5, backed='r')
n_total = a.shape[0]
obs_df = a.obs.copy()
obsm = {k: np.asarray(a.obsm[k]) for k in ('X_glue', 'X_glue_harmony', 'X_glue_harmony_nosamp') if k in a.obsm}
a.file.close()
log(f"  n_cells={n_total}, obsm keys loaded: {list(obsm.keys())}")

# Make a lightweight adata for downstream use
ad_lite = sc.AnnData(X=np.zeros((n_total, 1), dtype=np.float32), obs=obs_df)
for k, v in obsm.items():
    ad_lite.obsm[k] = v


# =========================================================================
# PART 1: cell-level UMAP (subsample for speed)
# =========================================================================
log("PART 1: cell-level UMAP (subsample 60000 cells for speed)")
rng = np.random.default_rng(42)
sub_idx = rng.choice(n_total, size=min(60000, n_total), replace=False)
sub_batch    = ad_lite.obs['batch'].astype(str).values[sub_idx]
sub_modality = ad_lite.obs['modality'].astype(str).values[sub_idx]

for emb_key in ('X_glue', 'X_glue_harmony', 'X_glue_harmony_nosamp'):
    if emb_key not in ad_lite.obsm:
        log(f"  {emb_key} missing — skip")
        continue
    log(f"  UMAP on {emb_key}…")
    t0 = time.time()
    import umap
    Z2 = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3)\
              .fit_transform(ad_lite.obsm[emb_key][sub_idx])
    log(f"    fit in {time.time()-t0:.1f}s")
    plot_scatter(Z2, sub_batch, BATCH_COLORS,
                 f"{emb_key} — UMAP coloured by batch (n={len(sub_idx)} cells)",
                 f"{OUT}/cell_umap/{emb_key}_batch.png")
    plot_scatter(Z2, sub_modality, MOD_COLORS,
                 f"{emb_key} — UMAP coloured by modality",
                 f"{OUT}/cell_umap/{emb_key}_modality.png", alpha=0.5)

log("PART 1 done — 6 PNG written")


# =========================================================================
# PART 2: replicate the SampleDisco blocks and analyse per-block batch effect
# =========================================================================
log("PART 2: building A1, A2, A3, CMD blocks (sample level)")

# Use X_glue_harmony for clustering/A2/A3, X_glue_harmony_nosamp for CMD
CLUSTER_KEY = 'X_glue_harmony'
CMD_KEY     = 'X_glue_harmony_nosamp'
SAMPLE_COL  = 'sample'
CELLTYPE_COL = 'cell_type'
MODALITY_COL = 'modality'
BATCH_COL    = 'batch'

units, unit_cellids_d, unit_ids, unit_groups, unit_batches, all_cellids, Z_clust = \
    assemble_units(ad_lite, SAMPLE_COL, CLUSTER_KEY, modality_col=MODALITY_COL,
                   batch_col=BATCH_COL)
n_units = len(unit_ids)
cellid_idx = {c: i for i, c in enumerate(all_cellids)}
log(f"  {n_units} units (samples × modalities)")
log(f"  per-unit groups: {sorted(set(unit_groups))}  batches: {len(set(unit_batches))}")

ct = ad_lite.obs[CELLTYPE_COL].astype(str).values
unique_ct = sorted(set(ct))
K_c = len(unique_ct)
log(f"  K_c (cell types) = {K_c}")

# ---- A1: coarse one-hot composition ----
L1 = {c: i for i, c in enumerate(unique_ct)}
soft1 = np.zeros((Z_clust.shape[0], K_c), dtype=np.float32)
for i, c in enumerate(ct):
    soft1[i, L1[c]] = 1.0
unit_cellids_list = [unit_cellids_d[u] for u in unit_ids]
A1 = composition_per_unit(unit_cellids_list, soft1, cellid_idx)
log(f"  A1 shape={A1.shape}")

# ---- A2: K_med soft k-means ----
K_med  = min(120, max(2, Z_clust.shape[0] // 200))
log(f"  A2 K-means K={K_med}…")
km2 = MiniBatchKMeans(n_clusters=K_med, random_state=42, batch_size=4096,
                       n_init=5, max_iter=200).fit(Z_clust)
A2 = composition_per_unit(unit_cellids_list, soft_assign(Z_clust, km2.cluster_centers_),
                            cellid_idx)
log(f"  A2 shape={A2.shape}")

# ---- A3: K_fine soft k-means ----
K_fine = min(300, max(2, Z_clust.shape[0] // 100))
log(f"  A3 K-means K={K_fine}…")
km3 = MiniBatchKMeans(n_clusters=K_fine, random_state=43, batch_size=4096,
                       n_init=5, max_iter=200).fit(Z_clust)
A3 = composition_per_unit(unit_cellids_list, soft_assign(Z_clust, km3.cluster_centers_),
                            cellid_idx)
log(f"  A3 shape={A3.shape}")

# ---- CMD: LOO displacement on the SAMPLE-PRESERVED embedding ----
log(f"  CMD on {CMD_KEY}…")
Z_cmd = ad_lite.obsm[CMD_KEY]
cmd_units = [(uid, g, Z_cmd[[cellid_idx[c] for c in unit_cellids_d[uid] if c in cellid_idx]])
              for uid, g in zip(unit_ids, unit_groups)]
coarse_label_map = dict(zip(all_cellids, ct))
CMD = loo_cmd(cmd_units, unit_cellids_d, coarse_label_map,
              max_dim_per_cluster=8, seed=42, loo=True, verbose=False)
log(f"  CMD shape={CMD.shape}")

# ---- Per-block batch metrics ----
def batch_R2(block: np.ndarray, batches: np.ndarray) -> float:
    """Average per-PC R² when regressing each top-10 PC of `block` on the one-hot
    batch design. Closer to 1 = more batch-explained variance."""
    n_pc = min(10, block.shape[1], block.shape[0] - 1)
    if n_pc < 1: return float('nan')
    P = PCA(n_components=n_pc, random_state=0).fit_transform(block)
    cats = pd.get_dummies(batches, drop_first=True).values.astype(np.float32)
    if cats.shape[1] == 0: return float('nan')
    # Build OLS via lstsq; for each PC compute R² = 1 - SSR/SST
    r2s = []
    for k in range(P.shape[1]):
        y = P[:, k]
        beta, *_ = np.linalg.lstsq(cats, y - y.mean(), rcond=None)
        yhat = cats @ beta
        ssr = float(np.sum((y - y.mean() - yhat) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2s.append(1.0 - ssr / sst if sst > 0 else 0.0)
    return float(np.mean(r2s))


def asw_batch(block: np.ndarray, batches: np.ndarray) -> float:
    if len(set(batches)) < 2 or block.shape[0] < 3:
        return float('nan')
    return float(silhouette_score(block, batches, metric='euclidean'))


log("computing per-block batch metrics")
unit_batch_arr = np.asarray(unit_batches)
unit_modality_arr = np.asarray(unit_groups)

rows = []
for name, block in [('A1', A1), ('A2', A2), ('A3', A3), ('CMD', CMD)]:
    asw_b = asw_batch(block, unit_batch_arr)
    r2_b  = batch_R2(block,  unit_batch_arr)
    asw_m = asw_batch(block, unit_modality_arr)
    r2_m  = batch_R2(block,  unit_modality_arr)
    rows.append({
        'block': name, 'shape': f"{block.shape}",
        'ASW_batch':       round(asw_b, 4),
        'ASW_modality':    round(asw_m, 4),
        'mean_PC_R2_batch':    round(r2_b, 4),
        'mean_PC_R2_modality': round(r2_m, 4),
    })

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(f"{OUT}/block_analysis/block_metrics.csv", index=False)
log(f"\n{metrics_df.to_string(index=False)}")

# ---- Plot per-block top-2 PC scatters coloured by batch ----
for name, block in [('A1', A1), ('A2', A2), ('A3', A3), ('CMD', CMD)]:
    P = PCA(n_components=2, random_state=0).fit_transform(block)
    plot_scatter(P, unit_batch_arr, BATCH_COLORS,
                 f"{name} (PC1, PC2) coloured by batch  —  ASW(batch)={asw_batch(block, unit_batch_arr):+.3f}",
                 f"{OUT}/block_analysis/{name}_batch_pca.png",
                 point_size=22, alpha=0.85)
    plot_scatter(P, unit_modality_arr, MOD_COLORS,
                 f"{name} (PC1, PC2) coloured by modality",
                 f"{OUT}/block_analysis/{name}_modality_pca.png",
                 point_size=22, alpha=0.85)

# Quick markdown summary
sorted_by_batch = metrics_df.sort_values('ASW_batch', ascending=False)
worst_block = sorted_by_batch.iloc[0]['block']
with open(f"{OUT}/block_analysis/summary.md", 'w') as f:
    f.write("# Block-by-block batch / modality analysis\n\n")
    f.write(f"Dataset: unpaired_test  ({n_units} units, {K_c} cell types, "
            f"{len(set(unit_batch_arr))} batches)\n\n")
    f.write("## Per-block metrics (1 row per block)\n\n")
    f.write(metrics_df.to_markdown(index=False))
    f.write(f"\n\n**Block carrying the most batch separation (ASW_batch):** "
            f"`{worst_block}`\n")
    f.write("\nBlock dim notes:\n")
    f.write(f"- A1 = one-hot per-(sample, modality) composition over {K_c} cell-types\n")
    f.write(f"- A2 = soft k-means composition at K_med={K_med}\n")
    f.write(f"- A3 = soft k-means composition at K_fine={K_fine}\n")
    f.write("- CMD = per-(modality, cluster) LOO displacement on X_glue_harmony_nosamp\n")
log(f"\nSummary written to {OUT}/block_analysis/summary.md")
log("DONE")
