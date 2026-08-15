#!/usr/bin/env python
"""Generalize fig2d (per-block cosine-similarity panel) to any SampleDisco dataset.

For one dataset it re-derives the four sample-embedding blocks (A1 coarse
composition, A2 medium k-means, A3 fine k-means, RMD displacement) plus the
final Harmony-corrected embedding, exactly as the production pipeline does, then
renders the five-panel cosine-similarity figure once per annotation label.

Block recipe and panel aesthetics are copied from
``figure/figure2/illustration/{build_blocks,make_plots}.py`` so the new panels
are directly comparable to the published ENCODE one.

Usage:  build_similarity_panel.py --dataset heart --outroot <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

sys.path.insert(0, "/users/hjiang/GenoDistance/code/src")

from sampledisco.sample_embedding.blocks import (  # noqa: E402
    assemble_units,
    build_emb_from_blocks,
    composition_per_unit,
    derive_weights,
    frobenius_stack,
    loo_rmd,
    soft_assign,
)

# Production knobs — identical to the ENCODE fig2d build.
MEDIUM_K, FINE_K, RMD_DIM, RMD_WEIGHT, PCA_N, SEED = 120, 300, 8, 0.60, 10, 42

R = "/dcs07/hongkai/data/harry/result"

# --------------------------------------------------------------------------- #
# Per-dataset recipe                                                           #
# --------------------------------------------------------------------------- #
# labels: (display_name, obs column, order) where order is
#   "count"   — groups sorted by descending size (the ENCODE/tissue convention)
#   "natural" — groups sorted by their own value (ordinal: severity, age, month)
DATASETS = {
    "ENCODE": dict(
        adata=f"{R}/multi_omics_ENCODE/multiomics/preprocess/adata_sample.h5ad",
        comp_key="Z_comp", rmd_key="X_glue", harmonize_comp=True,
        sample_col="sample", celltype_col="cell_type",
        modality_col="modality", batch_col=None, lite=True,
        meta_csv="/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
        labels=[("tissue", "tissue", "count")],
    ),
    "heart": dict(
        adata=f"{R}/multi_omics_heart/SD/multiomics/preprocess/atac_rna_integrated.h5ad",
        comp_key="Z_comp", rmd_key="X_glue", harmonize_comp=True,
        sample_col="sample", celltype_col="cell_type",
        modality_col="modality", batch_col=None, lite=True, meta_csv=None,
        labels=[("disease_state", "disease_state", "count"),
                ("modality", "modality", "count")],
    ),
    "retina": dict(
        adata=f"{R}/multi_omics_eye/benchmark_retina/retina/preprocess/atac_rna_integrated.h5ad",
        comp_key="Z_comp", rmd_key="X_glue", harmonize_comp=True,
        sample_col="sample", celltype_col="cell_type",
        modality_col="modality", batch_col=None, lite=True, meta_csv=None,
        labels=[("age", "age", "natural"), ("modality", "modality", "count")],
    ),
    "lutea": dict(
        adata=f"{R}/multi_omics_eye/benchmark_lutea/lutea/preprocess/atac_rna_integrated.h5ad",
        comp_key="Z_comp", rmd_key="X_glue", harmonize_comp=True,
        sample_col="sample", celltype_col="cell_type",
        modality_col="modality", batch_col=None, lite=True, meta_csv=None,
        labels=[("age", "age", "natural"), ("modality", "modality", "count")],
    ),
}

# The COVID benchmark exists as six nested subsamples; all share one recipe.
for _n in (25, 50, 100, 200, 279, 400):
    DATASETS[f"covid_{_n}"] = dict(
        adata=f"{R}/Benchmark_covid/covid_{_n}_sample/rna/preprocess/adata_cell.h5ad",
        comp_key="X_pca_harmony", rmd_key="X_pca_harmony_nosamp",
        sample_col="sample", celltype_col="cell_type",
        modality_col=None, batch_col="batch", lite=True, meta_csv=None,
        labels=[("severity", "severity", "natural"),
                ("sev.level", "sev.level", "natural"),
                ("study", "batch", "count")],
    )

# Per-study splits of the full 405-sample COVID cohort. Each study gets its own
# embedding built from its cells alone, so the composition anchors and the RMD
# leave-one-out reference are within-study — not borrowed from the pooled run.
# Counts in covid_400: Su 249, SS2 56, SS1 25, Lee 14, Zhu 14, Wilk 13,
# Aruna 12, Yu 9, Wen 5, Mudd 4, Guo 2, Silvin 2. Guo and Silvin are excluded:
# a 2x2 similarity matrix is not interpretable and LOO-RMD degenerates at n=2.
COVID_STUDIES = ("Su", "SS2", "SS1", "Lee", "Zhu", "Wilk", "Aruna", "Yu", "Wen", "Mudd")
for _s in COVID_STUDIES:
    DATASETS[f"covid_study_{_s}"] = dict(
        adata=f"{R}/Benchmark_covid/covid_400_sample/rna/preprocess/adata_cell.h5ad",
        comp_key="X_pca_harmony", rmd_key="X_pca_harmony_nosamp",
        sample_col="sample", celltype_col="cell_type",
        modality_col=None, batch_col=None, lite=True, meta_csv=None,
        subset=("batch", _s),
        labels=[("severity", "severity", "natural"),
                ("sev.level", "sev.level", "natural"),
                ("sex", "sex", "count")],
    )

DATASETS |= {
    "long_covid": dict(
        # No sample-preserved Harmony output on this object; X_pca (pre-Harmony)
        # is the only sample-preserved view, so it stands in as the RMD source
        # (same substitution the ENCODE ablation uses).
        adata=f"{R}/long_covid/rna/preprocess/adata_cell.h5ad",
        comp_key="X_pca_harmony", rmd_key="X_pca",
        sample_col="sample", celltype_col="cell_type",
        modality_col=None, batch_col=None, lite=True, meta_csv=None,
        labels=[("LC_vs_Recovered", "LC/Recovered", "count"),
                ("month", "month", "natural"),
                ("sex", "Sex", "count")],
    ),
    "health_aging": dict(
        adata=f"{R}/health_aging_PBMC/round1_batch/preprocess/adata_preprocessed.h5ad",
        comp_key="Z_clust", rmd_key="Z_cmd",
        sample_col="Tube_id", celltype_col="Cluster_names",
        modality_col=None, batch_col="Batch", lite=True, meta_csv=None,
        labels=[("age_group", "Age_group", "natural"),
                ("batch", "Batch", "count"),
                ("sex", "Sex", "count")],
    ),

    # ---------------- ATAC single-omics ----------------
    "covid_ATAC": dict(
        adata=f"{R}/Benchmark_covid/ATAC/preprocess/adata_preprocessed.h5ad",
        comp_key="X_lsi_harmony", rmd_key="X_lsi_harmony_nosamp",
        sample_col="sample", celltype_col="cell_type",
        modality_col=None, batch_col=None, lite=True, meta_csv=None,
        labels=[("sev.level", "sev.level", "natural"),
                ("age", "Age", "natural"), ("sex", "Sex", "count")],
    ),

    # ---------------- RNA-only arms of the multi-omics cohorts ----------------
    # batch_col stays None on both: their 'batch' column is one level per tissue
    # (ENCODE) or per sample (heart), so using it as the RMD comparison group
    # would either leak the plotted label into the reference or make every unit
    # its own group.
    "ENCODE_rna": dict(
        adata=f"{R}/archived_benchmarks/Benchmark_ENCODE_rna/rna/preprocess/adata_cell.h5ad",
        comp_key="X_pca_harmony", rmd_key="X_pca",
        sample_col="sample", celltype_col="cell_type",
        modality_col=None, batch_col=None, lite=True, meta_csv=None,
        labels=[("tissue", "tissue", "count")],
    ),
    "heart_rna": dict(
        adata=f"{R}/archived_benchmarks/Benchmark_heart_rna/rna/preprocess/adata_cell.h5ad",
        comp_key="X_pca_harmony", rmd_key="X_pca",
        sample_col="sample", celltype_col="cell_type",
        modality_col=None, batch_col=None, lite=True, meta_csv=None,
        labels=[("disease_state", "disease_state", "count")],
    ),
    # ENCODE RNA+ATAC as separate units WITHOUT scGLUE (PCA/Harmony only) —
    # the pre-integration counterpart of the "ENCODE" entry above.
    "ENCODE_rna_atac": dict(
        adata=f"{R}/Benchmark_multiomics/adata_cell.h5ad",
        comp_key="X_pca_harmony", rmd_key="X_pca",
        sample_col="sample", celltype_col="cell_type",
        modality_col="modality", batch_col=None, lite=True, meta_csv=None,
        labels=[("tissue", "tissue", "count"),
                ("modality", "modality", "count")],
    ),

    # ---------------- unpaired multi-omics (COVID RNA + ATAC) ----------------
    "unpaired_diemb": dict(
        adata=f"{R}/multi_omics_unpaired_diemb/multiomics/preprocess/adata_preprocessed.h5ad",
        comp_key="Z_clust", rmd_key="Z_cmd",
        sample_col="sample", celltype_col="cell_type",
        modality_col="modality", batch_col=None, lite=True, meta_csv=None,
        labels=[("sev.level", "sev.level", "natural"),
                ("study", "batch", "count"),
                ("modality", "modality", "count"),
                ("sex", "sex", "count")],
    ),
    "unpaired_paper": dict(
        adata=f"{R}/multi_omics_unpaired_paper/multiomics/preprocess/atac_rna_integrated.h5ad",
        comp_key="X_glue_harmony", rmd_key="X_glue",
        sample_col="sample", celltype_col="cell_type",
        modality_col="modality", batch_col=None, lite=True, meta_csv=None,
        labels=[("sev.level", "sev.level", "natural"),
                ("study", "batch", "count"),
                ("modality", "modality", "count")],
    ),
    "unpaired_test": dict(
        adata=f"{R}/multi_omics_unpaired_test/multiomics/preprocess/atac_rna_integrated.h5ad",
        comp_key="X_glue_harmony", rmd_key="X_glue_harmony_nosamp",
        sample_col="sample", celltype_col="cell_type",
        modality_col="modality", batch_col=None, lite=True, meta_csv=None,
        labels=[("sev.level", "sev.level", "natural"),
                ("study", "batch", "count"),
                ("modality", "modality", "count")],
    ),

    # ---------------- 1M-scBloodNL (stimulation time-course) ----------------
    # Sample unit is 'id' = donor x stimulus x timepoint ('D100_CA_24h');
    # 'assignment' is the donor alone. A stimulation experiment is the natural
    # stress test for RMD: 3 h of CA/MTB/PA changes within-cell-type state far
    # more than it changes cell-type proportions.
    "scBloodNL_V3": dict(
        adata=f"{R}/1M-scBloodNL/V3/rna/preprocess/adata_cell.h5ad",
        comp_key="X_pca_harmony", rmd_key="X_pca",
        sample_col="id", celltype_col="cell_type",
        modality_col=None, batch_col="seq_batch", lite=True, meta_csv=None,
        labels=[("stimulation", "stimulation_conditions", "count"),
                ("timepoint", "timepoint", "natural"),
                ("seq_lane", "seq_lane", "count")],
    ),
    "scBloodNL_V1": dict(
        adata=f"{R}/1M-scBloodNL/rna/preprocess/adata_cell.h5ad",
        comp_key="X_pca_harmony", rmd_key="X_pca",
        sample_col="id", celltype_col="cell_type",
        modality_col=None, batch_col="chem", lite=True, meta_csv=None,
        labels=[("stimulation", "stimulation_conditions", "count"),
                ("timepoint", "timepoint", "natural"),
                ("chem", "chem", "count")],
    ),

    # Finer 42-cluster cell typing of the same 35 long-COVID samples.
    "long_covid_fine": dict(
        adata=f"{R}/long_covid/analysis/preprocess/adata_cell.h5ad",
        comp_key="X_pca_harmony", rmd_key="X_pca",
        sample_col="sample", celltype_col="cell_type",
        modality_col=None, batch_col=None, lite=True, meta_csv=None,
        labels=[("LC_vs_Recovered", "LC/Recovered", "count"),
                ("month", "month", "natural"),
                ("sex", "Sex", "count")],
    ),
}

# Diagnostic: same V3 object, but the LOO comparison group is dropped. V3 splits
# 240 units across 31 seq_batch levels (~8 units per group), which may make the
# leave-one-out reference too noisy to form a usable displacement.
DATASETS["scBloodNL_V3_nobatch"] = dict(
    adata=f"{R}/1M-scBloodNL/V3/rna/preprocess/adata_cell.h5ad",
    comp_key="X_pca_harmony", rmd_key="X_pca",
    sample_col="id", celltype_col="cell_type",
    modality_col=None, batch_col=None, lite=True, meta_csv=None,
    labels=[("stimulation", "stimulation_conditions", "count"),
            ("timepoint", "timepoint", "natural")],
)

BLOCK_COLORS = {"A1": "#1f77b4", "A2": "#ff7f0e", "A3": "#2ca02c", "RMD": "#d62728"}
BLOCK_LONG = {
    "A1": "A1 · coarse cell-type composition",
    "A2": "A2 · medium k-means composition",
    "A3": "A3 · fine k-means composition",
    "RMD": "RMD · within-celltype state displacement",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Load                                                                         #
# --------------------------------------------------------------------------- #

def load_adata(cfg):
    """Full read, or an obsm+obs-only view for objects too large to hold in RAM.

    ``cfg['subset']`` = (obs column, value) restricts to those cells first, which
    is how the per-study COVID splits are built out of the one 405-sample object.
    """
    path = cfg["adata"]
    label_cols = [c for _, c, _ in cfg["labels"]]
    subset = cfg.get("subset")
    if not cfg["lite"]:
        _log(f"reading {path}")
        a = ad.read_h5ad(path)
    else:
        _log(f"reading LITE (obs + obsm only) {path}")
        ab = ad.read_h5ad(path, backed="r")
        keep = [cfg["sample_col"], cfg["celltype_col"]] + label_cols
        for extra in (cfg["batch_col"], cfg["modality_col"]):
            if extra:
                keep.append(extra)
        if subset:
            keep.append(subset[0])
        keep = [c for c in dict.fromkeys(keep) if c in ab.obs.columns]
        a = ad.AnnData(obs=ab.obs[keep].astype(str).copy())
        a.obs_names = ab.obs_names.astype(str)
        # comp_key may not exist on disk yet (harmonize_comp derives it below).
        for k in {cfg["comp_key"], cfg["rmd_key"]}:
            if k in ab.obsm:
                a.obsm[k] = np.asarray(ab.obsm[k], dtype=np.float32)
            elif not cfg.get("harmonize_comp"):
                raise SystemExit(f"obsm[{k!r}] missing from {path}")
        ab.file.close()

    if subset:
        col, val = subset
        mask = a.obs[col].astype(str).values == str(val)
        if mask.sum() == 0:
            raise SystemExit(f"subset {col}=={val} matched no cells")
        a = a[mask].copy()
        _log(f"subset {col}=={val}: {a.n_obs} cells, "
             f"{a.obs[cfg['sample_col']].nunique()} samples")

    # Objects that carry only raw X_glue have no sample-REMOVED view on disk.
    # The paper's z^comp is one Harmony pass on X_glue; derive it in memory
    # (nothing is written back to the source h5ad).
    if cfg.get("harmonize_comp"):
        from sampledisco.preparation.multi_omics_batch_correction import harmonize_xglue
        _log(f"deriving Z_comp = Harmony({cfg['rmd_key']}) on "
             f"sample_col={cfg['sample_col']!r}")
        harmonize_xglue(
            a, sample_col=cfg["sample_col"], batch_col=None,
            in_key=cfg["rmd_key"], out_key="Z_comp",
            use_gpu=False, max_iter=30, random_state=SEED, verbose=True,
        )
        if "Z_comp" not in a.obsm:
            raise SystemExit("harmonize_xglue did not produce Z_comp")
    return a


# --------------------------------------------------------------------------- #
# Blocks                                                                       #
# --------------------------------------------------------------------------- #

def build_blocks(adata, cfg):
    units, unit_cellids, unit_ids, unit_groups, unit_batches, all_cellids, Z = \
        assemble_units(adata, cfg["sample_col"], cfg["comp_key"],
                       modality_col=cfg["modality_col"], batch_col=cfg["batch_col"])
    cellid_idx = {c: i for i, c in enumerate(all_cellids)}
    ucl = [unit_cellids[u] for u in unit_ids]
    _log(f"units={len(unit_ids)}  cells={Z.shape[0]}  d={Z.shape[1]}")

    ct = adata.obs[cfg["celltype_col"]].astype(str).values
    uniq = sorted(set(ct))
    K_c = len(uniq)
    L1 = {c: i for i, c in enumerate(uniq)}
    soft1 = np.zeros((Z.shape[0], K_c), dtype=np.float32)
    for i, c in enumerate(ct):
        soft1[i, L1[c]] = 1.0
    A1 = composition_per_unit(ucl, soft1, cellid_idx)
    _log(f"A1 {A1.shape}")

    K_med = min(MEDIUM_K, max(2, Z.shape[0] // 200))
    km = MiniBatchKMeans(n_clusters=K_med, random_state=SEED, batch_size=4096,
                         n_init=5, max_iter=200).fit(Z)
    A2 = composition_per_unit(ucl, soft_assign(Z, km.cluster_centers_), cellid_idx)
    _log(f"A2 {A2.shape}")

    K_fine = min(FINE_K, max(2, Z.shape[0] // 100))
    km = MiniBatchKMeans(n_clusters=K_fine, random_state=SEED + 1, batch_size=4096,
                         n_init=5, max_iter=200).fit(Z)
    A3 = composition_per_unit(ucl, soft_assign(Z, km.cluster_centers_), cellid_idx)
    _log(f"A3 {A3.shape}")

    Zr = np.asarray(adata.obsm[cfg["rmd_key"]], dtype=np.float32)
    rmd_units = [(uid, grp, Zr[[cellid_idx[c] for c in unit_cellids[uid]]])
                 for uid, grp in zip(unit_ids, unit_groups)]
    RMD = loo_rmd(rmd_units, unit_cellids, dict(zip(all_cellids, ct)),
                  max_dim_per_cluster=RMD_DIM, seed=SEED, loo=True, verbose=True)
    _log(f"RMD {RMD.shape}")

    blocks = [A1, A2, A3, RMD]
    weights = derive_weights(K_c, K_med, K_fine, rmd_weight=RMD_WEIGHT,
                             n_blocks=len(blocks))
    F = frobenius_stack(blocks, weights)
    n_pc = min(PCA_N, F.shape[0] - 1, F.shape[1])
    Fp = PCA(n_components=n_pc, random_state=SEED).fit_transform(F)

    Zc = build_emb_from_blocks(
        blocks, weights, unit_ids=unit_ids, unit_groups=unit_groups,
        unit_batches=unit_batches, pca_components=PCA_N,
        batch_method="harmony", seed=SEED, verbose=True,
    ).values.astype(np.float32)
    _log(f"final embedding {Zc.shape}")

    meta = dict(K_c=int(K_c), K_med=int(K_med), K_fine=int(K_fine),
                n_units=len(unit_ids), n_cells=int(Z.shape[0]),
                weights={"A1": float(weights[0]), "A2": float(weights[1]),
                         "A3": float(weights[2]), "RMD": float(weights[3])},
                comp_key=cfg["comp_key"], rmd_key=cfg["rmd_key"],
                adata=cfg["adata"], seed=SEED)
    return dict(A1=A1, A2=A2, A3=A3, RMD=RMD, Fp=Fp, Zc=Zc,
                unit_ids=unit_ids, unit_groups=unit_groups,
                unit_batches=unit_batches, meta=meta)


def unit_labels(adata, cfg, unit_ids, unit_groups):
    """Per-unit value of each annotation column (majority vote over its cells)."""
    sample_arr = adata.obs[cfg["sample_col"]].astype(str).values
    modality_arr = (adata.obs[cfg["modality_col"]].astype(str).values
                    if cfg["modality_col"] else None)

    # unit_id -> row mask. assemble_units builds mo uids as f"{bio}_{modality}".
    df = pd.DataFrame({"sample": sample_arr})
    if modality_arr is not None:
        df["modality"] = modality_arr
        bio = df["sample"].copy()
        for m in df["modality"].unique():
            suf = f"_{m}"
            bio = bio.mask(df["modality"].eq(m) & bio.str.endswith(suf),
                           bio.str.slice(0, -len(suf)))
        df["uid"] = bio + "_" + df["modality"]
        df["bio"] = bio
    else:
        df["uid"] = df["sample"]
        df["bio"] = df["sample"]

    out = pd.DataFrame({"unit": unit_ids, "group": unit_groups})
    for name, col, _ in cfg["labels"]:
        if col in adata.obs.columns:
            vals = adata.obs[col].astype(str).values
            maj = (pd.DataFrame({"uid": df["uid"], "v": vals})
                   .groupby("uid")["v"]
                   .agg(lambda s: s.value_counts().idxmax()))
            out[name] = out["unit"].map(maj).fillna("unknown")
        elif cfg["meta_csv"]:
            # The ENCODE sheet is keyed by the modality-suffixed unit id; other
            # sheets may be keyed by the biological sample. Try both.
            meta = pd.read_csv(cfg["meta_csv"]).set_index("sample")
            uid2bio = dict(zip(df["uid"], df["bio"]))
            vals = []
            for u in out["unit"]:
                k = u if u in meta.index else uid2bio.get(u)
                vals.append(str(meta.loc[k, col]) if k in meta.index else "unknown")
            out[name] = vals
        else:
            out[name] = "unknown"
    return out


# --------------------------------------------------------------------------- #
# Panel (aesthetics verbatim from make_plots.fig_similarity_panel)             #
# --------------------------------------------------------------------------- #

def _cos_sim(M: np.ndarray) -> np.ndarray:
    Mc = M - M.mean(axis=0, keepdims=True)
    n = np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-8
    Mn = Mc / n
    return Mn @ Mn.T


def block_shares(d, weights):
    """Each block's share of the top-10 PC energy of the fused matrix.

    Without this the five panels read as equal contributors, when in practice
    A1 carries 65-93% and RMD 0.3-3% at the default w_RMD = 0.60.
    """
    blocks = [d["A1"], d["A2"], d["A3"], d["RMD"]]
    F = frobenius_stack(blocks, weights)
    n_pc = min(PCA_N, F.shape[0] - 1, F.shape[1])
    pca = PCA(n_components=n_pc, random_state=SEED).fit(F)
    edges = np.cumsum([0] + [b.shape[1] for b in blocks])
    energy = np.array([
        float((pca.explained_variance_[:, None]
               * pca.components_[:, edges[i]:edges[i + 1]] ** 2).sum())
        for i in range(4)
    ])
    energy = energy / energy.sum()
    return dict(zip(("A1", "A2", "A3", "RMD"), energy))


def _sort_key(order):
    """natural: sort by numeric value when the label parses as a number."""
    def key(v):
        try:
            return (0, float(v), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(v))
    return key if order == "natural" else None


def _group_order(units, col, order):
    counts = units[col].value_counts()
    if order == "natural":
        return sorted(units[col].unique(), key=_sort_key("natural"))
    return sorted(units[col].unique(), key=lambda t: (-counts[t], t))


def _row_order(units, col, order, has_modality):
    rows = []
    for g in _group_order(units, col, order):
        sub = units[units[col] == g]
        sub = sub.sort_values(["group", "unit"] if has_modality else ["unit"])
        rows.extend(sub.index.tolist())
    return np.array(rows, dtype=int)


def _palette(units, col, order):
    groups = _group_order(units, col, order)
    base = plt.get_cmap("tab10").colors + plt.get_cmap("Set3").colors
    return {g: base[i % len(base)] for i, g in enumerate(groups)}, groups


def _runs(units, col, order_idx):
    runs, cur, start = [], None, 0
    series = [units.loc[i, col] for i in order_idx]
    for k, t in enumerate(series):
        if cur is None:
            cur, start = t, k
        elif t != cur:
            runs.append((cur, start, k))
            cur, start = t, k
    runs.append((cur, start, len(series)))
    return runs


def similarity_panel(d, units, label_name, label_order, dataset, out_png, shares=None,
                     alpha=None):
    order = _row_order(units, label_name, label_order, "group" in units)
    colors, groups = _palette(units, label_name, label_order)
    row_colors = np.array([colors[units.loc[i, label_name]] for i in order])
    runs = _runs(units, label_name, order)

    panels = [("A1", d["A1"]), ("A2", d["A2"]), ("A3", d["A3"]),
              ("RMD", d["RMD"]), ("Combined", d["Zc"])]
    n = len(panels)

    rdbu = LinearSegmentedColormap.from_list(
        "rdbu_rich",
        ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
         "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"], N=256)

    # Geometry: matrices are drawn with aspect="auto" so they always fill their
    # axes box, and the figure height is chosen to make that box square. Letting
    # imshow enforce aspect="equal" instead collapses the matrix to a sliver
    # whenever the divider-managed axes box is wider than it is tall (which is
    # what happens for any dataset with fewer units than ENCODE's 88).
    n_leg_rows = int(np.ceil(len(groups) / 5))
    bottom_in = 1.15 + 0.22 * n_leg_rows          # colorbar + legend rows
    top_in = 1.15                                  # suptitle + block strips
    fig_w = 4.2 * n + 1.0
    panel_w = 0.965 * fig_w / (n + 0.20 * (n - 1))
    panel_h = panel_w                              # square matrices
    fig_h = panel_h * 1.10 + top_in + bottom_in    # 1.10 leaves room for strips

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(1, n, wspace=0.20, left=0.025, right=0.99,
                             top=1.0 - top_in / fig_h,
                             bottom=bottom_in / fig_h)

    last_im = None
    for col, (name, M) in enumerate(panels):
        S = _cos_sim(M)[np.ix_(order, order)]
        ax = fig.add_subplot(outer[0, col])
        v = np.percentile(np.abs(S), 99)
        last_im = ax.imshow(S, cmap=rdbu, vmin=-v, vmax=v,
                            interpolation="nearest", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])

        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="3.5%", pad=0.05, sharex=ax)
        ax_top.imshow(row_colors[None, :, :], aspect="auto", interpolation="nearest")
        ax_top.set_xticks([]); ax_top.set_yticks([])
        for sp in ax_top.spines.values():
            sp.set_visible(False)

        ax_acc = divider.append_axes("top", size="3.5%", pad=0.03, sharex=ax)
        color = BLOCK_COLORS.get(name, "#444444")
        # Must be n_units wide, not a fixed 100: the axis is x-shared with the
        # matrix, so a wider image stretches the shared xlim and squeezes the
        # matrix into n_units/width of the panel.
        ax_acc.imshow(np.ones((1, len(order), 4)) * mpl.colors.to_rgba(color),
                      aspect="auto", interpolation="nearest")
        ax_acc.set_xticks([]); ax_acc.set_yticks([])
        for sp in ax_acc.spines.values():
            sp.set_visible(False)
        title = BLOCK_LONG.get(name, "Combined · final 10-D embedding")
        if shares and name in shares:
            title += f"  [{shares[name] * 100:.1f}%]"
        elif name == "Combined":
            title += "  [100%]"
        ax_acc.set_title(title, fontsize=9.5, color=color, pad=4,
                         fontweight="bold", loc="left")

        ax_left = divider.append_axes("left", size="3.5%", pad=0.05, sharey=ax)
        ax_left.imshow(row_colors[:, None, :], aspect="auto", interpolation="nearest")
        ax_left.set_xticks([]); ax_left.set_yticks([])
        for sp in ax_left.spines.values():
            sp.set_visible(False)

        for _, s, _e in runs:
            if s > 0:
                ax.axhline(s - 0.5, color="white", linewidth=0.6, alpha=0.9)
                ax.axvline(s - 0.5, color="white", linewidth=0.6, alpha=0.9)
        for t, s, e in runs:
            if e - s < 2:
                continue
            ax.add_patch(mpatches.Rectangle((s - 0.5, s - 0.5), e - s, e - s,
                                            fill=False, edgecolor=colors[t],
                                            linewidth=1.6, alpha=0.95))
        for sp in ax.spines.values():
            sp.set_color("#444444")
            sp.set_linewidth(0.8)

    cbar_ax = fig.add_axes([0.34, (bottom_in - 0.62) / fig_h, 0.32, 0.12 / fig_h])
    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="horizontal")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_title("cosine similarity — each panel scaled independently "
                      "(99th pctile of |S|), so colour intensity is NOT "
                      "comparable across panels",
                      fontsize=9.5, pad=4)

    counts = units[label_name].value_counts()
    handles = [mpatches.Patch(color=colors[g], label=f"{g}  (n={counts[g]})")
               for g in groups]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.05 / fig_h),
               ncol=min(5, len(groups)), frameon=False, fontsize=9)

    fig.suptitle(
        f"{dataset} — pairwise cosine similarity of samples, each block alone vs "
        f"the combined embedding  (rows grouped by {label_name})\n"
        f"[%] = that block's share of the top-10 PC energy of the fused "
        f"embedding, at w_RMD = {RMD_WEIGHT if alpha is None else alpha:.3g}",
        fontsize=13, y=1.0 - 0.30 / fig_h, fontweight="bold")

    fig.savefig(out_png)
    plt.close(fig)
    _log(f"wrote {out_png}")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--outroot", default="/dcs07/hongkai/data/claude/fig2d_similarity")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    outdir = os.path.join(args.outroot, args.dataset)
    cache = os.path.join(outdir, "_cache")
    os.makedirs(cache, exist_ok=True)

    adata = load_adata(cfg)
    d = build_blocks(adata, cfg)
    units = unit_labels(adata, cfg, d["unit_ids"], d["unit_groups"])

    for k in ("A1", "A2", "A3", "RMD", "Fp", "Zc"):
        np.save(os.path.join(cache, f"{k}.npy"), d[k])
    units.to_csv(os.path.join(cache, "units.csv"), index=False)
    with open(os.path.join(cache, "block_meta.json"), "w") as fh:
        json.dump(d["meta"], fh, indent=2)

    weights = [d["meta"]["weights"][b] for b in ("A1", "A2", "A3", "RMD")]
    shares = block_shares(d, weights)
    _log("block share of top-10 PC energy: "
         + ", ".join(f"{k}={v * 100:.1f}%" for k, v in shares.items()))

    for name, _col, order in cfg["labels"]:
        if units[name].nunique() < 2:
            _log(f"skip label {name!r}: only one distinct value")
            continue
        similarity_panel(d, units, name, order, args.dataset,
                         os.path.join(outdir, f"fig2d_similarity_{args.dataset}_{name}.png"),
                         shares=shares)
    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
