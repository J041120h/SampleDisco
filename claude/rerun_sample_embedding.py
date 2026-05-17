"""
Re-run only the sample-embedding step for previously processed datasets.

Blocks (run sequentially):
  1) COVID RNA (×6 sample sizes): default-α + autotuned, reusing existing
     adata_cell.h5ad. Adds X_pca_harmony_nosamp via a harmony pass on 'batch'
     before the embedding, persisted into the original adata_cell.h5ad.
  2) COVID ATAC: full pipeline from scratch (preprocess + cell-type +
     default-α + tuned + a 2-D PCA viz colored by sev.level).
  3) Multi-omics (ENCODE, Lutea, Retina, Heart): default-α only, reusing the
     existing GLUE-integrated atac_rna_integrated.h5ad. Plus dimension
     association analysis on the resulting pseudo-adata.
  4) Unpaired (multi_omics_unpaired_paper): autotuned only, reusing the
     existing GLUE atac_rna_integrated.h5ad. Plus association analysis.

Outputs go into new sub-directories under each existing result folder
(named `sampledisco_default` and `sampledisco_tuned`); existing files are
not touched.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

CODE_DIR = "/users/hjiang/GenoDistance/code"
sys.path.insert(0, CODE_DIR)

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sample_embedding import compute_sample_embedding
from sample_embedding.sample_embedding import build_sample_adata
from parameter_selection.autotune import run_autotune
from sample_association.association import run_dimension_association_analysis


# --------------------------------------------------------------------------- #
# Path constants                                                              #
# --------------------------------------------------------------------------- #
COVID_BENCH_ROOT = "/dcs07/hongkai/data/harry/result/Benchmark_covid"
COVID_SAMPLE_SIZES = [25, 50, 100, 200, 279, 400]

ATAC_RAW = "/dcl01/hongkai/data/data/hjiang/Data/ATAC.h5ad"
ATAC_META = "/dcl01/hongkai/data/data/hjiang/Data/ATAC_Metadata.csv"
ATAC_OUT = f"{COVID_BENCH_ROOT}/ATAC"

MULTIOMICS = [
    ("ENCODE", "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/preprocess/atac_rna_integrated.h5ad",
                "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics"),
    ("Lutea",  "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/preprocess/atac_rna_integrated.h5ad",
                "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea"),
    ("Retina", "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/preprocess/atac_rna_integrated.h5ad",
                "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina"),
    ("Heart",  "/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/preprocess/atac_rna_integrated.h5ad",
                "/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics"),
]

UNPAIRED_H5 = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_paper/multiomics/preprocess/atac_rna_integrated.h5ad"
UNPAIRED_OUT = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_paper/multiomics"


# --------------------------------------------------------------------------- #
# Logging helpers                                                             #
# --------------------------------------------------------------------------- #
def _hdr(msg: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(msg, flush=True)
    print("=" * 78, flush=True)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Block 1: COVID RNA                                                          #
# --------------------------------------------------------------------------- #
def _ensure_pca_harmony_nosamp(adata, batch_keys, num_iter: int = 30) -> bool:
    """Compute X_pca_harmony_nosamp from X_pca with `batch_keys` (excluding
    the sample column). Returns True if added (i.e., wasn't already there)."""
    if "X_pca_harmony_nosamp" in adata.obsm:
        _log("X_pca_harmony_nosamp already present — skip recompute")
        return False
    if "X_pca" not in adata.obsm:
        raise KeyError("X_pca missing from adata.obsm; cannot run harmony pass.")
    from harmony import harmonize
    _log(f"Running harmony pass (no-sample) with batch_keys={batch_keys}, "
         f"max_iter={num_iter}")
    Z = harmonize(
        np.asarray(adata.obsm["X_pca"], dtype=np.float32),
        adata.obs,
        batch_key=batch_keys,
        max_iter_harmony=num_iter,
    )
    adata.obsm["X_pca_harmony_nosamp"] = np.asarray(Z, dtype=np.float32)
    _log(f"Added X_pca_harmony_nosamp shape={adata.obsm['X_pca_harmony_nosamp'].shape}")
    return True


def run_covid_rna_one(size: int) -> None:
    base = f"{COVID_BENCH_ROOT}/covid_{size}_sample/rna"
    cell_h5 = f"{base}/preprocess/adata_cell.h5ad"
    if not os.path.exists(cell_h5):
        _log(f"SKIP covid_{size}_sample — missing {cell_h5}")
        return

    _hdr(f"COVID RNA — size={size}")
    _log(f"Loading {cell_h5}")
    adata = sc.read(cell_h5)
    _log(f"adata: shape={adata.shape}, obsm={list(adata.obsm.keys())}, "
         f"n_samples={adata.obs['sample'].nunique()}, "
         f"n_batches={adata.obs['batch'].nunique() if 'batch' in adata.obs else 'NA'}")

    if "cell_type" not in adata.obs.columns and "celltype" in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs["celltype"].astype(str)

    added = _ensure_pca_harmony_nosamp(adata, batch_keys=["batch"])
    if added:
        _log(f"Re-saving {cell_h5} with X_pca_harmony_nosamp")
        sc.write(cell_h5, adata)

    # ----- Default-α -----
    out_default = f"{base}/sampledisco_default"
    os.makedirs(out_default, exist_ok=True)
    _log(f"Default-α sample embedding → {out_default}")
    compute_sample_embedding(
        adata, out_default,
        sample_col="sample",
        celltype_col="cell_type",
        cluster_emb_key="X_pca_harmony",
        cmd_emb_key="X_pca_harmony_nosamp",
        modality_col=None,
        batch_col="batch",
        save=True, verbose=True,
    )

    # ----- Tuned -----
    out_tuned = f"{base}/sampledisco_tuned"
    os.makedirs(out_tuned, exist_ok=True)
    _log(f"Autotuned sample embedding → {out_tuned}")
    run_autotune(
        adata, out_tuned,
        sample_col="sample",
        celltype_col="cell_type",
        cluster_emb_key="X_pca_harmony",
        cmd_emb_key="X_pca_harmony_nosamp",
        modality_col=None,
        batch_col="batch",
        grouping_col="sev.level",
        save=True, verbose=True,
    )
    _log(f"Done covid_{size}_sample")


def block_covid_rna() -> None:
    _hdr("BLOCK 1: COVID RNA — 6 sample sizes")
    for s in COVID_SAMPLE_SIZES:
        try:
            run_covid_rna_one(s)
        except Exception:
            _log(f"FAIL covid_{s}_sample:")
            traceback.print_exc()


# --------------------------------------------------------------------------- #
# Block 2: COVID ATAC                                                         #
# --------------------------------------------------------------------------- #
def _atac_visualize(out_dir: str, emb_csv: str, meta_csv: str, label: str) -> None:
    """2-D PCA scatter of the sample embedding colored by sev.level."""
    from sklearn.decomposition import PCA
    emb = pd.read_csv(emb_csv, index_col=0)
    meta = pd.read_csv(meta_csv)
    if "sample" not in meta.columns:
        _log(f"viz skipped — no 'sample' col in {meta_csv}")
        return
    meta = meta.set_index("sample")
    common = emb.index.intersection(meta.index)
    if len(common) == 0:
        _log("viz skipped — no overlapping sample IDs")
        return
    emb_a = emb.loc[common]
    meta_a = meta.loc[common]
    if "sev.level" not in meta_a.columns:
        _log("viz: 'sev.level' missing, coloring by index only")
        sev = pd.Series(range(len(meta_a)), index=meta_a.index)
    else:
        sev = pd.to_numeric(meta_a["sev.level"], errors="coerce")
    pca = PCA(n_components=2)
    P = pca.fit_transform(emb_a.values)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sc_ = ax.scatter(P[:, 0], P[:, 1], c=sev.values, cmap="viridis",
                       edgecolor="black", s=80)
    plt.colorbar(sc_, ax=ax, label="sev.level")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"COVID ATAC — sample embedding ({label})")
    out_png = os.path.join(out_dir, f"sample_embedding_pca_{label}.png")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    _log(f"Wrote {out_png}")


def block_covid_atac() -> None:
    _hdr("BLOCK 2: COVID ATAC (from scratch)")
    from wrapper.atac_wrapper import atac_wrapper

    Path(ATAC_OUT).mkdir(parents=True, exist_ok=True)

    # ----- Preprocess + cell-type + default-α sample embedding -----
    _log("Running atac_wrapper (preprocess + cell-type + default-α SE)")
    res = atac_wrapper(
        atac_count_data_path=ATAC_RAW,
        atac_output_dir=ATAC_OUT,
        atac_sample_meta_path=ATAC_META,
        preprocessing=True,
        cell_type_cluster=True,
        derive_sample_embedding=True,
        autotune_enable=False,
        sample_col="sample",
        sample_level_batch_col=None,
        celltype_col="cell_type",
        cell_level_batch_key=None,
        verbose=True,
    )
    adata = res["adata"]

    # Move the default-α sample embedding into a clearly-named subdir
    default_src = f"{ATAC_OUT}/sample_embedding/sample_embedding.csv"
    default_dst_dir = f"{ATAC_OUT}/sampledisco_default/sample_embedding"
    Path(default_dst_dir).mkdir(parents=True, exist_ok=True)
    if os.path.exists(default_src):
        os.replace(default_src, f"{default_dst_dir}/sample_embedding.csv")
        try:
            os.rmdir(f"{ATAC_OUT}/sample_embedding")
        except OSError:
            pass
        _log(f"Moved default-α embedding to {default_dst_dir}")

    # ----- Tuned -----
    out_tuned = f"{ATAC_OUT}/sampledisco_tuned"
    os.makedirs(out_tuned, exist_ok=True)
    _log("Running autotune for ATAC")
    run_autotune(
        adata, out_tuned,
        sample_col="sample",
        celltype_col="cell_type",
        cluster_emb_key="X_lsi_harmony" if "X_lsi_harmony" in adata.obsm else (
            list(adata.obsm.keys())[0]),
        cmd_emb_key=None,  # auto-resolve to *_nosamp or fall back
        modality_col=None,
        batch_col=None,
        grouping_col="sev.level",
        save=True, verbose=True,
    )

    # ----- Visualization (default + tuned) -----
    viz_dir = f"{ATAC_OUT}/visualization"
    os.makedirs(viz_dir, exist_ok=True)
    default_csv = f"{ATAC_OUT}/sampledisco_default/sample_embedding/sample_embedding.csv"
    tuned_csv   = f"{ATAC_OUT}/sampledisco_tuned/sample_embedding/sample_embedding.csv"
    if os.path.exists(default_csv):
        _atac_visualize(viz_dir, default_csv, ATAC_META, "default")
    if os.path.exists(tuned_csv):
        _atac_visualize(viz_dir, tuned_csv,   ATAC_META, "tuned")


# --------------------------------------------------------------------------- #
# Block 3: Multi-omics (default-α only) + association                          #
# --------------------------------------------------------------------------- #
def _run_assoc_for_pseudo(adata, output_dir: str, modality_col: Optional[str]) -> None:
    """Build sample-level pseudo-adata and run dimension association analysis."""
    pseudo = build_sample_adata(adata, sample_col="sample",
                                 modality_col=modality_col)
    # Rename .uns key so association sees a non-`sample` slug ("X_DR_se").
    if "X_DR_sample" in pseudo.uns:
        pseudo.uns["X_DR_se"] = pseudo.uns.pop("X_DR_sample")
    pseudo.obsm["X_DR_se"] = pseudo.obsm.get(
        "X_DR_sample", np.asarray(pseudo.X, dtype=np.float32))
    assoc_dir = os.path.join(output_dir, "sample_association")
    os.makedirs(assoc_dir, exist_ok=True)
    _log(f"Running association analysis → {assoc_dir}")
    run_dimension_association_analysis(
        pseudo_adata=pseudo,
        output_dir=assoc_dir,
        sample_col="sample",
        n_permutations=199,
        verbose=True,
    )


def run_multiomics_one(name: str, h5: str, base_out: str) -> None:
    _hdr(f"MULTI-OMICS — {name}")
    if not os.path.exists(h5):
        _log(f"SKIP {name} — missing {h5}")
        return
    _log(f"Loading {h5}")
    adata = sc.read(h5)
    _log(f"adata: shape={adata.shape}, obsm={list(adata.obsm.keys())}, "
         f"n_samples={adata.obs['sample'].nunique()}")

    if "cell_type" not in adata.obs.columns:
        raise KeyError(f"{name} missing cell_type column")
    if "X_glue" not in adata.obsm:
        raise KeyError(f"{name} missing X_glue obsm")

    out_default = f"{base_out}/sampledisco_default"
    os.makedirs(out_default, exist_ok=True)
    _log(f"Default-α sample embedding → {out_default}")
    compute_sample_embedding(
        adata, out_default,
        sample_col="sample",
        celltype_col="cell_type",
        cluster_emb_key="X_glue",
        cmd_emb_key="X_glue",
        modality_col="modality",
        batch_col=None,
        save=True, verbose=True,
    )
    _run_assoc_for_pseudo(adata, out_default, modality_col="modality")


def block_multiomics() -> None:
    _hdr("BLOCK 3: Multi-omics (ENCODE, Lutea, Retina, Heart)")
    for name, h5, base in MULTIOMICS:
        try:
            run_multiomics_one(name, h5, base)
        except Exception:
            _log(f"FAIL {name}:")
            traceback.print_exc()


# --------------------------------------------------------------------------- #
# Block 4: Unpaired (tuned only) + association                                #
# --------------------------------------------------------------------------- #
def block_unpaired() -> None:
    _hdr("BLOCK 4: Unpaired (tuned only)")
    if not os.path.exists(UNPAIRED_H5):
        _log(f"SKIP — missing {UNPAIRED_H5}")
        return
    _log(f"Loading {UNPAIRED_H5}")
    adata = sc.read(UNPAIRED_H5)
    _log(f"adata: shape={adata.shape}, obsm={list(adata.obsm.keys())}, "
         f"n_samples={adata.obs['sample'].nunique()}, "
         f"n_batches={adata.obs['batch'].nunique() if 'batch' in adata.obs else 'NA'}")

    out_tuned = f"{UNPAIRED_OUT}/sampledisco_tuned"
    os.makedirs(out_tuned, exist_ok=True)
    _log(f"Autotuned sample embedding → {out_tuned}")
    run_autotune(
        adata, out_tuned,
        sample_col="sample",
        celltype_col="cell_type",
        cluster_emb_key="X_glue",
        cmd_emb_key="X_glue",
        modality_col="modality",
        batch_col="batch",
        grouping_col="sev.level",
        save=True, verbose=True,
    )
    _run_assoc_for_pseudo(adata, out_tuned, modality_col="modality")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main(argv) -> int:
    blocks = {"rna": block_covid_rna, "atac": block_covid_atac,
              "mo": block_multiomics, "unpaired": block_unpaired}
    selected = argv[1:] if len(argv) > 1 else list(blocks.keys())
    bad = [b for b in selected if b not in blocks]
    if bad:
        print(f"Unknown blocks: {bad}; valid: {list(blocks.keys())}", file=sys.stderr)
        return 2
    t0 = time.time()
    for b in selected:
        try:
            blocks[b]()
        except Exception:
            _log(f"BLOCK {b} crashed:")
            traceback.print_exc()
    _log(f"All requested blocks done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
