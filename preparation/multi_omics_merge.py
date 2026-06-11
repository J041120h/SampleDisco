"""Multi-omics merge — replacement for compute_gene_activity_from_knn.

Three-AnnData design (see SampleDisco_Draft-4.pdf):

  ① ``build_embedding_union``
        UNION of RNA + ATAC cells, embeddings only (no expression X).
        Carries obs (sample, batch, modality, sev.level, …) + obsm
        (X_glue, plus Z_clust / Z_rmd when present from the 2-run scGLUE
        merge). This file is what cell typing, sample embedding, autotune,
        distance, trajectory all read.

  ② ``preprocess_rna_for_downstream``
        RNA cells only, with QC-filtered raw counts + normalized X. Used
        by RNA-side downstream (DGE / RAISIN / marker genes).

  ③ ``preprocess_atac_for_downstream``
        ATAC cells only, with QC-filtered raw peaks + TF-IDF + log1p X.
        Used by ATAC-side downstream.

Why three files instead of one synthetic gene-activity merge:
  - ATAC pseudo-expression via KNN to RNA (the old gene_activity step) is
    not a real measurement; using it for DGE is statistically meaningless.
  - The embedding-only union has no per-modality expression coupling, so
    the embedding pipeline (SE / autotune / distance) is completely
    independent of whether expression data is needed downstream.
  - Splitting RNA + ATAC preprocessing into separate per-modality files
    lets each modality apply its own QC (mito% for RNA, n_features /
    scrublet for ATAC) without the X-shape mismatch the old code hit.

QC logic is COPIED from rna_preprocess_cpu.preprocess / atac_preprocess_cpu
.preprocess (the QC blocks, lines L221-260 / L237-277 respectively) — by
design, to keep the single-omics pipelines untouched. Keep these in sync
manually if those single-omics QC blocks evolve.
"""
from __future__ import annotations

import contextlib
import io
import os
import time
from typing import Optional, Sequence

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse import issparse

from utils.merge_sample_meta import merge_sample_metadata
from utils.random_seed import set_global_seed
from utils.safe_save import safe_h5ad_write


# ────────────────────────────────────────────────────────────────────────── #
# ①   Embedding union (no expression X)                                      #
# ────────────────────────────────────────────────────────────────────────── #
def build_embedding_union(
    rna_emb_path: str,
    atac_emb_path: str,
    output_path: str,
    *,
    rna_sample_meta_path: Optional[str] = None,
    atac_sample_meta_path: Optional[str] = None,
    sample_column: str = "sample",
    rna_modality_value: str = "RNA",
    atac_modality_value: str = "ATAC",
    modality_col: str = "modality",
    obsm_keys_to_carry: Sequence[str] = ("X_glue", "Z_clust", "Z_rmd", "X_umap"),
    verbose: bool = True,
) -> ad.AnnData:
    """Build the embedding-only union AnnData from per-modality GLUE outputs.

    Reads ``glue-{rna,atac}-emb.h5ad`` (output of glue_train, possibly with
    Z_clust / Z_rmd merged in via the 2-run scGLUE flow), concatenates cells
    on the obs axis, carries obsm embeddings, and stores an empty-column
    placeholder X. Saves to ``output_path`` (typically
    ``<output_dir>/preprocess/adata_sample.h5ad``).

    Cell indices are suffixed with ``_{modality}`` to match the convention
    the SE / autotune code expects for multi-omics unit IDs. The original
    barcode is preserved in ``obs['original_barcode']`` to allow rejoining
    against per-modality h5ads later (cell typing label propagation).
    """
    if verbose:
        print(f"[merge] reading RNA emb: {rna_emb_path}")
    rna = ad.read_h5ad(rna_emb_path)
    if verbose:
        print(f"[merge] reading ATAC emb: {atac_emb_path}")
    atac = ad.read_h5ad(atac_emb_path)
    if verbose:
        print(f"[merge] RNA cells: {rna.n_obs:,}   ATAC cells: {atac.n_obs:,}")

    # Per-modality obs prep: infer sample column from barcode if absent
    # (matches single-omics _ensure_sample_column) and merge sample-level
    # metadata if a CSV is provided. Needed because glue-{rna,atac}-emb.h5ad
    # often lack sample / sev.level / batch — those are sample-level
    # annotations that join from a separate CSV.
    for a, meta_path in ((rna, rna_sample_meta_path), (atac, atac_sample_meta_path)):
        _ensure_sample_column(a, sample_column, verbose)
        if meta_path and os.path.exists(meta_path):
            merge_sample_metadata(adata=a, metadata_path=meta_path,
                                   sample_column=sample_column, verbose=verbose)

    def _embedding_view(a: ad.AnnData, modality_value: str) -> ad.AnnData:
        obs = a.obs.copy()
        obs[modality_col] = modality_value
        obs["original_barcode"] = obs.index.astype(str)
        obs.index = pd.Index([f"{idx}_{modality_value}" for idx in obs.index])

        carried = {k: np.asarray(a.obsm[k]) for k in obsm_keys_to_carry if k in a.obsm}
        if verbose:
            print(f"[merge]   {modality_value} obsm carried: {list(carried.keys())}")

        view = ad.AnnData(
            X=sparse.csr_matrix((a.n_obs, 0), dtype=np.float32),
            obs=obs,
        )
        for k, v in carried.items():
            view.obsm[k] = v.astype(np.float32, copy=False)
        return view

    rna_view = _embedding_view(rna, rna_modality_value)
    atac_view = _embedding_view(atac, atac_modality_value)

    # ad.concat(merge='same') requires obsm keys to match across views.
    common = set(rna_view.obsm) & set(atac_view.obsm)
    for view in (rna_view, atac_view):
        for k in [k for k in view.obsm if k not in common]:
            del view.obsm[k]

    union = ad.concat(
        [rna_view, atac_view],
        axis=0,
        join="outer",
        merge="same",
        index_unique=None,
    )

    if not union.obs.index.is_unique:
        union.obs_names_make_unique()

    if verbose:
        print(f"[merge] union shape: {union.shape}   "
              f"obsm: {list(union.obsm.keys())}   "
              f"obs cols: {len(union.obs.columns)}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    safe_h5ad_write(union, output_path)
    if verbose:
        print(f"[merge] wrote {output_path}")
    return union


# ────────────────────────────────────────────────────────────────────────── #
# Shared helpers (mirrored from rna/atac_preprocess_cpu)                     #
# ────────────────────────────────────────────────────────────────────────── #
def _ensure_sample_column(adata, sample_column, verbose=True):
    if sample_column not in adata.obs.columns:
        if verbose:
            print(f"   No '{sample_column}' in obs; inferring from obs_names")
        adata.obs[sample_column] = adata.obs_names.str.split(":").str[0]


def _to_float32(adata, verbose=True):
    if adata.X.dtype != np.float32:
        if verbose:
            print(f"   Converting X from {adata.X.dtype} to float32")
        adata.X = (
            adata.X.astype(np.float32)
            if issparse(adata.X)
            else np.asarray(adata.X, dtype=np.float32)
        )


# ────────────────────────────────────────────────────────────────────────── #
# ②   RNA per-modality preprocess for downstream DGE/RAISIN                   #
# ────────────────────────────────────────────────────────────────────────── #
def preprocess_rna_for_downstream(
    rna_emb_path: str,
    output_path: str,
    *,
    sample_column: str = "sample",
    sample_meta_path: Optional[str] = None,
    min_cells: int = 500,
    min_genes: int = 500,
    pct_mito_cutoff: float = 20.0,
    exclude_genes: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> ad.AnnData:
    """QC-filter the RNA cells from glue-rna-emb.h5ad and normalize for DGE.

    Mirrors the QC block of rna_preprocess_cpu.preprocess (L221-260): drop
    rare genes, low-gene cells, high-mito cells, mt + user genes, samples
    below cell-count threshold, then 0.1% gene re-filter. Layers/counts is
    preserved; X is normalized + log1p. Does NOT run HVG / PCA / Harmony —
    the cross-modal embedding lives in adata_sample.h5ad.

    Cell-typing labels are not added here; the wrapper joins ``cell_type``
    from the union after Leiden runs on Z_clust.
    """
    set_global_seed(seed=42)
    t0 = time.time()

    if verbose:
        print(f"[rna-preprocess] reading {rna_emb_path}")
    adata = sc.read_h5ad(rna_emb_path)
    if verbose:
        print(f"[rna-preprocess] raw shape: {adata.shape}")

    # Input X is raw counts; any pre-existing layers are redundant (the counts
    # layer is regenerated from X below) — drop them so the QC copies stay lean.
    adata.layers.clear()

    _ensure_sample_column(adata, sample_column, verbose)
    if sample_meta_path is not None and os.path.exists(sample_meta_path):
        adata = merge_sample_metadata(
            adata=adata, metadata_path=sample_meta_path,
            sample_column=sample_column, verbose=verbose,
        )

    _to_float32(adata, verbose)

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if verbose:
        print(f"[rna-preprocess] after initial filter: {adata.shape}")

    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                log1p=False, inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < pct_mito_cutoff].copy()
    if verbose:
        print(f"[rna-preprocess] after mt filter: {adata.shape}")

    mito_genes = adata.var_names[adata.var_names.str.startswith("MT-")]
    genes_to_exclude = set(mito_genes) | set(exclude_genes or [])
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()
    if verbose:
        print(f"[rna-preprocess] after gene exclusion: {adata.shape}")

    cells_per_sample = adata.obs.groupby(sample_column, observed=True).size()
    samples_to_keep = cells_per_sample[cells_per_sample >= min_cells].index
    adata = adata[adata.obs[sample_column].isin(samples_to_keep)].copy()
    if verbose:
        print(f"[rna-preprocess] after sample filter: {adata.shape}   "
              f"({len(samples_to_keep)} samples)")

    min_cells_for_gene = max(1, int(0.001 * adata.n_obs))
    sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)
    if verbose:
        print(f"[rna-preprocess] final shape: {adata.shape}")

    # Preserve raw counts before normalization; X becomes normalized log1p.
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    safe_h5ad_write(adata, output_path)
    if verbose:
        print(f"[rna-preprocess] wrote {output_path}   "
              f"runtime {time.time() - t0:.1f}s")
    return adata


# ────────────────────────────────────────────────────────────────────────── #
# ③   ATAC per-modality preprocess for downstream peak-level analyses        #
# ────────────────────────────────────────────────────────────────────────── #
def preprocess_atac_for_downstream(
    atac_emb_path: str,
    output_path: str,
    *,
    sample_column: str = "sample",
    sample_meta_path: Optional[str] = None,
    min_cells: int = 1,
    min_features: int = 2000,
    max_features: int = 15000,
    min_cells_per_sample: int = 1,
    exclude_features: Optional[Sequence[str]] = None,
    doublet_detection: bool = True,
    tfidf_scale_factor: float = 1e4,
    log_transform: bool = True,
    verbose: bool = True,
) -> ad.AnnData:
    """QC-filter ATAC cells from glue-atac-emb.h5ad and TF-IDF normalize.

    Mirrors the QC block of atac_preprocess_cpu.preprocess (L237-277):
    filter peaks by cell support, cells by n_features range, optional
    Scrublet doublet removal, per-sample filter, 0.1% feature re-filter.
    Then TF-IDF + log1p. Does NOT run LSI / Harmony — embedding is in
    adata_sample.h5ad.
    """
    from muon import atac as ac  # matches atac_preprocess_cpu's TF-IDF impl

    set_global_seed(seed=42)
    t0 = time.time()

    if verbose:
        print(f"[atac-preprocess] reading {atac_emb_path}")
    adata = sc.read_h5ad(atac_emb_path)
    if verbose:
        print(f"[atac-preprocess] raw shape: {adata.shape}")

    _ensure_sample_column(adata, sample_column, verbose)
    if sample_meta_path is not None and os.path.exists(sample_meta_path):
        adata = merge_sample_metadata(
            adata=adata, metadata_path=sample_meta_path,
            sample_column=sample_column, verbose=verbose,
        )

    _to_float32(adata, verbose)

    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(adata, "n_cells_by_counts", lambda x: x >= min_cells)
    mu.pp.filter_obs(adata, "n_genes_by_counts",
                     lambda x: (x >= min_features) & (x <= max_features))
    if verbose:
        print(f"[atac-preprocess] after initial filter: {adata.shape}")

    if doublet_detection and adata.n_vars >= 50:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                n_prin = min(30, adata.n_vars - 1, adata.n_obs - 1)
                sc.pp.scrublet(adata, batch_key=sample_column, n_prin_comps=n_prin)
                n_doublets = int(adata.obs["predicted_doublet"].sum())
                adata = adata[~adata.obs["predicted_doublet"]].copy()
            if verbose:
                print(f"[atac-preprocess] removed {n_doublets} doublets → {adata.shape}")
        except (ValueError, RuntimeError) as e:
            if verbose:
                print(f"[atac-preprocess] scrublet failed ({e}); continuing")

    if exclude_features:
        adata = adata[:, ~adata.var_names.isin(exclude_features)].copy()

    cells_per_sample = adata.obs.groupby(sample_column, observed=True).size()
    samples_to_keep = cells_per_sample[cells_per_sample >= min_cells_per_sample].index
    adata = adata[adata.obs[sample_column].isin(samples_to_keep)].copy()
    if verbose:
        print(f"[atac-preprocess] after sample filter: {adata.shape}   "
              f"({len(samples_to_keep)} samples)")

    min_cells_for_feature = max(1, int(0.001 * adata.n_obs))
    sc.pp.filter_genes(adata, min_cells=min_cells_for_feature)
    if verbose:
        print(f"[atac-preprocess] final shape: {adata.shape}")

    adata.layers["counts"] = adata.X.copy()
    ac.pp.tfidf(adata, scale_factor=tfidf_scale_factor)
    if log_transform:
        sc.pp.log1p(adata)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    safe_h5ad_write(adata, output_path)
    if verbose:
        print(f"[atac-preprocess] wrote {output_path}   "
              f"runtime {time.time() - t0:.1f}s")
    return adata


# ────────────────────────────────────────────────────────────────────────── #
# Cell-type label propagation                                                 #
# ────────────────────────────────────────────────────────────────────────── #
def propagate_cell_type(
    union_path: str,
    per_modality_paths: Sequence[str],
    *,
    celltype_col: str = "cell_type",
    barcode_col: str = "original_barcode",
    verbose: bool = True,
) -> None:
    """Copy cell_type from the embedding union onto each per-modality h5ad.

    Union obs has ``original_barcode`` set by ``build_embedding_union``;
    per-modality h5ads use the raw barcode as obs index. Join on that.
    """
    if verbose:
        print(f"[propagate] loading union {union_path}")
    union = sc.read_h5ad(union_path)
    if celltype_col not in union.obs.columns:
        raise KeyError(
            f"'{celltype_col}' not in union.obs — run cell typing first.")
    if barcode_col not in union.obs.columns:
        raise KeyError(
            f"'{barcode_col}' not in union.obs — was build_embedding_union "
            f"used to produce {union_path}?")
    ct_by_barcode = (union.obs.set_index(barcode_col)[celltype_col]
                              .groupby(level=0).first())

    for p in per_modality_paths:
        if not os.path.exists(p):
            if verbose:
                print(f"[propagate] skip (missing): {p}")
            continue
        if verbose:
            print(f"[propagate] joining cell_type → {p}")
        a = sc.read_h5ad(p)
        a.obs[celltype_col] = ct_by_barcode.reindex(a.obs.index.astype(str)).values
        a.obs[celltype_col] = a.obs[celltype_col].astype("category")
        safe_h5ad_write(a, p)
        if verbose:
            n_typed = int(a.obs[celltype_col].notna().sum())
            print(f"[propagate]   {n_typed}/{a.n_obs} cells received a label")
