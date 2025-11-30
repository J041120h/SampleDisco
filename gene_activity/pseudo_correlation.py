#!/usr/bin/env python3

# ----------------- Imports (headless plotting) -----------------
import os
import re
import json
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import matplotlib
matplotlib.use("Agg")  # ensure no X server is required
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr

# Try to import Harmony
try:
    from harmony import harmonize
    _HAS_HARMONY = True
except ImportError:
    _HAS_HARMONY = False
    print("[WARNING] harmony package not found. Will skip Harmony integration.")


# =====================================================================
#                     GENE ID / SYMBOL UTILITIES
# =====================================================================

_ENSEMBL_RE = re.compile(r"^ENSG\d+(?:\.\d+)?$", re.IGNORECASE)

def _looks_like_ensembl(x: str) -> bool:
    return bool(_ENSEMBL_RE.match(x or ""))

def _strip_ens_version(x: str) -> str:
    # ENSG00000141510.16 -> ENSG00000141510
    if x is None:
        return x
    return x.split(".", 1)[0] if x.upper().startswith("ENSG") else x

def _normalize_symbol(x: str) -> str:
    # standardize symbol casing; avoid None
    return (x or "").upper()

def _candidate_cols(var_df: pd.DataFrame) -> dict:
    """Return best-guess columns for ensembl and symbol in .var."""
    cols = {c.lower(): c for c in var_df.columns}
    ens_cols = [cols[k] for k in ["gene_id", "ensembl", "ensembl_id", "gene_ids"] if k in cols]
    sym_cols = [cols[k] for k in ["gene_name", "symbol", "gene_symbol", "genesymbol"] if k in cols]
    return {"ensembl": ens_cols, "symbol": sym_cols}

def _derive_ids(var_names: pd.Index, var_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a dataframe with columns:
      - ens_from_varname, sym_from_varname
      - ens_from_cols,   sym_from_cols
    (strings; may contain NaN where unavailable)
    """
    df = pd.DataFrame(index=var_names)

    # From var_names directly
    df["ens_from_varname"] = pd.Series(
        [_strip_ens_version(v) if _looks_like_ensembl(str(v)) else np.nan
         for v in var_names],
        index=var_names,
        dtype="object",
    )
    df["sym_from_varname"] = pd.Series(
        [_normalize_symbol(v) if not _looks_like_ensembl(str(v)) else np.nan
         for v in var_names],
        index=var_names,
        dtype="object",
    )

    # From known columns (pick the first available)
    cand = _candidate_cols(var_df)

    ens_col_val = None
    for c in cand["ensembl"]:
        if c in var_df.columns:
            ens_col_val = var_df[c].astype(str).map(_strip_ens_version)
            break

    sym_col_val = None
    for c in cand["symbol"]:
        if c in var_df.columns:
            sym_col_val = var_df[c].astype(str).map(_normalize_symbol)
            break

    df["ens_from_cols"] = ens_col_val.reindex(var_names) if ens_col_val is not None else np.nan
    df["sym_from_cols"] = sym_col_val.reindex(var_names) if sym_col_val is not None else np.nan

    return df

def _choose_unified_key(rna_ids: pd.DataFrame,
                        atac_ids: pd.DataFrame,
                        prefer: str = "auto") -> str:
    """
    Decide to unify on 'ensembl' or 'symbol'.
    prefer='auto' picks the ID space with the larger potential overlap.
    """
    # Potential usable keys
    rna_ens = pd.Series(rna_ids["ens_from_cols"]).fillna(rna_ids["ens_from_varname"])
    rna_sym = pd.Series(rna_ids["sym_from_cols"]).fillna(rna_ids["sym_from_varname"])

    atac_ens = pd.Series(atac_ids["ens_from_cols"]).fillna(atac_ids["ens_from_varname"])
    atac_sym = pd.Series(atac_ids["sym_from_cols"]).fillna(atac_ids["sym_from_varname"])

    ens_overlap = len(set(rna_ens.dropna()) & set(atac_ens.dropna()))
    sym_overlap = len(set(rna_sym.dropna()) & set(atac_sym.dropna()))

    if prefer in ("ensembl", "symbol"):
        return prefer
    # auto: pick the bigger
    return "ensembl" if ens_overlap >= sym_overlap else "symbol"


def unify_and_align_genes(
    adata_rna: sc.AnnData,
    adata_atac: sc.AnnData,
    output_dir: str,
    prefer: str = "auto",
    mapping_csv: str | None = None,
    atac_layer: str | None = "GeneActivity",
    verbose: bool = True,
):
    """
    Try multiple strategies to place RNA and ATAC into the same gene ID space.

    - Detect Ensembl IDs (with/without version) and symbols from var_names and common .var columns.
    - If mapping_csv is provided (columns must include both 'gene_id' and 'gene_name'),
      it is used as an extra source.
    - prefer='ensembl' | 'symbol' | 'auto' (default: auto)
    - If atac_layer is set and present, use that layer as the ATAC matrix.

    Returns: rna_sub, atac_sub, shared_ids, mapping_df

    DEBUG-friendly version:
      * Detects duplicate unified IDs per modality
      * Writes them to CSV and drops them (keeps first) before alignment
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use ATAC layer, e.g., GeneActivity, if provided
    if atac_layer and hasattr(adata_atac.layers, "keys") and atac_layer in adata_atac.layers.keys():
        if verbose:
            print(f"[unify] Using ATAC layer '{atac_layer}' as X")
        X = adata_atac.layers[atac_layer]
        # Preserve var/obs; replace X view only for correlation
        adata_atac = sc.AnnData(X=X, obs=adata_atac.obs.copy(), var=adata_atac.var.copy())
    elif verbose:
        print(f"[unify] ATAC layer '{atac_layer}' not found; using adata_atac.X")

    # Derive IDs for both
    if verbose:
        print("[unify] Deriving ID candidates from RNA and ATAC .var / var_names")

    rna_ids = _derive_ids(adata_rna.var_names, adata_rna.var)
    atac_ids = _derive_ids(adata_atac.var_names, adata_atac.var)

    # Optional external mapping
    sym2ens = {}
    ens2sym = {}
    if mapping_csv and os.path.exists(mapping_csv):
        if verbose:
            print(f"[unify] Reading mapping_csv: {mapping_csv}")
        mdf = pd.read_csv(mapping_csv)
        if {"gene_id", "gene_name"}.issubset(set(mdf.columns)):
            mdf = mdf.dropna(subset=["gene_id", "gene_name"]).copy()
            mdf["gene_id"] = mdf["gene_id"].astype(str).map(_strip_ens_version)
            mdf["gene_name"] = mdf["gene_name"].astype(str).map(_normalize_symbol)
            # prefer 1-1 (drop duplicates keeping first)
            mdf = mdf.drop_duplicates(subset=["gene_id"], keep="first")
            ens2sym = pd.Series(mdf["gene_name"].values, index=mdf["gene_id"].values).to_dict()
            sym2ens = pd.Series(mdf["gene_id"].values, index=mdf["gene_name"].values).to_dict()
            if verbose:
                print(f"[unify] Loaded mapping_csv with {len(mdf)} unique gene_id rows")
        else:
            if verbose:
                print("[unify] mapping_csv missing required columns 'gene_id' and 'gene_name' — ignoring")

    # Choose unified space
    target = _choose_unified_key(rna_ids, atac_ids, prefer=prefer)
    if verbose:
        print(f"[unify] Unifying on: {target.upper()}")

    # Build preferred ID columns for each modality
    if target == "ensembl":
        rna_id = rna_ids["ens_from_cols"].fillna(rna_ids["ens_from_varname"])
        atac_id = atac_ids["ens_from_cols"].fillna(atac_ids["ens_from_varname"])

        # If missing, try convert via symbol -> ensembl using mapping_csv
        if mapping_csv:
            if verbose:
                print("[unify] Filling missing Ensembl IDs using symbol→Ensembl mapping (if possible)")
            rna_missing = rna_id.isna()
            atac_missing = atac_id.isna()
            if rna_missing.any():
                sym = rna_ids.loc[rna_missing, "sym_from_cols"].fillna(rna_ids.loc[rna_missing, "sym_from_varname"])
                rna_id.loc[rna_missing] = sym.map(sym2ens)
            if atac_missing.any():
                sym = atac_ids.loc[atac_missing, "sym_from_cols"].fillna(atac_ids.loc[atac_missing, "sym_from_varname"])
                atac_id.loc[atac_missing] = sym.map(sym2ens)

        # Clean
        rna_id = rna_id.dropna().astype(str).map(_strip_ens_version)
        atac_id = atac_id.dropna().astype(str).map(_strip_ens_version)

    else:  # target == "symbol"
        rna_id = rna_ids["sym_from_cols"].fillna(rna_ids["sym_from_varname"])
        atac_id = atac_ids["sym_from_cols"].fillna(atac_ids["sym_from_varname"])

        # If missing, try ensembl -> symbol via mapping_csv
        if mapping_csv:
            if verbose:
                print("[unify] Filling missing symbols using Ensembl→symbol mapping (if possible)")
            rna_missing = rna_id.isna()
            atac_missing = atac_id.isna()
            if rna_missing.any():
                ens = rna_ids.loc[rna_missing, "ens_from_cols"].fillna(rna_ids.loc[rna_missing, "ens_from_varname"])
                rna_id.loc[rna_missing] = ens.map(ens2sym)
            if atac_missing.any():
                ens = atac_ids.loc[atac_missing, "ens_from_cols"].fillna(atac_ids.loc[atac_missing, "ens_from_varname"])
                atac_id.loc[atac_missing] = ens.map(ens2sym)

        # Clean
        rna_id = rna_id.dropna().astype(str).map(_normalize_symbol)
        atac_id = atac_id.dropna().astype(str).map(_normalize_symbol)

    # ------------------------------------------------------------------
    # Build mapping DataFrames with orig var_names and unified IDs
    # ------------------------------------------------------------------
    rna_uid = rna_id.reindex(adata_rna.var_names)
    atac_uid = atac_id.reindex(adata_atac.var_names)

    map_rna = pd.DataFrame({"orig": adata_rna.var_names, "unified_id": rna_uid.values})
    map_atac = pd.DataFrame({"orig": adata_atac.var_names, "unified_id": atac_uid.values})

    if verbose:
        print(f"[unify] Non-null unified IDs: RNA={map_rna['unified_id'].notna().sum()}, "
              f"ATAC={map_atac['unified_id'].notna().sum()}")

    # ------------------------------------------------------------------
    # DEBUG: detect and log duplicate unified IDs BEFORE alignment
    # ------------------------------------------------------------------
    dup_rna = map_rna[map_rna["unified_id"].notna() &
                      map_rna["unified_id"].duplicated(keep=False)]
    dup_atac = map_atac[map_atac["unified_id"].notna() &
                        map_atac["unified_id"].duplicated(keep=False)]

    if len(dup_rna) > 0:
        dup_path_rna = os.path.join(output_dir, "rna_unified_id_duplicates.csv")
        dup_rna.to_csv(dup_path_rna, index=False)
        if verbose:
            print(f"[unify][DEBUG] RNA duplicates detected for unified_id (n={len(dup_rna)} rows). "
                  f"Saved details → {dup_path_rna}")

    if len(dup_atac) > 0:
        dup_path_atac = os.path.join(output_dir, "atac_unified_id_duplicates.csv")
        dup_atac.to_csv(dup_path_atac, index=False)
        if verbose:
            print(f"[unify][DEBUG] ATAC duplicates detected for unified_id (n={len(dup_atac)} rows). "
                  f"Saved details → {dup_path_atac}")

    # Drop duplicates (keep first occurrence) for alignment purposes
    map_rna_unique = (
        map_rna
        .dropna(subset=["unified_id"])
        .drop_duplicates(subset="unified_id", keep="first")
        .copy()
    )
    map_atac_unique = (
        map_atac
        .dropna(subset=["unified_id"])
        .drop_duplicates(subset="unified_id", keep="first")
        .copy()
    )

    if verbose:
        print(f"[unify] After dropping duplicates: RNA unique IDs={len(map_rna_unique)}, "
              f"ATAC unique IDs={len(map_atac_unique)}")

    # Build Series unified_id -> orig var_name
    rna_uid_to_var = pd.Series(
        map_rna_unique["orig"].values,
        index=map_rna_unique["unified_id"].values,
        dtype="object",
    )
    atac_uid_to_var = pd.Series(
        map_atac_unique["orig"].values,
        index=map_atac_unique["unified_id"].values,
        dtype="object",
    )

    # Shared unified ids (after uniqueness)
    shared = sorted(set(rna_uid_to_var.index) & set(atac_uid_to_var.index))
    if verbose:
        print(f"[unify] Shared unified IDs after deduplication: {len(shared)}")

    if len(shared) == 0:
        raise ValueError("[unify] No overlap after ID harmonization (post-dedup). "
                         "Provide a mapping_csv or check species/build.")

    # ------------------------------------------------------------------
    # Use original var_names (which are unique) to get column indices
    # ------------------------------------------------------------------
    rna_var_index = pd.Index(adata_rna.var_names)
    atac_var_index = pd.Index(adata_atac.var_names)

    rna_cols = [rna_uid_to_var.loc[u] for u in shared]
    atac_cols = [atac_uid_to_var.loc[u] for u in shared]

    rna_idx = rna_var_index.get_indexer(rna_cols)
    atac_idx = atac_var_index.get_indexer(atac_cols)

    # Safety checks
    if (rna_idx < 0).any():
        bad = [shared[i] for i, idx in enumerate(rna_idx) if idx < 0]
        raise RuntimeError(f"[unify] Internal error: some shared IDs not found in RNA var_names: {bad[:10]} ...")
    if (atac_idx < 0).any():
        bad = [shared[i] for i, idx in enumerate(atac_idx) if idx < 0]
        raise RuntimeError(f"[unify] Internal error: some shared IDs not found in ATAC var_names: {bad[:10]} ...")

    if verbose:
        print(f"[unify] Will align RNA and ATAC to {len(shared)} shared genes")

    # Subset (columns) and overwrite var_names with unified ids
    rna_aligned = adata_rna[:, rna_idx].copy()
    atac_aligned = adata_atac[:, atac_idx].copy()
    rna_aligned.var_names = pd.Index(shared)
    atac_aligned.var_names = pd.Index(shared)

    # Save mapping CSV for provenance
    md = pd.DataFrame({
        "unified_id": shared,
        "rna_orig": [rna_uid_to_var.loc[u] for u in shared],
        "atac_orig": [atac_uid_to_var.loc[u] for u in shared],
        "target_space": target
    })
    md_path = os.path.join(output_dir, "gene_id_mapping_unified.csv")
    md.to_csv(md_path, index=False)
    if verbose:
        print(f"[unify] Saved mapping → {md_path}")

    return rna_aligned, atac_aligned, shared, md


# =====================================================================
#                       BASIC QC / CLUSTERING HELPERS
# =====================================================================

def _basic_qc_filter(
    adata: sc.AnnData,
    min_genes: int = 200,
    min_cells: int = 50,
    pct_mito_cutoff: float = 20.0,
    verbose: bool = True,
) -> sc.AnnData:
    """
    Basic QC on an AnnData object (RNA).
    """
    if verbose:
        print(f"[QC] Starting QC: {adata.n_obs} cells × {adata.n_vars} genes")

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    has_mt = adata.var_names.str.startswith("MT-").any()
    if has_mt:
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        before = adata.n_obs
        adata = adata[adata.obs["pct_counts_mt"] < pct_mito_cutoff].copy()
        if verbose:
            print(f"[QC] Mito filter ({pct_mito_cutoff}%): kept {adata.n_obs}/{before} cells")

    if verbose:
        print(f"[QC] Finished QC: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def _cluster_with_harmony_on_rna(
    adata_rna: sc.AnnData,
    batch_col: str | None = None,
    n_hvgs: int = 2000,
    n_pcs: int = 30,
    resolution: float = 1.0,
    celltype_key: str = "celltype",
    random_state: int = 0,
    verbose: bool = True,
) -> sc.AnnData:
    """
    On RNA data:
    - Normalize & log1p
    - HVG selection
    - PCA
    - Harmony (if available) using batch_col
    - Neighbors, UMAP, Leiden (igraph flavor)
    Adds cell type labels to adata_rna.obs[celltype_key].
    """
    if verbose:
        print("[Cluster] Normalizing RNA (normalize_total + log1p)...")
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)

    if verbose:
        print(f"[Cluster] Selecting {n_hvgs} HVGs...")
    sc.pp.highly_variable_genes(
        adata_rna,
        n_top_genes=n_hvgs,
        flavor="seurat_v3",
        batch_key=batch_col if (batch_col is not None and batch_col in adata_rna.obs.columns) else None,
    )
    adata_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()

    if verbose:
        print(f"[Cluster] Running PCA with {n_pcs} components...")
    sc.pp.pca(adata_rna, n_comps=n_pcs, svd_solver="arpack", random_state=random_state)

    use_rep = "X_pca"
    can_run_harmony = (
        _HAS_HARMONY
        and batch_col is not None
        and batch_col in adata_rna.obs.columns
        and adata_rna.obs[batch_col].notna().any()
    )
    if can_run_harmony:
        if verbose:
            print(f"[Cluster] Running Harmony on batch_col = '{batch_col}'...")
            print(f"[Cluster][DEBUG] batch_col unique values: {adata_rna.obs[batch_col].value_counts().to_dict()}")
        Z = harmonize(
            adata_rna.obsm["X_pca"],
            adata_rna.obs,
            batch_key=batch_col,
        )
        adata_rna.obsm["X_pca_harmony"] = Z
        use_rep = "X_pca_harmony"
    else:
        if verbose:
            print(f"[Cluster] Not running Harmony (HAS_HARMONY={_HAS_HARMONY}, "
                  f"batch_col='{batch_col}', in_obs={batch_col in adata_rna.obs.columns}). "
                  "Using PCA directly.")

    if verbose:
        print("[Cluster] Computing neighbors and UMAP...")
    sc.pp.neighbors(adata_rna, use_rep=use_rep, n_neighbors=15)
    sc.tl.umap(adata_rna, random_state=random_state)

    if verbose:
        print(f"[Cluster] Running Leiden (resolution={resolution}, igraph flavor)...")
    sc.tl.leiden(
        adata_rna,
        key_added=celltype_key,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )

    if verbose:
        n_ct = adata_rna.obs[celltype_key].nunique()
        print(f"[Cluster] Found {n_ct} Leiden clusters (cell types).")

    return adata_rna


# =====================================================================
#                        PSEUDOBULK HELPER
# =====================================================================

def _pseudobulk_by_group(
    adata: sc.AnnData,
    group_key: str,
    layer: str | None = None,
    agg: str = "mean",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pseudobulk per group (e.g., per cell type).
    Returns a DataFrame: rows = genes, columns = groups.
    Uses ALL genes in adata (no HVG restriction).
    """
    if group_key not in adata.obs.columns:
        raise KeyError(f"group_key '{group_key}' not found in adata.obs")

    valid = adata.obs[group_key].notna()
    adata = adata[valid].copy()

    if verbose:
        print(f"[Pseudobulk] Using {adata.n_obs} cells grouped by '{group_key}'.")

    X = adata.layers[layer] if layer is not None else adata.X
    if sparse.issparse(X):
        X = X.toarray()

    df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    df[group_key] = adata.obs[group_key].values

    # explicit observed=False to silence FutureWarning
    if agg == "mean":
        g = df.groupby(group_key, observed=False).mean()
    elif agg == "sum":
        g = df.groupby(group_key, observed=False).sum()
    else:
        raise ValueError("agg must be 'mean' or 'sum'")

    pseudobulk = g.T
    if verbose:
        print(f"[Pseudobulk] Result shape: {pseudobulk.shape[0]} genes × {pseudobulk.shape[1]} groups")

    return pseudobulk


# =====================================================================
#          MAIN FUNCTION: PSEUDOBULK + PER-CELLTYPE PEARSON
# =====================================================================

def compare_atac_rna_gene_activity(
    atac_h5ad_path: str,
    rna_h5ad_path: str,
    output_dir: str,
    sample_col: str = "sample",
    batch_col: str | None = "batch",
    celltype_key: str = "celltype",
    n_hvgs: int = 2000,
    n_pcs: int = 30,
    resolution: float = 1.0,
    min_genes: int = 200,
    min_cells: int = 50,
    pct_mito_cutoff: float = 20.0,
    random_state: int = 0,
    verbose: bool = True,
    # --- gene ID unification knobs ---
    unify_if_needed: bool = True,
    unify_prefer: str = "auto",              # 'ensembl' | 'symbol' | 'auto'
    unify_mapping_csv: str | None = None,    # optional (gene_id,gene_name)
    atac_layer_for_unify: str | None = "GeneActivity",  # use this layer if present
):
    """
    1. Load paired ATAC gene-activity and RNA AnnData objects.
    2. Align to shared cells (obs_names intersection).
    3. QC + HVG + PCA + Harmony + Leiden on RNA → cell types.
    4. Transfer cell types to RNA (full) and ATAC.
    5. Use unify_and_align_genes() to put RNA & ATAC in same gene ID space.
    6. Normalize (CPM+log1p), pseudobulk per cell type using ALL genes.
    7. Compute Pearson correlation between RNA and ATAC gene activity per cell type.
    8. Save UMAPs and correlation visualizations.

    Returns:
        dict with keys:
            'rna_adata_cluster', 'atac_adata_aligned',
            'rna_pseudobulk', 'atac_pseudobulk',
            'celltype_correlations'
    """
    os.makedirs(output_dir, exist_ok=True)
    sc.settings.autoshow = False
    sc.settings.figdir = output_dir

    # ------------------------------------------------------------------
    # 0. Load and pair cells
    # ------------------------------------------------------------------
    if verbose:
        print(f"[Main] Loading RNA from:  {rna_h5ad_path}")
        print(f"[Main] Loading ATAC from: {atac_h5ad_path}")
    adata_rna_full = sc.read_h5ad(rna_h5ad_path)
    adata_atac_full = sc.read_h5ad(atac_h5ad_path)

    if verbose:
        print(f"[Main] RNA shape:  {adata_rna_full.shape}")
        print(f"[Main] ATAC shape: {adata_atac_full.shape}")

    common_cells = adata_rna_full.obs_names.intersection(adata_atac_full.obs_names)
    if len(common_cells) == 0:
        raise ValueError("No overlapping cell barcodes between RNA and ATAC AnnData objects.")

    if verbose:
        print(f"[Main] Found {len(common_cells)} shared cells; subsetting both modalities.")
    adata_rna_full = adata_rna_full[common_cells].copy()
    adata_atac_full = adata_atac_full[common_cells].copy()

    # ------------------------------------------------------------------
    # 1. QC + clustering on RNA (to define cell types)
    # ------------------------------------------------------------------
    if verbose:
        print("[Main] Running QC + clustering on RNA (for cell types)...")

    adata_rna_cluster = _basic_qc_filter(
        adata_rna_full.copy(),
        min_genes=min_genes,
        min_cells=min_cells,
        pct_mito_cutoff=pct_mito_cutoff,
        verbose=verbose,
    )

    # After QC, align back to ATAC & full-RNA
    common_cells2 = adata_rna_cluster.obs_names.intersection(adata_atac_full.obs_names)
    adata_rna_cluster = adata_rna_cluster[common_cells2].copy()
    adata_rna_full = adata_rna_full[common_cells2].copy()
    adata_atac_full = adata_atac_full[common_cells2].copy()
    if verbose:
        print(f"[Main] After QC, keeping {len(common_cells2)} cells for both RNA and ATAC.")

    # Cluster RNA (HVG → PCA → Harmony → Leiden)
    adata_rna_cluster = _cluster_with_harmony_on_rna(
        adata_rna_cluster,
        batch_col=batch_col,
        n_hvgs=n_hvgs,
        n_pcs=n_pcs,
        resolution=resolution,
        celltype_key=celltype_key,
        random_state=random_state,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 2. Transfer cell types back to full RNA + ATAC objects
    # ------------------------------------------------------------------
    if verbose:
        print("[Main] Transferring cell types to full RNA and ATAC objects...")

    # Ensure same ordering
    adata_rna_full = adata_rna_full[adata_rna_cluster.obs_names].copy()
    adata_atac_full = adata_atac_full[adata_rna_cluster.obs_names].copy()

    adata_rna_full.obs[celltype_key] = adata_rna_cluster.obs[celltype_key].values
    adata_atac_full.obs[celltype_key] = adata_rna_cluster.obs[celltype_key].values

    # ------------------------------------------------------------------
    # 3. UMAP plots (RNA + ATAC; ATAC reuses RNA embedding)
    # ------------------------------------------------------------------
    if "X_umap" in adata_rna_cluster.obsm:
        if verbose:
            print("[Plot] Saving RNA UMAP colored by cell type...")
        sc.pl.umap(
            adata_rna_cluster,
            color=celltype_key,
            show=False,
            save="_rna_celltypes.png",
        )
        plt.close("all")

        if verbose:
            print("[Plot] Saving ATAC UMAP (using RNA UMAP coordinates) colored by cell type...")
        adata_atac_full.obsm["X_umap"] = adata_rna_cluster.obsm["X_umap"].copy()
        sc.pl.umap(
            adata_atac_full,
            color=celltype_key,
            show=False,
            save="_atac_celltypes.png",
        )
        plt.close("all")

    # ------------------------------------------------------------------
    # 4. Gene ID unification (your helper)
    # ------------------------------------------------------------------
    if verbose:
        print("[Main] Unifying gene IDs between RNA and ATAC (before pseudobulk)...")

    if unify_if_needed:
        rna_aligned, atac_aligned, shared_ids, mapping_df = unify_and_align_genes(
            adata_rna_full,
            adata_atac_full,
            output_dir=os.path.join(output_dir, "gene_mapping"),
            prefer=unify_prefer,
            mapping_csv=unify_mapping_csv,
            atac_layer=atac_layer_for_unify,
            verbose=verbose,
        )
        if verbose:
            print(f"[Main] After unification: {len(shared_ids)} shared gene IDs.")
    else:
        shared_var = adata_rna_full.var_names.intersection(adata_atac_full.var_names)
        if len(shared_var) == 0:
            raise ValueError("No overlapping genes between RNA and ATAC, and unify_if_needed=False.")
        rna_aligned = adata_rna_full[:, shared_var].copy()
        atac_aligned = adata_atac_full[:, shared_var].copy()
        shared_ids = list(shared_var)
        if verbose:
            print(f"[Main] Using direct name intersection without unification: {len(shared_ids)} genes")

    # Ensure celltype column survived
    assert celltype_key in rna_aligned.obs.columns
    assert celltype_key in atac_aligned.obs.columns

    # ------------------------------------------------------------------
    # 5. Normalize (CPM+log1p) using ALL genes, then pseudobulk per cell type
    # ------------------------------------------------------------------
    if verbose:
        print("[Main] Normalizing aligned RNA & ATAC (CPM + log1p) for pseudobulk...")

    rna_pb = rna_aligned.copy()
    atac_pb = atac_aligned.copy()

    sc.pp.normalize_total(rna_pb, target_sum=1e4)
    sc.pp.log1p(rna_pb)

    sc.pp.normalize_total(atac_pb, target_sum=1e4)
    sc.pp.log1p(atac_pb)

    # pseudobulk over ALL genes in shared_ids
    rna_pseudobulk = _pseudobulk_by_group(
        rna_pb,
        group_key=celltype_key,
        layer=None,
        agg="mean",
        verbose=verbose,
    )
    atac_pseudobulk = _pseudobulk_by_group(
        atac_pb,
        group_key=celltype_key,
        layer=None,
        agg="mean",
        verbose=verbose,
    )

    # Save pseudobulk matrices
    rna_pb_path = os.path.join(output_dir, "rna_pseudobulk_celltype.csv")
    atac_pb_path = os.path.join(output_dir, "atac_pseudobulk_celltype.csv")
    rna_pseudobulk.to_csv(rna_pb_path)
    atac_pseudobulk.to_csv(atac_pb_path)
    if verbose:
        print(f"[Main] Saved RNA pseudobulk to:  {rna_pb_path}")
        print(f"[Main] Saved ATAC pseudobulk to: {atac_pb_path}")

    # ------------------------------------------------------------------
    # 6. Pearson correlation per cell type (across genes)
    # ------------------------------------------------------------------
    common_genes = rna_pseudobulk.index.intersection(atac_pseudobulk.index)
    common_celltypes = rna_pseudobulk.columns.intersection(atac_pseudobulk.columns)
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between RNA and ATAC pseudobulk matrices "
                         "(after unification something is wrong).")
    if len(common_celltypes) == 0:
        raise ValueError("No overlapping cell types between RNA and ATAC pseudobulk matrices.")

    if verbose:
        print(f"[Main] For correlation: {len(common_genes)} common genes, "
              f"{len(common_celltypes)} common cell types.")

    rna_mat = rna_pseudobulk.loc[common_genes, common_celltypes]
    atac_mat = atac_pseudobulk.loc[common_genes, common_celltypes]

    corrs = {}
    for ct in common_celltypes:
        r = pearsonr(rna_mat[ct].values, atac_mat[ct].values)[0]
        corrs[ct] = r

    corr_df = pd.DataFrame.from_dict(corrs, orient="index", columns=["pearson_r"]).sort_index()
    corr_path = os.path.join(output_dir, "celltype_pearson_correlations.csv")
    corr_df.to_csv(corr_path)
    if verbose:
        print(f"[Main] Saved per-cell-type Pearson correlations to: {corr_path}")

    # ------------------------------------------------------------------
    # 7. Visualizations: heatmap + example scatter
    # ------------------------------------------------------------------
    if verbose:
        print("[Plot] Saving heatmap of per-cell-type Pearson correlations...")
    plt.figure(figsize=(max(6, len(common_celltypes) * 0.5), 3))
    sns.heatmap(
        corr_df.T,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0.0,
        cbar_kws={"label": "Pearson r (RNA vs ATAC)"},
    )
    plt.xlabel("Cell type")
    plt.ylabel("")
    plt.title("RNA vs ATAC gene activity correlation per cell type")
    heatmap_path = os.path.join(output_dir, "celltype_correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    example_ct = common_celltypes[0]
    if verbose:
        print(f"[Plot] Saving scatter plot for example cell type: {example_ct}")

    plt.figure(figsize=(5, 5))
    plt.scatter(
        rna_mat[example_ct].values,
        atac_mat[example_ct].values,
        s=5,
        alpha=0.5,
    )
    plt.xlabel("RNA pseudobulk (log1p CPM)")
    plt.ylabel("ATAC gene activity pseudobulk (log1p CPM)")
    plt.title(f"RNA vs ATAC gene activity per gene\nCell type: {example_ct}")
    scatter_path = os.path.join(output_dir, f"scatter_rna_vs_atac_{example_ct}.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    if verbose:
        print("[Main] Finished comparison of ATAC gene activity vs RNA.")
        print(f"[Main] Outputs saved to: {output_dir}")

    return {
        "rna_adata_cluster": adata_rna_cluster,
        "atac_adata_aligned": atac_aligned,
        "rna_pseudobulk": rna_pseudobulk,
        "atac_pseudobulk": atac_pseudobulk,
        "celltype_correlations": corr_df,
    }


# =====================================================================
#  OPTIONAL: CELL-LEVEL / GENE-LEVEL CORRELATION (YOUR ORIGINAL FUNC)
# =====================================================================

def compute_rna_atac_cell_gene_correlations(
    adata_rna,
    adata_atac,
    output_dir: str,
    min_cells_for_gene_corr: int = 3,
    sample_genes: int | None = 1000,   # set None to use all shared genes
    verbose: bool = True,
    # ---- flexible ID unifier knobs ----
    unify_if_needed: bool = True,
    unify_prefer: str = "auto",              # 'ensembl' | 'symbol' | 'auto'
    unify_mapping_csv: str | None = None,    # optional mapping with gene_id,gene_name
    atac_layer: str | None = "GeneActivity", # if present, use this ATAC layer as X
) -> dict:
    """
    Compute per-cell and per-gene RNA–ATAC correlations for paired cells.
    If gene names don't overlap, automatically unify ID spaces (Ensembl vs Symbol).
    DEBUG prints are enabled regardless of 'verbose' for core steps.
    """
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from scipy import stats as sp_stats

    os.makedirs(output_dir, exist_ok=True)

    def _to_array(X):
        if sparse.issparse(X):
            return X.toarray()
        return np.asarray(X)

    # -----------------------------
    # DEBUG: Print input data info
    # -----------------------------
    print("\n[DEBUG] ===== INPUT DATA INFO =====")
    print(f"[DEBUG] RNA data shape: {adata_rna.shape}")
    print(f"[DEBUG] First 5 RNA cell names: {list(adata_rna.obs_names[:5])}")
    print(f"[DEBUG] First 5 RNA gene names: {list(adata_rna.var_names[:5])}")
    print(f"[DEBUG] RNA X type: {type(adata_rna.X)}")

    print(f"\n[DEBUG] ATAC data shape: {adata_atac.shape}")
    print(f"[DEBUG] First 5 ATAC cell names: {list(adata_atac.obs_names[:5])}")
    print(f"[DEBUG] First 5 ATAC gene names: {list(adata_atac.var_names[:5])}")
    print(f"[DEBUG] ATAC X type: {type(adata_atac.X)}")

    # -----------------------------
    # 1) Pair cells by name
    # -----------------------------
    rna_cells = set(map(str, adata_rna.obs_names))
    atac_cells = set(map(str, adata_atac.obs_names))

    print(f"\n[DEBUG] ===== CELL PAIRING =====")
    print(f"[DEBUG] Total RNA cells: {len(rna_cells)} | Total ATAC cells: {len(atac_cells)}")

    common_cells = sorted(rna_cells & atac_cells)
    print(f"[DEBUG] Common cells found: {len(common_cells)}")
    if len(common_cells) == 0:
        raise ValueError("No paired cells found (no overlap in obs_names).")

    # Subset & align rows
    rna_sub = adata_rna[common_cells, :].copy()
    atac_sub = adata_atac[common_cells, :].copy()

    print(f"[DEBUG] After subsetting to common cells → RNA: {rna_sub.shape}, ATAC: {atac_sub.shape}")

    # -----------------------------
    # 2) Align genes by name or unify
    # -----------------------------
    rna_genes = set(map(str, rna_sub.var_names))
    atac_genes = set(map(str, atac_sub.var_names))
    shared_genes = sorted(rna_genes & atac_genes)

    print(f"\n[DEBUG] ===== GENE ALIGNMENT =====")
    print(f"[DEBUG] Overlap by current var_names: {len(shared_genes)}")

    if len(shared_genes) == 0 and unify_if_needed:
        print("[DEBUG] No shared genes. Attempting ID unification (Ensembl/Symbol)…")
        rna_sub, atac_sub, shared_genes, mapping_df = unify_and_align_genes(
            rna_sub, atac_sub,
            output_dir=output_dir,
            prefer=unify_prefer,
            mapping_csv=unify_mapping_csv,
            atac_layer=atac_layer,
            verbose=verbose,
        )
        print(f"[DEBUG] Unified & aligned genes: {len(shared_genes)}")

    if len(shared_genes) == 0:
        print("[DEBUG] !!! NO SHARED GENES FOUND EVEN AFTER UNIFICATION !!!")
        raise ValueError("No shared genes between RNA and ATAC.")

    # Optional downsampling
    if sample_genes is not None and sample_genes < len(shared_genes):
        rng = np.random.default_rng(42)
        shared_genes = sorted(rng.choice(shared_genes, size=sample_genes, replace=False).tolist())
        print(f"[DEBUG] Sampled down to {len(shared_genes)} genes")

    # Column-align
    rna_sub = rna_sub[:, shared_genes].copy()
    atac_sub = atac_sub[:, shared_genes].copy()

    # Extract dense arrays
    rna_X = _to_array(rna_sub.X)
    atac_X = _to_array(atac_sub.X)

    n_cells, n_genes = rna_X.shape
    print(f"\n[DEBUG] ===== FINAL MATRICES =====")
    print(f"[DEBUG] Final matrix shape: {n_cells} cells × {n_genes} genes")

    # -----------------------------
    # 3) Per-cell correlations
    # -----------------------------
    print(f"\n[DEBUG] ===== PER-CELL CORRELATIONS =====")
    per_cell_corr = np.full(n_cells, np.nan, dtype=float)
    n_valid_cells = 0
    for i in range(n_cells):
        r = rna_X[i, :]
        a = atac_X[i, :]
        mask = (r != 0) | (a != 0)
        if mask.sum() >= 3 and np.std(r[mask]) > 0 and np.std(a[mask]) > 0:
            per_cell_corr[i] = np.corrcoef(r[mask], a[mask])[0, 1]
            n_valid_cells += 1
            if i < 3:
                print(f"[DEBUG] Cell {i} ({common_cells[i]}): {mask.sum()} non-zero genes, corr={per_cell_corr[i]:.3f}")
    print(f"[DEBUG] Valid per-cell correlations: {n_valid_cells}/{n_cells}")

    per_cell_df = pd.DataFrame({"cell": common_cells, "pearson_corr": per_cell_corr})

    # -----------------------------
    # 4) Per-gene correlations (Spearman on co-expressing cells)
    # -----------------------------
    print(f"\n[DEBUG] ===== PER-GENE CORRELATIONS =====")
    per_gene_corr = np.full(n_genes, np.nan, dtype=float)
    per_gene_nco = np.zeros(n_genes, dtype=int)

    rna_T = rna_X.T
    atac_T = atac_X.T

    n_valid_genes = 0
    for j in tqdm(range(n_genes), desc="[correlation] per-gene", disable=not verbose):
        r = rna_T[j, :]
        a = atac_T[j, :]
        co_mask = (r != 0) & (a != 0)
        nco = int(co_mask.sum())
        per_gene_nco[j] = nco
        if nco >= min_cells_for_gene_corr and np.std(r[co_mask]) > 0 and np.std(a[co_mask]) > 0:
            try:
                per_gene_corr[j], _ = sp_stats.spearmanr(r[co_mask], a[co_mask])
                n_valid_genes += 1
                if j < 5:
                    print(f"[DEBUG] Gene {j} ({shared_genes[j]}): {nco} co-expressing cells, corr={per_gene_corr[j]:.3f}")
            except Exception:
                pass

    per_gene_df = pd.DataFrame({
        "gene": shared_genes,
        "spearman_corr": per_gene_corr,
        "n_coexpressing_cells": per_gene_nco
    })

    # -----------------------------
    # 5) Save CSVs
    # -----------------------------
    per_cell_csv = os.path.join(output_dir, "per_cell_correlations.csv")
    per_gene_csv = os.path.join(output_dir, "per_gene_correlations.csv")
    per_cell_df.to_csv(per_cell_csv, index=False)
    per_gene_df.to_csv(per_gene_csv, index=False)

    # -----------------------------
    # 6) Plots (PNG)
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Per-cell histogram
    valid_cc = np.isfinite(per_cell_corr)
    if valid_cc.any():
        axes[0, 0].hist(per_cell_corr[valid_cc], bins=50, edgecolor='black', alpha=0.8)
        axes[0, 0].axvline(np.nanmean(per_cell_corr), ls='--', color='r',
                           label=f"Mean={np.nanmean(per_cell_corr):.3f}")
        axes[0, 0].set_title("Per-cell Pearson correlation")
        axes[0, 0].set_xlabel("Correlation")
        axes[0, 0].set_ylabel("Cells")
        axes[0, 0].legend()

    # Per-gene histogram
    valid_gc = np.isfinite(per_gene_corr)
    if valid_gc.any():
        axes[0, 1].hist(per_gene_corr[valid_gc], bins=50, edgecolor='black', alpha=0.8, label="per-gene corr")
        axes[0, 1].axvline(np.nanmean(per_gene_corr), ls='--', color='r',
                           label=f"Mean={np.nanmean(per_gene_corr):.3f}")
        axes[0, 1].set_title(f"Per-gene Spearman correlation (n≥{min_cells_for_gene_corr} co-expressing cells)")
        axes[0, 1].set_xlabel("Correlation")
        axes[0, 1].set_ylabel("Genes")
        axes[0, 1].legend()

    # Co-expressing cells distribution (log scale on x)
    nz_co = per_gene_nco[per_gene_nco > 0]
    if nz_co.size > 0:
        axes[1, 0].hist(nz_co, bins=50, edgecolor='black', alpha=0.8)
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_title("Co-expressing cell counts per gene")
        axes[1, 0].set_xlabel("Number of co-expressing cells (log)")
        axes[1, 0].set_ylabel("Genes")

    # Corr vs co-expressing cells
    if valid_gc.any():
        axes[1, 1].scatter(per_gene_nco[valid_gc], per_gene_corr[valid_gc], s=8, alpha=0.5)
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_xlabel("Co-expressing cells (log)")
        axes[1, 1].set_ylabel("Gene correlation")
        axes[1, 1].set_title("Per-gene corr vs co-expressing cells")
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "correlation_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------
    # 7) Summary JSON
    # -----------------------------
    summary = {
        "n_paired_cells": int(n_cells),
        "n_shared_genes": int(n_genes),
        "per_cell_mean_corr": float(np.nanmean(per_cell_corr)) if valid_cc.any() else float("nan"),
        "per_cell_median_corr": float(np.nanmedian(per_cell_corr)) if valid_cc.any() else float("nan"),
        "per_gene_mean_corr": float(np.nanmean(per_gene_corr)) if valid_gc.any() else float("nan"),
        "per_gene_median_corr": float(np.nanmedian(per_gene_corr)) if valid_gc.any() else float("nan"),
        "min_cells_for_gene_corr": int(min_cells_for_gene_corr),
        "sample_genes": (int(sample_genes) if sample_genes is not None else None),
        "paths": {
            "per_cell_csv": per_cell_csv,
            "per_gene_csv": per_gene_csv,
            "plots_png": plot_path,
        },
    }
    with open(os.path.join(output_dir, "correlation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"[correlation] Saved:\n  {per_cell_csv}\n  {per_gene_csv}\n  {plot_path}")
    
    print("\n[DEBUG] ===== CORRELATION SUMMARY =====")
    print(f"[DEBUG] Per-cell correlation - Mean: {summary['per_cell_mean_corr']:.3f}, Median: {summary['per_cell_median_corr']:.3f}")
    print(f"[DEBUG] Per-gene correlation - Mean: {summary['per_gene_mean_corr']:.3f}, Median: {summary['per_gene_median_corr']:.3f}")

    return {
        "per_cell": per_cell_df,
        "per_gene": per_gene_df,
        "summary": summary,
    }


# =====================================================================
#                             ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("RNA–ATAC PSEUDOBULK + CORRELATION ANALYSIS (DEBUG VERSION)")
    print("="*60 + "\n")

    # Example paths (edit these to your dataset)
    rna_path = "/dcs07/hongkai/data/harry/result/gene_activity/old/rna_corrected.h5ad"
    atac_path = "/dcs07/hongkai/data/harry/result/gene_activity/signac_outputs/gene_activity.h5ad"
    out_dir = "/dcs07/hongkai/data/harry/result/gene_activity/signac_outputs/heart"

    # 1) Pseudobulk per cell type and Pearson correlation across genes
    result_pseudo = compare_atac_rna_gene_activity(
        atac_h5ad_path=atac_path,
        rna_h5ad_path=rna_path,
        output_dir=out_dir,
        batch_col="batch",          # adjust if needed
        celltype_key="celltype",
        unify_if_needed=True,
        unify_prefer="auto",
        unify_mapping_csv=None,
        atac_layer_for_unify="GeneActivity",
    )

    # 2) Optional: per-cell / per-gene correlation on the same aligned h5ad files
    print("\n" + "="*60)
    print("RUNNING CELL-LEVEL / GENE-LEVEL CORRELATION (OPTIONAL, DEBUG)")
    print("="*60 + "\n")

    adata_rna = sc.read_h5ad(rna_path)
    adata_atac = sc.read_h5ad(atac_path)

    result_single = compute_rna_atac_cell_gene_correlations(
        adata_rna,
        adata_atac,
        output_dir=os.path.join(out_dir, "cell_gene_corr"),
        unify_if_needed=True,
        unify_prefer="auto",
        unify_mapping_csv=None,
        atac_layer="GeneActivity",
    )

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60 + "\n")
