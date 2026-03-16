#!/usr/bin/env python3

import os
import gc
import signal
import io
import warnings
import contextlib
from typing import Tuple, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse, csr_matrix

import cupy as cp
import rapids_singlecell as rsc

from utils.tf_idf import tfidf_memory_efficient
from utils.random_seed import set_global_seed


def clear_gpu_memory():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def _as_list(x: Optional[Union[str, List[str]]]) -> List[str]:
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def _get_nan_mask(X) -> np.ndarray:
    if issparse(X):
        X = X.toarray()
    return np.isnan(X).any(axis=0)


def _extract_sample_metadata(
    cell_adata: sc.AnnData,
    sample_adata: sc.AnnData,
    sample_col: str,
    exclude_cols: Optional[List[str]] = None,
) -> sc.AnnData:
    exclude = set(exclude_cols or []) | {sample_col}
    grouped = cell_adata.obs.groupby(sample_col)

    meta = {}
    for col in cell_adata.obs.columns:
        if col in exclude:
            continue
        uniques = grouped[col].apply(lambda x: x.dropna().unique())
        if uniques.apply(lambda u: len(u) <= 1).all():
            meta[col] = uniques.apply(lambda u: u[0] if len(u) else np.nan)

    if meta:
        meta_df = pd.DataFrame(meta)
        overlap = sample_adata.obs.columns.intersection(meta_df.columns)
        if len(overlap) > 0:
            meta_df = meta_df.drop(columns=list(overlap))
        if meta_df.shape[1] > 0:
            sample_adata.obs = sample_adata.obs.join(meta_df, how="left")

    return sample_adata


def _try_combat(
    adata: sc.AnnData,
    batch_col: str,
    preserve_cols: List[str],
    timeout: float,
    verbose: bool
) -> bool:
    def timeout_handler(signum, frame):
        raise TimeoutError("ComBat timed out")

    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler) if timeout else None
        if timeout:
            signal.alarm(int(timeout))

        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                sc.pp.combat(
                    adata,
                    key=batch_col,
                    covariates=preserve_cols or None,
                    inplace=True
                )
            if verbose:
                print(f"    Applied ComBat" + (f" (kept={preserve_cols})" if preserve_cols else ""))
            return True
        finally:
            if timeout:
                signal.alarm(0)
                if old_handler:
                    signal.signal(signal.SIGALRM, old_handler)

    except (TimeoutError, Exception) as e:
        if verbose:
            print(f"    ComBat failed: {type(e).__name__}: {e}")
        return False


def _try_limma(
    adata: sc.AnnData,
    batch_col: str,
    preserve_cols: List[str],
    verbose: bool
) -> bool:
    try:
        from utils.limma import limma

        X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)

        if preserve_cols:
            terms = [f'Q("{c}")' for c in preserve_cols]
            keep_formula = "~ " + " + ".join(terms)
        else:
            keep_formula = "1"

        adata.X = limma(
            pheno=adata.obs,
            exprs=X,
            covariate_formula=keep_formula,
            design_formula=f'~ Q("{batch_col}")',
            rcond=1e-8,
            verbose=False
        )

        if verbose:
            print(f"    Applied limma" + (f" (kept={preserve_cols})" if preserve_cols else ""))
        return True

    except Exception as e:
        if verbose:
            print(f"    Limma failed: {type(e).__name__}: {e}")
        return False


def apply_batch_correction(
    adata: sc.AnnData,
    batch_col: str,
    preserve_cols: List[str],
    combat_timeout: float,
    verbose: bool
) -> bool:
    batch_counts = adata.obs[batch_col].value_counts()
    n_batches = len(batch_counts)

    if verbose:
        print(f"    Batches: {n_batches}, sizes: {batch_counts.min()}-{batch_counts.max()}")

    if n_batches <= 1:
        if verbose:
            print("    Skipping: only 1 batch")
        return False

    if batch_counts.min() < 2:
        if verbose:
            print("    Skipping ComBat: batches <2 samples")
        return _try_limma(adata, batch_col, preserve_cols, verbose)

    if _try_combat(adata, batch_col, preserve_cols, combat_timeout, verbose):
        return True
    return _try_limma(adata, batch_col, preserve_cols, verbose)


def aggregate_pseudobulk(
    adata: sc.AnnData,
    sample_col: str,
    celltype_col: str,
    batch_cols: List[str],
    verbose: bool,
    modality_col: Optional[str] = None,
    modality_filter: Optional[str] = None
) -> sc.AnnData:
    if modality_col and modality_filter and modality_col in adata.obs.columns:
        mask = adata.obs[modality_col] == modality_filter
        adata_work = adata[mask]
        if verbose:
            print(f"  Filtered to {modality_filter} cells: {adata_work.n_obs} / {adata.n_obs}")
    else:
        adata_work = adata

    samples = sorted(adata_work.obs[sample_col].unique())
    cell_types = sorted(adata_work.obs[celltype_col].unique())
    n_samples, n_genes = len(samples), adata_work.n_vars

    if verbose:
        print(f"  Aggregating {adata_work.n_obs} cells -> {n_samples} samples x {len(cell_types)} cell types")

    sample_idx = {s: i for i, s in enumerate(samples)}
    ct_idx = {ct: i for i, ct in enumerate(cell_types)}

    obs = pd.DataFrame(index=samples)
    obs.index.name = 'sample'
    for bc in batch_cols:
        mapping = adata_work.obs[[sample_col, bc]].drop_duplicates().set_index(sample_col)[bc].to_dict()
        obs[bc] = [mapping.get(s, 'Unknown') for s in samples]

    pb = sc.AnnData(
        X=np.zeros((n_samples, n_genes), dtype=np.float32),
        obs=obs,
        var=adata_work.var.copy()
    )

    cell_sample = adata_work.obs[sample_col].map(sample_idx).values
    cell_ct = adata_work.obs[celltype_col].map(ct_idx).values
    valid = ~(pd.isna(cell_sample) | pd.isna(cell_ct))
    valid_idx = np.where(valid)[0]
    cell_sample_valid = cell_sample[valid].astype(int)
    cell_ct_valid = cell_ct[valid].astype(int)

    X = adata_work.X.tocsr() if issparse(adata_work.X) else adata_work.X

    for i, ct in enumerate(cell_types):
        mask = cell_ct_valid == i
        ct_cells = valid_idx[mask]
        ct_samples = cell_sample_valid[mask]

        if len(ct_cells) == 0:
            pb.layers[ct] = np.zeros((n_samples, n_genes), dtype=np.float32)
            continue

        indicator = csr_matrix(
            (np.ones(len(ct_cells), dtype=np.float32), (ct_samples, np.arange(len(ct_cells)))),
            shape=(n_samples, len(ct_cells))
        )

        counts = np.array(indicator.sum(axis=1)).flatten()
        counts[counts == 0] = 1

        X_ct = X[ct_cells, :]
        sums = indicator @ X_ct
        if issparse(sums):
            sums = np.asarray(sums.todense())

        pb.layers[ct] = (sums / counts[:, None]).astype(np.float32)

    if verbose:
        print(f"  Created {len(cell_types)} cell type layers")

    return pb


def _normalize_layer(
    adata: sc.AnnData,
    normalize: bool,
    target_sum: float,
    atac: bool
) -> sc.AnnData:
    if not normalize:
        return adata
    
    if atac:
        tfidf_memory_efficient(adata, scale_factor=target_sum)
    else:
        rsc.get.anndata_to_GPU(adata)
        rsc.pp.normalize_total(adata, target_sum=target_sum)
        rsc.pp.log1p(adata)
        rsc.get.anndata_to_CPU(adata)
    return adata


def _remove_nan_genes(adata: sc.AnnData, verbose: bool, context: str = "") -> sc.AnnData:
    nan_mask = _get_nan_mask(adata.X)
    if nan_mask.any():
        if verbose and context:
            print(f"    Removing {nan_mask.sum()} NaN genes {context}")
        return adata[:, ~nan_mask].copy()
    return adata


def _apply_batch_and_clean(
    adata: sc.AnnData,
    batch_col: Optional[str],
    preserve_cols: List[str],
    combat_timeout: float,
    verbose: bool
) -> sc.AnnData:
    if batch_col and batch_col in adata.obs.columns:
        if apply_batch_correction(adata, batch_col, preserve_cols, combat_timeout, verbose):
            adata = _remove_nan_genes(adata, verbose, "post-correction")
    return adata


def process_celltype_layer(
    layer_data: np.ndarray,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    cell_type: str,
    batch_col: Optional[str],
    preserve_cols: List[str],
    n_features: int,
    normalize: bool,
    target_sum: float,
    atac: bool,
    combat_timeout: float,
    verbose: bool,
    hvg_genes: Optional[List[str]] = None
) -> Optional[Tuple[List[str], np.ndarray, List[str]]]:
    temp = sc.AnnData(X=layer_data.copy(), obs=obs.copy(), var=var.copy())

    sc.pp.filter_genes(temp, min_cells=1)
    if temp.n_vars == 0:
        if verbose:
            print(f"    Skipping: no genes after filtering")
        return None

    temp = _normalize_layer(temp, normalize, target_sum, atac)
    temp = _remove_nan_genes(temp, verbose)
    temp = _apply_batch_and_clean(temp, batch_col, preserve_cols, combat_timeout, verbose)

    if hvg_genes is None:
        n_hvgs = min(n_features, temp.n_vars)
        rsc.get.anndata_to_GPU(temp)
        rsc.pp.highly_variable_genes(temp, n_top_genes=n_hvgs)
        rsc.get.anndata_to_CPU(temp)

        hvg_mask = temp.var["highly_variable"].values
        selected_genes = temp.var.index[hvg_mask].tolist()
        
        if not selected_genes:
            if verbose:
                print(f"    Skipping: no HVGs found")
            return None

        hvg_expr = temp[:, hvg_mask].X
        if issparse(hvg_expr):
            hvg_expr = hvg_expr.toarray()

        if verbose:
            print(f"    Selected {len(selected_genes)} HVGs")

        prefixed = [f"{cell_type} - {g}" for g in selected_genes]
        return prefixed, hvg_expr, selected_genes
    else:
        available = [g for g in hvg_genes if g in temp.var.index]
        if not available:
            if verbose:
                print(f"    Skipping: no HVGs available")
            return None

        hvg_expr = temp[:, available].X
        if issparse(hvg_expr):
            hvg_expr = hvg_expr.toarray()

        if verbose:
            print(f"    Using {len(available)} HVGs")

        prefixed = [f"{cell_type} - {g}" for g in available]
        return prefixed, hvg_expr, None


def compute_proportions(
    adata: sc.AnnData,
    samples: List[str],
    cell_types: List[str],
    sample_col: str,
    celltype_col: str
) -> pd.DataFrame:
    sample_ct_counts = adata.obs.groupby([sample_col, celltype_col]).size().unstack(fill_value=0)
    sample_totals = sample_ct_counts.sum(axis=1)
    
    props = pd.DataFrame(index=cell_types, columns=samples, dtype=float)
    for ct in cell_types:
        if ct in sample_ct_counts.columns:
            for sample in samples:
                if sample in sample_ct_counts.index:
                    total = sample_totals[sample]
                    props.loc[ct, sample] = sample_ct_counts.loc[sample, ct] / total if total > 0 else 0.0
                else:
                    props.loc[ct, sample] = 0.0
        else:
            props.loc[ct, :] = 0.0
    
    return props


def _build_expression_dataframe(
    cell_types: List[str],
    samples: List[str],
    final_genes: set,
    hvg_data: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    ct_groups: Dict[str, List[str]] = {}
    for g in final_genes:
        ct = g.split(' - ', 1)[0]
        ct_groups.setdefault(ct, []).append(g)
    for ct in ct_groups:
        ct_groups[ct].sort()

    expr_df = pd.DataFrame(index=sorted(cell_types), columns=samples, dtype=object)
    
    for ct in expr_df.index:
        genes = ct_groups.get(ct, [])
        if not genes:
            for sample in samples:
                expr_df.at[ct, sample] = pd.Series(dtype=float)
            continue
            
        for sample in samples:
            sample_data = hvg_data.get(sample, {})
            vals = []
            names = []
            for g in genes:
                if g in sample_data:
                    vals.append(sample_data[g])
                    names.append(g)
            expr_df.at[ct, sample] = pd.Series(vals, index=names) if vals else pd.Series(dtype=float)

    return expr_df


def compute_pseudobulk_gpu(
    adata: sc.AnnData,
    batch_col: Optional[Union[str, List[str]]] = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './',
    n_features: int = 2000,
    normalize: bool = True,
    target_sum: float = 1e4,
    atac: bool = False,
    verbose: bool = False,
    combat_timeout: float = 20.0,
    preserve_cols: Optional[Union[str, List[str]]] = None,
    hvg_modality: str = 'RNA',
    modality_col: str = 'modality',
) -> Tuple[pd.DataFrame, pd.DataFrame, sc.AnnData]:
    if verbose:
        print(f"[Pseudobulk] Using GPU (rapids_singlecell)")
        print(f"[Pseudobulk] HVG selection will use {hvg_modality} cells only (both rounds)")
    clear_gpu_memory()

    os.makedirs(os.path.join(output_dir, "pseudobulk"), exist_ok=True)

    batch_cols = [b for b in _as_list(batch_col)
                  if b and b in adata.obs.columns and not adata.obs[b].isnull().all()]
    preserve_list = [c for c in _as_list(preserve_cols) if c in adata.obs.columns]

    for col in batch_cols + preserve_list:
        if adata.obs[col].isnull().any():
            adata.obs[col] = adata.obs[col].fillna("Unknown")

    samples = sorted(adata.obs[sample_col].unique())
    cell_types = sorted(adata.obs[celltype_col].unique())

    if verbose:
        print(f"[Pseudobulk] {len(samples)} samples, {len(cell_types)} cell types")
        print(f"[Pseudobulk] Batch correction: {batch_cols if batch_cols else 'disabled'}")

    has_modality = modality_col in adata.obs.columns
    if has_modality and verbose:
        print(f"[Pseudobulk] Modality distribution: {dict(adata.obs[modality_col].value_counts())}")

    if verbose:
        print("[Pseudobulk] Phase 1: Aggregating cells...")
    
    pb_all = aggregate_pseudobulk(adata, sample_col, celltype_col, batch_cols, verbose)
    
    if has_modality:
        pb_rna = aggregate_pseudobulk(
            adata, sample_col, celltype_col, batch_cols, verbose,
            modality_col=modality_col, modality_filter=hvg_modality
        )
    else:
        pb_rna = pb_all

    if preserve_list:
        pb_all = _extract_sample_metadata(adata, pb_all, sample_col)
        preserve_list = [c for c in preserve_list if c in pb_all.obs.columns]
        if has_modality:
            pb_rna = _extract_sample_metadata(
                adata[adata.obs[modality_col] == hvg_modality] if has_modality else adata,
                pb_rna, sample_col
            )

    combined_batch = None
    if batch_cols:
        combined_batch = "_combined_batch_"
        if len(batch_cols) == 1:
            pb_all.obs[combined_batch] = pb_all.obs[batch_cols[0]].astype(str)
            pb_rna.obs[combined_batch] = pb_rna.obs[batch_cols[0]].astype(str)
        else:
            pb_all.obs[combined_batch] = pb_all.obs[batch_cols].astype(str).agg("|".join, axis=1)
            pb_rna.obs[combined_batch] = pb_rna.obs[batch_cols].astype(str).agg("|".join, axis=1)

    if verbose:
        print("[Pseudobulk] Phase 2: Processing cell types...")

    rna_cell_types = set(pb_rna.layers.keys())
    
    hvg_data_all: Dict[str, Dict[str, float]] = {}
    hvg_data_rna: Dict[str, Dict[str, float]] = {}
    all_genes: List[str] = []
    valid_cts: List[str] = []

    for i, ct in enumerate(cell_types):
        if verbose:
            print(f"  [{i+1}/{len(cell_types)}] {ct}")

        try:
            if ct not in rna_cell_types:
                if verbose:
                    print(f"    Skipping: not in {hvg_modality} data")
                continue

            result_rna = process_celltype_layer(
                pb_rna.layers[ct], pb_rna.obs, pb_rna.var, ct,
                combined_batch, preserve_list, n_features,
                normalize, target_sum, atac, combat_timeout, verbose,
                hvg_genes=None
            )

            if result_rna is None:
                continue

            prefixed_rna, expr_rna, base_hvgs = result_rna

            for j, sample in enumerate(pb_rna.obs.index):
                hvg_data_rna.setdefault(sample, {}).update(
                    {g: float(expr_rna[j, k]) for k, g in enumerate(prefixed_rna)}
                )

            if has_modality and pb_all is not pb_rna:
                if verbose:
                    print(f"    Extracting from ALL cells...")
                
                result_all = process_celltype_layer(
                    pb_all.layers[ct], pb_all.obs, pb_all.var, ct,
                    combined_batch, preserve_list, n_features,
                    normalize, target_sum, atac, combat_timeout, verbose,
                    hvg_genes=base_hvgs
                )

                if result_all is not None:
                    prefixed_all, expr_all, _ = result_all
                    for j, sample in enumerate(pb_all.obs.index):
                        hvg_data_all.setdefault(sample, {}).update(
                            {g: float(expr_all[j, k]) for k, g in enumerate(prefixed_all)}
                        )
                    all_genes.extend(prefixed_all)
                else:
                    for j, sample in enumerate(pb_rna.obs.index):
                        hvg_data_all.setdefault(sample, {}).update(
                            {g: float(expr_rna[j, k]) for k, g in enumerate(prefixed_rna)}
                        )
                    all_genes.extend(prefixed_rna)
            else:
                for j, sample in enumerate(pb_rna.obs.index):
                    hvg_data_all.setdefault(sample, {}).update(
                        {g: float(expr_rna[j, k]) for k, g in enumerate(prefixed_rna)}
                    )
                all_genes.extend(prefixed_rna)

            valid_cts.append(ct)
            clear_gpu_memory()

        except Exception as e:
            if verbose:
                print(f"    Error: {type(e).__name__}: {e}")

    cell_types = valid_cts
    if verbose:
        print(f"[Pseudobulk] Retained {len(cell_types)} cell types")

    if verbose:
        print(f"[Pseudobulk] Phase 3: Round 2 HVG selection from {hvg_modality} cells...")

    unique_genes = sorted(set(all_genes))
    gene_idx = {g: j for j, g in enumerate(unique_genes)}

    concat_rna = np.zeros((len(samples), len(unique_genes)), dtype=np.float32)
    for i, sample in enumerate(samples):
        if sample in hvg_data_rna:
            for g, v in hvg_data_rna[sample].items():
                if g in gene_idx:
                    concat_rna[i, gene_idx[g]] = v

    concat_rna_ad = sc.AnnData(
        X=concat_rna,
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=unique_genes)
    )

    sc.pp.filter_genes(concat_rna_ad, min_cells=1)
    n_final = min(n_features, concat_rna_ad.n_vars)
    rsc.get.anndata_to_GPU(concat_rna_ad)
    rsc.pp.highly_variable_genes(concat_rna_ad, n_top_genes=n_final)
    rsc.get.anndata_to_CPU(concat_rna_ad)

    hvg_round2 = concat_rna_ad.var.index[concat_rna_ad.var["highly_variable"]].tolist()
    hvg_round2_set = set(hvg_round2)

    if verbose:
        print(f"[Pseudobulk] Phase 4: Building final matrix...")

    final_gene_idx = {g: j for j, g in enumerate(hvg_round2)}
    concat_final = np.zeros((len(samples), len(hvg_round2)), dtype=np.float32)
    
    for i, sample in enumerate(samples):
        if sample in hvg_data_all:
            for g, v in hvg_data_all[sample].items():
                if g in final_gene_idx:
                    concat_final[i, final_gene_idx[g]] = v

    final_adata = sc.AnnData(
        X=concat_final,
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=hvg_round2)
    )

    if verbose:
        print(f"[Pseudobulk] Final features: {final_adata.n_vars}")

    expr_df = _build_expression_dataframe(cell_types, samples, hvg_round2_set, hvg_data_all)

    if verbose:
        print("[Pseudobulk] Phase 5: Computing proportions...")
    
    props = compute_proportions(adata, samples, cell_types, sample_col, celltype_col)
    final_adata.uns["cell_proportions"] = props

    clear_gpu_memory()

    if verbose:
        print("[Pseudobulk] Complete")

    return expr_df, props, final_adata


def compute_pseudobulk_adata_linux(
    adata: sc.AnnData,
    batch_col: Optional[Union[str, List[str]]] = None,
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './',
    save: bool = True,
    n_features: int = 2000,
    normalize: bool = True,
    target_sum: float = 1e4,
    atac: bool = False,
    verbose: bool = False,
    combat_timeout: float = 20.0,
    preserve_cols: Optional[Union[str, List[str]]] = None,
    hvg_modality: str = 'RNA',
    modality_col: str = 'modality',
) -> Tuple[Dict, sc.AnnData]:
    if verbose:
        print(f"[Pseudobulk] Input: {adata.n_obs} cells, {adata.n_vars} genes")
        print(f"[Pseudobulk] HVG selection modality: {hvg_modality} (both rounds)")

    set_global_seed(seed=42)
    
    expr_df, props, final = compute_pseudobulk_gpu(
        adata=adata,
        batch_col=batch_col,
        sample_col=sample_col,
        celltype_col=celltype_col,
        output_dir=output_dir,
        n_features=n_features,
        normalize=normalize,
        target_sum=target_sum,
        atac=atac,
        verbose=verbose,
        combat_timeout=combat_timeout,
        preserve_cols=preserve_cols,
        hvg_modality=hvg_modality,
        modality_col=modality_col,
    )

    if verbose:
        print("[Pseudobulk] Extracting sample metadata...")

    final = _extract_sample_metadata(adata, final, sample_col)
    final.uns["cell_proportions"] = props

    if save:
        path = os.path.join(output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
        if verbose:
            print(f"[Pseudobulk] Saving to {path}")
        sc.write(path, final)

    if verbose:
        print(f"[Pseudobulk] Output: {final.n_obs} samples, {final.n_vars} features")

    return {
        "cell_expression_corrected": expr_df,
        "cell_proportion": props.T
    }, final