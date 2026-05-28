"""ARCHIVED: KNN-based synthetic ATAC gene-activity merge.

REMOVED FROM ACTIVE PIPELINE — kept for historical reference and
reproducibility of older results in `/dcs07/.../result/multi_omics_*`.

This function used to synthesise per-ATAC-cell pseudo-RNA expression by
KNN-weighted averaging of nearby RNA cells in the GLUE embedding space,
then concatenate it with the real RNA expression to form a single
``adata_sample.h5ad`` (cell-union, RNA-gene var-axis).

Why it was removed (2026-05-27 refactor):
  1. Pseudo-RNA expression for ATAC is not a measurement; using it for
     differential-expression analysis is statistically meaningless. The
     new pipeline runs DGE on RNA cells only.
  2. The cell-union AnnData no longer needs to carry expression at all —
     it carries only obs + obsm. Per-modality preprocessed h5ads
     (``adata_rna_preprocessed.h5ad`` / ``adata_atac_preprocessed.h5ad``)
     carry the real, modality-native expression for downstream analyses.
  3. The CPU fallback path introduced in commit 302b9be ("finish testing")
     had a bug where the resulting ``adata_sample.h5ad`` had ``X`` all
     zeros (RNA expression copy was being skipped), which then broke
     ``integrate_preprocess`` (filter_cells(min_genes=500) dropped every
     cell). Removing the function eliminates the bug class entirely.

Replacement: see ``preparation/multi_omics_merge.py``:
  - ``build_embedding_union`` (embedding-only union)
  - ``preprocess_rna_for_downstream`` (per-modality RNA QC + normalize)
  - ``preprocess_atac_for_downstream`` (per-modality ATAC QC + TF-IDF)

If you must re-run this for an older dataset, restore by:
  1. Re-import the dependencies (cupy/cuml-optional, cupyx.scipy.sparse,
     scipy.sparse, anndata, numpy, pandas, psutil, gc) at module top.
  2. Re-add the call site in multi_omics_glue.multiomics_preparation
     under `if run_gene_activity:`.
"""
from __future__ import annotations

# Original imports needed by the function below (some are optional GPU):
import contextlib
import gc
import io
import os
import time
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import scanpy as sc
from scipy import sparse

# Optional GPU stack — wrapped in try/except in the body, but listed here
# so the archived function is self-contained when restored.
# import cupy as cp
# from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
# from cupyx.scipy import sparse as cusparse

# `fix_sparse_matrix_dtype` and `safe_h5ad_write` originally came from utils.
from utils.safe_save import safe_h5ad_write


def compute_gene_activity_from_knn(
    glue_dir: str,
    output_path: str,
    raw_rna_path: str,
    k_neighbors: int = 1,
    use_rep: str = "X_glue",
    metric: str = "cosine",
    use_gpu: bool = True,
    verbose: bool = True,
) -> ad.AnnData:
    # Lazy GPU imports: RAPIDS (cuml/cupy/cupyx) can be broken on nodes with
    # a driver/runtime mismatch. Try to import; on failure transparently fall
    # back to a CPU implementation (sklearn k-NN + numpy einsum). Importing at
    # module level would block GLUE training itself on such nodes.
    gpu_ok = False
    cp = None
    cuNearestNeighbors = None
    cusparse = None
    if use_gpu:
        try:
            import cupy as cp  # noqa: F401
            from cuml.neighbors import NearestNeighbors as cuNearestNeighbors  # noqa: F401
            from cupyx.scipy import sparse as cusparse  # noqa: F401
            gpu_ok = True
        except Exception as exc:
            if verbose:
                print(f"   [gene_activity] GPU stack unavailable ({type(exc).__name__}: {exc}); "
                      f"falling back to CPU (sklearn k-NN).")

    def fix_sparse_matrix_dtype(X, verbose=False):
        if not sparse.issparse(X):
            return X
            
        if verbose:
            print(f"   Converting sparse matrix indices to int64...")
        
        coo = X.tocoo()
        X_fixed = sparse.csr_matrix(
            (coo.data.astype(np.float64), 
             (coo.row.astype(np.int64), coo.col.astype(np.int64))),
            shape=X.shape,
            dtype=np.float64
        )
        X_fixed.eliminate_zeros()
        X_fixed.sort_indices()
        
        return X_fixed
    
    if gpu_ok:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        gpu_mem = cp.cuda.Device().mem_info[0] / 1e9
    else:
        mempool = None
        pinned_mempool = None
        gpu_mem = 0.0
    cpu_mem = psutil.virtual_memory().available / 1e9
    
    rna_processed_path = os.path.join(glue_dir, "glue-rna-emb.h5ad")
    atac_path = os.path.join(glue_dir, "glue-atac-emb.h5ad")
    
    if not os.path.exists(rna_processed_path):
        raise FileNotFoundError(f"Processed RNA embedding file not found: {rna_processed_path}")
    if not os.path.exists(atac_path):
        raise FileNotFoundError(f"ATAC embedding file not found: {atac_path}")
    if not os.path.exists(raw_rna_path):
        raise FileNotFoundError(f"Raw RNA count file not found: {raw_rna_path}")
    
    if verbose:
        print(f"\n🧬 Computing gene activity using raw RNA counts...")
        print(f"   k_neighbors: {k_neighbors}")
        print(f"   metric: {metric}")
        print(f"   GPU acceleration: {'enabled' if use_gpu else 'disabled'}")
        print(f"   Available GPU memory: {gpu_mem:.2f} GB")
        print(f"   Available CPU memory: {cpu_mem:.2f} GB")
    
    if verbose:
        print("\n📂 Loading processed RNA embeddings and metadata...")
    
    rna_processed = ad.read_h5ad(rna_processed_path)
    rna_embedding = rna_processed.obsm[use_rep].copy()
    processed_rna_cells = rna_processed.obs.index.copy()
    rna_obsm_dict = {k: v.copy() for k, v in rna_processed.obsm.items()}
    processed_rna_obs = rna_processed.obs.copy()
    
    if verbose:
        print(f"   RNA cells: {len(processed_rna_cells)}")
        print(f"   Obs columns: {list(processed_rna_obs.columns)}")
    
    del rna_processed
    gc.collect()

    if verbose:
        print("\n📂 Loading ATAC embeddings...")
    
    atac = ad.read_h5ad(atac_path)
    atac_embedding = atac.obsm[use_rep].copy()
    atac_obs = atac.obs.copy()
    atac_obsm_dict = {k: v.copy() for k, v in atac.obsm.items()}
    n_atac_cells = atac.n_obs
    
    if verbose:
        print(f"   ATAC cells: {n_atac_cells}")
    
    del atac
    gc.collect()
    
    if verbose:
        print("\n📂 Loading raw RNA counts...")
    
    rna_raw = ad.read_h5ad(raw_rna_path)
    raw_rna_var = rna_raw.var.copy()
    raw_rna_varm_dict = {k: v.copy() for k, v in rna_raw.varm.items()} if hasattr(rna_raw, 'varm') else {}
    raw_rna_obs_index = rna_raw.obs.index.copy()
    
    if sparse.issparse(rna_raw.X):
        rna_X_full = rna_raw.X.tocsr()
    else:
        rna_X_full = rna_raw.X
    
    if verbose:
        print(f"   RNA matrix shape: {rna_X_full.shape}")
    
    del rna_raw
    gc.collect()

    if verbose:
        print("\n🔗 Aligning cells...")
    
    common_cells = processed_rna_cells.intersection(raw_rna_obs_index)
    
    if len(common_cells) == 0:
        raise ValueError("No common cells between processed and raw RNA!")
    
    if len(common_cells) != len(processed_rna_cells):
        if verbose:
            print(f"   Aligning to {len(common_cells)} common cells...")
        embedding_mask = np.isin(processed_rna_cells, common_cells)
        rna_embedding = rna_embedding[embedding_mask]
        
        for key in rna_obsm_dict:
            rna_obsm_dict[key] = rna_obsm_dict[key][embedding_mask]
    
    rna_obs = processed_rna_obs.loc[common_cells].copy()
    
    raw_rna_cell_to_idx = {cell: idx for idx, cell in enumerate(raw_rna_obs_index)}
    common_cells_list = list(common_cells)
    common_cells_raw_indices = np.array([raw_rna_cell_to_idx[cell] for cell in common_cells_list], dtype=np.int64)
    
    n_rna_cells = len(common_cells_list)
    n_genes = rna_X_full.shape[1]
    
    if verbose:
        print(f"   RNA cells: {n_rna_cells}, ATAC cells: {n_atac_cells}, Genes: {n_genes}")

    if verbose:
        print(f"\n🔍 Finding k-nearest RNA neighbors ({'GPU/cuML' if gpu_ok else 'CPU/sklearn'})...")

    is_sparse_rna = sparse.issparse(rna_X_full)
    gene_activity_matrix = np.zeros((n_atac_cells, n_genes), dtype=np.float64)

    if gpu_ok:
        rna_embedding_gpu = cp.asarray(rna_embedding, dtype=cp.float32)
        atac_embedding_gpu = cp.asarray(atac_embedding, dtype=cp.float32)
        del rna_embedding, atac_embedding
        gc.collect()

        nn = cuNearestNeighbors(
            n_neighbors=k_neighbors,
            metric=metric,
            algorithm='brute' if n_rna_cells < 50000 else 'auto'
        )
        nn.fit(rna_embedding_gpu)
        distances_gpu, indices_gpu = nn.kneighbors(atac_embedding_gpu)

        if verbose:
            print("\n📐 Computing similarity weights (GPU)...")
        if metric == 'cosine':
            similarities = 1 - (distances_gpu / 2)
        else:
            similarities = 1 / (1 + distances_gpu)
        min_sim = cp.min(similarities, axis=1, keepdims=True)
        max_sim = cp.max(similarities, axis=1, keepdims=True)
        sim_range = max_sim - min_sim
        all_equal = sim_range == 0
        if cp.any(all_equal):
            weights_gpu = cp.ones_like(similarities, dtype=cp.float32) / k_neighbors
            if not cp.all(all_equal):
                similarities = cp.where(all_equal, 0, (similarities - min_sim) / sim_range)
                similarities = similarities / cp.sum(similarities, axis=1, keepdims=True)
                weights_gpu = cp.where(all_equal, weights_gpu, similarities)
        else:
            similarities = (similarities - min_sim) / sim_range
            weights_gpu = similarities / cp.sum(similarities, axis=1, keepdims=True)
        if verbose:
            print(f"   Weight stats: min={float(cp.min(weights_gpu)):.6f}, max={float(cp.max(weights_gpu)):.6f}")
        del similarities, distances_gpu

        estimated_memory_per_cell = (k_neighbors * n_genes * 8) / 1e9
        optimal_batch_size = int(min(
            gpu_mem * 0.5 / estimated_memory_per_cell,
            10000,
            n_atac_cells
        ))
        optimal_batch_size = max(optimal_batch_size, 100)
        if verbose:
            print(f"\n🧮 Computing weighted gene activity (GPU)...")
            print(f"   Batch size: {optimal_batch_size}")
        n_batches = (n_atac_cells + optimal_batch_size - 1) // optimal_batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * optimal_batch_size
            end_idx = min((batch_idx + 1) * optimal_batch_size, n_atac_cells)
            batch_size_actual = end_idx - start_idx
            batch_indices_gpu = indices_gpu[start_idx:end_idx]
            batch_weights_gpu = weights_gpu[start_idx:end_idx]
            batch_indices_cpu = cp.asnumpy(batch_indices_gpu).flatten()
            unique_common_indices = np.unique(batch_indices_cpu)
            unique_raw_indices = common_cells_raw_indices[unique_common_indices]
            if is_sparse_rna:
                rna_expr_subset = rna_X_full[unique_raw_indices, :]
                if sparse.issparse(rna_expr_subset):
                    if rna_expr_subset.nnz / rna_expr_subset.size > 0.1:
                        rna_expr_subset = rna_expr_subset.toarray()
                    else:
                        rna_expr_subset = cusparse.csr_matrix(rna_expr_subset, dtype=cp.float32)
                else:
                    rna_expr_subset = np.asarray(rna_expr_subset)
            else:
                rna_expr_subset = rna_X_full[unique_raw_indices, :]
            raw_idx_to_subset_idx = {raw_idx: subset_idx for subset_idx, raw_idx in enumerate(unique_raw_indices)}
            common_to_subset = np.array([raw_idx_to_subset_idx[common_cells_raw_indices[ci]] for ci in unique_common_indices], dtype=np.int32)
            if not isinstance(rna_expr_subset, (cp.ndarray, cusparse.csr_matrix)):
                rna_expr_gpu = cp.asarray(rna_expr_subset, dtype=cp.float32)
            else:
                rna_expr_gpu = rna_expr_subset
            idx_map = cp.zeros(n_rna_cells, dtype=cp.int32) - 1
            idx_map[unique_common_indices] = cp.asarray(common_to_subset, dtype=cp.int32)
            mapped_indices_gpu = idx_map[batch_indices_gpu]
            batch_gene_activity_gpu = cp.zeros((batch_size_actual, n_genes), dtype=cp.float32)
            for i in range(batch_size_actual):
                cell_indices = mapped_indices_gpu[i]
                if isinstance(rna_expr_gpu, cusparse.csr_matrix):
                    neighbor_expr = rna_expr_gpu[cell_indices].toarray()
                else:
                    neighbor_expr = rna_expr_gpu[cell_indices]
                batch_gene_activity_gpu[i] = cp.einsum('n,ng->g',
                                                       batch_weights_gpu[i],
                                                       neighbor_expr)
            gene_activity_matrix[start_idx:end_idx] = cp.asnumpy(batch_gene_activity_gpu).astype(np.float64)
            del batch_gene_activity_gpu, rna_expr_gpu, mapped_indices_gpu
            if verbose and ((batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1):
                progress = (batch_idx + 1) / n_batches * 100
                print(f"   Progress: {progress:.1f}% ({batch_idx + 1}/{n_batches} batches)")
        del weights_gpu, indices_gpu, rna_embedding_gpu, atac_embedding_gpu
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    else:
        # CPU fallback: sklearn k-NN + numpy einsum
        from sklearn.neighbors import NearestNeighbors as skNearestNeighbors
        nn = skNearestNeighbors(
            n_neighbors=k_neighbors,
            metric=metric,
            algorithm='brute' if n_rna_cells < 50000 else 'auto',
            n_jobs=-1,
        )
        nn.fit(np.asarray(rna_embedding, dtype=np.float32))
        distances, indices = nn.kneighbors(np.asarray(atac_embedding, dtype=np.float32))
        del rna_embedding, atac_embedding
        gc.collect()

        if verbose:
            print("\n📐 Computing similarity weights (CPU)...")
        if metric == 'cosine':
            similarities = 1.0 - (distances / 2.0)
        else:
            similarities = 1.0 / (1.0 + distances)
        min_sim = similarities.min(axis=1, keepdims=True)
        max_sim = similarities.max(axis=1, keepdims=True)
        sim_range = max_sim - min_sim
        all_equal = (sim_range == 0)
        if np.any(all_equal):
            weights = np.ones_like(similarities, dtype=np.float32) / k_neighbors
            if not np.all(all_equal):
                normed = np.where(all_equal, 0.0, (similarities - min_sim) / np.where(sim_range == 0, 1, sim_range))
                normed = normed / np.where(normed.sum(axis=1, keepdims=True) == 0, 1, normed.sum(axis=1, keepdims=True))
                weights = np.where(all_equal, weights, normed)
        else:
            normed = (similarities - min_sim) / sim_range
            weights = normed / normed.sum(axis=1, keepdims=True)
        weights = weights.astype(np.float32)
        if verbose:
            print(f"   Weight stats: min={float(weights.min()):.6f}, max={float(weights.max()):.6f}")
        del similarities, distances

        # Batched weighted aggregation; cap RAM by limiting B*k*n_genes*4 bytes
        bytes_per_cell = k_neighbors * n_genes * 4
        cpu_budget_bytes = cpu_mem * 0.25 * 1e9
        optimal_batch_size = int(min(max(100, cpu_budget_bytes / max(bytes_per_cell, 1)), 5000, n_atac_cells))
        if verbose:
            print(f"\n🧮 Computing weighted gene activity (CPU)...")
            print(f"   Batch size: {optimal_batch_size}")
        n_batches = (n_atac_cells + optimal_batch_size - 1) // optimal_batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * optimal_batch_size
            end_idx = min((batch_idx + 1) * optimal_batch_size, n_atac_cells)
            batch_indices = indices[start_idx:end_idx]            # (B, k) into common_cells space
            batch_weights = weights[start_idx:end_idx]            # (B, k)
            flat = batch_indices.ravel()
            unique_common, inverse = np.unique(flat, return_inverse=True)
            unique_raw = common_cells_raw_indices[unique_common]
            if is_sparse_rna:
                rna_expr_subset = rna_X_full[unique_raw, :].toarray().astype(np.float32)
            else:
                rna_expr_subset = np.asarray(rna_X_full[unique_raw, :], dtype=np.float32)
            mapped = inverse.reshape(batch_indices.shape)         # (B, k) into rna_expr_subset rows
            neighbor_expr = rna_expr_subset[mapped]               # (B, k, n_genes)
            batch_activity = np.einsum('bk,bkg->bg', batch_weights, neighbor_expr)
            gene_activity_matrix[start_idx:end_idx] = batch_activity.astype(np.float64)
            del neighbor_expr, rna_expr_subset, batch_activity
            if verbose and ((batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1):
                progress = (batch_idx + 1) / n_batches * 100
                print(f"   Progress: {progress:.1f}% ({batch_idx + 1}/{n_batches} batches)")
        del weights, indices
    
    if verbose:
        print("\n📦 Creating gene activity AnnData...")
    
    gene_activity_matrix = np.nan_to_num(gene_activity_matrix, 0)
    np.clip(gene_activity_matrix, 0, None, out=gene_activity_matrix)
    
    gene_activity_sparse = sparse.csr_matrix(gene_activity_matrix, dtype=np.float64)
    gene_activity_sparse = fix_sparse_matrix_dtype(gene_activity_sparse, verbose=verbose)
    
    del gene_activity_matrix
    gc.collect()
    
    gene_activity_adata = ad.AnnData(
        X=gene_activity_sparse,
        obs=atac_obs.copy(),
        var=raw_rna_var.copy()
    )
    
    gene_activity_adata.obs['modality'] = 'ATAC'
    gene_activity_adata.layers['gene_activity'] = gene_activity_sparse.copy()
    
    for key, value in atac_obsm_dict.items():
        gene_activity_adata.obsm[key] = value
    
    for key, value in raw_rna_varm_dict.items():
        gene_activity_adata.varm[key] = value
    
    if verbose:
        print("\n📦 Creating RNA AnnData for merging...")
    
    if is_sparse_rna:
        rna_X = rna_X_full[common_cells_raw_indices, :]
        if sparse.issparse(rna_X):
            rna_X = rna_X.tocsr().astype(np.float64)
            rna_X = fix_sparse_matrix_dtype(rna_X, verbose=verbose)
        else:
            rna_X = np.asarray(rna_X).astype(np.float64)
            nnz = np.count_nonzero(rna_X)
            sparsity = 1 - (nnz / rna_X.size)
            if sparsity > 0.5:
                rna_X = sparse.csr_matrix(rna_X, dtype=np.float64)
                rna_X = fix_sparse_matrix_dtype(rna_X, verbose=verbose)
    else:
        rna_X = rna_X_full[common_cells_raw_indices, :].astype(np.float64)
        nnz = np.count_nonzero(rna_X)
        sparsity = 1 - (nnz / rna_X.size)
        if sparsity > 0.5:
            rna_X = sparse.csr_matrix(rna_X, dtype=np.float64)
            rna_X = fix_sparse_matrix_dtype(rna_X, verbose=verbose)
    
    del rna_X_full
    gc.collect()
    
    rna_for_merge = ad.AnnData(
        X=rna_X,
        obs=rna_obs.copy(),
        var=raw_rna_var.copy()
    )
    
    rna_for_merge.obs['modality'] = 'RNA'
    
    for key, value in rna_obsm_dict.items():
        rna_for_merge.obsm[key] = value
    
    for key, value in raw_rna_varm_dict.items():
        rna_for_merge.varm[key] = value
    
    if verbose:
        print("\n🔗 Merging RNA and ATAC datasets...")
    
    rna_indices = set(rna_for_merge.obs.index)
    atac_indices = set(gene_activity_adata.obs.index)
    overlap = rna_indices.intersection(atac_indices)
    
    if verbose and overlap:
        print(f"   Found {len(overlap)} overlapping indices, adding modality suffix...")
    
    rna_for_merge.obs['original_barcode'] = rna_for_merge.obs.index
    gene_activity_adata.obs['original_barcode'] = gene_activity_adata.obs.index
    
    rna_for_merge.obs.index = pd.Index([f"{idx}_RNA" for idx in rna_for_merge.obs.index])
    gene_activity_adata.obs.index = pd.Index([f"{idx}_ATAC" for idx in gene_activity_adata.obs.index])
    
    if verbose:
        print(f"   RNA cells: {rna_for_merge.n_obs}")
        print(f"   ATAC cells: {gene_activity_adata.n_obs}")
    
    merged_adata = ad.concat(
        [rna_for_merge, gene_activity_adata], 
        axis=0, 
        join='inner',
        merge='same',
        label=None,
        keys=None,
        index_unique=None
    )
    
    del rna_for_merge, gene_activity_adata
    gc.collect()
    
    if not merged_adata.obs.index.is_unique:
        if verbose:
            print("   ⚠️ Fixing non-unique indices...")
        merged_adata.obs_names_make_unique()
    
    if verbose:
        print("\n💾 Saving merged dataset...")
    
    if sparse.issparse(merged_adata.X):
        if not isinstance(merged_adata.X, sparse.csr_matrix):
            merged_adata.X = merged_adata.X.tocsr()
        merged_adata.X = fix_sparse_matrix_dtype(merged_adata.X, verbose=verbose)
        merged_adata.X.sort_indices()
        merged_adata.X.eliminate_zeros()
    
    output_dir_path = os.path.join(output_path, 'preprocess')
    os.makedirs(output_dir_path, exist_ok=True)
    output_path_anndata = os.path.join(output_dir_path, 'adata_sample.h5ad')
    safe_h5ad_write(merged_adata, output_path_anndata)
    
    if verbose:
        print(f"\n✅ Gene activity computation complete!")
        print(f"   Output: {output_path_anndata}")
        print(f"   Shape: {merged_adata.shape}")
        print(f"   RNA cells: {(merged_adata.obs['modality'] == 'RNA').sum()}")
        print(f"   ATAC cells: {(merged_adata.obs['modality'] == 'ATAC').sum()}")
        print(f"   Obs columns: {list(merged_adata.obs.columns)}")

    if gpu_ok and mempool is not None:
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    gc.collect()

    return merged_adata
