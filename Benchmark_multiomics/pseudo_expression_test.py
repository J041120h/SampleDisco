#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMPROVED GENE CORRELATION BUILD
Modified gene correlation to only consider cells where both RNA and ATAC express the gene.

Changes from previous version:
- Gene correlation now filters for cells with non-zero expression in both modalities
- More meaningful correlation by focusing on actual expression patterns
- Clearer debug output for gene correlation statistics
"""

import os
import gc
import re
import json
import h5py
import traceback
import warnings
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

warnings.filterwarnings('ignore')


# ───────────────────────── Debug helpers ─────────────────────────

def _open_debug_log(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    return open(os.path.join(output_dir, "debug.log"), "a", buffering=1)

def _dprint(fp, *args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    if fp:
        fp.write(msg + "\n")

def _dump_json(output_dir: str, name: str, obj: dict):
    with open(os.path.join(output_dir, name), "w") as f:
        json.dump(obj, f, indent=2, default=str)

def _dump_text(output_dir: str, name: str, lines):
    path = os.path.join(output_dir, name)
    with open(path, "w") as f:
        if isinstance(lines, (list, tuple, np.ndarray, pd.Index)):
            for ln in lines:
                f.write(str(ln) + "\n")
        else:
            f.write(str(lines))

def _dump_obs_preview(output_dir: str, obs_df: pd.DataFrame, dbg_fp):
    try:
        obs_df.head(50).to_csv(os.path.join(output_dir, "obs_preview.csv"))
        _dprint(dbg_fp, "[debug] Wrote obs_preview.csv (first 50 rows).")
    except Exception as e:
        _dprint(dbg_fp, "[debug] Failed to write obs_preview.csv:", repr(e))

def _dump_obs_value_counts(output_dir: str, obs_df: pd.DataFrame, col: str, dbg_fp):
    try:
        vc = obs_df[col].value_counts(dropna=False)
        vc.to_csv(os.path.join(output_dir, f"obs_value_counts_{col}.csv"))
        _dprint(dbg_fp, f"[debug] Wrote obs_value_counts_{col}.csv")
    except Exception as e:
        _dprint(dbg_fp, f"[debug] Failed to write obs_value_counts_{col}.csv:", repr(e))

def _dump_h5_structure(h5_path: str, output_dir: str, dbg_fp):
    def walk(name, obj, out_lines):
        if isinstance(obj, h5py.Dataset):
            out_lines.append(f"[DATASET] {name} shape={obj.shape} dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            out_lines.append(f"[GROUP]   {name}")
    try:
        lines = []
        with h5py.File(h5_path, "r") as f:
            f.visititems(lambda n, o: walk(n, o, lines))
        _dump_text(output_dir, "h5_structure.txt", lines)
        _dprint(dbg_fp, "[debug] Wrote h5_structure.txt")
    except Exception as e:
        _dprint(dbg_fp, "[debug] Failed to dump H5 structure:", repr(e))

def _dump_indices_preview(output_dir: str, name: str, arr: np.ndarray, dbg_fp, limit: int = 100):
    try:
        head = arr[:limit]
        _dump_text(output_dir, f"{name}_head.txt", head)
        _dprint(dbg_fp, f"[debug] Wrote {name}_head.txt (n={len(head)})")
    except Exception as e:
        _dprint(dbg_fp, f"[debug] Failed to write {name}_head.txt:", repr(e))

def _show_name_samples(rna_obs: pd.DataFrame, atac_obs: pd.DataFrame, dbg_fp, n: int = 10):
    """Print a few RNA/ATAC cell names (index + original_barcode if available)."""
    try:
        _dprint(dbg_fp, "\n[DEBUG] RNA index examples:", list(map(str, rna_obs.index[:n])))
        _dprint(dbg_fp, "[DEBUG] ATAC index examples:", list(map(str, atac_obs.index[:n])))
    except Exception as e:
        _dprint(dbg_fp, "[debug] Failed printing index examples:", repr(e))
    try:
        if 'original_barcode' in rna_obs.columns:
            _dprint(dbg_fp, "[DEBUG] RNA original_barcode examples:",
                    list(map(str, rna_obs['original_barcode'].head(n).tolist())))
        else:
            _dprint(dbg_fp, "[DEBUG] RNA has no 'original_barcode' column.")
    except Exception as e:
        _dprint(dbg_fp, "[debug] Failed printing RNA original_barcode:", repr(e))
    try:
        if 'original_barcode' in atac_obs.columns:
            _dprint(dbg_fp, "[DEBUG] ATAC original_barcode examples:",
                    list(map(str, atac_obs['original_barcode'].head(n).tolist())))
        else:
            _dprint(dbg_fp, "[DEBUG] ATAC has no 'original_barcode' column.")
    except Exception as e:
        _dprint(dbg_fp, "[debug] Failed printing ATAC original_barcode:", repr(e))


# ──────────────── Barcode normalization (NEW for pairing) ────────────────

_ENC_RE = re.compile(r"ENCSR[0-9A-Z]+_.+?-1$")

def _normalize_to_from_ENC(x: str) -> str:
    """
    Return the substring starting at the first 'ENCSR...' to the end if present;
    otherwise return the original string. This removes tissue prefixes like
    'adrenal_gland_' in ATAC names.
    """
    if not isinstance(x, str):
        x = str(x)
    # Fast path: if already starts with 'ENCSR', keep it
    idx = x.find("ENCSR")
    if idx != -1:
        return x[idx:]
    return x

# ───────────────────────── Core I/O ─────────────────────────

def safe_load_batch_from_hdf5(
    f: h5py.File,
    indices: np.ndarray,
    n_features: int,
    verbose: bool = False
) -> np.ndarray:
    """Safely load a batch of rows from an .h5ad-backed HDF5 storing X as dense or CSR."""
    if 'X' not in f:
        raise ValueError("No 'X' matrix found in HDF5 file")
    X_data = f['X']
    is_sparse = isinstance(X_data, h5py.Group)

    if is_sparse:
        if 'data' not in X_data or 'indices' not in X_data or 'indptr' not in X_data:
            raise ValueError("Sparse matrix missing required components")
        data_arr = X_data['data']
        indices_arr = X_data['indices']
        indptr_arr = X_data['indptr']
        batch_data = np.zeros((len(indices), n_features), dtype=np.float32)
        for i, idx in enumerate(indices):
            start = indptr_arr[idx]
            end = indptr_arr[idx + 1]
            if end > start:
                col_indices = indices_arr[start:end]
                values = data_arr[start:end]
                batch_data[i, col_indices] = values
        return batch_data
    else:
        if not isinstance(X_data, h5py.Dataset):
            raise ValueError("X is neither a sparse matrix group nor a dense dataset")
        if len(X_data.shape) != 2:
            raise ValueError(f"Expected 2D matrix, got shape {X_data.shape}")
        batch_data = np.zeros((len(indices), n_features), dtype=np.float32)
        chunk_size = min(10, len(indices))
        for i in range(0, len(indices), chunk_size):
            chunk_end = min(i + chunk_size, len(indices))
            chunk_indices = indices[i:chunk_end]
            try:
                for j, idx in enumerate(chunk_indices):
                    batch_data[i + j, :] = X_data[idx, :]
            except:
                if verbose:
                    print(f"Warning: Falling back to single-row loading for indices {i} to {chunk_end}")
                for j, idx in enumerate(chunk_indices):
                    try:
                        batch_data[i + j, :] = X_data[idx, :]
                    except Exception as e:
                        print(f"Error loading row {idx}: {e}")
        return batch_data


# ─────────────────────── Main compute (debug build) ───────────────────────

def compute_metrics_direct_hdf5_robust(
    integrated_path: str,
    output_dir: str = "./validation_results",
    batch_size: int = 100,
    verbose: bool = True,
    subsample_ratio: float = 1.0,
    max_memory_gb: float = 8.0,
    debug: bool = True
) -> Dict:

    # Guard: ensure output_dir is a directory, not a file path
    if output_dir.endswith(".h5ad") or (os.path.exists(output_dir) and os.path.isfile(output_dir)):
        base = os.path.dirname(output_dir) if output_dir.endswith(".h5ad") else os.path.dirname(integrated_path)
        output_dir = os.path.join(base, "validation_results_debug")
    os.makedirs(output_dir, exist_ok=True)

    dbg_fp = _open_debug_log(output_dir) if debug else None
    _dprint(dbg_fp, "=" * 80)
    _dprint(dbg_fp, "DIRECT HDF5 PROCESSING - IMPROVED GENE CORRELATION")
    _dprint(dbg_fp, "=" * 80)
    _dprint(dbg_fp, f"[debug] Timestamp       : {datetime.now().isoformat()}")
    _dprint(dbg_fp, f"[debug] integrated_path : {integrated_path}")
    _dprint(dbg_fp, f"[debug] output_dir      : {output_dir}")
    _dprint(dbg_fp, f"[debug] batch_size      : {batch_size}")
    _dprint(dbg_fp, f"[debug] subsample_ratio : {subsample_ratio}")
    _dprint(dbg_fp, f"[debug] max_memory_gb   : {max_memory_gb}")
    _dprint(dbg_fp, f"[debug] debug           : {debug}")

    if verbose:
        print("=" * 80)
        print("DIRECT HDF5 PROCESSING - IMPROVED GENE CORRELATION")
        print("=" * 80)
        print(f"\n📂 Processing: {integrated_path}")
        print(f"   Batch size: {batch_size}")
        print(f"   Subsample ratio: {subsample_ratio}")
        print(f"   Max memory: {max_memory_gb:.1f} GB")

    if debug:
        _dump_h5_structure(integrated_path, output_dir, dbg_fp)

    # Read metadata to find paired cells
    try:
        adata_meta = ad.read_h5ad(integrated_path, backed='r')
    except Exception as e:
        _dprint(dbg_fp, "[debug] ERROR: ad.read_h5ad failed:", repr(e))
        _dump_json(output_dir, "debug_context.json", {
            "stage": "read_h5ad",
            "error": repr(e),
            "traceback": traceback.format_exc(),
        })
        raise

    try:
        obs_df = adata_meta.obs.copy()
        var_names = adata_meta.var_names.copy()
        n_genes = len(var_names)
        matrix_shape = adata_meta.shape
        if debug:
            _dprint(dbg_fp, f"[debug] matrix_shape: {matrix_shape}")
            _dprint(dbg_fp, f"[debug] n_genes     : {n_genes}")
            _dump_obs_preview(output_dir, obs_df, dbg_fp)
            _dump_json(output_dir, "debug_context.json", {
                "matrix_shape": matrix_shape,
                "n_genes": n_genes,
                "obs_columns": list(obs_df.columns),
                "index_name": obs_df.index.name,
                "n_obs": obs_df.shape[0],
            })
    finally:
        del adata_meta
        gc.collect()

    # Modality masks
    try:
        rna_mask = obs_df['modality'] == 'RNA'
        atac_mask = obs_df['modality'] == 'ATAC'
        if debug:
            _dump_obs_value_counts(output_dir, obs_df, 'modality', dbg_fp)
    except KeyError:
        _dprint(dbg_fp, "[debug] ERROR: 'modality' column missing in .obs.")
        _dump_json(output_dir, "debug_context.json", {
            "stage": "modality_mask",
            "error": "'modality' column missing",
            "obs_columns": list(obs_df.columns)
        })
        raise

    rna_indices = np.where(rna_mask)[0]
    atac_indices = np.where(atac_mask)[0]

    if verbose:
        print(f"   Matrix shape: {matrix_shape}")
        print(f"   RNA cells: {len(rna_indices)}")
        print(f"   ATAC cells: {len(atac_indices)}")
    if debug:
        _dprint(dbg_fp, f"[debug] RNA cells : {len(rna_indices)}")
        _dprint(dbg_fp, f"[debug] ATAC cells: {len(atac_indices)}")

    # Split views
    rna_obs = obs_df[rna_mask]
    atac_obs = obs_df[atac_mask]

    # PRINT SAMPLE NAMES (indexes + original_barcode if available)
    _show_name_samples(rna_obs, atac_obs, dbg_fp, n=10)

    # ----------------- Pairing logic with ATAC prefix stripping -----------------
    # Prefer original_barcode if present; otherwise use index-based
    if 'original_barcode' in obs_df.columns:
        rna_barcodes_raw = rna_obs['original_barcode'].astype(str).values
        atac_barcodes_raw = atac_obs['original_barcode'].astype(str).values
        if debug:
            _dprint(dbg_fp, "[debug] Using 'original_barcode' for pairing.")
    else:
        rna_barcodes_raw = np.array([str(idx) for idx in rna_obs.index])
        atac_barcodes_raw = np.array([str(idx) for idx in atac_obs.index])
        if debug:
            _dprint(dbg_fp, "[debug] Using index-based pairing.")

    # Normalize: RNA left as is (already starts at ENCSR); ATAC strip prefix before 'ENCSR'
    rna_barcodes = np.array([s if s.startswith("ENCSR") else s for s in rna_barcodes_raw], dtype=object)
    atac_barcodes = np.array([_normalize_to_from_ENC(s) for s in atac_barcodes_raw], dtype=object)

    # Also normalize index-based fallback suffix removal (_RNA/_ATAC) **after** ENC trimming
    if 'original_barcode' not in obs_df.columns:
        rna_barcodes = np.array([s.replace('_RNA', '') for s in rna_barcodes], dtype=object)
        atac_barcodes = np.array([s.replace('_ATAC', '') for s in atac_barcodes], dtype=object)

    # Emit heads for inspection
    if debug:
        _dump_indices_preview(output_dir, "rna_barcodes", rna_barcodes, dbg_fp)
        _dump_indices_preview(output_dir, "atac_barcodes", atac_barcodes, dbg_fp)

    # Intersect
    common_barcodes = list(set(rna_barcodes) & set(atac_barcodes))

    # If still empty and we were using original_barcode, try index-based normalization as a fallback
    if len(common_barcodes) == 0 and 'original_barcode' in obs_df.columns:
        rna_idx_norm = np.array([_normalize_to_from_ENC(str(idx)).replace('_RNA', '') for idx in rna_obs.index], dtype=object)
        atac_idx_norm = np.array([_normalize_to_from_ENC(str(idx)).replace('_ATAC', '') for idx in atac_obs.index], dtype=object)
        common_barcodes = list(set(rna_idx_norm) & set(atac_idx_norm))
        if debug:
            _dprint(dbg_fp, "[debug] First pairing yielded 0; retried with index-based ENC-trim + suffix-strip.")
            _dump_indices_preview(output_dir, "rna_barcodes_index_norm", rna_idx_norm, dbg_fp)
            _dump_indices_preview(output_dir, "atac_barcodes_index_norm", atac_idx_norm, dbg_fp)

    n_paired = len(common_barcodes)
    if verbose:
        print(f"\n🔗 Found {n_paired} paired cells")
    if debug:
        _dprint(dbg_fp, f"[debug] n_paired={n_paired}")
        _dump_indices_preview(output_dir, "common_barcodes", np.array(common_barcodes, dtype=object), dbg_fp)

    if n_paired == 0:
        _dump_json(output_dir, "debug_context.json", {
            "stage": "pairing",
            "n_rna": int(len(rna_indices)),
            "n_atac": int(len(atac_indices)),
            "n_paired": 0,
            "pairing_note": "After stripping ATAC prefix to ENCSR, still no overlap. "
                            "Verify ENCSR accessions and cell barcode segments truly correspond between modalities.",
            "obs_columns": list(obs_df.columns),
        })
        raise ValueError("No paired cells found!")

    # Optional subsample
    if subsample_ratio < 1.0:
        n_subsample = max(1, int(n_paired * subsample_ratio))
        np.random.seed(42)
        common_barcodes = list(np.random.choice(common_barcodes, n_subsample, replace=False))
        n_paired = len(common_barcodes)
        if verbose:
            print(f"   Subsampled to {n_paired} cells")
        if debug:
            _dprint(dbg_fp, f"[debug] Subsampled to {n_paired} cells")

    # Build mapping from normalized barcodes back to local indices in each modality
    rna_bc_to_idx = {bc: i for i, bc in enumerate(rna_barcodes)}
    atac_bc_to_idx = {bc: i for i, bc in enumerate(atac_barcodes)}

    paired_rna_local = np.array([rna_bc_to_idx[bc] for bc in common_barcodes])
    paired_atac_local = np.array([atac_bc_to_idx[bc] for bc in common_barcodes])

    paired_rna_global = rna_indices[paired_rna_local]
    paired_atac_global = atac_indices[paired_atac_local]

    if debug:
        preview_df = pd.DataFrame({
            "barcode": common_barcodes[:100],
            "rna_local": paired_rna_local[:100],
            "atac_local": paired_atac_local[:100],
            "rna_global": paired_rna_global[:100],
            "atac_global": paired_atac_global[:100],
        })
        preview_df.to_csv(os.path.join(output_dir, "paired_indices_preview.csv"), index=False)
        _dprint(dbg_fp, "[debug] Wrote paired_indices_preview.csv")

    # Detect data format
    if verbose:
        print(f"\n🔍 Detecting data format...")
    with h5py.File(integrated_path, 'r') as f:
        if 'X' not in f:
            if debug:
                _dump_h5_structure(integrated_path, output_dir, dbg_fp)
            raise ValueError("No 'X' matrix found in HDF5 file")
        X_data = f['X']
        is_sparse = isinstance(X_data, h5py.Group)
        if is_sparse:
            if verbose:
                print("   Data format: Sparse CSR matrix")
                data_size = X_data['data'].size
                dtype_size = X_data['data'].dtype.itemsize
                estimated_gb = (data_size * dtype_size * 3) / (1024**3)
                print(f"   Estimated sparse matrix size: {estimated_gb:.2f} GB")
            if debug:
                _dprint(dbg_fp, f"[debug] Sparse CSR detected; nnz≈{X_data['data'].size}")
        else:
            if verbose:
                print("   Data format: Dense matrix")
                print(f"   Matrix dtype: {X_data.dtype}")
                print(f"   Matrix shape: {X_data.shape}")
            if debug:
                _dprint(dbg_fp, f"[debug] Dense dataset dtype={X_data.dtype} shape={X_data.shape}")

    # Process paired cells (unchanged logic)
    if verbose:
        print(f"\n📈 Processing paired cells in batches...")

    per_cell_corr = []
    gene_sums_rna = np.zeros(n_genes, dtype=np.float64)
    gene_sums_atac = np.zeros(n_genes, dtype=np.float64)
    gene_sums_sq_rna = np.zeros(n_genes, dtype=np.float64)
    gene_sums_sq_atac = np.zeros(n_genes, dtype=np.float64)
    gene_counts = np.zeros(n_genes, dtype=np.int64)
    n_processed = 0

    with h5py.File(integrated_path, 'r') as f:
        for batch_start in tqdm(range(0, n_paired, batch_size),
                                desc="Processing cells", disable=not verbose):
            batch_end = min(batch_start + batch_size, n_paired)
            rna_idx_batch = paired_rna_global[batch_start:batch_end]
            atac_idx_batch = paired_atac_global[batch_start:batch_end]
            try:
                rna_batch = safe_load_batch_from_hdf5(f, rna_idx_batch, n_genes, verbose=False)
                atac_batch = safe_load_batch_from_hdf5(f, atac_idx_batch, n_genes, verbose=False)

                for i in range(len(rna_idx_batch)):
                    rna_vec = rna_batch[i, :]
                    atac_vec = atac_batch[i, :]
                    mask = (rna_vec != 0) | (atac_vec != 0)
                    try:
                        corr, _ = stats.pearsonr(rna_vec[mask], atac_vec[mask])
                        if not np.isnan(corr) and not np.isinf(corr):
                            per_cell_corr.append(corr)
                    except:
                        pass

                gene_sums_rna += np.sum(rna_batch, axis=0, dtype=np.float64)
                gene_sums_atac += np.sum(atac_batch, axis=0, dtype=np.float64)
                gene_sums_sq_rna += np.sum(rna_batch.astype(np.float64) ** 2, axis=0)
                gene_sums_sq_atac += np.sum(atac_batch.astype(np.float64) ** 2, axis=0)
                gene_counts += len(rna_idx_batch)
                n_processed += len(rna_idx_batch)

                del rna_batch, atac_batch
                gc.collect()
            except Exception as e:
                if debug:
                    _dprint(dbg_fp, f"[debug] Error in batch {batch_start}-{batch_end}: {repr(e)}")
                print(f"\nError processing batch {batch_start}-{batch_end}: {e}")
                continue

    with np.errstate(divide='ignore', invalid='ignore'):
        gene_means_rna = np.where(gene_counts > 0, gene_sums_rna / gene_counts, 0)
        gene_means_atac = np.where(gene_counts > 0, gene_sums_atac / gene_counts, 0)
        gene_vars_rna = np.where(gene_counts > 0,
                                 (gene_sums_sq_rna / gene_counts) - (gene_means_rna ** 2), 0)
        gene_vars_atac = np.where(gene_counts > 0,
                                  (gene_sums_sq_atac / gene_counts) - (gene_means_atac ** 2), 0)
        gene_stds_rna = np.sqrt(np.maximum(gene_vars_rna, 0))
        gene_stds_atac = np.sqrt(np.maximum(gene_vars_atac, 0))

    per_cell_corr = np.array(per_cell_corr)
    if verbose:
        print(f"\n✅ Processed {n_processed} paired cells")
        print(f"   Computed {len(per_cell_corr)} cell correlations")
    if debug:
        _dprint(dbg_fp, f"[debug] n_processed={n_processed}, n_cell_corr={len(per_cell_corr)}")

    # ================== IMPROVED PER-GENE CORRELATIONS ==================
    if verbose:
        print(f"\n📈 Computing per-gene correlations (improved method)...")
    if debug:
        _dprint(dbg_fp, "[debug] Using improved gene correlation: filtering for co-expressed cells")

    # Sample genes for correlation analysis
    n_sample_genes = min(1000, n_genes)
    np.random.seed(42)
    sample_gene_idx = np.random.choice(n_genes, n_sample_genes, replace=False)
    
    # Initialize arrays to store correlation results
    gene_correlations = np.full(n_genes, np.nan)
    gene_n_coexpressing_cells = np.zeros(n_genes, dtype=int)  # Track how many cells co-express each gene
    
    # Load all paired data once
    with h5py.File(integrated_path, 'r') as f:
        if verbose:
            print("   Loading paired RNA and ATAC data...")
        rna_data = safe_load_batch_from_hdf5(f, paired_rna_global, n_genes)
        atac_data = safe_load_batch_from_hdf5(f, paired_atac_global, n_genes)
        
        # Process each sampled gene
        gene_batch_size = 50
        for gb_start in tqdm(range(0, len(sample_gene_idx), gene_batch_size),
                             desc="Gene correlations", disable=not verbose):
            gb_end = min(gb_start + gene_batch_size, len(sample_gene_idx))
            gene_batch = sample_gene_idx[gb_start:gb_end]
            
            try:
                for gene_idx in gene_batch:
                    # Get expression vectors for this gene across all paired cells
                    rna_vals = rna_data[:, gene_idx]
                    atac_vals = atac_data[:, gene_idx]
                    
                    # Filter for cells where BOTH RNA and ATAC express the gene (non-zero)
                    coexpress_mask = (rna_vals != 0) & (atac_vals != 0)
                    n_coexpress = np.sum(coexpress_mask)
                    gene_n_coexpressing_cells[gene_idx] = n_coexpress
                    
                    # Need at least 3 co-expressing cells for meaningful correlation
                    if n_coexpress >= 3:
                        rna_filtered = rna_vals[coexpress_mask]
                        atac_filtered = atac_vals[coexpress_mask]
                        
                        # Check for variance in filtered data
                        if np.std(rna_filtered) > 0 and np.std(atac_filtered) > 0:
                            try:
                                corr, _ = stats.pearsonr(rna_filtered, atac_filtered)
                                if not np.isnan(corr) and not np.isinf(corr):
                                    gene_correlations[gene_idx] = corr
                                    if debug and gene_idx < 5:  # Debug first few genes
                                        _dprint(dbg_fp, f"[debug] Gene {var_names[gene_idx]}: "
                                               f"n_coexpress={n_coexpress}, corr={corr:.4f}")
                            except Exception as e:
                                if debug:
                                    _dprint(dbg_fp, f"[debug] Correlation failed for gene {gene_idx}: {e}")
                        elif debug and gene_idx < 5:
                            _dprint(dbg_fp, f"[debug] Gene {var_names[gene_idx]}: "
                                   f"n_coexpress={n_coexpress}, but no variance in filtered data")
                    elif debug and gene_idx < 5:
                        _dprint(dbg_fp, f"[debug] Gene {var_names[gene_idx]}: "
                               f"only {n_coexpress} co-expressing cells (need >= 3)")
                        
            except Exception as e:
                if debug:
                    _dprint(dbg_fp, f"[debug] Error computing gene batch {gb_start}-{gb_end}:", repr(e))
                print(f"\nError processing gene batch: {e}")
                continue
        
        # Cleanup
        del rna_data, atac_data
        gc.collect()

    # Summary statistics for gene correlations
    valid_gene_corr = gene_correlations[~np.isnan(gene_correlations)]
    sampled_coexpress_counts = gene_n_coexpressing_cells[sample_gene_idx]
    genes_with_coexpress = np.sum(sampled_coexpress_counts > 0)
    genes_with_sufficient_coexpress = np.sum(sampled_coexpress_counts >= 3)
    
    if verbose:
        print(f"\n   Gene correlation statistics:")
        print(f"   - Genes sampled: {n_sample_genes}")
        print(f"   - Genes with co-expression: {genes_with_coexpress}")
        print(f"   - Genes with ≥3 co-expressing cells: {genes_with_sufficient_coexpress}")
        print(f"   - Valid correlations computed: {len(valid_gene_corr)}")
        if len(valid_gene_corr) > 0:
            print(f"   - Mean gene correlation: {valid_gene_corr.mean():.4f}")
            print(f"   - Median gene correlation: {np.median(valid_gene_corr):.4f}")
    
    if debug:
        _dprint(dbg_fp, f"[debug] Gene correlation summary:")
        _dprint(dbg_fp, f"[debug]   Sampled genes: {n_sample_genes}")
        _dprint(dbg_fp, f"[debug]   Genes with co-expression: {genes_with_coexpress}")
        _dprint(dbg_fp, f"[debug]   Genes with sufficient co-expression: {genes_with_sufficient_coexpress}")
        _dprint(dbg_fp, f"[debug]   Valid correlations: {len(valid_gene_corr)}")

    # ================== END IMPROVED PER-GENE CORRELATIONS ==================

    # Results DataFrame with additional co-expression info
    gene_results = pd.DataFrame({
        'gene': var_names,
        'pearson_corr': gene_correlations,
        'n_coexpressing_cells': gene_n_coexpressing_cells,
        'mean_rna': gene_means_rna,
        'mean_atac': gene_means_atac,
        'std_rna': gene_stds_rna,
        'std_atac': gene_stds_atac
    })

    # Plots (updated to show co-expression filtering effect)
    if verbose:
        print("\n🎨 Creating visualizations...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Cell correlations
    if len(per_cell_corr) > 0:
        axes[0, 0].hist(per_cell_corr, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(per_cell_corr.mean(), color='red', linestyle='--',
                           label=f'Mean: {per_cell_corr.mean():.3f}')
        axes[0, 0].set_xlabel('Pearson Correlation')
        axes[0, 0].set_ylabel('Number of Cells')
        axes[0, 0].set_title('Per-Cell Correlations')
        axes[0, 0].legend()

    # Gene correlations
    if len(valid_gene_corr) > 0:
        axes[0, 1].hist(valid_gene_corr, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].axvline(valid_gene_corr.mean(), color='red', linestyle='--',
                           label=f'Mean: {valid_gene_corr.mean():.3f}')
        axes[0, 1].set_xlabel('Pearson Correlation')
        axes[0, 1].set_ylabel('Number of Genes')
        axes[0, 1].set_title(f'Gene Correlations (n={len(valid_gene_corr)}, co-expressed cells only)')
        axes[0, 1].legend()

    # Co-expressing cells distribution
    coexpress_counts_nonzero = sampled_coexpress_counts[sampled_coexpress_counts > 0]
    if len(coexpress_counts_nonzero) > 0:
        axes[0, 2].hist(np.log10(coexpress_counts_nonzero + 1), bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0, 2].set_xlabel('log10(Number of Co-expressing Cells + 1)')
        axes[0, 2].set_ylabel('Number of Genes')
        axes[0, 2].set_title('Distribution of Co-expressing Cell Counts')
        axes[0, 2].axvline(np.log10(3), color='red', linestyle='--', label='Min for correlation (n=3)')
        axes[0, 2].legend()

    # Mean expression comparison
    mask_nonzero = (gene_means_rna > 0) | (gene_means_atac > 0)
    axes[1, 0].scatter(gene_means_rna[mask_nonzero], gene_means_atac[mask_nonzero], alpha=0.3, s=5)
    if mask_nonzero.any():
        max_val = max(gene_means_rna[mask_nonzero].max(), gene_means_atac[mask_nonzero].max())
        axes[1, 0].plot([0, max_val], [0, max_val], 'r--', label='Identity')
    axes[1, 0].set_xlabel('Mean RNA Expression')
    axes[1, 0].set_ylabel('Mean ATAC Gene Activity')
    axes[1, 0].set_title('Mean Expression Comparison')
    axes[1, 0].legend()
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')

    # Mean-variance relationship
    axes[1, 1].scatter(gene_means_rna[mask_nonzero], gene_stds_rna[mask_nonzero], alpha=0.3, s=5, label='RNA')
    axes[1, 1].scatter(gene_means_atac[mask_nonzero], gene_stds_atac[mask_nonzero], alpha=0.3, s=5, label='ATAC')
    axes[1, 1].set_xlabel('Mean Expression')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].set_title('Mean-Variance Relationship')
    axes[1, 1].legend()
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')

    # Correlation vs co-expression count
    valid_corr_mask = ~np.isnan(gene_correlations[sample_gene_idx])
    if np.any(valid_corr_mask):
        axes[1, 2].scatter(sampled_coexpress_counts[valid_corr_mask], 
                          gene_correlations[sample_gene_idx][valid_corr_mask],
                          alpha=0.5, s=10)
        axes[1, 2].set_xlabel('Number of Co-expressing Cells')
        axes[1, 2].set_ylabel('Gene Correlation')
        axes[1, 2].set_title('Correlation vs Co-expression Count')
        axes[1, 2].set_xscale('log')
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_results_improved.png'), dpi=150)
    plt.close()

    if verbose:
        print("\n💾 Saving results...")

    if len(per_cell_corr) > 0:
        pd.DataFrame({
            'barcode': common_barcodes[:len(per_cell_corr)],
            'correlation': per_cell_corr
        }).to_csv(os.path.join(output_dir, 'cell_correlations_improved.csv'), index=False)

    gene_results.to_csv(os.path.join(output_dir, 'gene_metrics_improved.csv'), index=False)

    summary = {
        'n_paired_cells': n_paired,
        'n_processed': n_processed,
        'n_genes': n_genes,
        'n_genes_sampled': int(n_sample_genes),
        'n_genes_with_coexpression': int(genes_with_coexpress),
        'n_genes_with_sufficient_coexpression': int(genes_with_sufficient_coexpress),
        'mean_cell_corr': float(per_cell_corr.mean()) if len(per_cell_corr) > 0 else np.nan,
        'median_cell_corr': float(np.median(per_cell_corr)) if len(per_cell_corr) > 0 else np.nan,
        'std_cell_corr': float(per_cell_corr.std()) if len(per_cell_corr) > 0 else np.nan,
        'mean_gene_corr': float(valid_gene_corr.mean()) if len(valid_gene_corr) > 0 else np.nan,
        'median_gene_corr': float(np.median(valid_gene_corr)) if len(valid_gene_corr) > 0 else np.nan,
        'std_gene_corr': float(valid_gene_corr.std()) if len(valid_gene_corr) > 0 else np.nan,
        'subsample_ratio': subsample_ratio
    }
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, 'summary_improved.csv'), index=False)

    if debug:
        _dump_json(output_dir, "debug_context.json", {"stage": "done", "summary": summary})
        _dprint(dbg_fp, "[debug] Saved summary_improved.csv and debug_context.json")

    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY - IMPROVED GENE CORRELATION")
        print("=" * 80)
        print(f"Paired cells: {n_paired}")
        print(f"Processed cells: {n_processed}")
        print(f"Cell correlations: {len(per_cell_corr)}")
        if len(per_cell_corr) > 0:
            print(f"Mean cell correlation: {summary['mean_cell_corr']:.4f}")
        print(f"\nGene correlation analysis:")
        print(f"  Genes sampled: {n_sample_genes}")
        print(f"  Genes with co-expression: {genes_with_coexpress}")
        print(f"  Valid correlations: {len(valid_gene_corr)}")
        if len(valid_gene_corr) > 0:
            print(f"  Mean gene correlation: {summary['mean_gene_corr']:.4f}")
            print(f"  Median gene correlation: {summary['median_gene_corr']:.4f}")
        print(f"\nResults saved to: {output_dir}")

    if dbg_fp:
        dbg_fp.close()
    return summary


if __name__ == "__main__":
    try:
        results = compute_metrics_direct_hdf5_robust(
            integrated_path="/dcs07/hongkai/data/harry/result/all/multiomics/preprocess/atac_rna_integrated.h5ad",
            output_dir="/dcs07/hongkai/data/harry/result/all/validation_results_pseudorna",
            batch_size=100,
            verbose=True,
            subsample_ratio=0.1,
            max_memory_gb=400.0,
            debug=True
        )
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()

    try:
        results = compute_metrics_direct_hdf5_robust(
            integrated_path="/dcs07/hongkai/data/harry/result/Benchmark_omics/multiomics/preprocess/atac_rna_integrated.h5ad",
            output_dir="/dcs07/hongkai/data/harry/result/Benchmark_omics/multiomics/validation_results_pseudorna",
            batch_size=100,
            verbose=True,
            subsample_ratio=0.1,
            max_memory_gb=400.0,
            debug=True
        )
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()