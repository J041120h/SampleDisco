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
                    except:
                        pass
        return batch_data


# ──────────────────────────── Main Function ────────────────────────────

def compute_metrics_direct_hdf5_robust(
    integrated_path: str,
    output_dir: str,
    batch_size: int = 100,
    verbose: bool = True,
    subsample_ratio: float = 1.0,
    max_memory_gb: float = 16.0,
    debug: bool = False
) -> Dict:
    """
    Compute RNA-ATAC correlation metrics directly from integrated H5AD file.
    
    Key improvements:
    - Gene correlations computed only from cells where both modalities express the gene
    - Spearman correlation used for robustness to outliers and non-linear relationships
    - Minimum threshold of 3 co-expressing cells required for correlation computation
    - Enhanced statistics and visualizations of co-expression patterns
    """
    dbg_fp = None
    if debug:
        dbg_fp = _open_debug_log(output_dir)
        _dprint(dbg_fp, f"\n{'=' * 80}")
        _dprint(dbg_fp, f"Started at: {datetime.now().isoformat()}")
        _dprint(dbg_fp, f"Integrated file: {integrated_path}")
        _dprint(dbg_fp, f"Output dir: {output_dir}")
        _dprint(dbg_fp, f"Batch size: {batch_size}, subsample: {subsample_ratio}")

    if verbose:
        print(f"\n{'=' * 80}")
        print("COMPUTING RNA-ATAC METRICS (IMPROVED GENE CORRELATION)")
        print(f"{'=' * 80}")
        print(f"Integrated file: {integrated_path}")

    if debug:
        _dump_h5_structure(integrated_path, output_dir, dbg_fp)

    # Load metadata using anndata (more robust for different HDF5 formats)
    if verbose:
        print("\n📖 Loading metadata...")
    
    # Load only metadata without loading the full matrix
    adata = ad.read_h5ad(integrated_path, backed='r')
    
    modality_key = 'modality'
    if modality_key not in adata.obs.columns:
        raise KeyError(f"Missing '{modality_key}' column in obs")
    
    modality = adata.obs[modality_key].values
    if modality.dtype == object:
        modality = modality.astype(str)
    
    # Get barcodes - try different column names
    barcode_key = None
    for key in ['original_barcode', 'barcode', 'cell_id']:
        if key in adata.obs.columns:
            barcode_key = key
            break
    
    if barcode_key:
        barcodes = adata.obs[barcode_key].values
    else:
        # Use the index as barcodes
        barcodes = adata.obs.index.values
    
    if barcodes.dtype == object:
        barcodes = barcodes.astype(str)
    
    var_names = adata.var.index.values
    if var_names.dtype == object:
        var_names = var_names.astype(str)
    
    # Close the backed connection
    adata.file.close()
    del adata

    n_genes = len(var_names)
    rna_mask = (modality == 'RNA')
    atac_mask = (modality == 'ATAC')

    rna_indices = np.where(rna_mask)[0]
    atac_indices = np.where(atac_mask)[0]
    rna_barcodes = barcodes[rna_indices]
    atac_barcodes = barcodes[atac_indices]

    if verbose:
        print(f"\n📊 Data overview:")
        print(f"   RNA cells: {len(rna_indices):,}")
        print(f"   ATAC cells: {len(atac_indices):,}")
        print(f"   Genes: {n_genes:,}")

    if debug:
        _dprint(dbg_fp, f"[debug] RNA cells: {len(rna_indices)}, ATAC cells: {len(atac_indices)}")
        _dprint(dbg_fp, f"[debug] Genes: {n_genes}")

    # Normalize barcodes
    rna_norm = np.array([_normalize_to_from_ENC(bc) for bc in rna_barcodes])
    atac_norm = np.array([_normalize_to_from_ENC(bc) for bc in atac_barcodes])

    # Find common barcodes
    rna_set = set(rna_norm)
    atac_set = set(atac_norm)
    common_bc_set = rna_set & atac_set
    common_barcodes = sorted(common_bc_set)
    n_paired = len(common_barcodes)

    if verbose:
        print(f"\n🔗 Pairing cells:")
        print(f"   Common barcodes: {n_paired:,}")

    if n_paired == 0:
        raise ValueError("No common barcodes found between RNA and ATAC")

    # Create mapping
    rna_bc_to_idx = {bc: i for i, bc in enumerate(rna_norm)}
    atac_bc_to_idx = {bc: i for i, bc in enumerate(atac_norm)}

    paired_rna_local = np.array([rna_bc_to_idx[bc] for bc in common_barcodes], dtype=np.int64)
    paired_atac_local = np.array([atac_bc_to_idx[bc] for bc in common_barcodes], dtype=np.int64)
    paired_rna_global = rna_indices[paired_rna_local]
    paired_atac_global = atac_indices[paired_atac_local]

    # Subsample
    if subsample_ratio < 1.0:
        np.random.seed(42)
        n_use = max(1, int(n_paired * subsample_ratio))
        sub_idx = np.random.choice(n_paired, size=n_use, replace=False)
        paired_rna_global = paired_rna_global[sub_idx]
        paired_atac_global = paired_atac_global[sub_idx]
        common_barcodes = [common_barcodes[i] for i in sub_idx]
        n_paired = n_use
        if verbose:
            print(f"   Subsampled to: {n_paired:,} cells")

    if debug:
        _dprint(dbg_fp, f"[debug] After subsample: {n_paired} cells")

    # Memory check
    bytes_per_matrix = n_paired * n_genes * 4
    total_gb = (2 * bytes_per_matrix) / (1024**3)
    if total_gb > max_memory_gb:
        raise MemoryError(f"Would need ~{total_gb:.1f} GB but limit is {max_memory_gb} GB")

    if verbose:
        print(f"   Estimated memory: {total_gb:.2f} GB")

    # Load data in batches
    if verbose:
        print("\n📥 Loading data in batches...")

    rna_mat = np.zeros((n_paired, n_genes), dtype=np.float32)
    atac_mat = np.zeros((n_paired, n_genes), dtype=np.float32)

    with h5py.File(integrated_path, "r") as f:
        n_batches = int(np.ceil(n_paired / batch_size))
        for batch_idx in tqdm(range(n_batches), desc="Loading", disable=not verbose):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_paired)
            rna_batch_global = paired_rna_global[start:end]
            atac_batch_global = paired_atac_global[start:end]
            rna_mat[start:end, :] = safe_load_batch_from_hdf5(f, rna_batch_global, n_genes, verbose=False)
            atac_mat[start:end, :] = safe_load_batch_from_hdf5(f, atac_batch_global, n_genes, verbose=False)

    if debug:
        _dprint(dbg_fp, f"[debug] Loaded matrices: RNA {rna_mat.shape}, ATAC {atac_mat.shape}")

    # Normalize
    if verbose:
        print("\n🔬 Normalizing data...")

    rna_totals = rna_mat.sum(axis=1, keepdims=True)
    rna_totals[rna_totals == 0] = 1
    rna_normed = (rna_mat / rna_totals) * 1e4
    rna_normed = np.log1p(rna_normed)

    atac_totals = atac_mat.sum(axis=1, keepdims=True)
    atac_totals[atac_totals == 0] = 1
    atac_normed = (atac_mat / atac_totals) * 1e4
    atac_normed = np.log1p(atac_normed)

    del rna_mat, atac_mat
    gc.collect()

    n_processed = rna_normed.shape[0]

    # Per-cell correlations
    if verbose:
        print("\n📊 Computing per-cell correlations...")

    per_cell_corr = []
    for i in tqdm(range(n_processed), desc="Cell correlations", disable=not verbose):
        rna_vec = rna_normed[i, :]
        atac_vec = atac_normed[i, :]
        if rna_vec.std() > 0 and atac_vec.std() > 0:
            corr, _ = stats.pearsonr(rna_vec, atac_vec)
            if not np.isnan(corr):
                per_cell_corr.append(corr)
    per_cell_corr = np.array(per_cell_corr)

    if verbose and len(per_cell_corr) > 0:
        print(f"\n   Per-cell correlation: {per_cell_corr.mean():.4f} ± {per_cell_corr.std():.4f}")
        print(f"   Valid cells: {len(per_cell_corr)}/{n_processed}")

    # ================== IMPROVED PER-GENE CORRELATIONS ==================
    """
    Enhanced gene-level correlation analysis with co-expression filtering.
    
    Methodology:
    1. Cell pairing: RNA and ATAC cells are matched by shared barcode identifiers
    2. Normalization: Expression values are library-size normalized (scaled to 10,000) 
       and log-transformed (log1p) for both modalities
    3. Co-expression filtering: For each gene, only cells with non-zero expression 
       in BOTH RNA and ATAC modalities are retained for correlation analysis
    4. Statistical requirement: Minimum of 3 co-expressing cells required to compute 
       correlation (prevents spurious correlations from insufficient data)
    5. Correlation metric: Spearman rank correlation is used for robustness to outliers
       and to capture monotonic (not strictly linear) relationships between modalities
    
    This approach ensures that correlations reflect genuine biological relationships
    where the gene is actively expressed in both assays, rather than being dominated
    by zero-inflation or technical noise.
    """
    
    if verbose:
        print("\n🧬 Computing gene-level correlations (co-expression filtered)...")
        print("   Using Spearman correlation on cells with non-zero expression in both modalities...")

    # Sample genes for analysis (to manage computational cost)
    n_sample_genes = min(5000, n_genes)
    np.random.seed(42)
    sample_gene_idx = np.random.choice(n_genes, size=n_sample_genes, replace=False)

    # Initialize result arrays
    gene_correlations = np.full(n_genes, np.nan, dtype=np.float32)
    gene_n_coexpressing_cells = np.zeros(n_genes, dtype=np.int32)
    gene_means_rna = np.zeros(n_genes, dtype=np.float32)
    gene_means_atac = np.zeros(n_genes, dtype=np.float32)
    gene_stds_rna = np.zeros(n_genes, dtype=np.float32)
    gene_stds_atac = np.zeros(n_genes, dtype=np.float32)

    # Compute correlations for sampled genes
    sampled_coexpress_counts = np.zeros(n_sample_genes, dtype=np.int32)
    
    for i, gene_idx in enumerate(tqdm(sample_gene_idx, desc="Gene correlations", disable=not verbose)):
        # Extract expression vectors for this gene across all paired cells
        rna_expr = rna_normed[:, gene_idx]
        atac_expr = atac_normed[:, gene_idx]
        
        # Identify cells with non-zero expression in BOTH modalities (co-expressing cells)
        coexpress_mask = (rna_expr > 0) & (atac_expr > 0)
        n_coexpress = coexpress_mask.sum()
        
        sampled_coexpress_counts[i] = n_coexpress
        gene_n_coexpressing_cells[gene_idx] = n_coexpress
        
        # Compute summary statistics across all paired cells
        gene_means_rna[gene_idx] = rna_expr.mean()
        gene_means_atac[gene_idx] = atac_expr.mean()
        gene_stds_rna[gene_idx] = rna_expr.std()
        gene_stds_atac[gene_idx] = atac_expr.std()
        
        # Require at least 3 co-expressing cells for meaningful correlation
        if n_coexpress >= 3:
            rna_coexpress = rna_expr[coexpress_mask]
            atac_coexpress = atac_expr[coexpress_mask]
            
            # Use Spearman correlation for robustness to outliers and non-linear relationships
            try:
                corr, _ = stats.spearmanr(rna_coexpress, atac_coexpress)
                if not np.isnan(corr):
                    gene_correlations[gene_idx] = corr
            except:
                pass  # Keep as NaN if correlation fails

    # Collect valid correlations for summary statistics
    valid_gene_corr = gene_correlations[sample_gene_idx]
    valid_gene_corr = valid_gene_corr[~np.isnan(valid_gene_corr)]
    
    # Count genes meeting criteria
    genes_with_coexpress = (sampled_coexpress_counts > 0).sum()
    genes_with_sufficient_coexpress = (sampled_coexpress_counts >= 3).sum()
    
    if verbose:
        print(f"\n   Gene correlation statistics:")
        print(f"   - Genes sampled: {n_sample_genes}")
        print(f"   - Genes with any co-expression: {genes_with_coexpress}")
        print(f"   - Genes with ≥3 co-expressing cells: {genes_with_sufficient_coexpress}")
        print(f"   - Valid correlations computed: {len(valid_gene_corr)}")
        if len(valid_gene_corr) > 0:
            print(f"   - Mean gene correlation: {valid_gene_corr.mean():.4f}")
            print(f"   - Median gene correlation: {np.median(valid_gene_corr):.4f}")
            print(f"   - Std gene correlation: {valid_gene_corr.std():.4f}")
    
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
        'spearman_corr': gene_correlations,
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
        axes[0, 1].set_xlabel('Spearman Correlation')
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