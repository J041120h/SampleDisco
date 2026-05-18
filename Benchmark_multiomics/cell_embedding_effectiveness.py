#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ====== EDIT THESE PARAMETERS ======
H5AD_PATH = "/dcs07/hongkai/data/harry/result/Benchmark_omics/multiomics/preprocess/atac_rna_integrated.h5ad"
OUTDIR    = "/dcs07/hongkai/data/harry/result/Benchmark_omics/multiomics"

# GPU and batch settings
USE_GPU             = True
BATCH_SIZE          = 10000
DEVICE              = "cuda:0"

# Optional speed knobs
MAX_CELLS_CORR      = None
MAX_GENES_CORR      = None
MIN_COEXPR_CELLS    = 3
RNG_SEED            = 0
# ==================================

import os, time, gc, json, re
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy import stats as sp_stats
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def ensure_csr64(X):
    if sp.issparse(X):
        X = X.tocsr().astype(np.float64)
        X.sort_indices()
        X.eliminate_zeros()
        return X
    X = np.asarray(X, dtype=np.float64)
    return sp.csr_matrix(X)


def to_dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def extract_sample_id(bc):
    """Extract ENCSR/ENCS sample ID from barcode string"""
    match = re.search(r'(ENC[SR]\w+)', str(bc))
    return match.group(1) if match else None


def extract_cell_barcode(bc):
    """Extract 16bp cell barcode (e.g., AAACAGCCAGCAAATA)"""
    match = re.search(r'([ACGT]{16})', str(bc))
    return match.group(1) if match else None


def strip_tissue_prefix(bc):
    """
    Remove tissue prefix from ATAC barcodes.
    e.g., 'adrenal_gland_ENCSR212VKB_AAACAGCCAAAGGTAC-1' -> 'ENCSR212VKB_AAACAGCCAAAGGTAC-1'
    """
    bc = str(bc)
    # Remove everything before ENC pattern
    stripped = re.sub(r'^.*?(?=ENC[SR])', '', bc)
    return stripped if stripped else bc


def pair_by_barcode(adata):
    """
    Pair RNA and ATAC cells using multiple strategies.
    """
    print("\n" + "="*60)
    print("BARCODE PAIRING DIAGNOSTICS")
    print("="*60)
    
    print(f"\n[INFO] obs columns: {list(adata.obs.columns)}")
    
    if 'modality' not in adata.obs:
        raise ValueError("Missing obs['modality'] column")
    
    # Find barcode column
    barcode_col = 'original_barcode' if 'original_barcode' in adata.obs.columns else None
    if barcode_col is None:
        for alt in ['barcode', 'cell_barcode', 'cell_id']:
            if alt in adata.obs.columns:
                barcode_col = alt
                break
    if barcode_col is None:
        barcode_col = 'obs_index'
        adata.obs[barcode_col] = adata.obs_names.astype(str)
    
    print(f"[INFO] Using barcode column: '{barcode_col}'")
    
    # Split by modality
    modality_counts = adata.obs['modality'].value_counts()
    print(f"\n[INFO] Modality distribution:\n{modality_counts.to_string()}")
    
    rna_mask = adata.obs['modality'].astype(str).str.upper().str.contains('RNA|GEX')
    atac_mask = adata.obs['modality'].astype(str).str.upper().str.contains('ATAC')
    
    rna = adata[rna_mask].copy()
    atac = adata[atac_mask].copy()
    
    print(f"\n[INFO] RNA cells: {rna.n_obs:,}, ATAC cells: {atac.n_obs:,}")
    
    # Show sample column info if available
    if 'sample' in adata.obs.columns:
        print(f"\n[INFO] Sample column present")
        print(f"  RNA samples (unique): {rna.obs['sample'].nunique()}")
        print(f"  ATAC samples (unique): {atac.obs['sample'].nunique()}")
        print(f"  RNA sample examples: {list(rna.obs['sample'].unique()[:5])}")
        print(f"  ATAC sample examples: {list(atac.obs['sample'].unique()[:5])}")
        
        # Check if samples overlap
        rna_samples = set(rna.obs['sample'].astype(str))
        atac_samples = set(atac.obs['sample'].astype(str))
        sample_overlap = rna_samples & atac_samples
        print(f"  Sample overlap: {len(sample_overlap)}")
    
    # Show barcode examples
    print(f"\n[INFO] Barcode examples:")
    print(f"  RNA:  {list(rna.obs[barcode_col].head(3))}")
    print(f"  ATAC: {list(atac.obs[barcode_col].head(3))}")
    
    # =====================================================
    # STRATEGY 1: Direct barcode match (after stripping tissue prefix from ATAC)
    # =====================================================
    print(f"\n--- Strategy 1: Direct match (strip ATAC tissue prefix) ---")
    
    rna.obs['bc_normalized'] = rna.obs[barcode_col].astype(str).apply(
        lambda x: re.sub(r'[-_](RNA|ATAC|GEX)$', '', x, flags=re.IGNORECASE)
    )
    atac.obs['bc_normalized'] = atac.obs[barcode_col].astype(str).apply(strip_tissue_prefix).apply(
        lambda x: re.sub(r'[-_](RNA|ATAC|GEX)$', '', x, flags=re.IGNORECASE)
    )
    
    print(f"  RNA normalized:  {list(rna.obs['bc_normalized'].head(3))}")
    print(f"  ATAC normalized: {list(atac.obs['bc_normalized'].head(3))}")
    
    overlap1 = len(set(rna.obs['bc_normalized']) & set(atac.obs['bc_normalized']))
    print(f"  Overlap: {overlap1:,}")
    
    if overlap1 > 0:
        use_col = 'bc_normalized'
    else:
        # =====================================================
        # STRATEGY 2: Match by sample + 16bp cell barcode
        # =====================================================
        print(f"\n--- Strategy 2: Sample + 16bp cell barcode ---")
        
        rna.obs['cell_bc_16'] = rna.obs[barcode_col].astype(str).apply(extract_cell_barcode)
        atac.obs['cell_bc_16'] = atac.obs[barcode_col].astype(str).apply(extract_cell_barcode)
        
        if 'sample' in rna.obs.columns and 'sample' in atac.obs.columns:
            rna.obs['sample_bc'] = rna.obs['sample'].astype(str) + '_' + rna.obs['cell_bc_16'].astype(str)
            atac.obs['sample_bc'] = atac.obs['sample'].astype(str) + '_' + atac.obs['cell_bc_16'].astype(str)
            
            print(f"  RNA sample_bc:  {list(rna.obs['sample_bc'].head(3))}")
            print(f"  ATAC sample_bc: {list(atac.obs['sample_bc'].head(3))}")
            
            overlap2 = len(set(rna.obs['sample_bc']) & set(atac.obs['sample_bc']))
            print(f"  Overlap: {overlap2:,}")
            
            if overlap2 > 0:
                use_col = 'sample_bc'
            else:
                # =====================================================
                # STRATEGY 3: Match by 16bp cell barcode only
                # =====================================================
                print(f"\n--- Strategy 3: 16bp cell barcode only ---")
                print(f"  RNA cell_bc:  {list(rna.obs['cell_bc_16'].head(3))}")
                print(f"  ATAC cell_bc: {list(atac.obs['cell_bc_16'].head(3))}")
                
                overlap3 = len(set(rna.obs['cell_bc_16'].dropna()) & set(atac.obs['cell_bc_16'].dropna()))
                print(f"  Overlap: {overlap3:,}")
                
                if overlap3 > 0:
                    print("  WARNING: Using cell barcode only - may have collisions across samples!")
                    use_col = 'cell_bc_16'
                else:
                    use_col = None
        else:
            # No sample column
            overlap3 = len(set(rna.obs['cell_bc_16'].dropna()) & set(atac.obs['cell_bc_16'].dropna()))
            print(f"  (No sample column) Cell barcode overlap: {overlap3:,}")
            use_col = 'cell_bc_16' if overlap3 > 0 else None
    
    # =====================================================
    # STRATEGY 4: Positional pairing (if equal counts)
    # =====================================================
    if use_col is None and rna.n_obs == atac.n_obs:
        print(f"\n--- Strategy 4: Positional pairing (equal counts: {rna.n_obs:,}) ---")
        print("  Assuming cells are already aligned by position!")
        
        # Create synthetic pairing key
        rna.obs['pos_idx'] = range(rna.n_obs)
        atac.obs['pos_idx'] = range(atac.n_obs)
        use_col = 'pos_idx'
        
        # Create pairs DataFrame for compatibility
        pairs = pd.DataFrame({
            'rna_idx': rna.obs_names,
            'atac_idx': atac.obs_names,
            'original_barcode': [f"pair_{i}" for i in range(rna.n_obs)]
        })
        
        # Gene alignment
        if not np.array_equal(rna.var_names.values, atac.var_names.values):
            print(f"\n[INFO] Aligning genes...")
            common_genes = sorted(set(rna.var_names) & set(atac.var_names))
            print(f"  Common genes: {len(common_genes):,}")
            if len(common_genes) == 0:
                raise ValueError("No common genes between RNA and ATAC")
            rna = rna[:, common_genes].copy()
            atac = atac[:, common_genes].copy()
        
        print(f"\n{'='*60}")
        print(f"PAIRING COMPLETE: {rna.n_obs:,} cells, {rna.n_vars:,} genes")
        print(f"{'='*60}\n")
        
        return rna, atac, pairs
    
    if use_col is None:
        print("\n" + "!"*60)
        print("ERROR: NO PAIRING STRATEGY WORKED")
        print("!"*60)
        print("\nThe RNA and ATAC data don't appear to be paired.")
        print("Possible reasons:")
        print("  1. Different experiments/samples concatenated together")
        print("  2. Barcodes use incompatible naming schemes")
        print("  3. Data requires custom preprocessing")
        raise ValueError("Could not find paired cells between RNA and ATAC")
    
    # Perform the merge
    print(f"\n[INFO] Merging on column: '{use_col}'")
    r = rna.obs[[use_col]].reset_index().rename(columns={'index': 'rna_idx'})
    a = atac.obs[[use_col]].reset_index().rename(columns={'index': 'atac_idx'})
    pairs = r.merge(a, on=use_col, how='inner')
    pairs['original_barcode'] = pairs[use_col]
    
    print(f"[INFO] Paired cells: {len(pairs):,}")
    
    if len(pairs) == 0:
        raise ValueError("Merge produced 0 pairs")
    
    rna = rna[pairs['rna_idx'].values].copy()
    atac = atac[pairs['atac_idx'].values].copy()
    
    # Gene alignment
    if not np.array_equal(rna.var_names.values, atac.var_names.values):
        print(f"\n[INFO] Aligning genes...")
        common_genes = sorted(set(rna.var_names) & set(atac.var_names))
        print(f"  Common genes: {len(common_genes):,}")
        if len(common_genes) == 0:
            raise ValueError("No common genes")
        rna = rna[:, common_genes].copy()
        atac = atac[:, common_genes].copy()
    
    print(f"\n{'='*60}")
    print(f"PAIRING COMPLETE: {rna.n_obs:,} cells, {rna.n_vars:,} genes")
    print(f"{'='*60}\n")
    
    return rna, atac, pairs


# ============================================================================
# PER-CELL PEARSON CORRELATION
# ============================================================================
def per_cell_pearson_gpu(X_rna, X_atac, batch_size, device, max_cells=None, seed=0):
    n_cells = X_rna.shape[0]
    if n_cells == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=int)
    
    idx = np.arange(n_cells)
    if max_cells is not None and max_cells < n_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_cells, size=max_cells, replace=False)
    
    m = len(idx)
    correlations = np.full(m, np.nan, dtype=np.float32)
    
    n_batches = (m + batch_size - 1) // batch_size
    print(f"[GPU] Processing {m:,} cells in {n_batches} batches")
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, m)
        batch_indices = idx[start:end]
        
        rna_batch = torch.from_numpy(X_rna[batch_indices].toarray()).float().to(device)
        atac_batch = torch.from_numpy(X_atac[batch_indices].toarray()).float().to(device)
        
        mask = (rna_batch != 0) | (atac_batch != 0)
        
        for i in range(end - start):
            cell_mask = mask[i]
            n_valid = cell_mask.sum().item()
            
            if n_valid >= 3:
                r = rna_batch[i, cell_mask]
                a = atac_batch[i, cell_mask]
                
                r_std = torch.std(r).item()
                a_std = torch.std(a).item()
                
                if r_std > 0 and a_std > 0:
                    r_c = r - r.mean()
                    a_c = a - a.mean()
                    cov = (r_c * a_c).mean()
                    correlations[start + i] = (cov / (r_std * a_std)).item()
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            print(f"  Batch {batch_idx + 1}/{n_batches}")
    
    return correlations, idx


def per_cell_pearson_cpu(X_rna, X_atac, max_cells=None, seed=0):
    n_cells = X_rna.shape[0]
    if n_cells == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=int)
    
    idx = np.arange(n_cells)
    if max_cells is not None and max_cells < n_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_cells, size=max_cells, replace=False)
    
    m = len(idx)
    correlations = np.full(m, np.nan, dtype=np.float64)
    
    rna_dense = to_dense(X_rna)
    atac_dense = to_dense(X_atac)
    
    print(f"[CPU] Processing {m:,} cells")
    
    for i, cell_idx in enumerate(tqdm(idx, desc="Per-cell correlation")):
        r = rna_dense[cell_idx, :]
        a = atac_dense[cell_idx, :]
        
        mask = (r != 0) | (a != 0)
        
        if mask.sum() >= 3:
            r_m, a_m = r[mask], a[mask]
            if np.std(r_m) > 0 and np.std(a_m) > 0:
                correlations[i] = np.corrcoef(r_m, a_m)[0, 1]
    
    return correlations, idx


def per_cell_random_baseline(X_rna, X_atac, idx, batch_size, device, seed=0, use_gpu=True):
    n_cells = X_rna.shape[0]
    m = len(idx)
    if m == 0:
        return np.array([], dtype=np.float32)
    
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n_cells)[:m]
    
    correlations = np.full(m, np.nan, dtype=np.float32 if use_gpu else np.float64)
    
    if use_gpu and device.type == "cuda":
        n_batches = (m + batch_size - 1) // batch_size
        print(f"[GPU] Random baseline in {n_batches} batches")
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, m)
            
            rna_batch = torch.from_numpy(X_rna[idx[start:end]].toarray()).float().to(device)
            atac_batch = torch.from_numpy(X_atac[perm[start:end]].toarray()).float().to(device)
            
            mask = (rna_batch != 0) | (atac_batch != 0)
            
            for i in range(end - start):
                cell_mask = mask[i]
                if cell_mask.sum().item() >= 3:
                    r = rna_batch[i, cell_mask]
                    a = atac_batch[i, cell_mask]
                    r_std, a_std = torch.std(r).item(), torch.std(a).item()
                    if r_std > 0 and a_std > 0:
                        correlations[start + i] = ((r - r.mean()) * (a - a.mean())).mean().item() / (r_std * a_std)
    else:
        rna_dense = to_dense(X_rna)
        atac_dense = to_dense(X_atac)
        
        for i, (ri, ai) in enumerate(zip(idx, perm)):
            r, a = rna_dense[ri, :], atac_dense[ai, :]
            mask = (r != 0) | (a != 0)
            if mask.sum() >= 3:
                r_m, a_m = r[mask], a[mask]
                if np.std(r_m) > 0 and np.std(a_m) > 0:
                    correlations[i] = np.corrcoef(r_m, a_m)[0, 1]
    
    return correlations


# ============================================================================
# PER-GENE SPEARMAN CORRELATION
# ============================================================================
def per_gene_spearman(X_rna, X_atac, gene_names, min_coexpr_cells=3, max_genes=None, seed=0):
    n_cells, n_genes = X_rna.shape
    
    if n_cells == 0:
        return pd.DataFrame(columns=['gene_idx', 'gene', 'n_coexpressing_cells',
                                     'spearman_corr', 'p_value', 'q_fdr_bh']), np.array([])
    
    gene_idx = np.arange(n_genes)
    if max_genes is not None and max_genes < n_genes:
        rng = np.random.default_rng(seed)
        gene_idx = np.sort(rng.choice(n_genes, size=max_genes, replace=False))
    
    rna_csc = X_rna.tocsc() if sp.issparse(X_rna) else sp.csc_matrix(X_rna)
    atac_csc = X_atac.tocsc() if sp.issparse(X_atac) else sp.csc_matrix(X_atac)
    
    results = []
    
    for g in tqdm(gene_idx, desc="Per-gene Spearman"):
        r = rna_csc.getcol(g).toarray().ravel()
        a = atac_csc.getcol(g).toarray().ravel()
        
        co_mask = (r != 0) & (a != 0)
        n_coexpr = int(co_mask.sum())
        
        corr, pval = np.nan, np.nan
        
        if n_coexpr >= min_coexpr_cells:
            r_m, a_m = r[co_mask], a[co_mask]
            if np.std(r_m) > 0 and np.std(a_m) > 0:
                try:
                    corr, pval = sp_stats.spearmanr(r_m, a_m)
                except:
                    pass
        
        results.append({
            'gene_idx': g,
            'gene': gene_names[g] if g < len(gene_names) else f"gene_{g}",
            'n_coexpressing_cells': n_coexpr,
            'spearman_corr': corr,
            'p_value': pval
        })
    
    res_df = pd.DataFrame(results)
    
    valid = res_df['p_value'].notna()
    if valid.sum() > 0:
        from statsmodels.stats.multitest import multipletests
        _, qvals, _, _ = multipletests(res_df.loc[valid, 'p_value'].values, method='fdr_bh')
        res_df.loc[valid, 'q_fdr_bh'] = qvals
    else:
        res_df['q_fdr_bh'] = np.nan
    
    return res_df, gene_idx


# ============================================================================
# PLOTTING
# ============================================================================
def save_hist(data, title, xlabel, path_png, bins=50):
    plt.figure(figsize=(8, 5))
    valid = data[np.isfinite(data)] if len(data) > 0 else np.array([])
    if len(valid) > 0:
        plt.hist(valid, bins=bins, edgecolor='black', alpha=0.8)
        plt.axvline(np.nanmean(data), ls='--', color='r', label=f"Mean={np.nanmean(data):.3f}")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.close()


def save_comparison_plots(corr_true, corr_rand, per_gene_df, min_coexpr_cells, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Per-cell true
    valid_cc = np.isfinite(corr_true) if len(corr_true) > 0 else np.array([])
    if len(valid_cc) > 0 and valid_cc.any():
        axes[0, 0].hist(corr_true[valid_cc], bins=50, edgecolor='black', alpha=0.8)
        axes[0, 0].axvline(np.nanmean(corr_true), ls='--', color='r', label=f"Mean={np.nanmean(corr_true):.3f}")
        axes[0, 0].legend()
    axes[0, 0].set_title("Per-cell Pearson (true pairs)")
    axes[0, 0].set_xlabel("Correlation")
    axes[0, 0].set_ylabel("Cells")
    
    # Per-gene
    gc = per_gene_df['spearman_corr'].values if len(per_gene_df) > 0 else np.array([])
    valid_gc = np.isfinite(gc) if len(gc) > 0 else np.array([])
    if len(valid_gc) > 0 and valid_gc.any():
        axes[0, 1].hist(gc[valid_gc], bins=50, edgecolor='black', alpha=0.8)
        axes[0, 1].axvline(np.nanmean(gc), ls='--', color='r', label=f"Mean={np.nanmean(gc):.3f}")
        axes[0, 1].legend()
    axes[0, 1].set_title(f"Per-gene Spearman (≥{min_coexpr_cells} co-expr cells)")
    axes[0, 1].set_xlabel("Correlation")
    axes[0, 1].set_ylabel("Genes")
    
    # Co-expressing dist
    nco = per_gene_df['n_coexpressing_cells'].values if len(per_gene_df) > 0 else np.array([])
    nz = nco[nco > 0] if len(nco) > 0 else np.array([])
    if len(nz) > 0:
        axes[1, 0].hist(nz, bins=50, edgecolor='black', alpha=0.8)
        axes[1, 0].set_xscale('log')
    axes[1, 0].set_title("Co-expressing cells per gene")
    axes[1, 0].set_xlabel("N cells (log)")
    axes[1, 0].set_ylabel("Genes")
    
    # Scatter
    if len(valid_gc) > 0 and valid_gc.any():
        axes[1, 1].scatter(nco[valid_gc], gc[valid_gc], s=8, alpha=0.5)
        axes[1, 1].set_xscale('log')
        axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_title("Corr vs co-expressing cells")
    axes[1, 1].set_xlabel("N co-expressing (log)")
    axes[1, 1].set_ylabel("Spearman corr")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_plots.png"), dpi=200)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    
    if USE_GPU and torch.cuda.is_available():
        device = torch.device(DEVICE)
        print(f"[INFO] GPU: {torch.cuda.get_device_name(device)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    
    print(f"[INFO] Loading: {H5AD_PATH}")
    adata = ad.read_h5ad(H5AD_PATH)
    print(f"[INFO] Total: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    
    rna, atac, pairs = pair_by_barcode(adata)
    
    if rna.n_obs == 0:
        print("[ERROR] No paired cells. Exiting.")
        return
    
    X_rna = ensure_csr64(rna.X)
    X_atac = ensure_csr64(atac.X)
    gene_names = rna.var_names.values
    
    # Per-cell correlation
    print("\n[INFO] Computing per-cell Pearson correlation...")
    t0 = time.time()
    
    if device.type == "cuda":
        corr_true, idx = per_cell_pearson_gpu(X_rna, X_atac, BATCH_SIZE, device, MAX_CELLS_CORR, RNG_SEED)
    else:
        corr_true, idx = per_cell_pearson_cpu(X_rna, X_atac, MAX_CELLS_CORR, RNG_SEED)
    
    corr_rand = per_cell_random_baseline(X_rna, X_atac, idx, BATCH_SIZE, device, RNG_SEED, device.type == "cuda")
    
    print(f"[INFO] Done in {time.time()-t0:.2f}s (n={len(corr_true):,})")
    
    cell_ids = pairs['original_barcode'].values[idx] if len(idx) > 0 else []
    per_cell_df = pd.DataFrame({
        'cell': cell_ids,
        'pearson_corr_true': corr_true,
        'pearson_corr_random': corr_rand
    })
    per_cell_csv = os.path.join(OUTDIR, "per_cell_correlations.csv")
    per_cell_df.to_csv(per_cell_csv, index=False)
    print(f"[SAVE] {per_cell_csv}")
    
    # Stats
    from scipy.stats import mannwhitneyu
    mt = np.isfinite(corr_true)
    mr = np.isfinite(corr_rand)
    if mt.sum() > 2 and mr.sum() > 2:
        u, p = mannwhitneyu(corr_true[mt], corr_rand[mr], alternative='greater')
        with open(os.path.join(OUTDIR, "per_cell_MWU.txt"), "w") as f:
            f.write(f"Mann-Whitney U (true > random): U={u}, p={p:.3e}\n")
            f.write(f"Mean(true)={np.nanmean(corr_true):.4f}, Mean(rand)={np.nanmean(corr_rand):.4f}\n")
        print(f"[STAT] MWU p={p:.3e}")
    
    save_hist(corr_true, "Per-cell Pearson (true)", "Correlation",
              os.path.join(OUTDIR, "per_cell_true_hist.png"))
    save_hist(corr_rand, "Per-cell Pearson (random)", "Correlation",
              os.path.join(OUTDIR, "per_cell_random_hist.png"))
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Per-gene correlation
    print("\n[INFO] Computing per-gene Spearman correlation...")
    t0 = time.time()
    per_gene_df, gene_idx = per_gene_spearman(X_rna, X_atac, gene_names, MIN_COEXPR_CELLS, MAX_GENES_CORR, RNG_SEED)
    print(f"[INFO] Done in {time.time()-t0:.2f}s (genes={len(gene_idx):,})")
    
    per_gene_csv = os.path.join(OUTDIR, "per_gene_correlations.csv")
    per_gene_df.to_csv(per_gene_csv, index=False)
    print(f"[SAVE] {per_gene_csv}")
    
    save_comparison_plots(corr_true, corr_rand, per_gene_df, MIN_COEXPR_CELLS, OUTDIR)
    
    # Summary
    valid_cell = np.isfinite(corr_true) if len(corr_true) > 0 else np.array([])
    valid_gene = np.isfinite(per_gene_df['spearman_corr'].values) if len(per_gene_df) > 0 else np.array([])
    
    summary = {
        "n_paired_cells": int(rna.n_obs),
        "n_cells_tested": int(len(corr_true)),
        "n_genes_tested": int(len(gene_idx)),
        "per_cell_mean": float(np.nanmean(corr_true)) if valid_cell.any() else None,
        "per_cell_median": float(np.nanmedian(corr_true)) if valid_cell.any() else None,
        "per_gene_mean": float(np.nanmean(per_gene_df['spearman_corr'])) if valid_gene.any() else None,
        "per_gene_median": float(np.nanmedian(per_gene_df['spearman_corr'])) if valid_gene.any() else None,
    }
    
    with open(os.path.join(OUTDIR, "correlation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    with open(os.path.join(OUTDIR, "summary.txt"), "w") as f:
        f.write(f"Paired cells: {rna.n_obs:,}\n")
        f.write(f"Genes tested: {len(gene_idx):,}\n")
        f.write(f"\n--- Per-cell Pearson ---\n")
        f.write(f"Mean (true): {summary['per_cell_mean']:.4f}\n" if summary['per_cell_mean'] else "Mean: N/A\n")
        f.write(f"Mean (rand): {np.nanmean(corr_rand):.4f}\n" if mr.any() else "")
        f.write(f"\n--- Per-gene Spearman ---\n")
        f.write(f"Mean: {summary['per_gene_mean']:.4f}\n" if summary['per_gene_mean'] else "Mean: N/A\n")
        sig = (per_gene_df['q_fdr_bh'] < 0.05).sum() if 'q_fdr_bh' in per_gene_df else 0
        f.write(f"Significant @ FDR<0.05: {sig:,}\n")
    
    print(f"\n[DONE] Results in: {OUTDIR}")
    if summary['per_cell_mean']:
        print(f"  Per-cell: Mean={summary['per_cell_mean']:.4f}")
    if summary['per_gene_mean']:
        print(f"  Per-gene: Mean={summary['per_gene_mean']:.4f}")


if __name__ == "__main__":
    main()