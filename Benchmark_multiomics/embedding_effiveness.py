#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ====== EDIT THESE PARAMETERS ======
H5AD_PATH = "/dcs07/hongkai/data/harry/result/Benchmark/multiomics/preprocess/atac_rna_integrated.h5ad"   # <-- change me
OUTDIR    = "/dcs07/hongkai/data/harry/result/Benchmark/multiomics"               # <-- change me

# GPU and batch settings
USE_GPU             = True      # Set to False to run on CPU
BATCH_SIZE          =  10000     # Number of cells to process at once for cosine similarity
DEVICE              = "cuda:0"  # "cuda:0", "cuda:1", or "cpu"

# Optional speed knobs for LARGE data (keep None for full run)
MAX_PAIRS_COSINE    = None     # e.g., 20000 to subsample cell-pairs for cosine
MAX_GENES_WILCOXON  = None     # e.g., 10000 to subsample genes for Wilcoxon
MIN_PAIRS_PER_GENE  = 10
RNG_SEED            = 0
# ==================================

import os, time, gc
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests
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

def pair_by_barcode(adata):
    if 'modality' not in adata.obs or 'original_barcode' not in adata.obs:
        raise ValueError("Expected obs['modality'] and obs['original_barcode'] in the integrated AnnData.")
    rna  = adata[adata.obs['modality'] == 'RNA'].copy()
    atac = adata[adata.obs['modality'] == 'ATAC'].copy()

    r = rna.obs[['original_barcode']].reset_index().rename(columns={'index':'rna_idx'})
    a = atac.obs[['original_barcode']].reset_index().rename(columns={'index':'atac_idx'})
    pairs = r.merge(a, on='original_barcode', how='inner')

    rna  = rna[pairs['rna_idx'].values].copy()
    atac = atac[pairs['atac_idx'].values].copy()
    if not np.array_equal(rna.var_names.values, atac.var_names.values):
        raise ValueError("Gene order mismatch between RNA and ATAC blocks. Align vars first.")
    return rna, atac, pairs

def sparse_to_torch(X_sparse, device):
    """Convert scipy sparse matrix to torch sparse tensor"""
    X_coo = sp.coo_matrix(X_sparse)
    indices = torch.from_numpy(np.vstack([X_coo.row, X_coo.col])).long()
    values = torch.from_numpy(X_coo.data).float()
    shape = X_coo.shape
    return torch.sparse_coo_tensor(indices, values, shape).to(device)

def batched_cosine_similarity_gpu(X_atac, X_rna, batch_size, device, max_pairs=None, seed=42):
    """GPU-accelerated batched cosine similarity computation"""
    n = X_atac.shape[0]
    idx = np.arange(n)
    if max_pairs is not None and max_pairs < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_pairs, replace=False)
    
    m = len(idx)
    cos_true = np.empty(m, dtype=np.float32)
    
    # Process in batches
    n_batches = (m + batch_size - 1) // batch_size
    print(f"[GPU] Processing {m:,} pairs in {n_batches} batches of size {batch_size}")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, m)
        batch_indices = idx[start_idx:end_idx]
        
        # Extract batch as dense tensors (more efficient for small-medium datasets)
        X_atac_batch = torch.from_numpy(X_atac[batch_indices].toarray()).float().to(device)
        X_rna_batch = torch.from_numpy(X_rna[batch_indices].toarray()).float().to(device)
        
        # Compute cosine similarity: dot / (norm_x * norm_y)
        dot_product = (X_atac_batch * X_rna_batch).sum(dim=1)
        norm_atac = torch.norm(X_atac_batch, dim=1)
        norm_rna = torch.norm(X_rna_batch, dim=1)
        
        # Handle zero norms
        cos_batch = dot_product / (norm_atac * norm_rna + 1e-8)
        cos_batch[(norm_atac == 0) | (norm_rna == 0)] = float('nan')
        
        cos_true[start_idx:end_idx] = cos_batch.cpu().numpy()
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            print(f"  Batch {batch_idx + 1}/{n_batches} complete")
    
    return cos_true, idx

def batched_cosine_random_gpu(X_atac, X_rna, idx, batch_size, device, seed=42):
    """GPU-accelerated random baseline cosine similarity"""
    n = X_atac.shape[0]
    m = len(idx)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)[:m]
    
    cos_rand = np.empty(m, dtype=np.float32)
    n_batches = (m + batch_size - 1) // batch_size
    
    print(f"[GPU] Computing random baseline in {n_batches} batches")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, m)
        
        batch_atac_idx = idx[start_idx:end_idx]
        batch_rna_idx = perm[start_idx:end_idx]
        
        X_atac_batch = torch.from_numpy(X_atac[batch_atac_idx].toarray()).float().to(device)
        X_rna_batch = torch.from_numpy(X_rna[batch_rna_idx].toarray()).float().to(device)
        
        dot_product = (X_atac_batch * X_rna_batch).sum(dim=1)
        norm_atac = torch.norm(X_atac_batch, dim=1)
        norm_rna = torch.norm(X_rna_batch, dim=1)
        
        cos_batch = dot_product / (norm_atac * norm_rna + 1e-8)
        cos_batch[(norm_atac == 0) | (norm_rna == 0)] = float('nan')
        
        cos_rand[start_idx:end_idx] = cos_batch.cpu().numpy()
    
    return cos_rand

def per_cell_cosine_gpu(X_atac, X_rna, batch_size, device, max_pairs=None, random_baseline=True, seed=0):
    """GPU-accelerated per-cell cosine similarity with batching"""
    cos_true, idx = batched_cosine_similarity_gpu(X_atac, X_rna, batch_size, device, max_pairs, seed)
    
    cos_rand = None
    if random_baseline:
        cos_rand = batched_cosine_random_gpu(X_atac, X_rna, idx, batch_size, device, seed)
    
    return cos_true, cos_rand

def row_cosine(x_row, y_row):
    """CPU fallback for single row cosine"""
    dot = x_row.multiply(y_row).sum()
    nx  = np.sqrt(x_row.multiply(x_row).sum())
    ny  = np.sqrt(y_row.multiply(y_row).sum())
    if nx == 0 or ny == 0:
        return np.nan
    return dot / (nx * ny)

def per_cell_cosine_cpu(X_atac, X_rna, max_pairs=None, random_baseline=True, seed=0):
    """CPU version (original implementation)"""
    n = X_atac.shape[0]
    idx = np.arange(n)
    if max_pairs is not None and max_pairs < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_pairs, replace=False)
    m = len(idx)

    cos_true = np.empty(m, dtype=np.float64)
    for i, k in enumerate(idx):
        cos_true[i] = row_cosine(X_atac.getrow(k), X_rna.getrow(k))
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{m} cells")

    cos_rand = None
    if random_baseline:
        rng = np.random.default_rng(seed + 1)
        perm = rng.permutation(n)[:m]
        cos_rand = np.empty(m, dtype=np.float64)
        for i, (k, j) in enumerate(zip(idx, perm)):
            cos_rand[i] = row_cosine(X_atac.getrow(k), X_rna.getrow(j))
    return cos_true, cos_rand

def gene_wise_wilcoxon(X_atac, X_rna, min_pairs=10, max_genes=None, seed=0):
    """Gene-wise Wilcoxon test (remains on CPU for statistical computation)"""
    G = X_atac.shape[1]
    if max_genes is not None and max_genes < G:
        rng = np.random.default_rng(seed)
        gene_idx = np.sort(rng.choice(G, size=max_genes, replace=False))
    else:
        gene_idx = np.arange(G)
    A = X_atac.tocsc(copy=False)
    R = X_rna.tocsc(copy=False)

    rec = []
    for i, g in enumerate(gene_idx):
        x = A.getcol(g).toarray().ravel()
        y = R.getcol(g).toarray().ravel()
        mask = ~((x == 0) & (y == 0))
        if mask.sum() < min_pairs:
            rec.append((g, int(mask.sum()), np.nan, np.nan))
            continue
        try:
            _, p = wilcoxon(x[mask], y[mask], zero_method='wilcox', alternative='two-sided', mode='auto')
        except ValueError:
            p = np.nan
        med_diff = np.median(x[mask] - y[mask])
        rec.append((g, int(mask.sum()), p, med_diff))
        
        if (i + 1) % 500 == 0:
            print(f"  Tested {i + 1}/{len(gene_idx)} genes")

    res = pd.DataFrame(rec, columns=['gene_idx','n_pairs_tested','p_wilcoxon','median_diff_atac_minus_rna'])
    valid = res['p_wilcoxon'].notna().values
    if valid.sum() > 0:
        _, qvals, _, _ = multipletests(res.loc[valid,'p_wilcoxon'].values, method='fdr_bh')
        res.loc[valid, 'q_fdr_bh'] = qvals
    else:
        res['q_fdr_bh'] = np.nan
    return res, gene_idx

def save_hist(data, title, xlabel, path_png, bins=50):
    plt.figure(figsize=(8,5))
    plt.hist(data[~np.isnan(data)], bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.close()

def save_scatter(x, y, title, xlabel, ylabel, path_png, s=6, alpha=0.6):
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, s=s, alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.close()

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    
    # Check GPU availability
    if USE_GPU:
        if torch.cuda.is_available():
            device = torch.device(DEVICE)
            print(f"[INFO] Using GPU: {torch.cuda.get_device_name(device)}")
            print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
            print(f"[INFO] Batch size: {BATCH_SIZE}")
        else:
            print("[WARNING] GPU requested but CUDA not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    
    print(f"[INFO] Loading: {H5AD_PATH}")
    adata = ad.read_h5ad(H5AD_PATH)

    print("[INFO] Pairing RNA/ATAC by original_barcode…")
    rna, atac, pairs = pair_by_barcode(adata)
    print(f"[INFO] Paired cells: {rna.n_obs:,}")

    X_rna  = ensure_csr64(rna.X)
    X_atac = ensure_csr64(atac.X)

    # ---------- Per-cell cosine similarity ----------
    print("[INFO] Computing per-cell cosine similarity…")
    t0 = time.time()
    
    if device.type == "cuda":
        cos_true, cos_rand = per_cell_cosine_gpu(
            X_atac, X_rna,
            batch_size=BATCH_SIZE,
            device=device,
            max_pairs=MAX_PAIRS_COSINE,
            random_baseline=True,
            seed=RNG_SEED
        )
    else:
        cos_true, cos_rand = per_cell_cosine_cpu(
            X_atac, X_rna,
            max_pairs=MAX_PAIRS_COSINE,
            random_baseline=True,
            seed=RNG_SEED
        )
    
    print(f"[INFO] Cosine done in {time.time()-t0:.2f}s (n={len(cos_true):,}).")

    cos_df = pd.DataFrame({'cosine_true_pair': cos_true})
    if cos_rand is not None:
        cos_df['cosine_random_mismatch'] = cos_rand
    cos_csv = os.path.join(OUTDIR, "per_cell_cosine_similarity.csv")
    cos_df.to_csv(cos_csv, index=False)
    print(f"[SAVE] {cos_csv}")

    if cos_rand is not None:
        mt, mr = ~np.isnan(cos_true), ~np.isnan(cos_rand)
        if mt.sum() > 2 and mr.sum() > 2:
            u, p = mannwhitneyu(cos_true[mt], cos_rand[mr], alternative='greater')
            with open(os.path.join(OUTDIR, "per_cell_cosine_MWU.txt"), "w") as f:
                f.write(f"Mann–Whitney U (true > random): U={u}, p={p:.3e}\n")
                f.write(f"Median(true)={np.nanmedian(cos_true):.4f}, Median(rand)={np.nanmedian(cos_rand):.4f}\n")
            print(f"[STAT] Mann–Whitney U p={p:.3e}")

    save_hist(cos_true,
              "Per-cell cosine similarity (true pairs)",
              "Cosine similarity",
              os.path.join(OUTDIR, "per_cell_cosine_true_hist.png"))
    if cos_rand is not None:
        save_hist(cos_rand,
                  "Per-cell cosine similarity (random mismatches)",
                  "Cosine similarity",
                  os.path.join(OUTDIR, "per_cell_cosine_random_hist.png"))

    # Clear GPU memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    # ---------- Gene-wise paired Wilcoxon ----------
    print("[INFO] Running gene-wise paired Wilcoxon across cells…")
    t0 = time.time()
    res, gene_idx = gene_wise_wilcoxon(
        X_atac, X_rna,
        min_pairs=MIN_PAIRS_PER_GENE,
        max_genes=MAX_GENES_WILCOXON,
        seed=RNG_SEED
    )
    print(f"[INFO] Wilcoxon done in {time.time()-t0:.2f}s (genes tested={len(gene_idx):,}).")

    # Attach gene names
    res['gene'] = rna.var_names.values[gene_idx]
    res = res[['gene','n_pairs_tested','p_wilcoxon','q_fdr_bh','median_diff_atac_minus_rna']]

    out_csv = os.path.join(OUTDIR, "gene_wise_paired_wilcoxon.csv")
    res.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}")

    # Volcano-style (trend view)
    q = res['q_fdr_bh'].values
    with np.errstate(divide='ignore'):
        neglog10q = -np.log10(q)
    save_scatter(
        x=res['median_diff_atac_minus_rna'].values,
        y=neglog10q,
        title="Gene-wise ATAC–RNA differences (Wilcoxon, FDR)",
        xlabel="Median(ATAC - RNA) per gene",
        ylabel="-log10(FDR q-value)",
        path_png=os.path.join(OUTDIR, "gene_wise_volcano.png"),
        s=6, alpha=0.6
    )

    # Global bias (overall pattern)
    save_hist(
        res['median_diff_atac_minus_rna'].values,
        "Distribution of median(ATAC - RNA) across genes",
        "Median difference (ATAC - RNA)",
        os.path.join(OUTDIR, "gene_wise_median_diff_hist.png")
    )

    # Short summary
    with open(os.path.join(OUTDIR, "summary.txt"), "w") as f:
        f.write(f"Paired cells: {rna.n_obs}\n")
        f.write(f"Genes tested: {res['gene'].notna().sum()}\n")
        f.write(f"Per-cell cosine median (true): {np.nanmedian(cos_true):.4f}\n")
        if cos_rand is not None:
            f.write(f"Per-cell cosine median (random): {np.nanmedian(cos_rand):.4f}\n")
        f.write(f"Median gene-wise (ATAC - RNA): {np.nanmedian(res['median_diff_atac_minus_rna'].values):.4f}\n")
        sig = (res['q_fdr_bh'].notna()) & (res['q_fdr_bh'] < 0.05)
        f.write(f"Significant genes @ FDR<0.05: {int(sig.sum())}\n")
    print(f"[DONE] Outputs in: {OUTDIR}")

if __name__ == "__main__":
    main()