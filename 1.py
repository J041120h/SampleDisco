#!/usr/bin/env python3
"""
decode_atac_to_rna.py

Load a trained SCGLUE model (.dill) and use its RNA decoder to generate
pseudo-RNA expression for ATAC cells.

Outputs an AnnData with:
  - X: decoded (expected) RNA counts for HV (or graphed) genes
  - layers["rna_decoded"]: same as X
  - obs: copied from ATAC
  - var: decoded RNA gene index (ordered to match the model's vertices)
"""
import time
import gc
import os 

def simple_mem_stress(mb: int = 100, hold_seconds: int = 10):
    MAX_MB = 819200  # safety cap
    if mb <= 0:
        raise ValueError("mb must be > 0")
    if mb > MAX_MB:
        print(f"[WARN] Requested {mb} MB exceeds safety cap of {MAX_MB} MB — capping to {MAX_MB} MB.")
        mb = MAX_MB

    chunk_size = 1024 * 1024  # 1 MiB
    allocated = []
    try:
        for i in range(mb):
            # allocate 1 MiB block and touch first byte so pages get committed
            b = bytearray(chunk_size)
            b[0] = 1
            allocated.append(b)
            # occasional progress print
            if (i + 1) % 50 == 0 or (i + 1) == mb:
                print(f"[INFO] Allocated {(i + 1)} / {mb} MiB")
        print(f"[INFO] Holding {mb} MiB for {hold_seconds}s...")
        time.sleep(hold_seconds)
    except MemoryError:
        print("[ERROR] MemoryError: allocation failed before reaching target size.")
    finally:
        # free and hint GC
        allocated.clear()
        gc.collect()
        print("[INFO] Memory released.")


# ────────────────────────────────────────────────────────────────────────────────
# 🔧 USER PATHS (edit these)
GLUE_DIR        = "/dcs07/hongkai/data/harry/result/all/multiomics/integration/glue"
MODEL_PATH      = os.path.join(GLUE_DIR, "glue.dill")
ATAC_INPUT_PATH = os.path.join(GLUE_DIR, "glue-atac-emb.h5ad")
RNA_REF_PATH    = os.path.join(GLUE_DIR, "glue-rna-emb.h5ad")   # processed RNA with embeddings
RAW_RNA_PATH    = "/dcl01/hongkai/data/data/hjiang/Data/paired/rna/all.h5ad"                 # NEW: raw RNA counts for merging
GRAPH_PATH      = os.path.join(GLUE_DIR, "glue-guidance-hvf.graphml.gz")
OUTPUT_DIR      = os.path.join(GLUE_DIR)           # Match KNN convention

# Training parameters
USE_BATCH_KEY   = "sample"   # set to None if you didn't use batches
USE_ATAC_REP    = "X_lsi"    # must match training
# ────────────────────────────────────────────────────────────────────────────────

import anndata as ad
import numpy as np
import pandas as pd
import networkx as nx
import scglue
import os
from scipy import sparse
import gc

def fix_sparse_matrix_dtype(X, verbose=False):
    """Fix sparse matrix by converting to int64 indices"""
    if not sparse.issparse(X):
        return X
        
    if verbose:
        print(f"   Converting sparse matrix indices to int64...")
        print(f"   Current dtypes - indices: {X.indices.dtype}, indptr: {X.indptr.dtype}")
    
    # Convert to COO then back to CSR with int64
    coo = X.tocoo()
    
    # Create new CSR matrix with int64 indices
    X_fixed = sparse.csr_matrix(
        (coo.data.astype(np.float64), 
         (coo.row.astype(np.int64), coo.col.astype(np.int64))),
        shape=X.shape,
        dtype=np.float64
    )
    
    # Clean up
    X_fixed.eliminate_zeros()
    X_fixed.sort_indices()
    
    if verbose:
        print(f"   Fixed dtypes - indices: {X_fixed.indices.dtype}, indptr: {X_fixed.indptr.dtype}")
    
    return X_fixed

def main():
    verbose = True
    
    if verbose:
        print("\n🧬 SCGLUE Decoder: ATAC → Pseudo-RNA Prediction")
        print("=" * 70)
    
    # ============================================================
    # 1) Load model & inputs
    # ============================================================
    if verbose:
        print("\n📂 Loading resources...")
    
    print("   Loading trained SCGLUE model...")
    glue = scglue.models.load_model(MODEL_PATH)
    
    print("   Loading ATAC AnnData...")
    atac = ad.read_h5ad(ATAC_INPUT_PATH)
    
    print("   Loading processed RNA reference (for embeddings)...")
    rna_processed = ad.read_h5ad(RNA_REF_PATH)
    
    print("   Loading raw RNA counts (for merging)...")
    rna_raw = ad.read_h5ad(RAW_RNA_PATH, backed='r')
    
    print("   Loading guidance graph...")
    G = nx.read_graphml(GRAPH_PATH)
    
    # ============================================================
    # 2) Validate and configure ATAC dataset
    # ============================================================
    if verbose:
        print("\n🔧 Configuring ATAC dataset...")
    
    if USE_ATAC_REP not in atac.obsm:
        raise RuntimeError(f"Expected '{USE_ATAC_REP}' in atac.obsm, but not found.")
    if USE_BATCH_KEY is not None and USE_BATCH_KEY not in atac.obs:
        print(f"   [WARN] '{USE_BATCH_KEY}' not found in atac.obs — proceeding without batch.")
    
    scglue.models.configure_dataset(
        atac, "NB",
        use_highly_variable=True,
        use_rep=USE_ATAC_REP,
        use_batch=USE_BATCH_KEY if USE_BATCH_KEY in atac.obs else None
    )
    
    # ============================================================
    # 3) Prepare target genes and library size
    # ============================================================
    if verbose:
        print("\n🎯 Preparing target genes...")
    
    model_vertices = list(getattr(glue, "vertices", []))
    rna_genes_set = set(rna_processed.var_names)
    target_genes = [g for g in model_vertices if g in rna_genes_set]
    
    if len(target_genes) == 0:
        raise RuntimeError("No RNA genes found in model vertices; check that RNA reference matches training.")
    
    print(f"   Target genes: {len(target_genes)}")
    
    # Estimate library size from raw RNA
    if "counts" in rna_raw.layers:
        target_libsize = float(np.median(np.asarray(rna_raw.layers["counts"].sum(axis=1)).ravel()))
    else:
        # Use raw counts if available
        rna_sample = rna_raw[:100, :].X
        if sparse.issparse(rna_sample):
            rna_sample = rna_sample.toarray()
        target_libsize = float(np.median(rna_sample.sum(axis=1)))
    
    print(f"   Target library size: {target_libsize:.2f}")
    
    # ============================================================
    # 4) Run cross-modal decoding
    # ============================================================
    if verbose:
        print("\n🔄 Decoding ATAC → pseudo-RNA...")
    
    decoded = glue.decode_data(
        source_key="atac",
        target_key="rna",
        adata=atac,
        graph=G,
        target_libsize=target_libsize,
        batch_size=256
    )
    
    # ============================================================
    # 5) Align decoded output to target genes
    # ============================================================
    if verbose:
        print("\n📐 Aligning decoded output...")
    
    vertex_index = {v: i for i, v in enumerate(model_vertices)}
    cols = [vertex_index[g] for g in target_genes]
    decoded_rna = decoded[:, cols]
    
    # Clean and validate
    decoded_rna = np.nan_to_num(decoded_rna, 0)
    np.clip(decoded_rna, 0, None, out=decoded_rna)
    
    # ============================================================
    # 6) Create pseudo-RNA AnnData (ATAC cells with predicted RNA)
    # ============================================================
    if verbose:
        print("\n🧪 Creating pseudo-RNA AnnData...")
    
    # Store ATAC obs and obsm
    atac_obs = atac.obs.copy()
    atac_obsm_dict = {k: v.copy() for k, v in atac.obsm.items()}
    
    # Convert to sparse if beneficial
    nnz = np.count_nonzero(decoded_rna)
    sparsity = 1 - (nnz / decoded_rna.size)
    
    if sparsity > 0.5:
        if verbose:
            print(f"   Converting to sparse (sparsity: {sparsity:.1%})")
        decoded_sparse = sparse.csr_matrix(decoded_rna, dtype=np.float64)
        decoded_sparse = fix_sparse_matrix_dtype(decoded_sparse, verbose=verbose)
    else:
        decoded_sparse = decoded_rna.astype(np.float64)
    
    # Get RNA var metadata from raw RNA
    rna_var = rna_raw.var.copy()
    # Filter to only target genes
    rna_var_filtered = rna_var.loc[target_genes].copy()
    
    # Create pseudo-RNA AnnData
    pseudo_rna_adata = ad.AnnData(
        X=decoded_sparse,
        obs=atac_obs,
        var=rna_var_filtered
    )
    
    # Add modality label
    pseudo_rna_adata.obs['modality'] = 'ATAC'
    
    # Store decoded data in layer
    pseudo_rna_adata.layers['pseudo_rna'] = pseudo_rna_adata.X.copy()
    
    # Preserve ATAC embeddings
    for key, value in atac_obsm_dict.items():
        pseudo_rna_adata.obsm[key] = value
    
    # Add decoding metadata
    pseudo_rna_adata.uns['decoder_info'] = {
        'source_modality': 'atac',
        'target_modality': 'rna',
        'model_path': MODEL_PATH,
        'graph_path': GRAPH_PATH,
        'use_rep': USE_ATAC_REP,
        'use_batch': (USE_BATCH_KEY if USE_BATCH_KEY in atac.obs else None),
        'target_libsize': target_libsize,
        'n_target_genes': len(target_genes)
    }
    
    del atac, decoded, decoded_rna
    gc.collect()
    
    # ============================================================
    # 7) Prepare RNA data for merging
    # ============================================================
    if verbose:
        print("\n🔗 Preparing RNA data for merging...")
    
    # Align cells between processed and raw RNA
    processed_cells = rna_processed.obs.index
    raw_cells = rna_raw.obs.index
    common_cells = processed_cells.intersection(raw_cells)
    
    if len(common_cells) != len(processed_cells):
        print(f"   Aligning to {len(common_cells)} common cells...")
    
    # Get RNA embeddings from processed data
    rna_obsm_dict = {}
    for key in rna_processed.obsm.keys():
        if key in rna_processed.obsm:
            embedding = rna_processed.obsm[key]
            # Align to common cells
            embedding_mask = np.isin(processed_cells, common_cells)
            rna_obsm_dict[key] = embedding[embedding_mask]
    
    # Load raw RNA counts for common cells
    is_sparse_rna = sparse.issparse(rna_raw.X)
    
    if is_sparse_rna:
        rna_X = rna_raw[common_cells, :].X
        if sparse.issparse(rna_X):
            rna_X = rna_X.tocsr().astype(np.float64)
            rna_X = fix_sparse_matrix_dtype(rna_X, verbose=verbose)
    else:
        rna_X = rna_raw[common_cells, :].X[:].astype(np.float64)
        nnz = np.count_nonzero(rna_X)
        sparsity = 1 - (nnz / rna_X.size)
        if sparsity > 0.5:
            if verbose:
                print(f"   Converting RNA to sparse (sparsity: {sparsity:.1%})")
            rna_X = sparse.csr_matrix(rna_X, dtype=np.float64)
            rna_X = fix_sparse_matrix_dtype(rna_X, verbose=verbose)
    
    # Close backed file
    rna_raw.file.close()
    del rna_raw, rna_processed
    gc.collect()
    
    # Create RNA AnnData
    rna_obs = raw_cells.to_frame() if isinstance(raw_cells, pd.Index) else pd.DataFrame(index=raw_cells)
    rna_obs = rna_obs.loc[common_cells]
    
    rna_for_merge = ad.AnnData(
        X=rna_X,
        obs=rna_obs,
        var=rna_var
    )
    
    rna_for_merge.obs['modality'] = 'RNA'
    
    # Add embeddings
    for key, value in rna_obsm_dict.items():
        rna_for_merge.obsm[key] = value
    
    # ============================================================
    # 8) Handle index overlaps and merge
    # ============================================================
    if verbose:
        print("\n🔗 Merging pseudo-RNA with real RNA data...")
        print("   Checking for index overlaps...")
    
    rna_indices = set(rna_for_merge.obs.index)
    atac_indices = set(pseudo_rna_adata.obs.index)
    overlap = rna_indices.intersection(atac_indices)
    
    if overlap:
        print(f"   ⚠️ Found {len(overlap)} overlapping indices")
    else:
        print(f"   No overlapping indices found, but adding suffix for safety...")
    
    # ALWAYS add modality suffix
    # Store original barcodes
    rna_for_merge.obs['original_barcode'] = rna_for_merge.obs.index
    pseudo_rna_adata.obs['original_barcode'] = pseudo_rna_adata.obs.index
    
    # Create unique indices with modality suffix
    rna_for_merge.obs.index = pd.Index([f"{idx}_RNA" for idx in rna_for_merge.obs.index])
    pseudo_rna_adata.obs.index = pd.Index([f"{idx}_ATAC" for idx in pseudo_rna_adata.obs.index])
    
    print(f"   RNA cells: {rna_for_merge.n_obs}")
    print(f"   ATAC cells (pseudo-RNA): {pseudo_rna_adata.n_obs}")
    
    # Merge datasets
    merged_adata = ad.concat(
        [rna_for_merge, pseudo_rna_adata],
        axis=0,
        join='inner',  # Only keep genes present in both
        merge='same',
        label=None,
        keys=None,
        index_unique=None
    )
    
    del rna_for_merge, pseudo_rna_adata
    gc.collect()
    
    # Verify unique indices
    if not merged_adata.obs.index.is_unique:
        print("   ⚠️ Warning: Non-unique indices detected, fixing...")
        merged_adata.obs_names_make_unique()
    
    # ============================================================
    # 9) Ensure proper sparse format
    # ============================================================
    if sparse.issparse(merged_adata.X):
        if verbose:
            print("\n📦 Ensuring merged matrix has int64 indices...")
        
        if not isinstance(merged_adata.X, sparse.csr_matrix):
            merged_adata.X = merged_adata.X.tocsr()
        
        merged_adata.X = fix_sparse_matrix_dtype(merged_adata.X, verbose=verbose)
        merged_adata.X.sort_indices()
        merged_adata.X.eliminate_zeros()
    
    # ============================================================
    # 10) Save merged output
    # ============================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'atac_pseudorna_integrated.h5ad')
    
    if verbose:
        print(f"\n💾 Saving integrated dataset...")
    
    merged_adata.write(output_path, compression='gzip', compression_opts=4)
    
    # ============================================================
    # Summary
    # ============================================================
    if verbose:
        print(f"\n✅ SCGLUE decoding and merging complete!")
        print(f"\n📊 Summary:")
        print(f"   Output path: {output_path}")
        print(f"   Merged dataset shape: {merged_adata.shape}")
        print(f"   RNA cells: {(merged_adata.obs['modality'] == 'RNA').sum()}")
        print(f"   ATAC cells (pseudo-RNA): {(merged_adata.obs['modality'] == 'ATAC').sum()}")
        print(f"   Unique cell indices: {merged_adata.obs.index.is_unique}")
        print(f"   Genes: {merged_adata.n_vars}")
        
        if sparse.issparse(merged_adata.X):
            matrix_type = type(merged_adata.X).__name__
            sparsity = 1 - (merged_adata.X.nnz / (merged_adata.X.shape[0] * merged_adata.X.shape[1]))
            print(f"   Matrix format: {matrix_type} (sparse)")
            print(f"   Matrix index dtype: {merged_adata.X.indices.dtype}")
            print(f"   Matrix indptr dtype: {merged_adata.X.indptr.dtype}")
            print(f"   Data type: {merged_adata.X.dtype}")
            print(f"   Sparsity: {sparsity:.1%} zeros")
            print(f"   Memory usage: {merged_adata.X.data.nbytes / 1e6:.2f} MB")
        else:
            print(f"   Matrix format: dense numpy array")
            print(f"   Data type: {merged_adata.X.dtype}")
            print(f"   Memory usage: {merged_adata.X.nbytes / 1e6:.2f} MB")
        
        print(f"\n   Available layers:")
        for layer in merged_adata.layers.keys():
            print(f"     - {layer}")
    
    return merged_adata

if __name__ == "__main__":
    main()
