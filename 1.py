#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
decode_atac_to_rna_debug.py

🧪 DEBUG BUILD — SCGLUE Decoder: ATAC → Pseudo-RNA Prediction

Adds extensive diagnostics:
  • Environment & CUDA report
  • Checkpoint capability checks (decoder presence, modalities, vertices)
  • Input AnnData integrity (indices, layers, obsm keys, shapes)
  • Graph sanity (node/edge counts, sampled node types)
  • Gene coverage & target library-size estimation paths
  • Matrix dtype/index sparsity validations
  • Optional dry-run / skip-merge / row-limits for fast iteration

Outputs an AnnData with:
  - X: decoded (expected) RNA counts
  - layers["pseudo_rna"]: same as X
  - obs: copied from ATAC (suffix _ATAC after merge)
  - var: RNA var for decoded genes
"""

# ────────────────────────────────────────────────────────────────────────────────
# 🔧 USER PATHS (edit these defaults; can also override via CLI flags)
GLUE_DIR_DEFAULT        = "/dcs07/hongkai/data/harry/result/all/multiomics/integration/glue"
MODEL_PATH_DEFAULT      = "glue.dill"
ATAC_INPUT_PATH_DEFAULT = "glue-atac-emb.h5ad"
RNA_REF_PATH_DEFAULT    = "glue-rna-emb.h5ad"   # processed RNA with embeddings
RAW_RNA_PATH_DEFAULT    = "/dcl01/hongkai/data/data/hjiang/Data/paired/rna/all.h5ad"  # raw RNA counts
GRAPH_PATH_DEFAULT      = "glue-guidance-hvf.graphml.gz"
OUTPUT_DIR_DEFAULT      = ""                    # default to GLUE_DIR

# Training-time conventions (override via CLI if needed)
USE_BATCH_KEY_DEFAULT   = "sample"   # set to None if you didn't use batches
USE_ATAC_REP_DEFAULT    = "X_lsi"    # must match training
# ────────────────────────────────────────────────────────────────────────────────

import os
import sys
import gc
import time
import json
import argparse
import warnings
import traceback
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import anndata as ad
import networkx as nx
from scipy import sparse

# Optional deps for richer debug
try:
    import torch
except Exception:
    torch = None

try:
    import psutil
except Exception:
    psutil = None

import scglue

# ────────────────────────────────────────────────────────────────────────────────
# Pretty logging helpers
# ────────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def section(title: str):
    print(f"\n{title}")
    print("=" * max(12, len(title)))

def log(msg: str):
    print(msg)

def kv(k: str, v: Any):
    print(f"   {k}: {v}")

def head_list(x, n=5):
    x = list(x)
    return x[:n]

def warn(msg: str):
    print(f"[WARN] {msg}")

def err(msg: str):
    print(f"[ERROR] {msg}")

# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────

def mem_report(note: str = ""):
    if psutil is None:
        return
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / (1024 ** 3)
    kv("Memory RSS (GB)" + (f" — {note}" if note else ""), f"{rss:.3f}")

def fix_sparse_matrix_dtype(X, verbose=False):
    """Ensure CSR with int64 indices/indptr and float64 data."""
    if not sparse.issparse(X):
        return X
    if not isinstance(X, sparse.csr_matrix):
        X = X.tocsr()
    if verbose:
        kv("CSR indices dtype", getattr(X.indices, "dtype", "n/a"))
        kv("CSR indptr dtype", getattr(X.indptr, "dtype", "n/a"))

    # Rebuild as float64 data + int64 idx
    coo = X.tocoo()
    X_fixed = sparse.csr_matrix(
        (coo.data.astype(np.float64),
         (coo.row.astype(np.int64), coo.col.astype(np.int64))),
        shape=X.shape,
        dtype=np.float64
    )
    X_fixed.sort_indices()
    X_fixed.eliminate_zeros()
    if verbose:
        kv("Fixed indices dtype", X_fixed.indices.dtype)
        kv("Fixed indptr dtype", X_fixed.indptr.dtype)
    return X_fixed

def env_report(args):
    section("🧬 SCGLUE Decoder: ATAC → Pseudo-RNA Prediction")
    kv("Timestamp", _ts())
    kv("Python", sys.version.split()[0])
    kv("Platform", sys.platform)
    kv("scglue", getattr(scglue, "__version__", "unknown"))
    kv("anndata", getattr(ad, "__version__", "unknown"))
    kv("numpy", np.__version__)
    kv("pandas", pd.__version__)
    kv("scipy", getattr(sparse, "__version__", "unknown"))
    if torch is not None:
        kv("torch", torch.__version__)
        kv("CUDA available", torch.cuda.is_available())
        if torch.cuda.is_available():
            kv("CUDA device count", torch.cuda.device_count())
            kv("CUDA current device", torch.cuda.current_device())
            kv("CUDA device name", torch.cuda.get_device_name(0))
    else:
        warn("torch not installed — CUDA report unavailable.")
    mem_report("start")

    section("📁 Paths & Settings")
    for k in ["glue_dir", "model_path", "atac_path", "rna_ref_path", "rna_raw_path", "graph_path", "output_dir",
              "use_batch_key", "use_atac_rep", "batch_size", "max_atac", "dry_run", "skip_merge", "quiet"]:
        kv(k, getattr(args, k))

def load_h5ad(path, backed=None, name="AnnData"):
    kv(f"Loading {name}", path)
    obj = ad.read_h5ad(path, backed=backed)
    kv(f"{name} shape", obj.shape)
    if isinstance(obj, ad._core.anndata.AnnData) and obj.isbacked:
        kv(f"{name} backed", obj.filename)
    return obj

def check_index_unique(adata: ad.AnnData, name: str):
    if not adata.obs_names.is_unique:
        warn(f"{name} obs index not unique; making unique.")
        adata.obs_names_make_unique()

def check_glue_checkpoint(glue) -> Dict[str, Any]:
    """
    Probe model to see if decoding is actually supported.
    Returns a dict with details and booleans for guards.
    """
    info = {}
    section("🧠 Checkpoint Diagnostics")
    has_decode_method = hasattr(glue, "decode_data") or hasattr(glue, "decode")
    info["has_decode_method"] = has_decode_method
    kv("Has decode method", has_decode_method)

    # Common attributes on SCGLUE models
    for attr in ["vertices", "modules", "modalities", "omics", "adatas", "decoders", "encoders"]:
        info[attr] = getattr(glue, attr, None)
        try:
            length = len(info[attr]) if info[attr] is not None else "n/a"
        except Exception:
            length = "n/a"
        kv(attr, f"{type(info[attr]).__name__ if info[attr] is not None else 'None'} (len={length})")

    # Try to infer modality keys expected by the checkpoint
    # Heuristic: often "rna" and "atac" are used; store lowercased keys we see
    modality_keys = set()
    for candidate in [info.get("modalities"), info.get("omics"), info.get("adatas")]:
        if isinstance(candidate, (dict, list, tuple)):
            keys = list(candidate.keys()) if isinstance(candidate, dict) else list(range(len(candidate)))
            try:
                if isinstance(candidate, dict):
                    modality_keys.update(map(str, candidate.keys()))
            except Exception:
                pass
    if not modality_keys:
        modality_keys = {"rna", "atac"}  # best-effort default
        warn("Could not read modality keys from checkpoint; assuming {'rna','atac'}")
    info["modality_keys"] = sorted({k.lower() for k in modality_keys})
    kv("Inferred/assumed modality keys", info["modality_keys"])

    # Decoder presence check (best-effort)
    has_rna_decoder = False
    dec = info.get("decoders")
    if isinstance(dec, dict):
        for k in dec.keys():
            if "rna" in str(k).lower():
                has_rna_decoder = True
    elif dec is not None:
        # If it's a ModuleList or similar, we can't reliably infer.
        warn("Decoders present but type not dict; cannot directly confirm target='rna'.")
    info["has_rna_decoder"] = has_rna_decoder
    kv("Has explicit RNA decoder", has_rna_decoder)

    # Vertices check
    vertices = getattr(glue, "vertices", [])
    info["n_vertices"] = len(vertices) if vertices is not None else 0
    kv("Vertices", info["n_vertices"])
    if info["n_vertices"] == 0:
        warn("No vertices in checkpoint — decoding cannot proceed.")

    return info

def sample_graph_info(G: nx.Graph, n=5):
    section("🕸️ Graph Diagnostics")
    kv("Nodes", G.number_of_nodes())
    kv("Edges", G.number_of_edges())
    # Try sampling node types if present
    sample_nodes = head_list(G.nodes(), n=n)
    kv("Sample nodes", sample_nodes)
    # Node attributes (best-effort)
    types = set()
    for u in sample_nodes:
        t = G.nodes[u].get("type", None) if isinstance(G.nodes, dict) else None
        if t is not None:
            types.add(t)
    if types:
        kv("Sampled node types", sorted(types))

def compute_target_libsize(rna_raw: ad.AnnData, verbose=True) -> float:
    # Preference: layer "counts" if exists
    if "counts" in rna_raw.layers:
        if verbose:
            kv("Library-size source", "layers['counts']")
        vals = np.asarray(rna_raw.layers["counts"].sum(axis=1)).ravel()
        return float(np.median(vals))
    # Else use X (beware normalization)
    # To avoid huge RAM, sample first 100 rows
    if verbose:
        kv("Library-size source", "X (first 100 rows sample)")
    sample = rna_raw[: min(100, rna_raw.n_obs), :].X
    if sparse.issparse(sample):
        sample = sample.toarray()
    return float(np.median(sample.sum(axis=1)))

def sparsity_of(A) -> float:
    if sparse.issparse(A):
        return 1.0 - (A.nnz / (A.shape[0] * A.shape[1]))
    else:
        return 1.0 - (np.count_nonzero(A) / A.size)

def main():
    parser = argparse.ArgumentParser(description="DEBUG — SCGLUE: decode ATAC → pseudo-RNA")
    parser.add_argument("--glue-dir", default=GLUE_DIR_DEFAULT, type=str)
    parser.add_argument("--model-path", default=MODEL_PATH_DEFAULT, type=str)
    parser.add_argument("--atac-path", default=ATAC_INPUT_PATH_DEFAULT, type=str)
    parser.add_argument("--rna-ref-path", default=RNA_REF_PATH_DEFAULT, type=str)
    parser.add_argument("--rna-raw-path", default=RAW_RNA_PATH_DEFAULT, type=str)
    parser.add_argument("--graph-path", default=GRAPH_PATH_DEFAULT, type=str)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, type=str)
    parser.add_argument("--use-batch-key", default=USE_BATCH_KEY_DEFAULT, type=str)
    parser.add_argument("--use-atac-rep", default=USE_ATAC_REP_DEFAULT, type=str)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--max-atac", default=0, type=int, help="Limit ATAC cells for quick debug (0 = no limit)")
    parser.add_argument("--dry-run", action="store_true", help="Stop after all checks; skip decoding")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merging with real RNA; only write pseudo-RNA")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Resolve paths
    if not os.path.isabs(args.model_path):
        args.model_path = os.path.join(args.glue_dir, args.model_path)
    if not os.path.isabs(args.atac_path):
        args.atac_path = os.path.join(args.glue_dir, args.atac_path)
    if not os.path.isabs(args.rna_ref_path):
        args.rna_ref_path = os.path.join(args.glue_dir, args.rna_ref_path)
    if not os.path.isabs(args.graph_path):
        args.graph_path = os.path.join(args.glue_dir, args.graph_path)
    if not args.output_dir:
        args.output_dir = args.glue_dir

    # Print environment & settings
    if not args.quiet:
        env_report(args)

    try:
        section("📂 Loading resources...")
        kv("Loading trained SCGLUE model", args.model_path)
        glue = scglue.models.load_model(args.model_path)

        kv("Loading ATAC AnnData", args.atac_path)
        atac = ad.read_h5ad(args.atac_path)
        check_index_unique(atac, "ATAC")

        kv("Loading processed RNA reference", args.rna_ref_path)
        rna_processed = ad.read_h5ad(args.rna_ref_path)
        check_index_unique(rna_processed, "RNA (processed)")

        kv("Loading raw RNA (backed)", args.rna_raw_path)
        rna_raw = ad.read_h5ad(args.rna_raw_path, backed="r")

        kv("Loading guidance graph", args.graph_path)
        G = nx.read_graphml(args.graph_path)

        mem_report("after loads")

        # Limit ATAC rows for quicker debug
        if args.max_atac and atac.n_obs > args.max_atac:
            warn(f"Limiting ATAC cells from {atac.n_obs} → {args.max_atac} for debug.")
            atac = atac[: args.max_atac].copy()

        # Basic input integrity
        section("🔧 Configuring ATAC dataset...")
        if args.use_atac_rep not in atac.obsm:
            raise RuntimeError(f"Expected '{args.use_atac_rep}' in atac.obsm but not found. "
                               f"Available: {list(atac.obsm.keys())}")
        if args.use_batch_key and args.use_batch_key not in atac.obs:
            warn(f"'{args.use_batch_key}' not found in atac.obs — proceeding without batch.")
            use_batch = None
        else:
            use_batch = args.use_batch_key

        kv("ATAC obsm keys", list(atac.obsm.keys())[:10])
        kv("ATAC obs columns", list(atac.obs.columns)[:10])

        scglue.models.configure_dataset(
            atac, "NB",
            use_highly_variable=True,
            use_rep=args.use_atac_rep,
            use_batch=use_batch
        )

        section("🎯 Preparing target genes...")
        info = check_glue_checkpoint(glue)

        if not info["has_decode_method"]:
            raise RuntimeError(
                "This SCGLUE checkpoint does not expose a decode method. "
                "It likely wasn't trained with a decoder head. "
                "Tip: ensure you trained with decoders for target='rna', "
                "or use an alternative pseudo-RNA strategy."
            )

        # Graph sanity
        sample_graph_info(G)

        # Determine target gene list from model vertices ∩ RNA reference
        model_vertices = list(getattr(glue, "vertices", [])) or []
        if len(model_vertices) == 0:
            raise RuntimeError("Model has 0 vertices — cannot align decoded outputs to RNA genes.")

        rna_genes_set = set(map(str, rna_processed.var_names))
        target_genes = [g for g in model_vertices if g in rna_genes_set]
        kv("Target genes (overlap)", len(target_genes))
        if len(target_genes) == 0:
            raise RuntimeError("No RNA genes in model vertices intersect rna_processed.var_names. "
                               "Check naming/versioning and training inputs.")

        # Estimate library size
        target_libsize = compute_target_libsize(rna_raw, verbose=not args.quiet)
        kv("Target library size", f"{target_libsize:.2f}")

        if args.dry_run:
            section("🛑 Dry run complete — skipping decode by request.")
            return 0

        # ─────────────────────────────────────────────────────────
        section("🔄 Decoding ATAC → pseudo-RNA...")
        decoded = None
        try:
            # Prefer a method named decode_data if present
            if hasattr(glue, "decode_data"):
                decoded = glue.decode_data(
                    source_key="atac",
                    target_key="rna",
                    adata=atac,
                    graph=G,
                    target_libsize=target_libsize,
                    batch_size=int(args.batch_size),
                )
            elif hasattr(glue, "decode"):
                # Fallback signature (rare)
                decoded = glue.decode(
                    source_key="atac",
                    target_key="rna",
                    adata=atac,
                    graph=G,
                    target_libsize=target_libsize,
                    batch_size=int(args.batch_size),
                )
            else:
                raise RuntimeError("No decode method found even though earlier probe said otherwise.")
        except Exception as e:
            err("Decoding raised an exception.")
            print(traceback.format_exc())
            raise

        if decoded is None:
            raise RuntimeError("Decoding returned None — cannot proceed.")

        kv("Decoded array shape", getattr(decoded, "shape", "unknown"))

        # ─────────────────────────────────────────────────────────
        section("📐 Aligning decoded output...")
        vertex_index = {v: i for i, v in enumerate(model_vertices)}
        cols = [vertex_index[g] for g in target_genes]
        decoded_rna = decoded[:, cols]

        if isinstance(decoded_rna, np.ndarray):
            pass
        else:
            decoded_rna = np.asarray(decoded_rna)

        nans = np.isnan(decoded_rna).sum()
        kv("Decoded NaN count", int(nans))
        if nans:
            warn("NaNs detected in decoded outputs — converting NaN→0 and clipping to ≥0.")
        decoded_rna = np.nan_to_num(decoded_rna, 0.0)
        np.clip(decoded_rna, 0, None, out=decoded_rna)

        # Sparsify if helpful
        sp = sparsity_of(decoded_rna)
        kv("Decoded sparsity", f"{sp:.1%}")
        if sp > 0.5:
            log("   Converting decoded matrix to CSR (sparse) for memory efficiency…")
            X_dec = sparse.csr_matrix(decoded_rna, dtype=np.float64)
            X_dec = fix_sparse_matrix_dtype(X_dec, verbose=not args.quiet)
        else:
            X_dec = decoded_rna.astype(np.float64, copy=False)

        mem_report("post-decode")

        # ─────────────────────────────────────────────────────────
        section("🧪 Creating pseudo-RNA AnnData...")
        # Build var from raw RNA (ensures same metadata & order by target_genes)
        rna_var = rna_raw.var.copy()
        # Guard: ensure all target_genes exist in raw var
        missing_in_raw = [g for g in target_genes if g not in rna_var.index]
        if missing_in_raw:
            warn(f"{len(missing_in_raw)} target genes missing in raw RNA var; proceeding with intersection.")
        present = [g for g in target_genes if g in rna_var.index]
        rna_var_filtered = rna_var.loc[present].copy()
        # Re-align X_dec columns
        if len(present) != len(target_genes):
            cols2 = [vertex_index[g] for g in present]
            X_dec = (decoded[:, cols2] if not sparse.issparse(X_dec)
                     else sparse.csr_matrix(decoded[:, cols2]))
            if not sparse.issparse(X_dec):
                X_dec = np.asarray(X_dec)
            # Clean again just in case
            if not sparse.issparse(X_dec):
                X_dec = np.nan_to_num(X_dec, 0.0)
                np.clip(X_dec, 0, None, out=X_dec)

        # Copy obs and obsm from ATAC
        atac_obs = atac.obs.copy()
        atac_obsm_dict = {k: v.copy() for k, v in atac.obsm.items()}

        pseudo_rna_adata = ad.AnnData(
            X=X_dec,
            obs=atac_obs,
            var=rna_var_filtered
        )
        pseudo_rna_adata.obs["modality"] = "ATAC"
        pseudo_rna_adata.layers["pseudo_rna"] = pseudo_rna_adata.X.copy()

        for k, v in atac_obsm_dict.items():
            pseudo_rna_adata.obsm[k] = v

        pseudo_rna_adata.uns["decoder_info"] = {
            "source_modality": "atac",
            "target_modality": "rna",
            "model_path": args.model_path,
            "graph_path": args.graph_path,
            "use_rep": args.use_atac_rep,
            "use_batch": (args.use_batch_key if args.use_batch_key in atac.obs else None),
            "target_libsize": target_libsize,
            "n_target_genes": pseudo_rna_adata.n_vars
        }

        kv("Pseudo-RNA shape", pseudo_rna_adata.shape)
        kv("Pseudo-RNA matrix type", type(pseudo_rna_adata.X).__name__)
        kv("Pseudo-RNA sparsity", f"{sparsity_of(pseudo_rna_adata.X):.1%}")

        mem_report("post-pseudoRNA")

        # Early exit if skipping merge
        if args.skip_merge:
            section("💾 Saving pseudo-RNA only (skip-merge)")
            os.makedirs(args.output_dir, exist_ok=True)
            out_pseudo = os.path.join(args.output_dir, "atac_pseudo_rna_only.h5ad")
            pseudo_rna_adata.write(out_pseudo, compression="gzip", compression_opts=4)
            kv("Saved", out_pseudo)
            return 0

        # ─────────────────────────────────────────────────────────
        section("🔗 Preparing RNA data for merging…")
        processed_cells = rna_processed.obs.index
        raw_cells = rna_raw.obs.index
        common_cells = processed_cells.intersection(raw_cells)
        kv("RNA processed cells", len(processed_cells))
        kv("RNA raw cells", len(raw_cells))
        kv("Common cells", len(common_cells))
        if len(common_cells) == 0:
            warn("0 common cells between processed and raw RNA. Embeddings will be empty unless indices match.")

        # Prepare embeddings from processed
        rna_obsm_dict = {}
        for key in rna_processed.obsm.keys():
            emb = rna_processed.obsm[key]
            if len(common_cells) > 0:
                mask = processed_cells.isin(common_cells).to_numpy()
                rna_obsm_dict[key] = emb[mask]
            else:
                rna_obsm_dict[key] = emb

        # Extract raw counts (aligned to common cells if any)
        if len(common_cells) > 0:
            rna_X = rna_raw[common_cells, :].X
            rna_obs = pd.DataFrame(index=common_cells)
        else:
            rna_X = rna_raw[:, :].X
            rna_obs = pd.DataFrame(index=rna_raw.obs.index)

        # Normalize dtypes/sparsity
        if sparse.issparse(rna_X):
            rna_X = fix_sparse_matrix_dtype(rna_X, verbose=not args.quiet)
        else:
            rna_X = np.asarray(rna_X, dtype=np.float64)
            sp_r = sparsity_of(rna_X)
            if sp_r > 0.5:
                log("   Converting RNA X to CSR due to high sparsity…")
                rna_X = sparse.csr_matrix(rna_X, dtype=np.float64)
                rna_X = fix_sparse_matrix_dtype(rna_X, verbose=not args.quiet)

        # Close backed file ASAP
        try:
            rna_raw.file.close()
        except Exception:
            pass
        del rna_raw, rna_processed
        gc.collect()

        rna_for_merge = ad.AnnData(
            X=rna_X,
            obs=rna_obs,
            var=rna_var
        )
        rna_for_merge.obs["modality"] = "RNA"
        for k, v in rna_obsm_dict.items():
            rna_for_merge.obsm[k] = v

        # ─────────────────────────────────────────────────────────
        section("🔗 Merging pseudo-RNA with real RNA…")
        # Always suffix to guarantee uniqueness
        rna_for_merge.obs["original_barcode"] = rna_for_merge.obs.index
        pseudo_rna_adata.obs["original_barcode"] = pseudo_rna_adata.obs.index
        rna_for_merge.obs_names = pd.Index([f"{x}_RNA" for x in rna_for_merge.obs_names])
        pseudo_rna_adata.obs_names = pd.Index([f"{x}_ATAC" for x in pseudo_rna_adata.obs_names])

        kv("RNA cells", rna_for_merge.n_obs)
        kv("ATAC (pseudo-RNA) cells", pseudo_rna_adata.n_obs)

        merged_adata = ad.concat(
            [rna_for_merge, pseudo_rna_adata],
            axis=0,
            join="inner",
            merge="same",
            label=None,
            keys=None,
            index_unique=None
        )
        del rna_for_merge, pseudo_rna_adata
        gc.collect()

        if not merged_adata.obs_names.is_unique:
            warn("Merged obs names not unique; making unique.")
            merged_adata.obs_names_make_unique()

        if sparse.issparse(merged_adata.X):
            merged_adata.X = fix_sparse_matrix_dtype(merged_adata.X, verbose=not args.quiet)

        # ─────────────────────────────────────────────────────────
        section("💾 Saving integrated dataset…")
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "atac_pseudorna_integrated.h5ad")
        merged_adata.write(output_path, compression="gzip", compression_opts=4)

        # ─────────────────────────────────────────────────────────
        section("📊 Summary")
        kv("Output path", output_path)
        kv("Merged shape", merged_adata.shape)
        kv("RNA cells", int((merged_adata.obs["modality"] == "RNA").sum()))
        kv("ATAC cells (pseudo-RNA)", int((merged_adata.obs["modality"] == "ATAC").sum()))
        kv("Genes", merged_adata.n_vars)
        if sparse.issparse(merged_adata.X):
            kv("Matrix", f"{type(merged_adata.X).__name__} (sparse)")
            kv("Index dtype", merged_adata.X.indices.dtype)
            kv("Indptr dtype", merged_adata.X.indptr.dtype)
            sp_m = 1 - merged_adata.X.nnz / (merged_adata.shape[0] * merged_adata.shape[1])
            kv("Sparsity", f"{sp_m:.1%}")
            kv("Data dtype", merged_adata.X.dtype)
            kv("Data bytes (MB)", f"{merged_adata.X.data.nbytes/1e6:.2f}")
        else:
            kv("Matrix", "dense numpy")
            kv("Data dtype", merged_adata.X.dtype)
            try:
                kv("Data bytes (MB)", f"{merged_adata.X.nbytes/1e6:.2f}")
            except Exception:
                pass

        layers = list(merged_adata.layers.keys())
        kv("Layers", layers if layers else "None")
        mem_report("end")
        return 0

    except Exception as e:
        section("❌ Debug Failure Report")
        err(str(e))
        print("\nTraceback:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
