"""Heart-only patch:
  - Reuse already-saved heart default-α SE.
  - Re-run heart autotune with grouping_col=None (disease_state is categorical,
    incompatible with the CCA-based autotune scorer).
  - Run heart benchmark for V2_tuned + V2_default + all competitors.
"""
from __future__ import annotations
import os, sys, time, gc
sys.path.insert(0, "/users/hjiang/GenoDistance/code")
sys.path.insert(0, "/users/hjiang/GenoDistance/code/Benchmark_multiomics")
sys.path.insert(0, "/users/hjiang/GenoDistance/code/claude")

import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import torch
from run_test_multiomics import (
    DATASETS, TEST_ROOT, load_minimal, cell_typing_v2, run_bench_one_dataset, log
)

cfg = DATASETS['heart']
out_dir = os.path.join(TEST_ROOT, 'heart')
os.makedirs(out_dir, exist_ok=True)

# Reload + redo Harmony + redo cell typing (cheap on heart: 198K cells)
log("loading heart h5ad (memory-safe)")
a = load_minimal(cfg['h5'])
log(f"  shape={a.shape}")

from preparation.multi_omics_batch_correction import (
    harmonize_xglue, XGLUE_HARMONY_KEY, XGLUE_HARMONY_NOSAMP,
)
log("dual harmony GPU")
a = harmonize_xglue(a, batch_col=cfg['batch_col'], sample_col=cfg['sample_col'],
                    use_gpu=True, max_iter=50, random_state=42, verbose=True)

cluster_emb_key = XGLUE_HARMONY_KEY
cmd_emb_key     = XGLUE_HARMONY_NOSAMP
log("cell typing on X_glue_harmony")
a = cell_typing_v2(a, cluster_key=cluster_emb_key, modality_col=cfg['modality_col'])

from sample_embedding import compute_sample_embedding
from parameter_selection.autotune import run_autotune

# Heart default-α SE (overwrite — quick, ensures consistency)
out_default = os.path.join(out_dir, "sampledisco_default_v2")
os.makedirs(out_default, exist_ok=True)
log(f"default-α SE → {out_default}")
compute_sample_embedding(
    a, out_default,
    sample_col=cfg['sample_col'], celltype_col='cell_type',
    cluster_emb_key=cluster_emb_key, cmd_emb_key=cmd_emb_key,
    modality_col=cfg['modality_col'], batch_col=cfg['batch_col'],
    save=True, verbose=True,
)

# Heart autotuned SE — grouping_col=None (unsupervised proxies only)
out_tuned = os.path.join(out_dir, "sampledisco_tuned_v2")
os.makedirs(out_tuned, exist_ok=True)
log(f"autotuned SE (grouping_col=None) → {out_tuned}")
run_autotune(
    a, out_tuned,
    sample_col=cfg['sample_col'], celltype_col='cell_type',
    cluster_emb_key=cluster_emb_key, cmd_emb_key=cmd_emb_key,
    modality_col=cfg['modality_col'], batch_col=cfg['batch_col'],
    grouping_col=None,                # FIX: skip CCA (disease_state is categorical)
    save=True, verbose=True,
)

# Benchmark heart
se_result = {
    "name": "heart",
    "K_c": int(a.obs['cell_type'].nunique()),
    "default_csv": f"{out_default}/sample_embedding/sample_embedding.csv",
    "tuned_csv":   f"{out_tuned}/sample_embedding/sample_embedding.csv",
    "timings": {},
}
del a; gc.collect(); torch.cuda.empty_cache()

run_bench_one_dataset('heart', cfg, se_result, TEST_ROOT)
log("HEART DONE")
