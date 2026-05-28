"""Tier-3 end-to-end test on /dcs07/.../result/test data.

Loads config.yaml, overrides flags so the wrapper:
  - reuses cached GLUE training (no rerun)
  - runs new merge + per-modality preprocess (build_embedding_union etc.)
  - runs cell typing on the new embedding union
  - runs SE + autotune
  - runs downstream sample distance + sample clustering
  - runs trajectory_DGE (verifies adata_cell_for_dge wiring)
  - runs cluster_DGE / RAISIN (verifies adata_sample_for_dge wiring)

Reference outputs are backed up at /dcs07/.../result/test_REFERENCE/.
"""
from __future__ import annotations
import sys, yaml, inspect, time, os
sys.path.insert(0, "/users/hjiang/GenoDistance/code")
from wrapper.wrapper import wrapper

CFG = "/users/hjiang/GenoDistance/code/config/config.yaml"
cfg = yaml.safe_load(open(CFG))

# Force the test to actually exercise the refactored pipeline end-to-end.
overrides = {
    # Re-run multiomics; cached GLUE training is reused via skip-if-exists.
    "multiomics_integration": True,
    "multiomics_run_glue_preprocessing": False,   # cached
    "multiomics_run_glue_training": False,        # cached
    # RESUME: merge / per-modality preprocess / cell typing already on disk
    # from the prior run (incl. propagated cell_type). Skip to SE + downstream.
    "multiomics_run_glue_merge": False,
    "multiomics_run_glue_preprocess_per_modality": False,
    "multiomics_cell_type_cluster": False,
    "multiomics_derive_sample_embedding": True,
    "multiomics_autotune_enable": True,
    "multiomics_autotune_grouping_col": "sev.level",
    # Downstream — exercise both DGE paths
    "multiomics_sample_distance_calculation": True,
    "multiomics_sample_cluster": True,
    "multiomics_trajectory_analysis": True,
    "multiomics_trajectory_dge": True,    # verifies adata_cell_for_dge (RNA only)
    "multiomics_cluster_dge": True,       # verifies adata_sample_for_dge (RAISIN on RNA pseudobulk)
    # CPU only: cuml/rapids broken on this env.
    "use_gpu": False,
    "multiomics_use_gpu": False,
}
cfg.update(overrides)

# Resume hint: clear so wrapper re-derives from output_dir.
cfg["multiomics_integrated_h5ad_path"] = None

allowed = set(inspect.signature(wrapper).parameters)
unknown = sorted(k for k in cfg if k not in allowed)
filtered = {k: v for k, v in cfg.items() if k in allowed}
print(f"[tier3] config keys:    {len(cfg)}")
print(f"[tier3] wrapper params: {len(allowed)}")
print(f"[tier3] dropped (unknown to wrapper): {len(unknown)}")
for k in unknown:
    print(f"    - {k}")
print(f"[tier3] overrides applied: {sorted(overrides)}")
print(f"[tier3] passing {len(filtered)} kwargs to wrapper(...)")
print()

t0 = time.time()
wrapper(**filtered)
print(f"\n[tier3] wrapper(...) returned after {time.time() - t0:.1f}s")
