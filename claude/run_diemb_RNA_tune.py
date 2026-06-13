"""Launcher for the RNA-only autotune variant of the unpaired_diemb run.

Same as claude/run_diemb_test.py but points at config_unpaired_RNA_tune.yaml
so the wrapper is called with multiomics_autotune_tune_on_modality='RNA'.
All upstream steps are cached on disk (GLUE training, gene_activity, cell
typing); only the autotune objective is rerun on RNA-only units. Output
lands in {output_dir}/multiomics/sample_embedding_tune-on-RNA/ to keep
the all-modality run intact.
"""
from __future__ import annotations
import sys, yaml, inspect
sys.path.insert(0, "/users/hjiang/GenoDistance/code")
from sampledisco.wrapper.wrapper import wrapper

CFG = "/users/hjiang/GenoDistance/code/config/config_unpaired_RNA_tune.yaml"

cfg = yaml.safe_load(open(CFG))
allowed = set(inspect.signature(wrapper).parameters)
unknown = sorted(k for k in cfg if k not in allowed)
filtered = {k: v for k, v in cfg.items() if k in allowed}

print(f"[run_diemb_RNA_tune] config keys:    {len(cfg)}")
print(f"[run_diemb_RNA_tune] wrapper params: {len(allowed)}")
print(f"[run_diemb_RNA_tune] dropped (unknown to current wrapper): {len(unknown)}")
for k in unknown:
    print(f"    - {k}")
print(f"[run_diemb_RNA_tune] tune_on_modality = "
      f"{filtered.get('multiomics_autotune_tune_on_modality', '(unset)')}")
print(f"[run_diemb_RNA_tune] passing {len(filtered)} kwargs to wrapper(...)")
print()

wrapper(**filtered)
