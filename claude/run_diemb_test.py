"""Test-run launcher for the di-embedding (Mode B) pipeline.

Loads config_unpaired.yaml, filters to keys the current wrapper accepts,
and invokes wrapper(...). Used because the user's config has historical
keys that SampleDisc.validate_config rejects. Production cleanup of those
stale keys is out-of-scope for this throughput-tuning test.
"""
from __future__ import annotations
import sys, yaml, inspect
sys.path.insert(0, "/users/hjiang/GenoDistance/code")
from wrapper.wrapper import wrapper

CFG = "/users/hjiang/GenoDistance/code/config/config_unpaired.yaml"

cfg = yaml.safe_load(open(CFG))
allowed = set(inspect.signature(wrapper).parameters)
unknown = sorted(k for k in cfg if k not in allowed)
filtered = {k: v for k, v in cfg.items() if k in allowed}

print(f"[run_diemb_test] config keys:    {len(cfg)}")
print(f"[run_diemb_test] wrapper params: {len(allowed)}")
print(f"[run_diemb_test] dropped (unknown to current wrapper): {len(unknown)}")
for k in unknown:
    print(f"    - {k}")
print(f"[run_diemb_test] passing {len(filtered)} kwargs to wrapper(...)")

wrapper(**filtered)
