#!/usr/bin/env python3
# rename_cells_strip_to_ENC.py
import re
import sys
import os
import shutil
from datetime import datetime

import anndata as ad
import pandas as pd

def make_unique(names):
    """Make index values unique by appending .1, .2, ... to duplicates (pandas-style)."""
    idx = pd.Index(names)
    if not idx.has_duplicates:
        return idx.tolist()
    counts = {}
    out = []
    for n in idx:
        if n not in counts:
            counts[n] = 0
            out.append(n)
        else:
            counts[n] += 1
            out.append(f"{n}.{counts[n]}")
    return out

def strip_prefix_before_ENC(name: str) -> str:
    """
    Keep from the first occurrence of 'ENC' onward.
    If 'ENC' not found, keep original name.
    """
    m = re.search(r'ENC', name)
    return name[m.start():] if m else name

def main(h5ad_path: str):
    if not os.path.isfile(h5ad_path):
        raise FileNotFoundError(f"No such file: {h5ad_path}")

    # Backup
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = f"{h5ad_path}.bak-{ts}"
    shutil.copy2(h5ad_path, backup_path)
    print(f"[INFO] Backup created: {backup_path}")

    # Load
    adata = ad.read_h5ad(h5ad_path)
    old_names = adata.obs_names.tolist()

    # Transform
    new_names = [strip_prefix_before_ENC(n) for n in old_names]

    # Report potential no-ENC cases
    no_enc_count = sum(1 for n in old_names if 'ENC' not in n)
    if no_enc_count:
        print(f"[WARN] {no_enc_count} cell IDs did not contain 'ENC' and were left unchanged.")

    # Ensure uniqueness
    if len(set(new_names)) != len(new_names):
        print("[WARN] Duplicate IDs detected after stripping; making names unique with numeric suffixes.")
        new_names = make_unique(new_names)

    # Apply
    adata.obs_names = new_names

    # (Optional) If .raw is present, it indexes genes (var_names), so no change needed.
    # Save back in place
    adata.write(h5ad_path)
    print(f"[OK] Wrote updated AnnData in place: {h5ad_path}")

    # Show a few examples of the change
    print("\n[EXAMPLES] First 10 renames (old → new):")
    for old, new in list(zip(old_names, new_names))[:10]:
        print(f"  {old}  →  {new}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_cells_strip_to_ENC.py /path/to/file.h5ad")
        sys.exit(1)
    main(sys.argv[1])
