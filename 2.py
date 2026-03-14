# Check whether test_ATAC.h5ad is compressed / has suspicious hidden payloads.
# If the file appears uncompressed (or if force_rewrite=True), rewrite it with gzip
# and atomically overwrite the original file.
#
# This does NOT intentionally modify AnnData contents; it just rewrites the file.
# To be safe, it first writes a temporary compressed copy, verifies it can be read,
# then renames the original to a backup and replaces it.

from pathlib import Path
import os
import shutil
import h5py
import anndata as ad

# =========================
# USER CONFIG
# =========================
h5ad_path = Path("/dcl01/hongkai/data/data/hjiang/Data/test_ATAC.h5ad")
compression = "gzip"          # use gzip for .h5ad rewrite
force_rewrite = False         # if True, rewrite even if file already looks compressed
keep_backup = True            # keep original as .bak after successful overwrite
verbose_max_datasets = 200    # just for printing

# =========================
# HELPER FUNCTIONS
# =========================
def format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024

def inspect_h5ad_storage(path: Path, verbose: bool = True, max_print: int = 200):
    """
    Inspect HDF5 dataset-level compression/chunking and summarize what is stored.
    """
    summary = {
        "datasets": [],
        "compressed_count": 0,
        "uncompressed_count": 0,
        "total_datasets": 0,
        "groups_present": set(),
    }

    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                rec = {
                    "name": name,
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "compression": obj.compression,
                    "chunks": obj.chunks,
                    "nbytes": getattr(obj, "nbytes", None),
                }
                summary["datasets"].append(rec)
                summary["total_datasets"] += 1
                if obj.compression is None:
                    summary["uncompressed_count"] += 1
                else:
                    summary["compressed_count"] += 1
            elif isinstance(obj, h5py.Group):
                summary["groups_present"].add(name)

        f.visititems(visitor)

    # Sort biggest first if sizes available
    summary["datasets"].sort(
        key=lambda x: (-1 if x["nbytes"] is None else -x["nbytes"], x["name"])
    )

    if verbose:
        print("=" * 100)
        print(f"Inspecting HDF5 storage: {path}")
        print(f"On-disk file size: {format_bytes(path.stat().st_size)}")
        print(f"Total datasets: {summary['total_datasets']}")
        print(f"Compressed datasets: {summary['compressed_count']}")
        print(f"Uncompressed datasets: {summary['uncompressed_count']}")
        print(f"Top-level-ish groups seen: {sorted(g for g in summary['groups_present'] if '/' not in g)}")
        print()

        print("Largest datasets:")
        for rec in summary["datasets"][:max_print]:
            print(
                f"{rec['name'][:60]:60s} "
                f"shape={str(rec['shape']):20s} "
                f"dtype={rec['dtype']:10s} "
                f"compression={str(rec['compression']):8s} "
                f"chunks={str(rec['chunks'])[:18]:18s} "
                f"nbytes={format_bytes(rec['nbytes']) if rec['nbytes'] is not None else 'NA'}"
            )

    return summary

def ann_data_summary(path: Path):
    """
    Read the file normally and summarize AnnData-visible content.
    """
    a = ad.read_h5ad(path)
    out = {
        "shape": a.shape,
        "layers": list(a.layers.keys()),
        "obsm": list(a.obsm.keys()),
        "varm": list(a.varm.keys()),
        "obsp": list(a.obsp.keys()),
        "varp": list(a.varp.keys()),
        "uns_keys": list(a.uns.keys()),
        "has_raw": a.raw is not None,
        "X_type": type(a.X).__name__,
        "X_dtype": str(a.X.dtype),
    }
    del a
    return out

def print_anndata_summary(label: str, summary: dict):
    print("=" * 100)
    print(label)
    for k, v in summary.items():
        if k == "uns_keys" and len(v) > 30:
            print(f"{k}: {v[:30]} ... ({len(v)} total)")
        else:
            print(f"{k}: {v}")

def rewrite_with_compression_inplace(path: Path, compression: str = "gzip", keep_backup: bool = True):
    """
    Rewrite .h5ad using compression and atomically replace original.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    bak_path = path.with_suffix(path.suffix + ".bak")

    print("=" * 100)
    print("Reading original AnnData...")
    adata = ad.read_h5ad(path)

    print("Writing temporary compressed file...")
    adata.write_h5ad(tmp_path, compression=compression)

    print("Verifying temporary file can be opened...")
    test = ad.read_h5ad(tmp_path, backed="r")
    _ = test.shape
    del test
    del adata

    print("Swapping files...")
    if bak_path.exists():
        bak_path.unlink()

    os.replace(path, bak_path)
    os.replace(tmp_path, path)

    if not keep_backup:
        bak_path.unlink()

    return bak_path if keep_backup else None

# =========================
# STEP 1: Inspect current file
# =========================
if not h5ad_path.exists():
    raise FileNotFoundError(h5ad_path)

before_size = h5ad_path.stat().st_size
storage_before = inspect_h5ad_storage(h5ad_path, verbose=True, max_print=verbose_max_datasets)
adata_before = ann_data_summary(h5ad_path)
print_anndata_summary("AnnData-visible summary BEFORE rewrite", adata_before)

# Determine whether rewrite is needed
looks_uncompressed = storage_before["compressed_count"] == 0
has_some_uncompressed = storage_before["uncompressed_count"] > 0

print("=" * 100)
print(f"looks_uncompressed = {looks_uncompressed}")
print(f"has_some_uncompressed = {has_some_uncompressed}")
print(f"force_rewrite = {force_rewrite}")

need_rewrite = looks_uncompressed or force_rewrite

if not need_rewrite:
    print("\nFile already has at least some compression. No overwrite performed.")
    print("Set force_rewrite=True if you still want to rewrite it cleanly with gzip.")
else:
    # =========================
    # STEP 2: Rewrite with compression
    # =========================
    backup_path = rewrite_with_compression_inplace(
        h5ad_path,
        compression=compression,
        keep_backup=keep_backup,
    )

    # =========================
    # STEP 3: Re-inspect rewritten file
    # =========================
    after_size = h5ad_path.stat().st_size
    storage_after = inspect_h5ad_storage(h5ad_path, verbose=True, max_print=verbose_max_datasets)
    adata_after = ann_data_summary(h5ad_path)
    print_anndata_summary("AnnData-visible summary AFTER rewrite", adata_after)

    print("=" * 100)
    print("SIZE COMPARISON")
    print(f"Before: {format_bytes(before_size)}")
    print(f"After : {format_bytes(after_size)}")
    delta = after_size - before_size
    sign = "+" if delta >= 0 else ""
    print(f"Delta : {sign}{format_bytes(delta)}")

    print("=" * 100)
    print("INTEGRITY CHECK")
    same_summary = (
        adata_before["shape"] == adata_after["shape"]
        and adata_before["layers"] == adata_after["layers"]
        and adata_before["obsm"] == adata_after["obsm"]
        and adata_before["varm"] == adata_after["varm"]
        and adata_before["obsp"] == adata_after["obsp"]
        and adata_before["varp"] == adata_after["varp"]
        and adata_before["has_raw"] == adata_after["has_raw"]
        and adata_before["X_dtype"] == adata_after["X_dtype"]
    )
    print(f"AnnData structure preserved: {same_summary}")

    if keep_backup:
        print(f"Backup kept at: {backup_path}")