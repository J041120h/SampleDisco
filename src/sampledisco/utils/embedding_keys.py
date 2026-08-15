"""Canonical cell-level embedding keys + legacy-name resolution.

Paper (sn-article.tex, Methods):
  z^comp — sample-REMOVED   → composition blocks A1/A2/A3, anchors, cell typing
  z^RMD  — sample-PRESERVED → RMD displacement block

Three generations of obsm names exist on disk. Reads go through the resolvers
below so every generation keeps working; writes emit the canonical name (plus,
until 1.0, the 0.2.0 legacy alias so new outputs stay readable by 0.2.0).
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

COMP_KEY = "Z_comp"
RMD_KEY = "Z_rmd"
XGLUE_KEY = "X_glue"

# Highest priority first; the second element is the version that wrote the key
# (None = canonical, no deprecation warning).
COMP_ALIASES: Tuple[Tuple[str, Optional[str]], ...] = (
    ("Z_comp", None),             # canonical (0.3.0+)
    ("Z_clust", "0.2.0"),
    ("X_pca_harmony", "pre-0.2"),   # RNA
    ("X_lsi_harmony", "pre-0.2"),   # ATAC
    ("X_glue_harmony", "pre-0.2"),  # multi-omics
)
RMD_ALIASES: Tuple[Tuple[str, Optional[str]], ...] = (
    ("Z_rmd", None),                       # canonical
    ("Z_cmd", "pre-0.2"),
    ("X_pca_harmony_nosamp", "pre-0.2"),   # RNA
    ("X_lsi_harmony_nosamp", "pre-0.2"),   # ATAC
)

# 0.3.0 also writes Z_clust so its outputs stay readable by installed 0.2.0.
# Remove in 1.0.
WRITE_LEGACY_ALIAS = True
LEGACY_COMP_KEY = "Z_clust"

_warned: set = set()


def _warn_once(key: str, context: str, msg: str, category) -> None:
    tag = (key, context)
    if tag in _warned:
        return
    _warned.add(tag)
    warnings.warn(msg, category, stacklevel=3)


def _deprecation_msg(old: str, canonical: str, version: str, context: str) -> str:
    where = f"[{context}] " if context else ""
    return (
        f"[sampledisco] {where}.obsm['{old}'] is a legacy key (written by "
        f"sampledisco {version}); the canonical name is '{canonical}'. Reading it "
        f"still works and will be removed in 1.0. To migrate in place: "
        f"from sampledisco.utils.embedding_keys import migrate_obsm_keys; "
        f"migrate_obsm_keys(adata)."
    )


def _check_override(adata, override: str, role: str) -> str:
    if override not in adata.obsm:
        raise KeyError(
            f"{role} embedding key {override!r} not in adata.obsm "
            f"(available: {list(adata.obsm.keys())})")
    return override


def _scan(adata, aliases, canonical: str, context: str) -> Optional[str]:
    for key, version in aliases:
        if key in adata.obsm:
            if version is not None:
                _warn_once(key, context,
                           _deprecation_msg(key, canonical, version, context),
                           FutureWarning)
            return key
    return None


def resolve_comp_key(adata, override: Optional[str] = None, *,
                     fallbacks: Sequence[str] = (), required: bool = True,
                     context: str = "") -> Optional[str]:
    """Resolve the sample-REMOVED (composition / anchor / clustering) obsm key."""
    if override:
        return _check_override(adata, override, "comp")
    key = _scan(adata, COMP_ALIASES, COMP_KEY, context)
    if key is not None:
        return key
    for key in fallbacks:
        if key in adata.obsm:
            return key
    if required:
        raise KeyError(
            f"No composition (sample-removed) embedding found in adata.obsm. "
            f"Looked for {[k for k, _ in COMP_ALIASES] + list(fallbacks)}; "
            f"available: {list(adata.obsm.keys())}. Run preprocessing first.")
    return None


def resolve_rmd_key(adata, override: Optional[str] = None, *,
                    comp_key: Optional[str] = None, fallbacks: Sequence[str] = (),
                    required: bool = True, context: str = "") -> Optional[str]:
    """Resolve the sample-PRESERVED (RMD displacement) obsm key.

    When no sample-preserved embedding exists, falls back to ``comp_key`` with a
    loud RuntimeWarning — the resulting RMD block is near-degenerate.
    """
    if override:
        return _check_override(adata, override, "rmd")
    key = _scan(adata, RMD_ALIASES, RMD_KEY, context)
    if key is not None:
        return key
    for key in fallbacks:
        if key in adata.obsm:
            return key
    if not required:
        return None
    if comp_key is None:
        raise KeyError(
            f"No sample-preserved (RMD) embedding found in adata.obsm. "
            f"Looked for {[k for k, _ in RMD_ALIASES] + list(fallbacks)}; "
            f"available: {list(adata.obsm.keys())}.")
    _warn_once("__degraded__", context, (
        f"[sampledisco] No sample-PRESERVED embedding found in .obsm (looked for "
        f"{', '.join(k for k, _ in RMD_ALIASES)}). Falling back to the composition "
        f"key '{comp_key}', which is sample-REMOVED — the RMD displacement block "
        f"will be near-degenerate and the resulting sample embedding is NOT the "
        f"method described in the paper. Re-run preprocessing, or pass use_rmd=False."
    ), RuntimeWarning)
    return comp_key


def resolve_embedding_keys(adata, comp_override: Optional[str] = None,
                           rmd_override: Optional[str] = None, *,
                           rmd_required: bool = True,
                           rmd_fallbacks: Sequence[str] = (),
                           comp_fallbacks: Sequence[str] = (),
                           context: str = "") -> Tuple[str, Optional[str]]:
    """Resolve both cell-level embedding keys in one call."""
    comp = resolve_comp_key(adata, comp_override, fallbacks=comp_fallbacks,
                            context=context)
    rmd = resolve_rmd_key(adata, rmd_override, comp_key=comp,
                          fallbacks=rmd_fallbacks, required=rmd_required,
                          context=context)
    return comp, rmd


_MIGRATIONS: Tuple[Tuple[str, str], ...] = (
    ("Z_clust", COMP_KEY),
    ("X_pca_harmony", COMP_KEY),
    ("X_lsi_harmony", COMP_KEY),
    ("X_glue_harmony", COMP_KEY),
    ("Z_cmd", RMD_KEY),
    ("X_pca_harmony_nosamp", RMD_KEY),
    ("X_lsi_harmony_nosamp", RMD_KEY),
)


def migrate_obsm_keys(adata, *, verbose: bool = True) -> List[Tuple[str, str]]:
    """Copy legacy obsm embeddings onto their canonical names, in memory only.

    Non-destructive: legacy keys are kept, and a canonical key that already
    exists is never overwritten. Writes no files.
    """
    applied = []
    for old, new in _MIGRATIONS:
        if old in adata.obsm and new not in adata.obsm:
            adata.obsm[new] = adata.obsm[old]
            applied.append((old, new))
            if verbose:
                print(f"[sampledisco] migrated obsm['{old}'] → obsm['{new}']")
    return applied


def write_comp_key(adata, value) -> None:
    """Write the canonical composition embedding (+ the 0.2.0 legacy alias)."""
    adata.obsm[COMP_KEY] = value
    if WRITE_LEGACY_ALIAS:
        adata.obsm[LEGACY_COMP_KEY] = adata.obsm[COMP_KEY]


__all__ = [
    "COMP_KEY", "RMD_KEY", "XGLUE_KEY", "LEGACY_COMP_KEY", "WRITE_LEGACY_ALIAS",
    "COMP_ALIASES", "RMD_ALIASES",
    "resolve_comp_key", "resolve_rmd_key", "resolve_embedding_keys",
    "migrate_obsm_keys", "write_comp_key",
]
