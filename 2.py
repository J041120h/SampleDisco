import os
import re
import json
import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse
from typing import Optional, Dict, Any, List, Tuple

# ---------------------------------------------------------------------
# Helper for suffix stripping (e.g. "GAPDH.1" -> "GAPDH")
# ---------------------------------------------------------------------

_SUFFIX_RE = re.compile(r"\.\d+$")
_ENSEMBL_RE = re.compile(r"^ENSG\d+(?:\.\d+)?$", re.IGNORECASE)
_ENSEMBL_VERSION_RE = re.compile(r"^(ENSG\d+)(?:\.\d+)?$", re.IGNORECASE)


def _looks_like_ensembl(x: str) -> bool:
    """Return True if x resembles an Ensembl gene ID (e.g., ENSG00000141510 or ENSG00000141510.12)."""
    try:
        return bool(_ENSEMBL_RE.match(x or ""))
    except Exception:
        return False


def _strip_suffix(name: str) -> str:
    """Strip numeric suffix like '.1', '.2' from gene names."""
    if name is None:
        return name
    return _SUFFIX_RE.sub("", str(name))


def _strip_ens_version(name: str) -> str:
    """Strip Ensembl version suffix (e.g., ENSG00000141510.12 -> ENSG00000141510)."""
    if name is None:
        return name
    match = _ENSEMBL_VERSION_RE.match(str(name))
    if match:
        return match.group(1)
    return str(name)


def load_ensembl_mapping(mapping_csv: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load Ensembl ID <-> gene symbol mapping from CSV.
    
    Expected columns: 'ensembl_id', 'gene_symbol' (or 'gene_name')
    """
    if mapping_csv is None or not os.path.exists(mapping_csv):
        return None
    
    df = pd.read_csv(mapping_csv)
    
    # Normalize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Try to identify the right columns
    ensembl_col = None
    symbol_col = None
    
    for col in df.columns:
        if "ensembl" in col or col == "gene_id":
            ensembl_col = col
        elif "symbol" in col or col == "gene_name" or col == "name":
            symbol_col = col
    
    if ensembl_col and symbol_col:
        return df[[ensembl_col, symbol_col]].rename(
            columns={ensembl_col: "ensembl_id", symbol_col: "gene_symbol"}
        ).dropna()
    
    return None


def unify_and_align_genes(
    adata_rna: sc.AnnData,
    adata_atac: sc.AnnData,
    output_dir: str,
    prefer: str = "auto",
    mapping_csv: Optional[str] = None,
    atac_layer: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[sc.AnnData, sc.AnnData, List[str], pd.DataFrame]:
    """
    Unify gene names between RNA and ATAC data and align to shared genes.
    
    Parameters
    ----------
    adata_rna : AnnData
        RNA expression data.
    adata_atac : AnnData
        ATAC gene activity data.
    output_dir : str
        Directory to save mapping information.
    prefer : str
        Strategy for unification: 'auto', 'ensembl', or 'symbol'.
        - 'auto': detect based on majority naming convention
        - 'ensembl': convert all to Ensembl IDs
        - 'symbol': convert all to gene symbols
    mapping_csv : str or None
        Path to Ensembl <-> symbol mapping CSV.
    atac_layer : str or None
        If provided and present, use this layer as X.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    Tuple of (rna_unified, atac_unified, shared_genes, mapping_df)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle ATAC layer
    if atac_layer and atac_layer in adata_atac.layers:
        atac = sc.AnnData(
            X=adata_atac.layers[atac_layer],
            obs=adata_atac.obs.copy(),
            var=adata_atac.var.copy(),
        )
    else:
        atac = adata_atac.copy()
    
    rna = adata_rna.copy()
    
    # Get gene names
    rna_genes = pd.Index([str(g) for g in rna.var_names])
    atac_genes = pd.Index([str(g) for g in atac.var_names])
    
    # Detect naming conventions
    rna_ensembl_frac = sum(_looks_like_ensembl(g) for g in rna_genes) / len(rna_genes)
    atac_ensembl_frac = sum(_looks_like_ensembl(g) for g in atac_genes) / len(atac_genes)
    
    if verbose:
        print(f"  RNA Ensembl fraction: {rna_ensembl_frac:.1%}")
        print(f"  ATAC Ensembl fraction: {atac_ensembl_frac:.1%}")
    
    # Determine unification strategy
    if prefer == "auto":
        # If both are mostly symbols or both are mostly Ensembl, just strip suffixes
        if (rna_ensembl_frac < 0.5 and atac_ensembl_frac < 0.5) or \
           (rna_ensembl_frac > 0.5 and atac_ensembl_frac > 0.5):
            prefer = "strip_suffix"
        elif rna_ensembl_frac > 0.5:
            prefer = "ensembl"
        else:
            prefer = "symbol"
        
        if verbose:
            print(f"  Auto-detected strategy: {prefer}")
    
    # Build mapping dataframe
    mapping_records = []
    
    # Load external mapping if provided
    external_mapping = load_ensembl_mapping(mapping_csv)
    ensembl_to_symbol = {}
    symbol_to_ensembl = {}
    
    if external_mapping is not None:
        for _, row in external_mapping.iterrows():
            ens_id = _strip_ens_version(row["ensembl_id"])
            sym = row["gene_symbol"]
            ensembl_to_symbol[ens_id] = sym
            symbol_to_ensembl[sym.upper()] = ens_id
        if verbose:
            print(f"  Loaded {len(ensembl_to_symbol)} mappings from {mapping_csv}")
    
    def _unify_name(name: str) -> str:
        """Unify a single gene name based on the chosen strategy."""
        name = str(name)
        
        if prefer == "strip_suffix":
            # Just strip version/suffix
            if _looks_like_ensembl(name):
                return _strip_ens_version(name)
            else:
                return _strip_suffix(name)
        
        elif prefer == "ensembl":
            if _looks_like_ensembl(name):
                return _strip_ens_version(name)
            else:
                # Try to convert symbol to Ensembl
                return symbol_to_ensembl.get(name.upper(), _strip_suffix(name))
        
        elif prefer == "symbol":
            if _looks_like_ensembl(name):
                ens_stripped = _strip_ens_version(name)
                return ensembl_to_symbol.get(ens_stripped, ens_stripped)
            else:
                return _strip_suffix(name)
        
        else:
            return _strip_suffix(name)
    
    # Unify RNA gene names
    rna_unified_names = [_unify_name(g) for g in rna_genes]
    rna_name_map = dict(zip(rna_genes, rna_unified_names))
    
    # Unify ATAC gene names
    atac_unified_names = [_unify_name(g) for g in atac_genes]
    atac_name_map = dict(zip(atac_genes, atac_unified_names))
    
    # Handle duplicates by keeping first occurrence
    def _deduplicate_adata(adata: sc.AnnData, unified_names: List[str], verbose: bool = True) -> sc.AnnData:
        """Remove duplicate genes, keeping first occurrence."""
        seen = set()
        keep_idx = []
        for i, name in enumerate(unified_names):
            if name not in seen:
                seen.add(name)
                keep_idx.append(i)
        
        n_dups = len(unified_names) - len(keep_idx)
        if verbose and n_dups > 0:
            print(f"    Removing {n_dups} duplicate genes after unification")
        
        adata_dedup = adata[:, keep_idx].copy()
        adata_dedup.var_names = pd.Index([unified_names[i] for i in keep_idx])
        return adata_dedup
    
    rna_unified = _deduplicate_adata(rna, rna_unified_names, verbose)
    atac_unified = _deduplicate_adata(atac, atac_unified_names, verbose)
    
    # Find shared genes
    shared_genes = sorted(set(rna_unified.var_names) & set(atac_unified.var_names))
    
    if verbose:
        print(f"  Shared genes after unification: {len(shared_genes)}")
    
    # Create mapping dataframe
    for orig, unified in rna_name_map.items():
        mapping_records.append({
            "original_name": orig,
            "unified_name": unified,
            "source": "RNA",
            "in_shared": unified in shared_genes,
        })
    
    for orig, unified in atac_name_map.items():
        mapping_records.append({
            "original_name": orig,
            "unified_name": unified,
            "source": "ATAC",
            "in_shared": unified in shared_genes,
        })
    
    mapping_df = pd.DataFrame(mapping_records)
    
    # Save mapping
    mapping_path = os.path.join(output_dir, "gene_name_unification_mapping.csv")
    mapping_df.to_csv(mapping_path, index=False)
    
    if verbose:
        print(f"  Saved mapping to: {mapping_path}")
    
    return rna_unified, atac_unified, shared_genes, mapping_df


def debug_gene_overlap(
    adata_rna: sc.AnnData,
    adata_atac: sc.AnnData,
    output_dir: str,
    atac_layer: Optional[str] = "GeneActivity",
    try_unify: bool = True,
    unify_prefer: str = "auto",
    unify_mapping_csv: Optional[str] = None,
    max_examples: int = 50,
    verbose: bool = True,
) -> dict:
    """
    Debug RNA vs ATAC gene-name overlap and naming inconsistencies.

    Parameters
    ----------
    adata_rna : AnnData
        RNA expression data (already annotated).
    adata_atac : AnnData
        ATAC gene activity data (from Signac/GeneActivity).
    output_dir : str
        Where to save CSVs and JSON summary.
    atac_layer : str or None
        If not None and present in adata_atac.layers, use that as X.
    try_unify : bool
        If True, attempt unify_and_align_genes and report improved overlap.
    unify_prefer : {'auto', 'ensembl', 'symbol'}
        Strategy for unify_and_align_genes.
    unify_mapping_csv : str or None
        Optional Ensembl <-> symbol mapping CSV for unify_and_align_genes.
    max_examples : int
        Max number of example genes to save per category.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict with keys:
        - basic_stats
        - direct_overlap
        - suffix_overlap
        - unify_overlap (if try_unify=True)
        - paths (where CSV/JSON were saved)
    """
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print("DEBUG: RNA vs ATAC gene-name overlap")
        print("=" * 70)

    # --------------------------------------------------------------
    # 0. Handle ATAC layer if requested
    # --------------------------------------------------------------
    atac = adata_atac
    if atac_layer and hasattr(adata_atac, "layers") and atac_layer in adata_atac.layers:
        if verbose:
            print(f"[0] Using ATAC layer '{atac_layer}' as X for overlap diagnostics")
        atac = sc.AnnData(
            X=adata_atac.layers[atac_layer],
            obs=adata_atac.obs.copy(),
            var=adata_atac.var.copy(),
        )

    # --------------------------------------------------------------
    # 1. Basic stats
    # --------------------------------------------------------------
    rna_cells = adata_rna.n_obs
    atac_cells = atac.n_obs
    rna_genes = adata_rna.n_vars
    atac_genes = atac.n_vars

    rna_names = pd.Index([str(g) for g in adata_rna.var_names])
    atac_names = pd.Index([str(g) for g in atac.var_names])

    if verbose:
        print(f"[1] Basic stats:")
        print(f"    RNA:  {rna_cells} cells × {rna_genes} genes")
        print(f"    ATAC: {atac_cells} cells × {atac_genes} genes")

    # --------------------------------------------------------------
    # 2. Direct overlap
    # --------------------------------------------------------------
    direct_shared = sorted(set(rna_names) & set(atac_names))
    direct_only_rna = sorted(set(rna_names) - set(atac_names))
    direct_only_atac = sorted(set(atac_names) - set(rna_names))

    direct_stats = {
        "rna_total_genes": int(rna_genes),
        "atac_total_genes": int(atac_genes),
        "direct_shared_genes": len(direct_shared),
        "direct_shared_pct_rna": round(len(direct_shared) / rna_genes * 100, 2) if rna_genes else 0.0,
        "direct_shared_pct_atac": round(len(direct_shared) / atac_genes * 100, 2) if atac_genes else 0.0,
    }

    if verbose:
        print("\n[2] Direct name-based overlap:")
        print(f"    Shared genes (exact string match): {direct_stats['direct_shared_genes']}")
        print(f"    → {direct_stats['direct_shared_pct_rna']:.1f}% of RNA")
        print(f"    → {direct_stats['direct_shared_pct_atac']:.1f}% of ATAC")

    # --------------------------------------------------------------
    # 3. Name pattern diagnostics (Ensembl vs symbol, suffixes)
    # --------------------------------------------------------------
    def _pattern_summary(names: pd.Index, label: str) -> dict:
        as_series = pd.Series(list(names), index=range(len(names)))
        is_ens = as_series.apply(lambda x: _looks_like_ensembl(str(x)))
        has_suffix = as_series.astype(str).str.contains(_SUFFIX_RE, na=False)

        # Count Ensembl IDs with version numbers
        ens_with_version = as_series[is_ens].apply(
            lambda x: "." in str(x) and _looks_like_ensembl(str(x))
        ).sum()

        stats = {
            "n_total": int(len(names)),
            "n_ensembl_like": int(is_ens.sum()),
            "n_non_ensembl": int((~is_ens).sum()),
            "n_ensembl_with_version": int(ens_with_version),
            "n_suffix_like": int(has_suffix.sum()),
            "example_ensembl_names": as_series[is_ens].head(5).tolist(),
            "example_symbol_names": as_series[~is_ens].head(5).tolist(),
            "example_suffix_names": as_series[has_suffix].head(10).tolist(),
        }

        if verbose:
            print(f"\n[3] Name patterns for {label}:")
            print(f"    Total:              {stats['n_total']}")
            print(f"    Ensembl-like:       {stats['n_ensembl_like']}")
            print(f"    Non-Ensembl:        {stats['n_non_ensembl']}")
            print(f"    Ensembl w/ version: {stats['n_ensembl_with_version']}")
            print(f"    With '.<digit>' suffix: {stats['n_suffix_like']}")
            
            if stats["example_ensembl_names"]:
                print("    Example Ensembl names:")
                for x in stats["example_ensembl_names"][:5]:
                    print(f"      - {x}")
            
            if stats["example_symbol_names"]:
                print("    Example symbol names:")
                for x in stats["example_symbol_names"][:5]:
                    print(f"      - {x}")
            
            if stats["example_suffix_names"]:
                print("    Example suffix names (up to 10):")
                for x in stats["example_suffix_names"]:
                    print(f"      - {x}")

        return stats

    rna_pattern_stats = _pattern_summary(rna_names, "RNA")
    atac_pattern_stats = _pattern_summary(atac_names, "ATAC")

    # --------------------------------------------------------------
    # 4. Overlap after stripping simple suffixes (e.g. '.1')
    # --------------------------------------------------------------
    rna_no_suffix = pd.Index([_strip_suffix(n) for n in rna_names])
    atac_no_suffix = pd.Index([_strip_suffix(n) for n in atac_names])

    shared_no_suffix = sorted(set(rna_no_suffix) & set(atac_no_suffix))

    suffix_stats = {
        "shared_genes_no_suffix": len(shared_no_suffix),
        "shared_pct_rna_no_suffix": round(len(shared_no_suffix) / rna_genes * 100, 2) if rna_genes else 0.0,
        "shared_pct_atac_no_suffix": round(len(shared_no_suffix) / atac_genes * 100, 2) if atac_genes else 0.0,
    }

    if verbose:
        print("\n[4] Overlap after stripping '.<digit>' suffixes:")
        print(f"    Shared (suffix-stripped): {suffix_stats['shared_genes_no_suffix']}")
        print(f"    → {suffix_stats['shared_pct_rna_no_suffix']:.1f}% of RNA")
        print(f"    → {suffix_stats['shared_pct_atac_no_suffix']:.1f}% of ATAC")

    # Save example genes where suffix is likely the only difference
    suffix_match_examples = []
    rna_map_ns = {}
    atac_map_ns = {}

    for n in rna_names:
        ns = _strip_suffix(n)
        rna_map_ns.setdefault(ns, []).append(n)
    for n in atac_names:
        ns = _strip_suffix(n)
        atac_map_ns.setdefault(ns, []).append(n)

    for ns in shared_no_suffix:
        r_list = rna_map_ns.get(ns, [])
        a_list = atac_map_ns.get(ns, [])
        if (len(r_list) > 0) and (len(a_list) > 0):
            if any(r != a for r in r_list for a in a_list):
                suffix_match_examples.append({
                    "base_name": ns,
                    "rna_names": ";".join(sorted(set(r_list))[:5]),
                    "atac_names": ";".join(sorted(set(a_list))[:5]),
                })
        if len(suffix_match_examples) >= max_examples:
            break

    suffix_examples_df = pd.DataFrame(suffix_match_examples)
    suffix_examples_path = os.path.join(output_dir, "gene_suffix_mismatch_examples.csv")
    suffix_examples_df.to_csv(suffix_examples_path, index=False)

    if verbose and len(suffix_match_examples) > 0:
        print(f"    Saved suffix-mismatch examples → {suffix_examples_path}")

    # --------------------------------------------------------------
    # 5. Also try stripping Ensembl versions
    # --------------------------------------------------------------
    def _normalize_gene_name(name: str) -> str:
        """Strip both Ensembl version and regular suffix."""
        name = str(name)
        if _looks_like_ensembl(name):
            return _strip_ens_version(name)
        return _strip_suffix(name)

    rna_normalized = pd.Index([_normalize_gene_name(n) for n in rna_names])
    atac_normalized = pd.Index([_normalize_gene_name(n) for n in atac_names])

    shared_normalized = sorted(set(rna_normalized) & set(atac_normalized))

    normalized_stats = {
        "shared_genes_normalized": len(shared_normalized),
        "shared_pct_rna_normalized": round(len(shared_normalized) / rna_genes * 100, 2) if rna_genes else 0.0,
        "shared_pct_atac_normalized": round(len(shared_normalized) / atac_genes * 100, 2) if atac_genes else 0.0,
    }

    if verbose:
        print("\n[5] Overlap after normalizing (strip suffix + Ensembl version):")
        print(f"    Shared (normalized): {normalized_stats['shared_genes_normalized']}")
        print(f"    → {normalized_stats['shared_pct_rna_normalized']:.1f}% of RNA")
        print(f"    → {normalized_stats['shared_pct_atac_normalized']:.1f}% of ATAC")

    # --------------------------------------------------------------
    # 6. Try unify_and_align_genes (if requested)
    # --------------------------------------------------------------
    unify_stats = None
    if try_unify:
        try:
            if verbose:
                print("\n[6] Trying unify_and_align_genes() to harmonize IDs...")

            rna_unified, atac_unified, shared_unified, mapping_df = unify_and_align_genes(
                adata_rna=adata_rna,
                adata_atac=atac,
                output_dir=output_dir,
                prefer=unify_prefer,
                mapping_csv=unify_mapping_csv,
                atac_layer=None,  # layer already handled above
                verbose=verbose,
            )

            unify_stats = {
                "shared_genes_after_unification": len(shared_unified),
                "shared_pct_rna_after_unification": round(len(shared_unified) / rna_genes * 100, 2) if rna_genes else 0.0,
                "shared_pct_atac_after_unification": round(len(shared_unified) / atac_genes * 100, 2) if atac_genes else 0.0,
                "rna_genes_after_dedup": rna_unified.n_vars,
                "atac_genes_after_dedup": atac_unified.n_vars,
            }

            if verbose:
                print("    unify_and_align_genes() completed.")
                print(f"    Shared genes after unification: {unify_stats['shared_genes_after_unification']}")
                print(f"    → {unify_stats['shared_pct_rna_after_unification']:.1f}% of RNA")
                print(f"    → {unify_stats['shared_pct_atac_after_unification']:.1f}% of ATAC")

        except Exception as e:
            import traceback
            unify_stats = {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            if verbose:
                print(f"    [WARN] unify_and_align_genes failed: {e}")

    # --------------------------------------------------------------
    # 7. Save example lists of only-RNA / only-ATAC genes
    # --------------------------------------------------------------
    rna_only_df = pd.DataFrame({"gene": direct_only_rna[:max_examples]})
    atac_only_df = pd.DataFrame({"gene": direct_only_atac[:max_examples]})

    rna_only_path = os.path.join(output_dir, "genes_only_in_rna_example.csv")
    atac_only_path = os.path.join(output_dir, "genes_only_in_atac_example.csv")

    rna_only_df.to_csv(rna_only_path, index=False)
    atac_only_df.to_csv(atac_only_path, index=False)

    # Also save the shared genes list
    shared_genes_path = os.path.join(output_dir, "shared_genes.csv")
    pd.DataFrame({"gene": direct_shared}).to_csv(shared_genes_path, index=False)

    if verbose:
        print("\n[7] Saved example gene lists:")
        print(f"    RNA-only genes (up to {max_examples}) → {rna_only_path}")
        print(f"    ATAC-only genes (up to {max_examples}) → {atac_only_path}")
        print(f"    Shared genes (all {len(direct_shared)}) → {shared_genes_path}")

    # --------------------------------------------------------------
    # 8. Bundle summary + save JSON
    # --------------------------------------------------------------
    summary = {
        "basic_stats": {
            "rna_cells": int(rna_cells),
            "atac_cells": int(atac_cells),
            "rna_genes": int(rna_genes),
            "atac_genes": int(atac_genes),
        },
        "direct_overlap": direct_stats,
        "rna_name_patterns": rna_pattern_stats,
        "atac_name_patterns": atac_pattern_stats,
        "suffix_overlap": suffix_stats,
        "normalized_overlap": normalized_stats,
        "unify_overlap": unify_stats,
        "paths": {
            "rna_only_examples_csv": rna_only_path,
            "atac_only_examples_csv": atac_only_path,
            "suffix_examples_csv": suffix_examples_path,
            "shared_genes_csv": shared_genes_path,
        },
    }

    summary_path = os.path.join(output_dir, "gene_overlap_debug_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print("\n[8] Debug summary saved to:", summary_path)
        print("\nDone.\n")

    return summary


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    rna_path = "/dcs07/hongkai/data/harry/result/multi_omics_heart/data/rna_raw.h5ad"
    atac_path = "/dcs07/hongkai/data/harry/result/multi_omics_heart/data/heart_gene_activity.h5ad"

    print("Loading RNA data...")
    adata_rna = sc.read_h5ad(rna_path)
    
    print("Loading ATAC data...")
    adata_atac = sc.read_h5ad(atac_path)

    debug_out = debug_gene_overlap(
        adata_rna=adata_rna,
        adata_atac=adata_atac,
        output_dir="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/gene_overlap_debug",
        atac_layer="GeneActivity",
        try_unify=True,
        unify_prefer="auto",
        unify_mapping_csv=None,  # or path to mapping CSV if you have one
        verbose=True,
    )
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Direct overlap: {debug_out['direct_overlap']['direct_shared_genes']} genes")
    print(f"After suffix stripping: {debug_out['suffix_overlap']['shared_genes_no_suffix']} genes")
    print(f"After normalization: {debug_out['normalized_overlap']['shared_genes_normalized']} genes")
    if debug_out['unify_overlap'] and 'error' not in debug_out['unify_overlap']:
        print(f"After unification: {debug_out['unify_overlap']['shared_genes_after_unification']} genes")
        
import scanpy as sc

rna_path = "/dcs07/hongkai/data/harry/result/multi_omics_heart/data/rna_raw.h5ad"
atac_path = "/dcs07/hongkai/data/harry/result/multi_omics_heart/data/heart_gene_activity.h5ad"

adata_rna = sc.read_h5ad(rna_path)
adata_atac = sc.read_h5ad(atac_path)

debug_out = debug_gene_overlap(
    adata_rna=adata_rna,
    adata_atac=adata_atac,
    output_dir="/dcs07/hongkai/data/harry/result/multi_omics_heart/data/gene_overlap_debug",
    atac_layer="GeneActivity",        # or None if you already set X to GeneActivity
    try_unify=True,                   # will use your unify_and_align_genes()
    unify_prefer="auto",
    unify_mapping_csv=None,           # or path to mapping CSV if you have one
    verbose=True,
)
