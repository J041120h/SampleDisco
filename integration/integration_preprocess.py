import os
import time
from pathlib import Path
from typing import Union, Optional

import pandas as pd
import scanpy as sc
from anndata import AnnData

from pseudo_adata import *
from DR import *
from preparation.Cell_type import *


def _store_original_sample_ids(
    adata: sc.AnnData,
    sample_column: str,
    original_sample_col: str,
    verbose: bool = True,
) -> None:
    if sample_column not in adata.obs.columns:
        raise KeyError(
            f"'{sample_column}' not found in adata.obs. Available columns: {list(adata.obs.columns)}"
        )

    if original_sample_col in adata.obs.columns:
        if verbose:
            print(
                f"'{original_sample_col}' already exists in .obs; leaving it unchanged."
            )
        return

    adata.obs[original_sample_col] = adata.obs[sample_column].astype(str)
    if verbose:
        print(f"Stored original sample IDs in adata.obs['{original_sample_col}'].")


def _maybe_append_modality_to_duplicates(
    adata: sc.AnnData,
    sample_column: str,
    modality_col: str,
    verbose: bool = True,
) -> None:
    if modality_col is None or modality_col not in adata.obs.columns:
        if verbose and modality_col is not None:
            print(f"'{modality_col}' not found in .obs; no modality suffix added.")
        return

    # Work on string views to detect duplicates safely
    s = adata.obs[sample_column].astype(str)
    dup_mask = s.duplicated(keep=False)

    if not dup_mask.any():
        if verbose:
            print(f"'{sample_column}' values are already unique; no modality suffix added.")
        return

    # 🔑 Ensure the target column is NOT categorical before assignment
    adata.obs[sample_column] = adata.obs[sample_column].astype(str)

    m = adata.obs[modality_col].astype(str)

    adata.obs.loc[dup_mask, sample_column] = (
        s[dup_mask] + "_" + m[dup_mask]
    )

    if verbose:
        n_dup_groups = s[dup_mask].nunique()
        n_dup_rows = int(dup_mask.sum())
        print(
            f"Detected non-unique '{sample_column}' values "
            f"({n_dup_groups} duplicated sample IDs across {n_dup_rows} rows). "
            f"Appended modality from '{modality_col}' only for those rows."
        )
    
    return adata


def multi_omics_merge_sample_meta(
    adata: sc.AnnData,
    sample_column: str = "sample",
    modality_col: str = "modality",
    rna_sample_meta_file: Optional[Union[str, Path]] = None,
    atac_sample_meta_file: Optional[Union[str, Path]] = None,
    original_sample_col: Optional[str] = None,
    verbose: bool = True,
) -> sc.AnnData:
    """
    Multi-omics aware sample metadata merge.

    - Handles RNA and ATAC separately based on `modality_col`.
    - Robust CSV reading (BOM stripping, auto-separator detection).
    - Deals with sample IDs that may or may not carry modality suffixes
      (e.g., 'S01', 'S01_RNA', 'S01_ATAC').
    - Only adds **new** metadata columns (does not overwrite existing obs columns).

    Parameters
    ----------
    adata : AnnData
        Integrated AnnData with both RNA and ATAC cells.
    sample_column : str
        Column in adata.obs denoting sample IDs (possibly with suffix).
    modality_col : str
        Column in adata.obs that marks modality, e.g., "RNA" or "ATAC".
    rna_sample_meta_file : str or Path, optional
        CSV-like metadata file for RNA samples.
    atac_sample_meta_file : str or Path, optional
        CSV-like metadata file for ATAC samples.
    original_sample_col : str, optional
        Column name in adata.obs containing original (pre-suffix) sample IDs.
        If None or missing, falls back to `sample_column`.
    verbose : bool
        Verbose logging.

    Returns
    -------
    AnnData
        AnnData with metadata added into .obs.
    """
    if not (rna_sample_meta_file or atac_sample_meta_file):
        if verbose:
            print("No RNA/ATAC sample metadata files provided; skipping multi-omics merge.")
        return adata

    if modality_col not in adata.obs.columns:
        if verbose:
            print(
                f"'{modality_col}' not found in adata.obs; cannot split by modality. "
                f"Skipping multi-omics metadata merge."
            )
        return adata

    def _merge_metadata_for_modality(
        modality_value: str,
        meta_file: Optional[Union[str, Path]],
    ) -> None:
        if meta_file is None:
            return

        meta_path = Path(meta_file)
        if not meta_path.exists():
            if verbose:
                print(
                    f"[{modality_value}] Metadata file not found: {meta_path}. "
                    f"Skipping metadata merge for this modality."
                )
            return

        mask = adata.obs[modality_col].astype(str) == modality_value
        n_cells_mod = int(mask.sum())
        if n_cells_mod == 0:
            if verbose:
                print(
                    f"[{modality_value}] No cells with {modality_col} == '{modality_value}'. "
                    f"Skipping metadata merge."
                )
            return

        if verbose:
            print(f"[{modality_value}] Merging sample metadata for {n_cells_mod} cells using file: {meta_path}")
            print(f"[{modality_value}] Reading metadata CSV...")

        # Read metadata, robust to BOM and delimiters
        meta_df = pd.read_csv(
            meta_path,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
        )
        meta_df = meta_df.copy()

        # Normalize column names
        meta_df.columns = (
            meta_df.columns.astype(str)
            .str.replace(r"^\ufeff", "", regex=True)  # strip BOM if present
            .str.strip()
        )

        # Clean string columns
        for col in meta_df.columns:
            if meta_df[col].dtype == "object":
                meta_df[col] = meta_df[col].fillna("Unknown").astype(str)

        # Determine metadata key column for sample IDs
        if sample_column in meta_df.columns:
            meta_key = sample_column
        elif "sample" in meta_df.columns:
            meta_key = "sample"
        else:
            raise KeyError(
                f"[{modality_value}] Could not find a sample column in metadata file. "
                f"Expected '{sample_column}' or 'sample' in columns: {list(meta_df.columns)}"
            )

        meta_df[meta_key] = meta_df[meta_key].astype(str)

        # Use original sample IDs if available; otherwise fall back to current sample_column
        if original_sample_col is not None and original_sample_col in adata.obs.columns:
            obs_base = adata.obs.loc[mask, original_sample_col].astype(str)
            base_col_used = original_sample_col
        else:
            obs_base = adata.obs.loc[mask, sample_column].astype(str)
            base_col_used = sample_column

        # Candidate 1: direct match
        obs_key1 = obs_base.copy()
        # Candidate 2: strip "_RNA" / "_ATAC" or corresponding modality suffix
        obs_key2 = obs_base.str.replace(fr"_{modality_value}$", "", regex=True)

        meta_keys_set = set(meta_df[meta_key].astype(str))

        n_match1 = len(set(obs_key1) & meta_keys_set)
        n_match2 = len(set(obs_key2) & meta_keys_set)

        if n_match1 == 0 and n_match2 == 0:
            if verbose:
                print(
                    f"[{modality_value}] Warning: no overlap between sample IDs in AnnData ("
                    f"column '{base_col_used}') and metadata file (key '{meta_key}'). "
                    f"Skipping metadata merge for this modality."
                )
            return

        if n_match1 >= n_match2:
            chosen_keys = obs_key1
            strategy = f"direct sample IDs from '{base_col_used}'"
            overlap = n_match1
        else:
            chosen_keys = obs_key2
            strategy = f"base sample IDs (suffix stripped) from '{base_col_used}'"
            overlap = n_match2

        if verbose:
            print(
                f"[{modality_value}] Using {strategy} to merge with metadata key '{meta_key}'. "
                f"Overlap: {overlap} samples."
            )

        meta_indexed = meta_df.set_index(meta_key)

        # Only add columns that do NOT already exist in adata.obs
        new_cols = [
            col for col in meta_indexed.columns
            if col not in adata.obs.columns
        ]
        if not new_cols:
            if verbose:
                print(
                    f"[{modality_value}] All metadata columns already exist in adata.obs; "
                    f"no new columns added."
                )
            return

        matched = chosen_keys.isin(meta_indexed.index).sum()
        if verbose:
            print(
                f"[{modality_value}] Attempting to map metadata for {n_cells_mod} cells; "
                f"{matched} cells have matching sample IDs in metadata."
            )

        for col in new_cols:
            mapping = meta_indexed[col].to_dict()
            adata.obs.loc[mask, col] = chosen_keys.map(mapping).values

        if verbose:
            print(f"[{modality_value}] Added metadata columns: {new_cols}")

    # Run for RNA and ATAC
    _merge_metadata_for_modality("RNA", rna_sample_meta_file)
    _merge_metadata_for_modality("ATAC", atac_sample_meta_file)

    if verbose:
        print("Multi-omics sample metadata merge finished.\n")

    return adata

def fill_missing_metadata_with_placeholder(
    adata: sc.AnnData,
    placeholder: str = "NA",
    verbose: bool = True,
) -> sc.AnnData:
    """
    Sweep the entire AnnData metadata (obs, var) and replace missing values
    with a placeholder string ('NA' by default) to ensure no NAs propagate to
    downstream analysis (design matrices, groupings, plotting).

    Parameters
    ----------
    adata : AnnData
        AnnData after metadata merging / preprocessing
    placeholder : str
        Placeholder value to fill NAs with
    verbose : bool
        Print summary of replacements

    Returns
    -------
    AnnData
        Same AnnData object with NA values replaced in metadata
    """

    # Handle .obs
    if verbose:
        print(f"[NA-SWEEP] Checking obs (rows): shape={adata.obs.shape}")

    obs_before = adata.obs.isna().sum().sum()
    adata.obs = adata.obs.astype('object').fillna(placeholder)
    obs_after = adata.obs.isna().sum().sum()

    if verbose:
        print(f"  - Filled {obs_before - obs_after} NA in obs")

    # Handle .var
    if verbose:
        print(f"[NA-SWEEP] Checking var (columns): shape={adata.var.shape}")

    var_before = adata.var.isna().sum().sum()
    adata.var = adata.var.astype('object').fillna(placeholder)
    var_after = adata.var.isna().sum().sum()

    if verbose:
        print(f"  - Filled {var_before - var_after} NA in var")

    # For .obsm we skip matrices, but treat DataFrames properly
    for key, mat in adata.obsm.items():
        if isinstance(mat, pd.DataFrame):
            before = mat.isna().sum().sum()
            adata.obsm[key] = mat.astype('object').fillna(placeholder)
            after = mat.isna().sum().sum()
            if verbose:
                print(f"[NA-SWEEP] obsm['{key}']: filled {before - after} NA")

    # Optional: varm too
    for key, mat in adata.varm.items():
        if isinstance(mat, pd.DataFrame):
            before = mat.isna().sum().sum()
            adata.varm[key] = mat.astype('object').fillna(placeholder)
            after = mat.isna().sum().sum()
            if verbose:
                print(f"[NA-SWEEP] varm['{key}']: filled {before - after} NA")

    if verbose:
        print("[NA-SWEEP] Completed metadata placeholder filling.\n")

    return adata


def integrate_preprocess(
    output_dir,
    h5ad_path=None,
    sample_column="sample",
    modality_col="modality",
    min_cells_sample=1,
    min_cell_gene=10,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
    verbose=True,
    original_sample_col=None,
    rna_sample_meta_file=None,
    atac_sample_meta_file=None,
):
    start_time = time.time()

    if h5ad_path is None:
        h5ad_path = os.path.join(output_dir, "glue/atac_rna_integrated.h5ad")

    os.makedirs(output_dir, exist_ok=True)
    preprocess_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(preprocess_dir, exist_ok=True)

    if verbose:
        if not os.path.exists(output_dir):
            print("Automatically generating output_dir")
        if not os.path.exists(preprocess_dir):
            print("Automatically generating preprocess subdirectory")

    if doublet and min_cells_sample < 30:
        min_cells_sample = 30
        print("Minimum dimension requested by scrublet is 30, raise sample standard accordingly")

    if verbose:
        print("=== Read input dataset ===")
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f"Dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}")

    if original_sample_col is None:
        original_sample_col = f"original_{sample_column}"

    # 1) Store original sample IDs (before we touch/append modality suffix)
    _store_original_sample_ids(
        adata=adata,
        sample_column=sample_column,
        original_sample_col=original_sample_col,
        verbose=verbose,
    )

    # 2) Re-merge sample metadata for RNA and ATAC separately
    #    This covers the case where sample names may or may not have modality suffixes.
    if rna_sample_meta_file or atac_sample_meta_file:
        if verbose:
            print("Re-merging sample metadata into integrated AnnData (per modality)...")
        adata = multi_omics_merge_sample_meta(
            adata=adata,
            sample_column=sample_column,
            modality_col=modality_col,
            rna_sample_meta_file=rna_sample_meta_file,
            atac_sample_meta_file=atac_sample_meta_file,
            original_sample_col=original_sample_col,
            verbose=verbose,
        )
        if verbose:
            print("Sample metadata re-merge complete.\n")
            
    adata = fill_missing_metadata_with_placeholder(
        adata,
        placeholder="NA",
        verbose=verbose
    )


    # 3) Ensure sample names are unique (append modality suffix only for duplicates)
    adata = _maybe_append_modality_to_duplicates(
        adata=adata,
        sample_column=sample_column,
        modality_col=modality_col,
        verbose=verbose,
    )

    # ---- below: original preprocessing logic, unchanged ----

    adata.var_names_make_unique()

    if isinstance(adata.var, pd.DataFrame):
        adata.var = adata.var.dropna(axis=1, how="all")

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    adata.var["MT"] = adata.var["mt"]

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        log1p=False,
        inplace=True,
    )

    sc.pp.filter_cells(adata, min_genes=min_features)
    if verbose:
        print(f"After cell filtering -- Cells remaining: {adata.n_obs}, Genes remaining: {adata.n_vars}")

    sc.pp.filter_genes(adata, min_cells=min_cell_gene)
    if verbose:
        print(f"After gene filtering -- Cells remaining: {adata.n_obs}, Genes remaining: {adata.n_vars}")

    if "pct_counts_mt" not in adata.obs.columns:
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    adata = adata[adata.obs["pct_counts_mt"] < pct_mito_cutoff].copy()

    mt_genes = adata.var_names[adata.var_names.str.upper().str.startswith("MT-")]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)

    adata = adata[:, ~adata.var_names.isin(list(genes_to_exclude))].copy()
    if verbose:
        print(
            f"After remove MT_gene and user input gene -- Cells remaining: {adata.n_obs}, Genes remaining: {adata.n_vars}"
        )

    cell_counts_per_patient = adata.obs.groupby(sample_column).size()
    if verbose:
        print("Sample counts BEFORE filtering:")
        print(cell_counts_per_patient.sort_values(ascending=False))

    patients_to_keep = cell_counts_per_patient[cell_counts_per_patient >= min_cells_sample].index
    if verbose:
        print(f"\nSamples retained (>= {min_cells_sample} cells): {list(patients_to_keep)}")

    adata = adata[adata.obs[sample_column].isin(patients_to_keep)].copy()

    if verbose:
        cell_counts_after = adata.obs[sample_column].value_counts()
        print("\nSample counts AFTER filtering:")
        print(cell_counts_after.sort_values(ascending=False))

    min_cells_for_gene = max(1, int(0.01 * adata.n_obs))
    sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)
    if verbose:
        print(f"Final filtering -- Cells remaining: {adata.n_obs}, Genes remaining: {adata.n_vars}")

    if verbose:
        print("Preprocessing complete!")

    output_h5ad_path = os.path.join(preprocess_dir, "adata_sample.h5ad")
    adata.write_h5ad(output_h5ad_path)

    if verbose:
        print(f"Preprocessed data saved to: {output_h5ad_path}")
        print(f"Original sample IDs stored in: adata.obs['{original_sample_col}']")
        print(f"Function execution time: {time.time() - start_time:.2f} seconds")

    return adata