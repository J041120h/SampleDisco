import os
import time
import pandas as pd
import scanpy as sc

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


def fill_missing_metadata_with_placeholder(
    adata: sc.AnnData,
    numeric_placeholder: float = -1.0,
    string_placeholder: str = "NA",
    bool_placeholder: bool = False,
    datetime_placeholder: pd.Timestamp = pd.Timestamp("1900-01-01"),
    verbose: bool = True,
) -> sc.AnnData:
    """
    Sweep the metadata parts of an AnnData object (obs, var, and any
    DataFrame-like obsm/varm entries) and fill missing values with
    dtype-appropriate placeholders.

    This avoids mixing numeric/string dtypes (which can break saving)
    while ensuring there are no NaNs for downstream analysis.
    """
    from pandas.api import types as ptypes

    def _fill_df(df: pd.DataFrame, label: str) -> None:
        if df is None or df.shape[1] == 0:
            return

        total_before = int(df.isna().sum().sum())
        if verbose:
            print(f"[NA-SWEEP] {label}: {total_before} NA before filling")

        for col in df.columns:
            s = df[col]
            n_before = int(s.isna().sum())
            if n_before == 0:
                continue

            # Categorical
            if ptypes.is_categorical_dtype(s):
                if string_placeholder not in s.cat.categories:
                    df[col] = s.cat.add_categories([string_placeholder])
                    s = df[col]
                df[col] = s.fillna(string_placeholder)

            # Boolean
            elif ptypes.is_bool_dtype(s):
                df[col] = s.fillna(bool_placeholder)

            # Numeric (int/float)
            elif ptypes.is_numeric_dtype(s):
                df[col] = s.fillna(numeric_placeholder)

            # Datetime-like
            elif ptypes.is_datetime64_any_dtype(s):
                df[col] = s.fillna(datetime_placeholder)

            # Everything else → treat as string
            else:
                df[col] = s.astype("object").fillna(string_placeholder)

            if verbose:
                n_after = int(df[col].isna().sum())
                print(
                    f"  - {label}['{col}']: filled {n_before - n_after} NA "
                    f"(dtype: {df[col].dtype})"
                )

        total_after = int(df.isna().sum().sum())
        if verbose:
            print(f"[NA-SWEEP] {label}: {total_before - total_after} NA filled total\n")

    # obs
    _fill_df(adata.obs, "obs")

    # var
    _fill_df(adata.var, "var")

    # obsm: only DataFrame-like entries
    for key, val in list(adata.obsm.items()):
        if isinstance(val, pd.DataFrame):
            _fill_df(adata.obsm[key], f"obsm['{key}']")

    # varm: only DataFrame-like entries
    for key, val in list(adata.varm.items()):
        if isinstance(val, pd.DataFrame):
            _fill_df(adata.varm[key], f"varm['{key}']")

    if verbose:
        print("[NA-SWEEP] Completed dtype-aware placeholder filling.\n")

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
    def _merge_metadata_for_modality(modality_value: str, meta_file: str):
        if meta_file is None:
            return
        if modality_col not in adata.obs.columns:
            if verbose:
                print(
                    f"[{modality_value}] '{modality_col}' not found in adata.obs; "
                    f"skipping metadata merge for this modality."
                )
            return

        if not os.path.exists(meta_file):
            if verbose:
                print(
                    f"[{modality_value}] Metadata file not found: {meta_file}. "
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
            print(f"[{modality_value}] Merging sample metadata for {n_cells_mod} cells using file: {meta_file}")

        meta_df = pd.read_csv(meta_file)
        meta_df = meta_df.copy()

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
        if original_sample_col in adata.obs.columns:
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
                    f"columns '{base_col_used}') and metadata file (key '{meta_key}'). "
                    f"Skipping metadata merge for this modality."
                )
            return

        if n_match1 >= n_match2:
            chosen_keys = obs_key1
            if verbose:
                print(
                    f"[{modality_value}] Using direct sample IDs from '{base_col_used}' "
                    f"to merge with metadata key '{meta_key}'. Overlap: {n_match1} samples."
                )
        else:
            chosen_keys = obs_key2
            if verbose:
                print(
                    f"[{modality_value}] Using base sample IDs (suffix stripped) from '{base_col_used}' "
                    f"to merge with metadata key '{meta_key}'. Overlap: {n_match2} samples."
                )

        # Only add columns that do NOT already exist in adata.obs
        new_cols = [
            col for col in meta_df.columns
            if col != meta_key and col not in adata.obs.columns
        ]
        if not new_cols:
            if verbose:
                print(
                    f"[{modality_value}] All metadata columns already exist in adata.obs; "
                    f"no new columns added."
                )
            return

        meta_indexed = meta_df.set_index(meta_key)

        for col in new_cols:
            mapping = meta_indexed[col].to_dict()
            # chosen_keys is indexed by the same index as adata.obs[mask]
            adata.obs.loc[mask, col] = chosen_keys.map(mapping).values

        if verbose:
            print(f"[{modality_value}] Added metadata columns: {new_cols}")

    if rna_sample_meta_file or atac_sample_meta_file:
        if verbose:
            print("Re-merging sample metadata into integrated AnnData (per modality)...")
        _merge_metadata_for_modality("RNA", rna_sample_meta_file)
        _merge_metadata_for_modality("ATAC", atac_sample_meta_file)
        if verbose:
            print("Sample metadata re-merge complete.\n")

    # 3) Ensure sample names are unique (append modality suffix only for duplicates)
    _maybe_append_modality_to_duplicates(
        adata=adata,
        sample_column=sample_column,
        modality_col=modality_col,
        verbose=verbose,
    )

    # ---- Original preprocessing logic ----

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

    # === NEW: dtype-aware NA filling in all metadata ===
    adata = fill_missing_metadata_with_placeholder(
        adata,
        numeric_placeholder=-1.0,
        string_placeholder="NA",
        bool_placeholder=False,
        datetime_placeholder=pd.Timestamp("1900-01-01"),
        verbose=verbose,
    )

    if verbose:
        print("Preprocessing complete!")

    output_h5ad_path = os.path.join(preprocess_dir, "adata_sample.h5ad")
    adata.write_h5ad(output_h5ad_path)

    if verbose:
        print(f"Preprocessed data saved to: {output_h5ad_path}")
        print(f"Original sample IDs stored in: adata.obs['{original_sample_col}']")
        print(f"Function execution time: {time.time() - start_time:.2f} seconds")

    return adata
