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

    _store_original_sample_ids(
        adata=adata,
        sample_column=sample_column,
        original_sample_col=original_sample_col,
        verbose=verbose,
    )

    _maybe_append_modality_to_duplicates(
        adata=adata,
        sample_column=sample_column,
        modality_col=modality_col,
        verbose=verbose,
    )

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

    # adata.raw = adata.copy()

    if verbose:
        print("Preprocessing complete!")

    output_h5ad_path = os.path.join(preprocess_dir, "adata_sample.h5ad")
    adata.write_h5ad(output_h5ad_path)

    if verbose:
        print(f"Preprocessed data saved to: {output_h5ad_path}")
        print(f"Original sample IDs stored in: adata.obs['{original_sample_col}']")
        print(f"Function execution time: {time.time() - start_time:.2f} seconds")

    return adata