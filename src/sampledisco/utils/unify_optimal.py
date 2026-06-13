import os
import scanpy as sc
import pandas as pd
import numpy as np


def replace_optimal_dimension_reduction(
    base_path: str,
    verbose: bool = True,
    modality: str = "RNA",
) -> sc.AnnData:
    """Replace DR results and pseudobulk expression in ``pseudobulk_sample.h5ad``.

    Uses ``optimal_expression`` as the template object (carries updated X, var, DR),
    merges original obs metadata and non-DR uns/obsm from ``pseudobulk_sample``,
    overwrites ``X_DR_expression`` and ``X_DR_proportion`` from the optimal files,
    saves with a ``.backup``, and writes CSV embeddings under
    ``<base_path>/embeddings/``.

    Parameters
    ----------
    base_path : str
        Run output directory, e.g. ``'.../covid_25_sample/rna'``.  Expects
        subdirs ``{MODALITY}_resolution_optimization_expression/summary/optimal.h5ad``,
        ``{MODALITY}_resolution_optimization_proportion/summary/optimal.h5ad``,
        and ``pseudobulk/pseudobulk_sample.h5ad``.
    verbose : bool
    modality : {"RNA", "ATAC"}

    Returns
    -------
    sc.AnnData
        Updated pseudobulk object (also written to disk).
    """

    modality_upper = str(modality).strip().upper()
    if modality_upper not in ("RNA", "ATAC"):
        raise ValueError(f"modality must be 'RNA' or 'ATAC', got {modality!r}")

    optimal_expression_path = os.path.join(
        base_path,
        f"{modality_upper}_resolution_optimization_expression",
        "summary",
        "optimal.h5ad",
    )

    optimal_proportion_path = os.path.join(
        base_path,
        f"{modality_upper}_resolution_optimization_proportion",
        "summary",
        "optimal.h5ad",
    )

    pseudobulk_path = os.path.join(
        base_path,
        "pseudobulk",
        "pseudobulk_sample.h5ad"
    )

    if verbose:
        print("[Replace DR] Starting dimension reduction + expression replacement...")
        print("[Replace DR] File paths:")
        print(f"  - Optimal expression: {optimal_expression_path}")
        print(f"  - Optimal proportion: {optimal_proportion_path}")
        print(f"  - Pseudobulk sample: {pseudobulk_path}")

    missing_files = []
    if not os.path.exists(optimal_expression_path):
        missing_files.append(f"Optimal expression: {optimal_expression_path}")
    if not os.path.exists(optimal_proportion_path):
        missing_files.append(f"Optimal proportion: {optimal_proportion_path}")
    if not os.path.exists(pseudobulk_path):
        missing_files.append(f"Pseudobulk: {pseudobulk_path}")

    if missing_files:
        error_msg = "The following files were not found:\n" + "\n".join(
            [f"  - {f}" for f in missing_files]
        )
        raise FileNotFoundError(error_msg)

    try:
        if verbose:
            print("\n[Replace DR] Loading files...")

        optimal_expression = sc.read_h5ad(optimal_expression_path)
        optimal_proportion = sc.read_h5ad(optimal_proportion_path)
        pseudobulk_sample_original = sc.read_h5ad(pseudobulk_path)

        if verbose:
            print(f"  ✓ Optimal expression loaded: {optimal_expression.shape}")
            print(f"  ✓ Optimal proportion loaded: {optimal_proportion.shape}")
            print(f"  ✓ Original pseudobulk loaded: {pseudobulk_sample_original.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load h5ad files: {str(e)}")

    # All three must share the same obs index order; sample embedding assumes aligned rows.
    if not np.array_equal(
        optimal_expression.obs.index, pseudobulk_sample_original.obs.index
    ):
        raise ValueError(
            "[ERROR] Sample indices in optimal_expression do not match pseudobulk_sample"
        )

    if not np.array_equal(
        optimal_proportion.obs.index, pseudobulk_sample_original.obs.index
    ):
        raise ValueError(
            "[ERROR] Sample indices in optimal_proportion do not match pseudobulk_sample"
        )

    if verbose:
        print(
            "\n[Replace DR] Building updated pseudobulk using optimal EXPRESSION as template..."
        )

    updated_pseudobulk = optimal_expression.copy()

    if verbose:
        print(f"  • Template (optimal_expression) shape: {updated_pseudobulk.shape}")
        print(f"  • Template genes: {updated_pseudobulk.var.shape[0]}")

    if verbose:
        print("  • Merging sample metadata (obs) from original pseudobulk...")

    original_obs = pseudobulk_sample_original.obs
    template_obs = updated_pseudobulk.obs

    original_obs_aligned = original_obs.reindex(template_obs.index)

    for col in original_obs_aligned.columns:
        if col not in template_obs.columns:
            template_obs[col] = original_obs_aligned[col]
        else:
            template_obs[col] = original_obs_aligned[col]

    updated_pseudobulk.obs = template_obs

    if verbose:
        print(f"  ✓ Updated obs columns: {list(updated_pseudobulk.obs.columns)}")

    if verbose:
        print("  • Merging .uns and .obsm from original pseudobulk (non-DR metadata)...")

    merged_uns = dict(pseudobulk_sample_original.uns)
    # Overlay with optimal_expression so DR-related uns keys reflect the new result.
    merged_uns.update(optimal_expression.uns)
    updated_pseudobulk.uns = merged_uns

    merged_obsm = dict(pseudobulk_sample_original.obsm)
    merged_obsm.update(optimal_expression.obsm)
    updated_pseudobulk.obsm = merged_obsm

    if verbose:
        print(f"  ✓ .uns keys after merge: {list(updated_pseudobulk.uns.keys())}")
        print(f"  ✓ .obsm keys after merge: {list(updated_pseudobulk.obsm.keys())}")

    if verbose:
        print("\n[Replace DR] Ensuring EXPRESSION dimension reduction results are from optimal_expression...")

    expression_uns_keys = [
        "X_DR_expression",
        "X_DR_expression_variance",
        "X_DR_expression_variance_ratio",
        "X_pca_expression_method",
        "X_lsi_expression_method",
        "X_spectral_expression_method",
    ]

    expression_obsm_keys = [
        "X_DR_expression",
        "X_pca_expression_method",
        "X_lsi_expression_method",
        "X_spectral_expression_method",
    ]

    copied_expression_count = 0
    for key in expression_uns_keys:
        if key in optimal_expression.uns:
            updated_pseudobulk.uns[key] = optimal_expression.uns[key]
            copied_expression_count += 1
            if verbose:
                shape_info = ""
                if hasattr(optimal_expression.uns[key], "shape"):
                    shape_info = f" (shape: {optimal_expression.uns[key].shape})"
                elif hasattr(optimal_expression.uns[key], "__len__"):
                    shape_info = f" (length: {len(optimal_expression.uns[key])})"
                print(f"  ✓ Ensured .uns['{key}'] from optimal_expression{shape_info}")

    for key in expression_obsm_keys:
        if key in optimal_expression.obsm:
            updated_pseudobulk.obsm[key] = optimal_expression.obsm[key]
            copied_expression_count += 1
            if verbose:
                print(
                    f"  ✓ Ensured .obsm['{key}'] from optimal_expression "
                    f"(shape: {optimal_expression.obsm[key].shape})"
                )

    if copied_expression_count == 0 and verbose:
        print("  ⚠ Warning: No expression DR keys found in optimal_expression file")
    elif verbose:
        print(f"  → Total expression DR-related keys ensured: {copied_expression_count}")

    if verbose:
        print("\n[Replace DR] Replacing PROPORTION dimension reduction results with optimal_proportion...")

    proportion_uns_keys = [
        "X_DR_proportion",
        "X_DR_proportion_variance_ratio",
        "pca_proportion_variance_ratio",
    ]

    proportion_obsm_keys = [
        "X_DR_proportion",
        "X_pca_proportion",
    ]

    copied_proportion_count = 0
    for key in proportion_uns_keys:
        if key in optimal_proportion.uns:
            updated_pseudobulk.uns[key] = optimal_proportion.uns[key]
            copied_proportion_count += 1
            if verbose:
                shape_info = ""
                if hasattr(optimal_proportion.uns[key], "shape"):
                    shape_info = f" (shape: {optimal_proportion.uns[key].shape})"
                elif hasattr(optimal_proportion.uns[key], "__len__"):
                    shape_info = f" (length: {len(optimal_proportion.uns[key])})"
                print(f"  ✓ Copied .uns['{key}'] from optimal_proportion{shape_info}")

    for key in proportion_obsm_keys:
        if key in optimal_proportion.obsm:
            updated_pseudobulk.obsm[key] = optimal_proportion.obsm[key]
            copied_proportion_count += 1
            if verbose:
                print(
                    f"  ✓ Copied .obsm['{key}'] from optimal_proportion "
                    f"(shape: {optimal_proportion.obsm[key].shape})"
                )

    if copied_proportion_count == 0 and verbose:
        print("  ⚠ Warning: No proportion DR keys found in optimal_proportion file")
    elif verbose:
        print(f"  → Total proportion DR-related keys copied: {copied_proportion_count}")

    try:
        if verbose:
            print("\n[Replace DR] Saving updated pseudobulk_sample...")
            print(f"  Destination: {pseudobulk_path}")

        backup_path = pseudobulk_path + ".backup"
        if os.path.exists(backup_path):
            if verbose:
                print(f"  Note: Backup file already exists: {backup_path}")
        else:
            import shutil

            shutil.copy2(pseudobulk_path, backup_path)
            if verbose:
                print(f"  ✓ Created backup: {backup_path}")

        updated_pseudobulk.write_h5ad(pseudobulk_path)

        if os.path.exists(pseudobulk_path):
            file_size = os.path.getsize(pseudobulk_path)
            if verbose:
                print(
                    f"  ✓ Successfully saved updated pseudobulk "
                    f"({file_size / (1024*1024):.1f} MB)"
                )
                print("\n[Replace DR] === SUMMARY ===")
                print(f"  Expression DR keys ensured:  {copied_expression_count}")
                print(f"  Proportion DR keys copied: {copied_proportion_count}")
                print("  Expression matrix source: optimal_expression (template)")
                print(f"  File updated: {pseudobulk_path}")
                print(f"  Backup saved: {backup_path}")
        else:
            raise RuntimeError("File was not created after save operation")

    except Exception as e:
        raise RuntimeError(f"Failed to save updated pseudobulk file: {str(e)}")

    try:
        if verbose:
            print("\n[Replace DR] Updating embedding CSV files...")

        embedding_dir = os.path.join(base_path, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)

        def _save_embedding_csv_from_uns(
            uns_key: str, filename: str, desc: str
        ) -> bool:
            if uns_key not in updated_pseudobulk.uns:
                if verbose:
                    print(
                        f"  ⚠ {desc} not found in updated_pseudobulk.uns['{uns_key}']; "
                        "skipping CSV update"
                    )
                return False

            data = updated_pseudobulk.uns[uns_key]

            if isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                try:
                    df = pd.DataFrame(data)
                except Exception:
                    if verbose:
                        print(
                            f"  ⚠ Skipping {desc}: could not convert type {type(data)} to DataFrame"
                        )
                    return False

            out_path = os.path.join(embedding_dir, filename)
            df.to_csv(out_path, index=False)

            if verbose:
                print(f"  ✓ Saved {desc} to {out_path} (shape: {df.shape})")
            return True

        _save_embedding_csv_from_uns(
            "X_DR_expression",
            "sample_expression_embedding.csv",
            "expression embedding",
        )
        _save_embedding_csv_from_uns(
            "X_DR_proportion",
            "sample_proportion_embedding.csv",
            "proportion embedding",
        )

    except Exception as e:
        if verbose:
            print(f"  ⚠ Failed to update embedding CSV files: {e}")

    return updated_pseudobulk
