import os
import scanpy as sc
import numpy as np
import pandas as pd
from typing import Optional, Union


def replace_optimal_dimension_reduction(
    base_path: str,
    expression_resolution_dir: Optional[str] = None,
    proportion_resolution_dir: Optional[str] = None,
    pseudobulk_path: Optional[str] = None,
    optimization_target: str = "rna",
    verbose: bool = True
) -> sc.AnnData:
    """
    Replaces dimension reduction results AND pseudobulk expression in pseudobulk_sample.h5ad
    with optimal results from resolution optimization for BOTH expression and proportion.

    New strategy:
    -------------
    - Use `optimal_expression` AnnData as the TEMPLATE for the updated pseudobulk.
    - Ensure sample indices match between `optimal_expression` and the original `pseudobulk_sample`.
    - Copy all sample metadata (obs columns) from `pseudobulk_sample` into the new object.
    - Merge .uns and .obsm from `pseudobulk_sample`, then overwrite DR-related keys with
      values from `optimal_expression` and `optimal_proportion`.
    - Save this updated AnnData back to pseudobulk_path (with a .backup created first).
    - Update unified embedding CSVs.

    File structure:
    ---------------
    resolution_optimization_{dr_type}/
        Integration_optimization_{target}_{dr_type}/
            summary/optimal_{target}_{dr_type}.h5ad

    Parameters
    ----------
    base_path : str
        Base path to the output directory
        Example: '/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_25_sample/multiomics'
    expression_resolution_dir : str, optional
        Path to the expression resolution optimization directory
        If None, defaults to base_path/resolution_optimization_expression
    proportion_resolution_dir : str, optional
        Path to the proportion resolution optimization directory
        If None, defaults to base_path/resolution_optimization_proportion
    pseudobulk_path : str, optional
        Path to the pseudobulk_sample.h5ad file
        If None, defaults to base_path/pseudobulk/pseudobulk_sample.h5ad
    optimization_target : str, default "rna"
        Optimization target ('rna' or 'atac') - used to construct filenames
    verbose : bool, default True
        Whether to print verbose output

    Returns
    -------
    sc.AnnData
        Updated pseudobulk AnnData with optimal expression and embeddings
    """

    # ------------------------------------------------------------------
    # Construct default file paths if not provided
    # ------------------------------------------------------------------
    if expression_resolution_dir is None:
        expression_resolution_dir = os.path.join(
            base_path,
            "resolution_optimization_expression"
        )

    if proportion_resolution_dir is None:
        proportion_resolution_dir = os.path.join(
            base_path,
            "resolution_optimization_proportion"
        )

    if pseudobulk_path is None:
        pseudobulk_path = os.path.join(
            base_path,
            "pseudobulk",
            "pseudobulk_sample.h5ad"
        )

    optimal_expression_path = os.path.join(
        expression_resolution_dir,
        f"Integration_optimization_{optimization_target}_expression",
        "summary",
        f"optimal_{optimization_target}_expression.h5ad"
    )

    optimal_proportion_path = os.path.join(
        proportion_resolution_dir,
        f"Integration_optimization_{optimization_target}_proportion",
        "summary",
        f"optimal_{optimization_target}_proportion.h5ad"
    )

    if verbose:
        print("\n" + "=" * 70)
        print("Replacing Dimension Reduction and Expression with Optimal Results")
        print("=" * 70)
        print("\n[Replace DR] Configuration:")
        print(f"  - Optimization target: {optimization_target.upper()}")
        print("\n[Replace DR] File paths:")
        print(f"  - Optimal expression: {optimal_expression_path}")
        print(f"  - Optimal proportion: {optimal_proportion_path}")
        print(f"  - Pseudobulk sample: {pseudobulk_path}")

    # ------------------------------------------------------------------
    # Check if required files exist
    # ------------------------------------------------------------------
    missing_files = []
    if not os.path.exists(optimal_expression_path):
        missing_files.append(f"Optimal expression: {optimal_expression_path}")
    if not os.path.exists(optimal_proportion_path):
        missing_files.append(f"Optimal proportion: {optimal_proportion_path}")
    if not os.path.exists(pseudobulk_path):
        missing_files.append(f"Pseudobulk: {pseudobulk_path}")

    if missing_files:
        error_msg = "\n[ERROR] The following required files were not found:\n"
        error_msg += "\n".join([f"  ✗ {f}" for f in missing_files])
        raise FileNotFoundError(error_msg)

    # ------------------------------------------------------------------
    # Load all three AnnData files
    # ------------------------------------------------------------------
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
        raise RuntimeError(f"[ERROR] Failed to load h5ad files: {str(e)}")

    # ------------------------------------------------------------------
    # Verify sample alignment (obs indices), but DO NOT require gene match
    # ------------------------------------------------------------------
    if not np.array_equal(optimal_expression.obs.index,
                          pseudobulk_sample_original.obs.index):
        raise ValueError(
            "[ERROR] Sample indices in optimal_expression do not match original pseudobulk_sample.\n"
            "        These must match to safely merge sample metadata."
        )

    if not np.array_equal(optimal_proportion.obs.index,
                          pseudobulk_sample_original.obs.index):
        raise ValueError(
            "[ERROR] Sample indices in optimal_proportion do not match original pseudobulk_sample."
        )

    # ------------------------------------------------------------------
    # NEW STRATEGY: Use optimal_expression as TEMPLATE for updated pseudobulk
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Replace DR] Building updated pseudobulk using optimal EXPRESSION as template...")

    # Start from optimal_expression: X, var, layers, obsm, uns, etc.
    updated_pseudobulk = optimal_expression.copy()

    if verbose:
        print(f"  • Template (optimal_expression) shape: {updated_pseudobulk.shape}")
        print(f"  • Template genes: {updated_pseudobulk.var.shape[0]}")

    # ------------------------------------------------------------------
    # Copy / merge obs: ensure we preserve all sample metadata from original pseudobulk
    # ------------------------------------------------------------------
    if verbose:
        print("  • Merging sample metadata (obs) from original pseudobulk...")

    # Start with the template obs, then add/overwrite columns from original
    original_obs = pseudobulk_sample_original.obs
    template_obs = updated_pseudobulk.obs

    # Align original obs to template index (they should match already)
    original_obs_aligned = original_obs.reindex(template_obs.index)

    # Add any missing columns from original to template
    for col in original_obs_aligned.columns:
        if col not in template_obs.columns:
            template_obs[col] = original_obs_aligned[col]
        else:
            # Overwrite existing column with original, to keep original metadata
            template_obs[col] = original_obs_aligned[col]

    updated_pseudobulk.obs = template_obs

    if verbose:
        print(f"  ✓ Updated obs columns: {list(updated_pseudobulk.obs.columns)}")

    # ------------------------------------------------------------------
    # Merge .uns and .obsm with original pseudobulk for non-DR metadata
    # ------------------------------------------------------------------
    if verbose:
        print("  • Merging .uns and .obsm from original pseudobulk (non-DR metadata)...")

    # For .uns: original keys, then overlay optimized expression .uns
    merged_uns = dict(pseudobulk_sample_original.uns)  # base: original
    merged_uns.update(optimal_expression.uns)          # overwrite with optimized expression info
    updated_pseudobulk.uns = merged_uns

    # For .obsm: original keys, then overlay optimized expression .obsm
    merged_obsm = dict(pseudobulk_sample_original.obsm)
    merged_obsm.update(optimal_expression.obsm)
    updated_pseudobulk.obsm = merged_obsm

    if verbose:
        print(f"  ✓ .uns keys after merge: {list(updated_pseudobulk.uns.keys())}")
        print(f"  ✓ .obsm keys after merge: {list(updated_pseudobulk.obsm.keys())}")

    # ------------------------------------------------------------------
    # Replace / ensure EXPRESSION DR keys from optimal_expression
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Replace DR] Ensuring EXPRESSION dimension reduction results are from optimal_expression...")

    expression_uns_keys = [
        'X_DR_expression',
        'X_DR_expression_variance',
        'X_DR_expression_variance_ratio',
        'X_pca_expression_method',
        'X_lsi_expression_method',
        'X_spectral_expression_method'
    ]

    expression_obsm_keys = [
        'X_DR_expression',
        'X_pca_expression_method',
        'X_lsi_expression_method',
        'X_spectral_expression_method'
    ]

    copied_expression_count = 0
    for key in expression_uns_keys:
        if key in optimal_expression.uns:
            updated_pseudobulk.uns[key] = optimal_expression.uns[key]
            copied_expression_count += 1
            if verbose:
                shape_info = ""
                if hasattr(optimal_expression.uns[key], 'shape'):
                    shape_info = f" (shape: {optimal_expression.uns[key].shape})"
                elif hasattr(optimal_expression.uns[key], '__len__'):
                    shape_info = f" (length: {len(optimal_expression.uns[key])})"
                print(f"  ✓ Ensured .uns['{key}'] from optimal_expression{shape_info}")

    for key in expression_obsm_keys:
        if key in optimal_expression.obsm:
            updated_pseudobulk.obsm[key] = optimal_expression.obsm[key]
            copied_expression_count += 1
            if verbose:
                print(f"  ✓ Ensured .obsm['{key}'] from optimal_expression "
                      f"(shape: {optimal_expression.obsm[key].shape})")

    if copied_expression_count == 0:
        print("  ⚠ Warning: No expression DR keys found in optimal_expression file")
    elif verbose:
        print(f"  → Total expression DR-related keys ensured: {copied_expression_count}")

    # ------------------------------------------------------------------
    # Replace PROPORTION DR with optimal_proportion DR
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Replace DR] Replacing PROPORTION dimension reduction results with optimal_proportion...")

    proportion_uns_keys = [
        'X_DR_proportion',
        'X_DR_proportion_variance',
        'X_DR_proportion_variance_ratio',
        'pca_proportion_variance_ratio'
    ]

    proportion_obsm_keys = [
        'X_DR_proportion',
        'X_pca_proportion'
    ]

    copied_proportion_count = 0
    for key in proportion_uns_keys:
        if key in optimal_proportion.uns:
            updated_pseudobulk.uns[key] = optimal_proportion.uns[key]
            copied_proportion_count += 1
            if verbose:
                shape_info = ""
                if hasattr(optimal_proportion.uns[key], 'shape'):
                    shape_info = f" (shape: {optimal_proportion.uns[key].shape})"
                elif hasattr(optimal_proportion.uns[key], '__len__'):
                    shape_info = f" (length: {len(optimal_proportion.uns[key])})"
                print(f"  ✓ Copied .uns['{key}'] from optimal_proportion{shape_info}")

    for key in proportion_obsm_keys:
        if key in optimal_proportion.obsm:
            updated_pseudobulk.obsm[key] = optimal_proportion.obsm[key]
            copied_proportion_count += 1
            if verbose:
                print(f"  ✓ Copied .obsm['{key}'] from optimal_proportion "
                      f"(shape: {optimal_proportion.obsm[key].shape})")

    if copied_proportion_count == 0:
        print("  ⚠ Warning: No proportion DR keys found in optimal_proportion file")
    elif verbose:
        print(f"  → Total proportion DR-related keys copied: {copied_proportion_count}")

    # ------------------------------------------------------------------
    # Save the updated pseudobulk (with backup)
    # ------------------------------------------------------------------
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
                print(f"  ✓ Successfully saved updated pseudobulk "
                      f"({file_size / (1024*1024):.1f} MB)")
                print("\n" + "-" * 70)
                print("SUMMARY")
                print("-" * 70)
                print(f"  Optimization target: {optimization_target.upper()}")
                print(f"  Expression DR keys ensured: {copied_expression_count}")
                print(f"  Proportion DR keys copied: {copied_proportion_count}")
                print(f"  Total DR keys updated: {copied_expression_count + copied_proportion_count}")
                print(f"  Expression matrix source: optimal_expression (template)")
                print(f"  File updated: {pseudobulk_path}")
                print(f"  Backup saved: {backup_path}")
                print("=" * 70 + "\n")
        else:
            raise RuntimeError("File was not created after save operation")

    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to save updated pseudobulk file: {str(e)}")

    # ------------------------------------------------------------------
    # Update CSV embeddings like the main pipeline
    # ------------------------------------------------------------------
    try:
        if verbose:
            print("[Replace DR] Updating embedding CSV files...")

        embedding_dir = os.path.join(base_path, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)

        def _save_embedding_csv_from_uns(uns_key: str, filename: str, desc: str) -> bool:
            if uns_key not in updated_pseudobulk.uns:
                if verbose:
                    print(f"  ⚠ {desc} not found in updated_pseudobulk.uns['{uns_key}']; skipping CSV update")
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
                        print(f"  ⚠ Skipping {desc}: could not convert type {type(data)} to DataFrame")
                    return False

            out_path = os.path.join(embedding_dir, filename)
            df.to_csv(out_path, index=False)

            if verbose:
                print(f"  ✓ Saved {desc} to {out_path} (shape: {df.shape})")
            return True

        _ = _save_embedding_csv_from_uns(
            "X_DR_expression",
            "sample_expression_embedding.csv",
            "expression embedding"
        )
        _ = _save_embedding_csv_from_uns(
            "X_DR_proportion",
            "sample_proportion_embedding.csv",
            "proportion embedding"
        )

    except Exception as e:
        if verbose:
            print(f"  ⚠ Failed to update embedding CSV files: {e}")

    return updated_pseudobulk
