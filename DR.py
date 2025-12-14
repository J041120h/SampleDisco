import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse as sparse
from muon import atac as ac
import snapatac2 as snap
from typing import Union, List, Optional


def run_lsi_expression(
    pseudobulk_anndata: sc.AnnData,
    n_components: int = 10,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Performs LSI (Latent Semantic Indexing) on pseudobulk expression data for ATAC-seq using scanpy.
    Parameters:
    -----------
    pseudobulk_anndata : sc.AnnData
        AnnData object with samples as observations and genes as variables (sample * gene)
    n_components : int, default 10
        Number of LSI components to compute
    verbose : bool, default False
        Whether to print verbose output
    Returns:
    --------
    pd.DataFrame
        LSI coordinates with samples as rows and LSI components as columns
    """
    if verbose:
        print(f"[LSI] Computing LSI with {n_components} components on {pseudobulk_anndata.shape} data")

    pb_adata = pseudobulk_anndata.copy()
    n_samples, n_genes = pb_adata.shape
    max_components = min(n_samples - 1, n_genes - 1)

    if n_components > max_components:
        n_components = max_components
    if n_components <= 0:
        raise ValueError(f"Cannot perform LSI: insufficient data dimensions (samples={n_samples}, genes={n_genes}).")

    try:
        if sparse.issparse(pb_adata.X):
            pb_adata.X = pb_adata.X.tocsr()

        ac.pp.tfidf(pb_adata, scale_factor=1e4)
        sc.tl.lsi(pb_adata, n_comps=n_components, random_state=42)
        lsi_coords = pb_adata.obsm['X_lsi']

        lsi_df = pd.DataFrame(
            data=lsi_coords,
            index=pb_adata.obs_names,
            columns=[f"LSI{i+1}" for i in range(lsi_coords.shape[1])]
        )

        if verbose:
            print(f"[LSI] Success. Shape: {lsi_df.shape}")
        return lsi_df

    except Exception as e:
        if verbose:
            print(f"[run_lsi_expression] Scanpy LSI failed: {str(e)}")

        # Method 2: Manual LSI implementation using TruncatedSVD with TF-IDF
        try:
            if verbose:
                print("[run_lsi_expression] Attempting manual LSI with TF-IDF")

            # Get the data matrix
            X = pb_adata.X
            if sparse.issparse(X):
                X = X.toarray()

            # Check for NaN values in genes (columns) and drop them
            if verbose:
                print(f"[run_lsi_expression] Original shape: {X.shape}")

            # Check for NaN values in each gene (column)
            nan_genes = np.isnan(X).any(axis=0)
            n_nan_genes = np.sum(nan_genes)

            if n_nan_genes > 0:
                if verbose:
                    print(f"[run_lsi_expression] Found {n_nan_genes} genes with NaN values, dropping them")

                # Keep only genes without NaN values
                X = X[:, ~nan_genes]

                # Update the AnnData object to reflect dropped genes
                pb_adata = pb_adata[:, ~nan_genes].copy()

                if verbose:
                    print(f"[run_lsi_expression] Shape after dropping NaN genes: {X.shape}")

                # Update max_components after potentially reducing gene count
                n_samples, n_genes = X.shape
                max_components = min(n_samples - 1, n_genes - 1)

                if n_components > max_components:
                    n_components = max_components
                    if verbose:
                        print(f"[run_lsi_expression] Reduced n_components to {n_components} due to data dimensions")

                if n_components <= 0:
                    if verbose:
                        print("[run_lsi_expression] Insufficient data dimensions after dropping NaN genes")
                    return None

            # Apply TF-IDF transformation (common for ATAC-seq LSI)
            tfidf = TfidfTransformer(norm='l2', use_idf=True, sublinear_tf=True)
            X_tfidf = tfidf.fit_transform(X)

            # CRITICAL FIX: Check for NaN/inf values AFTER TF-IDF transformation
            if sparse.issparse(X_tfidf):
                # For sparse matrices, check the data array
                nan_mask = np.isnan(X_tfidf.data) | np.isinf(X_tfidf.data)
                if np.any(nan_mask):
                    if verbose:
                        print(f"[run_lsi_expression] Found {np.sum(nan_mask)} NaN/inf values in TF-IDF data, replacing with 0")
                    X_tfidf.data[nan_mask] = 0.0
                    X_tfidf.eliminate_zeros()  # Remove explicit zeros

                # Convert to dense for final check and SVD
                X_tfidf_dense = X_tfidf.toarray()
            else:
                X_tfidf_dense = X_tfidf

            # Final check for NaN/inf values in dense matrix
            nan_mask = np.isnan(X_tfidf_dense) | np.isinf(X_tfidf_dense)
            if np.any(nan_mask):
                if verbose:
                    print(f"[run_lsi_expression] Found {np.sum(nan_mask)} NaN/inf values after TF-IDF, replacing with 0")
                X_tfidf_dense[nan_mask] = 0.0

            # Additional safety check: if any sample has all zeros, this can cause issues
            sample_sums = np.sum(X_tfidf_dense, axis=1)
            zero_samples = sample_sums == 0
            if np.any(zero_samples):
                if verbose:
                    print(f"[run_lsi_expression] Found {np.sum(zero_samples)} samples with all-zero TF-IDF values")
                    print("[run_lsi_expression] Adding small noise to prevent SVD issues")
                # Add very small random noise to zero samples
                noise_scale = 1e-10
                for i in np.where(zero_samples)[0]:
                    X_tfidf_dense[i, :] = np.random.normal(0, noise_scale, X_tfidf_dense.shape[1])

            if verbose:
                print(f"[run_lsi_expression] TF-IDF matrix stats: shape={X_tfidf_dense.shape}, "
                      f"min={X_tfidf_dense.min():.6f}, max={X_tfidf_dense.max():.6f}, "
                      f"mean={X_tfidf_dense.mean():.6f}")

            # Perform SVD (LSI is essentially SVD on term-document matrix)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            lsi_coords = svd.fit_transform(X_tfidf_dense)

            # Final check for NaN in results
            if np.any(np.isnan(lsi_coords)):
                if verbose:
                    print("[run_lsi_expression] NaN values found in LSI results, attempting cleanup")
                lsi_coords = np.nan_to_num(lsi_coords, nan=0.0, posinf=0.0, neginf=0.0)

            lsi_df = pd.DataFrame(
                data=lsi_coords,
                index=pb_adata.obs_names,
                columns=[f"LSI{i+1}" for i in range(lsi_coords.shape[1])]
            )

            if verbose:
                print(f"[run_lsi_expression] Manual LSI computation successful. Shape: {lsi_df.shape}")
                print(f"[run_lsi_expression] LSI result stats: min={lsi_coords.min():.6f}, max={lsi_coords.max():.6f}")

            return lsi_df

        except Exception as e2:
            if verbose:
                print(f"[run_lsi_expression] Manual LSI also failed: {str(e2)}")
            return None


def run_snapatac2_spectral(
    pseudobulk_anndata: sc.AnnData,
    n_components: int = 10,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Performs snapATAC2 spectral embedding on pseudobulk ATAC-seq data.
    """
    if verbose:
        print(f"[snapATAC2] Computing spectral embedding with {n_components} components on {pseudobulk_anndata.shape} data")

    pb_adata = pseudobulk_anndata.copy()

    try:
        # Fix data type issues - ensure data is in correct format for snapATAC2
        if sparse.issparse(pb_adata.X):
            pb_adata.X = pb_adata.X.tocsr().astype(np.float32)
        else:
            pb_adata.X = pb_adata.X.astype(np.float32)

        # Handle any infinite or NaN values
        if sparse.issparse(pb_adata.X):
            pb_adata.X.data = np.nan_to_num(pb_adata.X.data, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            pb_adata.X = np.nan_to_num(pb_adata.X, nan=0.0, posinf=0.0, neginf=0.0)

        if verbose:
            print(f"[snapATAC2] Data preprocessing complete. Matrix type: {type(pb_adata.X)}, dtype: {pb_adata.X.dtype}")

        # Select features (keeping all since they're already filtered)
        n_features_to_select = min(50000, pb_adata.shape[1])
        snap.pp.select_features(pb_adata, n_features=n_features_to_select)

        # Compute spectral embedding
        snap.tl.spectral(pb_adata, n_comps=n_components, random_state=42)

        # Extract spectral coordinates
        spectral_coords = pb_adata.obsm['X_spectral']
        spectral_df = pd.DataFrame(
            data=spectral_coords,
            index=pb_adata.obs_names,
            columns=[f"Spectral{i+1}" for i in range(spectral_coords.shape[1])]
        )

        if verbose:
            print(f"[snapATAC2] Success. Shape: {spectral_df.shape}")

        return spectral_df

    except Exception as e:
        if verbose:
            print(f"[snapATAC2] Failed: {str(e)}")
        return None


def _store_results_in_both_objects(adata, pseudobulk_anndata, key, df_result, obsm_key=None, verbose=False):
    """Helper function to store results consistently in both adata and pseudobulk_anndata objects."""
    if df_result is not None:
        adata.uns[key] = df_result
        pseudobulk_anndata.uns[key] = df_result
        if obsm_key is not None:
            pseudobulk_anndata.obsm[obsm_key] = df_result.values
        if verbose:
            print(f"[Storage] Stored {key} in both adata and pseudobulk_anndata (shape: {df_result.shape})")
    else:
        if verbose:
            print(f"[Storage] Skipped storing {key} - result was None")


def run_dimension_reduction_expression(
    adata: sc.AnnData,
    pseudobulk_anndata: sc.AnnData,
    n_components: int = 10,
    atac: bool = False,
    use_snapatac2_dimred: bool = False,
    verbose: bool = False
) -> None:
    """
    Performs dimension reduction on pseudobulk-corrected expression data.
    All results are stored under the unified key 'X_DR_expression'.
    """
    if pseudobulk_anndata is None:
        raise ValueError("pseudobulk_anndata parameter is required.")

    if verbose:
        if atac:
            if use_snapatac2_dimred:
                print(f"[DimRed] Computing snapATAC2 spectral (with LSI fallback) for ATAC data with {n_components} components on {pseudobulk_anndata.shape} data")
            else:
                print(f"[DimRed] Computing LSI for ATAC data with {n_components} components on {pseudobulk_anndata.shape} data")
        else:
            print(f"[DimRed] Computing PCA for RNA data with {n_components} components on {pseudobulk_anndata.shape} data")

    pb_adata = pseudobulk_anndata.copy()

    n_samples, n_genes = pb_adata.shape
    max_components = min(n_samples - 1, n_genes)
    if n_components > max_components:
        n_components = max_components

    if n_components <= 0:
        raise ValueError(f"Cannot perform dimension reduction: insufficient data dimensions (samples={n_samples}, genes={n_genes}).")

    if atac:
        primary_result = None
        method_used = None

        if use_snapatac2_dimred:
            spectral_df = run_snapatac2_spectral(
                pseudobulk_anndata=pseudobulk_anndata,
                n_components=n_components,
                verbose=verbose
            )

            if spectral_df is not None:
                primary_result = spectral_df
                method_used = "snapATAC2_spectral"

                _store_results_in_both_objects(
                    adata, pseudobulk_anndata,
                    "X_spectral_expression_method", spectral_df,
                    obsm_key="X_spectral_expression_method",
                    verbose=verbose
                )
            else:
                if verbose:
                    print("[DimRed] snapATAC2 spectral failed, falling back to LSI...")

        if primary_result is None:
            lsi_df = run_lsi_expression(
                pseudobulk_anndata=pseudobulk_anndata,
                n_components=n_components,
                verbose=verbose
            )

            if lsi_df is not None:
                primary_result = lsi_df
                method_used = "LSI"

                _store_results_in_both_objects(
                    adata, pseudobulk_anndata,
                    "X_lsi_expression_method", lsi_df,
                    obsm_key="X_lsi_expression_method",
                    verbose=verbose
                )
            else:
                raise RuntimeError("Both snapATAC2 spectral and LSI methods failed for ATAC data")

        if primary_result is not None:
            _store_results_in_both_objects(
                adata, pseudobulk_anndata,
                "X_DR_expression", primary_result,
                obsm_key="X_DR_expression",
                verbose=verbose
            )

            if verbose:
                print(f"[DimRed] Successfully used {method_used} for ATAC dimension reduction")

    else:
        try:
            if pb_adata.X.max() > 100:
                sc.pp.log1p(pb_adata)

            sc.tl.pca(pb_adata, n_comps=n_components, svd_solver='arpack', random_state=42)

            pca_coords = pb_adata.obsm['X_pca']
            pca_df = pd.DataFrame(
                data=pca_coords,
                index=pb_adata.obs_names,
                columns=[f"PC{i+1}" for i in range(pca_coords.shape[1])]
            )

            if verbose:
                print(f"[PCA] Success. Shape: {pca_df.shape}")

            _store_results_in_both_objects(
                adata, pseudobulk_anndata,
                "X_DR_expression", pca_df,
                obsm_key="X_DR_expression",
                verbose=verbose
            )

            _store_results_in_both_objects(
                adata, pseudobulk_anndata,
                "X_pca_expression_method", pca_df,
                obsm_key="X_pca_expression_method",
                verbose=verbose
            )

            if 'pca' in pb_adata.uns:
                adata.uns["X_DR_expression_variance"] = pb_adata.uns['pca']['variance']
                adata.uns["X_DR_expression_variance_ratio"] = pb_adata.uns['pca']['variance_ratio']
                pseudobulk_anndata.uns["X_DR_expression_variance"] = pb_adata.uns['pca']['variance']
                pseudobulk_anndata.uns["X_DR_expression_variance_ratio"] = pb_adata.uns['pca']['variance_ratio']

                if verbose:
                    print(f"[PCA] Stored variance information in both objects")

        except Exception as e:
            raise RuntimeError(f"PCA computation failed: {str(e)}")

    if verbose:
        print(f"[DimRed] Completed - results stored as X_DR_expression in both adata and pseudobulk_anndata")


def run_dimension_reduction_proportion(
    adata: sc.AnnData,
    pseudobulk: dict,
    pseudobulk_anndata: sc.AnnData = None,
    sample_col: str = "sample",
    batch_col: Union[str, List[str], None] = None,
    harmony_for_proportion: bool = False,
    preserve_cols: Optional[Union[str, List[str]]] = None,
    n_components: int = 10,
    verbose: bool = False,
) -> None:
    """
    Performs dimension reduction on cell proportion data with small dataset handling.

    UPDATED BEHAVIOR (as requested):
      - If preserve_cols is None: keep original behavior (PCA -> optional Harmony)
      - If preserve_cols is not None: try limma-style correction (remove batch, preserve covariates),
        then PCA. If limma fails, fall back to original PCA/Harmony path.

    IMPORTANT FIX:
      - utils.limma.limma() in your project expects exprs shaped as (samples x features).
        Proportions are (samples x celltypes), so we pass exprs WITHOUT transposing.

    Notes:
      - Keeps original output keys and storage behavior the same:
          adata.uns["X_DR_proportion"], adata.uns["X_DR_proportion_variance_ratio"]
          pseudobulk_anndata.uns["X_DR_proportion"], pseudobulk_anndata.obsm["X_DR_proportion"], ...
      - Keeps the original PCA/Harmony logic the same when not in limma mode or if limma fails.
    """
    # -------------------------
    # Validate inputs
    # -------------------------
    if "cell_proportion" not in pseudobulk:
        raise KeyError("Missing 'cell_proportion' key in pseudobulk dictionary.")

    proportion_df = pseudobulk["cell_proportion"].copy()
    proportion_df = proportion_df.fillna(0)

    # Normalize proportion sample IDs (keep stable normalization)
    proportion_df.index = proportion_df.index.astype(str).str.strip().str.lower()

    # Validate components
    n_samples, n_features = proportion_df.shape
    max_components = min(n_samples - 1, n_features)  # PCA needs <= n_samples-1

    if n_components > max_components:
        if verbose:
            print(f"[run_dimension_reduction_proportion] Adjusting n_components from {n_components} to {max_components}")
        n_components = max_components

    if n_components <= 0:
        raise ValueError(f"Insufficient data dimensions (samples={n_samples}, features={n_features})")

    # Normalize preserve_cols
    preserve_list: Optional[List[str]]
    if preserve_cols is None:
        preserve_list = None
    else:
        preserve_list = [preserve_cols] if isinstance(preserve_cols, str) else list(preserve_cols)
        preserve_list = [str(c) for c in preserve_list]

    # -------------------------
    # LIMMA MODE (only if preserve_cols provided)
    # -------------------------
    if preserve_list is not None:
        if verbose:
            print(f"[run_dimension_reduction_proportion] Limma mode requested (preserve_cols={preserve_list})")

        if pseudobulk_anndata is None:
            if verbose:
                print("[run_dimension_reduction_proportion] Limma requested but pseudobulk_anndata is None. Falling back.")
            preserve_list = None  # fall back
        elif batch_col is None:
            if verbose:
                print("[run_dimension_reduction_proportion] Limma requested but batch_col is None. Falling back.")
            preserve_list = None  # fall back
        else:
            try:
                # -------------------------
                # Determine batch column to remove (supports list -> combined)
                # -------------------------
                if isinstance(batch_col, list):
                    valid_b = [b for b in batch_col if b in pseudobulk_anndata.obs.columns]
                    if not valid_b:
                        raise ValueError(f"No valid batch columns found in pseudobulk_anndata.obs from batch_col={batch_col}")
                    batch_col_to_use = "_combined_batch_for_proportion_"
                    pseudobulk_anndata.obs[batch_col_to_use] = pseudobulk_anndata.obs[valid_b].astype(str).agg("|".join, axis=1)
                else:
                    batch_col_to_use = batch_col
                    if batch_col_to_use not in pseudobulk_anndata.obs.columns:
                        raise ValueError(f"batch_col '{batch_col_to_use}' not found in pseudobulk_anndata.obs")

                # -------------------------
                # Validate preserve columns exist
                # -------------------------
                preserve_present = [c for c in preserve_list if c in pseudobulk_anndata.obs.columns]
                preserve_missing = [c for c in preserve_list if c not in pseudobulk_anndata.obs.columns]

                if verbose:
                    if preserve_missing:
                        print(f"[run_dimension_reduction_proportion] Warning: preserve cols missing in obs: {preserve_missing}")
                        print("  (Common mistake: passing a category value instead of a column name.)")

                if len(preserve_present) == 0:
                    raise ValueError("preserve_cols was provided but none exist in pseudobulk_anndata.obs")

                preserve_list = preserve_present  # keep only valid

                # Fill NaNs in covariates
                for col in [batch_col_to_use] + preserve_list:
                    if pseudobulk_anndata.obs[col].isnull().any():
                        pseudobulk_anndata.obs[col] = pseudobulk_anndata.obs[col].fillna("Unknown")

                # -------------------------
                # Align sample IDs between proportion_df and pseudobulk_anndata
                # -------------------------
                prop_ids = proportion_df.index.astype(str).str.strip().str.lower()
                obs_ids = pseudobulk_anndata.obs_names.astype(str).str.strip().str.lower()

                obs_map = {}
                for orig, norm in zip(pseudobulk_anndata.obs_names, obs_ids):
                    if norm not in obs_map:
                        obs_map[norm] = orig

                common = [sid for sid in prop_ids if sid in obs_map]
                if len(common) < 2:
                    raise ValueError(
                        f"Not enough overlapping samples between proportion_df (n={len(prop_ids)}) "
                        f"and pseudobulk_anndata (n={len(obs_ids)}). overlap={len(common)}"
                    )

                # Subset in proportion order
                P = proportion_df.loc[common].copy()              # (samples x features)
                pheno_index = [obs_map[sid] for sid in common]    # original obs_names order aligned to P
                pheno = pseudobulk_anndata.obs.loc[pheno_index].copy()

                # exprs for utils.limma: (samples x features)  <-- DO NOT TRANSPOSE
                X = P.to_numpy(dtype=float)

                if verbose:
                    print(f"[run_dimension_reduction_proportion] Limma input: exprs={X.shape} (samples x features), pheno_rows={pheno.shape[0]}")
                    bc_counts = pheno[batch_col_to_use].value_counts()
                    print(f"[run_dimension_reduction_proportion] Batch '{batch_col_to_use}': n_batches={len(bc_counts)}, "
                          f"sizes={bc_counts.min()}-{bc_counts.max()}")
                    print(f"[run_dimension_reduction_proportion] Preserving columns: {preserve_list}")

                # sanity: pheno rows must equal samples
                if pheno.shape[0] != X.shape[0]:
                    raise ValueError(f"pheno rows ({pheno.shape[0]}) != exprs samples ({X.shape[0]}) after alignment")

                # -------------------------
                # Build patsy formulas
                # -------------------------
                terms = [f'Q("{c}")' for c in preserve_list]
                keep_formula = "~ " + " + ".join(terms) if terms else "1"
                remove_formula = f'~ Q("{batch_col_to_use}")'

                # -------------------------
                # Run limma
                # -------------------------
                from utils.limma import limma

                Xcorr = limma(
                    pheno=pheno,
                    exprs=X,  # (samples x features) expected by your utils.limma
                    covariate_formula=keep_formula,
                    design_formula=remove_formula,
                    rcond=1e-8,
                    verbose=False,
                )

                P_corr = np.asarray(Xcorr, dtype=float)  # (samples x features)

                if np.isnan(P_corr).any() or np.isinf(P_corr).any():
                    if verbose:
                        print("[run_dimension_reduction_proportion] Warning: NaN/Inf in limma output; replacing with 0")
                    P_corr = np.nan_to_num(P_corr, nan=0.0, posinf=0.0, neginf=0.0)

                # PCA on corrected proportions
                if verbose:
                    print(f"[run_dimension_reduction_proportion] Computing PCA on limma-corrected proportions with {n_components} components...")
                pca = PCA(n_components=n_components, random_state=42)
                pca_coords = pca.fit_transform(P_corr)

                # Store PCA intermediate (same as original behavior)
                if pseudobulk_anndata is not None:
                    pseudobulk_anndata.obsm["X_pca_proportion"] = pca_coords
                    pseudobulk_anndata.uns["pca_proportion_variance_ratio"] = pca.explained_variance_ratio_

                # Final outputs
                final_coords = pca_coords
                final_df = pd.DataFrame(
                    data=final_coords,
                    index=P.index,  # normalized sample IDs (lowercased) consistent with proportion_df
                    columns=[f"PC{i+1}" for i in range(final_coords.shape[1])],
                )

                adata.uns["X_DR_proportion"] = final_df
                adata.uns["X_DR_proportion_variance_ratio"] = pca.explained_variance_ratio_

                pseudobulk_anndata.uns["X_DR_proportion"] = final_df
                pseudobulk_anndata.obsm["X_DR_proportion"] = final_coords
                pseudobulk_anndata.uns["X_DR_proportion_variance_ratio"] = pca.explained_variance_ratio_

                if verbose:
                    print(f"[run_dimension_reduction_proportion] Completed using limma+PCA. Shape: {final_df.shape}")

                return  # IMPORTANT: keep original flow; stop if limma succeeded

            except Exception as e:
                if verbose:
                    print(f"[run_dimension_reduction_proportion] Limma failed: {e}. Falling back to original PCA/Harmony path.")
                # fall through to original PCA/Harmony path

    # -------------------------
    # ORIGINAL PCA + (optional) HARMONY PATH (unchanged)
    # -------------------------
    # Check if dataset is too small for Harmony
    MIN_SAMPLES_FOR_HARMONY = 10
    if harmony_for_proportion and n_samples < MIN_SAMPLES_FOR_HARMONY:
        if verbose:
            print(f"[run_dimension_reduction_proportion] Warning: Only {n_samples} samples. "
                  f"Skipping Harmony (requires >={MIN_SAMPLES_FOR_HARMONY} samples)")
        harmony_for_proportion = False

    # Step 1: PCA
    if verbose:
        print(f"[run_dimension_reduction_proportion] Computing PCA with {n_components} components...")

    pca = PCA(n_components=n_components, random_state=42)
    pca_coords = pca.fit_transform(proportion_df)

    if pseudobulk_anndata is not None:
        pseudobulk_anndata.obsm["X_pca_proportion"] = pca_coords
        pseudobulk_anndata.uns["pca_proportion_variance_ratio"] = pca.explained_variance_ratio_

    # Step 2: Harmony (if applicable)
    final_coords = pca_coords

    if harmony_for_proportion and batch_col and pseudobulk_anndata is not None:
        batch_cols = [batch_col] if isinstance(batch_col, str) else batch_col

        missing_cols = [col for col in batch_cols if col not in pseudobulk_anndata.obs.columns]
        if missing_cols:
            if verbose:
                print(f"[run_dimension_reduction_proportion] Warning: batch column(s) {missing_cols} not found")
            batch_cols = [col for col in batch_cols if col in pseudobulk_anndata.obs.columns]

        if batch_cols:
            try:
                import harmonypy as hm

                if verbose:
                    print(f"[run_dimension_reduction_proportion] Applying Harmony with batch column(s): {batch_cols}")

                valid_batch_cols = []
                for col in batch_cols:
                    n_batches = pseudobulk_anndata.obs[col].nunique()
                    if n_batches < 2:
                        if verbose:
                            print(f"[run_dimension_reduction_proportion] Skipping batch column '{col}': only {n_batches} batch(es)")
                    else:
                        valid_batch_cols.append(col)

                if valid_batch_cols:
                    min_batches = min(pseudobulk_anndata.obs[col].nunique() for col in valid_batch_cols)
                    nclust = max(2, min(min_batches, n_samples // 2))

                    harmony_out = hm.run_harmony(
                        pca_coords.T,
                        pseudobulk_anndata.obs,
                        valid_batch_cols,
                        max_iter_harmony=30,
                        nclust=nclust,
                    )
                    final_coords = harmony_out.Z_corr.T

                    if verbose:
                        print(f"[run_dimension_reduction_proportion] Harmony completed with {len(valid_batch_cols)} batch factor(s). Shape: {final_coords.shape}")
                else:
                    if verbose:
                        print("[run_dimension_reduction_proportion] No valid batch columns for Harmony. Using PCA only.")

            except Exception as e:
                if verbose:
                    print(f"[run_dimension_reduction_proportion] Harmony failed: {e}. Using PCA only.")
                final_coords = pca_coords

    # Step 3: Create DataFrame
    final_df = pd.DataFrame(
        data=final_coords,
        index=proportion_df.index,
        columns=[f"PC{i+1}" for i in range(final_coords.shape[1])],
    )

    # Step 4: Store results
    adata.uns["X_DR_proportion"] = final_df
    adata.uns["X_DR_proportion_variance_ratio"] = pca.explained_variance_ratio_

    if pseudobulk_anndata is not None:
        pseudobulk_anndata.uns["X_DR_proportion"] = final_df
        pseudobulk_anndata.obsm["X_DR_proportion"] = final_coords
        pseudobulk_anndata.uns["X_DR_proportion_variance_ratio"] = pca.explained_variance_ratio_

    if verbose:
        method = "Harmony-integrated PCA" if (final_coords is not pca_coords) else "PCA"
        print(f"[run_dimension_reduction_proportion] Completed using {method}. Shape: {final_df.shape}")



def _save_anndata_with_detailed_error_handling(file_path, adata, object_name, verbose=False):
    """Save an AnnData object with detailed error handling and reporting."""
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            if verbose:
                print(f"[Save] Created directory: {directory}")

        if verbose:
            print(f"[Save] Saving {object_name} to: {file_path}")

        sc.write(file_path, adata)

        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > 0:
                if verbose:
                    print(f"[Save] ✓ Successfully saved {object_name} ({file_size / (1024*1024):.1f} MB)")
                return True
            else:
                if verbose:
                    print(f"[Save] ✗ File was created but has zero size: {file_path}")
                return False
        else:
            if verbose:
                print(f"[Save] ✗ File was not created: {file_path}")
            return False

    except Exception as e:
        if verbose:
            print(f"[Save] ✗ Unexpected error saving {object_name}: {str(e)}")
        return False


def dimension_reduction(
    adata: sc.AnnData,
    pseudobulk: dict,
    pseudobulk_anndata: sc.AnnData,
    sample_col: str = 'sample',
    n_expression_components: int = 10,
    n_proportion_components: int = 10,
    batch_col: Union[str, List[str], None] = None,
    harmony_for_proportion: bool = False,
    preserve_cols: Optional[Union[str, List[str]]] = None,
    output_dir: str = "./",
    integrated_data: bool = False,
    not_save: bool = False,
    atac: bool = False,
    use_snapatac2_dimred: bool = False,
    verbose: bool = True
) -> None:
    """
    Computes dimension reduction for both cell expression and cell proportion data.

    Change: added preserve_cols and passed it to run_dimension_reduction_proportion().
    All other behavior unchanged; saving still uses pseudobulk/pseudobulk_sample.h5ad.
    """
    start_time = time.time() if verbose else None

    if "cell_expression_corrected" not in pseudobulk or "cell_proportion" not in pseudobulk:
        raise KeyError("Missing required keys ('cell_expression_corrected' or 'cell_proportion') in pseudobulk.")

    if pseudobulk_anndata is None:
        raise ValueError("pseudobulk_anndata parameter is required.")

    if verbose:
        print("[process_anndata_with_dimension_reduction] Starting dimension reduction computation...")
        print("[process_anndata_with_dimension_reduction] Results will be stored under unified keys:")
        print("  - X_DR_expression: for expression data")
        print("  - X_DR_proportion: for proportion data")
        if atac:
            if use_snapatac2_dimred:
                print("[process_anndata_with_dimension_reduction] ATAC mode: Will try snapATAC2 spectral first, fallback to LSI if needed")
            else:
                print("[process_anndata_with_dimension_reduction] ATAC mode: Will use LSI for expression data")
        else:
            print("[process_anndata_with_dimension_reduction] RNA mode: Will use PCA for expression data")

        if batch_col:
            if isinstance(batch_col, list):
                print(f"[process_anndata_with_dimension_reduction] Batch correction: Using {len(batch_col)} batch column(s): {batch_col}")
            else:
                print(f"[process_anndata_with_dimension_reduction] Batch correction: Using batch column: {batch_col}")

        if preserve_cols is not None:
            print(f"[process_anndata_with_dimension_reduction] Proportion correction: preserve_cols={preserve_cols} (limma mode)")

    output_dir = os.path.abspath(output_dir)
    pseudobulk_output_dir = os.path.join(output_dir, "pseudobulk")

    if not not_save:
        try:
            os.makedirs(pseudobulk_output_dir, exist_ok=True)
            if verbose:
                print(f"[process_anndata_with_dimension_reduction] ✓ Created output directories:")
                print(f"  - pseudobulk_output_dir: {pseudobulk_output_dir}")
        except Exception as e:
            if verbose:
                print(f"[process_anndata_with_dimension_reduction] ✗ Failed to create output directories: {str(e)}")
            if not verbose:
                print(f"ERROR: Cannot create output directories: {str(e)}")
            raise

    sample_proportion_df = pseudobulk["cell_proportion"]

    n_expression_components = min(n_expression_components, min(pseudobulk_anndata.shape) - 1)
    n_proportion_components = min(n_proportion_components, min(sample_proportion_df.shape))

    if verbose:
        print(f"[process_anndata_with_dimension_reduction] Using n_expression_components={n_expression_components}, n_proportion_components={n_proportion_components}")
        print(f"[process_anndata_with_dimension_reduction] Data dimensions: expression={pseudobulk_anndata.shape}, proportion={sample_proportion_df.shape}")

    expression_dr_successful = False
    proportion_dr_successful = False
    expression_error = None
    proportion_error = None

    try:
        run_dimension_reduction_expression(
            adata=adata,
            pseudobulk_anndata=pseudobulk_anndata,
            n_components=n_expression_components,
            atac=atac,
            use_snapatac2_dimred=use_snapatac2_dimred,
            verbose=verbose
        )
        expression_dr_successful = True
        if verbose:
            print("[process_anndata_with_dimension_reduction] ✓ Expression dimension reduction completed successfully")
    except Exception as e:
        expression_error = str(e)
        if verbose:
            print(f"[process_anndata_with_dimension_reduction] ✗ Expression dimension reduction failed: {expression_error}")

    try:
        run_dimension_reduction_proportion(
            adata=adata,
            pseudobulk=pseudobulk,
            pseudobulk_anndata=pseudobulk_anndata,
            sample_col=sample_col,
            n_components=n_proportion_components,
            batch_col=batch_col,
            harmony_for_proportion=harmony_for_proportion,
            preserve_cols=preserve_cols,
            verbose=verbose
        )
        proportion_dr_successful = True
        if verbose:
            print("[process_anndata_with_dimension_reduction] ✓ Proportion dimension reduction completed successfully")
    except Exception as e:
        proportion_error = str(e)
        if verbose:
            print(f"[process_anndata_with_dimension_reduction] ✗ Proportion dimension reduction failed: {proportion_error}")

    if not expression_dr_successful and not proportion_dr_successful:
        error_msg = "Both expression and proportion dimension reduction failed.\n"
        if expression_error:
            error_msg += f"Expression error: {expression_error}\n"
        if proportion_error:
            error_msg += f"Proportion error: {proportion_error}"
        raise RuntimeError(error_msg)

    if not not_save:
        if verbose:
            print("[process_anndata_with_dimension_reduction] Preparing to save results...")

        pb_filename = 'pseudobulk_sample.h5ad'
        pb_adata_path = os.path.join(pseudobulk_output_dir, pb_filename)

        if verbose:
            print(f"[process_anndata_with_dimension_reduction] Target file path:")
            print(f"  - pseudobulk_anndata: {pb_adata_path}")

        pb_save_success = _save_anndata_with_detailed_error_handling(
            pb_adata_path, pseudobulk_anndata, "pseudobulk_anndata", verbose
        )

        if pb_save_success:
            if verbose:
                print("[process_anndata_with_dimension_reduction] ✓ File saved successfully")
        else:
            if verbose:
                print("[process_anndata_with_dimension_reduction] ✗ File save failed")
            else:
                print("WARNING: Failed to save processed data file")

    if verbose and start_time is not None:
        elapsed_time = time.time() - start_time
        print(f"\n[process_anndata_with_dimension_reduction] === SUMMARY ===")
        print(f"Total runtime: {elapsed_time:.2f} seconds")
        print(f"Expression dimension reduction: {'✓ SUCCESS' if expression_dr_successful else '✗ FAILED'}")
        print(f"Proportion dimension reduction: {'✓ SUCCESS' if proportion_dr_successful else '✗ FAILED'}")

        if expression_dr_successful or proportion_dr_successful:
            print(f"Results available under unified keys:")
            if expression_dr_successful:
                print(f"  - X_DR_expression in both adata.uns and pseudobulk_anndata.uns")
            if proportion_dr_successful:
                print(f"  - X_DR_proportion in both adata.uns and pseudobulk_anndata.uns")

        if not not_save:
            print(f"File saving: ATTEMPTED (pseudobulk_anndata only)")

    return pseudobulk_anndata