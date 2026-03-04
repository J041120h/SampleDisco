import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse as sparse
from muon import atac as ac
from typing import Union, List, Optional


def _handle_nan_inf(X: np.ndarray, verbose: bool = False, context: str = "") -> np.ndarray:
    """Replace NaN/Inf values with 0 and report if verbose."""
    nan_mask = np.isnan(X) | np.isinf(X)
    if np.any(nan_mask):
        if verbose:
            print(f"[{context}] Found {np.sum(nan_mask)} NaN/inf values, replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _ensure_dense(X) -> np.ndarray:
    """Convert sparse matrix to dense array if needed."""
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _create_dr_dataframe(coords: np.ndarray, index, prefix: str = "PC") -> pd.DataFrame:
    """Create a DataFrame from dimension reduction coordinates."""
    return pd.DataFrame(
        data=coords,
        index=index,
        columns=[f"{prefix}{i+1}" for i in range(coords.shape[1])]
    )


def _validate_components(n_samples: int, n_features: int, n_components: int, verbose: bool = False) -> int:
    """Validate and adjust n_components based on data dimensions."""
    max_components = min(n_samples - 1, n_features - 1)
    if n_components > max_components:
        if verbose:
            print(f"[Validation] Adjusting n_components from {n_components} to {max_components}")
        n_components = max_components
    if n_components <= 0:
        raise ValueError(f"Cannot perform dimension reduction: insufficient data dimensions "
                        f"(samples={n_samples}, features={n_features}).")
    return n_components


def _manual_lsi(pb_adata: sc.AnnData, n_components: int, verbose: bool = False) -> Optional[pd.DataFrame]:
    """Manual LSI implementation using TruncatedSVD with TF-IDF."""
    try:
        if verbose:
            print("[LSI] Attempting manual LSI with TF-IDF")

        X = _ensure_dense(pb_adata.X)
        
        if verbose:
            print(f"[LSI] Original shape: {X.shape}")

        nan_genes = np.isnan(X).any(axis=0)
        if np.sum(nan_genes) > 0:
            if verbose:
                print(f"[LSI] Found {np.sum(nan_genes)} genes with NaN values, dropping them")
            X = X[:, ~nan_genes]
            if verbose:
                print(f"[LSI] Shape after dropping NaN genes: {X.shape}")

        n_components = _validate_components(X.shape[0], X.shape[1] + 1, n_components, verbose)

        tfidf = TfidfTransformer(norm='l2', use_idf=True, sublinear_tf=True)
        X_tfidf = tfidf.fit_transform(X)

        if sparse.issparse(X_tfidf):
            nan_mask = np.isnan(X_tfidf.data) | np.isinf(X_tfidf.data)
            if np.any(nan_mask):
                if verbose:
                    print(f"[LSI] Found {np.sum(nan_mask)} NaN/inf values in TF-IDF data, replacing with 0")
                X_tfidf.data[nan_mask] = 0.0
                X_tfidf.eliminate_zeros()

        X_tfidf_dense = _handle_nan_inf(_ensure_dense(X_tfidf), verbose, "LSI")

        zero_samples = np.sum(X_tfidf_dense, axis=1) == 0
        if np.any(zero_samples):
            if verbose:
                print(f"[LSI] Found {np.sum(zero_samples)} samples with all-zero TF-IDF values, adding noise")
            for i in np.where(zero_samples)[0]:
                X_tfidf_dense[i, :] = np.random.normal(0, 1e-10, X_tfidf_dense.shape[1])

        if verbose:
            print(f"[LSI] TF-IDF matrix stats: shape={X_tfidf_dense.shape}, "
                  f"min={X_tfidf_dense.min():.6f}, max={X_tfidf_dense.max():.6f}")

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        lsi_coords = _handle_nan_inf(svd.fit_transform(X_tfidf_dense), verbose, "LSI")

        lsi_df = _create_dr_dataframe(lsi_coords, pb_adata.obs_names, "LSI")

        if verbose:
            print(f"[LSI] Manual LSI computation successful. Shape: {lsi_df.shape}")

        return lsi_df

    except Exception as e:
        if verbose:
            print(f"[LSI] Manual LSI failed: {str(e)}")
        return None


def run_lsi_expression(
    pseudobulk_anndata: sc.AnnData,
    n_components: int = 10,
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """Performs LSI (Latent Semantic Indexing) on pseudobulk expression data for ATAC-seq."""
    if verbose:
        print(f"[LSI] Computing LSI with {n_components} components on {pseudobulk_anndata.shape} data")

    pb_adata = pseudobulk_anndata.copy()
    n_components = _validate_components(*pb_adata.shape, n_components, verbose)

    try:
        if sparse.issparse(pb_adata.X):
            pb_adata.X = pb_adata.X.tocsr()

        ac.pp.tfidf(pb_adata, scale_factor=1e4)
        sc.tl.lsi(pb_adata, n_comps=n_components, random_state=42)
        
        lsi_df = _create_dr_dataframe(pb_adata.obsm['X_lsi'], pb_adata.obs_names, "LSI")

        if verbose:
            print(f"[LSI] Success. Shape: {lsi_df.shape}")
        return lsi_df

    except Exception as e:
        if verbose:
            print(f"[LSI] Scanpy LSI failed: {str(e)}")
        return _manual_lsi(pb_adata, n_components, verbose)


def _store_dr_results(adata: sc.AnnData, pseudobulk_anndata: sc.AnnData, 
                      key: str, df_result: pd.DataFrame, verbose: bool = False) -> None:
    """Store dimension reduction results in both objects."""
    adata.uns[key] = df_result
    pseudobulk_anndata.uns[key] = df_result
    pseudobulk_anndata.obsm[key] = df_result.values
    if verbose:
        print(f"[Storage] Stored {key} in both objects (shape: {df_result.shape})")


def run_dimension_reduction_expression(
    adata: sc.AnnData,
    pseudobulk_anndata: sc.AnnData,
    n_components: int = 10,
    atac: bool = False,
    verbose: bool = False
) -> None:
    """
    Performs dimension reduction on pseudobulk-corrected expression data.
    Results stored under 'X_DR_expression'.
    """
    if pseudobulk_anndata is None:
        raise ValueError("pseudobulk_anndata parameter is required.")

    method = "LSI" if atac else "PCA"
    if verbose:
        print(f"[DimRed] Computing {method} for {'ATAC' if atac else 'RNA'} data with "
              f"{n_components} components on {pseudobulk_anndata.shape} data")

    pb_adata = pseudobulk_anndata.copy()
    n_components = _validate_components(*pb_adata.shape, n_components, verbose)

    if atac:
        lsi_df = run_lsi_expression(pseudobulk_anndata, n_components, verbose)
        if lsi_df is None:
            raise RuntimeError("LSI method failed for ATAC data")
        _store_dr_results(adata, pseudobulk_anndata, "X_DR_expression", lsi_df, verbose)
    else:
        try:
            if pb_adata.X.max() > 100:
                sc.pp.log1p(pb_adata)

            sc.tl.pca(pb_adata, n_comps=n_components, svd_solver='arpack', random_state=42)
            pca_df = _create_dr_dataframe(pb_adata.obsm['X_pca'], pb_adata.obs_names, "PC")

            if verbose:
                print(f"[PCA] Success. Shape: {pca_df.shape}")

            _store_dr_results(adata, pseudobulk_anndata, "X_DR_expression", pca_df, verbose)

        except Exception as e:
            raise RuntimeError(f"PCA computation failed: {str(e)}")

    if verbose:
        print(f"[DimRed] Completed - results stored as X_DR_expression")


def _align_samples(proportion_df: pd.DataFrame, pseudobulk_anndata: sc.AnnData) -> tuple:
    """Align samples between proportion DataFrame and pseudobulk AnnData."""
    prop_ids = proportion_df.index.astype(str).str.strip().str.lower()
    obs_ids = pseudobulk_anndata.obs_names.astype(str).str.strip().str.lower()

    obs_map = {norm: orig for orig, norm in zip(pseudobulk_anndata.obs_names, obs_ids)}
    
    common = [sid for sid in prop_ids if sid in obs_map]
    if len(common) < 2:
        raise ValueError(f"Not enough overlapping samples (n={len(common)})")

    P = proportion_df.loc[common].copy()
    pheno = pseudobulk_anndata.obs.loc[[obs_map[sid] for sid in common]].copy()

    return P, pheno


def _prepare_batch_column(pseudobulk_anndata: sc.AnnData, batch_col: Union[str, List[str]]) -> str:
    """Prepare and validate batch column(s), combining if multiple."""
    if isinstance(batch_col, list):
        valid_b = [b for b in batch_col if b in pseudobulk_anndata.obs.columns]
        if not valid_b:
            raise ValueError(f"No valid batch columns found from batch_col={batch_col}")
        batch_col_to_use = "_combined_batch_for_proportion_"
        pseudobulk_anndata.obs[batch_col_to_use] = (
            pseudobulk_anndata.obs[valid_b].astype(str).agg("|".join, axis=1)
        )
    else:
        batch_col_to_use = batch_col
        if batch_col_to_use not in pseudobulk_anndata.obs.columns:
            raise ValueError(f"batch_col '{batch_col_to_use}' not found in pseudobulk_anndata.obs")
    return batch_col_to_use


def _run_limma_correction(P: pd.DataFrame, pheno: pd.DataFrame, batch_col: str,
                          preserve_list: List[str], n_components: int, 
                          verbose: bool = False) -> np.ndarray:
    """Run limma correction and PCA on proportion data."""
    from utils.limma import limma

    for col in [batch_col] + preserve_list:
        if pheno[col].isnull().any():
            pheno[col] = pheno[col].fillna("Unknown")

    X = P.to_numpy(dtype=float)

    if verbose:
        print(f"[Limma] Input: exprs={X.shape}, pheno_rows={pheno.shape[0]}")
        bc_counts = pheno[batch_col].value_counts()
        print(f"[Limma] Batch '{batch_col}': n_batches={len(bc_counts)}, sizes={bc_counts.min()}-{bc_counts.max()}")

    terms = [f'Q("{c}")' for c in preserve_list]
    keep_formula = "~ " + " + ".join(terms) if terms else "1"

    Xcorr = limma(pheno=pheno, exprs=X, covariate_formula=keep_formula,
                  design_formula=f'~ Q("{batch_col}")', rcond=1e-8, verbose=False)

    P_corr = _handle_nan_inf(np.asarray(Xcorr, dtype=float), verbose, "Limma")

    if verbose:
        print(f"[Limma] Computing PCA on corrected proportions with {n_components} components...")
    
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(P_corr)


def _run_harmony(pca_coords: np.ndarray, pseudobulk_anndata: sc.AnnData,
                 batch_cols: List[str], n_samples: int, verbose: bool = False) -> np.ndarray:
    """Run Harmony batch correction on PCA coordinates."""
    import harmonypy as hm

    valid_batch_cols = [col for col in batch_cols 
                        if col in pseudobulk_anndata.obs.columns 
                        and pseudobulk_anndata.obs[col].nunique() >= 2]

    if not valid_batch_cols:
        if verbose:
            print("[Harmony] No valid batch columns. Using PCA only.")
        return pca_coords

    if verbose:
        print(f"[Harmony] Applying with batch column(s): {valid_batch_cols}")

    min_batches = min(pseudobulk_anndata.obs[col].nunique() for col in valid_batch_cols)

    harmony_out = hm.run_harmony(
        pca_coords.T,
        pseudobulk_anndata.obs,
        valid_batch_cols,
        max_iter_harmony=30,
        nclust=max(2, min(min_batches, n_samples // 2)),
    )

    if verbose:
        print(f"[Harmony] Completed with {len(valid_batch_cols)} batch factor(s)")

    return harmony_out.Z_corr.T


def run_dimension_reduction_proportion(
    adata: sc.AnnData,
    pseudobulk: dict,
    pseudobulk_anndata: sc.AnnData = None,
    sample_col: str = "sample",
    batch_col: Union[str, List[str], None] = None,
    harmony_for_proportion: bool = False,
    preserve_cols_in_sample_embedding: Optional[Union[str, List[str]]] = None,
    n_components: int = 10,
    verbose: bool = False,
) -> None:
    """
    Performs dimension reduction on cell proportion data.
    
    If preserve_cols_in_sample_embedding is provided: uses limma-style correction then PCA.
    Otherwise: uses PCA with optional Harmony batch correction.
    """
    if "cell_proportion" not in pseudobulk:
        raise KeyError("Missing 'cell_proportion' key in pseudobulk dictionary.")

    proportion_df = pseudobulk["cell_proportion"].copy().fillna(0)
    proportion_df.index = proportion_df.index.astype(str).str.strip().str.lower()

    n_samples, n_features = proportion_df.shape
    n_components = min(n_components, n_samples - 1, n_features)

    if n_components <= 0:
        raise ValueError(f"Insufficient data dimensions (samples={n_samples}, features={n_features})")

    preserve_list = None
    if preserve_cols_in_sample_embedding is not None:
        preserve_list = [preserve_cols_in_sample_embedding] if isinstance(preserve_cols_in_sample_embedding, str) else list(preserve_cols_in_sample_embedding)
        preserve_list = [str(c) for c in preserve_list]

    if preserve_list is not None and pseudobulk_anndata is not None and batch_col is not None:
        if verbose:
            print(f"[Proportion] Limma mode (preserve_cols_in_sample_embedding={preserve_list})")

        try:
            batch_col_to_use = _prepare_batch_column(pseudobulk_anndata, batch_col)

            preserve_list = [c for c in preserve_list if c in pseudobulk_anndata.obs.columns]
            if not preserve_list:
                raise ValueError("None of preserve_cols_in_sample_embedding exist in pseudobulk_anndata.obs")

            P, pheno = _align_samples(proportion_df, pseudobulk_anndata)
            
            pca_coords = _run_limma_correction(P, pheno, batch_col_to_use, preserve_list, n_components, verbose)

            final_df = _create_dr_dataframe(pca_coords, P.index, "PC")
            _store_dr_results(adata, pseudobulk_anndata, "X_DR_proportion", final_df, verbose)

            if verbose:
                print(f"[Proportion] Completed using limma+PCA. Shape: {final_df.shape}")
            return

        except Exception as e:
            if verbose:
                print(f"[Proportion] Limma failed: {e}. Falling back to PCA/Harmony.")

    MIN_SAMPLES_FOR_HARMONY = 10
    if harmony_for_proportion and n_samples < MIN_SAMPLES_FOR_HARMONY:
        if verbose:
            print(f"[Proportion] Only {n_samples} samples. Skipping Harmony.")
        harmony_for_proportion = False

    if verbose:
        print(f"[Proportion] Computing PCA with {n_components} components...")

    pca = PCA(n_components=n_components, random_state=42)
    pca_coords = pca.fit_transform(proportion_df)
    final_coords = pca_coords

    if harmony_for_proportion and batch_col and pseudobulk_anndata is not None:
        batch_cols = [batch_col] if isinstance(batch_col, str) else batch_col
        batch_cols = [col for col in batch_cols if col in pseudobulk_anndata.obs.columns]

        if batch_cols:
            try:
                final_coords = _run_harmony(pca_coords, pseudobulk_anndata, batch_cols, n_samples, verbose)
            except Exception as e:
                if verbose:
                    print(f"[Proportion] Harmony failed: {e}. Using PCA only.")

    final_df = _create_dr_dataframe(final_coords, proportion_df.index, "PC")
    _store_dr_results(adata, pseudobulk_anndata, "X_DR_proportion", final_df, verbose)

    if verbose:
        method = "Harmony-integrated PCA" if (final_coords is not pca_coords) else "PCA"
        print(f"[Proportion] Completed using {method}. Shape: {final_df.shape}")


def _save_embedding_csv(embedding_data, output_path: str, embedding_name: str, 
                        verbose: bool = False) -> bool:
    """Save embedding data to CSV file."""
    try:
        if isinstance(embedding_data, np.ndarray):
            df = pd.DataFrame(embedding_data)
        elif isinstance(embedding_data, pd.DataFrame):
            df = embedding_data
        else:
            if verbose:
                print(f"[SaveCSV] Skipping {embedding_name}: unsupported type {type(embedding_data)}")
            return False

        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        df.to_csv(output_path)

        if verbose:
            print(f"[SaveCSV] Saved {embedding_name}: {output_path} (shape: {df.shape})")
        return True

    except Exception as e:
        if verbose:
            print(f"[SaveCSV] Failed to save {embedding_name}: {str(e)}")
        return False


def _save_embeddings_to_csv(pseudobulk_anndata: sc.AnnData, output_dir: str, verbose: bool = False) -> None:
    """Save unified sample-level embeddings to CSV."""
    embedding_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(embedding_dir, exist_ok=True)

    embeddings = [
        ("X_DR_expression", "sample_expression_embedding.csv", "sample expression embedding"),
        ("X_DR_proportion", "sample_proportion_embedding.csv", "sample proportion embedding"),
    ]

    saved = 0
    for key, filename, name in embeddings:
        if key in pseudobulk_anndata.uns:
            if _save_embedding_csv(pseudobulk_anndata.uns[key], os.path.join(embedding_dir, filename), name, verbose):
                saved += 1
        elif verbose:
            print(f"[SaveEmbeddings] {key} not found; skipping")

    if verbose:
        print(f"[SaveEmbeddings] Saved {saved} embedding file(s) to: {embedding_dir}")


def _save_anndata(file_path: str, adata: sc.AnnData, object_name: str, 
                  verbose: bool = False) -> bool:
    """Save AnnData object with error handling."""
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if verbose:
            print(f"[Save] Saving {object_name} to: {file_path}")

        sc.write(file_path, adata)

        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            if verbose:
                print(f"[Save] Successfully saved {object_name} ({os.path.getsize(file_path) / (1024 * 1024):.1f} MB)")
            return True

        if verbose:
            print(f"[Save] File not created or empty: {file_path}")
        return False

    except Exception as e:
        if verbose:
            print(f"[Save] Error saving {object_name}: {str(e)}")
        return False


def dimension_reduction(
    adata: sc.AnnData,
    pseudobulk: dict,
    pseudobulk_anndata: sc.AnnData,
    sample_col: str = 'sample',
    n_expression_components: int = 10,
    n_proportion_components: int = 10,
    batch_col: Union[str, List[str], None] = None,
    harmony_for_proportion: bool = True,
    preserve_cols_in_sample_embedding: Optional[Union[str, List[str]]] = None,
    output_dir: str = "./",
    not_save: bool = False,
    atac: bool = False,
    verbose: bool = True
) -> sc.AnnData:
    """
    Computes dimension reduction for both cell expression and cell proportion data.
    Saves embeddings to CSV files and AnnData to h5ad.
    """
    start_time = time.time() if verbose else None

    required_keys = ["cell_expression_corrected", "cell_proportion"]
    missing = [k for k in required_keys if k not in pseudobulk]
    if missing:
        raise KeyError(f"Missing required keys in pseudobulk: {missing}")

    if pseudobulk_anndata is None:
        raise ValueError("pseudobulk_anndata parameter is required.")

    if verbose:
        print("[DimRed] Starting dimension reduction computation...")
        print(f"[DimRed] Mode: {'ATAC (LSI)' if atac else 'RNA (PCA)'}")
        if batch_col:
            print(f"[DimRed] Batch column(s): {batch_col}")
        if preserve_cols_in_sample_embedding:
            print(f"[DimRed] Preserve columns (limma mode): {preserve_cols_in_sample_embedding}")

    output_dir = os.path.abspath(output_dir)
    pseudobulk_output_dir = os.path.join(output_dir, "pseudobulk")

    if not not_save:
        os.makedirs(pseudobulk_output_dir, exist_ok=True)
        if verbose:
            print(f"[DimRed] Output directory: {pseudobulk_output_dir}")

    n_expression_components = min(n_expression_components, min(pseudobulk_anndata.shape) - 1)
    n_proportion_components = min(n_proportion_components, min(pseudobulk["cell_proportion"].shape))

    if verbose:
        print(f"[DimRed] Components: expression={n_expression_components}, proportion={n_proportion_components}")

    results = {"expression": False, "proportion": False}
    errors = {}

    try:
        run_dimension_reduction_expression(adata, pseudobulk_anndata, n_expression_components, atac, verbose)
        results["expression"] = True
        if verbose:
            print("[DimRed] Expression dimension reduction completed")
    except Exception as e:
        errors["expression"] = str(e)
        if verbose:
            print(f"[DimRed] Expression failed: {e}")

    try:
        run_dimension_reduction_proportion(
            adata, pseudobulk, pseudobulk_anndata, sample_col,
            batch_col, harmony_for_proportion, preserve_cols_in_sample_embedding, n_proportion_components, verbose
        )
        results["proportion"] = True
        if verbose:
            print("[DimRed] Proportion dimension reduction completed")
    except Exception as e:
        errors["proportion"] = str(e)
        if verbose:
            print(f"[DimRed] Proportion failed: {e}")

    if not any(results.values()):
        error_msg = "Both dimension reductions failed.\n" + "\n".join(f"{k}: {v}" for k, v in errors.items())
        raise RuntimeError(error_msg)

    if not not_save:
        try:
            _save_embeddings_to_csv(pseudobulk_anndata, output_dir, verbose)
        except Exception as e:
            if verbose:
                print(f"[DimRed] Warning: CSV save failed: {e}")

        _save_anndata(os.path.join(pseudobulk_output_dir, 'pseudobulk_sample.h5ad'), 
                     pseudobulk_anndata, "pseudobulk_anndata", verbose)

    if verbose and start_time:
        print(f"\n[DimRed] === SUMMARY ===")
        print(f"Runtime: {time.time() - start_time:.2f}s")
        print(f"Expression: {'Success' if results['expression'] else 'Failed'}")
        print(f"Proportion: {'Success' if results['proportion'] else 'Failed'}")

    return pseudobulk_anndata