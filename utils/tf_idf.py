import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, dia_matrix, issparse
from anndata import AnnData
from mudata import MuData
from warnings import warn
import gc

def tfidf_memory_efficient(
    data,
    log_tf: bool = True,
    log_idf: bool = True,
    log_tfidf: bool = False,
    scale_factor = 1e4,
    inplace: bool = True,
    copy: bool = False,
    from_layer = None,
    to_layer = None,
    chunk_size: int = 1000,
    verbose: bool = True
):
    """Chunk-wise sparse TF-IDF for scATAC-seq data.

    IDF convention: ``n_cells / document_frequency`` (no +1 smoothing), then
    optionally ``log1p``-transformed. This matches the muon/epiScanpy convention,
    not sklearn's smoothed variant.

    Parameters
    ----------
    data : AnnData or MuData
        Input. For MuData, the 'atac' modality is used.
    log_tf, log_idf : bool
        Apply log1p to TF and IDF respectively (default True each).
    log_tfidf : bool
        Apply log1p to the final product instead; mutually exclusive with log_tf/log_idf.
    scale_factor : float
        Multiply TF before log1p (default 1e4, analogous to CP10k).
    inplace : bool
        Write result back to adata.X (or to_layer). Default True.
    copy : bool
        Return a copy of the AnnData. Incompatible with inplace=False.
    from_layer, to_layer : str or None
        Read from / write to a named layer instead of .X.
    chunk_size : int
        Cells processed per IDF-multiplication chunk (memory vs speed tradeoff).
    """
    
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")
    
    if log_tfidf and (log_tf or log_idf):
        raise AttributeError("When returning log(TF*IDF), applying neither log(TF) nor log(IDF) is possible.")
    if copy and not inplace:
        raise ValueError("`copy=True` cannot be used with `inplace=False`.")
    if to_layer is not None and not inplace:
        raise ValueError(f"`to_layer='{str(to_layer)}'` cannot be used with `inplace=False`.")

    if copy:
        adata = adata.copy()

    counts = adata.X if from_layer is None else adata.layers[from_layer]

    if verbose:
        print(f"Processing {adata.n_obs} cells × {adata.n_vars} peaks")
        if issparse(counts):
            print(f"Input matrix sparsity: {1 - counts.nnz / (counts.shape[0] * counts.shape[1]):.3f}")
        else:
            print("Input matrix is dense")

    if not issparse(counts):
        if verbose:
            print("Converting to sparse matrix...")
        counts = csr_matrix(counts)

    if to_layer is not None and to_layer in adata.layers:
        warn(f"Existing layer '{str(to_layer)}' will be overwritten")

    if verbose:
        print("Computing TF...")

    cell_totals = np.asarray(counts.sum(axis=1)).flatten()
    cell_totals[cell_totals == 0] = 1  # avoid division by zero for empty cells

    tf = counts.copy().astype(np.float32)
    tf.data = tf.data.astype(np.float32)
    row_indices = tf.indptr
    
    for i in range(len(cell_totals)):
        start_idx = row_indices[i]
        end_idx = row_indices[i + 1]
        tf.data[start_idx:end_idx] /= cell_totals[i]
    
    if scale_factor is not None and scale_factor != 0 and scale_factor != 1:
        tf.data *= scale_factor

    if log_tf:
        tf.data = np.log1p(tf.data)

    if verbose:
        print("Computing IDF...")

    peak_totals = np.asarray(counts.sum(axis=0)).flatten()
    idf = adata.shape[0] / peak_totals  # raw n_cells/df, no +1 smoothing (muon convention)
    idf[peak_totals == 0] = 0

    if log_idf:
        idf = np.log1p(idf)

    if verbose:
        print("Computing TF-IDF...")

    tf_idf = tf.copy()
    n_cells = tf_idf.shape[0]

    for start_idx in range(0, n_cells, chunk_size):
        end_idx = min(start_idx + chunk_size, n_cells)
        chunk = tf_idf[start_idx:end_idx]
        chunk.data *= idf[chunk.indices]
        tf_idf[start_idx:end_idx] = chunk
        if verbose and start_idx % (chunk_size * 10) == 0:
            print(f"Processed {end_idx}/{n_cells} cells")

    if log_tfidf:
        tf_idf.data = np.log1p(tf_idf.data)

    tf_idf.data = np.nan_to_num(tf_idf.data, nan=0.0)

    del tf, counts
    gc.collect()

    if verbose:
        if issparse(tf_idf):
            print(f"Output matrix sparsity: {1 - tf_idf.nnz / (tf_idf.shape[0] * tf_idf.shape[1]):.3f}")
        else:
            print("Output matrix is dense")

    if not inplace:
        return tf_idf

    if to_layer is not None:
        adata.layers[to_layer] = tf_idf
    else:
        adata.X = tf_idf

    if copy:
        return adata


def run_scatac_pipeline_fixed(
    # ... all your existing parameters ...
    use_memory_efficient_tfidf=True,
    tfidf_chunk_size=1000,
    **kwargs
):
    if use_memory_efficient_tfidf:
        log("TF-IDF normalisation (memory-efficient)", verbose)
        try:
            tfidf_memory_efficient(
                atac,
                scale_factor=tfidf_scale_factor,
                chunk_size=tfidf_chunk_size,
                verbose=verbose
            )
        except MemoryError:
            log("Switching to ultra memory-efficient TF-IDF", verbose)
            tfidf_ultra_memory_efficient(
                atac,
                scale_factor=tfidf_scale_factor,
                verbose=verbose
            )
    else:
        ac.pp.tfidf(atac, scale_factor=tfidf_scale_factor)


def tfidf_ultra_memory_efficient(adata, scale_factor=1e4, log_tf=True, log_idf=True, verbose=True):
    """Single-cell-at-a-time TF-IDF fallback for extremely large datasets."""
    if verbose:
        print("Using ultra memory-efficient TF-IDF...")
    
    counts = adata.X
    if not issparse(counts):
        counts = csr_matrix(counts)

    n_cells, n_peaks = counts.shape

    # IDF is global; compute once before the per-cell loop.
    peak_totals = np.asarray(counts.sum(axis=0)).flatten()
    idf = n_cells / peak_totals
    idf[peak_totals == 0] = 0

    if log_idf:
        idf = np.log1p(idf)

    result_data = []
    result_indices = []
    result_indptr = [0]

    for i in range(n_cells):
        if verbose and i % 1000 == 0:
            print(f"Processing cell {i}/{n_cells}")

        cell_data = counts.getrow(i)

        if issparse(cell_data):
            if cell_data.nnz == 0:
                result_indptr.append(result_indptr[-1])
                continue

            cell_total = cell_data.sum()
            tf_values = cell_data.data / cell_total

            if scale_factor != 1:
                tf_values *= scale_factor

            if log_tf:
                tf_values = np.log1p(tf_values)

            tfidf_values = tf_values * idf[cell_data.indices]

            result_data.extend(tfidf_values)
            result_indices.extend(cell_data.indices)
            result_indptr.append(result_indptr[-1] + len(tfidf_values))
        else:
            cell_row = cell_data.toarray().flatten() if hasattr(cell_data, 'toarray') else cell_data.flatten()
            nonzero_indices = np.nonzero(cell_row)[0]

            if len(nonzero_indices) == 0:
                result_indptr.append(result_indptr[-1])
                continue

            cell_total = cell_row.sum()
            tf_values = cell_row[nonzero_indices] / cell_total

            if scale_factor != 1:
                tf_values *= scale_factor

            if log_tf:
                tf_values = np.log1p(tf_values)

            tfidf_values = tf_values * idf[nonzero_indices]

            result_data.extend(tfidf_values)
            result_indices.extend(nonzero_indices)
            result_indptr.append(result_indptr[-1] + len(tfidf_values))

    tf_idf = csr_matrix(
        (np.array(result_data), np.array(result_indices), np.array(result_indptr)),
        shape=(n_cells, n_peaks)
    )

    adata.X = tf_idf

    if verbose:
        print("Ultra memory-efficient TF-IDF complete")
    
    return adata


def safe_tfidf(adata, scale_factor=1e4, log_tf=True, log_idf=True, verbose=True):
    """TF-IDF dispatcher: chunk-wise for large matrices, single-cell fallback on OOM."""
    n_cells, n_peaks = adata.shape

    if verbose:
        print(f"Running safe TF-IDF on {n_cells} cells × {n_peaks} peaks")

    if n_cells * n_peaks < 1e6:
        if verbose:
            print("Using standard approach for small dataset")
        return tfidf_memory_efficient(
            adata,
            scale_factor=scale_factor,
            log_tf=log_tf,
            log_idf=log_idf,
            verbose=verbose
        )

    try:
        return tfidf_memory_efficient(
            adata,
            scale_factor=scale_factor,
            log_tf=log_tf,
            log_idf=log_idf,
            chunk_size=500,
            verbose=verbose
        )
    except (MemoryError, Exception) as e:
        if verbose:
            print(f"Memory-efficient failed ({e}), using ultra-efficient approach")
        return tfidf_ultra_memory_efficient(
            adata,
            scale_factor=scale_factor,
            log_tf=log_tf,
            log_idf=log_idf,
            verbose=verbose
        )