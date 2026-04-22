import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import issparse


def inspect_column(series: pd.Series, n_examples: int = 5) -> None:
    """Print detailed info about a single obs/var column."""
    dtype = series.dtype
    n_total = len(series)
    n_na = series.isna().sum()
    n_unique = series.nunique(dropna=True)

    print(f"      dtype: {dtype}")
    print(f"      non-NA: {n_total - n_na}/{n_total} ({100*(n_total-n_na)/n_total:.1f}%)")
    print(f"      unique: {n_unique}")

    if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
        # Categorical/string: show value counts
        vc = series.value_counts(dropna=False).head(n_examples)
        print(f"      top values:")
        for val, cnt in vc.items():
            pct = 100 * cnt / n_total
            print(f"        '{val}': {cnt} ({pct:.1f}%)")
    elif pd.api.types.is_numeric_dtype(series):
        # Numeric: show stats and examples
        print(f"      min: {series.min():.6g}, max: {series.max():.6g}, mean: {series.mean():.6g}")
        examples = series.dropna().unique()[:n_examples]
        print(f"      example values: {list(examples)}")
    elif pd.api.types.is_bool_dtype(series):
        vc = series.value_counts(dropna=False)
        print(f"      value counts: {dict(vc)}")
    else:
        # Fallback: just show some examples
        examples = series.dropna().unique()[:n_examples]
        print(f"      example values: {list(examples)}")


def _print_sparsity_info(mat, label: str = ".X") -> None:
    """
    Print sparsity information (nnz, total, density, sparsity) for a matrix.
    Works for both sparse and dense matrices.
    """
    try:
        n_rows, n_cols = mat.shape
        total = n_rows * n_cols
        if total == 0:
            print(f"  - {label} sparsity: empty matrix; skipping sparsity check.")
            return

        if issparse(mat):
            nnz = mat.nnz
        else:
            # convert to array view for count_nonzero, but avoid a full copy if possible
            arr = np.asarray(mat)
            nnz = int(np.count_nonzero(arr))

        density = nnz / total
        sparsity = 1.0 - density
        print(f"  - {label} nonzeros: {nnz} / {total} "
              f"({100*density:.4f}% nonzero, {100*sparsity:.4f}% zero)")
    except Exception as e:
        print(f"  - ⚠️ Could not compute sparsity for {label}: {e}")


def _maybe_print_unique_values(
    adata: sc.AnnData,
    column: str,
    where: str = "auto",
    max_unique: int | None = 200,
) -> None:
    """
    Optionally print *all* unique values for a requested column (in obs or var).

    Parameters
    ----------
    adata : AnnData
    column : str
        Column name to print unique values for.
    where : {"auto","obs","var"}
        Where to look for the column. "auto" checks obs then var.
    max_unique : int or None
        If not None, refuse to print if too many unique values (to avoid flooding logs).
        Set to None to always print all unique values.
    """
    if column is None:
        return

    if where not in {"auto", "obs", "var"}:
        print(f"\n⚠️ where='{where}' is invalid; use 'auto', 'obs', or 'var'. Skipping unique-values print.")
        return

    series = None
    found_in = None

    if where in {"auto", "obs"} and column in adata.obs.columns:
        series = adata.obs[column]
        found_in = "obs"
    elif where in {"auto", "var"} and column in adata.var.columns:
        series = adata.var[column]
        found_in = "var"

    if series is None:
        search_space = []
        if where in {"auto", "obs"}:
            search_space.append("obs")
        if where in {"auto", "var"}:
            search_space.append("var")
        print(f"\n⚠️ Column '{column}' not found in {', '.join(search_space)}.")
        return

    # Collect uniques (preserve NaN as a token in the output)
    uniques = pd.unique(series)
    # Make output stable-ish: sort when possible
    try:
        # keep NaN at the end if present
        uniques_no_na = [u for u in uniques if not pd.isna(u)]
        uniques_na = [u for u in uniques if pd.isna(u)]
        uniques_sorted = sorted(uniques_no_na)
        uniques = np.array(uniques_sorted + uniques_na, dtype=object)
    except Exception:
        pass

    n_unique = len(uniques)
    print("\n" + "="*60)
    print(f"🔎 Unique values for column '{column}' (found in adata.{found_in}):")
    print("="*60)
    print(f"  - n_unique: {n_unique}")

    if max_unique is not None and n_unique > max_unique:
        print(f"  - ⚠️ Too many unique values to print ({n_unique} > {max_unique}). "
              f"Increase max_unique or set max_unique=None to print all.")
        return

    for i, v in enumerate(uniques, start=1):
        # repr() keeps strings quoted and makes None/NaN obvious
        print(f"  {i:>4}. {repr(v)}")


def summarize_h5ad(
    h5ad_path: str,
    n_examples: int = 10,
    n_col_examples: int = 5,
    print_unique_column: str | None = None,
    unique_column_where: str = "auto",   # "auto" | "obs" | "var"
    unique_max_values: int | None = 200  # set None to print all
):
    """
    Summarize an AnnData .h5ad file by printing examples of cell names, obs, var,
    and inspecting .X (dtype, NaN/Inf, integer-like vs fractional, min/max, example values,
    and sparsity).
    Also prints sample values from any additional layers if present, and previews obsm/varm.

    NEW:
    ----
    User can optionally specify `print_unique_column` and all unique values of that column
    (from obs or var) will be printed.

    Parameters
    ----------
    h5ad_path : str
        Path to the .h5ad file.
    n_examples : int
        Number of example rows to show for matrices and dataframes.
    n_col_examples : int
        Number of example values to show per metadata column.
    print_unique_column : str or None
        If provided, print all unique values for this column (in obs/var).
    unique_column_where : {"auto","obs","var"}
        Where to look for `print_unique_column`.
    unique_max_values : int or None
        Guardrail to prevent printing an enormous number of uniques.
        Set to None to print all unique values regardless of count.
    """
    try:
        print(f"🔍 Loading AnnData from: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path, backed=None)

        print("\n📦 Basic Info")
        print(f"  - Shape (cells × genes): {adata.n_obs} × {adata.n_vars}")
        print(f"  - Layers: {list(adata.layers.keys()) if hasattr(adata, 'layers') else 'None'}")
        print(f"  - obs columns: {list(adata.obs.columns)}")
        print(f"  - var columns: {list(adata.var.columns)}")
        print(f"  - obsm keys: {list(adata.obsm.keys()) if hasattr(adata, 'obsm') else 'None'}")
        print(f"  - varm keys: {list(adata.varm.keys()) if hasattr(adata, 'varm') else 'None'}")

        # NEW: print unique values for a requested column (obs/var)
        _maybe_print_unique_values(
            adata=adata,
            column=print_unique_column,
            where=unique_column_where,
            max_unique=unique_max_values,
        )

        # 🔎 Inspect .X
        print("\n🔎 Inspecting .X matrix:")
        X = adata.X

        if issparse(X):
            print(f"  - storage type: sparse ({type(X).__name__})")
            dtype = X.dtype
            data_array_for_checks = X.data
        else:
            print(f"  - storage type: dense ({type(X).__name__})")
            dtype = X.dtype
            data_array_for_checks = np.asarray(X)

        print(f"  - dtype: {dtype}")

        # Sparsity info for .X
        _print_sparsity_info(X, label=".X")

        # Classify dtype using actual data content
        is_numeric = np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)
        if is_numeric:
            flat = data_array_for_checks.ravel()
            if flat.size == 0:
                print("  - value type: numeric (empty matrix, cannot inspect values)")
                is_integer_like = True
            else:
                max_check = min(100000, flat.size)
                sample = flat[:max_check]
                is_integer_like = np.allclose(sample, np.round(sample), atol=1e-8)

            if np.issubdtype(dtype, np.integer):
                msg = "integer"
            elif np.issubdtype(dtype, np.floating):
                msg = "float"
            elif np.issubdtype(dtype, np.bool_):
                msg = "bool"
            else:
                msg = "numeric (custom type)"

            if is_integer_like:
                msg += " — values appear integer-like (no fractional parts)"
            else:
                msg += " — fractional values detected"
            print(f"  - value type: {msg}")

        else:
            print("  - ⚠️ unsupported / non-numeric dtype (e.g. object/string)")

        # NaN / Inf checks
        if is_numeric:
            try:
                has_nan = bool(np.isnan(data_array_for_checks).any())
            except TypeError:
                has_nan = False
                print("  - ⚠️ np.isnan failed on this dtype; skipping NaN check")

            has_inf = False
            if np.issubdtype(dtype, np.floating):
                has_inf = bool(np.isinf(data_array_for_checks).any())

            print(f"  - contains NaN: {has_nan}")
            if np.issubdtype(dtype, np.floating):
                print(f"  - contains Inf: {has_inf}")
        else:
            print("  - Skipping NaN/Inf check due to non-numeric dtype.")

        # Min / Max values
        if is_numeric and data_array_for_checks.size > 0:
            try:
                min_val = float(np.nanmin(data_array_for_checks))
                max_val = float(np.nanmax(data_array_for_checks))
                print(f"  - min value: {min_val:.6g}")
                print(f"  - max value: {max_val:.6g}")
            except Exception as e:
                print(f"  - ⚠️ Could not compute min/max: {e}")

        # Example .X values
        n_rows = min(n_examples, adata.n_obs)
        n_cols = min(10, adata.n_vars)
        print(f"\n🧮 Example .X values (first {n_rows} cells × {n_cols} genes):")
        if issparse(X):
            X_sub = X[:n_rows, :n_cols].toarray()
        else:
            X_sub = np.asarray(X[:n_rows, :n_cols])
        print(X_sub)

        # 🔁 Inspect any additional layers
        if hasattr(adata, "layers") and len(adata.layers.keys()) > 0:
            print("\n📚 Inspecting additional layers:")
            for layer_name in adata.layers.keys():
                print(f"\n🔹 Layer '{layer_name}':")
                L = adata.layers[layer_name]

                if issparse(L):
                    print(f"  - storage type: sparse ({type(L).__name__})")
                    l_dtype = L.dtype
                    l_data_array_for_checks = L.data
                else:
                    print(f"  - storage type: dense ({type(L).__name__})")
                    L_arr = np.asarray(L)
                    l_dtype = L_arr.dtype
                    l_data_array_for_checks = L_arr

                print(f"  - dtype: {l_dtype}")

                # Sparsity info for this layer
                _print_sparsity_info(L, label=f"layer '{layer_name}'")

                # Only do light numeric checks here
                is_numeric_layer = np.issubdtype(l_dtype, np.number) or np.issubdtype(l_dtype, np.bool_)
                if is_numeric_layer and l_data_array_for_checks.size > 0:
                    try:
                        l_min_val = float(np.nanmin(l_data_array_for_checks))
                        l_max_val = float(np.nanmax(l_data_array_for_checks))
                        print(f"  - min value: {l_min_val:.6g}")
                        print(f"  - max value: {l_max_val:.6g}")
                    except Exception as e:
                        print(f"  - ⚠️ Could not compute min/max for layer '{layer_name}': {e}")
                elif is_numeric_layer:
                    print("  - numeric type but empty; skipping value checks.")
                else:
                    print("  - non-numeric dtype; skipping value checks.")

                # Example values from this layer
                n_rows_layer = min(n_examples, adata.n_obs)
                n_cols_layer = min(10, adata.n_vars)
                print(f"  - example values (first {n_rows_layer} cells × {n_cols_layer} genes):")
                if issparse(L):
                    L_sub = L[:n_rows_layer, :n_cols_layer].toarray()
                else:
                    L_sub = np.asarray(L[:n_rows_layer, :n_cols_layer])
                print(L_sub)

        # Example cell names
        print("\n🧫 Example cell names:")
        for name in adata.obs_names[:n_examples]:
            print("  -", name)

        # ════════════════════════════════════════════════════════════════
        # 📋 DETAILED OBS COLUMN INSPECTION
        # ════════════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("📋 Detailed obs columns inspection:")
        print("="*60)
        for col in adata.obs.columns:
            print(f"\n    [{col}]")
            inspect_column(adata.obs[col], n_examples=n_col_examples)

        print(f"\n📋 Example obs rows (head {n_examples}):")
        print(adata.obs.head(n_examples))

        # ════════════════════════════════════════════════════════════════
        # 🧬 DETAILED VAR COLUMN INSPECTION
        # ════════════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("🧬 Detailed var columns inspection:")
        print("="*60)
        for col in adata.var.columns:
            print(f"\n    [{col}]")
            inspect_column(adata.var[col], n_examples=n_col_examples)

        print(f"\n🧬 Example var rows (head {n_examples}):")
        print(adata.var.head(n_examples))

        # Example obsm
        if hasattr(adata, "obsm") and len(adata.obsm.keys()) > 0:
            print("\n📌 Example obsm entries:")
            for key, value in adata.obsm.items():
                try:
                    arr = np.asarray(value)
                except Exception as e:
                    print(f"  ▶ '{key}': could not convert to array ({e}); skipping preview.")
                    continue

                print(f"  ▶ '{key}': shape={arr.shape}, dtype={arr.dtype}")
                if arr.ndim == 2 and arr.size > 0:
                    r = min(n_examples, arr.shape[0])
                    c = min(5, arr.shape[1])
                    print(f"    first {r} rows × {c} cols:")
                    print(arr[:r, :c])
                else:
                    print("    (non-2D or empty; skipping preview)")

        # Example varm
        if hasattr(adata, "varm") and len(adata.varm.keys()) > 0:
            print("\n🧷 Example varm entries:")
            for key, value in adata.varm.items():
                try:
                    arr = np.asarray(value)
                except Exception as e:
                    print(f"  ▶ '{key}': could not convert to array ({e}); skipping preview.")
                    continue

                print(f"  ▶ '{key}': shape={arr.shape}, dtype={arr.dtype}")
                if arr.ndim == 2 and arr.size > 0:
                    r = min(n_examples, arr.shape[0])  # genes
                    c = min(5, arr.shape[1])
                    print(f"    first {r} genes × {c} cols:")
                    print(arr[:r, :c])
                else:
                    print("    (non-2D or empty; skipping preview)")

    except Exception as e:
        print(f"❌ Error reading {h5ad_path}: {e}")

if __name__ == "__main__":
    summarize_h5ad(
        h5ad_path='/dcs07/hongkai/data/harry/result/1M-scBloodNL/data/1M-scBloodNL.h5ad',
        # Optional: print unique values of a column in obs/var
        print_unique_column='cell_type',         # e.g., "cell_type" or "highly_variable"
        unique_column_where="auto",       # "auto" | "obs" | "var"
        unique_max_values=200             # set None to print all unique values
    )