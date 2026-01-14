import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from scipy import stats
from scipy import linalg
from scipy.optimize import root_scalar
from scipy.special import digamma, gamma, polygamma
import warnings
from multiprocessing import cpu_count
from functools import partial
import traceback
from joblib import Parallel, delayed


# ---------------------------------------------------------------------
#  Trigamma utilities
# ---------------------------------------------------------------------
def trigamma(x):
    """Wrapper around scipy.special.polygamma(1, x)."""
    try:
        return polygamma(1, x)
    except Exception as e:
        print(f"ERROR in trigamma: {e}")
        traceback.print_exc()
        raise


def trigamma_inverse(x):
    """Inverse of trigamma via Newton iterations."""
    try:
        if x <= 0:
            raise ValueError("trigamma_inverse requires x > 0")

        if x >= 1e7:
            return x / 2.0

        y = 0.5 + 1.0 / x          # starting guess
        tol = 1e-8

        for _ in range(100):
            delta = (trigamma(y) - x) / polygamma(2, y)
            y_old = y
            y -= delta
            if y <= 0:              # keep positive
                y = y_old / 2.0
            if abs(delta) < tol:
                break
        return y
    except Exception as e:
        print(f"ERROR in trigamma_inverse: {e}")
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------
#  Laguerre–Gauss nodes/weights
# ---------------------------------------------------------------------
def laguerre_quadrature(n=500, verbose=True):
    """Gauss–Laguerre quadrature rule with fallback for high n values."""
    import warnings
    from numpy.polynomial.laguerre import laggauss
    
    # Try with the requested number of points
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            nodes, weights = laggauss(n)
    except (RuntimeWarning, ValueError, OverflowError) as e:
        # If overflow occurs, try with fewer points
        if verbose:
            print(f"Warning: Overflow in Laguerre quadrature with n={n}, trying with n=100")
        try:
            nodes, weights = laggauss(100)
        except (RuntimeWarning, ValueError, OverflowError):
            # Final fallback
            if verbose:
                print("Warning: Using n=50 for Laguerre quadrature")
            nodes, weights = laggauss(50)
    
    return nodes, weights


# ---------------------------------------------------------------------
#  Main RAISIN function
# ---------------------------------------------------------------------
def raisinfit(
    adata_path,
    sample_col,
    batch_key=None,
    sample_to_clade=None,
    group_col=None,
    verbose=True,
    intercept=True,
    n_jobs=None,
):
    """
    Python port of RAISIN differential-expression model.

    Parameters
    ----------
    adata_path : str
        Path to AnnData object. Can be single-cell or pseudobulk.
    sample_col : str
        Column in adata.obs that identifies which sample each observation belongs to.
    batch_key : str, optional
        Column in adata.obs used as batch when no explicit group is provided.
    sample_to_clade : dict, optional
        Mapping {sample_id -> clade/group}. If provided, overrides `group_col`.
    group_col : str, optional
        Column in adata.obs containing sample-level grouping/clustering information
        (e.g. clade, condition). For each sample, the most frequent value of
        `group_col` across observations belonging to that sample will be used
        as its group/feature.
    verbose : bool
        Print progress.
    intercept : bool
        Include intercept in fixed-effect design matrix.
    n_jobs : int, optional
        Number of CPU cores for parallel per-gene estimation. If None or -1,
        use all available cores.
    """

    try:
        if verbose:
            print("\n===== Starting RAISIN fitting =====")

        # -----------------------------------------------------------------
        #  Thread count
        # -----------------------------------------------------------------
        if n_jobs in (None, -1):
            n_jobs = cpu_count()
        if verbose:
            print(f"Using {n_jobs} CPU cores")

        # -----------------------------------------------------------------
        #  Input argument sanity checks
        # -----------------------------------------------------------------
        if sample_to_clade is not None and group_col is not None:
            raise ValueError(
                "Provide at most one of `sample_to_clade` or `group_col`, not both."
            )

        # -----------------------------------------------------------------
        #  Load data
        # -----------------------------------------------------------------
        if verbose:
            print(f"Loading AnnData from {adata_path}")
        adata = sc.read(adata_path)
        
        # Print available columns to help users select the correct sample_col / group_col
        if verbose:
            print("Available columns in adata.obs:", list(adata.obs.columns))
            
        # Check if sample_col exists in adata.obs
        if sample_col not in adata.obs.columns:
            available_cols = list(adata.obs.columns)
            error_msg = (
                f"Error: Column '{sample_col}' not found in adata.obs. "
                f"Available columns are: {available_cols}"
            )
            raise KeyError(error_msg)

        if group_col is not None and group_col not in adata.obs.columns:
            available_cols = list(adata.obs.columns)
            error_msg = (
                f"Error: group_col '{group_col}' not found in adata.obs. "
                f"Available columns are: {available_cols}"
            )
            raise KeyError(error_msg)

        # -----------------------------------------------------------------
        #  Expression matrix
        # -----------------------------------------------------------------
        if adata.raw is not None and adata.raw.X is not None:
            expr = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else adata.raw.X
            gene_names = adata.raw.var_names
            if verbose:
                print("Using raw counts from adata.raw.X")
        else:
            expr = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
            gene_names = adata.var_names
            if verbose:
                print("Using counts from adata.X")

        # transpose to genes x cells/observations
        expr = expr.T

        # sample IDs per cell/observation
        sample = adata.obs[sample_col].values

        # deduplicate genes
        if len(gene_names) != len(set(gene_names)):
            if verbose:
                print("Removing duplicated gene names")
            _, keep_idx = np.unique(gene_names, return_index=True)
            expr = expr[keep_idx, :]
            gene_names = gene_names[keep_idx]

        # -----------------------------------------------------------------
        #  Build design
        # -----------------------------------------------------------------
        if sample_to_clade is not None:
            # -------------------------------------------------------------
            #  Case 1: User provided explicit mapping sample -> clade
            # -------------------------------------------------------------
            testtype = "unpaired"

            # keep only samples actually present in the data
            samples_in_data = np.unique(sample)
            mapping_keys = np.array(list(sample_to_clade.keys()))
            common = np.intersect1d(samples_in_data, mapping_keys)

            if verbose:
                print(f"Using sample_to_clade mapping for {len(common)} samples")
                missing = np.setdiff1d(samples_in_data, common)
                if missing.size > 0:
                    print(
                        f"Warning: {missing.size} samples in data not in sample_to_clade; "
                        f"they will be dropped."
                    )

            # mask cells/obs belonging to samples we keep
            valid = np.isin(sample, common)
            expr = expr[:, valid]
            sample = sample[valid]

            # Build design from the mapping but restricted to samples present
            common_list = common.tolist()
            design = pd.DataFrame({
                "sample": common_list,
                "feature": [sample_to_clade[s] for s in common_list],
            })

        else:
            # -------------------------------------------------------------
            #  Case 2: Build design from obs metadata
            # -------------------------------------------------------------
            testtype = "unpaired"
            uniq_samples = np.unique(sample)

            if group_col is not None:
                # -----------------------------------------
                # User-provided grouping column (preferred)
                # -----------------------------------------
                if verbose:
                    print(f"Using group_col='{group_col}' in adata.obs for sample groups")

                group_val = []
                for s in uniq_samples:
                    mask = sample == s
                    vals = adata.obs.loc[mask, group_col].values
                    if vals.size == 0:
                        raise ValueError(
                            f"No rows in adata.obs for sample '{s}' when using group_col='{group_col}'"
                        )
                    # For single-cell or pseudobulk, this collapses to the dominant value
                    most_common = pd.Series(vals).value_counts().idxmax()
                    group_val.append(most_common)

                design = pd.DataFrame({
                    "sample": uniq_samples,
                    "feature": group_val,
                })

            else:
                # -----------------------------------------
                # Fallback: use batch_key or dummy single group
                # -----------------------------------------
                if batch_key:
                    if batch_key not in adata.obs.columns:
                        raise KeyError(
                            f"batch_key '{batch_key}' not found in adata.obs. "
                            f"Available columns are: {list(adata.obs.columns)}"
                        )
                    batch_val = []
                    for s in uniq_samples:
                        mask = sample == s
                        batches = adata.obs[batch_key].values[mask]
                        if batches.size == 0:
                            raise ValueError(
                                f"No rows in adata.obs for sample '{s}' when using batch_key='{batch_key}'"
                            )
                        batch_val.append(pd.Series(batches).value_counts().idxmax())
                    design = pd.DataFrame({
                        "sample": uniq_samples,
                        "feature": batch_val
                    })
                    if verbose:
                        print(f"Using batch_key='{batch_key}' as grouping feature")
                else:
                    # All samples in one group
                    design = pd.DataFrame({
                        "sample": uniq_samples,
                        "feature": ["group1"] * len(uniq_samples)
                    })
                    if verbose:
                        print("No sample_to_clade, group_col, or batch_key provided; "
                              "all samples assigned to a single group 'group1'.")

        sample_names = design["sample"].values
        X_feature = design["feature"].values

        # ----- Z (random-effect dummy) -----
        Z_df = pd.get_dummies(design["sample"], prefix="", prefix_sep="")
        Z = Z_df.values
        group = design.loc[design["sample"].isin(Z_df.columns), "feature"].values
        group_names = np.array([str(g) for g in group])

        # ----- X (fixed effect) -----
        if intercept:
            X = pd.get_dummies(X_feature, drop_first=False)
            X.insert(0, "intercept", 1)
        else:
            X = pd.get_dummies(X_feature, drop_first=True)
        X = X.values

        # -----------------------------------------------------------------
        #  Per-sample means
        # -----------------------------------------------------------------
        G = expr.shape[0]
        means = np.zeros((G, len(sample_names)))
        for i, s in enumerate(sample_names):
            mask = sample == s
            if mask.any():
                means[:, i] = np.mean(expr[:, mask], axis=1)

        # -----------------------------------------------------------------
        #  Quadrature nodes
        # -----------------------------------------------------------------
        node, weight = laguerre_quadrature(500, verbose=verbose)  # Reduced from 1000 to 500
        pos = weight > 0
        node, weight = node[pos], weight[pos]
        log_node, log_weight = np.log(node), np.log(weight)

        # -----------------------------------------------------------------
        #  Cell-level variance w
        # -----------------------------------------------------------------
        w = np.zeros((G, len(sample_names)))
        for i, s in enumerate(sample_names):
            idx = np.where(sample == s)[0]
            n_cells = len(idx)
            if n_cells > 1:
                d = n_cells - 1
                s2 = (np.mean(expr[:, idx] ** 2, 1) - means[:, i] ** 2) * ((d + 1) / d)
                ok = s2 > 0
                if ok.any():
                    stat = np.var(np.log(s2[ok])) - trigamma(d / 2)
                    if stat > 0:
                        theta = trigamma_inverse(stat)
                        phi = np.exp(np.mean(np.log(s2[ok])) - digamma(d / 2) + digamma(theta)) * d / 2
                        if theta + d / 2 > 1:
                            w[:, i] = (d * s2 / 2 + phi) / (theta + d / 2 - 1)
                        else:
                            alpha = theta + d / 2
                            beta = d * s2 / 2 + phi
                            for g in range(G):
                                if s2[g] > 0:
                                    integrand = np.exp(node - alpha * log_node - beta[g] / node + log_weight)
                                    w[g, i] = (beta[g] ** alpha / gamma(alpha)) * np.sum(integrand)
                    else:
                        w[:, i] = np.exp(np.mean(np.log(s2[ok])))
            else:
                w[:, i] = np.nan

        # fill missing w using nearest neighbor in design space
        nan_cols = np.where(np.isnan(w).all(0))[0]
        ok_cols = np.setdiff1d(np.arange(w.shape[1]), nan_cols)
        if nan_cols.size and ok_cols.size:
            X_dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1) ** 2
            for i in nan_cols:
                nearest = ok_cols[np.argmin(X_dist[i, ok_cols])]
                w[:, i] = w[:, nearest]

        # per-cell normalization
        n_per_sample = np.array([(sample == s).sum() for s in sample_names])
        w /= n_per_sample

        # -----------------------------------------------------------------
        #  Initialise σ² container
        # -----------------------------------------------------------------
        unique_groups = np.unique(group_names)
        sigma2 = pd.DataFrame(0.0, index=range(G), columns=unique_groups)

        # -----------------------------------------------------------------
        #  Helper: variance component estimator
        # -----------------------------------------------------------------
        def sigma2_func(current_group, control_groups, done_groups,
                        X, Z, means, w, group, G, n_jobs,
                        node, log_node, log_weight):

            mask_ctrl = np.isin(group, control_groups)
            Xl = np.hstack([X, Z[:, mask_ctrl]])

            mask_curr = group == current_group
            Zl = Z[:, mask_curr]

            lid = np.where((Z[:, np.isin(group, [current_group] + control_groups)]).sum(1) > 0)[0]
            if lid.size == 0:
                warnings.warn(f"No data for variance of group {current_group}")
                return np.zeros(G)

            # reduce rank of Xl
            R = np.linalg.qr(Xl[lid], mode="r")
            rank = (np.abs(np.diag(R)) > 1e-10).sum()
            Xl = Xl[lid, :rank]

            n = lid.size
            p = n - Xl.shape[1]
            if p == 0:
                warnings.warn(f"Unable to estimate variance for group {current_group}")
                return np.zeros(G)

            K = np.random.normal(size=(n, p)).astype(np.float64)
            Xl = Xl.astype(np.float64)
            for i in range(p):
                b = Xl if i == 0 else np.hstack([Xl, K[:, :i]])
                b_transpose = b.T
                solve = np.linalg.lstsq(np.matmul(b_transpose, b), np.matmul(b_transpose, K[:, i]), rcond=None)[0]
                K[:, i] -= np.matmul(b, solve)
            K /= np.linalg.norm(K, axis=0, keepdims=True)
            K = K.T  # K shape becomes (p x n)

            # Get subset of Zl that matches the dimensions in lid
            Zl_lid = Z[lid][:, mask_curr]  # (n x num_curr_samples)

            pl = np.matmul(K, means[:, lid].T)                   # (p x G)
            
            # Compute qlm in steps to avoid dimension mismatch
            K_Zl = np.matmul(K, Zl_lid)                          # (p x num_curr_samples)
            Zl_T_K_T = np.matmul(Zl_lid.T, K.T)                  # (num_curr_samples x p)
            qlm = np.matmul(K_Zl, Zl_T_K_T)                      # (p x p)
            
            ql = np.diag(qlm)                                    # (p,)
            rl = np.matmul(w[:, lid], (K ** 2).T)                # (G x p)

            # already-pooled groups
            for sg in done_groups:
                mask = group == sg
                Z_lid_mask = Z[lid][:, group_names == sg]
                Z_lid_mask_T = Z_lid_mask.T
                
                KZ_temp = np.matmul(K, Z_lid_mask)
                Z_lid_mask_KT = np.matmul(Z_lid_mask_T, K.T)
                KZ = np.matmul(KZ_temp, Z_lid_mask_KT)
                
                for g in range(G):
                    rl[g] += sigma2.loc[g, sg] * np.diag(KZ)

            # moments
            pl2 = pl ** 2
            rl_T = rl.T
            M = np.mean(np.maximum(0, (pl2 - rl_T) / ql[:, None]))
            V = np.mean(np.maximum(0, (pl2 ** 2 - 3 * rl_T ** 2 - 6 * M * ql[:, None] * rl_T) /
                                    (3 * ql[:, None] ** 2)))

            try:
                alpha = M ** 2 / (V - M ** 2)
                gamma_ = M / (V - M ** 2)
                if any(np.isnan([alpha, gamma_])) or alpha <= 0 or gamma_ <= 0:
                    raise ValueError("Invalid hyper-parameters")

                # ----------------------------------------------------------
                #  Worker: EB estimate per gene
                # ----------------------------------------------------------
                def process_gene(g):
                    x_mat = np.outer(pl[:, g], pl[:, g])
                    w_g = w[g, lid]  # Shape (lid.size,)

                    # reshape for broadcasting
                    w_g_reshaped = w_g.reshape(1, -1)  # (1, n)
                    K_weighted = K * w_g_reshaped      # (p, n)
                    t2 = np.matmul(K_weighted, K.T)    # (p, p)
                    
                    res = np.zeros_like(node)
                    for i, gn in enumerate(node):
                        cm = gn * qlm + t2
                        try:
                            chol = np.linalg.cholesky(cm)
                            logdet = 2 * np.log(np.diag(chol)).sum()
                            inv_cm = np.linalg.inv(cm)
                        except np.linalg.LinAlgError:
                            eig = np.linalg.eigvalsh(cm)
                            logdet = np.log(eig).sum()
                            inv_cm = np.linalg.inv(cm)
                        res[i] = -logdet - np.sum(x_mat * inv_cm)
                    tmp = log_weight + node + res / 2 + (alpha - 1) * log_node - gamma_ * node
                    num = np.exp(tmp + log_node)
                    den = np.exp(tmp)
                    est = np.sum(num) / np.sum(den)
                    if not np.isfinite(est):
                        mv = tmp.max()
                        est = np.sum(np.exp(tmp + log_node - mv)) / np.sum(np.exp(tmp - mv))
                    return est

                # ---------- PARALLEL HERE ----------
                if n_jobs > 1:
                    est = Parallel(n_jobs=n_jobs)(
                        delayed(process_gene)(g) for g in range(G)
                    )
                else:
                    est = [process_gene(g) for g in range(G)]
                return np.array(est)

            except Exception as e:
                # -----------------------------------------------------------------
                #  Fallback: method-of-moments root-finding
                # -----------------------------------------------------------------
                if verbose:
                    print(f"Error in variance estimation ({current_group}): {e}\nProceeding without EB pooling")

                def root_func(g, s2):
                    return np.sum((s2 * ql ** 2 + ql * rl[g] - pl2[:, g] * ql) /
                                  (s2 * ql + rl[g]) ** 2)

                def process_gene_simple(g):
                    if np.allclose(pl[:, g], 0):
                        return 0.0
                    try:
                        sol = root_scalar(
                            lambda s: root_func(g, s),
                            bracket=[0, 1000],
                            method="brentq"
                        )
                        return sol.root
                    except Exception:
                        return 0.0

                # ---------- PARALLEL HERE ----------
                if n_jobs > 1:
                    est = Parallel(n_jobs=n_jobs)(
                        delayed(process_gene_simple)(g) for g in range(G)
                    )
                else:
                    est = [process_gene_simple(g) for g in range(G)]
                return np.array(est)

        # -----------------------------------------------------------------
        #  Iterate through groups
        # -----------------------------------------------------------------
        control_groups = list(unique_groups)
        done_groups = []
        n_para = {ug: (Z[:, group_names == ug] != 0).sum() for ug in unique_groups}
        for ug in sorted(unique_groups, key=lambda u: n_para[u], reverse=True):
            if verbose:
                print(f"\n===== Estimating variance component for: {ug} =====")
            sigma2[ug] = sigma2_func(
                ug,
                [g for g in control_groups if g != ug],
                done_groups,
                X, Z, means, w, group_names, G, n_jobs,
                node, log_node, log_weight
            )
            control_groups.remove(ug)
            done_groups.append(ug)

        # -----------------------------------------------------------------
        #  Assemble output
        # -----------------------------------------------------------------
        result = {
            "mean":   pd.DataFrame(means,  index=gene_names, columns=sample_names),
            "sigma2": sigma2,
            "omega2": pd.DataFrame(w,      index=gene_names, columns=sample_names),
            "X":      pd.DataFrame(X,      index=sample_names),
            "Z":      pd.DataFrame(Z,      index=sample_names),
            "group":  pd.Series(group_names, index=sample_names),
        }
        if verbose:
            print("\n===== Model fitting complete =====")
        return result

    except Exception as e:
        print(f"ERROR in raisinfit: {e}")
        traceback.print_exc()
        raise
