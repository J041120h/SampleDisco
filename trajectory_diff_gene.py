import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
from pygam import LinearGAM, s, f
from statsmodels.stats.multitest import multipletests
import warnings
import os
import datetime
import anndata as ad
from scipy.sparse import issparse, csr_matrix
import scanpy as sc

def prepare_trajectory_data(
    pseudobulk_adata: ad.AnnData,
    trajectory_results: Dict,
    trajectory_type: str = "TSCAN",  # "TSCAN" or "CCA"
    sample_col: str = "sample",
    verbose: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Extract pseudotime data from trajectory results and align with pseudobulk AnnData.
    
    Parameters
    ----------
    pseudobulk_adata : ad.AnnData
        Pseudobulk AnnData object (samples x genes)
    trajectory_results : Dict
        Results from TSCAN or CCA analysis
    trajectory_type : str
        Type of trajectory analysis ("TSCAN" or "CCA")
    sample_col : str
        Column name for sample identifiers
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    pseudotime_dict : Dict[str, Dict[str, float]]
        Dictionary containing pseudotime values for main and branching paths
    """
    pseudotime_dict = {}
    
    if trajectory_type == "TSCAN":
        # Extract TSCAN pseudotime
        if "pseudotime" in trajectory_results:
            # Main path
            if "main_path" in trajectory_results["pseudotime"]:
                pseudotime_dict["main_path"] = trajectory_results["pseudotime"]["main_path"]
                if verbose:
                    print(f"Found main path pseudotime for {len(pseudotime_dict['main_path'])} samples")
            
            # Branching paths
            if "branching_paths" in trajectory_results["pseudotime"]:
                pseudotime_dict["branching_paths"] = trajectory_results["pseudotime"]["branching_paths"]
                if verbose:
                    n_branches = len(pseudotime_dict["branching_paths"])
                    print(f"Found {n_branches} branching paths")
    
    elif trajectory_type == "CCA":
        # CCA returns pseudotime dictionaries directly
        # Expecting a tuple: (score_prop, score_expr, ptime_prop, ptime_expr)
        if len(trajectory_results) >= 4:
            # Use expression-based pseudotime as main path
            pseudotime_dict["main_path"] = trajectory_results[3]  # ptime_expression
            if verbose:
                print(f"Found CCA pseudotime for {len(pseudotime_dict['main_path'])} samples")
    
    # Validate pseudotime against pseudobulk samples
    pseudobulk_samples = set(pseudobulk_adata.obs.index.astype(str))
    
    # Check main path
    if "main_path" in pseudotime_dict:
        valid_samples = {}
        for sample, ptime in pseudotime_dict["main_path"].items():
            if str(sample) in pseudobulk_samples:
                valid_samples[str(sample)] = ptime
            elif verbose:
                print(f"Warning: Sample {sample} in pseudotime not found in pseudobulk data")
        
        pseudotime_dict["main_path"] = valid_samples
        
        if verbose:
            print(f"Validated main path: {len(valid_samples)} samples with pseudotime")
    
    # Check branching paths
    if "branching_paths" in pseudotime_dict:
        valid_branches = {}
        for branch_id, branch_ptime in pseudotime_dict["branching_paths"].items():
            valid_samples = {}
            for sample, ptime in branch_ptime.items():
                if str(sample) in pseudobulk_samples:
                    valid_samples[str(sample)] = ptime
            
            if valid_samples:  # Only keep branches with valid samples
                valid_branches[branch_id] = valid_samples
        
        pseudotime_dict["branching_paths"] = valid_branches
        
        if verbose:
            print(f"Validated {len(valid_branches)} branching paths")
    
    return pseudotime_dict


def prepare_gam_input_data_improved(
    pseudobulk_adata: ad.AnnData,
    ptime_expression: Dict[str, float],
    covariate_columns: Optional[List[str]] = None,
    sample_col: str = "sample",
    min_variance_threshold: float = 1e-6,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Improved version of prepare_gam_input_data with better handling of sparse matrices
    and data validation.
    """
    
    if ptime_expression is None or not ptime_expression:
        raise ValueError("Pseudotime expression values must be provided.")
    
    if verbose:
        print(f"Processing AnnData object with {pseudobulk_adata.n_obs} samples and {pseudobulk_adata.n_vars} genes")
        print(f"Processing pseudotime data for {len(ptime_expression)} samples")
    
    # Get sample metadata from .obs
    sample_meta = pseudobulk_adata.obs.copy()
    
    # Get sample names from index (pseudobulk data has samples as observations)
    sample_names = pseudobulk_adata.obs.index.astype(str)
    
    # Create lowercase versions for case-insensitive matching
    sample_names_lower = pd.Series(sample_names.str.lower(), index=sample_names)
    ptime_expression_lower = {k.lower(): v for k, v in ptime_expression.items()}
    
    # Find common samples
    common_samples_lower = set(sample_names_lower.values) & set(ptime_expression_lower.keys())
    
    if len(common_samples_lower) == 0:
        # Try exact matching
        common_samples = set(sample_names) & set(ptime_expression.keys())
        if len(common_samples) > 0:
            sample_mask = sample_names.isin(common_samples)
        else:
            raise ValueError(
                f"No common samples found between AnnData ({len(sample_names)} samples) "
                f"and pseudotime data ({len(ptime_expression)} samples). "
                f"First 5 AnnData samples: {list(sample_names[:5])}. "
                f"First 5 pseudotime samples: {list(ptime_expression.keys())[:5]}"
            )
    else:
        # Create mask for common samples using case-insensitive matching
        sample_mask = sample_names_lower.isin(common_samples_lower)
    
    if verbose:
        print(f"Found {sample_mask.sum()} common samples between AnnData and pseudotime data")
    
    # Filter AnnData to keep only common samples
    filtered_adata = pseudobulk_adata[sample_mask].copy()
    filtered_sample_names = sample_names[sample_mask]
    filtered_meta = sample_meta[sample_mask].copy()
    
    # Create expression DataFrame (samples x genes)
    # Handle both sparse and dense matrices
    if issparse(filtered_adata.X):
        if verbose:
            print(f"Converting sparse matrix (format: {type(filtered_adata.X).__name__}) to dense array")
        expression_matrix = filtered_adata.X.toarray()
    else:
        expression_matrix = np.array(filtered_adata.X)
    
    # Check for and handle NaN values
    if np.any(np.isnan(expression_matrix)):
        n_nan = np.sum(np.isnan(expression_matrix))
        if verbose:
            print(f"Warning: Found {n_nan} NaN values in expression matrix. Replacing with 0.")
        expression_matrix = np.nan_to_num(expression_matrix, nan=0.0)
    
    Y = pd.DataFrame(
        expression_matrix,
        index=filtered_sample_names,
        columns=filtered_adata.var_names
    )
    
    # Filter out genes with low variance
    gene_variances = Y.var(axis=0)
    low_var_genes = gene_variances < min_variance_threshold
    if low_var_genes.any():
        if verbose:
            print(f"Filtering out {low_var_genes.sum()} genes with variance < {min_variance_threshold}")
        Y = Y.loc[:, ~low_var_genes]
    
    # Add pseudotime values to metadata
    # Handle case-insensitive matching for pseudotime
    filtered_meta['pseudotime'] = np.nan
    for i, sample in enumerate(filtered_sample_names):
        # Try exact match first
        if sample in ptime_expression:
            filtered_meta.loc[sample, 'pseudotime'] = ptime_expression[sample]
        else:
            # Try case-insensitive match
            sample_lower = sample.lower()
            if sample_lower in ptime_expression_lower:
                filtered_meta.loc[sample, 'pseudotime'] = ptime_expression_lower[sample_lower]
    
    # Check for missing pseudotime values
    missing_ptime = filtered_meta['pseudotime'].isna().sum()
    if missing_ptime > 0:
        raise ValueError(f"Failed to assign pseudotime for {missing_ptime} samples")
    
    # Prepare design matrix X
    X = filtered_meta[['pseudotime']].copy()
    
    if covariate_columns is not None and len(covariate_columns) > 0:
        # Handle covariates from the pseudobulk AnnData metadata
        valid_covariate_columns = []
        
        for col in covariate_columns:
            if col in filtered_meta.columns and col != 'pseudotime':
                # Check if column has any non-null values
                if not filtered_meta[col].isna().all():
                    valid_covariate_columns.append(col)
                elif verbose:
                    print(f"Warning: Covariate column '{col}' has all null values, skipping")
            elif verbose and col != 'pseudotime':
                print(f"Warning: Covariate column '{col}' not found in metadata")
        
        if valid_covariate_columns:
            if verbose:
                print(f"Using covariates: {valid_covariate_columns}")
            
            # Get covariates
            covariates = filtered_meta[valid_covariate_columns].copy()
            
            # Handle categorical variables
            categorical_cols = covariates.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                if verbose:
                    print(f"Encoding categorical covariates: {list(categorical_cols)}")
                covariates_encoded = pd.get_dummies(covariates, columns=categorical_cols, drop_first=True)
            else:
                covariates_encoded = covariates
            
            # Combine pseudotime with encoded covariates
            X = pd.concat([X, covariates_encoded], axis=1)
    
    # Ensure X and Y have the same sample order
    X.index = Y.index
    
    # Get gene names
    gene_names = list(Y.columns)
    
    if verbose:
        print(f"Prepared input data:")
        print(f"  - {X.shape[0]} samples")
        print(f"  - {len(gene_names)} genes (after filtering)")
        print(f"  - {X.shape[1]} features in design matrix: {list(X.columns)}")
    
    return X, Y, gene_names

def fit_gam_models_for_genes(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    gene_names: List[str],
    *,
    spline_term: str = "pseudotime",
    num_splines: int = 5,
    spline_order: int = 3,
    fdr_threshold: float = 0.05,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, LinearGAM]]:
    """
    Fit GAM models for genes with pygam compatibility fix.
    """
    import numpy as np
    import pandas as pd
    from scipy.sparse import issparse
    from pygam import LinearGAM, s, f
    from statsmodels.stats.multitest import multipletests
    
    # ========================================================================= #
    # MONKEY PATCH: Fix pygam compatibility with newer SciPy versions
    # ========================================================================= #
    def patched_cholesky(A, sparse=None, verbose_chol=False):
        """
        Patched version of pygam.utils.cholesky that handles the .A attribute issue.
        """
        from scipy.sparse import issparse
        from scipy.linalg import cholesky as scipy_cholesky
        import numpy as np
        
        # Handle the .A attribute issue
        if issparse(A):
            A = A.toarray()
        elif hasattr(A, 'A'):
            A = A.A
        
        try:
            result = scipy_cholesky(A, lower=True)
            return result
        except Exception as e:
            raise

    # Apply monkey patch
    try:
        import pygam.utils
        original_cholesky = pygam.utils.cholesky
        pygam.utils.cholesky = patched_cholesky
    except ImportError:
        original_cholesky = None

    try:
        # ================================================================== #
        # Data preparation
        # ================================================================== #
        
        def _to_dense_2d(mat) -> np.ndarray:
            """DataFrame / ndarray / SciPy sparse  -> C-contiguous float64 (n×m)."""
            if isinstance(mat, pd.DataFrame):
                if hasattr(mat, "sparse"):
                    try:
                        mat = mat.sparse.to_dense()
                    except Exception:
                        pass
                mat = mat.to_numpy()
            if issparse(mat):
                mat = mat.toarray()
            return np.asarray(mat, dtype=np.float64, order="C")

        # Sanity checks
        if num_splines <= spline_order:
            num_splines = spline_order + 1

        if X.shape[0] < num_splines:
            num_splines = max(3, X.shape[0] - 1)

        # Densify data
        X_dense = _to_dense_2d(X)
        Y_dense = _to_dense_2d(Y)

        # Mask rows with any NaN / Inf in covariates
        finite_rows = np.isfinite(X_dense).all(axis=1)

        # Column index for spline term
        try:
            spline_idx = list(X.columns).index(spline_term)
        except ValueError:
            raise ValueError(f"spline_term '{spline_term}' not found in X.columns: {list(X.columns)}")

        # Build GAM term structure
        terms = s(spline_idx, n_splines=num_splines, spline_order=spline_order)
        
        for j in range(X_dense.shape[1]):
            if j != spline_idx:
                terms += f(j)

        # Gene-to-column map
        if isinstance(Y, pd.DataFrame):
            gene_to_col = {g: i for i, g in enumerate(Y.columns)}
        else:
            gene_to_col = {g: i for i, g in enumerate(gene_names)}

        results = []
        gam_models = {}
        total = len(gene_names)
        good = 0

        # Process genes
        for k, gene in enumerate(gene_names):
            if verbose and (k + 1) % 100 == 0:
                print(f"Processing gene {k + 1}/{total}")
            
            col_idx = gene_to_col.get(gene, None)
            if col_idx is None or col_idx >= Y_dense.shape[1]:
                continue

            y_raw = Y_dense[:, col_idx]

            # Align masks (finite in both X and y)
            mask = finite_rows & np.isfinite(y_raw)
            
            X_fit = X_dense[mask]
            y_fit = y_raw[mask]

            # Skip tiny / constant vectors
            if y_fit.size < (spline_order + 2):
                continue
                
            var_y = np.var(y_fit)
            if var_y < 1e-10:
                continue

            try:
                gam = LinearGAM(terms)
                gam = gam.fit(X_fit, y_fit)

                # Extract statistics
                stats = gam.statistics_
                pval = gam.statistics_["p_values"][spline_idx]
                dev = gam.statistics_["pseudo_r2"]["explained_deviance"]
                
                if np.isfinite(pval) and np.isfinite(dev):
                    results.append((gene, pval, dev))
                    gam_models[gene] = gam
                    good += 1

            except Exception as e:
                if verbose:
                    print(f"GAM fitting failed for {gene}: {str(e)}")
                continue

        # Assemble results table
        if not results:
            cols = ["gene", "pval", "dev_exp", "fdr", "significant"]
            return pd.DataFrame(columns=cols), {}

        res_df = pd.DataFrame(results, columns=["gene", "pval", "dev_exp"])
        
        try:
            _, fdrs, _, _ = multipletests(res_df["pval"], method="fdr_bh")
            res_df["fdr"] = fdrs
            res_df["significant"] = res_df["fdr"] < fdr_threshold
        except Exception:
            res_df["fdr"] = res_df["pval"]
            res_df["significant"] = res_df["fdr"] < fdr_threshold
        
        return res_df.sort_values("fdr").reset_index(drop=True), gam_models
    
    finally:
        # Restore original function to avoid side effects
        if original_cholesky is not None:
            try:
                pygam.utils.cholesky = original_cholesky
            except Exception:
                pass


def calculate_effect_size(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    gam_models: Dict[str, LinearGAM],
    genes: List[str],
    verbose: bool = False
) -> pd.DataFrame:
    """
    Calculate effect sizes for genes.
    """
    import numpy as np
    import pandas as pd
    
    effect_sizes = []
    
    for gene in genes:
        if gene not in gam_models or gene not in Y.columns:
            continue
        
        try:
            gam = gam_models[gene]
            
            # Convert X to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
                
            y_pred = gam.predict(X_values)
            y_true = Y[gene].values
            
            # Calculate effect size
            residuals = y_true - y_pred
            df_e = max(1, len(y_true) - gam.statistics_['edof'])
            
            max_pred = np.max(y_pred)
            min_pred = np.min(y_pred)
            sum_squared_residuals = np.sum(residuals**2)
            
            if sum_squared_residuals > 0 and df_e > 0:
                es = (max_pred - min_pred) / np.sqrt(sum_squared_residuals / df_e)
                effect_sizes.append((gene, es))
        except Exception as e:
            if verbose:
                print(f"Error calculating effect size for {gene}: {e}")
            continue
    
    if verbose:
        print(f"Calculated effect sizes for {len(effect_sizes)} genes")
    
    return pd.DataFrame(effect_sizes, columns=["gene", "effect_size"])

def run_integrated_differential_analysis(
    trajectory_results: Dict,
    pseudobulk_adata: ad.AnnData,
    trajectory_type: str = "TSCAN",
    sample_col: str = "sample",
    fdr_threshold: float = 0.05,
    effect_size_threshold: float = 1.0,
    top_n_genes: int = 100,
    covariate_columns: Optional[List[str]] = None,
    num_splines: int = 5,
    spline_order: int = 3,
    base_output_dir: str = "trajectory_diff_gene_results",
    visualization_gene_list: Optional[List[str]] = None,
    visualize_all_deg: bool = False,
    top_n_heatmap: int = 50,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Integrated function that handles both TSCAN and CCA trajectory results
    with improved error handling and data validation.
    """
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Extract pseudotime data based on trajectory type
    if verbose:
        print(f"Extracting pseudotime from {trajectory_type} results...")
    
    pseudotime_data = prepare_trajectory_data(
        pseudobulk_adata=pseudobulk_adata,
        trajectory_results=trajectory_results,
        trajectory_type=trajectory_type,
        sample_col=sample_col,
        verbose=verbose
    )
    
    if not pseudotime_data or "main_path" not in pseudotime_data:
        raise ValueError(f"No valid pseudotime data found in {trajectory_type} results")
    
    all_results = {}
    
    # Process main path
    if pseudotime_data["main_path"]:
        print("\nProcessing main trajectory path...")
        main_path_output_dir = os.path.join(base_output_dir, "main_path")
        os.makedirs(main_path_output_dir, exist_ok=True)
        
        try:
            # Use improved data preparation
            X, Y, gene_names = prepare_gam_input_data_improved(
                pseudobulk_adata=pseudobulk_adata,
                ptime_expression=pseudotime_data["main_path"],
                covariate_columns=covariate_columns,
                sample_col=sample_col,
                verbose=verbose
            )
            
            # Adjust spline parameters based on sample size
            n_samples = X.shape[0]
            adjusted_num_splines = min(num_splines, max(3, n_samples // 3))
            if adjusted_num_splines != num_splines and verbose:
                print(f"Adjusted num_splines from {num_splines} to {adjusted_num_splines} based on sample size")
            
            # Run GAM analysis
            if verbose:
                print(f"Fitting GAM models for {len(gene_names)} genes...")
            
            stat_results, gam_models = fit_gam_models_for_genes(
                X=X,
                Y=Y,
                gene_names=gene_names,
                spline_term="pseudotime",
                num_splines=adjusted_num_splines,
                spline_order=spline_order,
                fdr_threshold=fdr_threshold,
                verbose=verbose
            )
            
            # Calculate effect sizes for significant genes
            if len(stat_results) > 0:
                stat_sig_genes = stat_results[stat_results["fdr"] < fdr_threshold]["gene"].tolist()
                
                if verbose:
                    print(f"Calculating effect sizes for {len(stat_sig_genes)} significant genes...")
                
                effect_sizes = calculate_effect_size(
                    X=X,
                    Y=Y,
                    gam_models=gam_models,
                    genes=stat_sig_genes,
                    verbose=verbose
                )
                
                # Merge results
                results = stat_results.merge(effect_sizes, on="gene", how="left")
                
                # Determine pseudoDEGs
                results = determine_pseudoDEGs(
                    results=results,
                    fdr_threshold=fdr_threshold,
                    effect_size_threshold=effect_size_threshold,
                    top_n_genes=top_n_genes,
                    verbose=verbose
                )
                
                # Save results
                save_results(
                    results_df=results,
                    output_dir=main_path_output_dir,
                    fdr_threshold=fdr_threshold,
                    effect_size_threshold=effect_size_threshold,
                    top_n_genes=top_n_genes,
                    verbose=verbose
                )
                
                # Generate summary
                summarize_results(
                    results=results,
                    top_n=min(20, len(results)),
                    output_file=os.path.join(main_path_output_dir, "differential_gene_result.txt"),
                    verbose=verbose
                )
                
                all_results["main_path"] = results
                
                # Generate visualizations if requested
                if visualization_gene_list and len(gam_models) > 0:
                    generate_gene_visualizations(
                        gene_list=visualization_gene_list,
                        X=X,
                        Y=Y,
                        gam_models=gam_models,
                        results=results,
                        output_dir=main_path_output_dir,
                        verbose=verbose
                    )
            else:
                print("Warning: No genes were successfully analyzed for main path")
                all_results["main_path"] = pd.DataFrame()
                
        except Exception as e:
            print(f"Error processing main path: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
            all_results["main_path"] = pd.DataFrame()
    
    # Process branching paths if they exist (TSCAN only)
    if "branching_paths" in pseudotime_data and pseudotime_data["branching_paths"]:
        for branch_id, branch_ptime in pseudotime_data["branching_paths"].items():
            if not branch_ptime:  # Skip empty branches
                continue
                
            branch_name = f"branch_{branch_id}"
            print(f"\nProcessing {branch_name}...")
            
            branch_output_dir = os.path.join(base_output_dir, branch_name)
            os.makedirs(branch_output_dir, exist_ok=True)
            
            try:
                # Similar processing as main path
                X, Y, gene_names = prepare_gam_input_data_improved(
                    pseudobulk_adata=pseudobulk_adata,
                    ptime_expression=branch_ptime,
                    covariate_columns=covariate_columns,
                    sample_col=sample_col,
                    verbose=verbose
                )
                
                # Adjust splines for branch size
                n_samples = X.shape[0]
                adjusted_num_splines = min(num_splines, max(3, n_samples // 3))
                
                # Run analysis
                stat_results, gam_models = fit_gam_models_for_genes(
                    X=X,
                    Y=Y,
                    gene_names=gene_names,
                    spline_term="pseudotime",
                    num_splines=adjusted_num_splines,
                    spline_order=spline_order,
                    fdr_threshold=fdr_threshold,
                    verbose=verbose
                )
                
                if len(stat_results) > 0:
                    # Process results similar to main path
                    stat_sig_genes = stat_results[stat_results["fdr"] < fdr_threshold]["gene"].tolist()
                    effect_sizes = calculate_effect_size(X, Y, gam_models, stat_sig_genes, verbose)
                    results = stat_results.merge(effect_sizes, on="gene", how="left")
                    results = determine_pseudoDEGs(results, fdr_threshold, effect_size_threshold, top_n_genes, verbose)
                    
                    save_results(results, branch_output_dir, fdr_threshold, effect_size_threshold, top_n_genes, verbose)
                    summarize_results(results, min(20, len(results)), 
                                    os.path.join(branch_output_dir, "differential_gene_result.txt"), verbose)
                    
                    all_results[branch_name] = results
                else:
                    all_results[branch_name] = pd.DataFrame()
                    
            except Exception as e:
                print(f"Error processing {branch_name}: {str(e)}")
                all_results[branch_name] = pd.DataFrame()
    
    # Generate comparative analysis if multiple paths exist
    if len(all_results) > 1:
        try:
            create_comparative_analysis(all_results, base_output_dir, top_n=50)
        except Exception as e:
            print(f"Warning: Could not create comparative analysis: {str(e)}")
    
    if verbose:
        print(f"\nDifferential expression analysis completed.")
        print(f"Results saved to: {base_output_dir}")
        successful_paths = sum(1 for df in all_results.values() if len(df) > 0)
        print(f"Successfully analyzed {successful_paths} trajectory paths")
    
    return all_results


def determine_pseudoDEGs(
    results: pd.DataFrame,
    fdr_threshold: float,
    effect_size_threshold: float,
    top_n_genes: Optional[int],
    verbose: bool
) -> pd.DataFrame:
    """
    Determine pseudoDEGs based on FDR and effect size criteria.
    """
    if len(results) == 0:
        results["pseudoDEG"] = False
        return results
    
    if top_n_genes is not None:
        # Select top N genes by effect size from significant genes
        stat_sig_df = results[results["fdr"] < fdr_threshold].copy()
        
        if len(stat_sig_df) > 0:
            if 'effect_size' in stat_sig_df.columns and not stat_sig_df['effect_size'].isna().all():
                if len(stat_sig_df) > top_n_genes:
                    top_genes = stat_sig_df.nlargest(top_n_genes, 'effect_size')
                    results["pseudoDEG"] = results["gene"].isin(top_genes["gene"])
                else:
                    results["pseudoDEG"] = results["fdr"] < fdr_threshold
                
                if verbose:
                    n_selected = results["pseudoDEG"].sum()
                    print(f"Selected {n_selected} pseudoDEGs (top {top_n_genes} by effect size)")
            else:
                results["pseudoDEG"] = results["fdr"] < fdr_threshold
        else:
            results["pseudoDEG"] = False
    else:
        # Use effect size threshold
        if 'effect_size' in results.columns:
            results["pseudoDEG"] = (
                (results["fdr"] < fdr_threshold) & 
                (results["effect_size"] > effect_size_threshold)
            )
        else:
            results["pseudoDEG"] = results["fdr"] < fdr_threshold
        
        if verbose:
            n_selected = results["pseudoDEG"].sum()
            print(f"Selected {n_selected} pseudoDEGs (FDR < {fdr_threshold}, effect > {effect_size_threshold})")
    
    return results


def generate_gene_visualizations(
    gene_list: List[str],
    X: pd.DataFrame,
    Y: pd.DataFrame,
    gam_models: Dict[str, LinearGAM],
    results: pd.DataFrame,
    output_dir: str,
    verbose: bool = False
):
    """
    Generate visualizations for specified genes.
    """
    viz_dir = os.path.join(output_dir, "gene_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    if verbose:
        print(f"Generating visualizations for {len(gene_list)} genes...")
    
    # Import visualization function if available
    try:
        from DEG_visualization import visualize_gene_expression
        
        for gene in gene_list:
            if gene in gam_models and gene in Y.columns:
                try:
                    visualize_gene_expression(
                        gene=gene,
                        X=X,
                        Y=Y,
                        gam_model=gam_models[gene],
                        stats_df=results,
                        output_dir=output_dir,
                        gene_subfolder="gene_visualizations",
                        verbose=verbose
                    )
                except Exception as e:
                    if verbose:
                        print(f"Error visualizing gene {gene}: {e}")
            elif verbose:
                print(f"Gene {gene} not found in results")
                
    except ImportError:
        if verbose:
            print("DEG_visualization module not available, skipping visualizations")


def save_results(
    results_df: pd.DataFrame,
    output_dir: str,
    fdr_threshold: float,
    effect_size_threshold: float,
    top_n_genes: Optional[int] = None,
    verbose: bool = False
) -> None:
    """Save analysis results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all results
    all_results_file = os.path.join(output_dir, f"gam_all_genes_{timestamp}.tsv")
    results_df.to_csv(all_results_file, sep='\t', index=False)
    
    # Save significant genes
    if 'fdr' in results_df.columns:
        sig_genes = results_df[results_df["fdr"] < fdr_threshold]
        sig_file = os.path.join(output_dir, f"gam_significant_{timestamp}.tsv")
        sig_genes.to_csv(sig_file, sep='\t', index=False)
    
    # Save pseudoDEGs
    if 'pseudoDEG' in results_df.columns:
        degs = results_df[results_df["pseudoDEG"]]
        deg_file = os.path.join(output_dir, f"gam_pseudoDEGs_{timestamp}.tsv")
        degs.to_csv(deg_file, sep='\t', index=False)
    
    # Save summary
    summary_file = os.path.join(output_dir, f"gam_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("===== GAM ANALYSIS SUMMARY =====\n\n")
        f.write(f"Analysis date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"FDR threshold: {fdr_threshold}\n")
        
        if top_n_genes is not None:
            f.write(f"Selection: Top {top_n_genes} genes by effect size\n")
        else:
            f.write(f"Effect size threshold: {effect_size_threshold}\n")
        
        f.write(f"\nTotal genes analyzed: {len(results_df)}\n")
        
        if 'fdr' in results_df.columns:
            n_sig = (results_df['fdr'] < fdr_threshold).sum()
            f.write(f"Significant genes (FDR < {fdr_threshold}): {n_sig}\n")
        
        if 'pseudoDEG' in results_df.columns:
            n_deg = results_df['pseudoDEG'].sum()
            f.write(f"Selected pseudoDEGs: {n_deg}\n")
            
            # List top DEGs
            if n_deg > 0:
                f.write("\nTop pseudoDEGs:\n")
                top_degs = results_df[results_df['pseudoDEG']].nsmallest(20, 'fdr')
                for idx, row in top_degs.iterrows():
                    if 'effect_size' in row and pd.notna(row['effect_size']):
                        f.write(f"  {row['gene']}: FDR={row['fdr']:.4e}, Effect={row['effect_size']:.3f}\n")
                    else:
                        f.write(f"  {row['gene']}: FDR={row['fdr']:.4e}\n")
    
    if verbose:
        print(f"Results saved to {output_dir}")


def summarize_results(
    results: pd.DataFrame,
    top_n: int = 20,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> None:
    """Print and optionally save a summary of analysis results."""
    
    if len(results) == 0:
        summary = "No genes were successfully analyzed."
        if verbose:
            print(summary)
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(summary)
        return
    
    summary_lines = ["=== DIFFERENTIAL GENE EXPRESSION SUMMARY ==="]
    summary_lines.append(f"Total genes analyzed: {len(results)}")
    
    if 'fdr' in results.columns:
        n_sig = (results['fdr'] < 0.05).sum()
        summary_lines.append(f"Significant genes (FDR < 0.05): {n_sig}")
    
    if 'pseudoDEG' in results.columns:
        n_deg = results['pseudoDEG'].sum()
        summary_lines.append(f"Selected DEGs: {n_deg}")
        
        if n_deg > 0:
            summary_lines.append(f"\nTop {min(top_n, n_deg)} DEGs:")
            top_degs = results[results['pseudoDEG']].nsmallest(min(top_n, n_deg), 'fdr')
            
            for i, (idx, row) in enumerate(top_degs.iterrows(), 1):
                if 'effect_size' in row and pd.notna(row['effect_size']):
                    summary_lines.append(f"{i}. {row['gene']}: FDR={row['fdr']:.4e}, Effect={row['effect_size']:.3f}")
                else:
                    summary_lines.append(f"{i}. {row['gene']}: FDR={row['fdr']:.4e}")
    
    summary = "\n".join(summary_lines)
    
    if verbose:
        print(summary)
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(summary)


def create_comparative_analysis(
    all_results: Dict[str, pd.DataFrame],
    base_output_dir: str,
    top_n: int = 50
):
    """Create comparative analysis across trajectory paths."""
    
    comp_dir = os.path.join(base_output_dir, "comparative_analysis")
    os.makedirs(comp_dir, exist_ok=True)
    
    # Collect DEGs from each path
    path_degs = {}
    all_degs = set()
    
    for path_name, results in all_results.items():
        if len(results) > 0 and 'pseudoDEG' in results.columns:
            degs = results[results['pseudoDEG']]
            if len(degs) > 0:
                # Get top N DEGs by effect size or FDR
                if 'effect_size' in degs.columns:
                    top_degs = degs.nlargest(min(top_n, len(degs)), 'effect_size')
                else:
                    top_degs = degs.nsmallest(min(top_n, len(degs)), 'fdr')
                
                path_degs[path_name] = set(top_degs['gene'])
                all_degs.update(top_degs['gene'])
    
    if len(path_degs) < 2:
        print("Not enough paths with DEGs for comparative analysis")
        return
    
    # Create overlap matrix
    paths = sorted(path_degs.keys())
    overlap_matrix = pd.DataFrame(index=paths, columns=paths, dtype=int)
    
    for p1 in paths:
        for p2 in paths:
            if p1 == p2:
                overlap_matrix.loc[p1, p2] = len(path_degs[p1])
            else:
                overlap_matrix.loc[p1, p2] = len(path_degs[p1] & path_degs[p2])
    
    overlap_matrix.to_csv(os.path.join(comp_dir, "deg_overlap_matrix.csv"))
    
    # Find shared and unique DEGs
    if len(path_degs) > 1:
        shared_degs = set.intersection(*path_degs.values())
        
        with open(os.path.join(comp_dir, "deg_comparison.txt"), 'w') as f:
            f.write(f"=== DEG COMPARISON ACROSS {len(paths)} PATHS ===\n\n")
            f.write(f"Total unique DEGs: {len(all_degs)}\n")
            f.write(f"Shared DEGs: {len(shared_degs)}\n\n")
            
            if shared_degs:
                f.write("Shared DEGs:\n")
                for gene in sorted(shared_degs):
                    f.write(f"  {gene}\n")
                f.write("\n")
            
            # Path-specific DEGs
            for path in paths:
                other_paths = [p for p in paths if p != path]
                other_degs = set.union(*[path_degs[p] for p in other_paths])
                unique_degs = path_degs[path] - other_degs
                
                f.write(f"\n{path} specific DEGs ({len(unique_degs)}):\n")
                for gene in sorted(unique_degs)[:20]:  # Show top 20
                    f.write(f"  {gene}\n")
    
    # Create effect size comparison for shared genes
    if shared_degs:
        effect_comparison = pd.DataFrame(index=sorted(shared_degs), columns=paths)
        
        for path, results in all_results.items():
            if path in paths and 'effect_size' in results.columns:
                for gene in shared_degs:
                    if gene in results['gene'].values:
                        gene_data = results[results['gene'] == gene].iloc[0]
                        effect_comparison.loc[gene, path] = gene_data['effect_size']
        
        effect_comparison.to_csv(os.path.join(comp_dir, "shared_genes_effect_sizes.csv"))
    
    print(f"Comparative analysis saved to {comp_dir}")


# Additional utility function for handling pseudobulk data with cell type information
def extract_cell_type_specific_expression(
    pseudobulk_adata: ad.AnnData,
    cell_type: str,
    hvg_only: bool = True
) -> pd.DataFrame:
    """
    Extract expression data for a specific cell type from pseudobulk AnnData.
    
    Parameters
    ----------
    pseudobulk_adata : ad.AnnData
        Pseudobulk AnnData with concatenated HVGs
    cell_type : str
        Cell type to extract
    hvg_only : bool
        Whether to return only HVGs for this cell type
        
    Returns
    -------
    expression_df : pd.DataFrame
        Expression matrix for the specified cell type
    """
    
    # Get gene names for this cell type
    cell_type_genes = [g for g in pseudobulk_adata.var_names if g.startswith(f"{cell_type} - ")]
    
    if len(cell_type_genes) == 0:
        raise ValueError(f"No genes found for cell type '{cell_type}'")
    
    # Extract expression data
    gene_indices = [pseudobulk_adata.var_names.get_loc(g) for g in cell_type_genes]
    
    if issparse(pseudobulk_adata.X):
        expression_data = pseudobulk_adata.X[:, gene_indices].toarray()
    else:
        expression_data = pseudobulk_adata.X[:, gene_indices]
    
    # Create DataFrame
    clean_gene_names = [g.replace(f"{cell_type} - ", "") for g in cell_type_genes]
    expression_df = pd.DataFrame(
        expression_data,
        index=pseudobulk_adata.obs_names,
        columns=clean_gene_names
    )
    
    return expression_df


# Wrapper function that maintains compatibility with existing code
def run_differential_analysis_for_all_paths(
    TSCAN_results: Dict,
    pseudobulk_adata: ad.AnnData,
    sample_col: str = "sample",
    fdr_threshold: float = 0.05,
    effect_size_threshold: float = 1.0,
    top_n_genes: int = 100,
    covariate_columns: Optional[List[str]] = None,
    num_splines: int = 3,
    spline_order: int = 3,
    base_output_dir: str = "trajectory_diff_gene_results",
    top_gene_number: int = 100,
    visualization_gene_list: Optional[List[str]] = None,
    visualize_all_deg: bool = False,
    top_n_heatmap: int = 50,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Wrapper function that maintains compatibility with existing code while using improved implementation.
    """
    
    return run_integrated_differential_analysis(
        trajectory_results=TSCAN_results,
        pseudobulk_adata=pseudobulk_adata,
        trajectory_type="TSCAN",
        sample_col=sample_col,
        fdr_threshold=fdr_threshold,
        effect_size_threshold=effect_size_threshold,
        top_n_genes=top_n_genes,
        covariate_columns=covariate_columns,
        num_splines=num_splines,
        spline_order=spline_order,
        base_output_dir=base_output_dir,
        visualization_gene_list=visualization_gene_list,
        visualize_all_deg=visualize_all_deg,
        top_n_heatmap=top_n_heatmap,
        verbose=verbose
    )