import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from pygam import LinearGAM, s, f
from statsmodels.stats.multitest import multipletests
import warnings
import os
import datetime
from DEG_visualization import visualize_gene_expression, visualize_all_deg_genes, generate_deg_heatmap, generate_summary_trajectory_plot

def prepare_gam_input_data(
    pseudobulk: dict,
    sample_meta_path: Optional[str] = None,
    ptime_expression: Dict[str, float] = None,
    covariate_columns: Optional[List[str]] = None,
    expression_key: str = "cell_expression_corrected",
    sample_col: str = "sample",
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    
    metadata = None
    if sample_meta_path is not None and os.path.exists(sample_meta_path):
        if verbose:
            print("Reading metadata file...")
        try:
            # Read metadata file (auto-detect CSV or TSV)
            metadata = pd.read_csv(sample_meta_path, sep=None, engine="python")
            if sample_col not in metadata.columns:
                raise ValueError(f"Metadata file must contain a '{sample_col}' column.")
            # Convert sample names to lowercase for case-insensitive matching
            metadata[f'{sample_col}_lower'] = metadata[sample_col].str.lower()
            metadata = metadata.set_index(f"{sample_col}_lower")
        except Exception as e:
            print(f"Warning: Could not properly load metadata file: {e}")
            metadata = None
    elif sample_meta_path is not None:
        print(f"Warning: Metadata file {sample_meta_path} not found.")
    
    # Handle the expression data with more care
    expr_data = pseudobulk[expression_key]
    
    # Check the type of expr_data and handle accordingly
    if isinstance(expr_data, pd.DataFrame):
        # Already a DataFrame, just make a copy
        expr_df = expr_data.copy()
    else:
        # If it's not a DataFrame, we need to verify its structure before conversion
        try:
            # First, check if it's a simple dict that can be directly used as a DataFrame
            expr_df = pd.DataFrame(expr_data)
            
            # If the above didn't work, try different orientations or debug the structure
            if expr_df.empty and isinstance(expr_data, dict):
                # Try to determine the structure of the dict
                sample_keys = list(expr_data.keys())
                if len(sample_keys) > 0:
                    first_value = expr_data[sample_keys[0]]
                    if isinstance(first_value, dict):
                        # Convert nested dict to DataFrame
                        expr_df = pd.DataFrame.from_dict(expr_data, orient='index')
                    elif isinstance(first_value, (list, np.ndarray)):
                        # If values are arrays, need to ensure same length
                        # Create dict with gene names as keys first
                        if verbose:
                            print(f"Converting from list/array values with length {len(first_value)}")
                        expr_df = pd.DataFrame(expr_data, index=sample_keys)
        except Exception as e:
            raise ValueError(f"Failed to convert expression data to DataFrame: {e}\n"
                            f"Data type: {type(expr_data)}\n"
                            f"Please check the structure of pseudobulk['{expression_key}']")
    
    if expr_df.empty:
        raise ValueError(f"Expression data conversion resulted in an empty DataFrame. "
                        f"Please check the structure of pseudobulk['{expression_key}']")
    
    # Create a lowercase version of the sample index for matching
    sample_name_map = {s.lower(): s for s in expr_df.index}
    expr_df['sample_lower'] = [s.lower() for s in expr_df.index]
    expr_df = expr_df.set_index('sample_lower')

    if ptime_expression is None or not ptime_expression:
        raise ValueError("Pseudotime expression values must be provided.")
    if verbose:
        print(f"Processing pseudotime data for {len(ptime_expression)} samples")
    
    # Create a lowercase version of the pseudotime dictionary keys
    ptime_expression_lower = {k.lower(): v for k, v in ptime_expression.items()}
    # Add pseudotime as a column
    pseudotime_df = pd.DataFrame.from_dict(ptime_expression_lower, orient='index', columns=['pseudotime'])
    pseudotime_df.index.name = 'sample_lower'

    if metadata is not None:
        try:
            meta = metadata.join(pseudotime_df, how="inner")
            common_samples = meta.index.intersection(expr_df.index)
            if len(common_samples) == 0:
                raise ValueError("No common samples found between metadata/pseudotime and expression data. Check sample name formatting.")
            if verbose:
                print(f"Found {len(common_samples)} common samples between metadata and expression data")
            meta = meta.loc[common_samples]
            expr_df = expr_df.loc[common_samples]
            if covariate_columns is None:
                covariate_columns = [col for col in meta.columns 
                                    if col != 'pseudotime' 
                                    and col != sample_col 
                                    and col != f'{sample_col}_lower']

            # Ensure covariate_columns only includes columns that actually exist in meta
            covariate_columns = [col for col in covariate_columns if col in meta.columns]
            if not covariate_columns:
                # No valid covariates, just use pseudotime
                X = pd.DataFrame({'pseudotime': meta['pseudotime']})
            else:
                covariates = meta[covariate_columns]
                covariates_encoded = pd.get_dummies(covariates, drop_first=True)
                X = pd.concat([meta['pseudotime'], covariates_encoded], axis=1)
                
        except Exception as e:
            print(f"Warning: Error processing metadata: {e}")
            metadata = None
    
    if metadata is None:
        # If no metadata or metadata processing failed, just use pseudotime
        # Create a DataFrame with pseudotime values
        X = pd.DataFrame(index=expr_df.index)
        X['pseudotime'] = X.index.map(lambda s: ptime_expression_lower.get(s, np.nan))
        # Remove rows with missing pseudotime
        valid_samples = ~X['pseudotime'].isna()
        X = X[valid_samples]
        expr_df = expr_df[valid_samples]
        if len(X) == 0:
            raise ValueError("No samples with valid pseudotime values found.")
        if verbose:
            print(f"Using {len(X)} samples with valid pseudotime values")
    
    # Get original expression dataframe with original sample names
    # and make sure it has the same order as X
    Y = expr_df.copy()  # Already filtered to have the same index as X
    gene_names = list(Y.columns)
    
    # Remove any non-data columns that might have been added
    for col in ['sample_lower']:
        if col in gene_names:
            gene_names.remove(col)
            Y = Y.drop(col, axis=1)
    
    if verbose:
        print(f"Prepared input data with {X.shape[0]} samples and {len(gene_names)} genes")
    return X, Y, gene_names

def fit_gam_models_for_genes(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    gene_names: List[str],
    spline_term: str = "pseudotime",
    num_splines: int = 5,
    spline_order: int = 3,
    fdr_threshold: float = 0.05,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, LinearGAM]]:
    """
    Fits a GAM to each gene using pseudotime + covariates.
    
    Parameters
    ----------
    X : pd.DataFrame
        Design matrix with pseudotime and covariates (z_s).
    Y : pd.DataFrame
        Gene expression matrix, samples x genes.
    gene_names : list of str
        List of gene names (column names in Y).
    spline_term : str
        Name of the column in X to be modeled as a spline.
    num_splines : int
        Requested number of spline basis functions.
    spline_order : int
        Order of the spline (default cubic = 3).
    fdr_threshold : float
        Significance level for adjusted p-values.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with gene name, p-value, FDR, deviance explained, significance status.
    gam_models : dict
        Dictionary mapping gene names to fitted GAM models.
    """
    results = []
    gam_models = {}
    # Ensure num_splines is greater than spline_order
    if num_splines <= spline_order:
        corrected_num_splines = spline_order + 1
        warnings.warn(
            f"num_splines must be > spline_order. Adjusted from {num_splines} to {corrected_num_splines}."
        )
        num_splines = corrected_num_splines
    
    spline_idx = list(X.columns).index(spline_term)
    # Build GAM terms: pseudotime as a spline + other covariates as linear terms
    spline_term = s(spline_idx, n_splines=num_splines, spline_order=spline_order)
    linear_terms = [f(i) for i in range(X.shape[1]) if i != spline_idx]
    # Combine terms properly
    if linear_terms:
        terms = spline_term
        for term in linear_terms:
            terms += term
    else:
        terms = spline_term
    
    total_genes = len(gene_names)
    for i, gene in enumerate(gene_names):
        if verbose and (i % 100 == 0 or i == total_genes - 1):
            print(f"Fitting GAM model for gene {i+1}/{total_genes}: {gene}")
        y = Y[gene].values
        try:
            gam = LinearGAM(terms).fit(X.values, y)
            pval = gam.statistics_['p_values'][spline_idx]  # p-value for pseudotime
            dev_exp = gam.statistics_['pseudo_r2']['explained_deviance']
            results.append((gene, pval, dev_exp))
            gam_models[gene] = gam
        except Exception as e:
            if verbose:
                print(f"Failed to fit GAM for {gene}: {e}")
            results.append((gene, None, None))

    # Compile results
    results_df = pd.DataFrame(results, columns=["gene", "pval", "dev_exp"]).dropna()
    # FDR correction
    _, fdr, _, _ = multipletests(results_df["pval"], method="fdr_bh")
    results_df["fdr"] = fdr
    results_df["significant"] = results_df["fdr"] < fdr_threshold
    if verbose:
        sig_count = results_df["significant"].sum()
        print(f"Found {sig_count} statistically significant genes (FDR < {fdr_threshold})")
    return results_df.sort_values("fdr"), gam_models

def calculate_effect_size(
    X: pd.DataFrame,
    Y: pd.DataFrame, 
    gam_models: Dict[str, LinearGAM],
    genes: List[str],
    verbose: bool = False
) -> pd.DataFrame:
    """
    Calculate the effect size (ES) for each gene based on GAM results.
    
    Parameters
    ----------
    X : pd.DataFrame
        Design matrix with pseudotime and covariates.
    Y : pd.DataFrame
        Gene expression matrix.
    gam_models : dict
        Dictionary mapping gene names to fitted GAM models.
    genes : list
        List of genes to calculate effect size for.
    verbose : bool
        Whether to print progress messages.
        
    Returns
    -------
    effect_sizes : pd.DataFrame
        DataFrame with gene names and their effect sizes.
    """
    effect_sizes = []
    
    if verbose:
        print(f"Calculating effect sizes for {len(genes)} genes")
    
    for gene in genes:
        if gene not in gam_models:
            continue
            
        gam = gam_models[gene]
        y_pred = gam.predict(X.values)
        # Calculate effect size: (max_s ŷ_gs - min_s ŷ_gs) / √(∑_s e²_gs/df_e)
        # where ŷ is the GAM smoothed expression levels, e is the model residual
        # and df_e is the residual degrees of freedom
        y_true = Y[gene].values
        residuals = y_true - y_pred
        df_e = gam.statistics_['edof']
        max_pred = np.max(y_pred)
        min_pred = np.min(y_pred)
        sum_squared_residuals = np.sum(residuals**2)
        if sum_squared_residuals > 0 and df_e > 0:
            es = (max_pred - min_pred) / np.sqrt(sum_squared_residuals / df_e)
            effect_sizes.append((gene, es))
    result_df = pd.DataFrame(effect_sizes, columns=["gene", "effect_size"])
    if verbose:
        print(f"Calculated effect sizes for {len(result_df)} genes")
    return result_df

def identify_pseudoDEGs(
    pseudobulk=None,
    sample_meta_path=None,
    ptime_expression=None,
    fdr_threshold=0.05,
    effect_size_threshold=1.0,
    top_n_genes=None,
    covariate_columns=None,
    expression_key="cell_expression_corrected",
    sample_col="sample",
    num_splines=5,
    spline_order=3,
    visualization_gene_list=None,
    visualize_all_deg=False,
    top_n_heatmap=50,
    output_dir=None,
    verbose=False
):
    if pseudobulk is None or ptime_expression is None:
        raise ValueError("Either provide X and Y matrices or provide pseudobulk and ptime_expression")
    if verbose:
        print("Preparing input data for GAM...")
    X, Y, gene_names = prepare_gam_input_data(
        pseudobulk=pseudobulk,
        sample_meta_path=sample_meta_path,
        ptime_expression=ptime_expression,
        covariate_columns=covariate_columns,
        expression_key=expression_key,
        sample_col=sample_col,
        verbose=verbose
    )
    
    if verbose:
        print(f"Fitting GAM models for {len(gene_names)} genes...")
    
    stat_results, gam_models = fit_gam_models_for_genes(
        X=X,
        Y=Y,
        gene_names=gene_names,
        spline_term="pseudotime",
        num_splines=num_splines,
        spline_order=spline_order,
        fdr_threshold=fdr_threshold,
        verbose=verbose
    )
    
    stat_sig_genes = stat_results[stat_results["fdr"] < fdr_threshold]["gene"].tolist()
    if verbose:
        print(f"Found {len(stat_sig_genes)} statistically significant genes (FDR < {fdr_threshold}).")
    
    if verbose:
        print("Calculating effect sizes for significant genes...")
    effect_sizes = calculate_effect_size(
        X=X,
        Y=Y,
        gam_models=gam_models,
        genes=stat_sig_genes,
        verbose=verbose
    )
    
    results = stat_results.merge(effect_sizes, on="gene", how="left")
    
    if top_n_genes is not None:
        stat_sig_df = results[results["fdr"] < fdr_threshold].copy()
        
        if len(stat_sig_df) > top_n_genes:
            pseudoDEGs = stat_sig_df.sort_values("effect_size", ascending=False).head(top_n_genes)
            results["pseudoDEG"] = results["gene"].isin(pseudoDEGs["gene"])
            if verbose:
                print(f"Selected top {top_n_genes} genes by effect size from {len(stat_sig_df)} " +
                      f"statistically significant genes (FDR < {fdr_threshold}).")
        else:
            results["pseudoDEG"] = results["fdr"] < fdr_threshold
            if verbose:
                print(f"Found only {len(stat_sig_df)} statistically significant genes, " +
                      f"which is fewer than requested top {top_n_genes}. " +
                      f"Taking all genes with FDR < {fdr_threshold}.")
    else:
        results["pseudoDEG"] = (
            (results["fdr"] < fdr_threshold) & 
            (results["effect_size"] > effect_size_threshold)
        )
        if verbose:
            pseudoDEGs = results[results["pseudoDEG"]]
            print(f"Identified {len(pseudoDEGs)} pseudoDEGs with FDR < {fdr_threshold} " +
                  f"and effect size > {effect_size_threshold}.")
    
    if output_dir is not None:
        save_results(results, output_dir, fdr_threshold, effect_size_threshold, top_n_genes, verbose)
        
        if visualization_gene_list is not None and len(visualization_gene_list) > 0:
            if verbose:
                print(f"Generating visualizations for {len(visualization_gene_list)} specified genes...")
            viz_dir = os.path.join(output_dir, "gene_visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            for gene in visualization_gene_list:
                if gene in gam_models:
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
                    print(f"Warning: Gene {gene} not found in fitted models.")
        
        if visualize_all_deg:
            if verbose:
                print("Generating visualizations for all differentially expressed genes...")
            visualize_all_deg_genes(
                X=X,
                Y=Y,
                gam_models=gam_models,
                results_df=results,
                output_dir=output_dir,
                gene_subfolder="all_deg_visualizations",
                top_n_heatmap=top_n_heatmap,
                verbose=verbose
            )
            
            generate_summary_trajectory_plot(
                X=X,
                Y=Y,
                gam_models=gam_models,
                results_df=results,
                output_dir=output_dir,
                top_n=min(10, results["pseudoDEG"].sum()),
                verbose=verbose
            )
    
    return results

def run_differential_analysis_for_all_paths(
    TSCAN_results, 
    pseudobulk_df,
    sample_meta_path=None,
    sample_col="sample",
    fdr_threshold=0.05,
    effect_size_threshold=1.0,
    top_n_genes=100, 
    covariate_columns=None,
    num_splines=3,
    spline_order=3,
    base_output_dir="trajectory_diff_gene_results",
    top_gene_number=100,
    visualization_gene_list=None,
    visualize_all_deg=False,
    top_n_heatmap=50,
    verbose=True
):
    os.makedirs(base_output_dir, exist_ok=True)
    all_results = {}
    # Verify main path pseudotime exists
    if ('pseudotime' not in TSCAN_results or 
        'main_path' not in TSCAN_results['pseudotime'] or 
        not TSCAN_results['pseudotime']['main_path']):
        print("Error: No valid main path pseudotime found in TSCAN results.")
        return all_results
    
    # Process main path
    print("Processing main path...")
    main_path_output_dir = os.path.join(base_output_dir, "main_path")
    os.makedirs(main_path_output_dir, exist_ok=True)
    
    # Run differential analysis for main path
    main_path_results = identify_pseudoDEGs(
        pseudobulk=pseudobulk_df,
        sample_meta_path=sample_meta_path,
        ptime_expression=TSCAN_results['pseudotime']['main_path'],
        fdr_threshold=fdr_threshold,
        effect_size_threshold=effect_size_threshold,
        top_n_genes=top_n_genes,
        covariate_columns=covariate_columns,
        sample_col=sample_col,
        num_splines=num_splines,
        spline_order=spline_order,
        output_dir=main_path_output_dir,
        visualization_gene_list=visualization_gene_list,
        visualize_all_deg=visualize_all_deg,
        top_n_heatmap=top_n_heatmap,
        verbose=verbose
    )
    
    # Summarize main path results
    summarize_results(
        results=main_path_results,
        top_n=top_gene_number,
        output_file=os.path.join(main_path_output_dir, "differential_gene_result.txt"),
        verbose=verbose
    )
    
    all_results["main_path"] = main_path_results

    if ('pseudotime' in TSCAN_results and 
        'branching_paths' in TSCAN_results['pseudotime'] and 
        TSCAN_results['pseudotime']['branching_paths']):
        
        branching_paths = TSCAN_results['pseudotime']['branching_paths']
        if isinstance(branching_paths, dict):
            # Process each branch path by key
            for branch_key, branch_path in branching_paths.items():
                # Skip empty branch paths
                if not branch_path:
                    print(f"Warning: Branch {branch_key} has no pseudotime data. Skipping.")
                    continue
                    
                branch_name = f"branch_{branch_key}"
                print(f"Processing {branch_name}...")
                
                branch_output_dir = os.path.join(base_output_dir, branch_name)
                os.makedirs(branch_output_dir, exist_ok=True)
                
                try:
                    # Run differential analysis for this branch
                    branch_results = identify_pseudoDEGs(
                        pseudobulk=pseudobulk_df,
                        sample_meta_path=sample_meta_path,
                        ptime_expression=branch_path,
                        fdr_threshold=fdr_threshold,
                        effect_size_threshold=effect_size_threshold,
                        top_n_genes=top_n_genes,
                        covariate_columns=covariate_columns,
                        sample_col=sample_col,
                        num_splines=num_splines,
                        spline_order=spline_order,
                        output_dir=branch_output_dir,
                        visualization_gene_list=visualization_gene_list,
                        visualize_all_deg=visualize_all_deg,
                        top_n_heatmap=top_n_heatmap,
                        verbose=verbose
                    )
                    
                    # Summarize branch results
                    summarize_results(
                        results=branch_results,
                        top_n=top_gene_number,
                        output_file=os.path.join(branch_output_dir, "differential_gene_result.txt"),
                        verbose=verbose
                    )
                    
                    all_results[branch_name] = branch_results
                except Exception as e:
                    print(f"Error processing {branch_name}: {e}")
        else:
            print("Warning: 'branching_paths' is not a dictionary. Expected a dictionary with branch keys and pseudotime values.")
    else:
        print("No branching paths found in TSCAN results.")
    
    return all_results

def create_comparative_analysis(all_results, base_output_dir, top_n):
    """
    Create a comparative analysis of differentially expressed genes across different paths
    
    Parameters:
    -----------
    all_results : dict
        Dictionary of results for each path
    base_output_dir : str
        Base directory to save results
    top_n : int
        Number of top genes to include in comparison
    """
    print("Creating comparative analysis...")
    
    # Create a directory for comparative analysis
    comparative_dir = os.path.join(base_output_dir, "comparative_analysis")
    os.makedirs(comparative_dir, exist_ok=True)
    
    # Get top genes from each path
    path_top_genes = {}
    all_top_genes = set()
    
    for path_name, results in all_results.items():
        if hasattr(results, 'diff_genes') and results.diff_genes is not None:
            # Sort genes by significance and effect size
            top_genes = results.diff_genes.sort_values(by=['fdr', 'effect_size'], 
                                                      ascending=[True, False]).head(top_n)
            path_top_genes[path_name] = top_genes
            all_top_genes.update(top_genes.index)
    
    # Create a comparison matrix
    comparison_df = pd.DataFrame(index=list(all_top_genes), columns=list(path_top_genes.keys()))
    
    for path_name, top_genes in path_top_genes.items():
        for gene in comparison_df.index:
            if gene in top_genes.index:
                comparison_df.loc[gene, path_name] = top_genes.loc[gene, 'effect_size']
            else:
                comparison_df.loc[gene, path_name] = np.nan
    
    # Save comparison matrix
    comparison_df.to_csv(os.path.join(comparative_dir, "path_comparison_matrix.csv"))
    # Create a Venn diagram or overlap analysis of top genes
    create_gene_overlap_analysis(path_top_genes, comparative_dir)

def create_gene_overlap_analysis(path_top_genes, output_dir):
    """
    Create an analysis of gene overlaps between different paths
    
    Parameters:
    -----------
    path_top_genes : dict
        Dictionary of top genes for each path
    output_dir : str
        Directory to save results
    """
    # Create sets of genes for each path
    gene_sets = {path: set(genes.index) for path, genes in path_top_genes.items()}
    
    # Calculate pairwise overlaps
    paths = list(gene_sets.keys())
    overlap_matrix = pd.DataFrame(index=paths, columns=paths)
    
    for i, path1 in enumerate(paths):
        for path2 in paths:
            if path1 == path2:
                overlap_matrix.loc[path1, path2] = len(gene_sets[path1])
            else:
                overlap = len(gene_sets[path1].intersection(gene_sets[path2]))
                overlap_matrix.loc[path1, path2] = overlap
    
    # Save overlap matrix
    overlap_matrix.to_csv(os.path.join(output_dir, "gene_overlap_matrix.csv"))
    
    # Generate lists of shared and unique genes
    shared_genes = set.intersection(*gene_sets.values()) if gene_sets else set()
    
    unique_genes = {}
    for path, genes in gene_sets.items():
        other_paths = set(paths) - {path}
        other_genes = set.union(*[gene_sets[p] for p in other_paths]) if other_paths else set()
        unique_genes[path] = genes - other_genes
    
    # Write shared genes to file
    with open(os.path.join(output_dir, "shared_genes.txt"), "w") as f:
        f.write("Genes shared across all paths:\n")
        for gene in shared_genes:
            f.write(f"{gene}\n")

def save_results(
    results_df: pd.DataFrame, 
    output_dir: str, 
    fdr_threshold: float,
    effect_size_threshold: float,
    top_n_genes: Optional[int] = None,
    verbose: bool = False
) -> None:
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all results
    all_results_file = os.path.join(output_dir, f"gam_all_genes_{timestamp}.tsv")
    results_df.to_csv(all_results_file, sep='\t', index=False)
    
    # Save statistically significant genes (FDR < threshold)
    stat_sig_file = os.path.join(output_dir, f"gam_stat_sig_{timestamp}.tsv")
    stat_sig_genes = results_df[results_df["fdr"] < fdr_threshold]
    stat_sig_genes.to_csv(stat_sig_file, sep='\t', index=False)
    
    # Save pseudoDEGs (based on pseudoDEG column)
    pseudoDEG_file = os.path.join(output_dir, f"gam_pseudoDEGs_{timestamp}.tsv")
    pseudoDEGs = results_df[results_df["pseudoDEG"]]
    pseudoDEGs.to_csv(pseudoDEG_file, sep='\t', index=False)
    
    # Save a summary text file
    summary_file = os.path.join(output_dir, f"gam_summary_{timestamp}.txt")
    
    with open(summary_file, 'w') as f:
        f.write("===== GAM PSEUDOTIME ANALYSIS SUMMARY =====\n\n")
        f.write(f"Analysis performed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Parameters:\n")
        f.write(f"- FDR threshold: {fdr_threshold}\n")
        
        # Describe selection method based on whether top_n_genes was provided
        if top_n_genes is not None:
            f.write(f"- Selection method: Top {top_n_genes} genes by effect size from statistically significant genes\n\n")
        else:
            f.write(f"- Effect size threshold: {effect_size_threshold}\n")
            f.write(f"- Selection method: Genes with FDR < {fdr_threshold} AND effect size > {effect_size_threshold}\n\n")
        
        f.write("Results:\n")
        f.write(f"- Total genes analyzed: {len(results_df)}\n")
        f.write(f"- Statistically significant genes (FDR < {fdr_threshold}): {len(stat_sig_genes)}\n")
        f.write(f"- Selected pseudoDEGs: {len(pseudoDEGs)}\n\n")
        
        if len(pseudoDEGs) > 0:
            f.write("Top 20 pseudoDEGs:\n")
            top_genes = pseudoDEGs.sort_values(["fdr", "effect_size"], ascending=[True, False]).head(20)
            for i, (_, row) in enumerate(top_genes.iterrows(), 1):
                f.write(f"{i}. {row['gene']}: FDR = {row['fdr']:.4e}, Effect Size = {row['effect_size']:.2f}\n")
        
        f.write("\nOutput files:\n")
        f.write(f"- All genes: {os.path.basename(all_results_file)}\n")
        f.write(f"- Statistically significant genes: {os.path.basename(stat_sig_file)}\n")
        f.write(f"- PseudoDEGs: {os.path.basename(pseudoDEG_file)}\n")
    
    if verbose:
        print(f"Results saved to directory: {output_dir}")
        print(f"- All genes: {os.path.basename(all_results_file)}")
        print(f"- Statistically significant genes: {os.path.basename(stat_sig_file)}")
        print(f"- PseudoDEGs: {os.path.basename(pseudoDEG_file)}")
        print(f"- Summary: {os.path.basename(summary_file)}")

def summarize_results(
    results: pd.DataFrame, 
    top_n: int = 20,
    fdr_threshold: float = 0.05,
    effect_size_threshold: Optional[float] = None,
    top_n_genes: Optional[int] = None,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> None:
    total_genes = len(results)
    stat_sig = results[results["fdr"] < fdr_threshold].shape[0]
    pseudoDEGs = results[results["pseudoDEG"]].shape[0]
    
    # Create the summary text
    summary = []
    summary.append("=== SUMMARY OF RESULTS ===")
    summary.append(f"Total genes analyzed: {total_genes}")
    summary.append(f"Statistically significant genes (FDR < {fdr_threshold}): {stat_sig}")
    
    # Describe selection method based on whether top_n_genes was provided
    if top_n_genes is not None:
        summary.append(f"Selected pseudoDEGs (top {top_n_genes} by effect size): {pseudoDEGs}")
    else:
        summary.append(f"Selected pseudoDEGs (FDR < {fdr_threshold} AND effect size > {effect_size_threshold}): {pseudoDEGs}")
    
    if pseudoDEGs > 0:
        summary.append(f"\nTop {min(top_n, pseudoDEGs)} pseudoDEGs:")
        top_genes = results[results["pseudoDEG"]].sort_values(["fdr", "effect_size"], ascending=[True, False]).head(top_n)
        for i, (_, row) in enumerate(top_genes.iterrows(), 1):
            summary.append(f"{i}. {row['gene']}: FDR = {row['fdr']:.4e}, Effect Size = {row['effect_size']:.2f}")
    
    # Join all lines
    summary_text = "\n".join(summary)
    
    # Print to console if verbose
    if verbose:
        print(summary_text)
    
    # Write to file if output_file is provided
    if output_file is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(summary_text)
        
        if verbose:
            print(f"\nSummary saved to: {output_file}")