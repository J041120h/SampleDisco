import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
from pygam import LinearGAM

def visualize_gene_expression(
    gene: str,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    gam_model: LinearGAM,
    stats_df: pd.DataFrame,
    output_dir: str,
    gene_subfolder: str = "gene_plots",
    figsize: tuple = (10, 6),
    title_prefix: str = "Gene Expression Pattern:",
    point_size: int = 30,
    point_alpha: float = 0.6,
    line_width: int = 2,
    line_color: str = "red",
    dpi: int = 300,
    verbose: bool = False
) -> str:
    """
    Visualize the gene expression across pseudotime with GAM fit
    
    Parameters
    ----------
    gene : str
        Gene name to visualize
    X : pd.DataFrame
        Design matrix with pseudotime and covariates
    Y : pd.DataFrame
        Gene expression matrix
    gam_model : LinearGAM
        Fitted GAM model for the gene
    stats_df : pd.DataFrame
        DataFrame with statistics including FDR and effect size
    output_dir : str
        Base directory to save the visualization
    gene_subfolder : str
        Subfolder name for gene plots
    figsize : tuple
        Figure size
    title_prefix : str
        Prefix for plot title
    point_size : int
        Size of data points
    point_alpha : float
        Alpha (transparency) for data points
    line_width : int
        Width of the GAM fit line
    line_color : str
        Color of the GAM fit line
    dpi : int
        Resolution of saved figure
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    str
        Path to the saved figure
    """
    # Ensure the output directory exists
    plot_dir = os.path.join(output_dir, gene_subfolder)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Extract pseudotime and gene expression values
    pseudotime = X["pseudotime"].values
    expression = Y[gene].values
    
    # Get the GAM predictions
    X_pred = X.copy()
    y_pred = gam_model.predict(X.values)
    
    # Sort by pseudotime for smooth line plotting
    sort_idx = np.argsort(pseudotime)
    pseudotime_sorted = pseudotime[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    # Get statistics for this gene
    if gene in stats_df["gene"].values:
        gene_stats = stats_df[stats_df["gene"] == gene].iloc[0]
        fdr = gene_stats["fdr"]
        effect_size = gene_stats["effect_size"]
    else:
        fdr = np.nan
        effect_size = np.nan
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot the raw data points
    plt.scatter(pseudotime, expression, alpha=point_alpha, s=point_size, label="Expression")
    
    # Plot the GAM fit
    plt.plot(pseudotime_sorted, y_pred_sorted, color=line_color, linewidth=line_width, label="GAM fit")
    
    # Add title and labels
    plt.title(f"{title_prefix} {gene} (FDR: {fdr:.2e}, Effect Size: {effect_size:.2f})")
    plt.xlabel("Pseudotime")
    plt.ylabel("Expression")
    plt.legend()
    
    # Save the figure
    file_path = os.path.join(plot_dir, f"{gene}_pseudotime.png")
    plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    if verbose:
        print(f"Visualization for gene {gene} saved to {file_path}")
    
    return file_path

def visualize_all_deg_genes(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    gam_models: Dict[str, LinearGAM],
    results_df: pd.DataFrame,
    output_dir: str,
    gene_subfolder: str = "gene_plots",
    top_n_heatmap: int = 50,
    verbose: bool = False
) -> List[str]:
    """
    Visualize all differentially expressed genes
    
    Parameters
    ----------
    X : pd.DataFrame
        Design matrix with pseudotime and covariates
    Y : pd.DataFrame
        Gene expression matrix
    gam_models : Dict[str, LinearGAM]
        Dictionary of fitted GAM models for each gene
    results_df : pd.DataFrame
        DataFrame with statistics including FDR and effect size
    output_dir : str
        Base directory to save the visualizations
    gene_subfolder : str
        Subfolder name for gene plots
    top_n_heatmap : int
        Number of top genes to include in the heatmap
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    List[str]
        Paths to the saved figures
    """
    # Ensure the output directory exists
    plot_dir = os.path.join(output_dir, gene_subfolder)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get DEG genes
    deg_genes = results_df[results_df["pseudoDEG"]]["gene"].tolist()
    
    if verbose:
        print(f"Generating visualizations for {len(deg_genes)} differentially expressed genes...")
    
    # Create individual plots for each gene
    saved_paths = []
    for gene in deg_genes:
        if gene in gam_models:
            try:
                file_path = visualize_gene_expression(
                    gene=gene,
                    X=X,
                    Y=Y,
                    gam_model=gam_models[gene],
                    stats_df=results_df,
                    output_dir=output_dir,
                    gene_subfolder=gene_subfolder,
                    verbose=False  # Avoid too many print statements
                )
                saved_paths.append(file_path)
            except Exception as e:
                if verbose:
                    print(f"Error visualizing gene {gene}: {e}")
    
    # Generate a summary heatmap of top DEGs
    try:
        heatmap_path = generate_deg_heatmap(
            X=X,
            Y=Y,
            results_df=results_df,
            top_n=top_n_heatmap,
            output_dir=output_dir,
            verbose=verbose
        )
        saved_paths.append(heatmap_path)
    except Exception as e:
        if verbose:
            print(f"Error generating heatmap: {e}")
    
    if verbose:
        print(f"Generated {len(saved_paths)} visualizations")
    
    return saved_paths

def generate_deg_heatmap(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    results_df: pd.DataFrame,
    top_n: int = 50,
    output_dir: str = None,
    figsize: tuple = (12, 10),
    dpi: int = 300,
    verbose: bool = False
) -> str:
    """
    Generate a heatmap of top differentially expressed genes across pseudotime
    
    Parameters
    ----------
    X : pd.DataFrame
        Design matrix with pseudotime and covariates
    Y : pd.DataFrame
        Gene expression matrix
    results_df : pd.DataFrame
        DataFrame with statistics including FDR and effect size
    top_n : int
        Number of top genes to include in the heatmap
    output_dir : str
        Directory to save the visualization
    figsize : tuple
        Figure size
    dpi : int
        Resolution of saved figure
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    str
        Path to the saved figure
    """
    if verbose:
        print(f"Generating heatmap for top {top_n} differentially expressed genes...")
    
    # Get top DEGs
    top_degs = results_df[results_df["pseudoDEG"]].sort_values("effect_size", ascending=False).head(top_n)
    genes = top_degs["gene"].tolist()
    
    if len(genes) == 0:
        if verbose:
            print("No differentially expressed genes found for heatmap")
        return None
    
    # Prepare data for heatmap
    pseudotime = X["pseudotime"].values
    expr_data = Y[genes].values
    
    # Sort samples by pseudotime
    sort_idx = np.argsort(pseudotime)
    pseudotime_sorted = pseudotime[sort_idx]
    expr_data_sorted = expr_data[sort_idx, :]
    
    # Z-score normalize each gene
    from scipy.stats import zscore
    expr_data_norm = np.zeros_like(expr_data_sorted)
    for i in range(expr_data_sorted.shape[1]):
        expr_data_norm[:, i] = zscore(expr_data_sorted[:, i])
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create heatmap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    ax = sns.heatmap(expr_data_norm.T, cmap=cmap, center=0, 
                     xticklabels=False, yticklabels=genes if len(genes) <= 30 else False)
    
    # Add a secondary x-axis for pseudotime
    ax2 = ax.twiny()
    ax2.plot(np.arange(len(pseudotime_sorted)), pseudotime_sorted, alpha=0)
    ax2.set_xlabel("Pseudotime")
    
    # Set titles and labels
    plt.title(f"Top {len(genes)} Differentially Expressed Genes Across Pseudotime")
    ax.set_ylabel("Genes")
    ax.set_xlabel("Samples (sorted by pseudotime)")
    
    # Add colorbar label
    plt.colorbar(ax.collections[0], label="Z-score")
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "top_degs_heatmap.png")
    plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    if verbose:
        print(f"Heatmap saved to {file_path}")
    
    return file_path

def generate_summary_trajectory_plot(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    gam_models: Dict[str, LinearGAM],
    results_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 10,
    figsize: tuple = (12, 8),
    dpi: int = 300,
    verbose: bool = False
) -> str:
    """
    Generate a summary plot with the top DEGs trajectories
    
    Parameters
    ----------
    X : pd.DataFrame
        Design matrix with pseudotime and covariates
    Y : pd.DataFrame
        Gene expression matrix
    gam_models : Dict[str, LinearGAM]
        Dictionary of fitted GAM models for each gene
    results_df : pd.DataFrame
        DataFrame with statistics including FDR and effect size
    output_dir : str
        Directory to save the visualization
    top_n : int
        Number of top genes to include in the plot
    figsize : tuple
        Figure size
    dpi : int
        Resolution of saved figure
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    str
        Path to the saved figure
    """
    if verbose:
        print(f"Generating summary trajectory plot for top {top_n} genes...")
    
    # Get top DEGs
    top_degs = results_df[results_df["pseudoDEG"]].sort_values("effect_size", ascending=False).head(top_n)
    genes = top_degs["gene"].tolist()
    
    if len(genes) == 0:
        if verbose:
            print("No differentially expressed genes found for summary plot")
        return None
    
    # Extract pseudotime and sort it
    pseudotime = X["pseudotime"].values
    sort_idx = np.argsort(pseudotime)
    pseudotime_sorted = pseudotime[sort_idx]
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot GAM fits for each gene
    for gene in genes:
        if gene in gam_models:
            # Get the GAM predictions
            y_pred = gam_models[gene].predict(X.values)
            y_pred_sorted = y_pred[sort_idx]
            
            # Z-score normalize for better visualization
            from scipy.stats import zscore
            y_pred_norm = zscore(y_pred_sorted)
            
            # Get gene stats
            gene_stats = results_df[results_df["gene"] == gene].iloc[0]
            effect_size = gene_stats["effect_size"]
            
            # Plot the trajectory
            plt.plot(pseudotime_sorted, y_pred_norm, 
                     label=f"{gene} (ES: {effect_size:.2f})", 
                     linewidth=2, alpha=0.8)
    
    # Add title and labels
    plt.title(f"Top {len(genes)} Differentially Expressed Genes - Trajectory Comparison")
    plt.xlabel("Pseudotime")
    plt.ylabel("Normalized Expression (Z-score)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "top_degs_trajectories.png")
    plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    if verbose:
        print(f"Summary trajectory plot saved to {file_path}")
    
    return file_path