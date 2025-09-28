#!/usr/bin/env python3
"""
Trajectory Spearman Correlation Analysis with Disease Severity Scores
Tests the effectiveness of pseudotime trajectory correlation with disease severity levels
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data(meta_path, pseudotime_path):
    """
    Load metadata and pseudotime data from CSV files
    
    Parameters:
    -----------
    meta_path : str
        Path to sample metadata CSV
    pseudotime_path : str
        Path to pseudotime CSV
    
    Returns:
    --------
    tuple : (meta_df, pseudotime_df)
    """
    print("Loading data files...")
    
    # Load metadata
    meta_df = pd.read_csv(meta_path)
    print(f"  Loaded metadata: {meta_df.shape[0]} samples")
    
    # Load pseudotime
    pseudotime_df = pd.read_csv(pseudotime_path)
    print(f"  Loaded pseudotime: {pseudotime_df.shape[0]} samples")
    
    return meta_df, pseudotime_df


def merge_data(meta_df, pseudotime_df):
    """
    Merge metadata and pseudotime data on sample column
    
    Parameters:
    -----------
    meta_df : pd.DataFrame
        Metadata dataframe
    pseudotime_df : pd.DataFrame
        Pseudotime dataframe
    
    Returns:
    --------
    pd.DataFrame : Merged dataframe
    """
    print("\nMerging data...")
    
    # Check for sample column
    if 'sample' not in meta_df.columns:
        raise ValueError("'sample' column not found in metadata")
    
    if 'sample' not in pseudotime_df.columns:
        raise ValueError("'sample' column not found in pseudotime data")
    
    # Merge dataframes
    merged_df = pd.merge(meta_df, pseudotime_df, on='sample', how='inner')
    print(f"  Merged data: {merged_df.shape[0]} samples retained")
    
    return merged_df


def calculate_spearman_correlation(merged_df, pseudotime_col='pseudotime'):
    """
    Calculate Spearman correlation between pseudotime and severity level
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe with severity and pseudotime
    pseudotime_col : str
        Name of pseudotime column (default: 'Pseudotime')
    
    Returns:
    --------
    dict : Correlation statistics
    """
    print("\nCalculating Spearman correlation...")
    
    # Check for required columns
    if 'sev.level' not in merged_df.columns:
        raise ValueError("'sev.level' column not found in data")
    
    if pseudotime_col not in merged_df.columns:
        raise ValueError(f"'{pseudotime_col}' column not found in data. Available columns: {list(merged_df.columns)}")
    
    # Remove rows with NaN values
    clean_df = merged_df[['sample', 'sev.level', pseudotime_col]].dropna()
    print(f"  Analyzing {clean_df.shape[0]} samples with complete data")
    
    # Calculate Spearman correlation
    spearman_corr, p_value = stats.spearmanr(clean_df['sev.level'], 
                                             clean_df[pseudotime_col])
    
    # Calculate additional statistics
    kendall_tau, kendall_p = stats.kendalltau(clean_df['sev.level'], 
                                               clean_df[pseudotime_col])
    
    # Pearson correlation for comparison
    pearson_corr, pearson_p = stats.pearsonr(clean_df['sev.level'], 
                                             clean_df[pseudotime_col])
    
    # Group statistics by severity level
    severity_stats = clean_df.groupby('sev.level')[pseudotime_col].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    
    results = {
        'spearman_corr': spearman_corr,
        'spearman_p': p_value,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p,
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'n_samples': clean_df.shape[0],
        'severity_stats': severity_stats,
        'data': clean_df,
        'pseudotime_col': pseudotime_col
    }
    
    return results


def create_visualizations(results, output_dir):
    """
    Create visualization plots for the analysis
    
    Parameters:
    -----------
    results : dict
        Results from correlation analysis
    output_dir : str
        Output directory path
    """
    print("\nCreating visualizations...")
    
    data = results['data']
    pseudotime_col = results['pseudotime_col']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Scatter plot with regression line
    ax1 = axes[0, 0]
    ax1.scatter(data['sev.level'], data[pseudotime_col], alpha=0.6)
    z = np.polyfit(data['sev.level'], data[pseudotime_col], 1)
    p = np.poly1d(z)
    ax1.plot(data['sev.level'].unique(), p(data['sev.level'].unique()), 
            "r--", alpha=0.8, label='Linear fit')
    ax1.set_xlabel('Severity Level')
    ax1.set_ylabel('Pseudotime')
    ax1.set_title(f'Pseudotime vs Severity\n(Spearman ρ = {results["spearman_corr"]:.3f}, p = {results["spearman_p"]:.3e})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot by severity level
    ax2 = axes[0, 1]
    data.boxplot(column=pseudotime_col, by='sev.level', ax=ax2)
    ax2.set_xlabel('Severity Level')
    ax2.set_ylabel('Pseudotime')
    ax2.set_title('Pseudotime Distribution by Severity Level')
    plt.sca(ax2)
    plt.xticks(range(1, int(data['sev.level'].max()) + 1))
    
    # 3. Violin plot
    ax3 = axes[1, 0]
    severity_levels = sorted(data['sev.level'].unique())
    violin_data = [data[data['sev.level'] == level][pseudotime_col].values 
                  for level in severity_levels]
    positions = range(len(severity_levels))
    parts = ax3.violinplot(violin_data, positions=positions, showmeans=True)
    ax3.set_xticks(positions)
    ax3.set_xticklabels(severity_levels)
    ax3.set_xlabel('Severity Level')
    ax3.set_ylabel('Pseudotime')
    ax3.set_title('Pseudotime Distribution (Violin Plot)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation heatmap
    ax4 = axes[1, 1]
    corr_matrix = pd.DataFrame({
        'Spearman': [results['spearman_corr']],
        'Kendall': [results['kendall_tau']],
        'Pearson': [results['pearson_corr']]
    })
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, ax=ax4, cbar_kws={'label': 'Correlation'})
    ax4.set_title('Correlation Coefficients')
    ax4.set_yticklabels([''], rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, 'trajectory_correlation_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plots to: {plot_path}")


def generate_summary_report(results, output_dir):
    """
    Generate and save summary report
    
    Parameters:
    -----------
    results : dict
        Results from correlation analysis
    output_dir : str
        Output directory path
    """
    print("\nGenerating summary report...")
    
    report_lines = [
        "="*80,
        "TRAJECTORY SPEARMAN CORRELATION ANALYSIS REPORT",
        "="*80,
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-"*40,
        "SUMMARY STATISTICS",
        "-"*40,
        f"Total Samples Analyzed: {results['n_samples']}",
        "",
        "CORRELATION RESULTS:",
        f"  Spearman Correlation: {results['spearman_corr']:.4f} (p-value: {results['spearman_p']:.4e})",
        f"  Kendall Tau:          {results['kendall_tau']:.4f} (p-value: {results['kendall_p']:.4e})",
        f"  Pearson Correlation:  {results['pearson_corr']:.4f} (p-value: {results['pearson_p']:.4e})",
        "",
        "-"*40,
        "INTERPRETATION",
        "-"*40,
    ]
    
    # Add interpretation
    corr = results['spearman_corr']
    p_val = results['spearman_p']
    
    if abs(corr) < 0.3:
        strength = "weak"
    elif abs(corr) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    direction = "positive" if corr > 0 else "negative"
    
    if p_val < 0.001:
        significance = "highly significant (p < 0.001)"
    elif p_val < 0.01:
        significance = "significant (p < 0.01)"
    elif p_val < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p >= 0.05)"
    
    report_lines.extend([
        f"The Spearman correlation coefficient of {corr:.4f} indicates a {strength} {direction}",
        f"relationship between pseudotime and disease severity.",
        f"This correlation is {significance}.",
        "",
        "EFFECTIVENESS ASSESSMENT:",
    ])
    
    if abs(corr) > 0.5 and p_val < 0.05:
        report_lines.append("✓ The trajectory shows GOOD effectiveness in capturing disease progression")
        report_lines.append("  The pseudotime values correlate well with severity levels.")
    elif abs(corr) > 0.3 and p_val < 0.05:
        report_lines.append("⚠ The trajectory shows MODERATE effectiveness in capturing disease progression")
        report_lines.append("  There is a detectable but not strong relationship with severity.")
    else:
        report_lines.append("✗ The trajectory shows POOR effectiveness in capturing disease progression")
        report_lines.append("  The pseudotime values do not correlate well with severity levels.")
    
    report_lines.extend([
        "",
        "-"*40,
        "SEVERITY LEVEL STATISTICS",
        "-"*40,
        "Pseudotime statistics by severity level:",
        "",
        results['severity_stats'].to_string(),
        "",
        "-"*40,
        "RECOMMENDATIONS",
        "-"*40,
    ])
    
    if abs(corr) < 0.3:
        report_lines.extend([
            "1. Consider revising the trajectory inference method",
            "2. Check for batch effects or confounding variables",
            "3. Verify that the chosen root/starting point is appropriate",
            "4. Consider using alternative dimensionality reduction methods"
        ])
    elif abs(corr) < 0.7:
        report_lines.extend([
            "1. The trajectory captures some disease progression patterns",
            "2. Consider fine-tuning parameters for better resolution",
            "3. Investigate samples with high residuals for biological insights",
            "4. May benefit from incorporating additional biological features"
        ])
    else:
        report_lines.extend([
            "1. The trajectory effectively captures disease progression",
            "2. Can be used confidently for downstream analyses",
            "3. Consider validating with independent cohorts",
            "4. Investigate genes/features driving the trajectory"
        ])
    
    report_lines.extend([
        "",
        "="*80,
        "END OF REPORT",
        "="*80
    ])
    
    # Save report
    report_path = os.path.join(output_dir, 'trajectory_correlation_summary.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"  Saved summary report to: {report_path}")
    
    # Print key results to console
    print("\n" + "="*50)
    print("KEY RESULTS:")
    print(f"  Spearman ρ = {corr:.4f} (p = {p_val:.4e})")
    print(f"  Assessment: {strength.upper()} {direction} correlation")
    print(f"  Statistical significance: {significance}")
    print("="*50)


def run_trajectory_analysis(meta_csv_path, pseudotime_csv_path, output_dir_path):
    """
    Main wrapper function that runs the complete trajectory analysis pipeline
    
    Parameters:
    -----------
    meta_csv_path : str
        Path to sample metadata CSV file
    pseudotime_csv_path : str
        Path to pseudotime CSV file
    output_dir_path : str
        Output directory path
    
    Returns:
    --------
    dict : Results dictionary containing all analysis outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir_path, exist_ok=True)
    output_dir_path = os.path.join(output_dir_path, 'Spearman_Correlation')
    os.makedirs(output_dir_path, exist_ok=True)
    
    
    print("\n" + "="*50)
    print("STARTING TRAJECTORY CORRELATION ANALYSIS")
    print("="*50)
    
    try:
        # Step 1: Load data
        meta_df, pseudotime_df = load_data(meta_csv_path, pseudotime_csv_path)
        
        # Step 2: Merge data
        merged_df = merge_data(meta_df, pseudotime_df)
        
        # Step 3: Calculate correlations (using 'Pseudotime' column)
        results = calculate_spearman_correlation(merged_df, pseudotime_col='pseudotime')
        
        # Step 4: Create visualizations
        create_visualizations(results, output_dir_path)
        
        # Step 5: Generate summary report
        generate_summary_report(results, output_dir_path)
        
        print("\n✓ Analysis complete! Check the output directory for results.")
        print(f"  Output directory: {output_dir_path}")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the main analysis pipeline
    run_trajectory_analysis(
        meta_csv_path= "/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv",
        pseudotime_csv_path='/dcs07/hongkai/data/harry/result/Benchmark/covid_25_sample/rna/CCA/pseudotime_expression.csv',
        output_dir_path= '/dcs07/hongkai/data/harry/result/Benchmark/covid_25_sample'
    )