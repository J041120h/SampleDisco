#!/usr/bin/env python3
"""
Trajectory Two-way ANOVA Analysis with Batch and Disease Severity
Tests the effectiveness of pseudotime trajectory using Two-way ANOVA to assess
the effects of batch and severity level on pseudotime values
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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


def perform_twoway_anova(merged_df, pseudotime_col='pseudotime'):
    """
    Perform Two-way ANOVA to test effects of batch and severity on pseudotime
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe with batch, severity and pseudotime
    pseudotime_col : str
        Name of pseudotime column
    
    Returns:
    --------
    dict : ANOVA results and statistics
    """
    print("\nPerforming Two-way ANOVA...")
    
    # Check for required columns
    required_cols = ['batch', 'sev.level', pseudotime_col]
    for col in required_cols:
        if col not in merged_df.columns:
            raise ValueError(f"'{col}' column not found in data")
    
    # Prepare data - remove NaN and ensure factors are categorical
    anova_df = merged_df[['sample', 'batch', 'sev.level', pseudotime_col]].dropna()
    
    # IMPORTANT: Rename 'sev.level' to avoid formula parsing issues with the dot
    anova_df = anova_df.rename(columns={'sev.level': 'severity_level'})
    
    anova_df['batch'] = anova_df['batch'].astype('category')
    anova_df['severity_level'] = anova_df['severity_level'].astype('category')
    
    print(f"  Analyzing {anova_df.shape[0]} samples with complete data")
    print(f"  Number of batches: {anova_df['batch'].nunique()}")
    print(f"  Number of severity levels: {anova_df['severity_level'].nunique()}")
    
    # Create formula for two-way ANOVA with interaction
    formula = f'{pseudotime_col} ~ C(batch) + C(severity_level) + C(batch):C(severity_level)'
    
    # Fit the model
    model = ols(formula, data=anova_df).fit()
    
    # Perform ANOVA
    anova_table = anova_lm(model, typ=2)
    
    # Debug: Print available columns
    print(f"  ANOVA table columns: {list(anova_table.columns)}")
    
    # Calculate mean squares if not present
    if 'mean_sq' not in anova_table.columns:
        anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']
    
    # Calculate effect sizes (eta-squared)
    anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
    
    # Calculate partial eta-squared
    ss_total = anova_table['sum_sq'].sum()
    ss_residual = anova_table.loc['Residual', 'sum_sq']
    anova_table['partial_eta_sq'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + ss_residual)
    
    # Descriptive statistics by groups
    desc_stats = {
        'by_batch': anova_df.groupby('batch')[pseudotime_col].agg(['count', 'mean', 'std']),
        'by_severity': anova_df.groupby('severity_level')[pseudotime_col].agg(['count', 'mean', 'std']),
        'by_batch_severity': anova_df.groupby(['batch', 'severity_level'])[pseudotime_col].agg(['count', 'mean', 'std'])
    }
    
    # Post-hoc tests if main effects are significant
    posthoc_results = {}
    
    # Tukey HSD for severity levels if significant
    if 'PR(>F)' in anova_table.columns and anova_table.loc['C(severity_level)', 'PR(>F)'] < 0.05:
        tukey_severity = pairwise_tukeyhsd(anova_df[pseudotime_col], 
                                          anova_df['severity_level'], 
                                          alpha=0.05)
        posthoc_results['severity'] = tukey_severity
    
    # Tukey HSD for batch if significant
    if 'PR(>F)' in anova_table.columns and anova_table.loc['C(batch)', 'PR(>F)'] < 0.05:
        tukey_batch = pairwise_tukeyhsd(anova_df[pseudotime_col], 
                                       anova_df['batch'], 
                                       alpha=0.05)
        posthoc_results['batch'] = tukey_batch
    
    # Add back the original column name for compatibility with visualization
    anova_df['sev.level'] = anova_df['severity_level']
    
    results = {
        'anova_table': anova_table,
        'model': model,
        'desc_stats': desc_stats,
        'posthoc': posthoc_results,
        'data': anova_df,
        'pseudotime_col': pseudotime_col,
        'n_samples': anova_df.shape[0],
        'n_batches': anova_df['batch'].nunique(),
        'n_severity_levels': anova_df['severity_level'].nunique()
    }
    
    return results


def create_visualizations(results, output_dir):
    """
    Create visualization plots for the Two-way ANOVA analysis
    
    Parameters:
    -----------
    results : dict
        Results from ANOVA analysis
    output_dir : str
        Output directory path
    """
    print("\nCreating visualizations...")
    
    data = results['data']
    pseudotime_col = results['pseudotime_col']
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Interaction plot
    ax1 = plt.subplot(2, 3, 1)
    interaction_data = data.groupby(['sev.level', 'batch'])[pseudotime_col].mean().reset_index()
    for batch in interaction_data['batch'].unique():
        batch_data = interaction_data[interaction_data['batch'] == batch]
        ax1.plot(batch_data['sev.level'], batch_data[pseudotime_col], 
                marker='o', label=f'Batch {batch}', linewidth=2, markersize=8)
    ax1.set_xlabel('Severity Level')
    ax1.set_ylabel('Mean Pseudotime')
    ax1.set_title('Interaction Plot: Batch × Severity Level')
    ax1.legend(title='Batch')
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot by severity level
    ax2 = plt.subplot(2, 3, 2)
    data.boxplot(column=pseudotime_col, by='sev.level', ax=ax2)
    ax2.set_xlabel('Severity Level')
    ax2.set_ylabel('Pseudotime')
    ax2.set_title('Pseudotime by Severity Level')
    plt.sca(ax2)
    
    # 3. Box plot by batch
    ax3 = plt.subplot(2, 3, 3)
    data.boxplot(column=pseudotime_col, by='batch', ax=ax3)
    ax3.set_xlabel('Batch')
    ax3.set_ylabel('Pseudotime')
    ax3.set_title('Pseudotime by Batch')
    plt.sca(ax3)
    
    # 4. Violin plot with batch and severity
    ax4 = plt.subplot(2, 3, 4)
    sns.violinplot(data=data, x='sev.level', y=pseudotime_col, hue='batch', 
                   split=False, inner='box', ax=ax4)
    ax4.set_xlabel('Severity Level')
    ax4.set_ylabel('Pseudotime')
    ax4.set_title('Pseudotime Distribution by Severity and Batch')
    ax4.legend(title='Batch', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 5. Effect sizes visualization
    ax5 = plt.subplot(2, 3, 5)
    anova_table = results['anova_table'].iloc[:-1]  # Remove residual row
    effect_data = pd.DataFrame({
        'Factor': anova_table.index,
        'Partial Eta²': anova_table['partial_eta_sq'].values
    })
    bars = ax5.bar(range(len(effect_data)), effect_data['Partial Eta²'])
    ax5.set_xticks(range(len(effect_data)))
    ax5.set_xticklabels(['Batch', 'Severity', 'Batch×Severity'], rotation=45)
    ax5.set_ylabel('Partial Eta-Squared')
    ax5.set_title('Effect Sizes (Partial η²)')
    ax5.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, effect_data['Partial Eta²'])):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 6. Heatmap of mean pseudotime by batch and severity
    ax6 = plt.subplot(2, 3, 6)
    pivot_data = data.pivot_table(values=pseudotime_col, 
                                  index='sev.level', 
                                  columns='batch', 
                                  aggfunc='mean')
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax6)
    ax6.set_xlabel('Batch')
    ax6.set_ylabel('Severity Level')
    ax6.set_title('Mean Pseudotime Heatmap')
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, 'twoway_anova_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plots to: {plot_path}")


def generate_summary_report(results, output_dir):
    """
    Generate and save comprehensive summary report
    
    Parameters:
    -----------
    results : dict
        Results from ANOVA analysis
    output_dir : str
        Output directory path
    """
    print("\nGenerating summary report...")
    
    anova_table = results['anova_table']
    
    report_lines = [
        "="*80,
        "TWO-WAY ANOVA TRAJECTORY ANALYSIS REPORT",
        "="*80,
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-"*40,
        "DATASET SUMMARY",
        "-"*40,
        f"Total Samples Analyzed: {results['n_samples']}",
        f"Number of Batches: {results['n_batches']}",
        f"Number of Severity Levels: {results['n_severity_levels']}",
        "",
        "-"*40,
        "TWO-WAY ANOVA RESULTS",
        "-"*40,
        "",
        "ANOVA Table:",
        "="*70,
    ]
    
    # Format ANOVA table for report - check which columns are available
    available_cols = ['df', 'sum_sq']
    if 'mean_sq' in anova_table.columns:
        available_cols.append('mean_sq')
    if 'F' in anova_table.columns:
        available_cols.append('F')
    if 'PR(>F)' in anova_table.columns:
        available_cols.append('PR(>F)')
    available_cols.append('partial_eta_sq')
    
    anova_display = anova_table[available_cols].round(4)
    report_lines.append(anova_display.to_string())
    
    report_lines.extend([
        "",
        "-"*40,
        "MAIN EFFECTS INTERPRETATION",
        "-"*40,
    ])
    
    # Batch effect interpretation
    batch_p = anova_table.loc['C(batch)', 'PR(>F)'] if 'PR(>F)' in anova_table.columns else None
    batch_eta = anova_table.loc['C(batch)', 'partial_eta_sq']
    batch_f = anova_table.loc['C(batch)', 'F'] if 'F' in anova_table.columns else None
    
    report_lines.extend([
        "",
        f"1. BATCH EFFECT:",
    ])
    if batch_f is not None:
        report_lines.append(f"   F-statistic: {batch_f:.3f}")
    if batch_p is not None:
        report_lines.append(f"   p-value: {batch_p:.4e}")
    report_lines.append(f"   Partial η²: {batch_eta:.4f}")
    
    if batch_p is not None and batch_p < 0.05:
        effect_size = "large" if batch_eta > 0.14 else "medium" if batch_eta > 0.06 else "small"
        report_lines.append(f"   ✗ SIGNIFICANT batch effect detected ({effect_size} effect size)")
        report_lines.append("   → Batch correction may be necessary for reliable trajectory inference")
    elif batch_p is not None:
        report_lines.append("   ✓ No significant batch effect detected")
        report_lines.append("   → Trajectory is robust to batch variations")
    
    # Severity effect interpretation
    severity_p = anova_table.loc['C(severity_level)', 'PR(>F)'] if 'PR(>F)' in anova_table.columns else None
    severity_eta = anova_table.loc['C(severity_level)', 'partial_eta_sq']
    severity_f = anova_table.loc['C(severity_level)', 'F'] if 'F' in anova_table.columns else None
    
    report_lines.extend([
        "",
        f"2. SEVERITY LEVEL EFFECT:",
    ])
    if severity_f is not None:
        report_lines.append(f"   F-statistic: {severity_f:.3f}")
    if severity_p is not None:
        report_lines.append(f"   p-value: {severity_p:.4e}")
    report_lines.append(f"   Partial η²: {severity_eta:.4f}")
    
    if severity_p is not None and severity_p < 0.05:
        effect_size = "large" if severity_eta > 0.14 else "medium" if severity_eta > 0.06 else "small"
        report_lines.append(f"   ✓ SIGNIFICANT severity effect detected ({effect_size} effect size)")
        report_lines.append("   → Trajectory effectively captures disease progression")
    elif severity_p is not None:
        report_lines.append("   ✗ No significant severity effect detected")
        report_lines.append("   → Trajectory does NOT capture disease progression well")
    
    # Interaction effect interpretation
    interaction_p = anova_table.loc['C(batch):C(severity_level)', 'PR(>F)'] if 'PR(>F)' in anova_table.columns else None
    interaction_eta = anova_table.loc['C(batch):C(severity_level)', 'partial_eta_sq']
    interaction_f = anova_table.loc['C(batch):C(severity_level)', 'F'] if 'F' in anova_table.columns else None
    
    report_lines.extend([
        "",
        f"3. INTERACTION EFFECT (Batch × Severity):",
    ])
    if interaction_f is not None:
        report_lines.append(f"   F-statistic: {interaction_f:.3f}")
    if interaction_p is not None:
        report_lines.append(f"   p-value: {interaction_p:.4e}")
    report_lines.append(f"   Partial η²: {interaction_eta:.4f}")
    
    if interaction_p is not None and interaction_p < 0.05:
        report_lines.append("   ⚠ SIGNIFICANT interaction detected")
        report_lines.append("   → Batch effects vary across severity levels")
        report_lines.append("   → Interpretation of main effects should be cautious")
    elif interaction_p is not None:
        report_lines.append("   ✓ No significant interaction detected")
        report_lines.append("   → Batch effects are consistent across severity levels")
    
    # Overall effectiveness assessment
    report_lines.extend([
        "",
        "="*40,
        "TRAJECTORY EFFECTIVENESS ASSESSMENT",
        "="*40,
    ])
    
    # Determine overall effectiveness
    if severity_p is not None and batch_p is not None:
        if severity_p < 0.05 and batch_p >= 0.05:
            effectiveness = "EXCELLENT"
            effectiveness_desc = "The trajectory captures disease progression without batch confounding"
        elif severity_p < 0.05 and batch_p < 0.05 and batch_eta < severity_eta:
            effectiveness = "GOOD"
            effectiveness_desc = "The trajectory captures disease progression but shows some batch effects"
        elif severity_p < 0.05 and batch_p < 0.05 and batch_eta >= severity_eta:
            effectiveness = "MODERATE"
            effectiveness_desc = "Disease progression is detected but batch effects are substantial"
        elif severity_p >= 0.05 and batch_p < 0.05:
            effectiveness = "POOR"
            effectiveness_desc = "The trajectory is dominated by batch effects, not biological signal"
        else:
            effectiveness = "INEFFECTIVE"
            effectiveness_desc = "The trajectory does not capture meaningful biological variation"
    else:
        effectiveness = "UNDETERMINED"
        effectiveness_desc = "Unable to determine effectiveness due to missing p-values"
    
    report_lines.extend([
        f"Overall Assessment: {effectiveness}",
        f"→ {effectiveness_desc}",
        "",
        "Effect Size Guidelines:",
        "  Small effect:  η² = 0.01-0.06",
        "  Medium effect: η² = 0.06-0.14",
        "  Large effect:  η² > 0.14",
    ])
    
    # Add post-hoc results if available
    if results['posthoc']:
        report_lines.extend([
            "",
            "-"*40,
            "POST-HOC ANALYSIS (Tukey HSD)",
            "-"*40,
        ])
        
        if 'severity' in results['posthoc']:
            report_lines.extend([
                "",
                "Pairwise comparisons between severity levels:",
                str(results['posthoc']['severity']),
            ])
        
        if 'batch' in results['posthoc']:
            report_lines.extend([
                "",
                "Pairwise comparisons between batches:",
                str(results['posthoc']['batch']),
            ])
    
    # Add descriptive statistics
    report_lines.extend([
        "",
        "-"*40,
        "DESCRIPTIVE STATISTICS",
        "-"*40,
        "",
        "By Severity Level:",
        results['desc_stats']['by_severity'].round(4).to_string(),
        "",
        "By Batch:",
        results['desc_stats']['by_batch'].round(4).to_string(),
    ])
    
    # Recommendations
    report_lines.extend([
        "",
        "-"*40,
        "RECOMMENDATIONS",
        "-"*40,
    ])
    
    if effectiveness == "EXCELLENT":
        report_lines.extend([
            "1. The trajectory is ready for downstream analysis",
            "2. Consider using for gene expression dynamics studies",
            "3. Validate findings with independent cohorts",
            "4. Investigate key drivers of the trajectory"
        ])
    elif effectiveness in ["GOOD", "MODERATE"]:
        report_lines.extend([
            "1. Consider batch correction methods (e.g., ComBat, Harmony)",
            "2. Verify trajectory with batch-corrected data",
            "3. Include batch as covariate in downstream analyses",
            "4. Validate key findings within individual batches"
        ])
    elif effectiveness in ["POOR", "INEFFECTIVE"]:
        report_lines.extend([
            "1. Strongly recommend batch correction before trajectory inference",
            "2. Re-evaluate trajectory method parameters",
            "3. Consider alternative dimensionality reduction approaches",
            "4. Check for other confounding factors",
            "5. Verify data quality and preprocessing steps"
        ])
    else:  # UNDETERMINED
        report_lines.extend([
            "1. Check the ANOVA model fitting for potential issues",
            "2. Verify data format and column types",
            "3. Consider alternative statistical approaches"
        ])
    
    report_lines.extend([
        "",
        "="*80,
        "END OF REPORT",
        "="*80
    ])
    
    # Save report
    report_path = os.path.join(output_dir, 'twoway_anova_summary.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"  Saved summary report to: {report_path}")
    
    # Print key results to console
    print("\n" + "="*50)
    print("KEY RESULTS:")
    if batch_p is not None:
        print(f"  Batch Effect: p = {batch_p:.4e} (η² = {batch_eta:.3f})")
    else:
        print(f"  Batch Effect: η² = {batch_eta:.3f}")
    if severity_p is not None:
        print(f"  Severity Effect: p = {severity_p:.4e} (η² = {severity_eta:.3f})")
    else:
        print(f"  Severity Effect: η² = {severity_eta:.3f}")
    if interaction_p is not None:
        print(f"  Interaction: p = {interaction_p:.4e} (η² = {interaction_eta:.3f})")
    else:
        print(f"  Interaction: η² = {interaction_eta:.3f}")
    print(f"  Overall Assessment: {effectiveness}")
    print("="*50)


def run_trajectory_anova_analysis(meta_csv_path, pseudotime_csv_path, output_dir_path):
    """
    Main wrapper function that runs the complete Two-way ANOVA trajectory analysis pipeline
    
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
    output_dir_path = os.path.join(output_dir_path, 'ANOVA')
    os.makedirs(output_dir_path, exist_ok=True)
    
    print("\n" + "="*50)
    print("STARTING TWO-WAY ANOVA TRAJECTORY ANALYSIS")
    print("="*50)
    
    try:
        # Step 1: Load data
        meta_df, pseudotime_df = load_data(meta_csv_path, pseudotime_csv_path)
        
        # Step 2: Merge data
        merged_df = merge_data(meta_df, pseudotime_df)
        
        # Step 3: Perform Two-way ANOVA
        results = perform_twoway_anova(merged_df, pseudotime_col='pseudotime')
        
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
    # Users will directly modify these paths
    run_trajectory_anova_analysis(
        meta_csv_path="/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv",
        pseudotime_csv_path='/dcs07/hongkai/data/harry/result/Benchmark/covid_25_sample/rna/CCA/pseudotime_expression.csv',
        output_dir_path='/dcs07/hongkai/data/harry/result/Benchmark/covid_25_sample/twoway_anova_results'
    )