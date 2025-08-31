import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict, List
import warnings
import os
warnings.filterwarnings('ignore')

def parse_sample_names(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Parse sample names to extract sample IDs and modalities.
    
    Args:
        df: Distance matrix DataFrame
    
    Returns:
        sample_to_modality: Dict mapping full name to modality
        sample_pairs: Dict mapping sample ID to list of full names
    """
    sample_to_modality = {}
    sample_pairs = {}
    
    for name in df.index:
        # Split by underscore to get sample ID and modality
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            sample_id, modality = parts
            sample_to_modality[name] = modality
            
            if sample_id not in sample_pairs:
                sample_pairs[sample_id] = []
            sample_pairs[sample_id].append(name)
    
    return sample_to_modality, sample_pairs

def calculate_cross_modality_distance(df: pd.DataFrame, 
                                     sample_pairs: Dict) -> Tuple[float, List[float]]:
    """
    Calculate average distance between same samples from different modalities.
    
    Args:
        df: Distance matrix
        sample_pairs: Dict mapping sample ID to list of full names
    
    Returns:
        mean_distance: Average cross-modality distance
        distances: List of individual cross-modality distances
    """
    distances = []
    
    for sample_id, names in sample_pairs.items():
        if len(names) == 2:  # Should have exactly 2 modalities
            # Get distance between the two modalities for this sample
            dist = df.loc[names[0], names[1]]
            distances.append(dist)
    
    return np.mean(distances) if distances else 0, distances

def permutation_test(df: pd.DataFrame, 
                    sample_pairs: Dict,
                    n_permutations: int = 1000,
                    random_seed: int = 42) -> Tuple[float, float, np.ndarray, List[float]]:
    """
    Perform permutation test to assess significance of cross-modality distances.
    
    Args:
        df: Distance matrix
        sample_pairs: Dict mapping sample ID to list of full names
        n_permutations: Number of permutations
        random_seed: Random seed for reproducibility
    
    Returns:
        observed_mean: Observed mean cross-modality distance
        p_value: P-value from permutation test
        null_distribution: Array of permuted mean distances
        observed_distances: List of observed cross-modality distances
    """
    np.random.seed(random_seed)
    
    # Calculate observed statistic
    observed_mean, observed_distances = calculate_cross_modality_distance(df, sample_pairs)
    print(f"    Observed mean cross-modality distance: {observed_mean:.4f}")
    
    # Get all sample names and their modalities
    all_samples = list(df.index)
    atac_samples = [s for s in all_samples if s.endswith('_ATAC')]
    rna_samples = [s for s in all_samples if s.endswith('_RNA')]
    
    print(f"    Running {n_permutations} permutations...")
    # Permutation test
    null_distribution = []
    
    # Print progress for permutations
    progress_points = [int(n_permutations * p) for p in [0.25, 0.5, 0.75]]
    
    for i in range(n_permutations):
        if i in progress_points:
            print(f"      {int(100 * i / n_permutations)}% complete ({i}/{n_permutations})")
        
        # Shuffle the modality labels
        shuffled_atac = np.random.permutation(atac_samples)
        shuffled_rna = np.random.permutation(rna_samples)
        
        # Create new sample pairs based on shuffled labels
        shuffled_pairs = {}
        for j, (atac, rna) in enumerate(zip(shuffled_atac, shuffled_rna)):
            sample_id = f"shuffled_{j}"
            shuffled_pairs[sample_id] = [atac, rna]
        
        # Calculate mean distance for shuffled pairs
        shuffled_mean, _ = calculate_cross_modality_distance(df, shuffled_pairs)
        null_distribution.append(shuffled_mean)
    
    print(f"      100% complete ({n_permutations}/{n_permutations})")
    
    null_distribution = np.array(null_distribution)
    
    # Calculate p-value (two-tailed)
    p_value = np.mean(null_distribution <= observed_mean)
    print(f"    P-value: {p_value:.4f}")
    
    return observed_mean, p_value, null_distribution, observed_distances

def create_pvalue_plot(observed_mean: float, 
                      p_value: float, 
                      null_distribution: np.ndarray,
                      output_path: str):
    """
    Create and save p-value distribution plot.
    
    Args:
        observed_mean: Observed mean cross-modality distance
        p_value: P-value from permutation test
        null_distribution: Array of permuted mean distances
        output_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Histogram of null distribution
    plt.hist(null_distribution, bins=50, alpha=0.7, color='gray', 
             edgecolor='black', label='Null distribution')
    
    # Observed value line
    plt.axvline(observed_mean, color='red', linestyle='--', linewidth=2,
                label=f'Observed mean: {observed_mean:.4f}')
    
    plt.xlabel('Average Cross-Modality Distance')
    plt.ylabel('Frequency')
    plt.title(f'Permutation Test (p-value = {p_value:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def validate_cross_modality_distance(csv_path: str, 
                                    n_permutations: int = 1000,
                                    random_seed: int = 42) -> Dict:
    """
    Main function to validate cross-modality distances.
    
    Args:
        csv_path: Path to the distance matrix CSV file
        n_permutations: Number of permutations for the test
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing test results
    """
    print(f"  Loading distance matrix from: {csv_path}")
    # Load the distance matrix
    df = pd.read_csv(csv_path, index_col=0)
    print(f"  Matrix shape: {df.shape}")
    
    # Ensure the matrix is symmetric
    if not np.allclose(df.values, df.values.T, rtol=1e-5, atol=1e-8):
        print("  Symmetrizing distance matrix...")
        df = (df + df.T) / 2
    
    # Parse sample names
    print("  Parsing sample names...")
    sample_to_modality, sample_pairs = parse_sample_names(df)
    n_paired_samples = len([k for k, v in sample_pairs.items() if len(v) == 2])
    print(f"  Found {n_paired_samples} paired samples")
    
    # Perform permutation test
    print("  Starting permutation test...")
    observed_mean, p_value, null_distribution, observed_distances = permutation_test(
        df, sample_pairs, n_permutations, random_seed
    )
    
    # Return results
    results = {
        'observed_mean': observed_mean,
        'p_value': p_value,
        'null_mean': np.mean(null_distribution),
        'null_std': np.std(null_distribution),
        'effect_size': (observed_mean - np.mean(null_distribution)) / np.std(null_distribution),
        'n_samples': len(observed_distances),
        'null_distribution': null_distribution,
        'observed_distances': observed_distances
    }
    
    print(f"  Effect size: {results['effect_size']:.3f}")
    return results

def validate_multiple_methods(sample_distance_path: str,
                            methods: List[str],
                            output_dir: str,
                            n_permutations: int = 1000,
                            random_seed: int = 42) -> Dict[str, Dict[str, Dict]]:
    """
    Validate cross-modality distances for multiple distance methods and both DR types.
    
    Args:
        sample_distance_path: Path to the Sample_distance directory
        methods: List of distance methods to test (e.g., ['cosine', 'correlation', 'euclidean'])
        output_dir: Directory to save all output files
        n_permutations: Number of permutations for the test
        random_seed: Random seed for reproducibility
    
    Returns:
        Nested dictionary: {method: {dr_type: results}}
    """
    print(f"\n=== Starting validation for {len(methods)} methods ===")
    print(f"Methods: {methods}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # Process each method
    for i, method in enumerate(methods, 1):
        print(f"\n--- Processing method {i}/{len(methods)}: {method} ---")
        method_path = os.path.join(sample_distance_path, method)
        
        if not os.path.exists(method_path):
            print(f"  WARNING: Method directory not found: {method_path}")
            continue
        
        # Create method output directory
        method_output_dir = os.path.join(output_dir, method)
        os.makedirs(method_output_dir, exist_ok=True)
        
        # Run validation for this method
        method_results = validate_directory(
            base_dir=method_path,
            output_dir=method_output_dir,
            n_permutations=n_permutations,
            random_seed=random_seed
        )
        
        all_results[method] = method_results
        print(f"  Completed method: {method}")
    
    # Create comprehensive comparison
    print(f"\n--- Creating comprehensive comparison ---")
    create_comprehensive_comparison(all_results, output_dir)
    
    print(f"\n=== Validation complete! Results saved to: {output_dir} ===")
    return all_results

def validate_directory(base_dir: str,
                      output_dir: str,
                      n_permutations: int = 1000,
                      random_seed: int = 42) -> Dict[str, Dict]:
    """
    Validate cross-modality distances for both expression_DR and proportion_DR matrices
    in a given directory.
    
    Args:
        base_dir: Path to the base directory (e.g., 'Sample_distance/cosine')
        output_dir: Directory to save output files
        n_permutations: Number of permutations for the test
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing results for both distance types
    """
    results_all = {}
    
    # Define the two distance types and their paths
    distance_types = [
        ('expression_DR', 'expression_DR_distance', 'distance_matrix_expression_DR.csv'),
        ('proportion_DR', 'proportion_DR_distance', 'distance_matrix_proportion_DR.csv')
    ]
    
    for dist_name, folder_name, matrix_file in distance_types:
        print(f"\n Processing {dist_name}...")
        
        # Construct full path to the distance matrix
        matrix_path = os.path.join(base_dir, folder_name, matrix_file)
        
        # Check if file exists
        if not os.path.exists(matrix_path):
            print(f"  File not found: {matrix_path}")
            continue
        
        try:
            # Run validation for this distance matrix
            results = validate_cross_modality_distance(
                csv_path=matrix_path,
                n_permutations=n_permutations,
                random_seed=random_seed
            )
            
            # Save p-value plot
            print(f"  Saving p-value plot...")
            plot_path = os.path.join(output_dir, f'pvalue_plot_{dist_name}.png')
            create_pvalue_plot(
                results['observed_mean'],
                results['p_value'],
                results['null_distribution'],
                plot_path
            )
            
            # Store results
            results_all[dist_name] = results
            print(f"  ✓ Completed {dist_name}")
            
        except Exception as e:
            print(f"  ERROR processing {dist_name}: {str(e)}")
            results_all[dist_name] = None
    
    # Save summary for this method
    if any(r is not None for r in results_all.values()):
        print(f" Saving validation summary...")
        comparison_data = []
        for dist_type, results in results_all.items():
            if results is not None:
                comparison_data.append({
                    'Distance Type': dist_type,
                    'P-value': results['p_value'],
                    'Effect Size': results['effect_size'],
                    'Observed Mean': results['observed_mean'],
                    'Null Mean': results['null_mean'],
                    'Significant': results['p_value'] < 0.05
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_path = os.path.join(output_dir, 'validation_summary.csv')
            comparison_df.to_csv(comparison_path, index=False)
    
    return results_all

def create_comprehensive_comparison(all_results: Dict[str, Dict[str, Dict]], 
                                   output_dir: str):
    """
    Create comprehensive comparison across all methods and DR types.
    
    Args:
        all_results: Nested dictionary of all results
        output_dir: Directory to save outputs
    """
    print(" Compiling results across all methods...")
    
    # Collect all results in a flat format
    comparison_data = []
    
    for method, method_results in all_results.items():
        for dr_type, results in method_results.items():
            if results is not None:
                comparison_data.append({
                    'Method': method,
                    'DR Type': dr_type,
                    'P-value': results['p_value'],
                    'Effect Size': results['effect_size'],
                    'Observed Mean': results['observed_mean'],
                    'Null Mean': results['null_mean'],
                    'Significant': results['p_value'] < 0.05
                })
    
    if not comparison_data:
        print(" No valid results found for comparison")
        return
    
    print(f" Found {len(comparison_data)} valid results")
    
    # Create DataFrame and save
    df_comprehensive = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(output_dir, 'all_methods_comparison.csv')
    df_comprehensive.to_csv(comparison_path, index=False)
    print(f" Saved comprehensive comparison: {comparison_path}")
    
    # Create comprehensive visualization
    print(" Creating comprehensive plots...")
    create_comprehensive_plots(df_comprehensive, output_dir)
    
    # Create statistical summary
    summary_stats = {
        'total_tests': len(df_comprehensive),
        'significant_tests': df_comprehensive['Significant'].sum(),
        'best_method': df_comprehensive.loc[df_comprehensive['Effect Size'].idxmax()]['Method'],
        'best_dr_type': df_comprehensive.loc[df_comprehensive['Effect Size'].idxmax()]['DR Type'],
        'best_effect_size': df_comprehensive['Effect Size'].max(),
        'best_p_value': df_comprehensive.loc[df_comprehensive['Effect Size'].idxmax()]['P-value']
    }
    
    # Save summary statistics
    summary_path = os.path.join(output_dir, 'statistical_summary.txt')
    with open(summary_path, 'w') as f:
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f" Best method: {summary_stats['best_method']} ({summary_stats['best_dr_type']})")
    print(f" Effect size: {summary_stats['best_effect_size']:.3f}, p-value: {summary_stats['best_p_value']:.4f}")

def create_comprehensive_plots(df: pd.DataFrame, output_dir: str):
    """
    Create comprehensive visualization comparing all methods and DR types.
    
    Args:
        df: DataFrame with all comparison results
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Heatmap of p-values
    ax1 = axes[0, 0]
    pivot_pval = df.pivot(index='Method', columns='DR Type', values='P-value')
    sns.heatmap(pivot_pval, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                vmin=0, vmax=0.1, cbar_kws={'label': 'P-value'}, ax=ax1)
    ax1.set_title('P-values Across Methods and DR Types')
    
    # 2. Heatmap of effect sizes
    ax2 = axes[0, 1]
    pivot_effect = df.pivot(index='Method', columns='DR Type', values='Effect Size')
    sns.heatmap(pivot_effect, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                cbar_kws={'label': 'Effect Size'}, ax=ax2)
    ax2.set_title('Effect Sizes Across Methods and DR Types')
    
    # 3. Bar plot comparing methods
    ax3 = axes[1, 0]
    methods = df['Method'].unique()
    dr_types = df['DR Type'].unique()
    x = np.arange(len(methods))
    width = 0.35
    
    for i, dr_type in enumerate(dr_types):
        dr_data = df[df['DR Type'] == dr_type]
        effect_sizes = [dr_data[dr_data['Method'] == m]['Effect Size'].values[0] 
                       if len(dr_data[dr_data['Method'] == m]) > 0 else 0 
                       for m in methods]
        bars = ax3.bar(x + i*width, effect_sizes, width, label=dr_type)
        
        # Add significance markers
        for j, (method, es) in enumerate(zip(methods, effect_sizes)):
            method_dr_data = df[(df['Method'] == method) & (df['DR Type'] == dr_type)]
            if len(method_dr_data) > 0 and method_dr_data.iloc[0]['Significant']:
                ax3.text(x[j] + i*width, es + 0.02, '*', ha='center', fontsize=14)
    
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Effect Size')
    ax3.set_title('Effect Sizes by Method and DR Type (* = p<0.05)')
    ax3.set_xticks(x + width/2)
    ax3.set_xticklabels(methods)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 4. Scatter plot of p-value vs effect size
    ax4 = axes[1, 1]
    for dr_type in dr_types:
        dr_data = df[df['DR Type'] == dr_type]
        ax4.scatter(dr_data['Effect Size'], -np.log10(dr_data['P-value']), 
                   label=dr_type, s=100, alpha=0.7)
    
    # Add significance threshold line
    ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', 
               alpha=0.5, label='p=0.05')
    ax4.set_xlabel('Effect Size')
    ax4.set_ylabel('-log10(P-value)')
    ax4.set_title('Volcano Plot: Effect Size vs Significance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add method labels to points
    for _, row in df.iterrows():
        ax4.annotate(row['Method'][:3], 
                    (row['Effect Size'], -np.log10(row['P-value'])),
                    fontsize=8, alpha=0.7)
    
    plt.suptitle('Comprehensive Cross-Modality Validation Results', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, 'comprehensive_comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Comprehensive plots saved: {plot_path}")

# Example usage
if __name__ == "__main__":
    # Example: Validate multiple methods and save to output directory
    results = validate_multiple_methods(
        sample_distance_path='/dcl01/hongkai/data/data/hjiang/result/paired/rna/Sample_distance',
        methods=['cosine', 'correlation'],
        output_dir='/dcl01/hongkai/data/data/hjiang/result/paired/validation_results',
        n_permutations=1000
    )