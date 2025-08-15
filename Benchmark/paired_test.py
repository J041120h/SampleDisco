import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict, List
import warnings
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
    
    # Get all sample names and their modalities
    all_samples = list(df.index)
    atac_samples = [s for s in all_samples if s.endswith('_ATAC')]
    rna_samples = [s for s in all_samples if s.endswith('_RNA')]
    
    # Permutation test
    null_distribution = []
    
    for _ in range(n_permutations):
        # Shuffle the modality labels
        shuffled_atac = np.random.permutation(atac_samples)
        shuffled_rna = np.random.permutation(rna_samples)
        
        # Create new sample pairs based on shuffled labels
        shuffled_pairs = {}
        for i, (atac, rna) in enumerate(zip(shuffled_atac, shuffled_rna)):
            sample_id = f"shuffled_{i}"
            shuffled_pairs[sample_id] = [atac, rna]
        
        # Calculate mean distance for shuffled pairs
        shuffled_mean, _ = calculate_cross_modality_distance(df, shuffled_pairs)
        null_distribution.append(shuffled_mean)
    
    null_distribution = np.array(null_distribution)
    
    # Calculate p-value (two-tailed)
    p_value = np.mean(null_distribution <= observed_mean)
    
    return observed_mean, p_value, null_distribution, observed_distances

def visualize_results(observed_mean: float, 
                     p_value: float, 
                     null_distribution: np.ndarray,
                     observed_distances: List[float],
                     output_prefix: str = "validation"):
    """
    Create visualizations for the permutation test results.
    
    Args:
        observed_mean: Observed mean cross-modality distance
        p_value: P-value from permutation test
        null_distribution: Array of permuted mean distances
        observed_distances: List of observed cross-modality distances
        output_prefix: Prefix for output files
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Histogram of null distribution with observed value
    ax1 = axes[0, 0]
    ax1.hist(null_distribution, bins=50, alpha=0.7, color='gray', 
             edgecolor='black', label='Null distribution')
    ax1.axvline(observed_mean, color='red', linestyle='--', linewidth=2,
                label=f'Observed mean: {observed_mean:.4f}')
    ax1.set_xlabel('Average Cross-Modality Distance')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Permutation Test (p-value = {p_value:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q plot to check distribution
    ax2 = axes[0, 1]
    stats.probplot(null_distribution, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot of Null Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot comparing observed vs null
    ax3 = axes[1, 0]
    box_data = [observed_distances, null_distribution]
    box_positions = [1, 2]
    bp = ax3.boxplot(box_data, positions=box_positions, widths=0.6,
                      patch_artist=True, labels=['Observed\n(Same Sample)', 'Null\n(Shuffled)'])
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgray')
    ax3.set_ylabel('Distance')
    ax3.set_title('Distribution Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistical annotation
    y_max = max(max(observed_distances), max(null_distribution))
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'
    ax3.text(1.5, y_max * 1.05, sig_text, ha='center', fontsize=14, fontweight='bold')
    
    # 4. Cumulative distribution comparison
    ax4 = axes[1, 1]
    
    # Calculate empirical CDFs
    observed_sorted = np.sort(observed_distances)
    observed_cdf = np.arange(1, len(observed_sorted) + 1) / len(observed_sorted)
    
    null_sorted = np.sort(null_distribution)
    null_cdf = np.arange(1, len(null_sorted) + 1) / len(null_sorted)
    
    ax4.plot(observed_sorted, observed_cdf, 'r-', linewidth=2, label='Observed')
    ax4.plot(null_sorted, null_cdf, 'gray', linewidth=2, label='Null')
    ax4.set_xlabel('Distance')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Empirical CDF Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Cross-Modality Distance Statistical Validation', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_validation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("=" * 60)
    print("STATISTICAL VALIDATION RESULTS")
    print("=" * 60)
    print(f"Number of sample pairs: {len(observed_distances)}")
    print(f"Observed mean distance: {observed_mean:.6f}")
    print(f"Null distribution mean: {np.mean(null_distribution):.6f}")
    print(f"Null distribution std: {np.std(null_distribution):.6f}")
    print(f"Effect size (Cohen's d): {(observed_mean - np.mean(null_distribution)) / np.std(null_distribution):.4f}")
    print(f"P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("\n✓ SIGNIFICANT: Same samples from different modalities are")
        print("  significantly closer than expected by chance (p < 0.05)")
    else:
        print("\n✗ NOT SIGNIFICANT: No evidence that same samples from")
        print("  different modalities are closer than expected by chance")
    print("=" * 60)

def validate_cross_modality_distance(csv_path: str, 
                                    n_permutations: int = 1000,
                                    random_seed: int = 42,
                                    output_prefix: str = "validation") -> Dict:
    """
    Main function to validate cross-modality distances.
    
    Args:
        csv_path: Path to the distance matrix CSV file
        n_permutations: Number of permutations for the test
        random_seed: Random seed for reproducibility
        output_prefix: Prefix for output files
    
    Returns:
        Dictionary containing test results
    """
    # Load the distance matrix
    df = pd.read_csv(csv_path, index_col=0)
    
    # Ensure the matrix is symmetric
    if not np.allclose(df.values, df.values.T, rtol=1e-5, atol=1e-8):
        print("Warning: Distance matrix is not perfectly symmetric. Using average of upper and lower triangular.")
        df = (df + df.T) / 2
    
    # Parse sample names
    sample_to_modality, sample_pairs = parse_sample_names(df)
    
    print(f"Found {len(sample_pairs)} unique samples with cross-modality measurements")
    print(f"Total samples in matrix: {len(df)}")
    
    # Perform permutation test
    observed_mean, p_value, null_distribution, observed_distances = permutation_test(
        df, sample_pairs, n_permutations, random_seed
    )
    
    # Create visualizations
    visualize_results(observed_mean, p_value, null_distribution, 
                     observed_distances, output_prefix)
    
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
    
    return results

# Example usage
if __name__ == "__main__":
    # Example: Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 20
    
    # Create sample names
    sample_names = []
    for i in range(n_samples):
        sample_names.append(f"sample_{i}_ATAC")
        sample_names.append(f"sample_{i}_RNA")
    
    # Generate synthetic distance matrix
    # Make same-sample cross-modality distances smaller
    n_total = len(sample_names)
    dist_matrix = np.random.uniform(0.5, 1.0, (n_total, n_total))
    
    # Make it symmetric
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(dist_matrix, 0)
    
    # Make same-sample cross-modality distances smaller
    for i in range(n_samples):
        atac_idx = i * 2
        rna_idx = i * 2 + 1
        # Set smaller distance for same sample
        small_dist = np.random.uniform(0.1, 0.3)
        dist_matrix[atac_idx, rna_idx] = small_dist
        dist_matrix[rna_idx, atac_idx] = small_dist
    
    # Create DataFrame
    df_synthetic = pd.DataFrame(dist_matrix, index=sample_names, columns=sample_names)
    
    # Save synthetic data
    df_synthetic.to_csv('synthetic_distance_matrix.csv')
    
    # Run validation
    results = validate_cross_modality_distance('synthetic_distance_matrix.csv', 
                                              n_permutations=1000)