import os
import warnings
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import pairwise_distances

warnings.filterwarnings('ignore')

# =========================
# Existing code (unchanged)
# =========================

def parse_sample_names(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Parse sample names to extract sample IDs and modalities.
    """
    print("    Parsing sample names and identifying modalities...")
    sample_to_modality = {}
    sample_pairs = {}
    for name in df.index:
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            sample_id, modality = parts
            sample_to_modality[name] = modality
            if sample_id not in sample_pairs:
                sample_pairs[sample_id] = []
            sample_pairs[sample_id].append(name)
    print(f"    Found {len(sample_to_modality)} total samples across modalities")
    modalities = set(sample_to_modality.values())
    print(f"    Identified modalities: {list(modalities)}")
    return sample_to_modality, sample_pairs

def calculate_cross_modality_distance(df: pd.DataFrame, 
                                     sample_pairs: Dict) -> Tuple[float, List[float]]:
    distances = []
    for sample_id, names in sample_pairs.items():
        if len(names) == 2:
            dist = df.loc[names[0], names[1]]
            distances.append(dist)
    return np.mean(distances) if distances else 0, distances

def permutation_test(df: pd.DataFrame, 
                    sample_pairs: Dict,
                    n_permutations: int = 1000,
                    random_seed: int = 42) -> Tuple[float, float, np.ndarray, List[float]]:
    print("    Setting up permutation test...")
    np.random.seed(random_seed)
    print("    Calculating observed cross-modality distances...")
    observed_mean, observed_distances = calculate_cross_modality_distance(df, sample_pairs)
    print(f"    Observed mean cross-modality distance: {observed_mean:.4f}")
    print("    Preparing sample lists for permutation...")
    all_samples = list(df.index)
    atac_samples = [s for s in all_samples if s.endswith('_ATAC')]
    rna_samples = [s for s in all_samples if s.endswith('_RNA')]
    print(f"    ATAC samples: {len(atac_samples)}, RNA samples: {len(rna_samples)}")
    print(f"    Running {n_permutations} permutations...")
    null_distribution = []
    progress_points = [int(n_permutations * p) for p in [0.25, 0.5, 0.75]]
    for i in range(n_permutations):
        if i in progress_points:
            print(f"      {int(100 * i / n_permutations)}% complete ({i}/{n_permutations})")
        shuffled_atac = np.random.permutation(atac_samples)
        shuffled_rna = np.random.permutation(rna_samples)
        shuffled_pairs = {}
        for j, (atac, rna) in enumerate(zip(shuffled_atac, shuffled_rna)):
            sample_id = f"shuffled_{j}"
            shuffled_pairs[sample_id] = [atac, rna]
        shuffled_mean, _ = calculate_cross_modality_distance(df, shuffled_pairs)
        null_distribution.append(shuffled_mean)
    print(f"      100% complete ({n_permutations}/{n_permutations})")
    null_distribution = np.array(null_distribution)
    print("    Calculating p-value from null distribution...")
    p_value = np.mean(null_distribution <= observed_mean)
    print(f"    P-value: {p_value:.4f}")
    return observed_mean, p_value, null_distribution, observed_distances

def create_pvalue_plot(observed_mean: float, 
                      p_value: float, 
                      null_distribution: np.ndarray,
                      output_path: str):
    print("    Generating p-value visualization...")
    plt.figure(figsize=(8, 6))
    plt.hist(null_distribution, bins=50, alpha=0.7, color='gray', 
             edgecolor='black', label='Null distribution')
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
    print(f"    Plot saved to: {output_path}")

def validate_cross_modality_distance(csv_path: str, 
                                    n_permutations: int = 1000,
                                    random_seed: int = 42) -> Dict:
    print(f"  Loading distance matrix from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    print(f"  Matrix shape: {df.shape}")
    print("  Checking matrix symmetry...")
    if not np.allclose(df.values, df.values.T, rtol=1e-5, atol=1e-8):
        print("  Symmetrizing distance matrix...")
        df = (df + df.T) / 2
    else:
        print("  Matrix is already symmetric")
    print("  Parsing sample names...")
    sample_to_modality, sample_pairs = parse_sample_names(df)
    n_paired_samples = len([k for k, v in sample_pairs.items() if len(v) == 2])
    print(f"  Found {n_paired_samples} paired samples")
    print("  Starting permutation test...")
    observed_mean, p_value, null_distribution, observed_distances = permutation_test(
        df, sample_pairs, n_permutations, random_seed
    )
    print("  Calculating final statistics...")
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
    print("  ✓ Validation complete for this matrix")
    return results

def validate_multiple_methods(sample_distance_path: str,
                            methods: List[str],
                            output_dir: str,
                            n_permutations: int = 1000,
                            random_seed: int = 42) -> Dict[str, Dict[str, Dict]]:
    print(f"\n=== Starting validation for {len(methods)} methods ===")
    print(f"Methods: {methods}")
    print(f"Sample distance path: {sample_distance_path}")
    print(f"Output directory: {output_dir}")
    print(f"Permutations per test: {n_permutations}")
    print(f"Random seed: {random_seed}")
    print(f"\nPreparing output directory...")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ready: {output_dir}")
    all_results = {}
    for i, method in enumerate(methods, 1):
        print(f"\n--- Processing method {i}/{len(methods)}: {method} ---")
        method_path = os.path.join(sample_distance_path, method)
        print(f"Looking for method directory: {method_path}")
        if not os.path.exists(method_path):
            print(f"  WARNING: Method directory not found: {method_path}")
            print(f"  Skipping method: {method}")
            continue
        print(f"  ✓ Method directory found")
        method_output_dir = os.path.join(output_dir, method)
        print(f"  Creating method output directory: {method_output_dir}")
        os.makedirs(method_output_dir, exist_ok=True)
        print(f"  Starting validation for method: {method}")
        method_results = validate_directory(
            base_dir=method_path,
            output_dir=method_output_dir,
            n_permutations=n_permutations,
            random_seed=random_seed
        )
        all_results[method] = method_results
        print(f"  ✓ Completed method: {method}")
        valid_results = {k: v for k, v in method_results.items() if v is not None}
        print(f"  Summary: {len(valid_results)}/{len(method_results)} distance types processed successfully")
    print(f"\n--- Method processing complete ---")
    total_valid = sum(len([v for v in method_results.values() if v is not None]) 
                     for method_results in all_results.values())
    total_possible = len(methods) * 2
    print(f"Overall: {total_valid}/{total_possible} validations completed successfully")
    print(f"\n--- Creating comprehensive comparison ---")
    create_comprehensive_comparison(all_results, output_dir)
    print(f"\n=== Validation complete! Results saved to: {output_dir} ===")
    return all_results

def validate_directory(base_dir: str,
                      output_dir: str,
                      n_permutations: int = 1000,
                      random_seed: int = 42) -> Dict[str, Dict]:
    print(f"\n  Validating directory: {base_dir}")
    print(f"  Output will be saved to: {output_dir}")
    results_all = {}
    distance_types = [
        ('expression_DR', 'expression_DR_distance', 'distance_matrix_expression_DR.csv'),
        ('proportion_DR', 'proportion_DR_distance', 'distance_matrix_proportion_DR.csv')
    ]
    print(f"  Will process {len(distance_types)} distance types:")
    for dist_name, folder_name, matrix_file in distance_types:
        print(f"    - {dist_name}")
    for i, (dist_name, folder_name, matrix_file) in enumerate(distance_types, 1):
        print(f"\n Processing {dist_name} ({i}/{len(distance_types)})...")
        matrix_path = os.path.join(base_dir, folder_name, matrix_file)
        print(f"  Looking for matrix file: {matrix_path}")
        if not os.path.exists(matrix_path):
            print(f"  ❌ File not found: {matrix_path}")
            print(f"  Skipping {dist_name}")
            continue
        print(f"  ✓ Matrix file found")
        try:
            print(f"  Starting validation for {dist_name}...")
            results = validate_cross_modality_distance(
                csv_path=matrix_path,
                n_permutations=n_permutations,
                random_seed=random_seed
            )
            print(f"  Saving p-value plot...")
            plot_path = os.path.join(output_dir, f'pvalue_plot_{dist_name}.png')
            create_pvalue_plot(
                results['observed_mean'],
                results['p_value'],
                results['null_distribution'],
                plot_path
            )
            results_all[dist_name] = results
            print(f"  ✓ Completed {dist_name}")
            print(f"    - P-value: {results['p_value']:.4f}")
            print(f"    - Effect size: {results['effect_size']:.3f}")
            print(f"    - Significant: {'Yes' if results['p_value'] < 0.05 else 'No'}")
        except Exception as e:
            print(f"  ❌ ERROR processing {dist_name}: {str(e)}")
            print(f"  Skipping {dist_name}")
            results_all[dist_name] = None
    valid_results = [r for r in results_all.values() if r is not None]
    print(f"\n  Directory summary: {len(valid_results)}/{len(distance_types)} distance types processed")
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
            print(f"  Summary saved to: {comparison_path}")
    return results_all

def create_comprehensive_comparison(all_results: Dict[str, Dict[str, Dict]], 
                                   output_dir: str):
    print(" Compiling results across all methods...")
    comparison_data = []
    print(" Processing results from each method...")
    for method, method_results in all_results.items():
        print(f"   Processing method: {method}")
        for dr_type, results in method_results.items():
            if results is not None:
                print(f"     Adding {dr_type} results")
                comparison_data.append({
                    'Method': method,
                    'DR Type': dr_type,
                    'P-value': results['p_value'],
                    'Effect Size': results['effect_size'],
                    'Observed Mean': results['observed_mean'],
                    'Null Mean': results['null_mean'],
                    'Significant': results['p_value'] < 0.05
                })
            else:
                print(f"     Skipping {dr_type} (no valid results)")
    if not comparison_data:
        print(" ❌ No valid results found for comparison")
        return
    print(f" ✓ Found {len(comparison_data)} valid results for comparison")
    df_comprehensive = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(output_dir, 'all_methods_comparison.csv')
    df_comprehensive.to_csv(comparison_path, index=False)
    print(f" ✓ Comprehensive comparison saved: {comparison_path}")
    print(" Creating comprehensive plots...")
    create_comprehensive_plots(df_comprehensive, output_dir)
    print(" Generating statistical summary...")
    best_idx = df_comprehensive['Effect Size'].idxmax()
    summary_stats = {
        'total_tests': len(df_comprehensive),
        'significant_tests': df_comprehensive['Significant'].sum(),
        'best_method': df_comprehensive.loc[best_idx]['Method'],
        'best_dr_type': df_comprehensive.loc[best_idx]['DR Type'],
        'best_effect_size': df_comprehensive['Effect Size'].max(),
        'best_p_value': df_comprehensive.loc[best_idx]['P-value']
    }
    print(" Saving statistical summary...")
    summary_path = os.path.join(output_dir, 'statistical_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=== CROSS-MODALITY VALIDATION SUMMARY ===\n\n")
        for key, value in summary_stats.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
    print(f" ✓ Statistical summary saved: {summary_path}")
    print(f"\n ✓ Best performing combination:")
    print(f"   Method: {summary_stats['best_method']}")
    print(f"   DR Type: {summary_stats['best_dr_type']}")
    print(f"   Effect size: {summary_stats['best_effect_size']:.3f}")
    print(f"   P-value: {summary_stats['best_p_value']:.4f}")
    print(f" ✓ Significant results: {summary_stats['significant_tests']}/{summary_stats['total_tests']}")

def create_comprehensive_plots(df: pd.DataFrame, output_dir: str):
    print("   Setting up comprehensive visualization...")
    print("   Creating 4-panel comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax1 = axes[0, 0]
    pivot_pval = df.pivot(index='Method', columns='DR Type', values='P-value')
    sns.heatmap(pivot_pval, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                vmin=0, vmax=0.1, cbar_kws={'label': 'P-value'}, ax=ax1)
    ax1.set_title('P-values Across Methods and DR Types')
    ax2 = axes[0, 1]
    pivot_effect = df.pivot(index='Method', columns='DR Type', values='Effect Size')
    sns.heatmap(pivot_effect, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                cbar_kws={'label': 'Effect Size'}, ax=ax2)
    ax2.set_title('Effect Sizes Across Methods and DR Types')
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
    ax4 = axes[1, 1]
    for dr_type in dr_types:
        dr_data = df[df['DR Type'] == dr_type]
        ax4.scatter(dr_data['Effect Size'], -np.log10(dr_data['P-value']), 
                   label=dr_type, s=100, alpha=0.7)
    ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', 
               alpha=0.5, label='p=0.05')
    ax4.set_xlabel('Effect Size')
    ax4.set_ylabel('-log10(P-value)')
    ax4.set_title('Volcano Plot: Effect Size vs Significance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    for _, row in df.iterrows():
        ax4.annotate(row['Method'][:3], 
                    (row['Effect Size'], -np.log10(row['P-value'])),
                    fontsize=8, alpha=0.7)
    plt.suptitle('Comprehensive Cross-Modality Validation Results', fontsize=16)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comprehensive_comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Comprehensive plots saved: {plot_path}")

# =========================================================
# NEW — AnnData pseudobulk group-separation evaluation path
# =========================================================

def _select_embedding_from_adata(adata, obsm_keys_priority: List[str]) -> np.ndarray:
    """
    Try several obsm keys in order; fallback to .X if none found.
    """
    for k in obsm_keys_priority:
        if k in getattr(adata, "obsm", {}):
            X = adata.obsm[k]
            print(f"[INFO] Using adata.obsm['{k}'] with shape {X.shape} for distances.")
            return np.asarray(X)
    print("[WARN] None of the preferred obsm keys found; using adata.X.")
    return np.asarray(adata.X)

def _compute_distance_matrix(X: np.ndarray, metric: str = "cosine") -> pd.DataFrame:
    """
    Compute pairwise distances with sklearn pairwise_distances.
    metric ∈ {'cosine','euclidean','correlation','manhattan', ...}
    """
    D = pairwise_distances(X, metric=metric)
    return pd.DataFrame(D)

def _index_and_labels_from_adata(adata, index_col: Optional[str] = None) -> List[str]:
    """
    Provide row/col names for the distance matrix (defaults to adata.obs_names).
    """
    if index_col is not None and index_col in adata.obs.columns:
        labels = adata.obs[index_col].astype(str).tolist()
        # Ensure uniqueness to avoid duplicated index issues
        uniq = []
        counts = {}
        for s in labels:
            counts[s] = counts.get(s, 0) + 1
            uniq.append(s if counts[s] == 1 else f"{s}#{counts[s]}")
        return uniq
    else:
        return adata.obs_names.astype(str).tolist()

def compute_group_distance_stats(dist_df: pd.DataFrame,
                                 group_labels: pd.Series,
                                 stat: str = "diff") -> Dict[str, float]:
    """
    Compute within-group and between-group average distances and an effect size.
    stat: 'diff' (between - within) or 'ratio' (between / within)
    """
    assert dist_df.shape[0] == dist_df.shape[1] == len(group_labels), \
        "Distance matrix size must match number of labels."
    n = len(group_labels)
    tri_i, tri_j = np.triu_indices(n, k=1)
    same_mask = (group_labels.values[tri_i] == group_labels.values[tri_j])
    dists = dist_df.values[tri_i, tri_j]
    within = dists[same_mask]
    between = dists[~same_mask]
    within_mean = float(np.mean(within)) if within.size else np.nan
    between_mean = float(np.mean(between)) if between.size else np.nan
    if stat == "diff":
        effect = float(between_mean - within_mean)
    elif stat == "ratio":
        effect = float(between_mean / within_mean) if within_mean and not np.isnan(within_mean) else np.nan
    else:
        raise ValueError("stat must be 'diff' or 'ratio'")
    return {
        "within_mean": within_mean,
        "between_mean": between_mean,
        "effect": effect,
        "within_n_pairs": int(within.size),
        "between_n_pairs": int(between.size),
    }

def permutation_test_group_labels(dist_df: pd.DataFrame,
                                  group_labels: pd.Series,
                                  n_permutations: int = 1000,
                                  random_seed: int = 42,
                                  stat: str = "diff") -> Dict[str, object]:
    """
    Permute group labels to get null for (between - within) or (between/within).
    """
    np.random.seed(random_seed)
    observed = compute_group_distance_stats(dist_df, group_labels, stat=stat)
    obs_effect = observed["effect"]
    null = np.empty(n_permutations, dtype=float)
    print(f"[INFO] Running {n_permutations} permutations for group separation ({stat}).")
    checkpoints = [int(n_permutations * p) for p in [0.25, 0.5, 0.75]]
    labels = group_labels.values.copy()
    for i in range(n_permutations):
        if i in checkpoints:
            print(f"  ... {int(100*i/n_permutations)}%")
        np.random.shuffle(labels)
        perm_stats = compute_group_distance_stats(dist_df, pd.Series(labels), stat=stat)
        null[i] = perm_stats["effect"]
    # Right-tail if effect = between - within or ratio (>0 = better separation)
    p_value = float(np.mean(null >= obs_effect))
    return {
        "observed": observed,
        "null": null,
        "p_value": p_value
    }

def save_group_eval_outputs(results: Dict[str, object],
                            output_dir: str,
                            prefix: str = "group_eval"):
    """
    Save null distribution plot and a small CSV with stats.
    """
    os.makedirs(output_dir, exist_ok=True)
    # CSV of summary
    summary = {
        "within_mean": results["observed"]["within_mean"],
        "between_mean": results["observed"]["between_mean"],
        "effect": results["observed"]["effect"],
        "within_n_pairs": results["observed"]["within_n_pairs"],
        "between_n_pairs": results["observed"]["between_n_pairs"],
        "p_value": results["p_value"],
        "null_mean": float(np.mean(results["null"])),
        "null_std": float(np.std(results["null"]))
    }
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, f"{prefix}_summary.csv"), index=False)
    # Plot
    plt.figure(figsize=(8,6))
    plt.hist(results["null"], bins=50, alpha=0.7, color="gray", edgecolor="black", label="Null")
    plt.axvline(results["observed"]["effect"], color="red", linestyle="--", linewidth=2,
                label=f'Observed effect = {results["observed"]["effect"]:.4f}')
    plt.xlabel("Effect statistic")
    plt.ylabel("Frequency")
    plt.title(f"Group Separation Permutation (p = {results['p_value']:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_null.png"), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_group_separation_from_anndata(
        adata,
        group_col: str,
        metric: str = "cosine",
        obsm_keys_priority: Optional[List[str]] = None,
        index_col: Optional[str] = None,
        n_permutations: int = 1000,
        random_seed: int = 42,
        stat: str = "diff",
        output_dir: Optional[str] = None
    ) -> Dict[str, object]:
    """
    NEW: Accept a pseudobulk AnnData where adata.obs holds sample metadata.
    Compute a pairwise sample distance matrix from an embedding, then evaluate
    within-group vs between-group distance for a given obs column.

    Parameters
    ----------
    adata : AnnData
        Pseudobulk where each row is a sample; metadata in adata.obs.
    group_col : str
        Column in adata.obs (e.g., 'tissue') to define groups.
    metric : str
        Distance metric for pairwise_distances ('cosine','euclidean','correlation',...).
    obsm_keys_priority : list[str] | None
        Try these obsm keys in order; fallback to adata.X if none present.
        Default: ['X_embedding','PCA','X_pca'].
    index_col : str | None
        Optional obs column to use as distance-matrix row/col names (ensured unique).
        Default None → use adata.obs_names.
    n_permutations : int
    random_seed : int
    stat : str
        'diff' => between - within (larger is better separation)
        'ratio' => between / within (larger is better separation)
    output_dir : str | None
        If provided, saves a CSV/PNG with summary and null distribution.

    Returns
    -------
    results : dict
        {
          'distance_matrix': pd.DataFrame,
          'group_labels': pd.Series,
          'observed': {...},
          'null': np.ndarray,
          'p_value': float
        }
    """
    assert group_col in adata.obs.columns, f"[ERROR] group_col '{group_col}' not found in adata.obs"
    if obsm_keys_priority is None:
        obsm_keys_priority = ['X_embedding', 'PCA', 'X_pca']
    X = _select_embedding_from_adata(adata, obsm_keys_priority)
    names = _index_and_labels_from_adata(adata, index_col=index_col)
    dist_df = _compute_distance_matrix(X, metric=metric)
    dist_df.index = names
    dist_df.columns = names
    labels = adata.obs[group_col].astype(str)
    # compute and permute
    perm_results = permutation_test_group_labels(dist_df, labels,
                                                 n_permutations=n_permutations,
                                                 random_seed=random_seed,
                                                 stat=stat)
    out = {
        "distance_matrix": dist_df,
        "group_labels": labels,
        **perm_results
    }
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        # save distance matrix & labels
        dist_df.to_csv(os.path.join(output_dir, f"group_eval_distance_{metric}.csv"))
        labels.to_csv(os.path.join(output_dir, "group_labels.csv"), header=["group"])
        save_group_eval_outputs(perm_results, output_dir, prefix=f"group_eval_{metric}_{stat}")
    # quick print
    obs = perm_results["observed"]
    print(f"[RESULT] within={obs['within_mean']:.4f}, between={obs['between_mean']:.4f}, "
          f"effect({stat})={obs['effect']:.4f}, p={perm_results['p_value']:.4g}")
    return out

# =========================
# Example usage (unchanged)
# =========================
if __name__ == "__main__":
    print("=== CROSS-MODALITY VALIDATION SCRIPT ===")
    print("This script validates cross-modality distances using permutation testing.")
    print()

    # Example: Validate multiple methods and save to output directory (existing path)
    print("Starting example validation...")
    results = validate_multiple_methods(
        sample_distance_path='/dcs07/hongkai/data/harry/result/all/rna/Sample_distance',
        methods=['cosine', 'correlation'],
        output_dir='/dcs07/hongkai/data/harry/result/all/rna',
        n_permutations=1000
    )

    # NEW — Example: run group separation on an AnnData pseudobulk (uncomment & set your paths)
    import anndata as ad
    # adata = ad.read_h5ad('/dcs07/hongkai/data/harry/result/all/multiomics/pseudobulk/pseudobulk_sample.h5ad')
    # _ = evaluate_group_separation_from_anndata(
    #         adata=adata,
    #         group_col='tissue',              # <-- any obs column, e.g. 'tissue'
    #         metric='cosine',
    #         obsm_keys_priority=['X_embedding','PCA','X_pca'],
    #         index_col='sample',              # optional obs identifier for nice row names
    #         n_permutations=1000,
    #         random_seed=42,
    #         stat='diff',                     # 'diff' or 'ratio'
    #         output_dir='/tmp/group_eval'     # saves CSVs and a PNG
    #     )

    print("\n=== VALIDATION SCRIPT COMPLETE ===")
    print("Check the output directory for detailed results and visualizations.")
