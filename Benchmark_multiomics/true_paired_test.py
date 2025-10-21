#!/usr/bin/env python3
"""
Test how many true pairs are included in k-nearest neighbors
for varying values of k from 1 to 10.
"""

import os
import gc
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import GPU libraries (optional)
try:
    import cupy as cp
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU libraries not available, using CPU implementation")


def normalize_barcode(barcode: str) -> str:
    """
    Normalize barcode by removing modality suffixes and extracting core identifier.
    Handles both _RNA/_ATAC suffixes and finds ENCSR patterns.
    """
    if not isinstance(barcode, str):
        barcode = str(barcode)
    
    # Remove modality suffixes
    barcode = barcode.replace('_RNA', '').replace('_ATAC', '')
    
    # If barcode contains ENCSR, extract from there
    idx = barcode.find("ENCSR")
    if idx != -1:
        barcode = barcode[idx:]
    
    return barcode


def find_true_pairs(rna_obs, atac_obs, verbose=True):
    """
    Find true RNA-ATAC pairs based on matching barcodes.
    Returns dictionaries mapping from ATAC index to RNA index for paired cells.
    """
    # Get barcodes, preferring original_barcode if available
    if 'original_barcode' in rna_obs.columns and 'original_barcode' in atac_obs.columns:
        rna_barcodes = rna_obs['original_barcode'].astype(str)
        atac_barcodes = atac_obs['original_barcode'].astype(str)
        if verbose:
            print("Using 'original_barcode' column for pairing")
    else:
        rna_barcodes = pd.Series(rna_obs.index.astype(str), index=rna_obs.index)
        atac_barcodes = pd.Series(atac_obs.index.astype(str), index=atac_obs.index)
        if verbose:
            print("Using index for pairing")
    
    # Normalize barcodes
    rna_barcodes_norm = rna_barcodes.apply(normalize_barcode)
    atac_barcodes_norm = atac_barcodes.apply(normalize_barcode)
    
    # Find pairs
    rna_bc_to_idx = {bc: idx for idx, bc in zip(rna_obs.index, rna_barcodes_norm)}
    atac_bc_to_idx = {bc: idx for idx, bc in zip(atac_obs.index, atac_barcodes_norm)}
    
    # Build pairing dictionaries
    atac_to_rna_pairs = {}  # Maps ATAC cell index -> RNA cell index
    paired_barcodes = set(rna_barcodes_norm) & set(atac_barcodes_norm)
    
    for bc in paired_barcodes:
        if bc in rna_bc_to_idx and bc in atac_bc_to_idx:
            rna_idx = rna_bc_to_idx[bc]
            atac_idx = atac_bc_to_idx[bc]
            atac_to_rna_pairs[atac_idx] = rna_idx
    
    if verbose:
        print(f"Found {len(atac_to_rna_pairs)} paired cells")
        print(f"Total RNA cells: {len(rna_obs)}")
        print(f"Total ATAC cells: {len(atac_obs)}")
    
    return atac_to_rna_pairs


def test_knn_pair_inclusion_gpu(
    rna_embedding,
    atac_embedding,
    atac_to_rna_pairs,
    rna_indices,
    k_values=range(1, 11),
    metric='cosine',
    batch_size=1000
):
    """
    GPU-accelerated version of k-NN pair inclusion test.
    """
    import cupy as cp
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    
    results = {k: {'hits': 0, 'total': 0, 'hit_positions': []} for k in k_values}
    max_k = max(k_values)
    
    # Convert embeddings to GPU
    rna_embedding_gpu = cp.asarray(rna_embedding.astype(np.float32))
    atac_embedding_gpu = cp.asarray(atac_embedding.astype(np.float32))
    
    # Build k-NN model
    print(f"Building k-NN model on GPU with max_k={max_k}...")
    nn = cuNearestNeighbors(
        n_neighbors=min(max_k, len(rna_embedding)),
        metric=metric,
        algorithm='brute' if len(rna_embedding) < 50000 else 'auto'
    )
    nn.fit(rna_embedding_gpu)
    
    # Process in batches
    n_atac = len(atac_embedding)
    
    for batch_start in tqdm(range(0, n_atac, batch_size), desc="Processing ATAC cells"):
        batch_end = min(batch_start + batch_size, n_atac)
        batch_indices = list(range(batch_start, batch_end))
        
        # Find k-NN for batch
        batch_embedding = atac_embedding_gpu[batch_indices]
        _, indices_gpu = nn.kneighbors(batch_embedding)
        indices = cp.asnumpy(indices_gpu)  # Shape: (batch_size, max_k)
        
        # Check each ATAC cell in batch
        for i, atac_local_idx in enumerate(batch_indices):
            atac_idx = atac_embedding.index[atac_local_idx]
            
            if atac_idx not in atac_to_rna_pairs:
                continue
                
            true_rna_idx = atac_to_rna_pairs[atac_idx]
            
            # Find position of true RNA in RNA indices
            try:
                true_rna_local = list(rna_indices).index(true_rna_idx)
            except ValueError:
                continue
            
            # Get k-NN indices for this ATAC cell
            knn_indices = indices[i]  # Indices in RNA embedding space
            
            # Check for each k value
            for k in k_values:
                results[k]['total'] += 1
                
                # Check if true pair is in top-k
                if true_rna_local in knn_indices[:k]:
                    results[k]['hits'] += 1
                    # Find exact position (1-indexed)
                    position = np.where(knn_indices[:k] == true_rna_local)[0]
                    if len(position) > 0:
                        results[k]['hit_positions'].append(position[0] + 1)
    
    # Clean up GPU memory
    del rna_embedding_gpu, atac_embedding_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return results


def test_knn_pair_inclusion_cpu(
    rna_embedding,
    atac_embedding,
    atac_to_rna_pairs,
    rna_indices,
    k_values=range(1, 11),
    metric='cosine',
    batch_size=100
):
    """
    CPU version of k-NN pair inclusion test using sklearn.
    """
    from sklearn.neighbors import NearestNeighbors
    
    results = {k: {'hits': 0, 'total': 0, 'hit_positions': []} for k in k_values}
    max_k = max(k_values)
    
    # Build k-NN model
    print(f"Building k-NN model on CPU with max_k={max_k}...")
    nn = NearestNeighbors(
        n_neighbors=min(max_k, len(rna_embedding)),
        metric=metric,
        algorithm='auto'
    )
    nn.fit(rna_embedding)
    
    # Process all ATAC cells
    n_atac = len(atac_embedding)
    
    for batch_start in tqdm(range(0, n_atac, batch_size), desc="Processing ATAC cells"):
        batch_end = min(batch_start + batch_size, n_atac)
        batch_indices = list(range(batch_start, batch_end))
        
        # Find k-NN for batch
        batch_embedding = atac_embedding[batch_indices]
        _, indices = nn.kneighbors(batch_embedding)  # Shape: (batch_size, max_k)
        
        # Check each ATAC cell in batch
        for i, atac_local_idx in enumerate(batch_indices):
            atac_idx = atac_embedding.index[atac_local_idx]
            
            if atac_idx not in atac_to_rna_pairs:
                continue
                
            true_rna_idx = atac_to_rna_pairs[atac_idx]
            
            # Find position of true RNA in RNA indices
            try:
                true_rna_local = list(rna_indices).index(true_rna_idx)
            except ValueError:
                continue
            
            # Get k-NN indices for this ATAC cell
            knn_indices = indices[i]  # Indices in RNA embedding space
            
            # Check for each k value
            for k in k_values:
                results[k]['total'] += 1
                
                # Check if true pair is in top-k
                if true_rna_local in knn_indices[:k]:
                    results[k]['hits'] += 1
                    # Find exact position (1-indexed)
                    position = np.where(knn_indices[:k] == true_rna_local)[0]
                    if len(position) > 0:
                        results[k]['hit_positions'].append(position[0] + 1)
    
    return results


def plot_results(results, output_dir):
    """
    Create comprehensive visualizations of the k-NN pair inclusion results.
    """
    # Prepare data for plotting
    k_values = sorted(results.keys())
    hit_rates = []
    totals = []
    
    for k in k_values:
        total = results[k]['total']
        hits = results[k]['hits']
        hit_rate = hits / total * 100 if total > 0 else 0
        hit_rates.append(hit_rate)
        totals.append(total)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Main hit rate curve
    ax1 = axes[0, 0]
    ax1.plot(k_values, hit_rates, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('k (number of nearest neighbors)', fontsize=12)
    ax1.set_ylabel('True Pair Inclusion Rate (%)', fontsize=12)
    ax1.set_title('True Pair Inclusion in k-Nearest Neighbors', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Add value labels
    for k, rate in zip(k_values, hit_rates):
        ax1.text(k, rate + 1, f'{rate:.1f}%', ha='center', fontsize=9)
    
    # 2. Position distribution for k=1
    ax2 = axes[0, 1]
    if 1 in results and len(results[1]['hit_positions']) > 0:
        positions_k1 = results[1]['hit_positions']
        ax2.hist(positions_k1, bins=1, edgecolor='black', alpha=0.7, color='green')
        ax2.set_xlabel('Position of True Pair (1-indexed)', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title(f'k=1: True Pair Hit Rate = {hit_rates[0]:.1f}%', fontsize=14)
        ax2.set_xticks([1])
    else:
        ax2.text(0.5, 0.5, 'No data for k=1', ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Cumulative hit rate
    ax3 = axes[1, 0]
    cumulative_rates = []
    for k in k_values:
        rate = results[k]['hits'] / results[k]['total'] * 100 if results[k]['total'] > 0 else 0
        cumulative_rates.append(rate)
    
    ax3.bar(k_values, hit_rates, color='lightblue', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('k (number of nearest neighbors)', fontsize=12)
    ax3.set_ylabel('True Pair Inclusion Rate (%)', fontsize=12)
    ax3.set_title('True Pair Inclusion Rate by k (Bar Chart)', fontsize=14)
    ax3.set_xticks(k_values)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    summary_data = []
    for k in k_values[:5]:  # Show first 5 k values
        total = results[k]['total']
        hits = results[k]['hits']
        rate = hits / total * 100 if total > 0 else 0
        summary_data.append([f'k={k}', f'{hits}/{total}', f'{rate:.2f}%'])
    
    table = ax4.table(cellText=summary_data,
                      colLabels=['k value', 'Hits/Total', 'Hit Rate'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.3, 0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Color code the header
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('k-NN True Pair Inclusion Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'knn_true_pair_inclusion.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved to: {output_path}")
    
    return hit_rates


def run_knn_pair_test(
    glue_dir: str,
    output_dir: str = "./knn_validation",
    k_values=range(1, 11),
    use_rep: str = "X_glue",
    metric: str = "cosine",
    subsample_ratio: float = 1.0,
    use_gpu: bool = None,
    verbose: bool = True
):
    """
    Main function to run the k-NN pair inclusion test.
    
    Parameters:
    -----------
    glue_dir : str
        Directory containing GLUE output files
    output_dir : str
        Directory to save results
    k_values : range or list
        Values of k to test
    use_rep : str
        Which embedding to use (default: X_glue)
    metric : str
        Distance metric for k-NN
    subsample_ratio : float
        Fraction of cells to subsample for testing
    use_gpu : bool or None
        Whether to use GPU. If None, auto-detect
    verbose : bool
        Print progress messages
    """
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
    
    print("\n" + "="*80)
    print("k-NN TRUE PAIR INCLUSION TEST")
    print("="*80)
    print(f"GLUE directory: {glue_dir}")
    print(f"Output directory: {output_dir}")
    print(f"k values to test: {list(k_values)}")
    print(f"Embedding: {use_rep}")
    print(f"Metric: {metric}")
    print(f"GPU acceleration: {'enabled' if use_gpu and GPU_AVAILABLE else 'disabled'}")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    rna_path = os.path.join(glue_dir, "glue-rna-emb.h5ad")
    atac_path = os.path.join(glue_dir, "glue-atac-emb.h5ad")
    
    print("Loading RNA embeddings...")
    rna_adata = ad.read_h5ad(rna_path)
    rna_embedding = pd.DataFrame(
        rna_adata.obsm[use_rep],
        index=rna_adata.obs.index
    )
    rna_obs = rna_adata.obs.copy()
    
    print("Loading ATAC embeddings...")
    atac_adata = ad.read_h5ad(atac_path)
    atac_embedding = pd.DataFrame(
        atac_adata.obsm[use_rep],
        index=atac_adata.obs.index
    )
    atac_obs = atac_adata.obs.copy()
    
    print(f"RNA cells: {len(rna_embedding)}")
    print(f"ATAC cells: {len(atac_embedding)}")
    print(f"Embedding dimensions: {rna_embedding.shape[1]}")
    
    # Find true pairs
    print("\n🔍 Finding true paired cells...")
    atac_to_rna_pairs = find_true_pairs(rna_obs, atac_obs, verbose=verbose)
    
    if len(atac_to_rna_pairs) == 0:
        raise ValueError("No paired cells found! Check barcode matching.")
    
    # Optional subsampling
    if subsample_ratio < 1.0:
        n_sample = max(1, int(len(atac_to_rna_pairs) * subsample_ratio))
        np.random.seed(42)
        sampled_atac = np.random.choice(
            list(atac_to_rna_pairs.keys()), 
            size=n_sample, 
            replace=False
        )
        atac_to_rna_pairs = {k: atac_to_rna_pairs[k] for k in sampled_atac}
        print(f"Subsampled to {len(atac_to_rna_pairs)} paired cells")
    
    # Run k-NN test
    print(f"\n🔬 Testing k-NN pair inclusion for k={list(k_values)}...")
    
    start_time = time.time()
    
    if use_gpu and GPU_AVAILABLE:
        results = test_knn_pair_inclusion_gpu(
            rna_embedding,
            atac_embedding,
            atac_to_rna_pairs,
            rna_embedding.index,
            k_values=k_values,
            metric=metric
        )
    else:
        results = test_knn_pair_inclusion_cpu(
            rna_embedding,
            atac_embedding,
            atac_to_rna_pairs,
            rna_embedding.index,
            k_values=k_values,
            metric=metric
        )
    
    elapsed = time.time() - start_time
    print(f"Testing completed in {elapsed:.2f} seconds")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    hit_rates = plot_results(results, output_dir)
    
    # Save detailed results
    print("\n💾 Saving results...")
    
    # Summary CSV
    summary_df = pd.DataFrame([
        {
            'k': k,
            'hits': results[k]['hits'],
            'total': results[k]['total'],
            'hit_rate': results[k]['hits'] / results[k]['total'] * 100 if results[k]['total'] > 0 else 0,
            'mean_position': np.mean(results[k]['hit_positions']) if results[k]['hit_positions'] else np.nan
        }
        for k in k_values
    ])
    summary_path = os.path.join(output_dir, 'knn_pair_inclusion_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Detailed results JSON
    import json
    detailed_results = {
        'parameters': {
            'glue_dir': glue_dir,
            'k_values': list(k_values),
            'metric': metric,
            'use_rep': use_rep,
            'subsample_ratio': subsample_ratio,
            'total_paired_cells': len(atac_to_rna_pairs)
        },
        'results': {
            k: {
                'hits': results[k]['hits'],
                'total': results[k]['total'],
                'hit_rate': results[k]['hits'] / results[k]['total'] * 100 if results[k]['total'] > 0 else 0
            }
            for k in k_values
        }
    }
    
    json_path = os.path.join(output_dir, 'knn_pair_inclusion_detailed.json')
    with open(json_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Total paired cells tested: {results[1]['total']}")
    print("\nTrue Pair Inclusion Rates:")
    for k in k_values:
        rate = results[k]['hits'] / results[k]['total'] * 100 if results[k]['total'] > 0 else 0
        print(f"  k={k:2d}: {rate:6.2f}% ({results[k]['hits']}/{results[k]['total']} cells)")
    
    print(f"\nResults saved to: {output_dir}")
    print("="*80)
    
    return results, summary_df


if __name__ == "__main__":
    # Example usage
    results, summary = run_knn_pair_test(
        glue_dir="/path/to/glue/output",  # Update this path
        output_dir="./knn_validation_results",
        k_values=range(1, 11),
        use_rep="X_glue",
        metric="cosine",
        subsample_ratio=0.1,  # Test on 10% of cells for speed
        use_gpu=True,
        verbose=True
    )