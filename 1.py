import numpy as np
import anndata as ad
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def check_embedding_whitening(glue_dir, embedding_key="X_glue", verbose=True):
    """
    Check if scGLUE embeddings are whitened (zero mean, unit variance).
    
    Parameters:
    -----------
    glue_dir : str
        Path to the scGLUE output directory
    embedding_key : str
        Key to access embeddings in obsm (default: "X_glue")
    verbose : bool
        Print detailed statistics
    
    Returns:
    --------
    dict : Dictionary containing whitening statistics and assessment
    """
    
    results = {}
    
    # Construct file paths
    rna_path = os.path.join(glue_dir, "glue-rna-emb.h5ad")
    atac_path = os.path.join(glue_dir, "glue-atac-emb.h5ad")
    
    # Check if files exist
    if not os.path.exists(rna_path):
        raise FileNotFoundError(f"RNA embedding file not found: {rna_path}")
    if not os.path.exists(atac_path):
        raise FileNotFoundError(f"ATAC embedding file not found: {atac_path}")
    
    print("=" * 60)
    print("CHECKING SCGLUE EMBEDDING WHITENING STATUS")
    print("=" * 60)
    
    # Check both RNA and ATAC embeddings
    for modality, file_path in [("RNA", rna_path), ("ATAC", atac_path)]:
        print(f"\n📊 Analyzing {modality} embeddings...")
        print("-" * 40)
        
        # Load the data
        adata = ad.read_h5ad(file_path)
        
        # Check if embedding exists
        if embedding_key not in adata.obsm:
            print(f"❌ {embedding_key} not found in {modality} data")
            continue
        
        # Get embeddings
        embeddings = adata.obsm[embedding_key]
        n_cells, n_dims = embeddings.shape
        
        print(f"Embedding shape: {n_cells} cells × {n_dims} dimensions")
        
        # Calculate statistics per dimension
        means = np.mean(embeddings, axis=0)
        stds = np.std(embeddings, axis=0)
        vars = np.var(embeddings, axis=0)
        
        # Overall statistics
        overall_mean = np.mean(embeddings)
        overall_std = np.std(embeddings)
        
        # Check for whitening characteristics
        mean_of_means = np.mean(means)
        std_of_means = np.std(means)
        mean_of_stds = np.mean(stds)
        std_of_stds = np.std(stds)
        
        # Detailed per-dimension analysis
        near_zero_mean_dims = np.sum(np.abs(means) < 0.01)
        near_unit_var_dims = np.sum(np.abs(vars - 1.0) < 0.1)
        
        # Check correlation between dimensions (whitened should be uncorrelated)
        corr_matrix = np.corrcoef(embeddings.T)
        off_diagonal_mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        off_diagonal_corrs = corr_matrix[off_diagonal_mask]
        mean_abs_corr = np.mean(np.abs(off_diagonal_corrs))
        max_abs_corr = np.max(np.abs(off_diagonal_corrs))
        
        # Assessment
        is_centered = np.abs(mean_of_means) < 0.01 and std_of_means < 0.01
        is_standardized = np.abs(mean_of_stds - 1.0) < 0.1 and std_of_stds < 0.1
        is_decorrelated = mean_abs_corr < 0.1
        is_whitened = is_centered and is_standardized and is_decorrelated
        
        # Store results
        modality_results = {
            'n_cells': n_cells,
            'n_dims': n_dims,
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'mean_of_dim_means': mean_of_means,
            'std_of_dim_means': std_of_means,
            'mean_of_dim_stds': mean_of_stds,
            'std_of_dim_stds': std_of_stds,
            'dims_near_zero_mean': near_zero_mean_dims,
            'dims_near_unit_var': near_unit_var_dims,
            'mean_abs_correlation': mean_abs_corr,
            'max_abs_correlation': max_abs_corr,
            'is_centered': is_centered,
            'is_standardized': is_standardized,
            'is_decorrelated': is_decorrelated,
            'is_whitened': is_whitened,
            'dim_means': means,
            'dim_stds': stds
        }
        results[modality] = modality_results
        
        # Print results
        print(f"\n🔍 STATISTICS:")
        print(f"  Overall mean: {overall_mean:.6f}")
        print(f"  Overall std: {overall_std:.6f}")
        print(f"\n  Per-dimension analysis:")
        print(f"    Mean of dimension means: {mean_of_means:.6f}")
        print(f"    Std of dimension means: {std_of_means:.6f}")
        print(f"    Mean of dimension stds: {mean_of_stds:.6f}")
        print(f"    Std of dimension stds: {std_of_stds:.6f}")
        print(f"    Dims with |mean| < 0.01: {near_zero_mean_dims}/{n_dims}")
        print(f"    Dims with |var - 1| < 0.1: {near_unit_var_dims}/{n_dims}")
        print(f"\n  Correlation analysis:")
        print(f"    Mean |correlation|: {mean_abs_corr:.6f}")
        print(f"    Max |correlation|: {max_abs_corr:.6f}")
        
        # Whitening assessment
        print(f"\n✅ WHITENING STATUS:")
        print(f"  Centered (zero mean): {'✓' if is_centered else '✗'}")
        print(f"  Standardized (unit var): {'✓' if is_standardized else '✗'}")
        print(f"  Decorrelated: {'✓' if is_decorrelated else '✗'}")
        print(f"  → Fully whitened: {'YES' if is_whitened else 'NO'}")
        
        if verbose and n_dims <= 20:
            print(f"\n  First 5 dimension statistics:")
            for i in range(min(5, n_dims)):
                print(f"    Dim {i}: mean={means[i]:.4f}, std={stds[i]:.4f}")
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if results:
        any_whitened = any(r.get('is_whitened', False) for r in results.values())
        any_centered = any(r.get('is_centered', False) for r in results.values())
        
        if any_whitened:
            print("⚠️ WARNING: Embeddings appear to be WHITENED!")
            print("This can cause issues with cosine similarity-based k-NN:")
            print("  - Neighbors may come from opposite sides of axes")
            print("  - Gene activity profiles may be anti-correlated")
            print("\nRECOMMENDATIONS:")
            print("  1. Consider using Euclidean distance instead of cosine")
            print("  2. Or use the raw embeddings before whitening")
            print("  3. Or apply inverse whitening transformation")
        elif any_centered:
            print("⚠️ WARNING: Embeddings appear to be CENTERED (zero mean)!")
            print("This can affect cosine similarity calculations.")
            print("\nRECOMMENDATION: Consider adding back the mean or using Euclidean distance.")
        else:
            print("✅ Embeddings do not appear to be whitened.")
            print("Cosine similarity should work as expected.")
    
    return results


def visualize_whitening_check(glue_dir, embedding_key="X_glue", output_dir=None):
    """
    Create visualizations to check whitening status of embeddings.
    
    Parameters:
    -----------
    glue_dir : str
        Path to the scGLUE output directory
    embedding_key : str
        Key to access embeddings in obsm
    output_dir : str, optional
        Directory to save plots. If None, uses glue_dir
    """
    
    if output_dir is None:
        output_dir = glue_dir
    
    # Get whitening statistics
    results = check_embedding_whitening(glue_dir, embedding_key, verbose=False)
    
    # Create visualization for each modality
    for modality in results.keys():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{modality} Embedding Whitening Analysis', fontsize=16)
        
        # Plot 1: Distribution of dimension means
        ax = axes[0, 0]
        dim_means = results[modality]['dim_means']
        ax.hist(dim_means, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', label='Zero mean')
        ax.set_xlabel('Dimension Mean')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Dimension Means')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of dimension stds
        ax = axes[0, 1]
        dim_stds = results[modality]['dim_stds']
        ax.hist(dim_stds, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=1, color='red', linestyle='--', label='Unit std')
        ax.set_xlabel('Dimension Std Dev')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Dimension Std Devs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Mean vs Std scatter
        ax = axes[1, 0]
        ax.scatter(dim_means, dim_stds, alpha=0.6)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Dimension Mean')
        ax.set_ylabel('Dimension Std Dev')
        ax.set_title('Mean vs Std Dev per Dimension')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
        Whitening Status Summary:
        
        • Centered: {'✓' if results[modality]['is_centered'] else '✗'}
        • Standardized: {'✓' if results[modality]['is_standardized'] else '✗'}
        • Decorrelated: {'✓' if results[modality]['is_decorrelated'] else '✗'}
        
        Overall Status: {'WHITENED' if results[modality]['is_whitened'] else 'NOT WHITENED'}
        
        Mean of means: {results[modality]['mean_of_dim_means']:.6f}
        Mean of stds: {results[modality]['mean_of_dim_stds']:.6f}
        Mean |correlation|: {results[modality]['mean_abs_correlation']:.6f}
        """
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(output_dir, f'whitening_check_{modality.lower()}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to: {save_path}")


# Example usage
if __name__ == "__main__":
    import sys
    
    glue_dir = "/dcs07/hongkai/data/harry/result/all/multiomics/integration/glue"
    
    # Check whitening status
    results = check_embedding_whitening(glue_dir)
    
    # Optional: Generate visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    visualize_whitening_check(glue_dir)
    
    # Check for the specific issue mentioned
    if any(r.get('is_centered', False) for r in results.values()):
        print("\n" + "⚠️" * 20)
        print("IMPORTANT: The embeddings are centered/whitened!")
        print("This explains why cosine similarity might pick anti-correlated neighbors.")
        print("The 'opposite side of axis' phenomenon you mentioned is likely occurring.")
        print("⚠️" * 20)