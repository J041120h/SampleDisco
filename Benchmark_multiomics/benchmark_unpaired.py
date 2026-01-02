"""
Multiomics Embedding Visualization with Statistical Analysis (v2)

This script visualizes integrated ATAC and RNA embeddings, computing:
1. Modality mixing (ASW scores with permutation p-values)
2. Batch effects visualization
3. CCA trajectory analysis with severity levels - showing SEPARATE directions for RNA and ATAC

Key update: CCA directions are computed separately for each modality to check alignment.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cross_decomposition import CCA
from scipy import stats
import warnings
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150

# Output directory
OUTPUT_DIR = '/dcl01/hongkai/data/data/hjiang/result/visualization'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load and merge all data files."""
    # Load expression embedding
    expr_df = pd.read_csv('/dcl01/hongkai/data/data/hjiang/result/integration/X_DR_expression.csv', index_col=0)
    
    # Load proportion embedding
    prop_df = pd.read_csv('/dcl01/hongkai/data/data/hjiang/result/integration/X_DR_proportion.csv', index_col=0)
    
    # Load metadata
    sample_data = pd.read_csv('/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv', encoding='utf-8-sig')
    atac_meta = pd.read_csv('/dcl01/hongkai/data/data/hjiang/Data/covid_data/ATAC_Metadata.csv', encoding='utf-8-sig')
    
    return expr_df, prop_df, sample_data, atac_meta


def create_merged_metadata(expr_df, sample_data, atac_meta):
    """Create merged metadata with modality and batch info."""
    # ATAC samples
    atac_samples = set(atac_meta['sample'].values)
    
    # Create modality labels
    modality = []
    batch = []
    sev_level = []
    
    for sample in expr_df.index:
        if sample in atac_samples:
            modality.append('ATAC')
            # Get batch from ATAC metadata
            atac_row = atac_meta[atac_meta['sample'] == sample]
            if len(atac_row) > 0:
                batch.append('ATAC')  # Single ATAC batch
                sev_val = atac_row['sev.level'].values[0]
                sev_level.append(sev_val)
            else:
                batch.append('Unknown')
                sev_level.append(np.nan)
        else:
            modality.append('RNA')
            # Get batch from sample_data
            rna_row = sample_data[sample_data['sample'] == sample]
            if len(rna_row) > 0:
                batch.append(rna_row['batch'].values[0])
                sev_level.append(rna_row['sev.level'].values[0])
            else:
                batch.append('Unknown')
                sev_level.append(np.nan)
    
    metadata = pd.DataFrame({
        'sample': expr_df.index,
        'modality': modality,
        'batch': batch,
        'sev.level': sev_level
    })
    metadata.set_index('sample', inplace=True)
    
    return metadata


def compute_asw(embedding, labels, metric='euclidean'):
    """Compute Average Silhouette Width."""
    valid_mask = pd.notna(labels)
    if valid_mask.sum() < 10:
        return np.nan
    
    X = embedding[valid_mask]
    y = np.array(labels)[valid_mask]
    
    # Need at least 2 unique labels
    if len(np.unique(y)) < 2:
        return np.nan
    
    return silhouette_score(X, y, metric=metric)


def compute_asw_pvalue(embedding, labels, n_permutations=1000, seed=42):
    """Compute ASW and permutation p-value."""
    np.random.seed(seed)
    
    valid_mask = pd.notna(labels)
    if valid_mask.sum() < 10:
        return np.nan, np.nan
    
    X = embedding[valid_mask]
    y = np.array(labels)[valid_mask]
    
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    
    # Observed ASW
    observed_asw = silhouette_score(X, y)
    
    # Permutation test
    perm_asw = []
    for _ in range(n_permutations):
        perm_labels = np.random.permutation(y)
        try:
            perm_score = silhouette_score(X, perm_labels)
            perm_asw.append(perm_score)
        except:
            continue
    
    if len(perm_asw) == 0:
        return observed_asw, np.nan
    
    # Two-tailed p-value
    perm_asw = np.array(perm_asw)
    p_value = (np.sum(np.abs(perm_asw) >= np.abs(observed_asw)) + 1) / (len(perm_asw) + 1)
    
    return observed_asw, p_value


def plot_embedding_by_modality(embedding_df, metadata, output_path, title_prefix=""):
    """Plot embedding colored by modality (ATAC vs RNA)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use first two PCs
    x = embedding_df['PC1'].values
    y = embedding_df['PC2'].values
    
    modality = metadata.loc[embedding_df.index, 'modality'].values
    
    # Color palette
    colors = {'ATAC': '#E74C3C', 'RNA': '#3498DB'}
    
    for mod in ['RNA', 'ATAC']:  # Plot RNA first, then ATAC on top
        mask = modality == mod
        ax.scatter(x[mask], y[mask], 
                   c=colors[mod], 
                   label=mod, 
                   alpha=0.7,
                   s=80 if mod == 'ATAC' else 50,
                   edgecolors='white',
                   linewidths=0.5,
                   zorder=10 if mod == 'ATAC' else 5)
    
    # Set equal aspect with reasonable limits
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    max_range = max(x_range, y_range) * 1.1
    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    
    ax.set_xlabel('PC1', fontweight='bold')
    ax.set_ylabel('PC2', fontweight='bold')
    ax.set_title(f'{title_prefix}Embedding by Modality', fontweight='bold', pad=15)
    
    # Legend in top right with larger font
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                       shadow=True, fontsize=12, markerscale=1.5)
    legend.get_frame().set_alpha(0.9)
    
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_embedding_by_batch(embedding_df, metadata, output_path, title_prefix=""):
    """Plot embedding with ATAC as gray and RNA colored by batch."""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    x = embedding_df['PC1'].values
    y = embedding_df['PC2'].values
    
    modality = metadata.loc[embedding_df.index, 'modality'].values
    batch = metadata.loc[embedding_df.index, 'batch'].values
    
    # Plot ATAC samples in gray first
    atac_mask = modality == 'ATAC'
    ax.scatter(x[atac_mask], y[atac_mask],
               c='#7F8C8D',
               label='ATAC',
               alpha=0.8,
               s=90,
               edgecolors='white',
               linewidths=0.5,
               zorder=10,
               marker='D')
    
    # Get unique RNA batches
    rna_mask = modality == 'RNA'
    rna_batches = np.unique(batch[rna_mask])
    
    # Color palette for batches
    n_batches = len(rna_batches)
    batch_colors = sns.color_palette("Set2", n_batches)
    batch_color_map = {b: batch_colors[i] for i, b in enumerate(rna_batches)}
    
    # Plot RNA samples by batch
    for b in rna_batches:
        mask = (modality == 'RNA') & (batch == b)
        ax.scatter(x[mask], y[mask],
                   c=[batch_color_map[b]],
                   label=f'RNA-{b}',
                   alpha=0.7,
                   s=50,
                   edgecolors='white',
                   linewidths=0.3,
                   zorder=5)
    
    # Set equal aspect
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    max_range = max(x_range, y_range) * 1.1
    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    
    ax.set_xlabel('PC1', fontweight='bold')
    ax.set_ylabel('PC2', fontweight='bold')
    ax.set_title(f'{title_prefix}Embedding by Batch', fontweight='bold', pad=15)
    
    # Legend
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True,
                       shadow=True, fontsize=11, markerscale=1.3,
                       ncol=1 if n_batches <= 6 else 2)
    legend.get_frame().set_alpha(0.9)
    
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def run_cca_for_modality(embedding_df, metadata, modality_filter, n_components=10):
    """
    Run CCA to find trajectory direction correlated with severity for a specific modality.
    
    Parameters:
    -----------
    embedding_df : pd.DataFrame
        Embedding data with samples as index
    metadata : pd.DataFrame
        Metadata with 'modality' and 'sev.level' columns
    modality_filter : str
        'RNA', 'ATAC', or 'all' for which samples to use
    n_components : int
        Number of PCs to use for CCA
        
    Returns:
    --------
    cca_score : float
        CCA correlation score
    direction_2d : np.ndarray
        Direction vector in first 2 PCs (for visualization)
    direction_full : np.ndarray
        Full direction vector
    valid_mask : np.ndarray
        Boolean mask of valid samples used
    cca_model : CCA
        Fitted CCA model
    """
    # Get modality and severity
    modality = metadata.loc[embedding_df.index, 'modality'].values
    sev_levels = metadata.loc[embedding_df.index, 'sev.level'].values
    
    # Filter by modality
    if modality_filter == 'all':
        modality_mask = np.ones(len(modality), dtype=bool)
    else:
        modality_mask = modality == modality_filter
    
    # Combined mask: modality filter AND valid severity
    valid_mask = modality_mask & pd.notna(sev_levels)
    
    if valid_mask.sum() < 5:
        print(f"Warning: Only {valid_mask.sum()} samples for {modality_filter}, skipping CCA")
        return np.nan, None, None, valid_mask, None
    
    X = embedding_df.values[valid_mask]
    sev = sev_levels[valid_mask].astype(float)
    
    # Use available PCs
    n_pcs = min(n_components, X.shape[1])
    X_subset = X[:, :n_pcs]
    
    # Check if we have enough variation in severity
    if len(np.unique(sev)) < 2:
        print(f"Warning: Not enough severity variation for {modality_filter}")
        return np.nan, None, None, valid_mask, None
    
    # Run CCA
    try:
        cca = CCA(n_components=1)
        cca.fit(X_subset, sev.reshape(-1, 1))
        
        # Get CCA scores
        U, V = cca.transform(X_subset, sev.reshape(-1, 1))
        cca_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        
        # Direction vector in PC space (for first 2 PCs)
        direction_2d = cca.x_weights_[:2, 0] if len(cca.x_weights_) >= 2 else cca.x_weights_[:, 0]
        direction_full = cca.x_weights_[:, 0]
        
        return cca_score, direction_2d, direction_full, valid_mask, cca
    
    except Exception as e:
        print(f"CCA failed for {modality_filter}: {e}")
        return np.nan, None, None, valid_mask, None


def compute_cca_pvalue(embedding_df, metadata, modality_filter='all', n_permutations=1000, seed=42):
    """Compute CCA score and permutation p-value for a specific modality."""
    np.random.seed(seed)
    
    modality = metadata.loc[embedding_df.index, 'modality'].values
    sev_levels = metadata.loc[embedding_df.index, 'sev.level'].values
    
    # Filter by modality
    if modality_filter == 'all':
        modality_mask = np.ones(len(modality), dtype=bool)
    else:
        modality_mask = modality == modality_filter
    
    valid_mask = modality_mask & pd.notna(sev_levels)
    
    if valid_mask.sum() < 5:
        return np.nan, np.nan
    
    X = embedding_df.values[valid_mask]
    sev = sev_levels[valid_mask].astype(float)
    
    n_pcs = min(10, X.shape[1])
    X_subset = X[:, :n_pcs]
    
    if len(np.unique(sev)) < 2:
        return np.nan, np.nan
    
    # Observed CCA score
    try:
        cca = CCA(n_components=1)
        cca.fit(X_subset, sev.reshape(-1, 1))
        U, V = cca.transform(X_subset, sev.reshape(-1, 1))
        observed_score = np.abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1])
    except:
        return np.nan, np.nan
    
    # Permutation test
    perm_scores = []
    for _ in range(n_permutations):
        perm_sev = np.random.permutation(sev)
        try:
            cca_perm = CCA(n_components=1)
            cca_perm.fit(X_subset, perm_sev.reshape(-1, 1))
            U_perm, V_perm = cca_perm.transform(X_subset, perm_sev.reshape(-1, 1))
            score = np.abs(np.corrcoef(U_perm[:, 0], V_perm[:, 0])[0, 1])
            perm_scores.append(score)
        except:
            continue
    
    if len(perm_scores) == 0:
        return observed_score, np.nan
    
    # One-tailed p-value (we want high correlation)
    p_value = (np.sum(np.array(perm_scores) >= observed_score) + 1) / (len(perm_scores) + 1)
    
    return observed_score, p_value


def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    if v1 is None or v2 is None:
        return np.nan
    
    # Make sure both vectors have same length
    min_len = min(len(v1), len(v2))
    v1 = v1[:min_len]
    v2 = v2[:min_len]
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        return np.nan
    
    return np.dot(v1, v2) / (norm1 * norm2)


def plot_embedding_with_dual_cca(embedding_df, metadata, output_path, title_prefix=""):
    """
    Plot embedding colored by severity with SEPARATE CCA directions for RNA and ATAC.
    
    This allows visual comparison of whether the two modalities agree on the 
    direction of disease progression.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    x = embedding_df['PC1'].values
    y = embedding_df['PC2'].values
    
    modality = metadata.loc[embedding_df.index, 'modality'].values
    sev_levels = metadata.loc[embedding_df.index, 'sev.level'].values
    valid_sev_mask = pd.notna(sev_levels)
    
    # Normalize severity for coloring (for samples with valid severity)
    sev_valid = sev_levels[valid_sev_mask].astype(float)
    norm_sev = (sev_valid - sev_valid.min()) / (sev_valid.max() - sev_valid.min() + 1e-16)
    
    # Plot samples colored by severity
    scatter = ax.scatter(x[valid_sev_mask], y[valid_sev_mask],
                         c=norm_sev,
                         cmap='viridis_r',
                         alpha=0.8,
                         s=60,
                         edgecolors='white',
                         linewidths=0.5,
                         zorder=5)
    
    # Plot samples without severity in gray
    if not valid_sev_mask.all():
        ax.scatter(x[~valid_sev_mask], y[~valid_sev_mask],
                   c='lightgray',
                   alpha=0.5,
                   s=40,
                   edgecolors='gray',
                   linewidths=0.3,
                   label='No severity data',
                   zorder=1)
    
    # Compute CCA directions for each modality
    rna_score, rna_dir_2d, rna_dir_full, _, _ = run_cca_for_modality(
        embedding_df, metadata, 'RNA'
    )
    atac_score, atac_dir_2d, atac_dir_full, _, _ = run_cca_for_modality(
        embedding_df, metadata, 'ATAC'
    )
    
    # Also compute overall CCA for reference
    all_score, all_dir_2d, all_dir_full, _, _ = run_cca_for_modality(
        embedding_df, metadata, 'all'
    )
    
    # Compute scale for direction lines
    scale = 0.35 * max(np.ptp(x), np.ptp(y))
    x_center = np.mean(x)
    y_center = np.mean(y)
    
    # Helper function to draw direction line with arrow
    def draw_direction(direction, color, label, linestyle='--', linewidth=3, offset=(0, 0)):
        if direction is None or len(direction) < 2:
            return
        
        dx, dy = direction[0], direction[1]
        norm = np.sqrt(dx**2 + dy**2)
        if norm < 1e-10:
            return
        dx, dy = dx/norm * scale, dy/norm * scale
        
        # Apply offset to avoid overlapping lines
        ox, oy = offset
        
        # Draw line
        ax.plot([x_center - dx + ox, x_center + dx + ox], 
                [y_center - dy + oy, y_center + dy + oy],
                color=color, linestyle=linestyle, linewidth=linewidth, 
                alpha=0.9, label=label, zorder=15)
        
        # Add arrow to show direction of increasing severity
        ax.annotate('', 
                    xy=(x_center + dx*0.85 + ox, y_center + dy*0.85 + oy),
                    xytext=(x_center + ox, y_center + oy),
                    arrowprops=dict(arrowstyle='->', color=color, lw=linewidth-0.5),
                    zorder=16)
    
    # Draw RNA direction (blue)
    draw_direction(rna_dir_2d, '#3498DB', f'RNA CCA (r={rna_score:.3f})', 
                   linestyle='--', linewidth=3.5)
    
    # Draw ATAC direction (red) with slight offset if needed
    draw_direction(atac_dir_2d, '#E74C3C', f'ATAC CCA (r={atac_score:.3f})', 
                   linestyle='-.', linewidth=3.5)
    
    # Optionally draw overall direction (gray, thinner)
    # draw_direction(all_dir_2d, '#7F8C8D', f'All CCA (r={all_score:.3f})', 
    #                linestyle=':', linewidth=2)
    
    # Compute cosine similarity between RNA and ATAC directions
    cosine_sim = compute_cosine_similarity(rna_dir_full, atac_dir_full)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Severity Level (normalized)', rotation=270, labelpad=20, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Set equal aspect
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    max_range = max(x_range, y_range) * 1.1
    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    
    ax.set_xlabel('PC1', fontweight='bold')
    ax.set_ylabel('PC2', fontweight='bold')
    ax.set_title(f'{title_prefix}Severity Trajectory\nRNA vs ATAC CCA Directions', 
                 fontweight='bold', pad=15)
    
    # Legend with statistics
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True,
                       shadow=True, fontsize=12, markerscale=1.3)
    legend.get_frame().set_alpha(0.9)
    
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return {
        'rna_score': rna_score,
        'atac_score': atac_score,
        'all_score': all_score,
        'rna_direction': rna_dir_full,
        'atac_direction': atac_dir_full,
        'cosine_similarity': cosine_sim
    }


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("MULTIOMICS EMBEDDING VISUALIZATION (v2)")
    print("With separate CCA directions for RNA and ATAC")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    expr_df, prop_df, sample_data, atac_meta = load_data()
    
    # Check if proportion data is available
    prop_has_data = not prop_df.isnull().all().all()
    print(f"Expression embedding shape: {expr_df.shape}")
    print(f"Proportion embedding has data: {prop_has_data}")
    
    # Create merged metadata
    print("\nCreating merged metadata...")
    metadata = create_merged_metadata(expr_df, sample_data, atac_meta)
    
    print(f"Total samples: {len(metadata)}")
    print(f"ATAC samples: {(metadata['modality'] == 'ATAC').sum()}")
    print(f"RNA samples: {(metadata['modality'] == 'RNA').sum()}")
    print(f"Unique batches: {metadata['batch'].nunique()}")
    
    # Dictionary to store all statistics
    stats_results = {
        'expression': {},
        'proportion': {}
    }
    
    # ==================== EXPRESSION EMBEDDING ====================
    print("\n" + "=" * 60)
    print("EXPRESSION EMBEDDING ANALYSIS")
    print("=" * 60)
    
    # 1. Modality mixing analysis
    print("\n1. Modality mixing analysis...")
    plot_embedding_by_modality(expr_df, metadata, 
                                os.path.join(OUTPUT_DIR, 'expression_modality.png'),
                                title_prefix='Expression ')
    
    asw_modality, pval_modality = compute_asw_pvalue(
        expr_df.values, 
        metadata['modality'].values,
        n_permutations=1000
    )
    stats_results['expression']['modality_asw'] = asw_modality
    stats_results['expression']['modality_pvalue'] = pval_modality
    print(f"   Modality ASW: {asw_modality:.4f}, p-value: {pval_modality:.4f}")
    
    # 2. Batch analysis
    print("\n2. Batch analysis...")
    plot_embedding_by_batch(expr_df, metadata,
                            os.path.join(OUTPUT_DIR, 'expression_batch.png'),
                            title_prefix='Expression ')
    
    asw_batch, pval_batch = compute_asw_pvalue(
        expr_df.values,
        metadata['batch'].values,
        n_permutations=1000
    )
    stats_results['expression']['batch_asw'] = asw_batch
    stats_results['expression']['batch_pvalue'] = pval_batch
    print(f"   Batch ASW: {asw_batch:.4f}, p-value: {pval_batch:.4f}")
    
    # 3. CCA trajectory analysis with DUAL directions
    print("\n3. CCA trajectory analysis (RNA and ATAC separately)...")
    cca_results = plot_embedding_with_dual_cca(
        expr_df, metadata,
        os.path.join(OUTPUT_DIR, 'expression_cca_trajectory.png'),
        title_prefix='Expression '
    )
    
    # Store CCA results
    stats_results['expression']['cca_rna_score'] = cca_results['rna_score']
    stats_results['expression']['cca_atac_score'] = cca_results['atac_score']
    stats_results['expression']['cca_all_score'] = cca_results['all_score']
    stats_results['expression']['cca_cosine_similarity'] = cca_results['cosine_similarity']
    
    if cca_results['rna_direction'] is not None:
        stats_results['expression']['cca_rna_direction'] = cca_results['rna_direction'].tolist()
    if cca_results['atac_direction'] is not None:
        stats_results['expression']['cca_atac_direction'] = cca_results['atac_direction'].tolist()
    
    # Compute p-values for each modality
    _, rna_pval = compute_cca_pvalue(expr_df, metadata, 'RNA', n_permutations=1000)
    _, atac_pval = compute_cca_pvalue(expr_df, metadata, 'ATAC', n_permutations=1000)
    _, all_pval = compute_cca_pvalue(expr_df, metadata, 'all', n_permutations=1000)
    
    stats_results['expression']['cca_rna_pvalue'] = rna_pval
    stats_results['expression']['cca_atac_pvalue'] = atac_pval
    stats_results['expression']['cca_all_pvalue'] = all_pval
    
    print(f"   RNA CCA Score: {cca_results['rna_score']:.4f}, p-value: {rna_pval:.4f}")
    print(f"   ATAC CCA Score: {cca_results['atac_score']:.4f}, p-value: {atac_pval:.4f}")
    print(f"   All CCA Score: {cca_results['all_score']:.4f}, p-value: {all_pval:.4f}")
    print(f"   Cosine similarity (RNA vs ATAC): {cca_results['cosine_similarity']:.4f}")
    
    # ==================== PROPORTION EMBEDDING ====================
    if prop_has_data:
        print("\n" + "=" * 60)
        print("PROPORTION EMBEDDING ANALYSIS")
        print("=" * 60)
        
        # 1. Modality mixing analysis
        print("\n1. Modality mixing analysis...")
        plot_embedding_by_modality(prop_df, metadata,
                                    os.path.join(OUTPUT_DIR, 'proportion_modality.png'),
                                    title_prefix='Proportion ')
        
        asw_modality_p, pval_modality_p = compute_asw_pvalue(
            prop_df.values,
            metadata['modality'].values,
            n_permutations=1000
        )
        stats_results['proportion']['modality_asw'] = asw_modality_p
        stats_results['proportion']['modality_pvalue'] = pval_modality_p
        
        # 2. Batch analysis
        print("\n2. Batch analysis...")
        plot_embedding_by_batch(prop_df, metadata,
                                os.path.join(OUTPUT_DIR, 'proportion_batch.png'),
                                title_prefix='Proportion ')
        
        asw_batch_p, pval_batch_p = compute_asw_pvalue(
            prop_df.values,
            metadata['batch'].values,
            n_permutations=1000
        )
        stats_results['proportion']['batch_asw'] = asw_batch_p
        stats_results['proportion']['batch_pvalue'] = pval_batch_p
        
        # 3. CCA trajectory analysis with DUAL directions
        print("\n3. CCA trajectory analysis (RNA and ATAC separately)...")
        cca_results_prop = plot_embedding_with_dual_cca(
            prop_df, metadata,
            os.path.join(OUTPUT_DIR, 'proportion_cca_trajectory.png'),
            title_prefix='Proportion '
        )
        
        stats_results['proportion']['cca_rna_score'] = cca_results_prop['rna_score']
        stats_results['proportion']['cca_atac_score'] = cca_results_prop['atac_score']
        stats_results['proportion']['cca_all_score'] = cca_results_prop['all_score']
        stats_results['proportion']['cca_cosine_similarity'] = cca_results_prop['cosine_similarity']
        
        # P-values for proportion embedding
        _, rna_pval_p = compute_cca_pvalue(prop_df, metadata, 'RNA', n_permutations=1000)
        _, atac_pval_p = compute_cca_pvalue(prop_df, metadata, 'ATAC', n_permutations=1000)
        _, all_pval_p = compute_cca_pvalue(prop_df, metadata, 'all', n_permutations=1000)
        
        stats_results['proportion']['cca_rna_pvalue'] = rna_pval_p
        stats_results['proportion']['cca_atac_pvalue'] = atac_pval_p
        stats_results['proportion']['cca_all_pvalue'] = all_pval_p
        
        # Cross-embedding comparison: cosine between expression and proportion directions
        if cca_results['rna_direction'] is not None and cca_results_prop['rna_direction'] is not None:
            cross_cosine_rna = compute_cosine_similarity(
                cca_results['rna_direction'], 
                cca_results_prop['rna_direction']
            )
            stats_results['cross_embedding_cosine_rna'] = cross_cosine_rna
        
        if cca_results['atac_direction'] is not None and cca_results_prop['atac_direction'] is not None:
            cross_cosine_atac = compute_cosine_similarity(
                cca_results['atac_direction'], 
                cca_results_prop['atac_direction']
            )
            stats_results['cross_embedding_cosine_atac'] = cross_cosine_atac
    else:
        print("\n" + "=" * 60)
        print("PROPORTION EMBEDDING: NO DATA AVAILABLE")
        print("=" * 60)
        print("The proportion embedding file contains only empty values.")
        stats_results['proportion']['status'] = 'No data available'
    
    # ==================== WRITE SUMMARY ====================
    print("\n" + "=" * 60)
    print("WRITING SUMMARY")
    print("=" * 60)
    
    summary_path = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MULTIOMICS EMBEDDING VISUALIZATION - STATISTICAL SUMMARY (v2)\n")
        f.write("With Separate CCA Directions for RNA and ATAC\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("DATA OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total samples: {len(metadata)}\n")
        f.write(f"ATAC samples: {(metadata['modality'] == 'ATAC').sum()}\n")
        f.write(f"RNA samples: {(metadata['modality'] == 'RNA').sum()}\n")
        f.write(f"Unique batches: {metadata['batch'].nunique()}\n")
        f.write(f"Batches: {', '.join(metadata['batch'].unique())}\n")
        sev_levels = sorted([int(x) for x in metadata['sev.level'].dropna().unique()])
        f.write(f"Severity levels: {sev_levels} (1=Healthy, 2=Mild, 3=Moderate, 4=Severe)\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("EXPRESSION EMBEDDING RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. MODALITY MIXING (ATAC vs RNA)\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Average Silhouette Width (ASW): {stats_results['expression']['modality_asw']:.4f}\n")
        f.write(f"   Permutation p-value (n=1000):   {stats_results['expression']['modality_pvalue']:.4f}\n")
        f.write("   Interpretation: Lower ASW indicates better mixing.\n")
        f.write("   Negative ASW suggests modalities are well-integrated.\n\n")
        
        f.write("2. BATCH EFFECTS (RNA batches + ATAC)\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Average Silhouette Width (ASW): {stats_results['expression']['batch_asw']:.4f}\n")
        f.write(f"   Permutation p-value (n=1000):   {stats_results['expression']['batch_pvalue']:.4f}\n")
        f.write("   Interpretation: Lower ASW indicates reduced batch effects.\n\n")
        
        f.write("3. CCA TRAJECTORY ANALYSIS (Severity) - SEPARATE MODALITIES\n")
        f.write("-" * 40 + "\n")
        f.write("   RNA samples only:\n")
        f.write(f"      CCA Correlation Score:        {stats_results['expression']['cca_rna_score']:.4f}\n")
        f.write(f"      Permutation p-value (n=1000): {stats_results['expression']['cca_rna_pvalue']:.4f}\n")
        f.write("   ATAC samples only:\n")
        f.write(f"      CCA Correlation Score:        {stats_results['expression']['cca_atac_score']:.4f}\n")
        f.write(f"      Permutation p-value (n=1000): {stats_results['expression']['cca_atac_pvalue']:.4f}\n")
        f.write("   All samples combined:\n")
        f.write(f"      CCA Correlation Score:        {stats_results['expression']['cca_all_score']:.4f}\n")
        f.write(f"      Permutation p-value (n=1000): {stats_results['expression']['cca_all_pvalue']:.4f}\n\n")
        
        f.write("4. CCA DIRECTION AGREEMENT (RNA vs ATAC)\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Cosine Similarity:              {stats_results['expression']['cca_cosine_similarity']:.4f}\n")
        f.write("   Interpretation:\n")
        f.write("      +1: Directions perfectly aligned (same trajectory)\n")
        f.write("      -1: Directions opposite (reversed trajectory)\n")
        f.write("       0: Directions orthogonal (unrelated trajectories)\n\n")
        
        if 'cca_rna_direction' in stats_results['expression']:
            f.write("   RNA CCA Direction (first 5 PCs):\n")
            f.write(f"      {[f'{x:.4f}' for x in stats_results['expression']['cca_rna_direction'][:5]]}\n")
        if 'cca_atac_direction' in stats_results['expression']:
            f.write("   ATAC CCA Direction (first 5 PCs):\n")
            f.write(f"      {[f'{x:.4f}' for x in stats_results['expression']['cca_atac_direction'][:5]]}\n\n")
        
        if prop_has_data:
            f.write("=" * 70 + "\n")
            f.write("PROPORTION EMBEDDING RESULTS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("1. MODALITY MIXING (ATAC vs RNA)\n")
            f.write("-" * 40 + "\n")
            f.write(f"   Average Silhouette Width (ASW): {stats_results['proportion']['modality_asw']:.4f}\n")
            f.write(f"   Permutation p-value (n=1000):   {stats_results['proportion']['modality_pvalue']:.4f}\n\n")
            
            f.write("2. BATCH EFFECTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"   Average Silhouette Width (ASW): {stats_results['proportion']['batch_asw']:.4f}\n")
            f.write(f"   Permutation p-value (n=1000):   {stats_results['proportion']['batch_pvalue']:.4f}\n\n")
            
            f.write("3. CCA TRAJECTORY ANALYSIS (Severity) - SEPARATE MODALITIES\n")
            f.write("-" * 40 + "\n")
            f.write("   RNA samples only:\n")
            f.write(f"      CCA Correlation Score:        {stats_results['proportion']['cca_rna_score']:.4f}\n")
            f.write(f"      Permutation p-value (n=1000): {stats_results['proportion']['cca_rna_pvalue']:.4f}\n")
            f.write("   ATAC samples only:\n")
            f.write(f"      CCA Correlation Score:        {stats_results['proportion']['cca_atac_score']:.4f}\n")
            f.write(f"      Permutation p-value (n=1000): {stats_results['proportion']['cca_atac_pvalue']:.4f}\n")
            f.write("   All samples combined:\n")
            f.write(f"      CCA Correlation Score:        {stats_results['proportion']['cca_all_score']:.4f}\n")
            f.write(f"      Permutation p-value (n=1000): {stats_results['proportion']['cca_all_pvalue']:.4f}\n\n")
            
            f.write("4. CCA DIRECTION AGREEMENT (RNA vs ATAC)\n")
            f.write("-" * 40 + "\n")
            f.write(f"   Cosine Similarity:              {stats_results['proportion']['cca_cosine_similarity']:.4f}\n\n")
            
            # Cross-embedding comparison
            if 'cross_embedding_cosine_rna' in stats_results:
                f.write("=" * 70 + "\n")
                f.write("CROSS-EMBEDDING COMPARISON\n")
                f.write("=" * 70 + "\n\n")
                f.write("Cosine similarity between Expression and Proportion CCA directions:\n")
                f.write(f"   RNA directions:  {stats_results.get('cross_embedding_cosine_rna', 'N/A'):.4f}\n")
                f.write(f"   ATAC directions: {stats_results.get('cross_embedding_cosine_atac', 'N/A'):.4f}\n\n")
        else:
            f.write("=" * 70 + "\n")
            f.write("PROPORTION EMBEDDING RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write("   STATUS: No data available.\n")
            f.write("   The proportion embedding file contains only empty values.\n")
            f.write("   Analysis was performed on expression embedding only.\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("GENERATED FILES\n")
        f.write("=" * 70 + "\n\n")
        f.write("Expression Embedding:\n")
        f.write("   - expression_modality.png: Colored by modality (ATAC/RNA)\n")
        f.write("   - expression_batch.png: ATAC gray, RNA colored by batch\n")
        f.write("   - expression_cca_trajectory.png: Colored by severity with\n")
        f.write("     SEPARATE CCA directions for RNA (blue) and ATAC (red)\n\n")
        
        if prop_has_data:
            f.write("Proportion Embedding:\n")
            f.write("   - proportion_modality.png: Colored by modality (ATAC/RNA)\n")
            f.write("   - proportion_batch.png: ATAC gray, RNA colored by batch\n")
            f.write("   - proportion_cca_trajectory.png: Colored by severity with\n")
            f.write("     SEPARATE CCA directions for RNA (blue) and ATAC (red)\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("METHODOLOGY NOTES\n")
        f.write("=" * 70 + "\n\n")
        f.write("ASW (Average Silhouette Width):\n")
        f.write("   - Measures how well samples cluster by labels\n")
        f.write("   - Range: -1 to 1\n")
        f.write("   - For modality mixing: LOWER is better (want mixing)\n")
        f.write("   - P-value from 1000 permutations of labels\n\n")
        
        f.write("CCA (Canonical Correlation Analysis):\n")
        f.write("   - Finds linear combination of PCs maximally correlated with severity\n")
        f.write("   - Computed SEPARATELY for RNA and ATAC samples\n")
        f.write("   - Score: Pearson correlation between CCA projections\n")
        f.write("   - P-value from 1000 permutations of severity labels\n\n")
        
        f.write("Cosine Similarity:\n")
        f.write("   - Measures agreement between CCA direction vectors\n")
        f.write("   - Range: -1 to 1\n")
        f.write("   - |1|: Perfect alignment (directions agree or are opposite)\n")
        f.write("   - 0: Orthogonal (directions are unrelated)\n")
        f.write("   - High |cosine| suggests both modalities capture similar\n")
        f.write("     disease progression patterns\n")
    
    print(f"Summary written to: {summary_path}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return stats_results


if __name__ == "__main__":
    results = main()