"""
Multiomics Embedding Visualization with Statistical Analysis (v2.3)

Updates:
1. Cosine Similarity is now ABSOLUTE (0 to 1).
   - We check if the 'axis' of severity agrees, ignoring the +/- direction.
   - P-value checks if random axes are as aligned as the observed ones.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cross_decomposition import CCA
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
                batch.append('ATAC')
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
    """Compute ASW-batch (0 to 1, Higher is Better)."""
    valid_mask = pd.notna(labels)
    if valid_mask.sum() < 10: return np.nan
    X = embedding[valid_mask]
    y = np.array(labels)[valid_mask]
    if len(np.unique(y)) < 2: return np.nan
    
    raw_score = silhouette_score(X, y, metric=metric)
    asw_batch = np.clip((1.0 - raw_score) / 2.0, 0.0, 1.0)
    return asw_batch


def compute_asw_pvalue(embedding, labels, n_permutations=1000, seed=42):
    """Compute ASW-batch and permutation p-value."""
    np.random.seed(seed)
    valid_mask = pd.notna(labels)
    if valid_mask.sum() < 10: return np.nan, np.nan
    X = embedding[valid_mask]
    y = np.array(labels)[valid_mask]
    if len(np.unique(y)) < 2: return np.nan, np.nan
    
    # Observed
    raw_obs = silhouette_score(X, y)
    observed_asw = np.clip((1.0 - raw_obs) / 2.0, 0.0, 1.0)
    
    perm_asw = []
    for _ in range(n_permutations):
        perm_labels = np.random.permutation(y)
        try:
            raw_perm = silhouette_score(X, perm_labels)
            score_perm = np.clip((1.0 - raw_perm) / 2.0, 0.0, 1.0)
            perm_asw.append(score_perm)
        except: continue
    
    if len(perm_asw) == 0: return observed_asw, np.nan
    
    perm_asw = np.array(perm_asw)
    p_value = (np.sum(perm_asw >= observed_asw) + 1) / (len(perm_asw) + 1)
    return observed_asw, p_value


def plot_embedding_by_modality(embedding_df, metadata, output_path, title_prefix=""):
    """Plot embedding colored by modality."""
    fig, ax = plt.subplots(figsize=(10, 8))
    x, y = embedding_df['PC1'].values, embedding_df['PC2'].values
    modality = metadata.loc[embedding_df.index, 'modality'].values
    colors = {'ATAC': '#E74C3C', 'RNA': '#3498DB'}
    
    for mod in ['RNA', 'ATAC']:
        mask = modality == mod
        ax.scatter(x[mask], y[mask], c=colors[mod], label=mod, alpha=0.7,
                   s=80 if mod == 'ATAC' else 50, edgecolors='white', linewidths=0.5, zorder=10 if mod == 'ATAC' else 5)
    
    x_range, y_range = x.max() - x.min(), y.max() - y.min()
    max_range = max(x_range, y_range) * 1.1
    x_mid, y_mid = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    
    ax.set_xlabel('PC1', fontweight='bold')
    ax.set_ylabel('PC2', fontweight='bold')
    ax.set_title(f'{title_prefix}Embedding by Modality', fontweight='bold', pad=15)
    
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12, markerscale=1.5)
    legend.get_frame().set_alpha(0.9)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_embedding_by_batch(embedding_df, metadata, output_path, title_prefix=""):
    """Plot embedding by batch."""
    fig, ax = plt.subplots(figsize=(12, 9))
    x, y = embedding_df['PC1'].values, embedding_df['PC2'].values
    modality = metadata.loc[embedding_df.index, 'modality'].values
    batch = metadata.loc[embedding_df.index, 'batch'].values
    
    atac_mask = modality == 'ATAC'
    ax.scatter(x[atac_mask], y[atac_mask], c='#7F8C8D', label='ATAC', alpha=0.8,
               s=90, edgecolors='white', linewidths=0.5, zorder=10, marker='D')
    
    rna_mask = modality == 'RNA'
    rna_batches = np.unique(batch[rna_mask])
    n_batches = len(rna_batches)
    batch_colors = sns.color_palette("Set2", n_batches)
    batch_color_map = {b: batch_colors[i] for i, b in enumerate(rna_batches)}
    
    for b in rna_batches:
        mask = (modality == 'RNA') & (batch == b)
        ax.scatter(x[mask], y[mask], c=[batch_color_map[b]], label=f'RNA-{b}',
                   alpha=0.7, s=50, edgecolors='white', linewidths=0.3, zorder=5)
    
    x_range, y_range = x.max() - x.min(), y.max() - y.min()
    max_range = max(x_range, y_range) * 1.1
    x_mid, y_mid = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    
    ax.set_xlabel('PC1', fontweight='bold')
    ax.set_ylabel('PC2', fontweight='bold')
    ax.set_title(f'{title_prefix}Embedding by Batch', fontweight='bold', pad=15)
    
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
                       fontsize=11, markerscale=1.3, ncol=1 if n_batches <= 6 else 2)
    legend.get_frame().set_alpha(0.9)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def run_cca_for_modality(embedding_df, metadata, modality_filter, n_components=10):
    """Run CCA to find trajectory direction."""
    modality = metadata.loc[embedding_df.index, 'modality'].values
    sev_levels = metadata.loc[embedding_df.index, 'sev.level'].values
    
    if modality_filter == 'all': modality_mask = np.ones(len(modality), dtype=bool)
    else: modality_mask = modality == modality_filter
    
    valid_mask = modality_mask & pd.notna(sev_levels)
    if valid_mask.sum() < 5: return np.nan, None, None, valid_mask, None
    
    X = embedding_df.values[valid_mask]
    sev = sev_levels[valid_mask].astype(float)
    n_pcs = min(n_components, X.shape[1])
    X_subset = X[:, :n_pcs]
    
    if len(np.unique(sev)) < 2: return np.nan, None, None, valid_mask, None
    
    try:
        cca = CCA(n_components=1)
        cca.fit(X_subset, sev.reshape(-1, 1))
        U, V = cca.transform(X_subset, sev.reshape(-1, 1))
        
        # Taking ABS for correlation score (standard CCA metric)
        cca_score = np.abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1])
        
        direction_2d = cca.x_weights_[:2, 0] if len(cca.x_weights_) >= 2 else cca.x_weights_[:, 0]
        direction_full = cca.x_weights_[:, 0]
        return cca_score, direction_2d, direction_full, valid_mask, cca
    except Exception as e:
        print(f"CCA failed for {modality_filter}: {e}")
        return np.nan, None, None, valid_mask, None


def compute_cca_pvalue(embedding_df, metadata, modality_filter='all', n_permutations=1000, seed=42):
    """Compute CCA correlation p-value."""
    np.random.seed(seed)
    modality = metadata.loc[embedding_df.index, 'modality'].values
    sev_levels = metadata.loc[embedding_df.index, 'sev.level'].values
    
    if modality_filter == 'all': modality_mask = np.ones(len(modality), dtype=bool)
    else: modality_mask = modality == modality_filter
    
    valid_mask = modality_mask & pd.notna(sev_levels)
    if valid_mask.sum() < 5: return np.nan, np.nan
    
    X = embedding_df.values[valid_mask]
    sev = sev_levels[valid_mask].astype(float)
    n_pcs = min(10, X.shape[1])
    X_subset = X[:, :n_pcs]
    
    if len(np.unique(sev)) < 2: return np.nan, np.nan
    
    try:
        cca = CCA(n_components=1)
        cca.fit(X_subset, sev.reshape(-1, 1))
        U, V = cca.transform(X_subset, sev.reshape(-1, 1))
        observed_score = np.abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1])
    except: return np.nan, np.nan
    
    perm_scores = []
    for _ in range(n_permutations):
        perm_sev = np.random.permutation(sev)
        try:
            cca_perm = CCA(n_components=1)
            cca_perm.fit(X_subset, perm_sev.reshape(-1, 1))
            U_perm, V_perm = cca_perm.transform(X_subset, perm_sev.reshape(-1, 1))
            score = np.abs(np.corrcoef(U_perm[:, 0], V_perm[:, 0])[0, 1])
            perm_scores.append(score)
        except: continue
        
    if len(perm_scores) == 0: return observed_score, np.nan
    p_value = (np.sum(np.array(perm_scores) >= observed_score) + 1) / (len(perm_scores) + 1)
    return observed_score, p_value


def compute_abs_cosine_similarity(v1, v2):
    """
    Compute ABSOLUTE cosine similarity.
    Returns value between 0 and 1.
    """
    if v1 is None or v2 is None: return np.nan
    min_len = min(len(v1), len(v2))
    v1, v2 = v1[:min_len], v2[:min_len]
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10: return np.nan
    
    # Calculate raw cosine then take Absolute value
    raw_cos = np.dot(v1, v2) / (norm1 * norm2)
    return np.abs(raw_cos)


def compute_cosine_pvalue(embedding_df, metadata, observed_abs_cosine, n_permutations=1000, seed=42):
    """
    Compute p-value for the ABSOLUTE Cosine Similarity.
    Tests if the magnitude of alignment is significantly greater than random.
    """
    if pd.isna(observed_abs_cosine): return np.nan
    
    np.random.seed(seed)
    
    # Pre-extract data
    def get_data_for_mod(mod_name):
        mod_mask = metadata['modality'] == mod_name
        sev_levels = metadata['sev.level']
        valid_mask = mod_mask & pd.notna(sev_levels)
        if valid_mask.sum() < 5: return None, None
        X = embedding_df.loc[valid_mask].values
        n_pcs = min(10, X.shape[1])
        X = X[:, :n_pcs]
        y = sev_levels[valid_mask].values.astype(float)
        return X, y

    X_rna, y_rna = get_data_for_mod('RNA')
    X_atac, y_atac = get_data_for_mod('ATAC')
    
    if X_rna is None or X_atac is None: return np.nan
    
    perm_sims = []
    
    for _ in range(n_permutations):
        try:
            # Shuffle independently
            y_rna_perm = np.random.permutation(y_rna)
            y_atac_perm = np.random.permutation(y_atac)
            
            cca_rna = CCA(n_components=1)
            cca_rna.fit(X_rna, y_rna_perm.reshape(-1, 1))
            dir_rna = cca_rna.x_weights_[:, 0]
            
            cca_atac = CCA(n_components=1)
            cca_atac.fit(X_atac, y_atac_perm.reshape(-1, 1))
            dir_atac = cca_atac.x_weights_[:, 0]
            
            # Compute ABSOLUTE Cosine
            sim = compute_abs_cosine_similarity(dir_rna, dir_atac)
            if not pd.isna(sim):
                perm_sims.append(sim)
        except: continue
            
    if len(perm_sims) == 0: return np.nan
    
    perm_sims = np.array(perm_sims)
    
    # One-sided p-value: Fraction of random abs_similarities >= observed abs_similarity
    p_value = (np.sum(perm_sims >= observed_abs_cosine) + 1) / (len(perm_sims) + 1)
    return p_value


def plot_embedding_with_dual_cca(embedding_df, metadata, output_path, title_prefix=""):
    """Plot embedding and compute stats."""
    fig, ax = plt.subplots(figsize=(12, 10))
    x, y = embedding_df['PC1'].values, embedding_df['PC2'].values
    sev_levels = metadata.loc[embedding_df.index, 'sev.level'].values
    valid_sev_mask = pd.notna(sev_levels)
    
    sev_valid = sev_levels[valid_sev_mask].astype(float)
    norm_sev = (sev_valid - sev_valid.min()) / (sev_valid.max() - sev_valid.min() + 1e-16)
    
    scatter = ax.scatter(x[valid_sev_mask], y[valid_sev_mask], c=norm_sev, cmap='viridis_r',
                         alpha=0.8, s=60, edgecolors='white', linewidths=0.5, zorder=5)
    
    if not valid_sev_mask.all():
        ax.scatter(x[~valid_sev_mask], y[~valid_sev_mask], c='lightgray', alpha=0.5,
                   s=40, edgecolors='gray', linewidths=0.3, label='No severity data', zorder=1)
    
    rna_score, rna_dir_2d, rna_dir_full, _, _ = run_cca_for_modality(embedding_df, metadata, 'RNA')
    atac_score, atac_dir_2d, atac_dir_full, _, _ = run_cca_for_modality(embedding_df, metadata, 'ATAC')
    all_score, _, _, _, _ = run_cca_for_modality(embedding_df, metadata, 'all')
    
    scale = 0.35 * max(np.ptp(x), np.ptp(y))
    x_center, y_center = np.mean(x), np.mean(y)
    
    def draw_direction(direction, color, label, linestyle='--', linewidth=3, offset=(0, 0)):
        if direction is None or len(direction) < 2: return
        dx, dy = direction[0], direction[1]
        norm = np.sqrt(dx**2 + dy**2)
        if norm < 1e-10: return
        dx, dy = dx/norm * scale, dy/norm * scale
        ox, oy = offset
        ax.plot([x_center - dx + ox, x_center + dx + ox], 
                [y_center - dy + oy, y_center + dy + oy],
                color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.9, label=label, zorder=15)
        ax.annotate('', xy=(x_center + dx*0.85 + ox, y_center + dy*0.85 + oy),
                    xytext=(x_center + ox, y_center + oy),
                    arrowprops=dict(arrowstyle='->', color=color, lw=linewidth-0.5), zorder=16)
    
    draw_direction(rna_dir_2d, '#3498DB', f'RNA CCA (r={rna_score:.3f})', linestyle='--', linewidth=3.5)
    draw_direction(atac_dir_2d, '#E74C3C', f'ATAC CCA (r={atac_score:.3f})', linestyle='-.', linewidth=3.5)
    
    # Compute Absolute Cosine Similarity
    abs_cosine_sim = compute_abs_cosine_similarity(rna_dir_full, atac_dir_full)
    
    # Compute P-value for Absolute Cosine
    cosine_pval = compute_cosine_pvalue(embedding_df, metadata, abs_cosine_sim, n_permutations=1000)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Severity Level (normalized)', rotation=270, labelpad=20, fontweight='bold')
    
    x_range, y_range = x.max() - x.min(), y.max() - y.min()
    max_range = max(x_range, y_range) * 1.1
    x_mid, y_mid = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    
    ax.set_xlabel('PC1', fontweight='bold')
    ax.set_ylabel('PC2', fontweight='bold')
    ax.set_title(f'{title_prefix}Severity Trajectory\nRNA vs ATAC CCA Directions', fontweight='bold', pad=15)
    
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
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
        'cosine_similarity': abs_cosine_sim,
        'cosine_pvalue': cosine_pval
    }


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("MULTIOMICS EMBEDDING VISUALIZATION (v2.3)")
    print("Features: Abs Cosine Sim, scIB-style ASW")
    print("=" * 60)
    
    print("\nLoading data...")
    expr_df, prop_df, sample_data, atac_meta = load_data()
    prop_has_data = not prop_df.isnull().all().all()
    metadata = create_merged_metadata(expr_df, sample_data, atac_meta)
    
    print(f"Total samples: {len(metadata)}")
    print(f"ATAC samples: {(metadata['modality'] == 'ATAC').sum()}")
    print(f"RNA samples: {(metadata['modality'] == 'RNA').sum()}")
    
    stats_results = {'expression': {}, 'proportion': {}}
    
    # ==================== EXPRESSION EMBEDDING ====================
    print("\n" + "=" * 60)
    print("EXPRESSION EMBEDDING ANALYSIS")
    print("=" * 60)
    
    plot_embedding_by_modality(expr_df, metadata, os.path.join(OUTPUT_DIR, 'expression_modality.png'), 'Expression ')
    asw_mod, pval_mod = compute_asw_pvalue(expr_df.values, metadata['modality'].values)
    stats_results['expression']['modality_asw'] = asw_mod
    stats_results['expression']['modality_pvalue'] = pval_mod
    print(f"   Modality ASW (Higher=Better): {asw_mod:.4f} (p={pval_mod:.4f})")
    
    plot_embedding_by_batch(expr_df, metadata, os.path.join(OUTPUT_DIR, 'expression_batch.png'), 'Expression ')
    asw_batch, pval_batch = compute_asw_pvalue(expr_df.values, metadata['batch'].values)
    stats_results['expression']['batch_asw'] = asw_batch
    stats_results['expression']['batch_pvalue'] = pval_batch
    print(f"   Batch ASW (Higher=Better): {asw_batch:.4f} (p={pval_batch:.4f})")
    
    print("\n3. CCA trajectory analysis...")
    cca_res = plot_embedding_with_dual_cca(expr_df, metadata, os.path.join(OUTPUT_DIR, 'expression_cca_trajectory.png'), 'Expression ')
    stats_results['expression'].update(cca_res)
    
    _, rna_p = compute_cca_pvalue(expr_df, metadata, 'RNA')
    _, atac_p = compute_cca_pvalue(expr_df, metadata, 'ATAC')
    _, all_p = compute_cca_pvalue(expr_df, metadata, 'all')
    stats_results['expression'].update({'cca_rna_pvalue': rna_p, 'cca_atac_pvalue': atac_p, 'cca_all_pvalue': all_p})
    
    print(f"   RNA CCA Score: {cca_res['rna_score']:.4f}, p: {rna_p:.4f}")
    print(f"   ATAC CCA Score: {cca_res['atac_score']:.4f}, p: {atac_p:.4f}")
    print(f"   Abs Cosine Sim: {cca_res['cosine_similarity']:.4f}, p: {cca_res['cosine_pvalue']:.4f}")

    # ==================== PROPORTION EMBEDDING ====================
    if prop_has_data:
        print("\n" + "=" * 60)
        print("PROPORTION EMBEDDING ANALYSIS")
        print("=" * 60)
        
        plot_embedding_by_modality(prop_df, metadata, os.path.join(OUTPUT_DIR, 'proportion_modality.png'), 'Proportion ')
        asw_mod_p, pval_mod_p = compute_asw_pvalue(prop_df.values, metadata['modality'].values)
        stats_results['proportion']['modality_asw'] = asw_mod_p
        stats_results['proportion']['modality_pvalue'] = pval_mod_p
        
        plot_embedding_by_batch(prop_df, metadata, os.path.join(OUTPUT_DIR, 'proportion_batch.png'), 'Proportion ')
        asw_batch_p, pval_batch_p = compute_asw_pvalue(prop_df.values, metadata['batch'].values)
        stats_results['proportion']['batch_asw'] = asw_batch_p
        stats_results['proportion']['batch_pvalue'] = pval_batch_p
        
        cca_res_p = plot_embedding_with_dual_cca(prop_df, metadata, os.path.join(OUTPUT_DIR, 'proportion_cca_trajectory.png'), 'Proportion ')
        stats_results['proportion'].update(cca_res_p)
        
        _, rna_pp = compute_cca_pvalue(prop_df, metadata, 'RNA')
        _, atac_pp = compute_cca_pvalue(prop_df, metadata, 'ATAC')
        _, all_pp = compute_cca_pvalue(prop_df, metadata, 'all')
        stats_results['proportion'].update({'cca_rna_pvalue': rna_pp, 'cca_atac_pvalue': atac_pp, 'cca_all_pvalue': all_pp})
    else:
        stats_results['proportion']['status'] = 'No data available'

    # ==================== WRITE SUMMARY ====================
    print("\nWriting summary...")
    summary_path = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("MULTIOMICS EMBEDDING VISUALIZATION - STATISTICAL SUMMARY (v2.3)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. MODALITY MIXING & BATCH EFFECTS (ASW)\n")
        f.write("-" * 40 + "\n")
        f.write("   Metric: ASW-batch (0 to 1, Higher is Better)\n\n")
        f.write("   EXPRESSION:\n")
        f.write(f"      Modality Mixing ASW: {stats_results['expression']['modality_asw']:.4f} (p={stats_results['expression']['modality_pvalue']:.4f})\n")
        f.write(f"      Batch Effect ASW:    {stats_results['expression']['batch_asw']:.4f} (p={stats_results['expression']['batch_pvalue']:.4f})\n\n")
        
        if prop_has_data:
            f.write("   PROPORTION:\n")
            f.write(f"      Modality Mixing ASW: {stats_results['proportion']['modality_asw']:.4f} (p={stats_results['proportion']['modality_pvalue']:.4f})\n")
            f.write(f"      Batch Effect ASW:    {stats_results['proportion']['batch_asw']:.4f} (p={stats_results['proportion']['batch_pvalue']:.4f})\n\n")
            
        f.write("2. CCA TRAJECTORY ALIGNMENT (Abs Cosine Similarity)\n")
        f.write("-" * 40 + "\n")
        f.write("   Tests if RNA and ATAC severity axes are aligned (ignoring sign).\n")
        f.write("   Range: 0 (Orthogonal) to 1 (Parallel/Anti-parallel).\n\n")
        
        f.write(f"   Expression Abs Cosine Sim: {stats_results['expression']['cosine_similarity']:.4f}\n")
        f.write(f"   Expression Cosine Pval:    {stats_results['expression']['cosine_pvalue']:.4f}\n\n")
        
        if prop_has_data:
            f.write(f"   Proportion Abs Cosine Sim: {stats_results['proportion']['cosine_similarity']:.4f}\n")
            f.write(f"   Proportion Cosine Pval:    {stats_results['proportion']['cosine_pvalue']:.4f}\n")
            
    print(f"Summary written to: {summary_path}")
    return stats_results

if __name__ == "__main__":
    main()