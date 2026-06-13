import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from harmony import harmonize
import time
import contextlib
import io
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu, chi2_contingency
import warnings
from matplotlib.gridspec import GridSpec
from copy import deepcopy

def align_gene_distributions(adata_rna, adata_activity, method='zero_aware_quantile', 
                            scale_factor=10000, verbose=True):
    """
    Align gene distributions between RNA and gene activity data before integration.
    
    Parameters:
    -----------
    adata_rna : AnnData
        RNA expression data (should be raw counts)
    adata_activity : AnnData
        Gene activity/accessibility data (should be raw scores)
    method : str
        Method for alignment: 'zero_aware_quantile', 'quantile', 'zscore', 'minmax', or 'rank'
    scale_factor : float
        Scaling factor for normalization (used in quantile methods)
    verbose : bool
        Print progress messages
    
    Returns:
    --------
    adata_rna_aligned, adata_activity_aligned : tuple of AnnData
        Aligned datasets ready for concatenation
    """
    
    if verbose:
        print("=== Aligning Gene Distributions ===")
        print(f"Method: {method}")
    
    # Create copies to avoid modifying original data
    adata_rna_aligned = adata_rna.copy()
    adata_activity_aligned = adata_activity.copy()
    
    # Find common genes
    common_genes = list(set(adata_rna_aligned.var_names) & 
                       set(adata_activity_aligned.var_names))
    
    if verbose:
        print(f"Found {len(common_genes)} common genes between modalities")
        print(f"RNA unique genes: {len(set(adata_rna_aligned.var_names) - set(common_genes))}")
        print(f"ATAC unique genes: {len(set(adata_activity_aligned.var_names) - set(common_genes))}")
    
    # Store raw data before transformation
    adata_rna_aligned.layers['raw'] = adata_rna_aligned.X.copy()
    adata_activity_aligned.layers['raw'] = adata_activity_aligned.X.copy()
    
    if method == 'zero_aware_quantile':
        # Zero-aware quantile normalization - NEW DEFAULT METHOD
        from sklearn.preprocessing import QuantileTransformer
        
        if verbose:
            print("Applying zero-aware quantile matching (preserves ATAC sparsity)")
        
        # First normalize by library size
        sc.pp.normalize_total(adata_rna_aligned, target_sum=scale_factor)
        sc.pp.normalize_total(adata_activity_aligned, target_sum=scale_factor)
        
        # Apply log transformation
        sc.pp.log1p(adata_rna_aligned)
        sc.pp.log1p(adata_activity_aligned)
        
        # Track alignment statistics
        alignment_stats = {
            'genes_processed': 0,
            'genes_skipped_no_positive': 0,
            'avg_atac_sparsity': 0,
            'avg_rna_sparsity': 0
        }
        
        # For common genes, apply zero-aware quantile matching
        for gene in common_genes:
            if gene in adata_rna_aligned.var_names and gene in adata_activity_aligned.var_names:
                # Get gene expression vectors
                rna_expr = adata_rna_aligned[:, gene].X.toarray().flatten()
                atac_expr = adata_activity_aligned[:, gene].X.toarray().flatten()
                
                # Identify zero and non-zero values
                atac_zero_mask = atac_expr == 0
                atac_nonzero_mask = ~atac_zero_mask
                rna_zero_mask = rna_expr == 0
                rna_nonzero_mask = ~rna_zero_mask
                
                # Track sparsity
                alignment_stats['avg_atac_sparsity'] += np.mean(atac_zero_mask)
                alignment_stats['avg_rna_sparsity'] += np.mean(rna_zero_mask)
                
                # Only proceed if both modalities have some positive values
                if np.sum(atac_nonzero_mask) > 1 and np.sum(rna_nonzero_mask) > 1:
                    # Extract positive values only
                    rna_positive = rna_expr[rna_nonzero_mask]
                    atac_positive = atac_expr[atac_nonzero_mask]
                    
                    # Fit quantile transformer on RNA positive values (reference)
                    qt_rna_to_uniform = QuantileTransformer(
                        n_quantiles=min(1000, len(rna_positive)), 
                        output_distribution='uniform', 
                        random_state=42
                    )
                    qt_rna_to_uniform.fit(rna_positive.reshape(-1, 1))
                    
                    # Transform ATAC positive values to uniform distribution
                    try:
                        atac_uniform = qt_rna_to_uniform.transform(atac_positive.reshape(-1, 1))
                        
                        # Inverse transform to match RNA distribution
                        qt_uniform_to_rna = QuantileTransformer(
                            n_quantiles=min(1000, len(rna_positive)), 
                            output_distribution='normal', 
                            random_state=42
                        )
                        qt_uniform_to_rna.fit(rna_positive.reshape(-1, 1))
                        atac_matched = qt_uniform_to_rna.inverse_transform(atac_uniform)
                        
                        # Reconstruct full ATAC expression vector
                        atac_aligned = atac_expr.copy()
                        atac_aligned[atac_nonzero_mask] = atac_matched.flatten()
                        # Keep zeros as zeros: atac_aligned[atac_zero_mask] remains 0
                        
                        # Update ATAC expression
                        adata_activity_aligned[:, gene].X = atac_aligned.reshape(-1, 1)
                        alignment_stats['genes_processed'] += 1
                        
                    except Exception as e:
                        if verbose and alignment_stats['genes_processed'] < 5:  # Only warn for first few
                            print(f"  Warning: Could not align gene {gene}: {str(e)}")
                        alignment_stats['genes_skipped_no_positive'] += 1
                else:
                    alignment_stats['genes_skipped_no_positive'] += 1
        
        # Calculate average sparsity
        if len(common_genes) > 0:
            alignment_stats['avg_atac_sparsity'] /= len(common_genes)
            alignment_stats['avg_rna_sparsity'] /= len(common_genes)
        
        if verbose:
            print(f"  Genes successfully aligned: {alignment_stats['genes_processed']}")
            print(f"  Genes skipped (insufficient positive values): {alignment_stats['genes_skipped_no_positive']}")
            print(f"  Average ATAC sparsity: {alignment_stats['avg_atac_sparsity']:.1%}")
            print(f"  Average RNA sparsity: {alignment_stats['avg_rna_sparsity']:.1%}")
    
    elif method == 'quantile':
        # Original quantile normalization (transforms all values including zeros)
        from sklearn.preprocessing import QuantileTransformer
        
        if verbose:
            print("Applying standard quantile matching (transforms all values)")
        
        # First normalize by library size
        sc.pp.normalize_total(adata_rna_aligned, target_sum=scale_factor)
        sc.pp.normalize_total(adata_activity_aligned, target_sum=scale_factor)
        
        # Apply log transformation
        sc.pp.log1p(adata_rna_aligned)
        sc.pp.log1p(adata_activity_aligned)
        
        # For common genes, apply quantile matching
        for gene in common_genes:
            if gene in adata_rna_aligned.var_names and gene in adata_activity_aligned.var_names:
                # Get gene expression vectors
                rna_expr = adata_rna_aligned[:, gene].X.toarray().flatten()
                atac_expr = adata_activity_aligned[:, gene].X.toarray().flatten()
                
                # Fit quantile transformer on RNA data (reference)
                qt = QuantileTransformer(n_quantiles=min(1000, len(rna_expr)), 
                                        output_distribution='uniform', 
                                        random_state=42)
                qt.fit(rna_expr.reshape(-1, 1))
                
                # Transform ATAC to match RNA distribution
                atac_transformed = qt.transform(atac_expr.reshape(-1, 1))
                
                # Inverse transform to get back to RNA scale
                qt_rna = QuantileTransformer(n_quantiles=min(1000, len(rna_expr)), 
                                            output_distribution='normal', 
                                            random_state=42)
                qt_rna.fit(rna_expr.reshape(-1, 1))
                atac_matched = qt_rna.inverse_transform(atac_transformed)
                
                # Update ATAC expression
                adata_activity_aligned[:, gene].X = atac_matched.reshape(-1, 1)
    
    elif method == 'zscore':
        # Z-score normalization
        sc.pp.normalize_total(adata_rna_aligned, target_sum=scale_factor)
        sc.pp.normalize_total(adata_activity_aligned, target_sum=scale_factor)
        sc.pp.log1p(adata_rna_aligned)
        sc.pp.log1p(adata_activity_aligned)
        
        # Scale to unit variance and zero mean
        sc.pp.scale(adata_rna_aligned, zero_center=True)
        sc.pp.scale(adata_activity_aligned, zero_center=True)
    
    elif method == 'minmax':
        # Min-max scaling to [0, 1] range
        sc.pp.normalize_total(adata_rna_aligned, target_sum=scale_factor)
        sc.pp.normalize_total(adata_activity_aligned, target_sum=scale_factor)
        sc.pp.log1p(adata_rna_aligned)
        sc.pp.log1p(adata_activity_aligned)
        
        from sklearn.preprocessing import MinMaxScaler
        
        # Apply min-max scaling per gene
        for gene in common_genes:
            if gene in adata_rna_aligned.var_names and gene in adata_activity_aligned.var_names:
                scaler = MinMaxScaler()
                
                # Scale RNA
                rna_expr = adata_rna_aligned[:, gene].X.toarray()
                rna_scaled = scaler.fit_transform(rna_expr)
                adata_rna_aligned[:, gene].X = rna_scaled
                
                # Scale ATAC
                atac_expr = adata_activity_aligned[:, gene].X.toarray()
                atac_scaled = scaler.fit_transform(atac_expr)
                adata_activity_aligned[:, gene].X = atac_scaled
    
    elif method == 'rank':
        # Rank-based transformation
        from scipy.stats import rankdata
        
        sc.pp.normalize_total(adata_rna_aligned, target_sum=scale_factor)
        sc.pp.normalize_total(adata_activity_aligned, target_sum=scale_factor)
        sc.pp.log1p(adata_rna_aligned)
        sc.pp.log1p(adata_activity_aligned)
        
        # Convert to ranks and normalize
        for gene in common_genes:
            if gene in adata_rna_aligned.var_names and gene in adata_activity_aligned.var_names:
                # Rank transform RNA
                rna_expr = adata_rna_aligned[:, gene].X.toarray().flatten()
                rna_ranks = rankdata(rna_expr, method='average') / len(rna_expr)
                adata_rna_aligned[:, gene].X = rna_ranks.reshape(-1, 1)
                
                # Rank transform ATAC
                atac_expr = adata_activity_aligned[:, gene].X.toarray().flatten()
                atac_ranks = rankdata(atac_expr, method='average') / len(atac_expr)
                adata_activity_aligned[:, gene].X = atac_ranks.reshape(-1, 1)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'zero_aware_quantile', 'quantile', 'zscore', 'minmax', or 'rank'")
    
    if verbose:
        print(f"Alignment complete using {method} method")
        
        # Print distribution statistics for validation
        print("\n=== Distribution Statistics (subset of common genes) ===")
        sample_genes = common_genes[:min(5, len(common_genes))]
        for gene in sample_genes:
            if gene in adata_rna_aligned.var_names and gene in adata_activity_aligned.var_names:
                rna_vals = adata_rna_aligned[:, gene].X.toarray().flatten()
                atac_vals = adata_activity_aligned[:, gene].X.toarray().flatten()
                print(f"{gene}:")
                print(f"  RNA  - mean: {np.mean(rna_vals):.3f}, std: {np.std(rna_vals):.3f}, zeros: {np.mean(rna_vals==0):.1%}")
                print(f"  ATAC - mean: {np.mean(atac_vals):.3f}, std: {np.std(atac_vals):.3f}, zeros: {np.mean(atac_vals==0):.1%}")
    
    return adata_rna_aligned, adata_activity_aligned


def combine_rna_and_activity_data(
    adata_rna,
    adata_activity,
    rna_cell_meta_path=None,
    activity_cell_meta_path=None,
    rna_sample_meta_path=None,
    activity_sample_meta_path=None,
    rna_sample_column='sample',
    activity_sample_column='sample',
    unified_sample_column='sample',
    align_distributions=True,
    alignment_method='zero_aware_quantile',  # Changed default to new method
    rna_batch_key='batch',
    activity_batch_key='batch',
    unified_batch_key='batch',
    rna_prefix='RNA',
    activity_prefix='ATAC',
    verbose=True
):
    """
    Combine RNA and gene activity data into a single AnnData object.
    Now defaults to zero-aware quantile matching which preserves ATAC sparsity.
    """
    
    if verbose:
        print(f"RNA data shape: {adata_rna.shape}")
        print(f"Gene activity data shape: {adata_activity.shape}")
    
    # Load cell metadata
    if verbose:
        print("=== Loading Cell Metadata ===")
    
    # Handle RNA cell metadata
    if rna_cell_meta_path is not None:
        rna_cell_meta = pd.read_csv(rna_cell_meta_path)
        if 'barcode' not in rna_cell_meta.columns:
            rna_cell_meta['barcode'] = rna_cell_meta.index.astype(str)
    else:
        if verbose:
            print("No RNA cell metadata provided, creating from obs_names")
        rna_cell_meta = pd.DataFrame({
            'barcode': adata_rna.obs_names.astype(str)
        })
        if rna_sample_column not in adata_rna.obs.columns:
            rna_cell_meta[rna_sample_column] = adata_rna.obs_names.str.split(':').str[0]
    
    # Handle gene activity cell metadata
    if activity_cell_meta_path is not None:
        activity_cell_meta = pd.read_csv(activity_cell_meta_path)
        if 'barcode' not in activity_cell_meta.columns:
            activity_cell_meta['barcode'] = activity_cell_meta.index.astype(str)
    else:
        if verbose:
            print("No gene activity cell metadata provided, creating from obs_names")
        activity_cell_meta = pd.DataFrame({
            'barcode': adata_activity.obs_names.astype(str)
        })
        if activity_sample_column not in adata_activity.obs.columns:
            activity_cell_meta[activity_sample_column] = adata_activity.obs_names.str.split(':').str[0]
    
    # Add data type column
    rna_cell_meta['data_type'] = 'RNA'
    activity_cell_meta['data_type'] = 'Gene_Activity'
    
    # Add prefixes to cell barcodes
    rna_cell_meta['barcode'] = rna_prefix + '_' + rna_cell_meta['barcode'].astype(str)
    activity_cell_meta['barcode'] = activity_prefix + '_' + activity_cell_meta['barcode'].astype(str)
    
    # Update AnnData obs_names
    adata_rna.obs_names = [rna_prefix + '_' + str(x) for x in adata_rna.obs_names]
    adata_activity.obs_names = [activity_prefix + '_' + str(x) for x in adata_activity.obs_names]
    
    # Attach cell metadata
    rna_cell_meta.set_index('barcode', inplace=True)
    activity_cell_meta.set_index('barcode', inplace=True)
    
    adata_rna.obs = adata_rna.obs.join(rna_cell_meta, how='left')
    adata_activity.obs = adata_activity.obs.join(activity_cell_meta, how='left')
    
    # Load sample metadata
    if verbose:
        print("=== Loading Sample Metadata ===")
    
    if rna_sample_meta_path is not None:
        rna_sample_meta = pd.read_csv(rna_sample_meta_path)
        adata_rna.obs = adata_rna.obs.merge(rna_sample_meta, on=rna_sample_column, how='left')
        if verbose:
            print("RNA sample metadata loaded and merged")
    
    if activity_sample_meta_path is not None:
        activity_sample_meta = pd.read_csv(activity_sample_meta_path)
        adata_activity.obs = adata_activity.obs.merge(activity_sample_meta, on=activity_sample_column, how='left')
        if verbose:
            print("Gene activity sample metadata loaded and merged")
    
    # Standardize column names
    if verbose:
        print("=== Standardizing Column Names ===")
    
    # Ensure required columns exist
    if rna_sample_column not in adata_rna.obs.columns:
        adata_rna.obs[rna_sample_column] = 'RNA_sample'
    
    if activity_sample_column not in adata_activity.obs.columns:
        adata_activity.obs[activity_sample_column] = 'ATAC_sample'
    
    if rna_batch_key not in adata_rna.obs.columns:
        adata_rna.obs[rna_batch_key] = 'RNA_batch'
    
    if activity_batch_key not in adata_activity.obs.columns:
        adata_activity.obs[activity_batch_key] = 'ATAC_batch'
    
    # Rename columns to unified names
    if rna_sample_column != unified_sample_column:
        if unified_sample_column in adata_rna.obs.columns and unified_sample_column != rna_sample_column:
            adata_rna.obs.drop(columns=[unified_sample_column], inplace=True)
        adata_rna.obs[unified_sample_column] = adata_rna.obs[rna_sample_column]
    
    if activity_sample_column != unified_sample_column:
        if unified_sample_column in adata_activity.obs.columns and unified_sample_column != activity_sample_column:
            adata_activity.obs.drop(columns=[unified_sample_column], inplace=True)
        adata_activity.obs[unified_sample_column] = adata_activity.obs[activity_sample_column]
    
    if rna_batch_key != unified_batch_key:
        if unified_batch_key in adata_rna.obs.columns and unified_batch_key != rna_batch_key:
            adata_rna.obs.drop(columns=[unified_batch_key], inplace=True)
        adata_rna.obs[unified_batch_key] = adata_rna.obs[rna_batch_key]
    
    if activity_batch_key != unified_batch_key:
        if unified_batch_key in adata_activity.obs.columns and unified_batch_key != activity_batch_key:
            adata_activity.obs.drop(columns=[unified_batch_key], inplace=True)
        adata_activity.obs[unified_batch_key] = adata_activity.obs[activity_batch_key]
    
    # Apply distribution alignment
    if align_distributions:
        adata_rna, adata_activity = align_gene_distributions(
            adata_rna, 
            adata_activity, 
            method=alignment_method,
            verbose=verbose
        )

    if verbose:
        print("=== Combining Datasets ===")
        print(f"RNA data shape: {adata_rna.shape}")
        print(f"Gene activity data shape: {adata_activity.shape}")

    adata_combined = sc.concat([adata_rna, adata_activity], axis=0, join='outer')
    
    if verbose:
        print(f"Combined data shape: {adata_combined.shape}")
        print(f"RNA cells: {sum(adata_combined.obs['data_type'] == 'RNA')}")
        print(f"Gene activity cells: {sum(adata_combined.obs['data_type'] == 'Gene_Activity')}")
        print(f"Total unique genes: {adata_combined.n_vars}")
    
    return adata_combined


def clean_obs_for_writing(adata):
    """
    Clean obs dataframe to ensure it can be written to H5AD format.
    """
    obs_clean = adata.obs.copy()
    
    for col in obs_clean.columns:
        if obs_clean[col].dtype == 'object':
            obs_clean[col] = obs_clean[col].fillna('')
            obs_clean[col] = obs_clean[col].astype(str)
            obs_clean[col] = obs_clean[col].replace('nan', '')
        elif obs_clean[col].dtype == 'bool':
            obs_clean[col] = obs_clean[col].astype(str)
        elif obs_clean[col].dtype.name.startswith('category'):
            obs_clean[col] = obs_clean[col].astype(str)
    
    adata.obs = obs_clean
    return adata


def run_leiden_clustering(adata, resolution=1.0, key_added='leiden', random_state=42):
    """
    Run Leiden clustering on the integrated data.
    
    Parameters:
    -----------
    adata : AnnData
        Integrated AnnData object with computed neighbors
    resolution : float
        Resolution parameter for Leiden clustering
    key_added : str
        Key to add to adata.obs for cluster labels
    random_state : int
        Random seed for reproducibility
    """
    print(f"Running Leiden clustering with resolution={resolution}")
    sc.tl.leiden(adata, resolution=resolution, key_added=key_added, random_state=random_state)
    
    # Print cluster statistics
    cluster_counts = adata.obs[key_added].value_counts().sort_index()
    print(f"Found {len(cluster_counts)} clusters")
    print("Cluster sizes:")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} cells")
    
    return adata


def visualize_integration_quality(adata, output_dir, integration_method='harmony', 
                                 leiden_key='leiden', sample_key='sample', 
                                 batch_key='batch', figsize=(20, 16)):
    """
    Create comprehensive visualizations to assess integration quality and modality mixing.
    
    Parameters:
    -----------
    adata : AnnData
        Integrated AnnData object
    output_dir : str
        Directory to save figures
    integration_method : str
        Name of integration method for labeling
    leiden_key : str
        Key in adata.obs containing Leiden clusters
    sample_key : str
        Key in adata.obs containing sample information
    batch_key : str
        Key in adata.obs containing batch information
    """
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, f'{integration_method}_visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Main UMAP visualizations
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # UMAP colored by modality
    ax1 = fig.add_subplot(gs[0, 0])
    sc.pl.umap(adata, color='data_type', ax=ax1, show=False, legend_loc='right margin',
               title='UMAP by Modality', frameon=True)
    
    # UMAP colored by Leiden clusters
    ax2 = fig.add_subplot(gs[0, 1])
    sc.pl.umap(adata, color=leiden_key, ax=ax2, show=False, legend_loc='right margin',
               title='UMAP by Leiden Clusters', frameon=True)
    
    # UMAP colored by sample
    ax3 = fig.add_subplot(gs[0, 2])
    sc.pl.umap(adata, color=sample_key, ax=ax3, show=False, legend_loc=None,
               title='UMAP by Sample', frameon=True)
    
    # UMAP colored by batch
    ax4 = fig.add_subplot(gs[1, 0])
    sc.pl.umap(adata, color=batch_key, ax=ax4, show=False, legend_loc='right margin',
               title='UMAP by Batch', frameon=True)
    
    # Split UMAP by modality
    ax5 = fig.add_subplot(gs[1, 1:])
    for i, modality in enumerate(['RNA', 'Gene_Activity']):
        mask = adata.obs['data_type'] == modality
        ax5.scatter(adata.obsm['X_umap'][mask, 0], 
                   adata.obsm['X_umap'][mask, 1],
                   s=1, alpha=0.5, label=modality)
    ax5.set_xlabel('UMAP1')
    ax5.set_ylabel('UMAP2')
    ax5.set_title('UMAP Split by Modality')
    ax5.legend()
    
    # Modality proportion per cluster
    ax6 = fig.add_subplot(gs[2, :])
    modality_props = pd.crosstab(adata.obs[leiden_key], adata.obs['data_type'], normalize='index')
    modality_props.plot(kind='bar', stacked=True, ax=ax6, color=['#1f77b4', '#ff7f0e'])
    ax6.set_xlabel('Leiden Cluster')
    ax6.set_ylabel('Proportion')
    ax6.set_title('Modality Proportions per Cluster')
    ax6.legend(title='Modality', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0)
    
    plt.suptitle(f'{integration_method.upper()} Integration Overview', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'integration_overview.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Detailed mixing metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Cluster composition heatmap
    ax = axes[0, 0]
    cluster_composition = pd.crosstab(adata.obs[leiden_key], adata.obs['data_type'])
    sns.heatmap(cluster_composition.T, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Cell Count'})
    ax.set_title('Cell Count Heatmap: Clusters vs Modality')
    ax.set_xlabel('Leiden Cluster')
    ax.set_ylabel('Modality')
    
    # Mixing entropy per cluster
    ax = axes[0, 1]
    def calculate_entropy(props):
        """Calculate Shannon entropy for mixing assessment"""
        props = props[props > 0]
        return -np.sum(props * np.log2(props))
    
    mixing_entropy = modality_props.apply(calculate_entropy, axis=1)
    mixing_entropy.plot(kind='bar', ax=ax, color='steelblue')
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Perfect mixing (entropy=1)')
    ax.set_xlabel('Leiden Cluster')
    ax.set_ylabel('Mixing Entropy')
    ax.set_title('Modality Mixing Entropy per Cluster')
    ax.legend()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Sample distribution per modality
    ax = axes[1, 0]
    sample_modality = pd.crosstab(adata.obs[sample_key], adata.obs['data_type'])
    sample_modality.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_xlabel('Sample')
    ax.set_ylabel('Cell Count')
    ax.set_title('Sample Distribution by Modality')
    ax.legend(title='Modality')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Chi-square test for independence
    ax = axes[1, 1]
    chi2, p_value, dof, expected = chi2_contingency(cluster_composition)
    
    # Plot observed vs expected
    x = np.arange(len(cluster_composition))
    width = 0.35
    
    observed_rna = cluster_composition['RNA'].values
    observed_atac = cluster_composition['Gene_Activity'].values
    expected_rna = expected[:, 0]
    expected_atac = expected[:, 1]
    
    ax.bar(x - width/2, observed_rna, width, label='Observed RNA', alpha=0.8, color='#1f77b4')
    ax.bar(x + width/2, observed_atac, width, label='Observed ATAC', alpha=0.8, color='#ff7f0e')
    ax.plot(x - width/2, expected_rna, 'r--', marker='o', label='Expected RNA', alpha=0.7)
    ax.plot(x + width/2, expected_atac, 'r--', marker='s', label='Expected ATAC', alpha=0.7)
    
    ax.set_xlabel('Leiden Cluster')
    ax.set_ylabel('Cell Count')
    ax.set_title(f'Observed vs Expected Distribution\n(χ² = {chi2:.2f}, p = {p_value:.2e})')
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_composition.index)
    ax.legend()
    
    plt.suptitle(f'{integration_method.upper()} Mixing Metrics', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'mixing_metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Integration quality metrics summary
    print(f"\n=== {integration_method.upper()} Integration Quality Metrics ===")
    
    # Calculate overall mixing score
    overall_entropy = calculate_entropy(adata.obs['data_type'].value_counts(normalize=True))
    print(f"Overall modality entropy: {overall_entropy:.3f} (max=1.0 for 2 modalities)")
    
    # Average cluster entropy
    avg_cluster_entropy = mixing_entropy.mean()
    print(f"Average cluster mixing entropy: {avg_cluster_entropy:.3f}")
    
    # Proportion of well-mixed clusters (entropy > 0.5)
    well_mixed = (mixing_entropy > 0.5).sum() / len(mixing_entropy)
    print(f"Proportion of well-mixed clusters: {well_mixed:.2%}")
    
    # Chi-square test results
    print(f"Chi-square test for independence: χ² = {chi2:.2f}, p = {p_value:.2e}")
    if p_value < 0.05:
        print("  → Significant association between clusters and modality (poor mixing)")
    else:
        print("  → No significant association (good mixing)")
    
    # Save metrics to file
    metrics_dict = {
        'integration_method': integration_method,
        'overall_entropy': overall_entropy,
        'avg_cluster_entropy': avg_cluster_entropy,
        'well_mixed_proportion': well_mixed,
        'chi_square': chi2,
        'p_value': p_value,
        'n_clusters': len(cluster_composition),
        'n_cells_rna': (adata.obs['data_type'] == 'RNA').sum(),
        'n_cells_atac': (adata.obs['data_type'] == 'Gene_Activity').sum()
    }
    
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(os.path.join(vis_dir, 'integration_metrics.csv'), index=False)
    
    return metrics_dict

def combined_integration_analysis(
    adata_rna,
    adata_activity,
    rna_cell_meta_path=None,
    activity_cell_meta_path=None,
    rna_sample_meta_path=None,
    activity_sample_meta_path=None,
    output_dir=None,
    rna_sample_column='sample',
    align_distributions=True,
    alignment_method='zero_aware_quantile',  # Changed default to new method
    activity_sample_column='sample',
    unified_sample_column='sample',
    rna_batch_key='batch',
    activity_batch_key='batch',
    unified_batch_key='batch',
    rna_prefix='RNA',
    activity_prefix='ATAC',
    num_PCs=20,
    num_harmony=30,
    num_features=2000,
    min_cells=500,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=False,
    vars_to_regress=[],
    leiden_resolution=1.0,
    run_combat=True,
    verbose=True
):
    """
    Enhanced pipeline with both Harmony and ComBat integration, Leiden clustering, and visualization.
    Now defaults to zero-aware quantile matching for better modality alignment.
    """
    start_time = time.time()
    
    # Create output directories
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            if verbose:
                print("Created output directory")
    
    # Step 1: Combine RNA and gene activity data
    adata_combined = combine_rna_and_activity_data(
        adata_rna=adata_rna,
        adata_activity=adata_activity,
        rna_cell_meta_path=rna_cell_meta_path,
        activity_cell_meta_path=activity_cell_meta_path,
        align_distributions=align_distributions,
        alignment_method=alignment_method,  # Pass through the new parameter
        rna_sample_meta_path=rna_sample_meta_path,
        activity_sample_meta_path=activity_sample_meta_path,
        rna_sample_column=rna_sample_column,
        activity_sample_column=activity_sample_column,
        unified_sample_column=unified_sample_column,
        rna_batch_key=rna_batch_key,
        activity_batch_key=activity_batch_key,
        unified_batch_key=unified_batch_key,
        rna_prefix=rna_prefix,
        activity_prefix=activity_prefix,
        verbose=verbose
    )
    
    # Prepare vars_to_regress for harmony
    vars_to_regress_for_harmony = vars_to_regress.copy()
    if unified_sample_column not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append(unified_sample_column)
    if 'data_type' not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append('data_type')
    
    # Error checking
    all_required_columns = vars_to_regress_for_harmony + [unified_batch_key]
    missing_vars = [col for col in all_required_columns if col not in adata_combined.obs.columns]
    if missing_vars:
        raise KeyError(f"Missing variables in adata_combined.obs: {missing_vars}")
    
    if verbose:
        print("=== Starting Quality Control and Filtering ===")
    
    # Basic filtering
    sc.pp.filter_cells(adata_combined, min_genes=min_features)
    sc.pp.filter_genes(adata_combined, min_cells=min_cells)
    if verbose:
        print(f"After basic filtering -- Cells: {adata_combined.n_obs}, Genes: {adata_combined.n_vars}")
    
    # Mitochondrial QC
    adata_combined.var['mt'] = adata_combined.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_combined, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata_combined = adata_combined[adata_combined.obs['pct_counts_mt'] < pct_mito_cutoff].copy()
    
    # Exclude genes
    mt_genes = adata_combined.var_names[adata_combined.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    adata_combined = adata_combined[:, ~adata_combined.var_names.isin(genes_to_exclude)].copy()
    
    if verbose:
        print(f"After MT filtering -- Cells: {adata_combined.n_obs}, Genes: {adata_combined.n_vars}")
    
    # Sample filtering
    cell_counts_per_sample = adata_combined.obs.groupby(unified_sample_column).size()
    samples_to_keep = cell_counts_per_sample[cell_counts_per_sample >= min_cells].index
    adata_combined = adata_combined[adata_combined.obs[unified_sample_column].isin(samples_to_keep)].copy()
    
    if verbose:
        print(f"Samples retained: {list(samples_to_keep)}")
    
    # Final gene filtering
    min_cells_for_gene = int(0.01 * adata_combined.n_obs)
    sc.pp.filter_genes(adata_combined, min_cells=min_cells_for_gene)
    
    if verbose:
        print(f"Final dimensions -- Cells: {adata_combined.n_obs}, Genes: {adata_combined.n_vars}")
    
    # Optional doublet detection
    if doublet:
        if verbose:
            print("=== Running Doublet Detection ===")
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            sc.pp.scrublet(adata_combined)
        adata_combined = adata_combined[~adata_combined.obs['predicted_doublet']].copy()
        if verbose:
            print(f"After doublet removal -- Cells: {adata_combined.n_obs}")
    
    # Save raw data
    adata_combined.raw = adata_combined.copy()
    
    # Conditional normalization based on alignment method
    if align_distributions and alignment_method == 'zero_aware_quantile':
        if verbose:
            print("=== Skipping normalization (data already aligned with zero-aware quantile) ===")
    else:
        if verbose:
            print("=== Performing standard normalization ===")
        # Standard normalization only if NOT aligned or using different method
        sc.pp.normalize_total(adata_combined, target_sum=1e4)
        sc.pp.log1p(adata_combined)
    
    # HVG selection
    sc.pp.highly_variable_genes(
        adata_combined,
        n_top_genes=num_features,
        flavor='seurat_v3',
        batch_key=unified_sample_column
    )
    
    # Create a copy for parallel processing
    adata_hvg = adata_combined[:, adata_combined.var['highly_variable']].copy()
    
    # === HARMONY INTEGRATION ===
    if verbose:
        print('\n=== HARMONY INTEGRATION PIPELINE ===')
        print(f'Using alignment method: {alignment_method}')
    
    harmony_dir = os.path.join(output_dir, 'harmony_integration')
    if not os.path.exists(harmony_dir):
        os.makedirs(harmony_dir)
    
    # Make a copy for Harmony
    adata_harmony = adata_hvg.copy()
    
    # PCA for Harmony
    sc.tl.pca(adata_harmony, n_comps=num_PCs, svd_solver='arpack')
    
    # Run Harmony
    if verbose:
        print(f'Running Harmony with variables: {", ".join(vars_to_regress_for_harmony)}')
    
    Z = harmonize(
        adata_harmony.obsm['X_pca'],
        adata_harmony.obs,
        batch_key=vars_to_regress_for_harmony,
        max_iter_harmony=num_harmony,
        use_gpu=True
    )
    adata_harmony.obsm['X_pca_harmony'] = Z
    
    # Compute neighbors and UMAP for Harmony
    sc.pp.neighbors(adata_harmony, use_rep='X_pca_harmony', n_neighbors=15)
    sc.tl.umap(adata_harmony)
    
    # Run Leiden clustering for Harmony
    adata_harmony = run_leiden_clustering(adata_harmony, resolution=leiden_resolution, 
                                         key_added='leiden_harmony')
    
    # Visualize Harmony results
    if verbose:
        print("Creating Harmony visualizations...")
    harmony_metrics = visualize_integration_quality(
        adata_harmony, harmony_dir, 'harmony', 
        leiden_key='leiden_harmony',
        sample_key=unified_sample_column,
        batch_key=unified_batch_key
    )
    
    # Save Harmony results
    clean_obs_for_writing(adata_harmony)
    adata_harmony.write(os.path.join(harmony_dir, 'adata_harmony.h5ad'))
    
    # === COMBAT INTEGRATION ===
    if run_combat:
        if verbose:
            print('\n=== COMBAT INTEGRATION PIPELINE ===')
        
        combat_dir = os.path.join(output_dir, 'combat_integration')
        if not os.path.exists(combat_dir):
            os.makedirs(combat_dir)
        
        # Make a fresh copy for ComBat (to avoid contamination)
        adata_combat = adata_hvg.copy()
        sc.pp.combat(adata_combat, key=unified_batch_key)
        
        # PCA on ComBat-corrected data
        sc.tl.pca(adata_combat, n_comps=num_PCs, svd_solver='arpack')
        
        # Compute neighbors and UMAP for ComBat
        sc.pp.neighbors(adata_combat, use_rep='X_pca', n_neighbors=15)
        sc.tl.umap(adata_combat)
        
        # Run Leiden clustering for ComBat
        adata_combat = run_leiden_clustering(adata_combat, resolution=leiden_resolution,
                                            key_added='leiden_combat')
        
        # Visualize ComBat results
        if verbose:
            print("Creating ComBat visualizations...")
        combat_metrics = visualize_integration_quality(
            adata_combat, combat_dir, 'combat',
            leiden_key='leiden_combat',
            sample_key=unified_sample_column,
            batch_key=unified_batch_key
        )
        
        # Save ComBat results
        clean_obs_for_writing(adata_combat)
        adata_combat.write(os.path.join(combat_dir, 'adata_combat.h5ad'))
        
        # === COMPARE INTEGRATION METHODS ===
        if verbose:
            print('\n=== INTEGRATION COMPARISON ===')
        
        compare_integration_methods(adata_harmony, adata_combat, output_dir, 
                                   harmony_metrics, combat_metrics)
    
    # Print summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if verbose:
        print(f"\n=== Analysis Complete ===")
        print(f"Execution time: {elapsed_time:.2f} seconds")
        print(f"Final data shape: {adata_harmony.shape}")
        print(f"Alignment method used: {alignment_method}")
        if run_combat:
            print(f"Both Harmony and ComBat integration completed")
        else:
            print(f"Harmony integration completed")
    
    # Return both integrated objects
    if run_combat:
        return adata_harmony, adata_combat
    else:
        return adata_harmony, None


def compare_integration_methods(adata_harmony, adata_combat, output_dir, 
                               harmony_metrics, combat_metrics):
    """
    Compare Harmony and ComBat integration results.
    
    Parameters:
    -----------
    adata_harmony : AnnData
        Harmony-integrated data
    adata_combat : AnnData
        ComBat-integrated data
    output_dir : str
        Output directory for comparison results
    harmony_metrics : dict
        Metrics from Harmony integration
    combat_metrics : dict
        Metrics from ComBat integration
    """
    
    comparison_dir = os.path.join(output_dir, 'integration_comparison')
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Side-by-side UMAPs colored by modality
    ax = axes[0, 0]
    for i, modality in enumerate(['RNA', 'Gene_Activity']):
        mask = adata_harmony.obs['data_type'] == modality
        ax.scatter(adata_harmony.obsm['X_umap'][mask, 0],
                  adata_harmony.obsm['X_umap'][mask, 1],
                  s=0.5, alpha=0.5, label=modality)
    ax.set_title('Harmony - Modality Distribution')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.legend()
    
    ax = axes[0, 1]
    for i, modality in enumerate(['RNA', 'Gene_Activity']):
        mask = adata_combat.obs['data_type'] == modality
        ax.scatter(adata_combat.obsm['X_umap'][mask, 0],
                  adata_combat.obsm['X_umap'][mask, 1],
                  s=0.5, alpha=0.5, label=modality)
    ax.set_title('ComBat - Modality Distribution')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.legend()
    
    # Plot 2: Metrics comparison
    ax = axes[0, 2]
    metrics_comparison = pd.DataFrame({
        'Harmony': [harmony_metrics['overall_entropy'],
                   harmony_metrics['avg_cluster_entropy'],
                   harmony_metrics['well_mixed_proportion']],
        'ComBat': [combat_metrics['overall_entropy'],
                  combat_metrics['avg_cluster_entropy'],
                  combat_metrics['well_mixed_proportion']]
    }, index=['Overall Entropy', 'Avg Cluster Entropy', 'Well-Mixed Proportion'])
    
    metrics_comparison.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Score')
    ax.set_title('Integration Metrics Comparison')
    ax.legend()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 3: Cluster counts
    ax = axes[1, 0]
    cluster_counts = pd.DataFrame({
        'Harmony': [harmony_metrics['n_clusters']],
        'ComBat': [combat_metrics['n_clusters']]
    })
    cluster_counts.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Number of Leiden Clusters')
    ax.set_xticklabels([''], rotation=0)
    ax.legend()
    
    # Plot 4: Chi-square test p-values
    ax = axes[1, 1]
    p_values = pd.DataFrame({
        'Method': ['Harmony', 'ComBat'],
        'p-value': [harmony_metrics['p_value'], combat_metrics['p_value']]
    })
    bars = ax.bar(p_values['Method'], -np.log10(p_values['p-value']), 
                  color=['#1f77b4', '#ff7f0e'])
    ax.axhline(y=-np.log10(0.05), color='r', linestyle='--', alpha=0.5, 
              label='p=0.05 threshold')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Chi-square Test Results\n(Higher = Better Mixing)')
    ax.legend()
    
    # Plot 5: Summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
    Integration Summary:
    
    HARMONY:
    • Clusters: {harmony_metrics['n_clusters']}
    • Overall Entropy: {harmony_metrics['overall_entropy']:.3f}
    • Avg Cluster Entropy: {harmony_metrics['avg_cluster_entropy']:.3f}
    • Well-Mixed: {harmony_metrics['well_mixed_proportion']:.1%}
    • χ² p-value: {harmony_metrics['p_value']:.2e}
    
    COMBAT:
    • Clusters: {combat_metrics['n_clusters']}
    • Overall Entropy: {combat_metrics['overall_entropy']:.3f}
    • Avg Cluster Entropy: {combat_metrics['avg_cluster_entropy']:.3f}
    • Well-Mixed: {combat_metrics['well_mixed_proportion']:.1%}
    • χ² p-value: {combat_metrics['p_value']:.2e}
    """
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Harmony vs ComBat Integration Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'methods_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save comparison metrics
    comparison_df = pd.DataFrame([harmony_metrics, combat_metrics])
    comparison_df['method'] = ['Harmony', 'ComBat']
    comparison_df.to_csv(os.path.join(comparison_dir, 'comparison_metrics.csv'), index=False)
    
    print("\n=== Integration Method Comparison ===")
    print(comparison_df[['method', 'n_clusters', 'overall_entropy', 
                         'avg_cluster_entropy', 'well_mixed_proportion', 'p_value']])
    
    # Determine which method performed better
    harmony_score = (harmony_metrics['avg_cluster_entropy'] + 
                    harmony_metrics['well_mixed_proportion']) / 2
    combat_score = (combat_metrics['avg_cluster_entropy'] + 
                   combat_metrics['well_mixed_proportion']) / 2
    
    if harmony_score > combat_score:
        print(f"\n→ Harmony shows better integration (score: {harmony_score:.3f} vs {combat_score:.3f})")
    else:
        print(f"\n→ ComBat shows better integration (score: {combat_score:.3f} vs {harmony_score:.3f})")

# Example usage:
if __name__ == "__main__":
    # Load your data
    adata_rna = sc.read("/dcl01/hongkai/data/data/hjiang/Test/gene_activity/rna_gene_id.h5ad")
    adata_activity = sc.read("/dcl01/hongkai/data/data/hjiang/Test/gene_activity/gene_activity_weighted_gpu.h5ad")
    
    # Run the enhanced pipeline with zero-aware quantile matching as default
    adata_harmony, adata_combat = combined_integration_analysis(
        adata_rna,
        adata_activity,
        unified_batch_key='sample',
        rna_cell_meta_path=None,
        activity_cell_meta_path=None,
        output_dir="/dcl01/hongkai/data/data/hjiang/Test/gene_activity/",
        align_distributions=True,  # Enable alignment (default True)
        alignment_method='zero_aware_quantile',  # New default method
        leiden_resolution=0.8,  # Adjust for more/fewer clusters
        run_combat=True,  # Set to True to also run ComBat
        verbose=True
    )