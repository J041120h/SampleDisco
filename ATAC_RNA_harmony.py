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
from scipy.stats import mannwhitneyu
import warnings

def combine_rna_and_activity_data(
    rna_h5ad_path,
    activity_h5ad_path,
    rna_cell_meta_path=None,
    activity_cell_meta_path=None,
    rna_sample_meta_path=None,
    activity_sample_meta_path=None,
    rna_sample_column='sample',
    activity_sample_column='sample',
    unified_sample_column='sample',
    rna_batch_key='batch',
    activity_batch_key='batch',
    unified_batch_key='batch',
    rna_prefix='RNA',
    activity_prefix='ATAC',
    verbose=True
):
    """
    Combine RNA and gene activity data into a single AnnData object.
    
    Parameters:
    -----------
    rna_h5ad_path : str
        Path to RNA H5AD file
    activity_h5ad_path : str
        Path to gene activity H5AD file
    rna_cell_meta_path : str, optional
        Path to RNA cell metadata CSV (if None, will extract from obs_names)
    activity_cell_meta_path : str, optional
        Path to gene activity cell metadata CSV (if None, will extract from obs_names)
    rna_sample_meta_path : str, optional
        Path to RNA sample metadata CSV (if None, no additional sample metadata)
    activity_sample_meta_path : str, optional
        Path to gene activity sample metadata CSV (if None, no additional sample metadata)
    rna_sample_column : str
        Column name for sample identification in RNA data
    activity_sample_column : str
        Column name for sample identification in gene activity data
    unified_sample_column : str
        Column name for sample identification in combined data
    rna_batch_key : str
        Column name for batch identification in RNA data
    activity_batch_key : str
        Column name for batch identification in gene activity data
    unified_batch_key : str
        Column name for batch identification in combined data
    rna_prefix : str
        Prefix to add to RNA cell barcodes
    activity_prefix : str
        Prefix to add to gene activity cell barcodes
    verbose : bool
        Print progress messages
    
    Returns:
    --------
    adata_combined : AnnData
        Combined AnnData object with both RNA and gene activity data
    """
    
    if verbose:
        print("=== Loading RNA and Gene Activity Data ===")
    
    # Load the raw count data
    adata_rna = sc.read_h5ad(rna_h5ad_path)
    adata_activity = sc.read_h5ad(activity_h5ad_path)
    
    if verbose:
        print(f"RNA data shape: {adata_rna.shape}")
        print(f"Gene activity data shape: {adata_activity.shape}")
    
    # Load cell metadata
    if verbose:
        print("=== Loading Cell Metadata ===")
    
    # Handle RNA cell metadata
    if rna_cell_meta_path is not None:
        rna_cell_meta = pd.read_csv(rna_cell_meta_path)
        # Ensure barcode column exists or create from index
        if 'barcode' not in rna_cell_meta.columns:
            rna_cell_meta['barcode'] = rna_cell_meta.index.astype(str)
    else:
        # Create minimal cell metadata from obs_names
        if verbose:
            print("No RNA cell metadata provided, creating from obs_names")
        rna_cell_meta = pd.DataFrame({
            'barcode': adata_rna.obs_names.astype(str)
        })
        # Extract sample from barcode if sample column not already in obs
        if rna_sample_column not in adata_rna.obs.columns:
            rna_cell_meta[rna_sample_column] = adata_rna.obs_names.str.split(':').str[0]
    
    # Handle gene activity cell metadata
    if activity_cell_meta_path is not None:
        activity_cell_meta = pd.read_csv(activity_cell_meta_path)
        # Ensure barcode column exists or create from index
        if 'barcode' not in activity_cell_meta.columns:
            activity_cell_meta['barcode'] = activity_cell_meta.index.astype(str)
    else:
        # Create minimal cell metadata from obs_names
        if verbose:
            print("No gene activity cell metadata provided, creating from obs_names")
        activity_cell_meta = pd.DataFrame({
            'barcode': adata_activity.obs_names.astype(str)
        })
        # Extract sample from barcode if sample column not already in obs
        if activity_sample_column not in adata_activity.obs.columns:
            activity_cell_meta[activity_sample_column] = adata_activity.obs_names.str.split(':').str[0]
    
    # Add data type column to distinguish RNA vs gene activity cells
    rna_cell_meta['data_type'] = 'RNA'
    activity_cell_meta['data_type'] = 'Gene_Activity'
    
    # Add prefixes to cell barcodes to make them unique
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
    
    # Load sample metadata (optional)
    if verbose:
        print("=== Loading Sample Metadata ===")
    
    if rna_sample_meta_path is not None:
        rna_sample_meta = pd.read_csv(rna_sample_meta_path)
        adata_rna.obs = adata_rna.obs.merge(rna_sample_meta, on=rna_sample_column, how='left')
        if verbose:
            print("RNA sample metadata loaded and merged")
    else:
        if verbose:
            print("No RNA sample metadata provided")
    
    if activity_sample_meta_path is not None:
        activity_sample_meta = pd.read_csv(activity_sample_meta_path)
        adata_activity.obs = adata_activity.obs.merge(activity_sample_meta, on=activity_sample_column, how='left')
        if verbose:
            print("Gene activity sample metadata loaded and merged")
    else:
        if verbose:
            print("No gene activity sample metadata provided")
    
    # Standardize column names to unified names
    if verbose:
        print("=== Standardizing Column Names ===")
    
    # Ensure required columns exist (create default values if needed)
    # Handle sample column
    if rna_sample_column not in adata_rna.obs.columns:
        if verbose:
            print(f"RNA sample column '{rna_sample_column}' not found, creating from data_type")
        adata_rna.obs[rna_sample_column] = 'RNA_sample'
    
    if activity_sample_column not in adata_activity.obs.columns:
        if verbose:
            print(f"Gene activity sample column '{activity_sample_column}' not found, creating from data_type")
        adata_activity.obs[activity_sample_column] = 'ATAC_sample'
    
    # Handle batch column (create default if not provided)
    if rna_batch_key not in adata_rna.obs.columns:
        if verbose:
            print(f"RNA batch column '{rna_batch_key}' not found, creating default")
        adata_rna.obs[rna_batch_key] = 'RNA_batch'
    
    if activity_batch_key not in adata_activity.obs.columns:
        if verbose:
            print(f"Gene activity batch column '{activity_batch_key}' not found, creating default")
        adata_activity.obs[activity_batch_key] = 'ATAC_batch'
    
    # Rename sample columns to unified name
    if rna_sample_column != unified_sample_column:
        if unified_sample_column in adata_rna.obs.columns and unified_sample_column != rna_sample_column:
            adata_rna.obs.drop(columns=[unified_sample_column], inplace=True)
        adata_rna.obs[unified_sample_column] = adata_rna.obs[rna_sample_column]
    
    if activity_sample_column != unified_sample_column:
        if unified_sample_column in adata_activity.obs.columns and unified_sample_column != activity_sample_column:
            adata_activity.obs.drop(columns=[unified_sample_column], inplace=True)
        adata_activity.obs[unified_sample_column] = adata_activity.obs[activity_sample_column]
    
    # Rename batch columns to unified name
    if rna_batch_key != unified_batch_key:
        if unified_batch_key in adata_rna.obs.columns and unified_batch_key != rna_batch_key:
            adata_rna.obs.drop(columns=[unified_batch_key], inplace=True)
        adata_rna.obs[unified_batch_key] = adata_rna.obs[rna_batch_key]
    
    if activity_batch_key != unified_batch_key:
        if unified_batch_key in adata_activity.obs.columns and unified_batch_key != activity_batch_key:
            adata_activity.obs.drop(columns=[unified_batch_key], inplace=True)
        adata_activity.obs[unified_batch_key] = adata_activity.obs[activity_batch_key]
    
    if verbose:
        print("=== Combining Datasets ===")
        print(f"RNA data shape: {adata_rna.shape}")
        print(f"Gene activity data shape: {adata_activity.shape}")
    
    # Combine the datasets using scanpy concat with outer join
    # This automatically handles missing genes by filling with zeros
    adata_combined = sc.concat([adata_rna, adata_activity], axis=0, join='outer')
    
    if verbose:
        print(f"Combined data shape: {adata_combined.shape}")
        print(f"RNA cells: {sum(adata_combined.obs['data_type'] == 'RNA')}")
        print(f"Gene activity cells: {sum(adata_combined.obs['data_type'] == 'Gene_Activity')}")
        print(f"Total unique genes: {adata_combined.n_vars}")
    
    # Return the combined AnnData object
    return adata_combined


def clean_obs_for_writing(adata):
    """
    Clean obs dataframe to ensure it can be written to H5AD format.
    Converts problematic data types and handles missing values.
    """
    import pandas as pd
    import numpy as np
    
    # Create a copy of obs to avoid modifying the original
    obs_clean = adata.obs.copy()
    
    for col in obs_clean.columns:
        # Check if column has object dtype
        if obs_clean[col].dtype == 'object':
            # Replace NaN values with empty string
            obs_clean[col] = obs_clean[col].fillna('')
            
            # Convert all values to string
            obs_clean[col] = obs_clean[col].astype(str)
            
            # Replace 'nan' string with empty string
            obs_clean[col] = obs_clean[col].replace('nan', '')
        
        # Handle boolean columns
        elif obs_clean[col].dtype == 'bool':
            # Convert to string representation
            obs_clean[col] = obs_clean[col].astype(str)
        
        # Handle any other problematic dtypes
        elif obs_clean[col].dtype.name.startswith('category'):
            # Convert categorical to string
            obs_clean[col] = obs_clean[col].astype(str)
    
    # Update the adata obs
    adata.obs = obs_clean
    
    return adata


def visualize_rna_atac_integration(adata_combined, output_dir, verbose=True):
    """
    Create comprehensive visualizations to assess RNA and ATAC integration quality.
    
    Parameters:
    -----------
    adata_combined : AnnData
        Combined and processed AnnData object
    output_dir : str
        Directory to save plots
    verbose : bool
        Print progress messages
    """
    
    if verbose:
        print("=== Creating Integration Quality Visualizations ===")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'integration_plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # After integration (Harmony)
    sc.pl.umap(adata_combined, color='data_type',
              show=False,
              title='Harmony')
    
    # Clustering
    sc.pl.umap(adata_combined, color='leiden',show=False,
              title='Leiden Clusters', legend_loc='right margin')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'umap_integration_overview.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mixing metrics visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Integration Quality Metrics', fontsize=16, fontweight='bold')
    
    # Calculate mixing metrics
    def calculate_mixing_metrics(adata, use_rep='X_pca_harmony'):
        """Calculate various mixing metrics"""
        embedding = adata.obsm[use_rep]
        data_types = adata.obs['data_type'].values
        samples = adata.obs['sample'].values
        
        # k-NN mixing score
        def knn_mixing_score(embedding, labels, k=50):
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(embedding)
            _, indices = nbrs.kneighbors(embedding)
            
            mixing_scores = []
            for i, cell_label in enumerate(labels):
                neighbors = labels[indices[i][1:]]  # Exclude self
                mixing_score = np.mean(neighbors != cell_label)
                mixing_scores.append(mixing_score)
            return np.array(mixing_scores)
        
        # Local mixing entropy
        def local_mixing_entropy(embedding, labels, k=50):
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(embedding)
            _, indices = nbrs.kneighbors(embedding)
            
            entropies = []
            for i in range(len(labels)):
                neighbors = labels[indices[i][1:]]  # Exclude self
                unique, counts = np.unique(neighbors, return_counts=True)
                probs = counts / len(neighbors)
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                entropies.append(entropy)
            return np.array(entropies)
        
        # Calculate metrics
        data_type_mixing = knn_mixing_score(embedding, data_types)
        sample_mixing = knn_mixing_score(embedding, samples)
        data_type_entropy = local_mixing_entropy(embedding, data_types)
        
        return {
            'data_type_mixing': data_type_mixing,
            'sample_mixing': sample_mixing,
            'data_type_entropy': data_type_entropy
        }
    
    # Before and after metrics
    if 'X_pca' in adata_combined.obsm:
        metrics_before = calculate_mixing_metrics(adata_combined, 'X_pca')
    else:
        metrics_before = None
    metrics_after = calculate_mixing_metrics(adata_combined, 'X_pca_harmony')
    
    # Plot data type mixing scores
    if metrics_before:
        axes[0,0].hist(metrics_before['data_type_mixing'], bins=50, alpha=0.6, 
                      label='Before (PCA)', color='lightcoral')
    axes[0,0].hist(metrics_after['data_type_mixing'], bins=50, alpha=0.6,
                  label='After (Harmony)', color='skyblue')
    axes[0,0].set_xlabel('Data Type Mixing Score')
    axes[0,0].set_ylabel('Number of Cells')
    axes[0,0].set_title('Data Type Mixing\n(Higher = Better Mixing)')
    axes[0,0].legend()
    
    # Plot sample mixing scores
    if metrics_before:
        axes[0,1].hist(metrics_before['sample_mixing'], bins=50, alpha=0.6,
                      label='Before (PCA)', color='lightcoral')
    axes[0,1].hist(metrics_after['sample_mixing'], bins=50, alpha=0.6,
                  label='After (Harmony)', color='skyblue')
    axes[0,1].set_xlabel('Sample Mixing Score')
    axes[0,1].set_ylabel('Number of Cells')
    axes[0,1].set_title('Sample Mixing\n(Higher = Better Mixing)')
    axes[0,1].legend()
    
    # Plot entropy
    axes[0,2].hist(metrics_after['data_type_entropy'], bins=50, alpha=0.7, color='green')
    axes[0,2].set_xlabel('Local Mixing Entropy')
    axes[0,2].set_ylabel('Number of Cells')
    axes[0,2].set_title('Local Data Type Entropy\n(Higher = Better Mixing)')
    
    # 3. Distance-based analysis
    def plot_distance_analysis(adata, ax_within, ax_between, use_rep='X_pca_harmony'):
        """Plot within vs between group distances"""
        embedding = adata.obsm[use_rep]
        data_types = adata.obs['data_type'].values
        
        # Sample cells for faster computation
        n_sample = min(2000, len(adata))
        idx = np.random.choice(len(adata), n_sample, replace=False)
        embedding_sample = embedding[idx]
        data_types_sample = data_types[idx]
        
        # Calculate pairwise distances
        distances = pdist(embedding_sample, metric='euclidean')
        dist_matrix = squareform(distances)
        
        within_rna = []
        within_atac = []
        between = []
        
        for i in range(len(data_types_sample)):
            for j in range(i+1, len(data_types_sample)):
                dist = dist_matrix[i, j]
                if data_types_sample[i] == data_types_sample[j]:
                    if data_types_sample[i] == 'RNA':
                        within_rna.append(dist)
                    else:
                        within_atac.append(dist)
                else:
                    between.append(dist)
        
        # Plot distributions
        ax_within.hist(within_rna, bins=50, alpha=0.6, label='Within RNA', density=True)
        ax_within.hist(within_atac, bins=50, alpha=0.6, label='Within ATAC', density=True)
        ax_within.set_xlabel('Distance')
        ax_within.set_ylabel('Density')
        ax_within.set_title('Within Data Type Distances')
        ax_within.legend()
        
        ax_between.hist(between, bins=50, alpha=0.7, color='purple', density=True)
        ax_between.set_xlabel('Distance')
        ax_between.set_ylabel('Density')
        ax_between.set_title('Between Data Type Distances')
        
        return np.mean(within_rna + within_atac), np.mean(between)
    
    # Distance analysis
    if 'X_pca' in adata_combined.obsm:
        within_before, between_before = plot_distance_analysis(adata_combined, axes[1,0], axes[1,1], 'X_pca')
    within_after, between_after = plot_distance_analysis(adata_combined, axes[1,0], axes[1,1], 'X_pca_harmony')
    
    # Integration score
    integration_score = between_after / (within_after + 1e-10)
    axes[1,2].bar(['Integration\nScore'], [integration_score], color='orange')
    axes[1,2].set_ylabel('Score')
    axes[1,2].set_title(f'Integration Score\n(Lower = Better)\nScore: {integration_score:.3f}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mixing_metrics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Cluster composition analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cluster Composition Analysis', fontsize=16, fontweight='bold')
    
    # Create cluster composition matrix
    cluster_composition = pd.crosstab(adata_combined.obs['leiden'], 
                                     adata_combined.obs['data_type'], 
                                     normalize='index')
    
    # Heatmap of cluster composition
    sns.heatmap(cluster_composition, annot=True, cmap='RdYlBu_r', 
                ax=axes[0], cbar_kws={'label': 'Proportion'})
    axes[0].set_title('Cluster Composition\n(Row-normalized)')
    axes[0].set_xlabel('Data Type')
    axes[0].set_ylabel('Leiden Cluster')
    
    # Stacked bar chart
    cluster_composition.plot(kind='bar', stacked=True, ax=axes[1], 
                           color=['lightcoral', 'skyblue'])
    axes[1].set_title('Cluster Composition\n(Stacked)')
    axes[1].set_xlabel('Leiden Cluster')
    axes[1].set_ylabel('Proportion')
    axes[1].legend(title='Data Type')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    
    # Balance score per cluster
    balance_scores = []
    for cluster in cluster_composition.index:
        rna_prop = cluster_composition.loc[cluster, 'RNA']
        atac_prop = cluster_composition.loc[cluster, 'Gene_Activity']
        balance = 1 - abs(rna_prop - atac_prop)  # 1 = perfect balance, 0 = completely imbalanced
        balance_scores.append(balance)
    
    axes[2].bar(range(len(balance_scores)), balance_scores, color='green', alpha=0.7)
    axes[2].set_xlabel('Leiden Cluster')
    axes[2].set_ylabel('Balance Score')
    axes[2].set_title('Cluster Balance Score\n(1.0 = Perfect Balance)')
    axes[2].set_xticks(range(len(balance_scores)))
    axes[2].set_xticklabels(cluster_composition.index, rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cluster_composition.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Quality control comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Quality Control Metrics by Data Type', fontsize=16, fontweight='bold')
    
    # QC metrics to compare
    qc_metrics = ['total_counts', 'n_genes_by_counts', 'pct_counts_mt']
    
    for i, metric in enumerate(qc_metrics):
        if metric in adata_combined.obs.columns:
            # Box plot
            data_for_plot = [adata_combined.obs[adata_combined.obs['data_type'] == 'RNA'][metric].values,
                           adata_combined.obs[adata_combined.obs['data_type'] == 'Gene_Activity'][metric].values]
            
            axes[0, i].boxplot(data_for_plot, labels=['RNA', 'Gene Activity'])
            axes[0, i].set_title(f'{metric}')
            axes[0, i].set_ylabel(metric.replace('_', ' ').title())
            
            # Violin plot
            sns.violinplot(data=adata_combined.obs, x='data_type', y=metric, ax=axes[1, i])
            axes[1, i].set_title(f'{metric} Distribution')
            axes[1, i].set_xlabel('Data Type')
            plt.setp(axes[1, i].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'qc_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Summary statistics
    if verbose:
        print("\n=== Integration Quality Summary ===")
        
        # Overall statistics
        total_cells = len(adata_combined)
        rna_cells = sum(adata_combined.obs['data_type'] == 'RNA')
        atac_cells = sum(adata_combined.obs['data_type'] == 'Gene_Activity')
        
        print(f"Total cells: {total_cells}")
        print(f"RNA cells: {rna_cells} ({rna_cells/total_cells*100:.1f}%)")
        print(f"ATAC cells: {atac_cells} ({atac_cells/total_cells*100:.1f}%)")
        print(f"Number of clusters: {len(adata_combined.obs['leiden'].unique())}")
        print(f"Number of samples: {len(adata_combined.obs['sample'].unique())}")
        
        # Mixing statistics
        avg_data_mixing = np.mean(metrics_after['data_type_mixing'])
        avg_sample_mixing = np.mean(metrics_after['sample_mixing'])
        avg_entropy = np.mean(metrics_after['data_type_entropy'])
        avg_balance = np.mean(balance_scores)
        
        print(f"\nMixing Quality Metrics:")
        print(f"Average data type mixing score: {avg_data_mixing:.3f}")
        print(f"Average sample mixing score: {avg_sample_mixing:.3f}")
        print(f"Average local entropy: {avg_entropy:.3f}")
        print(f"Average cluster balance: {avg_balance:.3f}")
        print(f"Integration distance ratio: {integration_score:.3f}")
        
        # Save summary to file
        summary_stats = {
            'total_cells': total_cells,
            'rna_cells': rna_cells,
            'atac_cells': atac_cells,
            'n_clusters': len(adata_combined.obs['leiden'].unique()),
            'n_samples': len(adata_combined.obs['sample'].unique()),
            'avg_data_type_mixing': avg_data_mixing,
            'avg_sample_mixing': avg_sample_mixing,
            'avg_entropy': avg_entropy,
            'avg_cluster_balance': avg_balance,
            'integration_score': integration_score
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(os.path.join(plots_dir, 'integration_summary.csv'), index=False)
        
        print(f"\nAll plots saved to: {plots_dir}")
        print(f"Summary statistics saved to: {os.path.join(plots_dir, 'integration_summary.csv')}")
    
    return summary_stats


def combined_harmony_analysis(
    rna_h5ad_path,
    activity_h5ad_path,
    rna_cell_meta_path=None,
    activity_cell_meta_path=None,
    rna_sample_meta_path=None,
    activity_sample_meta_path=None,
    output_dir=None,
    rna_sample_column='sample',
    activity_sample_column='sample',
    unified_sample_column='sample',
    rna_batch_key='batch',
    activity_batch_key='batch',
    unified_batch_key='batch',
    cell_column='cell_type',
    rna_prefix='RNA',
    activity_prefix='ATAC',
    markers=None,
    cluster_resolution=0.8,
    num_PCs=20,
    num_harmony=30,
    num_features=2000,
    min_cells=500,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
    method='average',
    metric='euclidean',
    distance_mode='centroid',
    vars_to_regress=[],
    verbose=True
):
    """
    Complete pipeline for combining RNA and gene activity data and performing Harmony integration.
    
    Parameters:
    -----------
    rna_h5ad_path : str
        Path to RNA H5AD file
    activity_h5ad_path : str
        Path to gene activity H5AD file
    rna_cell_meta_path : str, optional
        Path to RNA cell metadata CSV (if None, will extract from obs_names)
    activity_cell_meta_path : str, optional
        Path to gene activity cell metadata CSV (if None, will extract from obs_names)
    rna_sample_meta_path : str, optional
        Path to RNA sample metadata CSV (if None, no additional sample metadata)
    activity_sample_meta_path : str, optional
        Path to gene activity sample metadata CSV (if None, no additional sample metadata)
    output_dir : str, optional
        Output directory for results (if None, will not save files)
    rna_sample_column : str
        Column name for sample identification in RNA data
    activity_sample_column : str
        Column name for sample identification in gene activity data
    unified_sample_column : str
        Column name for sample identification in combined data
    rna_batch_key : str
        Column name for batch identification in RNA data
    activity_batch_key : str
        Column name for batch identification in gene activity data
    unified_batch_key : str
        Column name for batch identification in combined data
    cell_column : str
        Column name for cell type
    rna_prefix : str
        Prefix for RNA cell barcodes
    activity_prefix : str
        Prefix for gene activity cell barcodes
    Other parameters: same as original harmony function
    
    Returns:
    --------
    adata_final : AnnData
        Final processed AnnData object with cell type annotations
    integration_stats : dict (optional)
        Integration quality statistics (only if output_dir is provided)
    """
    
    start_time = time.time()
    
    # Create output directories (optional)
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            if verbose:
                print("Created output directory")
        
        output_dir = os.path.join(output_dir, 'combined_harmony')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            if verbose:
                print("Created combined_harmony subdirectory")
    
    # Step 1: Combine RNA and gene activity data
    adata_combined = combine_rna_and_activity_data(
        rna_h5ad_path=rna_h5ad_path,
        activity_h5ad_path=activity_h5ad_path,
        rna_cell_meta_path=rna_cell_meta_path,
        activity_cell_meta_path=activity_cell_meta_path,
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
    
    # Add 'data_type' to vars_to_regress if not already present
    vars_to_regress_for_harmony = vars_to_regress.copy()
    if unified_sample_column not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append(unified_sample_column)
    if 'data_type' not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append('data_type')
    
    # Error checking for required columns
    all_required_columns = vars_to_regress_for_harmony + [unified_batch_key]
    missing_vars = [col for col in all_required_columns if col not in adata_combined.obs.columns]
    if missing_vars:
        raise KeyError(f"The following variables are missing from adata_combined.obs: {missing_vars}")
    
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
    if verbose:
        print("Sample counts before filtering:")
        print(cell_counts_per_sample.sort_values(ascending=False))
    
    samples_to_keep = cell_counts_per_sample[cell_counts_per_sample >= min_cells].index
    adata_combined = adata_combined[adata_combined.obs[unified_sample_column].isin(samples_to_keep)].copy()
    
    if verbose:
        print(f"Samples retained: {list(samples_to_keep)}")
        print("Sample counts after filtering:")
        print(adata_combined.obs[unified_sample_column].value_counts().sort_values(ascending=False))
    
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
            sc.pp.scrublet(adata_combined)  # Removed batch_key parameter
        adata_combined = adata_combined[~adata_combined.obs['predicted_doublet']].copy()
        if verbose:
            print(f"After doublet removal -- Cells: {adata_combined.n_obs}")
    
    # Save raw data
    adata_combined.raw = adata_combined.copy()
    
    # Step 2: Run the clustering pipeline (modified anndata_cluster)
    if verbose:
        print('=== Processing Combined Data for Clustering ===')
    
    # Normalization and log transformation
    sc.pp.normalize_total(adata_combined, target_sum=1e4)
    sc.pp.log1p(adata_combined)
    
    # HVG selection
    sc.pp.highly_variable_genes(
        adata_combined,
        n_top_genes=num_features,
        flavor='seurat_v3',
        batch_key=unified_sample_column
    )
    adata_combined = adata_combined[:, adata_combined.var['highly_variable']].copy()
    
    # PCA
    sc.tl.pca(adata_combined, n_comps=num_PCs, svd_solver='arpack')
    
    if verbose:
        print('=== Running Harmony Integration ===')
        print(f'Variables to regress: {", ".join(vars_to_regress_for_harmony)}')
        print(f'Cluster resolution: {cluster_resolution}')
    
    # Harmony integration
    Z = harmonize(
        adata_combined.obsm['X_pca'],
        adata_combined.obs,
        batch_key=vars_to_regress_for_harmony,
        max_iter_harmony=num_harmony,
        use_gpu=True
    )
    adata_combined.obsm['X_pca_harmony'] = Z
    
    # Clustering and UMAP
    if verbose:
        print('=== Clustering and Visualization ===')
    
    sc.pp.neighbors(adata_combined, use_rep='X_pca_harmony', n_pcs=num_PCs, n_neighbors=15, metric='cosine')
    sc.tl.leiden(adata_combined, resolution=cluster_resolution, key_added='leiden')
    sc.tl.umap(adata_combined, min_dist=0.3, spread=1.0)
    
    # Marker gene analysis
    if verbose:
        print('=== Finding Marker Genes ===')
    sc.tl.rank_genes_groups(adata_combined, 'leiden', method='wilcoxon')
    
    # Save final result
    if output_dir is not None:
        # Clean obs dataframe before saving
        adata_combined = clean_obs_for_writing(adata_combined)
        
        try:
            sc.write(os.path.join(output_dir, 'adata_combined_final.h5ad'), adata_combined)
            if verbose:
                print(f"Results saved to: {output_dir}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not save H5AD file due to: {e}")
                print("Attempting alternative save method...")
            
            # Alternative: save as pickle if H5AD fails
            import pickle
            pickle_path = os.path.join(output_dir, 'adata_combined_final.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(adata_combined, f)
            if verbose:
                print(f"Results saved as pickle to: {pickle_path}")
    
    # Create integration quality visualizations (optional)
    integration_stats = None
    if output_dir is not None:
        if verbose:
            print('=== Creating Integration Quality Visualizations ===')
        integration_stats = visualize_rna_atac_integration(adata_combined, output_dir, verbose)
    
    # Print summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if verbose:
        print(f"\n=== Analysis Complete ===")
        print(f"Execution time: {elapsed_time:.2f} seconds")
        print(f"Final data shape: {adata_combined.shape}")
        print(f"Number of clusters: {len(adata_combined.obs['leiden'].unique())}")
        print(f"Data types: {adata_combined.obs['data_type'].value_counts().to_dict()}")
    
    if integration_stats is not None:
        return adata_combined, integration_stats
    else:
        return adata_combined


# Example usage:
if __name__ == "__main__":
    # Example parameters
    adata_final, integration_stats = combined_harmony_analysis(
        rna_h5ad_path="/Users/harry/Desktop/GenoDistance/Data/count_data.h5ad",
        activity_h5ad_path="/Users/harry/Desktop/GenoDistance/result/gene_activity/gene_activity_weighted.h5ad",
        rna_cell_meta_path=None,
        activity_cell_meta_path=None,
        rna_sample_meta_path="/Users/harry/Desktop/GenoDistance/Data/sample_data.csv",
        activity_sample_meta_path="/Users/harry/Desktop/GenoDistance/Data/ATAC_Metadata.csv",
        output_dir="/Users/harry/Desktop/GenoDistance/result",
        rna_sample_column='sample',  # Different sample column names
        activity_sample_column='sample',
        unified_sample_column='sample',
        rna_batch_key='batch',  # Different batch column names
        activity_batch_key='batch',
        unified_batch_key='batch',
        cluster_resolution=0.8,
        num_PCs=20,
        num_harmony=30,
        num_features=2000,
        verbose=True
    )