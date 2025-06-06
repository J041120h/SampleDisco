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

def visualize_rna_atac_integration(adata_combined, output_dir, cell_type_column='cell_type', 
                                   quantitative_measures=True, verbose=True):
    """
    Create comprehensive visualizations to assess RNA and ATAC integration quality.
    
    Parameters:
    -----------
    adata_combined : AnnData
        Combined and processed AnnData object
    output_dir : str
        Directory to save plots
    cell_type_column : str, default 'cell_type'
        Column name in adata_combined.obs containing cell type annotations
    quantitative_measures : bool
        Whether to compute quantitative mixing metrics (k-NN mixing, entropy, distance ratios)
    verbose : bool
        Print progress messages
    
    Returns:
    --------
    summary_stats : dict
        Summary statistics from the analysis
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
    
    # Check if cell type column exists
    has_cell_types = cell_type_column in adata_combined.obs.columns
    if not has_cell_types and verbose:
        print(f"Warning: Cell type column '{cell_type_column}' not found in adata.obs")
    
    # 1. Basic UMAP visualization
    n_plots = 3 if has_cell_types else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    # After integration (Harmony)
    sc.pl.umap(adata_combined, color='data_type', ax=axes[0],
              show=False, title='Data Type Distribution')
    
    # Cell type visualization (if available)
    if has_cell_types:
        sc.pl.umap(adata_combined, color=cell_type_column, ax=axes[2], show=False,
                  title='Cell Types', legend_loc='right margin')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'umap_integration_overview.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Initialize summary stats
    summary_stats = {
        'total_cells': len(adata_combined),
        'rna_cells': sum(adata_combined.obs['data_type'] == 'RNA'),
        'atac_cells': sum(adata_combined.obs['data_type'] == 'Gene_Activity'),
        'n_samples': len(adata_combined.obs['sample'].unique()) if 'sample' in adata_combined.obs.columns else 0,
        'has_cell_types': has_cell_types,
        'n_cell_types': len(adata_combined.obs[cell_type_column].unique()) if has_cell_types else 0
    }
    
    # 2. Quantitative mixing metrics (optional)
    if quantitative_measures:
        if verbose:
            print("Computing quantitative mixing metrics...")
            
        # Determine number of subplots based on available data
        n_rows = 3 if has_cell_types else 2
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
        if n_rows == 2:
            axes = np.vstack([axes, [None, None, None]])  # Add empty row for consistency
        
        fig.suptitle('Integration Quality Metrics', fontsize=16, fontweight='bold')
        
        # Calculate mixing metrics
        def calculate_mixing_metrics(adata, use_rep='X_pca_harmony'):
            """Calculate various mixing metrics"""
            embedding = adata.obsm[use_rep]
            data_types = adata.obs['data_type'].values
            samples = adata.obs['sample'].values if 'sample' in adata.obs.columns else np.array(['sample'] * len(data_types))
            cell_types = adata.obs[cell_type_column].values if has_cell_types else np.array(['cell_type'] * len(data_types))
            
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
            
            metrics = {
                'data_type_mixing': data_type_mixing,
                'sample_mixing': sample_mixing,
                'data_type_entropy': data_type_entropy
            }
            
            # Add cell type metrics if available
            if has_cell_types:
                cell_type_mixing = knn_mixing_score(embedding, cell_types)
                cell_type_entropy = local_mixing_entropy(embedding, cell_types)
                metrics['cell_type_mixing'] = cell_type_mixing
                metrics['cell_type_entropy'] = cell_type_entropy
            
            return metrics
        
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
        
        # Cell type mixing plots (if available)
        if has_cell_types:
            # Cell type mixing score
            if metrics_before and 'cell_type_mixing' in metrics_before:
                axes[2,0].hist(metrics_before['cell_type_mixing'], bins=50, alpha=0.6,
                              label='Before (PCA)', color='lightcoral')
            axes[2,0].hist(metrics_after['cell_type_mixing'], bins=50, alpha=0.6,
                          label='After (Harmony)', color='skyblue')
            axes[2,0].set_xlabel('Cell Type Mixing Score')
            axes[2,0].set_ylabel('Number of Cells')
            axes[2,0].set_title('Cell Type Mixing\n(Higher = Better Mixing)')
            axes[2,0].legend()
            
            # Cell type entropy
            axes[2,1].hist(metrics_after['cell_type_entropy'], bins=50, alpha=0.7, color='orange')
            axes[2,1].set_xlabel('Local Cell Type Entropy')
            axes[2,1].set_ylabel('Number of Cells')
            axes[2,1].set_title('Local Cell Type Entropy\n(Higher = Better Mixing)')
            
            # Cell type composition by data type
            cell_type_data_crosstab = pd.crosstab(adata_combined.obs[cell_type_column], 
                                                  adata_combined.obs['data_type'])
            cell_type_data_crosstab_norm = cell_type_data_crosstab.div(cell_type_data_crosstab.sum(axis=1), axis=0)
            
            sns.heatmap(cell_type_data_crosstab_norm, annot=True, cmap='RdYlBu_r', 
                       ax=axes[2,2], cbar_kws={'label': 'Proportion'})
            axes[2,2].set_title('Cell Type vs Data Type\n(Row-normalized)')
            axes[2,2].set_xlabel('Data Type')
            axes[2,2].set_ylabel('Cell Type')
        
        # Distance-based analysis
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
        
        # Remove empty subplots if cell types not available
        if not has_cell_types:
            for ax in axes[2]:
                if ax is not None:
                    ax.remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'mixing_metrics.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add quantitative metrics to summary
        metrics_summary = {
            'avg_data_type_mixing': np.mean(metrics_after['data_type_mixing']),
            'avg_sample_mixing': np.mean(metrics_after['sample_mixing']),
            'avg_entropy': np.mean(metrics_after['data_type_entropy']),
            'integration_score': integration_score
        }
        
        if has_cell_types:
            metrics_summary.update({
                'avg_cell_type_mixing': np.mean(metrics_after['cell_type_mixing']),
                'avg_cell_type_entropy': np.mean(metrics_after['cell_type_entropy'])
            })
        
        summary_stats.update(metrics_summary)
    
    # 3. Cluster composition analysis (only if clustering exists)
    if 'leiden' in adata_combined.obs.columns:
        n_plots = 4 if has_cell_types else 3
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
        if n_plots == 1:
            axes = [axes]
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
        
        # Cell type composition by cluster (if available)
        if has_cell_types:
            cluster_celltype_composition = pd.crosstab(adata_combined.obs['leiden'], 
                                                      adata_combined.obs[cell_type_column])
            cluster_celltype_composition_norm = cluster_celltype_composition.div(
                cluster_celltype_composition.sum(axis=1), axis=0)
            
            sns.heatmap(cluster_celltype_composition_norm, annot=True, cmap='viridis', 
                       ax=axes[3], cbar_kws={'label': 'Proportion'})
            axes[3].set_title('Cluster vs Cell Type\n(Row-normalized)')
            axes[3].set_xlabel('Cell Type')
            axes[3].set_ylabel('Leiden Cluster')
            plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'cluster_composition.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add cluster info to summary
        summary_stats['n_clusters'] = len(adata_combined.obs['leiden'].unique())
        summary_stats['avg_cluster_balance'] = np.mean(balance_scores)
    
    # 4. Quality control comparison
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
    
    # 5. Cell type specific analysis (if available)
    if has_cell_types:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cell Type Analysis', fontsize=16, fontweight='bold')
        
        # Cell type distribution by data type
        celltype_data_counts = pd.crosstab(adata_combined.obs[cell_type_column], 
                                          adata_combined.obs['data_type'])
        
        # Stacked bar chart
        celltype_data_counts.plot(kind='bar', stacked=True, ax=axes[0,0], 
                                 color=['lightcoral', 'skyblue'])
        axes[0,0].set_title('Cell Type Distribution by Data Type')
        axes[0,0].set_xlabel('Cell Type')
        axes[0,0].set_ylabel('Number of Cells')
        axes[0,0].legend(title='Data Type')
        plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45)
        
        # Proportion plot
        celltype_data_prop = celltype_data_counts.div(celltype_data_counts.sum(axis=1), axis=0)
        celltype_data_prop.plot(kind='bar', ax=axes[0,1], 
                               color=['lightcoral', 'skyblue'])
        axes[0,1].set_title('Cell Type Composition (Proportions)')
        axes[0,1].set_xlabel('Cell Type')
        axes[0,1].set_ylabel('Proportion')
        axes[0,1].legend(title='Data Type')
        plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45)
        
        # Cell type balance scores
        celltype_balance = []
        for celltype in celltype_data_counts.index:
            if celltype_data_counts.loc[celltype].sum() > 10:  # Only for cell types with >10 cells
                rna_prop = celltype_data_prop.loc[celltype, 'RNA']
                atac_prop = celltype_data_prop.loc[celltype, 'Gene_Activity']
                balance = 1 - abs(rna_prop - atac_prop)
                celltype_balance.append((celltype, balance))
        
        if celltype_balance:
            celltypes, balances = zip(*celltype_balance)
            axes[1,0].bar(range(len(balances)), balances, color='purple', alpha=0.7)
            axes[1,0].set_xlabel('Cell Type')
            axes[1,0].set_ylabel('Balance Score')
            axes[1,0].set_title('Cell Type Balance Score\n(1.0 = Perfect Balance)')
            axes[1,0].set_xticks(range(len(balances)))
            axes[1,0].set_xticklabels(celltypes, rotation=45)
        
        # Cell type size distribution
        celltype_sizes = adata_combined.obs[cell_type_column].value_counts()
        axes[1,1].bar(range(len(celltype_sizes)), celltype_sizes.values, 
                     color='teal', alpha=0.7)
        axes[1,1].set_xlabel('Cell Type (Ranked by Size)')
        axes[1,1].set_ylabel('Number of Cells')
        axes[1,1].set_title('Cell Type Size Distribution')
        axes[1,1].set_xticks(range(len(celltype_sizes)))
        axes[1,1].set_xticklabels(celltype_sizes.index, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'celltype_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add cell type balance to summary
        if celltype_balance:
            summary_stats['avg_celltype_balance'] = np.mean([b for _, b in celltype_balance])
    
    # Print summary
    if verbose:
        print("\n=== Integration Quality Summary ===")
        
        # Overall statistics
        total_cells = summary_stats['total_cells']
        rna_cells = summary_stats['rna_cells']
        atac_cells = summary_stats['atac_cells']
        
        print(f"Total cells: {total_cells}")
        print(f"RNA cells: {rna_cells} ({rna_cells/total_cells*100:.1f}%)")
        print(f"ATAC cells: {atac_cells} ({atac_cells/total_cells*100:.1f}%)")
        
        if 'n_clusters' in summary_stats:
            print(f"Number of clusters: {summary_stats['n_clusters']}")
        if 'n_samples' in summary_stats:
            print(f"Number of samples: {summary_stats['n_samples']}")
        if has_cell_types:
            print(f"Number of cell types: {summary_stats['n_cell_types']}")
        
        # Mixing statistics (if calculated)
        if quantitative_measures:
            print(f"\nMixing Quality Metrics:")
            print(f"Average data type mixing score: {summary_stats['avg_data_type_mixing']:.3f}")
            print(f"Average sample mixing score: {summary_stats['avg_sample_mixing']:.3f}")
            print(f"Average local entropy: {summary_stats['avg_entropy']:.3f}")
            if has_cell_types:
                print(f"Average cell type mixing score: {summary_stats['avg_cell_type_mixing']:.3f}")
                print(f"Average cell type entropy: {summary_stats['avg_cell_type_entropy']:.3f}")
            if 'avg_cluster_balance' in summary_stats:
                print(f"Average cluster balance: {summary_stats['avg_cluster_balance']:.3f}")
            if 'avg_celltype_balance' in summary_stats:
                print(f"Average cell type balance: {summary_stats['avg_celltype_balance']:.3f}")
            print(f"Integration distance ratio: {summary_stats['integration_score']:.3f}")
        
        # Save summary to file
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(os.path.join(plots_dir, 'integration_summary.csv'), index=False)
        
        print(f"\nAll plots saved to: {plots_dir}")
        print(f"Summary statistics saved to: {os.path.join(plots_dir, 'integration_summary.csv')}")
    
    return summary_stats