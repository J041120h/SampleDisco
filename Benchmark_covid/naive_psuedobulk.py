import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import time
import contextlib
import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pseudobulk_samples(adata, sample_column='sample', min_cells_per_sample=10):
    """
    Pseudobulk the data by summing counts per sample.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with cells x genes
    sample_column : str
        Column in adata.obs containing sample identifiers
    min_cells_per_sample : int
        Minimum number of cells required per sample
        
    Returns:
    --------
    pseudobulk_df : DataFrame
        Pseudobulked expression matrix (samples x genes)
    sample_metadata : DataFrame
        Metadata for each sample
    """
    # Get unique samples
    samples = adata.obs[sample_column].unique()
    
    # Initialize pseudobulk matrix
    pseudobulk_data = []
    sample_info = []
    
    for sample in samples:
        # Get cells for this sample
        sample_mask = adata.obs[sample_column] == sample
        sample_cells = adata[sample_mask]
        
        # Skip if too few cells
        if sample_cells.n_obs < min_cells_per_sample:
            print(f"Skipping sample {sample}: only {sample_cells.n_obs} cells")
            continue
            
        # Sum expression across cells (pseudobulk)
        if hasattr(sample_cells.X, 'toarray'):
            pseudobulk_expr = sample_cells.X.toarray().sum(axis=0)
        else:
            pseudobulk_expr = sample_cells.X.sum(axis=0)
        
        pseudobulk_data.append(pseudobulk_expr)
        
        # Collect sample metadata (take first row as representative)
        sample_meta = sample_cells.obs.iloc[0].to_dict()
        sample_meta['n_cells'] = sample_cells.n_obs
        sample_info.append(sample_meta)
    
    # Create DataFrame
    pseudobulk_df = pd.DataFrame(
        pseudobulk_data,
        index=[s[sample_column] for s in sample_info],
        columns=adata.var_names
    )
    
    sample_metadata = pd.DataFrame(sample_info)
    
    return pseudobulk_df, sample_metadata

def normalize_pseudobulk(pseudobulk_df, method='CPM'):
    """
    Normalize pseudobulk expression data.
    
    Parameters:
    -----------
    pseudobulk_df : DataFrame
        Raw pseudobulk counts (samples x genes)
    method : str
        Normalization method ('CPM', 'TPM', 'log1p_CPM')
        
    Returns:
    --------
    normalized_df : DataFrame
        Normalized expression matrix
    """
    if method == 'CPM':
        # Counts per million
        total_counts = pseudobulk_df.sum(axis=1)
        normalized_df = pseudobulk_df.div(total_counts, axis=0) * 1e6
    elif method == 'log1p_CPM':
        # Log-transformed CPM
        total_counts = pseudobulk_df.sum(axis=1)
        cpm = pseudobulk_df.div(total_counts, axis=0) * 1e6
        normalized_df = np.log1p(cpm)
    else:
        normalized_df = pseudobulk_df.copy()
    
    return normalized_df

def perform_pca(normalized_df, n_components=20, scale=True):
    """
    Perform PCA on normalized pseudobulk data.
    
    Parameters:
    -----------
    normalized_df : DataFrame
        Normalized expression matrix (samples x genes)
    n_components : int
        Number of principal components
    scale : bool
        Whether to scale features before PCA
        
    Returns:
    --------
    pca_result : dict
        Dictionary containing PCA results and model
    """
    # Prepare data
    if scale:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(normalized_df)
    else:
        scaled_data = normalized_df.values
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, min(scaled_data.shape)-1))
    pca_coords = pca.fit_transform(scaled_data)
    
    # Create DataFrame with PCA coordinates
    pca_df = pd.DataFrame(
        pca_coords,
        index=normalized_df.index,
        columns=[f'PC{i+1}' for i in range(pca_coords.shape[1])]
    )
    
    # Get loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=normalized_df.columns,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])]
    )
    
    return {
        'coords': pca_df,
        'loadings': loadings,
        'explained_variance': pca.explained_variance_ratio_,
        'pca_model': pca,
        'scaler': scaler if scale else None
    }

def visualize_pseudobulk_results(
    pseudobulk_df,
    normalized_df,
    pca_result,
    sample_metadata,
    output_dir,
    color_by=None,
    top_n_genes=50
):
    """
    Generate visualizations for pseudobulk analysis.
    
    Parameters:
    -----------
    pseudobulk_df : DataFrame
        Raw pseudobulk counts
    normalized_df : DataFrame
        Normalized expression
    pca_result : dict
        PCA results from perform_pca()
    sample_metadata : DataFrame
        Sample metadata
    output_dir : str
        Directory to save plots
    color_by : str
        Column in sample_metadata to use for coloring PCA plot
    top_n_genes : int
        Number of top variable genes to show in heatmap
    """
    # Create figure directory
    fig_dir = os.path.join(output_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. PCA Scree plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    variance_explained = pca_result['explained_variance'] * 100
    ax.bar(range(1, len(variance_explained)+1), variance_explained)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('PCA Scree Plot')
    ax.set_xticks(range(1, min(21, len(variance_explained)+1)))
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'pca_scree_plot.png'), dpi=300)
    plt.close()
    
    # 2. PCA scatter plots (PC1 vs PC2, PC2 vs PC3, etc.)
    pca_coords = pca_result['coords']
    
    # Merge with metadata for coloring
    plot_data = pca_coords.merge(sample_metadata, left_index=True, right_on='sample')
    
    # Create multiple PCA plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    pc_pairs = [(1, 2), (1, 3), (2, 3), (1, 4)]
    
    for ax, (pc1, pc2) in zip(axes.flat, pc_pairs):
        if f'PC{pc2}' not in pca_coords.columns:
            ax.axis('off')
            continue
            
        if color_by and color_by in plot_data.columns:
            # Color by specified column
            unique_vals = plot_data[color_by].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_vals)))
            
            for i, val in enumerate(unique_vals):
                mask = plot_data[color_by] == val
                ax.scatter(
                    plot_data.loc[mask, f'PC{pc1}'],
                    plot_data.loc[mask, f'PC{pc2}'],
                    label=val,
                    alpha=0.7,
                    s=100,
                    color=colors[i]
                )
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(
                plot_data[f'PC{pc1}'],
                plot_data[f'PC{pc2}'],
                alpha=0.7,
                s=100
            )
        
        # Add sample labels
        for idx, row in plot_data.iterrows():
            ax.annotate(
                row['sample'],
                (row[f'PC{pc1}'], row[f'PC{pc2}']),
                fontsize=8,
                alpha=0.5
            )
        
        ax.set_xlabel(f'PC{pc1} ({pca_result["explained_variance"][pc1-1]*100:.1f}%)')
        ax.set_ylabel(f'PC{pc2} ({pca_result["explained_variance"][pc2-1]*100:.1f}%)')
        ax.set_title(f'PC{pc1} vs PC{pc2}')
    
    plt.suptitle('PCA of Pseudobulk Samples', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'pca_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Sample correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sample_corr = normalized_df.T.corr()
    sns.heatmap(
        sample_corr,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Correlation'}
    )
    ax.set_title('Sample-Sample Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'sample_correlation.png'), dpi=300)
    plt.close()
    
    # 4. Top variable genes heatmap
    gene_var = normalized_df.var(axis=0)
    top_var_genes = gene_var.nlargest(top_n_genes).index
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Z-score normalize for visualization
    top_genes_data = normalized_df[top_var_genes].T
    z_scores = (top_genes_data - top_genes_data.mean(axis=1, keepdims=True)) / top_genes_data.std(axis=1, keepdims=True)
    
    sns.heatmap(
        z_scores,
        cmap='RdBu_r',
        center=0,
        yticklabels=True,
        xticklabels=True,
        ax=ax,
        cbar_kws={'label': 'Z-score'}
    )
    ax.set_title(f'Top {top_n_genes} Variable Genes (Z-score normalized)')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Genes')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'top_variable_genes.png'), dpi=300)
    plt.close()
    
    # 5. Sample statistics plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Total counts per sample
    total_counts = pseudobulk_df.sum(axis=1)
    axes[0, 0].bar(range(len(total_counts)), total_counts.values)
    axes[0, 0].set_xticks(range(len(total_counts)))
    axes[0, 0].set_xticklabels(total_counts.index, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Total Counts')
    axes[0, 0].set_title('Total Counts per Sample')
    
    # Number of detected genes per sample
    detected_genes = (pseudobulk_df > 0).sum(axis=1)
    axes[0, 1].bar(range(len(detected_genes)), detected_genes.values)
    axes[0, 1].set_xticks(range(len(detected_genes)))
    axes[0, 1].set_xticklabels(detected_genes.index, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Number of Genes')
    axes[0, 1].set_title('Detected Genes per Sample')
    
    # Number of cells per sample
    if 'n_cells' in sample_metadata.columns:
        n_cells = sample_metadata.set_index('sample')['n_cells']
        axes[1, 0].bar(range(len(n_cells)), n_cells.values)
        axes[1, 0].set_xticks(range(len(n_cells)))
        axes[1, 0].set_xticklabels(n_cells.index, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Number of Cells')
        axes[1, 0].set_title('Cells per Sample')
    else:
        axes[1, 0].axis('off')
    
    # Distribution of library sizes
    axes[1, 1].hist(np.log10(total_counts.values + 1), bins=20, edgecolor='black')
    axes[1, 1].set_xlabel('Log10(Total Counts + 1)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Library Sizes')
    
    plt.suptitle('Sample Statistics', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'sample_statistics.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {fig_dir}")

def preprocess_pseudobulk(
    h5ad_path,
    sample_meta_path,
    output_dir,
    sample_column='sample',
    cell_meta_path=None,
    batch_key='batch',
    min_cells=500,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
    normalization_method='log1p_CPM',
    n_pcs=20,
    color_by=None,
    verbose=True
):
    """
    Preprocess single-cell data and perform pseudobulk analysis with PCA.
    
    This function:
      1. Reads and preprocesses the data (filter genes/cells, remove MT genes, etc.)
      2. Performs pseudobulking by sample
      3. Normalizes pseudobulk data
      4. Performs PCA
      5. Generates visualizations
      
    Returns:
      - adata: Preprocessed single-cell data
      - pseudobulk_df: Raw pseudobulk counts
      - normalized_df: Normalized pseudobulk expression
      - pca_result: PCA results
    """
    # Start timing
    start_time = time.time()

    # 0. Create output directories if not present
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating output directory")

    # 1. Read the raw count data from an existing H5AD
    if verbose:
        print('=== Read input dataset ===')
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f'Dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # Attach sample info
    if cell_meta_path is None:
        if sample_column not in adata.obs.columns: 
            adata.obs[sample_column] = adata.obs_names.str.split(':').str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path)
        cell_meta.set_index('barcode', inplace=True)
        adata.obs = adata.obs.join(cell_meta, how='left')

    # Merge sample metadata if provided
    if sample_meta_path is not None:
        sample_meta = pd.read_csv(sample_meta_path)
        adata.obs = adata.obs.merge(sample_meta, on=sample_column, how='left')
    
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=min_features)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if verbose:
        print(f"After basic filtering -- Cells remaining: {adata.n_obs}, Genes remaining: {adata.n_vars}")

    # Mito QC
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()

    # Exclude genes if needed
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()
    if verbose:
        print(f"After removing MT genes and excluded genes -- Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # Filter samples with too few cells
    cell_counts_per_patient = adata.obs.groupby(sample_column).size()
    if verbose:
        print("\nSample counts BEFORE filtering:")
        print(cell_counts_per_patient.sort_values(ascending=False))
    
    patients_to_keep = cell_counts_per_patient[cell_counts_per_patient >= min_cells].index
    if verbose:
        print(f"\nSamples retained (>= {min_cells} cells): {list(patients_to_keep)}")
    
    adata = adata[adata.obs[sample_column].isin(patients_to_keep)].copy()
    
    cell_counts_after = adata.obs[sample_column].value_counts()
    if verbose:
        print("\nSample counts AFTER filtering:")
        print(cell_counts_after.sort_values(ascending=False))

    # Drop genes that are too rare
    min_cells_for_gene = int(0.01 * adata.n_obs)
    sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)
    if verbose:
        print(f"Final filtering -- Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # Optional doublet detection
    if doublet:
        if verbose:
            print("Performing doublet detection...")
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            sc.pp.scrublet(adata, batch_key=sample_column)
            adata = adata[~adata.obs['predicted_doublet']].copy()
        if verbose:
            print(f"After doublet removal -- Cells: {adata.n_obs}")
    
    # Save raw data
    adata.raw = adata.copy()
    
    # ========== PSEUDOBULKING ==========
    if verbose:
        print("\n=== Performing Pseudobulking ===")
    
    # Pseudobulk by sample
    pseudobulk_df, sample_metadata = pseudobulk_samples(adata, sample_column=sample_column)
    if verbose:
        print(f"Pseudobulk matrix shape: {pseudobulk_df.shape}")
    
    # Normalize pseudobulk data
    normalized_df = normalize_pseudobulk(pseudobulk_df, method=normalization_method)
    if verbose:
        print(f"Normalization method: {normalization_method}")
    
    # Perform PCA
    if verbose:
        print("\n=== Performing PCA ===")
    pca_result = perform_pca(normalized_df, n_components=n_pcs)
    
    # Print variance explained
    if verbose:
        print("\nVariance explained by top PCs:")
        for i in range(min(5, len(pca_result['explained_variance']))):
            print(f"  PC{i+1}: {pca_result['explained_variance'][i]*100:.2f}%")
        print(f"  Total (first {n_pcs} PCs): {sum(pca_result['explained_variance'])*100:.2f}%")
    
    # Generate visualizations
    if verbose:
        print("\n=== Generating Visualizations ===")
    visualize_pseudobulk_results(
        pseudobulk_df=pseudobulk_df,
        normalized_df=normalized_df,
        pca_result=pca_result,
        sample_metadata=sample_metadata,
        output_dir=output_dir,
        color_by=color_by
    )
    
    # Save results
    if verbose:
        print("\n=== Saving Results ===")
    
    # Save processed single-cell data
    sc.write(os.path.join(output_dir, 'adata_processed.h5ad'), adata)
    
    # Save pseudobulk data
    pseudobulk_df.to_csv(os.path.join(output_dir, 'pseudobulk_counts.csv'))
    normalized_df.to_csv(os.path.join(output_dir, 'pseudobulk_normalized.csv'))
    
    # Save PCA results
    pca_result['coords'].to_csv(os.path.join(output_dir, 'pca_coordinates.csv'))
    pca_result['loadings'].to_csv(os.path.join(output_dir, 'pca_loadings.csv'))
    
    # Save sample metadata
    sample_metadata.to_csv(os.path.join(output_dir, 'sample_metadata.csv'), index=False)
    
    # Save variance explained
    var_explained_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(pca_result['explained_variance']))],
        'Variance_Explained': pca_result['explained_variance']
    })
    var_explained_df.to_csv(os.path.join(output_dir, 'pca_variance_explained.csv'), index=False)
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if verbose:
        print(f"\n=== Function execution time: {elapsed_time:.2f} seconds ===")
        print(f"Results saved to: {output_dir}")
    
    return adata, pseudobulk_df, normalized_df, pca_result

if __name__ == "__main__":
    # Example usage
    h5ad_path = 'path/to/input_data.h5ad'
    sample_meta_path = 'path/to/sample_metadata.csv'  # Optional
    output_dir = 'path/to/output_directory'
    
    adata, pseudobulk_df, normalized_df, pca_result = preprocess_pseudobulk(
        h5ad_path=h5ad_path,
        sample_meta_path=sample_meta_path,
        output_dir=output_dir,
        sample_column='sample',
        min_cells=500,
        min_features=500,
        pct_mito_cutoff=20,
        exclude_genes=None,
        doublet=True,
        normalization_method='log1p_CPM',
        n_pcs=20,
        color_by='condition',  # Column in sample metadata for coloring PCA
        verbose=True
    )