import numpy as np
import pandas as pd
import os
import anndata as ad
import rapids_singlecell as rsc
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.safe_save import safe_h5ad_write, ensure_cpu_arrays
from utils.imbalance_cell_type_handeler import filter_modality_imbalanced_clusters


def cell_types_multiomics(
    adata,
    modality_column="modality",
    rna_modality_value="RNA",
    atac_modality_value="ATAC",
    cell_type_column="cell_type",
    cluster_resolution=0.8,
    use_rep="X_glue",
    num_PCs=50,
    k_neighbors=15,
    transfer_metric="cosine",
    compute_umap=True,
    save=False,
    output_dir=None,
    defined_output_path=None,
    verbose=True,
    generate_plots=True,
):
    """
    Cell type assignment for multi-omics data.
    
    This function:
    1. Clusters only RNA cells using Leiden clustering
    2. Transfers cell type labels to ATAC cells using k-NN in the scGLUE embedding space
    
    Parameters:
    -----------
    adata : AnnData
        Integrated AnnData object containing both RNA and ATAC cells
    modality_column : str
        Column in adata.obs indicating modality
    rna_modality_value : str
        Value in modality_column for RNA cells
    atac_modality_value : str
        Value in modality_column for ATAC cells
    cell_type_column : str
        Column name to store cell type assignments
    cluster_resolution : float
        Resolution parameter for Leiden clustering
    use_rep : str
        Representation to use for clustering and label transfer (e.g., 'X_glue')
    num_PCs : int
        Number of components to use from the representation
    k_neighbors : int
        Number of neighbors to use for label transfer
    transfer_metric : str
        Distance metric for k-NN label transfer
    compute_umap : bool
        Whether to compute UMAP embedding
    save : bool
        Whether to save the result
    output_dir : str
        Output directory for saving
    defined_output_path : str
        Specific path for saving the output file
    verbose : bool
        Whether to print progress messages
    generate_plots : bool
        Whether to generate visualization plots
        
    Returns:
    --------
    AnnData
        AnnData object with cell type assignments
    """
    
    if verbose:
        print("\n" + "="*60)
        print("Cell Type Assignment for Multi-omics Data")
        print("="*60)
    
    # Set random seed
    from utils.random_seed import set_global_seed
    set_global_seed(seed=42, verbose=verbose)
    
    # Validate inputs
    if modality_column not in adata.obs.columns:
        raise ValueError(f"Modality column '{modality_column}' not found in adata.obs")
    
    if use_rep not in adata.obsm:
        raise ValueError(f"Representation '{use_rep}' not found in adata.obsm")
    
    # Separate RNA and ATAC cells
    rna_mask = adata.obs[modality_column] == rna_modality_value
    atac_mask = adata.obs[modality_column] == atac_modality_value
    
    n_rna = rna_mask.sum()
    n_atac = atac_mask.sum()
    
    if verbose:
        print(f"\nData composition:")
        print(f"  RNA cells: {n_rna}")
        print(f"  ATAC cells: {n_atac}")
        print(f"  Total cells: {adata.n_obs}")
    
    if n_rna == 0:
        raise ValueError("No RNA cells found in the data")
    
    if n_atac == 0:
        raise ValueError("No ATAC cells found in the data")
    
    # Get embeddings
    embedding = adata.obsm[use_rep]
    if hasattr(embedding, 'get'):
        embedding = embedding.get()
    
    # Limit to num_PCs if specified
    if num_PCs is not None and embedding.shape[1] > num_PCs:
        embedding = embedding[:, :num_PCs]
        if verbose:
            print(f"\nUsing first {num_PCs} components from {use_rep}")
    
    # =========================================
    # Step 1: Cluster RNA cells only
    # =========================================
    if verbose:
        print(f"\n--- Step 1: Clustering RNA cells ---")
        print(f"  Resolution: {cluster_resolution}")
    
    # Create a subset for RNA cells
    rna_adata = adata[rna_mask].copy()
    
    # Move to GPU for clustering
    rsc.get.anndata_to_GPU(rna_adata)
    
    # Build neighbors graph for RNA cells
    if verbose:
        print("  Building neighborhood graph for RNA cells...")
    rsc.pp.neighbors(rna_adata, use_rep=use_rep, n_pcs=num_PCs, random_state=42)
    
    # Perform Leiden clustering
    if verbose:
        print("  Performing Leiden clustering...")
    rsc.tl.leiden(
        rna_adata,
        resolution=cluster_resolution,
        key_added=cell_type_column,
        random_state=42,
    )
    
    # Convert cluster labels to 1-indexed strings
    rna_adata.obs[cell_type_column] = (
        (rna_adata.obs[cell_type_column].astype(int) + 1).astype(str).astype("category")
    )
    
    n_clusters = rna_adata.obs[cell_type_column].nunique()
    if verbose:
        print(f"  Found {n_clusters} clusters in RNA cells")
    
    # Move back to CPU
    rsc.get.anndata_to_CPU(rna_adata)
    
    # Get RNA cell type assignments
    rna_cell_types = rna_adata.obs[cell_type_column].copy()
    
    # =========================================
    # Step 2: Transfer labels to ATAC cells
    # =========================================
    if verbose:
        print(f"\n--- Step 2: Transferring labels to ATAC cells ---")
        print(f"  Using k={k_neighbors} nearest neighbors")
        print(f"  Metric: {transfer_metric}")
    
    # Get RNA and ATAC embeddings
    rna_embedding = embedding[rna_mask]
    atac_embedding = embedding[atac_mask]
    
    # Use GPU for k-NN
    import cupy as cp
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    
    rna_embedding_gpu = cp.asarray(rna_embedding, dtype=cp.float32)
    atac_embedding_gpu = cp.asarray(atac_embedding, dtype=cp.float32)
    
    # Find k nearest RNA neighbors for each ATAC cell
    nn = cuNearestNeighbors(
        n_neighbors=k_neighbors,
        metric=transfer_metric,
        algorithm='brute' if n_rna < 50000 else 'auto'
    )
    nn.fit(rna_embedding_gpu)
    distances_gpu, indices_gpu = nn.kneighbors(atac_embedding_gpu)
    
    # Transfer to CPU
    indices = cp.asnumpy(indices_gpu)
    distances = cp.asnumpy(distances_gpu)
    
    # Clean up GPU memory
    del rna_embedding_gpu, atac_embedding_gpu, distances_gpu, indices_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    # Convert RNA cell types to array for indexing
    rna_cell_types_array = rna_cell_types.values
    
    # Perform majority voting for label transfer
    atac_cell_types = []
    atac_confidence = []
    
    for i in range(n_atac):
        neighbor_indices = indices[i]
        neighbor_labels = rna_cell_types_array[neighbor_indices]
        
        # Majority voting
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]
        confidence = np.max(counts) / k_neighbors
        
        atac_cell_types.append(majority_label)
        atac_confidence.append(confidence)
    
    atac_cell_types = pd.Series(atac_cell_types, index=adata.obs.index[atac_mask])
    atac_confidence = pd.Series(atac_confidence, index=adata.obs.index[atac_mask])
    
    if verbose:
        mean_confidence = np.mean(atac_confidence)
        print(f"  Label transfer complete")
        print(f"  Mean transfer confidence: {mean_confidence:.3f}")
    
    # =========================================
    # Step 3: Combine cell type assignments
    # =========================================
    if verbose:
        print(f"\n--- Step 3: Combining cell type assignments ---")
    
    # Initialize cell type column
    adata.obs[cell_type_column] = pd.NA
    
    # Assign RNA cell types
    adata.obs.loc[rna_mask, cell_type_column] = rna_cell_types.values
    
    # Assign ATAC cell types (transferred)
    adata.obs.loc[atac_mask, cell_type_column] = atac_cell_types.values
    
    # Add transfer confidence for ATAC cells
    adata.obs['label_transfer_confidence'] = np.nan
    adata.obs.loc[atac_mask, 'label_transfer_confidence'] = atac_confidence.values
    
    # Convert to category
    adata.obs[cell_type_column] = adata.obs[cell_type_column].astype("category")
    
    if verbose:
        print(f"  Cell type assignments complete")
        print(f"\n  Cell type distribution:")
        for ct in sorted(adata.obs[cell_type_column].unique()):
            n_rna_ct = ((adata.obs[modality_column] == rna_modality_value) & 
                        (adata.obs[cell_type_column] == ct)).sum()
            n_atac_ct = ((adata.obs[modality_column] == atac_modality_value) & 
                         (adata.obs[cell_type_column] == ct)).sum()
            print(f"    Cluster {ct}: RNA={n_rna_ct}, ATAC={n_atac_ct}")
    
    # =========================================
    # Step 4: Filter modality-imbalanced clusters
    # =========================================
    if verbose:
        print(f"\n--- Step 4: Filtering modality-imbalanced clusters ---")
    
    adata = filter_modality_imbalanced_clusters(
        adata=adata,
        modality_column=modality_column,
        cluster_column=cell_type_column,
        min_proportion_of_expected=0.05,
        verbose=verbose
    )
    
    # =========================================
    # Step 5: Compute UMAP (optional)
    # =========================================
    if compute_umap:
        if verbose:
            print(f"\n--- Step 5: Computing UMAP ---")
        
        # Move to GPU
        rsc.get.anndata_to_GPU(adata)
        
        # Build neighbors for full dataset
        rsc.pp.neighbors(adata, use_rep=use_rep, n_pcs=num_PCs, random_state=42)
        
        # Compute UMAP
        rsc.tl.umap(adata, min_dist=0.5)
        
        # Move back to CPU
        rsc.get.anndata_to_CPU(adata)
    
    # Ensure all arrays are on CPU
    adata = ensure_cpu_arrays(adata)
    
    # =========================================
    # Step 6: Generate visualizations
    # =========================================
    if generate_plots and output_dir:
        if verbose:
            print(f"\n--- Step 6: Generating visualizations ---")
        
        vis_dir = os.path.join(output_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        
        _generate_multiomics_celltype_plots(
            adata=adata,
            output_dir=vis_dir,
            cell_type_column=cell_type_column,
            modality_column=modality_column,
            rna_modality_value=rna_modality_value,
            atac_modality_value=atac_modality_value,
            verbose=verbose
        )
    
    # =========================================
    # Step 7: Save results
    # =========================================
    if save and output_dir and not defined_output_path:
        out_pre = os.path.join(output_dir, "preprocess")
        os.makedirs(out_pre, exist_ok=True)
        save_path = os.path.join(out_pre, "atac_rna_integrated.h5ad")
        safe_h5ad_write(adata, save_path, verbose=verbose)
    
    if defined_output_path:
        safe_h5ad_write(adata, defined_output_path, verbose=verbose)
    
    if verbose:
        print("\n" + "="*60)
        print("Cell type assignment complete!")
        print("="*60 + "\n")
    
    return adata


def _generate_multiomics_celltype_plots(
    adata,
    output_dir,
    cell_type_column="cell_type",
    modality_column="modality",
    rna_modality_value="RNA",
    atac_modality_value="ATAC",
    verbose=True
):
    """
    Generate visualization plots for multi-omics cell type assignment.
    """
    
    sc.settings.set_figure_params(dpi=80, facecolor='white', figsize=(8, 6))
    plt.rcParams['figure.max_open_warning'] = 50
    
    # Check if UMAP exists
    has_umap = 'X_umap' in adata.obsm
    
    # 1. UMAP colored by cell type
    if has_umap:
        if verbose:
            print("  Generating cell type UMAP...")
        
        plt.figure(figsize=(12, 8))
        sc.pl.umap(adata, color=cell_type_column,
                   title="Cell Types (RNA clustering + ATAC transfer)",
                   save=False, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "umap_cell_type.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. UMAP colored by modality
    if has_umap:
        if verbose:
            print("  Generating modality UMAP...")
        
        plt.figure(figsize=(12, 8))
        sc.pl.umap(adata, color=modality_column,
                   title="Modality Distribution",
                   save=False, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "umap_modality.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Split UMAP by modality
    if has_umap:
        if verbose:
            print("  Generating split modality UMAPs...")
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # RNA cells
        rna_mask = adata.obs[modality_column] == rna_modality_value
        rna_adata = adata[rna_mask]
        
        ax = axes[0]
        umap_coords = rna_adata.obsm['X_umap']
        cell_types = rna_adata.obs[cell_type_column].astype('category')
        
        for i, ct in enumerate(sorted(cell_types.cat.categories)):
            mask = cell_types == ct
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                      label=ct, s=10, alpha=0.6)
        
        ax.set_title(f"RNA Cells (n={rna_mask.sum()})", fontsize=14)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
        
        # ATAC cells
        atac_mask = adata.obs[modality_column] == atac_modality_value
        atac_adata = adata[atac_mask]
        
        ax = axes[1]
        umap_coords = atac_adata.obsm['X_umap']
        cell_types = atac_adata.obs[cell_type_column].astype('category')
        
        for i, ct in enumerate(sorted(cell_types.cat.categories)):
            mask = cell_types == ct
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                      label=ct, s=10, alpha=0.6)
        
        ax.set_title(f"ATAC Cells (n={atac_mask.sum()})", fontsize=14)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "umap_split_by_modality.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Label transfer confidence distribution (ATAC cells only)
    if 'label_transfer_confidence' in adata.obs.columns:
        if verbose:
            print("  Generating label transfer confidence plot...")
        
        atac_mask = adata.obs[modality_column] == atac_modality_value
        confidence = adata.obs.loc[atac_mask, 'label_transfer_confidence'].dropna()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(confidence, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=confidence.mean(), color='red', linestyle='--', 
                       label=f'Mean: {confidence.mean():.3f}')
        axes[0].set_xlabel("Transfer Confidence")
        axes[0].set_ylabel("Number of Cells")
        axes[0].set_title("Label Transfer Confidence Distribution")
        axes[0].legend()
        
        # Box plot by cell type
        atac_adata = adata[atac_mask]
        conf_by_ct = pd.DataFrame({
            'confidence': atac_adata.obs['label_transfer_confidence'],
            'cell_type': atac_adata.obs[cell_type_column]
        })
        
        conf_by_ct.boxplot(column='confidence', by='cell_type', ax=axes[1])
        axes[1].set_xlabel("Cell Type")
        axes[1].set_ylabel("Transfer Confidence")
        axes[1].set_title("Transfer Confidence by Cell Type")
        plt.suptitle('')  # Remove automatic title
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "label_transfer_confidence.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Cell type composition heatmap
    if verbose:
        print("  Generating cell type composition heatmap...")
    
    # Cross-tabulation
    crosstab = pd.crosstab(
        adata.obs[cell_type_column],
        adata.obs[modality_column]
    )
    
    # Proportions
    crosstab_props = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Raw counts
    sns.heatmap(
        crosstab,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Number of cells'},
        ax=ax1
    )
    ax1.set_title('Cell Count Distribution', fontsize=14)
    ax1.set_xlabel('Modality')
    ax1.set_ylabel('Cell Type')
    
    # Proportions
    sns.heatmap(
        crosstab_props,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Percentage (%)'},
        ax=ax2
    )
    ax2.set_title('Modality Distribution within Each Cell Type (%)', fontsize=14)
    ax2.set_xlabel('Modality')
    ax2.set_ylabel('Cell Type')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "celltype_modality_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save distribution table
    csv_path = os.path.join(output_dir, "celltype_modality_distribution.csv")
    crosstab.to_csv(csv_path)
    
    if verbose:
        print(f"  Saved distribution table to: {csv_path}")
        print("  Visualizations complete!")