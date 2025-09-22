def integrate_and_visualize_paired_data(
    rna_embed_path: str,
    atac_embed_path: str,
    output_dir: str,
    use_rep: str = "X_glue",
    harmony_key: str = "modality",
    tissue_key: str = "tissue",
    sample_id_key: str = "sample_id",
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    figsize: tuple = (12, 5),
    verbose: bool = True
) -> None:
    """
    Integrate paired RNA and ATAC embeddings using Harmony and visualize results.
    
    Parameters
    ----------
    rna_embed_path : str
        Path to RNA embedding h5ad file
    atac_embed_path : str
        Path to ATAC embedding h5ad file
    output_dir : str
        Directory to save outputs
    use_rep : str
        Key in obsm to use for embeddings (default: "X_glue")
    harmony_key : str
        Key to use for Harmony batch correction (default: "modality")
    tissue_key : str
        Key in obs for tissue labels (default: "tissue")
    sample_id_key : str
        Key in obs for sample IDs (default: "sample_id")
    n_neighbors : int
        Number of neighbors for UMAP (default: 30)
    min_dist : float
        Minimum distance for UMAP (default: 0.3)
    figsize : tuple
        Figure size for plots (default: (12, 5))
    verbose : bool
        Whether to print progress messages
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import os
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set plotting parameters
    sc.settings.verbosity = 3 if verbose else 0
    sc.settings.set_figure_params(dpi=100, facecolor='white')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("🧬 Starting Harmony Integration of Paired RNA-ATAC Data")
        print("=" * 60)
    
    # Step 1: Load embedding files
    if verbose:
        print("\n📂 Loading embedding files...")
        print(f"   RNA: {rna_embed_path}")
        print(f"   ATAC: {atac_embed_path}")
    
    # Check if files exist
    if not os.path.exists(rna_embed_path):
        raise FileNotFoundError(f"RNA embedding file not found: {rna_embed_path}")
    if not os.path.exists(atac_embed_path):
        raise FileNotFoundError(f"ATAC embedding file not found: {atac_embed_path}")
    
    # Load data
    rna_adata = ad.read_h5ad(rna_embed_path)
    atac_adata = ad.read_h5ad(atac_embed_path)
    
    if verbose:
        print(f"   RNA shape: {rna_adata.shape}")
        print(f"   ATAC shape: {atac_adata.shape}")
    
    # Step 2: Extract X_glue embeddings
    if verbose:
        print(f"\n🔍 Extracting {use_rep} embeddings...")
    
    if use_rep not in rna_adata.obsm:
        raise ValueError(f"{use_rep} not found in RNA obsm. Available keys: {list(rna_adata.obsm.keys())}")
    if use_rep not in atac_adata.obsm:
        raise ValueError(f"{use_rep} not found in ATAC obsm. Available keys: {list(atac_adata.obsm.keys())}")
    
    rna_embedding = rna_adata.obsm[use_rep].copy()
    atac_embedding = atac_adata.obsm[use_rep].copy()
    
    if verbose:
        print(f"   RNA embedding shape: {rna_embedding.shape}")
        print(f"   ATAC embedding shape: {atac_embedding.shape}")
    
    # Step 3: Modify sample names with modality prefix
    if verbose:
        print("\n🏷️ Modifying sample names with modality prefix...")
    
    # Store original sample IDs for pairing
    rna_original_ids = rna_adata.obs.index.copy()
    atac_original_ids = atac_adata.obs.index.copy()
    
    # Create new index with modality prefix
    rna_adata.obs.index = 'RNA_' + rna_adata.obs.index.astype(str)
    atac_adata.obs.index = 'ATAC_' + atac_adata.obs.index.astype(str)
    
    # Add modality column
    rna_adata.obs[harmony_key] = 'RNA'
    atac_adata.obs[harmony_key] = 'ATAC'
    
    # Store original IDs for pairing visualization
    rna_adata.obs['original_id'] = rna_original_ids.values
    atac_adata.obs['original_id'] = atac_original_ids.values
    
    if verbose:
        print(f"   RNA samples: {rna_adata.n_obs} (prefixed with 'RNA_')")
        print(f"   ATAC samples: {atac_adata.n_obs} (prefixed with 'ATAC_')")
    
    # Step 4: Concatenate datasets
    if verbose:
        print("\n🔗 Concatenating RNA and ATAC data...")
    
    # Create simplified AnnData objects with just embeddings
    rna_simple = ad.AnnData(
        X=rna_embedding,
        obs=rna_adata.obs.copy()
    )
    rna_simple.obsm[use_rep] = rna_embedding
    
    atac_simple = ad.AnnData(
        X=atac_embedding,
        obs=atac_adata.obs.copy()
    )
    atac_simple.obsm[use_rep] = atac_embedding
    
    # Concatenate
    adata_merged = ad.concat([rna_simple, atac_simple], axis=0, join='outer')
    
    # Use embedding as X for processing
    adata_merged.X = adata_merged.obsm[use_rep].copy()
    
    if verbose:
        print(f"   Merged shape: {adata_merged.shape}")
        print(f"   Modalities: {adata_merged.obs[harmony_key].value_counts().to_dict()}")
    
    # Step 5: Run Harmony integration
    if verbose:
        print(f"\n🎵 Running Harmony integration on '{harmony_key}'...")
    
    try:
        import harmonypy as hm
        
        # Run Harmony
        ho = hm.run_harmony(
            adata_merged.X, 
            adata_merged.obs, 
            harmony_key,
            max_iter_harmony=20
        )
        
        # Store harmonized embedding
        adata_merged.obsm['X_harmony'] = ho.Z_corr.T
        
        if verbose:
            print(f"   Harmony integration complete!")
            print(f"   Output shape: {adata_merged.obsm['X_harmony'].shape}")
    
    except ImportError:
        print("⚠️ harmonypy not installed. Install with: pip install harmonypy")
        print("   Skipping Harmony integration, using original embeddings...")
        adata_merged.obsm['X_harmony'] = adata_merged.X.copy()
    
    # Step 6: Compute UMAP on harmonized embeddings
    if verbose:
        print(f"\n🗺️ Computing UMAP visualization...")
    
    # Compute neighbors on harmonized embedding
    sc.pp.neighbors(
        adata_merged, 
        use_rep='X_harmony',
        n_neighbors=n_neighbors,
        metric='cosine'
    )
    
    # Compute UMAP
    sc.tl.umap(adata_merged, min_dist=min_dist)
    
    if verbose:
        print(f"   UMAP computation complete!")
    
    # Step 7: Create visualizations
    if verbose:
        print(f"\n📊 Creating visualizations...")
    
    # Prepare color palette
    n_tissues = adata_merged.obs[tissue_key].nunique() if tissue_key in adata_merged.obs else 0
    if n_tissues > 0:
        tissue_palette = dict(zip(
            adata_merged.obs[tissue_key].unique(),
            sns.color_palette("tab20", n_tissues)
        ))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Visualization 1: Sample embedding labeled by tissue
    if tissue_key in adata_merged.obs:
        if verbose:
            print(f"   Creating tissue visualization...")
        
        for tissue in adata_merged.obs[tissue_key].unique():
            mask = adata_merged.obs[tissue_key] == tissue
            ax1.scatter(
                adata_merged.obsm['X_umap'][mask, 0],
                adata_merged.obsm['X_umap'][mask, 1],
                label=tissue,
                s=10,
                alpha=0.7,
                color=tissue_palette[tissue]
            )
        
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.set_title(f'Sample Embedding by {tissue_key.capitalize()}')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., markerscale=2)
        ax1.grid(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    else:
        # If no tissue info, color by modality
        for modality in ['RNA', 'ATAC']:
            mask = adata_merged.obs[harmony_key] == modality
            ax1.scatter(
                adata_merged.obsm['X_umap'][mask, 0],
                adata_merged.obsm['X_umap'][mask, 1],
                label=modality,
                s=10,
                alpha=0.7
            )
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.set_title('Sample Embedding by Modality')
        ax1.legend()
        ax1.grid(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    
    # Visualization 2: Paired samples connected with lines
    if verbose:
        print(f"   Creating paired sample visualization...")
    
    # Get paired samples (same original ID)
    rna_df = adata_merged.obs[adata_merged.obs[harmony_key] == 'RNA'][['original_id']].copy()
    atac_df = adata_merged.obs[adata_merged.obs[harmony_key] == 'ATAC'][['original_id']].copy()
    
    # Find common samples
    common_samples = set(rna_df['original_id']) & set(atac_df['original_id'])
    
    if verbose:
        print(f"   Found {len(common_samples)} paired samples")
    
    # Plot points first
    for modality, color, marker in [('RNA', 'blue', 'o'), ('ATAC', 'red', '^')]:
        mask = adata_merged.obs[harmony_key] == modality
        ax2.scatter(
            adata_merged.obsm['X_umap'][mask, 0],
            adata_merged.obsm['X_umap'][mask, 1],
            label=modality,
            s=20,
            alpha=0.7,
            color=color,
            marker=marker,
            zorder=2
        )
    
    # Draw lines connecting paired samples
    for sample_id in common_samples:
        # Get RNA position
        rna_mask = (adata_merged.obs[harmony_key] == 'RNA') & (adata_merged.obs['original_id'] == sample_id)
        rna_pos = adata_merged.obsm['X_umap'][rna_mask][0]
        
        # Get ATAC position
        atac_mask = (adata_merged.obs[harmony_key] == 'ATAC') & (adata_merged.obs['original_id'] == sample_id)
        atac_pos = adata_merged.obsm['X_umap'][atac_mask][0]
        
        # Draw line
        ax2.plot(
            [rna_pos[0], atac_pos[0]],
            [rna_pos[1], atac_pos[1]],
            'gray',
            alpha=0.3,
            linewidth=0.5,
            zorder=1
        )
    
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.set_title(f'Paired Samples Connected ({len(common_samples)} pairs)')
    ax2.legend()
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, 'harmony_integration_visualization.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    if verbose:
        print(f"   Saved visualization to: {fig_path}")
    
    plt.show()
    
    # Step 8: Save integrated data
    output_path = os.path.join(output_dir, 'integrated_harmonized.h5ad')
    adata_merged.write(output_path)
    
    if verbose:
        print(f"\n💾 Saved integrated data to: {output_path}")
    
    # Print summary statistics
    if verbose:
        print("\n" + "=" * 60)
        print("✅ Integration and Visualization Complete!")
        print("=" * 60)
        print(f"\n📊 Summary Statistics:")
        print(f"   Total samples: {adata_merged.n_obs}")
        print(f"   RNA samples: {(adata_merged.obs[harmony_key] == 'RNA').sum()}")
        print(f"   ATAC samples: {(adata_merged.obs[harmony_key] == 'ATAC').sum()}")
        print(f"   Paired samples: {len(common_samples)}")
        print(f"   Embedding dimensions: {adata_merged.obsm['X_harmony'].shape[1]}")
        
        if tissue_key in adata_merged.obs:
            print(f"\n   Tissues found:")
            tissue_counts = adata_merged.obs[tissue_key].value_counts()
            for tissue, count in tissue_counts.items():
                print(f"      - {tissue}: {count} samples")
        
        print(f"\n   Output files:")
        print(f"      - Integrated data: {output_path}")
        print(f"      - Visualization: {fig_path}")


# Example usage
if __name__ == "__main__":
    # Example paths - replace with your actual file paths
    integrate_and_visualize_paired_data(
        rna_embed_path="/dcl01/hongkai/data/data/hjiang/Data/multi_omics_testing/multiomics/integration/glue/glue-rna-emb.h5ad",
        atac_embed_path="/dcl01/hongkai/data/data/hjiang/Data/multi_omics_testing/multiomics/integration/glue/glue-atac-emb.h5ad",
        output_dir="/dcl01/hongkai/data/data/hjiang/Data/multi_omics_testing",
        use_rep="X_glue",
        harmony_key="modality",
        tissue_key="tissue",
        sample_id_key="sample_id",
        n_neighbors=30,
        min_dist=0.3,
        figsize=(12, 5),
        verbose=True
    )