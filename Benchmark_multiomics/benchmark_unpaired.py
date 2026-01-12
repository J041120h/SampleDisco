import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_sample_embedding(
    embedding_file,
    metadata_file,
    output_file="figures/sample_embedding_plot.png",
    color_by="tissue",
):
    """
    Plot sample embedding colored by tissue or modality and save to local file
    
    Args:
        embedding_file: path to embedding CSV
        metadata_file: path to metadata CSV
        output_file: path to save output figure
        color_by: 'tissue' or 'modality'
    """
    print(f"[INFO] Starting plot_sample_embedding (color_by={color_by})")
    
    # Ensure output directory exists
    outdir = os.path.dirname(output_file)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        print(f"[INFO] Output directory ensured: {outdir}")
    
    # Color palettes
    tissue_color_palette = [
        '#D7462F', '#F28E2D', '#E8B219', '#8FBD5A', '#2CA02C',
        '#17BECF', '#00BBFF', '#E377C2', '#986DBF', '#7D7BDA'
    ]
    
    modality_colors = {
        'RNA': '#6CC6D8',
        'ATAC': '#EE7564'
    }
    
    # Read data
    print(f"[INFO] Reading embedding file: {embedding_file}")
    embedding = pd.read_csv(embedding_file, index_col=0)
    print(f"[INFO] Embedding loaded: {embedding.shape[0]} samples")
    
    print(f"[INFO] Reading metadata file: {metadata_file}")
    metadata = pd.read_csv(metadata_file, index_col=0)
    print(f"[INFO] Metadata loaded: {metadata.shape[0]} samples")
    
    # Normalize sample names
    embedding.index = embedding.index.astype(str).str.lower()
    metadata.index = metadata.index.astype(str).str.lower()
    print("[INFO] Sample names normalized to lowercase")
    
    # Merge
    merged = embedding.join(metadata, how="inner")
    print(f"[INFO] Samples after merge: {merged.shape[0]}")
    
    # Determine grouping variable and colors
    if color_by == "tissue":
        groups = sorted(merged["tissue"].unique())
        print(f"[INFO] Number of tissue groups: {len(groups)}")
        group_colors = {group: tissue_color_palette[i % len(tissue_color_palette)] 
                       for i, group in enumerate(groups)}
        group_column = "tissue"
    elif color_by == "modality":
        groups = sorted(merged["modality"].unique())
        print(f"[INFO] Number of modality groups: {len(groups)}")
        group_colors = modality_colors
        group_column = "modality"
    else:
        raise ValueError(f"color_by must be 'tissue' or 'modality', got {color_by}")
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax_main = plt.subplot2grid((1, 6), (0, 0), colspan=5)
    ax_legend = plt.subplot2grid((1, 6), (0, 5))
    
    # If modality plot, draw connecting lines first
    if color_by == "modality":
        # Extract base sample names (without modality suffix)
        merged['base_sample'] = merged.index.str.replace('_rna$|_atac$', '', regex=True)
        
        # Find paired samples
        paired_samples = merged.groupby('base_sample').filter(lambda x: len(x) == 2)
        base_samples = paired_samples['base_sample'].unique()
        
        print(f"[INFO] Found {len(base_samples)} paired samples")
        
        # Draw lines connecting paired samples
        for base_sample in base_samples:
            pair = paired_samples[paired_samples['base_sample'] == base_sample]
            if len(pair) == 2:
                coords = pair[['PC1', 'PC2']].values
                ax_main.plot(coords[:, 0], coords[:, 1], 
                           color='gray', alpha=0.3, linewidth=0.5, zorder=1)
    
    # Plot points
    for group in groups:
        d = merged[merged[group_column] == group]
        ax_main.scatter(
            d["PC1"], d["PC2"],
            s=50,
            alpha=0.7,
            color=group_colors[group],
            label=group,
            zorder=2
        )
    
    # Formatting - equal aspect ratio
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.spines["left"].set_linewidth(1.5)
    ax_main.spines["bottom"].set_linewidth(1.5)
    ax_main.set_xlabel("UMAP1", fontsize=12, fontweight="bold")
    ax_main.set_ylabel("UMAP2", fontsize=12, fontweight="bold")
    ax_main.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    ax_main.set_aspect('equal', adjustable='box')
    
    # Legend
    handles, labels = ax_main.get_legend_handles_labels()
    ax_legend.axis("off")
    ax_legend.legend(handles, labels, loc="center left", frameon=False, fontsize=10)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[INFO] Plot successfully saved to: {output_file}")
    print("[INFO] plot_sample_embedding finished")
    
    return output_file


if __name__ == "__main__":
    # Plot by tissue
    out_tissue = plot_sample_embedding(
        "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/embeddings/sample_proportion_embedding.csv",
        "/dcl01/hongkai/data/data/hjiang/Data/paired/sample_metadata.csv",
        output_file="/users/hjiang/GenoDistance/figure/embedding/sample_embedding_tissue.png",
        color_by="tissue"
    )
    print(f"Tissue plot saved to {out_tissue}")
    
    # Plot by modality
    out_modality = plot_sample_embedding(
        "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/embeddings/sample_proportion_embedding.csv",
        "/dcl01/hongkai/data/data/hjiang/Data/paired/sample_metadata.csv",
        output_file="/users/hjiang/GenoDistance/figure/embedding/sample_embedding_modality.png",
        color_by="modality"
    )
    print(f"Modality plot saved to {out_modality}")