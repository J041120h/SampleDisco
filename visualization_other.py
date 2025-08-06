import os
import matplotlib.pyplot as plt
import scanpy as sc
from Grouping import find_sample_grouping
from visualization_emebedding import plot_sample_cell_proportions_embedding, plot_sample_cell_expression_embedding

def _preprocessing(
    adata_sample_diff,
    adata_pseudobulk,
    output_dir,
    grouping_columns,
    age_bin_size,
    age_column,
    verbose
):
    # 1. Create output sub-directory
    output_dir = os.path.join(output_dir, "visualization")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    required_columns = grouping_columns.copy()
    if age_bin_size is not None:
        required_columns.append(age_column)
    missing_columns = []

    for col in required_columns:
        if col not in adata_pseudobulk.obs.columns:
            missing_columns.append(col)
    
    if missing_columns:
        raise ValueError(f"Missing required columns in pseudobulk AnnData: {missing_columns}")
    
    if verbose:
        print(f"[_preprocessing] Verified required columns in pseudobulk AnnData: {required_columns}")
    
    return output_dir

def _plot_umap_by_cell_type(adata_sample_diff, output_dir, dot_size, verbose):
    """
    UMAP colored by 'cell_type' with cell type labels on cluster centroids.
    """
    if verbose:
        print("[plot_umap_by_cell_type] UMAP colored by 'cell_type'.")
    
    plt.figure(figsize=(12, 10))
    
    # Create the UMAP plot
    sc.pl.umap(
        adata_sample_diff,
        color='cell_type',
        legend_loc=None,
        frameon=False,
        size=dot_size,
        show=False
    )
    
    # Get UMAP coordinates
    umap_coords = adata_sample_diff.obsm['X_umap']
    
    # Get cell type labels
    cell_types = adata_sample_diff.obs['cell_type']
    
    # Calculate centroids for each cell type and add labels
    unique_cell_types = cell_types.unique()
    
    for cell_type in unique_cell_types:
        # Get indices for this cell type
        mask = cell_types == cell_type
        
        # Calculate centroid coordinates
        centroid_x = umap_coords[mask, 0].mean()
        centroid_y = umap_coords[mask, 1].mean()
        
        # Add text label at centroid
        plt.text(
            centroid_x, centroid_y, 
            str(cell_type),
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor='black',
                alpha=0.8
            )
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_umap_by_cell_type.pdf'), bbox_inches='tight')
    plt.close()

def visualization(
    AnnData_cell,
    AnnData_sample,
    pseudobulk_anndata,
    output_dir,
    grouping_columns=['sev.level'],
    age_bin_size=None,
    age_column='age',
    verbose=True,
    dot_size=3,

    plot_dendrogram_flag=True,
    plot_umap_by_cell_type_flag=True,
    plot_cell_type_proportions_pca_flag=False,
    plot_cell_type_expression_umap_flag=False,
):
    """
    Main function to handle all steps. Sub-functions are called conditionally based on flags.
    """
    # 1. Preprocessing
    if grouping_columns:
        output_dir = _preprocessing(
            AnnData_cell,
            pseudobulk_anndata,
            output_dir,
            grouping_columns,
            age_bin_size,
            age_column,
            verbose
        )

    # # 2. Dendrogram
    # if plot_dendrogram_flag:
    #     _plot_dendrogram(adata_sample_diff, output_dir, verbose)

    # # 3. UMAP by cell type
    # if plot_umap_by_cell_type_flag:
    #     _plot_umap_by_cell_type(adata_sample_diff, output_dir, dot_size, verbose)

    # # 4. Cell type proportions PCA embedding
    # if plot_cell_type_proportions_pca_flag:
    #     plot_sample_cell_proportions_embedding(
    #         adata_sample_diff,
    #         os.path.dirname(output_dir),  # pass parent (function itself appends 'harmony')
    #         grouping_columns=grouping_columns,
    #         verbose=verbose
    #     )

    # # 5. Cell expression UMAP embedding
    # if plot_cell_type_expression_umap_flag:
    #     plot_sample_cell_expression_embedding(
    #         adata_sample_diff, 
    #         os.path.dirname(output_dir),  # pass parent (function itself appends 'harmony')
    #         grouping_columns=grouping_columns,
    #         verbose=verbose
    #     )

    if verbose:
        print("[visualization] All requested visualizations saved.")

if __name__ == "__main__":
    rna_output_dir = "/Users/harry/Desktop/GenoDistance/result/rna"
    temp_cell_path = os.path.join(rna_output_dir, "preprocess", "adata_cell.h5ad")
    temp_sample_path = os.path.join(rna_output_dir, "preprocess", "adata_sample.h5ad")
    if not os.path.exists(temp_cell_path) or not os.path.exists(temp_sample_path):
        raise ValueError("Preprocessed data paths are not provided and default files path do not exist.")
    AnnData_cell_path = temp_cell_path
    AnnData_sample_path = temp_sample_path
    
    AnnData_cell = sc.read(AnnData_cell_path)
    AnnData_sample = sc.read(AnnData_sample_path)
    temp_pseudobulk_path = os.path.join(rna_output_dir, "pseudobulk", "pseudobulk_sample.h5ad")
    if not os.path.exists(temp_pseudobulk_path):
        raise ValueError("dimensionality_reduction data paths are not provided and default files path do not exist.")
    pseudobulk_anndata = sc.read(temp_pseudobulk_path)

    visualization(
    AnnData_cell,
    AnnData_sample,
    pseudobulk_anndata,
    rna_output_dir,
    grouping_columns=['sev.level'],
    age_bin_size=None,
    verbose=True,
    dot_size=3,

    plot_dendrogram_flag=True,
    plot_umap_by_cell_type_flag=True,
    plot_cell_type_proportions_pca_flag=False,
    plot_cell_type_expression_umap_flag=False,
)