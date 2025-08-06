import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import scanpy as sc
from Grouping import find_sample_grouping
from visualization_emebedding import plot_proportion_embedding, plot_expression_embedding

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

def plot_dendrogram(AnnData_cell, output_dir, verbose=True):
    if 'X_pca_harmony' not in AnnData_cell.obsm.keys():
        raise ValueError("X_pca_harmony not found in AnnData_cell.obsm.")
    
    X_harmony = AnnData_cell.obsm['X_pca_harmony']
    cell_type_col = next((col for col in ['cell_type', 'celltype', 'cluster', 'leiden', 'seurat_clusters'] if col in AnnData_cell.obs.columns), None)
    if cell_type_col is None:
        raise ValueError("No cell type column found.")
    
    cell_types = AnnData_cell.obs[cell_type_col].astype(str)
    unique_cell_types = cell_types.unique()
    if verbose:
        print(f"Found {len(unique_cell_types)} unique cell types.")

    cell_type_embeddings = [np.mean(X_harmony[cell_types == ct], axis=0) for ct in unique_cell_types if np.sum(cell_types == ct) > 0]
    condensed_distances = pdist(cell_type_embeddings, metric='euclidean')
    linkage_matrix = linkage(condensed_distances, method='ward')

    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=unique_cell_types, leaf_rotation=45, leaf_font_size=10, color_threshold=0.7 * max(linkage_matrix[:, 2]))
    plt.title('Cell Type Relationship Dendrogram', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Cell Types', fontsize=12, fontweight='bold')
    plt.ylabel('Distance', fontsize=12, fontweight='bold')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'cell_type_dendrogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"Dendrogram saved to: {os.path.join(output_dir, 'cell_type_dendrogram.png')}")

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

    # 2. Dendrogram
    if plot_dendrogram_flag:
        plot_dendrogram(AnnData_cell, output_dir, verbose=verbose)

    if plot_cell_type_proportions_pca_flag:
        for col in grouping_columns:
            plot_proportion_embedding(
                adata = pseudobulk_anndata,
                color_col = col,
                output_dir = output_dir, 
                grouping_columns=grouping_columns,
                verbose=verbose
            )

    if plot_cell_type_expression_umap_flag:
        for col in grouping_columns:
            plot_expression_embedding(
                adata = pseudobulk_anndata,
                color_col = col,
                output_dir = output_dir, 
                grouping_columns=grouping_columns,
                verbose=verbose
            )

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
    grouping_columns=['batch', 'sev.level'],
    age_bin_size=None,
    verbose=True,
    dot_size=3,

    plot_dendrogram_flag=True,
    plot_umap_by_cell_type_flag=True,
    plot_cell_type_proportions_pca_flag=True,
    plot_cell_type_expression_umap_flag=True,
)