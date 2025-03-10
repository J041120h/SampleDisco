import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import harmonypy as hm
import matplotlib.pyplot as plt
from pseudobulk import compute_pseudobulk_dataframes
from combat.pycombat import pycombat
from Harmony import harmony
from EMD import EMD_distances
from VectorDistance import sample_distance
from ChiSquare import chi_square_distance
from jensenshannon import jensen_shannon_distance
from Test import sample_anndata_by_sample, treecor_seurat_mapping,count_samples_in_adata, test_harmony
from Visualization import visualization_harmony, plot_cell_type_proportions_pca, plot_pseudobulk_pca, plot_pseudobulk_batch_test_pca, plot_pseudobulk_umap, plot_pseudobulk_batch_umap

def main():
    output_dir = "/users/hjiang/GenoDistance/result"
    h5ad_path = "/users/hjiang/GenoDistance/Data/count_data.h5ad"
    cell_meta_path="/users/hjiang/GenoDistance/Data/cell_data.csv"
    sample_meta_path = "/users/hjiang/GenoDistance/Data/sample_data.csv"
    method = "hamming"
    cell_group_weight = 0.8
    min_cells= 100
    min_features= 3
    pct_mito_cutoff=20
    exclude_genes=None
    vars_to_regress=['sample']
    resolution=0.5
    verbose=True
    num_PCs=10
    num_harmony=20
    markers = [
            'CD3D', 'CD14', 'CD19', 'NCAM1', 'CD4', 'CD8A',
            'FCGR3A', 'CD1C', 'CD68', 'CD79A', 'CSF3R',
            'CD33', 'CCR7', 'CD38', 'CD27', 'KLRD1'
        ]
    proportion_weight: float = 1.0,
    expression_weight: float = 1.0,
    # methods = [
    #     'euclidean',
    #     'minkowski',
    #     'cityblock',
    #     'chebyshev',
    #     'cosine',
    #     'correlation',
    #     'hamming',
    #     'jaccard',
    #     'canberra',
    #     'braycurtis',
    #     'sqeuclidean',
    #     'matching',
    # ]
    methods = [
        'euclidean',
        'minkowski',
        'cityblock',
        'chebyshev',
        'cosine',
        'correlation',
        'canberra',
        'braycurtis',
        'sqeuclidean',
    ]
    summary_cell_csv_path = "/users/hjiang/GenoDistance/result/summary_cell.csv"
    summary_sample_csv_path = "/users/hjiang/GenoDistance/result/summary_sample.csv"
    AnnData_cell_path = '/users/hjiang/GenoDistance/result/harmony/adata_cell.h5ad'
    AnnData_sample_path = '/users/hjiang/GenoDistance/result/harmony/adata_sample.h5ad'
    vars_to_regress= ['sample']

    #on local mac
    # output_dir = "/Users/harry/Desktop/GenoDistance/result"
    # h5ad_path = "/Users/harry/Desktop/GenoDistance/Data/count_data.h5ad"
    # cell_meta_path="/Users/harry/Desktop/GenoDistance/Data/cell_data.csv"
    # sample_meta_path = "/Users/harry/Desktop/GenoDistance/Data/sample_data.csv"
    # AnnData_cell_path = '/Users/harry/Desktop/GenoDistance/result/harmony/adata_cell.h5ad'
    # AnnData_sample_path = '/Users/harry/Desktop/GenoDistance/result/harmony/adata_sample.h5ad'
    # summary_cell_csv_path = "/Users/harry/Desktop/GenoDistance/result/summary_cell.csv"
    # summary_sample_csv_path = "/Users/harry/Desktop/GenoDistance/result/summary_sample.csv"
    # vars_to_regress = []

    # in /dcs04/hongkai/data/HarryJ
    # output_dir = "/dcs04/hongkai/data/HarryJ/harmony_after_combat"
    # h5ad_path = "/Users/harry/Desktop/GenoDistance/Data/count_data.h5ad"
    # cell_meta_path="/Users/harry/Desktop/GenoDistance/Data/cell_data.csv"
    # sample_meta_path = "/Users/harry/Desktop/GenoDistance/Data/sample_data.csv"
    # AnnData_cell_path = '/dcs04/hongkai/data/HarryJ/harmony_after_combat/harmony/adata_cell.h5ad'
    # AnnData_sample_path = '/dcs04/hongkai/data/HarryJ/harmony_after_combat/harmony/adata_sample.h5ad'
    # summary_cell_csv_path = "/dcs04/hongkai/data/HarryJ/harmony_after_combat/summary_cell.csv"
    # summary_sample_csv_path = "/dcs04/hongkai/data/HarryJ/harmony_after_combat/summary_sample.csv"

    AnnData_cell,AnnData_sample = harmony(h5ad_path, sample_meta_path, output_dir,cell_meta_path, vars_to_regress = vars_to_regress)
    # pseudobulk = compute_pseudobulk_dataframes(AnnData_sample, 'batch', 'sample', 'cell_type', output_dir)
    # plot_pseudobulk_pca(AnnData_sample, output_dir, pseudobulk, verbose=True)
    # plot_pseudobulk_batch_test_pca(AnnData_sample, output_dir, pseudobulk, verbose=True)

    # visualization_harmony(
    #     AnnData_sample,
    #     AnnData_sample,
    #     output_dir,
    #     grouping_columns=['sev.level'],
    #     verbose=True,
    #     dot_size = 3
    # )

    # if os.path.exists(summary_sample_csv_path):
    #     os.remove(summary_sample_csv_path)
    # if os.path.exists(summary_cell_csv_path):
    #     os.remove(summary_cell_csv_path)

    # for md in methods:
    #     print("\n\n\n\n" + md + "\n\n\n\n")
    #     sample_distance(AnnData_sample, os.path.join(output_dir, 'Sample'), f'{md}', summary_sample_csv_path, pseudobulk)
    # EMD_distances(AnnData_sample, os.path.join(output_dir, 'sample_level_EMD'), summary_sample_csv_path)
    # chi_square_distance(AnnData_sample, os.path.join(output_dir, 'Chi_square_sample'), summary_sample_csv_path)
    # jensen_shannon_distance(AnnData_sample, os.path.join(output_dir, 'jensen_shannon_sample'), summary_sample_csv_path)

    print("End of Process")
    print("End of Process")
    print("End of Process")

if __name__ == '__main__':
    main()