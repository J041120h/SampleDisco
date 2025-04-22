import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import harmonypy as hm
import matplotlib.pyplot as plt 
from pseudobulk import compute_pseudobulk_dataframes
from Harmony import harmony
from EMD import EMD_distances
from VectorDistance import sample_distance
from ChiSquare import chi_square_distance
from jensenshannon import jensen_shannon_distance
from PCA import process_anndata_with_pca
from CCA import CCA_Call
from CellType import cell_types, cell_type_assign
from CCA_test import find_optimal_cell_resolution, cca_pvalue_test
from TSCAN import TSCAN
# from linux.harmony_linux import harmony_linux
# from linux.CellType_linux import cell_types_linux, cell_type_assign_linux
# from linux.CCA_test_linux import find_optimal_cell_resolution_linux
from resolution_parallel import find_optimal_cell_resolution_parallel
from sample_clustering.tree_cut import cut_tree_by_group_count
from sample_clustering.cluster_DEG_visualization import cluster_dge_visualization

def main():
    output_dir = "/users/hjiang/GenoDistance/Test/result"
    h5ad_path = "/users/hjiang/GenoDistance/Test/test_count_data.h5ad"
    sample_meta_path = "/users/hjiang/GenoDistance/Test/test_sample_data.csv"
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
    AnnData_cell_path = '/users/hjiang/GenoDistance/Test/result/harmony/adata_cell.h5ad'
    AnnData_sample_path = '/users/hjiang/GenoDistance/Test/result/harmony/adata_sample.h5ad'
    vars_to_regress= ['sample']
    cell_column = "celltype"

    #on local mac
    output_dir = "/Users/harry/Desktop/GenoDistance/result"
    h5ad_path = "/Users/harry/Desktop/GenoDistance/Data/count_data.h5ad"
    cell_meta_path="/Users/harry/Desktop/GenoDistance/Data/cell_data.csv"
    sample_meta_path = "/Users/harry/Desktop/GenoDistance/Data/sample_data.csv"
    AnnData_cell_path = '/Users/harry/Desktop/GenoDistance/result/harmony/adata_cell.h5ad'
    AnnData_sample_path = '/Users/harry/Desktop/GenoDistance/result/harmony/adata_sample.h5ad'
    summary_cell_csv_path = "/Users/harry/Desktop/GenoDistance/result/summary_cell.csv"
    summary_sample_csv_path = "/Users/harry/Desktop/GenoDistance/result/summary_sample.csv"
    vars_to_regress = []
    cell_column = "cell_type"

    # in /dcs04/hongkai/data/HarryJ
    # output_dir = "/dcs04/hongkai/data/HarryJ/harmony_after_combat"
    # h5ad_path = "/Users/harry/Desktop/GenoDistance/Data/count_data.h5ad"
    # cell_meta_path="/Users/harry/Desktop/GenoDistance/Data/cell_data.csv"
    # sample_meta_path = "/Users/harry/Desktop/GenoDistance/Data/sample_data.csv"
    # AnnData_cell_path = '/dcs04/hongkai/data/HarryJ/harmony_after_combat/harmony/adata_cell.h5ad'
    # AnnData_sample_path = '/dcs04/hongkai/data/HarryJ/harmony_after_combat/harmony/adata_sample.h5ad'
    # summary_cell_csv_path = "/dcs04/hongkai/data/HarryJ/harmony_after_combat/summary_cell.csv"
    # summary_sample_csv_path = "/dcs04/hongkai/data/HarryJ/harmony_after_combat/summary_sample.csv"

    # AnnData_cell,AnnData_sample = harmony_linux(h5ad_path, sample_meta_path, output_dir, cell_column, vars_to_regress = vars_to_regress)
    # AnnData_cell = sc.read(AnnData_cell_path)
    AnnData_sample = sc.read(AnnData_sample_path)
    tree_path = "/Users/harry/Desktop/GenoDistance/result/Tree/Combined/Consensus.nex"
    desired_groups = 4
    result = cut_tree_by_group_count(tree_path, desired_groups, format='nexus', verbose=True, tol=0)
    cluster_dge_visualization(sample_to_clade = result, folder_path = output_dir)
    # AnnData_cell = cell_types_linux(
    #         adata=AnnData_cell,
    #         cell_column=cell_column,
    #         Save=True,
    #         output_dir=output_dir,
    #         cluster_resolution=0.82,
    #         markers=markers,
    #         method=method,
    #         num_PCs=num_PCs,
    #         verbose=verbose
    #     )

    # cell_type_assign_linux(
    #     adata_cluster=AnnData_cell,
    #     adata=AnnData_sample,
    #     Save=True,
    #     output_dir=output_dir,
    #     verbose=verbose
    # )
    # cell_type_assign(AnnData_cell, AnnData_sample, Save=True, output_dir=output_dir,verbose = True)  
    # pseudobulk_df = compute_pseudobulk_dataframes(AnnData_sample, 'batch', 'sample', 'cell_type', output_dir, verbose = True)
    # process_anndata_with_pca(adata = AnnData_sample, pseudobulk = pseudobulk_df, output_dir = output_dir, adata_path=AnnData_sample_path, verbose = True)
    # result = TSCAN(AnnData_sample, "X_pca_expression", 2, output_dir, grouping_columns = ["sev.level"], verbose = True, origin=None)


    # first_component_score_proportion, first_component_score_expression, ptime_proportion, ptime_expression= CCA_Call(adata = AnnData_sample, sample_meta_path=sample_meta_path, output_dir=output_dir, sample_col = 'sample', sev_col = 'sev.level', ptime = True)
    # from trajectory_diff_gene import identify_pseudoDEGs, summarize_results, run_differential_analysis_for_all_paths
    # results = identify_pseudoDEGs(
    #                 pseudobulk= pseudobulk_df,
    #                 sample_meta_path=sample_meta_path,
    #                 ptime_expression=ptime_expression,
    #                 fdr_threshold= 0.1,
    #                 effect_size_threshold= 0.1,
    #                 top_n_genes = 100,
    #                 sample_col='sample',
    #                 visualize_all_deg = True,
    #                 output_dir=os.path.join(output_dir, 'visualization'),
    #                 verbose=True
    #             )
    
    # all_path_results = run_differential_analysis_for_all_paths(
    #     TSCAN_results=result,
    #     pseudobulk_df=pseudobulk_df,
    #     sample_meta_path=sample_meta_path,
    #     fdr_threshold=0.1,
    #     effect_size_threshold=0.1,
    #     visualize_all_deg =True,
    #     top_gene_number=30,
    #     base_output_dir=os.path.join(output_dir, 'TSCAN_DEF'),
    # )
    # cca_pvalue_test(AnnData_sample, sample_meta_path, "X_pca_expression", 0.5807686668238389, output_dir)
    # cca_pvalue_test(AnnData_sample, sample_meta_path, "X_pca_proportion", 0.44774005254663607, output_dir)

    # column = "X_pca_proportion"
    # find_optimal_cell_resolution(AnnData_cell, AnnData_sample, output_dir, sample_meta_path, column) 

    # plot_cell_type_proportions_pca(AnnData_sample, output_dir)
    # plot_pseudobulk_pca(AnnData_sample, output_dir)
    # plot_pseudobulk_batch_test_pca(AnnData_sample, output_dir)
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