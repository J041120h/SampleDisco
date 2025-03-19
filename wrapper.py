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
from Visualization import visualization_harmony, plot_cell_type_proportions_pca, plot_pseudobulk_batch_test_pca
from PCA import process_anndata_with_pca
from CCA import CCA_Call
from CellType import cell_types, cell_type_assign
from CCA_test import find_optimal_cell_resolution, cca_pvalue_test

def wrapper(output_dir, cell_column, cluster_resolution, markers, num_features=2000, num_PCs=20, num_harmony=30, vars_to_regress_for_harmony=None, method='average', metric='euclidean', distance_mode='centroid', verbose=True):
    AnnData_cell,AnnData_sample = harmony(h5ad_path, sample_meta_path, output_dir, cell_column, cell_meta_path, vars_to_regress = vars_to_regress)

    
if __name__ == "__main__":
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
    proportion_weight: float = 1.0
    expression_weight: float = 1.0
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
    cell_column = "celltype"
    wrapper()