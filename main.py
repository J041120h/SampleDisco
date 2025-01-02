import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import harmonypy as hm
import matplotlib.pyplot as plt
from Harmony import treecor_harmony
from EMD import EMD_distances
from VectorDistance import sample_distance
from ChiSquare import chi_square_distance
from jensenshannon import jensen_shannon_distance
def main():
    output_dir = "/users/harry/desktop/GenoDistance/result"
    count_path = "/users/harry/output_matrix.csv"
    sample_meta_path = "/users/harry/desktop/GenoDistance/Data/raw_data_sample_meta.csv"
    summary_sample_csv_path = "/users/harry/desktop/GenoDistance/result/summary_sample.csv"
    summary_cell_csv_path = "/users/harry/desktop/GenoDistance/result/summary_cell.csv"
    method = "hamming"
    cell_group_weight = 0.8
    min_cells= 100
    min_features= 3
    pct_mito_cutoff=20
    exclude_genes=None
    vars_to_regress=['sample']
    resolution=0.5
    verbose=True
    cell_meta_path="/users/harry/desktop/GenoDistance/Data/integrated_cellmeta.csv"
    num_PCs=20
    num_harmony=20
    markers = [
            'CD3D', 'CD14', 'CD19', 'NCAM1', 'CD4', 'CD8A',
            'FCGR3A', 'CD1C', 'CD68', 'CD79A', 'CSF3R',
            'CD33', 'CCR7', 'CD38', 'CD27', 'KLRD1'
        ]
    proportion_weight: float = 1.0,
    expression_weight: float = 1.0,
    methods = [
        'euclidean',
        'minkowski',
        'cityblock',
        'chebyshev',
        'cosine',
        'correlation',
        'hamming',
        'jaccard',
        'canberra',
        'braycurtis',
        'sqeuclidean',
        'matching',
        'dice',
    ]


    treecor_harmony(count_path, sample_meta_path, output_dir,cell_meta_path, markers)
    AnnData_cell = sc.read_h5ad("/users/harry/desktop/GenoDistance/result/harmony/adata_cell.h5ad")
    AnnData_sample = sc.read_h5ad("/users/harry/desktop/GenoDistance/result/harmony/adata_sample.h5ad")
    EMD_distances(AnnData_sample, os.path.join(output_dir, 'sample_level_EMD'), summary_sample_csv_path)
    EMD_distances(AnnData_cell, os.path.join(output_dir, 'cell_level_EMD'), summary_cell_csv_path)
    for md in methods:
        print("\n\n\n\n" + md + "\n\n\n\n")
        sample_distance(AnnData_cell, os.path.join(output_dir, 'Cell'), f'{md}', summary_cell_csv_path)
        sample_distance(AnnData_sample, os.path.join(output_dir, 'Sample'), f'{md}', summary_sample_csv_path)
    chi_square_distance(AnnData_sample, os.path.join(output_dir, 'Chi_square_sample'), summary_sample_csv_path)
    jensen_shannon_distance(AnnData_sample, os.path.join(output_dir, 'jensen_shannon_sample'), summary_sample_csv_path)
    chi_square_distance(AnnData_sample, os.path.join(output_dir, 'Chi_square_cell'), summary_cell_csv_path)
    jensen_shannon_distance(AnnData_sample, os.path.join(output_dir, 'jensen_shannon_cell'), summary_cell_csv_path)

if __name__ == '__main__':
    main()