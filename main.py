import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import harmonypy as hm
import matplotlib.pyplot as plt
from Harmony import treecor_harmony
from SampleSimilarityCellProportion import calculate_sample_distances_cell_proprotion
from SampleSimilarityCellExpression import calculate_sample_distances_cell_expression
from SampleSimilarity import Sample_distances, calculate_sample_distances_weighted_expression

def main():
    output_dir = "/users/harry/desktop/GenoDistance/result"
    count_path = "/users/harry/output_matrix.csv"
    sample_meta_path = "/users/harry/desktop/GenoDistance/Data/raw_data_sample_meta.csv"
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

    # treecor_harmony(count_path, sample_meta_path, output_dir,cell_meta_path, markers)
    AnnData = sc.read_h5ad("/users/harry/desktop/GenoDistance/result/integrate.h5ad")
    # for proportion_weight in np.arange(1.0, 10.0, 1.0):
    #     temo_output_dir = os.path.join(output_dir, 'weight' + str(proportion_weight))
    Sample_distances(AnnData, output_dir)
    # calculate_sample_distances_weighted_expression(AnnData, output_dir)
    # calculate_sample_distances_cell_proprotion(AnnData, output_dir)
    # calculate_sample_distances_cell_expression(AnnData, output_dir)

if __name__ == '__main__':
    main()