import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import harmonypy as hm
import matplotlib.pyplot as plt
import subprocess
import sys
import platform
import shutil
from pseudobulk import compute_pseudobulk_dataframes
from Harmony import harmony
from EMD import EMD_distances
from VectorDistance import sample_distance
from ChiSquare import chi_square_distance
from jensenshannon import jensen_shannon_distance
from Visualization import visualization
from PCA import process_anndata_with_pca
from CCA import CCA_Call
from CellType import cell_types, cell_type_assign
from CCA_test import find_optimal_cell_resolution, cca_pvalue_test
from TSCAN import TSCAN
from resolution_parallel import find_optimal_cell_resolution_parallel
from trajectory_diff_gene import identify_pseudoDEGs, summarize_results, run_differential_analysis_for_all_paths
from cluster import cluster
from sample_clustering.RAISIN import *
from sample_clustering.RAISIN_TEST import *
from ATAC_general_pipeline import *

if __name__ == "__main__":
    # Example 1: Standard usage (scanpy LSI)
    # atac_sample, atac_cluster = run_scatac_pipeline(
    #     filepath     = "/Users/harry/Desktop/GenoDistance/Data/test_ATAC.h5ad",
    #     output_dir  = "/Users/harry/Desktop/GenoDistance/result",
    #     metadata_path= "/Users/harry/Desktop/GenoDistance/Data/ATAC_Metadata.csv",
    #     sample_column= "sample",
    #     batch_key    = 'Donor',
    #     leiden_resolution = 0.8,
    #     use_snapatac2_dimred = True  # Use snapATAC2 for dim reduction
    # )

    atac_sample = sc.read_h5ad("/Users/harry/Desktop/GenoDistance/result/ATAC/harmony/ATAC_sample.h5ad")
    pseudobulk_df = compute_pseudobulk_dataframes(
            adata=atac_sample,
            batch_col='Donor',
            sample_col="sample",
            celltype_col='cell_type',
            output_dir="/Users/harry/Desktop/GenoDistance/result/ATAC",
            n_features=20000,
            frac=0.3,
            verbose=True
        )
    
    print("end of process")
        # process_anndata_with_pca(
        #     adata=AnnData_sample,
        #     pseudobulk=pseudobulk_df,
        #     sample_col = sample_col,
        #     grouping_columns = [batch_col],
        #     n_expression_pcs=n_expression_pcs,
        #     n_proportion_pcs=n_proportion_pcs,
        #     output_dir=pca_output_dir,
        #     verbose=pca_verbose
        # )