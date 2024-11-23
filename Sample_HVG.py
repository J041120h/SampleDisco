import os
from anndata import AnnData
# A function that is used to find the sample HVG based on the given cell_cluster and single cell data

def sample_hvg(adata: AnnData, output_dir: str, cell_type_column: str = 'resolution_based_cell_cluster', sample_column: str = 'sample'):
    #Based on the given method find the hvg, and return an anndata only with the selected hvg
    print("a")