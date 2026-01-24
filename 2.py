import pandas as pd
import sys

def print_unique_cell_types(csv_path):
    df = pd.read_csv(csv_path)
    unique_values = df['cell_type'].unique()
    for v in unique_values:
        print(v)

if __name__ == "__main__":
    print_unique_cell_types('/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/resolution_optimization_expression/Integration_optimization_rna_expression/resolutions/resolution_0.050_expression/preprocess/cell_type.csv')
